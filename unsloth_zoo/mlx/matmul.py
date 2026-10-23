# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Affine `x @ W^T` on the stock packed weights: small-row 4- and 8-bit qmm for Apple GPUs without stock NAX
kernels, and multi-row qmv that gives each row its one-row stock bits."""

import functools
import hashlib
import itertools
import time

import mlx.core as mx

from . import nax


# From 6 rows, simdgroup matrices on exact `offset + q` operands; each group folds
# `s * (P - offset * sum(x)) + b * sum(x)` in fp32, stock qmv's factorization.
_SIMD_QMM_BITS = (4, 8)
_SIMD_QMM_GROUP_SIZES = (32, 64, 128)
_SIMD_QMM_MAX_ROWS = 24
_SIMD_QMM_SIMDGROUPS = 8   # 256 threads: M2 and M3 GPUs cap a threadgroup at 896
# (operand, its bits' type, the code's shift into them, the offset's bits, in both halves for a 16-bit
# operand, offset). bf16 cannot hold `256 + q`, so 8-bit codes multiply as fp32.
_SIMD_QMM_OPERANDS = {
    (4, mx.bfloat16): (mx.bfloat16, mx.uint16, 3, 0x41804180, 16),
    (4, mx.float16): (mx.float16, mx.uint16, 6, 0x4C004C00, 16),
    (8, mx.bfloat16): (mx.float32, mx.uint32, 15, 0x43800000, 256),
    (8, mx.float16): (mx.float32, mx.uint32, 15, 0x43800000, 256),
}

_SIMD_QMM_SOURCE = """
    constexpr int G = K / GS;
    constexpr int CODES = 32 / BITS;   // per packed word
    constexpr int STEPS = CODES / 2;
    constexpr bool WIDE = sizeof(O) == 4;
    const int M = x_shape[0];
    const int N = w_shape[0];
    const uint simd = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    // A lane holds elements (fm, fn) and (fm, fn + 1) of every 8 x 8 simdgroup matrix.
    const uint fm = ((lane >> 2) & 4) | ((lane >> 1) & 3);
    const uint fn = ((lane >> 1) & 4) | ((lane & 1) << 1);
    const uint n0 = threadgroup_position_in_grid.y * (8 * CT) + fm;
    const uint g0 = simd * G / NSG, g1 = (simd + 1) * G / NSG;
    // Step j of four packed words multiplies codes j and j + STEPS of each: both halves of a lane's word.
    const uint kx = (fm >> 1) * CODES + (fm & 1) * STEPS;
    const device T* rows[RT][2];
    for (int r = 0; r < RT; ++r) {   // rows past M read the last one and are never stored
        rows[r][0] = x + ulong(min(r * 8 + int(fn), M - 1)) * K + kx;
        rows[r][1] = x + ulong(min(r * 8 + int(fn) + 1, M - 1)) * K + kx;
    }
    float2 out[RT][CT];
    for (int r = 0; r < RT; ++r) for (int c = 0; c < CT; ++c) out[r][c] = float2(0.0f);

    for (uint g = g0; g < g1; ++g) {
        simdgroup_matrix<float, 8, 8> acc[RT][CT];
        float2 sum[RT];
        for (int r = 0; r < RT; ++r) {
            sum[r] = float2(0.0f);
            for (int c = 0; c < CT; ++c) acc[r][c] = simdgroup_matrix<float, 8, 8>(0.0f);
        }
        for (uint k = g * GS; k < (g + 1) * GS; k += 4 * CODES) {
            vec<T, STEPS> xa[RT], xb[RT];
            for (int r = 0; r < RT; ++r) {
                xa[r] = *(const device vec<T, STEPS>*)(rows[r][0] + k);
                xb[r] = *(const device vec<T, STEPS>*)(rows[r][1] + k);
                for (int j = 0; j < STEPS; ++j) sum[r] += float2(float(xa[r][j]), float(xb[r][j]));
            }
            for (int c = 0; c < CT; ++c) {
                const uint word = w[(ulong(n0 + c * 8) * K + k) / CODES + (fn >> 1)];
                for (int j = 0; j < STEPS; ++j) {
                    const uint pair = (word >> (BITS * j)) & (((1u << BITS) - 1u) * 0x00010001u);
                    simdgroup_matrix<O, 8, 8> a, b;
                    // 16-bit operands take both elements from one shift of the pair.
                    const uint lo = (WIDE ? (pair & 0xFFFFu) << SHIFT : pair << SHIFT) | OFFSET_BITS;
                    const uint hi = WIDE ? ((pair >> 16) << SHIFT) | OFFSET_BITS : lo >> 16;
                    a.thread_elements()[0] = as_type<O>(OB(lo));
                    a.thread_elements()[1] = as_type<O>(OB(hi));
                    for (int r = 0; r < RT; ++r) {
                        b.thread_elements()[0] = O(xa[r][j]);
                        b.thread_elements()[1] = O(xb[r][j]);
                        simdgroup_multiply_accumulate(acc[r][c], a, b, acc[r][c]);
                    }
                }
            }
        }
        for (int r = 0; r < RT; ++r) {   // the lanes sharing fn differ in the fm bits of the lane index
            sum[r] += float2(simd_shuffle_xor(sum[r].x, 2), simd_shuffle_xor(sum[r].y, 2));
            sum[r] += float2(simd_shuffle_xor(sum[r].x, 4), simd_shuffle_xor(sum[r].y, 4));
            sum[r] += float2(simd_shuffle_xor(sum[r].x, 16), simd_shuffle_xor(sum[r].y, 16));
        }
        for (int c = 0; c < CT; ++c) {
            const float s = float(scales[ulong(n0 + c * 8) * G + g]);
            const float bias = float(biases[ulong(n0 + c * 8) * G + g]);
            for (int r = 0; r < RT; ++r) {
                const float2 p = float2(acc[r][c].thread_elements()[0], acc[r][c].thread_elements()[1]);
                out[r][c] += s * (p - float(OFFSET) * sum[r]) + bias * sum[r];
            }
        }
    }

    if (NSG > 1) {
        threadgroup float2 partials[RT * CT * NSG * 32];
        for (int i = 0; i < RT * CT; ++i) partials[(i * NSG + simd) * 32 + lane] = out[i / CT][i % CT];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd != 0) return;
        for (int i = 0; i < RT * CT; ++i) {
            float2 total = float2(0.0f);
            for (int p = 0; p < NSG; ++p) total += partials[(i * NSG + p) * 32 + lane];
            out[i / CT][i % CT] = total;
        }
    }
    for (int r = 0; r < RT; ++r) {
        const int m = r * 8 + int(fn);
        for (int c = 0; c < CT; ++c) {
            if (m < M) y[ulong(m) * N + n0 + c * 8] = T(out[r][c].x);
            if (m + 1 < M) y[ulong(m + 1) * N + n0 + c * 8] = T(out[r][c].y);
        }
    }
"""

# Up to this many rows an 8 x 8 tile is mostly padding: scalar lanes split K (`s * q + b` in fp32); wider tiles spill.
_SIMD_QMM_FEW_ROWS = 5
_SIMD_QMM_FEW_ROW_SOURCE = """
    constexpr int G = K / GS;
    constexpr int PACKS = K / 8;
    constexpr int WORDS = BITS / 4;   // per pack
    const int N = w_shape[0];
    const int simd = int(simdgroup_index_in_threadgroup);
    const int lane = int(thread_index_in_simdgroup);
    const int n0 = int(threadgroup_position_in_grid.y) * 2;
    const int p1 = (simd + 1) * PACKS / NSG;
    float acc[2 * M];
    _Pragma("unroll") for (int i = 0; i < 2 * M; ++i) acc[i] = 0.0f;
    for (int p = simd * PACKS / NSG + lane; p < p1; p += 32) {
        const int g = p * 8 / GS;
        vec<T, 8> xv[M];
        _Pragma("unroll") for (int r = 0; r < M; ++r) xv[r] = *(const device vec<T, 8>*)(x + (r * K + p * 8));
        // Every load of the step is issued before its arithmetic.
        uint word[2 * WORDS];
        float s[2], bias[2];
        _Pragma("unroll") for (int c = 0; c < 2; ++c) {
            _Pragma("unroll") for (int h = 0; h < WORDS; ++h) word[c * WORDS + h] = w[((n0 + c) * PACKS + p) * WORDS + h];
            s[c] = float(scales[(n0 + c) * G + g]);
            bias[c] = float(biases[(n0 + c) * G + g]);
        }
        _Pragma("unroll") for (int c = 0; c < 2; ++c) {
            _Pragma("unroll") for (int h = 0; h < WORDS; ++h) {
                _Pragma("unroll") for (int i = 0; i < 8 / WORDS; ++i) {
                    const float weight = float((word[c * WORDS + h] >> (BITS * i)) & ((1u << BITS) - 1u)) * s[c] + bias[c];
                    _Pragma("unroll") for (int r = 0; r < M; ++r)
                        acc[c * M + r] += float(xv[r][h * (8 / WORDS) + i]) * weight;
                }
            }
        }
    }
    _Pragma("unroll") for (int i = 0; i < 2 * M; ++i) acc[i] = simd_sum(acc[i]);
    threadgroup float partials[NSG * 2 * M];
    if (lane == 0) { _Pragma("unroll") for (int i = 0; i < 2 * M; ++i) partials[simd * (2 * M) + i] = acc[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0 && lane < 2 * M) {
        float total = 0.0f;
        _Pragma("unroll") for (int q = 0; q < NSG; ++q) total += partials[q * (2 * M) + lane];
        y[(lane % M) * N + n0 + lane / M] = T(total);
    }
"""

# Constants in the source (MLX re-parses template args per call), one kernel name per configuration.
_SIMD_QMM_TYPE, _SIMD_QMM_CONSTANT = "typedef {1} {0};\n", "constexpr int {} = {};\n"
_SIMD_QMM_METAL_TYPES = {mx.bfloat16: "bfloat16_t", mx.float16: "float16_t", mx.float32: "float",
                         mx.uint16: "uint16_t", mx.uint32: "uint32_t"}

SIMD_QMM_PROBE_KEY = "simd_qmm:" + hashlib.sha256(
    (_SIMD_QMM_TYPE + _SIMD_QMM_CONSTANT + _SIMD_QMM_SOURCE + _SIMD_QMM_FEW_ROW_SOURCE).encode()).hexdigest()[:12]


@functools.cache
def _simd_qmm_launch(M, N, K, group_size, bits, dtype):
    rows, simdgroups, column_tiles = simd_qmm_geometry(M, N, K, group_size)
    if rows <= _SIMD_QMM_FEW_ROWS:
        name, source, grid = "unsloth_simd_few_row_qmm", _SIMD_QMM_FEW_ROW_SOURCE, N // column_tiles
        constants = dict(T = dtype, K = K, GS = group_size, BITS = bits, M = M, NSG = simdgroups)
    else:
        name, source, grid = "unsloth_simd_small_m_qmm", _SIMD_QMM_SOURCE, N // (8 * column_tiles)
        operand, operand_bits, shift, offset_bits, offset = _SIMD_QMM_OPERANDS[bits, dtype]
        constants = dict(T = dtype, K = K, GS = group_size, RT = rows // 8, CT = column_tiles, NSG = simdgroups,
                         BITS = bits, O = operand, OB = operand_bits, SHIFT = shift, OFFSET_BITS = offset_bits,
                         OFFSET = offset)
    values = {key: _SIMD_QMM_METAL_TYPES.get(value, value) for key, value in constants.items()}
    header = "".join((_SIMD_QMM_TYPE if isinstance(constants[key], mx.Dtype) else _SIMD_QMM_CONSTANT).format(key, value)
                     for key, value in values.items())
    kernel = mx.fast.metal_kernel(name = "_".join(map(str, (name, *values.values()))),
                                  input_names = ["x", "w", "scales", "biases"], output_names = ["y"],
                                  source = header + source)
    return functools.partial(kernel, grid = (32 * simdgroups, grid, 1), threadgroup = (32 * simdgroups, 1, 1),
                             output_shapes = [(M, N)], output_dtypes = [dtype])


def simd_qmm_available():
    """Whether this is an Apple GPU (not a paravirtual one) whose stock MLX dispatches no NAX kernels."""
    return (mx.metal.is_available() and nax._gpu_generation() is not None
            and not (nax._nax_gpu() and nax._stock_nax_kernels()))


def simd_qmm_supported(N, K, group_size, bits, mode):
    """Whether the small-row kernels cover an `[N, K]` weight quantized this way."""
    return (mode == "affine" and bits in _SIMD_QMM_BITS and group_size in _SIMD_QMM_GROUP_SIZES
            and N > 0 and N % 8 == 0 and K > 0 and K % group_size == 0
            # the few-row kernel indexes in 32 bits: packed words, and M rows of x and of the output
            and N * K * bits // 32 < 1 << 31 and max(N, K) * _SIMD_QMM_FEW_ROWS < 1 << 31)


def simd_qmm_geometry(M, N, K, group_size):
    """(rows, simdgroups, column tiles per simdgroup) for a call of M rows; the few-row kernel takes exactly M."""
    if M <= _SIMD_QMM_FEW_ROWS:
        return M, 2, 2
    rows = -(-M // 8) * 8
    # A second column tile reuses each input read, until three row tiles leave no registers for it.
    return rows, min(_SIMD_QMM_SIMDGROUPS, K // group_size), 2 if N % 16 == 0 and rows <= 16 else 1


def simd_qmm(x, w, scales, biases, group_size, bits):
    """`quantized_matmul(x, w, scales, biases, transpose=True)` for a 2-D x of at most 24 rows."""
    M, K = x.shape
    return _simd_qmm_launch(M, w.shape[0], K, group_size, bits, x.dtype)(inputs = [x, w, scales, biases])[0]


# `nax._QMM_ROWS_BY_GPU` format; measured on an M3 Max, used for every generation.
_SIMD_QMM_ROWS = ((4, None, 1 << 23, 7168, 3, 24), (4, None, 1 << 24, 65536, 3, 24), (4, None, 1 << 24, None, 3, 16),
                  (8, None, 1 << 23, None, 3, 16))


def simd_qmm_row_range(N, K, group_size, bits):
    """The row counts (lowest, highest) where the kernel is measured faster than stock for `[N, K]`."""
    return nax._measured_row_range(_SIMD_QMM_ROWS, _SIMD_QMM_MAX_ROWS, N, K, group_size, bits)


def probe_simd_qmm():
    """Subprocess probe: each bit width, every row tile count full and ragged, every group size, bf16 and fp16."""
    for bits, dtype in itertools.product(_SIMD_QMM_BITS, (mx.bfloat16, mx.float16)):
        for (N, K), group_size in itertools.product(((384, 1024), (72, 128)), _SIMD_QMM_GROUP_SIZES):
            w = mx.random.normal((N, K), key = mx.random.key(group_size)) * 0.05
            w, scales, biases = mx.quantize(w.astype(dtype), group_size = group_size, bits = bits)
            for M in (2, 5, 6, 8, 13, 16, 24):
                x = mx.random.normal((M, K), key = mx.random.key(M)).astype(dtype)
                native = mx.quantized_matmul(x, w, scales, biases, transpose = True,
                                             group_size = group_size, bits = bits).astype(mx.float32)
                got = simd_qmm(x, w, scales, biases, group_size, bits).astype(mx.float32)
                error = mx.abs(got - native).max().item()
                assert error <= 0.02 * mx.abs(native).max().item(), (bits, group_size, M, error)


# Rows of affine `x @ W^T` with stock qmv_fast's one-row arithmetic, so each row equals its one-row call bit for bit
# while each weight is read once per launch, where mlx-vlm's exact verifier re-reads or re-decodes it per row.
QMV_ROWS_MAX = 8
_QMV_ROWS_BITS = (4, 5, 6, 8)
_QMV_ROWS_GROUP_SIZES = (32, 64, 128)

# A row's inputs as the installed MLX's `load_vector` forms them, and `xt`, each code product's factor.
_QMV_ROWS_INPUTS = """
    float x_thread[VPT];
    const float sum = load_vector<T, float, VPT, BITS>(xr, x_thread);
    float xt[NP];
    if (BITS == 5) {
        _Pragma("clang loop unroll(full)")
        for (int p = 0; p < 2; ++p) {
            const thread float* v = x_thread + 8 * p;
            thread float* f = xt + 12 * p;
            f[0] = v[0]; f[1] = v[1]; f[2] = v[1] * 256.0f; f[3] = v[2]; f[4] = v[3]; f[5] = v[3] * 256.0f;
            f[6] = v[4]; f[7] = v[4] * 256.0f; f[8] = v[5]; f[9] = v[6]; f[10] = v[6] * 256.0f; f[11] = v[7];
        }
    } else if (BITS == 6) {
        _Pragma("clang loop unroll(full)")
        for (int p = 0; p < 2; ++p) {
            const thread float* v = x_thread + 4 * p;
            thread float* f = xt + 6 * p;
            f[0] = v[0]; f[1] = v[1]; f[2] = v[1] * 256.0f; f[3] = v[2]; f[4] = v[2] * 256.0f; f[5] = v[3];
        }
    } else {
        _Pragma("clang loop unroll(full)")
        for (int j = 0; j < VPT; ++j) xt[j] = x_thread[j];
    }
"""

# Multiply-add bound on smaller GPUs: a lane decodes its codes once for all rows (half codes are exact) and forms
# each row's inputs per four outputs or reads them from `qmv_rows_inputs`. A row count compiles once for all N and K.
_QMV_ROWS_SOURCE = """
    constexpr int VPT = (BITS == 4 || BITS == 5) ? 16 : 8;
    constexpr int NP = BITS == 5 ? 24 : (BITS == 6 ? 12 : VPT);
    constexpr int LANE_BYTES = VPT * BITS / 8;
    const int K = w_shape[w_ndim - 1] * 32 / BITS;
    const int N = w_shape[w_ndim - 2];
    const int ROW_BYTES = K * BITS / 8;
    const int KG = K / GS;
    const uint lane = thread_index_in_simdgroup;
    const int n0 = int(thread_position_in_grid.y) * 4;
    const device uint8_t* ws = (const device uint8_t*)w + n0 * ROW_BYTES + int(lane) * LANE_BYTES;
    const device T* sc = scales + n0 * KG + int(lane) / (GS / VPT);
    const device T* bs = biases + n0 * KG + int(lane) / (GS / VPT);
    float result[ROWS][4];
    _Pragma("clang loop unroll(full)")
    for (int t = 0; t < ROWS; ++t) _Pragma("clang loop unroll(full)") for (int r = 0; r < 4; ++r) result[t][r] = 0;
    for (int k = 0; k < K; k += 32 * VPT) {
        half code[4][NP];
        float s[4], b[4];
        _Pragma("clang loop unroll(full)")
        for (int r = 0; r < 4; ++r) {
            const device uint8_t* wl = ws + r * ROW_BYTES;
            if (BITS == 4) {
                _Pragma("clang loop unroll(full)")
                for (int p = 0; p < 4; ++p) {
                    const uint16_t v = ((const device uint16_t*)wl)[p];
                    code[r][4 * p] = v & 0x000f; code[r][4 * p + 1] = v & 0x00f0;
                    code[r][4 * p + 2] = v & 0x0f00; code[r][4 * p + 3] = v & 0xf000;
                }
            } else if (BITS == 5) {
                _Pragma("clang loop unroll(full)")
                for (int p = 0; p < 2; ++p) {
                    const device uint8_t* wp = wl + 5 * p;
                    const uint8_t w0 = wp[0], w1 = wp[1], w2 = wp[2], w3 = wp[3], w4 = wp[4];
                    code[r][12 * p] = w0 & 0x1f; code[r][12 * p + 1] = w0 & 0xe0; code[r][12 * p + 2] = w1 & 0x03;
                    code[r][12 * p + 3] = w1 & 0x7c; code[r][12 * p + 4] = w1 & 0x80; code[r][12 * p + 5] = w2 & 0x0f;
                    code[r][12 * p + 6] = w2 & 0xf0; code[r][12 * p + 7] = w3 & 0x01; code[r][12 * p + 8] = w3 & 0x3e;
                    code[r][12 * p + 9] = w3 & 0xc0; code[r][12 * p + 10] = w4 & 0x07; code[r][12 * p + 11] = w4 & 0xf8;
                }
            } else if (BITS == 6) {
                const uint8_t w0 = wl[0], w1 = wl[1], w2 = wl[2], w3 = wl[3], w4 = wl[4], w5 = wl[5];
                code[r][0] = w0 & 0x3f; code[r][1] = w0 & 0xc0; code[r][2] = w1 & 0x0f;
                code[r][3] = w1 & 0xf0; code[r][4] = w2 & 0x03; code[r][5] = w2 & 0xfc;
                code[r][6] = w3 & 0x3f; code[r][7] = w3 & 0xc0; code[r][8] = w4 & 0x0f;
                code[r][9] = w4 & 0xf0; code[r][10] = w5 & 0x03; code[r][11] = w5 & 0xfc;
            } else {
                _Pragma("clang loop unroll(full)")
                for (int j = 0; j < 8; ++j) code[r][j] = wl[j];
            }
            s[r] = sc[r * KG];
            b[r] = bs[r * KG];
        }
        _Pragma("clang loop unroll(full)")
        for (int t = 0; t < ROWS; ++t) {
            const int row = t;
            ROW_INPUTS
            _Pragma("clang loop unroll(full)")
            for (int r = 0; r < 4; ++r) {
                float accum = 0;
                if (BITS == 4) {
                    _Pragma("clang loop unroll(full)")
                    for (int i = 0; i < 16; i += 4) {
                        accum += (xt[i] * float(code[r][i]) + xt[i + 1] * float(code[r][i + 1])
                                  + xt[i + 2] * float(code[r][i + 2]) + xt[i + 3] * float(code[r][i + 3]));
                    }
                } else {
                    _Pragma("clang loop unroll(full)")
                    for (int j = 0; j < NP; ++j) accum += float(code[r][j]) * xt[j];
                }
                result[t][r] += s[r] * accum + sum * b[r];
            }
        }
        ws += 32 * LANE_BYTES;
        sc += 32 * VPT / GS;
        bs += 32 * VPT / GS;
    }
    _Pragma("clang loop unroll(full)")
    for (int r = 0; r < 4; ++r) {
        _Pragma("clang loop unroll(full)")
        for (int t = 0; t < ROWS; ++t) {
            const float v = simd_sum(result[t][r]);
            if (lane == 0) y[t * N + n0 + r] = T(v);
        }
    }
"""
_QMV_ROWS_READ = """const int chunk = row * (K / VPT) + (k / VPT) + int(lane);
            const float sum = sums[chunk];
            float xt[NP];
            _Pragma("clang loop unroll(full)")
            for (int j = 0; j < NP; j += 4) {
                const float4 v = *(const device float4*)(xs + chunk * NP + j);
                xt[j] = v.x; xt[j + 1] = v.y; xt[j + 2] = v.z; xt[j + 3] = v.w;
            }"""

_QMV_ROWS_PREPARE = """
    constexpr int VPT = (BITS == 4 || BITS == 5) ? 16 : 8;
    constexpr int NP = BITS == 5 ? 24 : (BITS == 6 ? 12 : VPT);
    const int K = x_shape[1];
    const int t = int(thread_position_in_grid.y);
    const int chunk = int(thread_position_in_grid.x);
    const device T* xr = x + t * K + chunk * VPT;
""" + _QMV_ROWS_INPUTS + """
    device float* out = xs + (t * (K / VPT) + chunk) * NP;
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < NP; ++j) out[j] = xt[j];
    sums[t * (K / VPT) + chunk] = sum;
"""


def _qmv_rows_source(prepared):
    inputs = _QMV_ROWS_READ if prepared else "const device T* xr = x + row * K + k + int(lane) * VPT;" + _QMV_ROWS_INPUTS
    return _QMV_ROWS_SOURCE.replace("ROW_INPUTS", inputs)


QMV_ROWS_PROBE_KEY = "qmv_rows:" + hashlib.sha256((_QMV_ROWS_SOURCE + _QMV_ROWS_INPUTS + _QMV_ROWS_READ
                                                   + _QMV_ROWS_PREPARE).encode()).hexdigest()[:12]


@functools.cache
def _qmv_rows_kernel(prepared):
    header = nax._mlx_kernel_header("quantized.h", "qdot")
    return header and mx.fast.metal_kernel(
        name = f"unsloth_qmv_rows{'_prepared' if prepared else ''}",
        input_names = [*(("xs", "sums") if prepared else ("x",)), "w", "scales", "biases"],
        output_names = ["y"], header = header, source = _qmv_rows_source(prepared))


@functools.cache
def _qmv_rows_inputs_kernel():
    header = nax._mlx_kernel_header("quantized.h", "qdot")
    return header and mx.fast.metal_kernel(name = "unsloth_qmv_rows_inputs", input_names = ["x"],
                                           output_names = ["xs", "sums"], header = header, source = _QMV_ROWS_PREPARE)


def _qmv_rows_values(bits):  # qmv_fast's values per lane per K step, and the code products they make
    values = 16 if bits in (4, 5) else 8
    return values, values * 3 // 2 if bits in (5, 6) else values


def qmv_rows_supported(N, K, group_size, bits, mode):
    """Whether `qmv_rows` covers an `[N, K]` weight quantized this way: where stock runs qmv_fast for one row."""
    return (mode == "affine" and bits in _QMV_ROWS_BITS and group_size in _QMV_ROWS_GROUP_SIZES and N > 0
            and N % 8 == 0 and K > 0 and K % (32 * _qmv_rows_values(bits)[0]) == 0 and N * K * bits // 8 < 1 << 31)


def qmv_rows_inputs(x, bits):
    """Each row's prescaled inputs and input sums as qmv_fast forms them, for `qmv_rows` over several weights of x."""
    (M, K), (values, products) = x.shape, _qmv_rows_values(bits)
    return tuple(_qmv_rows_inputs_kernel()(
        inputs = [x], template = [("T", x.dtype), ("BITS", bits)],
        grid = (K // values, M, 1), threadgroup = (min(256, K // values), 1, 1),
        output_shapes = [(M, K // values * products), (M, K // values)], output_dtypes = [mx.float32, mx.float32]))


def qmv_rows(x, w, scales, biases, group_size, bits, inputs = None):
    """Each row of a 2-D x of at most `QMV_ROWS_MAX` rows as `quantized_matmul(row, w, ..., transpose=True)`;
    `inputs`, its `qmv_rows_inputs`, are read rather than formed per output."""
    M, N = x.shape[0], w.shape[0]
    return _qmv_rows_kernel(inputs is not None)(
        inputs = [*([x] if inputs is None else inputs), w, scales, biases],
        template = [("T", x.dtype), ("ROWS", M), ("GS", group_size), ("BITS", bits)],
        grid = (32, N // 4, 1), threadgroup = (32, next(g for g in (8, 4, 2) if N % (4 * g) == 0), 1),
        output_shapes = [(M, N)], output_dtypes = [x.dtype])[0]


@functools.cache
def compile_qmv_rows(dtype, group_size, bits):
    """Build `qmv_rows` both ways for every row count of `dtype` inputs and these weights, on a tiny weight."""
    K = 32 * _qmv_rows_values(bits)[0]
    w, scales, biases = mx.quantize(mx.zeros((8, K), dtype = dtype), group_size = group_size, bits = bits)
    x = mx.zeros((QMV_ROWS_MAX, K), dtype = dtype)
    mx.eval([qmv_rows(x[:M], w, scales, biases, group_size, bits, inputs) for M in range(1, QMV_ROWS_MAX + 1)
             for inputs in (None, qmv_rows_inputs(x[:M], bits))])


def probe_qmv_rows():
    """Subprocess probe: every variant the kernel compiles, each row bitwise its one-row stock call."""
    for dtype, group_size, bits in itertools.product((mx.bfloat16, mx.float16), _QMV_ROWS_GROUP_SIZES, _QMV_ROWS_BITS):
        w = mx.random.normal((72, 1024), key = mx.random.key(group_size)) * 0.05
        w, scales, biases = mx.quantize(w.astype(dtype), group_size = group_size, bits = bits)
        for M in range(1, QMV_ROWS_MAX + 1):
            x = mx.random.normal((M, 1024), key = mx.random.key(M)).astype(dtype)
            native = mx.concatenate([mx.quantized_matmul(x[i:i + 1], w, scales, biases, transpose = True,
                                                         group_size = group_size, bits = bits) for i in range(M)])
            for inputs in (None, qmv_rows_inputs(x, bits)):
                got = qmv_rows(x, w, scales, biases, group_size, bits, inputs)
                assert mx.array_equal(got.view(mx.uint16), native.view(mx.uint16)).item(), (dtype, group_size, bits, M)


def qmv_rows_launch_costs(dtype, group_size, bits, clock = time.perf_counter):
    # A weight of 64-128 MB and four launches per timed run, so each run is well above timer and launch noise.
    w = mx.quantize(mx.zeros((16384, 8192), dtype = dtype), group_size = group_size, bits = bits)
    x = mx.zeros((QMV_ROWS_MAX, 8192), dtype = dtype)
    launch = lambda m, prepared: [qmv_rows(x[:m], *w, group_size, bits,
                                               qmv_rows_inputs(x[:m], bits) if prepared else None) for _ in range(4)]
    launches = [(m, prepared) for m in range(1, QMV_ROWS_MAX + 1) for prepared in (False, True)]
    mx.eval(w, [launch(*plan) for plan in launches])
    seconds = {plan: [] for plan in launches}
    for _ in range(3):
        for plan in launches:
            mx.synchronize()
            start = clock()
            mx.eval(launch(*plan))
            seconds[plan].append((clock() - start) / 4)
    return {plan: sorted(spent)[1] for plan, spent in seconds.items()}


def qmv_rows_sizes(rows, most):
    count = -(-rows // most)
    size, extra = divmod(rows, count)
    return [size + (i < extra) for i in range(count)]


@functools.cache
def qmv_rows_plan(costs, rows):
    """The (rows per launch at most, inputs prepared) whose balanced launches of `rows` rows cost least by `costs`."""
    costs = dict(costs)
    return min(((most, prepared) for most in range(1, QMV_ROWS_MAX + 1) for prepared in (False, True)),
               key = lambda plan: sum(costs[(m, plan[1])] for m in qmv_rows_sizes(rows, plan[0])))
