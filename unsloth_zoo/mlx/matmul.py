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

"""Affine `x @ W^T` on stock packed weights: small-row 4/8-bit qmm for Apple GPUs without stock NAX kernels,
multi-row qmv whose rows match one-row stock calls bit for bit, and row-invariant lane qmm."""

import functools
import hashlib
import itertools
import struct
import time

import mlx.core as mx

from . import nax


# From 6 rows, simdgroup matrices run on exact `offset + q` operands; each group folds `s * (P - offset * sum(x)) + b * sum(x)` in fp32, as stock qmv does.
_SIMD_QMM_BITS = (4, 8)
_SIMD_QMM_GROUP_SIZES = (32, 64, 128)
_SIMD_QMM_MAX_ROWS = 24
_SIMD_QMM_SIMDGROUPS = 8   # 256 threads: M2 and M3 GPUs cap a threadgroup at 896
# Tuple is (operand, bits type, code shift, offset bits (both halves for 16-bit operands), offset); bf16 cannot hold `256 + q`, so 8-bit codes multiply as fp32.
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


# Rows of affine `x @ W^T` use stock qmv_fast's one-row arithmetic, so each row equals its one-row call bit for bit;
# each weight is read once per launch, where mlx-vlm's exact verifier re-reads or re-decodes it per row.
QMV_ROWS_MAX = 8  # rows a simdgroup holds; a staged launch runs up to twice as many in two row groups
QMV_ROWS_STAGED_MAX = 16
_QMV_ROWS_BITS = (4, 5, 6, 8)
_QMV_ROWS_GROUP_SIZES = (32, 64, 128)

# A row's inputs as the installed MLX's `load_vector` forms them, and `xt`, each code product's factor.
_QMV_ROWS_INPUTS = """
    float sum = 0;
    float xt[NP];
    if (BITS == 4) {  // subnormal half codes take x unscaled
        _Pragma("clang loop unroll(full)")
        for (int i = 0; i < VPT; i += 4) {
            sum += float(xr[i]) + float(xr[i + 1]) + float(xr[i + 2]) + float(xr[i + 3]);
            xt[i] = xr[i]; xt[i + 1] = xr[i + 1]; xt[i + 2] = xr[i + 2]; xt[i + 3] = xr[i + 3];
        }
    } else {
        float x_thread[VPT];
        sum = load_vector<T, float, VPT, BITS>(xr, x_thread);
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
    }
"""

# Multiply-add bound on smaller GPUs: a lane decodes its codes once for all rows (half codes are exact) and forms each
# row's inputs per four outputs, read from `qmv_rows_inputs` or staged per threadgroup; a row count compiles once for
# all N and K.
_QMV_ROWS_SOURCE = """
    constexpr int VPT = (BITS == 4 || BITS == 5) ? 16 : 8;
    constexpr int NP = BITS == 5 ? 24 : (BITS == 6 ? 12 : VPT);
    constexpr int LANE_BYTES = VPT * BITS / 8;
    const int K = w_shape[w_ndim - 1] * 32 / BITS;
    const int N = w_shape[w_ndim - 2];
    const int ROW_BYTES = K * BITS / 8;
    const int KG = K / GS;
    const uint lane = thread_index_in_simdgroup;
    const int SG = int(threads_per_threadgroup.y), group = int(thread_position_in_grid.y) / SG % GROUPS;
    const int n0 = (int(thread_position_in_grid.y) / SG / GROUPS * SG + int(thread_position_in_grid.y) % SG) * 4;
    const int row0 = group * ROWS;
    const device uint8_t* ws = (const device uint8_t*)w + n0 * ROW_BYTES + int(lane) * LANE_BYTES;
    const device T* sc = scales + n0 * KG + int(lane) / (GS / VPT);
    const device T* bs = biases + n0 * KG + int(lane) / (GS / VPT);
    float result[ROWS][4];
    _Pragma("clang loop unroll(full)")
    for (int t = 0; t < ROWS; ++t) _Pragma("clang loop unroll(full)") for (int r = 0; r < 4; ++r) result[t][r] = 0;
    ROW_SHARED
    for (int k = 0; k < K; k += 32 * VPT) {
        ROW_STAGE
        half code[4][NP];
        float s[4], b[4];
        _Pragma("clang loop unroll(full)")
        for (int r = 0; r < 4; ++r) {
            const device uint8_t* wl = ws + r * ROW_BYTES;
            if (BITS == 4) {
                _Pragma("clang loop unroll(full)")
                for (int p = 0; p < 2; ++p) {
                    // q * 2^-24 with no integer conversion; products are stock's times 2^-24 unless they underflow fp32.
                    const uint v = ((const device uint*)wl)[p];
                    _Pragma("clang loop unroll(full)")
                    for (int m = 0; m < 4; ++m) {
                        const half2 c = as_type<half2>((v >> (4 * m)) & 0x000F000Fu);
                        code[r][8 * p + m] = c.x;
                        code[r][8 * p + 4 + m] = c.y;
                    }
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
            s[r] = BITS == 4 ? float(sc[r * KG]) * 16777216.0f : float(sc[r * KG]);
            b[r] = bs[r * KG];
        }
        _Pragma("clang loop unroll(full)")
        for (int t = 0; t < ROWS; ++t) {
            const int row = min(row0 + t, M - 1);
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
            if (lane == 0 && row0 + t < M) y[(row0 + t) * N + n0 + r] = T(v);
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


_QMV_ROWS_SHARED = """threadgroup float4 staged[ROWS * NP / 4 * 32];
    threadgroup float staged_sums[ROWS * 32];"""
_QMV_ROWS_STAGE = """threadgroup_barrier(mem_flags::mem_threadgroup);
        const int threads = 32 * int(threads_per_threadgroup.y);
        for (int chunk = int(thread_index_in_threadgroup); chunk < ROWS * 32; chunk += threads) {
            const int t = chunk / 32;
            const device T* xr = x + min(row0 + t, M - 1) * K + k + (chunk % 32) * VPT;""" + _QMV_ROWS_INPUTS + """
            _Pragma("clang loop unroll(full)")
            for (int j = 0; j < NP; j += 4)
                staged[(t * NP + j) * 8 + chunk % 32] = float4(xt[j], xt[j + 1], xt[j + 2], xt[j + 3]);
            staged_sums[chunk] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);"""
_QMV_ROWS_STAGED_READ = """const float sum = staged_sums[t * 32 + int(lane)];
            float xt[NP];
            _Pragma("clang loop unroll(full)")
            for (int j = 0; j < NP; j += 4) {
                const float4 v = staged[(t * NP + j) * 8 + int(lane)];
                xt[j] = v.x; xt[j + 1] = v.y; xt[j + 2] = v.z; xt[j + 3] = v.w;
            }"""
QMV_ROWS_KINDS = ("inline", "prepared", "staged")


def _qmv_rows_source(kind):
    inputs = {"inline": "const device T* xr = x + row * K + k + int(lane) * VPT;" + _QMV_ROWS_INPUTS,
              "prepared": _QMV_ROWS_READ, "staged": _QMV_ROWS_STAGED_READ}[kind]
    staged = kind == "staged"
    return (_QMV_ROWS_SOURCE.replace("ROW_INPUTS", inputs).replace("ROW_SHARED", _QMV_ROWS_SHARED if staged else "")
            .replace("ROW_STAGE", _QMV_ROWS_STAGE if staged else ""))


QMV_ROWS_PROBE_KEY = "qmv_rows:" + hashlib.sha256((_QMV_ROWS_SOURCE + _QMV_ROWS_INPUTS + _QMV_ROWS_READ
                                                   + _QMV_ROWS_PREPARE + _QMV_ROWS_SHARED + _QMV_ROWS_STAGE
                                                   + _QMV_ROWS_STAGED_READ).encode()).hexdigest()[:12]


@functools.cache
def _qmv_rows_kernel(kind):
    header = nax._mlx_kernel_header("quantized.h", "qdot")
    return header and mx.fast.metal_kernel(
        name = f"unsloth_qmv_rows{'' if kind == 'inline' else '_' + kind}",
        input_names = [*(("xs", "sums") if kind == "prepared" else ("x",)), "w", "scales", "biases"],
        output_names = ["y"], header = header, source = _qmv_rows_source(kind))


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


def qmv_rows(x, w, scales, biases, group_size, bits, inputs = None, staged = False):
    """Each row of a 2-D x as `quantized_matmul(row, w, ..., transpose=True)`, x having at most `qmv_rows_most(kind)`
    rows; `inputs`, its `qmv_rows_inputs`, are read rather than formed per output, and `staged` forms them once per
    threadgroup instead."""
    M, N = x.shape[0], w.shape[0]
    groups = -(-M // QMV_ROWS_MAX)
    kind = "prepared" if inputs is not None else "staged" if staged else "inline"
    simdgroups = next(g for g in ((4, 2) if staged else (8, 4, 2)) if N % (4 * g) == 0)
    return _qmv_rows_kernel(kind)(
        inputs = [*([x] if inputs is None else inputs), w, scales, biases],
        template = [("T", x.dtype), ("M", M), ("ROWS", -(-M // groups)), ("GROUPS", groups), ("GS", group_size),
                    ("BITS", bits)],
        grid = (32, N // 4 * groups, 1), threadgroup = (32, simdgroups, 1),
        output_shapes = [(M, N)], output_dtypes = [x.dtype])[0]


def qmv_rows_most(kind):
    return QMV_ROWS_STAGED_MAX if kind == "staged" else QMV_ROWS_MAX


def qmv_rows_args(x, bits, kind):
    return qmv_rows_inputs(x, bits) if kind == "prepared" else None, kind == "staged"


@functools.cache
def compile_qmv_rows(dtype, group_size, bits):
    """Build every kind of `qmv_rows` for every row count of `dtype` inputs and these weights, on a tiny weight."""
    K = 32 * _qmv_rows_values(bits)[0]
    w, scales, biases = mx.quantize(mx.zeros((8, K), dtype = dtype), group_size = group_size, bits = bits)
    x = mx.zeros((QMV_ROWS_STAGED_MAX, K), dtype = dtype)
    mx.eval([qmv_rows(x[:M], w, scales, biases, group_size, bits, *qmv_rows_args(x[:M], bits, kind))
             for kind in QMV_ROWS_KINDS for M in range(1, qmv_rows_most(kind) + 1)])


def qmv_rows_matches(x, w, scales, biases, group_size, bits):
    """Whether every kind of `qmv_rows` that takes x's rows gives each row bit for bit its one-row stock call."""
    native = mx.concatenate([mx.quantized_matmul(x[i:i + 1], w, scales, biases, transpose = True,
                                                 group_size = group_size, bits = bits) for i in range(x.shape[0])])
    return all(mx.array_equal(qmv_rows(x, w, scales, biases, group_size, bits, *qmv_rows_args(x, bits, kind))
                              .view(mx.uint16), native.view(mx.uint16)).item()
               for kind in QMV_ROWS_KINDS if x.shape[0] <= qmv_rows_most(kind))


_QMV_ROWS_DTYPES = {"bfloat16": mx.bfloat16, "float16": mx.float16}


def qmv_rows_probe(dtype, group_size, bits):
    """The (probe key, `probe_qmv_rows_*` name) of one kernel: probing all 24 at once outlasts the probe timeout on M1."""
    name = f"probe_qmv_rows_{next(k for k, v in _QMV_ROWS_DTYPES.items() if v == dtype)}_{group_size}_{bits}"
    return f"{QMV_ROWS_PROBE_KEY}:{name[len('probe_qmv_rows_'):]}", name


def qmv_rows_probe_passed(dtype, group_size, bits):
    key, function = qmv_rows_probe(dtype, group_size, bits)
    return nax.kernel_probe_passed(key, __name__, function, mx.metal.is_available)


def _probe_qmv_rows(dtype, group_size, bits):
    """Subprocess probe: every variant of one kernel, each row bitwise its one-row stock call."""
    # 72 and 80 outputs launch two and four simdgroups per staged threadgroup.
    for N in (72, 80):
        w = mx.random.normal((N, 1024), key = mx.random.key(group_size)) * 0.05
        w, scales, biases = mx.quantize(w.astype(dtype), group_size = group_size, bits = bits)
        for M in range(1, QMV_ROWS_STAGED_MAX + 1):
            x = mx.random.normal((M, 1024), key = mx.random.key(M)).astype(dtype)
            assert qmv_rows_matches(x, w, scales, biases, group_size, bits), (dtype, group_size, bits, N, M)


def probe_qmv_rows():
    for dtype, group_size, bits in itertools.product(_QMV_ROWS_DTYPES.values(), _QMV_ROWS_GROUP_SIZES, _QMV_ROWS_BITS):
        _probe_qmv_rows(dtype, group_size, bits)


def __getattr__(name):  # `probe_qmv_rows_<dtype>_<group size>_<bits>`, the names `qmv_rows_probe` gives the probe runner
    dtype, _, rest = name.removeprefix("probe_qmv_rows_").partition("_")
    group_size, _, bits = rest.partition("_")
    if (name.startswith("probe_qmv_rows_") and dtype in _QMV_ROWS_DTYPES and group_size.isdigit() and bits.isdigit()
            and int(group_size) in _QMV_ROWS_GROUP_SIZES and int(bits) in _QMV_ROWS_BITS):
        return functools.partial(_probe_qmv_rows, _QMV_ROWS_DTYPES[dtype], int(group_size), int(bits))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def qmv_rows_launch_costs(dtype, group_size, bits, clock = time.perf_counter):
    # A weight of 64-128 MB and four launches per timed run, so each run is well above timer and launch noise.
    w = mx.quantize(mx.zeros((16384, 8192), dtype = dtype), group_size = group_size, bits = bits)
    x = mx.zeros((QMV_ROWS_STAGED_MAX, 8192), dtype = dtype)
    launch = lambda m, kind: [qmv_rows(x[:m], *w, group_size, bits, *qmv_rows_args(x[:m], bits, kind))
                              for _ in range(4)]
    launches = [(m, kind) for kind in QMV_ROWS_KINDS for m in range(1, qmv_rows_most(kind) + 1)]
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
    """The (rows per launch at most, launch kind) whose balanced launches of `rows` rows cost least by `costs`."""
    costs = dict(costs)
    return min(sorted(costs), key = lambda plan: sum(costs[(m, plan[1])] for m in qmv_rows_sizes(rows, plan[0])))


# Row-invariant small-row affine `x @ W^T` for every Apple GPU and 2- to 8-bit codes, on simdgroup matrices
# or, on M5 for 4- and 8-bit codes, the tensor units: a row's bits do not depend on the call's other rows, and are within
# fp32 reordering of stock's products.
LANE_QMM_MAX_ROWS = 32
_LANE_BITS = (2, 3, 4, 5, 6, 8)
_LANE_GROUP_SIZES = (32, 64, 128)
_LANE_SIMDGROUPS = 8

# A lane holds elements (fm, fn) and (fm, fn + 1) of every 8 x 8 simdgroup matrix: codes at chunk positions
# j = fn, fn + 1 of its column, x at position j = fm of its rows fn, fn + 1. Each group folds its fp32 product
# `s * (P - OFFSET * sum(x)) + b * sum(x)` into the row's output.
_LANE_SIMD_ONE = """
    const int N = w_shape[0];
    const uint n0 = threadgroup_position_in_grid.y * (8 * CT);
"""
_LANE_SIMD_PROLOGUE = """
    constexpr int G = K / GS;
    constexpr int C8 = GS / 8;
    constexpr bool WIDE = sizeof(O) == 4;
    const int M = x_shape[0];
    const uint simd = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint fm = ((lane >> 2) & 4) | ((lane >> 1) & 3);
    const uint fn = ((lane >> 1) & 4) | ((lane & 1) << 1);
    const uint nb = n0;
    const int mb = int(threadgroup_position_in_grid.z) * (8 * RT);
    const uint g0 = simd * RG / NSG, g1 = (simd + 1) * RG / NSG;
    float2 out[RT][CT];
    _Pragma("unroll") for (int r = 0; r < RT; ++r) _Pragma("unroll") for (int c = 0; c < CT; ++c) out[r][c] = float2(0.0f);
"""

# Accumulates the operand `offset + q`, which is exact: the code shifted under the offset's exponent.
_LANE_SIMD_MMA = """
                    simdgroup_matrix<O, 8, 8> a;
                    if constexpr (WIDE) {
                        a.thread_elements()[0] = as_type<O>(OB(((p & 0xFFFFu) << SHIFT) | OFFSET_BITS));
                        a.thread_elements()[1] = as_type<O>(OB(((p >> 16) << SHIFT) | OFFSET_BITS));
                    } else {
                        const uint both = (p << SHIFT) | OFFSET_BITS;
                        a.thread_elements()[0] = as_type<O>(OB(both));
                        a.thread_elements()[1] = as_type<O>(OB(both >> 16));
                    }
                    _Pragma("unroll") for (int r = 0; r < RT; ++r) {
                        simdgroup_matrix<O, 8, 8> b;
                        b.thread_elements()[0] = O(xa[r][i]);
                        b.thread_elements()[1] = O(xb[r][i]);
                        simdgroup_multiply_accumulate(acc[r][c], a, b, acc[r][c]);
                    }
"""

_LANE_SIMD_FOLD = """
        _Pragma("unroll") for (int r = 0; r < RT; ++r) {   // the lanes sharing fn differ in the fm bits of the lane index
            sum[r] += float2(simd_shuffle_xor(sum[r].x, 2), simd_shuffle_xor(sum[r].y, 2));
            sum[r] += float2(simd_shuffle_xor(sum[r].x, 4), simd_shuffle_xor(sum[r].y, 4));
            sum[r] += float2(simd_shuffle_xor(sum[r].x, 16), simd_shuffle_xor(sum[r].y, 16));
            _Pragma("unroll") for (int c = 0; c < CT; ++c) {
                const float px = fma(-OFFSET, sum[r].x, acc[r][c].thread_elements()[0]);
                const float py = fma(-OFFSET, sum[r].y, acc[r][c].thread_elements()[1]);
                out[r][c].x = fma(s[c], px, fma(bias[c], sum[r].x, out[r][c].x));
                out[r][c].y = fma(s[c], py, fma(bias[c], sum[r].y, out[r][c].y));
            }
        }
"""

# Any width: chunk t's position j is k = j * GS / 8 + t, so a lane's codes are one span of GS / 4 codes of its column
# and its x one run of GS / 8 values of each row.
_LANE_GENERIC_LOOP = """
    constexpr uint MASK = (1u << BITS) - 1u;
    constexpr int SPAN_BITS = GS / 4 * BITS;
    constexpr int WORDS = (SPAN_BITS + 31) / 32 + 1;
    constexpr ulong ROW_BYTES = ulong(K) * BITS / 8;
    const device T* rows[RT][2];
    _Pragma("unroll") for (int r = 0; r < RT; ++r) {   // rows past M read the last one and are never stored
        rows[r][0] = x + ulong(min(mb + r * 8 + int(fn), M - 1)) * K + fm * C8;
        rows[r][1] = x + ulong(min(mb + r * 8 + int(fn) + 1, M - 1)) * K + fm * C8;
    }
    const device uchar* cols[CT];
    _Pragma("unroll") for (int c = 0; c < CT; ++c) cols[c] = (const device uchar*)w + ulong(nb + c * 8 + fm) * ROW_BYTES;

    for (uint g = g0; g < g1; ++g) {
        uint pair[CT][C8];   // chunk t's two codes of a column: position fn low, fn + 1 high
        float s[CT], bias[CT];
        _Pragma("unroll") for (int c = 0; c < CT; ++c) {
            uint word[WORDS];
            const device U* src = (const device U*)(cols[c] + (ulong(g) * GS + fn * C8) * BITS / 8);
            _Pragma("unroll") for (int i = 0; i < WORDS; ++i) word[i] = 0u;
            _Pragma("unroll") for (int i = 0; i < SPAN_BITS / (8 * int(sizeof(U))); ++i) {
                const int bit = i * 8 * int(sizeof(U));
                word[bit >> 5] |= uint(src[i]) << (bit & 31);
            }
            _Pragma("unroll") for (int t = 0; t < C8; ++t) {
                uint q[2];
                _Pragma("unroll") for (int e = 0; e < 2; ++e) {
                    const int bit = (e * C8 + t) * BITS;
                    q[e] = word[bit >> 5] >> (bit & 31);
                    if ((bit & 31) + BITS > 32) q[e] |= word[(bit >> 5) + 1] << (32 - (bit & 31));
                    q[e] &= MASK;
                }
                pair[c][t] = q[0] | (q[1] << 16);
            }
            s[c] = float(scales[ulong(nb + c * 8 + fm) * G + g]);
            bias[c] = float(biases[ulong(nb + c * 8 + fm) * G + g]);
        }
        simdgroup_matrix<float, 8, 8> acc[RT][CT];
        float2 sum[RT];
        _Pragma("unroll") for (int r = 0; r < RT; ++r) {
            sum[r] = float2(0.0f);
            _Pragma("unroll") for (int c = 0; c < CT; ++c) acc[r][c] = simdgroup_matrix<float, 8, 8>(0.0f);
        }
        _Pragma("unroll") for (int t0 = 0; t0 < C8; t0 += 4) {
            vec<T, 4> xa[RT], xb[RT];
            _Pragma("unroll") for (int r = 0; r < RT; ++r) {
                xa[r] = *(const device vec<T, 4>*)(rows[r][0] + g * GS + t0);
                xb[r] = *(const device vec<T, 4>*)(rows[r][1] + g * GS + t0);
                _Pragma("unroll") for (int i = 0; i < 4; ++i) sum[r] += float2(float(xa[r][i]), float(xb[r][i]));
            }
            _Pragma("unroll") for (int i = 0; i < 4; ++i) {
                _Pragma("unroll") for (int c = 0; c < CT; ++c) {
                    const uint p = pair[c][t0 + i];
""" + _LANE_SIMD_MMA + """
                }
            }
        }
""" + _LANE_SIMD_FOLD + """
    }
"""

# 2- and 4-bit codes: one shift and mask of a word gives its codes i and i + H, the operands of j = fn and
# fn + 1, so chunk t's position j is k = (j / 2) * GS / 4 + CPW * (t / H) + t % H + H * (j % 2) for both operands.
_LANE_PAIRED_LOOP = """
    constexpr int CPW = 32 / BITS;
    constexpr int H = CPW / 2;
    constexpr int SPAN = GS / 4;
    constexpr int WN = SPAN / CPW;
    constexpr uint MASK2 = ((1u << BITS) - 1u) * 0x10001u;
    const uint xoff = (fm >> 1) * SPAN + H * (fm & 1);
    const device T* rows[RT][2];
    _Pragma("unroll") for (int r = 0; r < RT; ++r) {   // rows past M read the last one and are never stored
        rows[r][0] = x + ulong(min(mb + r * 8 + int(fn), M - 1)) * K + xoff;
        rows[r][1] = x + ulong(min(mb + r * 8 + int(fn) + 1, M - 1)) * K + xoff;
    }
    const device uint* cols[CT];
    _Pragma("unroll") for (int c = 0; c < CT; ++c)
        cols[c] = (const device uint*)w + ulong(nb + c * 8 + fm) * (K / CPW) + (fn >> 1) * WN;

    for (uint g = g0; g < g1; ++g) {
        uint word[CT][WN];
        float s[CT], bias[CT];
        _Pragma("unroll") for (int c = 0; c < CT; ++c) {
            const device uint* src = cols[c] + ulong(g) * (GS / CPW);
            if (WN == 4) { const uint4 v = *(const device uint4*)src; word[c][0] = v.x; word[c][1] = v.y; word[c][2] = v.z; word[c][3] = v.w; }
            else if (WN == 2) { const uint2 v = *(const device uint2*)src; word[c][0] = v.x; word[c][1] = v.y; }
            else _Pragma("unroll") for (int i = 0; i < WN; ++i) word[c][i] = src[i];
            s[c] = float(scales[ulong(nb + c * 8 + fm) * G + g]);
            bias[c] = float(biases[ulong(nb + c * 8 + fm) * G + g]);
        }
        simdgroup_matrix<float, 8, 8> acc[RT][CT];
        float2 sum[RT];
        _Pragma("unroll") for (int r = 0; r < RT; ++r) {
            sum[r] = float2(0.0f);
            _Pragma("unroll") for (int c = 0; c < CT; ++c) acc[r][c] = simdgroup_matrix<float, 8, 8>(0.0f);
        }
        _Pragma("unroll") for (int wi = 0; wi < WN; ++wi) {
            vec<T, H> xa[RT], xb[RT];
            _Pragma("unroll") for (int r = 0; r < RT; ++r) {
                xa[r] = *(const device vec<T, H>*)(rows[r][0] + g * GS + CPW * wi);
                xb[r] = *(const device vec<T, H>*)(rows[r][1] + g * GS + CPW * wi);
                _Pragma("unroll") for (int i = 0; i < H; ++i) sum[r] += float2(float(xa[r][i]), float(xb[r][i]));
            }
            _Pragma("unroll") for (int i = 0; i < H; ++i) {
                _Pragma("unroll") for (int c = 0; c < CT; ++c) {
                    const uint p = (word[c][wi] >> (BITS * i)) & MASK2;
""" + _LANE_SIMD_MMA + """
                }
            }
        }
""" + _LANE_SIMD_FOLD + """
    }
"""

_LANE_SIMD_EPILOGUE = """
    if (NSG > 1) {
        threadgroup float2 partials[RT * CT * NSG * 32];
        _Pragma("unroll") for (int i = 0; i < RT * CT; ++i) partials[(i * NSG + simd) * 32 + lane] = out[i / CT][i % CT];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd != 0) return;
        _Pragma("unroll") for (int i = 0; i < RT * CT; ++i) {
            float2 total = partials[i * NSG * 32 + lane];
            for (int p = 1; p < NSG; ++p) total += partials[(i * NSG + p) * 32 + lane];
            out[i / CT][i % CT] = total;
        }
    }
    _Pragma("unroll") for (int r = 0; r < RT; ++r) {
        const int m = mb + r * 8 + int(fn);
        _Pragma("unroll") for (int c = 0; c < CT; ++c) {
            if (m < M) y[ulong(m) * N + nb + c * 8 + fm] = Y(out[r][c].x);
            if (m + 1 < M) y[ulong(m + 1) * N + nb + c * 8 + fm] = Y(out[r][c].y);
        }
    }
"""

# On the tensor units each simdgroup runs one 16 x 32 x GS matmul per group and 16-row tile, reading 4- and 8-bit codes
# in place, and folds by the destination's layout: lane l holds rows r0, r0 + 8 of columns c0 + 0..3 and c0 + 16..19,
# element i at column c0 + i % 4 + 16 * (i / 8) and row r0 + 8 * (i / 4 % 2). The readers differ in which simdgroup
# takes which group and how scales and biases are read; each fixes its order by the weight alone.
_LANE_TENSOR_ONE = """
    const int N = w_shape[0];
    const uint n0 = threadgroup_position_in_grid.y * 32;
"""
_LANE_TENSOR_HEAD = """
    constexpr int G = K / GS;
    using W = metal::conditional_t<BITS == 4, uint4b_format, uchar>;
    const int M = x_shape[0];
    const uint simd = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint c0 = 4 * (lane & 1) + 8 * ((lane >> 3) & 1);
    const uint r0 = ((lane >> 1) & 3) + 4 * ((lane >> 4) & 1);
    // Dynamic row extents keep each tile's reads within M.
    auto a0 = tensor((device T*)x, dextents<int, 2>{K, min(M, 16)}, array<int, 2>{1, K});
    auto a1 = tensor((device T*)x + (RT > 1 ? 16 * K : 0), dextents<int, 2>{K, RT > 1 ? max(M - 16, 1) : 1},
                     array<int, 2>{1, K});
    tensor<device W, dextents<int, 2>, tensor_inline> b(
        (device uchar*)w + ulong(n0) * K * BITS / 8, dextents<int, 2>{K, 32}, array<int, 2>{1, K});
    constexpr auto desc = matmul2d_descriptor(16, 32, GS, false, true, false);
    matmul2d<desc, execution_simdgroup> op;
    auto a_first = a0.slice(0, 0);
    auto b_first = b.template slice<GS, 32>(0, 0);
    using P = decltype(op.template get_destination_cooperative_tensor<decltype(a_first), decltype(b_first), float>());
    P p;
    float out[RT][16];
    for (int t = 0; t < RT; ++t) for (int i = 0; i < 16; ++i) out[t][i] = 0.0f;
    decltype(sums) xs[RT][2];   // rows r0 and r0 + 8 of each tile
    for (int t = 0; t < RT; ++t)
        for (int h = 0; h < 2; ++h) xs[t][h] = sums + ulong(min(16 * t + int(r0) + 8 * h, M - 1)) * G;
"""
_LANE_TENSOR_FOLD = """
            for (int t = 0; t < RT; ++t) {
                if (t == 0) { auto as = a0.slice(int(g * GS), 0); op.run(as, bs, p); }
                else { auto as = a1.slice(int(g * GS), 0); op.run(as, bs, p); }
                for (int h = 0; h < H; ++h) {
                    if (16 * t + int(r0) + 8 * h >= M) continue;
                    const float sum = xs[t][h][g];
                    for (int k = 0; k < 8; ++k) {
                        const int i = (k & 3) + 4 * h + 8 * (k >> 2);
                        out[t][i] = fma(sv[k], p[i], fma(cv[k], sum, out[t][i]));
                    }
                }
            }
"""
# A K slice per simdgroup or, interleaved, groups dealt round-robin, so a threadgroup's simdgroups read adjacent
# groups of its columns at once.
_LANE_SLICES_LOOP = """
    const device T* sp = scales + ulong(n0 + c0) * G;
    const device T* bp = biases + ulong(n0 + c0) * G;
    threadgroup float partials[(NSG > 1 ? NSG - 1 : 1) * 16 * 32];
    const uint g0 = ILV ? simd : simd * RG / NSG, g1 = ILV ? uint(RG) : (simd + 1) * RG / NSG, step = ILV ? NSG : 1;
    for (uint g = g0; g < g1; g += step) {
        float sv[8], cv[8];
        auto bs = b.template slice<GS, 32>(int(g * GS), 0);
        for (int j = 0; j < 8; ++j) {
            const uint col = (j & 3) + 16 * (j >> 2);
            sv[j] = float(sp[col * G + g]);
            cv[j] = float(bp[col * G + g]);
        }
""" + _LANE_TENSOR_FOLD + """
    }
"""
# Groups dealt round-robin; each round of CG groups has its 32 columns' scales and biases loaded one round ahead by the
# whole threadgroup, four groups a load, and staged by group in double-buffered threadgroup memory.
_LANE_STAGED_LOOP = """
    constexpr int THREADS = 32 * NSG;
    constexpr int LOADS = (32 * CG / 4 + THREADS - 1) / THREADS;
    constexpr int ROUNDS = (RG + CG - 1) / CG;
    constexpr int BUF = 2 * 32 * CG;   // scales then biases, [group][column]
    constexpr int STAGE = 2 * BUF * int(sizeof(T)), PART = (NSG > 1 ? NSG - 1 : 1) * 16 * 32 * 4;
    threadgroup uint4 shared[((STAGE > PART ? STAGE : PART) + 15) / 16];
    threadgroup T* stage = (threadgroup T*)shared;
    threadgroup float* partials = (threadgroup float*)shared;
    const uint tid = simd * 32 + lane;
    const device T* srow = scales + ulong(n0) * G;
    const device T* brow = biases + ulong(n0) * G;
    vec<T, 4> rs[LOADS], rb[LOADS];
    auto load = [&](int c) {
        for (int i = 0; i < LOADS; ++i) {
            const int v = int(tid) + i * THREADS, r = v / (CG / 4), g = c * CG + v % (CG / 4) * 4;
            if (v < 32 * CG / 4 && g < RG) {
                rs[i] = *(const device vec<T, 4>*)(srow + ulong(r) * G + g);
                rb[i] = *(const device vec<T, 4>*)(brow + ulong(r) * G + g);
            }
        }
    };
    auto store = [&](int buf) {
        for (int i = 0; i < LOADS; ++i) {
            const int v = int(tid) + i * THREADS, r = v / (CG / 4), j = v % (CG / 4) * 4;
            if (v < 32 * CG / 4)
                for (int e = 0; e < 4; ++e) { stage[buf * BUF + (j + e) * 32 + r] = rs[i][e]; stage[buf * BUF + 32 * CG + (j + e) * 32 + r] = rb[i][e]; }
        }
    };
    load(0);
    for (int c = 0; c < ROUNDS; ++c) {
        const int buf = c & 1;
        store(buf);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (c + 1 < ROUNDS) load(c + 1);
        const threadgroup T* tail = stage + buf * BUF;
        for (int j = int(simd); j < CG; j += NSG) {
            const uint g = uint(c * CG + j);
            if (g >= uint(RG)) break;
            float sv[8], cv[8];
            auto bs = b.template slice<GS, 32>(int(g * GS), 0);
            for (int h = 0; h < 2; ++h) {
                const vec<T, 4> s4 = *(const threadgroup vec<T, 4>*)(tail + j * 32 + c0 + 16 * h);
                const vec<T, 4> b4 = *(const threadgroup vec<T, 4>*)(tail + 32 * CG + j * 32 + c0 + 16 * h);
                for (int q = 0; q < 4; ++q) { sv[4 * h + q] = float(s4[q]); cv[4 * h + q] = float(b4[q]); }
            }
""" + _LANE_TENSOR_FOLD + """
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);   // the partials reuse the staging
"""
# Groups dealt round-robin, folding only their scales; the bias term sum_g b[n][g] * sum[m][g] runs on the tensor units as
# one matmul over the groups, reading the biases in place, with each sum split into T's high and low halves.
_LANE_BIAS_LOOP = """
    const device T* sp = scales + ulong(n0 + c0) * G;
    threadgroup float partials[(NSG > 1 ? NSG - 1 : 1) * 16 * 32];
    for (uint g = simd; g < uint(RG); g += NSG) {
        float sv[8];
        auto bs = b.template slice<GS, 32>(int(g * GS), 0);
        for (int j = 0; j < 8; ++j) sv[j] = float(sp[((j & 3) + 16 * (j >> 2)) * G + g]);
        for (int t = 0; t < RT; ++t) {
            if (t == 0) { auto as = a0.slice(int(g * GS), 0); op.run(as, bs, p); }
            else { auto as = a1.slice(int(g * GS), 0); op.run(as, bs, p); }
            for (int h = 0; h < H; ++h) {
                if (16 * t + int(r0) + 8 * h >= M) continue;
                for (int k = 0; k < 8; ++k) {
                    const int i = (k & 3) + 4 * h + 8 * (k >> 2);
                    out[t][i] = fma(sv[k], p[i], out[t][i]);
                }
            }
        }
    }
    {
        tensor<device T, dextents<int, 2>, tensor_inline> bb((device T*)biases + ulong(n0) * G, dextents<int, 2>{G, 32}, array<int, 2>{1, G});
        auto h0 = tensor((device T*)sums_hi, dextents<int, 2>{G, min(M, 16)}, array<int, 2>{1, G});
        auto l0 = tensor((device T*)sums_lo, dextents<int, 2>{G, min(M, 16)}, array<int, 2>{1, G});
        auto h1 = tensor((device T*)sums_hi + (RT > 1 ? 16 * G : 0), dextents<int, 2>{G, RT > 1 ? max(M - 16, 1) : 1}, array<int, 2>{1, G});
        auto l1 = tensor((device T*)sums_lo + (RT > 1 ? 16 * G : 0), dextents<int, 2>{G, RT > 1 ? max(M - 16, 1) : 1}, array<int, 2>{1, G});
        constexpr auto bdesc = matmul2d_descriptor(16, 32, 16, false, true, false);
        matmul2d<bdesc, execution_simdgroup> bop;
        auto bf = bb.template slice<16, 32>(0, 0);
        auto hf = h0.slice(0, 0);
        using PB = decltype(bop.template get_destination_cooperative_tensor<decltype(hf), decltype(bf), float>());
        PB pb;
        for (uint c = simd; c < uint(RG / 16); c += NSG) {
            auto bs = bb.template slice<16, 32>(int(c * 16), 0);
            for (int t = 0; t < RT; ++t) {
                for (int part = 0; part < 2; ++part) {
                    if (t == 0) { auto as = part ? l0.slice(int(c * 16), 0) : h0.slice(int(c * 16), 0); bop.run(as, bs, pb); }
                    else { auto as = part ? l1.slice(int(c * 16), 0) : h1.slice(int(c * 16), 0); bop.run(as, bs, pb); }
                    for (int i = 0; i < 16; ++i) out[t][i] += pb[i];
                }
            }
        }
    }
    if (RG % 16 && simd == NSG - 1) {   // groups past the last whole 16: their bias term per lane, in order
        const device T* bq = biases + ulong(n0 + c0) * G;
        for (uint g = RG / 16 * 16; g < uint(RG); ++g)
            for (int t = 0; t < RT; ++t)
                for (int h = 0; h < H; ++h) {
                    if (16 * t + int(r0) + 8 * h >= M) continue;
                    const float sum = xs[t][h][g];
                    for (int k = 0; k < 8; ++k) {
                        const int i = (k & 3) + 4 * h + 8 * (k >> 2);
                        out[t][i] = fma(float(bq[((k & 3) + 16 * (k >> 2)) * G + g]), sum, out[t][i]);
                    }
                }
    }
"""
_LANE_TENSOR_REDUCE = """
    if (NSG > 1) {
        for (int t = 0; t < RT; ++t) {
            if (t) threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd > 0) for (int i = 0; i < 16; ++i) partials[((simd - 1) * 16 + i) * 32 + lane] = out[t][i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd == 0)
                for (int q = 0; q < NSG - 1; ++q) for (int i = 0; i < 16; ++i) out[t][i] += partials[(q * 16 + i) * 32 + lane];
        }
        if (simd != 0) return;
    }
"""
_LANE_TENSOR_TAIL = """    for (int t = 0; t < RT; ++t)
        for (int i = 0; i < 16; ++i) {
            const int m = 16 * t + int(r0) + 8 * ((i >> 2) & 1);
            if (m < M) y[ulong(m) * N + n0 + c0 + (i & 3) + 16 * (i >> 3)] = Y(out[t][i]);
        }
"""
# K slices as `slices` reads them, each simdgroup staging its 32 columns' scales and biases eight groups at a time: lane l
# loads column l's 16-byte run of eight 2-byte values and stores it by group, so the fold reads four columns a vector, and
# its partials hold only the row halves a call fills. The same operations in the same order as `slices`, so the same bits.
# Two stages per simdgroup, so one barrier per run orders a run's stores after every lane's reads of the run before last.
_LANE_SCALES_LOOP = """
    constexpr int E = 8 * H;
    constexpr int STAGE = NSG * 2 * 2 * 8 * 32 * int(sizeof(T)), PART = (NSG > 1 ? NSG - 1 : 1) * E * 32 * 4;
    threadgroup uint4 shared[((REUSE ? (STAGE > PART ? STAGE : PART) : STAGE + PART) + 15) / 16];
    threadgroup T (*stage)[2][2][8][32] = (threadgroup T (*)[2][2][8][32])shared;
    threadgroup float* partials = (threadgroup float*)shared + (REUSE ? 0 : STAGE / 4);
    const device uint4* scol = (const device uint4*)(scales + ulong(n0 + lane) * G);
    const device uint4* bcol = (const device uint4*)(biases + ulong(n0 + lane) * G);
    const uint g0 = simd * RG / NSG, g1 = (simd + 1) * RG / NSG;
    uint4 rs = scol[g0 / 8], rb = bcol[g0 / 8];
    for (uint g = g0; g < g1; ++g) {
        const uint buf = (g / 8) & 1;
        if (g == g0 || g % 8 == 0) {
            for (int j = 0; j < 8; ++j) {
                stage[simd][buf][0][j][lane] = as_type<T>(ushort(rs[j / 2] >> (16 * (j & 1))));
                stage[simd][buf][1][j][lane] = as_type<T>(ushort(rb[j / 2] >> (16 * (j & 1))));
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if ((g / 8 + 1) * 8 < g1) { rs = scol[g / 8 + 1]; rb = bcol[g / 8 + 1]; }
        }
        float sv[8], cv[8];
        auto bs = b.template slice<GS, 32>(int(g * GS), 0);
        for (int h = 0; h < 2; ++h) {
            const vec<T, 4> s4 = *(const threadgroup vec<T, 4>*)(&stage[simd][buf][0][g % 8][c0 + 16 * h]);
            const vec<T, 4> b4 = *(const threadgroup vec<T, 4>*)(&stage[simd][buf][1][g % 8][c0 + 16 * h]);
            for (int q = 0; q < 4; ++q) { sv[4 * h + q] = float(s4[q]); cv[4 * h + q] = float(b4[q]); }
        }
""" + _LANE_TENSOR_FOLD + """
    }
    if (NSG > 1) {
        if (REUSE) threadgroup_barrier(mem_flags::mem_threadgroup);   // the partials reuse the stages
        for (int t = 0; t < RT; ++t) {
            if (t) threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd > 0)
                for (int e = 0; e < E; ++e) partials[((simd - 1) * E + e) * 32 + lane] = out[t][H == 2 ? e : (e & 3) + 8 * (e >> 2)];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd == 0)
                for (int q = 0; q < NSG - 1; ++q)
                    for (int e = 0; e < E; ++e) out[t][H == 2 ? e : (e & 3) + 8 * (e >> 2)] += partials[(q * E + e) * 32 + lane];
        }
        if (simd != 0) return;
    }
"""
_LANE_TENSOR_LOOPS = {"slices": _LANE_SLICES_LOOP, "interleaved": _LANE_SLICES_LOOP, "staged": _LANE_STAGED_LOOP,
                      "bias": _LANE_BIAS_LOOP, "scales": _LANE_SCALES_LOOP}
# Each row's sum over each group, which every tensor-unit threadgroup would otherwise read again: two halves, each in
# order, then added.
_LANE_SUMS_SOURCE = """
    const uint g = thread_position_in_grid.x, m = thread_position_in_grid.y;
    if (g >= G) return;
    float half_sums[2];
    for (int h = 0; h < 2; ++h) {
        const device T* row = x + ulong(m) * K + g * GS + h * (GS / 2);
        float v = 0.0f;
        for (int q = 0; q < GS / 2; q += 4) {
            const vec<T, 4> u = *(const device vec<T, 4>*)(row + q);
            v += float(u[0]); v += float(u[1]); v += float(u[2]); v += float(u[3]);
        }
        half_sums[h] = v;
    }
    sums[ulong(m) * G + g] = half_sums[0] + half_sums[1];
"""
_LANE_SUMS_SPLIT = """
    const float total = sums[ulong(m) * G + g];
    sums_hi[ulong(m) * G + g] = T(total);
    sums_lo[ulong(m) * G + g] = T(total - float(T(total)));
"""
# Several weights of one input in one launch: a threadgroup's column tile picks its weight, the rest reads as one weight.
_LANE_PARTS = """
    const uint tile = threadgroup_position_in_grid.y;
    int part = 0;
    for (int i = 1; i < PARTS; ++i) part += int(tile >= TILES[i]);
    const int N = NS[part];
    const uint n0 = (tile - TILES[part]) * COLUMNS;
"""
_LANE_TENSOR_BITS = (4, 8)
_LANE_TENSOR_COLUMNS = 32
_LANE_TENSOR_WIDE = 16384   # outputs from which 8-bit weights read K slices, and 4-bit bf16 ones outside `scales` fold the bias term
_LANE_STAGED_GROUPS = 16
_LANE_DEEP_GROUPS = 256
_LANE_WIDE_COLUMNS = 8192   # from here four column tiles per threadgroup read paired codes faster than two

# Contraction off, so every variant rounds a row the same way; MLX inlines the source, so the pragma opens the body.
_LANE_PREAMBLE = "#pragma clang fp contract(off)\n"
_LANE_TYPE, _LANE_CONSTANT = "typedef {1} {0};\n", "constexpr {2} {0} = {1};\n"
_LANE_METAL_TYPES = {mx.bfloat16: "bfloat16_t", mx.float16: "float16_t", mx.float32: "float"}
# Unrolling the group loop by two lets more code loads issue ahead of the arithmetic: the same bytes and bits, less DRAM
# energy per byte on M3. The "slices" reader (the drafter's, and the default path's shortlist's) keeps the one-group loop.
_LANE_GROUP_HEADER = "    for (uint g = g0; g < g1; ++g) {"
_LANE_TWO_GROUP_HEADER = '    _Pragma("clang loop unroll_count(2)")\n' + _LANE_GROUP_HEADER
assert _LANE_GENERIC_LOOP.count(_LANE_GROUP_HEADER) == _LANE_PAIRED_LOOP.count(_LANE_GROUP_HEADER) == 1

LANE_QMM_PROBE_KEY = "lane_qmm:" + hashlib.sha256((
    _LANE_PREAMBLE + _LANE_TYPE + _LANE_CONSTANT + _LANE_SIMD_PROLOGUE + _LANE_GENERIC_LOOP + _LANE_PAIRED_LOOP + _LANE_TWO_GROUP_HEADER + _LANE_SIMD_EPILOGUE + nax._QMM_HEADER
    + _LANE_TENSOR_ONE + _LANE_PARTS + _LANE_TENSOR_HEAD + _LANE_TENSOR_FOLD + _LANE_SLICES_LOOP + _LANE_STAGED_LOOP + _LANE_BIAS_LOOP + _LANE_SCALES_LOOP
    + _LANE_TENSOR_REDUCE + _LANE_TENSOR_TAIL
    + _LANE_SUMS_SOURCE + _LANE_SUMS_SPLIT).encode()).hexdigest()[:12]


def _lane_operand(bits, dtype):
    """`_SIMD_QMM_OPERANDS` for any width: bf16 cannot hold `256 + q`, so 8-bit codes multiply as fp32."""
    if dtype == mx.bfloat16 and bits == 8:
        return mx.float32, "uint32_t", 15, struct.unpack("<I", struct.pack("<f", 256.0))[0], 256
    if dtype == mx.bfloat16:
        return mx.bfloat16, "uint16_t", 7 - bits, (struct.unpack("<I", struct.pack("<f", 2.0 ** bits))[0] >> 16) * 0x10001, 2 ** bits
    return mx.float16, "uint16_t", 10 - bits, struct.unpack("<H", struct.pack("<e", 2.0 ** bits))[0] * 0x10001, 2 ** bits


def _lane_tensor_units(N, bits):
    return bits in _LANE_TENSOR_BITS and N % _LANE_TENSOR_COLUMNS == 0 and nax.nax_available()


def _lane_tensor_simdgroups(K, group_size, bits):
    """K slices, from the weight's shape alone: the most that leave each slice four or more groups, up to 8 (16 for byte codes)."""
    simdgroups = 1
    while simdgroups < _LANE_SIMDGROUPS * (2 if bits == 8 else 1) and K // group_size >= 8 * simdgroups:
        simdgroups *= 2
    return simdgroups


def _lane_tensor_reader(N, K, group_size, bits, dtype):
    """The tensor-unit reader of an `[N, K]` weight's calls, from its shape alone so that a row's bits do not depend on
    its call: the one measured fastest when each call waits on the one before, as in a forward. 4-bit weights read K
    slices with their scales staged eight groups at a time, where each column's scales are whole 16-byte runs.
    Otherwise narrower weights read K slices at 4 bits and stage their scales and biases at 8; wider ones fold the bias
    term as one matmul at 4 bits (bf16 only, where a sum's two halves cannot overflow) and read K slices at 8."""
    if K // group_size % 4:   # the staged rounds load four groups at once
        return "slices"
    if bits == 4 and K // group_size % 8 == 0:
        return "scales"
    if N < _LANE_TENSOR_WIDE:
        return "slices" if bits == 4 else "staged"
    return "slices" if bits == 8 else "bias" if dtype == mx.bfloat16 else "staged"


def _lane_paired(group_size, bits):
    """Sub-byte codes whose span fills whole words; byte codes read as fast unpaired, and faster at 16 rows."""
    return bits in (2, 4) and group_size // 4 % (32 // bits) == 0


def _lane_parts(Ns, columns = _LANE_TENSOR_COLUMNS):
    """(header, prelude, weight inputs, outputs) of a launch over weights of `Ns` outputs, in column tiles of `columns`."""
    names = ("w", "scales", "biases")
    if len(Ns) == 1:
        return "", None, list(names), ["y"]
    tiles = [sum(Ns[:i]) // columns for i in range(len(Ns))]
    header = (_LANE_CONSTANT.format("PARTS", len(Ns), "constant int") + _LANE_CONSTANT.format("COLUMNS", columns, "constant int")
              + "constexpr constant int TILES[] = {%s};\nconstexpr constant int NS[] = {%s};\n" % (", ".join(map(str, tiles)), ", ".join(map(str, Ns))))
    pick = lambda name: "    %s %s = %s;\n" % ("device Y*" if name == "y" else "const device uint32_t*" if name == "w" else "const device T*",
                                              name, " : ".join(f"part == {i} ? {name}{i}" for i in range(len(Ns) - 1)) + f" : {name}{len(Ns) - 1}")
    prelude = _LANE_PARTS + "".join(pick(name) for name in (*names, "y"))
    return header, prelude, [f"{name}{i}" for i in range(len(Ns)) for name in names], [f"y{i}" for i in range(len(Ns))]


@functools.cache
def _lane_tensor_launch(M, Ns, K, group_size, bits, dtype, output_dtype, groups, reader):
    """The tensor-unit kernel of `x @ W^T` for weights of `Ns` outputs each, one launch, every output's bits those of
    its weight's own launch."""
    simdgroups = {"slices": _lane_tensor_simdgroups(K, group_size, bits), "interleaved": _LANE_SIMDGROUPS,
                  "staged": 4 if bits == 4 else _LANE_SIMDGROUPS, "bias": 4,
                  "scales": _lane_tensor_simdgroups(K, group_size, bits)}[reader]
    # H: row halves a tile can fill; up to 8 rows the second half is padding, and skipping it per group is not free.
    constants = dict(K = K, GS = group_size, BITS = bits, RT = 1 if M <= 16 else 2, NSG = simdgroups, H = 1 if M <= 8 else 2, RG = groups,
                     ILV = int(reader == "interleaved"), CG = _LANE_STAGED_GROUPS,
                     # Deep weights run faster with the partials in the spent scale stages; the rest slower.
                     **(dict(REUSE = int(K // group_size >= _LANE_DEEP_GROUPS)) if reader == "scales" else {}))
    header = (nax._QMM_HEADER + _LANE_TYPE.format("T", _LANE_METAL_TYPES[dtype])
              + _LANE_TYPE.format("Y", _LANE_METAL_TYPES[output_dtype])
              + "".join(_LANE_CONSTANT.format(key, value, "constant int") for key, value in constants.items()))
    parts, prelude, weights, outputs = _lane_parts(Ns)
    header += parts
    kernel = mx.fast.metal_kernel(name = "_".join(map(str, ("unsloth_lane_tensor_qmm", reader, _LANE_METAL_TYPES[dtype],
                                                            _LANE_METAL_TYPES[output_dtype], *constants.values(), *Ns[1:]))),
                                  input_names = ["x", *weights, *(("sums", "sums_hi", "sums_lo") if reader == "bias" else ("sums",))],
                                  output_names = outputs, header = header,
                                  source = _LANE_PREAMBLE + (prelude or _LANE_TENSOR_ONE) + _LANE_TENSOR_HEAD + _LANE_TENSOR_LOOPS[reader]
                                  + ("" if reader == "scales" else _LANE_TENSOR_REDUCE) + _LANE_TENSOR_TAIL)
    return functools.partial(kernel, grid = (32 * simdgroups, sum(Ns) // _LANE_TENSOR_COLUMNS, 1), threadgroup = (32 * simdgroups, 1, 1),
                             output_shapes = [(M, N) for N in Ns], output_dtypes = [output_dtype] * len(Ns))


@functools.cache
def _lane_launch(M, N, K, group_size, bits, dtype, *, output_dtype = None, groups = None, reader = None):
    output_dtype = dtype if output_dtype is None else output_dtype
    groups = K // group_size if groups is None else groups
    if _lane_tensor_units(N, bits):
        return _lane_tensor_launch(M, (N,), K, group_size, bits, dtype, output_dtype, groups,
                                   reader or _lane_tensor_reader(N, K, group_size, bits, dtype))
    return _lane_simd_launch(M, (N,), K, group_size, bits, dtype, output_dtype, groups, reader)


def _lane_column_tiles(N, group_size, bits):
    return 4 if _lane_paired(group_size, bits) and N % 32 == 0 and N >= _LANE_WIDE_COLUMNS else 2 if N % 16 == 0 else 1


def _lane_simd_launch(M, Ns, K, group_size, bits, dtype, output_dtype, groups, reader):
    """The simdgroup-matrix kernel of `x @ W^T` for weights of `Ns` outputs each, one launch, every output's bits those
    of its weight's own launch."""
    # Row and column tiles only regroup independent outputs; the K split, which orders the sums, follows the weight.
    rows = 8 if M <= 8 else 16
    paired = _lane_paired(group_size, bits)
    column_tiles = _lane_column_tiles(Ns[0], group_size, bits)
    simdgroups = min(_LANE_SIMDGROUPS, K // group_size)
    span_bits = group_size // 4 * bits   # a lane's span starts at a multiple of its length: load it this wide
    operand, operand_bits, shift, offset_bits, offset = _lane_operand(bits, dtype)
    types = dict(T = _LANE_METAL_TYPES[dtype], Y = _LANE_METAL_TYPES[output_dtype], O = _LANE_METAL_TYPES[operand], OB = operand_bits,
                 U = {1: "uchar", 2: "ushort", 4: "uint"}[min(4, (span_bits & -span_bits) // 8)])
    constants = dict(K = K, GS = group_size, BITS = bits, RT = rows // 8, CT = column_tiles, NSG = simdgroups, RG = groups,
                     SHIFT = shift)
    header = (_LANE_PREAMBLE + "".join(_LANE_TYPE.format(*item) for item in types.items())
              + "".join(_LANE_CONSTANT.format(key, value, "int") for key, value in constants.items())
              + _LANE_CONSTANT.format("OFFSET_BITS", f"{offset_bits}u", "uint") + _LANE_CONSTANT.format("OFFSET", f"{offset}.0f", "float"))
    parts, prelude, weights, outputs = _lane_parts(Ns, 8 * column_tiles)
    loop = _LANE_PAIRED_LOOP if paired else _LANE_GENERIC_LOOP
    two_groups = reader != "slices"
    if two_groups:
        loop = loop.replace(_LANE_GROUP_HEADER, _LANE_TWO_GROUP_HEADER)
    kernel = mx.fast.metal_kernel(name = "_".join(map(str, ("unsloth_lane_qmm", "paired" if paired else "generic", *(["twogroups"] if two_groups else []),
                                                            *types.values(), *constants.values(), *Ns[1:]))),
                                  input_names = ["x", *weights], output_names = outputs, header = parts,
                                  source = header + (prelude or _LANE_SIMD_ONE) + _LANE_SIMD_PROLOGUE + loop + _LANE_SIMD_EPILOGUE)
    return functools.partial(kernel, grid = (32 * simdgroups, sum(Ns) // (8 * column_tiles), -(-M // rows)),
                             threadgroup = (32 * simdgroups, 1, 1), output_shapes = [(M, N) for N in Ns],
                             output_dtypes = [output_dtype] * len(Ns))


@functools.cache
def _lane_sums_launch(M, K, group_size, dtype, split = False):
    """Each row's fp32 sum over each group and, `split`, its high and low halves in `dtype`."""
    constants = dict(K = K, GS = group_size, G = K // group_size)
    kernel = mx.fast.metal_kernel(name = "_".join(map(str, ("unsloth_lane_sums" + "_split" * split, _LANE_METAL_TYPES[dtype], *constants.values()))),
                                  input_names = ["x"], output_names = ["sums", *(("sums_hi", "sums_lo") if split else ())],
                                  header = _LANE_TYPE.format("T", _LANE_METAL_TYPES[dtype]) + "".join(
                                      _LANE_CONSTANT.format(key, value, "constant int") for key, value in constants.items()),
                                  source = _LANE_PREAMBLE + _LANE_SUMS_SOURCE + _LANE_SUMS_SPLIT * split)
    return functools.partial(kernel, grid = (-(-(K // group_size) // 32) * 32, M, 1), threadgroup = (32, 1, 1),
                             output_shapes = [(M, K // group_size)] * (3 if split else 1),
                             output_dtypes = [mx.float32, *(dtype,) * (2 * split)])


def lane_qmm_available():
    """Whether this is an Apple GPU (not a paravirtual one), whose simdgroup matrices the kernel's layout assumes."""
    return mx.metal.is_available() and nax._gpu_generation() is not None


def lane_qmm_supported(N, K, group_size, bits, mode):
    """Whether the kernel covers an `[N, K]` weight quantized this way."""
    return (mode == "affine" and bits in _LANE_BITS and group_size in _LANE_GROUP_SIZES and N > 0 and N % 8 == 0
            and K > 0 and K % group_size == 0)


def lane_qmm_geometry(M, N, K, group_size):
    """(rows,) that tell apart every kernel variant a call of M rows can select."""
    return (8 if M <= 8 else 16 if M <= 16 else 32,)


def lane_qmm(x, w, scales, biases, group_size, bits, *, output_dtype = None, groups = None, reader = None):
    """`x @ W^T` of a 2-D x of 1 to `LANE_QMM_MAX_ROWS` rows, each row's bits independent of the others; `groups`
    reads only K's first groups, and `reader` picks another tensor-unit reader ("slices" also keeps the simdgroup-matrix
    kernel's one-group loop, whose bits are the same)."""
    (M, K), N = x.shape, w.shape[0]
    inputs = [x, w, scales, biases]
    if _lane_tensor_units(N, bits):
        reader = reader or _lane_tensor_reader(N, K, group_size, bits, x.dtype)
        inputs += _lane_sums_launch(M, K, group_size, x.dtype, reader == "bias")(inputs = [x])
    return _lane_launch(M, N, K, group_size, bits, x.dtype, output_dtype = output_dtype, groups = groups,
                        reader = reader)(inputs = inputs)[0]


@functools.cache
def _lane_shared_launch(M, Ns, K, group_size, bits, dtype):
    return _lane_simd_launch(M, Ns, K, group_size, bits, dtype, dtype, K // group_size, None)


def lane_qmms_reader(N, K, group_size, bits, dtype):
    """The reader an `[N, K]` weight's calls take: weights of one reader can share a `lane_qmms` launch."""
    if _lane_tensor_units(N, bits):
        return _lane_tensor_reader(N, K, group_size, bits, dtype)
    return "simd", _lane_column_tiles(N, group_size, bits)


def lane_qmms(x, weights, group_size, bits):
    """`lane_qmm` of x by each `(w, scales, biases)` of `weights` in one launch, each output's bits those of its own
    call; None unless every weight takes one `lane_qmms_reader`."""
    (M, K), Ns = x.shape, tuple(w.shape[0] for w, _, _ in weights)
    readers = {lane_qmms_reader(N, K, group_size, bits, x.dtype) for N in Ns}
    if len(readers) > 1 or None in readers:
        return None
    reader = readers.pop()
    if isinstance(reader, tuple):
        return _lane_shared_launch(M, Ns, K, group_size, bits, x.dtype)(inputs = [x, *(a for weight in weights for a in weight)])
    sums = _lane_sums_launch(M, K, group_size, x.dtype, reader == "bias")(inputs = [x])
    return _lane_tensor_launch(M, Ns, K, group_size, bits, x.dtype, x.dtype, K // group_size, reader)(
        inputs = [x, *(a for weight in weights for a in weight), *sums])


def lane_shortlist_reader(M, bits):
    """The tensor-unit reader measured fastest for a coarse shortlist of M rows, which needs no row invariance."""
    return "interleaved" if bits == 4 and M <= 2 else None


_LANE_INDEXED_SOURCE = """
    uint row = threadgroup_position_in_grid.y;
    uint candidate = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    uint n = ids[row * ids_shape[1] + candidate];
    float total = 0.0f;
    for (uint j = lane; j < K; j += 32) {
        uint g = j / GS;
        uint word = w[ulong(n) * (K * BITS / 32) + j / (32 / BITS)];
        float s = float(scales[ulong(n) * G + g]);
        float b = float(biases[ulong(n) * G + g]);
        uint q = (word >> ((j % (32 / BITS)) * BITS)) & ((1u << BITS) - 1u);
        total += float(x[ulong(row) * K + j]) * (s * float(q) + b);
    }
    total = simd_sum(total);
    if (lane == 0) out[row * ids_shape[1] + candidate] = T(total);
"""


@functools.cache
def _lane_indexed_kernel(K, group_size, bits, dtype):
    header = _LANE_TYPE.format("T", _LANE_METAL_TYPES[dtype]) + f"constant int K={K}, GS={group_size}, BITS={bits}, G=K/GS;\n"
    return mx.fast.metal_kernel(name = f"unsloth_lane_indexed_{K}_{group_size}_{bits}_{_LANE_METAL_TYPES[dtype]}",
                                input_names = ["x", "w", "scales", "biases", "ids"], output_names = ["out"],
                                header = header, source = _LANE_INDEXED_SOURCE)


def _lane_indexed(x, w, scales, biases, ids, group_size, bits):
    ids = ids.astype(mx.uint32)
    return _lane_indexed_kernel(x.shape[-1], group_size, bits, x.dtype)(
        inputs = [x, w, scales, biases, ids], grid = (32 * ids.shape[1], ids.shape[0], 1), threadgroup = (32, 1, 1),
        output_shapes = [ids.shape], output_dtypes = [x.dtype])[0]


_SHORTLIST_K, _SHORTLIST_CHUNK = 512, 8192

_SHORTLIST_HEADER = """
template<typename U> inline uint order_key(U value) {
    ushort raw = as_type<ushort>(value);
    if (isnan(float(value))) return 0;
    if ((raw & 0x7fff) == 0) raw = 0;
    return raw & 0x8000 ? uint(ushort(~raw)) : uint(raw ^ 0x8000);
}
inline uint block_prefix(uint value, threadgroup uint* scratch, uint lane, uint sg) {
    uint before = simd_prefix_exclusive_sum(value);
    if (lane == 31) scratch[sg] = before + value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < sg; ++i) before += scratch[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return before;
}
"""

_SHORTLIST_HISTOGRAM = """
    uint chunk = threadgroup_position_in_grid.x, row = threadgroup_position_in_grid.y;
    uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup;
    threadgroup atomic_uint bins[256];
    atomic_store_explicit(bins + tid, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = tid; j < CHUNK; j += 256) {
        uint col = chunk * CHUNK + j;
        bool valid = col < N;
        uint key = valid ? order_key(values[row * N + col]) : 0;
"""

_SHORTLIST_HIGH_COUNT = """
        while (true) {
            uint first = simd_min(valid ? lane : 32u);
            if (first == 32) break;
            uint bucket = simd_shuffle(key >> 8, first);
            bool match = valid && (key >> 8) == bucket;
            uint count = simd_sum(uint(match));
            if (lane == first) atomic_fetch_add_explicit(bins + bucket, count, memory_order_relaxed);
            valid = valid && !match;
        }
"""

_SHORTLIST_LOW_COUNT = """
        if (valid && (key >> 8) == high[row * 2])
            atomic_fetch_add_explicit(bins + (key & 255), 1u, memory_order_relaxed);
"""

_SHORTLIST_HISTOGRAM_STORE = """
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hist[(row * SHARDS + chunk) * 256 + tid] = atomic_load_explicit(bins + tid, memory_order_relaxed);
"""

_SHORTLIST_THRESHOLD = """
    uint row = threadgroup_position_in_grid.x, tid = thread_index_in_threadgroup;
    threadgroup uint bins[256];
    uint count = 0;
    for (uint c = 0; c < SHARDS; ++c) count += hist[(row * SHARDS + c) * 256 + tid];
    bins[tid] = count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint remaining = REMAINING;
        uint bin = 255;
        while (bin && remaining > bins[bin]) remaining -= bins[bin--];
        cut[row * 2] = bin;
        cut[row * 2 + 1] = remaining;
    }
"""

_SHORTLIST_COLLECT = """
    uint chunk = threadgroup_position_in_grid.x, row = threadgroup_position_in_grid.y;
    uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    uint hi = high[row * 2], lo = low[row * 2], cut = (hi << 8) | lo, ties = low[row * 2 + 1];
    threadgroup uint scratch[8];
    uint before = 0, equal_before = 0;
    for (uint c = 0; c < chunk; ++c) {
        if (tid > hi) before += high_hist[(row * SHARDS + c) * 256 + tid];
        if (tid > lo) before += low_hist[(row * SHARDS + c) * 256 + tid];
        equal_before += low_hist[(row * SHARDS + c) * 256 + lo];
    }
    uint prefix = block_prefix(before, scratch, lane, sg);
    threadgroup uint greater_before;
    if (tid == 255) greater_before = prefix + before;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint first = chunk * CHUNK + tid * (CHUNK / 256);
    uint greater = 0, equal = 0;
    for (uint j = 0; j < CHUNK / 256; ++j) {
        uint col = first + j;
        if (col < N) {
            uint key = order_key(values[row * N + col]);
            greater += key > cut;
            equal += key == cut;
        }
    }
    uint prior_equal = equal_before + block_prefix(equal, scratch, lane, sg);
    uint take_equal = prior_equal < ties ? min(equal, ties - prior_equal) : 0;
    uint offset = greater_before + min(equal_before, ties) + block_prefix(greater + take_equal, scratch, lane, sg);
    for (uint j = 0; j < CHUNK / 256; ++j) {
        uint col = first + j;
        if (col < N) {
            uint key = order_key(values[row * N + col]);
            bool take = key > cut || (key == cut && prior_equal < ties);
            prior_equal += key == cut;
            if (take) ids[row * K_SELECT + offset++] = col;
        }
    }
"""


@functools.cache
def _shortlist_histogram(low):
    count = _SHORTLIST_LOW_COUNT if low else _SHORTLIST_HIGH_COUNT
    return mx.fast.metal_kernel(name = f"unsloth_shortlist_histogram_{int(low)}", input_names = ["values", *(["high"] if low else [])],
                                output_names = ["hist"], header = _SHORTLIST_HEADER, compile_options = {"math_mode": "safe"},
                                source = _SHORTLIST_HISTOGRAM + count + _SHORTLIST_HISTOGRAM_STORE)


@functools.cache
def _shortlist_cut(low):
    return mx.fast.metal_kernel(name = f"unsloth_shortlist_cut_{int(low)}", input_names = ["hist", *(["high"] if low else [])],
                                output_names = ["cut"], compile_options = {"math_mode": "safe"},
                                source = _SHORTLIST_THRESHOLD.replace("REMAINING", "high[row * 2 + 1]" if low else "K_SELECT"))


@functools.cache
def _shortlist_collect():
    return mx.fast.metal_kernel(name = "unsloth_shortlist_collect", input_names = ["values", "high_hist", "low_hist", "high", "low"],
                                output_names = ["ids"], header = _SHORTLIST_HEADER, compile_options = {"math_mode": "safe"},
                                source = _SHORTLIST_COLLECT)


def radix_shortlist(values):
    """The 512 largest entries of each row of 16-bit ``values`` [M, N > 512] as ascending uint32 column ids; equal values
    at the cut keep the lowest columns, and NaN ranks below every number."""
    M, N = values.shape
    shards = -(-N // _SHORTLIST_CHUNK)
    template = [("T", values.dtype), ("N", N), ("CHUNK", _SHORTLIST_CHUNK), ("SHARDS", shards), ("K_SELECT", _SHORTLIST_K)]
    launch = dict(template = template, threadgroup = (256, 1, 1), output_dtypes = [mx.uint32])
    per_chunk, per_row = (256 * shards, M, 1), (256 * M, 1, 1)
    high_hist = _shortlist_histogram(False)(inputs = [values], grid = per_chunk, output_shapes = [(M, shards, 256)], **launch)[0]
    high = _shortlist_cut(False)(inputs = [high_hist], grid = per_row, output_shapes = [(M, 2)], **launch)[0]
    low_hist = _shortlist_histogram(True)(inputs = [values, high], grid = per_chunk, output_shapes = [(M, shards, 256)], **launch)[0]
    low = _shortlist_cut(True)(inputs = [low_hist, high], grid = per_row, output_shapes = [(M, 2)], **launch)[0]
    return _shortlist_collect()(inputs = [values, high_hist, low_hist, high, low], grid = per_chunk,
                                output_shapes = [(M, _SHORTLIST_K)], **launch)[0]


def probe_lane_qmm():
    """Subprocess probe: every width and group size, bf16 and fp16, one, two and four column tiles (or the tensor-unit
    readers each shape takes, and the interleaved one; on simdgroup matrices also the "slices" one-group loop), both row
    tiles and ragged ones; within 2% of stock's largest output, and every row of a block, wherever
    it starts, bit for bit its one-row call."""
    wide = (_LANE_TENSOR_WIDE,) if nax.nax_available() else ()
    for bits, group_size, dtype, N in itertools.product(_LANE_BITS, _LANE_GROUP_SIZES, (mx.bfloat16, mx.float16),
                                                        (72, 128, _LANE_WIDE_COLUMNS, *wide)):
        w = mx.random.normal((N, 1024), key = mx.random.key(bits)) * 0.05
        w, scales, biases = mx.quantize(w.astype(dtype), group_size = group_size, bits = bits)
        x = mx.random.normal((LANE_QMM_MAX_ROWS, 1024), key = mx.random.key(group_size)).astype(dtype)
        native = mx.quantized_matmul(x, w, scales, biases, transpose = True, group_size = group_size, bits = bits)
        got = lane_qmm(x, w, scales, biases, group_size, bits)
        for reader in ("interleaved",) if _lane_tensor_units(N, bits) else ("slices",):
            assert mx.abs(lane_qmm(x[:2], w, scales, biases, group_size, bits, reader = reader).astype(mx.float32)
                          - native[:2].astype(mx.float32)).max().item() <= 0.02 * mx.abs(native).max().item(), reader
        error = mx.abs(got.astype(mx.float32) - native.astype(mx.float32)).max().item()
        assert error <= 0.02 * mx.abs(native).max().item(), (bits, group_size, dtype, error)
        for start, M in ((0, 1), (0, 5), (0, 17), (LANE_QMM_MAX_ROWS - 1, 1), (LANE_QMM_MAX_ROWS - 5, 5), (3, 17)):
            part = lane_qmm(x[start:start + M], w, scales, biases, group_size, bits)
            assert mx.array_equal(mx.view(part, mx.uint16), mx.view(got[start:start + M], mx.uint16)).item(), (bits, start, M)
