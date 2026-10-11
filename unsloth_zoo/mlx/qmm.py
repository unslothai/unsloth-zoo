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

"""Small-row 4- and 8-bit affine `x @ W^T` on the stock packed weights, for Apple GPUs without stock NAX kernels."""

import functools
import hashlib
import itertools

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

# Up to this many rows an 8 x 8 tile is mostly padding, so scalar lanes split K, dequantizing
# each code once as `s * q + b` in fp32. More columns a threadgroup, or a rolled code loop, spill the accumulators.
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

# Constants go into the source because MLX re-parses template arguments on every call; it also recompiles
# a library whose name returns with another source, hence one name per configuration.
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


# Rows where the kernels beat stock end to end, in `nax._QMM_ROWS_BY_GPU`'s format; measured on an M3 Max,
# taken by every generation.
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
