# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# The three kernels below are adapted from JetBrains' int8 NAX prefill patch for
# mlx-vlm (MIT licensed):
#   https://github.com/JetBrains/mlx-vlm/tree/feature/int8-prefill
#   mlx_vlm/int8_prefill.py, Copyright (c) 2025 Prince Canuma and contributors.
#
# Changes from that original:
#   * the MetalPerformancePrimitives include is isolated to the GEMM kernel, so the
#     activation-quant and weight-requant kernels compile and run on any Metal device;
#   * the requant kernel is parameterized on bits and group size rather than assuming
#     4-bit / group_size 64 / eight nibbles per word;
#   * the bias epilogue is gone -- patching at op level means the caller adds bias after
#     the matmul, halving the GEMM variants we compile;
#   * the GEMM emits int32 so callers can compare it exactly against an integer
#     reference; scale application moved out to the caller.
"""Metal kernels for the W8A8 int8 prefill path.

Only `int8_gemm` needs Metal Performance Primitives (Metal 4, and the int8 arithmetic
rate that makes any of this worthwhile arrives with the M5 neural accelerators). The
other two are ordinary Metal, which is deliberate: it means an M1 CI runner can validate
the packed-weight unpacking -- the nibble ordering and word-boundary arithmetic that is
the likeliest place for a silent wrong-answer bug, and that Linux cannot check at all.
"""

import mlx.core as mx

# Ordinary Metal. No MPP, so these compile on any Metal device.
_PLAIN_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

# Metal 4 tensor ops. Compiling this at all requires macOS 26+; running it usefully
# requires the M5 neural accelerators.
_MPP_HEADER = """
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
using namespace metal;
"""

# One threadgroup (256 threads) per row: absmax reduce, then quantize.
_QUANT_SRC = """
    constexpr int K = {K};
    constexpr int NTH = 256;

    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = tid % 32;
    uint sg = tid / 32;

    const device {T}* xrow = x + size_t(row) * K;

    float amax = 0.0f;
    for (int i = tid; i < K; i += NTH) {{
        amax = max(amax, fabs(float(xrow[i])));
    }}
    amax = simd_max(amax);

    threadgroup float tg_max[NTH / 32];
    if (lane == 0) tg_max[sg] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    amax = tg_max[lane % (NTH / 32)];
    amax = simd_max(amax);

    float scale = max(amax, 1e-8f) / 127.0f;
    float inv = 1.0f / scale;
    if (tid == 0) xs[row] = scale;

    device int8_t* qrow = xq + size_t(row) * K;
    for (int i = tid; i < K; i += NTH) {{
        qrow[i] = int8_t(clamp(rint(float(xrow[i]) * inv), -127.0f, 127.0f));
    }}
"""

# Fused requantization: packed affine weights -> per-channel symmetric int8, one pass,
# no high-precision intermediate. One threadgroup (256 threads) per output channel.
#
# VPW values per uint32 word and WPG words per group are both integral for every
# (bits, group_size) the eligibility table admits, so a word never straddles two groups
# and the group index is a plain division.
_REQUANT_SRC = """
    constexpr int KW = {KW};     // packed words per row
    constexpr int VPW = {VPW};   // values per word (32 / bits)
    constexpr int WPG = {WPG};   // words per group (group_size / VPW)
    constexpr int BITS = {BITS};
    constexpr uint MASK = {MASK}u;

    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;

    const device uint* prow = packed + size_t(row) * KW;
    const device {T}* srow = scales + size_t(row) * (KW / WPG);
    const device {T}* brow = biases + size_t(row) * (KW / WPG);
    device int8_t* orow = out + size_t(row) * KW * VPW;

    float inv = 1.0f / ws[row];

    for (int i = tid; i < KW; i += 256) {{
        uint wrd = prow[i];
        int g = i / WPG;
        float s = float(srow[g]);
        float b = float(brow[g]);
        device int8_t* o = orow + i * VPW;
#pragma unroll
        for (int j = 0; j < VPW; ++j) {{
            float v = float((wrd >> (BITS * j)) & MASK) * s + b;
            o[j] = int8_t(clamp(rint(v * inv), -127.0f, 127.0f));
        }}
    }}
"""

# Threadgroup computes a 128x128 output tile with 8 simdgroups; matmul2d loops over K
# internally (dynamic_extent). Edge tiles in M are handled by the epilogue guard; N is
# required to be a multiple of 128 by the eligibility rules, because the launch grid
# below does not round up and a partial N tile would silently drop columns.
_GEMM_SRC = """
    constexpr int N = {N};
    constexpr int K = {K};
    constexpr int TM = 128;
    constexpr int TN = 128;

    uint2 tgid = threadgroup_position_in_grid.xy;
    const int M = m_dim[0];

    constexpr auto desc = matmul2d_descriptor(
        TM, TN, static_cast<int>(dynamic_extent),
        /*transpose_left=*/false, /*transpose_right=*/true,
        /*relaxed_precision=*/false,
        matmul2d_descriptor::mode::multiply_accumulate);

    matmul2d<desc, execution_simdgroups<8>> op;

    // Row-major X[M,K] -> extents (K, M); row-major W[N,K] used as the transposed right
    // operand -> extents (K, N).
    auto A = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        (device int8_t*)xq, dextents<int32_t, 2>(K, M));
    auto B = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        (device int8_t*)wq, dextents<int32_t, 2>(K, N));

    auto tA = A.slice(0, int(tgid.y) * TM);
    auto tB = B.slice(0, int(tgid.x) * TN);

    auto cT = op.get_destination_cooperative_tensor<
        decltype(tA), decltype(tB), int32_t>();

#pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {{
        if (cT.is_valid_element(i)) cT[i] = 0;
    }}

    op.run(tA, tB, cT);

#pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {{
        if (cT.is_valid_element(i)) {{
            auto idx = cT.get_multidimensional_index(i);
            int n = int(tgid.x) * TN + idx[0];
            int m = int(tgid.y) * TM + idx[1];
            if (m < M && n < N) {{
                out[size_t(m) * N + n] = cT[i];
            }}
        }}
    }}
"""

_TM, _TN, _NSIMD = 128, 128, 8
_DTYPE_NAMES = {mx.bfloat16: "bfloat", mx.float16: "half", mx.float32: "float"}

_quant_kernels = {}
_requant_kernels = {}
_gemm_kernels = {}


def _quant_kernel(k, tname):
    key = (k, tname)
    if key not in _quant_kernels:
        _quant_kernels[key] = mx.fast.metal_kernel(
            name=f"unsloth_i8_rowquant_{k}_{tname}",
            input_names=["x"],
            output_names=["xq", "xs"],
            header=_PLAIN_HEADER,
            source=_QUANT_SRC.format(K=k, T=tname),
        )
    return _quant_kernels[key]


def _requant_kernel(kw, vpw, wpg, bits, tname):
    key = (kw, vpw, wpg, bits, tname)
    if key not in _requant_kernels:
        _requant_kernels[key] = mx.fast.metal_kernel(
            name=f"unsloth_i8_requant_{kw}_{bits}_{wpg}_{tname}",
            input_names=["packed", "scales", "biases", "ws"],
            output_names=["out"],
            header=_PLAIN_HEADER,
            source=_REQUANT_SRC.format(
                KW=kw, VPW=vpw, WPG=wpg, BITS=bits, MASK=(1 << bits) - 1, T=tname
            ),
        )
    return _requant_kernels[key]


def _gemm_kernel(n, k):
    key = (n, k)
    if key not in _gemm_kernels:
        _gemm_kernels[key] = mx.fast.metal_kernel(
            name=f"unsloth_i8_gemm_{n}x{k}",
            input_names=["xq", "wq", "m_dim"],
            output_names=["out"],
            header=_MPP_HEADER,
            source=_GEMM_SRC.format(N=n, K=k),
        )
    return _gemm_kernels[key]


def build_probe_kernel(N, K):
    """The GEMM kernel alone, for the capability probe in capability.py."""
    return _gemm_kernel(N, K)


def quantize_rows(x):
    """Per-row symmetric int8 activation quantization. Returns (xq int8, xs float32)."""
    m, k = x.shape
    return _quant_kernel(k, _DTYPE_NAMES[x.dtype])(
        inputs=[x],
        grid=(m * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(m, k), (m,)],
        output_dtypes=[mx.int8, mx.float32],
    )


def requantize_weight(weight, scales, biases, ws, bits, group_size):
    """Packed affine weights -> per-channel symmetric int8 [N, K], fused."""
    n, kw = weight.shape[-2:]
    vpw = 32 // bits
    wpg = group_size // vpw
    kernel = _requant_kernel(kw, vpw, wpg, bits, _DTYPE_NAMES[scales.dtype])
    return kernel(
        inputs=[weight, scales, biases, ws],
        grid=(n * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n, kw * vpw)],
        output_dtypes=[mx.int8],
    )[0]


def int8_gemm_raw(xq, wq):
    """Raw int32 (xq @ wq.T). No scales -- the caller folds those in."""
    m, k = xq.shape
    n = wq.shape[0]
    return _gemm_kernel(n, k)(
        inputs=[xq, wq, mx.array([m], dtype=mx.int32)],
        grid=(n // _TN * 32 * _NSIMD, (m + _TM - 1) // _TM, 1),
        threadgroup=(32 * _NSIMD, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[mx.int32],
    )[0]


def matmul(x, entry, out_dtype=mx.bfloat16):
    """Full W8A8 path for a registered weight."""
    k = x.shape[-1]
    flat = x.reshape(-1, k)
    xq, xs = quantize_rows(flat)
    wq = requantize_weight(
        entry.w, entry.scales, entry.biases, entry.ws, entry.bits, entry.group_size
    )
    acc = int8_gemm_raw(xq, wq)
    out = acc.astype(mx.float32) * xs[:, None] * entry.ws[None, :]
    return out.astype(out_dtype).reshape(*x.shape[:-1], entry.n)
