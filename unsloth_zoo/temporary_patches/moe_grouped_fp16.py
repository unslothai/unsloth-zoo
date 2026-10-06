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
"""Triton grouped GEMMs over expert-sorted rows, for MoE training on float16 GPUs.

torch._grouped_mm takes neither fp32 operands nor an fp32 output for fp16 operands, and
off sm90 / sm100 it copies the offsets to the host. These kernels take per-expert row
counts on device (no host sync), accumulate in fp32 and store in any dtype, so a gpt-oss
down projection, whose outputs overflow fp16, runs on fp16 operands with an fp32 output.

  grouped_linear(x, w, counts)   y[m] = x[m] @ w[e(m)].T     w: [E, N, K], rows sorted by expert
    * x and w of one dtype: that dtype on the tensor cores (fp32: IEEE, no TF32);
    * x fp32, w fp16: x rounded to fp16 in the kernel, or split hi + lo (two dots);
    * backward: dX by the same kernel; an fp32 dY against an fp16 w is scaled per row by a
      power of two before the fp16 rounding (no underflow of unscaled gradients, exact
      unscale), single or hi + lo; dW by a per-expert reduction kernel, no atomics.
No TMA, no warp specialization: plain tl.dot tiles.

Backends: Triton lowers tl.dot to CUDA-core FMA on sm75 (T4: no mma.sync in the PTX, ~0.8
TFLOPs against cuBLAS's ~43), so below sm80 the same calls run per-expert cuBLAS GEMMs over
host row counts read once per MoE layer (`host` / Groups), fp16 operands with an fp32 output
through torch.mm(out_dtype=float32) where torch has it. UNSLOTH_GPTOSS_FP16_GEMM=triton|cublas
overrides the choice."""

__all__ = [
    "Groups",
    "use_cublas",
    "grouped_gemm",
    "grouped_wgrad",
    "generic_gemm_config",
    "generic_wgrad_config",
    "grouped_linear",
    "grouped_frozen_linear",
    "fp16_grouped_available",
    "row_pow2_scale",
]

import contextlib
import os

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None

_DISABLED_REASON = None
CALLS = {"gemm": 0, "wgrad": 0}

# A operand handling when x is fp32 and w is fp16 / bf16.
A_DIRECT, A_CAST, A_SPLIT = 0, 1, 2


if triton is not None:

    @triton.jit
    def _grouped_gemm_kernel(
        A, B, C, COUNTS, SCALE, BIAS,
        E, E_LO, E_HI, N, K,
        stride_am, stride_ak, stride_be, stride_bk, stride_bn, stride_cm, stride_bias_e,
        A_MODE: tl.constexpr, ROW_SCALE: tl.constexpr, HAS_BIAS: tl.constexpr, IEEE: tl.constexpr,
        E_POW2: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr = 0, EVEN_K: tl.constexpr = False,
    ):
        # C[m, n] = sum_k A[m, k] * B[e(m) - E_LO, k, n] for the rows of experts in [E_LO, E_HI).
        # GROUP_M > 0: a 1D grid walked GROUP_M row tiles at a time (L2 reuse of A and B tiles);
        # EVEN_K: K % BLOCK_K == 0, no K masks. Both only from generic_gemm_config.
        if GROUP_M > 0:
            pid = tl.program_id(0)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            num_pid_m = tl.num_programs(0) // num_pid_n
            width = GROUP_M * num_pid_n
            first_m = (pid // width) * GROUP_M
            size_m = tl.minimum(num_pid_m - first_m, GROUP_M)
            pid_m = first_m + (pid % width) % size_m
            pid_n = (pid % width) // size_m
        else:
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)
        e_offs = tl.arange(0, E_POW2)
        counts = tl.load(COUNTS + e_offs, mask = e_offs < E, other = 0).to(tl.int32)
        tiles = tl.where((e_offs >= E_LO) & (e_offs < E_HI), (counts + BLOCK_M - 1) // BLOCK_M, 0)
        tile_end = tl.cumsum(tiles, 0)
        e = tl.sum((tile_end <= pid_m).to(tl.int32), 0)
        if e >= E_HI:
            return
        sel = e_offs == e
        row_end = tl.sum(tl.where(sel, tl.cumsum(counts, 0), 0), 0)
        row_start = row_end - tl.sum(tl.where(sel, counts, 0), 0)
        tile_start = tl.sum(tl.where(sel, tile_end - tiles, 0), 0)
        offs_m = row_start + (pid_m - tile_start) * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < row_end
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        offs_m64 = offs_m.to(tl.int64)
        b_base = B + (e - E_LO).to(tl.int64) * stride_be
        if ROW_SCALE:
            s = tl.load(SCALE + offs_m64, mask = mask_m, other = 1.0)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            if EVEN_K:
                a = tl.load(A + offs_m64[:, None] * stride_am + offs_k[None, :] * stride_ak,
                            mask = mask_m[:, None], other = 0.0)
                b = tl.load(b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                            mask = mask_n[None, :], other = 0.0)
            else:
                mask_k = offs_k < K
                a = tl.load(A + offs_m64[:, None] * stride_am + offs_k[None, :] * stride_ak,
                            mask = mask_m[:, None] & mask_k[None, :], other = 0.0)
                b = tl.load(b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                            mask = mask_k[:, None] & mask_n[None, :], other = 0.0)
            if ROW_SCALE:
                a = a * s[:, None]
            if A_MODE == 0:
                if IEEE:
                    acc = tl.dot(a, b, acc, input_precision = "ieee")
                else:
                    acc = tl.dot(a, b, acc)
            elif A_MODE == 1:
                acc = tl.dot(a.to(b.dtype), b, acc)
            else:
                hi = a.to(b.dtype)
                lo = (a - hi.to(tl.float32)).to(b.dtype)
                acc = tl.dot(hi, b, acc)
                acc = tl.dot(lo, b, acc)
        if ROW_SCALE:
            acc = acc / s[:, None]   # a power of two: exact
        if HAS_BIAS:
            bias = tl.load(BIAS + e.to(tl.int64) * stride_bias_e + offs_n, mask = mask_n, other = 0.0)
            acc += bias.to(tl.float32)[None, :]
        c = C + offs_m64[:, None] * stride_cm + offs_n[None, :]
        tl.store(c, acc.to(C.dtype.element_ty), mask = mask_m[:, None] & mask_n[None, :])

    @triton.jit
    def _grouped_wgrad_kernel(
        G, X, DW, COUNTS,
        E, N, K,
        stride_gm, stride_xm, stride_de, stride_dn, stride_dk,
        IEEE: tl.constexpr, E_POW2: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        # DW[e, n, k] = sum over the rows m of expert e of G[m, n] * X[m, k]; zero for an empty expert.
        e = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_k = tl.program_id(2)
        e_offs = tl.arange(0, E_POW2)
        counts = tl.load(COUNTS + e_offs, mask = e_offs < E, other = 0).to(tl.int32)
        sel = e_offs == e
        row_end = tl.sum(tl.where(sel, tl.cumsum(counts, 0), 0), 0)
        row_start = row_end - tl.sum(tl.where(sel, counts, 0), 0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_n = offs_n < N
        mask_k = offs_k < K
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype = tl.float32)
        for m0 in range(row_start, row_end, BLOCK_M):
            offs_m = m0 + tl.arange(0, BLOCK_M)
            mask_m = offs_m < row_end
            offs_m64 = offs_m.to(tl.int64)
            g = tl.load(G + offs_m64[None, :] * stride_gm + offs_n[:, None],
                        mask = mask_n[:, None] & mask_m[None, :], other = 0.0)
            x = tl.load(X + offs_m64[:, None] * stride_xm + offs_k[None, :],
                        mask = mask_m[:, None] & mask_k[None, :], other = 0.0)
            if IEEE:
                acc = tl.dot(g, x, acc, input_precision = "ieee")
            else:
                acc = tl.dot(g, x, acc)
        d = DW + e.to(tl.int64) * stride_de + offs_n[:, None] * stride_dn + offs_k[None, :] * stride_dk
        tl.store(d, acc.to(DW.dtype.element_ty), mask = mask_n[:, None] & mask_k[None, :])


def _disable(exc):
    global _DISABLED_REASON
    _DISABLED_REASON = f"{type(exc).__name__}: {exc}"
    import logging
    logging.getLogger(__name__).warning(
        "Unsloth: the float16 grouped MoE GEMM failed its self-check and is disabled for this "
        f"process; the per-expert loop is used instead. Reason: {_DISABLED_REASON}"
    )


_SM = {}


def _capability(device):
    if device.type != "cuda":
        return (8, 0)
    cap = _SM.get(device.index)
    if cap is None:
        index = device.index if device.index is not None else torch.cuda.current_device()
        cap = _SM[device.index] = torch.cuda.get_device_capability(index)
    return cap


def _on(device):
    return torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext()


def _pow2(n):
    return 1 << max(int(n) - 1, 0).bit_length()


# fp32 gpt-oss down GEMM operands per backend when UNSLOTH_GPTOSS_FP16_DOWN_OPERAND=auto.
# "fp32" is the loop's exact math (IEEE fp32, fp32 dequant stack); "fp16" rounds x / W / dY to fp16
# (dY per-row power-of-two scaled) with an fp32 accumulate and output. Colab T4 (cuBLAS), 20B layer
# fwd + bwd vs the loop: fp16 1.9x / 1.7x at 512 / 2048 tokens, fp32 1.33x / 0.97x (no fp32 tensor
# cores). Triton: fp16 beats fp32 operands on A100 (1.9x / 2.0x at 512 / 2048 tokens), RTX PRO 6000
# (1.3x / 1.5x), L4 (1.3x / 1.2x); fp16 runs on the tensor cores, IEEE fp32 on CUDA-core FMA.
DOWN_OPERAND_AUTO = {"triton": "fp16", "cublas": "fp16"}


def use_cublas(device) -> bool:
    mode = os.environ.get("UNSLOTH_GPTOSS_FP16_GEMM", "auto")
    if mode in ("cublas", "triton"):
        return mode == "cublas"
    # Measured, gpt-oss-20b MoE layer fwd + bwd (Colab, torch 2.11, Triton 3.6): Triton wins on
    # A100 (sm80) and RTX PRO 6000 (sm120) and B200 (sm100); on L4 (sm89) cuBLAS ties at 512
    # tokens and is 1.43x faster at 2048; on T4 (sm75) Triton has no MMA at all.
    return device.type == "cuda" and (_capability(device) < (8, 0) or _capability(device) == (8, 9))


class Groups:
    """Per-expert row counts on device, with the host copy read at most once (cuBLAS backend)."""

    __slots__ = ("t", "_host")

    def __init__(self, t, host = None):
        self.t, self._host = t, host

    def host(self):
        if self._host is None:
            self._host = [int(c) for c in self.t.tolist()]
        return self._host


def _counts_tensor(counts):
    return counts.t if isinstance(counts, Groups) else counts


def _host(counts):
    return counts.host() if isinstance(counts, Groups) else [int(c) for c in counts.tolist()]


_MM_OUT_DTYPE = {}


def _mm(x, w, out_dtype):
    """x @ w with fp32 accumulation, stored in out_dtype (cuBLAS keeps an fp16 x fp16 -> fp32 GEMM
    on the tensor cores when torch.mm takes out_dtype; else an fp32 GEMM, the loop's own math).
    Callers run it with autocast off: an fp16-autocast fp32 GEMM would round the down output to
    fp16 (the per-expert loop disables autocast around down for the same reason)."""
    if x.dtype == out_dtype:
        return x @ w
    if out_dtype == torch.float32 and x.dtype in (torch.float16, torch.bfloat16):
        ok = _MM_OUT_DTYPE.get(x.dtype)
        if ok is not False:
            try:
                y = torch.mm(x, w, out_dtype = torch.float32)
                _MM_OUT_DTYPE[x.dtype] = True
                return y
            except Exception:
                _MM_OUT_DTYPE[x.dtype] = False
        return x.float() @ w.float()
    return (x @ w).to(out_dtype)


def _gemm_cublas(a, b, counts, out_dtype, b_trans, e_lo, e_hi, out, a_mode, row_scale, bias):
    host = _host(counts)
    if a.dtype != b.dtype:
        if row_scale is not None:
            a = a * row_scale[:, None]
        hi = a.to(b.dtype)
        lo = (a - hi.float()).to(b.dtype) if a_mode == A_SPLIT else None
    else:
        hi, lo = a, None
    start = sum(host[:e_lo])
    for e in range(e_lo, e_hi):
        c = host[e]
        if c == 0:
            continue
        rows = slice(start, start + c)
        start += c
        w = b[e - e_lo]
        w = w.T if b_trans else w
        if lo is None and row_scale is None and bias is None and hi.dtype == out_dtype:
            torch.mm(hi[rows], w, out = out[rows])
            continue
        y = _mm(hi[rows], w, torch.float32 if (lo is not None or row_scale is not None or bias is not None) else out_dtype)
        if lo is not None:
            y = y + _mm(lo[rows], w, torch.float32)
        if row_scale is not None:
            y = y / row_scale[rows, None]
        if bias is not None:
            y = y + bias[e].to(y.dtype)
        out[rows] = y
    return out


def _wgrad_cublas(g, x, counts, out_dtype, E):
    host = _host(counts)
    dw = torch.zeros((E, g.shape[1], x.shape[1]), dtype = out_dtype, device = g.device)
    start = 0
    for e, c in enumerate(host[:E]):
        if c:
            dw[e] = _mm(g[start:start + c].T, x[start:start + c], out_dtype)
        start += c
    return dw


def _gemm_config(M, E, N, K, device, split):
    """(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages). mma.sync tiles (no TMA / wgmma
    assumptions), two stages and <= 32 KB of shared memory below sm80 (T4: 64 KB / SM)."""
    per_expert = M / max(E, 1)
    BM = 16 if per_expert <= 16 else (32 if per_expert <= 32 else 64)
    BK = 32
    if _capability(device) < (8, 0):
        BN = 64
        return BM, max(16, min(BN, _pow2(N))), max(16, min(BK, _pow2(K))), 4, 2
    BN = 128 if not split else 64
    if per_expert > 128 and not split:
        BM = 128
    return BM, max(16, min(BN, _pow2(N))), max(16, min(BK if BM == 128 else 64, _pow2(K))), 4 if BM < 128 else 8, 3


# Generic MoE path tiles (generic_gemm_config / generic_wgrad_config), from a sweep over rows per expert
# 16..2048 on Qwen3-30B-A3B, Qwen3.5-35B-A3B, Mixtral 8x7B, gpt-oss-20b, GLM-4.5-Air and DeepSeek-V2-Lite
# expert shapes (base gate_up / down fwd + dX, LoRA r=16 A / B fwd + dX + dW, base dW), bf16 + fp16,
# Colab A100 (sm80), L4 (sm89) and RTX PRO 6000 (sm120), torch 2.11 / Triton 3.6. Each entry is the
# config with the best geomean time over all shapes in its rows-per-expert bucket (1-4% off the per-shape
# best). Keys: "big" (N > 32 and K > 32), "N" (N <= 32: LoRA A forward, B dX, dA), "K" (K <= 32: LoRA B
# forward, A dX, dB). Rows: (max rows per expert or None, (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages
# [, GROUP_M])). GROUP_M > 0 walks the output GROUP_M row tiles at a time (L2 reuse), the main win
# above ~256 rows per expert. For the wgrad kernel BLOCK_M is the row (reduction) step.
_GENERIC_GEMM = {
    "sm80": {
        "big": ((16, (32, 256, 64, 8, 3, 0)), (128, (64, 256, 64, 8, 3, 0)), (None, (128, 128, 64, 4, 3, 8))),
        "N": ((32, (16, 16, 256, 4, 4, 0)), (None, (64, 16, 256, 4, 4, 0))),
        "K": ((128, (32, 256, 16, 4, 2, 0)), (None, (64, 256, 16, 8, 2, 0))),
    },
    "sm89": {
        "big": ((32, (32, 128, 128, 4, 3, 0)), (128, (128, 256, 64, 8, 3, 0)), (None, (128, 256, 64, 8, 3, 8))),
        "N": ((None, (32, 16, 256, 4, 4, 0)),),
        "K": ((32, (32, 256, 16, 4, 2, 0)), (64, (128, 128, 16, 4, 2, 0)), (None, (64, 256, 16, 8, 2, 0))),
    },
    "sm120": {
        "big": ((128, (32, 128, 64, 4, 3, 0)), (256, (128, 256, 64, 8, 3, 8)), (None, (128, 128, 32, 4, 4, 8))),
        "N": ((128, (32, 16, 256, 4, 3, 0)), (None, (64, 16, 128, 4, 3, 0))),
        "K": ((None, (32, 256, 16, 4, 2, 0)),),
    },
}
_GENERIC_WGRAD = {
    "sm80": {
        "big": ((256, (32, 128, 256, 8, 4)), (None, (64, 128, 256, 8, 3))),
        "N": ((None, (32, 16, 256, 4, 3)),),
        "K": ((None, (32, 256, 16, 4, 3)),),
    },
    "sm89": {
        "big": ((None, (32, 128, 256, 8, 3)),),
        "N": ((None, (32, 16, 256, 8, 3)),),
        "K": ((None, (32, 256, 16, 8, 3)),),
    },
    "sm120": {
        "big": ((64, (32, 128, 128, 8, 4)), (None, (32, 128, 256, 8, 4))),
        "N": ((None, (32, 16, 128, 4, 3)),),
        "K": ((None, (32, 128, 16, 4, 3)),),
    },
}
_SMEM = {}


def _generic_family(device):
    """Table for a device: measured sm80 / sm89 / sm120; sm86 / sm87 (~100 KB shared memory per block,
    like sm89) use sm89's, sm12x sm120's, sm90+ (>= 228 KB) sm80's."""
    cap = _capability(device)
    if cap[0] == 12:
        return "sm120"
    if cap[0] >= 9 or cap == (8, 0):
        return "sm80"
    return "sm89"


def _smem_limit(device):
    key = device.index
    lim = _SMEM.get(key)
    if lim is None:
        lim = 101376
        if device.type == "cuda":
            index = device.index if device.index is not None else torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            lim = getattr(props, "shared_memory_per_block_optin", 0) or lim
        _SMEM[key] = lim
    return lim


def _pick(table, rows):
    for hi, cfg in table:
        if hi is None or rows <= hi:
            return cfg
    return table[-1][1]


def _fit_smem(BM, BN, BK, stages, elt, limit, wgrad):
    """Triton keeps (stages - 1) pairs of operand tiles in shared memory (gemm: BM x BK + BK x BN,
    wgrad: BN x BM + BM x BK): drop stages, then halve BLOCK_K (gemm) or the BLOCK_M row step (wgrad)
    until they fit (ranks, dtypes or arches the sweep did not cover)."""
    while (stages - 1) * (BM * (BN + BK) if wgrad else BK * (BM + BN)) * elt > limit:
        if stages > 2:
            stages -= 1
        elif not wgrad and BK > 16:
            BK //= 2
        elif wgrad and BM > 16:
            BM //= 2
        else:
            break
    return BM, BK, stages


def generic_gemm_config(M, E, N, K, device, dtype):
    """(BLOCK_M, BLOCK_N, BLOCK_K, warps, stages, GROUP_M) for grouped_gemm(config = ...) on the generic
    MoE path: out[M, N] = a[M, K] @ b[e], M rows over E experts. Only shapes and the device, no counts."""
    if device.type == "cuda" and _capability(device) < (8, 0):
        return _gemm_config(M, E, N, K, device, False) + (0,)
    rows = M / max(E, 1)
    cls = "N" if N <= 32 else ("K" if K <= 32 else "big")
    BM, BN, BK, warps, stages, group_m = _pick(_GENERIC_GEMM[_generic_family(device)][cls], rows)
    BN = max(16, _pow2(N)) if cls == "N" else max(16, min(BN, _pow2(N)))
    BK = max(16, _pow2(K)) if cls == "K" else max(16, min(BK, _pow2(K)))
    elt = dtype.itemsize
    BM, BK, stages = _fit_smem(BM, BN, BK, stages, elt, _smem_limit(device), False)
    return BM, BN, BK, warps, stages, group_m


def generic_wgrad_config(M, E, N, K, device, dtype):
    """(BLOCK_M, BLOCK_N, BLOCK_K, warps, stages) for grouped_wgrad(config = ...) on the generic MoE
    path: dw[e] = g[rows of e].T @ x[rows of e] -> [E, N, K] with g [M, N], x [M, K]."""
    if device.type == "cuda" and _capability(device) < (8, 0):
        return 32, max(16, min(64, _pow2(N))), max(16, min(64, _pow2(K))), 4, 2
    rows = M / max(E, 1)
    cls = "N" if N <= 32 else ("K" if K <= 32 else "big")
    BM, BN, BK, warps, stages = _pick(_GENERIC_WGRAD[_generic_family(device)][cls], rows)
    BN = max(16, _pow2(N)) if cls == "N" else max(16, min(BN, _pow2(N)))
    BK = max(16, _pow2(K)) if cls == "K" else max(16, min(BK, _pow2(K)))
    elt = dtype.itemsize
    BM, BK, stages = _fit_smem(BM, BN, BK, stages, elt, _smem_limit(device), True)
    return BM, BN, BK, warps, stages


def row_pow2_scale(t, target_exp = 15):
    """Per-row power of two s with max|t[m]| * s in [2^(target-1), 2^target): fp16-safe,
    exactly invertible. Rows of zeros / non-finite rows get a finite scale."""
    amax = t.detach().abs().amax(1).float()
    _, ex = torch.frexp(amax)
    ex = ex.clamp(-100, 120)
    return torch.ldexp(torch.ones_like(amax), (target_exp - ex)).contiguous()


def grouped_gemm(a, b, counts, out_dtype, *, b_trans, e_lo = 0, e_hi = None, num_experts = None,
                 out = None, a_mode = None, row_scale = None, bias = None, config = None):
    """out[m] = a[m] @ (b[e - e_lo].T if b_trans else b[e - e_lo]) for rows of experts in [e_lo, e_hi).

    a: [M, K] rows sorted by expert; b: [E', N, K] (b_trans) or [E', K, N]; counts: [E] rows per
    expert on device. Rows outside the window are not written (pass `out` across windows)."""
    M, K = a.shape
    E = int(num_experts if num_experts is not None else _counts_tensor(counts).numel())
    e_hi = E if e_hi is None else int(e_hi)
    if b_trans:
        N = b.shape[1]
        assert b.shape[2] == K
        stride_bk, stride_bn = b.stride(2), b.stride(1)
    else:
        N = b.shape[2]
        assert b.shape[1] == K
        stride_bk, stride_bn = b.stride(1), b.stride(2)
    if out is None:
        out = torch.empty((M, N), dtype = out_dtype, device = a.device)
    if M == 0 or N == 0:
        return out
    if a_mode is None:
        a_mode = A_DIRECT if a.dtype == b.dtype else A_CAST
    ieee = a.dtype == torch.float32 and b.dtype == torch.float32
    if a_mode == A_DIRECT and a.dtype != b.dtype:
        raise TypeError(f"grouped_gemm: {a.dtype} x {b.dtype} needs a_mode cast / split")
    if use_cublas(a.device):
        with torch.autocast(device_type = a.device.type, enabled = False):
            _gemm_cublas(a, b, counts, out_dtype, b_trans, int(e_lo), e_hi, out, a_mode, row_scale, bias)
        CALLS["gemm"] += 1
        return out
    counts = _counts_tensor(counts)
    # config: (BM, BN, BK, warps, stages) or generic_gemm_config's (..., group_m), which also drops
    # the K masks when K % BK == 0. None: _gemm_config, the #1591 tiles.
    config = config or _gemm_config(M, E, N, K, a.device, a_mode == A_SPLIT)
    BM, BN, BK, warps, stages = config[:5]
    extra = {}
    grid = (-(-M // BM) + min(E, M), -(-N // BN))
    if len(config) > 5:
        extra = dict(GROUP_M = int(config[5]), EVEN_K = K % BK == 0)
        if config[5]:
            grid = (grid[0] * grid[1],)
    with _on(a.device):
        _grouped_gemm_kernel[grid](
            a, b, out, counts, row_scale if row_scale is not None else a, bias if bias is not None else a,
            E, int(e_lo), e_hi, N, K,
            a.stride(0), a.stride(1), b.stride(0), stride_bk, stride_bn, out.stride(0),
            bias.stride(0) if bias is not None else 0,
            A_MODE = a_mode, ROW_SCALE = row_scale is not None, HAS_BIAS = bias is not None, IEEE = ieee,
            E_POW2 = _pow2(max(E, 2)), BLOCK_M = BM, BLOCK_N = BN, BLOCK_K = BK,
            num_warps = warps, num_stages = stages, **extra,
        )
    CALLS["gemm"] += 1
    return out


def grouped_wgrad(g, x, counts, out_dtype, num_experts = None, config = None):
    """dw[e] = g[rows of e].T @ x[rows of e] -> [E, N, K]; zeros for an expert with no rows.
    config: None (the #1591 tile) or generic_wgrad_config's (BM, BN, BK, warps, stages)."""
    assert g.dtype == x.dtype and g.stride(1) == 1 and x.stride(1) == 1
    M, N = g.shape
    K = x.shape[1]
    E = int(num_experts if num_experts is not None else _counts_tensor(counts).numel())
    if use_cublas(g.device):
        CALLS["wgrad"] += 1
        with torch.autocast(device_type = g.device.type, enabled = False):
            return _wgrad_cublas(g, x, counts, out_dtype, E)
    counts = _counts_tensor(counts)
    ieee = g.dtype == torch.float32
    dw = torch.empty((E, N, K), dtype = out_dtype, device = g.device)
    if config is not None:
        BM, BN, BK, warps, stages = config
        with _on(g.device):
            _grouped_wgrad_kernel[(E, -(-N // BN), -(-K // BK))](
                g, x, dw, counts, E, N, K,
                g.stride(0), x.stride(0), dw.stride(0), dw.stride(1), dw.stride(2),
                IEEE = ieee, E_POW2 = _pow2(max(E, 2)), BLOCK_M = BM, BLOCK_N = BN, BLOCK_K = BK,
                num_warps = warps, num_stages = stages,
            )
        CALLS["wgrad"] += 1
        return dw
    BM = 32
    BN = max(16, min(64, _pow2(N)))
    BK = max(16, min(64, _pow2(K)))
    stages = 2 if _capability(g.device) < (8, 0) else 3
    with _on(g.device):
        _grouped_wgrad_kernel[(E, -(-N // BN), -(-K // BK))](
            g, x, dw, counts, E, N, K,
            g.stride(0), x.stride(0), dw.stride(0), dw.stride(1), dw.stride(2),
            IEEE = ieee, E_POW2 = _pow2(max(E, 2)), BLOCK_M = BM, BLOCK_N = BN, BLOCK_K = BK,
            num_warps = 4, num_stages = stages,
        )
    CALLS["wgrad"] += 1
    return dw


def _bwd_dx(g, w, counts, x_dtype, dy_mode):
    """dX = g @ w[e] for a frozen or trainable w ([E, N, K]); fp32 g against a 16-bit w is
    scaled per row before rounding (dy_mode: "scale" single dot, "split" hi + lo)."""
    g = g.contiguous()
    if g.dtype == w.dtype:
        return grouped_gemm(g, w, counts, x_dtype, b_trans = False)
    if g.dtype != torch.float32:
        g = g.float()
    scale = row_pow2_scale(g)
    mode = A_SPLIT if dy_mode == "split" else A_CAST
    return grouped_gemm(g, w, counts, x_dtype, b_trans = False, a_mode = mode, row_scale = scale)


class _GroupedLinear(torch.autograd.Function):
    """y = x @ w[e].T with grads for x and w (LoRA A / B stacks). x and w share a dtype."""

    @staticmethod
    def forward(ctx, x, w, counts, out_dtype):
        x = x.contiguous()
        w = w.contiguous()
        ctx.save_for_backward(x, w)
        ctx.counts = counts
        return grouped_gemm(x, w, counts, out_dtype, b_trans = True)

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        counts = ctx.counts
        g = g.contiguous()
        if g.dtype != x.dtype:
            g = g.to(x.dtype)   # what autograd of the per-expert matmul does to a cast output
        dx = dw = None
        if ctx.needs_input_grad[0]:
            dx = grouped_gemm(g, w, counts, x.dtype, b_trans = False)
        if ctx.needs_input_grad[1]:
            dw = grouped_wgrad(g, x, counts, w.dtype, num_experts = w.shape[0])
        return dx, dw, None, None


def grouped_linear(x, w, counts, out_dtype = None):
    return _GroupedLinear.apply(x, w, counts, out_dtype or x.dtype)


class _GroupedFrozenLinear(torch.autograd.Function):
    """y = x @ W[e].T + bias[e] for a frozen W from provider(lo, hi) -> [hi - lo, N, K]
    (rebuilt in backward, in expert windows when memory is short). x may be fp32 against a
    16-bit W (x_mode "cast" / "split"); the backward dX keeps x's dtype."""

    @staticmethod
    def forward(ctx, x, counts, provider, bias, out_dtype, x_mode, dy_mode):
        x = x.contiguous()
        M = x.shape[0]
        E = provider.num_experts
        out = torch.empty((M, provider.N), dtype = out_dtype, device = x.device)
        a_mode = None
        if x.dtype != provider.dtype:
            a_mode = A_SPLIT if x_mode == "split" else A_CAST
        pinned = []
        for lo, hi in provider.windows():
            w = provider(lo, hi)
            grouped_gemm(x, w, counts, out_dtype, b_trans = True, e_lo = lo, e_hi = hi, num_experts = E,
                         out = out, a_mode = a_mode, bias = bias)
            if provider.pin:
                pinned.append(w)
            del w
        ctx.provider = provider
        ctx.pinned = pinned
        ctx.x_dtype = x.dtype
        ctx.dy_mode = dy_mode
        ctx.counts = counts
        return out

    @staticmethod
    def backward(ctx, g):
        counts = ctx.counts
        provider = ctx.provider
        E = provider.num_experts
        g = g.contiguous()
        dx = torch.empty((g.shape[0], provider.K), dtype = ctx.x_dtype, device = g.device)
        scale = mode = None
        if g.dtype != provider.dtype:
            g = g.float()
            scale = row_pow2_scale(g)
            mode = A_SPLIT if ctx.dy_mode == "split" else A_CAST
        pinned = ctx.pinned
        for i, (lo, hi) in enumerate(provider.windows()):
            w = pinned[i] if pinned else provider(lo, hi)
            grouped_gemm(g, w, counts, ctx.x_dtype, b_trans = False, e_lo = lo, e_hi = hi, num_experts = E,
                         out = dx, a_mode = mode, row_scale = scale)
            del w
        ctx.pinned = None
        return dx, None, None, None, None, None, None


def grouped_frozen_linear(x, counts, provider, bias = None, out_dtype = None, x_mode = "cast", dy_mode = "scale"):
    return _GroupedFrozenLinear.apply(x, counts, provider, bias, out_dtype or x.dtype, x_mode, dy_mode)


_SELF_CHECKED = {}


def _self_check(device):
    """Compile and check every kernel variant once per device against fp64 on a small ragged
    problem (empty expert, tails); any failure disables the path for the process."""
    g = torch.Generator(device = "cpu").manual_seed(0)
    E, N, K = 3, 40, 24
    counts = torch.tensor([5, 0, 19], dtype = torch.int32, device = device)
    M = 24
    e_of = torch.repeat_interleave(torch.arange(E), counts.cpu().long()).to(device)
    w = (torch.randn(E, N, K, generator = g) * 0.1).to(device)
    x = torch.randn(M, K, generator = g).to(device)
    ref = torch.einsum("mk,mnk->mn", x.double(), w.double()[e_of])
    for xd, wd, od, mode in ((torch.float16, torch.float16, torch.float16, None),
                             (torch.float32, torch.float16, torch.float32, A_CAST),
                             (torch.float32, torch.float16, torch.float32, A_SPLIT),
                             (torch.float32, torch.float32, torch.float32, None)):
        y = grouped_gemm(x.to(xd), w.to(wd), counts, od, b_trans = True, a_mode = mode).double()
        tol = 1e-5 if wd == torch.float32 else 2e-2
        if not torch.allclose(y, ref, rtol = tol, atol = tol * 4):
            raise RuntimeError(f"grouped_gemm self-check {xd}/{wd}/{mode}: {(y - ref).abs().max().item()}")
    dw = grouped_wgrad(x.float(), x.float(), counts, torch.float32).double()
    ref_dw = torch.stack([x.double()[e_of == e].T @ x.double()[e_of == e] for e in range(E)])
    if not torch.allclose(dw, ref_dw, rtol = 1e-5, atol = 1e-4):
        raise RuntimeError(f"grouped_wgrad self-check: {(dw - ref_dw).abs().max().item()}")


def fp16_grouped_available(device) -> bool:
    if (
        triton is None
        or _DISABLED_REASON is not None
        or os.environ.get("UNSLOTH_GPTOSS_GROUPED_FP16", "1") == "0"
        or getattr(device, "type", None) != "cuda"
        or torch.version.hip is not None
        or os.environ.get("TRITON_INTERPRET", "0") == "1"
    ):
        return False
    key = device.index
    ok = _SELF_CHECKED.get(key)
    if ok is None:
        try:
            with torch.cuda.device(device):
                _self_check(device)
            ok = True
        except Exception as exc:
            if isinstance(exc, torch.OutOfMemoryError):
                raise
            _disable(exc)
            ok = False
        _SELF_CHECKED[key] = ok
    return ok


def unavailable_reason():
    if triton is None:
        return "Triton is not installed"
    if _DISABLED_REASON is not None:
        return f"float16 grouped GEMM disabled: {_DISABLED_REASON}"
    if os.environ.get("UNSLOTH_GPTOSS_GROUPED_FP16", "1") == "0":
        return "UNSLOTH_GPTOSS_GROUPED_FP16=0"
    if torch.version.hip is not None:
        return "float16 grouped GEMM is CUDA-only"
    return "float16 grouped GEMM unavailable on this device"
