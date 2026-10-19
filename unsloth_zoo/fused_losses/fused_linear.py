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

"""Fused `hidden @ lm_head.T` + token cross entropy / log-probabilities, one row chunk of logits at a time.

Each chunk is projected by cuBLAS into one reused buffer and reduced by a Triton row kernel (online logsumexp in
float32, plus the requested statistics). Where the gradient of a row is known in forward it is written over the
logits in place and projected back right away, so backward never recomputes the logits:

* cross entropy: d loss / d z is known up to the incoming scalar gradient;
* log-probs with a frozen head: d logp_i / d h_i = (onehot - softmax) @ W does not depend on the downstream loss,
  so forward keeps that vector per row and backward is `grad_logps[:, None] * u`.

A trainable head, a trainable bias or a differentiable entropy recompute the logits in backward instead, vocab slice
by vocab slice, accumulating the head gradient in float32. Ignored / masked rows never reach the head.

Everything data dependent runs inside opaque `torch.library` custom ops, so a compiled caller has no graph break.
Without Triton (CPU, MPS, no Triton wheel) the same ops run torch bodies with the same math.
"""

__all__ = [
    "fused_linear_cross_entropy",
    "fused_linear_logprobs",
    "fused_linear_rows",
    "CALLS",
]

import functools
import math
import os
import warnings
import torch
from typing import Optional

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = os.environ.get("TRITON_INTERPRET", "0") != "1"
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


_TRITON = os.environ.get("UNSLOTH_FUSED_LINEAR_TRITON", "1") != "0"
_FORWARD_GRAD = os.environ.get("UNSLOTH_LOGPROB_FORWARD_GRAD", "1") != "0"
# 1: fp32 logits where the fp32-output GEMM is fast (CUDA, torch >= 2.8), plain half logits elsewhere (ROCm: an
# fp32-output GEMM is 6-13x slower on gfx1151); 2: half logits + the exact fp32 target logit in the returned value
# (opt in: on gfx1151 its values sharpened the DPO loss but, paired with the half logits' Jacobian, made GRPO
# gradients 1.5x worse than plain half, which matched the old path exactly); 0: plain half logits.
_ACCURACY = os.environ.get("UNSLOTH_LOGPROB_FP32_LOGITS", "1")
# d logp / d hidden kept in the hidden dtype: an fp32 copy costs 4 * tokens * hidden bytes (335 MB at 32k tokens,
# which put DPO above TRL's peak) for ~1.25x less hidden-gradient error, already ~25x below TRL / stock logits.
_U_FP32 = os.environ.get("UNSLOTH_LOGPROB_U_FP32", "0") == "1"
# Recompute paths (trainable head / bias, differentiable entropy) read their logits twice and write a gradient in
# between. fp32 logits there (default, where mode 1 allows) keep the returned values equal to a no-grad pass of the
# same rows, so a policy agrees with its reference / old log-probs, and the gradient 3x closer than TRL; it costs
# 7-11% vs TRL on GDDR / A100 cards (G4, 8k tokens: GRPO + entropy 57.8 ms vs TRL 53.6; B200 at parity). 0 = half
# logits in both passes: TRL's speed and precision, but no longer bit-consistent with no-grad passes. (fp32 forward
# + half backward was tried: a half softmax under an fp32 normaliser made the GRPO + entropy gradient 3x worse than
# TRL.) Forward and backward read the same flag, so the saved lse matches the recomputed logits.
_RECOMPUTE_FP32 = os.environ.get("UNSLOTH_FUSED_LINEAR_RECOMPUTE_FP32", "1") != "0"
_CHUNK_GB = _env_float("UNSLOTH_LOGPROB_CHUNK_GB", 2.0)
# Smallest chunk, in logits: 512 rows of a 151936 vocabulary, and never under 512 rows. Fewer rows starve the
# K = vocab hidden-gradient GEMM (G4, 8k tokens, V 262144: DPO 98 ms at 256 rows vs 81 at 512, TRL 92); more only
# grow the transient. A small vocab takes more rows (32k: 2048, which the row sweep also found faster).
_MIN_LOGITS = 512 * 151936
_MIN_ROWS = int(_env_float("UNSLOTH_FUSED_LINEAR_MIN_ROWS", 0))
_MAX_ROWS = max(int(_env_float("UNSLOTH_FUSED_LINEAR_MAX_ROWS", 4096)), 1)
_BLOCK = int(_env_float("UNSLOTH_FUSED_LINEAR_BLOCK", 4096))
# Row statistics of a vocab wider than 2 splits run as (rows x splits of this many columns) programs plus a merge:
# one program per row left a 256-512 row chunk latency bound (B200 kernel time at V = 151936: 1.8 / 1.2 ms per 8k
# tokens vs 0.9 / 0.8 split; 262144: 2.9 / 2.0 vs 1.4 / 1.3; also 4-28% less at 4096 rows). 0 = one program per row.
_SPLIT_COLS = max(int(_env_float("UNSLOTH_FUSED_LINEAR_SPLIT_COLS", 16384)), 0)
_SPLIT_BLOCK = max(int(_env_float("UNSLOTH_FUSED_LINEAR_SPLIT_BLOCK", 2048)), 16)
# Recompute backward tiles (trainable head / bias, differentiable entropy): rows x vocab columns per GEMM. ROCm takes
# small tiles: on gfx1151 (GRPO + entropy, 4k tokens, V 151936) 4096 x 256 MB ran 609 ms vs 348 at 1024 x 8 MB
# (zoo main's recompute 360; at 1k tokens 89.5 vs 115.5); CUDA keeps TRL's shape, at or below TRL there.
_HIP_TILES = torch.version.hip is not None
_BWD_ROWS = max(int(_env_float("UNSLOTH_FUSED_LINEAR_BWD_ROWS", 1024 if _HIP_TILES else 4096)), 1)
_BWD_TILE_BYTES = max(int(_env_float("UNSLOTH_FUSED_LINEAR_BWD_TILE_MB", 8 if _HIP_TILES else 256) * 1024**2), 1)
_HALF = (torch.float16, torch.bfloat16)
# ROCm's half-in / fp32-out GEMM is a slow fallback (gfx1151: 6-14x a half-out GEMM).
_IS_HIP = torch.version.hip is not None

# Engagement counters (op calls and which body ran).
CALLS = {"ce": 0, "logprobs": 0, "logprobs_bwd": 0, "triton": 0, "torch": 0, "tiled": 0}


@functools.cache
def _budget_bytes(device):
    # Half the device's free memory read once per device (as the fused CE does), capped at UNSLOTH_LOGPROB_CHUNK_GB.
    try:
        from unsloth_zoo.tiled_mlp import _default_target_gb

        free = float(_default_target_gb(device))
    except Exception:
        free = 0.0
    gb = min(_CHUNK_GB, free) if free > 0 else _CHUNK_GB
    return max(int(gb * 1024**3), 1)


def _pow2(x):
    return 1 << max(int(math.ceil(math.log2(max(x, 1)))), 0)


def fused_linear_rows(n_rows, hidden_size, vocab_size, bytes_per_logit = 4, device = None):
    """Rows per chunk: the power of two at or below `n_rows * hidden / vocab` (a chunk's logits stay within the
    hidden states' own size, as Liger does), clamped to [min rows (by vocabulary, or UNSLOTH_FUSED_LINEAR_MIN_ROWS), 4096] and to the device
    budget. Rounding up instead doubled the transient at 32k tokens for ~4% speed."""
    rows = _pow2(n_rows * hidden_size / max(vocab_size, 1))
    if rows > n_rows * hidden_size / max(vocab_size, 1):
        rows //= 2
    min_rows = _MIN_ROWS
    if min_rows <= 0:
        min_rows = _pow2(_MIN_LOGITS / max(vocab_size, 1))
        if min_rows > _MIN_LOGITS / max(vocab_size, 1):
            min_rows //= 2
        min_rows = min(max(min_rows, 512), _MAX_ROWS)
    rows = min(max(rows, min_rows), _MAX_ROWS)
    cap = _budget_bytes(device) // max(vocab_size * bytes_per_logit, 1)
    rows = min(rows, max(cap, 2))
    return max(min(rows, max(n_rows, 1)), 1)


def _chunk_rows(rows, n_rows, hidden_size, vocab_size, bytes_per_logit, device):
    # rows > 0 sets the chunk, rows < 0 caps the automatic one at -rows, 0 is automatic.
    if rows > 0:
        return rows
    auto = fused_linear_rows(n_rows, hidden_size, vocab_size, bytes_per_logit, device)
    return min(auto, -rows) if rows < 0 else auto


@functools.cache
def _fp32_out_ok(device, dtype = torch.float16):
    # torch.mm(out_dtype = fp32) for half inputs: torch >= 2.8 on CUDA, and bf16 inputs only on sm >= 80 (a T4 raises
    # for bf16, which used to send the whole process to the torch path). Without it fp32 logits would need an fp32 GEMM.
    try:
        a = torch.ones((8, 8), device = device, dtype = dtype)
        return torch.mm(a, a, out_dtype = torch.float32).dtype == torch.float32
    except torch.OutOfMemoryError:
        raise
    except Exception:
        return False


def _logits_mode(device, dtype = torch.float16):
    # (fp32 logits buffer, exact fp32 target logit) for this device and compute dtype.
    if device.type != "cuda":
        return False, False
    if _ACCURACY == "0":
        return False, False
    if _ACCURACY == "2":
        return False, True
    if _IS_HIP or not _fp32_out_ok(device, dtype):
        return False, False
    return True, False


def _mm_fp32(a, b, out = None):
    """a @ b with an fp32 result, rounded once (CUDA half inputs); elsewhere an fp32 matmul of the upcast inputs."""
    if a.is_cuda and a.dtype in _HALF:
        if not _IS_HIP and _fp32_out_ok(a.device, a.dtype):
            return torch.mm(a, b, out_dtype = torch.float32, out = out) if out is not None \
                else torch.mm(a, b, out_dtype = torch.float32)
        # Half GEMM rounded once: an upcast fp32 GEMM would be several times slower.
        r = (a @ b).float()
    else:
        r = a.float() @ b.float() if a.dtype != torch.float64 else a @ b
    if out is not None:
        out.copy_(r)
        return out
    return r


def _addmm_fp32(acc, a, b):
    """acc += a @ b with fp32 accumulation."""
    if acc.dtype == torch.float32 and a.is_cuda and a.dtype in _HALF:
        if not _IS_HIP and _fp32_out_ok(a.device, a.dtype):
            torch.addmm(acc, a, b, out_dtype = torch.float32, out = acc)
        elif a.dtype == torch.bfloat16:
            acc.add_(a @ b)
        else:
            # fp16: one tile's / row chunk's partial can pass 65504 before the partials cancel in acc.
            acc.addmm_(a.float(), b.float())
        return
    if acc.dtype == a.dtype:
        acc.addmm_(a, b)
    else:
        acc.add_(a.to(acc.dtype) @ b.to(acc.dtype))


# ---------------------------------------------------------------------------------------------------------------
# Triton row kernels
# ---------------------------------------------------------------------------------------------------------------
if _HAS_TRITON:

    try:
        # libdevice tanh (CUDA and HIP): 2 sigmoid(2u) - 1 cancels for small u, so an fp32 softcapped logit lost
        # ~cap * 1e-7, 2.5x Inductor's error.
        from triton.language.extra import libdevice as _libdevice

        @triton.jit
        def _fl_tanh(x):
            return _libdevice.tanh(x)
    except Exception:
        @triton.jit
        def _fl_tanh(x):
            return 2.0 * tl.sigmoid(2.0 * x) - 1.0

    @triton.jit
    def _fl_transform(x, scale, cap, HAS_SCALE: tl.constexpr, HAS_CAP: tl.constexpr):
        # Returns (transformed logit before temperature, d transformed / d raw).
        d = tl.full(x.shape, 1.0, tl.float32)
        if HAS_SCALE:
            x = x * scale
            d = d * scale
        if HAS_CAP:
            t = _fl_tanh(x / cap)
            x = cap * t
            d = d * (1.0 - t * t)
        return x, d

    @triton.jit
    def _fl_forward_kernel(
        x_ptr, x_stride, n_cols,
        tgt_ptr, exact_ptr, bias_ptr,
        logp_ptr, lse_ptr, ent_ptr, sum_ptr, lssp_ptr, top1_ptr,
        inv_t, scale, cap,
        HAS_BIAS: tl.constexpr, HAS_SCALE: tl.constexpr, HAS_CAP: tl.constexpr, HAS_EXACT: tl.constexpr,
        WANT_ENT: tl.constexpr, WANT_SUM: tl.constexpr, WANT_LSSP: tl.constexpr, WANT_TOP1: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # Row statistics only. The gradient pass is a separate launch of _fl_backward_kernel, so these statistics are
        # bit-identical with and without a gradient (a policy and its no-grad reference agree exactly).
        row = tl.program_id(0).to(tl.int64)
        x_row = x_ptr + row * x_stride
        cols = tl.arange(0, BLOCK)
        tgt = tl.load(tgt_ptr + row).to(tl.int64)
        valid = (tgt >= 0) & (tgt < n_cols)
        tgt_c = tl.where(valid, tgt, 0)

        m = tl.full((), -float("inf"), tl.float32)
        s = tl.zeros((), tl.float32)
        a = tl.zeros((), tl.float32)  # sum e^(z - m) (z - m), for entropy
        q = tl.zeros((), tl.float32)  # sum e^(2 (z - m)), for log sum p^2
        tot = tl.zeros((), tl.float32)  # sum of transformed logits before temperature
        best = tl.full((), -float("inf"), tl.float32)
        best_i = tl.zeros((), tl.int64)
        for start in range(0, n_cols, BLOCK):
            offs = start + cols
            ok = offs < n_cols
            x = tl.load(x_row + offs, mask = ok, other = 0.0).to(tl.float32)
            if HAS_BIAS:
                x += tl.load(bias_ptr + offs, mask = ok, other = 0.0).to(tl.float32)
            x, _d = _fl_transform(x, scale, cap, HAS_SCALE, HAS_CAP)
            if WANT_SUM:
                tot += tl.sum(tl.where(ok, x, 0.0), axis = 0)
            z = tl.where(ok, x * inv_t, -float("inf"))
            bm = tl.max(z, axis = 0)
            if WANT_TOP1:
                bi = tl.argmax(z, axis = 0).to(tl.int64) + start
                take = bm > best
                best_i = tl.where(take, bi, best_i)
                best = tl.where(take, bm, best)
            m_new = tl.maximum(m, bm)
            r = tl.exp(m - m_new)
            e = tl.exp(z - m_new)
            if WANT_ENT:
                # Rebase sum e^(z - m)(z - m) onto m_new; nothing to rebase before the first block (s == 0, m = -inf).
                shift = tl.where(s > 0, (m - m_new) * s, 0.0)
                a = r * (a + shift) + tl.sum(tl.where(ok, e * (z - m_new), 0.0), axis = 0)
            if WANT_LSSP:
                q = q * r * r + tl.sum(e * e, axis = 0)
            s = s * r + tl.sum(e, axis = 0)
            m = m_new

        xt = tl.load(x_row + tgt_c).to(tl.float32)
        if HAS_BIAS:
            xt += tl.load(bias_ptr + tgt_c).to(tl.float32)
        xt, _d = _fl_transform(xt, scale, cap, HAS_SCALE, HAS_CAP)
        zt = xt * inv_t
        # The stored lse stays consistent with the logits the gradient pass reads, so its softmax sums to 1 (an
        # lse that mixes in the exact target logit biased every gradient: 2.5x the error on gfx1151).
        lse = m + tl.log(s)
        tl.store(lse_ptr + row, lse)
        if HAS_EXACT:
            # Only the returned value swaps the rounded target logit for the exact fp32 one, normaliser included.
            xe = tl.load(exact_ptr + row).to(tl.float32)
            xe, _d = _fl_transform(xe, scale, cap, HAS_SCALE, HAS_CAP)
            ze = xe * inv_t
            lse_value = m + tl.log(s + tl.exp(ze - m) - tl.exp(zt - m))
            tl.store(logp_ptr + row, tl.where(valid, ze - lse_value, 0.0))
        else:
            tl.store(logp_ptr + row, tl.where(valid, zt - lse, 0.0))
        if WANT_ENT:
            tl.store(ent_ptr + row, tl.log(s) - a / s)
        if WANT_SUM:
            tl.store(sum_ptr + row, tot)
        if WANT_LSSP:
            tl.store(lssp_ptr + row, tl.log(q) - 2.0 * tl.log(s))
        if WANT_TOP1:
            tl.store(top1_ptr + row, (best_i == tgt) & valid)


    @triton.jit
    def _fl_partial_kernel(
        x_ptr, x_stride, n_cols, n_rows, split_cols, stride_k, col0, split0, bias_ptr, tgt_ptr, part_ptr,
        inv_t, scale, cap,
        HAS_BIAS: tl.constexpr, HAS_SCALE: tl.constexpr, HAS_CAP: tl.constexpr,
        WANT_ENT: tl.constexpr, WANT_SUM: tl.constexpr, WANT_LSSP: tl.constexpr, WANT_TOP1: tl.constexpr,
        WANT_TGT: tl.constexpr, BLOCK: tl.constexpr,
    ):
        # Online softmax statistics of one row over one vocab split (columns split * split_cols ..), so a chunk of a
        # few hundred rows still fills the GPU; _fl_combine_kernel merges the splits in column order. The split
        # width depends only on the vocabulary, so a row's statistics do not depend on how many rows share a chunk.
        # x may be one vocab tile (global columns col0 ..) whose first split is global split split0; WANT_TGT then
        # also keeps the transformed target logit (part[7]) for the merge, since the tile is gone by then.
        row = tl.program_id(0).to(tl.int64)
        split = tl.program_id(1).to(tl.int64)
        x_row = x_ptr + row * x_stride
        cols = tl.arange(0, BLOCK)
        c0 = split * split_cols
        c1 = tl.minimum(c0 + split_cols, n_cols)
        m = tl.full((), -float("inf"), tl.float32)
        s = tl.zeros((), tl.float32)
        a = tl.zeros((), tl.float32)
        q = tl.zeros((), tl.float32)
        tot = tl.zeros((), tl.float32)
        best = tl.full((), -float("inf"), tl.float32)
        best_i = tl.zeros((), tl.int64)
        for start in range(c0, c1, BLOCK):
            offs = start + cols
            ok = offs < c1
            x = tl.load(x_row + offs, mask = ok, other = 0.0).to(tl.float32)
            if HAS_BIAS:
                x += tl.load(bias_ptr + offs, mask = ok, other = 0.0).to(tl.float32)
            x, _d = _fl_transform(x, scale, cap, HAS_SCALE, HAS_CAP)
            if WANT_SUM:
                tot += tl.sum(tl.where(ok, x, 0.0), axis = 0)
            z = tl.where(ok, x * inv_t, -float("inf"))
            bm = tl.max(z, axis = 0)
            if WANT_TOP1:
                bi = tl.argmax(z, axis = 0).to(tl.int64) + start + col0
                take = bm > best
                best_i = tl.where(take, bi, best_i)
                best = tl.where(take, bm, best)
            m_new = tl.maximum(m, bm)
            r = tl.exp(m - m_new)
            e = tl.exp(z - m_new)
            if WANT_ENT:
                shift = tl.where(s > 0, (m - m_new) * s, 0.0)
                a = r * (a + shift) + tl.sum(tl.where(ok, e * (z - m_new), 0.0), axis = 0)
            if WANT_LSSP:
                q = q * r * r + tl.sum(e * e, axis = 0)
            s = s * r + tl.sum(e, axis = 0)
            m = m_new
        # part[k, split, row] (stride_k = splits * rows): m, s, a, q, tot, best, best_i (as float: < 2^24 columns)
        p = part_ptr + (split0 + split) * n_rows + row
        if WANT_TGT:
            tgt = tl.load(tgt_ptr + row).to(tl.int64) - col0
            if (tgt >= c0) & (tgt < c1):
                xt = tl.load(x_row + tgt).to(tl.float32)
                if HAS_BIAS:
                    xt += tl.load(bias_ptr + tgt).to(tl.float32)
                xt, _d = _fl_transform(xt, scale, cap, HAS_SCALE, HAS_CAP)
                tl.store(p + 7 * stride_k, xt)
        tl.store(p, m)
        tl.store(p + stride_k, s)
        if WANT_ENT:
            tl.store(p + 2 * stride_k, a)
        if WANT_LSSP:
            tl.store(p + 3 * stride_k, q)
        if WANT_SUM:
            tl.store(p + 4 * stride_k, tot)
        if WANT_TOP1:
            tl.store(p + 5 * stride_k, best)
            tl.store(p + 6 * stride_k, best_i.to(tl.float32))

    @triton.jit
    def _fl_combine_kernel(
        x_ptr, x_stride, n_cols, n_rows, n_splits, stride_k, split_cols, part_ptr,
        tgt_ptr, exact_ptr, bias_ptr,
        logp_ptr, lse_ptr, ent_ptr, sum_ptr, lssp_ptr, top1_ptr,
        inv_t, scale, cap,
        HAS_BIAS: tl.constexpr, HAS_SCALE: tl.constexpr, HAS_CAP: tl.constexpr, HAS_EXACT: tl.constexpr,
        WANT_ENT: tl.constexpr, WANT_SUM: tl.constexpr, WANT_LSSP: tl.constexpr, WANT_TOP1: tl.constexpr,
        HAS_TGT: tl.constexpr, SPLITS: tl.constexpr,
    ):
        # Merges the per-split statistics of one row, then writes the same outputs as _fl_forward_kernel.
        row = tl.program_id(0).to(tl.int64)
        k = tl.arange(0, SPLITS)
        okk = k < n_splits
        p = part_ptr + k.to(tl.int64) * n_rows + row
        ms = tl.load(p, mask = okk, other = -float("inf"))
        m = tl.max(ms, axis = 0)
        w = tl.where(okk, tl.exp(ms - m), 0.0)
        ss = tl.load(p + stride_k, mask = okk, other = 0.0)
        s = tl.sum(ss * w, axis = 0)
        tgt = tl.load(tgt_ptr + row).to(tl.int64)
        valid = (tgt >= 0) & (tgt < n_cols)
        tgt_c = tl.where(valid, tgt, 0)
        if HAS_TGT:
            # Kept by the partial kernel of the split holding the target (an invalid target's slot is never written;
            # its value is masked below).
            xt = tl.load(part_ptr + 7 * stride_k + (tgt_c // split_cols) * n_rows + row)
        else:
            xt = tl.load(x_ptr + row * x_stride + tgt_c).to(tl.float32)
            if HAS_BIAS:
                xt += tl.load(bias_ptr + tgt_c).to(tl.float32)
            xt, _d = _fl_transform(xt, scale, cap, HAS_SCALE, HAS_CAP)
        zt = xt * inv_t
        lse = m + tl.log(s)
        tl.store(lse_ptr + row, lse)
        if HAS_EXACT:
            xe = tl.load(exact_ptr + row).to(tl.float32)
            xe, _d = _fl_transform(xe, scale, cap, HAS_SCALE, HAS_CAP)
            ze = xe * inv_t
            lse_value = m + tl.log(s + tl.exp(ze - m) - tl.exp(zt - m))
            tl.store(logp_ptr + row, tl.where(valid, ze - lse_value, 0.0))
        else:
            tl.store(logp_ptr + row, tl.where(valid, zt - lse, 0.0))
        if WANT_ENT:
            # Each split's sum e^(z - m_k)(z - m_k) rebased onto the row max m.
            aa = tl.load(p + 2 * stride_k, mask = okk, other = 0.0)
            a = tl.sum(w * (aa + tl.where(okk, (ms - m) * ss, 0.0)), axis = 0)
            tl.store(ent_ptr + row, tl.log(s) - a / s)
        if WANT_SUM:
            tl.store(sum_ptr + row, tl.sum(tl.load(p + 4 * stride_k, mask = okk, other = 0.0), axis = 0))
        if WANT_LSSP:
            qq = tl.load(p + 3 * stride_k, mask = okk, other = 0.0)
            q = tl.sum(qq * w * w, axis = 0)
            tl.store(lssp_ptr + row, tl.log(q) - 2.0 * tl.log(s))
        if WANT_TOP1:
            bb = tl.load(p + 5 * stride_k, mask = okk, other = -float("inf"))
            bi = tl.load(p + 6 * stride_k, mask = okk, other = 0.0)
            top = tl.max(bb, axis = 0)
            # First index among the splits holding the maximum: ties go to the earliest column, as torch.argmax.
            first = tl.min(tl.where(bb == top, bi, float(2 ** 30)), axis = 0)
            tl.store(top1_ptr + row, (first.to(tl.int64) == tgt) & valid)

    @triton.jit
    def _fl_backward_kernel(
        x_ptr, x_stride, dz_ptr, dz_stride, n_cols, col0, n_total,
        tgt_ptr, bias_ptr, lse_ptr, ent_ptr, g_ptr, g_stride, ge_ptr,
        inv_t, scale, cap, smooth_keep, smooth_each,
        HAS_BIAS: tl.constexpr, HAS_SCALE: tl.constexpr, HAS_CAP: tl.constexpr, HAS_ENT_GRAD: tl.constexpr,
        ALIASED: tl.constexpr, BLOCK: tl.constexpr,
    ):
        # Write d loss / d raw logits of a [rows, n_cols] slice (vocab columns col0 + j) into dz, which is either x
        # itself or (fp32 logits) a half view of the same row's bytes:
        #   dz = g * (keep * onehot + each - p) - ge * p * (z - lse + H), chained through temperature and transforms.
        # g is per row (g_stride 1: recompute backward) or one scalar (g_stride 0: CE's -scaling / divisor, or the
        # forward-gradient -1 of d logp / d z).
        row = tl.program_id(0).to(tl.int64)
        x_row = x_ptr + row * x_stride
        dz_row = dz_ptr + row * dz_stride
        cols = tl.arange(0, BLOCK)
        tgt = tl.load(tgt_ptr + row).to(tl.int64)
        valid = (tgt >= 0) & (tgt < n_total)
        lse = tl.load(lse_ptr + row)
        g = tl.where(valid, tl.load(g_ptr + row * g_stride).to(tl.float32), 0.0)
        if HAS_ENT_GRAD:
            ge = tl.where(valid, tl.load(ge_ptr + row).to(tl.float32), 0.0)
            ent = tl.load(ent_ptr + row)
        for start in range(0, n_cols, BLOCK):
            offs = start + cols
            ok = offs < n_cols
            x = tl.load(x_row + offs, mask = ok, other = 0.0).to(tl.float32)
            if ALIASED:
                # Same hazard as the forward's in-place write: all loads of this block land before any store.
                tl.debug_barrier()
            if HAS_BIAS:
                # bias_ptr is already this slice's bias (vocab columns col0..), as is x.
                x += tl.load(bias_ptr + offs, mask = ok, other = 0.0).to(tl.float32)
            x, d = _fl_transform(x, scale, cap, HAS_SCALE, HAS_CAP)
            z = x * inv_t
            p = tl.exp(z - lse)
            dz = g * (tl.where(offs + col0 == tgt, smooth_keep, 0.0) + smooth_each - p)
            if HAS_ENT_GRAD:
                dz -= ge * p * (z - lse + ent)
            dz = dz * inv_t * d
            tl.store(dz_row + offs, dz.to(dz_ptr.dtype.element_ty), mask = ok)


    @triton.jit
    def _fl_scale_kernel(x_ptr, g_ptr, out_ptr, n, cols, PER_ROW: tl.constexpr, BLOCK: tl.constexpr):
        # out = x * g (one value, or g[row] of a [rows, cols] x) in fp32, rounded once into out: one read and one write
        # of x, where a chunked torch upcast moved it three times (trainable CE: 4.5 vs 1.7 ms per backward).
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        ok = offs < n
        x = tl.load(x_ptr + offs, mask = ok, other = 0.0).to(tl.float32)
        if PER_ROW:
            g = tl.load(g_ptr + offs // cols, mask = ok, other = 0.0)
        else:
            g = tl.load(g_ptr)
        tl.store(out_ptr + offs, (x * g).to(out_ptr.dtype.element_ty), mask = ok)


_TRITON_BROKEN = set()


def _use_triton(device):
    return _TRITON and _HAS_TRITON and device.type == "cuda" and device not in _TRITON_BROKEN


def _launch(device, fn):
    with torch.cuda.device(device):
        return fn()


def _num_warps():
    return 8 if _IS_HIP else 16


def _split_block():
    return min(triton.next_power_of_2(_SPLIT_BLOCK), triton.next_power_of_2(max(_SPLIT_COLS, 16)))


# ---------------------------------------------------------------------------------------------------------------
# torch bodies (no Triton), same math in float32 (float64 for float64 inputs)
# ---------------------------------------------------------------------------------------------------------------
def _stat_dtype(dtype):
    return torch.float64 if dtype == torch.float64 else torch.float32


def _torch_transform(x, scale, cap):
    d = None
    if scale != 1.0:
        x = x * scale
    if cap != 0.0:
        t = torch.tanh(x / cap)
        x = cap * t
        d = 1.0 - t * t
    return x, d


def _torch_forward(x, bias, tgt, inv_t, scale, cap, exact, want):
    """Row statistics of raw logits x [R, V] (any dtype). want = (ent, sum, lssp, top1)."""
    sd = _stat_dtype(x.dtype)
    x = x.to(sd)
    if bias is not None:
        x = x + bias.to(sd)
    xt, _ = _torch_transform(x, scale, cap)
    z = xt * inv_t
    lse = torch.logsumexp(z, dim = -1)
    V = z.shape[-1]
    valid = (tgt >= 0) & (tgt < V)
    tc = torch.where(valid, tgt, torch.zeros_like(tgt))
    zt = z.gather(-1, tc.unsqueeze(-1)).squeeze(-1)
    lse_value = lse
    if exact is not None:
        # As the kernel: only the returned value uses the exact target logit; lse stays consistent for gradients.
        ze = _torch_transform(exact.to(sd), scale, cap)[0] * inv_t
        m = z.max(-1).values
        lse_value = m + torch.log(torch.exp(lse - m) + torch.exp(ze - m) - torch.exp(zt - m))
        zt = ze
    out = {"lse": lse.float(), "logp": torch.where(valid, zt - lse_value, torch.zeros_like(lse)).float()}
    want_ent, want_sum, want_lssp, want_top1 = want
    logp_all = None
    if want_ent or want_lssp:
        logp_all = z - lse.unsqueeze(-1)
    if want_ent:
        out["ent"] = (-(logp_all.exp() * logp_all).sum(-1)).float()
    if want_sum:
        out["sum"] = xt.sum(-1).float()
    if want_lssp:
        out["lssp"] = torch.logsumexp(2 * logp_all, dim = -1).float()
    if want_top1:
        out["top1"] = (z.argmax(-1) == tgt) & valid
    return out, z, lse, valid


def _torch_dz(x, bias, tgt, lse, inv_t, scale, cap, coef_row, smooth_keep, smooth_each, col0 = 0,
              ge = None, ent = None, n_total = None):
    """d loss / d raw logits for raw logits x [R, Vs] covering vocab columns col0..col0 + Vs.
    dz = coef_row * (p - keep onehot - each) - ge * p * (z - lse + ent), chained through temperature and transforms."""
    sd = _stat_dtype(x.dtype)
    x = x.to(sd)
    if bias is not None:
        x = x + bias.to(sd)
    xt, d = _torch_transform(x, scale, cap)
    z = xt * inv_t
    p = torch.exp(z - lse.to(sd).unsqueeze(-1))
    n_total = z.shape[-1] if n_total is None else n_total
    valid = (tgt >= 0) & (tgt < n_total)
    cols = torch.arange(col0, col0 + z.shape[-1], device = z.device)
    onehot = (cols.unsqueeze(0) == tgt.unsqueeze(-1)).to(sd)
    coef = torch.where(valid, coef_row.to(sd), torch.zeros_like(coef_row, dtype = sd))
    dz = coef.unsqueeze(-1) * (p - smooth_keep * onehot - smooth_each)
    if ge is not None:
        gev = torch.where(valid, ge.to(sd), torch.zeros_like(ge, dtype = sd))
        dz = dz - gev.unsqueeze(-1) * p * (z - lse.to(sd).unsqueeze(-1) + ent.to(sd).unsqueeze(-1))
    dz = dz * inv_t
    if d is not None:
        dz = dz * d
    if scale != 1.0:
        dz = dz * scale
    return dz


# ---------------------------------------------------------------------------------------------------------------
# Chunk engine
# ---------------------------------------------------------------------------------------------------------------
class _Buffers:
    """One logits buffer per call, reused by every chunk; odd vocab rows padded to a multiple of 8 for cuBLAS."""

    def __init__(self, rows, n_cols, device, fp32, compute_dtype):
        self.n_cols = n_cols
        self.fp32 = fp32
        self.stride = -(-n_cols // 8) * 8
        self.dtype = torch.float32 if fp32 else compute_dtype
        self.buf = torch.empty((rows, self.stride), device = device, dtype = self.dtype)

    def logits(self, n):
        return self.buf[:n, : self.n_cols]

    def grad_view(self, n, half_dtype):
        # fp32 logits: the half gradient of row i is written into the first half of row i's own bytes.
        if self.fp32:
            return self.buf.view(half_dtype)[:n, : self.n_cols]
        return self.buf[:n, : self.n_cols]


def _aligned(V):
    # cuBLAS drops to a slow non-vectorised kernel when the vocab dimension is not a multiple of 8 (an sm75 cutlass
    # kernel on B200 at V = 32001): run the aligned columns as one GEMM and the <= 7 left over as a tiny one.
    V8 = V - V % 8
    return V8 if 0 < V8 < V else V


def _logits_into(h, w, out, fp32_logits):
    """out[n, V] = h @ w.T (fp32 result when fp32_logits)."""
    V8 = _aligned(w.shape[0])
    parts = ((0, V8), (V8, w.shape[0])) if V8 < w.shape[0] else ((0, V8),)
    for a, b in parts:
        if fp32_logits:
            _mm_fp32(h, w[a:b].t(), out = out[:, a:b])
        else:
            torch.mm(h, w[a:b].t(), out = out[:, a:b])
    return out


def _project(h, w, buf, n, fp32_logits):
    return _logits_into(h, w, buf.logits(n), fp32_logits)


def _dz_w(dz, w):
    """dz[n, V] @ w[V, H] with an fp32 result rounded once."""
    V8 = _aligned(w.shape[0])
    out = _mm_fp32(dz[:, :V8], w[:V8])
    if V8 < w.shape[0]:
        _addmm_fp32(out, dz[:, V8:], w[V8:])
    return out


def _dz_w_acc(acc, dz, w):
    """acc[n, H] (fp32) += dz[n, V] @ w[V, H], in place."""
    V8 = _aligned(w.shape[0])
    _addmm_fp32(acc, dz[:, :V8], w[:V8])
    if V8 < w.shape[0]:
        _addmm_fp32(acc, dz[:, V8:], w[V8:])


def _dz_t_h(acc, dz, h):
    """acc[V, H] += dz[n, V].T @ h."""
    V8 = _aligned(dz.shape[1])
    _addmm_fp32(acc[:V8], dz[:, :V8].t(), h)
    if V8 < dz.shape[1]:
        _addmm_fp32(acc[V8:], dz[:, V8:].t(), h)


def _exact_target(h, w, bias, tgt):
    # fp32 dot product of each row with its target's head row (+ bias), O(rows * hidden).
    V = w.shape[0]
    tc = tgt.clamp(0, V - 1)
    x = (h.float() * w.index_select(0, tc).float()).sum(-1)
    if bias is not None:
        x = x + bias.float().index_select(0, tc)
    return x


def _row_forward(device, x, dz_out, tgt, bias, exact, coef, scalars, want, write_grad, outs = None):
    """Statistics of a chunk of raw logits x [n, V]; with write_grad also writes dz into dz_out (may alias x).
    `outs` optionally gives contiguous [n] destinations per statistic, written directly by the kernel."""
    inv_t, scale, cap, smooth_keep, smooth_each = scalars
    n, V = x.shape
    want_ent, want_sum, want_lssp, want_top1 = want
    outs = outs or {}
    if _use_triton(device) and n > 0:
        f32 = dict(device = device, dtype = torch.float32)
        logp = outs.get("logp") if outs.get("logp") is not None else torch.empty(n, **f32)
        lse = outs.get("lse") if outs.get("lse") is not None else torch.empty(n, **f32)
        ent = (outs.get("ent") if outs.get("ent") is not None else torch.empty(n, **f32)) if want_ent else lse
        tot = (outs.get("sum") if outs.get("sum") is not None else torch.empty(n, **f32)) if want_sum else lse
        lssp = (outs.get("lssp") if outs.get("lssp") is not None else torch.empty(n, **f32)) if want_lssp else lse
        top1 = (outs.get("top1") if outs.get("top1") is not None else torch.empty(n, device = device, dtype = torch.bool)) \
            if want_top1 else logp
        block = min(_BLOCK, triton.next_power_of_2(V))
        splits = -(-V // _SPLIT_COLS) if _SPLIT_COLS > 0 else 1
        if splits <= 2:
            splits = 1  # a 32k vocab: the merge costs more than the extra parallelism gains

        def run_split():
            part = torch.empty((7, splits, n), **f32)
            flags = dict(HAS_BIAS = bias is not None, HAS_SCALE = scale != 1.0, HAS_CAP = cap != 0.0,
                         WANT_ENT = want_ent, WANT_SUM = want_sum, WANT_LSSP = want_lssp, WANT_TOP1 = want_top1)
            scal = (float(inv_t), float(scale), float(cap) if cap != 0.0 else 1.0)
            _fl_partial_kernel[(n, splits)](
                x, x.stride(0), V, n, _SPLIT_COLS, splits * n, 0, 0, bias if bias is not None else lse, tgt, part,
                *scal, WANT_TGT = False, BLOCK = _split_block(), num_warps = 4, **flags,
            )
            _fl_combine_kernel[(n,)](
                x, x.stride(0), V, n, splits, splits * n, _SPLIT_COLS, part,
                tgt, exact if exact is not None else lse, bias if bias is not None else lse,
                logp, lse, ent, tot, lssp, top1, *scal,
                HAS_EXACT = exact is not None, HAS_TGT = False, SPLITS = max(triton.next_power_of_2(splits), 2),
                num_warps = 1, **flags,
            )

        def run():
            if splits > 1:
                return run_split()
            _fl_forward_kernel[(n,)](
                x, x.stride(0), V,
                tgt, exact if exact is not None else lse, bias if bias is not None else lse,
                logp, lse, ent, tot, lssp, top1,
                float(inv_t), float(scale), float(cap) if cap != 0.0 else 1.0,
                HAS_BIAS = bias is not None, HAS_SCALE = scale != 1.0, HAS_CAP = cap != 0.0,
                HAS_EXACT = exact is not None,
                WANT_ENT = want_ent, WANT_SUM = want_sum, WANT_LSSP = want_lssp, WANT_TOP1 = want_top1,
                BLOCK = block, num_warps = 8,
            )
        _launch(device, run)
        CALLS["triton"] += 1
        out = {"logp": logp, "lse": lse}
        if want_ent: out["ent"] = ent
        if want_sum: out["sum"] = tot
        if want_lssp: out["lssp"] = lssp
        if want_top1: out["top1"] = top1
        dz = None
        if write_grad:
            # dz = coef * (p - keep onehot - each): the gradient kernel with the scalar g = -coef.
            dz = _row_backward(device, x, dz_out, tgt, bias, lse, None, -coef.reshape(()), None,
                               (inv_t, scale, cap), 0, V, (smooth_keep, smooth_each))
        return out, dz
    CALLS["torch"] += 1
    out, z, lse, valid = _torch_forward(x, bias, tgt, inv_t, scale, cap, exact, want)
    for key, dst in outs.items():
        if dst is not None and key in out:
            dst.copy_(out[key])
            out[key] = dst
    dz = None
    if write_grad:
        dz = _row_backward(device, x, dz_out, tgt, bias, out["lse"], None, -coef.reshape(()), None,
                           (inv_t, scale, cap), 0, V, (smooth_keep, smooth_each))
    return out, dz


def _tiled_forward(device, hidden, idx, n_keep, w, bias, targets, exact_mode, scalars, want, fp32_logits,
                   compute_dtype, outs, cap_rows):
    """Row statistics of every kept row from TRL-shaped tiles (up to 4096 rows x a vocab tile), merged over the
    vocab splits. For the recompute paths, which keep nothing per row but the statistics: a 4096-row logits GEMM
    instead of the few-hundred-row full-vocab chunks the forward-gradient path needs (16% faster GEMM on B200)."""
    inv_t, scale, cap = scalars
    want_ent, want_sum, want_lssp, want_top1 = want
    N, H = hidden.shape
    V = w.shape[0]
    f32 = dict(device = device, dtype = torch.float32)
    bpl = 4 if fp32_logits else 2
    R = max(min(_BWD_ROWS, n_keep, cap_rows if cap_rows > 0 else n_keep), 1)
    tile = max(_BWD_TILE_BYTES // max(R * bpl, 1), _SPLIT_COLS)
    tile = min(tile - tile % _SPLIT_COLS, -(-V // _SPLIT_COLS) * _SPLIT_COLS)
    splits = -(-V // _SPLIT_COLS)
    buf = _Buffers(R, tile, device, fp32_logits, compute_dtype)
    flags = dict(HAS_BIAS = bias is not None, HAS_SCALE = scale != 1.0, HAS_CAP = cap != 0.0,
                 WANT_ENT = want_ent, WANT_SUM = want_sum, WANT_LSSP = want_lssp, WANT_TOP1 = want_top1)
    scal = (float(inv_t), float(scale), float(cap) if cap != 0.0 else 1.0)
    for a in range(0, n_keep, R):
        b = min(a + R, n_keep)
        n = b - a
        h = _rows_of(hidden, idx, a, b, n_keep, N).to(compute_dtype)
        t = targets[a:b] if n_keep == N else targets.index_select(0, idx[a:b])
        part = torch.empty((8, splits, n), **f32)
        for first in range(0, V, tile):
            last = min(first + tile, V)
            x = _logits_into(h, w[first:last], buf.buf[:n, : last - first], fp32_logits)
            bias_t = bias[first:last] if bias is not None else None

            def run():
                _fl_partial_kernel[(n, -(-(last - first) // _SPLIT_COLS))](
                    x, x.stride(0), last - first, n, _SPLIT_COLS, splits * n, first, first // _SPLIT_COLS,
                    bias_t if bias_t is not None else part, t, part, *scal,
                    WANT_TGT = True, BLOCK = _split_block(), num_warps = 4, **flags,
                )
            _launch(device, run)
        exact = _exact_target(h, w, bias, t) if exact_mode else None
        o = {k: outs[k][a:b] for k in ("logp", "lse", "ent", "sum", "lssp", "top1") if outs.get(k) is not None}
        lse = o["lse"]

        def run():
            _fl_combine_kernel[(n,)](
                part, 0, V, n, splits, splits * n, _SPLIT_COLS, part,
                t, exact if exact is not None else lse, lse,
                o["logp"], lse, o.get("ent", lse), o.get("sum", lse), o.get("lssp", lse), o.get("top1", o["logp"]),
                *scal, HAS_EXACT = exact is not None, HAS_TGT = True,
                SPLITS = max(triton.next_power_of_2(splits), 2), num_warps = 1, **flags,
            )
        _launch(device, run)
        CALLS["triton"] += 1
    CALLS["tiled"] += 1


def _row_backward(device, x, dz_out, tgt, bias_slice, lse, ent, g, ge, scalars, col0, n_total, smooth = (1.0, 0.0)):
    """Writes d loss / d raw logits of x [n, Vs] (vocab columns col0..) into dz_out (x itself, or a half view of the
    same bytes); returns dz_out. g is per row [n] or one 0-dim coefficient for every row."""
    inv_t, scale, cap = scalars
    smooth_keep, smooth_each = smooth
    n, Vs = x.shape
    if _use_triton(device) and n > 0:
        block = min(_BLOCK, triton.next_power_of_2(Vs))

        def run():
            _fl_backward_kernel[(n,)](
                x, x.stride(0), dz_out, dz_out.stride(0), Vs, col0, n_total,
                tgt, bias_slice if bias_slice is not None else lse, lse, ent if ent is not None else lse,
                g, 0 if g.dim() == 0 else 1, ge if ge is not None else g,
                float(inv_t), float(scale), float(cap) if cap != 0.0 else 1.0, float(smooth_keep), float(smooth_each),
                HAS_BIAS = bias_slice is not None, HAS_SCALE = scale != 1.0, HAS_CAP = cap != 0.0,
                HAS_ENT_GRAD = ge is not None, ALIASED = dz_out.dtype != x.dtype, BLOCK = block,
                num_warps = _num_warps(),
            )
        _launch(device, run)
        CALLS["triton"] += 1
        return dz_out
    CALLS["torch"] += 1
    dz = _torch_dz(x, bias_slice, tgt, lse, inv_t, scale, cap, -g, smooth_keep, smooth_each, col0 = col0, ge = ge,
                   ent = ent, n_total = n_total)
    dz_out.copy_(dz.to(dz_out.dtype))
    return dz_out


def _scalars(temperature, mul, div, cap):
    scale = 1.0
    if mul != 0.0:
        scale *= mul
    if div != 0.0:
        scale /= div
    return 1.0 / temperature, scale, cap


def _keep_index(targets, mask, vocab_size, ignore_index):
    keep = (targets >= 0) & (targets < vocab_size)
    if ignore_index is not None:
        keep &= targets != ignore_index
    if mask is not None:
        keep &= mask.to(torch.bool)
    idx = keep.nonzero().squeeze(-1)
    return idx, int(idx.numel())


def _rows_of(x, idx, a, b, n_keep, n_all):
    if n_keep == n_all:
        return x[a:b]
    return x.index_select(0, idx[a:b])


def _guarded(fn):
    """Runs the Triton body; a Triton failure that is not OOM / dynamo falls back to torch on that device."""
    @functools.wraps(fn)
    def wrapper(hidden, *args, **kwargs):
        device = args[0].device  # weight
        if not _use_triton(device):
            return fn(hidden, *args, **kwargs)
        err = None
        # Two tries: one transient JIT failure (a race on a shared Triton cache) must not send a whole run to the
        # slower torch bodies.
        for _ in range(2):
            try:
                return fn(hidden, *args, **kwargs)
            except torch.OutOfMemoryError:
                raise
            except Exception as e:
                if type(e).__module__.startswith("torch._dynamo"):
                    raise
                err = e
        _TRITON_BROKEN.add(device)
        try:
            out = fn(hidden, *args, **kwargs)
        except BaseException:
            # The torch body fails as well: the inputs are at fault, not Triton, so keep Triton for later calls.
            _TRITON_BROKEN.discard(device)
            raise
        warnings.warn(f"Unsloth: fused linear Triton kernel failed on {device} ({type(err).__name__}: {err}); "
                      f"using the torch path.", stacklevel = 2)
        return out
    return wrapper


# ---------------------------------------------------------------------------------------------------------------
# Op bodies
# ---------------------------------------------------------------------------------------------------------------
@_guarded
def _ce_impl(hidden, weight, bias, targets, divisor, scaling, ignore_index, mul, div, cap, label_smoothing,
             logits_bias, compute_dtype, need_h, need_w, need_b, fp32_head_grad, rows):
    device = weight.device
    N, H = hidden.shape
    V = weight.shape[0]
    targets = targets.contiguous()
    idx, n_keep = _keep_index(targets, None, V, ignore_index)
    f32 = dict(device = device, dtype = torch.float32)
    grad_h = torch.zeros((N, H), device = device, dtype = hidden.dtype) if need_h else hidden.new_empty(0)
    acc_w = None
    if need_w:
        acc_w = torch.zeros(weight.shape, device = device,
                            dtype = torch.float32 if fp32_head_grad and weight.dtype in _HALF else weight.dtype)
    acc_b = torch.zeros(V, **f32) if need_b else None
    loss = torch.zeros((), **f32)
    inv_t, scale, cap = _scalars(1.0, mul, div, cap)
    eps = float(label_smoothing)
    full_bias = None
    if bias is not None or logits_bias is not None:
        full_bias = torch.zeros(V, **f32)
        if bias is not None: full_bias += bias.float()
        if logits_bias is not None: full_bias += logits_bias.float()
    if n_keep > 0:
        grad = need_h or need_w or need_b
        fp32_logits, exact_mode = _logits_mode(device, compute_dtype)
        if compute_dtype not in _HALF:
            fp32_logits, exact_mode = False, False
        R = _chunk_rows(rows, n_keep, H, V, 4 if fp32_logits else 2, device)
        R = min(R, n_keep)
        w = weight.to(compute_dtype)
        buf = _Buffers(R, V, device, fp32_logits, compute_dtype)
        coef = (scaling.to(**f32) / divisor.to(**f32)).reshape(())
        # Per-row statistics land straight in these; the loss is summed once at the end, not per chunk.
        lp_all = torch.empty(n_keep, **f32)
        lse_all = torch.empty(n_keep, **f32) if eps != 0.0 else None
        sum_all = torch.empty(n_keep, **f32) if eps != 0.0 else None
        with torch.autocast(device.type, enabled = False):
            for a in range(0, n_keep, R):
                b = min(a + R, n_keep)
                n = b - a
                h = _rows_of(hidden, idx, a, b, n_keep, N).to(compute_dtype)
                t = targets[a:b] if n_keep == N else targets.index_select(0, idx[a:b])
                x = _project(h, w, buf, n, fp32_logits)
                exact = _exact_target(h, w, full_bias, t) if exact_mode else None
                dz_out = buf.grad_view(n, compute_dtype) if grad else None
                outs = {"logp": lp_all[a:b]}
                if eps != 0.0:
                    outs["lse"], outs["sum"] = lse_all[a:b], sum_all[a:b]
                st, dz = _row_forward(
                    device, x, dz_out, t, full_bias, exact, coef, (inv_t, scale, cap, 1.0 - eps, eps / V),
                    (False, eps != 0.0, False, False), grad, outs,
                )
                if grad:
                    if need_h:
                        gh = _dz_w(dz, w)
                        if n_keep == N: grad_h[a:b].copy_(gh)
                        else: grad_h.index_copy_(0, idx[a:b], gh.to(grad_h.dtype))
                    if need_w:
                        _dz_t_h(acc_w, dz, h)
                    if need_b:
                        acc_b += dz.sum(0, dtype = torch.float32)
        # The [n] per-row values are summed in float64 (not on MPS, which has none): an fp32 sum of thousands of
        # ~log(V) terms rounds the scalar loss by an ulp or more.
        acc = torch.float64 if device.type in ("cuda", "cpu") else torch.float32
        if eps != 0.0:
            nll = (1.0 - eps) * (-lp_all.to(acc)) + eps * (lse_all.to(acc) - sum_all.to(acc) * (inv_t / V))
        else:
            nll = -lp_all.to(acc)
        loss = (nll.sum() / divisor.to(acc)).float()
    grad_w = acc_w.to(weight.dtype) if need_w else hidden.new_empty(0)
    grad_b = acc_b.to(bias.dtype) if need_b else hidden.new_empty(0)
    return loss, grad_h, grad_w, grad_b


@_guarded
def _logprobs_impl(hidden, weight, bias, targets, mask, temperature, mul, div, cap, compute_dtype,
                   want_ent, want_mean, want_lssp, want_top1, want_u, rows, tiled = False):
    device = weight.device
    N, H = hidden.shape
    V = weight.shape[0]
    targets = targets.contiguous()
    bias = bias.contiguous() if bias is not None else None
    idx, n_keep = _keep_index(targets, mask, V, None)
    f32 = dict(device = device, dtype = torch.float32)
    logps = torch.zeros(N, **f32)
    lse = torch.zeros(N, **f32)
    ent = torch.zeros(N, **f32) if want_ent else logps.new_empty(0)
    mean = torch.zeros(N, **f32) if want_mean else logps.new_empty(0)
    lssp = torch.zeros(N, **f32) if want_lssp else logps.new_empty(0)
    top1 = torch.zeros(N, device = device, dtype = torch.bool) if want_top1 else torch.empty(0, device = device,
                                                                                               dtype = torch.bool)
    u_dtype = torch.float32 if _U_FP32 and hidden.dtype in _HALF else hidden.dtype
    u = torch.zeros((N, H), device = device, dtype = u_dtype) if want_u else hidden.new_empty(0)
    inv_t, scale, cap = _scalars(temperature, mul, div, cap)
    if n_keep > 0:
        fp32_logits, exact_mode = _logits_mode(device, compute_dtype)
        if compute_dtype not in _HALF:
            fp32_logits, exact_mode = False, False
        if tiled and not _RECOMPUTE_FP32:
            fp32_logits = False
        w = weight.to(compute_dtype)
        outs = (("ent", ent, want_ent), ("sum", mean, want_mean), ("lssp", lssp, want_lssp), ("top1", top1, want_top1))
        if tiled and rows <= 0 and _SPLIT_COLS > 0 and _use_triton(device):
            # Recompute paths (trainable head / bias, differentiable entropy): only the statistics survive forward.
            # (No-grad calls stay on the chunked path, bit-identical to a forward-gradient policy pass.)
            full = n_keep == N
            dst = {"logp": logps, "lse": lse, "ent": ent if want_ent else None, "sum": mean if want_mean else None,
                   "lssp": lssp if want_lssp else None, "top1": top1 if want_top1 else None}
            if not full:
                dst = {k: (torch.empty(n_keep, dtype = v.dtype, device = device) if v is not None else None)
                       for k, v in dst.items()}
            with torch.autocast(device.type, enabled = False):
                _tiled_forward(device, hidden, idx, n_keep, w, bias, targets, exact_mode, (inv_t, scale, cap),
                               (want_ent, want_mean, want_lssp, want_top1), fp32_logits, compute_dtype, dst, -rows)
            if want_mean:
                dst["sum"].div_(V)
            if not full:
                for key, out in (("logp", logps), ("lse", lse), ("ent", ent), ("sum", mean), ("lssp", lssp),
                                 ("top1", top1)):
                    if dst[key] is not None:
                        out.index_copy_(0, idx, dst[key])
            return logps, ent, mean, lssp, top1, u, lse
        R = _chunk_rows(rows, n_keep, H, V, 4 if fp32_logits else 2, device)
        R = min(R, n_keep)
        buf = _Buffers(R, V, device, fp32_logits, compute_dtype)
        coef = torch.full((), -1.0, **f32)
        with torch.autocast(device.type, enabled = False):
            for a in range(0, n_keep, R):
                b = min(a + R, n_keep)
                n = b - a
                h = _rows_of(hidden, idx, a, b, n_keep, N).to(compute_dtype)
                t = targets[a:b] if n_keep == N else targets.index_select(0, idx[a:b])
                x = _project(h, w, buf, n, fp32_logits)
                exact = _exact_target(h, w, bias, t) if exact_mode else None
                dz_out = buf.grad_view(n, compute_dtype) if want_u else None
                # Every row scored: the kernel writes the outputs in place; compacted rows are scattered after.
                direct = {"logp": logps[a:b], "lse": lse[a:b]} if n_keep == N else None
                if direct is not None:
                    for key, dst, on in outs:
                        if on and key != "sum":
                            direct[key] = dst[a:b]
                st, dz = _row_forward(
                    device, x, dz_out, t, bias, exact, coef, (inv_t, scale, cap, 1.0, 0.0),
                    (want_ent, want_mean, want_lssp, want_top1), want_u, direct,
                )
                put = (lambda dst, src: dst[a:b].copy_(src)) if n_keep == N \
                    else (lambda dst, src: dst.index_copy_(0, idx[a:b], src.to(dst.dtype)))
                if direct is None:
                    put(logps, st["logp"])
                    put(lse, st["lse"])
                for key, dst, on in outs:
                    if on and (direct is None or key == "sum"):
                        put(dst, st[key] / V if key == "sum" else st[key])
                if want_u:
                    put(u, _dz_w(dz, w))
    return logps, ent, mean, lssp, top1, u, lse


@_guarded
def _logprobs_bwd_impl(hidden, weight, bias, targets, mask, lse, ent, g, ge, temperature, mul, div, cap,
                       compute_dtype, need_h, need_w, need_b, rows):
    device = weight.device
    N, H = hidden.shape
    V = weight.shape[0]
    idx, n_keep = _keep_index(targets, mask, V, None)
    f32 = dict(device = device, dtype = torch.float32)
    inv_t, scale, cap = _scalars(temperature, mul, div, cap)
    has_ge = ge.numel() > 0
    # Autograd hands in stride-0 gradients (from .sum()); the kernels index rows directly.
    g, ge, lse, ent, targets = g.contiguous(), ge.contiguous(), lse.contiguous(), ent.contiguous(), targets.contiguous()
    bias = bias.contiguous() if bias is not None else None
    grad_h = torch.zeros((N, H), device = device, dtype = hidden.dtype) if need_h else hidden.new_empty(0)
    grad_w = torch.zeros_like(weight) if need_w else hidden.new_empty(0)
    grad_b = torch.zeros(V, device = device, dtype = bias.dtype) if need_b else hidden.new_empty(0)
    if n_keep == 0:
        return grad_h, grad_w, grad_b
    fp32_logits, _ = _logits_mode(device, compute_dtype)
    if compute_dtype not in _HALF or not _RECOMPUTE_FP32:
        fp32_logits = False
    bpl = 4 if fp32_logits else 2
    w = weight.to(compute_dtype)
    # Tiles of up to 4096 rows x a vocab tile (UNSLOTH_FUSED_LINEAR_BWD_TILE_MB), as TRL: the head gradient then
    # accumulates over K = rows per GEMM (512-row chunks re-read a whole fp32 slice accumulator 16x per 8k tokens,
    # 2.5x TRL's dW time) and the hidden gradient over K = tile columns.
    # rows < 0 (max_rows_per_chunk) caps the tile rows, as in forward.
    R = max(min(rows if rows > 0 else (min(_BWD_ROWS, -rows) if rows < 0 else _BWD_ROWS), n_keep), 1)
    tile = _BWD_TILE_BYTES // max(R * bpl, 1)
    if need_w:
        tile = min(tile, _BWD_TILE_BYTES // (4 * H))  # the per-tile fp32 head-gradient accumulator, same budget
    tile = max(tile, 256)
    tile = min(V, tile - tile % 256)
    buf = _Buffers(R, tile, device, fp32_logits, compute_dtype)
    gather = (lambda v, a, b: v[a:b]) if n_keep == N else (lambda v, a, b: v.index_select(0, idx[a:b]))
    acc_b = torch.zeros(V, **f32) if need_b else None

    def tile_grad(h, a, b, first, last):
        n = b - a
        x = _logits_into(h, w[first:last], buf.buf[:n, : last - first], fp32_logits)
        dz_out = buf.buf.view(compute_dtype)[:n, : last - first] if fp32_logits else x
        dz = _row_backward(
            device, x, dz_out, gather(targets, a, b), bias[first:last] if bias is not None else None,
            gather(lse, a, b), gather(ent, a, b) if has_ge else None,
            gather(g, a, b), gather(ge, a, b) if has_ge else None, (inv_t, scale, cap), first, V,
        )
        if acc_b is not None:
            acc_b[first:last] += dz.sum(0, dtype = torch.float32)
        return dz

    def put_h(a, b, gh):
        if n_keep == N: grad_h[a:b].copy_(gh)
        else: grad_h.index_copy_(0, idx[a:b], gh.to(grad_h.dtype))

    with torch.autocast(device.type, enabled = False):
        if need_w:
            # Vocab-major: each fp32 head-gradient tile is finished over every row before it is cast into place.
            acc_h = torch.zeros((n_keep, H), **f32) if need_h else None
            for first in range(0, V, tile):
                last = min(first + tile, V)
                acc_w = torch.zeros((last - first, H), **f32)
                for a in range(0, n_keep, R):
                    b = min(a + R, n_keep)
                    h = _rows_of(hidden, idx, a, b, n_keep, N).to(compute_dtype)
                    dz = tile_grad(h, a, b, first, last)
                    if acc_h is not None:
                        _dz_w_acc(acc_h[a:b], dz, w[first:last])
                    _dz_t_h(acc_w, dz, h)
                grad_w[first:last].copy_(acc_w)  # cast on copy: no half temp + device copy
                del acc_w
            if acc_h is not None:
                put_h(0, n_keep, acc_h)
        else:
            # Row-major: one fp32 hidden-gradient tile per row chunk, finished over the vocabulary.
            acc_h = torch.zeros((R, H), **f32) if (need_h and tile < V) else None
            for a in range(0, n_keep, R):
                b = min(a + R, n_keep)
                h = _rows_of(hidden, idx, a, b, n_keep, N).to(compute_dtype)
                if acc_h is not None:
                    acc_h[: b - a].zero_()
                for first in range(0, V, tile):
                    last = min(first + tile, V)
                    dz = tile_grad(h, a, b, first, last)
                    if not need_h:
                        continue
                    if acc_h is None:
                        put_h(a, b, _dz_w(dz, w[first:last]))
                    else:
                        _dz_w_acc(acc_h[: b - a], dz, w[first:last])
                if acc_h is not None:
                    put_h(a, b, acc_h[: b - a])
    if acc_b is not None:
        grad_b.copy_(acc_b)
    return grad_h, grad_w, grad_b


# ---------------------------------------------------------------------------------------------------------------
# torch.library registration (once per process: a reloaded module reuses the registered ops)
# ---------------------------------------------------------------------------------------------------------------
_OPS = getattr(torch, "_unsloth_fused_linear_ops", None)
if _OPS is None:
    from torch.library import custom_op

    @custom_op("unsloth_zoo::fused_linear_ce", mutates_args = ())
    def _ce_op(
        hidden: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], targets: torch.Tensor,
        divisor: torch.Tensor, scaling: torch.Tensor, ignore_index: int, mul: float, div: float, cap: float,
        label_smoothing: float, logits_bias: Optional[torch.Tensor], compute_dtype: torch.dtype,
        need_h: bool, need_w: bool, need_b: bool, fp32_head_grad: bool, rows: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        CALLS["ce"] += 1
        return _OPS["ce_impl"](hidden, weight, bias, targets, divisor, scaling, ignore_index, mul, div, cap,
                               label_smoothing, logits_bias, compute_dtype, need_h, need_w, need_b, fp32_head_grad,
                               rows)

    @_ce_op.register_fake
    def _ce_fake(hidden, weight, bias, targets, divisor, scaling, ignore_index, mul, div, cap, label_smoothing,
                 logits_bias, compute_dtype, need_h, need_w, need_b, fp32_head_grad, rows):
        # A fresh empty per slot: the real op returns distinct tensors, and outputs must not alias each other.
        return (
            torch.empty((), device = weight.device, dtype = torch.float32),
            hidden.new_empty(hidden.shape) if need_h else hidden.new_empty(0),
            weight.new_empty(weight.shape) if need_w else hidden.new_empty(0),
            bias.new_empty(bias.shape) if need_b else hidden.new_empty(0),
        )

    def _ce_setup(ctx, inputs, output):
        loss, grad_h, grad_w, grad_b = output
        ctx.mark_non_differentiable(grad_h, grad_w, grad_b)
        ctx.need = inputs[13:16]
        ctx.save_for_backward(grad_h, grad_w, grad_b, inputs[5])

    def _ce_backward(ctx, grad_loss, *_):
        grad_h, grad_w, grad_b, scaling = ctx.saved_tensors
        # Stored gradients carry `scaling`; the incoming gradient replaces it, multiplied in fp32 a row chunk at a time
        # (no fp32 copy of a [V, H] head gradient, and the factor is not rounded to half first).
        acc = torch.float64 if grad_loss.device.type in ("cuda", "cpu") else torch.float32  # MPS: no fp64
        s = grad_loss.to(acc) / scaling.to(acc)
        need_h, need_w, need_b = ctx.need
        scale = lambda x: torch.ops.unsloth_zoo.fused_linear_scale_rows(x, s, x.dtype)
        out_h = scale(grad_h) if need_h else None
        out_w = scale(grad_w) if need_w else None
        out_b = scale(grad_b) if need_b else None
        return (out_h, out_w, out_b) + (None,) * 15

    _ce_op.register_autograd(_ce_backward, setup_context = _ce_setup)

    @custom_op("unsloth_zoo::fused_linear_logprobs", mutates_args = ())
    def _lp_op(
        hidden: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], targets: torch.Tensor,
        mask: Optional[torch.Tensor], temperature: float, mul: float, div: float, cap: float,
        compute_dtype: torch.dtype, want_ent: bool, want_mean: bool, want_lssp: bool, want_top1: bool,
        want_u: bool, need_h: bool, need_w: bool, need_b: bool, rows: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        CALLS["logprobs"] += 1
        return _OPS["lp_impl"](hidden, weight, bias, targets, mask, temperature, mul, div, cap, compute_dtype,
                               want_ent, want_mean, want_lssp, want_top1, want_u, rows,
                               (need_h or need_w or need_b) and not want_u)

    @_lp_op.register_fake
    def _lp_fake(hidden, weight, bias, targets, mask, temperature, mul, div, cap, compute_dtype,
                 want_ent, want_mean, want_lssp, want_top1, want_u, need_h, need_w, need_b, rows):
        N = hidden.shape[0]
        f32 = dict(device = weight.device, dtype = torch.float32)
        e = lambda: torch.empty(0, **f32)  # noqa: E731  (a fresh one per slot: outputs must not alias)
        u_dtype = torch.float32 if _U_FP32 and hidden.dtype in _HALF else hidden.dtype
        return (
            torch.empty(N, **f32),
            torch.empty(N, **f32) if want_ent else e(),
            torch.empty(N, **f32) if want_mean else e(),
            torch.empty(N, **f32) if want_lssp else e(),
            torch.empty(N if want_top1 else 0, device = weight.device, dtype = torch.bool),
            torch.empty((N, hidden.shape[1]), device = weight.device, dtype = u_dtype) if want_u
            else hidden.new_empty(0),
            torch.empty(N, **f32),
        )

    def _lp_setup(ctx, inputs, output):
        logps, ent, mean, lssp, top1, u, lse = output
        ctx.mark_non_differentiable(mean, lssp, top1, u, lse)
        # An unused (or detached) entropy then arrives as None, not zeros that run the entropy backward term.
        ctx.set_materialize_grads(False)
        (hidden, weight, bias, targets, mask, temperature, mul, div, cap, compute_dtype,
         want_ent, want_mean, want_lssp, want_top1, want_u, need_h, need_w, need_b, rows) = inputs
        ctx.want_u = want_u
        ctx.need = (need_h, need_w, need_b)
        ctx.args = (temperature, mul, div, cap, compute_dtype, rows)
        ctx.has_bias = bias is not None
        ctx.has_mask = mask is not None
        ctx.hidden_dtype = hidden.dtype
        if want_u:
            ctx.save_for_backward(u)
        else:
            ctx.save_for_backward(hidden, weight, bias, targets, mask, lse, ent)

    def _lp_backward(ctx, g_logps, g_ent, *_):
        need_h, need_w, need_b = ctx.need
        none = (None,) * 18
        if ctx.want_u:
            (u,) = ctx.saved_tensors
            if g_logps is None or not need_h:
                return (None,) + none
            return (torch.ops.unsloth_zoo.fused_linear_scale_rows(u, g_logps, ctx.hidden_dtype),) + none
        hidden, weight, bias, targets, mask, lse, ent = ctx.saved_tensors
        if g_logps is None and g_ent is None:
            return (None,) + none
        g = g_logps.float() if g_logps is not None else torch.zeros_like(lse)
        ge = g_ent.float() if (g_ent is not None and ent.numel() > 0) else lse.new_empty(0)
        temperature, mul, div, cap, compute_dtype, rows = ctx.args
        gh, gw, gb = torch.ops.unsloth_zoo.fused_linear_logprobs_bwd(
            hidden, weight, bias, targets, mask, lse, ent, g, ge, temperature, mul, div, cap, compute_dtype,
            need_h, need_w, need_b, rows,
        )
        return (gh if need_h else None, gw if need_w else None, gb if need_b else None) + (None,) * 16

    _lp_op.register_autograd(_lp_backward, setup_context = _lp_setup)

    @custom_op("unsloth_zoo::fused_linear_logprobs_bwd", mutates_args = ())
    def _lp_bwd_op(
        hidden: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], targets: torch.Tensor,
        mask: Optional[torch.Tensor], lse: torch.Tensor, ent: torch.Tensor, g: torch.Tensor, ge: torch.Tensor,
        temperature: float, mul: float, div: float, cap: float, compute_dtype: torch.dtype,
        need_h: bool, need_w: bool, need_b: bool, rows: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        CALLS["logprobs_bwd"] += 1
        return _OPS["lp_bwd_impl"](hidden, weight, bias, targets, mask, lse, ent, g, ge, temperature, mul, div,
                                   cap, compute_dtype, need_h, need_w, need_b, rows)

    @_lp_bwd_op.register_fake
    def _lp_bwd_fake(hidden, weight, bias, targets, mask, lse, ent, g, ge, temperature, mul, div, cap,
                     compute_dtype, need_h, need_w, need_b, rows):
        return (
            hidden.new_empty(hidden.shape) if need_h else hidden.new_empty(0),
            weight.new_empty(weight.shape) if need_w else hidden.new_empty(0),
            bias.new_empty(bias.shape) if need_b else hidden.new_empty(0),
        )

    @custom_op("unsloth_zoo::fused_linear_scale_rows", mutates_args = ())
    def _scale_rows_op(u: torch.Tensor, g: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return _OPS["scale_rows_impl"](u, g, dtype)

    @_scale_rows_op.register_fake
    def _scale_rows_fake(u, g, dtype):
        return u.new_empty(u.shape, dtype = dtype)

    _OPS = {}
    torch._unsloth_fused_linear_ops = _OPS

# Bodies are looked up at call time, so a reloaded module serves the registered ops with its own code.
_OPS["ce_impl"] = _ce_impl
_OPS["lp_impl"] = _logprobs_impl
_OPS["lp_bwd_impl"] = _logprobs_bwd_impl


def _scale_rows_impl(u, g, dtype):
    # (u * g[:, None]).to(dtype) in fp32 (fp64 for fp64), a row chunk at a time: an fp32 u times g would otherwise
    # materialise a second fp32 [N, H] before the cast. g is per row, or one value for every row (CE). A half tensor
    # times a 0-dim fp32 tensor would round that value to half first (0.37 -> 0.369: 0.2% off every gradient).
    sd = _stat_dtype(u.dtype)
    g = g.reshape(-1, 1).to(device = u.device, dtype = sd)
    one = g.shape[0] == 1
    out = torch.empty(u.shape, device = u.device, dtype = dtype)
    if sd == torch.float32 and u.is_contiguous() and u.numel() > 0 and _use_triton(u.device):
        n = u.numel()
        block = 4096
        g = g.contiguous()

        def run():
            _fl_scale_kernel[(triton.cdiv(n, block),)](
                u, g, out, n, max(n // max(u.shape[0], 1), 1) if u.dim() > 1 else 1, PER_ROW = not one,
                BLOCK = block, num_warps = 8,
            )
        try:
            _launch(u.device, run)
            return out
        except torch.OutOfMemoryError:
            raise
        except Exception as e:
            if type(e).__module__.startswith("torch._dynamo"):
                raise
            _TRITON_BROKEN.add(u.device)
            warnings.warn(f"Unsloth: fused linear Triton kernel failed on {u.device} ({type(e).__name__}: {e}); "
                          f"using the torch path.", stacklevel = 2)
    u2, out2 = (u.reshape(-1, 1), out.reshape(-1, 1)) if u.dim() == 1 else (u, out)
    step = 8192
    for a in range(0, u2.shape[0], step):
        b = min(a + step, u2.shape[0])
        ga = g if one else g[a:b]
        torch.mul(u2[a:b].to(sd), ga, out = out2[a:b]) if dtype == sd else out2[a:b].copy_(u2[a:b].to(sd) * ga)
    return out


_OPS["scale_rows_impl"] = _scale_rows_impl
if "calls" in _OPS:
    CALLS = _OPS["calls"]
else:
    _OPS["calls"] = CALLS


# ---------------------------------------------------------------------------------------------------------------
# Public wrappers (traceable)
# ---------------------------------------------------------------------------------------------------------------
def _compute_dtype(hidden, weight):
    device_type = weight.device.type
    if torch.is_autocast_enabled(device_type):
        return torch.get_autocast_dtype(device_type)
    if weight.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return weight.dtype
    return hidden.dtype


def _requires_grad(x):
    return x is not None and x.requires_grad


def fused_linear_cross_entropy(
    hidden, weight, bias, targets, *,
    divisor,
    scaling = None,
    ignore_index = -100,
    logit_scale_multiply = 0.0, logit_scale_divide = 0.0, logit_softcapping = 0.0,
    label_smoothing = 0.0,
    logits_bias = None,
    fp32_head_grad = False,
    rows_per_chunk = None,
):
    """sum(CE(hidden @ weight.T + bias, targets)) / divisor over rows whose target != ignore_index, float32 scalar.

    The gradient is taken in forward chunk by chunk (no recompute). `scaling` (fp16 GradScaler) multiplies the stored
    gradients so they do not underflow in half precision; backward divides it back out. A zero or missing divisor or
    scaling counts as 1."""
    device = weight.device
    hidden2 = hidden.reshape(-1, hidden.shape[-1]).to(device)
    targets2 = targets.reshape(-1).to(device)
    grad_on = torch.is_grad_enabled()
    need_h = grad_on and hidden.requires_grad
    need_w = grad_on and weight.requires_grad
    need_b = grad_on and _requires_grad(bias)
    f32 = dict(device = device, dtype = torch.float32)
    divisor = torch.as_tensor(divisor).to(**f32).reshape(-1)[:1].reshape(())
    divisor = torch.where(divisor == 0, torch.ones_like(divisor), divisor)
    if scaling is None:
        scaling = torch.ones((), **f32)
    else:
        scaling = torch.as_tensor(scaling).to(**f32).reshape(())
        scaling = torch.where(scaling == 0, torch.ones_like(scaling), scaling)
    if logits_bias is not None:
        # Bart's final_logits_bias is [1, vocab], and a split model may keep it off the head's device.
        logits_bias = logits_bias.reshape(-1).to(device)
    loss = torch.ops.unsloth_zoo.fused_linear_ce(
        hidden2, weight, bias, targets2, divisor, scaling, int(ignore_index),
        float(logit_scale_multiply or 0.0), float(logit_scale_divide or 0.0), float(logit_softcapping or 0.0),
        float(label_smoothing or 0.0), logits_bias, _compute_dtype(hidden, weight),
        need_h, need_w, need_b, bool(fp32_head_grad), int(rows_per_chunk or 0),
    )[0]
    return loss


def fused_linear_logprobs(
    hidden, weight, targets, *,
    bias = None,
    mask = None,
    temperature = 1.0,
    logit_scale_multiply = 0.0, logit_scale_divide = 0.0, logit_softcapping = 0.0,
    return_entropy = False,
    return_mean_logits = False,
    return_log_sum_sq_probs = False,
    return_is_top1 = False,
    rows_per_chunk = None,
    entropy_requires_grad = True,
    max_rows_per_chunk = None,
):
    """Per-token log_softmax(transform(hidden @ weight.T + bias) / temperature)[targets], float32 [...].

    Rows outside `mask` (and rows with an out-of-range target) return 0 and never reach the head. Extras in order:
    entropy (differentiable unless entropy_requires_grad=False), vocab mean of the transformed logits before
    temperature, log sum p^2, argmax == target. `max_rows_per_chunk` caps the automatic chunk size
    (`rows_per_chunk` sets it). With a frozen head, forward keeps d logp / d hidden per row and
    backward is one multiply; a trainable head / bias or a differentiable entropy recompute the logits instead."""
    device = weight.device
    lead = targets.shape
    hidden2 = hidden.reshape(-1, hidden.shape[-1]).to(device)
    targets2 = targets.reshape(-1).to(device)
    mask2 = mask.reshape(-1).to(device = device, dtype = torch.bool) if mask is not None else None
    grad_on = torch.is_grad_enabled()
    need_h = grad_on and hidden.requires_grad
    need_w = grad_on and weight.requires_grad
    need_b = grad_on and _requires_grad(bias)
    ent_grad = bool(return_entropy) and bool(entropy_requires_grad)
    want_u = _FORWARD_GRAD and need_h and not need_w and not need_b and not ent_grad
    logps, ent, mean, lssp, top1, u, lse = torch.ops.unsloth_zoo.fused_linear_logprobs(
        hidden2, weight, bias, targets2, mask2, float(temperature),
        float(logit_scale_multiply or 0.0), float(logit_scale_divide or 0.0), float(logit_softcapping or 0.0),
        _compute_dtype(hidden, weight), bool(return_entropy), bool(return_mean_logits),
        bool(return_log_sum_sq_probs), bool(return_is_top1), want_u, need_h, need_w, need_b,
        int(rows_per_chunk or 0) or -int(max_rows_per_chunk or 0),
    )
    out_device = hidden.device
    result = [logps.reshape(lead).to(out_device)]
    if return_entropy:
        e = ent if entropy_requires_grad else ent.detach()
        result.append(e.reshape(lead).to(out_device))
    if return_mean_logits:
        result.append(mean.reshape(lead).to(out_device))
    if return_log_sum_sq_probs:
        result.append(lssp.reshape(lead).to(out_device))
    if return_is_top1:
        result.append(top1.reshape(lead).to(out_device))
    return result[0] if len(result) == 1 else tuple(result)
