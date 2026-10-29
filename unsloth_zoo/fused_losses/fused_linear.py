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

CE and frozen-head log-probs write the gradient in forward (no logits recompute in backward); a trainable head /
bias or differentiable entropy recompute the logits per vocab slice. Opaque `torch.library` ops, so no graph break;
torch bodies with the same math when Triton is unavailable.
"""

__all__ = [
    "fused_linear_cross_entropy",
    "fused_linear_logprobs",
    "fused_linear_jsd",
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
# 1: fp32 logits where fp32-out GEMM is fast; 2: half logits + exact fp32 target logit; 0: half logits.
_ACCURACY = os.environ.get("UNSLOTH_LOGPROB_FP32_LOGITS", "1")
_U_FP32 = os.environ.get("UNSLOTH_LOGPROB_U_FP32", "0") == "1"
# fp32 forward + half backward is WRONG (3x worse gradient); both passes read this so saved lse matches recompute.
_RECOMPUTE_FP32 = os.environ.get("UNSLOTH_FUSED_LINEAR_RECOMPUTE_FP32", "1") != "0"
# Half backward recompute with each _TOP_COLS block's two largest logits restored to fp32; plain half logits under
# an fp32 lse are 3x worse than TRL (top logits' rounding no longer cancels).
_BWD_EXACT_TOP = os.environ.get("UNSLOTH_FUSED_LINEAR_BWD_EXACT_TOP", "1") != "0"
_SINGLE_MIN_COLS = max(int(_env_float("UNSLOTH_FUSED_LINEAR_SINGLE_MIN_COLS", 8192)), 0)
# 128 not 2048: frequent tokens cluster at low ids, so wide blocks give 3x worse gradients.
_TOP_COLS = 1 << max(int(math.log2(max(int(_env_float("UNSLOTH_FUSED_LINEAR_TOP_COLS", 128)), 16))), 4)
_CHUNK_GB = _env_float("UNSLOTH_LOGPROB_CHUNK_GB", 2.0)
# Fewer than 512 rows starve the K = vocab GEMM.
_MIN_LOGITS = 512 * 151936
_MIN_ROWS = int(_env_float("UNSLOTH_FUSED_LINEAR_MIN_ROWS", 0))
_MAX_ROWS = max(int(_env_float("UNSLOTH_FUSED_LINEAR_MAX_ROWS", 4096)), 1)
_BLOCK = int(_env_float("UNSLOTH_FUSED_LINEAR_BLOCK", 4096))
_SPLIT_COLS = max(int(_env_float("UNSLOTH_FUSED_LINEAR_SPLIT_COLS", 16384)), 0)
# A multiple of 16: split bounds, vocab tiles and their buffers keep Triton's divisibility-by-16 specialisation
# (see _split_launches), or the chunked and vocab-tiled passes of one row stop agreeing in the last bit.
_SPLIT_COLS = -(-_SPLIT_COLS // 16) * 16
_SPLIT_BLOCK = max(int(_env_float("UNSLOTH_FUSED_LINEAR_SPLIT_BLOCK", 2048)), 16)
# ROCm: small tiles are ~2x faster on gfx1151.
_HIP_TILES = torch.version.hip is not None
_BWD_ROWS = max(int(_env_float("UNSLOTH_FUSED_LINEAR_BWD_ROWS", 1024 if _HIP_TILES else 4096)), 1)
_BWD_TILE_BYTES = max(int(_env_float("UNSLOTH_FUSED_LINEAR_BWD_TILE_MB", 8 if _HIP_TILES else 256) * 1024**2), 1)
_HALF = (torch.float16, torch.bfloat16)
# Frozen CUDA head: flash-style tiled forward gradient; no-grad makes the same calls so policy == reference bitwise.
# Cards > 24 GiB keep chunks below 4096 rows (faster there on A100).
_FLASH = os.environ.get("UNSLOTH_FUSED_LINEAR_FLASH", "1") != "0"
_FLASH_MIN_ROWS = (max(int(_env_float("UNSLOTH_FUSED_LINEAR_FLASH_MIN_ROWS", 0)), 0)
                   if os.environ.get("UNSLOTH_FUSED_LINEAR_FLASH_MIN_ROWS") is not None else None)
_SMALL_CARD_BYTES = 24.5 * 1024**3


def _flash_min_rows(device):
    """Per device, so a policy pass and its reference always take the same route."""
    if _FLASH_MIN_ROWS is not None:
        return _FLASH_MIN_ROWS
    return _flash_min_rows_auto(device)


@functools.cache
def _flash_min_rows_auto(device):
    if device.type != "cuda" or _IS_HIP:
        return 4096
    try:
        total = torch.cuda.get_device_properties(device).total_memory
    except Exception:
        return 4096
    return 0 if total <= _SMALL_CARD_BYTES else 4096
_FWD_TILE_COLS = (max(int(_env_float("UNSLOTH_FUSED_LINEAR_FWD_TILE_COLS", 0)), 0)
                  if os.environ.get("UNSLOTH_FUSED_LINEAR_FWD_TILE_COLS") is not None else None)
_IS_HIP = torch.version.hip is not None
# Bump the suffix on any schema / backward change: Inductor / AOTAutograd caches key on neither.
_LP_OP = "fused_linear_logprobs_v6"
_LP_BWD_OP = "fused_linear_logprobs_bwd_v4"
_CE_OP = "fused_linear_ce_v2"

CALLS = {"ce": 0, "logprobs": 0, "logprobs_bwd": 0, "triton": 0, "torch": 0, "tiled": 0, "exact_top": 0,
         "bwd_single": 0, "flash": 0}


@functools.cache
def _budget_bytes(device):
    try:
        from unsloth_zoo.tiled_mlp import _default_target_gb

        free = float(_default_target_gb(device))
    except Exception:
        free = 0.0
    gb = min(_CHUNK_GB, free) if free > 0 else _CHUNK_GB
    return max(int(gb * 1024**3), 1)


@functools.cache
def _flash_ce_ok(device):
    # CE chunks beat the tiled pass on Turing (log-prob passes still gain from it there).
    return device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8


def _pow2(x):
    return 1 << max(int(math.ceil(math.log2(max(x, 1)))), 0)


def fused_linear_rows(n_rows, hidden_size, vocab_size, bytes_per_logit = 4, device = None):
    """Power of two at or below `n_rows * hidden / vocab` (as Liger), clamped; rounding up doubles the transient."""
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


def _bytes_per_logit(fp32_logits, compute_dtype):
    return 4 if fp32_logits else compute_dtype.itemsize


def _chunk_rows(rows, n_rows, hidden_size, vocab_size, bytes_per_logit, device):
    # rows > 0 sets the chunk, rows < 0 caps the automatic one at -rows, 0 is automatic.
    if rows > 0:
        return rows
    auto = fused_linear_rows(n_rows, hidden_size, vocab_size, bytes_per_logit, device)
    return min(auto, -rows) if rows < 0 else auto


@functools.cache
def _fp32_out_ok(device, dtype = torch.float16):
    # bf16 only on sm >= 80 (a T4 raises).
    try:
        a = torch.ones((8, 8), device = device, dtype = dtype)
        return torch.mm(a, a, out_dtype = torch.float32).dtype == torch.float32
    except torch.OutOfMemoryError:
        raise
    except Exception:
        return False


def _logits_mode(device, dtype = torch.float16):
    if device.type != "cuda":
        return False, False
    if _ACCURACY == "0":
        return False, False
    if _ACCURACY == "2":
        return False, True
    if _IS_HIP or not _fp32_out_ok(device, dtype):
        return False, False
    return True, False


def _exact_lpt(device, dtype):
    # Static (the fake uses it): True iff _logits_mode picks the exact target logit, so logps is not log p_t of
    # the logits the gradient reads and the op returns that separately.
    return device.type == "cuda" and _ACCURACY == "2" and dtype in _HALF


def _capturing(x):
    return x.is_cuda and torch.cuda.is_current_stream_capturing()


def _half_mm_bounded(a, b):
    """fp16 a @ b as a half-result GEMM (fp32 accumulation), returned in fp32, that cannot pass 65504.

    |(a @ b)[i, j]| <= ||a[i]||_1 * max|b|. Rows under 32768 cannot overflow, so the plain half GEMM is safe; if any
    row is over, eager mode falls back to the exact fp32 GEMM, and CUDA graph capture scales those rows down by a power
    of two (exact, undone after the upcast; no host sync).
    For HIP, whose fp32-out GEMM is 6-14x slower and whose fp32 GEMM of the upcast operands (a full fp32 copy of the
    head per chunk) made fp16 3-6x slower than plain torch on gfx1151."""
    # Reductions without an |a| / |b| temporary (b is a full head slice).
    bound = torch.linalg.vector_norm(a, 1, dim = -1, keepdim = True, dtype = torch.float32) * \
        torch.linalg.vector_norm(b, math.inf, dtype = torch.float32)
    scale = torch.exp2(torch.floor(torch.log2(32768.0 / bound.clamp_min(1e-30)))).clamp_(min = 2.0 ** -38, max = 1.0)
    if not _capturing(a):
        # The bound is worst case, so it almost never trips (hidden rows sum to ~1e2-1e3, scaled dz rows to ~1e4).
        # When it does, scaling a row down can flush its small entries below fp16's subnormal range (a +-16384 row
        # with a 1e-4 entry loses that entry's contribution), so take the exact fp32 GEMM of upcast operands for this
        # chunk. One host read of a flag; under CUDA graph capture (no sync allowed) the scaled path below runs.
        if bool((scale < 1.0).any()):
            return torch.mm(a.float(), b.float())
        return (a @ b).float()
    # Capture: a scale below fp16's smallest subnormal (2^-24) would round to 0 and drop the row, so apply it as two
    # power-of-two factors that are each representable (2^-14 is the smallest normal); the product stays exact.
    s1 = scale.clamp(min = 2.0 ** -14)
    a = a * s1.to(a.dtype)
    a.mul_((scale / s1).to(a.dtype))
    return (a @ b).float().div_(scale)


def _mm_fp32(a, b, out = None):
    """a @ b with an fp32 result, rounded once (CUDA half inputs); elsewhere an fp32 matmul of the upcast inputs."""
    if a.is_cuda and a.dtype in _HALF:
        if not _IS_HIP and _fp32_out_ok(a.device, a.dtype):
            return torch.mm(a, b, out_dtype = torch.float32, out = out) if out is not None \
                else torch.mm(a, b, out_dtype = torch.float32)
        if a.dtype == torch.bfloat16:
            r = (a @ b).float()
        elif _IS_HIP:
            r = _half_mm_bounded(a, b)
        else:
            # An fp16 result can pass 65504 (scaled forward gradient), so upcast.
            r = torch.mm(a.float(), b.float())
    else:
        r = a.float() @ b.float() if a.dtype != torch.float64 else a @ b
    if out is not None:
        out.copy_(r)
        return out
    return r


def _addmm_fp32(acc, a, b):
    if acc.dtype == torch.float32 and a.is_cuda and a.dtype in _HALF:
        if not _IS_HIP and _fp32_out_ok(a.device, a.dtype):
            torch.addmm(acc, a, b, out_dtype = torch.float32, out = acc)
        elif a.dtype == torch.bfloat16:
            acc.add_(a @ b)
        elif _IS_HIP:
            acc.add_(_half_mm_bounded(a, b))
        else:
            # fp16 partials can pass 65504 before cancelling in acc.
            acc.addmm_(a.float(), b.float())
        return
    if acc.dtype == a.dtype:
        acc.addmm_(a, b)
    else:
        acc.add_(a.to(acc.dtype) @ b.to(acc.dtype))


if _HAS_TRITON:

    try:
        # 2 sigmoid(2u) - 1 cancels for small u.
        from triton.language.extra import libdevice as _libdevice

        @triton.jit
        def _fl_tanh(x):
            return _libdevice.tanh(x)

        @triton.jit
        def _fl_expm1(x):
            return _libdevice.expm1(x)

        @triton.jit
        def _fl_log1p(x):
            return _libdevice.log1p(x)
    except Exception:
        @triton.jit
        def _fl_tanh(x):
            return 2.0 * tl.sigmoid(2.0 * x) - 1.0

        # Kahan's forms: exact to an ulp for tiny x with tl.exp / tl.log only.
        @triton.jit
        def _fl_expm1(x):
            u = tl.exp(x)
            um1 = u - 1.0
            return tl.where(u == 1.0, x, tl.where(um1 == -1.0, -1.0, um1 * (x / tl.log(tl.where(u == 1.0, 2.0, u)))))

        @triton.jit
        def _fl_log1p(x):
            u = 1.0 + x
            return tl.where(u == 1.0, x, tl.log(u) * (x / tl.where(u == 1.0, 1.0, u - 1.0)))

    @triton.jit
    def _fl_nll(zt, m, so):
        # lse - zt from the max m and so = sum_{j != t} exp(z_j - m): softplus(log(so) + m - zt) never adds e^0 = 1
        # to e^-margin, so it stays exact past the fp32 cliff (margin ~16.6, arXiv 2610.31642) and is never < 0.
        r = tl.log(so) + m - zt
        return tl.maximum(r, 0.0) + _fl_log1p(tl.exp(-tl.abs(r)))

    @triton.jit
    def _fl_transform(x, scale, cap, HAS_SCALE: tl.constexpr, HAS_CAP: tl.constexpr):
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
    def _fl_block_others(x_row, bias_ptr, bstart, n_end, tgt, m, inv_t, scale, cap, cols,
                         HAS_BIAS: tl.constexpr, HAS_SCALE: tl.constexpr, HAS_CAP: tl.constexpr):
        # sum_{j != tgt} exp(z_j - m) over the block at bstart: the sweep leaves the target's block out entirely (a
        # scalar choice per block, no per-element mask in the hot loop) and this re-read adds the rest of it back.
        offs = bstart + cols
        ok = offs < n_end
        x = tl.load(x_row + offs, mask = ok, other = 0.0).to(tl.float32)
        if HAS_BIAS:
            x += tl.load(bias_ptr + offs, mask = ok, other = 0.0).to(tl.float32)
        x, _d = _fl_transform(x, scale, cap, HAS_SCALE, HAS_CAP)
        z = tl.where(ok & (offs != tgt), x * inv_t, -float("inf"))
        return tl.sum(tl.exp(z - m), axis = 0)

    @triton.jit
    def _fl_row_stats(row, valid, zt, m, so, a, exact_ptr, logp_ptr, lse_ptr, ent_ptr, lpt_ptr, inv_t, scale, cap,
                      HAS_SCALE: tl.constexpr, HAS_CAP: tl.constexpr, HAS_EXACT: tl.constexpr, WANT_ENT: tl.constexpr):
        # so excludes the target. lse must match the logits the gradient reads; mixing in the exact target logit biases
        # gradients. lpt = log p_t of those logits (logp unless HAS_EXACT): the backward's 1 - p_t = -expm1(lpt).
        s = so + tl.where(valid, tl.exp(zt - m), 0.0)
        tl.store(lse_ptr + row, m + tl.log(s))
        nll = _fl_nll(zt, m, so)
        lpt = tl.where(valid, -nll, 0.0)
        if HAS_EXACT:
            xe = tl.load(exact_ptr + row).to(tl.float32)
            xe, _d = _fl_transform(xe, scale, cap, HAS_SCALE, HAS_CAP)
            tl.store(logp_ptr + row, tl.where(valid, -_fl_nll(xe * inv_t, m, so), 0.0))
            tl.store(lpt_ptr + row, lpt)
        else:
            tl.store(logp_ptr + row, lpt)
        if WANT_ENT:
            # log s = lse - m, which is nll when the target is the max (log(s) reads 0 past the cliff).
            log_s = tl.where(valid & (zt >= m), nll, tl.log(s))
            tl.store(ent_ptr + row, log_s - a / s)
        return s

    @triton.jit
    def _fl_forward_kernel(
        x_ptr, x_stride, n_cols,
        tgt_ptr, exact_ptr, bias_ptr,
        logp_ptr, lse_ptr, ent_ptr, sum_ptr, lssp_ptr, top1_ptr, lpt_ptr,
        inv_t, scale, cap,
        HAS_BIAS: tl.constexpr, HAS_SCALE: tl.constexpr, HAS_CAP: tl.constexpr, HAS_EXACT: tl.constexpr,
        WANT_ENT: tl.constexpr, WANT_SUM: tl.constexpr, WANT_LSSP: tl.constexpr, WANT_TOP1: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        x_row = x_ptr + row * x_stride
        cols = tl.arange(0, BLOCK)
        tgt = tl.load(tgt_ptr + row).to(tl.int64)
        valid = (tgt >= 0) & (tgt < n_cols)
        tgt_c = tl.where(valid, tgt, 0)

        m = tl.full((), -float("inf"), tl.float32)
        s = tl.zeros((), tl.float32)
        a = tl.zeros((), tl.float32)
        q = tl.zeros((), tl.float32)
        tot = tl.zeros((), tl.float32)
        best = tl.full((), -float("inf"), tl.float32)
        best_i = tl.zeros((), tl.int64)
        sb = tl.zeros((), tl.float32)
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
                shift = tl.where(s + sb > 0, (m - m_new) * (s + sb), 0.0)
                a = r * (a + shift) + tl.sum(tl.where(ok, e * (z - m_new), 0.0), axis = 0)
            if WANT_LSSP:
                q = q * r * r + tl.sum(e * e, axis = 0)
            # s skips the target's block (sb holds it); its other classes are added back after the sweep.
            es = tl.sum(e, axis = 0)
            inb = valid & (tgt >= start) & (tgt < start + BLOCK)
            s = s * r + tl.where(inb, 0.0, es)
            sb = sb * r + tl.where(inb, es, 0.0)
            m = m_new

        if valid:
            s += _fl_block_others(x_row, bias_ptr, (tgt // BLOCK) * BLOCK, n_cols, tgt, m, inv_t, scale, cap, cols,
                                  HAS_BIAS, HAS_SCALE, HAS_CAP)
        xt = tl.load(x_row + tgt_c).to(tl.float32)
        if HAS_BIAS:
            xt += tl.load(bias_ptr + tgt_c).to(tl.float32)
        xt, _dt = _fl_transform(xt, scale, cap, HAS_SCALE, HAS_CAP)
        zt = xt * inv_t
        s = _fl_row_stats(row, valid, zt, m, s, a, exact_ptr, logp_ptr, lse_ptr, ent_ptr, lpt_ptr, inv_t, scale, cap,
                          HAS_SCALE, HAS_CAP, HAS_EXACT, WANT_ENT)
        if WANT_SUM:
            tl.store(sum_ptr + row, tot)
        if WANT_LSSP:
            tl.store(lssp_ptr + row, tl.log(q) - 2.0 * tl.log(s))
        if WANT_TOP1:
            tl.store(top1_ptr + row, (best_i == tgt) & valid)


    @triton.jit
    def _fl_partial_kernel(
        x_ptr, x_stride, n_cols, n_rows, split_cols, stride_k, col0, split0, bias_ptr, tgt_ptr, part_ptr,
        inv_t, scale, cap, top_ptr, top_stride,
        HAS_BIAS: tl.constexpr, HAS_SCALE: tl.constexpr, HAS_CAP: tl.constexpr,
        WANT_ENT: tl.constexpr, WANT_SUM: tl.constexpr, WANT_LSSP: tl.constexpr, WANT_TOP1: tl.constexpr,
        WANT_TGT: tl.constexpr, WANT_BLOCK_TOP: tl.constexpr, SUB: tl.constexpr, BLOCK: tl.constexpr,
    ):
        # Split width depends only on the vocabulary, so a row's statistics do not depend on chunk size.
        # WANT_BLOCK_TOP: top_ptr[row, 2 k + {0, 1}] = two largest z of the k-th SUB-aligned column group.
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
        tgt = tl.load(tgt_ptr + row).to(tl.int64) - col0
        here = (tgt >= c0) & (tgt < c1)
        sb = tl.zeros((), tl.float32)
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
            if WANT_BLOCK_TOP:
                zg = tl.reshape(z, (BLOCK // SUB, SUB))
                t1 = tl.max(zg, axis = 1)
                # Drop one max position only: tied maxima (repeated or zero-initialised head rows) keep t2 == t1.
                i1 = tl.argmax(zg, axis = 1)
                t2 = tl.max(tl.where(tl.arange(0, SUB)[None, :] == i1[:, None], -float("inf"), zg), axis = 1)
                k = tl.arange(0, BLOCK // SUB)
                tp = top_ptr + row * top_stride + ((start + col0) // SUB + k) * 2
                kok = start + k * SUB < c1
                tl.store(tp, t1, mask = kok)
                tl.store(tp + 1, t2, mask = kok)
            if WANT_TOP1:
                bi = tl.argmax(z, axis = 0).to(tl.int64) + start + col0
                take = bm > best
                best_i = tl.where(take, bi, best_i)
                best = tl.where(take, bm, best)
            m_new = tl.maximum(m, bm)
            r = tl.exp(m - m_new)
            e = tl.exp(z - m_new)
            if WANT_ENT:
                shift = tl.where(s + sb > 0, (m - m_new) * (s + sb), 0.0)
                a = r * (a + shift) + tl.sum(tl.where(ok, e * (z - m_new), 0.0), axis = 0)
            if WANT_LSSP:
                q = q * r * r + tl.sum(e * e, axis = 0)
            es = tl.sum(e, axis = 0)
            # here: a split narrower than BLOCK must not drop a target that lies past its end.
            inb = here & (tgt >= start) & (tgt < start + BLOCK)
            s = s * r + tl.where(inb, 0.0, es)
            sb = sb * r + tl.where(inb, es, 0.0)
            m = m_new
        if here:
            s += _fl_block_others(x_row, bias_ptr, c0 + ((tgt - c0) // BLOCK) * BLOCK, c1, tgt, m, inv_t, scale, cap,
                                  cols, HAS_BIAS, HAS_SCALE, HAS_CAP)
        # part[k, split, row]: m, s (without the target), a, q, tot, best, best_i (float: < 2^24 columns), target logit
        p = part_ptr + (split0 + split) * n_rows + row
        if WANT_TGT:
            if here:
                xt = tl.load(x_row + tgt).to(tl.float32)
                if HAS_BIAS:
                    xt += tl.load(bias_ptr + tgt).to(tl.float32)
                xt, _dt = _fl_transform(xt, scale, cap, HAS_SCALE, HAS_CAP)
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
        logp_ptr, lse_ptr, ent_ptr, sum_ptr, lssp_ptr, top1_ptr, lpt_ptr,
        inv_t, scale, cap,
        HAS_BIAS: tl.constexpr, HAS_SCALE: tl.constexpr, HAS_CAP: tl.constexpr, HAS_EXACT: tl.constexpr,
        WANT_ENT: tl.constexpr, WANT_SUM: tl.constexpr, WANT_LSSP: tl.constexpr, WANT_TOP1: tl.constexpr,
        HAS_TGT: tl.constexpr, SPLITS: tl.constexpr,
    ):
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
            xt = tl.load(part_ptr + 7 * stride_k + (tgt_c // split_cols) * n_rows + row)
        else:
            xt = tl.load(x_ptr + row * x_stride + tgt_c).to(tl.float32)
            if HAS_BIAS:
                xt += tl.load(bias_ptr + tgt_c).to(tl.float32)
            xt, _d = _fl_transform(xt, scale, cap, HAS_SCALE, HAS_CAP)
        zt = xt * inv_t
        a = 0.0
        if WANT_ENT:
            # The (ms - m) * ss shift uses each split's full sum: add the target back to its own split.
            aa = tl.load(p + 2 * stride_k, mask = okk, other = 0.0)
            st = tl.where(valid & (k == tgt_c // split_cols), tl.exp(zt - ms), 0.0)
            a = tl.sum(w * (aa + tl.where(okk, (ms - m) * (ss + st), 0.0)), axis = 0)
        s = _fl_row_stats(row, valid, zt, m, s, a, exact_ptr, logp_ptr, lse_ptr, ent_ptr, lpt_ptr, inv_t, scale, cap,
                          HAS_SCALE, HAS_CAP, HAS_EXACT, WANT_ENT)
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
            # Ties go to the earliest column, as torch.argmax.
            first = tl.min(tl.where(bb == top, bi, float(2 ** 30)), axis = 0)
            tl.store(top1_ptr + row, (first.to(tl.int64) == tgt) & valid)

    @triton.jit
    def _fl_backward_kernel(
        x_ptr, x_stride, dz_ptr, dz_stride, n_cols, col0, n_total,
        tgt_ptr, bias_ptr, lse_ptr, ent_ptr, g_ptr, g_stride, ge_ptr, lpt_ptr,
        inv_t, scale, cap, smooth_keep, smooth_each, top_ptr, top_stride,
        HAS_BIAS: tl.constexpr, HAS_SCALE: tl.constexpr, HAS_CAP: tl.constexpr, HAS_ENT_GRAD: tl.constexpr,
        ALIASED: tl.constexpr, HAS_TOP: tl.constexpr, SUB: tl.constexpr, BLOCK: tl.constexpr,
    ):
        # dz = g * (keep * onehot + each - p) - ge * p * (z - lse + H). HAS_TOP: half x; each SUB group's two largest
        # z take the forward's fp32 values (rounding keeps order; ties in half take their mean). The target column
        # takes the forward's log p_t: 1 - p_t = -expm1(lpt) is the others' mass (exp(z_t - lse) rounds to 1).
        row = tl.program_id(0).to(tl.int64)
        x_row = x_ptr + row * x_stride
        dz_row = dz_ptr + row * dz_stride
        cols = tl.arange(0, BLOCK)
        tgt = tl.load(tgt_ptr + row).to(tl.int64)
        valid = (tgt >= 0) & (tgt < n_total)
        lse = tl.load(lse_ptr + row)
        g = tl.where(valid, tl.load(g_ptr + row * g_stride).to(tl.float32), 0.0)
        lpt = tl.load(lpt_ptr + row)
        pt = tl.exp(lpt)
        ct = g * ((smooth_keep + smooth_each - 1.0) - _fl_expm1(lpt))
        if HAS_ENT_GRAD:
            ge = tl.where(valid, tl.load(ge_ptr + row).to(tl.float32), 0.0)
            ent = tl.load(ent_ptr + row)
            ct -= ge * pt * (lpt + ent)
        for start in range(0, n_cols, BLOCK):
            offs = start + cols
            ok = offs < n_cols
            x = tl.load(x_row + offs, mask = ok, other = 0.0).to(tl.float32)
            if ALIASED:
                # dz aliases x: all loads land before any store.
                tl.debug_barrier()
            if HAS_BIAS:
                x += tl.load(bias_ptr + offs, mask = ok, other = 0.0).to(tl.float32)
            x, d = _fl_transform(x, scale, cap, HAS_SCALE, HAS_CAP)
            z = x * inv_t
            if HAS_TOP:
                k = tl.arange(0, BLOCK // SUB)
                kok = start + k * SUB < n_cols
                tp = top_ptr + row * top_stride + ((col0 + start) // SUB + k) * 2
                t1 = tl.load(tp, mask = kok, other = 0.0)[:, None]
                t2 = tl.load(tp + 1, mask = kok, other = 0.0)[:, None]
                zg = tl.reshape(tl.where(ok, z, -float("inf")), (BLOCK // SUB, SUB))
                h1 = tl.max(zg, axis = 1)[:, None]
                is1 = zg == h1
                one = (tl.sum(is1.to(tl.int32), axis = 1) == 1)[:, None]
                h2 = tl.max(tl.where(is1, -float("inf"), zg), axis = 1)[:, None]
                zg = tl.where(is1, tl.where(one, t1, 0.5 * (t1 + t2)), tl.where(one & (zg == h2), t2, zg))
                z = tl.where(ok, tl.reshape(zg, (BLOCK,)), z)
            p = tl.exp(z - lse)
            dz = g * (smooth_each - p)
            if HAS_ENT_GRAD:
                dz -= ge * p * (z - lse + ent)
            dz = tl.where(offs + col0 == tgt, ct, dz)
            dz = dz * inv_t * d
            tl.store(dz_row + offs, dz.to(dz_ptr.dtype.element_ty), mask = ok)


    @triton.jit
    def _fl_flash_p_kernel(
        x_ptr, x_stride, p_ptr, p_stride, n_cols, n_rows, col0, split0, n_splits, stride_k, part_ptr,
        tgt_ptr, bias_ptr, m_ptr, acc_ptr, acc_stride, H,
        inv_t, scale, cap, gscale,
        HAS_BIAS: tl.constexpr, HAS_SCALE: tl.constexpr, HAS_CAP: tl.constexpr, SPLITS: tl.constexpr,
        BLOCK: tl.constexpr, BLOCK_H: tl.constexpr,
    ):
        # p overwrites the tile in place (half view of x); target column zeroed, its term is added exactly later.
        row = tl.program_id(0).to(tl.int64)
        k = tl.arange(0, SPLITS)
        ms = tl.load(part_ptr + (split0 + k).to(tl.int64) * n_rows + row, mask = k < n_splits, other = -float("inf"))
        m_old = tl.load(m_ptr + row)
        m_new = tl.maximum(m_old, tl.max(ms, axis = 0))
        if (m_new > m_old) & (m_old > -float("inf")):
            r = tl.exp(m_old - m_new)
            hc = tl.arange(0, BLOCK_H)
            for hs in range(0, H, BLOCK_H):
                ho = hs + hc
                a = tl.load(acc_ptr + row * acc_stride + ho, mask = ho < H, other = 0.0)
                tl.store(acc_ptr + row * acc_stride + ho, a * r, mask = ho < H)
        x_row = x_ptr + row * x_stride
        p_row = p_ptr + row * p_stride
        tgt = tl.load(tgt_ptr + row).to(tl.int64) - col0
        cols = tl.arange(0, BLOCK)
        c = inv_t * gscale
        for start in range(0, n_cols, BLOCK):
            offs = start + cols
            ok = offs < n_cols
            x = tl.load(x_row + offs, mask = ok, other = 0.0).to(tl.float32)
            tl.debug_barrier()
            if HAS_BIAS:
                x += tl.load(bias_ptr + offs, mask = ok, other = 0.0).to(tl.float32)
            x, d = _fl_transform(x, scale, cap, HAS_SCALE, HAS_CAP)
            e = tl.exp(x * inv_t - m_new) * c * d
            e = tl.where(offs == tgt, 0.0, e)
            tl.store(p_row + offs, e.to(p_ptr.dtype.element_ty), mask = ok)
        # Every warp has read m_old before it is replaced.
        tl.debug_barrier()
        tl.store(m_ptr + row, m_new)

    @triton.jit
    def _fl_flash_u_kernel(
        acc_ptr, acc_stride, w_ptr, w_stride, w_stride_h, u_ptr, u_stride, map_ptr, row0, H, n_rows, stride_k, split_cols,
        part_ptr, tgt_ptr, logp_ptr, lse_ptr, m_ptr, g_ptr,
        inv_t, scale, cap, inv_gscale,
        HAS_SCALE: tl.constexpr, HAS_CAP: tl.constexpr, HAS_MAP: tl.constexpr, HAS_G: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        # u = inv_t d_t (1 - p_t) w_t - sum_{j != t} p_j inv_t d_j w_j; target term in fp32, the rest from acc.
        row = tl.program_id(0).to(tl.int64)
        tgt = tl.load(tgt_ptr + row).to(tl.int64)
        xt = tl.load(part_ptr + 7 * stride_k + (tgt // split_cols) * n_rows + row)
        dt = 1.0
        if HAS_SCALE:
            dt = scale
        if HAS_CAP:
            tt = xt / cap
            dt = dt * (1.0 - tt * tt)
        ct = -inv_t * dt * _fl_expm1(tl.load(logp_ptr + row))
        r = tl.exp(tl.load(m_ptr + row) - tl.load(lse_ptr + row)) * inv_gscale
        if HAS_G:
            g = tl.load(g_ptr)
            ct = ct * g
            r = r * g
        if HAS_MAP:
            out = tl.load(map_ptr + row).to(tl.int64)
        else:
            out = row0 + row
        hc = tl.arange(0, BLOCK_H)
        for hs in range(0, H, BLOCK_H):
            ho = hs + hc
            ok = ho < H
            w = tl.load(w_ptr + tgt * w_stride + ho * w_stride_h, mask = ok, other = 0.0).to(tl.float32)
            a = tl.load(acc_ptr + row * acc_stride + ho, mask = ok, other = 0.0)
            tl.store(u_ptr + out * u_stride + ho, (ct * w - a * r).to(u_ptr.dtype.element_ty), mask = ok)

    @triton.jit
    def _fl_scale_kernel(x_ptr, g_ptr, out_ptr, n, cols, PER_ROW: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        ok = offs < n
        x = tl.load(x_ptr + offs, mask = ok, other = 0.0).to(tl.float32)
        if PER_ROW:
            g = tl.load(g_ptr + offs // cols, mask = ok, other = 0.0)
        else:
            g = tl.load(g_ptr)
        tl.store(out_ptr + offs, (x * g).to(out_ptr.dtype.element_ty), mask = ok)


    # GKD generalized JSD over two heads' fp32 logits [R, V]; grid (rows, vocab blocks) fills the card at small R.
    @triton.jit
    def _fl_jsd_load(s_ptr, t_ptr, row, cols, ok, stride, s_scale, s_cap, t_scale, t_cap, inv_t,
                     S_SCALE: tl.constexpr, S_CAP: tl.constexpr, T_SCALE: tl.constexpr, T_CAP: tl.constexpr):
        base = row.to(tl.int64) * stride
        s = tl.load(s_ptr + base + cols, mask = ok, other = 0.0).to(tl.float32)
        t = tl.load(t_ptr + base + cols, mask = ok, other = 0.0).to(tl.float32)
        s, ds = _fl_transform(s, s_scale, s_cap, S_SCALE, S_CAP)
        t, _ = _fl_transform(t, t_scale, t_cap, T_SCALE, T_CAP)
        return s * inv_t, t * inv_t, ds

    @triton.jit
    def _fl_jsd_stats_kernel(s_ptr, t_ptr, stride, V, stat_ptr, NB, s_scale, s_cap, t_scale, t_cap, inv_t,
                             S_SCALE: tl.constexpr, S_CAP: tl.constexpr, T_SCALE: tl.constexpr,
                             T_CAP: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        blk = tl.program_id(1)
        cols = blk * BLOCK + tl.arange(0, BLOCK)
        ok = cols < V
        s, t, _ = _fl_jsd_load(s_ptr, t_ptr, row, cols, ok, stride, s_scale, s_cap, t_scale, t_cap, inv_t,
                               S_SCALE, S_CAP, T_SCALE, T_CAP)
        s = tl.where(ok, s, -float("inf"))
        t = tl.where(ok, t, -float("inf"))
        ms = tl.max(s, 0)
        mt = tl.max(t, 0)
        ls = tl.sum(tl.exp(s - ms), 0)
        lt = tl.sum(tl.exp(t - mt), 0)
        o = stat_ptr + (row.to(tl.int64) * NB + blk) * 4
        tl.store(o, ms)
        tl.store(o + 1, ls)
        tl.store(o + 2, mt)
        tl.store(o + 3, lt)

    @triton.jit
    def _fl_jsd_row_kernel(s_ptr, t_ptr, stride, V, lse_ptr, c_ptr, part_ptr, NB, dz_ptr, dz_stride, coef_ptr,
                           beta, log1mb, logb, s_scale, s_cap, t_scale, t_cap, inv_t,
                           MODE: tl.constexpr, WRITE_DZ: tl.constexpr, S_SCALE: tl.constexpr,
                           S_CAP: tl.constexpr, T_SCALE: tl.constexpr, T_CAP: tl.constexpr, BLOCK: tl.constexpr):
        # MODE 0: beta = 0, partial loss and (WRITE_DZ) dz = p - q in one pass. MODE 1: beta = 1, partial
        # sum p (ls - lt). MODE 2: interior, partial sum q (lt - m) and sum p (ls - m).
        row = tl.program_id(0)
        blk = tl.program_id(1)
        cols = blk * BLOCK + tl.arange(0, BLOCK)
        ok = cols < V
        s, t, ds = _fl_jsd_load(s_ptr, t_ptr, row, cols, ok, stride, s_scale, s_cap, t_scale, t_cap, inv_t,
                                S_SCALE, S_CAP, T_SCALE, T_CAP)
        ls = s - tl.load(lse_ptr + row * 2)
        lt = t - tl.load(lse_ptr + row * 2 + 1)
        p = tl.exp(ls)
        q = tl.exp(lt)
        o = part_ptr + (row.to(tl.int64) * NB + blk) * 2
        if MODE == 0:
            tl.store(o, tl.sum(tl.where(ok, q * (lt - ls), 0.0), 0))
            if WRITE_DZ:
                dz = (p - q) * (tl.load(coef_ptr) * inv_t) * ds
                tl.store(dz_ptr + row.to(tl.int64) * dz_stride + cols, dz.to(dz_ptr.dtype.element_ty), mask = ok)
        elif MODE == 1:
            tl.store(o, tl.sum(tl.where(ok, p * (ls - lt - tl.load(c_ptr + row)), 0.0), 0))
        else:
            a = ls + log1mb
            b = lt + logb
            mx = tl.maximum(a, b)
            m = mx + tl.log(tl.exp(a - mx) + tl.exp(b - mx))
            tl.store(o, tl.sum(tl.where(ok, q * (lt - m), 0.0), 0))
            tl.store(o + 1, tl.sum(tl.where(ok, p * (ls - m - tl.load(c_ptr + row)), 0.0), 0))

    @triton.jit
    def _fl_jsd_dz_kernel(s_ptr, t_ptr, stride, V, lse_ptr, c_ptr, rs_ptr, dz_ptr, dz_stride, coef_ptr, scale_g,
                          log1mb, logb, s_scale, s_cap, t_scale, t_cap, inv_t,
                          MODE: tl.constexpr, S_SCALE: tl.constexpr, S_CAP: tl.constexpr, T_SCALE: tl.constexpr,
                          T_CAP: tl.constexpr, BLOCK: tl.constexpr):
        # dz = scale_g * p * (d - sum p d); MODE 1: d = ls - lt, MODE 2: d = ls - m, scale_g = 1 - beta.
        row = tl.program_id(0)
        blk = tl.program_id(1)
        cols = blk * BLOCK + tl.arange(0, BLOCK)
        ok = cols < V
        s, t, ds = _fl_jsd_load(s_ptr, t_ptr, row, cols, ok, stride, s_scale, s_cap, t_scale, t_cap, inv_t,
                                S_SCALE, S_CAP, T_SCALE, T_CAP)
        ls = s - tl.load(lse_ptr + row * 2)
        lt = t - tl.load(lse_ptr + row * 2 + 1)
        p = tl.exp(ls)
        if MODE == 1:
            d = ls - lt
        else:
            a = ls + log1mb
            b = lt + logb
            mx = tl.maximum(a, b)
            d = ls - (mx + tl.log(tl.exp(a - mx) + tl.exp(b - mx)))
        # d is shifted by its value at the student's argmax (c): the softmax-collapse-safe form of d - sum p d.
        g = p * ((d - tl.load(c_ptr + row)) - tl.load(rs_ptr + row))
        dz = g * (tl.load(coef_ptr) * (scale_g * inv_t)) * ds
        tl.store(dz_ptr + row.to(tl.int64) * dz_stride + cols, dz.to(dz_ptr.dtype.element_ty), mask = ok)


_TRITON_BROKEN = set()


@functools.cache
def _triton_arch_ok(device):
    # Pre-Volta and gfx906 run with torch.compile off (Triton codegen is unsupported or broken there): torch bodies.
    if _IS_HIP:
        arch = str(getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")).split(":")[0].lower()
        return arch != "gfx906"
    return torch.cuda.get_device_capability(device)[0] >= 7


def _use_triton(device):
    return _TRITON and _HAS_TRITON and device.type == "cuda" and device not in _TRITON_BROKEN and _triton_arch_ok(device)


def _launch(device, fn):
    with torch.cuda.device(device):
        return fn()


def _num_warps():
    return 8 if _IS_HIP else 16


def _split_block():
    return min(triton.next_power_of_2(_SPLIT_BLOCK), triton.next_power_of_2(max(_SPLIT_COLS, 16)))


def _split_launches(width):
    """(offset, n_cols, splits) partial-kernel launches over width columns. Triton specialises n_cols on
    divisibility by 16 (load width, so the fp32 summation order): full splits get a bound divisible by 16 and only a
    ragged last split the remainder, so chunked and vocab-tiled passes run every split with the same kernel."""
    k = -(-width // _SPLIT_COLS)
    if k == 1 or width % 16 == 0:
        return ((0, width, k),)
    c = (k - 1) * _SPLIT_COLS
    return ((0, c, k - 1), (c, width - c, 1))


def _tiled_vocab(V):
    # The chunked pass merges splits only past two, so only there do tiled and chunked statistics agree.
    return _SPLIT_COLS > 0 and -(-V // _SPLIT_COLS) > 2


def _block_top_shape(N, V, device, compute_dtype, recompute):
    # Static (the fake uses it). Trainable heads only: frozen-head entropy passes were slower with it.
    if not (recompute and _BWD_EXACT_TOP and _RECOMPUTE_FP32 and _TRITON and _HAS_TRITON and not _IS_HIP
            and device.type == "cuda" and compute_dtype in _HALF and _tiled_vocab(V)):
        return (0,)
    return (N, 2 * -(-V // _top_cols()))


def _block_top_live(device, compute_dtype, rows, fp32_logits, V):
    # Evaluated alike by forward and backward: True iff the tiled Triton forward wrote the block tops.
    return (_BWD_EXACT_TOP and _RECOMPUTE_FP32 and bool(fp32_logits) and compute_dtype in _HALF and rows <= 0
            and _use_triton(device) and _tiled_vocab(V) and _SPLIT_COLS % _split_block() == 0)


def _top_cols():
    return min(_TOP_COLS, _split_block())


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
    V = z.shape[-1]
    valid = (tgt >= 0) & (tgt < V)
    tc = torch.where(valid, tgt, torch.zeros_like(tgt)).unsqueeze(-1)
    zt = z.gather(-1, tc).squeeze(-1)
    # lse without the target, then log p_t = -softplus(lse_o - z_t): exact past the softmax collapse cliff.
    # z is this call's own tensor: mask the target in place and put it back.
    lse_o = torch.logsumexp(z.scatter_(-1, tc, -float("inf")), dim = -1)
    z.scatter_(-1, tc, zt.unsqueeze(-1))
    lse = torch.logaddexp(lse_o, zt)
    zero = torch.zeros_like(lse)
    lpt = torch.where(valid, -torch.nn.functional.softplus(lse_o - zt), zero)
    logp = lpt
    if exact is not None:
        ze = _torch_transform(exact.to(sd), scale, cap)[0] * inv_t
        logp = torch.where(valid, -torch.nn.functional.softplus(lse_o - ze), zero)
    out = {"lse": lse.float(), "logp": logp.float(), "lpt": lpt.float()}
    want_ent, want_sum, want_lssp, want_top1 = want
    logp_all = None
    if want_ent or want_lssp:
        logp_all = (z - lse.unsqueeze(-1)).scatter(-1, tc, (lpt + torch.where(valid, zero, zt - lse)).unsqueeze(-1))
    if want_ent:
        out["ent"] = (-(logp_all.exp() * logp_all).sum(-1)).float()
    if want_sum:
        out["sum"] = xt.sum(-1).float()
    if want_lssp:
        out["lssp"] = torch.logsumexp(2 * logp_all, dim = -1).float()
    if want_top1:
        out["top1"] = (z.argmax(-1) == tgt) & valid
    return out, z, lse, valid


def _torch_dz(x, bias, tgt, lse, lpt, inv_t, scale, cap, coef_row, smooth_keep, smooth_each, col0 = 0,
              ge = None, ent = None, n_total = None):
    """dz = coef_row * (p - keep onehot - each) - ge * p * (z - lse + ent) for x [R, Vs] at columns col0..; the
    target column takes p_t - 1 = expm1(lpt) and log p_t = lpt."""
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
    lp = lpt.to(sd)
    dzt = coef * (torch.expm1(lp) + (1.0 - smooth_keep - smooth_each))
    if ge is not None:
        gev = torch.where(valid, ge.to(sd), torch.zeros_like(ge, dtype = sd))
        dz = dz - gev.unsqueeze(-1) * p * (z - lse.to(sd).unsqueeze(-1) + ent.to(sd).unsqueeze(-1))
        dzt = dzt - gev * torch.exp(lp) * (lp + ent.to(sd))
    inside = valid & (tgt >= col0) & (tgt < col0 + z.shape[-1])
    lc = (tgt - col0).clamp(0, z.shape[-1] - 1).unsqueeze(-1)
    dz = dz.scatter_(-1, lc, torch.where(inside, dzt, dz.gather(-1, lc).squeeze(-1)).unsqueeze(-1))
    dz = dz * inv_t
    if d is not None:
        dz = dz * d
    if scale != 1.0:
        dz = dz * scale
    return dz


# Forward statistics buffers: Triton specialises the row stride on divisibility by 16 (load width, so the fp32
# summation order), so the chunked [rows, V] buffer and a vocab tile must both have it or a no-grad pass and its
# trainable-head twin disagree in the last bit (V % 16 != 0).
_FWD_PAD = 16


def _padded(n_cols, pad = 8):
    return -(-n_cols // pad) * pad


class _Buffers:
    """One logits buffer per call, reused by every chunk; odd vocab rows padded to a multiple of pad (8 for cuBLAS)."""

    def __init__(self, rows, n_cols, device, fp32, compute_dtype, pad = 8):
        self.n_cols = n_cols
        self.fp32 = fp32
        self.stride = _padded(n_cols, pad)
        self.dtype = torch.float32 if fp32 else compute_dtype
        self.buf = torch.empty((rows, self.stride), device = device, dtype = self.dtype)

    def logits(self, n):
        return self.buf[:n, : self.n_cols]

    def grad_view(self, n, half_dtype):
        if self.fp32:
            return self.buf.view(half_dtype)[:n, : self.n_cols]
        return self.buf[:n, : self.n_cols]


def _aligned(V):
    V8 = V - V % 8
    return V8 if 0 < V8 < V else V


def _logits_into(h, w, out, fp32_logits):
    V8 = _aligned(w.shape[0])
    parts = ((0, V8), (V8, w.shape[0])) if V8 < w.shape[0] else ((0, V8),)
    for a, b in parts:
        if fp32_logits:
            _mm_fp32(h, w[a:b].t(), out = out[:, a:b])
        elif h.is_cuda:
            torch.mm(h, w[a:b].t(), out = out[:, a:b])
        else:
            # CPU fp16 mm with out= into this strided column slice reads the slice's stale contents on
            # avx512_fp16 hosts (torch 2.10, oneDNN): NaN logits from an uninitialised buffer, wrong ones otherwise.
            out[:, a:b].copy_(torch.mm(h, w[a:b].t()))
    return out


def _project(h, w, buf, n, fp32_logits):
    return _logits_into(h, w, buf.logits(n), fp32_logits)


def _dz_w(dz, w):
    """dz[n, V] @ w[V, H] with an fp32 result rounded once."""
    V8 = _aligned(w.shape[0])
    if V8 < w.shape[0] and dz.is_cuda and dz.dtype == torch.float16 and (_IS_HIP or not _fp32_out_ok(dz.device, dz.dtype)):
        # fp16 partials can pass 65504 before the rest cancels.
        return _half_mm_bounded(dz, w) if _IS_HIP else torch.mm(dz.float(), w.float())
    out = _mm_fp32(dz[:, :V8], w[:V8])
    if V8 < w.shape[0]:
        _addmm_fp32(out, dz[:, V8:], w[V8:])
    return out


def _dz_w_acc(acc, dz, w):
    V8 = _aligned(w.shape[0])
    _addmm_fp32(acc, dz[:, :V8], w[:V8])
    if V8 < w.shape[0]:
        _addmm_fp32(acc, dz[:, V8:], w[V8:])


def _dz_t_h(acc, dz, h):
    V8 = _aligned(dz.shape[1])
    _addmm_fp32(acc[:V8], dz[:, :V8].t(), h)
    if V8 < dz.shape[1]:
        _addmm_fp32(acc[V8:], dz[:, V8:].t(), h)


def _exact_target(h, w, bias, tgt):
    V = w.shape[0]
    tc = tgt.clamp(0, V - 1)
    x = (h.float() * w.index_select(0, tc).float()).sum(-1)
    if bias is not None:
        x = x + bias.float().index_select(0, tc)
    return x


def _u_dtype(hidden_dtype):
    # Stored before the GradScaler-scaled gradient exists: fp16 flushes small entries, so keep fp32.
    if hidden_dtype == torch.float16 or (_U_FP32 and hidden_dtype in _HALF):
        return torch.float32
    return hidden_dtype


def _row_forward(device, x, dz_out, tgt, bias, exact, coef, scalars, want, write_grad, outs = None):
    """Statistics of raw logits x [n, V]; with write_grad also writes dz into dz_out (may alias x)."""
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
        lpt = logp
        if exact is not None:
            lpt = outs.get("lpt") if outs.get("lpt") is not None else torch.empty(n, **f32)
        block = min(_BLOCK, triton.next_power_of_2(V))
        splits = -(-V // _SPLIT_COLS) if _SPLIT_COLS > 0 else 1
        if splits <= 2:
            splits = 1

        def run_split():
            # The same kernel binaries as _tiled_forward (target logit kept in part[7]): with temperature != 1 the
            # WANT_TGT / HAS_TGT variants contract x * inv_t into FMAs differently, so entropy and log-sum-p^2 of a
            # no-grad pass would otherwise drift from its trainable-head twin in the last bit.
            part = torch.empty((8, splits, n), **f32)
            flags = dict(HAS_BIAS = bias is not None, HAS_SCALE = scale != 1.0, HAS_CAP = cap != 0.0,
                         WANT_ENT = want_ent, WANT_SUM = want_sum, WANT_LSSP = want_lssp, WANT_TOP1 = want_top1)
            scal = (float(inv_t), float(scale), float(cap) if cap != 0.0 else 1.0)
            for c, n_c, k in _split_launches(V):
                _fl_partial_kernel[(n, k)](
                    x[:, c:], x.stride(0), n_c, n, _SPLIT_COLS, splits * n, c, c // _SPLIT_COLS,
                    bias[c:] if bias is not None else part, tgt, part,
                    *scal, part, 0, WANT_TGT = True, WANT_BLOCK_TOP = False, SUB = _top_cols(), BLOCK = _split_block(),
                    num_warps = 4, enable_fp_fusion = True, **flags,
                )
            _fl_combine_kernel[(n,)](
                part, 0, V, n, splits, splits * n, _SPLIT_COLS, part,
                tgt, exact if exact is not None else lse, lse,
                logp, lse, ent, tot, lssp, top1, lpt, *scal,
                HAS_EXACT = exact is not None, HAS_TGT = True,
                SPLITS = max(triton.next_power_of_2(splits), 2),
                num_warps = 1, enable_fp_fusion = True, **flags,
            )

        def run():
            if splits > 1:
                return run_split()
            _fl_forward_kernel[(n,)](
                x, x.stride(0), V,
                tgt, exact if exact is not None else lse, bias if bias is not None else lse,
                logp, lse, ent, tot, lssp, top1, lpt,
                float(inv_t), float(scale), float(cap) if cap != 0.0 else 1.0,
                HAS_BIAS = bias is not None, HAS_SCALE = scale != 1.0, HAS_CAP = cap != 0.0,
                HAS_EXACT = exact is not None,
                WANT_ENT = want_ent, WANT_SUM = want_sum, WANT_LSSP = want_lssp, WANT_TOP1 = want_top1,
                BLOCK = block, num_warps = 8, enable_fp_fusion = True,
            )
        _launch(device, run)
        CALLS["triton"] += 1
        out = {"logp": logp, "lse": lse, "lpt": lpt}
        if want_ent:
            out["ent"] = ent
        if want_sum:
            out["sum"] = tot
        if want_lssp:
            out["lssp"] = lssp
        if want_top1:
            out["top1"] = top1
        dz = None
        if write_grad:
            dz = _row_backward(device, x, dz_out, tgt, bias, lse, lpt, None, -coef.reshape(()), None,
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
        dz = _row_backward(device, x, dz_out, tgt, bias, out["lse"], out["lpt"], None, -coef.reshape(()), None,
                           (inv_t, scale, cap), 0, V, (smooth_keep, smooth_each))
    return out, dz


def _tiled_layout(n_keep, V, cap_rows, bpl, device):
    """(rows, vocab tile) of the tiled forward; identical for no-grad, frozen and trainable passes of one shape."""
    budget = min(_BWD_TILE_BYTES, _budget_bytes(device))
    R = max(min(_BWD_ROWS, n_keep, cap_rows if cap_rows > 0 else n_keep, budget // (_SPLIT_COLS * bpl)), 1)
    tile = max(budget // max(R * bpl, 1), _SPLIT_COLS)
    cols = _FWD_TILE_COLS if _FWD_TILE_COLS is not None else (16384 if _flash_min_rows(device) == 0 else 0)
    if cols > 0 and bpl == 4:
        tile = min(tile, max(cols, _SPLIT_COLS))
    # A multiple of _SPLIT_COLS (itself of 16), so the _FWD_PAD-padded buffer is exactly R x tile: the budget holds.
    tile = min(tile - tile % _SPLIT_COLS, -(-V // _SPLIT_COLS) * _SPLIT_COLS)
    return R, tile


def _tiled_forward(device, hidden, idx, n_keep, w, bias, targets, exact_mode, scalars, want, fp32_logits,
                   compute_dtype, outs, cap_rows, block_top = None, u = None, gscale = 1.0, g = None):
    """Recompute-path statistics from TRL-shaped tiles; u [N, H] (fp32 logits only) gets d logp / d hidden
    flash-style over vocab tiles. g (0-dim fp32) scales u before rounding."""
    inv_t, scale, cap = scalars
    want_ent, want_sum, want_lssp, want_top1 = want
    N, H = hidden.shape
    V = w.shape[0]
    f32 = dict(device = device, dtype = torch.float32)
    bpl = _bytes_per_logit(fp32_logits, compute_dtype)
    R, tile = _tiled_layout(n_keep, V, cap_rows, bpl, device)
    splits = -(-V // _SPLIT_COLS)
    buf = _Buffers(R, tile, device, fp32_logits, compute_dtype, _FWD_PAD)
    flags = dict(HAS_BIAS = bias is not None, HAS_SCALE = scale != 1.0, HAS_CAP = cap != 0.0,
                 WANT_ENT = want_ent, WANT_SUM = want_sum, WANT_LSSP = want_lssp, WANT_TOP1 = want_top1)
    scal = (float(inv_t), float(scale), float(cap) if cap != 0.0 else 1.0)
    if u is not None:
        assert fp32_logits, "the flash forward gradient writes p into the fp32 logits' bytes"
        acc = torch.empty((R, H), **f32)
        m_run = torch.empty(R, **f32)
        block_h = min(triton.next_power_of_2(H), 1024)
    for a in range(0, n_keep, R):
        b = min(a + R, n_keep)
        n = b - a
        h = _rows_of(hidden, idx, a, b, n_keep, N).to(compute_dtype)
        t = targets[a:b] if n_keep == N else targets.index_select(0, idx[a:b])
        part = torch.empty((8, splits, n), **f32)
        top = block_top[a:b] if block_top is not None else part
        if u is not None:
            acc[:n].zero_()
            m_run[:n].fill_(-float("inf"))
        for first in range(0, V, tile):
            last = min(first + tile, V)
            x = _logits_into(h, w[first:last], buf.buf[:n, : last - first], fp32_logits)
            bias_t = bias[first:last] if bias is not None else None

            def run():
                for c, n_c, k in _split_launches(last - first):
                    _fl_partial_kernel[(n, k)](
                        x[:, c:], x.stride(0), n_c, n, _SPLIT_COLS, splits * n, first + c, (first + c) // _SPLIT_COLS,
                        bias_t[c:] if bias_t is not None else part, t, part, *scal,
                        top, top.stride(0) if block_top is not None else 0,
                        WANT_TGT = True, WANT_BLOCK_TOP = block_top is not None, SUB = _top_cols(),
                        BLOCK = _split_block(), num_warps = 4, enable_fp_fusion = True, **flags,
                    )
                if u is None:
                    return
                n_split = -(-(last - first) // _SPLIT_COLS)
                p = buf.buf.view(compute_dtype)[:n, : last - first]
                _fl_flash_p_kernel[(n,)](
                    x, x.stride(0), p, p.stride(0), last - first, n, first, first // _SPLIT_COLS, n_split,
                    splits * n, part, t, bias_t if bias_t is not None else part, m_run, acc, acc.stride(0), H,
                    *scal, float(gscale), HAS_BIAS = bias_t is not None, HAS_SCALE = scale != 1.0,
                    HAS_CAP = cap != 0.0, SPLITS = max(triton.next_power_of_2(n_split), 2),
                    BLOCK = min(_BLOCK, triton.next_power_of_2(last - first)), BLOCK_H = block_h,
                    num_warps = _num_warps(), enable_fp_fusion = True,
                )
            _launch(device, run)
            if u is not None:
                _dz_w_acc(acc[:n], buf.buf.view(compute_dtype)[:n, : last - first], w[first:last])
        exact = _exact_target(h, w, bias, t) if exact_mode else None
        o = {k: outs[k][a:b] for k in ("logp", "lse", "ent", "sum", "lssp", "top1", "lpt") if outs.get(k) is not None}
        lse = o["lse"]
        assert exact is None or "lpt" in o, "the exact target logit needs its own lpt"
        lpt = o.get("lpt", o["logp"]) if exact is not None else o["logp"]

        def run():
            _fl_combine_kernel[(n,)](
                part, 0, V, n, splits, splits * n, _SPLIT_COLS, part,
                t, exact if exact is not None else lse, lse,
                o["logp"], lse, o.get("ent", lse), o.get("sum", lse), o.get("lssp", lse), o.get("top1", o["logp"]), lpt,
                *scal, HAS_EXACT = exact is not None, HAS_TGT = True,
                SPLITS = max(triton.next_power_of_2(splits), 2), num_warps = 1, enable_fp_fusion = True, **flags,
            )
            if u is None:
                return
            rowmap = idx[a:b] if n_keep != N else None
            _fl_flash_u_kernel[(n,)](
                acc, acc.stride(0), w, w.stride(0), w.stride(1), u, u.stride(0), rowmap if rowmap is not None else t, a, H, n,
                splits * n, _SPLIT_COLS, part, t, lpt, lse, m_run, g if g is not None else lse,
                float(inv_t), float(scale), scal[2], 1.0 / float(gscale), HAS_SCALE = scale != 1.0,
                HAS_CAP = cap != 0.0, HAS_MAP = rowmap is not None, HAS_G = g is not None, BLOCK_H = block_h, num_warps = 4, enable_fp_fusion = True,
            )
        _launch(device, run)
        CALLS["triton"] += 1
    CALLS["tiled"] += 1
    if u is not None:
        CALLS["flash"] += 1


def _row_backward(device, x, dz_out, tgt, bias_slice, lse, lpt, ent, g, ge, scalars, col0, n_total, smooth = (1.0, 0.0),
                  block_top = None):
    """Writes dz of x [n, Vs] into dz_out (x or a half view of it); block_top swaps group tops for fp32 values.
    lpt: the forward's log p_t of these logits (the target column's 1 - p_t)."""
    inv_t, scale, cap = scalars
    smooth_keep, smooth_each = smooth
    n, Vs = x.shape
    if _use_triton(device) and n > 0:
        block = _split_block() if block_top is not None else min(_BLOCK, triton.next_power_of_2(Vs))

        def run():
            _fl_backward_kernel[(n,)](
                x, x.stride(0), dz_out, dz_out.stride(0), Vs, col0, n_total,
                tgt, bias_slice if bias_slice is not None else lse, lse, ent if ent is not None else lse,
                g, 0 if g.dim() == 0 else 1, ge if ge is not None else g, lpt,
                float(inv_t), float(scale), float(cap) if cap != 0.0 else 1.0, float(smooth_keep), float(smooth_each),
                block_top if block_top is not None else lse, block_top.stride(0) if block_top is not None else 0,
                HAS_BIAS = bias_slice is not None, HAS_SCALE = scale != 1.0, HAS_CAP = cap != 0.0,
                HAS_ENT_GRAD = ge is not None, ALIASED = dz_out.dtype != x.dtype, HAS_TOP = block_top is not None,
                SUB = _top_cols() if block_top is not None else 1, BLOCK = block, num_warps = _num_warps(), enable_fp_fusion = True,
            )
        _launch(device, run)
        CALLS["triton"] += 1
        if block_top is not None:
            CALLS["exact_top"] += 1
        return dz_out
    assert block_top is None, "exact-top backward needs the Triton kernel"
    CALLS["torch"] += 1
    dz = _torch_dz(x, bias_slice, tgt, lse, lpt, inv_t, scale, cap, -g, smooth_keep, smooth_each, col0 = col0, ge = ge,
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


def _saved_index(idx, n_keep, n_all):
    # Fixed length (no data dependent op output) + the count on the host, so the backward needs no nonzero / sync.
    full = idx if n_keep == n_all else torch.cat([idx, idx.new_zeros(n_all - n_keep)])
    return full, torch.tensor([n_keep], dtype = torch.long)


def _rows_of(x, idx, a, b, n_keep, n_all):
    if n_keep == n_all:
        return x[a:b]
    return x.index_select(0, idx[a:b])


def _guarded(fn):
    """Runs the Triton body; a Triton failure that is not OOM / dynamo falls back to torch on that device."""
    @functools.wraps(fn)
    def wrapper(hidden, *args, **kwargs):
        device = args[0].device
        if not _use_triton(device):
            return fn(hidden, *args, **kwargs)
        err = None
        # Two tries: a transient JIT race on a shared Triton cache must not demote the run to torch.
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
            # Torch fails too: the inputs are at fault.
            _TRITON_BROKEN.discard(device)
            raise
        warnings.warn(f"Unsloth: fused linear Triton kernel failed on {device} ({type(err).__name__}: {err}); "
                      f"using the torch path.", stacklevel = 2)
        return out
    return wrapper


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
        if bias is not None:
            full_bias += bias.float()
        if logits_bias is not None:
            full_bias += logits_bias.float()
    if n_keep > 0:
        grad = need_h or need_w or need_b
        fp32_logits, exact_mode = _logits_mode(device, compute_dtype)
        if compute_dtype not in _HALF:
            fp32_logits, exact_mode = False, False
        R = _chunk_rows(rows, n_keep, H, _padded(V, _FWD_PAD), _bytes_per_logit(fp32_logits, compute_dtype), device)
        R = min(R, n_keep)
        w = weight.to(compute_dtype)
        coef = (scaling.to(**f32) / divisor.to(**f32)).reshape(())
        lp_all = torch.empty(n_keep, **f32)
        lse_all = torch.empty(n_keep, **f32) if eps != 0.0 else None
        sum_all = torch.empty(n_keep, **f32) if eps != 0.0 else None
        # Frozen head, no label smoothing: the log-prob op's tiled forward (no [rows, V] buffer).
        if (_FLASH and fp32_logits and not _IS_HIP and compute_dtype in _HALF and eps == 0.0 and rows <= 0
                and n_keep >= _flash_min_rows(device) and not need_w and not need_b and _tiled_vocab(V) and _use_triton(device) and _flash_ce_ok(device)):
            gscale = 1.0
            if need_h and compute_dtype == torch.float16:
                gscale = 2.0 ** max(math.floor(math.log2(16384.0 / max(abs(scale), 1e-30))), 0)
            with torch.autocast(device.type, enabled = False):
                _tiled_forward(device, hidden, idx, n_keep, w, full_bias, targets, False, (inv_t, scale, cap),
                               (False, False, False, False), True, compute_dtype,
                               {"logp": lp_all, "lse": torch.empty(n_keep, **f32)}, -rows, None,
                               grad_h if need_h else None, gscale, -coef if need_h else None)
            n_chunked = 0
        else:
            n_chunked = n_keep
            buf = _Buffers(R, V, device, fp32_logits, compute_dtype, _FWD_PAD)
        ratio = None
        if n_chunked and grad and compute_dtype in _HALF and (not need_w or acc_w.dtype == torch.float32):
            # half(coef) per row is a 0.4% bf16 scale error: buffer takes a power of two, fp32 the rest.
            pow2 = torch.exp2(torch.round(torch.log2(coef)))
            ratio, coef = coef / pow2, pow2
        with torch.autocast(device.type, enabled = False):
            for a in range(0, n_chunked, R):
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
                        if ratio is not None:
                            gh.mul_(ratio)
                        if n_keep == N:
                            grad_h[a:b].copy_(gh)
                        else:
                            grad_h.index_copy_(0, idx[a:b], gh.to(grad_h.dtype))
                    if need_w:
                        _dz_t_h(acc_w, dz, h)
                    if need_b:
                        acc_b += dz.sum(0, dtype = torch.float32)
        # float64 (MPS has none): fp32 over thousands of terms is off by an ulp or more.
        acc = torch.float64 if device.type in ("cuda", "cpu") else torch.float32
        if eps != 0.0:
            nll = (1.0 - eps) * (-lp_all.to(acc)) + eps * (lse_all.to(acc) - sum_all.to(acc) * (inv_t / V))
        else:
            nll = -lp_all.to(acc)
        loss = (nll.sum() / divisor.to(acc)).float()
    if n_keep > 0 and ratio is not None:
        if need_w:
            acc_w.mul_(ratio)
        if need_b:
            acc_b.mul_(ratio)
    grad_w = acc_w.to(weight.dtype) if need_w else hidden.new_empty(0)
    grad_b = acc_b.to(bias.dtype) if need_b else hidden.new_empty(0)
    return loss, grad_h, grad_w, grad_b


@_guarded
def _logprobs_impl(hidden, weight, bias, targets, mask, temperature, mul, div, cap, compute_dtype,
                   want_ent, want_mean, want_lssp, want_top1, want_u, rows, tiled = False, head_grad = False):
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
    u_dtype = _u_dtype(hidden.dtype)
    u = torch.zeros((N, H), device = device, dtype = u_dtype) if want_u else hidden.new_empty(0)
    # A softcap keeps the fp32 recompute (half recompute error 3.4x at cap 5).
    block_top = torch.empty(_block_top_shape(N, V, device, compute_dtype, tiled and head_grad and not cap), device = device,
                            dtype = torch.float32)
    inv_t, scale, cap = _scalars(temperature, mul, div, cap)
    fp32_logits, exact_mode = _logits_mode(device, compute_dtype)
    if compute_dtype not in _HALF:
        fp32_logits, exact_mode = False, False
    # log p_t of the logits the gradient reads; only the exact target logit makes it differ from logps (else a
    # copy at the end: an op output may not alias another).
    exact_lpt = _exact_lpt(device, compute_dtype)
    lpt = torch.zeros(N, **f32) if exact_lpt else None
    if n_keep > 0:
        if tiled and not _RECOMPUTE_FP32:
            fp32_logits = False
        w = weight.to(compute_dtype)
        outs = (("ent", ent, want_ent), ("sum", mean, want_mean), ("lssp", lssp, want_lssp), ("top1", top1, want_top1))
        # fp16: onehot - p flushes to 0, so scale by a power of two and divide out after the GEMM.
        u_scale = 1.0
        if want_u and compute_dtype == torch.float16:
            u_scale = 2.0 ** max(math.floor(math.log2(16384.0 / max(abs(inv_t * scale), 1e-30))), 0)
        # Decided without want_u / tiled, so every pass kind of one shape takes the same route.
        flash = _FLASH and fp32_logits and not _IS_HIP and compute_dtype in _HALF and n_keep >= _flash_min_rows(device)
        if (tiled or flash) and rows <= 0 and _tiled_vocab(V) and _use_triton(device):
            full = n_keep == N
            dst = {"logp": logps, "lse": lse, "ent": ent if want_ent else None, "sum": mean if want_mean else None,
                   "lssp": lssp if want_lssp else None, "top1": top1 if want_top1 else None,
                   "lpt": lpt if exact_mode else None}
            if not full:
                dst = {k: (torch.empty(n_keep, dtype = v.dtype, device = device) if v is not None else None)
                       for k, v in dst.items()}
            top = None
            if block_top.numel() > 0 and _block_top_live(device, compute_dtype, rows, fp32_logits, V):
                top = block_top if full else torch.empty((n_keep,) + block_top.shape[1:], device = device,
                                                         dtype = torch.float32)
            with torch.autocast(device.type, enabled = False):
                _tiled_forward(device, hidden, idx, n_keep, w, bias, targets, exact_mode, (inv_t, scale, cap),
                               (want_ent, want_mean, want_lssp, want_top1), fp32_logits, compute_dtype, dst, -rows,
                               top, u if want_u else None, u_scale)
            if want_mean:
                dst["sum"].div_(V)
            if not full:
                for key, out in (("logp", logps), ("lse", lse), ("ent", ent), ("sum", mean), ("lssp", lssp),
                                 ("top1", top1), ("lpt", lpt)):
                    if dst[key] is not None:
                        out.index_copy_(0, idx, dst[key])
                if top is not None:
                    block_top.index_copy_(0, idx, top)
            return (logps, ent, mean, lssp, top1, u, lse, block_top) + _saved_index(idx, n_keep, N) + \
                (lpt if exact_lpt else logps.clone(),)
        R = _chunk_rows(rows, n_keep, H, _padded(V, _FWD_PAD), _bytes_per_logit(fp32_logits, compute_dtype), device)
        R = min(R, n_keep)
        buf = _Buffers(R, V, device, fp32_logits, compute_dtype, _FWD_PAD)
        coef = torch.full((), -u_scale, **f32)
        with torch.autocast(device.type, enabled = False):
            for a in range(0, n_keep, R):
                b = min(a + R, n_keep)
                n = b - a
                h = _rows_of(hidden, idx, a, b, n_keep, N).to(compute_dtype)
                t = targets[a:b] if n_keep == N else targets.index_select(0, idx[a:b])
                x = _project(h, w, buf, n, fp32_logits)
                exact = _exact_target(h, w, bias, t) if exact_mode else None
                dz_out = buf.grad_view(n, compute_dtype) if want_u else None
                direct = {"logp": logps[a:b], "lse": lse[a:b]} if n_keep == N else None
                if direct is not None and exact_mode:
                    direct["lpt"] = lpt[a:b]
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
                    if exact_mode:
                        put(lpt, st["lpt"])
                for key, dst, on in outs:
                    if on and (direct is None or key == "sum"):
                        put(dst, st[key] / V if key == "sum" else st[key])
                if want_u:
                    put(u, _dz_w(dz, w) if u_scale == 1.0 else _dz_w(dz, w).div_(u_scale))
    return (logps, ent, mean, lssp, top1, u, lse, block_top) + _saved_index(idx, n_keep, N) + \
        (lpt if exact_lpt else logps.clone(),)


def _mm_into(out, a, b):
    """out = a @ b accumulated in fp32 and rounded once to out's dtype (no reduced-precision split-K reduction)."""
    if out.dtype == torch.float32:
        _mm_fp32(a, b, out = out)
        return
    if a.is_cuda and a.dtype == out.dtype == b.dtype and a.dtype in _HALF:
        flags = torch.backends.cuda.matmul
        name = "allow_bf16_reduced_precision_reduction" if a.dtype == torch.bfloat16 else \
            "allow_fp16_reduced_precision_reduction"
        old = getattr(flags, name)
        setattr(flags, name, False)
        try:
            torch.mm(a, b, out = out)
        finally:
            setattr(flags, name, old)
        return
    out.copy_(_mm_fp32(a, b))


def _dz_t_h_into(out, dz, h):
    V8 = _aligned(dz.shape[1])
    _mm_into(out[:V8], dz[:, :V8].t(), h)
    if V8 < dz.shape[1]:
        _mm_into(out[V8:], dz[:, V8:].t(), h)


def _bwd_layout(n_keep, V, H, rows, bpl, budget, need_w, single, align):
    if single:
        R, tile = n_keep, budget // max(n_keep * bpl, 1)
    else:
        R = max(min(rows if rows > 0 else (min(_BWD_ROWS, -rows) if rows < 0 else _BWD_ROWS), n_keep), 1)
        if rows <= 0:
            R = min(R, max(budget // (min(V, 256) * bpl), 1))
        tile = budget // max(R * bpl, 1)
        if need_w:
            tile = min(tile, budget // (4 * H))
    tile = max(tile, 8)
    tile = min(V, tile - tile % (align if tile >= align else 8))
    return R, tile


@_guarded
def _logprobs_bwd_impl(hidden, weight, bias, targets, idx, n_keep, lse, ent, g, ge, block_top, lpt, temperature, mul,
                       div, cap, compute_dtype, need_h, need_w, need_b, rows):
    device = weight.device
    N, H = hidden.shape
    V = weight.shape[0]
    n_keep = int(n_keep[0])
    idx = idx[:n_keep]
    f32 = dict(device = device, dtype = torch.float32)
    inv_t, scale, cap = _scalars(temperature, mul, div, cap)
    has_ge = ge.numel() > 0
    # Autograd hands in stride-0 gradients (from .sum()).
    g, ge, lse, ent, targets = g.contiguous(), ge.contiguous(), lse.contiguous(), ent.contiguous(), targets.contiguous()
    lpt = lpt.contiguous()
    bias = bias.contiguous() if bias is not None else None
    if n_keep == 0:
        return (torch.zeros((N, H), device = device, dtype = hidden.dtype) if need_h else hidden.new_empty(0),
                torch.zeros_like(weight) if need_w else hidden.new_empty(0),
                torch.zeros(V, device = device, dtype = bias.dtype) if need_b else hidden.new_empty(0))
    grad_h = (torch.empty if n_keep == N else torch.zeros)((N, H), device = device, dtype = hidden.dtype) \
        if need_h else hidden.new_empty(0)
    grad_w = torch.empty_like(weight) if need_w else hidden.new_empty(0)
    grad_b = torch.zeros(V, device = device, dtype = bias.dtype) if need_b else hidden.new_empty(0)
    fp32_logits, _ = _logits_mode(device, compute_dtype)
    if compute_dtype not in _HALF or not _RECOMPUTE_FP32:
        fp32_logits = False
    budget = min(_BWD_TILE_BYTES, _budget_bytes(device))
    head_acc_mode = need_w and need_h and n_keep > V
    exact_top = block_top.numel() > 0 and _block_top_live(device, compute_dtype, rows, fp32_logits, V)
    for _ in range(2):
        bpl = _bytes_per_logit(fp32_logits and not exact_top, compute_dtype)
        single = (need_w and not head_acc_mode and not _HIP_TILES and _SINGLE_MIN_COLS > 0
                  and (rows == 0 or (rows < 0 and n_keep <= -rows))
                  and n_keep * bpl * min(V, _SINGLE_MIN_COLS) <= budget)
        align = _top_cols() if exact_top else 256
        R, tile = _bwd_layout(n_keep, V, H, rows, bpl, budget, need_w, single, align)
        if not exact_top or tile == V or tile % align == 0:
            break
        exact_top = False
    if exact_top:
        fp32_logits = False
    w = weight.to(compute_dtype)
    buf = _Buffers(R, tile, device, fp32_logits, compute_dtype)
    if n_keep == N:
        tk, lk, ek, gk, gek, topk, ptk = targets, lse, ent, g, ge, block_top, lpt
    else:
        tk, lk, gk = targets.index_select(0, idx), lse.index_select(0, idx), g.index_select(0, idx)
        ptk = lpt.index_select(0, idx)
        ek = ent.index_select(0, idx) if has_ge else ent
        gek = ge.index_select(0, idx) if has_ge else ge
        topk = block_top.index_select(0, idx) if exact_top else block_top
    acc_b = torch.zeros(V, **f32) if need_b else None

    def tile_grad(h, a, b, first, last):
        n = b - a
        x = _logits_into(h, w[first:last], buf.buf[:n, : last - first], fp32_logits)
        dz_out = buf.buf.view(compute_dtype)[:n, : last - first] if fp32_logits else x
        dz = _row_backward(
            device, x, dz_out, tk[a:b], bias[first:last] if bias is not None else None,
            lk[a:b], ptk[a:b], ek[a:b] if has_ge else None, gk[a:b], gek[a:b] if has_ge else None, (inv_t, scale, cap), first,
            V, block_top = topk[a:b] if exact_top else None,
        )
        if acc_b is not None:
            acc_b[first:last] += dz.sum(0, dtype = torch.float32)
        return dz

    def put_h(a, b, gh):
        if n_keep == N:
            grad_h[a:b].copy_(gh)
        else:
            grad_h.index_copy_(0, idx[a:b], gh.to(grad_h.dtype))

    head_acc = torch.zeros((V, H), **f32) if head_acc_mode else None
    with torch.autocast(device.type, enabled = False):
        if single:
            CALLS["bwd_single"] += 1
            hk = _rows_of(hidden, idx, 0, n_keep, n_keep, N).to(compute_dtype)
            acc_h = torch.zeros((n_keep, H), **f32) if need_h else None
            for first in range(0, V, tile):
                last = min(first + tile, V)
                dz = tile_grad(hk, 0, n_keep, first, last)
                if acc_h is not None:
                    _dz_w_acc(acc_h, dz, w[first:last])
                _dz_t_h_into(grad_w[first:last], dz, hk)
            if acc_h is not None:
                put_h(0, n_keep, acc_h)
        elif need_w and head_acc is None:
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
                grad_w[first:last].copy_(acc_w)
                del acc_w
            if acc_h is not None:
                put_h(0, n_keep, acc_h)
        else:
            acc_h = torch.zeros((R, H), **f32) if (need_h and tile < V) else None
            for a in range(0, n_keep, R):
                b = min(a + R, n_keep)
                h = _rows_of(hidden, idx, a, b, n_keep, N).to(compute_dtype)
                if acc_h is not None:
                    acc_h[: b - a].zero_()
                for first in range(0, V, tile):
                    last = min(first + tile, V)
                    dz = tile_grad(h, a, b, first, last)
                    if head_acc is not None:
                        _dz_t_h(head_acc[first:last], dz, h)
                    if not need_h:
                        continue
                    if acc_h is None:
                        put_h(a, b, _dz_w(dz, w[first:last]))
                    else:
                        _dz_w_acc(acc_h[: b - a], dz, w[first:last])
                if acc_h is not None:
                    put_h(a, b, acc_h[: b - a])
            if head_acc is not None:
                grad_w.copy_(head_acc)
    if acc_b is not None:
        grad_b.copy_(acc_b)
    return grad_h, grad_w, grad_b


_OPS = getattr(torch, "_unsloth_fused_linear_ops", None)
if _OPS is None:
    from torch.library import custom_op

    @custom_op(f"unsloth_zoo::{_CE_OP}", mutates_args = ())
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
        return (
            torch.empty((), device = weight.device, dtype = torch.float32),
            hidden.new_empty(hidden.shape) if need_h else hidden.new_empty(0),
            weight.new_empty(weight.shape) if need_w else hidden.new_empty(0),
            bias.new_empty(bias.shape) if need_b else hidden.new_empty(0),
        )

    def _ce_setup(ctx, inputs, output):
        loss, grad_h, grad_w, grad_b = output
        ctx.mark_non_differentiable(grad_h, grad_w, grad_b)
        # Else autograd materializes zero gradients for the stored-gradient outputs: [N, H] (and [V, H]) per backward.
        ctx.set_materialize_grads(False)
        ctx.need = inputs[13:16]
        ctx.save_for_backward(grad_h, grad_w, grad_b, inputs[5])

    def _ce_backward(ctx, grad_loss, *_):
        if grad_loss is None:
            return (None,) * 18
        grad_h, grad_w, grad_b, scaling = ctx.saved_tensors
        # fp32 factor, not rounded to half.
        acc = torch.float64 if grad_loss.device.type in ("cuda", "cpu") else torch.float32
        s = grad_loss.to(acc) / scaling.to(acc)
        need_h, need_w, need_b = ctx.need
        scale = _stored_grad_scaler(s, (grad_h, grad_w, grad_b))
        out_h = scale(grad_h) if need_h else None
        out_w = scale(grad_w) if need_w else None
        out_b = scale(grad_b) if need_b else None
        return (out_h, out_w, out_b) + (None,) * 15

    _ce_op.register_autograd(_ce_backward, setup_context = _ce_setup)

    @custom_op(f"unsloth_zoo::{_LP_OP}", mutates_args = ())
    def _lp_op(
        hidden: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], targets: torch.Tensor,
        mask: Optional[torch.Tensor], temperature: float, mul: float, div: float, cap: float,
        compute_dtype: torch.dtype, want_ent: bool, want_mean: bool, want_lssp: bool, want_top1: bool,
        want_u: bool, need_h: bool, need_w: bool, need_b: bool, rows: int, ent_lazy: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        CALLS["logprobs"] += 1
        return _OPS["lp_impl"](hidden, weight, bias, targets, mask, temperature, mul, div, cap, compute_dtype,
                               want_ent, want_mean, want_lssp, want_top1, want_u, rows,
                               (need_h or need_w or need_b) and not want_u, need_w)

    @_lp_op.register_fake
    def _lp_fake(hidden, weight, bias, targets, mask, temperature, mul, div, cap, compute_dtype,
                 want_ent, want_mean, want_lssp, want_top1, want_u, need_h, need_w, need_b, rows, ent_lazy):
        N = hidden.shape[0]
        f32 = dict(device = weight.device, dtype = torch.float32)
        e = lambda: torch.empty(0, **f32)  # noqa: E731
        u_dtype = _u_dtype(hidden.dtype)
        return (
            torch.empty(N, **f32),
            torch.empty(N, **f32) if want_ent else e(),
            torch.empty(N, **f32) if want_mean else e(),
            torch.empty(N, **f32) if want_lssp else e(),
            torch.empty(N if want_top1 else 0, device = weight.device, dtype = torch.bool),
            torch.empty((N, hidden.shape[1]), device = weight.device, dtype = u_dtype) if want_u
            else hidden.new_empty(0),
            torch.empty(N, **f32),
            torch.empty(_block_top_shape(N, weight.shape[0], weight.device, compute_dtype,
                                         (need_h or need_w or need_b) and not want_u and need_w and not cap), **f32),
            torch.empty(N, device = targets.device, dtype = torch.long),
            torch.empty(1, device = "cpu", dtype = torch.long),
            torch.empty(N, **f32),
        )

    def _lp_setup(ctx, inputs, output):
        logps, ent, mean, lssp, top1, u, lse, block_top, idx, n_keep, lpt = output
        ctx.mark_non_differentiable(mean, lssp, top1, u, lse, block_top, idx, n_keep, lpt)
        ctx.set_materialize_grads(False)
        (hidden, weight, bias, targets, mask, temperature, mul, div, cap, compute_dtype,
         want_ent, want_mean, want_lssp, want_top1, want_u, need_h, need_w, need_b, rows, ent_lazy) = inputs
        ctx.want_u = want_u
        # Lazy entropy: its gradient is recomputed only when one arrives (a logged entropy costs nothing).
        ctx.ent_lazy = bool(want_u and ent_lazy and want_ent)
        ctx.need = (need_h, need_w, need_b)
        ctx.args = (temperature, mul, div, cap, compute_dtype, rows)
        ctx.has_bias = bias is not None
        ctx.has_mask = mask is not None
        ctx.hidden_dtype = hidden.dtype
        # lpt: log p_t for the backward's target column (logps unless the exact target logit is on).
        if ctx.ent_lazy:
            ctx.save_for_backward(u, hidden, weight, bias, targets, idx, n_keep, lse, ent, lpt)
        elif want_u:
            ctx.save_for_backward(u)
        else:
            ctx.save_for_backward(hidden, weight, bias, targets, idx, n_keep, lse, ent, block_top, lpt)

    def _lp_backward(ctx, g_logps, g_ent, *_):
        need_h, need_w, need_b = ctx.need
        none = (None,) * 19
        if ctx.want_u:
            if not need_h:
                return (None,) + none
            if ctx.ent_lazy and g_ent is not None:
                _, hidden, weight, bias, targets, idx, n_keep, lse, ent, lpt = ctx.saved_tensors
                temperature, mul, div, cap, compute_dtype, rows = ctx.args
                g = g_logps.float() if g_logps is not None else torch.zeros_like(lse)
                gh, _, _ = getattr(torch.ops.unsloth_zoo, _LP_BWD_OP)(
                    hidden, weight, bias, targets, idx, n_keep, lse, ent, g, g_ent.float(), lse.new_empty(0), lpt,
                    temperature, mul, div, cap, compute_dtype, True, False, False, rows,
                )
                return (gh,) + none
            if g_logps is None:
                return (None,) + none
            return (torch.ops.unsloth_zoo.fused_linear_scale_rows(ctx.saved_tensors[0], g_logps, ctx.hidden_dtype),) + none
        hidden, weight, bias, targets, idx, n_keep, lse, ent, block_top, lpt = ctx.saved_tensors
        if g_logps is None and g_ent is None:
            return (None,) + none
        g = g_logps.float() if g_logps is not None else torch.zeros_like(lse)
        ge = g_ent.float() if (g_ent is not None and ent.numel() > 0) else lse.new_empty(0)
        temperature, mul, div, cap, compute_dtype, rows = ctx.args
        gh, gw, gb = getattr(torch.ops.unsloth_zoo, _LP_BWD_OP)(
            hidden, weight, bias, targets, idx, n_keep, lse, ent, g, ge, block_top, lpt, temperature, mul, div, cap,
            compute_dtype, need_h, need_w, need_b, rows,
        )
        return (gh if need_h else None, gw if need_w else None, gb if need_b else None) + (None,) * 17

    _lp_op.register_autograd(_lp_backward, setup_context = _lp_setup)

    @custom_op(f"unsloth_zoo::{_LP_BWD_OP}", mutates_args = ())
    def _lp_bwd_op(
        hidden: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], targets: torch.Tensor,
        idx: torch.Tensor, n_keep: torch.Tensor, lse: torch.Tensor, ent: torch.Tensor, g: torch.Tensor,
        ge: torch.Tensor, block_top: torch.Tensor, lpt: torch.Tensor, temperature: float, mul: float, div: float, cap: float,
        compute_dtype: torch.dtype, need_h: bool, need_w: bool, need_b: bool, rows: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        CALLS["logprobs_bwd"] += 1
        return _OPS["lp_bwd_impl"](hidden, weight, bias, targets, idx, n_keep, lse, ent, g, ge, block_top, lpt,
                                   temperature, mul, div, cap, compute_dtype, need_h, need_w, need_b, rows)

    @_lp_bwd_op.register_fake
    def _lp_bwd_fake(hidden, weight, bias, targets, idx, n_keep, lse, ent, g, ge, block_top, lpt, temperature, mul, div,
                     cap, compute_dtype, need_h, need_w, need_b, rows):
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

_ASSERT_ASYNC_MSG = "msg" in torch.ops.aten._assert_async.overloads()


def _ce_checked(hidden, weight, bias, targets, divisor, scaling, ignore_index, *args):
    # Raise like F.cross_entropy; on GPU a device-side assert, since a host read would sync every step.
    V = weight.shape[0]
    bad = (targets != ignore_index) & ((targets < 0) | (targets >= V))
    if targets.is_cuda:
        ok = torch.logical_not(bad.any())
        if _ASSERT_ASYNC_MSG:
            torch._assert_async(ok, f"Unsloth: cross entropy target out of bounds for a vocabulary of {V}")
        else:
            torch._assert_async(ok)
    elif bool(bad.any()):
        raise IndexError(f"Unsloth: cross entropy target {int(targets[bad][0])} is out of bounds for a vocabulary "
                         f"of {V} (ignore_index = {ignore_index}).")
    return _ce_impl(hidden, weight, bias, targets, divisor, scaling, ignore_index, *args)


# Bodies are looked up at call time, so a reloaded module serves the registered ops.
_OPS["ce_impl"] = _ce_checked
_OPS["lp_impl"] = _logprobs_impl
_OPS["lp_bwd_impl"] = _logprobs_bwd_impl


def _scale_rows_impl(u, g, dtype, out = None):
    # half * 0-dim fp32 would round g to half (0.2% off).
    sd = _stat_dtype(u.dtype)
    g = g.reshape(-1, 1).to(device = u.device, dtype = sd)
    one = g.shape[0] == 1
    if out is None:
        out = torch.empty(u.shape, device = u.device, dtype = dtype)
    if sd == torch.float32 and u.is_contiguous() and u.numel() > 0 and _use_triton(u.device):
        n = u.numel()
        block = 4096
        g = g.contiguous()

        def run():
            _fl_scale_kernel[(triton.cdiv(n, block),)](
                u, g, out, n, max(n // max(u.shape[0], 1), 1) if u.dim() > 1 else 1, PER_ROW = not one,
                BLOCK = block, num_warps = 8, enable_fp_fusion = True,
            )
        err = None
        for _ in range(2):
            try:
                _launch(u.device, run)
                return out
            except torch.OutOfMemoryError:
                raise
            except Exception as e:
                if type(e).__module__.startswith("torch._dynamo"):
                    raise
                err = e
        _TRITON_BROKEN.add(u.device)
        warnings.warn(f"Unsloth: fused linear Triton kernel failed on {u.device} ({type(err).__name__}: {err}); "
                      f"using the torch path.", stacklevel = 2)
    u2, out2 = (u.reshape(-1, 1), out.reshape(-1, 1)) if u.dim() == 1 else (u, out)
    step = 8192
    for a in range(0, u2.shape[0], step):
        b = min(a + step, u2.shape[0])
        ga = g if one else g[a:b]
        torch.mul(u2[a:b].to(sd), ga, out = out2[a:b]) if dtype == sd else out2[a:b].copy_(u2[a:b].to(sd) * ga)
    return out


def _scale_rows_inplace_impl(u, g):
    # Triton retry is safe in place: a launch that raises wrote nothing.
    return _scale_rows_impl(u, g, u.dtype, out = u)


_KEEP_GRAPH = getattr(torch._C._autograd, "_get_current_graph_task_keep_graph", None)


def _stored_grad_scaler(s, saved):
    """In place only in eager backward without retain_graph (else a second [V, H] grad coexists with .grad).
    Compiled autograd traces this backward and cannot trace _KEEP_GRAPH, so it keeps the out-of-place op."""
    inplace = _OPS.get("scale_rows_inplace")
    if inplace is not None and _KEEP_GRAPH is not None and not torch.compiler.is_compiling() and all(
        type(x) is torch.Tensor and not x.requires_grad for x in saved
    ):
        try:
            keep = _KEEP_GRAPH()
        except RuntimeError:  # not inside a backward pass
            keep = True
        if not keep:
            return lambda x: inplace(x, s)
    return lambda x: torch.ops.unsloth_zoo.fused_linear_scale_rows(x, s, x.dtype)


_OPS["scale_rows_impl"] = _scale_rows_impl
_OPS["scale_rows_inplace"] = _scale_rows_inplace_impl
if "calls" in _OPS:
    CALLS = _OPS["calls"]
else:
    _OPS["calls"] = CALLS
CALLS.setdefault("jsd", 0)


# Generalized JSD between a trained student head and a frozen teacher head (TRL GKDTrainer.generalized_jsd_loss).
_JSD_OP = "fused_linear_jsd_v2"
# Live [rows, vocab] buffers per eager chunk in the statistic dtype; the chunk is sized to this budget.
_JSD_BUFFERS = 4
_JSD_CHUNK_BYTES = max(int(_env_float("UNSLOTH_FUSED_LINEAR_JSD_CHUNK_MB", 1024) * 1024**2), 1)


def _jsd_rows(rows, n_rows, V, bytes_per_row):
    # rows > 0 sets the chunk; otherwise a power of two within the budget, in [16, _MAX_ROWS].
    if rows > 0:
        return rows
    fit = max(_JSD_CHUNK_BYTES // max(bytes_per_row * V, 1), 1)
    auto = min(max(1 << (fit.bit_length() - 1), 16), _MAX_ROWS)
    return min(auto, -rows) if rows < 0 else auto


def _jsd_logprobs(h, w, bias, scale, cap, inv_t, sd, device):
    """log_softmax(transform(h @ w.T + bias) * inv_t) in sd on `device`, and its row lse; projected on w's device."""
    x = _jsd_logits(h, w, bias, sd)
    if scale != 1.0:
        x.mul_(scale)
    if cap != 0.0:
        x.div_(cap).tanh_().mul_(cap)
    if inv_t != 1.0:
        x.mul_(inv_t)
    x = x.to(device)
    lse = torch.logsumexp(x, dim = -1, keepdim = True)
    x.sub_(lse)
    return x, lse


def _jsd_logits(h, w, bias, sd):
    # On HIP fp16 _mm_fp32 is the bounded half-result GEMM: fast, and raw logits past 65504 stay finite.
    x = _mm_fp32(h, w.t()) if sd == torch.float32 else (h.to(sd) @ w.to(sd).t())
    if bias is not None:
        x.add_(bias.to(sd))
    return x


def _jsd_apply(dzc, h, w, grad_h, acc_w, idx, a, b, n_keep, N, ratio, need_h, need_w, sd):
    if need_h:
        gh = _dz_w(dzc, w) if sd == torch.float32 else dzc @ w
        if ratio is not None:
            gh.mul_(ratio)
        if n_keep == N:
            grad_h[a:b].copy_(gh)
        else:
            grad_h.index_copy_(0, idx[a:b], gh.to(grad_h.dtype))
    if need_w:
        _dz_t_h(acc_w, dzc, h)


def _jsd_chunk(ls, lt, beta, want_grad):
    """Per-row generalized JSD of log-probs ls (student) / lt (teacher), and d loss / d student logits.

    Buffers ls / lt are consumed. Interior beta: m = log((1 - b) p + b q),
    loss = b KL(q || M) + (1 - b) KL(p || M), dz = (1 - b) (g - p sum g) with g = p (ls - m)."""
    if beta == 0.0:
        q = lt.exp()
        lt.sub_(ls).mul_(q)
        row = lt.sum(-1)
        if not want_grad:
            return row, None
        return row, ls.exp_().sub_(q)
    if beta == 1.0:
        p = ls.exp()
        ls.sub_(lt)
        g = torch.mul(p, ls, out = lt)
        row = g.sum(-1)
        if not want_grad:
            return row, None
        p.mul_(row.unsqueeze(-1))
        return row, g.sub_(p)
    m = torch.add(ls, math.log1p(-beta))
    q = torch.add(lt, math.log(beta))
    torch.logaddexp(m, q, out = m)
    torch.exp(lt, out = q)
    lt.sub_(m).mul_(q)
    row_t = lt.sum(-1)
    m.neg_().add_(ls)
    p = torch.exp(ls, out = lt)
    g = torch.mul(p, m, out = q)
    row_s = g.sum(-1)
    row = beta * row_t + (1.0 - beta) * row_s
    if not want_grad:
        return row, None
    p.mul_(row_s.unsqueeze(-1))
    return row, g.sub_(p).mul_(1.0 - beta)


def _jsd_body(s, t, factor, beta: float, inv_t: float, s_scale: float, s_cap: float, t_scale: float, t_cap: float,
              want_grad: bool, grad_dtype: torch.dtype):
    """Raw fp32 logits (bias added) of both heads -> per-row JSD and dz in grad_dtype; the compiled (fused) form."""
    s = s * s_scale
    u = None
    if s_cap != 0.0:
        u = torch.tanh(s / s_cap)
        s = s_cap * u
    s = s * inv_t
    t = t * t_scale
    if t_cap != 0.0:
        t = t_cap * torch.tanh(t / t_cap)
    t = t * inv_t
    ls = s - torch.logsumexp(s, dim = -1, keepdim = True)
    lt = t - torch.logsumexp(t, dim = -1, keepdim = True)
    p = torch.exp(ls)
    if beta == 0.0:
        q = torch.exp(lt)
        row = (q * (lt - ls)).sum(-1)
        dz = p - q
    elif beta == 1.0:
        g = p * (ls - lt)
        row = g.sum(-1)
        dz = g - p * row.unsqueeze(-1)
    else:
        m = torch.logaddexp(ls + math.log1p(-beta), lt + math.log(beta))
        row_t = (torch.exp(lt) * (lt - m)).sum(-1)
        g = p * (ls - m)
        row_s = g.sum(-1)
        row = beta * row_t + (1.0 - beta) * row_s
        dz = (1.0 - beta) * (g - p * row_s.unsqueeze(-1))
    if not want_grad:
        return row, None
    dz = dz * (factor * (inv_t * s_scale))
    if u is not None:
        dz = dz * (1.0 - u * u)
    return row, dz.to(grad_dtype)


_JSD_BLOCK = 4096
# An explicit UNSLOTH_FUSED_LINEAR_JSD_CHUNK_MB sizes the Triton chunk by budget instead.
_JSD_TRITON_ROWS = os.environ.get("UNSLOTH_FUSED_LINEAR_JSD_CHUNK_MB") is None


def _jsd_triton(s, t, factor, beta, inv_t, s_scale, s_cap, t_scale, t_cap, want_grad, grad_dtype):
    """_jsd_body on CUDA with Triton: two (beta = 0) or three passes over the raw fp32 logits, dz written once."""
    # Triton launches on the current device: else a head on another card falls back and marks it broken.
    return _launch(s.device, lambda: _jsd_triton_launch(s, t, factor, beta, inv_t, s_scale, s_cap, t_scale, t_cap,
                                                         want_grad, grad_dtype))


def _jsd_triton_launch(s, t, factor, beta, inv_t, s_scale, s_cap, t_scale, t_cap, want_grad, grad_dtype):
    R, V = s.shape
    NB = triton.cdiv(V, _JSD_BLOCK)
    flags = dict(S_SCALE = s_scale != 1.0, S_CAP = s_cap != 0.0, T_SCALE = t_scale != 1.0, T_CAP = t_cap != 0.0,
                 BLOCK = _JSD_BLOCK, num_warps = 8, enable_fp_fusion = True)
    sc = (float(s_scale), float(s_cap) or 1.0, float(t_scale), float(t_cap) or 1.0, float(inv_t))
    stats = torch.empty((R, NB, 4), device = s.device, dtype = torch.float32)
    grid = (R, NB)
    _fl_jsd_stats_kernel[grid](s, t, s.stride(0), V, stats, NB, *sc, **flags)
    m = stats[..., 0::2]
    sums = stats[..., 1::2]
    top = m.amax(1, keepdim = True)
    lse = (top.squeeze(1) + torch.log((sums * torch.exp(m - top)).sum(1))).contiguous()  # [R, 2] student, teacher
    mode = 0 if beta == 0.0 else (1 if beta == 1.0 else 2)
    log1mb = math.log1p(-beta) if mode == 2 else 0.0
    logb = math.log(beta) if mode == 2 else 0.0
    part = torch.empty((R, NB, 2), device = s.device, dtype = torch.float32)
    # c = d at the student's argmax. With p ~ 1 there, d - sum_k p_k d_k cancels to rounding noise in fp32 (the
    # softmax-collapse failure); written as (d - c) - sum_k p_k (d_k - c) the peak's own term drops out exactly.
    c = torch.zeros(R, device = s.device, dtype = torch.float32)
    if mode != 0:
        j = s.argmax(-1, keepdim = True)
        sj = _torch_transform(s.gather(1, j).squeeze(1), s_scale, s_cap)[0] * inv_t - lse[:, 0]
        tj = _torch_transform(t.gather(1, j).squeeze(1), t_scale, t_cap)[0] * inv_t - lse[:, 1]
        if mode == 1:
            c = sj - tj
        else:
            c = sj - torch.logaddexp(sj + log1mb, tj + logb)
    dz = torch.empty((R, V), device = s.device, dtype = grad_dtype) if want_grad else s.new_empty((1, 1))
    factor = factor.reshape(1).to(torch.float32)
    _fl_jsd_row_kernel[grid](s, t, s.stride(0), V, lse, c, part, NB, dz, dz.stride(0), factor, float(beta), log1mb, logb,
                             *sc, MODE = mode, WRITE_DZ = want_grad and mode == 0, **flags)
    if mode == 0:
        row = part[..., 0].sum(1)
    elif mode == 1:
        rs = part[..., 0].sum(1)
        row = rs + c
    else:
        row_t = part[..., 0].sum(1)
        rs = part[..., 1].sum(1)
        row = beta * row_t + (1.0 - beta) * (rs + c)
    if not want_grad:
        return row, None
    if mode != 0:
        rs = rs.contiguous()
        _fl_jsd_dz_kernel[grid](s, t, s.stride(0), V, lse, c, rs, dz, dz.stride(0), factor,
                                1.0 if mode == 1 else 1.0 - beta, log1mb, logb, *sc, MODE = mode, **flags)
    return row, dz


_JSD_COMPILE = os.environ.get("UNSLOTH_FUSED_LINEAR_JSD_COMPILE", "1") != "0"
_JSD_COMPILED = {}


def _jsd_compiled(device):
    """Inductor-fused _jsd_body on CUDA (reads each chunk's logits a few times, writes only dz); None = eager."""
    if not _JSD_COMPILE or device.type != "cuda":
        return None
    fn = _JSD_COMPILED.get("fn", False)
    if fn is False:
        fn = None
        try:
            from unsloth_zoo.temporary_patches.common import _maybe_compile, torch_compile_options

            # Static shapes (every chunk padded to the same rows): faster and fewer live buffers than dynamic.
            compiled = _maybe_compile(dynamic = False, fullgraph = True, options = torch_compile_options)(_jsd_body)
            fn = None if compiled is _jsd_body else compiled
        except Exception:
            fn = None
        _JSD_COMPILED["fn"] = fn
    return fn


def _jsd_impl(hidden, weight, bias, t_hidden, t_weight, t_bias, mask, divisor, scaling, beta, temperature,
              s_scale, s_cap, t_scale, t_cap, compute_dtype, t_dtype, need_h, need_w, need_b, rows):
    device = weight.device
    N, H = hidden.shape
    V = weight.shape[0]
    sd = _stat_dtype(compute_dtype)
    acc = torch.float64 if device.type in ("cuda", "cpu") else torch.float32
    keep = mask.to(device = device, dtype = torch.bool).reshape(-1)
    idx = keep.nonzero().squeeze(-1)
    n_keep = int(idx.numel())
    grad = need_h or need_w or need_b
    grad_h = torch.zeros((N, H), device = device, dtype = hidden.dtype) if need_h else hidden.new_empty(0)
    acc_w = None
    if need_w:
        acc_w = torch.zeros(weight.shape, device = device,
                            dtype = torch.float32 if weight.dtype in _HALF else weight.dtype)
    acc_b = torch.zeros(V, device = device, dtype = sd) if need_b else None
    total = torch.zeros((), device = device, dtype = acc)
    inv_t = 1.0 / temperature
    divisor = divisor.to(device = device, dtype = torch.float32)
    coef = (scaling.to(device = device, dtype = torch.float32) / divisor).reshape(())
    ratio = None
    if n_keep > 0:
        fused = None
        if sd == torch.float32:
            fused = _jsd_triton if _use_triton(device) else _jsd_compiled(device)
        # Compiled body: both logits + about 1.5 buffers (measured); eager four, five with softcapping.
        n_buf = 4 if fused is not None else _JSD_BUFFERS + (s_cap != 0.0)
        R = max(_jsd_rows(rows, n_keep, V, n_buf * sd.itemsize), 1)
        if fused is _jsd_triton and rows <= 0 and _JSD_TRITON_ROWS:
            # Triton keeps 2.5 [rows, vocab] fp32 buffers; rows as in the torch.func chunked JSD (2**26 logits, [64, 1024]).
            R = min(1024, max(64, 1 << max(((1 << 26) // max(V, 1)).bit_length() - 1, 0)))
            if rows < 0:
                R = min(R, -rows)
        pad = fused is not None and fused is not _jsd_triton
        if not pad:
            R = min(R, n_keep)
        w = weight.to(compute_dtype)
        t_device = t_weight.device
        wt = t_weight.to(t_dtype)
        t_idx = idx.to(t_hidden.device)
        coef_dz = coef.to(sd)
        if grad and compute_dtype in _HALF:
            # half(coef) rounds 0.4%: dz takes a power of two, fp32 the rest.
            pow2 = torch.exp2(torch.round(torch.log2(coef)))
            ratio, coef_dz = coef / pow2, pow2
        with torch.autocast(device.type, enabled = False):
            for a in range(0, n_keep, R):
                b = min(a + R, n_keep)
                h = (hidden[a:b] if n_keep == N else hidden.index_select(0, idx[a:b])).to(compute_dtype)
                ht = t_hidden[a:b] if n_keep == N else t_hidden.index_select(0, t_idx[a:b])
                if fused is not None:
                    n = b - a
                    # One static shape for the compiled body: pad the tail (rows of zeros, sliced off below).
                    hp = h if n == R or not pad else torch.cat([h, h.new_zeros(R - n, h.shape[1])])
                    htp = ht if n == R or not pad else torch.cat([ht, ht.new_zeros(R - n, ht.shape[1])])
                    s_raw = _jsd_logits(hp, w, bias, sd)
                    with torch.no_grad():
                        t_raw = _jsd_logits(htp.to(device = t_device, dtype = t_dtype), wt, t_bias, sd).to(device)
                    del hp, htp
                    try:
                        row, dzc = fused(s_raw, t_raw, coef_dz, beta, inv_t, s_scale, s_cap, t_scale, t_cap, grad,
                                         compute_dtype)
                    except Exception as e:
                        if type(e).__name__ == "OutOfMemoryError":
                            raise
                        if fused is _jsd_triton:
                            _TRITON_BROKEN.add(device)
                        else:
                            _JSD_COMPILED["fn"] = None
                        fused = None
                        warnings.warn(f"Unsloth: fused GKD JSD body failed ({type(e).__name__}: {e}); using eager.",
                                      stacklevel = 2)
                        row, dzc = _jsd_body(s_raw, t_raw, coef_dz, beta, inv_t, s_scale, s_cap, t_scale, t_cap,
                                             grad, compute_dtype)
                    del s_raw, t_raw
                    if row.shape[0] != n:
                        row = row[:n]
                        dzc = dzc[:n] if dzc is not None else None
                    total += row.sum(dtype = acc)
                    if not grad:
                        continue
                    if need_b:
                        acc_b += dzc.sum(0, dtype = sd)
                    _jsd_apply(dzc, h, w, grad_h, acc_w, idx, a, b, n_keep, N, ratio, need_h, need_w, sd)
                    # Free this chunk's dz before the next chunk's logits are allocated.
                    del dzc, row
                    continue
                ls, lse_s = _jsd_logprobs(h, w, bias, s_scale, s_cap, inv_t, sd, device)
                with torch.no_grad():
                    lt, _ = _jsd_logprobs(ht.to(device = t_device, dtype = t_dtype), wt, t_bias, t_scale, t_cap,
                                          inv_t, sd, device)
                if grad and s_cap != 0.0:
                    # 1 - tanh^2 of the capped logit, recovered from the log-probs before they are consumed.
                    dcap = ls.add(lse_s).mul_(1.0 / (inv_t * s_cap)).square_().neg_().add_(1.0)
                else:
                    dcap = None
                row, dz = _jsd_chunk(ls, lt, beta, grad)
                total += row.sum(dtype = acc)
                if not grad:
                    continue
                factor = coef_dz * (inv_t * s_scale)
                dz.mul_(factor)
                if dcap is not None:
                    dz.mul_(dcap)
                    del dcap
                if need_b:
                    acc_b += dz.sum(0)
                dzc = dz.to(compute_dtype) if compute_dtype != sd else dz
                _jsd_apply(dzc, h, w, grad_h, acc_w, idx, a, b, n_keep, N, ratio, need_h, need_w, sd)
    loss = (total / divisor.to(acc)).float()
    if ratio is not None:
        if need_w:
            acc_w.mul_(ratio)
        if need_b:
            acc_b.mul_(ratio)
    grad_w = acc_w.to(weight.dtype) if need_w else hidden.new_empty(0)
    grad_b = acc_b.to(bias.dtype) if need_b else hidden.new_empty(0)
    return loss, grad_h, grad_w, grad_b


_OPS["jsd_impl"] = _jsd_impl
if not _OPS.get("jsd_registered", False):
    from torch.library import custom_op as _custom_op

    @_custom_op(f"unsloth_zoo::{_JSD_OP}", mutates_args = ())
    def _jsd_op(
        hidden: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor],
        t_hidden: torch.Tensor, t_weight: torch.Tensor, t_bias: Optional[torch.Tensor],
        mask: torch.Tensor, divisor: torch.Tensor, scaling: torch.Tensor, beta: float, temperature: float,
        s_scale: float, s_cap: float, t_scale: float, t_cap: float, compute_dtype: torch.dtype,
        t_dtype: torch.dtype, need_h: bool, need_w: bool, need_b: bool, rows: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _OPS["calls"]["jsd"] += 1
        return _OPS["jsd_impl"](hidden, weight, bias, t_hidden, t_weight, t_bias, mask, divisor, scaling, beta,
                                temperature, s_scale, s_cap, t_scale, t_cap, compute_dtype, t_dtype, need_h, need_w,
                                need_b, rows)

    @_jsd_op.register_fake
    def _jsd_fake(hidden, weight, bias, t_hidden, t_weight, t_bias, mask, divisor, scaling, beta, temperature,
                  s_scale, s_cap, t_scale, t_cap, compute_dtype, t_dtype, need_h, need_w, need_b, rows):
        return (
            torch.empty((), device = weight.device, dtype = torch.float32),
            hidden.new_empty(hidden.shape) if need_h else hidden.new_empty(0),
            weight.new_empty(weight.shape) if need_w else hidden.new_empty(0),
            bias.new_empty(bias.shape) if need_b else hidden.new_empty(0),
        )

    def _jsd_setup(ctx, inputs, output):
        loss, grad_h, grad_w, grad_b = output
        ctx.mark_non_differentiable(grad_h, grad_w, grad_b)
        ctx.set_materialize_grads(False)  # as the CE op
        ctx.need = inputs[17:20]
        ctx.save_for_backward(grad_h, grad_w, grad_b, inputs[8])

    def _jsd_backward(ctx, grad_loss, *_):
        if grad_loss is None:
            return (None,) * 21
        grad_h, grad_w, grad_b, scaling = ctx.saved_tensors
        acc = torch.float64 if grad_loss.device.type in ("cuda", "cpu") else torch.float32
        s = grad_loss.to(acc) / scaling.to(acc)
        need_h, need_w, need_b = ctx.need
        scale = _stored_grad_scaler(s, (grad_h, grad_w, grad_b))
        return ((scale(grad_h) if need_h else None, scale(grad_w) if need_w else None,
                 scale(grad_b) if need_b else None) + (None,) * 18)

    _jsd_op.register_autograd(_jsd_backward, setup_context = _jsd_setup)
    _OPS["jsd_registered"] = True


def _compute_dtype(hidden, weight):
    device_type = weight.device.type
    if torch.is_autocast_enabled(device_type):
        return torch.get_autocast_dtype(device_type)
    if weight.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return weight.dtype
    return hidden.dtype


def _requires_grad(x):
    return x is not None and x.requires_grad


def _head_bias(bias, weight):
    if bias is None:
        return None
    if bias.numel() != weight.shape[0]:
        raise ValueError(f"Unsloth: lm_head bias has {bias.numel()} entries for a vocabulary of {weight.shape[0]}.")
    return bias.reshape(-1).to(weight.device)


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

    The gradient is taken in forward (no recompute). `scaling` (fp16 GradScaler) multiplies stored gradients;
    backward divides it out. A zero or missing divisor or scaling counts as 1."""
    device = weight.device
    hidden2 = hidden.reshape(-1, hidden.shape[-1]).to(device)
    targets2 = targets.reshape(-1).to(device)
    grad_on = torch.is_grad_enabled()
    need_h = grad_on and hidden.requires_grad
    need_w = grad_on and weight.requires_grad
    need_b = grad_on and _requires_grad(bias)
    bias = _head_bias(bias, weight)
    label_smoothing = float(label_smoothing or 0.0)
    if not 0.0 <= label_smoothing <= 1.0:
        raise ValueError(f"Unsloth: label_smoothing must be between 0.0 and 1.0, got {label_smoothing}.")
    f32 = dict(device = device, dtype = torch.float32)
    divisor = torch.as_tensor(divisor).to(**f32).reshape(-1)[:1].reshape(())
    divisor = torch.where(divisor == 0, torch.ones_like(divisor), divisor)
    if scaling is None:
        scaling = torch.ones((), **f32)
    else:
        scaling = torch.as_tensor(scaling).to(**f32).reshape(())
        scaling = torch.where(scaling == 0, torch.ones_like(scaling), scaling)
    if logits_bias is not None:
        # Bart's final_logits_bias is [1, vocab]; a split model may keep it off the head's device.
        logits_bias = logits_bias.reshape(-1).to(device)
    loss = getattr(torch.ops.unsloth_zoo, _CE_OP)(
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

    Rows outside `mask` (or with an out-of-range target) return 0. Extras: entropy (None: gradient recomputed only
    if one arrives), vocab mean of transformed logits, log sum p^2, argmax == target."""
    device = weight.device
    lead = targets.shape
    hidden2 = hidden.reshape(-1, hidden.shape[-1]).to(device)
    targets2 = targets.reshape(-1).to(device)
    mask2 = mask.reshape(-1).to(device = device, dtype = torch.bool) if mask is not None else None
    grad_on = torch.is_grad_enabled()
    need_h = grad_on and hidden.requires_grad
    need_w = grad_on and weight.requires_grad
    need_b = grad_on and _requires_grad(bias)
    bias = _head_bias(bias, weight)
    lazy = entropy_requires_grad is None
    ent_grad = bool(return_entropy) and (lazy or bool(entropy_requires_grad))
    head_grad = need_w or need_b
    want_u = _FORWARD_GRAD and need_h and not head_grad and (not ent_grad or lazy)
    logps, ent, mean, lssp, top1, u, lse, _, _, _, _ = getattr(torch.ops.unsloth_zoo, _LP_OP)(
        hidden2, weight, bias, targets2, mask2, float(temperature),
        float(logit_scale_multiply or 0.0), float(logit_scale_divide or 0.0), float(logit_softcapping or 0.0),
        _compute_dtype(hidden, weight), bool(return_entropy), bool(return_mean_logits),
        bool(return_log_sum_sq_probs), bool(return_is_top1), want_u, need_h, need_w, need_b,
        int(rows_per_chunk or 0) or -int(max_rows_per_chunk or 0), bool(lazy and ent_grad),
    )
    out_device = hidden.device
    result = [logps.reshape(lead).to(out_device)]
    if return_entropy:
        e = ent if (lazy or entropy_requires_grad) else ent.detach()
        result.append(e.reshape(lead).to(out_device))
    if return_mean_logits:
        result.append(mean.reshape(lead).to(out_device))
    if return_log_sum_sq_probs:
        result.append(lssp.reshape(lead).to(out_device))
    if return_is_top1:
        result.append(top1.reshape(lead).to(out_device))
    return result[0] if len(result) == 1 else tuple(result)


def fused_linear_jsd(
    student_hidden, student_weight, teacher_hidden, teacher_weight, *,
    mask = None,
    labels = None,
    ignore_index = -100,
    beta = 0.5,
    temperature = 1.0,
    student_bias = None,
    teacher_bias = None,
    student_logit_scale = 1.0, student_logit_softcapping = 0.0,
    teacher_logit_scale = 1.0, teacher_logit_softcapping = 0.0,
    num_items_in_batch = None,
    scaling = None,
    rows_per_chunk = None,
):
    """TRL ``GKDTrainer.generalized_jsd_loss`` over ``transform(hidden @ weight.T + bias) / temperature`` of a student
    and a frozen teacher, one row chunk of both logits at a time; float32 scalar.

    beta 0 is forward KL(teacher || student), 1 reverse KL, else the generalized JSD with mixture
    ``(1 - beta) p_student + beta p_teacher``. Rows are kept where ``mask`` is non zero and ``labels != ignore_index``
    (either may be omitted). The sum is divided by ``num_items_in_batch`` when given (TRL >= 1.x), else by the kept
    row count (``reduction = "batchmean"``); a zero divisor counts as 1. With neither labels, mask nor count, TRL 0.x
    divides 3D logits by batch * sequence (as here), 1.x by batch: pass ``num_items_in_batch`` to pin one. The
    transform is ``* logit_scale`` then softcapping, as each model's forward. The student gradient (hidden, and head / bias when they require grad) is
    taken in forward; teacher inputs are never differentiated and may have their own hidden size and device.
    ``scaling`` (fp16 GradScaler) multiplies stored gradients; backward divides it out."""
    beta = float(beta)
    temperature = float(temperature)
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"Unsloth: GKD beta must be in [0, 1], got {beta}.")
    # Not math.isfinite: it graph-breaks on a SymFloat (dynamic = True); NaN / inf fail these comparisons.
    if not (0.0 < temperature < math.inf):
        raise ValueError(f"Unsloth: GKD temperature must be finite and > 0, got {temperature}.")
    if teacher_weight.shape[0] != student_weight.shape[0]:
        raise ValueError(f"Unsloth: GKD student vocabulary {student_weight.shape[0]} != teacher vocabulary "
                         f"{teacher_weight.shape[0]}.")
    device = student_weight.device
    hidden2 = student_hidden.reshape(-1, student_hidden.shape[-1]).to(device)
    t_hidden2 = teacher_hidden.detach().reshape(-1, teacher_hidden.shape[-1])
    if t_hidden2.shape[0] != hidden2.shape[0]:
        raise ValueError(f"Unsloth: GKD student has {hidden2.shape[0]} rows, teacher {t_hidden2.shape[0]}.")
    keep = torch.ones(hidden2.shape[0], device = device, dtype = torch.bool)
    if mask is not None:
        keep = keep & (mask.reshape(-1).to(device) != 0)
    if labels is not None:
        keep = keep & (labels.reshape(-1).to(device) != ignore_index)
    grad_on = torch.is_grad_enabled()
    need_h = grad_on and student_hidden.requires_grad
    need_w = grad_on and student_weight.requires_grad
    need_b = grad_on and _requires_grad(student_bias)
    student_bias = _head_bias(student_bias, student_weight)
    teacher_weight = teacher_weight.detach()
    teacher_bias = _head_bias(teacher_bias.detach(), teacher_weight) if teacher_bias is not None else None
    f32 = dict(device = device, dtype = torch.float32)
    if num_items_in_batch is None:
        divisor = keep.sum().to(**f32)
    else:
        divisor = torch.as_tensor(num_items_in_batch).to(**f32).reshape(-1)[:1].reshape(())
    divisor = torch.where(divisor == 0, torch.ones_like(divisor), divisor)
    if scaling is None:
        scaling = torch.ones((), **f32)
    else:
        scaling = torch.as_tensor(scaling).to(**f32).reshape(())
        scaling = torch.where(scaling == 0, torch.ones_like(scaling), scaling)
    loss = getattr(torch.ops.unsloth_zoo, _JSD_OP)(
        hidden2, student_weight, student_bias, t_hidden2, teacher_weight, teacher_bias, keep, divisor, scaling,
        beta, temperature, float(student_logit_scale or 1.0), float(student_logit_softcapping or 0.0),
        float(teacher_logit_scale or 1.0), float(teacher_logit_softcapping or 0.0),
        _compute_dtype(student_hidden, student_weight), _compute_dtype(teacher_hidden, teacher_weight),
        need_h, need_w, need_b, int(rows_per_chunk or 0),
    )[0]
    return loss
