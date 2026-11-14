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

"""Index-gather attention with one shared KV head and a per-head sink, as Triton forward + backward kernels.

out[b, t, h] = sum_i softmax_i(scale * q[b, t, h] . kv[b, idx[b, t, i]]) * kv[b, idx[b, t, i]], where the sink
logit (unscaled) only joins the softmax denominator, -1 / out-of-range slots are skipped, and a row with no valid
slot returns zero (DeepSeek-V3.2 / V4.1 sparse_attn semantics).

Nothing gathered is materialised: each program walks its own index list in BLOCK_K tiles, loads every gathered KV
row once and uses it as both key and value (tl.dot for q.K^T and P.V), with an online softmax. The backward
recomputes P from the saved log-sum-exp; dq stays in registers and the key + value gradient of every gathered row
is accumulated with fp32 atomics into a [b, n, d] buffer (no index_put / sort; summation order is not
deterministic). d(sink) is a reduction over the saved log-sum-exp in torch.

Ops are versioned custom ops with fakes and autograd (no graph break under torch.compile; bump _V on any schema or
formula change: Inductor caches key the op name). Sequence length, slot count and pool length are runtime
arguments (do_not_specialize); the compile key is (head_dim, tile sizes, has_sink, dtype). Head dims are powers of
two from 16 to 512. Configs are fixed per shared-memory tier of the operand's device, never autotuned; a launch
refused for shared memory steps down to the next config (on 64 KB parts the head_dim 512 backward splits D across
two programs, each recomputing the full-D logits), and the last refusal propagates.
"""

__all__ = ["sparse_mqa_attention", "sparse_mqa_supported", "SPARSE_MQA_STATS"]

from typing import Tuple

import torch
import triton
import triton.language as tl

SPARSE_MQA_STATS = {"fwd": 0, "bwd": 0}
_V = "v2"


@triton.jit(do_not_specialize = ["S", "N", "TOPK", "H", "stride_ib", "stride_is"])
def _sparse_mqa_fwd(
    Q, KV, IDX, SINK, O, LSE,
    S, N, TOPK, H,
    stride_qb, stride_qs, stride_qh,
    stride_kvb, stride_kvn,
    stride_ib, stride_is,
    stride_ob, stride_os, stride_oh,
    stride_lb, stride_ls,
    scale,
    D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr, HAS_SINK: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    hb = tl.program_id(1)
    b = tl.program_id(2).to(tl.int64)
    offs_h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    h_ok = offs_h < H
    offs_d = tl.arange(0, D)
    q = tl.load(Q + b * stride_qb + t * stride_qs + offs_h[:, None] * stride_qh + offs_d[None, :],
                mask = h_ok[:, None], other = 0.0)
    # Finite floor, not -inf: a tile with no valid slot then rescales by exp(0) instead of exp(nan).
    m_i = tl.full((BLOCK_H,), -1.0e30, tl.float32)
    l_i = tl.zeros((BLOCK_H,), tl.float32)
    acc = tl.zeros((BLOCK_H, D), tl.float32)
    idx_row = IDX + b * stride_ib + t * stride_is
    kv_base = KV + b * stride_kvb
    for k0 in tl.range(0, TOPK, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        idx = tl.load(idx_row + offs_k, mask = offs_k < TOPK, other = -1)
        valid = (idx >= 0) & (idx < N)
        kv = tl.load(kv_base + idx.to(tl.int64)[:, None] * stride_kvn + offs_d[None, :],
                     mask = valid[:, None], other = 0.0)
        s = tl.dot(q, tl.trans(kv)) * scale
        s = tl.where(valid[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
    denom = l_i
    if HAS_SINK:
        sink = tl.load(SINK + offs_h, mask = h_ok, other = float("-inf")).to(tl.float32)
        denom = l_i + tl.exp(sink - m_i)
    # No valid slot: zero output and lse = +inf, so the backward's p = exp(s - lse) and the sink weight are 0.
    nonempty = l_i > 0
    safe = tl.where(nonempty, denom, 1.0)
    out = tl.where(nonempty[:, None], acc / safe[:, None], 0.0)
    lse = tl.where(nonempty, m_i + tl.log(safe), float("inf"))
    tl.store(O + b * stride_ob + t * stride_os + offs_h[:, None] * stride_oh + offs_d[None, :],
             out.to(O.dtype.element_ty), mask = h_ok[:, None])
    tl.store(LSE + b * stride_lb + t * stride_ls + offs_h, lse, mask = h_ok)


@triton.jit(do_not_specialize = ["S", "N", "TOPK", "H", "stride_ib", "stride_is"])
def _sparse_mqa_bwd(
    Q, KV, IDX, DO, LSE, DELTA, DQ, DKV,
    S, N, TOPK, H,
    stride_qb, stride_qs, stride_qh,
    stride_kvb, stride_kvn,
    stride_ib, stride_is,
    stride_ob, stride_os, stride_oh,
    stride_lb, stride_ls,
    stride_dkb, stride_dkn,
    scale,
    D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_D: tl.constexpr,
):
    # SPLIT_D == D: one program per (token, head block). SPLIT_D < D (64 KB shared-memory parts, e.g. T4 at
    # head_dim 512): D // SPLIT_D programs per (token, head block), each recomputing the full-D logits and dO.V^T
    # in SPLIT_D-wide slices and writing only its own SPLIT_D columns of dq and dkv.
    NSPLIT: tl.constexpr = D // SPLIT_D
    t = tl.program_id(0).to(tl.int64)
    hb = tl.program_id(1) // NSPLIT
    dc = tl.program_id(1) % NSPLIT
    b = tl.program_id(2).to(tl.int64)
    offs_h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    h_ok = offs_h < H
    offs_c = tl.arange(0, SPLIT_D)
    offs_d = dc * SPLIT_D + offs_c
    q_row = b * stride_qb + t * stride_qs + offs_h[:, None] * stride_qh
    q_base = Q + q_row
    do_base = DO + b * stride_ob + t * stride_os + offs_h[:, None] * stride_oh
    q = tl.load(q_base + offs_d[None, :], mask = h_ok[:, None], other = 0.0)
    do = tl.load(do_base + offs_d[None, :], mask = h_ok[:, None], other = 0.0)
    lse = tl.load(LSE + b * stride_lb + t * stride_ls + offs_h, mask = h_ok, other = float("inf"))
    delta = tl.load(DELTA + b * stride_lb + t * stride_ls + offs_h, mask = h_ok, other = 0.0)
    dq = tl.zeros((BLOCK_H, SPLIT_D), tl.float32)
    idx_row = IDX + b * stride_ib + t * stride_is
    kv_base = KV + b * stride_kvb
    dkv_base = DKV + b * stride_dkb
    for k0 in tl.range(0, TOPK, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        idx = tl.load(idx_row + offs_k, mask = offs_k < TOPK, other = -1)
        valid = (idx >= 0) & (idx < N)
        rows = idx.to(tl.int64)
        kv = tl.load(kv_base + rows[:, None] * stride_kvn + offs_d[None, :], mask = valid[:, None], other = 0.0)
        if NSPLIT == 1:
            s = tl.dot(q, tl.trans(kv))
            dp = tl.dot(do, tl.trans(kv))
        else:
            s = tl.zeros((BLOCK_H, BLOCK_K), tl.float32)
            dp = tl.zeros((BLOCK_H, BLOCK_K), tl.float32)
            for c in tl.static_range(NSPLIT):
                cols = c * SPLIT_D + offs_c
                kv_c = tl.load(kv_base + rows[:, None] * stride_kvn + cols[None, :], mask = valid[:, None], other = 0.0)
                s += tl.dot(tl.load(q_base + cols[None, :], mask = h_ok[:, None], other = 0.0), tl.trans(kv_c))
                dp += tl.dot(tl.load(do_base + cols[None, :], mask = h_ok[:, None], other = 0.0), tl.trans(kv_c))
        p = tl.where(valid[None, :], tl.exp(s * scale - lse[:, None]), 0.0)
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(kv.dtype), kv)
        # K == V: value gradient p^T dO plus key gradient scale * ds^T q, one atomic add per gathered element.
        dkv = tl.dot(tl.trans(p.to(kv.dtype)), do) + tl.dot(tl.trans(ds.to(kv.dtype)), q) * scale
        tl.atomic_add(dkv_base + rows[:, None] * stride_dkn + offs_d[None, :], dkv, mask = valid[:, None])
    tl.store(DQ + q_row + offs_d[None, :], (dq * scale).to(DQ.dtype.element_ty), mask = h_ok[:, None])


_SMEM = {}


def _shared_memory(device):
    if device.type != "cuda":  # TRITON_INTERPRET=1 tests on CPU
        return 48 * 1024
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _SMEM:
        props = torch.cuda.get_device_properties(index)
        _SMEM[index] = getattr(props, "shared_memory_per_block_optin", 48 * 1024)
    return _SMEM[index]


def _device_ctx(device):
    import contextlib
    return torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext()


def _configs(device, d, backward):
    """Launch configs (BLOCK_K, num_warps, num_stages, SPLIT_D), first choice first, by the operand device's shared
    memory per block (not device 0). Shared memory needed (Triton 3.8, cross-compiled per target with
    scripts/sparse_attn/smem_targets.py): head_dim 512 backward without a split is 96 KB on sm75 and 64 KB on
    sm80+, so 64 KB parts (T4) split D in two (48 KB); the forward is 64 KB on sm75 and 36 KB on sm8x / sm12x.
    Later entries are the fallbacks taken when a launch is refused for shared memory."""
    smem = _shared_memory(device)
    if d >= 512:
        if backward:
            full, split = (16, 8, 1, d), (16, 4, 1, d // 2)
            return (full, split) if smem >= 96 * 1024 else (split,)
        return ((32, 8, 1, d), (16, 4, 1, d)) if smem >= 160 * 1024 else ((16, 4, 1, d),)
    if d >= 256:
        if backward:
            return ((32, 4, 1, d), (16, 4, 1, d)) if smem >= 160 * 1024 else ((16, 4, 1, d), (16, 4, 1, d // 2))
        return ((32, 4, 1, d), (16, 4, 1, d))
    return ((32, 4, 2, d), (32, 4, 1, d)) if smem >= 96 * 1024 else ((32, 4, 1, d), (16, 4, 1, d))


_CHOSEN = {}


def _launch(kernel, grid_fn, args, device, d, backward, **const):
    """Launch with the first config that fits; a config refused for shared memory moves to the next one and the
    choice is remembered per (device, head_dim, direction, dtype). The last refusal propagates (sparse_attn then
    latches this backend for the shape class and falls through)."""
    key = (device.type, device.index, d, backward, const.pop("_dtype"))
    configs = _configs(device, d, backward)
    start = _CHOSEN.get(key, 0)
    for i in range(start, len(configs)):
        block_k, warps, stages, split = configs[i]
        extra = dict(SPLIT_D = split) if backward else {}
        try:
            with _device_ctx(device):
                kernel[grid_fn(split)](*args, D = d, BLOCK_K = block_k, num_warps = warps, num_stages = stages,
                                       **extra, **const)
        except triton.runtime.errors.OutOfResources:
            if i + 1 == len(configs):
                raise
            continue
        _CHOSEN[key] = i
        return configs[i]


_BLOCK_H = 16


def sparse_mqa_supported(q, kv):
    """CUDA, fp16 / bf16 (bf16 needs sm80+), head_dim a power of two in [16, 512], not ROCm."""
    if not (q.is_cuda and kv.is_cuda and torch.version.hip is None):
        return False
    if q.dtype not in (torch.float16, torch.bfloat16) or kv.dtype != q.dtype:
        return False
    d = q.shape[-1]
    if d < 16 or d > 512 or d & (d - 1):
        return False
    if q.dtype == torch.bfloat16 and torch.cuda.get_device_capability(q.device)[0] < 8:
        return False
    return True


@torch.library.custom_op(f"unsloth_zoo::sparse_mqa_fwd_{_V}", mutates_args = ())
def _fwd_op(q: torch.Tensor, kv: torch.Tensor, idx: torch.Tensor, sink: torch.Tensor, scale: float,
            has_sink: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    b, s, h, d = q.shape
    q = q if q.stride(-1) == 1 else q.contiguous()
    kv = kv if kv.stride(-1) == 1 else kv.contiguous()
    idx = idx.contiguous()
    out = torch.empty_like(q)
    lse = torch.empty((b, s, h), device = q.device, dtype = torch.float32)
    if s == 0 or b == 0:
        return out, lse
    _launch(_sparse_mqa_fwd, lambda split: (s, triton.cdiv(h, _BLOCK_H), b), (
            q, kv, idx, sink, out, lse,
            s, kv.shape[1], idx.shape[-1], h,
            q.stride(0), q.stride(1), q.stride(2),
            kv.stride(0), kv.stride(1),
            idx.stride(0), idx.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            lse.stride(0), lse.stride(1),
            scale,
        ), q.device, d, False, BLOCK_H = _BLOCK_H, HAS_SINK = has_sink, _dtype = q.dtype)
    SPARSE_MQA_STATS["fwd"] += 1
    return out, lse


@_fwd_op.register_fake
def _(q, kv, idx, sink, scale, has_sink):
    b, s, h, d = q.shape
    return torch.empty_like(q), q.new_empty((b, s, h), dtype = torch.float32)


@torch.library.custom_op(f"unsloth_zoo::sparse_mqa_bwd_{_V}", mutates_args = ())
def _bwd_op(q: torch.Tensor, kv: torch.Tensor, idx: torch.Tensor, grad_out: torch.Tensor, lse: torch.Tensor,
            delta: torch.Tensor, scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
    b, s, h, d = q.shape
    q = q if q.stride(-1) == 1 else q.contiguous()
    kv = kv if kv.stride(-1) == 1 else kv.contiguous()
    grad_out = grad_out.contiguous()
    idx = idx.contiguous()
    lse, delta = lse.contiguous(), delta.contiguous()
    dq = torch.empty_like(q)
    dkv = torch.zeros(kv.shape, device = kv.device, dtype = torch.float32)
    if s and b:
        _launch(_sparse_mqa_bwd, lambda split: (s, triton.cdiv(h, _BLOCK_H) * (d // split), b), (
                q, kv, idx, grad_out, lse, delta, dq, dkv,
                s, kv.shape[1], idx.shape[-1], h,
                q.stride(0), q.stride(1), q.stride(2),
                kv.stride(0), kv.stride(1),
                idx.stride(0), idx.stride(1),
                grad_out.stride(0), grad_out.stride(1), grad_out.stride(2),
                lse.stride(0), lse.stride(1),
                dkv.stride(0), dkv.stride(1),
                scale,
            ), q.device, d, True, BLOCK_H = _BLOCK_H, _dtype = q.dtype)
    SPARSE_MQA_STATS["bwd"] += 1
    return dq, dkv.to(kv.dtype)


@_bwd_op.register_fake
def _(q, kv, idx, grad_out, lse, delta, scale):
    return torch.empty_like(q), torch.empty_like(kv)


def _setup_context(ctx, inputs, output):
    q, kv, idx, sink, scale, has_sink = inputs
    out, lse = output
    ctx.save_for_backward(q, kv, idx, sink, out, lse)
    ctx.scale, ctx.has_sink = scale, has_sink


def _backward(ctx, grad_out, grad_lse):
    q, kv, idx, sink, out, lse = ctx.saved_tensors
    delta = (grad_out.float() * out.float()).sum(-1)
    dq, dkv = _bwd_op(q, kv, idx, grad_out, lse, delta, ctx.scale)
    dsink = None
    if ctx.has_sink and ctx.needs_input_grad[3]:
        # d softmax / d sink = -p_sink * delta; empty rows carry lse = +inf, so p_sink = 0 there.
        dsink = -(torch.exp(sink.float().view(1, 1, -1) - lse) * delta).sum((0, 1)).to(sink.dtype)
    return dq, dkv, None, dsink, None, None


_fwd_op.register_autograd(_backward, setup_context = _setup_context)


def sparse_mqa_attention(q, kv, attn_sink, topk_idxs, softmax_scale):
    """q [b, s, h, d], kv [b, n, d], attn_sink [h] or None, topk_idxs [b, s, k] int (-1 = unused) -> [b, s, h, d]."""
    # A -inf sink contributes exp(-inf) = 0 in the kernel: no value inspection (host sync) needed.
    has_sink = attn_sink is not None
    sink = attn_sink if has_sink else q.new_full((q.shape[2],), float("-inf"), dtype = torch.float32)
    out, _ = _fwd_op(q, kv, topk_idxs, sink, float(softmax_scale), has_sink)
    return out
