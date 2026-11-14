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

__all__ = ["sparse_mqa_attention", "sparse_mqa_window_attention", "sparse_mqa_window_supported",
           "window_hd512_attention", "window_hd512_supported", "sparse_mqa_supported", "SPARSE_MQA_STATS"]

import importlib.util
import math
import os
import sys
from typing import Tuple

import torch
import triton
import triton.language as tl

SPARSE_MQA_STATS = {"fwd": 0, "bwd": 0, "window_fwd": 0, "window_bwd": 0, "hd512": 0}
_V = "v3"


@triton.jit(do_not_specialize = ["S", "N", "TOPK", "H", "IDX_GROUP", "stride_ib", "stride_is"])
def _sparse_mqa_fwd(
    Q, KV, IDX, SINK, O, LSE,
    S, N, TOPK, H, IDX_GROUP,
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
    # IDX_GROUP batches share one index row (the GQA fold: hk KV heads per batch read one selection).
    idx_row = IDX + (b // IDX_GROUP) * stride_ib + t * stride_is
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


@triton.jit(do_not_specialize = ["S", "N", "TOPK", "H", "IDX_GROUP", "stride_ib", "stride_is"])
def _sparse_mqa_bwd(
    Q, KV, IDX, DO, LSE, DELTA, DQ, DKV,
    S, N, TOPK, H, IDX_GROUP,
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
    # IDX_GROUP batches share one index row (the GQA fold: hk KV heads per batch read one selection).
    idx_row = IDX + (b // IDX_GROUP) * stride_ib + t * stride_is
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


@triton.jit(do_not_specialize = ["S", "H", "W"])
def _window_fwd(
    Q, KV, OC, LSEC, SINK, O, LSE,
    S, H, W,
    stride_qb, stride_qs, stride_qh,
    stride_kvb, stride_kvn,
    stride_lb, stride_ls,
    scale,
    D: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr,
    HAS_PRIOR: tl.constexpr, HAS_SINK: tl.constexpr,
):
    # Query t sees window rows (t - W, t] of KV (one shared head). BLOCK_T consecutive tokens x BLOCK_H heads form
    # the M rows of one program, so each window KV tile is loaded once for all of them. HAS_PRIOR: start the online
    # softmax from a normalised partial (out, lse) over other keys (the compressed slots), merging exactly.
    M: tl.constexpr = BLOCK_T * BLOCK_H
    t0 = tl.program_id(0) * BLOCK_T
    hb = tl.program_id(1)
    b = tl.program_id(2).to(tl.int64)
    r = tl.arange(0, M)
    tok = t0 + r // BLOCK_H
    head = hb * BLOCK_H + r % BLOCK_H
    ok = (tok < S) & (head < H)
    offs_d = tl.arange(0, D)
    row = b * stride_qb + tok.to(tl.int64) * stride_qs + head * stride_qh
    q = tl.load(Q + row[:, None] + offs_d[None, :], mask = ok[:, None], other = 0.0)
    lrow = b * stride_lb + tok.to(tl.int64) * stride_ls + head
    if HAS_PRIOR:
        lse_c = tl.load(LSEC + lrow, mask = ok, other = float("inf"))
        prior = lse_c < float("inf")
        m_i = tl.where(prior, lse_c, -1.0e30)
        l_i = tl.where(prior, 1.0, 0.0)
        acc = tl.load(OC + row[:, None] + offs_d[None, :], mask = (ok & prior)[:, None], other = 0.0).to(tl.float32)
    else:
        m_i = tl.full((M,), -1.0e30, tl.float32)
        l_i = tl.zeros((M,), tl.float32)
        acc = tl.zeros((M, D), tl.float32)
    lo = tl.maximum(t0 - W + 1, 0)
    hi = tl.minimum(t0 + BLOCK_T, S)
    kv_base = KV + b * stride_kvb
    for n0 in tl.range(lo, hi, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        kv = tl.load(kv_base + offs_n.to(tl.int64)[:, None] * stride_kvn + offs_d[None, :],
                     mask = (offs_n < hi)[:, None], other = 0.0)
        s = tl.dot(q, tl.trans(kv)) * scale
        valid = (offs_n[None, :] <= tok[:, None]) & (offs_n[None, :] > tok[:, None] - W) & (offs_n[None, :] < hi)
        s = tl.where(valid, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
    denom = l_i
    if HAS_SINK:
        sink = tl.load(SINK + head, mask = ok, other = float("-inf")).to(tl.float32)
        denom = l_i + tl.exp(sink - m_i)
    nonempty = l_i > 0
    safe = tl.where(nonempty, denom, 1.0)
    out = tl.where(nonempty[:, None], acc / safe[:, None], 0.0)
    lse = tl.where(nonempty, m_i + tl.log(safe), float("inf"))
    tl.store(O + row[:, None] + offs_d[None, :], out.to(O.dtype.element_ty), mask = ok[:, None])
    tl.store(LSE + lrow, lse, mask = ok)


@triton.jit(do_not_specialize = ["S", "H", "W"])
def _window_bwd_dq(
    Q, KV, DO, LSE, DELTA, DQ_IN, DQ,
    S, H, W,
    stride_qb, stride_qs, stride_qh,
    stride_kvb, stride_kvn,
    stride_lb, stride_ls,
    scale,
    D: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, HAS_DQ_IN: tl.constexpr,
):
    M: tl.constexpr = BLOCK_T * BLOCK_H
    t0 = tl.program_id(0) * BLOCK_T
    hb = tl.program_id(1)
    b = tl.program_id(2).to(tl.int64)
    r = tl.arange(0, M)
    tok = t0 + r // BLOCK_H
    head = hb * BLOCK_H + r % BLOCK_H
    ok = (tok < S) & (head < H)
    offs_d = tl.arange(0, D)
    row = b * stride_qb + tok.to(tl.int64) * stride_qs + head * stride_qh
    q = tl.load(Q + row[:, None] + offs_d[None, :], mask = ok[:, None], other = 0.0)
    do = tl.load(DO + row[:, None] + offs_d[None, :], mask = ok[:, None], other = 0.0)
    lrow = b * stride_lb + tok.to(tl.int64) * stride_ls + head
    lse = tl.load(LSE + lrow, mask = ok, other = float("inf"))
    delta = tl.load(DELTA + lrow, mask = ok, other = 0.0)
    dq = tl.zeros((M, D), tl.float32)
    lo = tl.maximum(t0 - W + 1, 0)
    hi = tl.minimum(t0 + BLOCK_T, S)
    kv_base = KV + b * stride_kvb
    for n0 in tl.range(lo, hi, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        kv = tl.load(kv_base + offs_n.to(tl.int64)[:, None] * stride_kvn + offs_d[None, :],
                     mask = (offs_n < hi)[:, None], other = 0.0)
        s = tl.dot(q, tl.trans(kv)) * scale
        valid = (offs_n[None, :] <= tok[:, None]) & (offs_n[None, :] > tok[:, None] - W) & (offs_n[None, :] < hi)
        p = tl.where(valid, tl.exp(s - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(kv))
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(kv.dtype), kv)
    dq = dq * scale
    if HAS_DQ_IN:
        dq += tl.load(DQ_IN + row[:, None] + offs_d[None, :], mask = ok[:, None], other = 0.0).to(tl.float32)
    tl.store(DQ + row[:, None] + offs_d[None, :], dq.to(DQ.dtype.element_ty), mask = ok[:, None])


@triton.jit(do_not_specialize = ["S", "H", "W"])
def _window_bwd_dkv(
    Q, KV, DO, LSE, DELTA, DKV,
    S, H, W,
    stride_qb, stride_qs, stride_qh,
    stride_kvb, stride_kvn,
    stride_lb, stride_ls,
    scale,
    D: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # One program per BLOCK_N window rows: walks every (token, head) that sees them, tokens [n0, n0 + BLOCK_N - 1 + W),
    # and accumulates key + value gradient in registers (no atomics).
    M: tl.constexpr = BLOCK_T * BLOCK_H
    n0 = tl.program_id(0) * BLOCK_N
    b = tl.program_id(1).to(tl.int64)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    n_ok = offs_n < S
    offs_d = tl.arange(0, D)
    kv = tl.load(KV + b * stride_kvb + offs_n.to(tl.int64)[:, None] * stride_kvn + offs_d[None, :],
                 mask = n_ok[:, None], other = 0.0)
    dkv = tl.zeros((BLOCK_N, D), tl.float32)
    r = tl.arange(0, M)
    t_hi = tl.minimum(n0 + BLOCK_N - 1 + W, S)
    for hb in tl.range(0, tl.cdiv(H, BLOCK_H)):
        head = hb * BLOCK_H + r % BLOCK_H
        for t0 in tl.range(n0, t_hi, BLOCK_T):
            tok = t0 + r // BLOCK_H
            ok = (tok < S) & (head < H)
            row = b * stride_qb + tok.to(tl.int64) * stride_qs + head * stride_qh
            q = tl.load(Q + row[:, None] + offs_d[None, :], mask = ok[:, None], other = 0.0)
            do = tl.load(DO + row[:, None] + offs_d[None, :], mask = ok[:, None], other = 0.0)
            lrow = b * stride_lb + tok.to(tl.int64) * stride_ls + head
            lse = tl.load(LSE + lrow, mask = ok, other = float("inf"))
            delta = tl.load(DELTA + lrow, mask = ok, other = 0.0)
            s = tl.dot(q, tl.trans(kv)) * scale
            valid = (offs_n[None, :] <= tok[:, None]) & (offs_n[None, :] > tok[:, None] - W) & n_ok[None, :] & ok[:, None]
            p = tl.where(valid, tl.exp(s - lse[:, None]), 0.0)
            dp = tl.dot(do, tl.trans(kv))
            ds = p * (dp - delta[:, None])
            dkv += tl.dot(tl.trans(p.to(kv.dtype)), do) + tl.dot(tl.trans(ds.to(kv.dtype)), q) * scale
    tl.store(DKV + b * stride_kvb + offs_n.to(tl.int64)[:, None] * stride_kvn + offs_d[None, :],
             dkv.to(DKV.dtype.element_ty), mask = n_ok[:, None])


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
    return _index_fwd(q, kv, idx, sink, scale, has_sink)


def _index_fwd(q, kv, idx, sink, scale, has_sink):
    group = q.shape[0] // idx.shape[0] if idx.shape[0] else 1
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
            s, kv.shape[1], idx.shape[-1], h, group,
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
    return _index_bwd(q, kv, idx, grad_out, lse, delta, scale)


def _index_bwd(q, kv, idx, grad_out, lse, delta, scale):
    b, s, h, d = q.shape
    group = b // idx.shape[0] if idx.shape[0] else 1
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
                s, kv.shape[1], idx.shape[-1], h, group,
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
    """q [b, s, h, d], kv [b, n, d], attn_sink [h] or None, topk_idxs [b, s, k] int (-1 = unused) -> [b, s, h, d].
    topk_idxs may also be [b // g, s, k]: each run of g consecutive batches reads one index row (no repeat)."""
    if topk_idxs.shape[0] == 0 or q.shape[0] % topk_idxs.shape[0]:
        raise ValueError(f"sparse_mqa_attention: batch {q.shape[0]} is not a multiple of the index batch {topk_idxs.shape[0]}")
    # A -inf sink contributes exp(-inf) = 0 in the kernel: no value inspection (host sync) needed.
    has_sink = attn_sink is not None
    sink = attn_sink if has_sink else q.new_full((q.shape[2],), float("-inf"), dtype = torch.float32)
    out, _ = _fwd_op(q, kv, topk_idxs, sink, float(softmax_scale), has_sink)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Window form: per query t, the window rows (t - W, t] of window_kv plus listed compressed rows, one sink.
# The window part is a banded kernel over contiguous rows (tokens x heads share every KV tile; key + value gradient
# per KV tile in registers, no atomics); the compressed part is the index kernel above without a sink. The two
# merge exactly: the window forward starts its online softmax from the compressed (out, lse), and both backwards
# take the merged lse, so each recomputes its own share of the probabilities.

def _window_configs(device, d, kind):
    """(BLOCK_T, BLOCK_N, num_warps, num_stages) per kernel kind ("fwd", "dq", "dkv"); first choice first."""
    smem = _shared_memory(device)
    big = smem >= 96 * 1024
    if d >= 512:
        if kind == "dkv":
            return ((1, 16, 8, 1),)
        return ((2, 16, 8, 1), (1, 16, 8, 1)) if big else ((1, 16, 8, 1),)
    if d >= 256:
        return ((2, 32, 4, 1), (1, 16, 4, 1)) if big else ((1, 16, 4, 1),)
    return ((4, 32, 4, 2), (2, 32, 4, 1)) if big else ((2, 32, 4, 1), (1, 16, 4, 1))


def _launch_window(kernel, kind, grid_fn, args, device, d, dtype, **const):
    key = ("window", kind, device.type, device.index, d, dtype)
    configs = _window_configs(device, d, kind)
    for i in range(_CHOSEN.get(key, 0), len(configs)):
        block_t, block_n, warps, stages = configs[i]
        try:
            with _device_ctx(device):
                kernel[grid_fn(block_t, block_n)](*args, D = d, BLOCK_T = block_t, BLOCK_H = _BLOCK_H,
                                                  BLOCK_N = block_n, num_warps = warps, num_stages = stages, **const)
        except triton.runtime.errors.OutOfResources:
            if i + 1 == len(configs):
                raise
            continue
        _CHOSEN[key] = i
        return configs[i]


@torch.library.custom_op(f"unsloth_zoo::sparse_mqa_window_fwd_{_V}", mutates_args = ())
def _wfwd_op(q: torch.Tensor, wkv: torch.Tensor, ckv: torch.Tensor, idx: torch.Tensor, sink: torch.Tensor,
             scale: float, window: int, has_sink: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    b, s, h, d = q.shape
    q, wkv, ckv = q.contiguous(), wkv.contiguous(), ckv.contiguous()
    out = torch.empty_like(q)
    lse = torch.empty((b, s, h), device = q.device, dtype = torch.float32)
    if s == 0 or b == 0:
        return out, lse
    has_prior = idx.shape[-1] > 0 and ckv.shape[1] > 0
    if has_prior:
        no_sink = torch.full((h,), float("-inf"), device = q.device, dtype = torch.float32)
        out_c, lse_c = _index_fwd(q, ckv, idx, no_sink, scale, False)
    else:
        out_c, lse_c = out, lse
    _launch_window(_window_fwd, "fwd", lambda bt, bn: (triton.cdiv(s, bt), triton.cdiv(h, _BLOCK_H), b), (
        q, wkv, out_c, lse_c, sink, out, lse,
        s, h, int(window),
        q.stride(0), q.stride(1), q.stride(2), wkv.stride(0), wkv.stride(1),
        lse.stride(0), lse.stride(1),
        scale,
    ), q.device, d, q.dtype, HAS_PRIOR = has_prior, HAS_SINK = has_sink)
    SPARSE_MQA_STATS["window_fwd"] += 1
    return out, lse


@_wfwd_op.register_fake
def _(q, wkv, ckv, idx, sink, scale, window, has_sink):
    b, s, h, d = q.shape
    return torch.empty_like(q), q.new_empty((b, s, h), dtype = torch.float32)


@torch.library.custom_op(f"unsloth_zoo::sparse_mqa_window_bwd_{_V}", mutates_args = ())
def _wbwd_op(q: torch.Tensor, wkv: torch.Tensor, ckv: torch.Tensor, idx: torch.Tensor, grad_out: torch.Tensor,
             lse: torch.Tensor, delta: torch.Tensor, scale: float,
             window: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, s, h, d = q.shape
    q, wkv, ckv = q.contiguous(), wkv.contiguous(), ckv.contiguous()
    grad_out, lse, delta = grad_out.contiguous(), lse.contiguous(), delta.contiguous()
    dq = torch.empty_like(q)
    dwkv = torch.empty_like(wkv)
    if s == 0 or b == 0:
        return dq, dwkv.zero_(), torch.zeros_like(ckv)
    has_prior = idx.shape[-1] > 0 and ckv.shape[1] > 0
    if has_prior:
        dq_c, dckv = _index_bwd(q, ckv, idx, grad_out, lse, delta, scale)
    else:
        dq_c, dckv = dq, torch.zeros_like(ckv)
    args = (s, h, int(window), q.stride(0), q.stride(1), q.stride(2), wkv.stride(0), wkv.stride(1),
            lse.stride(0), lse.stride(1), scale)
    _launch_window(_window_bwd_dq, "dq", lambda bt, bn: (triton.cdiv(s, bt), triton.cdiv(h, _BLOCK_H), b),
                   (q, wkv, grad_out, lse, delta, dq_c, dq) + args, q.device, d, q.dtype, HAS_DQ_IN = has_prior)
    _launch_window(_window_bwd_dkv, "dkv", lambda bt, bn: (triton.cdiv(s, bn), b),
                   (q, wkv, grad_out, lse, delta, dwkv) + args, q.device, d, q.dtype)
    SPARSE_MQA_STATS["window_bwd"] += 1
    return dq, dwkv, dckv


@_wbwd_op.register_fake
def _(q, wkv, ckv, idx, grad_out, lse, delta, scale, window):
    return torch.empty_like(q), torch.empty_like(wkv), torch.empty_like(ckv)


def _wsetup_context(ctx, inputs, output):
    q, wkv, ckv, idx, sink, scale, window, has_sink = inputs
    out, lse = output
    ctx.save_for_backward(q, wkv, ckv, idx, sink, out, lse)
    ctx.scale, ctx.window, ctx.has_sink = scale, window, has_sink


def _wbackward(ctx, grad_out, grad_lse):
    q, wkv, ckv, idx, sink, out, lse = ctx.saved_tensors
    delta = (grad_out.float() * out.float()).sum(-1)
    dq, dwkv, dckv = _wbwd_op(q, wkv, ckv, idx, grad_out, lse, delta, ctx.scale, ctx.window)
    dsink = None
    if ctx.has_sink and ctx.needs_input_grad[4]:
        dsink = -(torch.exp(sink.float().view(1, 1, -1) - lse) * delta).sum((0, 1)).to(sink.dtype)
    return dq, dwkv, dckv, None, dsink, None, None, None


_wfwd_op.register_autograd(_wbackward, setup_context = _wsetup_context)


def sparse_mqa_window_supported(q, kv):
    """The window kernels at head_dim 512 need 88 KB (dq) / 65 KB (dkv) of shared memory on sm80+ and 96 KB on sm75
    (cross-compiled, scripts/sparse_attn/smem_targets.py window): 64 KB parts keep the index path for that size."""
    if not sparse_mqa_supported(q, kv):
        return False
    return q.shape[-1] < 512 or _shared_memory(q.device) >= 96 * 1024


def sparse_mqa_window_attention(q, window_kv, compress_kv, compress_idxs, attn_sink, softmax_scale, window_size):
    """q [b, s, h, d]; window_kv [b, s, d] (query t sees rows (t - window_size, t]); compress_kv [b, nc, d] with
    compress_idxs [b, s, k] into it (-1 = unused); attn_sink [h] or None -> [b, s, h, d]."""
    has_sink = attn_sink is not None
    sink = attn_sink if has_sink else q.new_full((q.shape[2],), float("-inf"), dtype = torch.float32)
    out, _ = _wfwd_op(q, window_kv, compress_kv, compress_idxs, sink, float(softmax_scale), int(window_size), has_sink)
    return out



# ---------------------------------------------------------------------------------------------------------------
# Window-only layers (no compressed slots) on Unsloth's head_dim 512 flash kernel (unsloth/kernels/
# large_head_attention.py: query blocks x KV tiles over per-row intervals [lo, hi], GQA heads read in place), with
# the sink applied exactly around it. The kernel returns the window-only log-sum-exp; with
# c = exp(lse_window - lse_total) = sigmoid(lse_window - sink), the output is c * o, and its backward is the
# kernel's own backward called with dO * c and the final output (P' = c P and delta' = c * rowsum(dO . o) make
# P (c dP - c^2 rowsum(dO . o)) = P' (dP - delta')). d(sink) = -sum(exp(sink - lse_total) * delta').

_LH_NAME = "unsloth.kernels.large_head_attention"
_LH = None


def _large_head_module():
    """The installed Unsloth's large_head_attention, loaded without importing (and patching through) unsloth
    itself; registered under its own name so a later `import unsloth` reuses it. None when absent or its private
    entry points (_fwd_impl / _bwd_impl) differ."""
    global _LH
    if _LH is None:
        module = sys.modules.get(_LH_NAME)
        if module is None:
            try:
                spec = importlib.util.find_spec("unsloth")
                path = os.path.join(os.path.dirname(spec.origin), "kernels", "large_head_attention.py") if spec and spec.origin else None
                if path and os.path.isfile(path):
                    file_spec = importlib.util.spec_from_file_location(_LH_NAME, path)
                    module = importlib.util.module_from_spec(file_spec)
                    sys.modules[_LH_NAME] = module
                    file_spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(_LH_NAME, None)
                module = None
        ok = module is not None and callable(getattr(module, "_fwd_impl", None)) and callable(getattr(module, "_bwd_impl", None))
        _LH = module if ok else False
    return _LH or None


def window_hd512_supported(q, kv):
    """head_dim 512, fp16 / bf16 (bf16 on sm80+), CUDA (not ROCm), >= 96 KB shared memory per block (T4 cannot
    launch it: 196 KB asked), Unsloth's kernel importable."""
    if not (sparse_mqa_supported(q, kv) and q.shape[-1] == 512):
        return False
    if _shared_memory(q.device) < 96 * 1024:
        return False
    return _large_head_module() is not None


def _intervals(b, s, window, device):
    t = torch.arange(s, device = device, dtype = torch.int32)
    lo = (t - int(window) + 1).clamp(min = 0)
    return lo.expand(b, s).contiguous(), t.expand(b, s).contiguous()


@torch.library.custom_op(f"unsloth_zoo::window_hd512_fwd_{_V}", mutates_args = ())
def _hfwd_op(q: torch.Tensor, kv: torch.Tensor, sink: torch.Tensor, scale: float, window: int,
             has_sink: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, s, h, d = q.shape
    lo, hi = _intervals(b, s, window, q.device)
    k = kv.unsqueeze(1)
    o, lse2 = _large_head_module()._fwd_impl(q.transpose(1, 2), k, k, lo, hi, float(scale))
    lse = (lse2 * math.log(2.0)).transpose(1, 2)                                  # [b, s, h], natural log
    if has_sink:
        sink_ = sink.float().view(1, 1, h)
        total = torch.logaddexp(lse, sink_)
        out = (o.transpose(1, 2).float() * torch.exp(lse - total).unsqueeze(-1)).to(q.dtype)
    else:
        total = lse.contiguous()
        out = o.transpose(1, 2).contiguous()
    return out, total, lse2


@_hfwd_op.register_fake
def _(q, kv, sink, scale, window, has_sink):
    b, s, h, d = q.shape
    return (torch.empty_like(q), q.new_empty((b, s, h), dtype = torch.float32),
            q.new_empty((b, h, s), dtype = torch.float32))


@torch.library.custom_op(f"unsloth_zoo::window_hd512_bwd_{_V}", mutates_args = ())
def _hbwd_op(do_eff: torch.Tensor, q: torch.Tensor, kv: torch.Tensor, out: torch.Tensor, lse2: torch.Tensor,
             scale: float, window: int) -> Tuple[torch.Tensor, torch.Tensor]:
    b, s, h, d = q.shape
    lo, hi = _intervals(b, s, window, q.device)
    k = kv.unsqueeze(1)
    dq, dk, dv = _large_head_module()._bwd_impl(do_eff.transpose(1, 2), q.transpose(1, 2), k, k,
                                                out.transpose(1, 2), lse2, lo, hi, float(scale))
    dkv = (dk.float() + dv.float()).squeeze(1).to(kv.dtype)
    return dq.transpose(1, 2).contiguous(), dkv


@_hbwd_op.register_fake
def _(do_eff, q, kv, out, lse2, scale, window):
    return torch.empty_like(q), torch.empty_like(kv)


def _hsetup_context(ctx, inputs, output):
    q, kv, sink, scale, window, has_sink = inputs
    out, total, lse2 = output
    ctx.save_for_backward(q, kv, sink, out, total, lse2)
    ctx.scale, ctx.window, ctx.has_sink = scale, window, has_sink


def _hbackward(ctx, grad_out, grad_total, grad_lse2):
    q, kv, sink, out, total, lse2 = ctx.saved_tensors
    if ctx.has_sink:
        c = torch.exp(lse2.transpose(1, 2) * math.log(2.0) - total)               # sigmoid(lse_window - sink)
        do_eff = (grad_out.float() * c.unsqueeze(-1)).to(grad_out.dtype)
    else:
        do_eff = grad_out
    dq, dkv = _hbwd_op(do_eff.contiguous(), q, kv, out, lse2, ctx.scale, ctx.window)
    dsink = None
    if ctx.has_sink and ctx.needs_input_grad[2]:
        delta = (grad_out.float() * out.float()).sum(-1)
        dsink = -(torch.exp(sink.float().view(1, 1, -1) - total) * delta).sum((0, 1)).to(sink.dtype)
    return dq, dkv, dsink, None, None, None


_hfwd_op.register_autograd(_hbackward, setup_context = _hsetup_context)


def window_hd512_attention(q, window_kv, attn_sink, softmax_scale, window_size):
    """q [b, s, h, 512], window_kv [b, s, 512] (query t sees rows (t - window_size, t]), attn_sink [h] or None."""
    has_sink = attn_sink is not None
    sink = attn_sink if has_sink else q.new_full((q.shape[2],), float("-inf"), dtype = torch.float32)
    out, _, _ = _hfwd_op(q.contiguous(), window_kv.contiguous(), sink, float(softmax_scale), int(window_size), has_sink)
    SPARSE_MQA_STATS["hd512"] += 1
    return out
