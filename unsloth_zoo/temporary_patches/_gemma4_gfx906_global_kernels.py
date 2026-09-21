# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Memory-efficient Gemma-4 31B global attention kernels for AMD gfx906.

These kernels intentionally target one narrow training shape:
  * causal, dense/global attention
  * query heads = 32, KV heads = 4 (GQA ratio 8)
  * head dimension = 512
  * equal Q/K/V sequence lengths (no KV-cache decoding)
  * fp16, bf16 or fp32

The implementation uses tiled online softmax and never materializes the S x S
attention matrix. It exists as a memory-enabling fallback for gfx906, where
PyTorch SDPA can fall back to the math backend for D=512 and scale quadratically
in memory.
"""

import torch
import triton
import triton.language as tl

_HQ = 32
_HKV = 4
_D = 512
_GQA = 8


@triton.jit
def _fwd_kernel(
    Q, K, V, O, LSE,
    sqb, sqh, sqn, sqd,
    skb, skh, skn, skd,
    svb, svh, svn, svd,
    sob, soh, son, sod,
    slb, slh, sln,
    SEQ_LEN,
    scale,
    N_Q_HEADS: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    q_block = tl.program_id(0)
    bh = tl.program_id(1)
    qh = bh % N_Q_HEADS
    b = bh // N_Q_HEADS
    kvh = qh // GQA_RATIO

    q_base = Q + b * sqb + qh * sqh
    k_base = K + b * skb + kvh * skh
    v_base = V + b * svb + kvh * svh
    o_base = O + b * sob + qh * soh

    q_idx = q_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_idx < SEQ_LEN
    d = tl.arange(0, HEAD_DIM)

    # D=512 is register constrained on gfx906. The tuned path intentionally
    # reloads Q per KV tile rather than retaining both a 16x512 Q tile and the
    # online-softmax accumulator for the whole loop.
    m = tl.full([BLOCK_Q], -float("inf"), tl.float32)
    l = tl.zeros([BLOCK_Q], tl.float32)
    acc = tl.zeros([BLOCK_Q, HEAD_DIM], tl.float32)
    LOG2E: tl.constexpr = 1.4426950408889634
    scale_log2 = scale * LOG2E

    kv_end = (q_block + 1) * BLOCK_Q
    for kv_start in range(0, kv_end, BLOCK_KV):
        kv_idx = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_idx < SEQ_LEN

        q_ptr = q_base + q_idx[:, None] * sqn + d[None, :] * sqd
        k_ptr = k_base + kv_idx[:, None] * skn + d[None, :] * skd
        q = tl.load(q_ptr, mask=q_mask[:, None], other=0.0)
        k = tl.load(k_ptr, mask=kv_mask[:, None], other=0.0)
        scores = tl.dot(q, tl.trans(k)) * scale_log2

        valid = (kv_idx[None, :] <= q_idx[:, None]) & kv_mask[None, :]
        scores = tl.where(valid, scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(m, block_max)
        alpha = tl.math.exp2(m - new_max)
        p = tl.math.exp2(scores - new_max[:, None])
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptr = v_base + kv_idx[:, None] * svn + d[None, :] * svd
        vv = tl.load(v_ptr, mask=kv_mask[:, None], other=0.0)
        acc += tl.dot(p.to(vv.dtype), vv)
        m = new_max

    out = acc / l[:, None]
    o_ptr = o_base + q_idx[:, None] * son + d[None, :] * sod
    tl.store(o_ptr, out, mask=q_mask[:, None])

    LN2: tl.constexpr = 0.6931471805599453
    lse = m * LN2 + tl.log(l)
    lse_ptr = LSE + b * slb + qh * slh + q_idx * sln
    tl.store(lse_ptr, lse, mask=q_mask)


@triton.jit
def _bwd_dq_kernel(
    Q, K, V, dO, O, dQ, LSE, Delta,
    sqb, sqh, sqn, sqd,
    skb, skh, skn, skd,
    svb, svh, svn, svd,
    sdob, sdoh, sdon, sdod,
    sob, soh, son, sod,
    sdqb, sdqh, sdqn, sdqd,
    slb, slh, sln,
    sdb, sdh, sdn,
    SEQ_LEN,
    scale,
    N_Q_HEADS: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    q_block = tl.program_id(0)
    bh = tl.program_id(1)
    qh = bh % N_Q_HEADS
    b = bh // N_Q_HEADS
    kvh = qh // GQA_RATIO

    q_base = Q + b * sqb + qh * sqh
    k_base = K + b * skb + kvh * skh
    v_base = V + b * svb + kvh * svh
    do_base = dO + b * sdob + qh * sdoh
    o_base = O + b * sob + qh * soh
    dq_base = dQ + b * sdqb + qh * sdqh

    q_idx = q_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_idx < SEQ_LEN
    d = tl.arange(0, HEAD_DIM)

    q_ptr = q_base + q_idx[:, None] * sqn + d[None, :] * sqd
    do_ptr = do_base + q_idx[:, None] * sdon + d[None, :] * sdod
    o_ptr = o_base + q_idx[:, None] * son + d[None, :] * sod
    q = tl.load(q_ptr, mask=q_mask[:, None], other=0.0)
    do = tl.load(do_ptr, mask=q_mask[:, None], other=0.0)
    oo = tl.load(o_ptr, mask=q_mask[:, None], other=0.0)

    lse_ptr = LSE + b * slb + qh * slh + q_idx * sln
    lse = tl.load(lse_ptr, mask=q_mask, other=0.0)
    delta = tl.sum(do.to(tl.float32) * oo.to(tl.float32), axis=1)
    delta_ptr = Delta + b * sdb + qh * sdh + q_idx * sdn
    tl.store(delta_ptr, delta, mask=q_mask)

    LOG2E: tl.constexpr = 1.4426950408889634
    lse_log2 = lse * LOG2E
    scale_log2 = scale * LOG2E
    dq = tl.zeros([BLOCK_Q, HEAD_DIM], tl.float32)
    kv_end = (q_block + 1) * BLOCK_Q

    for kv_start in range(0, kv_end, BLOCK_KV):
        kv_idx = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_idx < SEQ_LEN
        k_ptr = k_base + kv_idx[:, None] * skn + d[None, :] * skd
        v_ptr = v_base + kv_idx[:, None] * svn + d[None, :] * svd
        k = tl.load(k_ptr, mask=kv_mask[:, None], other=0.0)
        vv = tl.load(v_ptr, mask=kv_mask[:, None], other=0.0)

        scores = tl.dot(q, tl.trans(k)).to(tl.float32) * scale_log2
        valid = (kv_idx[None, :] <= q_idx[:, None]) & kv_mask[None, :]
        scores = tl.where(valid, scores, -float("inf"))
        p = tl.math.exp2(scores - lse_log2[:, None])

        dp = tl.dot(do, tl.trans(vv)).to(tl.float32)
        ds = p * (dp - delta[:, None])
        ds = tl.where(valid, ds, 0.0)
        dq += tl.dot(ds.to(k.dtype), k).to(tl.float32)

    dq *= scale
    dq_ptr = dq_base + q_idx[:, None] * sdqn + d[None, :] * sdqd
    tl.store(dq_ptr, dq.to(q.dtype), mask=q_mask[:, None])


@triton.jit
def _bwd_dkv_kernel(
    Q, K, V, dO, dK, dV, LSE, Delta,
    sqb, sqh, sqn, sqd,
    skb, skh, skn, skd,
    svb, svh, svn, svd,
    sdob, sdoh, sdon, sdod,
    sdkb, sdkh, sdkn, sdkd,
    sdvb, sdvh, sdvn, sdvd,
    slb, slh, sln,
    sdb, sdh, sdn,
    SEQ_LEN,
    scale,
    N_KV_HEADS: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    kv_block = tl.program_id(0)
    bkv = tl.program_id(1)
    kvh = bkv % N_KV_HEADS
    b = bkv // N_KV_HEADS

    k_base = K + b * skb + kvh * skh
    v_base = V + b * svb + kvh * svh
    dk_base = dK + b * sdkb + kvh * sdkh
    dv_base = dV + b * sdvb + kvh * sdvh

    kv_idx = kv_block * BLOCK_KV + tl.arange(0, BLOCK_KV)
    kv_mask = kv_idx < SEQ_LEN
    d = tl.arange(0, HEAD_DIM)
    k_ptr = k_base + kv_idx[:, None] * skn + d[None, :] * skd
    v_ptr = v_base + kv_idx[:, None] * svn + d[None, :] * svd
    k = tl.load(k_ptr, mask=kv_mask[:, None], other=0.0)
    vv = tl.load(v_ptr, mask=kv_mask[:, None], other=0.0)

    dk = tl.zeros([BLOCK_KV, HEAD_DIM], tl.float32)
    dv = tl.zeros([BLOCK_KV, HEAD_DIM], tl.float32)
    LOG2E: tl.constexpr = 1.4426950408889634
    scale_log2 = scale * LOG2E

    # Causal reverse traversal: key block j receives gradients only from query
    # positions at or after its first key position.
    q_start = (kv_block * BLOCK_KV // BLOCK_Q) * BLOCK_Q
    for qh_off in tl.static_range(GQA_RATIO):
        qh = kvh * GQA_RATIO + qh_off
        q_base = Q + b * sqb + qh * sqh
        do_base = dO + b * sdob + qh * sdoh
        lse_base = LSE + b * slb + qh * slh
        delta_base = Delta + b * sdb + qh * sdh

        for qs in range(q_start, SEQ_LEN, BLOCK_Q):
            q_idx = qs + tl.arange(0, BLOCK_Q)
            q_mask = q_idx < SEQ_LEN
            q_ptr = q_base + q_idx[:, None] * sqn + d[None, :] * sqd
            do_ptr = do_base + q_idx[:, None] * sdon + d[None, :] * sdod
            q = tl.load(q_ptr, mask=q_mask[:, None], other=0.0)
            do = tl.load(do_ptr, mask=q_mask[:, None], other=0.0)
            lse = tl.load(lse_base + q_idx * sln, mask=q_mask, other=0.0)
            delta = tl.load(delta_base + q_idx * sdn, mask=q_mask, other=0.0)
            lse_log2 = lse * LOG2E

            scores = tl.dot(q, tl.trans(k)).to(tl.float32) * scale_log2
            valid = (kv_idx[None, :] <= q_idx[:, None]) & kv_mask[None, :] & q_mask[:, None]
            scores = tl.where(valid, scores, -float("inf"))
            p = tl.math.exp2(scores - lse_log2[:, None])

            dv += tl.dot(tl.trans(p.to(do.dtype)), do).to(tl.float32)
            dp = tl.dot(do, tl.trans(vv)).to(tl.float32)
            ds = p * (dp - delta[:, None])
            ds = tl.where(valid, ds, 0.0)
            dk += tl.dot(tl.trans(ds.to(q.dtype)), q).to(tl.float32)

    dk *= scale
    dk_ptr = dk_base + kv_idx[:, None] * sdkn + d[None, :] * sdkd
    dv_ptr = dv_base + kv_idx[:, None] * sdvn + d[None, :] * sdvd
    tl.store(dk_ptr, dk.to(k.dtype), mask=kv_mask[:, None])
    tl.store(dv_ptr, dv.to(vv.dtype), mask=kv_mask[:, None])


class _Gemma4Gfx906GlobalAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale):
        B, HQ, N, D = q.shape
        if (
            HQ != _HQ
            or k.shape[1] != _HKV
            or v.shape[1] != _HKV
            or D != _D
            or k.shape[3] != _D
            or v.shape[3] != _D
        ):
            raise ValueError("gfx906 Gemma-4 global kernel only supports Hq=32/Hkv=4/D=512")
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            raise ValueError("gfx906 Gemma-4 global kernel requires matching Q/K/V batch size")
        if q.shape[2] != k.shape[2] or q.shape[2] != v.shape[2]:
            raise ValueError("gfx906 Gemma-4 global kernel requires equal Q/K/V sequence lengths")
        if q.dtype not in (torch.float16, torch.bfloat16, torch.float32) or k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError("gfx906 Gemma-4 global kernel requires matching fp16/bf16/fp32 Q/K/V")
        if k.device != q.device or v.device != q.device:
            raise ValueError("gfx906 Gemma-4 global kernel requires Q/K/V on one device")

        # Triton launches on the current accelerator/stream. The caller may have
        # tensors on cuda:1 while cuda:0 is current, so select q.device around
        # allocation and launch instead of relying on ambient device state.
        with torch.cuda.device(q.device):
            out = torch.empty_like(q)
            lse = torch.empty((B, _HQ, N), dtype=torch.float32, device=q.device)
            grid = (triton.cdiv(N, 16), B * _HQ)
            _fwd_kernel[grid](
                q, k, v, out, lse,
                *q.stride(), *k.stride(), *v.stride(), *out.stride(), *lse.stride(),
                SEQ_LEN=N, scale=float(scale), N_Q_HEADS=_HQ, GQA_RATIO=_GQA, HEAD_DIM=_D,
                BLOCK_Q=16, BLOCK_KV=32,
                num_warps=4, num_stages=1,
            )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.scale = float(scale)
        return out

    @staticmethod
    def backward(ctx, do):
        # This hand-written Triton backward returns detached first derivatives;
        # allowing create_graph=True would silently construct a partial Hessian
        # when the loss also contains differentiable non-attention terms.
        if torch.is_grad_enabled():
            raise RuntimeError(
                "gfx906 Gemma4 global attention supports first-order gradients only"
            )
        q, k, v, out, lse = ctx.saved_tensors
        with torch.cuda.device(q.device):
            do = do.contiguous()
            B, _, N, _ = q.shape
            delta = torch.empty((B, _HQ, N), dtype=torch.float32, device=q.device)
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)

            grid_dq = (triton.cdiv(N, 16), B * _HQ)
            _bwd_dq_kernel[grid_dq](
                q, k, v, do, out, dq, lse, delta,
                *q.stride(), *k.stride(), *v.stride(), *do.stride(), *out.stride(),
                *dq.stride(), *lse.stride(), *delta.stride(),
                SEQ_LEN=N, scale=ctx.scale, N_Q_HEADS=_HQ, GQA_RATIO=_GQA, HEAD_DIM=_D,
                BLOCK_Q=16, BLOCK_KV=16,
                num_warps=4, num_stages=2,
            )

            # D=512 backward is shared-memory constrained on MI50 (65,536 B limit).
            # fp16/bf16 fit with BKV=8; fp32 needs BKV=4 and a single pipeline stage.
            if q.dtype == torch.float32:
                dkv_block = 4
                dkv_stages = 1
            else:
                dkv_block = 8
                dkv_stages = 2
            grid_dkv = (triton.cdiv(N, dkv_block), B * _HKV)
            _bwd_dkv_kernel[grid_dkv](
                q, k, v, do, dk, dv, lse, delta,
                *q.stride(), *k.stride(), *v.stride(), *do.stride(), *dk.stride(),
                *dv.stride(), *lse.stride(), *delta.stride(),
                SEQ_LEN=N, scale=ctx.scale, N_KV_HEADS=_HKV, GQA_RATIO=_GQA, HEAD_DIM=_D,
                BLOCK_Q=16, BLOCK_KV=dkv_block,
                num_warps=4, num_stages=dkv_stages,
            )
        return dq, dk, dv, None


def gemma4_gfx906_global_attention(q, k, v, scale):
    """Return causal full-attention output with shape (B, Hq, S, 512)."""
    return _Gemma4Gfx906GlobalAttention.apply(q, k, v, scale)
