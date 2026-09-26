# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# ============================================================================
# Tiled causal attention for head_dim 512, forward and backward, in Triton.
#
# Dense causal self-attention over (B, Hq, S, 512) queries and (B, Hkv, S, 512)
# keys/values with any GQA ratio Hq % Hkv == 0. Online softmax in the forward
# and FlashAttention-2 style recomputation in the backward, so memory is O(S)
# rather than the O(S^2) of an SDPA backend that materialises the score matrix.
# fp16 / bf16 / fp32 inputs, fp32 accumulation, first-order gradients only.
#
# The backward is split in two launches so every output has exactly one owner:
# one program per (query block, q head) writes dQ, one program per (key block,
# kv head) writes dK/dV after summing over the GQA group. No atomics.
#
# Tile sizes are chosen to fit a 64 KiB LDS budget at D=512; bf16/fp32 sit at
# that limit on gfx906. Routing (which devices and models use this) lives in
# the callers, not here.
# ============================================================================

import torch
import triton
import triton.language as tl

__all__ = ["causal_attention_d512"]

HEAD_DIM = 512


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

    m = tl.full([BLOCK_Q], -float("inf"), tl.float32)
    l = tl.zeros([BLOCK_Q], tl.float32)
    acc = tl.zeros([BLOCK_Q, HEAD_DIM], tl.float32)
    LOG2E: tl.constexpr = 1.4426950408889634
    scale_log2 = scale * LOG2E

    kv_end = (q_block + 1) * BLOCK_Q
    for kv_start in range(0, kv_end, BLOCK_KV):
        kv_idx = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_idx < SEQ_LEN

        # Q is reloaded per KV tile: holding a 16x512 Q tile next to the
        # 16x512 accumulator for the whole loop does not fit in registers.
        q = tl.load(q_base + q_idx[:, None] * sqn + d[None, :] * sqd, mask = q_mask[:, None], other = 0.0)
        k = tl.load(k_base + kv_idx[:, None] * skn + d[None, :] * skd, mask = kv_mask[:, None], other = 0.0)
        scores = tl.dot(q, tl.trans(k)) * scale_log2

        valid = (kv_idx[None, :] <= q_idx[:, None]) & kv_mask[None, :]
        scores = tl.where(valid, scores, -float("inf"))

        new_max = tl.maximum(m, tl.max(scores, axis = 1))
        alpha = tl.math.exp2(m - new_max)
        p = tl.math.exp2(scores - new_max[:, None])
        l = l * alpha + tl.sum(p, axis = 1)
        acc = acc * alpha[:, None]

        vv = tl.load(v_base + kv_idx[:, None] * svn + d[None, :] * svd, mask = kv_mask[:, None], other = 0.0)
        acc += tl.dot(p.to(vv.dtype), vv)
        m = new_max

    out = acc / l[:, None]
    tl.store(o_base + q_idx[:, None] * son + d[None, :] * sod, out, mask = q_mask[:, None])

    # Natural-log LSE; the backward converts back to base 2.
    LN2: tl.constexpr = 0.6931471805599453
    lse = m * LN2 + tl.log(l)
    tl.store(LSE + b * slb + qh * slh + q_idx * sln, lse, mask = q_mask)


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

    k_base = K + b * skb + kvh * skh
    v_base = V + b * svb + kvh * svh

    q_idx = q_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_idx < SEQ_LEN
    d = tl.arange(0, HEAD_DIM)

    q = tl.load(Q + b * sqb + qh * sqh + q_idx[:, None] * sqn + d[None, :] * sqd, mask = q_mask[:, None], other = 0.0)
    do = tl.load(dO + b * sdob + qh * sdoh + q_idx[:, None] * sdon + d[None, :] * sdod, mask = q_mask[:, None], other = 0.0)
    oo = tl.load(O + b * sob + qh * soh + q_idx[:, None] * son + d[None, :] * sod, mask = q_mask[:, None], other = 0.0)
    lse = tl.load(LSE + b * slb + qh * slh + q_idx * sln, mask = q_mask, other = 0.0)

    # Delta is written here and consumed by _bwd_dkv_kernel on the same stream.
    delta = tl.sum(do.to(tl.float32) * oo.to(tl.float32), axis = 1)
    tl.store(Delta + b * sdb + qh * sdh + q_idx * sdn, delta, mask = q_mask)

    LOG2E: tl.constexpr = 1.4426950408889634
    lse_log2 = lse * LOG2E
    scale_log2 = scale * LOG2E
    dq = tl.zeros([BLOCK_Q, HEAD_DIM], tl.float32)

    kv_end = (q_block + 1) * BLOCK_Q
    for kv_start in range(0, kv_end, BLOCK_KV):
        kv_idx = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_idx < SEQ_LEN
        k = tl.load(k_base + kv_idx[:, None] * skn + d[None, :] * skd, mask = kv_mask[:, None], other = 0.0)
        vv = tl.load(v_base + kv_idx[:, None] * svn + d[None, :] * svd, mask = kv_mask[:, None], other = 0.0)

        scores = tl.dot(q, tl.trans(k)).to(tl.float32) * scale_log2
        valid = (kv_idx[None, :] <= q_idx[:, None]) & kv_mask[None, :]
        scores = tl.where(valid, scores, -float("inf"))
        p = tl.math.exp2(scores - lse_log2[:, None])

        dp = tl.dot(do, tl.trans(vv)).to(tl.float32)
        ds = tl.where(valid, p * (dp - delta[:, None]), 0.0)
        dq += tl.dot(ds.to(k.dtype), k).to(tl.float32)

    dq *= scale
    tl.store(dQ + b * sdqb + qh * sdqh + q_idx[:, None] * sdqn + d[None, :] * sdqd, dq.to(q.dtype), mask = q_mask[:, None])


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

    kv_idx = kv_block * BLOCK_KV + tl.arange(0, BLOCK_KV)
    kv_mask = kv_idx < SEQ_LEN
    d = tl.arange(0, HEAD_DIM)
    k = tl.load(K + b * skb + kvh * skh + kv_idx[:, None] * skn + d[None, :] * skd, mask = kv_mask[:, None], other = 0.0)
    vv = tl.load(V + b * svb + kvh * svh + kv_idx[:, None] * svn + d[None, :] * svd, mask = kv_mask[:, None], other = 0.0)

    dk = tl.zeros([BLOCK_KV, HEAD_DIM], tl.float32)
    dv = tl.zeros([BLOCK_KV, HEAD_DIM], tl.float32)
    LOG2E: tl.constexpr = 1.4426950408889634
    scale_log2 = scale * LOG2E

    # Causal: this key block only receives gradient from queries at or after it.
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
            q = tl.load(q_base + q_idx[:, None] * sqn + d[None, :] * sqd, mask = q_mask[:, None], other = 0.0)
            do = tl.load(do_base + q_idx[:, None] * sdon + d[None, :] * sdod, mask = q_mask[:, None], other = 0.0)
            lse = tl.load(lse_base + q_idx * sln, mask = q_mask, other = 0.0)
            delta = tl.load(delta_base + q_idx * sdn, mask = q_mask, other = 0.0)

            scores = tl.dot(q, tl.trans(k)).to(tl.float32) * scale_log2
            valid = (kv_idx[None, :] <= q_idx[:, None]) & kv_mask[None, :] & q_mask[:, None]
            scores = tl.where(valid, scores, -float("inf"))
            p = tl.math.exp2(scores - (lse * LOG2E)[:, None])

            dv += tl.dot(tl.trans(p.to(do.dtype)), do).to(tl.float32)
            dp = tl.dot(do, tl.trans(vv)).to(tl.float32)
            ds = tl.where(valid, p * (dp - delta[:, None]), 0.0)
            dk += tl.dot(tl.trans(ds.to(q.dtype)), q).to(tl.float32)

    dk *= scale
    tl.store(dK + b * sdkb + kvh * sdkh + kv_idx[:, None] * sdkn + d[None, :] * sdkd, dk.to(k.dtype), mask = kv_mask[:, None])
    tl.store(dV + b * sdvb + kvh * sdvh + kv_idx[:, None] * sdvn + d[None, :] * sdvd, dv.to(vv.dtype), mask = kv_mask[:, None])


def _check_inputs(q, k, v):
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("causal_attention_d512 expects (B, H, S, D) q/k/v")
    B, Hq, S, D = q.shape
    if D != HEAD_DIM or k.shape[3] != HEAD_DIM or v.shape[3] != HEAD_DIM:
        raise ValueError(f"causal_attention_d512 requires head_dim {HEAD_DIM}")
    if k.shape[:3] != v.shape[:3]:
        raise ValueError("causal_attention_d512 requires k and v with the same (B, Hkv, S)")
    if k.shape[0] != B or k.shape[2] != S:
        raise ValueError("causal_attention_d512 requires matching batch and sequence length across q/k/v")
    Hkv = k.shape[1]
    if Hkv == 0 or Hq % Hkv != 0:
        raise ValueError(f"causal_attention_d512 requires Hq % Hkv == 0, got Hq={Hq} Hkv={Hkv}")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("causal_attention_d512 requires fp16/bf16/fp32 q/k/v of one dtype")
    if k.device != q.device or v.device != q.device:
        raise ValueError("causal_attention_d512 requires q/k/v on one device")
    return B, Hq, Hkv, S


def _dkv_config(dtype):
    # fp32 at BLOCK_KV=8 needs ~96 KiB LDS; BLOCK_KV=4 with one stage fits 64 KiB.
    if dtype == torch.float32:
        return 4, 1
    return 8, 2


class _CausalAttentionD512(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale):
        B, Hq, Hkv, S = _check_inputs(q, k, v)
        # Launch on q's device, not whatever device happens to be current.
        with torch.cuda.device(q.device):
            out = torch.empty_like(q)
            lse = torch.empty((B, Hq, S), dtype = torch.float32, device = q.device)
            _fwd_kernel[(triton.cdiv(S, 16), B * Hq)](
                q, k, v, out, lse,
                *q.stride(), *k.stride(), *v.stride(), *out.stride(), *lse.stride(),
                SEQ_LEN = S, scale = float(scale),
                N_Q_HEADS = Hq, GQA_RATIO = Hq // Hkv, HEAD_DIM = HEAD_DIM,
                BLOCK_Q = 16, BLOCK_KV = 32,
                num_warps = 4, num_stages = 1,
            )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.scale = float(scale)
        return out

    @staticmethod
    def backward(ctx, do):
        # The kernels return detached first derivatives. With create_graph=True a
        # composed loss would get a silently partial second derivative, so refuse.
        if torch.is_grad_enabled():
            raise RuntimeError("causal_attention_d512 supports first-order gradients only")
        q, k, v, out, lse = ctx.saved_tensors
        B, Hq, S, _ = q.shape
        Hkv = k.shape[1]
        with torch.cuda.device(q.device):
            do = do.contiguous()
            delta = torch.empty((B, Hq, S), dtype = torch.float32, device = q.device)
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)

            _bwd_dq_kernel[(triton.cdiv(S, 16), B * Hq)](
                q, k, v, do, out, dq, lse, delta,
                *q.stride(), *k.stride(), *v.stride(), *do.stride(), *out.stride(),
                *dq.stride(), *lse.stride(), *delta.stride(),
                SEQ_LEN = S, scale = ctx.scale,
                N_Q_HEADS = Hq, GQA_RATIO = Hq // Hkv, HEAD_DIM = HEAD_DIM,
                BLOCK_Q = 16, BLOCK_KV = 16,
                num_warps = 4, num_stages = 2,
            )

            block_kv, stages = _dkv_config(q.dtype)
            _bwd_dkv_kernel[(triton.cdiv(S, block_kv), B * Hkv)](
                q, k, v, do, dk, dv, lse, delta,
                *q.stride(), *k.stride(), *v.stride(), *do.stride(), *dk.stride(),
                *dv.stride(), *lse.stride(), *delta.stride(),
                SEQ_LEN = S, scale = ctx.scale,
                N_KV_HEADS = Hkv, GQA_RATIO = Hq // Hkv, HEAD_DIM = HEAD_DIM,
                BLOCK_Q = 16, BLOCK_KV = block_kv,
                num_warps = 4, num_stages = stages,
            )
        return dq, dk, dv, None


def causal_attention_d512(q, k, v, scale):
    """Causal attention for (B, Hq, S, 512) q and (B, Hkv, S, 512) k/v.

    Returns (B, Hq, S, 512) in q's dtype. ``scale`` multiplies q @ k^T.
    """
    return _CausalAttentionD512.apply(q, k, v, scale)
