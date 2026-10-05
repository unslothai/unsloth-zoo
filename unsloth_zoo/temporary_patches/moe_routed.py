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

"""Routed MoE expert inference: only the experts the router picked are read.

Decode-sized no-grad calls skip dequantize-all + grouped_mm; no host sync, fixed grid.

NF4 (bitsandbytes), two launches per layer, dequantized inside the GEMV:
  gate_up: one GEMV per (token, slot) -> [P, I] fp32 activation (bias, LoRA B, gated act fused).
  down:    per token, GEMV per slot, bias, LoRA B, routing weight and the top-k sum in a fixed
           order, written once in the output dtype (fp32 accumulation, no atomics).
Constexpr axes:
  STACKED: one stacked Params4bit [E, N, K], or a pointer table of per-expert Linear4bit.
  ACT:     gpt-oss clamp-swiglu, silu, gelu (tanh) or gelu (erf), gate * up.
  INTERLEAVED: gate / up rows alternate, else [gate; up] halves.
  LORA_KIND: LoRA B as one [E, N, r] tensor, or a pointer table to each live lora_B.
BF16 / FP16 3D experts use the expert-major kernel.
"""

__all__ = [
    "ROUTED_MAX_SLOTS",
    "NF4_MAX_SLOTS",
    "NF4_SLOTS_PER_EXPERT",
    "nf4_slot_limit",
    "BF16_MAX_SLOTS",
    "ACT_GPTOSS",
    "ACT_SILU",
    "ACT_GELU_TANH",
    "ACT_GELU",
    "routed_gate_up",
    "routed_down",
    "routed_bf16_gemm",
    "routed_bf16_moe",
    "routed_lora_h",
    "lora_pointer_table",
    "routed_moe_forward",
    "prepare_stacked_nf4",
    "nf4_select_dequant",
    "routed_mode",
]

import functools
import os
from typing import List, Optional

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - no Triton, no routed path
    triton = None


ACT_GPTOSS, ACT_SILU, ACT_GELU_TANH, ACT_GELU = 0, 1, 2, 3


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# Above this many (token, expert) slots the routed kernels lose to dequant + grouped_mm.
# gpt-oss (gpt_oss_routed.py, top-4):
ROUTED_MAX_SLOTS = _env_int("UNSLOTH_MOE_ROUTED_MAX_SLOTS", 64)
# Generic 3D experts: routed NF4 cost grows with slots P, dequantize-all with E. On B200,
# routed / current time is ~0.67 at P = E and ~1.2 at 2E across E 32-256, so route while P <= E.
# BF16 routed wins only up to 32 slots. UNSLOTH_MOE_ROUTED_MAX_SLOTS sets an absolute limit.
NF4_MAX_SLOTS = _env_int("UNSLOTH_MOE_ROUTED_MAX_SLOTS", -1)
NF4_MAX_SLOTS = None if NF4_MAX_SLOTS < 0 else NF4_MAX_SLOTS
NF4_SLOTS_PER_EXPERT = 1.0
BF16_MAX_SLOTS = _env_int("UNSLOTH_MOE_ROUTED_MAX_SLOTS", 32)


def nf4_slot_limit(num_experts):
    """Most (token, expert) slots a stacked NF4 call routes for a layer with num_experts experts."""
    if NF4_MAX_SLOTS is not None:
        return NF4_MAX_SLOTS
    return int(NF4_SLOTS_PER_EXPERT * num_experts)


def routed_mode():
    """UNSLOTH_MOE_ROUTED_KERNEL: "1" (default, fused routed kernels), "0" (off, the
    dequantize-all + grouped_mm path) or "grouped" (selective dequant of the routed experts
    into a scratch + torch._grouped_mm; comparator arm)."""
    mode = os.environ.get("UNSLOTH_MOE_ROUTED_KERNEL", "1").strip().lower()
    if mode in ("0", "false", "off", "no"):
        return "0"
    if mode == "grouped":
        return "grouped"
    return "1"


if triton is not None:

    @triton.jit
    def _nf4_dot(W, A, A2, C2, offset, LUT, rows64, rmask, x_chunk, k0, K,
                 BLOCKSIZE: tl.constexpr, BLOCKSIZE2: tl.constexpr, NESTED: tl.constexpr,
                 BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        # sum_k dequant(W[rows, k0:k0 + BLOCK_K]) * x_chunk, as bitsandbytes dequantizes.
        # rows64 index the (flat) buffer's rows: absmax / state2 blocks are global indices.
        HALF_K: tl.constexpr = BLOCK_K // 2
        NB: tl.constexpr = BLOCK_K // BLOCKSIZE
        half_k = tl.arange(0, HALF_K)
        nbs = tl.arange(0, NB)
        kmask = (k0 + 2 * half_k) < K
        b = tl.load(W + rows64[:, None] * (K // 2) + (k0 // 2 + half_k)[None, :],
                    mask = rmask[:, None] & kmask[None, :], other = 0).to(tl.int32)
        # High nibble is the earlier element, as bitsandbytes packs it.
        w = tl.reshape(tl.join(tl.load(LUT + (b >> 4)), tl.load(LUT + (b & 15))), (BLOCK_N, BLOCK_K))
        part = tl.sum(tl.reshape(w * x_chunk[None, :], (BLOCK_N, NB, BLOCKSIZE)), axis = 2)
        bmask = rmask[:, None] & ((k0 + nbs * BLOCKSIZE) < K)[None, :]
        blk = rows64[:, None] * (K // BLOCKSIZE) + (k0 // BLOCKSIZE + nbs)[None, :]
        if NESTED:
            q = tl.load(A + blk, mask = bmask, other = 0).to(tl.int32)
            a = tl.load(C2 + q, mask = bmask, other = 0.0) * tl.load(A2 + blk // BLOCKSIZE2, mask = bmask, other = 0.0) + offset
        else:
            a = tl.load(A + blk, mask = bmask, other = 0.0)
        return tl.sum(part * a, axis = 1)

    @triton.jit
    def _expert_tables(e, W_PTRS, A_PTRS, A2_PTRS, C2_PTRS, OFFSETS, NESTED: tl.constexpr, STACKED: tl.constexpr):
        if STACKED:
            # One stacked buffer per tensor: the expert offset goes into the row index.
            W = W_PTRS
            A = A_PTRS
            A2 = A2_PTRS
            C2 = C2_PTRS
            if NESTED:
                offset = tl.load(OFFSETS)
            else:
                offset = 0.0
        else:
            W = tl.load(W_PTRS + e).to(tl.pointer_type(tl.uint8))
            if NESTED:
                A = tl.load(A_PTRS + e).to(tl.pointer_type(tl.uint8))
                A2 = tl.load(A2_PTRS + e).to(tl.pointer_type(tl.float32))
                C2 = tl.load(C2_PTRS + e).to(tl.pointer_type(tl.float32))
                offset = tl.load(OFFSETS + e)
            else:
                A = tl.load(A_PTRS + e).to(tl.pointer_type(tl.float32))
                A2 = A
                C2 = A
                offset = 0.0
        return W, A, A2, C2, offset

    @triton.jit
    def _lora_ptr(PTRS, e, LORA_KIND: tl.constexpr):
        # Expert e's own adapter weight from an int64 table of pointers (1 / 2 / 3: fp32 / bf16 / fp16).
        if LORA_KIND == 2:
            return tl.load(PTRS + e).to(tl.pointer_type(tl.bfloat16))
        elif LORA_KIND == 3:
            return tl.load(PTRS + e).to(tl.pointer_type(tl.float16))
        else:
            return tl.load(PTRS + e).to(tl.pointer_type(tl.float32))

    @triton.jit
    def _lora_b(acc, LORA_B, LORA_H, e, s, rows64, rmask, scaling, stride_be, stride_bn, stride_br,
                R: tl.constexpr, R_PAD: tl.constexpr, LORA_KIND: tl.constexpr):
        # acc += scaling * B[e][rows] @ H[s], H = A[e] @ x precomputed per slot.
        j = tl.arange(0, R_PAD)
        jmask = j < R
        if LORA_KIND == 0:
            Be = LORA_B + e * stride_be
        else:
            Be = _lora_ptr(LORA_B, e, LORA_KIND)
        b = tl.load(Be + rows64[:, None] * stride_bn + j[None, :] * stride_br,
                    mask = rmask[:, None] & jmask[None, :], other = 0.0)
        h = tl.load(LORA_H + s * R + j, mask = jmask, other = 0.0)
        return acc + tl.sum(b.to(tl.float32) * h.to(tl.float32)[None, :], axis = 1) * scaling

    @triton.jit
    def _expert_bias(BIAS, e, N, rows64, rmask, BIAS_KIND: tl.constexpr):
        # Live bias, never a copy. 1 / 2 / 3: pointer table (fp32 / bf16 / fp16), 4: stacked [E, N].
        if BIAS_KIND == 4:
            B = BIAS + e * N
        elif BIAS_KIND == 2:
            B = tl.load(BIAS + e).to(tl.pointer_type(tl.bfloat16))
        elif BIAS_KIND == 3:
            B = tl.load(BIAS + e).to(tl.pointer_type(tl.float16))
        else:
            B = tl.load(BIAS + e).to(tl.pointer_type(tl.float32))
        return tl.load(B + rows64, mask = rmask, other = 0.0).to(tl.float32)

    @triton.jit
    def _gated_act(gate, up, alpha, limit, ACT: tl.constexpr):
        if ACT == 0:
            # gpt-oss: clamp, then (up + 1) * gate * sigmoid(alpha * gate).
            gate = tl.minimum(gate, limit)
            up = tl.minimum(tl.maximum(up, -limit), limit)
            inter = (up + 1.0) * (gate * tl.sigmoid(gate * alpha))
        elif ACT == 1:
            inter = gate * tl.sigmoid(gate) * up
        elif ACT == 2:
            # 0.5 * g * (1 + tanh(u)) == g * sigmoid(2 u), u = sqrt(2 / pi) * (g + 0.044715 g^3).
            u = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
            inter = gate * tl.sigmoid(2.0 * u) * up
        else:
            inter = 0.5 * gate * (1.0 + tl.erf(gate * 0.7071067811865476)) * up
        return inter

    @triton.jit
    def _routed_gate_up_kernel(
        X, IDX, W_PTRS, A_PTRS, A2_PTRS, C2_PTRS, OFFSETS, LUT, BIAS, LORA_B, LORA_H, OUT, N, K,
        alpha, limit, scaling, stride_be, stride_bn, stride_br,
        TOP_K: tl.constexpr, BLOCKSIZE: tl.constexpr, BLOCKSIZE2: tl.constexpr, NESTED: tl.constexpr,
        STACKED: tl.constexpr, ACT: tl.constexpr, INTERLEAVED: tl.constexpr,
        BIAS_KIND: tl.constexpr, HAS_LORA: tl.constexpr, R: tl.constexpr, R_PAD: tl.constexpr,
        LORA_KIND: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        # Each program owns BLOCK_N // 2 intermediate columns; its rows alternate gate, up.
        alpha = alpha.to(tl.float32)
        limit = limit.to(tl.float32)
        p = tl.program_id(0)
        e = tl.load(IDX + p).to(tl.int64)
        W, A, A2, C2, offset = _expert_tables(e, W_PTRS, A_PTRS, A2_PTRS, C2_PTRS, OFFSETS, NESTED, STACKED)
        r = tl.arange(0, BLOCK_N)
        j = tl.program_id(1) * (BLOCK_N // 2) + r // 2
        if INTERLEAVED:
            rows = 2 * j + r % 2
        else:
            rows = j + (r % 2) * (N // 2)
        rmask = j < N // 2
        rows64 = rows.to(tl.int64)
        if STACKED:
            wrows = e * N + rows64
        else:
            wrows = rows64
        ks = tl.arange(0, BLOCK_K)
        x_row = X + (p // TOP_K).to(tl.int64) * K
        acc = tl.zeros([BLOCK_N], dtype = tl.float32)
        for k0 in range(0, K, BLOCK_K):
            x = tl.load(x_row + k0 + ks, mask = (k0 + ks) < K, other = 0.0).to(tl.float32)
            acc += _nf4_dot(W, A, A2, C2, offset, LUT, wrows, rmask, x, k0, K,
                            BLOCKSIZE, BLOCKSIZE2, NESTED, BLOCK_N, BLOCK_K)
        if BIAS_KIND != 0:
            acc += _expert_bias(BIAS, e, N, rows64, rmask, BIAS_KIND)
        if HAS_LORA:
            acc = _lora_b(acc, LORA_B, LORA_H, e, p.to(tl.int64), rows64, rmask, scaling,
                          stride_be, stride_bn, stride_br, R, R_PAD, LORA_KIND)
        gate, up = tl.split(tl.reshape(acc, (BLOCK_N // 2, 2)))
        inter = _gated_act(gate, up, alpha, limit, ACT)
        half = tl.program_id(1) * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
        tl.store(OUT + p.to(tl.int64) * (N // 2) + half, inter, mask = half < N // 2)

    @triton.jit
    def _routed_down_kernel(
        X, IDX, RW, W_PTRS, A_PTRS, A2_PTRS, C2_PTRS, OFFSETS, LUT, BIAS, LORA_B, LORA_H, OUT, N, K,
        RW_STRIDE, scaling, stride_be, stride_bn, stride_br,
        TOP_K: tl.constexpr, DENSE_RW: tl.constexpr, BLOCKSIZE: tl.constexpr, BLOCKSIZE2: tl.constexpr,
        NESTED: tl.constexpr, STACKED: tl.constexpr, BIAS_KIND: tl.constexpr, HAS_LORA: tl.constexpr,
        R: tl.constexpr, R_PAD: tl.constexpr, LORA_KIND: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # OUT[t] = sum_k rw[t, k] * (dequant(W[e_k]) @ X[t * TOP_K + k] + BIAS[e_k] (+ LoRA)),
        # fp32 accumulation, one store in the output dtype: no atomics, deterministic.
        t = tl.program_id(0)
        rows = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        rmask = rows < N
        rows64 = rows.to(tl.int64)
        ks = tl.arange(0, BLOCK_K)
        out = tl.zeros([BLOCK_N], dtype = tl.float32)
        for k in tl.static_range(TOP_K):
            s = t.to(tl.int64) * TOP_K + k
            e = tl.load(IDX + s).to(tl.int64)
            if DENSE_RW:
                rw = tl.load(RW + t.to(tl.int64) * RW_STRIDE + e).to(tl.float32)
            else:
                rw = tl.load(RW + t.to(tl.int64) * RW_STRIDE + k).to(tl.float32)
            W, A, A2, C2, offset = _expert_tables(e, W_PTRS, A_PTRS, A2_PTRS, C2_PTRS, OFFSETS, NESTED, STACKED)
            if STACKED:
                wrows = e * N + rows64
            else:
                wrows = rows64
            acc = tl.zeros([BLOCK_N], dtype = tl.float32)
            for k0 in range(0, K, BLOCK_K):
                x = tl.load(X + s * K + k0 + ks, mask = (k0 + ks) < K, other = 0.0).to(tl.float32)
                acc += _nf4_dot(W, A, A2, C2, offset, LUT, wrows, rmask, x, k0, K,
                                BLOCKSIZE, BLOCKSIZE2, NESTED, BLOCK_N, BLOCK_K)
            if BIAS_KIND != 0:
                acc += _expert_bias(BIAS, e, N, rows64, rmask, BIAS_KIND)
            if HAS_LORA:
                acc = _lora_b(acc, LORA_B, LORA_H, e, s, rows64, rmask, scaling,
                              stride_be, stride_bn, stride_br, R, R_PAD, LORA_KIND)
            out += rw * acc
        tl.store(OUT + t.to(tl.int64) * N + rows, out.to(OUT.dtype.element_ty), mask = rmask)

    @triton.jit
    def _lora_h_kernel(X, IDX, A_PTRS, OUT, K, ROW_DIV, R: tl.constexpr, LORA_KIND: tl.constexpr,
                       BLOCK_K: tl.constexpr):
        # OUT[p, j] = A[IDX[p]][j] @ X[p // ROW_DIV], fp32. One program per (slot, rank row).
        p = tl.program_id(0)
        j = tl.program_id(1)
        e = tl.load(IDX + p).to(tl.int64)
        a_row = _lora_ptr(A_PTRS, e, LORA_KIND) + j.to(tl.int64) * K
        x_row = X + (p // ROW_DIV).to(tl.int64) * K
        ks = tl.arange(0, BLOCK_K)
        acc = tl.zeros([BLOCK_K], dtype = tl.float32)
        for k0 in range(0, K, BLOCK_K):
            kmask = (k0 + ks) < K
            x = tl.load(x_row + k0 + ks, mask = kmask, other = 0.0).to(tl.float32)
            a = tl.load(a_row + k0 + ks, mask = kmask, other = 0.0).to(tl.float32)
            acc += a * x
        tl.store(OUT + p.to(tl.int64) * R + j, tl.sum(acc, axis = 0))

    @triton.jit
    def _nf4_select_dequant_kernel(
        Q, OUT, UNIQ, LUT, ABSMAX, CODE2, ABSMAX2, OFFSET, BYTES_PER_EXPERT,
        BLOCKSIZE: tl.constexpr, BLOCKSIZE2: tl.constexpr, NESTED: tl.constexpr, BLOCK: tl.constexpr,
    ):
        # OUT[g] = dequant(expert UNIQ[g]) in OUT's dtype, bit-identical to bitsandbytes
        # (launched with fp fusion off: product, then add, two roundings); UNIQ[g] < 0 skips.
        g = tl.program_id(0)
        e = tl.load(UNIQ + g).to(tl.int64)
        if e < 0:
            return
        offs = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        mask = offs < BYTES_PER_EXPERT
        gofs = e * BYTES_PER_EXPERT + offs
        qw = tl.load(Q + gofs, mask = mask, other = 0)
        blk = gofs // (BLOCKSIZE // 2)
        if NESTED:
            aq = tl.load(ABSMAX + blk, mask = mask, other = 0).to(tl.int32)
            am2 = tl.load(ABSMAX2 + blk // BLOCKSIZE2, mask = mask, other = 0.0).to(tl.float32)
            am = tl.load(CODE2 + aq) * am2 + tl.load(OFFSET).to(tl.float32)
        else:
            am = tl.load(ABSMAX + blk, mask = mask, other = 0.0).to(tl.float32)
        vh = (tl.load(LUT + (qw >> 4).to(tl.int32)) * am).to(OUT.dtype.element_ty)
        vl = (tl.load(LUT + (qw & 15).to(tl.int32)) * am).to(OUT.dtype.element_ty)
        w = tl.reshape(tl.join(vh, vl), (2 * BLOCK,))
        offs2 = tl.program_id(1).to(tl.int64) * (2 * BLOCK) + tl.arange(0, 2 * BLOCK).to(tl.int64)
        tl.store(OUT + g.to(tl.int64) * (2 * BYTES_PER_EXPERT) + offs2, w, mask = offs2 < 2 * BYTES_PER_EXPERT)

    # BF16 / FP16 experts: 3D [E, K, N] weights, one program per (expert, N block, K split).
    @triton.jit
    def _routed_bf16_expert_kernel(
        X, IDX, W, BIAS, OUT,
        P, N, K,
        ROW_DIV,
        stride_xr,
        stride_we, stride_wk, stride_wn,
        stride_be,
        HAS_BIAS: tl.constexpr,
        P_PAD: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
    ):
        # Reads the expert's weight tile once for every slot routed to it (masked dot over P).
        e = tl.program_id(0)
        nb = tl.program_id(1)
        sk = tl.program_id(2)
        slots = tl.arange(0, P_PAD)
        ids = tl.load(IDX + slots, mask = slots < P, other = -1)
        hit = ids == e
        if tl.sum(hit.to(tl.int32), axis = 0) == 0:
            return
        rows = (slots // ROW_DIV).to(tl.int64)
        n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        nmask = n < N
        ks = tl.arange(0, BLOCK_K)
        w_base = W + e.to(tl.int64) * stride_we
        acc = tl.zeros([P_PAD, BLOCK_N], dtype = tl.float32)
        k_per = tl.cdiv(tl.cdiv(K, SPLIT_K), BLOCK_K) * BLOCK_K
        k_lo = sk * k_per
        for k0 in range(k_lo, tl.minimum(k_lo + k_per, K), BLOCK_K):
            k = k0 + ks
            kmask = k < K
            x = tl.load(X + rows[:, None] * stride_xr + k[None, :], mask = hit[:, None] & kmask[None, :], other = 0.0)
            w = tl.load(
                w_base + k[:, None] * stride_wk + n[None, :] * stride_wn,
                mask = kmask[:, None] & nmask[None, :], other = 0.0,
            )
            acc = tl.dot(x.to(w.dtype), w, acc)
        if HAS_BIAS:
            if sk == 0:
                b = tl.load(BIAS + e.to(tl.int64) * stride_be + n, mask = nmask, other = 0.0).to(tl.float32)
                acc += b[None, :]
        out_ptr = OUT + (sk * P + slots.to(tl.int64))[:, None] * N + n[None, :]
        tl.store(out_ptr, acc.to(OUT.dtype.element_ty), mask = hit[:, None] & nmask[None, :])


# ---------------------------------------------------------------------------------------
# NF4 launches. A table `tb` is a dict: w / a / a2 / c2 / off (pointer tables, or the stacked
# tensors themselves), lut, bias, N, K, blocksize, blocksize2, nested, stacked.
# ---------------------------------------------------------------------------------------

def _block_k(K, blocksize):
    return max(blocksize, min(1024, triton.next_power_of_2(K)))


def _lora_args(lora, dummy):
    # (B, H [P, r] fp32, scaling, r, strides, LORA_KIND) or dummies.
    if lora is None:
        return dummy, dummy, 0.0, 1, (0, 0, 0), 0
    B, H, scaling = lora[:3]
    if isinstance(B, (list, tuple)):
        got = lora_pointer_table(B)
        if got is not None:
            r = H.shape[-1]
            return got[0], H, float(scaling), r, (0, r, 1), got[1]
        B = torch.stack(B)
    return B, H, float(scaling), B.shape[-1], (B.stride(0), B.stride(1), B.stride(2)), 0


def _bias_args(tb):
    # Per-expert bias tables are resolved on every launch: bitsandbytes recasts Linear4bit.bias
    # into new storage whenever the input dtype differs.
    bias = tb["bias"]
    if isinstance(bias, (list, tuple)):
        got = lora_pointer_table(bias)
        if got is not None:
            return got
        return torch.stack([b.float() for b in bias]), 4
    return bias, tb["bias_kind"]


def _gate_up(x, idx, tb, top_k, lora, alpha, limit, out, act = ACT_GPTOSS, interleaved = True):
    B, H, scaling, r, bs, kind = _lora_args(lora, tb["lut"])
    bias, bias_kind = _bias_args(tb)
    # Triton launches on the current device, not the tensors' (multi-GPU device_map).
    with torch.cuda.device(x.device):
        _routed_gate_up_kernel[(idx.numel(), triton.cdiv(tb["N"], 4))](
            x, idx, tb["w"], tb["a"], tb["a2"], tb["c2"], tb["off"], tb["lut"],
            bias if bias is not None else tb["lut"], B, H, out, tb["N"], tb["K"],
            float(alpha), float(limit), scaling, bs[0], bs[1], bs[2],
            TOP_K = top_k, BLOCKSIZE = tb["blocksize"], BLOCKSIZE2 = tb["blocksize2"], NESTED = tb["nested"],
            STACKED = tb.get("stacked", False), ACT = act, INTERLEAVED = interleaved,
            BIAS_KIND = bias_kind, HAS_LORA = lora is not None, R = r, R_PAD = triton.next_power_of_2(r),
            LORA_KIND = kind, BLOCK_N = 4, BLOCK_K = _block_k(tb["K"], tb["blocksize"]), num_warps = 4,
        )
    return out


def _down(x, idx, rw, dense_rw, tb, top_k, lora, out):
    B, H, scaling, r, bs, kind = _lora_args(lora, tb["lut"])
    bias, bias_kind = _bias_args(tb)
    with torch.cuda.device(x.device):
        _routed_down_kernel[(out.shape[0], triton.cdiv(tb["N"], 4))](
            x, idx, rw, tb["w"], tb["a"], tb["a2"], tb["c2"], tb["off"], tb["lut"],
            bias if bias is not None else tb["lut"], B, H, out, tb["N"], tb["K"],
            rw.stride(0), scaling, bs[0], bs[1], bs[2],
            TOP_K = top_k, DENSE_RW = dense_rw, BLOCKSIZE = tb["blocksize"], BLOCKSIZE2 = tb["blocksize2"],
            NESTED = tb["nested"], STACKED = tb.get("stacked", False), BIAS_KIND = bias_kind,
            HAS_LORA = lora is not None, R = r, R_PAD = triton.next_power_of_2(r), LORA_KIND = kind,
            BLOCK_N = 4, BLOCK_K = _block_k(tb["K"], tb["blocksize"]), num_warps = 4,
        )
    return out


def _tb(w, a, a2, c2, off, lut, bias, N, K, blocksize, blocksize2, nested, bias_kind, stacked):
    return dict(w = w, a = a, a2 = a2, c2 = c2, off = off, lut = lut, bias = bias, N = N, K = K,
                blocksize = blocksize, blocksize2 = blocksize2, nested = nested, bias_kind = bias_kind,
                stacked = stacked)


if triton is not None:
    # custom_op, not triton_op: Inductor before torch 2.12 dropped the gate_up kernel from the graph.

    @torch.library.custom_op("unsloth_zoo::routed_nf4_gate_up", mutates_args = ())
    def _gate_up_op(
        x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor, a: torch.Tensor, a2: torch.Tensor,
        c2: torch.Tensor, off: torch.Tensor, lut: torch.Tensor, bias: Optional[torch.Tensor],
        N: int, K: int, blocksize: int, blocksize2: int, nested: bool, bias_kind: int, stacked: bool,
        top_k: int, lora_b: Optional[torch.Tensor], lora_b_list: List[torch.Tensor],
        lora_h: Optional[torch.Tensor], scaling: float, alpha: float, limit: float, act: int, interleaved: bool,
        bias_list: List[torch.Tensor],
    ) -> torch.Tensor:
        tb = _tb(w, a, a2, c2, off, lut, list(bias_list) if bias_list else bias, N, K, blocksize, blocksize2,
                 nested, bias_kind, stacked)
        lora = _op_lora(lora_b, lora_b_list, lora_h, scaling)
        out = torch.empty((idx.numel(), N // 2), dtype = torch.float32, device = x.device)
        return _gate_up(x, idx, tb, top_k, lora, alpha, limit, out, act, interleaved)

    @_gate_up_op.register_fake
    def _routed_nf4_gate_up_fake(x, idx, w, a, a2, c2, off, lut, bias, N, K, blocksize, blocksize2, nested,
                                 bias_kind, stacked, top_k, lora_b, lora_b_list, lora_h, scaling, alpha, limit,
                                 act, interleaved, bias_list):
        return x.new_empty((idx.numel(), N // 2), dtype = torch.float32)

    @torch.library.custom_op("unsloth_zoo::routed_nf4_down", mutates_args = ())
    def _down_op(
        x: torch.Tensor, idx: torch.Tensor, rw: torch.Tensor, dense_rw: bool, w: torch.Tensor,
        a: torch.Tensor, a2: torch.Tensor, c2: torch.Tensor, off: torch.Tensor, lut: torch.Tensor,
        bias: Optional[torch.Tensor], N: int, K: int, blocksize: int, blocksize2: int, nested: bool,
        bias_kind: int, stacked: bool, top_k: int, lora_b: Optional[torch.Tensor],
        lora_b_list: List[torch.Tensor], lora_h: Optional[torch.Tensor], scaling: float,
        out_dtype: torch.dtype, bias_list: List[torch.Tensor],
    ) -> torch.Tensor:
        tb = _tb(w, a, a2, c2, off, lut, list(bias_list) if bias_list else bias, N, K, blocksize, blocksize2,
                 nested, bias_kind, stacked)
        lora = _op_lora(lora_b, lora_b_list, lora_h, scaling)
        out = torch.empty((idx.numel() // top_k, N), dtype = out_dtype, device = x.device)
        return _down(x, idx, rw, dense_rw, tb, top_k, lora, out)

    @_down_op.register_fake
    def _routed_nf4_down_fake(x, idx, rw, dense_rw, w, a, a2, c2, off, lut, bias, N, K, blocksize, blocksize2,
                              nested, bias_kind, stacked, top_k, lora_b, lora_b_list, lora_h, scaling, out_dtype,
                              bias_list):
        return x.new_empty((idx.numel() // top_k, N), dtype = out_dtype)


def _table_args(tb):
    bias = tb["bias"]
    return (tb["w"], tb["a"], tb["a2"], tb["c2"], tb["off"], tb["lut"], None if isinstance(bias, (list, tuple)) else bias,
            tb["N"], tb["K"], tb["blocksize"], tb["blocksize2"], tb["nested"], tb["bias_kind"],
            tb.get("stacked", False))


def _bias_list(tb):
    bias = tb["bias"]
    return list(bias) if isinstance(bias, (list, tuple)) else []


def _op_lora(lora_b, lora_b_list, lora_h, scaling):
    if lora_b is None and not lora_b_list:
        return None
    return (lora_b if lora_b is not None else list(lora_b_list), lora_h, scaling)


def _lora_op_args(lora):
    if lora is None:
        return None, [], None, 0.0
    B = lora[0]
    if isinstance(B, (list, tuple)):
        return None, list(B), lora[1], float(lora[2])
    return B, [], lora[1], float(lora[2])


def routed_gate_up(x, idx, tb, top_k, alpha, limit, lora = None, act = ACT_GPTOSS, interleaved = True):
    """[T * top_k, N // 2] fp32: act(dequant(W[idx[p]]) @ x[p // top_k] + bias[idx[p]] (+ LoRA)).
    lora: (B, H [P, r] fp32, scaling) or None; B: [E, N, r] or a list of per-expert [N, r]."""
    if torch.compiler.is_compiling():
        return torch.ops.unsloth_zoo.routed_nf4_gate_up(
            x, idx, *_table_args(tb), top_k, *_lora_op_args(lora), float(alpha), float(limit),
            int(act), bool(interleaved), _bias_list(tb))
    # Eager launches directly: the dispatcher costs tens of us per call at decode sizes.
    out = torch.empty((idx.numel(), tb["N"] // 2), dtype = torch.float32, device = x.device)
    return _gate_up(x, idx, tb, top_k, lora, alpha, limit, out, act, interleaved)


def routed_down(x, idx, rw, dense_rw, tb, top_k, out_dtype, lora = None):
    """[T, N]: sum over the top-k slots of rw * (dequant(W[e]) @ x[slot] + bias[e] (+ LoRA))."""
    if torch.compiler.is_compiling():
        return torch.ops.unsloth_zoo.routed_nf4_down(
            x, idx, rw, dense_rw, *_table_args(tb), top_k, *_lora_op_args(lora), out_dtype, _bias_list(tb))
    out = torch.empty((idx.numel() // top_k, tb["N"]), dtype = out_dtype, device = x.device)
    return _down(x, idx, rw, dense_rw, tb, top_k, lora, out)


LORA_KINDS = {torch.float32: 1, torch.bfloat16: 2, torch.float16: 3}
# (device, dtype, shape, addresses) -> int64 table. Kept (a captured CUDA graph may still read a
# table); a new entry needs new storage (a .to() / cast / reload, or bitsandbytes recasting a bias).
_PTR_TABLES = {}
_PTR_TABLES_MAX = 1 << 16


def lora_pointer_table(weights):
    """(int64 [E] table of the weights' addresses, LORA_KIND), or None unless every weight is a
    contiguous CUDA tensor of one shape, dtype and device in LORA_KINDS. Resolved per call, so
    the kernels always read each expert's live adapter."""
    w0 = weights[0]
    kind = LORA_KINDS.get(w0.dtype)
    if kind is None or not w0.is_cuda:
        return None
    ptrs = tuple(w.data_ptr() for w in weights)
    key = (w0.device, w0.dtype, tuple(w0.shape), ptrs)
    table = _PTR_TABLES.get(key)
    if table is None:
        for w in weights:
            if w.dtype != w0.dtype or w.device != w0.device or w.shape != w0.shape or not w.is_contiguous():
                return None
        if torch.cuda.is_current_stream_capturing():
            return None  # no host copy inside a capture: the stacked path runs instead
        with torch.inference_mode(False):
            table = torch.tensor(ptrs, dtype = torch.int64, device = w0.device)
        if len(_PTR_TABLES) >= _PTR_TABLES_MAX:
            _PTR_TABLES.pop(next(iter(_PTR_TABLES)))
        _PTR_TABLES[key] = table
    return table, kind


def _lora_h_launch(x, idx, a_ptrs, kind, r, row_div, out):
    K = x.shape[-1]
    with torch.cuda.device(x.device):
        _lora_h_kernel[(idx.numel(), r)](
            x, idx, a_ptrs, out, K, row_div, R = r, LORA_KIND = kind,
            BLOCK_K = max(16, min(1024, triton.next_power_of_2(K))), num_warps = 4,
        )
    return out


def _lora_h_eager(x, idx, a_list, row_div):
    got = lora_pointer_table(a_list)
    r = a_list[0].shape[0]
    if got is None:
        x_slots = x[torch.arange(idx.numel(), device = x.device) // row_div] if row_div > 1 else x
        A = torch.stack(a_list)
        return torch.bmm(A[idx].float(), x_slots.float().unsqueeze(-1)).squeeze(-1)
    out = torch.empty((idx.numel(), r), dtype = torch.float32, device = x.device)
    return _lora_h_launch(x, idx, got[0], got[1], r, row_div, out)


if triton is not None:

    @torch.library.custom_op("unsloth_zoo::routed_lora_h", mutates_args = ())
    def _lora_h_op(x: torch.Tensor, idx: torch.Tensor, a_list: List[torch.Tensor], row_div: int) -> torch.Tensor:
        return _lora_h_eager(x, idx, list(a_list), row_div)

    @_lora_h_op.register_fake
    def _routed_lora_h_fake(x, idx, a_list, row_div):
        return x.new_empty((idx.numel(), a_list[0].shape[0]), dtype = torch.float32)


def routed_lora_h(x, idx, a_list, row_div = 1):
    """fp32 [P, r]: row p = A[idx[p]] @ x[p // row_div] for a list of per-expert [r, K] weights,
    read in place through lora_pointer_table. x: [R, K] contiguous."""
    if torch.compiler.is_compiling():
        return torch.ops.unsloth_zoo.routed_lora_h(x, idx, list(a_list), int(row_div))
    return _lora_h_eager(x, idx, a_list, row_div)


# ---------------------------------------------------------------------------------------
# Selective NF4 dequant (comparator arm): only the experts in `uniq`, sync-free.
# ---------------------------------------------------------------------------------------

def _select_dequant_launch(tb, uniq, out):
    E_n = out.shape[0]
    bytes_per_expert = tb["N"] * tb["K"] // 2
    BLOCK = 1024
    with torch.cuda.device(out.device):
        _nf4_select_dequant_kernel[(E_n, triton.cdiv(bytes_per_expert, BLOCK))](
            tb["w"], out, uniq, tb["lut"], tb["a"], tb["c2"], tb["a2"], tb["off"], bytes_per_expert,
            BLOCKSIZE = tb["blocksize"], BLOCKSIZE2 = tb["blocksize2"], NESTED = tb["nested"],
            BLOCK = BLOCK, num_warps = 4, enable_fp_fusion = False,
        )
    return out


if triton is not None:

    @torch.library.custom_op("unsloth_zoo::nf4_select_dequant", mutates_args = ())
    def _select_dequant_op(
        w: torch.Tensor, a: torch.Tensor, a2: torch.Tensor, c2: torch.Tensor, off: torch.Tensor,
        lut: torch.Tensor, uniq: torch.Tensor, N: int, K: int, blocksize: int, blocksize2: int,
        nested: bool, dtype: torch.dtype,
    ) -> torch.Tensor:
        tb = _tb(w, a, a2, c2, off, lut, None, N, K, blocksize, blocksize2, nested, 0, True)
        out = torch.empty((uniq.numel(), N, K), dtype = dtype, device = w.device)
        return _select_dequant_launch(tb, uniq, out)

    @_select_dequant_op.register_fake
    def _nf4_select_dequant_fake(w, a, a2, c2, off, lut, uniq, N, K, blocksize, blocksize2, nested, dtype):
        return w.new_empty((uniq.numel(), N, K), dtype = dtype)


def nf4_select_dequant(tb, uniq, dtype):
    """[len(uniq), N, K] in dtype: rows g hold expert uniq[g] of a stacked NF4 table, bit-exact
    to bitsandbytes; entries with uniq[g] < 0 are left unwritten."""
    if torch.compiler.is_compiling():
        return torch.ops.unsloth_zoo.nf4_select_dequant(
            tb["w"], tb["a"], tb["a2"], tb["c2"], tb["off"], tb["lut"], uniq,
            tb["N"], tb["K"], tb["blocksize"], tb["blocksize2"], tb["nested"], dtype)
    out = torch.empty((uniq.numel(), tb["N"], tb["K"]), dtype = dtype, device = tb["w"].device)
    return _select_dequant_launch(tb, uniq, out)


# ---------------------------------------------------------------------------------------
# BF16 / FP16 expert-major GEMM.
# ---------------------------------------------------------------------------------------

_EXPERT_CONFIG = None  # tests / sweeps may force one config


def _expert_config(P):
    # Tuned on B200 at gpt-oss-20b shapes (H = I = 2880, E = 32, top-4), L2 defeated.
    if _EXPERT_CONFIG is not None:
        return _EXPERT_CONFIG
    if P <= 16:
        return {"BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 4, "num_warps": 4, "num_stages": 3}
    if P <= 64:
        return {"BLOCK_N": 64, "BLOCK_K": 64, "SPLIT_K": 4, "num_warps": 4, "num_stages": 3}
    return {"BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1, "num_warps": 4, "num_stages": 3}


def _launch_expert(x, idx, weight, bias, out, row_div):
    S, P, N = out.shape
    E, K = weight.shape[0], weight.shape[1]
    c = _expert_config(P)
    grid = (E, triton.cdiv(N, c["BLOCK_N"]), S)
    with torch.cuda.device(x.device):
        _routed_bf16_expert_kernel[grid](
            x, idx, weight, bias if bias is not None else weight, out,
            P, N, K, row_div,
            x.stride(0),
            weight.stride(0), weight.stride(1), weight.stride(2),
            bias.stride(0) if bias is not None else 0,
            HAS_BIAS = bias is not None,
            P_PAD = max(16, triton.next_power_of_2(P)),
            BLOCK_N = c["BLOCK_N"], BLOCK_K = c["BLOCK_K"], SPLIT_K = S,
            num_warps = c["num_warps"], num_stages = c["num_stages"],
        )
    return out


if triton is not None:

    # custom_op (torch >= 2.4), as for NF4; triton_op only exists from torch 2.6.
    @torch.library.custom_op("unsloth_zoo::routed_bf16_gemm", mutates_args = ())
    def _routed_bf16_gemm_op(
        x: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], row_div: int,
    ) -> torch.Tensor:
        part = torch.empty(_expert_config(idx.numel())["SPLIT_K"], idx.numel(), weight.shape[2], dtype = torch.float32, device = x.device)
        return _launch_expert(x, idx, weight, bias, part, row_div)

    @_routed_bf16_gemm_op.register_fake
    def _routed_bf16_gemm_fake(x, idx, weight, bias, row_div):
        return x.new_empty((_expert_config(idx.numel())["SPLIT_K"], idx.numel(), weight.shape[2]), dtype = torch.float32)


def routed_bf16_gemm(x, idx, weight, bias = None, row_div = 1):
    """fp32 [P, N]: row p = x[p // row_div] @ weight[idx[p]] + bias[idx[p]].
    x: [R, K]; idx: [P] expert ids (on device); weight: [E, K, N] (any strides); bias: [E, N] or None.
    Each active expert's weights are read once; no host sync, so it compiles and captures."""
    idx = idx.reshape(-1)
    if x.stride(-1) != 1:
        x = x.contiguous()
    if torch.compiler.is_compiling():
        part = _routed_bf16_gemm_op(x, idx, weight, bias, row_div)
    else:
        part = torch.empty(_expert_config(idx.numel())["SPLIT_K"], idx.numel(), weight.shape[2], dtype = torch.float32, device = x.device)
        _launch_expert(x, idx, weight, bias, part, row_div)
    # Fixed-order sum of the k-split partials: deterministic.
    return part.sum(0) if part.shape[0] > 1 else part[0]


def _act_torch(gate_up, act, interleaved, alpha = 1.702, limit = 7.0):
    """Gated activation over an fp32 [P, 2I] gate_up, as the grouped_mm path computes it."""
    if interleaved:
        gate, up = gate_up[:, ::2], gate_up[:, 1::2]
    else:
        gate, up = gate_up.chunk(2, dim = -1)
    if act == ACT_GPTOSS:
        gate = gate.clamp(max = limit)
        up = up.clamp(min = -limit, max = limit)
        return (up + 1) * (gate * torch.sigmoid(gate * alpha))
    if act == ACT_SILU:
        return F.silu(gate) * up
    if act == ACT_GELU_TANH:
        return F.gelu(gate, approximate = "tanh") * up
    return F.gelu(gate) * up


def _lora_delta(x_slots, idx, terms):
    """sum over terms of (x_slots @ first[idx] @ second[idx]) * scaling, fp32 [P, out].
    first: [E, in, r]; second: [E, r, out]."""
    delta = None
    for first, second, scaling in terms:
        h = torch.bmm(x_slots.to(first.dtype)[:, None, :], first[idx])
        d = torch.bmm(h, second[idx])[:, 0].float() * scaling
        delta = d if delta is None else delta + d
    return delta


def routed_bf16_moe(x, idx, routing_weights, w_gu, w_dn, b_gu, b_dn, act, interleaved,
                    gu_lora = (), dn_lora = (), alpha = 1.702, limit = 7.0, out_dtype = None):
    """MoE output from only the routed BF16 / FP16 experts.
    x: [T, H]; idx: [T, top_k]; routing_weights: [T, top_k]; w_gu: [E, H, 2I] and w_dn: [E, I, H]
    (any strides); gu_lora / dn_lora: [(first, second, scaling), ...]. Returns [T, H]."""
    T, top_k = idx.shape
    flat = idx.reshape(-1)
    gate_up = routed_bf16_gemm(x.to(w_gu.dtype), flat, w_gu, b_gu, row_div = top_k)
    if gu_lora:
        gate_up = gate_up + _lora_delta(x.repeat_interleave(top_k, 0), flat, gu_lora)
    inter = _act_torch(gate_up, act, interleaved, alpha, limit)
    down = routed_bf16_gemm(inter.to(w_dn.dtype), flat, w_dn, b_dn, row_div = 1)
    if dn_lora:
        down = down + _lora_delta(inter, flat, dn_lora)
    out = (down.view(T, top_k, -1) * routing_weights.float()[..., None]).sum(1)
    return out.to(out_dtype if out_dtype is not None else x.dtype)


# ---------------------------------------------------------------------------------------
# Generic experts modules (transformers v5 3D experts: Qwen3 / Qwen3.5 MoE, Gemma 4, Mixtral, ...)
# ---------------------------------------------------------------------------------------

def _semantics(experts):
    """Everything besides the weights that _act_code / the plans read; compared on every reuse."""
    from .moe_utils import _uses_own_apply_gate, _gate_up_is_interleaved
    d, cls = experts.__dict__, type(experts)
    # Instance dict / submodules / class, as nn.Module.__getattr__ would find them, without its cost.
    get = lambda n, default: d.get(n, d["_modules"].get(n, d["_parameters"].get(n, getattr(cls, n, default))))
    return (
        get("act_fn", None), bool(_gate_up_is_interleaved(experts)), bool(_uses_own_apply_gate(experts)),
        bool(get("is_transposed", False)), bool(get("has_gate", True)), get("alpha", None), get("limit", None),
    )


def _same_semantics(a, b):
    # Objects (act_fn, tensor alpha / limit) by identity, plain numbers and flags by value.
    for x, y in zip(a, b):
        if x is y:
            continue
        if not (isinstance(x, (bool, int, float)) and isinstance(y, (bool, int, float)) and x == y):
            return False
    return True


def _act_code(experts):
    """ACT constexpr for this experts module, or None when its activation is not one the
    kernels implement. Probed by value, not class name (ACT2FN has many spellings)."""
    from .moe_utils import _uses_own_apply_gate
    if "GptOssExperts" in type(experts).__name__:
        return ACT_GPTOSS
    if _uses_own_apply_gate(experts):
        return None
    fn = getattr(experts, "act_fn", None)
    if fn is None:
        return ACT_SILU  # forward_native_grouped_mm's own default
    if not callable(fn):
        return None
    try:
        with torch.no_grad():
            g = torch.linspace(-8, 8, 257, dtype = torch.float64)
            got = fn(g).double()
            for code, ref in ((ACT_SILU, F.silu(g)), (ACT_GELU_TANH, F.gelu(g, approximate = "tanh")),
                              (ACT_GELU, F.gelu(g))):
                if torch.allclose(got, ref, rtol = 1e-6, atol = 1e-9):
                    return code
    except Exception:
        pass
    return None


def _is_input_major(experts, name, shape, hidden_dim):
    """True / False as the grouped_mm path would orient this weight ([E, in, out] / [E, out, in]),
    None when it would reshape it some other way (a registered preprocessor)."""
    from .moe_utils import preprocess_weight
    meta = torch.empty(shape, device = "meta")
    got = preprocess_weight(meta, "gate_up" if name == "gate_up_proj" else "down", hidden_dim,
                            getattr(experts, "_unsloth_model_type", None), experts_module = experts)
    if got is meta:
        return True
    if got.shape == meta.shape[:-2] + (meta.shape[-1], meta.shape[-2]) and got.stride(-2) == 1:
        return False
    return None


def _nf4_table(param, device):
    qs = getattr(param, "quant_state", None)
    if qs is None or getattr(qs, "quant_type", None) != "nf4":
        return None
    shape = tuple(getattr(param, "_original_shape", None) or qs.shape)
    if len(shape) != 3:
        return None
    E, N, K = (int(s) for s in shape)
    blocksize = int(qs.blocksize)
    # Blocks must not straddle a row, and expert e's rows start at row e * N of the flat buffer:
    # packed bytes e*N*K/2, absmax e*N*K/blocksize, state2 absmax (e*N*K/blocksize)/blocksize2.
    if blocksize not in (32, 64, 128, 256, 512, 1024) or K % blocksize != 0 or N % 2 != 0:
        return None
    packed = param.data
    if packed.device != device or not packed.is_contiguous():
        return None
    packed = packed.view(torch.uint8).reshape(-1)
    if packed.numel() * 2 != E * N * K or qs.code.numel() != 16:
        return None
    nested = bool(qs.nested)
    if nested:
        s2 = qs.state2
        a, a2, c2 = qs.absmax, s2.absmax, s2.code
        off = qs.offset
        if not isinstance(off, torch.Tensor):
            off = torch.tensor(float(off), dtype = torch.float32, device = device)
        if a.dtype != torch.uint8 or a.numel() != E * N * K // blocksize:
            return None
        a2, c2 = a2.float().contiguous(), c2.float().contiguous()
        off = off.reshape(1).float().contiguous()
        blocksize2 = int(s2.blocksize)
    else:
        a = qs.absmax
        if a.dtype != torch.float32 or a.numel() != E * N * K // blocksize:
            return None
        a2 = c2 = off = a
        blocksize2 = 1
    for t in (a, a2, c2, off):
        if t.device != device:
            return None
    return {
        "w": packed, "a": a.contiguous(), "a2": a2, "c2": c2, "off": off,
        "lut": qs.code.to(device = device, dtype = torch.float32).contiguous(),
        "bias": None, "bias_kind": 0, "N": N, "K": K, "E": E, "blocksize": blocksize, "blocksize2": blocksize2,
        "nested": nested, "stacked": True,
    }


def _stacked_sources(qs):
    # Every quant-state tensor _nf4_table reads; held by the state so a replaced one gets a new address.
    ts = [qs.absmax, qs.code]
    if qs.nested:
        ts += [qs.state2.absmax, qs.state2.code]
        if isinstance(qs.offset, torch.Tensor):
            ts.append(qs.offset)
    return ts


def _stacked_key(experts):
    gu, dn = experts.gate_up_proj, experts.down_proj
    biases = tuple(
        b.data_ptr() if isinstance(b, torch.Tensor) else 0
        for b in (getattr(experts, "gate_up_proj_bias", None), getattr(experts, "down_proj_bias", None)))
    key = [gu.data_ptr(), dn.data_ptr()]
    for qs in (gu.quant_state, dn.quant_state):
        # Version counters catch in-place edits of buffers the tables copy (offset, code, state2).
        key += [(t.data_ptr(), t._version) for t in _stacked_sources(qs)]
        key += [qs.blocksize, bool(qs.nested)]
        if qs.nested:
            key += [qs.state2.blocksize, None if isinstance(qs.offset, torch.Tensor) else float(qs.offset)]
    return tuple(key) + biases


def prepare_stacked_nf4(experts, hidden_dim = None):
    """Tables for an experts module holding one stacked NF4 Params4bit per projection,
    built (or revalidated against the buffers' addresses) eagerly; None when unsupported."""
    state = experts.__dict__.get("_unsloth_routed_moe")
    if state is False:
        return None
    try:
        key = _stacked_key(experts)
    except Exception:
        return None
    if state is not None and state["key"] == key and _same_semantics(state["sem"], _semantics(experts)):
        return state
    state = _build_stacked_nf4(experts, hidden_dim)
    if state is None:
        experts.__dict__["_unsloth_routed_moe"] = False
        return None
    state["key"], state["sem"] = key, _semantics(experts)
    state["held"] = [_stacked_sources(p.quant_state) for p in (experts.gate_up_proj, experts.down_proj)]
    experts.__dict__["_unsloth_routed_moe"] = state
    return state


_DECLINED = set()


def _decline(experts, reason):
    """None, logging once per (experts class, reason) under UNSLOTH_ENABLE_LOGGING=1."""
    key = (type(experts).__name__, reason)
    if key not in _DECLINED:
        _DECLINED.add(key)
        if os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1":
            from unsloth_zoo.log import logger
            logger.info(f"Unsloth: routed MoE decode kernels skip {key[0]}: {reason}.")
    return None


def _build_stacked_nf4(experts, hidden_dim):
    from .moe_utils import _gate_up_is_interleaved
    if triton is None:
        return _decline(experts, "no Triton")
    gu_p, dn_p = experts.gate_up_proj, experts.down_proj
    device = gu_p.data.device
    if device.type != "cuda" or getattr(experts, "is_transposed", False) or not getattr(experts, "has_gate", True):
        return _decline(experts, "not CUDA, transposed or ungated")
    act = _act_code(experts)
    if act is None:
        return _decline(experts, f"activation {type(getattr(experts, 'act_fn', None)).__name__} / own gate")
    gu, dn = _nf4_table(gu_p, device), _nf4_table(dn_p, device)
    if gu is None or dn is None:
        return _decline(experts, "not a stacked NF4 [E, N, K] with whole absmax blocks per row")
    E = gu["E"]
    if dn["E"] != E or gu["N"] != 2 * dn["K"] or gu["K"] != dn["N"]:
        return _decline(experts, "gate_up / down shapes do not pair")
    H = gu["K"]
    if hidden_dim is not None and hidden_dim != H:
        return _decline(experts, "hidden size mismatch")
    # The kernels read [E, out, in] rows; take only modules the grouped_mm path orients the same way.
    if _is_input_major(experts, "gate_up_proj", (E, gu["N"], H), H) is not False:
        return _decline(experts, "gate_up_proj not [E, out, in]")
    if _is_input_major(experts, "down_proj", (E, H, dn["K"]), H) is not False:
        return _decline(experts, "down_proj not [E, out, in]")
    if not torch.equal(gu["lut"], dn["lut"]):
        return _decline(experts, "different NF4 codebooks")
    for name, tb in (("gate_up_proj_bias", gu), ("down_proj_bias", dn)):
        bias = getattr(experts, name, None)
        if bias is None:
            continue
        if (
            not isinstance(bias, torch.Tensor) or tuple(bias.shape) != (E, tb["N"]) or bias.device != device
            or not bias.is_contiguous() or bias.dtype not in (torch.float32, torch.bfloat16, torch.float16)
        ):
            return _decline(experts, f"{name} not a contiguous [E, N] float tensor")
        # The live [E, N] tensor; a swapped one changes the table key (_stacked_key).
        tb["bias"], tb["bias_kind"] = bias.detach(), 4
    return {
        "gate_up": gu, "down": dn, "E": E, "act": act,
        "interleaved": bool(_gate_up_is_interleaved(experts)),
        "alpha": float(getattr(experts, "alpha", 1.702)), "limit": float(getattr(experts, "limit", 7.0)),
    }


# Dynamo before torch 2.10 traces Params4bit as a plain object (no .detach / .view), so compiled
# NF4 calls keep the current path there.
_LIVE_QUANT = tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 10)


def _live_quant(experts, state):
    """Compiled calls skip the eager key check, so read the quant tensors from the module (graph
    inputs, guarded by Dynamo); None (current path) if they no longer fit the cached layout."""
    out = dict(state)
    for name, key in (("gate_up_proj", "gate_up"), ("down_proj", "down")):
        p, tb = getattr(experts, name), state[key]
        qs = getattr(p, "quant_state", None)
        if qs is None or bool(qs.nested) != tb["nested"] or int(qs.blocksize) != tb["blocksize"]:
            return None
        live = {"w": p.detach().view(torch.uint8).reshape(-1), "a": qs.absmax, "lut": qs.code}
        if tb["nested"]:
            if not isinstance(qs.offset, torch.Tensor):
                return None
            live.update(a2 = qs.state2.absmax, c2 = qs.state2.code, off = qs.offset.reshape(1))
            if int(qs.state2.blocksize) != tb["blocksize2"]:
                return None
        for k, t in live.items():
            ref = tb[k]
            if t.dtype != ref.dtype or t.shape != ref.shape or t.device != ref.device or not t.is_contiguous():
                return None
        if not tb["nested"]:
            live.update(a2 = live["a"], c2 = live["a"], off = live["a"])
        out[key] = dict(tb, **live)
    return out


def _live_biases(experts, state):
    """The state with each stacked [E, N] bias re-read from the module; None if it no longer fits."""
    out = state
    for name, key in (("gate_up_proj_bias", "gate_up"), ("down_proj_bias", "down")):
        bias = getattr(experts, name, None)
        if state[key]["bias"] is None:
            if bias is not None:
                return None  # attached after the tables were built
            continue
        tb = state[key]
        if (
            not isinstance(bias, torch.Tensor) or tuple(bias.shape) != (state["E"], tb["N"])
            or not bias.is_contiguous() or bias.dtype not in (torch.float32, torch.bfloat16, torch.float16)
        ):
            return None
        if out is state:
            out = dict(state)
        out[key] = dict(tb, bias = bias.detach())
    return out


def _stash_lora(experts):
    """(gate_up terms, down terms) from the ParamWrapper stash; False when an adapter is attached
    some other way (a param-level wrapper the stash does not carry)."""
    from .moe_utils import take_moe_lora_stash, _has_lora_adapters
    gu = take_moe_lora_stash(experts, "gate_up_proj")
    dn = take_moe_lora_stash(experts, "down_proj")
    if (gu is None and _has_lora_adapters(experts.gate_up_proj)) or (dn is None and _has_lora_adapters(experts.down_proj)):
        return False
    for got in (gu, dn):
        # Anything but (first [E, in, r], second [E, r, out], scaling, ...) is not per-expert LoRA.
        if got is not None and (
            len(got) < 3 or not isinstance(got[0], torch.Tensor) or not isinstance(got[1], torch.Tensor)
            or got[0].dim() != 3 or got[1].dim() != 3 or got[0].shape[0] != got[1].shape[0]
            or got[0].shape[2] != got[1].shape[1]
        ):
            return False
    return gu, dn


def _lora_h(first, idx, x_slots):
    # H[p] = x_slots[p] @ first[idx[p]], [P, r] fp32; first: [E, in, r].
    return torch.bmm(x_slots.float()[:, None, :], first[idx].float())[:, 0]


def _nf4_routed(state, x, idx, rw, top_k, gu_lora, dn_lora, out_dtype):
    gu_tb, dn_tb = state["gate_up"], state["down"]
    flat = idx.reshape(-1)
    lora = None
    if gu_lora is not None:
        first, second, scaling = gu_lora[:3]
        # second [E, r, 2I] -> B [E, 2I, r] view; H from x per slot.
        lora = (second.transpose(1, 2), _lora_h(first, flat, x.repeat_interleave(top_k, 0)), scaling)
    inter = routed_gate_up(x, flat, gu_tb, top_k, state["alpha"], state["limit"], lora,
                           state["act"], state["interleaved"])
    lora = None
    if dn_lora is not None:
        first, second, scaling = dn_lora[:3]
        lora = (second.transpose(1, 2), _lora_h(first, flat, inter), scaling)
    return routed_down(inter, flat, rw, False, dn_tb, top_k, out_dtype, lora)


def _nf4_grouped(state, x, idx, rw, top_k, gu_lora, dn_lora, out_dtype):
    """Comparator: dequantize only the routed experts (sync-free, bit-exact to bitsandbytes)
    into a [min(P, E), N, K] scratch, then torch._grouped_mm."""
    gu_tb, dn_tb = state["gate_up"], state["down"]
    flat = idx.reshape(-1)
    P = flat.numel()
    G = min(P, state["E"])
    sorted_e, order = torch.sort(flat, stable = True)
    new = torch.ones_like(sorted_e, dtype = torch.bool)
    new[1:] = sorted_e[1:] != sorted_e[:-1]
    gid = torch.cumsum(new, 0) - 1
    uniq = torch.full((G,), -1, dtype = torch.int64, device = x.device).scatter_(0, gid, sorted_e)
    counts = torch.zeros(G, dtype = torch.int32, device = x.device).scatter_add_(0, gid, torch.ones_like(gid, dtype = torch.int32))
    offs = torch.cumsum(counts, 0, dtype = torch.int32)
    dtype = x.dtype
    xs = x[order // top_k]
    w_gu = nf4_select_dequant(gu_tb, uniq, dtype)
    mm1 = torch._grouped_mm(xs, w_gu.transpose(1, 2), offs = offs).float()
    if gu_tb["bias"] is not None:
        mm1 = mm1 + gu_tb["bias"][sorted_e].float()
    if gu_lora is not None:
        mm1 = mm1 + _lora_delta(xs, sorted_e, [gu_lora[:3]])
    inter = _act_torch(mm1, state["act"], state["interleaved"], state["alpha"], state["limit"]).to(dtype)
    del w_gu
    w_dn = nf4_select_dequant(dn_tb, uniq, dtype)
    mm2 = torch._grouped_mm(inter, w_dn.transpose(1, 2), offs = offs).float()
    if dn_tb["bias"] is not None:
        mm2 = mm2 + dn_tb["bias"][sorted_e].float()
    if dn_lora is not None:
        mm2 = mm2 + _lora_delta(inter, sorted_e, [dn_lora[:3]])
    slots = torch.empty_like(mm2)
    slots[order] = mm2 * rw.reshape(-1)[order].float()[:, None]
    return slots.view(-1, top_k, slots.shape[-1]).sum(1).to(out_dtype)


@functools.lru_cache(maxsize = None)
def _bf16_supported_index(device_index):
    major, _ = torch.cuda.get_device_capability(device_index)
    return major >= 8


def _bf16_supported(device):
    return _bf16_supported_index(device.index if device.index is not None else torch.cuda.current_device())


def _bf16_plan(experts, gu, dn, hidden_dim):
    """(gate_up input-major, down input-major, ACT, interleaved) for 3D BF16 / FP16 expert
    weights, or None. Pure metadata: cached per module by _bf16_views."""
    if gu.dim() != 3 or dn.dim() != 3:
        return None
    if getattr(gu, "quant_state", None) is not None or gu.dtype not in (torch.bfloat16, torch.float16) or dn.dtype != gu.dtype:
        return None
    if not gu.is_cuda or (gu.dtype == torch.bfloat16 and not _bf16_supported(gu.device)):
        return None
    if getattr(experts, "is_transposed", False) or not getattr(experts, "has_gate", True):
        return None
    act = _act_code(experts)
    if act is None:
        return None
    gu_in = _is_input_major(experts, "gate_up_proj", tuple(gu.shape), hidden_dim)
    dn_in = _is_input_major(experts, "down_proj", tuple(dn.shape), hidden_dim)
    if gu_in is None or dn_in is None:
        return None
    gu_shape = tuple(gu.shape) if gu_in else (gu.shape[0], gu.shape[2], gu.shape[1])
    dn_shape = tuple(dn.shape) if dn_in else (dn.shape[0], dn.shape[2], dn.shape[1])
    if gu_shape[1] != hidden_dim or dn_shape[2] != hidden_dim or gu_shape[2] != 2 * dn_shape[1]:
        return None
    return gu_in, dn_in, act, bool(_gate_up_interleaved(experts))


def _bf16_views(experts, hidden_dim):
    """([E, H, 2I], [E, I, H]) views of 3D BF16 / FP16 expert weights plus (ACT, interleaved),
    or None. Cached per module, keyed on the tensors' identity."""
    gu, dn = experts.__dict__.get("_parameters", {}).get("gate_up_proj"), experts.__dict__.get("_parameters", {}).get("down_proj")
    if gu is None or dn is None:
        gu, dn = getattr(experts, "gate_up_proj", None), getattr(experts, "down_proj", None)
    if not isinstance(gu, torch.Tensor) or not isinstance(dn, torch.Tensor):
        return None
    cache = experts.__dict__.get("_unsloth_routed_bf16")
    meta = (gu.dtype, dn.dtype, gu.device, dn.device, tuple(gu.shape), tuple(dn.shape), hidden_dim)
    if torch.compiler.is_compiling():
        # Built on an eager call first; a retrace after a recast or reshape must not reuse its verdict.
        if cache is None or cache[0][2:] != meta or not _same_semantics(cache[2], _semantics(experts)):
            return None
        plan = cache[1]
    else:
        key = (gu.data_ptr(), dn.data_ptr()) + meta
        sem = _semantics(experts)
        if cache is not None and cache[0] == key and _same_semantics(cache[2], sem):
            plan = cache[1]
        else:
            plan = _bf16_plan(experts, gu, dn, hidden_dim)
            experts.__dict__["_unsloth_routed_bf16"] = (key, plan, sem)
    if plan is None:
        return None
    gu_in, dn_in, act, interleaved = plan
    w_gu = gu if gu_in else gu.transpose(1, 2)
    w_dn = dn if dn_in else dn.transpose(1, 2)
    return w_gu, w_dn, act, interleaved


def routed_moe_forward(experts, hidden_states, top_k_index, top_k_weights):
    """Experts output from only the routed experts for a decode-sized no-grad call, or None
    (callers then run their usual path). top_k_index / top_k_weights: [T, top_k]."""
    if triton is None or torch.is_grad_enabled() or not hidden_states.is_cuda:
        return None
    mode = routed_mode()
    if mode == "0" or top_k_index.dim() != 2:
        return None
    hidden_dim = hidden_states.shape[-1]
    gu_param = getattr(experts, "gate_up_proj", None)
    nf4 = getattr(gu_param, "quant_state", None) is not None
    if nf4:
        # Built eagerly (data_ptr is not traceable), even on calls too large to route.
        if torch.compiler.is_compiling():
            # Without a live read (torch < 2.10) a compiled call cannot see a requantize: fall back.
            state = experts.__dict__.get("_unsloth_routed_moe")
            ok = isinstance(state, dict) and _LIVE_QUANT and _same_semantics(state["sem"], _semantics(experts))
            state = _live_quant(experts, state) if ok else None
        else:
            state = prepare_stacked_nf4(experts, hidden_dim)
        if not isinstance(state, dict) or top_k_index.numel() > nf4_slot_limit(state["E"]):
            return None
        if mode == "grouped":
            # torch._grouped_mm takes bf16 only (its fake impl rejects fp16 under compile).
            from .moe_utils import _check_torch_grouped_mm_supported, moe_compute_dtype
            if moe_compute_dtype(hidden_states) != torch.bfloat16 or not _check_torch_grouped_mm_supported():
                return None
    else:
        # BF16 / FP16: nothing to dequantize, so the "grouped" comparator is the current path.
        if mode == "grouped":
            return None
        views = _bf16_views(experts, hidden_dim)
        if views is None or top_k_index.numel() > BF16_MAX_SLOTS:
            return None
    lora = _stash_lora(experts)
    if lora is False:
        return None
    gu_lora, dn_lora = lora
    if nf4:
        E, I2 = state["E"], state["gate_up"]["N"]
    else:
        E, I2 = views[0].shape[0], views[0].shape[2]
    # Each term must index this module's experts: first [E, in, r], second [E, r, out].
    for got, n_in, n_out in ((gu_lora, hidden_dim, I2), (dn_lora, I2 // 2, hidden_dim)):
        if got is not None and (tuple(got[0].shape[:2]) != (E, n_in) or got[1].shape[0] != E or got[1].shape[2] != n_out):
            return None
    from .moe_utils import moe_compute_dtype
    T, top_k = top_k_index.shape
    x = hidden_states.reshape(T, hidden_dim)
    rw = top_k_weights.reshape(T, top_k)
    if rw.stride(-1) != 1:
        rw = rw.contiguous()
    if nf4:
        x = x.to(moe_compute_dtype(hidden_states)).contiguous()
        out_dtype = hidden_states.dtype if hidden_states.dtype.is_floating_point else x.dtype
        fn = _nf4_grouped if mode == "grouped" else _nf4_routed
        state = _live_biases(experts, state)
        if state is None:
            return None
        out = fn(state, x, top_k_index, rw, top_k, gu_lora, dn_lora, out_dtype)
    else:
        w_gu, w_dn, act, interleaved = views
        out = routed_bf16_moe(
            x, top_k_index, rw, w_gu, w_dn,
            getattr(experts, "gate_up_proj_bias", None), getattr(experts, "down_proj_bias", None),
            act, interleaved,
            [gu_lora[:3]] if gu_lora is not None else (), [dn_lora[:3]] if dn_lora is not None else (),
            float(getattr(experts, "alpha", 1.702)), float(getattr(experts, "limit", 7.0)),
            out_dtype = hidden_states.dtype,
        )
    return out.view(hidden_states.shape[:-1] + (out.shape[-1],))


def _gate_up_interleaved(experts):
    from .moe_utils import _gate_up_is_interleaved
    return _gate_up_is_interleaved(experts)
