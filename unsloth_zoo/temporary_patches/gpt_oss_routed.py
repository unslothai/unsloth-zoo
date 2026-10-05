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

"""Routed gpt-oss expert inference for bitsandbytes NF4 experts.

Only the experts the router picked are read, in place, through per-layer pointer tables
to each Linear4bit's packed bytes: no host sync, no stacked or dequantized copy, and a
fixed grid for a given shape, so it compiles fullgraph and captures in CUDA graphs.
Two launches per layer:
  gate_up: one GEMV per (token, slot) -> [P, 2I] fp32, bias added.
  down:    per token, swiglu on the fly, GEMV, bias, routing weight and the top-k sum,
           written once in the output dtype (fp32 accumulation, no atomics).
"""

__all__ = [
    "routed_experts_forward",
    "prepare_routed_experts",
    "routed_bf16_gemm",
    "routed_bf16_forward",
    "routed_bf16_eligible",
    "routed_mlp_forward",
]

import functools
import os
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - no Triton, no routed path
    triton = None


if triton is not None:

    @triton.jit
    def _nf4_dot(W, A, A2, C2, offset, LUT, rows64, rmask, x_chunk, k0, K,
                 BLOCKSIZE: tl.constexpr, BLOCKSIZE2: tl.constexpr, NESTED: tl.constexpr,
                 BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        # sum_k dequant(W[rows, k0:k0 + BLOCK_K]) * x_chunk, as bitsandbytes dequantizes.
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
    def _expert_tables(e, W_PTRS, A_PTRS, A2_PTRS, C2_PTRS, OFFSETS, NESTED: tl.constexpr):
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
    def _lora_b(acc, LORA_B, LORA_H, e, s, rows64, rmask, N, scaling, R: tl.constexpr, R_PAD: tl.constexpr):
        # acc += scaling * B[e][rows] @ H[s], H = A[e] @ x precomputed per slot.
        j = tl.arange(0, R_PAD)
        jmask = j < R
        b = tl.load(LORA_B + e * N * R + rows64[:, None] * R + j[None, :], mask = rmask[:, None] & jmask[None, :], other = 0.0)
        h = tl.load(LORA_H + s * R + j, mask = jmask, other = 0.0)
        return acc + tl.sum(b.to(tl.float32) * h.to(tl.float32)[None, :], axis = 1) * scaling

    @triton.jit
    def _routed_gate_up_kernel(
        X, IDX, W_PTRS, A_PTRS, A2_PTRS, C2_PTRS, OFFSETS, LUT, BIAS, LORA_B, LORA_H, OUT, N, K,
        alpha, limit, scaling,
        TOP_K: tl.constexpr, BLOCKSIZE: tl.constexpr, BLOCKSIZE2: tl.constexpr, NESTED: tl.constexpr,
        HAS_BIAS: tl.constexpr, HAS_LORA: tl.constexpr, R: tl.constexpr, R_PAD: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        # gu = dequant(W[IDX[p]]) @ X[p // TOP_K] + BIAS[IDX[p]] (+ LoRA), then gpt-oss swiglu over
        # its interleaved (gate, up) rows: OUT[p] is the [N // 2] fp32 expert intermediate.
        alpha = alpha.to(tl.float32)
        limit = limit.to(tl.float32)
        p = tl.program_id(0)
        e = tl.load(IDX + p).to(tl.int64)
        W, A, A2, C2, offset = _expert_tables(e, W_PTRS, A_PTRS, A2_PTRS, C2_PTRS, OFFSETS, NESTED)
        rows = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        rmask = rows < N
        rows64 = rows.to(tl.int64)
        ks = tl.arange(0, BLOCK_K)
        x_row = X + (p // TOP_K).to(tl.int64) * K
        acc = tl.zeros([BLOCK_N], dtype = tl.float32)
        for k0 in range(0, K, BLOCK_K):
            x = tl.load(x_row + k0 + ks, mask = (k0 + ks) < K, other = 0.0).to(tl.float32)
            acc += _nf4_dot(W, A, A2, C2, offset, LUT, rows64, rmask, x, k0, K,
                            BLOCKSIZE, BLOCKSIZE2, NESTED, BLOCK_N, BLOCK_K)
        if HAS_BIAS:
            acc += tl.load(BIAS + e * N + rows, mask = rmask, other = 0.0).to(tl.float32)
        if HAS_LORA:
            acc = _lora_b(acc, LORA_B, LORA_H, e, p.to(tl.int64), rows64, rmask, N, scaling, R, R_PAD)
        gate, up = tl.split(tl.reshape(acc, (BLOCK_N // 2, 2)))
        gate = tl.minimum(gate, limit)
        up = tl.minimum(tl.maximum(up, -limit), limit)
        inter = (up + 1.0) * (gate * tl.sigmoid(gate * alpha))
        half = tl.program_id(1) * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
        tl.store(OUT + p.to(tl.int64) * (N // 2) + half, inter, mask = half < N // 2)

    @triton.jit
    def _routed_down_kernel(
        X, IDX, RW, W_PTRS, A_PTRS, A2_PTRS, C2_PTRS, OFFSETS, LUT, BIAS, LORA_B, LORA_H, OUT, N, K,
        RW_STRIDE, scaling,
        TOP_K: tl.constexpr, DENSE_RW: tl.constexpr, BLOCKSIZE: tl.constexpr, BLOCKSIZE2: tl.constexpr,
        NESTED: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_LORA: tl.constexpr, R: tl.constexpr,
        R_PAD: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
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
            W, A, A2, C2, offset = _expert_tables(e, W_PTRS, A_PTRS, A2_PTRS, C2_PTRS, OFFSETS, NESTED)
            acc = tl.zeros([BLOCK_N], dtype = tl.float32)
            for k0 in range(0, K, BLOCK_K):
                x = tl.load(X + s * K + k0 + ks, mask = (k0 + ks) < K, other = 0.0).to(tl.float32)
                acc += _nf4_dot(W, A, A2, C2, offset, LUT, rows64, rmask, x, k0, K,
                                BLOCKSIZE, BLOCKSIZE2, NESTED, BLOCK_N, BLOCK_K)
            if HAS_BIAS:
                acc += tl.load(BIAS + e * N + rows, mask = rmask, other = 0.0).to(tl.float32)
            if HAS_LORA:
                acc = _lora_b(acc, LORA_B, LORA_H, e, s, rows64, rmask, N, scaling, R, R_PAD)
            out += rw * acc
        tl.store(OUT + t.to(tl.int64) * N + rows, out.to(OUT.dtype.element_ty), mask = rmask)


def _routed_disabled():
    # UNSLOTH_GPTOSS_ROUTED_INFERENCE is the earlier name of the same switch.
    return "0" in (
        os.environ.get("UNSLOTH_GPTOSS_ROUTED_KERNEL", "1"),
        os.environ.get("UNSLOTH_GPTOSS_ROUTED_INFERENCE", "1"),
    )


def _block_k(K, blocksize):
    return max(blocksize, min(1024, triton.next_power_of_2(K)))


def _lora_args(lora, dummy):
    # (B stack [E, N, r], H [P, r] fp32, scaling, r) or dummies.
    if lora is None:
        return dummy, dummy, 0.0, 1
    B, H, scaling = lora
    return B, H, float(scaling), B.shape[-1]


def _gate_up(kernel, x, idx, tb, top_k, lora, alpha, limit, out):
    B, H, scaling, r = _lora_args(lora, tb["lut"])
    kernel[(idx.numel(), triton.cdiv(tb["N"], 4))](
        x, idx, tb["w"], tb["a"], tb["a2"], tb["c2"], tb["off"], tb["lut"],
        tb["bias"] if tb["bias"] is not None else tb["lut"], B, H, out, tb["N"], tb["K"],
        float(alpha), float(limit), scaling,
        TOP_K = top_k, BLOCKSIZE = tb["blocksize"], BLOCKSIZE2 = tb["blocksize2"], NESTED = tb["nested"],
        HAS_BIAS = tb["bias"] is not None, HAS_LORA = lora is not None, R = r, R_PAD = triton.next_power_of_2(r),
        BLOCK_N = 4, BLOCK_K = _block_k(tb["K"], tb["blocksize"]), num_warps = 4,
    )
    return out


def _down(kernel, x, idx, rw, dense_rw, tb, top_k, lora, out):
    B, H, scaling, r = _lora_args(lora, tb["lut"])
    kernel[(out.shape[0], triton.cdiv(tb["N"], 4))](
        x, idx, rw, tb["w"], tb["a"], tb["a2"], tb["c2"], tb["off"], tb["lut"],
        tb["bias"] if tb["bias"] is not None else tb["lut"], B, H, out, tb["N"], tb["K"],
        rw.stride(0), scaling,
        TOP_K = top_k, DENSE_RW = dense_rw, BLOCKSIZE = tb["blocksize"], BLOCKSIZE2 = tb["blocksize2"],
        NESTED = tb["nested"], HAS_BIAS = tb["bias"] is not None, HAS_LORA = lora is not None, R = r,
        R_PAD = triton.next_power_of_2(r), BLOCK_N = 4, BLOCK_K = _block_k(tb["K"], tb["blocksize"]),
        num_warps = 4,
    )
    return out


def _tb(w, a, a2, c2, off, lut, bias, N, K, blocksize, blocksize2, nested):
    return dict(w = w, a = a, a2 = a2, c2 = c2, off = off, lut = lut, bias = bias, N = N, K = K,
                blocksize = blocksize, blocksize2 = blocksize2, nested = nested)


if triton is not None:
    # custom_op, not triton_op: Inductor before torch 2.12 dropped the gate_up kernel from the
    # graph (its output reached down uninitialized). Opaque calls stay fullgraph and graph safe.

    @torch.library.custom_op("unsloth_zoo::routed_nf4_gate_up", mutates_args = ())
    def _gate_up_op(
        x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor, a: torch.Tensor, a2: torch.Tensor,
        c2: torch.Tensor, off: torch.Tensor, lut: torch.Tensor, bias: Optional[torch.Tensor],
        N: int, K: int, blocksize: int, blocksize2: int, nested: bool, top_k: int,
        lora_b: Optional[torch.Tensor], lora_h: Optional[torch.Tensor], scaling: float,
        alpha: float, limit: float,
    ) -> torch.Tensor:
        tb = _tb(w, a, a2, c2, off, lut, bias, N, K, blocksize, blocksize2, nested)
        lora = None if lora_b is None else (lora_b, lora_h, scaling)
        out = torch.empty((idx.numel(), N // 2), dtype = torch.float32, device = x.device)
        return _gate_up(_routed_gate_up_kernel, x, idx, tb, top_k, lora, alpha, limit, out)

    @_gate_up_op.register_fake
    def _(x, idx, w, a, a2, c2, off, lut, bias, N, K, blocksize, blocksize2, nested, top_k,
          lora_b, lora_h, scaling, alpha, limit):
        return x.new_empty((idx.numel(), N // 2), dtype = torch.float32)

    @torch.library.custom_op("unsloth_zoo::routed_nf4_down", mutates_args = ())
    def _down_op(
        x: torch.Tensor, idx: torch.Tensor, rw: torch.Tensor, dense_rw: bool, w: torch.Tensor,
        a: torch.Tensor, a2: torch.Tensor, c2: torch.Tensor, off: torch.Tensor, lut: torch.Tensor,
        bias: Optional[torch.Tensor], N: int, K: int, blocksize: int, blocksize2: int, nested: bool,
        top_k: int, lora_b: Optional[torch.Tensor], lora_h: Optional[torch.Tensor], scaling: float,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        tb = _tb(w, a, a2, c2, off, lut, bias, N, K, blocksize, blocksize2, nested)
        lora = None if lora_b is None else (lora_b, lora_h, scaling)
        out = torch.empty((idx.numel() // top_k, N), dtype = out_dtype, device = x.device)
        return _down(_routed_down_kernel, x, idx, rw, dense_rw, tb, top_k, lora, out)

    @_down_op.register_fake
    def _(x, idx, rw, dense_rw, w, a, a2, c2, off, lut, bias, N, K, blocksize, blocksize2, nested,
          top_k, lora_b, lora_h, scaling, out_dtype):
        return x.new_empty((idx.numel() // top_k, N), dtype = out_dtype)


def _table_args(tb):
    return (tb["w"], tb["a"], tb["a2"], tb["c2"], tb["off"], tb["lut"], tb["bias"],
            tb["N"], tb["K"], tb["blocksize"], tb["blocksize2"], tb["nested"])


def routed_gate_up(x, idx, tb, top_k, alpha, limit, lora = None):
    """[T * top_k, N // 2] fp32: swiglu(dequant(W[idx[p]]) @ x[p // top_k] + bias[idx[p]] (+ LoRA))."""
    if torch.compiler.is_compiling():
        b, h, scaling = lora if lora is not None else (None, None, 0.0)
        return torch.ops.unsloth_zoo.routed_nf4_gate_up(
            x, idx, *_table_args(tb), top_k, b, h, float(scaling), float(alpha), float(limit))
    # Eager launches directly: the dispatcher costs tens of us per call at decode sizes.
    out = torch.empty((idx.numel(), tb["N"] // 2), dtype = torch.float32, device = x.device)
    return _gate_up(_routed_gate_up_kernel, x, idx, tb, top_k, lora, alpha, limit, out)


def routed_down(x, idx, rw, dense_rw, tb, top_k, out_dtype, lora = None):
    """[T, N]: sum over the top-k slots of rw * (dequant(W[e]) @ x[slot] + bias[e] (+ LoRA))."""
    if torch.compiler.is_compiling():
        b, h, scaling = lora if lora is not None else (None, None, 0.0)
        return torch.ops.unsloth_zoo.routed_nf4_down(
            x, idx, rw, dense_rw, *_table_args(tb), top_k, b, h, float(scaling), out_dtype)
    out = torch.empty((idx.numel() // top_k, tb["N"]), dtype = out_dtype, device = x.device)
    return _down(_routed_down_kernel, x, idx, rw, dense_rw, tb, top_k, lora, out)


def _base_linear4bit(module):
    """The bitsandbytes Linear4bit under an optional PEFT LoRA wrapper, else None."""
    base = getattr(module, "base_layer", module)
    weight = getattr(base, "weight", None)
    if weight is None or getattr(weight, "quant_state", None) is None:
        return None
    return base


def _build_table(projs, device):
    states, w, a, a2, c2, off, biases = [], [], [], [], [], [], []
    for proj in projs:
        base = _base_linear4bit(proj)
        if base is None:
            return None
        qs = base.weight.quant_state
        if qs.quant_type != "nf4" or base.weight.device != device:
            return None
        states.append(qs)
        w.append(base.weight.data_ptr())
        if qs.nested:
            a.append(qs.absmax.data_ptr())
            a2.append(qs.state2.absmax.data_ptr())
            c2.append(qs.state2.code.data_ptr())
            off.append(float(qs.offset))
        else:
            if qs.absmax.dtype != torch.float32:
                return None
            a.append(qs.absmax.data_ptr())
            a2.append(0)
            c2.append(0)
            off.append(0.0)
        biases.append(base.bias)
    q0 = states[0]
    N, K = int(q0.shape[0]), int(q0.shape[1])
    nested = bool(q0.nested)
    for qs in states:
        if (
            tuple(qs.shape) != (N, K) or qs.blocksize != q0.blocksize or bool(qs.nested) != nested
            or (nested and qs.state2.blocksize != q0.state2.blocksize)
            or not torch.equal(qs.code, q0.code)
        ):
            return None
    if K % q0.blocksize != 0 or q0.blocksize not in (32, 64, 128, 256, 512, 1024):
        return None
    if all(b is None for b in biases):
        bias = None
    elif any(b is None or b.requires_grad for b in biases):
        # The stacked bias is a one-time copy; a trainable bias would go stale.
        return None
    else:
        bias = torch.stack([b.detach() for b in biases]).contiguous()
    as_i64 = lambda v: torch.tensor(v, dtype = torch.int64, device = device)
    return {
        "w": as_i64(w), "a": as_i64(a), "a2": as_i64(a2), "c2": as_i64(c2),
        "off": torch.tensor(off, dtype = torch.float32, device = device),
        "lut": q0.code.to(device = device, dtype = torch.float32).contiguous(),
        "bias": bias, "N": N, "K": K, "blocksize": int(q0.blocksize),
        "blocksize2": int(q0.state2.blocksize) if nested else 1, "nested": nested,
    }


def _key(projs):
    # A moved, reloaded or replaced expert changes its packed buffer's address. Checking the
    # first and last of each list keeps the per-call cost to a few us.
    return tuple(_base_linear4bit(p).weight.data_ptr() for p in (projs[0], projs[-1]))


def prepare_routed_experts(experts):
    """Build (or validate) the pointer tables for a ModuleList NF4 experts module.

    Eager only (data_ptr is not traceable); returns None when the layout is unsupported."""
    gate_up_projs = getattr(experts, "gate_up_projs", None)
    down_projs = getattr(experts, "down_projs", None)
    if gate_up_projs is None or down_projs is None or len(gate_up_projs) == 0:
        return None
    state = getattr(experts, "_unsloth_routed_nf4", None)
    if state is False:
        return None
    try:
        key = (_key(gate_up_projs), _key(down_projs))
    except Exception:
        return None
    if state is not None and state["key"] == key:
        return state
    device = _base_linear4bit(gate_up_projs[0]).weight.device
    if triton is None or device.type != "cuda":
        experts._unsloth_routed_nf4 = False
        return None
    gate_up, down = _build_table(gate_up_projs, device), _build_table(down_projs, device)
    if gate_up is None or down is None or down["K"] * 2 != gate_up["N"] or gate_up["K"] != down["N"]:
        experts._unsloth_routed_nf4 = False
        return None
    state = {"gate_up": gate_up, "down": down, "key": key}
    experts._unsloth_routed_nf4 = state
    return state


def _lora(projs):
    """(A [E, r, in], B [E, out, r], scaling) of the single active adapter; None without LoRA
    (or adapters disabled); False for a LoRA setup this path does not cover."""
    first = projs[0]
    if not hasattr(first, "lora_A"):
        return None
    if first.disable_adapters:
        return None
    active = first.active_adapters
    if len(active) != 1:
        return False if len(active) > 1 else None
    name = active[0]
    if first.merged or (first.training and getattr(first.lora_dropout[name], "p", 0) > 0):
        return False
    cache = getattr(projs, "_unsloth_routed_lora", None)
    if cache is None or cache["name"] != name or cache["first"] is not first.lora_A[name].weight:
        for proj in projs:
            if (
                name not in proj.lora_A or proj.scaling[name] != first.scaling[name]
                or proj.use_dora.get(name, False) or getattr(proj, "lora_variant", {}).get(name) is not None
            ):
                return False
        cache = {"name": name, "first": first.lora_A[name].weight, "versions": None,
                 "A": [proj.lora_A[name].weight for proj in projs],
                 "B": [proj.lora_B[name].weight for proj in projs]}
        projs._unsloth_routed_lora = cache
    if torch.compiler.is_compiling():
        return torch.stack(cache["A"]), torch.stack(cache["B"]), first.scaling[name]
    # Restack only after an optimizer step (or any in-place edit) bumps a version counter.
    versions = tuple(p._version for p in cache["A"]) + tuple(p._version for p in cache["B"])
    if versions != cache["versions"]:
        cache["stacked"] = (torch.stack(cache["A"]).contiguous(), torch.stack(cache["B"]).contiguous())
        cache["versions"] = versions
    return cache["stacked"][0], cache["stacked"][1], first.scaling[name]


def _lora_h(A, idx, x_slots):
    # H[p] = A[idx[p]] @ x_slots[p], [P, r] fp32.
    return torch.bmm(A[idx].float(), x_slots.float().unsqueeze(-1)).squeeze(-1)


def routed_experts_forward(experts, hidden_states, router_indices, routing_weights):
    """Routed eval forward for a ModuleList NF4 gpt-oss experts module, or None if ineligible.

    routing_weights is dense [T, E] (zeros off the picked experts) or already [T, top_k]."""
    if _routed_disabled():
        return None
    if torch.is_grad_enabled() or experts.training or not hidden_states.is_cuda:
        return None
    if torch.compiler.is_compiling():
        state = getattr(experts, "_unsloth_routed_nf4", None)
    else:
        state = prepare_routed_experts(experts)
    if not state:
        return None
    gu_lora, dn_lora = _lora(experts.gate_up_projs), _lora(experts.down_projs)
    if gu_lora is False or dn_lora is False:
        return None

    shape = hidden_states.shape
    x = hidden_states.reshape(-1, state["gate_up"]["K"]).contiguous()
    top_k = router_indices.shape[-1]
    idx = router_indices.reshape(-1)
    rw = routing_weights.reshape(x.shape[0], -1).contiguous()
    dense_rw = rw.shape[-1] == len(experts.gate_up_projs)  # [T, E] dense, else [T, top_k]

    if gu_lora is not None:
        A, B, scaling = gu_lora
        gu_lora = (B, _lora_h(A, idx, x.repeat_interleave(top_k, 0)), scaling)
    inter = routed_gate_up(x, idx, state["gate_up"], top_k, experts.alpha, experts.limit, gu_lora)
    if dn_lora is not None:
        A, B, scaling = dn_lora
        dn_lora = (B, _lora_h(A, idx, inter), scaling)
    out = routed_down(inter, idx, rw, dense_rw, state["down"], top_k, hidden_states.dtype, dn_lora)
    return out.view(shape[:-1] + (out.shape[-1],))


# BF16 / FP16 experts: 3D [E, K, N] weights, one program per (expert, N block, K split).
if triton is not None:

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
        # One program per (expert, n block, k split): reads the expert's weight tile once and
        # multiplies it with every slot routed to that expert (masked dot over all P slots).
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


def _launch_expert(kernel, x, idx, weight, bias, out, row_div):
    S, P, N = out.shape
    E, K = weight.shape[0], weight.shape[1]
    c = _expert_config(P)
    grid = (E, triton.cdiv(N, c["BLOCK_N"]), S)
    kernel[grid](
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

    @torch.library.triton_op("unsloth_zoo::routed_bf16_gemm", mutates_args = ())
    def _routed_bf16_gemm_op(
        x: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, row_div: int,
    ) -> torch.Tensor:
        part = torch.empty(_expert_config(idx.numel())["SPLIT_K"], idx.numel(), weight.shape[2], dtype = torch.float32, device = x.device)
        return _launch_expert(torch.library.wrap_triton(_routed_bf16_expert_kernel), x, idx, weight, bias, part, row_div)


def routed_bf16_gemm(x, idx, weight, bias = None, row_div = 1):
    """fp32 [P, N]: row p = x[p // row_div] @ weight[idx[p]] + bias[idx[p]].
    x: [R, K]; idx: [P] expert ids (on device); weight: [E, K, N] (any strides); bias: [E, N] or None.
    Each active expert's weights are read once; no host sync, so it compiles and captures."""
    idx = idx.reshape(-1)
    if x.stride(-1) != 1:
        x = x.contiguous()
    if bias is None:
        bias = weight.new_zeros(weight.shape[0], weight.shape[2])
    if torch.compiler.is_compiling():
        part = _routed_bf16_gemm_op(x, idx, weight, bias, row_div)
    else:
        part = torch.empty(_expert_config(idx.numel())["SPLIT_K"], idx.numel(), weight.shape[2], dtype = torch.float32, device = x.device)
        _launch_expert(_routed_bf16_expert_kernel, x, idx, weight, bias, part, row_div)
    # Fixed-order sum of the k-split partials: deterministic.
    return part.sum(0) if part.shape[0] > 1 else part[0]


# Above this many (token, expert) slots most experts are active and the dense path is as fast.
ROUTED_MAX_SLOTS = 64


def _expert_weight_3d(p):
    """[E, K, N] view of a gpt-oss expert weight: a 3D Parameter, or zoo's ParameterModule
    (2D storage for PEFT) viewed without the copy get_param() makes."""
    if hasattr(p, "get_param") and hasattr(p, "shape_3d"):
        unflat = [p.shape_3d[i] for i in p.permute_to_2d]
        return p.weight.view(*unflat).permute(*p.permute_to_3d)
    if hasattr(p, "weight") and not isinstance(p, torch.Tensor):
        return p.weight
    return p


def _lora_terms(experts):
    """(base experts module, {param_name: [(first [E,in,R], second [E,R,out], scaling), ...]}),
    or None when an adapter state is not representable here (caller falls back)."""
    from .moe_utils import _extract_lora_from_wrapper
    terms = {}
    m = experts
    while hasattr(m, "base_layer"):
        name = getattr(m, "parameter_name", None)
        if hasattr(m, "lora_A") and name is not None:
            if getattr(m, "disable_adapters", False):
                if getattr(m, "merged", False):
                    return None  # PEFT would unmerge first
            elif not getattr(m, "merged", False):
                for adapter in getattr(m, "active_adapters", []):
                    if adapter not in m.lora_A:
                        continue
                    got = _extract_lora_from_wrapper(m, adapter, experts_module = None)
                    if got is None:
                        return None
                    terms.setdefault(name, []).append((got[0], got[1], got[2]))
        elif hasattr(m, "lora_A"):
            return None
        m = m.base_layer
    return m, terms


def _lora_delta(x_slots, idx, terms):
    delta = None
    for first, second, scaling in terms:
        h = torch.bmm(x_slots.to(first.dtype)[:, None, :], first[idx])
        d = torch.bmm(h, second[idx])[:, 0].float() * scaling
        delta = d if delta is None else delta + d
    return delta


@functools.lru_cache(maxsize = None)
def _bf16_supported(device_index):
    major, _ = torch.cuda.get_device_capability(device_index)
    return major >= 8


def routed_bf16_eligible(experts, hidden_states):
    """True when routed_bf16_forward can run this call exactly."""
    if triton is None or not hidden_states.is_cuda or torch.is_grad_enabled():
        return False
    if _routed_disabled():
        return False
    got = _lora_terms(experts)
    if got is None:
        return False
    base, _ = got
    for name in ("gate_up_proj", "down_proj"):
        w = _expert_weight_3d(getattr(base, name, None))
        # fp32 weights would hit tl.dot's TF32 path; bf16 needs hardware support (not T4).
        if not isinstance(w, torch.Tensor) or w.dim() != 3 or w.dtype not in (torch.bfloat16, torch.float16):
            return False
        if w.dtype == torch.bfloat16 and not _bf16_supported(w.device.index or 0):
            return False
    return True


def routed_bf16_forward(experts, hidden_states, router_indices, routing_weights):
    """gpt-oss experts on only the routed experts. routing_weights: [T, E] (dense, zero off
    the picked experts) or [T, top_k]; returns [B, S, H] in hidden_states.dtype."""
    base, lora = _lora_terms(experts)
    shape = hidden_states.shape
    H = base.hidden_size if hasattr(base, "hidden_size") else shape[-1]
    x = hidden_states.reshape(-1, H)
    top_k = router_indices.shape[-1]
    T = router_indices.numel() // top_k
    idx = router_indices.reshape(T, top_k)
    # Dense [.., E] (zero off the picked experts) or top-k [.., top_k], any leading dims.
    routing_weights = routing_weights.reshape(T, -1)
    if routing_weights.shape[-1] != top_k:
        routing_weights = routing_weights.gather(1, idx)
    flat = idx.reshape(-1)
    w_gu = _expert_weight_3d(base.gate_up_proj)
    w_dn = _expert_weight_3d(base.down_proj)
    gate_up = routed_bf16_gemm(x.to(w_gu.dtype), flat, w_gu, base.gate_up_proj_bias, row_div = top_k)
    if "gate_up_proj" in lora:
        gate_up = gate_up + _lora_delta(x.repeat_interleave(top_k, 0), flat, lora["gate_up_proj"])
    limit, alpha = getattr(base, "limit", 7.0), getattr(base, "alpha", 1.702)
    gate = gate_up[:, ::2].clamp(max = limit)
    up = gate_up[:, 1::2].clamp(min = -limit, max = limit)
    act = (up + 1) * (gate * torch.sigmoid(gate * alpha))
    down = routed_bf16_gemm(act.to(w_dn.dtype), flat, w_dn, base.down_proj_bias, row_div = 1)
    if "down_proj" in lora:
        down = down + _lora_delta(act, flat, lora["down_proj"])
    out = (down.view(T, top_k, -1) * routing_weights.float()[..., None]).sum(1)
    return out.view(shape).to(hidden_states.dtype)


def routed_mlp_forward(mlp, hidden_states, max_slots = ROUTED_MAX_SLOTS):
    """gpt-oss MLP output from the router and only the routed experts, or None for calls larger
    than max_slots (token, expert) pairs or experts this path does not cover."""
    if triton is None or torch.is_grad_enabled() or not hidden_states.is_cuda:
        return None
    if _routed_disabled():
        return None
    experts = mlp.experts
    nf4 = False
    if hasattr(experts, "gate_up_projs"):
        # Pointer tables are built eagerly (data_ptr is not traceable), also on calls too large to
        # route, so a compiled decode step that follows an eager prefill finds them ready.
        state = getattr(experts, "_unsloth_routed_nf4", None) if torch.compiler.is_compiling() else prepare_routed_experts(experts)
        nf4 = bool(state)
    if hidden_states.numel() // hidden_states.shape[-1] * getattr(mlp.router, "top_k", 4) > max_slots:
        return None
    if not nf4 and not routed_bf16_eligible(experts, hidden_states):
        return None
    # The router takes [tokens, hidden]: transformers 5 normalizes the top-k scores with
    # softmax(dim=1), which on a [batch, seq, k] input would run over the sequence instead.
    router_out = mlp.router(hidden_states.reshape(-1, hidden_states.shape[-1]))
    scores, indices = router_out[-2], router_out[-1]
    if nf4:
        out = routed_experts_forward(experts, hidden_states, indices, scores)
        # An adapter setup the NF4 kernels do not cover: the module forward, same router output.
        return out if out is not None else experts(hidden_states, router_indices = indices, routing_weights = scores)
    return routed_bf16_forward(experts, hidden_states, indices, scores)
