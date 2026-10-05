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
"""Grouped QLoRA training for gpt-oss bnb NF4 experts (ModuleList of Linear4bit): one
bnb-exact Triton dequant of all experts per projection, base via _base_grouped_mm, LoRA via
torch._grouped_mm over stacked per-expert A / B. No host sync, so an expert routed no tokens
gets zero LoRA grads (the loop leaves None). Unsupported setups return a reason string and
the caller keeps the per-expert loop."""

__all__ = [
    "nf4_dequant_expert_stack",
    "ready_signature",
    "expert_lora_state",
    "grouped_qlora_forward",
]

import os

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except Exception:  # pragma: no cover - no Triton / libdevice, no stacked dequant
    triton = None

_DISABLED_REASON = None
CALLS = {"forward": 0, "forward_lora": 0, "stacked_dequant": 0, "bnb_fallback_dequant": 0}


if triton is not None:

    @triton.jit
    def _nf4_dequant_stack_kernel(
        W_PTRS, A_PTRS, A2_PTRS, C2_PTRS, OFFSETS, LUT, OUT,
        n_bytes,
        BLOCKSIZE: tl.constexpr, BLOCKSIZE2: tl.constexpr, NESTED: tl.constexpr, BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        e = tl.program_id(1).to(tl.int64)
        W = tl.load(W_PTRS + e).to(tl.pointer_type(tl.uint8))
        offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        mask = offs < n_bytes
        qw = tl.load(W + offs, mask = mask, other = 0)
        blk = offs // (BLOCKSIZE // 2)
        if NESTED:
            A = tl.load(A_PTRS + e).to(tl.pointer_type(tl.uint8))
            A2 = tl.load(A2_PTRS + e).to(tl.pointer_type(tl.float32))
            C2 = tl.load(C2_PTRS + e).to(tl.pointer_type(tl.float32))
            aq = tl.load(A + blk, mask = mask, other = 0).to(tl.int32)
            am2 = tl.load(A2 + blk // BLOCKSIZE2, mask = mask, other = 0.0)
            # code2[q] * absmax2, then + offset: two roundings, as bitsandbytes does.
            am = libdevice.mul_rn(tl.load(C2 + aq, mask = mask, other = 0.0), am2) + tl.load(OFFSETS + e)
        else:
            A = tl.load(A_PTRS + e).to(tl.pointer_type(tl.float32))
            am = tl.load(A + blk, mask = mask, other = 0.0)
        # mul_rn (mul.rn.ftz.f32): no FMA contraction, subnormals flushed as bitsandbytes does.
        vh = libdevice.mul_rn(tl.load(LUT + (qw >> 4).to(tl.int32)), am).to(OUT.dtype.element_ty)
        vl = libdevice.mul_rn(tl.load(LUT + (qw & 15).to(tl.int32)), am).to(OUT.dtype.element_ty)
        w = tl.reshape(tl.join(vh, vl), (2 * BLOCK,))
        offs2 = pid * (2 * BLOCK) + tl.arange(0, 2 * BLOCK).to(tl.int64)
        tl.store(OUT + e * (2 * n_bytes) + offs2, w, mask = offs2 < 2 * n_bytes)


def _disable(exc):
    global _DISABLED_REASON
    _DISABLED_REASON = f"{type(exc).__name__}: {exc}"
    import logging
    logging.getLogger(__name__).warning(
        "Unsloth: the stacked NF4 expert dequant kernel failed and is disabled for this "
        f"process; bitsandbytes is used instead. Reason: {_DISABLED_REASON}"
    )


def stacked_dequant_available(device) -> bool:
    return (
        triton is not None
        and _DISABLED_REASON is None
        and os.environ.get("UNSLOTH_MOE_TRITON_KERNELS", "1") != "0"
        and getattr(device, "type", None) == "cuda"
        and torch.version.hip is None   # HIP has no libdevice mul_rn
        and os.environ.get("TRITON_INTERPRET", "0") != "1"
    )


@torch.compiler.disable
def nf4_dequant_expert_stack(tb, dtype, out = None):
    """All experts of one projection as [E, N, K] in `dtype`, from a gpt_oss_routed pointer table.

    Bit-identical to stacking bitsandbytes.functional.dequantize_4bit of each expert (the
    quant state dtype must equal `dtype`). None when unsupported, so callers fall back."""
    w_ptrs = tb["w"]
    device = w_ptrs.device
    if not stacked_dequant_available(device):
        return None
    E, N, K = int(w_ptrs.numel()), int(tb["N"]), int(tb["K"])
    n_bytes = N * K // 2
    blocksize = int(tb["blocksize"])
    if (N * K) % 2 != 0 or blocksize % 2 != 0 or tb["lut"].numel() != 16:
        return None
    if out is None:
        out = torch.empty((E, N, K), dtype = dtype, device = device)
    BLOCK = 1024
    try:
        with torch.cuda.device(device):
            _nf4_dequant_stack_kernel[(-(-n_bytes // BLOCK), E)](
                w_ptrs, tb["a"], tb["a2"], tb["c2"], tb["off"], tb["lut"], out, n_bytes,
                BLOCKSIZE = blocksize, BLOCKSIZE2 = int(tb["blocksize2"]), NESTED = bool(tb["nested"]),
                BLOCK = BLOCK, num_warps = 4, enable_fp_fusion = False,
            )
    except Exception as exc:
        from torch.utils import checkpoint
        if isinstance(exc, tuple(c for c in (getattr(checkpoint, "_StopRecomputationError", None),
                                             getattr(checkpoint, "CheckpointError", None)) if c is not None)):
            raise
        if isinstance(exc, torch.OutOfMemoryError):
            raise
        _disable(exc)
        return None
    CALLS["stacked_dequant"] += 1
    return out


class _StackProvider:
    """weight_provider for moe_utils._base_grouped_mm: () -> [E, K, N] view; .transposed() -> [E, N, K]."""

    __slots__ = ("tb", "dtype", "fallback")

    def __init__(self, tb, dtype, fallback):
        self.tb, self.dtype, self.fallback = tb, dtype, fallback

    def transposed(self):
        w = None
        if self.tb is not None and self.tb.get("dtype") == self.dtype:
            w = nf4_dequant_expert_stack(self.tb, self.dtype)
        if w is None:
            w = self.fallback()
        return w

    def __call__(self):
        return self.transposed().transpose(1, 2)


def _projs_lora(projs):
    """(name, scaling, rank) of the one active adapter on every expert of `projs`;
    None when no expert is LoRA-wrapped; a reason string when unsupported."""
    wrapped = [hasattr(p, "lora_A") for p in projs]
    if not any(wrapped):
        return None
    if not all(wrapped):
        return "some experts are LoRA-wrapped and some are not"
    first = projs[0]
    for h in getattr(first, "_forward_pre_hooks", {}).values():
        if "adapter_names" in getattr(h, "keywords", ()):
            return "mixed-adapter batch"
    for proj in projs:
        if getattr(proj, "disable_adapters", False):
            return "adapters disabled"
        if getattr(proj, "merged", False):
            return "adapter merged into the base weight"
    active = list(first.active_adapters)
    if len(active) != 1:
        return f"{len(active)} active adapters"
    name = active[0]
    scaling = None
    rank = None
    for proj in projs:
        if list(proj.active_adapters) != active or name not in proj.lora_A:
            return "adapters differ across experts"
        lora_A, lora_B = proj.lora_A[name], proj.lora_B[name]
        if not (isinstance(lora_A, torch.nn.Linear) and isinstance(lora_B, torch.nn.Linear)):
            return f"lora_A is {type(lora_A).__name__}"
        if proj.use_dora.get(name, False) or getattr(proj, "lora_variant", {}).get(name) is not None:
            return "DoRA / LoRA variant"
        if getattr(lora_B, "bias", None) is not None or getattr(lora_A, "bias", None) is not None:
            return "lora_bias"
        drop = proj.lora_dropout[name]
        if not isinstance(drop, torch.nn.Identity) and getattr(drop, "p", 0) > 0 and drop.training:
            return "lora_dropout > 0"
        s = proj.scaling[name]
        if not isinstance(s, (int, float)):
            return "non-scalar scaling"
        if scaling is None:
            scaling, rank = float(s), lora_A.weight.shape[0]
        elif float(s) != scaling or lora_A.weight.shape[0] != rank:
            return "scaling / rank differ across experts"
        if lora_A.weight.dtype != lora_B.weight.dtype:
            return "lora_A / lora_B dtypes differ"
    return (name, scaling, rank)


@torch.compiler.disable
def expert_lora_state(experts):
    """None (no LoRA), a reason string (unsupported), or {"gate_up": (...), "down": (...)}."""
    gu, dn = _projs_lora(experts.gate_up_projs), _projs_lora(experts.down_projs)
    if gu is None and dn is None:
        return None
    for v in (gu, dn):
        if isinstance(v, str):
            return v
    return {"gate_up": gu, "down": dn}


def _proj_signature(proj, out):
    # Read through __dict__ / _modules / _parameters: nn.Module.__getattr__ and Params4bit's
    # __torch_function__ would dominate the cost.
    d = proj.__dict__
    mods = d["_modules"]
    base = mods.get("base_layer", proj)
    w = base._parameters.get("weight")
    b = base._parameters.get("bias")
    # Tensors / QuantState by id: their __eq__ is elementwise, not identity.
    out += (proj, id(w), id(getattr(w, "quant_state", None)), w is not None and w.requires_grad,
            id(b), b is not None and b.requires_grad)
    if "lora_A" in mods:
        active = d.get("_active_adapter")
        out += (active if isinstance(active, str) else tuple(active), tuple(d.get("merged_adapters", ())),
                d.get("_disable_adapters"), tuple(mods["lora_A"]._modules.values()),
                tuple(mods["lora_B"]._modules.values()), tuple(d.get("use_dora", {}).items()),
                tuple(d.get("scaling", {}).items()), len(d.get("lora_variant") or ()),
                tuple(proj._forward_pre_hooks))
        for m in mods["lora_dropout"]._modules.values():
            out += (m, m.training, m.__dict__.get("p"))


@torch.compiler.disable
def ready_signature(experts):
    """Key for caching _grouped_bnb4bit_ready over every expert of both projections, so a
    change on any one expert (dropout, adapter switch, merge, reload) re-runs the full check.
    None when it cannot be built (no caching)."""
    out = []
    try:
        with torch._C.DisableTorchFunctionSubclass():
            for projs in (experts.gate_up_projs, experts.down_projs):
                out.append(len(projs))
                for p in projs:
                    _proj_signature(p, out)
    except Exception:
        return None
    return tuple(out)


def _lora_operands(projs, name, dtype):
    """[E, in, R'] and [E, R', out] in `dtype`, rank zero-padded by moe_utils'
    _pad_lora_rank_for_grouped_mm (torch._grouped_mm rejects ranks 4 / 6 in bf16)."""
    from unsloth_zoo.temporary_patches.moe_utils import _pad_lora_rank_for_grouped_mm
    A = torch.stack([p.lora_A[name].weight for p in projs])   # [E, r, in]
    B = torch.stack([p.lora_B[name].weight for p in projs])   # [E, out, r]
    A = A.to(dtype).transpose(1, 2)                           # [E, in, r]
    B = B.to(dtype).transpose(1, 2)                           # [E, r, out]
    A, B = _pad_lora_rank_for_grouped_mm(A, B)
    return A.contiguous(), B.contiguous()


def _lora_delta(x, offsets, projs, lora, dtype):
    from unsloth_zoo.temporary_patches.moe_utils import _grouped_mm_with_backward_fix
    name, scaling, _ = lora
    A, B = _lora_operands(projs, name, dtype)
    h = _grouped_mm_with_backward_fix(x, A, offsets)
    return _grouped_mm_with_backward_fix(h, B, offsets) * scaling


def _fallback_key(weight):
    qs = weight.quant_state
    key = (weight.data_ptr(), qs.absmax.data_ptr())
    if getattr(qs, "nested", False):
        offset = qs.offset
        offset = offset.data_ptr() if isinstance(offset, torch.Tensor) else offset
        key += (qs.state2.absmax.data_ptr(), qs.state2.code.data_ptr(), offset)
    return key


def _bnb_fallback_stack(projs, dtype):
    """Concat-and-dequant through bitsandbytes (the pre-existing grouped path)."""
    import bitsandbytes as bnb
    from bitsandbytes.functional import QuantState
    # Keyed on every expert's packed and (nested) absmax storage, so a reload or requantize rebuilds it.
    key = tuple(_fallback_key(getattr(p, "base_layer", p).weight) for p in projs)
    cached = getattr(projs, "_unsloth_grouped_cat_qs", None)
    cache = cached[1] if cached is not None and cached[0] == key else None
    if cache is None:
        states = [getattr(p, "base_layer", p).weight.quant_state for p in projs]
        absmax = []
        for qs in states:
            if getattr(qs, "nested", False):
                a = bnb.functional.dequantize_blockwise(qs.absmax, qs.state2)
                absmax.append((a + qs.offset).float())
            else:
                absmax.append(qs.absmax.float())
        q0 = states[0]
        cache = QuantState(
            absmax = torch.cat(absmax), shape = torch.Size((len(projs),) + tuple(q0.shape)),
            code = q0.code, blocksize = q0.blocksize, quant_type = q0.quant_type, dtype = q0.dtype,
        )
        projs._unsloth_grouped_cat_qs = (key, cache)
    CALLS["bnb_fallback_dequant"] += 1
    data = torch.cat([getattr(p, "base_layer", p).weight.data.reshape(-1, 1) for p in projs])
    return bnb.functional.dequantize_4bit(data, cache).to(dtype)


def _storage_key(experts):
    out = []
    for projs in (experts.gate_up_projs, experts.down_projs):
        for p in projs:
            weight = getattr(p, "base_layer", p).weight
            qs = weight.quant_state
            out.append(weight.data_ptr())
            out.append(qs.absmax.data_ptr())
            if getattr(qs, "nested", False):
                out.append(qs.state2.absmax.data_ptr())
    return tuple(out)


def _tables(experts, dtype):
    """Pointer tables for both projections, or None for the bnb fallback."""
    try:
        from unsloth_zoo.temporary_patches.gpt_oss_routed import prepare_routed_experts
        # prepare_routed_experts only re-checks the end experts; a replaced middle expert
        # would leave the kernel reading a freed buffer, so check every expert here.
        key = _storage_key(experts)
        state = getattr(experts, "_unsloth_routed_nf4", None)
        if isinstance(state, dict) and state.get("storage_key") != key:
            experts._unsloth_routed_nf4 = None
        state = prepare_routed_experts(experts)
    except Exception:
        return None
    if not isinstance(state, dict):
        return None
    state["storage_key"] = key
    for key in ("gate_up", "down"):
        tb = state[key]
        if "dtype" not in tb:
            projs = experts.gate_up_projs if key == "gate_up" else experts.down_projs
            tb["dtype"] = getattr(projs[0], "base_layer", projs[0]).weight.quant_state.dtype
    return state


# Opaque to Dynamo (data_ptr-keyed tables, raw Triton launch): one graph break per call.
@torch.compiler.disable
def grouped_qlora_forward(experts, hidden_states, router_indices, routing_weights,
                          batch_size, num_tokens, num_experts, top_k, lora = None):
    """Grouped training forward of GptOssExpertsBnb4bit, with or without per-expert LoRA.
    Returns fp32 [batch, seq, hidden], as the per-expert training loop does, or None when
    the experts do not compute in bf16 (the caller keeps the loop)."""
    from unsloth_zoo.temporary_patches.moe_utils import (
        _base_grouped_mm, _moe_recompute_default, count_tokens_per_expert,
    )
    from unsloth_zoo.temporary_patches.gpt_oss import swiglu_torch_forward

    device = hidden_states.device
    # Linear4bit computes in compute_dtype whatever the input dtype; _grouped_mm backward needs bf16.
    for proj in (*experts.gate_up_projs, *experts.down_projs):
        base = getattr(proj, "base_layer", proj)
        if (
            getattr(base, "compute_dtype", None) is not torch.bfloat16
            or getattr(base, "_pre_set_compute_dtype", None) not in (None, torch.bfloat16)
            or base.weight.quant_state.dtype is not torch.bfloat16
        ):
            return None
    dtype = torch.bfloat16
    if hidden_states.dtype not in (torch.bfloat16, torch.float32):
        return None
    # The loop's dtypes: gate_up in the input dtype, swiglu and the down sum in fp32.
    acc_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(dtype)
    CALLS["forward"] += 1
    if lora is not None:
        CALLS["forward_lora"] += 1
    with torch.no_grad():
        flat_experts = router_indices.flatten()
        token_ids = torch.arange(num_tokens, device = device).repeat_interleave(top_k)
        sorted_idx = flat_experts.argsort(stable = True)
        sorted_tokens = token_ids[sorted_idx]
        expert_ids = flat_experts[sorted_idx]
        counts = count_tokens_per_expert(flat_experts, num_experts, torch.int64)
        offsets = counts.cumsum(0, dtype = torch.int32)

    recompute = _moe_recompute_default()
    state = _tables(experts, dtype)
    gu_projs, dn_projs = experts.gate_up_projs, experts.down_projs
    # Read live every call, so in-place bias edits are never stale.
    gu_bias = torch.stack([getattr(p, "base_layer", p).bias for p in gu_projs]).detach()
    dn_bias = torch.stack([getattr(p, "base_layer", p).bias for p in dn_projs]).detach()
    gu_tb = state["gate_up"] if state is not None else None
    dn_tb = state["down"] if state is not None else None
    gu_w = _StackProvider(gu_tb, dtype, lambda: _bnb_fallback_stack(gu_projs, dtype))
    dn_w = _StackProvider(dn_tb, dtype, lambda: _bnb_fallback_stack(dn_projs, dtype))

    xg = hidden_states[sorted_tokens]
    gate_up = _base_grouped_mm(xg, offsets, gu_w, recompute).to(acc_dtype)
    gate_up = gate_up + gu_bias[expert_ids].to(acc_dtype)
    if lora is not None and lora["gate_up"] is not None:
        gate_up = gate_up + _lora_delta(xg, offsets, gu_projs, lora["gate_up"], dtype).to(acc_dtype)
    gated = swiglu_torch_forward(gate_up, experts.alpha, experts.limit, dtype = torch.float32).to(dtype)
    out = _base_grouped_mm(gated, offsets, dn_w, recompute).float()
    out = out + dn_bias[expert_ids].float()
    if lora is not None and lora["down"] is not None:
        out = out + _lora_delta(gated, offsets, dn_projs, lora["down"], dtype).float()

    weighted = out.to(torch.float32) * routing_weights[sorted_tokens, expert_ids, None].to(torch.float32)
    next_states = torch.zeros(num_tokens, experts.hidden_size, dtype = torch.float32, device = device)
    next_states.index_add_(0, sorted_tokens, weighted)
    return next_states.view(batch_size, -1, experts.hidden_size)
