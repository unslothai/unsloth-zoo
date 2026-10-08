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
the caller keeps the per-expert loop.

float16 (T4): the loader computes the down experts in fp32 (_pre_set_compute_dtype), since
their outputs overflow fp16. torch._grouped_mm has no fp32 output for fp16 operands, so this
path runs moe_grouped_fp16's Triton grouped GEMMs: fp16 operands, fp32 accumulate, the
loop's output dtype per projection (gate_up fp16, down fp32), LoRA in the loop's dtypes.

Stacked LoRA: the loader entry point (stack_expert_lora) gives each projection list one lora_A /
lora_B Parameter (moe_grouped_modulelist._StackedLoraLinear) when a grouped path trains it; both
paths read the stacks. UNSLOTH_MOE_STACKED_LORA=0 keeps PEFT's per-expert Parameters."""

__all__ = [
    "nf4_dequant_expert_stack",
    "ready_signature",
    "cached_ready",
    "ready_record",
    "expert_lora_state",
    "grouped_qlora_forward",
    "stack_expert_lora",
]

import os

import torch

from . import moe_ready_epoch as _ready_epoch
# Model-agnostic expert-LoRA helpers, shared with the ModuleList grouped MoE path.
from .moe_grouped_modulelist import (
    _STACK_NAME, _lora_delta, _lora_operands, _lora_stacks, _projs_lora, _stack_projs_lora, _stackable_lora,
)

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except Exception:  # pragma: no cover - no Triton / libdevice, no stacked dequant
    triton = None

_DISABLED_REASON = None
CALLS = {"forward": 0, "forward_lora": 0, "stacked_dequant": 0, "bnb_fallback_dequant": 0,
         "forward_fp16": 0, "forward_fp16_lora": 0, "declined": 0}
# Last reason the grouped forward handed a layer back to the per-expert loop.
LAST_DECLINE = {"reason": None}


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
    # Read through __dict__ / _modules / _parameters: __getattr__ / __torch_function__ dominate otherwise.
    d = proj.__dict__
    mods = d["_modules"]
    base = mods.get("base_layer", proj)
    w = base._parameters.get("weight")
    b = base._parameters.get("bias")
    # Tensors / QuantState by id: their __eq__ is elementwise, not identity.
    out += (proj, id(w), id(getattr(w, "quant_state", None)), w is not None and w.requires_grad,
            id(b), b is not None and b.requires_grad, "_hf_hook" in d or "_hf_hook" in base.__dict__)
    if "lora_A" in mods:
        active = d.get("_active_adapter")
        out += (active if isinstance(active, str) else tuple(active), tuple(d.get("merged_adapters", ())),
                d.get("_disable_adapters"), tuple(mods["lora_A"]._modules.values()),
                tuple(mods["lora_B"]._modules.values()), tuple(d.get("use_dora", {}).items()),
                tuple(d.get("scaling", {}).items()), len(d.get("lora_variant") or ()),
                tuple(proj._forward_pre_hooks))
        for m in mods["lora_dropout"]._modules.values():
            out += (m, m.training, m.__dict__.get("p"))
        # Readiness reads the adapter weights' dtype / rank: a stack (stacked LoRA) or a
        # per-expert weight, whose `.data =` / `.to()` keeps the Parameter's identity.
        for lm in (*mods["lora_A"]._modules.values(), *mods["lora_B"]._modules.values()):
            mp = lm._parameters
            lw = mp.get("weight", mp.get(_STACK_NAME))
            out += (id(lw), getattr(lw, "shape", None), getattr(lw, "dtype", None), getattr(lw, "device", None))


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
                    # Readiness validates the quant format, so quant edits must re-run it too.
                    out.append(_quant_key(getattr(p, "base_layer", p).weight))
    except Exception:
        return None
    return tuple(out)


def _tensor_key(t):
    # Storage and version: a replaced tensor or an in-place edit both change it.
    return (t.data_ptr(), t._version) if isinstance(t, torch.Tensor) else t


def _quant_key(weight):
    qs = weight.quant_state
    key = (weight.data_ptr(), _tensor_key(qs.absmax), _tensor_key(qs.code))
    if getattr(qs, "nested", False):
        key += (_tensor_key(qs.state2.absmax), _tensor_key(qs.state2.code), _tensor_key(qs.offset))
    return key


def _storage_lean(experts, refs = None):
    """moe_ready_epoch's lean key: per expert projection, the base weight's identity, address and
    requires_grad, its quant state's identity and absmax address, and the bias' identity and
    requires_grad. Re-read once per optimizer step; None when it cannot be built."""
    ptr = torch._C.TensorBase.data_ptr
    out = []
    try:
        with torch._C.DisableTorchFunctionSubclass():
            for projs in (experts._modules["gate_up_projs"], experts._modules["down_projs"]):
                for p in projs._modules.values():
                    base = p._modules.get("base_layer", p)
                    bp = base._parameters
                    w, b = bp["weight"], bp.get("bias")
                    qs = w.__dict__.get("quant_state")
                    out += (id(w), ptr(w), w.requires_grad, id(qs), None if qs is None else ptr(qs.absmax),
                            id(b), b is not None and b.requires_grad)
                    if refs is not None:
                        refs.append((w, qs, b))
    except Exception:
        return None
    return tuple(out)


def _spot_signature(experts):
    """ready_signature's fields over the first and last expert of each projection list."""
    out = []
    try:
        with torch._C.DisableTorchFunctionSubclass():
            for projs in (experts._modules["gate_up_projs"], experts._modules["down_projs"]):
                out += (projs, len(projs))
                for p in (projs[0], projs[-1]):
                    _proj_signature(p, out)
                    out.append(_quant_key(getattr(p, "base_layer", p).weight))
    except Exception:
        return None
    return tuple(out)


@torch.compiler.disable
def cached_ready(experts):
    """(verdict, lora) of the last full _grouped_bnb4bit_ready check while moe_ready_epoch vouches
    for it (no expert-state change, end experts unchanged), else None."""
    if not _ready_epoch.enabled():
        experts.__dict__.pop(_ready_epoch.VALID, None)
        return None
    cached = experts.__dict__.get("_unsloth_grouped_ready")
    rec = cached[3] if cached is not None and len(cached) > 3 else None
    if _ready_epoch.valid(rec, (experts,), lambda: _spot_signature(experts), lambda: _storage_lean(experts)):
        _ready_epoch.mark_valid(experts, rec)
        return cached[1], cached[2]
    experts.__dict__.pop(_ready_epoch.VALID, None)   # the tables re-key until a full check vouches again
    return None


@torch.compiler.disable
def ready_record(experts):
    """A moe_ready_epoch.Record for the full check just run, or None (switch off, accelerate hooks)."""
    if not _ready_epoch.enabled():
        return None
    _ready_epoch.COUNTS["full"] += 1
    rec = None
    hooked, drops = _ready_epoch.scan(experts)
    if not hooked:
        refs = []
        spot, lean = _spot_signature(experts), _storage_lean(experts, refs)
        if spot is not None and lean is not None:
            _ready_epoch.wrap_peft()   # PEFT tuner classes imported since the last full check
            rec = _ready_epoch.Record((experts,), spot, lean, drops, refs)
    if rec is not None:
        _ready_epoch.mark_valid(experts, rec)
    else:
        experts.__dict__.pop(_ready_epoch.VALID, None)
    return rec


def _bnb_fallback_stack(projs, dtype):
    """Concat-and-dequant through bitsandbytes (the pre-existing grouped path)."""
    import bitsandbytes as bnb
    from bitsandbytes.functional import QuantState
    # Keyed on every expert's quant tensors (storage and version), so any requantize or edit rebuilds it.
    key = tuple(_quant_key(getattr(p, "base_layer", p).weight) for p in projs)
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
    return tuple(
        _quant_key(getattr(p, "base_layer", p).weight)
        for projs in (experts.gate_up_projs, experts.down_projs)
        for p in projs
    )


def _table_spot(experts):
    out = []
    with torch._C.DisableTorchFunctionSubclass():
        for projs in (experts._modules["gate_up_projs"], experts._modules["down_projs"]):
            for p in (projs[0], projs[-1]):
                w = getattr(p, "base_layer", p).weight
                out += (id(w), id(w.quant_state), _quant_key(w))
    return tuple(out)


def _tables(experts, dtype):
    """Pointer tables for both projections, or None for the bnb fallback. While readiness vouches
    for the experts' storage at the current moe_ready_epoch stamp, the O(E) storage key and the
    snapshot check are skipped for a spot key over the end experts."""
    gen = _ready_epoch.current_gen(experts) if _ready_epoch.enabled() else None
    spot = None
    if gen is not None:
        try:
            spot = _table_spot(experts)
        except Exception:
            spot = None
        state = experts.__dict__.get("_unsloth_routed_nf4")
        if isinstance(state, dict) and state.get("ready_gen") is gen and spot is not None \
                and state.get("ready_spot") == spot:
            _ready_epoch.COUNTS["table_cheap"] += 1
            return state
    _ready_epoch.COUNTS["table_full"] += 1
    try:
        from unsloth_zoo.temporary_patches.gpt_oss_routed import prepare_routed_experts
        # prepare_routed_experts only re-checks the end experts: check every expert here.
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
    state["ready_gen"], state["ready_spot"] = gen, spot
    if "storage_hold" not in state:
        # The weights' storages: a hook-less `.data =` swap then reads old values, not freed memory.
        with torch._C.DisableTorchFunctionSubclass():
            state["storage_hold"] = [w.data for w in state.get("weights", ())]
    for key in ("gate_up", "down"):
        tb = state[key]
        if "dtype" not in tb:
            projs = experts.gate_up_projs if key == "gate_up" else experts.down_projs
            tb["dtype"] = getattr(projs[0], "base_layer", projs[0]).weight.quant_state.dtype
    return state


def _proj_dtypes(projs):
    """{(compute_dtype, _pre_set_compute_dtype, quant_state.dtype)} over the experts of one projection."""
    out = set()
    for proj in projs:
        base = getattr(proj, "base_layer", proj)
        out.add((getattr(base, "compute_dtype", None), getattr(base, "_pre_set_compute_dtype", None),
                 base.weight.quant_state.dtype))
    return out


def _is16(kind, dtype):
    # Linear4bit computing in `dtype` with no conflicting override.
    return kind[0] is dtype and kind[1] in (None, dtype) and kind[2] is dtype


def compute_mode(experts, input_dtype):
    """("bf16", None) / ("fp16", down_fp32) when a grouped path reproduces the per-expert
    loop's dtypes, else (None, reason). Read per call: per-projection, so a dtype edit on
    any one expert sends the layer back to the loop."""
    gu, dn = _proj_dtypes(experts.gate_up_projs), _proj_dtypes(experts.down_projs)
    if len(gu) != 1 or len(dn) != 1:
        return None, "experts of one projection compute in different dtypes"
    (gu,), (dn,) = gu, dn
    if _is16(gu, torch.bfloat16) and _is16(dn, torch.bfloat16):
        if input_dtype not in (torch.bfloat16, torch.float32):
            return None, f"bf16 experts with {input_dtype} input"
        return "bf16", None
    if _is16(gu, torch.float16):
        # The loader's fp16 rule: down computes, dequantizes and keeps its bias in fp32.
        down_fp32 = dn == (torch.float32, torch.float32, torch.float32)
        if not (down_fp32 or _is16(dn, torch.float16)):
            return None, f"fp16 gate_up with down compute / override / quant dtypes {dn}"
        if input_dtype not in (torch.float16, torch.float32):
            return None, f"fp16 experts with {input_dtype} input"
        return "fp16", down_fp32
    return None, f"gate_up compute / override / quant dtypes {gu}, down {dn}"


def _decline(reason):
    CALLS["declined"] += 1
    if LAST_DECLINE["reason"] != reason:
        LAST_DECLINE["reason"] = reason
        if os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1":
            import logging
            logging.getLogger(__name__).info(f"Unsloth: gpt-oss grouped QLoRA keeps the per-expert loop: {reason}")
    return None


# Opaque to Dynamo (data_ptr-keyed tables, raw Triton launch): one graph break per call.
@torch.compiler.disable
def grouped_qlora_forward(experts, hidden_states, router_indices, routing_weights,
                          batch_size, num_tokens, num_experts, top_k, lora = None):
    """Grouped training forward of GptOssExpertsBnb4bit, with or without per-expert LoRA.
    Returns fp32 [batch, seq, hidden], as the per-expert training loop does, or None when
    no grouped path matches the experts' dtypes (the caller keeps the loop)."""
    mode, why = compute_mode(experts, hidden_states.dtype)
    if mode == "fp16":
        return _grouped_qlora_forward_fp16(
            experts, hidden_states, router_indices, routing_weights,
            batch_size, num_tokens, num_experts, top_k, lora, down_fp32 = why,
        )
    if mode != "bf16":
        return _decline(why)
    from unsloth_zoo.temporary_patches.moe_utils import (
        _base_grouped_mm, _check_torch_grouped_mm_supported, _moe_recompute_default,
        combine_permuted_moe_outputs, count_tokens_per_expert,
    )
    from unsloth_zoo.temporary_patches.gpt_oss import swiglu_torch_forward

    # Readiness also admits a device with only the fp16 Triton path.
    if not _check_torch_grouped_mm_supported():
        return _decline("torch._grouped_mm unsupported")
    device = hidden_states.device
    dtype = torch.bfloat16
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
    # Fixed-order top_k sum (an index_add_ over repeated tokens is atomic, so not reproducible).
    next_states = combine_permuted_moe_outputs(weighted, sorted_idx, num_tokens, top_k, out_dtype = torch.float32)
    return next_states.view(batch_size, -1, experts.hidden_size)


class _Fp16StackProvider:
    """Frozen NF4 experts of one projection as fp16 [hi - lo, N, K] stacks, from the pointer
    table (one Triton launch; an fp32 quant state rounds once, fp32 -> fp16) or bitsandbytes.
    windows() splits the experts when a whole stack does not fit in free memory."""

    __slots__ = ("tb", "projs", "N", "K", "num_experts", "dtype", "chunk", "pin")

    def __init__(self, tb, projs, recompute, dtype = torch.float16):
        qs = getattr(projs[0], "base_layer", projs[0]).weight.quant_state
        self.tb, self.projs = tb, projs
        self.N, self.K = int(qs.shape[0]), int(qs.shape[1])
        self.num_experts = len(projs)
        self.dtype = dtype
        device = qs.absmax.device
        from unsloth_zoo.temporary_patches.moe_grouped_fp16 import use_cublas
        # Per-expert cuBLAS GEMMs gain nothing from a whole stack: keep its transient small
        # (a T4 running 20B sits at ~95% of its memory with a 1 GiB gate_up stack).
        cap = int(os.environ.get("UNSLOTH_GPTOSS_FP16_CUBLAS_STACK_MB", "256")) << 20 if use_cublas(device) else None
        self.chunk = _expert_window(self.num_experts, self.N * self.K * dtype.itemsize, device, cap)
        # Pin (skip the backward rebuild) only when asked to and the whole stack fits.
        self.pin = (not recompute) and self.chunk >= self.num_experts

    def windows(self):
        E, c = self.num_experts, self.chunk
        return [(lo, min(lo + c, E)) for lo in range(0, E, c)]

    def __call__(self, lo, hi):
        tb = self.tb
        # fp16 from an fp16 or fp32 quant state (one rounding); fp32 only from an fp32 one (exact).
        ok = (torch.float16, torch.float32) if self.dtype is torch.float16 else (torch.float32,)
        if tb is not None and tb.get("dtype") in ok:
            if (lo, hi) != (0, self.num_experts):
                tb = dict(tb)
                for k in ("w", "a", "a2", "c2", "off"):
                    tb[k] = tb[k][lo:hi]
            w = nf4_dequant_expert_stack(tb, self.dtype)
            if w is not None:
                return w
        import bitsandbytes as bnb
        CALLS["bnb_fallback_dequant"] += 1
        return torch.stack([
            bnb.functional.dequantize_4bit(b.weight.data, b.weight.quant_state).to(self.dtype)
            for b in (getattr(p, "base_layer", p) for p in self.projs[lo:hi])
        ])


def _expert_window(E, bytes_per_expert, device, cap_bytes = None):
    """Experts per dequant window: all of them unless the stack would take more than half of
    the free memory (T4: a 20B gate_up stack is 1 GiB) or cap_bytes. UNSLOTH_GPTOSS_FP16_EXPERT_WINDOW pins it."""
    pinned = os.environ.get("UNSLOTH_GPTOSS_FP16_EXPERT_WINDOW")
    if pinned:
        return max(1, min(E, int(pinned)))
    try:
        free, _ = torch.cuda.mem_get_info(device)
        free += torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    except Exception:
        free = None
    budget = [b for b in (None if free is None else free // 2, cap_bytes) if b is not None]
    if not budget:
        return E
    fit = int(min(budget) // max(bytes_per_expert, 1))
    return max(1, min(E, fit))


def _stack_lora(projs, name, which, dtype):
    # [E, out, in] in `dtype`; autograd routes each slice's grad to its expert's parameter.
    stacks = _lora_stacks(projs, name)
    if stacks is not None:   # stacked LoRA: the stack itself, no per-call torch.stack
        return stacks[which == "lora_B"].to(dtype)
    mods = [getattr(p, which)[name].weight for p in projs]
    return torch.stack(mods).to(dtype)


def _lora_add_fp16(result, x, counts, projs, lora):
    """result + scaling * (x.half() @ A.half().T) @ B.T as Unsloth's forced-float32 LoRA forward
    (compiler.COMPILED_LORA_FORWARD_forced_float32: one addmm in the result's dtype). The second GEMM,
    the scaling and the add stay fp32 and round once, so an unscaled product past the fp16 range
    cannot become inf before scaling."""
    from unsloth_zoo.temporary_patches.moe_grouped_fp16 import grouped_linear
    name, scaling, _ = lora
    res_dtype = result.dtype
    A = _stack_lora(projs, name, "lora_A", torch.float16)
    B = _stack_lora(projs, name, "lora_B", res_dtype)
    xa = grouped_linear(x.to(torch.float16), A, counts)
    delta = grouped_linear(xa.to(res_dtype), B, counts, out_dtype = torch.float32)
    return (result.float() + delta * scaling).to(res_dtype)


def down_operand_dtype(device):
    """Operand dtype of the fp32 down GEMM. UNSLOTH_GPTOSS_FP16_DOWN_OPERAND = fp16 | fp32 | auto;
    auto picks from the measured speed per backend (see moe_grouped_fp16.DOWN_OPERAND_AUTO)."""
    from unsloth_zoo.temporary_patches.moe_grouped_fp16 import DOWN_OPERAND_AUTO, use_cublas
    mode = os.environ.get("UNSLOTH_GPTOSS_FP16_DOWN_OPERAND", "auto")
    if mode not in ("fp16", "fp32"):
        mode = DOWN_OPERAND_AUTO["cublas" if use_cublas(device) else "triton"]
    return torch.float32 if mode == "fp32" else torch.float16


def _grouped_qlora_forward_fp16(experts, hidden_states, router_indices, routing_weights,
                                batch_size, num_tokens, num_experts, top_k, lora, down_fp32):
    """The per-expert loop's dtypes on float16 GPUs: gate_up in fp16 (returned in the input's
    dtype, as Linear4bit does), swiglu in fp32, down on fp16 operands with an fp32 output
    and fp32 bias when the loader kept down in fp32 (else fp16 output, as the loop)."""
    from unsloth_zoo.temporary_patches.moe_grouped_fp16 import (
        Groups, fp16_grouped_available, grouped_frozen_linear, unavailable_reason,
    )
    from unsloth_zoo.temporary_patches.moe_utils import (
        _moe_recompute_default, combine_permuted_moe_outputs, count_tokens_per_expert,
    )
    from unsloth_zoo.temporary_patches.gpt_oss import swiglu_torch_forward

    device = hidden_states.device
    if not fp16_grouped_available(device):
        return _decline(unavailable_reason())
    f16 = torch.float16
    acc_dtype = hidden_states.dtype
    dn_out_dtype = torch.float32 if down_fp32 else f16
    x_mode = os.environ.get("UNSLOTH_GPTOSS_FP16_X_MODE", "cast")
    dy_mode = os.environ.get("UNSLOTH_GPTOSS_FP16_DY_MODE", "scale")
    # fp32 down (loader rule): fp32 operands reproduce the loop's fp32 GEMM (IEEE, no TF32, no
    # narrowing of x or dY); fp16 operands with an fp32 output are the speed option.
    dn_operand = down_operand_dtype(device) if down_fp32 else f16
    CALLS["forward_fp16"] += 1
    if lora is not None:
        CALLS["forward_fp16_lora"] += 1
    with torch.no_grad():
        flat_experts = router_indices.flatten()
        token_ids = torch.arange(num_tokens, device = device).repeat_interleave(top_k)
        sorted_idx = flat_experts.argsort(stable = True)
        sorted_tokens = token_ids[sorted_idx]
        expert_ids = flat_experts[sorted_idx]
        # Triton backend: counts stay on device. cuBLAS backend (below sm80): one host read
        # per layer, shared by every GEMM of its forward and backward (the loop reads once too).
        counts = Groups(count_tokens_per_expert(flat_experts, num_experts, torch.int32))

    recompute = _moe_recompute_default()
    state = _tables(experts, f16)
    gu_projs, dn_projs = experts.gate_up_projs, experts.down_projs
    gu_bias = torch.stack([getattr(p, "base_layer", p).bias for p in gu_projs]).detach()
    dn_bias = torch.stack([getattr(p, "base_layer", p).bias for p in dn_projs]).detach()
    gu_w = _Fp16StackProvider(state["gate_up"] if state is not None else None, gu_projs, recompute)
    dn_w = _Fp16StackProvider(state["down"] if state is not None else None, dn_projs, recompute, dn_operand)

    xs = hidden_states[sorted_tokens]
    # Linear4bit: x.to(compute dtype), bias in the epilogue, output back in the input's dtype.
    gate_up = grouped_frozen_linear(xs.to(f16), counts, gu_w, bias = gu_bias, out_dtype = f16).to(acc_dtype)
    if lora is not None and lora["gate_up"] is not None:
        gate_up = _lora_add_fp16(gate_up, xs, counts, gu_projs, lora["gate_up"])
    gated = swiglu_torch_forward(gate_up, experts.alpha, experts.limit, dtype = torch.float32)
    out = grouped_frozen_linear(
        gated, counts, dn_w, bias = dn_bias, out_dtype = dn_out_dtype, x_mode = x_mode, dy_mode = dy_mode,
    ).float()
    if lora is not None and lora["down"] is not None:
        out = _lora_add_fp16(out, gated, counts, dn_projs, lora["down"])

    weighted = out * routing_weights[sorted_tokens, expert_ids, None].to(torch.float32)
    next_states = combine_permuted_moe_outputs(weighted, sorted_idx, num_tokens, top_k, out_dtype = torch.float32)
    return next_states.view(batch_size, -1, experts.hidden_size)


def _grouped_training_applies(experts):
    """True when a gpt-oss experts module trains on a grouped path: bnb NF4 experts ready for it
    (_grouped_bnb4bit_ready) whose compute dtype has its backend, torch._grouped_mm for bf16 or
    moe_grouped_fp16 for fp16. The per-expert loop leaves an unrouted expert's LoRA grad None,
    a stack would give it zeros (weight decay / momentum still move it), so only these stack."""
    ready = getattr(experts, "_grouped_bnb4bit_ready", None)
    if ready is None or not ready():
        return False
    kinds = _proj_dtypes(experts.gate_up_projs)
    if len(kinds) != 1:
        return False
    dtype = next(iter(kinds))[0]
    mode, _ = compute_mode(experts, dtype)
    if mode == "bf16":
        from unsloth_zoo.temporary_patches.moe_utils import _check_torch_grouped_mm_supported
        return bool(_check_torch_grouped_mm_supported())
    if mode == "fp16":
        from unsloth_zoo.temporary_patches.moe_grouped_fp16 import fp16_grouped_available
        base = getattr(experts.gate_up_projs[0], "base_layer", experts.gate_up_projs[0])
        return bool(fp16_grouped_available(base.weight.device))
    return False


def stack_expert_lora(model, verbose = True):
    """Stack the expert LoRA of every gpt-oss ModuleList experts module on a grouped training path
    (one Parameter per projection, see moe_grouped_modulelist._StackedLoraLinear); returns
    #stacked projection lists. Loader entry point only (auto_enable_grouped_moe): it must run
    before an optimizer or DDP wrapper holds the per-expert Parameters. Idempotent;
    UNSLOTH_MOE_STACKED_LORA=0 keeps PEFT's per-expert Parameters."""
    if os.environ.get("UNSLOTH_MOE_STACKED_LORA", "1") == "0":
        return 0
    n = 0
    for experts in list(model.modules()):
        projs = (getattr(experts, "gate_up_projs", None), getattr(experts, "down_projs", None))
        if not all(isinstance(p, torch.nn.ModuleList) and len(p) for p in projs) \
                or not hasattr(experts, "_grouped_bnb4bit_ready"):
            continue
        try:
            if not _grouped_training_applies(experts):
                continue
            stacked = 0
            for p in projs:
                name = _stackable_lora(p)
                if name is not None:
                    _stack_projs_lora(p, name)
                    p.__dict__.pop("_unsloth_routed_lora", None)
                    stacked += 1
            if stacked:
                experts.__dict__.pop("_unsloth_grouped_ready", None)
            n += stacked
        except Exception as e:
            if os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1":
                import logging
                logging.getLogger(__name__).info(f"Unsloth: gpt-oss stacked expert LoRA skipped: {e}")
    if verbose and n:
        print(f"Unsloth: Stacked gpt-oss expert LoRA on {n} projection list(s).", flush = True)
    return n
