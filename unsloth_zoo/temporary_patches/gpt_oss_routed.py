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


"""Routed gpt-oss expert inference: the gpt-oss adapter over moe_routed's kernels.

NF4 experts are a ModuleList of bitsandbytes Linear4bit, read in place through per-layer
pointer tables to each expert's packed bytes (moe_routed's pointer-table layout): no host
sync, no stacked or dequantized copy, a fixed grid for a given shape, so it compiles
fullgraph and captures in CUDA graphs. BF16 / FP16 experts use the expert-major kernel.
"""

__all__ = [
    "routed_experts_forward",
    "prepare_routed_experts",
    "routed_bf16_gemm",
    "routed_bf16_forward",
    "routed_bf16_eligible",
    "routed_mlp_forward",
]

import os

import torch

from .moe_routed import (
    ACT_GPTOSS,
    ROUTED_MAX_SLOTS,
    _bf16_supported,
    _lora_delta,
    routed_bf16_gemm,
    routed_bf16_moe,
    routed_down,
    routed_gate_up,
    routed_mode,
    triton,
)


def _mixed_adapter_batch(module):
    # PEFT mixed-adapter batches inject adapter_names through a pre-hook on each LoRA layer;
    # the routed kernels never call those layers, so they must leave such batches alone.
    return any("adapter_names" in getattr(h, "keywords", ()) for h in getattr(module, "_forward_pre_hooks", {}).values())


def _routed_disabled():
    # UNSLOTH_GPTOSS_ROUTED_INFERENCE is the earlier name of the same switch.
    return "0" in (
        os.environ.get("UNSLOTH_GPTOSS_ROUTED_KERNEL", "1"),
        os.environ.get("UNSLOTH_GPTOSS_ROUTED_INFERENCE", "1"),
    ) or routed_mode() == "0"


def _base_linear4bit(module):
    """The bitsandbytes Linear4bit under an optional PEFT LoRA wrapper, else None."""
    base = getattr(module, "base_layer", module)
    weight = getattr(base, "weight", None)
    if weight is None or getattr(weight, "quant_state", None) is None:
        return None
    return base


_BIAS_KINDS = {torch.float32: 1, torch.bfloat16: 2, torch.float16: 3}


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
    as_i64 = lambda v: torch.tensor(v, dtype = torch.int64, device = device)
    bias, bias_kind = None, 0
    if any(b is not None for b in biases):
        b0 = biases[0]
        if any(
            b is None or b.dtype != b0.dtype or b.device != device or b.shape != (N,) or not b.is_contiguous()
            for b in biases
        ) or b0.dtype not in _BIAS_KINDS:
            return None
        # Pointers to each expert's live bias, not a stacked copy: load_state_dict and other
        # in-place updates are seen without a rebuild.
        bias, bias_kind = as_i64([b.data_ptr() for b in biases]), _BIAS_KINDS[b0.dtype]
    return {
        "w": as_i64(w), "a": as_i64(a), "a2": as_i64(a2), "c2": as_i64(c2),
        "off": torch.tensor(off, dtype = torch.float32, device = device),
        "lut": q0.code.to(device = device, dtype = torch.float32).contiguous(),
        "bias": bias, "bias_kind": bias_kind, "N": N, "K": K, "blocksize": int(q0.blocksize),
        "blocksize2": int(q0.state2.blocksize) if nested else 1, "nested": nested, "stacked": False,
    }


def _key(projs):
    # A moved, reloaded or replaced expert changes its packed buffer's address. Checking the
    # first and last of each list keeps the per-call cost to a few us.
    ends = [_base_linear4bit(p) for p in (projs[0], projs[-1])]
    return tuple(b.weight.data_ptr() for b in ends) + tuple(
        b.bias.data_ptr() if b.bias is not None else 0 for b in ends)


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
    if _mixed_adapter_batch(first):
        return False
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
                # lora_bias=True: the kernels add only B @ A @ x, not lora_B's bias.
                or getattr(proj.lora_B[name], "bias", None) is not None
            ):
                return False
        cache = {"name": name, "first": first.lora_A[name].weight, "versions": None,
                 "A": [proj.lora_A[name].weight for proj in projs],
                 "B": [proj.lora_B[name].weight for proj in projs]}
        projs._unsloth_routed_lora = cache
    if torch.compiler.is_compiling():
        # The persistent buffers an eager call filled: a compiled step reads them instead of
        # restacking every expert's A / B (E x r x (in + out) per layer per call).
        stacked = cache.get("stacked")
        if stacked is None:
            return torch.stack(cache["A"]), torch.stack(cache["B"]), first.scaling[name]
        return stacked[0], stacked[1], first.scaling[name]
    _restack(cache)
    return cache["stacked"][0], cache["stacked"][1], first.scaling[name]


def _restack(cache):
    """Eager: refill the persistent stacked A / B buffers in place after an optimizer step (or
    any in-place edit) bumps a version counter, or a `.data` swap moves a buffer; compiled
    steps and captured CUDA graphs keep reading the same tensors."""
    A, B = cache["A"], cache["B"]
    # In-place edits bump _version; a .to() move / cast swaps .data without bumping it, so each
    # storage pointer is checked too.
    key = tuple((p._version, p.data_ptr()) for p in A) + tuple((p._version, p.data_ptr()) for p in B)
    if key == cache["versions"]:
        return
    stacked = cache.get("stacked")
    with torch.no_grad():
        if (
            stacked is None or stacked[0].dtype != A[0].dtype or stacked[0].device != A[0].device
            or stacked[0].shape[1:] != A[0].shape or stacked[1].shape[1:] != B[0].shape
        ):
            # Normal tensors even when first built under inference_mode: later refills write them.
            with torch.inference_mode(False):
                cache["stacked"] = (torch.stack(A), torch.stack(B))
        else:
            torch.stack(A, out = stacked[0])
            torch.stack(B, out = stacked[1])
    cache["versions"] = key


def refresh_routed_lora(experts):
    """Eager calls the routed kernels do not take (prefill, the dense path) still refresh the
    stacked adapters, so a compiled decode step that follows reads current weights."""
    if torch.compiler.is_compiling():
        return
    for projs in (getattr(experts, "gate_up_projs", None), getattr(experts, "down_projs", None)):
        if projs is not None and len(projs) and hasattr(projs[0], "lora_A"):
            _lora(projs)


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
    # isinstance, not truthiness: Dynamo before torch 2.12 cannot trace bool() of a dict.
    if not isinstance(state, dict):
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
        if _mixed_adapter_batch(m):
            return None
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
        if w.dtype == torch.bfloat16 and not _bf16_supported(w.device):
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
    out = routed_bf16_moe(
        x, idx, routing_weights, _expert_weight_3d(base.gate_up_proj), _expert_weight_3d(base.down_proj),
        base.gate_up_proj_bias, base.down_proj_bias, ACT_GPTOSS, True,
        lora.get("gate_up_proj", ()), lora.get("down_proj", ()),
        getattr(base, "alpha", 1.702), getattr(base, "limit", 7.0), out_dtype = torch.float32,
    )
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
        nf4 = isinstance(state, dict)
        if nf4:
            refresh_routed_lora(experts)
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
