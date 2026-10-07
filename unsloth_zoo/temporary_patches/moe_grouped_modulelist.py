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
"""Grouped-GEMM MoE forward for transformers < 5 (the nn.ModuleList expert layout).

Replaces the Python per-expert loop with the v5 grouped path: route -> sort by expert ->
grouped_mm(gate_up) -> act(gate)*up -> grouped_mm(down) -> router scale -> fp32 scatter-add.
Same bf16 math as the loop, so accuracy is neutral. Activates only for a known block class
with no shared expert, frozen bnb-4bit / plain-frozen experts, and torch._grouped_mm support
(CUDA); otherwise the original forward runs unchanged. Patches the live instance forward so it
wins over the compiled-cache class patch.

Expert LoRA (PEFT-wrapped gate / up / down, each with or without an adapter): the frozen base
runs as above through `base_layer`, and each adapter adds a grouped delta over the stacked
per-expert A / B (_lora_delta, bf16 operands as the loop under autocast). Unsupported adapter
states (dropout > 0 while training, DoRA, lora bias, several / disabled / merged adapters,
mixed-adapter batches, heterogeneous rank or scaling) keep the loop; readiness is re-checked
every call, cached on a signature of every expert projection.

Env: UNSLOTH_MOE_GROUPED=0 disables; UNSLOTH_MOE_GROUPED_LORA=0 keeps expert-LoRA blocks on the
loop; UNSLOTH_MOE_GROUPED_RECOMPUTE=1 rebuilds the dequant stack in backward (auto-on without
gradient checkpointing); UNSLOTH_MOE_GROUPED_CACHE=1 holds the dequantized experts resident.
"""
from __future__ import annotations
import os
import types
import torch
import torch.nn.functional as F

__all__ = [
    "enable_grouped_moe",
    "disable_grouped_moe",
    "auto_enable_grouped_moe",
    "wrap_loader_for_grouped_moe",
]

try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Params4bit
    # The zoo injects a permissive bitsandbytes stub wherever the real package is absent,
    # macOS arm64 among others, and every attribute of that stub is a placeholder object
    # rather than a class. `isinstance(x, Params4bit)` against one raises TypeError, so a
    # non-class Params4bit has to count as no bitsandbytes at all.
    HAS_BNB = isinstance(Params4bit, type)
    if not HAS_BNB:
        Params4bit = None
except Exception:
    HAS_BNB = False
    bnb = None
    Params4bit = None

_GROUPED_MM_SUPPORTED = None
# Engagement counters: grouped forwards (with / without expert LoRA) and declines to the loop.
CALLS = {"grouped": 0, "grouped_lora": 0, "declined": 0}
# Last reason a patched block handed a call back to the per-expert loop.
LAST_DECLINE = {"reason": None}


def _grouped_mm_supported() -> bool:
    """Cached probe: reuse unsloth's check, else a local tiny torch._grouped_mm call."""
    global _GROUPED_MM_SUPPORTED
    if _GROUPED_MM_SUPPORTED is not None:
        return _GROUPED_MM_SUPPORTED
    ok = False
    try:
        from .moe_utils import _check_torch_grouped_mm_supported
        ok = bool(_check_torch_grouped_mm_supported())
    except Exception:
        try:
            if torch.cuda.is_available():
                dev = "cuda"
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                dev = "xpu"
            else:
                dev = None
            if hasattr(torch, "_grouped_mm") and dev is not None:
                x = torch.ones((1, 8), device=dev, dtype=torch.bfloat16)
                w = torch.ones((1, 8, 8), device=dev, dtype=torch.bfloat16)
                torch._grouped_mm(x, w, offs=torch.tensor([1], device=dev, dtype=torch.int32))
                ok = True
        except Exception:
            ok = False
    _GROUPED_MM_SUPPORTED = ok
    return ok


try:
    from .moe_utils import _GROUPED_MM_FP16_OP
except Exception:
    _GROUPED_MM_FP16_OP = None

try:
    from .moe_utils import _triton_grouped_mm, _triton_grouped_mm_wanted
except Exception:
    _triton_grouped_mm = _triton_grouped_mm_wanted = None


def _grouped_mm_fix(x: torch.Tensor, w: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """torch._grouped_mm with a per-group matmul fallback for the 16-byte stride error."""
    if _triton_grouped_mm_wanted is not None and _triton_grouped_mm_wanted(x, w):
        return _triton_grouped_mm(x, w, offs)
    # aten._grouped_mm's fake impl rejects float16 under torch.compile; the opaque op (moe_utils) does not.
    if (
        x.dtype == torch.float16 and w.dtype == torch.float16
        and _GROUPED_MM_FP16_OP is not None and torch.compiler.is_compiling()
    ):
        return _GROUPED_MM_FP16_OP(x, w, offs)
    x = x.contiguous()
    w = w.contiguous()
    try:
        return torch._grouped_mm(x, w, offs=offs)
    except RuntimeError as e:
        if "strides should be multiple of 16 bytes" not in str(e):
            raise
        # Shared with moe_utils, not looped here: a loop tapes every per-group SLICE,
        # whose shape is the router-decided group size, so non-reentrant checkpointing
        # aborts the backward. Two callers tape this, so the hazard is reachable.
        try:
            from .moe_utils import _manual_grouped_mm
        except Exception:
            outs, start = [], 0
            for i, end in enumerate(offs.detach().cpu().tolist()):
                if start < end:
                    outs.append(torch.matmul(x[start:end], w[i]))
                start = end
            return torch.cat(outs, 0) if outs else x.new_empty((0, w.shape[-1]))
        return _manual_grouped_mm(x, w, offs)


def _base_lin(lin):
    """The frozen base projection: PEFT's `base_layer` when LoRA-wrapped, else `lin`."""
    base = getattr(lin, "base_layer", None)
    return lin if base is None else base


def _expert_weight(lin, dtype):
    """Logical 2D base weight [out, in], dequantized if 4-bit (read through PEFT's base_layer)."""
    w = _base_lin(lin).weight
    if HAS_BNB and isinstance(w, Params4bit):
        return bnb.functional.dequantize_4bit(w.data, w.quant_state).to(dtype)
    return w.to(dtype)


def _route_softmax_topk(self, router_logits, top_k):
    """softmax(fp32) -> top_k -> optional renorm (Mixtral has no norm_topk_prob attr -> True)."""
    rw = F.softmax(router_logits, dim=1, dtype=torch.float32)
    rw, sel = torch.topk(rw, top_k, dim=-1)
    if getattr(self, "norm_topk_prob", True):
        rw = rw / rw.sum(dim=-1, keepdim=True)
    return rw, sel


# block class name -> (gate, up, down attr names, router fn), one row per model. Blocks with a
# shared expert (e.g. Qwen2Moe) are absent: it is not part of the routed GEMM.
_BLOCK_SPECS = {
    "Qwen3MoeSparseMoeBlock": ("gate_proj", "up_proj", "down_proj", _route_softmax_topk),
    "MixtralSparseMoeBlock":  ("w1",        "w3",      "w2",        _route_softmax_topk),
    # OLMoE has the same routed structure as Qwen3-MoE (see the parity test).
    "OlmoeSparseMoeBlock":    ("gate_proj", "up_proj", "down_proj", _route_softmax_topk),
}


def _build_gate_up_stack(experts, spec, dtype):
    """[E, hidden, 2*inter]: per expert cat(gate^T, up^T)."""
    g_name, u_name = spec[0], spec[1]
    rows = []
    for ex in experts:
        g = _expert_weight(getattr(ex, g_name), dtype)
        u = _expert_weight(getattr(ex, u_name), dtype)
        rows.append(torch.cat((g, u), dim=0).t())
    return torch.stack(rows, 0).contiguous()


def _build_down_stack(experts, spec, dtype):
    """[E, inter, hidden]: per expert down^T."""
    d_name = spec[2]
    return torch.stack([_expert_weight(getattr(ex, d_name), dtype).t() for ex in experts], 0).contiguous()


class _GroupedFrozenMM(torch.autograd.Function):
    """grouped_mm(x, W) for frozen experts: W is rebuilt by weight_fn in backward, not saved."""
    @staticmethod
    def forward(ctx, x, offsets, weight_fn):
        ctx.weight_fn = weight_fn
        ctx.save_for_backward(offsets)   # x is unused in backward (frozen base -> dX only)
        with torch.no_grad():
            out = _grouped_mm_fix(x, weight_fn(), offsets)
        return out

    @staticmethod
    def backward(ctx, g):
        (offsets,) = ctx.saved_tensors
        with torch.no_grad():
            Wt = ctx.weight_fn().transpose(1, 2).contiguous()
            dX = _grouped_mm_fix(g.contiguous(), Wt, offsets)
        return dX, None, None


def _grouped_expert_gemm(x, offsets, weight_fn, recompute):
    if recompute:
        return _GroupedFrozenMM.apply(x, offsets, weight_fn)
    return _grouped_mm_fix(x, weight_fn(), offsets)


def _lin_compute_dtype(lin):
    """Matmul dtype of one projection: Linear4bit.compute_dtype (else quant_state dtype), else weight dtype."""
    lin = _base_lin(lin)
    w = getattr(lin, "weight", None)
    if HAS_BNB and isinstance(w, Params4bit):
        compute_dtype = getattr(lin, "compute_dtype", None)
        if compute_dtype is not None:
            return compute_dtype
        return getattr(getattr(w, "quant_state", None), "dtype", torch.bfloat16)
    return getattr(w, "dtype", torch.bfloat16)


def _expert_compute_dtype(experts, spec):
    """The compute dtype of the first expert's gate projection (see _lin_compute_dtype)."""
    return _lin_compute_dtype(getattr(experts[0], spec[0], None))


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


def _ready_signature(experts, spec):
    """Key for caching _experts_grouped_state, built from every expert projection: identities
    (module, base weight, quant state, bias, LoRA modules and weights), frozen flags, and the
    PEFT state the check reads (active / disabled / merged adapters, forward pre-hooks for
    mixed-adapter batches, dropout mode and p, scaling, DoRA / variants). dtype / device / compute
    dtype are read from the first expert only: `.to()` keeps a Parameter's identity but moves
    every expert at once. The fields of gpt-oss' _proj_signature, read through __dict__
    (~3 us per projection; E=128 has 384).

    Returns (key, refs) or None when it cannot be built (no caching). Modules sit in the key as
    objects (identity equality); tensors and quant states by id (their __eq__ is elementwise),
    kept alive in `refs` beside the cached verdict so a freed id cannot be reused by a new one."""
    out = [len(experts)]
    append = out.append
    refs = []
    keep = refs.append
    try:
        with torch._C.DisableTorchFunctionSubclass():
            first = True
            for ex in experts:
                mods = ex.__dict__["_modules"]
                for name in spec[:3]:
                    p = mods.get(name)
                    if p is None:
                        return None
                    d = p.__dict__
                    pm = d["_modules"]
                    base = pm.get("base_layer", p)
                    bp = base.__dict__["_parameters"]
                    w = bp.get("weight")
                    b = bp.get("bias")
                    qs = getattr(w, "quant_state", None)
                    keep((w, qs, b))
                    append((p, id(w), id(qs), w is not None and w.requires_grad,
                            id(b), b is not None and b.requires_grad))
                    if first:
                        append((getattr(w, "dtype", None), getattr(w, "device", None),
                                base.__dict__.get("compute_dtype")))
                    la = pm.get("lora_A")
                    if la is None:
                        continue
                    hooks = d["_forward_pre_hooks"]
                    merged = d.get("merged_adapters")
                    active = d.get("_active_adapter")
                    append((active if isinstance(active, str) else tuple(active),
                            tuple(merged) if merged else (), d.get("_disable_adapters"),
                            tuple(hooks) if hooks else (), tuple(d.get("scaling", {}).items()),
                            tuple(d.get("use_dora", {}).items()), len(d.get("lora_variant") or ())))
                    for m in (*la.__dict__["_modules"].values(), *pm["lora_B"].__dict__["_modules"].values()):
                        mp = m.__dict__["_parameters"]
                        lw, lb = mp.get("weight"), mp.get("bias")
                        keep((lw, lb))
                        append((m, id(lw), id(lb)))
                        if first:
                            append((getattr(lw, "dtype", None), getattr(lw, "device", None)))
                    for m in pm["lora_dropout"].__dict__["_modules"].values():
                        md = m.__dict__
                        append((m, md.get("training"), md.get("p")))
                first = False
    except Exception:
        return None
    return tuple(out), refs


def _experts_grouped_state(experts, spec, device, dtype):
    """{"gate", "up", "down"} -> None / (adapter, scaling, rank) when the grouped path reproduces
    the loop, else a reason string. Every expert projection must be present, frozen, on `device`
    and matmul in `dtype` (read through PEFT's base_layer); a LoRA-wrapped projection needs one
    supported adapter on every expert (_projs_lora), on `device`. Checks all experts (not just
    experts[0]): the stacks read them all, and state can change after enable."""
    for ex in experts:
        for name in spec[:3]:
            lin = getattr(ex, name, None)
            base = _base_lin(lin) if lin is not None else None
            w = getattr(base, "weight", None)
            if w is None:
                return f"expert projection {name} has no weight"
            if w.requires_grad:
                return "trainable expert weight"
            if getattr(w, "device", device) != device:
                return "expert weight on another device"
            if _lin_compute_dtype(base) != dtype:
                return "expert compute dtype differs from the input"
    lora = {}
    for key, name in zip(("gate", "up", "down"), spec[:3]):
        projs = [getattr(ex, name) for ex in experts]
        st = _projs_lora(projs)
        if isinstance(st, str):
            return f"{name} LoRA: {st}"
        if st is not None:
            if os.environ.get("UNSLOTH_MOE_GROUPED_LORA", "1") == "0":
                return "UNSLOTH_MOE_GROUPED_LORA=0"
            if any(p.lora_A[st[0]].weight.device != device or p.lora_B[st[0]].weight.device != device
                   for p in projs):
                return f"{name} LoRA weights on another device"
        lora[key] = st
    return lora


def _experts_grouped_ready(experts, spec, device, dtype) -> bool:
    """True when grouped_moe_forward reproduces the loop for these experts (see _experts_grouped_state)."""
    return isinstance(_experts_grouped_state(experts, spec, device, dtype), dict)


def _decline(reason, count=True):
    """Record a hand-back to the per-expert loop; logged once per new reason under UNSLOTH_ENABLE_LOGGING."""
    if count:
        CALLS["declined"] += 1
    if LAST_DECLINE["reason"] != reason:
        LAST_DECLINE["reason"] = reason
        if os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1":
            import logging
            logging.getLogger(__name__).info(f"Unsloth: grouped MoE keeps the per-expert loop: {reason}")
    return None


@torch.compiler.disable
def _cached_state(block, experts, spec, device, dtype):
    """_experts_grouped_state, reused while the signature of every expert projection, the input
    device / dtype and UNSLOTH_MOE_GROUPED_LORA are unchanged (the full check is O(experts))."""
    sig = _ready_signature(experts, spec)
    key = None if sig is None else (device, dtype, os.environ.get("UNSLOTH_MOE_GROUPED_LORA", "1"), sig[0])
    cached = block.__dict__.get("_moe_ready")
    if key is not None and cached is not None and cached[0] == key:
        return cached[1]
    state = _experts_grouped_state(experts, spec, device, dtype)
    if cached is not None:
        # Something changed (e.g. a merge / unmerge edits the base in place): rebuild resident stacks.
        block.__dict__.pop("_cached_gate_up", None)
        block.__dict__.pop("_cached_down", None)
    block.__dict__["_moe_ready"] = (key, state, sig[1]) if key is not None else None
    return state


def _grouped_state(block, experts, spec, device, dtype):
    # Dynamo traces the plain check and guards on what it reads; eager reuses the cached verdict.
    if torch.compiler.is_compiling():
        return _experts_grouped_state(experts, spec, device, dtype)
    return _cached_state(block, experts, spec, device, dtype)


def grouped_moe_forward(self, hidden_states: torch.Tensor):
    spec = self._unsloth_moe_spec
    experts = self.experts
    # Run the original loop unless this is the frozen-base CUDA path in a low-precision dtype
    # with every expert grouped-ready (CPU/offload, unsupported LoRA, fp32, dtype/device
    # mismatch fall back).
    if hidden_states.device.type not in ("cuda", "xpu") \
            or hidden_states.dtype not in (torch.bfloat16, torch.float16):
        if not torch.compiler.is_compiling():
            _decline(f"{hidden_states.device.type} {hidden_states.dtype} input")
        return self._orig_moe_forward(hidden_states)
    lora = _grouped_state(self, experts, spec, hidden_states.device, hidden_states.dtype)
    if not isinstance(lora, dict):
        if not torch.compiler.is_compiling():
            _decline(lora)
        return self._orig_moe_forward(hidden_states)
    # Dynamo replays these global-dict updates after each compiled call (no graph break).
    CALLS["grouped"] += 1
    if lora["gate"] is not None or lora["up"] is not None or lora["down"] is not None:
        CALLS["grouped_lora"] += 1
    is_3d = hidden_states.dim() == 3
    if is_3d:
        bsz, seqlen, hidden_dim = hidden_states.shape
    else:
        seqlen, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    if self.training and getattr(self, "jitter_noise", 0):   # Mixtral router jitter
        hidden_states = hidden_states * torch.empty_like(hidden_states).uniform_(
            1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
    T = hidden_states.shape[0]
    num_experts = self.num_experts
    top_k = self.top_k
    dev = hidden_states.device
    dtype = hidden_states.dtype

    router_logits = self.gate(hidden_states)
    rw, sel = spec[3](self, router_logits, top_k)
    rw = rw.to(dtype)

    flat_e = sel.reshape(-1)
    flat_w = rw.reshape(-1)
    tok_of_pair = torch.arange(T, device=dev).repeat_interleave(top_k)
    from .moe_utils import count_tokens_per_expert
    # int64 matches what bincount returned, so cumsum is unchanged.
    counts = count_tokens_per_expert(flat_e, num_experts, torch.int64)
    order = torch.argsort(flat_e, stable=True)
    sorted_tok = tok_of_pair[order]
    sorted_w = flat_w[order]
    offsets = torch.cumsum(counts, dim=0).to(torch.int32)
    permuted = hidden_states[sorted_tok]

    recompute = getattr(self, "_moe_recompute", False)
    cache = getattr(self, "_moe_cache", False)
    act = getattr(experts[0], "act_fn", None) or F.silu

    if cache:
        cu = getattr(self, "_cached_gate_up", None)
        if cu is None or cu.device != dev or cu.dtype != dtype:
            with torch.no_grad():
                self._cached_gate_up = _build_gate_up_stack(experts, spec, dtype)
                self._cached_down = _build_down_stack(experts, spec, dtype)
        gate_up = _grouped_mm_fix(permuted, self._cached_gate_up, offsets)
    else:
        gate_up = _grouped_expert_gemm(permuted, offsets, lambda: _build_gate_up_stack(experts, spec, dtype), recompute)
    gate, up = gate_up.chunk(2, dim=-1)
    # Expert LoRA: each wrapped projection adds its grouped delta to its base output, as the loop does.
    if lora["gate"] is not None:
        gate = gate + _lora_delta(permuted, offsets, [getattr(ex, spec[0]) for ex in experts], lora["gate"], dtype)
    if lora["up"] is not None:
        up = up + _lora_delta(permuted, offsets, [getattr(ex, spec[1]) for ex in experts], lora["up"], dtype)
    inter = act(gate) * up
    if cache:
        down = _grouped_mm_fix(inter, self._cached_down, offsets)
    else:
        down = _grouped_expert_gemm(inter, offsets, lambda: _build_down_stack(experts, spec, dtype), recompute)
    if lora["down"] is not None:
        down = down + _lora_delta(inter, offsets, [getattr(ex, spec[2]) for ex in experts], lora["down"], dtype)

    down = down * sorted_w.unsqueeze(-1)
    final = torch.zeros((T, hidden_dim), dtype=torch.float32, device=dev)
    final.index_add_(0, sorted_tok, down.to(torch.float32))
    final = final.to(dtype)
    if is_3d:
        final = final.reshape(bsz, seqlen, hidden_dim)
    return final, router_logits


def _block_is_eligible(block):
    """Return the spec if this is a ModuleList MoE with frozen experts we can speed up, else None.
    LoRA-wrapped expert projections qualify when their adapter state is supported."""
    spec = _BLOCK_SPECS.get(type(block).__name__)
    if spec is None:
        return None
    experts = getattr(block, "experts", None)
    if experts is None or not hasattr(experts, "__len__") or len(experts) == 0:
        return None
    if not hasattr(block, "gate") or not hasattr(block, "num_experts") or not hasattr(block, "top_k"):
        return None
    for attr in ("shared_expert", "shared_experts", "shared_expert_gate"):  # not routed -> bail
        if getattr(block, attr, None) is not None:
            return None
    g_name, u_name, d_name, _ = spec
    for ex in experts:   # frozen base only (returns dX): a trainable expert -> original loop
        for name in (g_name, u_name, d_name):
            lin = getattr(ex, name, None)
            w = getattr(_base_lin(lin), "weight", None) if lin is not None else None
            if w is None:
                return None
            is_4bit = (HAS_BNB and isinstance(w, Params4bit)
                       and getattr(w, "quant_state", None) is not None and not w.requires_grad)
            # fp32 omitted: torch._grouped_mm targets low-precision CUDA inputs.
            is_plain_frozen = (not w.requires_grad) and w.dtype in (torch.bfloat16, torch.float16)
            if not (is_4bit or is_plain_frozen):
                return None
    for name in (g_name, u_name, d_name):
        st = _projs_lora([getattr(ex, name) for ex in experts])
        if st is None:
            continue
        if isinstance(st, str):
            _decline(f"{name} LoRA: {st}", count=False)
            return None
        if os.environ.get("UNSLOTH_MOE_GROUPED_LORA", "1") == "0":
            _decline("UNSLOTH_MOE_GROUPED_LORA=0", count=False)
            return None
    return spec


def _uses_grad_checkpointing(model) -> bool:
    if getattr(model, "is_gradient_checkpointing", False):
        return True
    return any(getattr(m, "gradient_checkpointing", False) for m in model.modules())


def _restore_block(module):
    if not hasattr(module, "_orig_moe_forward"):
        return False
    if getattr(getattr(module, "forward", None), "__func__", None) is grouped_moe_forward:
        module.forward = module._orig_moe_forward   # only restore our own patch
    for attr in ("_orig_moe_forward", "_unsloth_moe_spec", "_moe_recompute",
                 "_moe_cache", "_cached_gate_up", "_cached_down", "_moe_ready"):
        if hasattr(module, attr):
            delattr(module, attr)
    return True


def enable_grouped_moe(model, recompute=None, cache=None, verbose=True):
    """Patch eligible ModuleList MoE blocks (frozen experts, optionally with a supported expert
    LoRA) to the grouped forward; returns #patched. Re-entrant (a now-ineligible block is
    restored, so it runs again after get_peft_model), and a no-op without grouped_mm support."""
    if os.environ.get("UNSLOTH_MOE_GROUPED", "1") == "0":
        for module in model.modules():
            _restore_block(module)
        return 0
    if not _grouped_mm_supported():
        return 0
    if recompute is None:
        env = os.environ.get("UNSLOTH_MOE_GROUPED_RECOMPUTE")
        recompute = (env == "1") if env is not None else (not _uses_grad_checkpointing(model))
    if cache is None:
        cache = os.environ.get("UNSLOTH_MOE_GROUPED_CACHE") == "1"
    n = 0
    warmed = False
    for module in model.modules():
        spec = _block_is_eligible(module)
        if spec is None:
            _restore_block(module)
            continue
        if not hasattr(module, "_orig_moe_forward"):
            module._orig_moe_forward = module.forward
        module._unsloth_moe_spec = spec
        module._moe_recompute = recompute
        module._moe_cache = cache
        module.__dict__.pop("_moe_ready", None)
        module.forward = types.MethodType(grouped_moe_forward, module)
        n += 1
        if not warmed and any(hasattr(getattr(module.experts[0], name), "lora_A") for name in spec[:3]):
            warmed = True
            # The LoRA GEMMs (moe_utils._grouped_mm_eager) read a one-time eager probe; run it now,
            # so a compiled first forward does not graph-break on it.
            try:
                from .moe_utils import _transposed_view_grouped_mm_is_safe
                _transposed_view_grouped_mm_is_safe()
            except Exception:
                pass
    if verbose and n:
        print(f"Unsloth: Grouped MoE enabled on {n} block(s) (recompute={recompute}, cache={cache}).", flush=True)
    return n


def disable_grouped_moe(model):
    return sum(_restore_block(m) for m in list(model.modules()))


def auto_enable_grouped_moe(model):
    """Loader entry point; fully guarded so it never raises into model loading."""
    try:
        if model is not None and hasattr(model, "modules"):
            enable_grouped_moe(model, verbose=True)
    except Exception:
        pass  # optional speedup; never block model loading


def wrap_loader_for_grouped_moe(func):
    """Wrap a from_pretrained / get_peft_model leaf (returns model or (model, tokenizer))
    so grouped MoE is enabled before it returns. Idempotent."""
    if func is None or getattr(func, "_unsloth_grouped_moe_wrapped", False):
        return func
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        try:
            auto_enable_grouped_moe(result[0] if isinstance(result, tuple) and result else result)
        except Exception:
            pass  # optional speedup; never block model loading
        return result

    wrapper._unsloth_grouped_moe_wrapped = True
    return wrapper
