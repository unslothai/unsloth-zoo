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
every call, cached on a signature of every expert projection, which is itself only re-read after
an expert-state change (moe_ready_epoch; UNSLOTH_MOE_FAST_READY=0 re-reads it every call).

Env: UNSLOTH_MOE_GROUPED=0 disables; UNSLOTH_MOE_GROUPED_LORA=0 keeps expert-LoRA blocks on the
loop; UNSLOTH_MOE_GROUPED_RECOMPUTE=1 rebuilds the dequant stack in backward (auto-on without
gradient checkpointing); UNSLOTH_MOE_GROUPED_CACHE=1 holds the dequantized experts resident;
UNSLOTH_MOE_STACKED_LORA=0 keeps PEFT's per-expert LoRA Parameters (see _StackedLoraLinear; an
optimizer state saved before stacking does not load into a stacked model).
"""
from __future__ import annotations
import collections
import os
import types
import torch
import torch.nn.functional as F

from . import moe_ready_epoch as _ready_epoch

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

# Hot-path helpers are read as module attributes (one lookup per call, no import machinery);
# monkeypatching moe_utils / gpt_oss_grouped_qlora still reaches them.
try:
    from . import moe_utils as _moe_utils
except Exception:
    _moe_utils = None
# gpt_oss_grouped_qlora imports this module at its top: resolved on first use.
_GOQ = None


def _goq():
    global _GOQ
    if _GOQ is None:
        from . import gpt_oss_grouped_qlora
        _GOQ = gpt_oss_grouped_qlora
    return _GOQ


def _view_weight_ok(w):
    """True for a 16-byte aligned transpose(1, 2) view of a contiguous [E, N, K] stack, which
    torch._grouped_mm takes without a copy once moe_utils' one-time view probe agrees."""
    if w.dim() != 3:
        return False
    E, K, N = w.shape
    n = w.element_size()
    if w.stride() != (K * N, 1, K) or (K * n) % 16 or (K * N * n) % 16:
        return False
    try:
        from .moe_utils import _transposed_view_grouped_mm_is_safe
        return bool(_transposed_view_grouped_mm_is_safe())
    except Exception:
        return False


def _grouped_mm_fix(x: torch.Tensor, w: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """torch._grouped_mm with a per-group matmul fallback for the 16-byte stride error. The
    builders' transposed-view stacks go in uncopied when the view probe passes (_view_weight_ok)."""
    if _triton_grouped_mm_wanted is not None and _triton_grouped_mm_wanted(x, w):
        return _triton_grouped_mm(x, w, offs)
    # aten._grouped_mm's fake impl rejects float16 under torch.compile; the opaque op (moe_utils) does not.
    if (
        x.dtype == torch.float16 and w.dtype == torch.float16
        and _GROUPED_MM_FP16_OP is not None and torch.compiler.is_compiling()
    ):
        return _GROUPED_MM_FP16_OP(x, w, offs)
    x = x.contiguous()
    if not _view_weight_ok(w):
        w = w.contiguous()
    try:
        return torch._grouped_mm(x, w, offs=offs)
    except RuntimeError as e:
        if "strides should be multiple of 16 bytes" not in str(e):
            raise
        if not w.is_contiguous():   # an unaligned view (e.g. storage offset): the copy may pass
            return _grouped_mm_fix(x, w.contiguous(), offs)
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


_NF4_STACK_CALLS = {"stacked": 0, "fallback": 0}


def _nf4_stack_key(projs):
    """Everything the pointer table depends on: addresses (raw data_ptr, no __torch_function__),
    the copied code / offset (by address and version), and object identity. None if not NF4."""
    ptr = torch._C.TensorBase.data_ptr
    key = []
    for p in projs:
        base = p._modules.get("base_layer", p)
        w = base._parameters.get("weight")
        qs = getattr(w, "quant_state", None)
        if qs is None:
            return None
        a = qs.absmax
        key += (id(w), id(qs), ptr(w), ptr(a), a._version, ptr(qs.code), qs.code._version)
        if qs.nested:
            s2, off = qs.state2, qs.offset
            key += (ptr(s2.absmax), s2.absmax._version, ptr(s2.code), s2.code._version,
                    (ptr(off), off._version) if isinstance(off, torch.Tensor) else off)
    return tuple(key)


def _nf4_table(experts, kind, projs):
    """Cached pointer table (gpt_oss_routed._build_table) over `projs`, or None. The cache holds
    the projections' weights / quant states and their storages, so the addresses in the table
    cannot be recycled. While readiness vouches for the experts' storage at the current
    moe_ready_epoch stamp, the O(E) key is skipped for a spot key over the end projections."""
    cache = experts.__dict__.setdefault("_unsloth_nf4_stack_tables", {})
    hit = cache.get(kind)
    gen = _ready_epoch.current_gen(experts) if _ready_epoch.enabled() else None
    spot = None
    if gen is not None:
        try:
            spot = _nf4_stack_key([*projs[:2], *projs[-2:]])
        except Exception:
            spot = None
        if hit is not None and hit[2] is gen and hit[3] == len(projs) and spot is not None and hit[4] == spot:
            _ready_epoch.COUNTS["table_cheap"] += 1
            return hit[1]
    _ready_epoch.COUNTS["table_full"] += 1
    try:
        key = _nf4_stack_key(projs)
    except Exception:
        return None
    if key is None:
        return None
    if hit is not None and hit[0] == key:
        cache[kind] = (key, hit[1], gen, len(projs), spot)
        return hit[1]
    tb = None
    try:
        from .gpt_oss_routed import _build_table
        base0 = getattr(projs[0], "base_layer", projs[0])
        tb = _build_table(projs, base0.weight.device)
        if tb is not None:
            tb["dtype"] = base0.weight.quant_state.dtype
            ws = [getattr(p, "base_layer", p).weight for p in projs]
            tb["_hold"] = ws + [w.quant_state for w in ws]
            # The storages too: a hook-less `.data =` / absmax swap then reads old values, not freed memory.
            with torch._C.DisableTorchFunctionSubclass():
                for w in ws:
                    qs = w.quant_state
                    tb["_hold"] += (w.data, qs.absmax, qs.code)
                    if qs.nested:
                        tb["_hold"] += (qs.state2.absmax, qs.state2.code)
    except Exception:
        tb = None
    cache[kind] = (key, tb, gen, len(projs), spot)   # a None table is a negative cache under the same key
    return tb


def _nf4_stack(experts, kind, projs, dtype):
    """[E', N, K] in `dtype` from one Triton launch over `projs`, bit-identical to stacking
    bnb dequantize_4bit(...).to(dtype), or None (the caller keeps the bitsandbytes builder)."""
    # Traced code keeps the bitsandbytes builder (data_ptr tables are eager only).
    if not HAS_BNB or os.environ.get("UNSLOTH_MOE_GROUPED_NF4_STACK", "1") == "0" \
            or torch.compiler.is_compiling():
        return None
    w0 = getattr(getattr(projs[0], "base_layer", projs[0]), "weight", None)
    if not isinstance(w0, Params4bit) or getattr(w0, "device", None) is None or w0.device.type != "cuda":
        return None
    try:
        goq = _goq()
    except Exception:
        return None
    if not goq.stacked_dequant_available(w0.device):
        return None
    tb = _nf4_table(experts, kind, projs)
    # One rounding on both sides: a quant state in `dtype`, or fp32 rounded once to `dtype`.
    if tb is None or tb["dtype"] not in (dtype, torch.float32):
        return None
    return goq.nf4_dequant_expert_stack(tb, dtype)


def _nf4_build_gate_up_stack(experts, spec, dtype, projs = None):
    """_build_gate_up_stack from the pointer-table kernel, or None. The table interleaves
    [g0, u0, g1, u1, ...], so the [2E, inter, hidden] output is [E, 2*inter, hidden] =
    per expert cat(gate, up, dim=0), returned as its transposed view (no copy)."""
    projs = (_expert_projs(experts, spec) if projs is None else projs).gate_up
    w = _nf4_stack(experts, "gate_up", projs, dtype)
    if w is None:
        return None
    E, N, K = len(experts), w.shape[1], w.shape[2]
    return w.view(E, 2 * N, K).transpose(1, 2)


def _nf4_build_down_stack(experts, spec, dtype, projs = None):
    """_build_down_stack from the pointer-table kernel (transposed view), or None."""
    w = _nf4_stack(experts, "down", (_expert_projs(experts, spec) if projs is None else projs).down, dtype)
    return None if w is None else w.transpose(1, 2)


def _build_gate_up_stack(experts, spec, dtype, projs = None):
    """[E, hidden, 2*inter]: per expert cat(gate^T, up^T), a transposed view of the contiguous
    [E, 2*inter, hidden] stack (the GEMMs take the view, backward its contiguous transpose).
    `projs` is the block's _ExpertProjs (read from `experts` when omitted)."""
    w = _nf4_build_gate_up_stack(experts, spec, dtype, projs)
    # Not counted while tracing: this can run inside _GroupedFrozenMM, where Dynamo cannot
    # replay a global-dict update (fullgraph fails), and traced code never takes the NF4 kernel.
    counting = not torch.compiler.is_compiling()
    if w is not None:
        if counting:
            _NF4_STACK_CALLS["stacked"] += 1
        return w
    if counting:
        _NF4_STACK_CALLS["fallback"] += 1
    return _bnb_build_gate_up_stack(experts, spec, dtype, projs)


def _build_down_stack(experts, spec, dtype, projs = None):
    """[E, inter, hidden]: per expert down^T, a transposed view of the contiguous [E, hidden, inter] stack."""
    w = _nf4_build_down_stack(experts, spec, dtype, projs)
    # Not counted while tracing: this can run inside _GroupedFrozenMM, where Dynamo cannot
    # replay a global-dict update (fullgraph fails), and traced code never takes the NF4 kernel.
    counting = not torch.compiler.is_compiling()
    if w is not None:
        if counting:
            _NF4_STACK_CALLS["stacked"] += 1
        return w
    if counting:
        _NF4_STACK_CALLS["fallback"] += 1
    return _bnb_build_down_stack(experts, spec, dtype, projs)


def _bnb_build_gate_up_stack(experts, spec, dtype, projs = None):
    """[E, hidden, 2*inter]: per expert cat(gate^T, up^T), as a transposed view."""
    if projs is None:
        projs = _expert_projs(experts, spec)
    rows = []
    for g, u in zip(projs.gate, projs.up):
        rows.append(torch.cat((_expert_weight(g, dtype), _expert_weight(u, dtype)), dim=0))
    return torch.stack(rows, 0).transpose(1, 2)


def _bnb_build_down_stack(experts, spec, dtype, projs = None):
    """[E, inter, hidden]: per expert down^T, as a transposed view."""
    if projs is None:
        projs = _expert_projs(experts, spec)
    return torch.stack([_expert_weight(d, dtype) for d in projs.down], 0).transpose(1, 2)


# The gate / up / down projection of every expert, plus the [g0, u0, g1, u1, ...] list the
# gate_up pointer table reads. Built once per readiness verdict (_block_projs), not per call.
_ExpertProjs = collections.namedtuple("_ExpertProjs", ("gate", "up", "down", "gate_up"))


def _expert_projs(experts, spec):
    gate = [getattr(ex, spec[0]) for ex in experts]
    up = [getattr(ex, spec[1]) for ex in experts]
    down = [getattr(ex, spec[2]) for ex in experts]
    gate_up = []
    for g, u in zip(gate, up):
        gate_up += (g, u)
    return _ExpertProjs(gate, up, down, gate_up)


def _block_projs(block, experts, spec):
    """_ExpertProjs of `block`, reused while the block's readiness cache entry (_moe_ready) is the
    same object: _cached_state replaces it whenever any expert projection's identity changes, so
    the lists cannot go stale. Rebuilt every call when there is no entry (uncached readiness)."""
    ready = block.__dict__.get("_moe_ready")
    if ready is None:
        return _expert_projs(experts, spec)
    hit = block.__dict__.get("_moe_projs")
    if hit is not None and hit[0] is ready and hit[1] is experts and hit[2] == len(experts):
        projs = hit[3]
        # O(1) spot check of the end experts, in case a readiness cache keeps its entry across a swap.
        first, last = experts[0]._modules, experts[-1]._modules
        if first.get(spec[0]) is projs.gate[0] and first.get(spec[1]) is projs.up[0] \
                and first.get(spec[2]) is projs.down[0] and last.get(spec[0]) is projs.gate[-1] \
                and last.get(spec[1]) is projs.up[-1] and last.get(spec[2]) is projs.down[-1]:
            return projs
    projs = _expert_projs(experts, spec)
    block.__dict__["_moe_projs"] = (ready, experts, len(experts), projs)
    return projs


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
            # The builders return transposed views, so this is the contiguous [E, N, K] stack.
            Wt = ctx.weight_fn().transpose(1, 2)
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
        # IA3, LoHa, ... wrap with base_layer but no lora_A: the grouped path would drop the tuner.
        if any(hasattr(p, "base_layer") for p in projs):
            return "PEFT wrapper is not LoRA"
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
    stacks = _lora_stacks(projs, name)
    if stacks is not None:
        A, B = stacks
    else:
        A = torch.stack([p.lora_A[name].weight for p in projs])   # [E, r, in]
        B = torch.stack([p.lora_B[name].weight for p in projs])   # [E, out, r]
    A = A.to(dtype).transpose(1, 2)                           # [E, in, r]
    B = B.to(dtype).transpose(1, 2)                           # [E, r, out]
    A, B = _moe_utils._pad_lora_rank_for_grouped_mm(A, B)
    return A.contiguous(), B.contiguous()


def _lora_delta(x, offsets, projs, lora, dtype):
    gmm = _moe_utils._grouped_mm_with_backward_fix
    name, scaling, _ = lora
    A, B = _lora_operands(projs, name, dtype)
    h = gmm(x, A, offsets)
    return gmm(h, B, offsets) * scaling


# Stacked expert LoRA. PEFT gives every expert its own lora_A / lora_B Parameter (36,864 tensors on
# Qwen3-30B-A3B), so the optimizer, clipping and autograd pay per tensor. A converted block holds
# one [E, r, in] / [E, out, r] Parameter per projection, registered on expert 0's lora_A[n] /
# lora_B[n] as `weight_stack`; every expert's module reads `weight` as its slice (a view, autograd
# reaches the stack) and still saves / loads PEFT's per-expert `weight` keys.
_STACK_NAME = "weight_stack"


class _StackedLoraLinear(torch.nn.Linear):
    """PEFT lora_A[n] / lora_B[n] of one expert whose weight is a slice of a block-wide stack."""
    @property
    def weight(self):
        return getattr(self._unsloth_stack_owner, _STACK_NAME)[self._unsloth_stack_index]


def _stacked_state_dict_pre_hook(module, prefix, keep_vars):
    module.__dict__["_unsloth_keep_vars"] = keep_vars


def _stacked_state_dict_hook(module, destination, prefix, local_metadata):
    # Per-expert key and shape as PEFT saves them; the stack key never leaves the model.
    destination.pop(prefix + _STACK_NAME, None)
    w = module.weight
    destination[prefix + "weight"] = w if module.__dict__.pop("_unsloth_keep_vars", False) else w.detach().clone()


def _stacked_load_pre_hook(module, state_dict, prefix, local_metadata, strict,
                           missing_keys, unexpected_keys, error_msgs):
    key = prefix + "weight"
    if module._unsloth_stack_owner is module:
        module.__dict__["_unsloth_load_prefix"] = prefix
    if key not in state_dict:
        missing_keys.append(key)   # torch reports every module's keys as strict here
        return
    value = state_dict.pop(key)    # torch loads from a copy of the caller's dict
    dst = module.weight
    if not isinstance(value, torch.Tensor) or value.shape != dst.shape:
        error_msgs.append(f"size mismatch for {key}: copying a param with shape "
                          f"{getattr(value, 'shape', type(value))} from checkpoint, "
                          f"the shape in current model is {dst.shape}.")
        return
    with torch.no_grad():
        dst.copy_(value)


def _stacked_load_post_hook(module, incompatible_keys):
    prefix = module.__dict__.pop("_unsloth_load_prefix", None)
    if prefix is not None and prefix + _STACK_NAME in incompatible_keys.missing_keys:
        incompatible_keys.missing_keys.remove(prefix + _STACK_NAME)


def _lora_stacks(projs, name):
    """(A [E, r, in], B [E, out, r]) when `projs` are, in order, the projections one stack pair was
    built over, else None (per-expert Parameters, another adapter, another projection list)."""
    a0, b0 = projs[0].lora_A[name], projs[0].lora_B[name]
    if type(a0) is not _StackedLoraLinear or type(b0) is not _StackedLoraLinear:
        return None
    owner_a, owner_b = a0._unsloth_stack_owner, b0._unsloth_stack_owner
    members_a, members_b = owner_a._unsloth_stack_projs, owner_b._unsloth_stack_projs
    if len(members_a) != len(projs) or len(members_b) != len(projs):
        return None
    for p, qa, qb in zip(projs, members_a, members_b):
        if p is not qa or p is not qb:
            return None
    # PEFT replaces lora_A[n] / lora_B[n] for all experts together; the last one confirms it.
    if projs[-1].lora_A[name]._unsloth_stack_owner is not owner_a \
            or projs[-1].lora_B[name]._unsloth_stack_owner is not owner_b:
        return None
    return getattr(owner_a, _STACK_NAME), getattr(owner_b, _STACK_NAME)


def _stackable_lora(projs):
    """The one adapter name to stack over `projs`, else None. Declines whatever the grouped path
    declines (_projs_lora), several adapters on a projection, PEFT side state, non-Linear / biased /
    mismatched / non-plain-Parameter weights (DTensor, meta), and already stacked projections."""
    st = _projs_lora(projs)
    if not isinstance(st, tuple):
        return None
    name = st[0]
    shapes = {}
    for p in projs:
        if list(p.lora_A.keys()) != [name] or list(p.lora_B.keys()) != [name]:
            return None
        for attr in ("lora_embedding_A", "lora_embedding_B", "lora_magnitude_vector"):
            if len(getattr(p, attr, ())):
                return None
        if getattr(p, "lora_variant", None) or getattr(p, "lora_bias", {}).get(name, False):
            return None
        # Dropout > 0 runs the per-expert loop in train mode; a stack would turn an unrouted
        # expert's None grad into zeros (weight decay / momentum still move it).
        drop = p.lora_dropout[name] if name in p.lora_dropout else None
        if drop is not None and not isinstance(drop, torch.nn.Identity) and getattr(drop, "p", 0) > 0:
            return None
        for kind in ("A", "B"):
            m = getattr(p, "lora_" + kind)[name]
            if type(m) is not torch.nn.Linear or m.bias is not None:
                return None
            w = m._parameters.get("weight")
            if type(w) is not torch.nn.Parameter or w.is_meta or w.dim() != 2:
                return None
            if w.grad is not None:   # mid-training: an optimizer may already hold this Parameter
                return None
            sig = (tuple(w.shape), w.dtype, w.device, w.requires_grad)
            if shapes.setdefault(kind, sig) != sig:
                return None
    return name


def _stack_projs_lora(projs, name):
    """Replace every expert's lora_A[name] / lora_B[name] weight Parameter with a slice of one
    stacked Parameter (see _StackedLoraLinear). Values, dtype, device and requires_grad carry over."""
    groups = []
    for attr in ("lora_A", "lora_B"):
        mods = [getattr(p, attr)[name] for p in projs]
        ws = [m.weight for m in mods]
        with torch.no_grad():
            stack = torch.nn.Parameter(torch.stack([w.detach() for w in ws]), requires_grad=ws[0].requires_grad)
        groups.append((mods, stack))
    members = tuple(projs)
    for mods, stack in groups:   # everything allocated: mutate
        owner = mods[0]
        for i, m in enumerate(mods):
            del m._parameters["weight"]
            m.__class__ = _StackedLoraLinear
            m.__dict__["_unsloth_stack_owner"] = owner   # not a child module
            m.__dict__["_unsloth_stack_index"] = i
            m.register_state_dict_pre_hook(_stacked_state_dict_pre_hook)
            m._register_state_dict_hook(_stacked_state_dict_hook)
            m._register_load_state_dict_pre_hook(_stacked_load_pre_hook, with_module=True)
        owner.register_parameter(_STACK_NAME, stack)
        owner.__dict__["_unsloth_stack_projs"] = members
        owner.register_load_state_dict_post_hook(_stacked_load_post_hook)
    try:   # the `del m._parameters` above skips torch's registration hooks
        from unsloth_zoo.fast_grad_params import bump
        bump()
    except Exception:
        pass


def _stack_block_lora(block, spec):
    """Stack the expert LoRA of each projection of a patched block; returns #stacked projections.
    Idempotent; UNSLOTH_MOE_STACKED_LORA=0 keeps PEFT's per-expert Parameters."""
    if os.environ.get("UNSLOTH_MOE_STACKED_LORA", "1") == "0":
        return 0
    n = 0
    for attr in spec[:3]:
        projs = [getattr(ex, attr) for ex in block.experts]
        name = _stackable_lora(projs)
        if name is not None:
            _stack_projs_lora(projs, name)
            n += 1
    return n


def _ready_signature(experts, spec):
    """Key for caching _experts_grouped_state, built from every expert projection: identities
    (module, base weight, quant state, bias, LoRA modules and weights), frozen flags, and the
    PEFT state the check reads (active / disabled / merged adapters, forward pre-hooks for
    mixed-adapter batches, dropout mode and p, scaling, DoRA, which variant entries are set), and
    every weight's shape / dtype / device (`.data =` and `.to()` keep a Parameter's identity, and
    one expert or adapter can be changed alone), and accelerate hooks.
    The fields of gpt-oss' _proj_signature, read through __dict__ (~3 us per projection; E=128
    has 384).

    Returns (key, refs) or None when it cannot be built (no caching). Modules sit in the key as
    objects (identity equality); tensors and quant states by id (their __eq__ is elementwise),
    kept alive in `refs` beside the cached verdict so a freed id cannot be reused by a new one."""
    out = [len(experts)]
    append = out.append
    refs = []
    keep = refs.append
    try:
        with torch._C.DisableTorchFunctionSubclass():
            for ex in experts:
                exd = ex.__dict__
                mods = exd["_modules"]
                if "_hf_hook" in exd:
                    append("_hf_hook")
                for name in spec[:3]:
                    p = mods.get(name)
                    if p is None:
                        return None
                    d = p.__dict__
                    pm = d["_modules"]
                    base = pm.get("base_layer", p)
                    bd = base.__dict__
                    bp = bd["_parameters"]
                    w = bp.get("weight")
                    b = bp.get("bias")
                    qs = getattr(w, "quant_state", None)
                    keep((w, qs, b))
                    append((p, id(w), id(qs), getattr(w, "shape", None), w is not None and w.requires_grad,
                            id(b), b is not None and b.requires_grad,
                            getattr(w, "dtype", None), getattr(w, "device", None),
                            bd.get("compute_dtype"), "_hf_hook" in d or "_hf_hook" in bd))
                    la = pm.get("lora_A")
                    if la is None:
                        continue
                    hooks = d["_forward_pre_hooks"]
                    merged = d.get("merged_adapters")
                    active = d.get("_active_adapter")
                    variants = d.get("lora_variant") or {}
                    append((active if isinstance(active, str) else tuple(active),
                            tuple(merged) if merged else (), d.get("_disable_adapters"),
                            tuple(hooks) if hooks else (), tuple(d.get("scaling", {}).items()),
                            tuple(d.get("use_dora", {}).items()),
                            tuple((k, v is not None) for k, v in variants.items())))
                    for m in (*la.__dict__["_modules"].values(), *pm["lora_B"].__dict__["_modules"].values()):
                        mp = m.__dict__["_parameters"]
                        lw, lb = mp.get("weight", mp.get(_STACK_NAME)), mp.get("bias")
                        keep((lw, lb))
                        append((m, id(lw), id(lb), getattr(lw, "shape", None), getattr(lw, "dtype", None),
                                getattr(lw, "device", None)))
                    for m in pm["lora_dropout"].__dict__["_modules"].values():
                        md = m.__dict__
                        append((m, md.get("training"), md.get("p")))
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
            if getattr(base, "bias", None) is not None:   # the grouped GEMMs carry no bias
                return f"expert projection {name} has a bias"
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


def _base_params(experts, spec):
    """The `__dict__` of every expert's base projection (moe_ready_epoch.Lean reads them)."""
    out = []
    for ex in experts:
        mods = ex.__dict__["_modules"]
        for name in spec[:3]:
            p = mods[name]
            out.append(p.__dict__["_modules"].get("base_layer", p).__dict__)
    return out


def _spot_signature(experts, spec):
    sig = _ready_signature((experts[0], experts[-1]), spec)
    return None if sig is None else sig[0]


def _ready_record(block, experts, spec, ctx):
    """A moe_ready_epoch.Record for the full check just run, or None (accelerate hooks, no key)."""
    hooked, drops = _ready_epoch.scan(experts)
    if hooked:
        return None
    _ready_epoch.track((block,))
    spot = _ready_signature((experts[0], experts[-1]), spec)
    if spot is None:
        return None
    _ready_epoch.wrap_peft()   # PEFT tuner classes imported since the last full check
    try:
        return _ready_epoch.Record(ctx, spot[0], _base_params(experts, spec), drops, spot[1])
    except Exception:
        return None


@torch.compiler.disable
def _cached_state(block, experts, spec, device, dtype):
    """_experts_grouped_state, reused while the signature of every expert projection, the input
    device / dtype and UNSLOTH_MOE_GROUPED_LORA are unchanged (the full check is O(experts)).
    Between expert-state changes (moe_ready_epoch) only the end experts are re-read."""
    env = os.environ.get("UNSLOTH_MOE_GROUPED_LORA", "1")
    cached = block.__dict__.get("_moe_ready")
    fast = _ready_epoch.enabled()
    ctx = (device, dtype, env, experts, len(experts))
    if fast:
        rec = cached[3] if cached is not None else None
        if _ready_epoch.valid(rec, ctx, lambda: _spot_signature(experts, spec)):
            _ready_epoch.mark_valid(experts, rec)
            return cached[1]
        _ready_epoch.COUNTS["full"] += 1
    else:
        experts.__dict__.pop(_ready_epoch.VALID, None)
    sig = _ready_signature(experts, spec)
    key = None if sig is None else (device, dtype, env, sig[0])
    if key is not None and cached is not None and cached[0] == key:
        state = cached[1]
    else:
        state = _experts_grouped_state(experts, spec, device, dtype)
        if cached is not None:
            # Something changed (e.g. a merge / unmerge edits the base in place): rebuild resident stacks.
            block.__dict__.pop("_cached_gate_up", None)
            block.__dict__.pop("_cached_down", None)
    rec = _ready_record(block, experts, spec, ctx) if fast and key is not None else None
    block.__dict__["_moe_ready"] = (key, state, sig[1], rec) if key is not None else None
    if rec is not None:
        _ready_epoch.mark_valid(experts, rec)
    else:
        experts.__dict__.pop(_ready_epoch.VALID, None)
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
    # int64 matches what bincount returned, so cumsum is unchanged.
    counts = _moe_utils.count_tokens_per_expert(flat_e, num_experts, torch.int64)
    order = torch.argsort(flat_e, stable=True)
    sorted_tok = tok_of_pair[order]
    sorted_w = flat_w[order]
    offsets = torch.cumsum(counts, dim=0).to(torch.int32)
    permuted = hidden_states[sorted_tok]

    recompute = getattr(self, "_moe_recompute", False)
    cache = getattr(self, "_moe_cache", False)
    act = getattr(experts[0], "act_fn", None) or F.silu
    # Traced code reads the experts live (Dynamo guards on them); eager reuses the cached lists.
    projs = _expert_projs(experts, spec) if torch.compiler.is_compiling() else _block_projs(self, experts, spec)

    if cache:
        cu = getattr(self, "_cached_gate_up", None)
        if cu is None or cu.device != dev or cu.dtype != dtype:
            with torch.no_grad():
                # Resident stacks: one contiguous copy at build time.
                self._cached_gate_up = _build_gate_up_stack(experts, spec, dtype, projs).contiguous()
                self._cached_down = _build_down_stack(experts, spec, dtype, projs).contiguous()
        gate_up = _grouped_mm_fix(permuted, self._cached_gate_up, offsets)
    else:
        gate_up = _grouped_expert_gemm(permuted, offsets, lambda: _build_gate_up_stack(experts, spec, dtype, projs), recompute)
    # Expert LoRA: each wrapped projection adds its grouped delta to its base output, as the loop does.
    gate, up = gate_up.chunk(2, dim=-1)
    if lora["gate"] is not None:
        gate = gate + _lora_delta(permuted, offsets, projs.gate, lora["gate"], dtype)
    if lora["up"] is not None:
        up = up + _lora_delta(permuted, offsets, projs.up, lora["up"], dtype)
    inter = act(gate) * up
    if cache:
        down = _grouped_mm_fix(inter, self._cached_down, offsets)
    else:
        down = _grouped_expert_gemm(inter, offsets, lambda: _build_down_stack(experts, spec, dtype, projs), recompute)
    if lora["down"] is not None:
        down = down + _lora_delta(inter, offsets, projs.down, lora["down"], dtype)

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
            if getattr(_base_lin(lin), "bias", None) is not None:   # the grouped GEMMs carry no bias
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
                 "_moe_cache", "_cached_gate_up", "_cached_down", "_moe_ready", "_moe_projs"):
        if hasattr(module, attr):
            delattr(module, attr)
    return True


def enable_grouped_moe(model, recompute=None, cache=None, verbose=True, stack_lora=False):
    """Patch eligible ModuleList MoE blocks (frozen experts, optionally with a supported expert
    LoRA) to the grouped forward; returns #patched. Re-entrant (a now-ineligible block is
    restored, so it runs again after get_peft_model), and a no-op without grouped_mm support.
    stack_lora replaces the per-expert LoRA Parameters with stacked ones, so it must run before
    an optimizer or DDP wrapper holds them: only the loader entry points pass it."""
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
        # One Parameter per projection instead of one per expert (optimizer / clip / autograd cost).
        try:
            if stack_lora:
                _stack_block_lora(module, spec)
        except Exception as e:
            _decline(f"stacked expert LoRA skipped: {e}", count=False)
        if not warmed:
            warmed = True
            # The base and LoRA GEMMs (_grouped_mm_fix, moe_utils._grouped_mm_eager) read a one-time
            # eager probe; run it now, so a compiled first forward does not graph-break on it.
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
            enable_grouped_moe(model, verbose=True, stack_lora=True)
    except Exception:
        pass  # optional speedup; never block model loading
    try:
        # gpt-oss experts have their own grouped path (no torch._grouped_mm needed on fp16).
        if model is not None and hasattr(model, "modules"):
            from .gpt_oss_grouped_qlora import stack_expert_lora
            stack_expert_lora(model)
    except Exception:
        pass


def wrap_loader_for_grouped_moe(func):
    """Wrap a from_pretrained / get_peft_model leaf (returns model or (model, tokenizer))
    so grouped MoE and fast grad params are enabled before it returns. Idempotent."""
    if func is None or getattr(func, "_unsloth_grouped_moe_wrapped", False):
        return func
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        model = result[0] if isinstance(result, tuple) and result else result
        try:
            auto_enable_grouped_moe(model)
        except Exception:
            pass  # optional speedup; never block model loading
        try:   # Trainer's per-step clip / zero_grad without module-tree walks
            from unsloth_zoo.fast_grad_params import enable_fast_grad_params
            enable_fast_grad_params(model)
        except Exception:
            pass
        return result

    wrapper._unsloth_grouped_moe_wrapped = True
    return wrapper
