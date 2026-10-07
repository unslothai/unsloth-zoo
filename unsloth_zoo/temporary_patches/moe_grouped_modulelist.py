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
with no shared expert, frozen bnb-4bit / plain-frozen experts, no expert LoRA, and
torch._grouped_mm support (CUDA); otherwise the original forward runs unchanged. Patches the
live instance forward so it wins over the compiled-cache class patch.

Env: UNSLOTH_MOE_GROUPED=0 disables; UNSLOTH_MOE_GROUPED_RECOMPUTE=1 rebuilds the dequant
stack in backward (auto-on without gradient checkpointing); UNSLOTH_MOE_GROUPED_CACHE=1 holds
the dequantized experts resident.
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


def _expert_weight(lin, dtype):
    """Logical 2D weight [out, in], dequantized if 4-bit."""
    w = lin.weight
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
    the projections' weights / quant states, so the addresses in the table cannot be recycled."""
    try:
        key = _nf4_stack_key(projs)
    except Exception:
        return None
    if key is None:
        return None
    cache = experts.__dict__.setdefault("_unsloth_nf4_stack_tables", {})
    hit = cache.get(kind)
    if hit is not None and hit[0] == key:
        return hit[1]
    tb = None
    try:
        from .gpt_oss_routed import _build_table
        base0 = getattr(projs[0], "base_layer", projs[0])
        tb = _build_table(projs, base0.weight.device)
        if tb is not None:
            tb["dtype"] = base0.weight.quant_state.dtype
            tb["_hold"] = [getattr(p, "base_layer", p).weight for p in projs]
            tb["_hold"] += [w.quant_state for w in tb["_hold"]]
    except Exception:
        tb = None
    cache[kind] = (key, tb)   # a None table is a negative cache under the same key
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
        from .gpt_oss_grouped_qlora import nf4_dequant_expert_stack, stacked_dequant_available
    except Exception:
        return None
    if not stacked_dequant_available(w0.device):
        return None
    tb = _nf4_table(experts, kind, projs)
    # One rounding on both sides: a quant state in `dtype`, or fp32 rounded once to `dtype`.
    if tb is None or tb["dtype"] not in (dtype, torch.float32):
        return None
    return nf4_dequant_expert_stack(tb, dtype)


def _nf4_build_gate_up_stack(experts, spec, dtype):
    """_build_gate_up_stack from the pointer-table kernel, or None. The table interleaves
    [g0, u0, g1, u1, ...], so the [2E, inter, hidden] output is [E, 2*inter, hidden] =
    per expert cat(gate, up, dim=0)."""
    g_name, u_name = spec[0], spec[1]
    projs = []
    for ex in experts:
        projs += (getattr(ex, g_name), getattr(ex, u_name))
    w = _nf4_stack(experts, "gate_up", projs, dtype)
    if w is None:
        return None
    E, N, K = len(experts), w.shape[1], w.shape[2]
    return w.view(E, 2 * N, K).transpose(1, 2).contiguous()


def _nf4_build_down_stack(experts, spec, dtype):
    """_build_down_stack from the pointer-table kernel, or None."""
    w = _nf4_stack(experts, "down", [getattr(ex, spec[2]) for ex in experts], dtype)
    return None if w is None else w.transpose(1, 2).contiguous()


def _build_gate_up_stack(experts, spec, dtype):
    """[E, hidden, 2*inter]: per expert cat(gate^T, up^T)."""
    w = _nf4_build_gate_up_stack(experts, spec, dtype)
    if w is not None:
        _NF4_STACK_CALLS["stacked"] += 1
        return w
    _NF4_STACK_CALLS["fallback"] += 1
    return _bnb_build_gate_up_stack(experts, spec, dtype)


def _build_down_stack(experts, spec, dtype):
    """[E, inter, hidden]: per expert down^T."""
    w = _nf4_build_down_stack(experts, spec, dtype)
    if w is not None:
        _NF4_STACK_CALLS["stacked"] += 1
        return w
    _NF4_STACK_CALLS["fallback"] += 1
    return _bnb_build_down_stack(experts, spec, dtype)


def _bnb_build_gate_up_stack(experts, spec, dtype):
    """[E, hidden, 2*inter]: per expert cat(gate^T, up^T)."""
    g_name, u_name = spec[0], spec[1]
    rows = []
    for ex in experts:
        g = _expert_weight(getattr(ex, g_name), dtype)
        u = _expert_weight(getattr(ex, u_name), dtype)
        rows.append(torch.cat((g, u), dim=0).t())
    return torch.stack(rows, 0).contiguous()


def _bnb_build_down_stack(experts, spec, dtype):
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


def _experts_grouped_ready(experts, spec, device, dtype) -> bool:
    """Every expert projection must be present, LoRA-free, frozen, on `device`, and matmul in
    `dtype`, else fall back. Checks all experts (not just experts[0]): the stacks read them all,
    and state can change after enable (heterogeneous device_map / offload / unfrozen / dtype)."""
    for ex in experts:
        for name in spec[:3]:
            lin = getattr(ex, name, None)
            w = getattr(lin, "weight", None)
            if w is None:
                return False
            if getattr(lin, "lora_A", None) is not None or getattr(lin, "base_layer", None) is not None:
                return False
            if w.requires_grad:
                return False
            if getattr(w, "device", device) != device:
                return False
            if _lin_compute_dtype(lin) != dtype:
                return False
    return True


def grouped_moe_forward(self, hidden_states: torch.Tensor):
    spec = self._unsloth_moe_spec
    experts = self.experts
    # Run the original loop unless this is the frozen-expert CUDA path in a low-precision dtype
    # with every expert grouped-ready (CPU/offload, LoRA, fp32, dtype/device mismatch fall back).
    if hidden_states.device.type not in ("cuda", "xpu") \
            or hidden_states.dtype not in (torch.bfloat16, torch.float16) \
            or not _experts_grouped_ready(experts, spec, hidden_states.device, hidden_states.dtype):
        return self._orig_moe_forward(hidden_states)
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
        gate, up = gate_up.chunk(2, dim=-1)
        inter = act(gate) * up
        down = _grouped_mm_fix(inter, self._cached_down, offsets)
    else:
        gate_up = _grouped_expert_gemm(permuted, offsets, lambda: _build_gate_up_stack(experts, spec, dtype), recompute)
        gate, up = gate_up.chunk(2, dim=-1)
        inter = act(gate) * up
        down = _grouped_expert_gemm(inter, offsets, lambda: _build_down_stack(experts, spec, dtype), recompute)

    down = down * sorted_w.unsqueeze(-1)
    final = torch.zeros((T, hidden_dim), dtype=torch.float32, device=dev)
    final.index_add_(0, sorted_tok, down.to(torch.float32))
    final = final.to(dtype)
    if is_3d:
        final = final.reshape(bsz, seqlen, hidden_dim)
    return final, router_logits


def _block_is_eligible(block):
    """Return the spec if this is a frozen ModuleList MoE we can speed up, else None."""
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
    for ex in experts:   # frozen-only path (returns dX): any LoRA / trainable expert -> original loop
        for name in (g_name, u_name, d_name):
            lin = getattr(ex, name, None)
            w = getattr(lin, "weight", None)
            if w is None:
                return None
            if getattr(lin, "lora_A", None) is not None or getattr(lin, "base_layer", None) is not None:
                return None
            is_4bit = (HAS_BNB and isinstance(w, Params4bit)
                       and getattr(w, "quant_state", None) is not None and not w.requires_grad)
            # fp32 omitted: torch._grouped_mm targets low-precision CUDA inputs.
            is_plain_frozen = (not w.requires_grad) and w.dtype in (torch.bfloat16, torch.float16)
            if not (is_4bit or is_plain_frozen):
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
                 "_moe_cache", "_cached_gate_up", "_cached_down"):
        if hasattr(module, attr):
            delattr(module, attr)
    return True


def enable_grouped_moe(model, recompute=None, cache=None, verbose=True):
    """Patch eligible frozen ModuleList MoE blocks to the grouped forward; returns #patched.
    Re-entrant (a now-ineligible block is restored), and a no-op without grouped_mm support."""
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
        module.forward = types.MethodType(grouped_moe_forward, module)
        n += 1
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
