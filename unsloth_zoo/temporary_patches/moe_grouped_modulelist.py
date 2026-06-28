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

On transformers < 5 a MoE block (Qwen3MoeSparseMoeBlock, MixtralSparseMoeBlock, ...)
keeps its experts as a nn.ModuleList of per-expert MLPs and loops over them in Python,
launching O(num_experts) tiny matmuls + a data-dependent sync per layer and never
calling torch._grouped_mm. The v5 stacked layout already uses a grouped path; this
brings the same recipe to the < 5 ModuleList layout:

    route -> sort tokens by expert -> grouped_mm (gate_up) -> act(gate)*up
    -> grouped_mm (down) -> router-weight scale -> float32 scatter-add

Experts stay 4-bit; the bf16 dequant stack is rebuilt in backward (recompute) or held
resident (cache). Same bf16 math as the loop, so accuracy is neutral.

Activates only when: a known block class with no shared expert, frozen bnb-4bit (or
plain-frozen) ModuleList experts, no LoRA on the experts, and torch._grouped_mm is
supported (CUDA). Otherwise the original forward runs unchanged (so v5, non-bnb,
LoRA-on-experts and Mac/MLX/AMD/Intel/CPU are no-ops).

Patches the live instance forward after the model is built, so it wins over the
compiled-cache class patch. Disable with UNSLOTH_MOE_GROUPED=0. Force backward-recompute
with UNSLOTH_MOE_GROUPED_RECOMPUTE=1 (auto-on when gradient checkpointing is off); hold
the dequantized experts resident with UNSLOTH_MOE_GROUPED_CACHE=1 (big-GPU opt-in).
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
    HAS_BNB = True
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
            if hasattr(torch, "_grouped_mm") and torch.cuda.is_available():
                x = torch.ones((1, 8), device="cuda", dtype=torch.bfloat16)
                w = torch.ones((1, 8, 8), device="cuda", dtype=torch.bfloat16)
                torch._grouped_mm(x, w, offs=torch.tensor([1], device="cuda", dtype=torch.int32))
                ok = True
        except Exception:
            ok = False
    _GROUPED_MM_SUPPORTED = ok
    return ok


def _grouped_mm_fix(x: torch.Tensor, w: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """torch._grouped_mm with a contiguity guard + per-group matmul fallback for the
    rare 16-byte stride error (mirrors moe_utils._grouped_mm_with_backward_fix)."""
    x = x.contiguous()
    w = w.contiguous()
    try:
        return torch._grouped_mm(x, w, offs=offs)
    except RuntimeError as e:
        if "strides should be multiple of 16 bytes" not in str(e):
            raise
        outs, start = [], 0
        for i, end in enumerate(offs.detach().cpu().tolist()):
            if start < end:
                outs.append(torch.matmul(x[start:end], w[i]))
            start = end
        return torch.cat(outs, 0) if outs else x.new_empty((0, w.shape[-1]))


def _expert_weight(lin, dtype):
    """Logical 2D weight [out, in], dequantized if 4-bit."""
    w = lin.weight
    if HAS_BNB and isinstance(w, Params4bit):
        return bnb.functional.dequantize_4bit(w.data, w.quant_state).to(dtype)
    return w.to(dtype)


def _route_softmax_topk(self, router_logits, top_k):
    """softmax(fp32) -> top_k -> optional renorm. Covers Qwen3-MoE / Mixtral (Mixtral
    has no norm_topk_prob attr -> default True == its routing_weights /= sum)."""
    rw = F.softmax(router_logits, dim=1, dtype=torch.float32)
    rw, sel = torch.topk(rw, top_k, dim=-1)
    if getattr(self, "norm_topk_prob", True):
        rw = rw / rw.sum(dim=-1, keepdim=True)
    return rw, sel


# block class name -> (gate, up, down attr names, router fn). Add a row per model.
# Qwen2MoeSparseMoeBlock is intentionally absent: its shared expert is not part of the
# routed grouped GEMM, so it is skipped (see the shared-expert bail in _block_is_eligible).
_BLOCK_SPECS = {
    "Qwen3MoeSparseMoeBlock": ("gate_proj", "up_proj", "down_proj", _route_softmax_topk),
    "MixtralSparseMoeBlock":  ("w1",        "w3",      "w2",        _route_softmax_topk),
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
    """grouped_mm(x, W) for FROZEN experts. W is rebuilt by weight_fn in backward
    instead of saved, so the bf16 dequant stack is not held across the step."""
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


def grouped_moe_forward(self, hidden_states: torch.Tensor):
    spec = self._unsloth_moe_spec
    experts = self.experts
    # Fall back to the original forward for CPU/offloaded inputs or experts that gained
    # LoRA after patching (grouped stays on the frozen-expert CUDA path only).
    lin0 = getattr(experts[0], spec[0], None)
    if (not hidden_states.is_cuda) or getattr(lin0, "lora_A", None) is not None \
            or getattr(lin0, "base_layer", None) is not None:
        return self._orig_moe_forward(hidden_states)
    is_3d = hidden_states.dim() == 3
    if is_3d:
        bsz, seqlen, hidden_dim = hidden_states.shape
    else:
        seqlen, hidden_dim = hidden_states.shape
        bsz = 1
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
    rw, sel = spec[3](self, router_logits, top_k)   # exact per-model router
    rw = rw.to(dtype)

    flat_e = sel.reshape(-1)
    flat_w = rw.reshape(-1)
    tok_of_pair = torch.arange(T, device=dev).repeat_interleave(top_k)
    counts = torch.bincount(flat_e, minlength=num_experts)
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
        if cu is None or cu.device != dev or cu.dtype != dtype:   # (re)build on device/dtype change
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
    for ex in experts:   # every expert must be frozen with no LoRA, else run the original loop
        for name in (g_name, u_name, d_name):
            lin = getattr(ex, name, None)
            w = getattr(lin, "weight", None)
            if w is None:
                return None
            if getattr(lin, "lora_A", None) is not None or getattr(lin, "base_layer", None) is not None:
                return None
            is_4bit = HAS_BNB and isinstance(w, Params4bit) and getattr(w, "quant_state", None) is not None
            is_plain_frozen = (not w.requires_grad) and w.dtype in (torch.bfloat16, torch.float16, torch.float32)
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
    """Patch eligible frozen ModuleList MoE blocks to the grouped forward. Re-entrant
    (a block that became ineligible, e.g. LoRA was added, is restored). Safe no-op when
    torch._grouped_mm is unsupported or no eligible block exists. Returns #patched."""
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
