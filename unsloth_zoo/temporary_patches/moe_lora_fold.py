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
"""Routed-expert LoRA folded into the frozen base grouped GEMM.

The separated path runs the base GEMM, then T = X @ A and D = T @ B as two more grouped GEMMs, then
Y = base + D * s, and autograd sums the base and LoRA dX. Folded, on the generic Triton grouped GEMM
(moe_grouped_fp16) that the base GEMM already takes on sm80 / sm89 / sm120:

  forward  Y  = [X | s T] @ [W ; B]          one GEMM, T = X @ A (rank-r GEMM) is its second K segment
  dX       dX = [dY | s U] @ [W^T ; A^T]     one GEMM, U = dY @ B^T
  dA = s X^T U, dB = s T^T dY                per-expert reductions, as before

The rank segments are read from their own pointers, so no [E, K + r, N] stack is copied. dX takes the K
concatenation rather than dY @ [W^T | B^T]: the N concatenation would still need a separate s U @ A^T GEMM
and an [R, K] add. Every term accumulates in fp32 and rounds once; the separated path rounds D, D * s and
the sum, so the two agree to bf16 rounding, not bitwise.

UNSLOTH_MOE_LORA_FOLD=0 restores the separated path. The fold engages on a frozen base whose GEMM the Triton
policy already sends to the generic kernel, with a Python float scaling and rank <= 128, in eager code and in
traced code (the MoE experts forward is compiled on transformers 5.x) through the opaque op
unsloth_zoo::grouped_mm_lora_fold_v1; everything else keeps the separated path, as do the native_torch and
unsloth_triton backends."""

__all__ = [
    "LORA_FOLD_STATS",
    "lora_fold_enabled",
    "lora_fold_decline_reason",
    "lora_fold_base_mm",
    "lora_fold_prepare",
    "lora_fold_onto_base",
]

import collections
import os

import torch

# Eager calls only (traced ones show in moe_grouped_fp16.LORA_CALLS, counted inside the op): fold / accumulate:
# forward launches; backward: folded dX launches; declined: eligibility misses (by reason in DECLINE_REASONS).
LORA_FOLD_STATS = {"fold": 0, "accumulate": 0, "backward": 0, "declined": 0}
LAST_DECLINE = {"reason": None}
DECLINE_REASONS = collections.Counter()


def lora_fold_enabled() -> bool:
    return os.environ.get("UNSLOTH_MOE_LORA_FOLD", "1").strip().lower() not in ("0", "false", "off", "no")


def _mu():
    from unsloth_zoo.temporary_patches import moe_utils
    return moe_utils


def _mg():
    from unsloth_zoo.temporary_patches import moe_grouped_fp16
    return moe_grouped_fp16


def _base_param(source):
    param = source
    while hasattr(param, "base_layer"):
        param = param.base_layer
    return param


def lora_fold_decline_reason(inputs, first, second, source):
    """None when the fold applies to inputs [R, K] with LoRA first [E, K, r] / second [E, r, N] over a frozen base
    `source`, else a short reason. Shapes, dtypes and flags only: no host sync."""
    compiling = torch.compiler.is_compiling()
    if compiling and _FOLD_OP is None:
        return "compiling"
    if not lora_fold_enabled():
        return "disabled"
    if not isinstance(inputs, torch.Tensor) or inputs.device.type != "cuda" or inputs.dim() != 2:
        return "device"
    if type(inputs) not in _PLAIN or type(first) not in _PLAIN or type(second) not in _PLAIN:
        return "subclass"   # DTensor / quantized wrappers keep their own dispatch
    dtype = inputs.dtype
    if dtype not in (torch.bfloat16, torch.float16):
        return "dtype"
    if first.dim() != 3 or second.dim() != 3:
        return "shape"
    E, K, r = first.shape
    if second.shape[0] != E or second.shape[1] != r or inputs.shape[1] != K or E == 0 or second.shape[2] == 0:
        return "shape"
    if r > _mg().LORA_FOLD_MAX_RANK:
        return "rank"
    if first.device != inputs.device or second.device != inputs.device:
        return "device"
    param = _base_param(source)
    if getattr(param, "requires_grad", False):
        return "trainable_base"   # the fold returns no dW
    if getattr(param, "device", inputs.device) != inputs.device:
        return "device"           # an offloaded / other-device stack: the kernel reads every operand on this GPU
    mu = _mu()
    if not (mu.HAS_BNB and mu.Params4bit is not None and isinstance(param, mu.Params4bit)):
        if not isinstance(param, torch.Tensor) or param.dtype != dtype:
            return "base_dtype"
    if mu._triton_grouped_mm_max_rows(inputs.device.index, "lora") == 0:
        return "policy"
    # The fold replaces the base GEMM, so it engages only where that GEMM already takes the generic kernel.
    if not mu._triton_grouped_mm_rows_ok(inputs.device, inputs.shape[0], E, K, second.shape[2]):
        return "policy"
    mg = _mg()
    if compiling:
        # Never run the self-check while tracing, and never key the graph on whether it has run yet (that recompiled
        # once it had): the op body checks at run time and falls back to the separated arithmetic.
        if mg.triton is None or torch.version.hip is not None:
            return "unavailable"
    elif not mg.triton_grouped_available(inputs.device):
        return "unavailable"
    return None


def _fold_mm_impl(inputs, weight, offsets, lora_in, lora_w, scale):
    mg = _mg()
    if mg.triton_grouped_available(inputs.device):
        return mg.generic_grouped_mm_lora(inputs, weight, offsets, lora_in, lora_w, scale)
    mu = _mu()
    mg.GENERIC_CALLS["fallback"] += 1
    return (mu._grouped_mm_eager(inputs, weight, offsets) + mu._grouped_mm_eager(lora_in, lora_w, offsets) * scale).contiguous()


_PLAIN = (torch.Tensor, torch.nn.Parameter)


def _fold_mm(inputs, weight, offsets, lora_in, lora_w, scale):
    """inputs @ weight[e] + scale * lora_in @ lora_w[e]: the opaque op when traced, the kernel directly in eager."""
    if torch.compiler.is_compiling():
        return _FOLD_OP(inputs, weight, offsets, lora_in, lora_w, scale)
    return _mg().generic_grouped_mm_lora(inputs, weight, offsets, lora_in, lora_w, scale)


def _register_fold_op():
    """unsloth_zoo::grouped_mm_lora_fold_v1 for traced callers (the autograd.Function around it is traced, the op
    is opaque); no `from __future__ import annotations` in this module (torch 2.4 schema inference)."""
    if not hasattr(torch, "library") or not hasattr(torch.library, "custom_op"):
        return None
    try:
        return torch.ops.unsloth_zoo.grouped_mm_lora_fold_v1.default
    except (AttributeError, RuntimeError):
        pass
    try:
        @torch.library.custom_op("unsloth_zoo::grouped_mm_lora_fold_v1", mutates_args = ())
        def grouped_mm_lora_fold_v1(inputs: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor,
                                    lora_in: torch.Tensor, lora_w: torch.Tensor, scale: float) -> torch.Tensor:
            return _fold_mm_impl(inputs, weight, offsets, lora_in, lora_w, scale)

        @grouped_mm_lora_fold_v1.register_fake
        def _(inputs, weight, offsets, lora_in, lora_w, scale):
            return inputs.new_empty((inputs.shape[0], lora_w.shape[-1]))

        return torch.ops.unsloth_zoo.grouped_mm_lora_fold_v1.default
    except Exception:
        return None


_FOLD_OP = _register_fold_op()


def _wgrad(inputs, grad, offsets):
    mu = _mu()
    if torch.compiler.is_compiling():
        return torch.ops.unsloth_zoo.grouped_mm_triton_wgrad(inputs.contiguous(), grad.contiguous(), offsets)
    return mu._grouped_mm_triton_wgrad_impl(inputs.contiguous(), grad.contiguous(), offsets)


def _rank_mm(inputs, weight, offsets):
    # The rank-r GEMMs go through the dispatcher (Triton or torch._grouped_mm by policy), as in the separated path.
    with torch.no_grad():
        return _mu()._grouped_mm_with_backward_fix(inputs, weight, offsets).contiguous()


def _weight_t(provider, weight):
    if weight is not None:
        return weight.transpose(-2, -1)
    transposed = getattr(provider, "transposed", None)
    return transposed() if transposed is not None else provider().transpose(-2, -1)


class _GroupedLoraFold(torch.autograd.Function):
    """Y = X @ W + s (X @ A) @ B over a frozen W from weight_provider (rebuilt in backward when recompute), or
    Y = base + s (X @ A) @ B when base_box = [keep_base, base] hands over an already computed base output (a list, so
    the tensor is not an autograd input): written into it, or into a new tensor when keep_base."""

    @staticmethod
    def forward(ctx, inputs, first, second, offsets, scaling, weight_provider, recompute, base_box):
        mg = _mg()
        x = inputs.contiguous()
        t = _rank_mm(x, first, offsets)
        with torch.no_grad():
            if base_box is not None:
                base, keep = base_box.pop(), base_box.pop() if base_box else False
                # keep: the base is held elsewhere (the checkpoint's kept output), so write a new tensor from it.
                dst = torch.empty_like(base) if keep else base
                out = mg.generic_lora_accumulate_(dst, t, second, offsets, scaling, base = base if keep else None)
                weight = None
                LORA_FOLD_STATS["accumulate"] += 1
            else:
                weight = weight_provider()
                out = _fold_mm(x, weight, offsets, t, second, scaling)
                if not torch.compiler.is_compiling():
                    LORA_FOLD_STATS["fold"] += 1
        need_x, need_a, need_b = ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2]
        keep_weight = need_x and weight is not None and not recompute
        ctx.save_for_backward(
            x if need_a else None, t if need_b else None, first if (need_x or need_a) else None,
            second if (need_x or need_a) else None, weight if keep_weight else None, offsets,
        )
        ctx.weight_provider = weight_provider if (need_x and not keep_weight) else None
        ctx.scaling = scaling
        return out

    @staticmethod
    def backward(ctx, grad):
        x, t, first, second, weight, offsets = ctx.saved_tensors
        need_x, need_a, need_b = ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2]
        s = ctx.scaling
        g = grad.contiguous()
        if torch.is_grad_enabled():
            return _differentiable_backward(ctx, g, x, t, first, second, weight, offsets, s)
        gx = ga = gb = None
        with torch.no_grad():
            u = None
            if need_x or need_a:
                u = _rank_mm(g, second.transpose(-2, -1), offsets)
            if need_x:
                weight_t = _weight_t(ctx.weight_provider, weight)
                gx = _fold_mm(g, weight_t, offsets, u, first.transpose(-2, -1), s)
                if not torch.compiler.is_compiling():
                    LORA_FOLD_STATS["backward"] += 1
            if need_a:
                ga = _wgrad(x, u, offsets).mul_(s)
            if need_b:
                gb = _wgrad(t, g, offsets).mul_(s)
        return gx, ga, gb, None, None, None, None, None


def _differentiable_backward(ctx, g, x, t, first, second, weight, offsets, s):
    # create_graph=True: build the same terms from torch._grouped_mm so they can be differentiated again.
    mu = _mu()
    need_x, need_a, need_b = ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2]
    gx = ga = gb = None
    u = mu._grouped_mm_eager(g, second.transpose(-2, -1), offsets) if (need_x or need_a) else None
    if need_x:
        gx = mu._grouped_mm_eager(g, _weight_t(ctx.weight_provider, weight), offsets) \
            + mu._grouped_mm_eager(u, first.transpose(-2, -1), offsets) * s
    if need_a:
        ga = mu._grouped_mm_fp16_wgrad_eager(x, u, offsets) * s
    if need_b:
        gb = mu._grouped_mm_fp16_wgrad_eager(t, g, offsets) * s
    return gx, ga, gb, None, None, None, None, None


def _prepare_lora(inputs, lora):
    first, second, scaling = lora
    if isinstance(scaling, bool) or not isinstance(scaling, (int, float)):
        return None   # a tensor scaling would need a host read
    first = first.to(inputs.dtype).contiguous()
    second = second.to(inputs.dtype).contiguous()
    first, second = _mu()._pad_lora_rank_for_grouped_mm(first, second, inputs)
    return first, second, float(scaling)


def _decline(reason):
    if not torch.compiler.is_compiling():   # a traced global write would re-trace every call
        LORA_FOLD_STATS["declined"] += 1
        DECLINE_REASONS[reason] += 1
        LAST_DECLINE["reason"] = reason
    return None


def lora_fold_base_mm(inputs, offsets, weight_provider, recompute, lora, source):
    """Base GEMM with its LoRA folded in, or None (the caller keeps the separated path)."""
    first, second, scaling = lora
    reason = lora_fold_decline_reason(inputs, first, second, source)
    if reason is not None:
        return _decline(reason)
    prepared = _prepare_lora(inputs, lora)
    if prepared is None:
        return _decline("scaling")
    first, second, scaling = prepared
    return _GroupedLoraFold.apply(inputs, first, second, offsets, scaling, weight_provider, recompute, None)



def lora_fold_prepare(inputs, lora, source):
    """(first, second, scaling) cast and padded for the fold, or None (the separated path applies)."""
    first, second, scaling = lora
    reason = lora_fold_decline_reason(inputs, first, second, source)
    if reason is not None:
        return _decline(reason)
    prepared = _prepare_lora(inputs, lora)
    if prepared is None:
        return _decline("scaling")
    return prepared


def lora_fold_onto_base(inputs, offsets, weight_provider, prepared, base, keep_base = False):
    """base + s (X @ A) @ B (computed earlier, no grad) written into `base`, or into a new tensor when keep_base, with
    the folded backward: dX through the frozen weight rebuilt from weight_provider plus the LoRA term, dA, dB."""
    first, second, scaling = prepared
    return _GroupedLoraFold.apply(inputs, first, second, offsets, scaling, weight_provider, True, [keep_base, base])
