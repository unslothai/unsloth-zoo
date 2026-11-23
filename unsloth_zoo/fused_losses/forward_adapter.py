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

"""HF-call-convention adapter for unsloth_fused_ce_loss.

The fused kernel takes hidden_states + lm_head.weight directly, skipping the
lm_head matmul and fp32 logits materialisation the HF template does before
`self.loss_function(...)`. `EMPTY_LOGITS` is the 0-element sentinel for the
`logits=` return field.
"""

from __future__ import annotations

__all__ = [
    "EMPTY_LOGITS",
    "lm_head_fake_quantize",
    "lm_head_live_adapter",
    "lm_head_needs_module_forward",
    "unsloth_fused_lm_head_loss",
    "unsloth_module_head_ce_loss",
]

import inspect

import torch

from . import cross_entropy_loss as _ce
from .cross_entropy_loss import unsloth_fsdp_unshard, unsloth_fused_ce_loss


EMPTY_LOGITS = torch.empty(0)

# Kwargs the fused CE reads; extras (HF output_attentions / output_hidden_states) make _use_chunked_ce refuse the kernel.
_KERNEL_KWARGS = frozenset(_ce._FUSED_LINEAR_CE_KWARGS) | frozenset(("n_chunks",)) | frozenset(
    inspect.signature(_ce.unsloth_fused_ce_loss).parameters
) - frozenset(("trainer", "hidden_states", "lm_head_weight", "lm_head_bias", "labels", "n_items", "shift_labels",
               "kwargs"))


def lm_head_on_old_cuda_arch(weight):
    """Replaced by unsloth_zoo.compiler.lm_head_on_old_cuda_arch, which owns the pre-Turing card set."""
    return False


def _adapter_names_hooked(lm_head):
    """PEFT's per-example `adapter_names` pre-hook: each row picks its own adapter inside the head's forward."""
    for hook in getattr(lm_head, "_forward_pre_hooks", {}).values():
        func = getattr(hook, "func", hook)
        if getattr(func, "__name__", "") == "_adapter_names_pre_forward_hook":
            return True
    return False


def lm_head_live_adapter(lm_head):
    """Why lm_head.forward() differs from a matmul with lm_head.weight (a PEFT tuner layer whose forward adds an
    adapter), else None. lm_head.weight on such a layer is the base layer's weight, so the fused kernel would drop
    the adapter: no gradient for it and the base head's loss."""
    if type(lm_head) is torch.nn.Linear or getattr(lm_head, "base_layer", None) is None:
        return None
    cls = type(lm_head)
    name = f"{cls.__module__}.{cls.__name__}"
    merged = bool(getattr(lm_head, "merged", False))
    if getattr(lm_head, "disable_adapters", False):
        # Disabled: forward unmerges a merged adapter first, so only the unmerged state reads the base weight.
        return f"{name} (merged, then disabled)" if merged else None
    if _adapter_names_hooked(lm_head):
        return f"{name} (per-example adapter_names)"
    if merged:
        # PEFT's forward is base_layer(x) once merged: the merged weight is what the kernel reads.
        return None
    active = getattr(lm_head, "active_adapters", None)
    if active is None:
        active = getattr(lm_head, "active_adapter", None)
    if isinstance(active, str):
        active = (active,)
    held = set()
    for layer_name in getattr(lm_head, "adapter_layer_names", None) or ():
        held.update(getattr(lm_head, layer_name, None) or ())
    if any(adapter in held for adapter in (active or ())):
        return name
    return None


_WARNED_MODULE_HEADS = set()


def _warn_module_head(reason):
    if reason in _WARNED_MODULE_HEADS:
        return
    _WARNED_MODULE_HEADS.add(reason)
    print(
        f"Unsloth: the LM head carries a live adapter ({reason}), which the fused cross entropy kernel cannot "
        "apply, so the loss calls the head's own forward and materialises the full logits (more memory). "
        "modules_to_save on lm_head, or merging the adapter, keeps the fused loss."
    )


def lm_head_needs_module_forward(lm_head):
    """True when only lm_head.forward() gives the right logits: a live PEFT adapter on the head."""
    if type(lm_head) is torch.nn.Linear:
        return False
    reason = lm_head_live_adapter(lm_head)
    if reason is None:
        return False
    # Printing inside a trace would break fullgraph capture; the eager call (or the first warm-up) prints it.
    if not torch.compiler.is_compiling():
        _warn_module_head(reason)
    return True


def _align_lm_head_input(lm_head, hidden_states):
    """Hidden states on the head's device, as the fused kernel moves them; accelerate-hooked heads move their own."""
    weight = getattr(lm_head, "weight", None)
    if (
        weight is not None
        and weight.device.type != "meta"
        and hidden_states.device != weight.device
        and getattr(lm_head, "_hf_hook", None) is None
    ):
        return hidden_states.to(device = weight.device)
    return hidden_states


def _scale_logits(logits, multiply, divide, softcap):
    """The kernel's fp32 logit transforms; 0 means off. Tensor scales stay traceable (no Python truth test)."""
    if torch.is_tensor(multiply):
        mul = multiply.to(logits)
        logits = logits * torch.where(mul == 0, torch.ones_like(mul), mul)
    elif multiply:
        logits = logits * multiply
    if torch.is_tensor(divide):
        div = divide.to(logits)
        logits = logits / torch.where(div == 0, torch.ones_like(div), div)
    elif divide:
        logits = logits / divide
    if torch.is_tensor(softcap):
        cap = softcap.to(logits)
        safe = torch.where(cap == 0, torch.ones_like(cap), cap)
        logits = torch.where(cap == 0, logits, torch.tanh(logits / safe) * safe)
    elif softcap:
        logits = torch.tanh(logits / softcap) * softcap
    return logits


def unsloth_module_head_ce_loss(
    hidden_states,
    lm_head,
    labels,
    mask = None,
    n_items = None,
    shift_labels = True,
    logit_scale_multiply = 0,
    logit_scale_divide = 0,
    logit_softcapping = 0,
    logits_bias = None,
    ignore_index = -100,
    label_smoothing = 0.0,
    precomputed_logits = None,
    return_logits = False,
    **kwargs,
):
    """unsloth_fused_ce_loss's loss with the logits from lm_head(hidden_states), for heads whose forward the kernel
    cannot reproduce (lm_head_needs_module_forward): same targets (shift, mask on shifted or pre-aligned labels),
    divisor (n_items, else the labelled token count; an all-ignored batch gives 0), fp32 logit scale / softcap,
    extra vocab bias and label smoothing (including the trainer's override). Memory heavier but exact.

    precomputed_logits: the head output the caller already has (one projection, one dropout draw).
    return_logits: also return the transformed logits."""
    from .chunked_logprob import _ce_divisor, _ce_targets

    if _ce._LABEL_SMOOTHING.active:
        n_items, smoothing = _ce._apply_label_smoothing_override(n_items, {"label_smoothing": label_smoothing})
        label_smoothing = smoothing["label_smoothing"]
    logits = precomputed_logits
    if logits is None:
        logits = lm_head(_align_lm_head_input(lm_head, hidden_states))
    if logits_bias is not None:
        logits_bias = logits_bias.to(device = logits.device)
        logits = logits.to(torch.promote_types(logits.dtype, logits_bias.dtype)) + logits_bias
    logits = _scale_logits(logits.float(), logit_scale_multiply, logit_scale_divide, logit_softcapping)
    targets = _ce_targets(labels.to(dtype = torch.long), mask, shift_labels, ignore_index, logits.device)
    # Keeps the vocab width: a token-count mismatch with the targets raises, as the stock loss does.
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets, ignore_index = ignore_index, reduction = "sum",
        label_smoothing = float(label_smoothing or 0.0),
    )
    loss = loss / _ce_divisor(n_items, targets, ignore_index, loss.device)
    return (loss, logits) if return_logits else loss


def lm_head_fake_quantize(lm_head, hidden_states):
    """(hidden_states, weight) after the TorchAO QAT QDQ that lm_head.forward() applies and fused CE skips."""
    weight = lm_head.weight
    if type(lm_head) is torch.nn.Linear:
        return hidden_states, weight
    activation_fake_quantizer = getattr(lm_head, "activation_fake_quantizer", None)
    if activation_fake_quantizer is not None:
        hidden_states = activation_fake_quantizer(hidden_states)
    weight_fake_quantizer = getattr(lm_head, "weight_fake_quantizer", None)
    if weight_fake_quantizer is not None:
        weight = weight_fake_quantizer(weight)
    return hidden_states, weight


def unsloth_fused_lm_head_loss(
    hidden_states,
    lm_head,
    labels,
    vocab_size=None,
    **kwargs,
):
    """Replacement for the canonical `self.loss_function(logits=..., labels=...,
    vocab_size=..., **kwargs)` call site, routing through the chunked fused CE
    kernel without materialising fp32 logits.

    Args:
        hidden_states: tensor that was about to be fed into `self.lm_head`.
        lm_head: the lm_head Linear module; weight + bias pulled off it.
        labels: integer label tensor.
        vocab_size: ignored; kernel reads it from `lm_head.weight.shape[0]`.
        **kwargs: forwarded to the kernel. `num_items_in_batch` is renamed to
            `n_items`; also accepts `logit_scale_multiply`, `logit_scale_divide`,
            `logit_softcapping`, `logits_bias` (an extra vocab bias added after the
            head), plus extras the original loss_function ignored.
    """
    n_items = kwargs.pop("num_items_in_batch", None)
    if n_items is None:
        n_items = kwargs.pop("n_items", None)
    else:
        kwargs.pop("n_items", None)
    # vocab_size is read from lm_head_weight.shape[0]; drop the keyword.
    kwargs.pop("vocab_size", None)
    # Caller may pass a pre-shifted target (e.g. trl padding_free + packing
    # gives shift_labels=<tensor>); route it through with shift_labels=False.
    shift_labels_kw = kwargs.pop("shift_labels", None)
    pre_shifted_tensor = (
        shift_labels_kw is not None and not isinstance(shift_labels_kw, bool)
    )
    if pre_shifted_tensor:
        target = shift_labels_kw
        do_shift = False
    elif shift_labels_kw is False:
        target = labels
        do_shift = False
    else:
        target = labels
        do_shift = True

    # fp16: grads are computed in the forward and underflow without the loss scale (#3529).
    accelerator_scaler = kwargs.pop("accelerator_scaler", None)
    if kwargs.get("scaling", None) is None and accelerator_scaler is not None:
        kwargs["scaling"] = accelerator_scaler

    kwargs = {k: v for k, v in kwargs.items() if k in _KERNEL_KWARGS}

    unsloth_fsdp_unshard(lm_head)
    if lm_head_needs_module_forward(lm_head):
        # A live PEFT adapter on the head: the same loss through the head's own forward.
        loss = unsloth_module_head_ce_loss(
            hidden_states, lm_head, target, n_items = n_items, shift_labels = do_shift, **kwargs,
        )
        return loss.clone()
    # A head on a pre-Turing card skips Inductor. Asked after unshard: FSDP2 CPUOffloadPolicy keeps
    # the shard on CPU until then.
    if kwargs.get("torch_compile", True) and lm_head_on_old_cuda_arch(lm_head.weight):
        kwargs["torch_compile"] = False
    hidden_states, weight = lm_head_fake_quantize(lm_head, hidden_states)
    loss = unsloth_fused_ce_loss(
        trainer        = None,
        hidden_states  = hidden_states,
        lm_head_weight = weight,
        lm_head_bias   = getattr(lm_head, "bias", None),
        labels         = target,
        n_items        = n_items,
        shift_labels   = do_shift,
        **kwargs,
    )
    # The autograd.Function output is a view; MoE forwards then do an in-place
    # loss += aux_loss, which autograd rejects. Return a non-view tensor.
    if isinstance(loss, torch.Tensor):
        loss = loss.clone()
    return loss
