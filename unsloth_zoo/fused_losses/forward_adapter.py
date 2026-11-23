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


def _head_adapter_plan(lm_head):
    """How the fused CE reproduces a head's forward: (None, ()) base weight as is, ("fold", adapters) plain LoRA
    folded into the kernel inputs, ("module", reason) only the head's own forward can (DoRA, lora_bias, other LoRA
    variants, fan_in_fan_out, non-LoRA tuners)."""
    if type(lm_head) is torch.nn.Linear or getattr(lm_head, "base_layer", None) is None:
        return None, ()
    # PEFT's forward returns base_layer(x) alone once any adapter is merged or adapters are disabled.
    if getattr(lm_head, "merged", False) or getattr(lm_head, "disable_adapters", False):
        return None, ()
    active = getattr(lm_head, "active_adapters", None) or ()
    if isinstance(active, str):
        active = (active,)
    lora_A = getattr(lm_head, "lora_A", None)
    if lora_A is None:
        held = set()
        for layer_name in getattr(lm_head, "adapter_layer_names", None) or ():
            held.update(getattr(lm_head, layer_name, None) or ())
        if any(name in held for name in active):
            return "module", type(lm_head).__name__
        return None, ()
    adapters = tuple(name for name in active if name in lora_A)
    if not adapters:
        return None, ()
    if getattr(lm_head, "fan_in_fan_out", False):
        return "module", "fan_in_fan_out LoRA"
    use_dora = getattr(lm_head, "use_dora", None) or {}
    lora_bias = getattr(lm_head, "lora_bias", None) or {}
    lora_variant = getattr(lm_head, "lora_variant", None) or {}
    for name in adapters:
        if use_dora.get(name, False):
            return "module", "DoRA"
        if lora_bias.get(name, False):
            return "module", "LoRA with lora_bias"
        if name in lora_variant:
            return "module", f"LoRA variant {type(lora_variant[name]).__name__}"
    return "fold", adapters


_WARNED_MODULE_HEADS = set()


@torch.compiler.disable
def _warn_module_head(reason):
    if reason in _WARNED_MODULE_HEADS:
        return
    _WARNED_MODULE_HEADS.add(reason)
    print(
        f"Unsloth: the LM head carries a {reason} adapter, which the fused cross entropy kernel cannot apply, so "
        "the loss calls the head's own forward and materialises the full logits (more memory). A plain LoRA "
        "adapter or modules_to_save on lm_head keeps the fused loss."
    )


def lm_head_needs_module_forward(lm_head):
    """True when only lm_head.forward() gives the right logits (a live adapter the fused kernel cannot fold)."""
    if type(lm_head) is torch.nn.Linear:
        return False
    plan, reason = _head_adapter_plan(lm_head)
    if plan == "module":
        _warn_module_head(reason)
        return True
    return False


def _fold_lora_head(lm_head, base_hidden_states, hidden_states, weight, adapters):
    """Logits of base + LoRA as one matmul: [x, s * A(dropout(x))] @ [W, B]^T. Exact, dropout included, and it
    keeps the head fused (r extra columns, never a V x V or full logits tensor)."""
    # PEFT feeds the adapters the head's raw input; only the base layer sees its own activation fake quantizer.
    xs, ws = [base_hidden_states], [weight]
    for name in adapters:
        A = lm_head.lora_A[name]
        B = lm_head.lora_B[name]
        dropout = lm_head.lora_dropout[name] if name in getattr(lm_head, "lora_dropout", {}) else None
        x = hidden_states.to(A.weight.dtype)
        if dropout is not None:
            x = dropout(x)
        xs.append((A(x) * lm_head.scaling[name]).to(base_hidden_states.dtype))
        ws.append(B.weight.to(weight.dtype))
    return torch.cat(xs, dim = -1), torch.cat(ws, dim = 1)


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
    **kwargs,
):
    """unsloth_fused_ce_loss's loss with the logits from lm_head(hidden_states): same label shift and mask, divisor
    (n_items, else the labelled token count), fp32 logit scale / softcap and label smoothing. For heads whose
    forward the kernel cannot fold (lm_head_needs_module_forward); memory heavier but exact."""
    if _ce._LABEL_SMOOTHING.active:
        n_items, smoothing = _ce._apply_label_smoothing_override(n_items, {"label_smoothing": label_smoothing})
        label_smoothing = smoothing["label_smoothing"]
    logits = lm_head(hidden_states)
    labels = labels.to(logits.device)
    if shift_labels:
        _labels = torch.full_like(labels, ignore_index)
        _labels[..., :-1] = labels[..., 1:]
        if mask is not None:
            _labels[..., :-1][mask.to(logits.device)[..., 1:] == 0] = ignore_index
        labels = _labels
    logits = logits.float()
    if logits_bias is not None:
        logits = logits + logits_bias.float()
    if logit_scale_multiply:
        logits = logits * logit_scale_multiply
    if logit_scale_divide:
        logits = logits / logit_scale_divide
    if logit_softcapping:
        logits = torch.tanh(logits / logit_softcapping) * logit_softcapping
    labels = labels.reshape(-1)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(labels.shape[0], -1), labels, ignore_index = ignore_index, reduction = "sum",
        label_smoothing = float(label_smoothing or 0.0),
    )
    divisor = n_items if n_items is not None else (labels != ignore_index).sum()
    if not torch.is_tensor(divisor):
        divisor = torch.full((), divisor, dtype = torch.float32, device = loss.device)
    if divisor.numel() != 1:
        divisor = divisor.ravel()[0]
    return loss / _ce._nonzero_count(divisor.to(dtype = torch.float32, device = loss.device))


def lm_head_fake_quantize(lm_head, hidden_states):
    """(hidden_states, weight) whose product is lm_head.forward()'s logits: the TorchAO QAT QDQ that the head's
    forward applies and fused CE skips, plus any live PEFT LoRA adapter on the head, which the fused kernel would
    otherwise drop (wrong loss, no adapter gradient)."""
    weight = lm_head.weight
    if type(lm_head) is torch.nn.Linear:
        return hidden_states, weight
    plan, adapters = _head_adapter_plan(lm_head)
    if plan == "fold":
        base_hidden_states, weight = lm_head_fake_quantize(lm_head.base_layer, hidden_states)
        return _fold_lora_head(lm_head, base_hidden_states, hidden_states, weight, adapters)
    # plan == "module": callers route such heads through unsloth_module_head_ce_loss / their exact branch first.
    if getattr(lm_head, "base_layer", None) is not None:
        lm_head = lm_head.base_layer
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
        # An adapter the kernel cannot fold (DoRA, ...): the same loss through the head's own forward.
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
