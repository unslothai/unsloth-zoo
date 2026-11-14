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
    "unsloth_fused_lm_head_loss",
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
