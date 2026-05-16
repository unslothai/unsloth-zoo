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

The fused kernel takes hidden_states + lm_head.weight directly, skipping
both the lm_head matmul and the fp32 logits materialisation that the HF
template performs before `self.loss_function(...)`. `EMPTY_LOGITS` is the
0-element sentinel slotted into the `logits=` return field.
"""

from __future__ import annotations

__all__ = [
    "EMPTY_LOGITS",
    "unsloth_fused_lm_head_loss",
]

import torch

from .cross_entropy_loss import unsloth_fused_ce_loss


EMPTY_LOGITS = torch.empty(0)


def unsloth_fused_lm_head_loss(
    hidden_states,
    lm_head,
    labels,
    vocab_size=None,
    **kwargs,
):
    """Replacement for the canonical `self.loss_function(logits=..., labels=...,
    vocab_size=..., **kwargs)` call site. Routes through the chunked fused
    cross-entropy kernel without materialising fp32 logits.

    Args:
        hidden_states: the tensor that was about to be fed into `self.lm_head`.
        lm_head: the lm_head module (Linear). Weight + bias pulled off it.
        labels: integer label tensor.
        vocab_size: ignored. Kernel reads it from `lm_head.weight.shape[0]`.
        **kwargs: forwarded to the kernel. Accepts `num_items_in_batch`
            (renamed to `n_items`), `logit_scale_multiply`, `logit_scale_divide`,
            `logit_softcapping`, plus any other extras the original
            `self.loss_function` would have ignored.
    """
    n_items = kwargs.pop("num_items_in_batch", None)
    if n_items is None:
        n_items = kwargs.pop("n_items", None)
    else:
        kwargs.pop("n_items", None)
    # vocab_size is read from lm_head_weight.shape[0]; drop the keyword.
    kwargs.pop("vocab_size", None)
    # Caller already shifted (either `shift_labels=<tensor>` or `shift_labels=False`):
    # the fused kernel always re-shifts, so route to stock CE for correctness.
    shift_labels_kw = kwargs.pop("shift_labels", None)
    pre_shifted_tensor = (
        shift_labels_kw is not None and not isinstance(shift_labels_kw, bool)
    )
    if pre_shifted_tensor or shift_labels_kw is False:
        logits = torch.nn.functional.linear(
            hidden_states.to(dtype=lm_head.weight.dtype, device=lm_head.weight.device),
            lm_head.weight,
            getattr(lm_head, "bias", None),
        )
        ignore_index = int(kwargs.get("ignore_index", -100))
        label_smoothing = float(kwargs.get("label_smoothing", 0.0))
        target = shift_labels_kw if pre_shifted_tensor else labels
        reduction = "sum" if n_items is not None else "mean"
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]).float(),
            target.view(-1).to(logits.device),
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction=reduction,
        )
        if n_items is not None:
            if torch.is_tensor(n_items):
                n_items = n_items.to(device=loss.device, dtype=loss.dtype)
            loss = loss / n_items
        return loss

    return unsloth_fused_ce_loss(
        trainer        = None,
        hidden_states  = hidden_states,
        lm_head_weight = lm_head.weight,
        lm_head_bias   = getattr(lm_head, "bias", None),
        labels         = labels,
        n_items        = n_items,
        **kwargs,
    )
