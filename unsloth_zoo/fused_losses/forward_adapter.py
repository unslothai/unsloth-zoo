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
    vocab_size=..., **kwargs)` call site, routing through the chunked fused CE
    kernel without materialising fp32 logits.

    Args:
        hidden_states: tensor that was about to be fed into `self.lm_head`.
        lm_head: the lm_head Linear module; weight + bias pulled off it.
        labels: integer label tensor.
        vocab_size: ignored; kernel reads it from `lm_head.weight.shape[0]`.
        **kwargs: forwarded to the kernel. `num_items_in_batch` is renamed to
            `n_items`; also accepts `logit_scale_multiply`, `logit_scale_divide`,
            `logit_softcapping`, plus extras the original loss_function ignored.
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

    loss = unsloth_fused_ce_loss(
        trainer        = None,
        hidden_states  = hidden_states,
        lm_head_weight = lm_head.weight,
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
