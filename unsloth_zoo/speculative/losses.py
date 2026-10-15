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

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

__all__ = [
    "fastmtp_step_weights",
    "chunked_lm_loss",
    "LOSS_TYPES",
]

LOSS_TYPES = ("ce", "kl_div", "tv", "lk_hybrid")


def fastmtp_step_weights(beta = 0.6, num_steps = 3):
    """FastMTP (arXiv 2509.18362, Eq. 2): normalized beta**k over k = 0..K-1."""
    if num_steps < 1:
        raise ValueError(f"Unsloth: num_steps must be >= 1, got {num_steps}.")
    if not beta > 0:
        raise ValueError(f"Unsloth: beta must be > 0, got {beta}.")
    raw = [beta ** k for k in range(num_steps)]
    total = sum(raw)
    return [w / total for w in raw]


def _logits(hidden, weight, logit_softcap):
    logits = F.linear(hidden.to(weight.dtype), weight)
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    if logit_softcap is not None:
        logits = logit_softcap * torch.tanh(logits / logit_softcap)
    return logits


def _chunk_loss(hidden, labels, teacher_hidden, weight, teacher_weight, loss_type, lk_eta, logit_softcap):
    logits = _logits(hidden, weight, logit_softcap)
    n_correct = (logits.detach().argmax(-1) == labels).sum()
    if loss_type == "ce":
        return F.cross_entropy(logits, labels, reduction = "sum"), n_correct
    with torch.no_grad():
        teacher_logits = _logits(teacher_hidden, teacher_weight, logit_softcap).to(logits.dtype)
        log_p = F.log_softmax(teacher_logits, dim = -1)
        p = log_p.exp()
    log_q = F.log_softmax(logits, dim = -1)
    q = log_q.exp()
    kl = (p * (log_p - log_q)).sum(-1)
    if loss_type == "kl_div":
        return kl.sum(), n_correct
    tv = 0.5 * (p - q).abs().sum(-1)
    if loss_type == "tv":
        return tv.sum(), n_correct
    overlap = torch.minimum(p, q.detach()).sum(-1)
    lam = torch.exp(-lk_eta * overlap)
    return (lam * kl + (1.0 - lam) * tv).sum(), n_correct


LOGITS_BUDGET_BYTES = 256 << 20


def chunked_lm_loss(
    hidden,
    lm_head_weight,
    labels,
    loss_type = "ce",
    teacher_hidden = None,
    teacher_lm_head_weight = None,
    ignore_index = -100,
    chunk_size = 1024,
    lk_eta = 3.0,
    logit_softcap = None,
):
    """Draft-head loss over the vocabulary, a row chunk at a time.

    Returns `(loss_sum, n_tokens, n_correct)` summed over rows whose label is not
    `ignore_index`, so callers can normalize across microbatches. Each chunk's logits are
    recomputed in backward, so peak memory follows `chunk_size`, not the row count.
    """
    if loss_type not in LOSS_TYPES:
        raise ValueError(f"Unsloth: loss_type must be one of {LOSS_TYPES}, got {loss_type!r}.")
    if loss_type != "ce" and teacher_hidden is None:
        raise ValueError(f"Unsloth: loss_type = {loss_type!r} needs teacher_hidden.")
    if chunk_size < 1:
        raise ValueError(f"Unsloth: chunk_size must be >= 1, got {chunk_size}.")
    # A chunk holds ~2 (ce) or ~6 (teacher losses) fp32 [rows, V] tensors: keep that near the budget.
    per_row = lm_head_weight.shape[0] * 4 * (2 if loss_type == "ce" else 6)
    chunk_size = max(1, min(chunk_size, LOGITS_BUDGET_BYTES // per_row))
    hidden = hidden.reshape(-1, hidden.shape[-1])
    labels = labels.reshape(-1).to(hidden.device)
    if hidden.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Unsloth: hidden has {hidden.shape[0]} rows but labels has {labels.shape[0]}."
        )
    if teacher_lm_head_weight is None:
        teacher_lm_head_weight = lm_head_weight
    valid = labels != ignore_index
    hidden = hidden[valid]
    labels = labels[valid]
    if teacher_hidden is not None:
        teacher_hidden = teacher_hidden.reshape(-1, teacher_hidden.shape[-1])[valid.to(teacher_hidden.device)]
        teacher_hidden = teacher_hidden.detach()
    n_tokens = int(labels.shape[0])
    loss_sum = hidden.new_zeros((), dtype = torch.float32 if hidden.dtype in (torch.float16, torch.bfloat16) else hidden.dtype)
    n_correct = None
    if n_tokens == 0:
        # Keep the graph connected so backward on an all-ignored batch is a no-op, not an error.
        return loss_sum + hidden.sum() * 0.0, 0, 0
    needs_grad = torch.is_grad_enabled() and (hidden.requires_grad or lm_head_weight.requires_grad)
    for start in range(0, n_tokens, chunk_size):
        end = min(start + chunk_size, n_tokens)
        args = (
            hidden[start:end],
            labels[start:end],
            None if teacher_hidden is None else teacher_hidden[start:end],
            lm_head_weight,
            teacher_lm_head_weight,
            loss_type,
            lk_eta,
            logit_softcap,
        )
        if needs_grad:
            chunk_sum, chunk_correct = checkpoint(_chunk_loss, *args, use_reentrant = False)
        else:
            chunk_sum, chunk_correct = _chunk_loss(*args)
        loss_sum = loss_sum + chunk_sum.to(loss_sum.dtype)
        n_correct = chunk_correct if n_correct is None else n_correct + chunk_correct
    return loss_sum, n_tokens, int(n_correct)
