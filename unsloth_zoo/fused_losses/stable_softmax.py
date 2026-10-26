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

"""Target log-probabilities without softmax collapse (arXiv 2610.31642, Section 2).

``z_t - logsumexp(z)`` rounds to 0 once the target leads the logsumexp of the others by ~16.6 nats in
float32: the denominator is 1 + exp(-margin) and the small term is lost, so the loss and the target gradient
become exactly 0 while every other class keeps its gradient. Excluding the target *before* the reduction keeps
both: ``log p_t = -softplus(logsumexp_{j != t}(z_j) - z_t)`` and ``d log p_t / d z_t = -expm1(log p_t)``.
"""

import torch
import torch.nn.functional as F

__all__ = [
    "target_logprob",
    "target_logprob_",
    "target_grad",
    "smoothed_nll",
]


def _others_lse(s_others, m):
    # log(s) with a zero-safe argument: an empty support (only the target kept) gives -inf, and log's
    # backward at 0 would turn the zero upstream gradient into NaN.
    nonzero = s_others > 0
    safe = torch.where(nonzero, s_others, torch.ones_like(s_others))
    return torch.where(nonzero, m + torch.log(safe), torch.full_like(m, -float("inf")))


def target_logprob(z, target):
    """(log softmax(z)[..., target], logsumexp(z)) over the last dim, traceable and autograd exact.

    ``target`` must index a column of every row (map ignored labels to 0 first and mask the result); columns
    set to -inf (a truncated support) drop out. One masked reduction, the same work as logsumexp.
    """
    m = z.detach().amax(-1, keepdim = True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    index = target.unsqueeze(-1)
    z_t = z.gather(-1, index)
    is_target = torch.arange(z.shape[-1], device = z.device) == index
    s_others = torch.exp(z - m).masked_fill(is_target, 0.0).sum(-1, keepdim = True)
    logp = -F.softplus(_others_lse(s_others, m) - z_t)
    return logp.squeeze(-1), (z_t - logp).squeeze(-1)


@torch.no_grad()
def target_logprob_(z, target):
    """``target_logprob`` for hand-written forwards: overwrites ``z`` (a local float chunk) in place, so take
    every other statistic of ``z`` first. Returns (logp, lse) in z's dtype."""
    m = z.amax(-1, keepdim = True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    index = target.unsqueeze(-1)
    z_t = z.gather(-1, index)
    z.sub_(m).exp_()
    z.scatter_(-1, index, 0.0)
    s_others = z.sum(-1, keepdim = True)
    logp = -F.softplus(_others_lse(s_others, m) - z_t)
    return logp.squeeze(-1), (z_t - logp).squeeze(-1)


def target_grad(logp):
    """d log p_t / d z_t = 1 - p_t, from an accurate log p_t (``1 - exp(logp)`` cancels to 0)."""
    return -torch.expm1(logp)


def smoothed_nll(z, target, label_smoothing = 0.0):
    """Per row F.cross_entropy(z, target, label_smoothing) (uniform mass over all V classes), collapse free."""
    logp, lse = target_logprob(z, target)
    nll = -logp
    if label_smoothing:
        nll = (1.0 - label_smoothing) * nll + label_smoothing * (lse - z.mean(-1))
    return nll
