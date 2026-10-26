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

__all__ = [
    "target_logprob",
    "target_logprob_",
    "target_grad",
    "smoothed_nll",
    "cross_entropy_rows",
]


def _softplus(x):
    # Exact log(1 + exp(x)): F.softplus returns x itself past its threshold of 20, off by up to 2e-9.
    return x.clamp_min(0) + torch.log1p(torch.exp(-x.abs()))


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
    logp = -_softplus(_others_lse(s_others, m) - z_t)
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
    logp = -_softplus(_others_lse(s_others, m) - z_t)
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


# float32 bytes per chunk of the eager row loop: bounds the transient upcast of half precision logits.
_ROWS_CHUNK_BYTES = 256 * 1024 * 1024


def _row_chunks(n_rows, vocab_size):
    if torch.compiler.is_compiling():
        return n_rows  # one chunk: Inductor fuses the upcast instead of materialising it
    return max(1, min(n_rows, _ROWS_CHUNK_BYTES // max(4 * vocab_size, 1)))


class _CrossEntropyRows(torch.autograd.Function):
    """Per row cross entropy of logits [N, V] in any dtype, collapse free, saving only the logits and two
    row vectors (stock F.cross_entropy saves a same sized log_softmax), float32 math one chunk at a time."""

    @staticmethod
    def forward(ctx, logits, target):
        n_rows, vocab_size = logits.shape
        rows = _row_chunks(n_rows, vocab_size)
        logp = torch.empty(n_rows, dtype = torch.float32, device = logits.device)
        lse = torch.empty(n_rows, dtype = torch.float32, device = logits.device)
        for start in range(0, n_rows, rows):
            end = min(start + rows, n_rows)
            chunk = logits[start:end].to(torch.float32, copy = True)
            logp[start:end], lse[start:end] = target_logprob_(chunk, target[start:end])
        ctx.save_for_backward(logits, target, logp, lse)
        return -logp

    @staticmethod
    def backward(ctx, grad_nll):
        logits, target, logp, lse = ctx.saved_tensors
        n_rows, vocab_size = logits.shape
        rows = _row_chunks(n_rows, vocab_size)
        grad = torch.empty_like(logits)
        for start in range(0, n_rows, rows):
            end = min(start + rows, n_rows)
            # d nll / dz = p - onehot; the target entry p_t - 1 = expm1(logp) keeps a confident target.
            p = torch.exp(logits[start:end].to(torch.float32) - lse[start:end].unsqueeze(-1))
            p.scatter_(-1, target[start:end].unsqueeze(-1), torch.expm1(logp[start:end]).unsqueeze(-1))
            grad[start:end] = p.mul_(grad_nll[start:end].to(torch.float32).unsqueeze(-1))
        return grad, None


def cross_entropy_rows(logits, target, ignore_index = -100):
    """Per row float32 nll of logits [N, V] (0 on ignored rows), F.cross_entropy(reduction = "none") without
    softmax collapse and without a float32 copy of the logits."""
    keep = target != ignore_index
    safe = torch.where(keep, target, torch.zeros_like(target))
    nll = _CrossEntropyRows.apply(logits, safe)
    return torch.where(keep, nll, torch.zeros_like(nll))
