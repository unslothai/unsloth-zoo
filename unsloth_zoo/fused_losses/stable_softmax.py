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

"""Target log-probabilities without softmax collapse (arXiv 2610.31642, Section 2): ``z_t - logsumexp(z)``
rounds to 0 past a ~16.6 nat margin in float32, so ``log p_t = -softplus(logsumexp_{j != t}(z_j) - z_t)``."""

import torch

__all__ = [
    "target_logprob",
    "target_logprob_",
    "target_grad",
    "smoothed_nll",
    "cross_entropy_rows",
]


def _softplus(x):
    # Not F.softplus (returns x past 20), clamp_min(x, 0) + ... (derivative 1 at 0) or logaddexp (backward
    # underflows for x << 0). The clamps keep the untaken branch finite.
    positive = x > 0
    return torch.where(
        positive, x + torch.log1p(torch.exp(-x.clamp_min(0))), torch.log1p(torch.exp(x.clamp_max(0))),
    )


def _centred(z, target):
    """(log p_t, logsumexp(z) - m, m, z - m) with m the detached row max. Centred so a shared offset cannot
    round the normaliser away; the normaliser is logaddexp(lse_others, z_t), never z_t - logp (cancels when
    the target trails); target by masked sum, not gather (elementwise backward, no saved float32 copy);
    finfo.min, not -inf, keeps a target-only support NaN free."""
    m = z.detach().amax(-1, keepdim = True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    z = z - m
    is_target = torch.arange(z.shape[-1], device = z.device) == target.unsqueeze(-1)
    z_t = torch.where(is_target, z, 0.0).sum(-1)
    lse_others = torch.logsumexp(torch.where(is_target, torch.finfo(z.dtype).min, z), -1)
    logp = -_softplus(lse_others - z_t)
    return logp, torch.logaddexp(lse_others, z_t), m.squeeze(-1), z


def target_logprob(z, target):
    """(log softmax(z)[..., target], logsumexp(z)), traceable. ``target`` must be a valid column of every row."""
    logp, lse_c, m, _ = _centred(z, target)
    return logp, lse_c + m


def _others_lse(s_others, m):
    # An empty support (only the target kept) gives -inf.
    nonzero = s_others > 0
    safe = torch.where(nonzero, s_others, torch.ones_like(s_others))
    return torch.where(nonzero, m + torch.log(safe), torch.full_like(m, -float("inf")))


@torch.no_grad()
def target_logprob_(z, target):
    """``target_logprob`` that overwrites ``z`` (a local float chunk): take every other statistic of ``z`` first."""
    m = z.amax(-1, keepdim = True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    index = target.unsqueeze(-1)
    z_t = z.gather(-1, index)
    z.sub_(m).exp_()
    z.scatter_(-1, index, 0.0)
    s_others = z.sum(-1, keepdim = True)
    lse_others = _others_lse(s_others, m)
    logp = -_softplus(lse_others - z_t)
    return logp.squeeze(-1), torch.logaddexp(lse_others, z_t).squeeze(-1)


def target_grad(logp):
    """d log p_t / d z_t; ``1 - exp(logp)`` would cancel to 0."""
    return -torch.expm1(logp)


def smoothed_nll(z, target, label_smoothing = 0.0):
    """Per row F.cross_entropy(z, target, label_smoothing = label_smoothing)."""
    logp, lse_c, _, z_c = _centred(z, target)
    nll = -logp
    if label_smoothing:
        nll = (1.0 - label_smoothing) * nll + label_smoothing * (lse_c - z_c.mean(-1))
    return nll


# Bounds the float32 upcast of half precision logits in the eager row loop.
_ROWS_CHUNK_BYTES = 256 * 1024 * 1024


def _row_chunks(n_rows, vocab_size):
    if torch.compiler.is_compiling():
        return max(n_rows, 1)  # one chunk: Inductor fuses the upcast instead of materialising it
    return max(1, min(n_rows, _ROWS_CHUNK_BYTES // max(4 * vocab_size, 1)))


def _acc(dtype):
    return torch.promote_types(dtype, torch.float32)  # half -> float32, float64 stays float64


def _rows_forward(chunk, target):
    # Saves the row max and the logsumexp relative to it: an absolute lse rounds away the normaliser once
    # the logits share a large offset, and exp(z - lse) would then disagree with the forward.
    logp, lse_c, m, _ = _centred(chunk.to(_acc(chunk.dtype)), target)
    return logp, lse_c, m


def _rows_backward(chunk, target, logp, lse_c, m, grad_nll):
    # d nll / dz = p - onehot, with p_t - 1 = expm1(logp) for the target.
    z = chunk.to(logp.dtype)
    p = torch.exp((z - m.unsqueeze(-1)) - lse_c.unsqueeze(-1))
    is_target = torch.arange(z.shape[-1], device = z.device) == target.unsqueeze(-1)
    grad = torch.where(is_target, torch.expm1(logp).unsqueeze(-1), p) * grad_nll.to(logp.dtype).unsqueeze(-1)
    return grad.to(chunk.dtype)


@torch.no_grad()
def _rows_forward_eager(chunk, target):
    z = chunk.to(_acc(chunk.dtype), copy = True)
    m = z.amax(-1)
    logp, lse_c = target_logprob_(z.sub_(m.unsqueeze(-1)), target)
    return logp, lse_c, m


def _rows_backward_eager(chunk, target, logp, lse_c, m, grad_nll):
    p = torch.exp((chunk.to(logp.dtype) - m.unsqueeze(-1)) - lse_c.unsqueeze(-1))
    p.scatter_(-1, target.unsqueeze(-1), torch.expm1(logp).unsqueeze(-1))
    return p.mul_(grad_nll.to(logp.dtype).unsqueeze(-1)).to(chunk.dtype)


_COMPILED_ROWS = {}


def _rows_impl(logits):
    """(forward, backward) per chunk: one fused Inductor kernel each on GPUs (eager here reads the chunk ~6
    times, 1.9x stock's time on an A100), the in place eager pair elsewhere or when compiling fails."""
    if torch.compiler.is_compiling():
        return _rows_forward, _rows_backward
    if logits.device.type not in ("cuda", "xpu") or _COMPILED_ROWS.get("failed") or _compile_disabled():
        return _rows_forward_eager, _rows_backward_eager
    if not _COMPILED_ROWS:
        from unsloth_zoo.temporary_patches.common import torch_compile_options
        _COMPILED_ROWS.update({
            name: torch.compile(fn, dynamic = True, fullgraph = True, options = torch_compile_options)
            for name, fn in (("forward", _rows_forward), ("backward", _rows_backward))
        })
    return _COMPILED_ROWS["forward"], _COMPILED_ROWS["backward"]


def _compile_disabled():
    from unsloth_zoo.temporary_patches.common import UNSLOTH_COMPILE_DISABLE
    return bool(UNSLOTH_COMPILE_DISABLE)


def _run_rows(step, eager_step, *args):
    try:
        return step(*args)
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception:
        if step is eager_step or torch.compiler.is_compiling():
            raise
        _COMPILED_ROWS["failed"] = True  # no Triton / compiler on this host: stay eager from now on
        return eager_step(*args)


class _CrossEntropyRows(torch.autograd.Function):
    """Saves the logits and two row vectors; stock F.cross_entropy saves a same sized log_softmax."""

    @staticmethod
    def forward(ctx, logits, target):
        n_rows, vocab_size = logits.shape
        rows = _row_chunks(n_rows, vocab_size)
        forward, _ = _rows_impl(logits)
        logp, lse_c, m = (torch.empty(n_rows, dtype = _acc(logits.dtype), device = logits.device) for _ in range(3))
        for start in range(0, n_rows, rows):
            end = min(start + rows, n_rows)
            logp[start:end], lse_c[start:end], m[start:end] = _run_rows(
                forward, _rows_forward_eager, logits[start:end], target[start:end])
        ctx.save_for_backward(logits, target, logp, lse_c, m)
        return -logp

    @staticmethod
    def backward(ctx, grad_nll):
        logits, target, logp, lse_c, m = ctx.saved_tensors
        n_rows, vocab_size = logits.shape
        rows = _row_chunks(n_rows, vocab_size)
        _, backward = _rows_impl(logits)
        grad = torch.empty_like(logits)
        for start in range(0, n_rows, rows):
            end = min(start + rows, n_rows)
            grad[start:end] = _run_rows(
                backward, _rows_backward_eager,
                logits[start:end], target[start:end], logp[start:end], lse_c[start:end], m[start:end],
                grad_nll[start:end])
        return grad, None


def cross_entropy_rows(logits, target, ignore_index = -100):
    """Per row nll of logits [N, V] (float32, float64 for float64 logits), 0 on ignored rows."""
    keep = target != ignore_index
    safe = torch.where(keep, target, torch.zeros_like(target))
    nll = _CrossEntropyRows.apply(logits, safe)
    return torch.where(keep, nll, torch.zeros_like(nll))
