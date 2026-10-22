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

"""Per-token log-probabilities of `hidden @ lm_head.T` without keeping `[tokens, vocab]` logits for backward.

Forward projects a fixed number of rows at a time and keeps only the per-token logsumexp; backward recomputes each
chunk's logits and turns them into the logits gradient. Peak memory is one chunk, whatever the sequence length.
Autograd over a chunked loop instead keeps every chunk's logits until backward (bf16 tokens x vocab).
"""

__all__ = [
    "UnslothChunkedLogProbs",
    "UnslothChunkedCE",
    "unsloth_chunked_ce_loss",
    "unsloth_chunked_logprobs",
    "unsloth_masked_logprobs",
    "chunked_logprob_rows",
]

import functools
import os
import torch
from typing import Optional

from unsloth_zoo.temporary_patches.common import CHUNK_ALIGN_ROWS, torch_compile_options


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


# Per chunk: fp32 logits (log-probs) or half logits (CE) plus a half logits gradient; the compiled bodies fuse the rest.
_BYTES_PER_LOGIT = 8.0
_CHUNK_GB = _env_float("UNSLOTH_LOGPROB_CHUNK_GB", 2.0)
_MIN_ROWS = int(_env_float("UNSLOTH_LOGPROB_MIN_ROWS", 512))


def _default_chunk_gb(device = None):
    # Half the head device's free memory, read per call: a first call before the optimizer states exist would
    # otherwise size every later chunk for memory that is gone.
    try:
        from unsloth_zoo.tiled_mlp import _default_target_gb

        free = float(_default_target_gb(device))
    except Exception:
        return _CHUNK_GB
    return min(_CHUNK_GB, free) if free > 0 else _CHUNK_GB


def chunked_logprob_rows(vocab_size, target_gb = None, n_rows = None, device = None):
    """Rows per chunk so that one chunk's logits stay near `target_gb`, and with `n_rows` at most an eighth of them
    (min `UNSLOTH_LOGPROB_MIN_ROWS`, 512), so a short batch is not projected in one or two
    near-full chunks. At least 2: a one-row head gradient
    is a K=1 matmul Inductor mis-lowers under dynamic shapes (see chunked_hidden_states_selective_log_softmax)."""
    target_gb = _default_chunk_gb(device) if target_gb is None else target_gb
    rows = int(target_gb * (1024**3) / (max(int(vocab_size), 1) * _BYTES_PER_LOGIT))
    if n_rows is not None:
        rows = min(rows, max(_MIN_ROWS, -(-int(n_rows) // 8)))
    # UNSLOTH_CE_CHUNK_ALIGN=1: whole 128-row tiles, as chunk_row_bounds does, unless that shrinks a chunk by > 1/4.
    aligned = (rows // CHUNK_ALIGN_ROWS) * CHUNK_ALIGN_ROWS if CHUNK_ALIGN_ROWS > 1 else 0
    if aligned and 4 * aligned >= 3 * rows:
        rows = aligned
    return max(rows, 2)


def _transform(z, scalars, has_mul: bool, has_div: bool, has_cap: bool):
    # scalars = [1 / temperature, multiply, divide, softcap] as a tensor, so new values never recompile.
    if has_mul:
        z = z * scalars[1]
    if has_div:
        z = z / scalars[2]
    if has_cap:
        z = scalars[3] * torch.tanh(z / scalars[3])
    return z


def _forward_stats(logits, targets, scalars, has_mul: bool, has_div: bool, has_cap: bool, want_mean: bool):
    z = _transform(logits.float(), scalars, has_mul, has_div, has_cap)
    mean_logits = z.mean(-1) if want_mean else None
    z = z * scalars[0]
    lse = torch.logsumexp(z, dim = -1)
    selected = z.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return selected - lse, lse, mean_logits


def _logits_grad(logits, targets, lse, grad, scalars, has_mul: bool, has_div: bool, has_cap: bool, out_dtype):
    # d logp_t / d z_v = onehot(t)_v - softmax(z)_v, chained back through temperature, softcap and scale.
    x = logits.float()
    if has_mul:
        x = x * scalars[1]
    if has_div:
        x = x / scalars[2]
    if has_cap:
        t = torch.tanh(x / scalars[3])
        z = scalars[3] * t
    else:
        z = x
    p = torch.exp(z * scalars[0] - lse.unsqueeze(-1))
    onehot = torch.arange(z.shape[-1], device = z.device).unsqueeze(0) == targets.unsqueeze(-1)
    dz = (onehot.to(p.dtype) - p) * (grad.unsqueeze(-1) * scalars[0])
    if has_cap:
        dz = dz * (1.0 - t * t)
    if has_mul:
        dz = dz * scalars[1]
    if has_div:
        dz = dz / scalars[2]
    return dz.to(out_dtype)


_COMPILE = os.environ.get("UNSLOTH_COMPILE_DISABLE", "0") not in ("1", "partial")
_compiled = {}
_eager = set()


def _run(name, fn, *args):
    if not _COMPILE or name in _eager:
        return fn(*args)
    compiled = _compiled.get(name)
    if compiled is None:
        # Only rows are dynamic: a symbolic vocab makes Inductor's row reductions ~3x slower.
        compiled = _compiled[name] = torch.compile(fn, dynamic = None, fullgraph = True, options = torch_compile_options)
    for x in args:
        # Else the second row count recompiles each body.
        if torch.is_tensor(x) and x.dim() >= 1 and x.shape[0] > 1:
            torch._dynamo.maybe_mark_dynamic(x, 0)
    try:
        return compiled(*args)
    except torch.OutOfMemoryError:
        raise
    except Exception:
        # No compiler (CPU without a toolchain), an unsupported backend, or the recompile budget.
        _eager.add(name)
        return fn(*args)


# ROCm's half-in / fp32-out GEMM is a slow fallback (gfx1151: 6-14x a bf16-out GEMM), so HIP keeps half outputs.
_FP32_OUT_GEMM = torch.version.hip is None


def _mm_fp32(a, b):
    """a @ b with an fp32 result for half inputs, so a vocab-long sum is rounded once, when cast to the grad dtype."""
    if _FP32_OUT_GEMM and a.dtype in (torch.float16, torch.bfloat16) and a.is_cuda:
        try:
            return torch.mm(a, b, out_dtype = torch.float32)
        except torch.OutOfMemoryError:
            raise
        except (TypeError, RuntimeError):
            pass
    return a @ b


_FP32_LOGITS = os.environ.get("UNSLOTH_LOGPROB_FP32_LOGITS", "1") != "0"


def _project(hidden, weight, bias, compute_dtype, fp32_out = False):
    h = hidden.to(compute_dtype)
    w = weight.to(compute_dtype)
    if fp32_out and _FP32_LOGITS and _FP32_OUT_GEMM and compute_dtype in (torch.float16, torch.bfloat16) and h.is_cuda:
        # bf16 logits bias a sequence's summed log-prob, which preference losses take differences of.
        try:
            out = torch.mm(h, w.t(), out_dtype = torch.float32)
            return out if bias is None else out + bias.float()
        except torch.OutOfMemoryError:
            raise
        except (TypeError, RuntimeError):
            pass
    if bias is None:
        return h @ w.t()
    return torch.nn.functional.linear(h, w, bias.to(compute_dtype))


class UnslothChunkedLogProbs(torch.autograd.Function):
    """logp[i] = log_softmax(transform(hidden[i] @ weight.T + bias) / temperature)[targets[i]] for hidden [N, H].

    Returns (logps [N] float32, mean_logits [N] float32 or None). mean_logits is the vocab mean of the transformed
    logits before temperature (what `logits.mean()` metrics read), not differentiable.
    """

    @staticmethod
    def forward(
        ctx, hidden, weight, bias, targets, rows_per_chunk, temperature,
        logit_scale_multiply, logit_scale_divide, logit_softcapping, want_mean,
    ):
        ctx.set_materialize_grads(False)
        device = hidden.device
        if torch.is_autocast_enabled(device.type):
            compute_dtype = torch.get_autocast_dtype(device.type)
        else:
            compute_dtype = weight.dtype if weight.dtype in (torch.float16, torch.bfloat16, torch.float32) else hidden.dtype
        n = hidden.shape[0]
        mul, div, cap = float(logit_scale_multiply), float(logit_scale_divide), float(logit_softcapping)
        scalars = torch.tensor([1.0 / float(temperature), mul, div, cap], device = device, dtype = torch.float32)
        flags = (mul != 0.0, div != 0.0, cap != 0.0)
        logps = torch.empty(n, device = device, dtype = torch.float32)
        lse = torch.empty(n, device = device, dtype = torch.float32)
        mean_logits = torch.empty(n, device = device, dtype = torch.float32) if want_mean else None
        w = weight.to(compute_dtype)
        with torch.autocast(device.type, enabled = False):
            for a in range(0, n, rows_per_chunk):
                b = min(a + rows_per_chunk, n)
                logits = _project(hidden[a:b], w, bias, compute_dtype, fp32_out = True)
                lp, ls, mean = _run(
                    "fwd_mean" if want_mean else "fwd", _forward_stats,
                    logits, targets[a:b], scalars, *flags, bool(want_mean),
                )
                logps[a:b] = lp
                lse[a:b] = ls
                if want_mean:
                    mean_logits[a:b] = mean
                del logits
        ctx.save_for_backward(hidden, weight, bias, targets, lse, scalars)
        ctx.rows_per_chunk = rows_per_chunk
        ctx.flags = flags
        ctx.compute_dtype = compute_dtype
        if mean_logits is not None:
            ctx.mark_non_differentiable(mean_logits)
        return logps, mean_logits

    @staticmethod
    def backward(ctx, grad_logps, grad_mean):
        hidden, weight, bias, targets, lse, scalars = ctx.saved_tensors
        need_h, need_w, need_b = ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2]
        if grad_logps is None or not (need_h or need_w or need_b):
            return (None,) * 10
        compute_dtype = ctx.compute_dtype
        grad_logps = grad_logps.float()
        n = hidden.shape[0]
        grad_hidden = torch.empty_like(hidden) if need_h else None
        # In the head's dtype, like the fused CE path: an fp32 [V, H] buffer costs GiBs on a trainable half head.
        grad_weight = torch.zeros_like(weight) if need_w else None
        grad_bias = torch.zeros(bias.shape, device = bias.device, dtype = torch.float32) if need_b else None
        w = weight.to(compute_dtype)
        with torch.autocast(hidden.device.type, enabled = False):
            for a in range(0, n, ctx.rows_per_chunk):
                b = min(a + ctx.rows_per_chunk, n)
                h = hidden[a:b].to(compute_dtype)
                logits = _project(h, w, bias, compute_dtype, fp32_out = True)
                dz = _run(
                    "bwd", _logits_grad,
                    logits, targets[a:b], lse[a:b], grad_logps[a:b], scalars, *ctx.flags, compute_dtype,
                )
                del logits
                if need_h:
                    grad_hidden[a:b] = _mm_fp32(dz, w).to(grad_hidden.dtype)
                if need_w:
                    grad_weight.add_(dz.t() @ h)
                if need_b:
                    grad_bias.add_(dz.float().sum(0))
                del dz
        if grad_weight is not None:
            grad_weight = grad_weight.to(weight.dtype)
        if grad_bias is not None:
            grad_bias = grad_bias.to(bias.dtype)
        return grad_hidden, grad_weight, grad_bias, None, None, None, None, None, None, None


def unsloth_chunked_logprobs(
    hidden_states : torch.Tensor,
    lm_head       : torch.Tensor,
    index         : torch.Tensor,
    temperature   : float = 1.0,
    logit_scale_multiply : float = 0.0,
    logit_scale_divide   : float = 0.0,
    logit_softcapping    : float = 0.0,
    lm_head_bias  : Optional[torch.Tensor] = None,
    rows_per_chunk: Optional[int] = None,
    return_mean_logits: bool = False,
):
    """Per-token log-probs for hidden_states [..., H] and index [...] (valid token ids), shape [...] float32.

    Same contract as chunked_hidden_states_selective_log_softmax, but backward recomputes logits per chunk instead of
    keeping them. On a split model the math runs on the head's device and the result comes back to the caller's.
    """
    lead_shape = index.shape
    hidden = hidden_states.reshape(-1, hidden_states.shape[-1]).to(lm_head.device)
    targets = index.reshape(-1).to(lm_head.device)
    if rows_per_chunk is None:
        rows_per_chunk = chunked_logprob_rows(lm_head.shape[0], n_rows = hidden.shape[0], device = lm_head.device)
    logps, mean_logits = UnslothChunkedLogProbs.apply(
        hidden, lm_head, lm_head_bias, targets, int(rows_per_chunk), float(temperature),
        float(logit_scale_multiply or 0.0), float(logit_scale_divide or 0.0), float(logit_softcapping or 0.0),
        bool(return_mean_logits),
    )
    logps = logps.reshape(lead_shape).to(hidden_states.device)
    if return_mean_logits:
        return logps, mean_logits.reshape(lead_shape).to(hidden_states.device)
    return logps


def unsloth_masked_logprobs(hidden_states, lm_head, labels, mask = None, ignore_index = -100, **kwargs):
    """Like unsloth_chunked_logprobs, but only rows where `mask` (default labels != ignore_index) is set reach the head;
    the rest are 0. `masked_scatter` keeps an all-masked batch connected to the graph as a differentiable zero."""
    if mask is None:
        mask = labels != ignore_index
    mask = mask.to(hidden_states.device)
    out = unsloth_chunked_logprobs(hidden_states[mask], lm_head, labels.to(mask.device)[mask], **kwargs)
    if isinstance(out, tuple):
        return tuple(x.new_zeros(mask.shape).masked_scatter(mask, x) for x in out)
    return out.new_zeros(mask.shape).masked_scatter(mask, out)


class UnslothChunkedCE(torch.autograd.Function):
    """sum(-logp[i] for labelled rows) / divisor, with the gradient taken chunk by chunk in forward.

    Cross entropy is a sum over tokens, so unlike preference losses its gradient is known in forward: each chunk's
    logits gradient is (softmax - onehot) / divisor, turned into the hidden (and head) gradient right away and the
    logits freed. No recompute, one chunk resident. `scaling` (fp16 GradScaler) multiplies the stored gradients so
    they do not underflow in half precision; backward divides it back out.
    """

    @staticmethod
    def forward(
        ctx, hidden, weight, bias, targets, divisor, scaling, rows_per_chunk,
        logit_scale_multiply, logit_scale_divide, logit_softcapping,
    ):
        device = hidden.device
        if torch.is_autocast_enabled(device.type):
            compute_dtype = torch.get_autocast_dtype(device.type)
        else:
            compute_dtype = weight.dtype if weight.dtype in (torch.float16, torch.bfloat16, torch.float32) else hidden.dtype
        mul, div, cap = float(logit_scale_multiply), float(logit_scale_divide), float(logit_softcapping)
        scalars = torch.tensor([1.0, mul, div, cap], device = device, dtype = torch.float32)
        flags = (mul != 0.0, div != 0.0, cap != 0.0)
        need_h = hidden.requires_grad
        need_w = weight.requires_grad
        need_b = bias is not None and bias.requires_grad
        n = hidden.shape[0]
        divisor = divisor.to(device = device, dtype = torch.float32).reshape(())
        loss = torch.zeros((), device = device, dtype = torch.float32)
        grad_hidden = torch.empty_like(hidden) if need_h else None
        grad_weight = torch.zeros_like(weight) if need_w else None
        grad_bias = torch.zeros(bias.shape, device = bias.device, dtype = torch.float32) if need_b else None
        w = weight.to(compute_dtype)
        g_row = (-float(scaling) / divisor).expand(min(rows_per_chunk, max(n, 1)))
        with torch.autocast(device.type, enabled = False):
            for a in range(0, n, rows_per_chunk):
                b = min(a + rows_per_chunk, n)
                h = hidden[a:b].to(compute_dtype)
                logits = _project(h, w, bias, compute_dtype)
                lp, lse, _ = _run("fwd", _forward_stats, logits, targets[a:b], scalars, *flags, False)
                loss -= lp.sum()
                if need_h or need_w or need_b:
                    dz = _run("bwd", _logits_grad, logits, targets[a:b], lse, g_row[: b - a], scalars, *flags, compute_dtype)
                    del logits
                    if need_h:
                        grad_hidden[a:b] = (dz @ w).to(grad_hidden.dtype)
                    if need_w:
                        grad_weight.add_(dz.t() @ h)
                    if need_b:
                        grad_bias.add_(dz.float().sum(0))
                    del dz
        if grad_weight is not None:
            grad_weight = grad_weight.to(weight.dtype)
        if grad_bias is not None:
            grad_bias = grad_bias.to(bias.dtype)
        ctx.save_for_backward(*(x for x in (grad_hidden, grad_weight, grad_bias) if x is not None))
        ctx.has = (need_h, need_w, need_b)
        ctx.scaling = float(scaling)
        return loss / divisor

    @staticmethod
    def backward(ctx, grad_output):
        saved = iter(ctx.saved_tensors)
        scale = grad_output.detach().float().mean() / ctx.scaling
        grads = [next(saved) * scale if has else None for has in ctx.has]
        return grads[0], grads[1], grads[2], None, None, None, None, None, None, None


def unsloth_chunked_ce_loss(
    hidden_states, lm_head_weight, lm_head_bias, labels, mask = None, n_items = None, scaling = None,
    shift_labels = True, ignore_index = -100, logit_scale_multiply = 0.0, logit_scale_divide = 0.0,
    logit_softcapping = 0.0, rows_per_chunk = None,
    target_gb = None,
):
    """Drop-in for the chunked fused CE (same shift / mask / n_items / scaling contract), scoring only labelled rows.
    A zero `scaling` stores unscaled gradients, so a collapsed GradScaler gives zeros rather than 0 * inf."""
    device = lm_head_weight.device
    labels = labels.to(device)
    if shift_labels:
        shifted = torch.full_like(labels, ignore_index)
        shifted[..., :-1] = labels[..., 1:]
        if mask is not None:
            shifted[..., :-1][mask.to(device)[..., 1:] == 0] = ignore_index
        labels = shifted
    labels = labels.reshape(-1)
    hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    keep = labels != ignore_index
    divisor = n_items if n_items is not None else keep.sum()
    if not torch.is_tensor(divisor):
        divisor = torch.tensor(divisor, dtype = torch.float32)
    if divisor.numel() != 1:
        divisor = divisor.ravel()[0]
    divisor = divisor.to(device = device, dtype = torch.float32)
    divisor = torch.where(divisor == 0, torch.ones_like(divisor), divisor)  # all-masked batch: loss 0, not 0 / 0
    # Unlabelled rows (prompts in completion-only training) never reach the head.
    hidden = hidden[keep.to(hidden.device)].to(device)
    if rows_per_chunk is None:
        rows_per_chunk = chunked_logprob_rows(
            lm_head_weight.shape[0], target_gb = target_gb, n_rows = hidden.shape[0], device = device
        )
    return UnslothChunkedCE.apply(
        hidden, lm_head_weight, lm_head_bias, labels[keep], divisor, float(scaling or 1.0),
        int(rows_per_chunk), float(logit_scale_multiply or 0.0), float(logit_scale_divide or 0.0),
        float(logit_softcapping or 0.0),
    )
