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

"""Chunked linear token logps for preference trainers (DPO, ORPO, CPO / SimPO).

``fused_linear_token_logps`` scores completion tokens from the final hidden states through
the output head one chunk of rows at a time and recomputes each chunk in backward, so the
``batch x seq_len x vocab`` logits are never materialised. The preference loss itself stays
with TRL's trainer.
"""

__all__ = [
    "fused_linear_token_logps",
    "linear_mean_logits",
]

import os
from typing import Optional

import torch
import torch.nn.functional as F


def _target_chunk_bytes():
    value = os.environ.get("UNSLOTH_FUSED_PREFERENCE_CHUNK_MB", "256").strip()
    try:
        return max(int(value), 1) * 1024 * 1024
    except ValueError:
        return 256 * 1024 * 1024


def _rows_per_chunk(n_rows, vocab_size, chunk_size):
    if chunk_size is not None and chunk_size > 0:
        return max(1, min(int(chunk_size), max(n_rows, 1)))
    # About four float32 rows x vocab buffers are live per chunk in backward.
    rows = _target_chunk_bytes() // max(4 * 4 * vocab_size, 1)
    return max(1, min(int(rows), max(n_rows, 1)))


def _transform_logits(logits, scale_multiply, scale_divide, softcapping):
    if scale_multiply != 0.0:
        logits = logits * scale_multiply
    if scale_divide != 0.0:
        logits = logits / scale_divide
    if softcapping != 0.0:
        logits = softcapping * torch.tanh(logits / softcapping)
    return logits


_MM_OUT_DTYPE_OK = {}

# UNSLOTH_FUSED_LINEAR=0 or a float64 head (kernel scores in fp32) keep this module's chunked Function.
_FUSED_LINEAR = os.environ.get("UNSLOTH_FUSED_LINEAR", "1") != "0"
# So does an explicit UNSLOTH_FUSED_PREFERENCE_CHUNK_MB.
_EXPLICIT_CHUNK_MB = "UNSLOTH_FUSED_PREFERENCE_CHUNK_MB" in os.environ


def _use_fused_linear(weight):
    return _FUSED_LINEAR and not _EXPLICIT_CHUNK_MB and weight.dtype != torch.float64


def _linear_float32(hidden, weight, bias):
    # A half-precision head rounds its logits to bf16 / fp16 before the softmax; that rounding
    # dominates the logp error, so ask the GEMM for float32 output (torch >= 2.8) where it can.
    key = (hidden.device.type, weight.dtype)
    # Not on ROCm: half-in / fp32-out GEMM is a slow fallback.
    if torch.version.hip is None and weight.dtype in (torch.float16, torch.bfloat16) and _MM_OUT_DTYPE_OK.get(key, True):
        try:
            logits = torch.mm(hidden, weight.t(), out_dtype = torch.float32)
            _MM_OUT_DTYPE_OK[key] = True
            return logits if bias is None else logits + bias.float()
        except torch.cuda.OutOfMemoryError:
            raise
        except (TypeError, RuntimeError, NotImplementedError):
            _MM_OUT_DTYPE_OK[key] = False
    logits = F.linear(hidden, weight, bias)
    return logits.to(torch.promote_types(logits.dtype, torch.float32))


def _mm_float32(a, b):
    # Half GEMM with float32 output: per slice / row group partial sums can exceed the fp16 range
    # before they cancel in the float32 accumulators.
    key = (a.device.type, a.dtype)
    if _MM_OUT_DTYPE_OK.get(key, True):
        try:
            out = torch.mm(a, b, out_dtype = torch.float32)
            _MM_OUT_DTYPE_OK[key] = True
            return out
        except torch.cuda.OutOfMemoryError:
            raise
        except (TypeError, RuntimeError, NotImplementedError):
            _MM_OUT_DTYPE_OK[key] = False
    return torch.mm(a.float(), b.float())


def _chunk_logits(hidden, weight, bias, transforms):
    hidden = hidden.to(device = weight.device, dtype = weight.dtype)
    return _transform_logits(_linear_float32(hidden, weight, bias), *transforms)


_HALF_DTYPES = (torch.float16, torch.bfloat16)


def _sliced_backward(ctx, hidden, weight, bias, labels, lse, grad_logps):
    # A trainable half precision head: a float32 vocab x hidden accumulator plus per chunk GEMM and
    # cast copies cost about 10 x vocab x hidden bytes, and summing chunks in the head dtype loses
    # bits every chunk. Loop over vocab slices instead, each summed over every row chunk in a float32
    # accumulator of about one chunk's size: the head gradient costs what autograd's own does, with
    # the float32 accumulation of the per chunk loop.
    need_hidden, need_weight, need_bias = ctx.needs_input_grad[:3]
    scale_multiply, scale_divide, softcapping = ctx.transforms
    scale = 1.0
    if scale_multiply != 0.0:
        scale *= scale_multiply
    if scale_divide != 0.0:
        scale /= scale_divide
    n_rows = hidden.shape[0]
    vocab_size, hidden_size = weight.shape
    acc_dtype = torch.promote_types(weight.dtype, torch.float32)
    n_slices = max(1, min(vocab_size, -(-vocab_size * hidden_size * 4 // _target_chunk_bytes())))
    slice_size = -(-vocab_size // n_slices)
    # Keep each rows x slice logits block the size of a rows x vocab chunk.
    rows = max(1, min(ctx.rows * (-(-vocab_size // slice_size)), max(n_rows, 1)))
    grad_hidden = (
        torch.zeros((n_rows, hidden_size), dtype = acc_dtype, device = weight.device)
        if need_hidden else None
    )
    grad_weight = torch.empty_like(weight)
    grad_bias = (
        torch.zeros(bias.shape, dtype = acc_dtype, device = bias.device)
        if (need_bias and bias is not None) else None
    )
    if grad_logps is None:
        grad_logps = torch.zeros(n_rows, dtype = acc_dtype, device = hidden.device)
    with torch.autocast(device_type = hidden.device.type, enabled = False):
        for first in range(0, vocab_size, slice_size):
            last = min(first + slice_size, vocab_size)
            weight_slice = weight[first:last]
            bias_slice = bias[first:last] if bias is not None else None
            weight_acc = torch.zeros((last - first, hidden_size), dtype = acc_dtype, device = weight.device)
            for start in range(0, n_rows, rows):
                end = min(start + rows, n_rows)
                chunk_hidden = hidden[start:end].to(device = weight.device, dtype = weight.dtype)
                logits = _chunk_logits(chunk_hidden, weight_slice, bias_slice, ctx.transforms)
                grad_logits = (logits - lse[start:end].to(logits).unsqueeze(-1)).exp_()
                grad_logits.neg_()
                target = labels[start:end].to(logits.device) - first
                inside = (target >= 0) & (target < last - first)
                grad_logits.scatter_add_(
                    -1, target.clamp(0, last - first - 1).unsqueeze(-1),
                    inside.to(grad_logits.dtype).unsqueeze(-1),
                )
                grad_logits.mul_(grad_logps[start:end].to(grad_logits).unsqueeze(-1))
                if softcapping != 0.0:
                    grad_logits.mul_(1.0 - (logits / softcapping).square())
                del logits
                if scale != 1.0:
                    grad_logits.mul_(scale)
                if grad_bias is not None:
                    grad_bias[first:last] += grad_logits.sum(0)
                grad_logits = grad_logits.to(weight.dtype)
                if need_hidden:
                    grad_hidden[start:end] += _mm_float32(grad_logits, weight_slice)
                weight_acc += _mm_float32(grad_logits.t(), chunk_hidden)
                del grad_logits
            grad_weight[first:last] = weight_acc
            del weight_acc
    if grad_hidden is not None:
        grad_hidden = grad_hidden.to(device = hidden.device, dtype = hidden.dtype)
    if grad_bias is not None:
        grad_bias = grad_bias.to(bias.dtype)
    return grad_hidden, grad_weight, grad_bias, None, None, None, None, None


class _ChunkedLinearTokenLogps(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, bias, labels, rows, transforms, want_mean, want_lssp):
        n_rows = hidden.shape[0]
        device = hidden.device
        dtype = torch.promote_types(weight.dtype, torch.float32)
        logps = torch.empty(n_rows, dtype = dtype, device = device)
        mean_logits = torch.empty(n_rows if want_mean else 0, dtype = dtype, device = device)
        log_sum_sq_probs = torch.empty(n_rows if want_lssp else 0, dtype = dtype, device = device)
        # A trainable half precision head runs the vocab sliced backward, which needs each row's lse.
        sliced = ctx.needs_input_grad[1] and weight.dtype in _HALF_DTYPES
        if sliced and ctx.needs_input_grad[0]:
            # Sliced backward holds a float32 rows x hidden gradient plus its cast (6 N H bytes for a
            # half hidden) and a head dtype head gradient (2 V H); the per chunk loop holds float32
            # head sized copies (10 V H) and a hidden dtype gradient. Long contexts favour the latter.
            vocab_size = weight.shape[0]
            hidden_bytes = hidden.element_size()
            cast_bytes = hidden_bytes if hidden.dtype != torch.float32 else 0
            sliced = 2 * vocab_size + n_rows * (4 + cast_bytes) <= 10 * vocab_size + n_rows * hidden_bytes
        saved_lse = torch.empty(n_rows if sliced else 0, dtype = dtype, device = device)
        with torch.autocast(device_type = device.type, enabled = False):
            for start in range(0, n_rows, rows):
                end = min(start + rows, n_rows)
                logits = _chunk_logits(hidden[start:end], weight, bias, transforms)
                lse = logits.logsumexp(-1)
                if sliced:
                    saved_lse[start:end] = lse.to(device)
                target = labels[start:end].to(logits.device).unsqueeze(-1)
                logps[start:end] = (logits.gather(-1, target).squeeze(-1) - lse).to(device)
                if want_mean:
                    mean_logits[start:end] = logits.mean(-1).to(device)
                if want_lssp:
                    log_sum_sq_probs[start:end] = ((2.0 * logits).logsumexp(-1) - 2.0 * lse).to(device)
                del logits
        ctx.save_for_backward(hidden, weight, bias, labels, saved_lse)
        ctx.sliced = sliced
        ctx.rows = rows
        ctx.transforms = transforms
        ctx.mark_non_differentiable(mean_logits, log_sum_sq_probs)
        return logps, mean_logits, log_sum_sq_probs

    @staticmethod
    def backward(ctx, grad_logps, grad_mean, grad_lssp):
        hidden, weight, bias, labels, saved_lse = ctx.saved_tensors
        need_hidden, need_weight, need_bias = ctx.needs_input_grad[:3]
        if ctx.sliced and need_weight:
            return _sliced_backward(ctx, hidden, weight, bias, labels, saved_lse, grad_logps)
        scale_multiply, scale_divide, softcapping = ctx.transforms
        scale = 1.0
        if scale_multiply != 0.0:
            scale *= scale_multiply
        if scale_divide != 0.0:
            scale /= scale_divide
        n_rows = hidden.shape[0]
        grad_hidden = torch.zeros_like(hidden) if need_hidden else None
        # float32 accumulators: summing per-chunk head gradients in bf16 loses bits every chunk.
        acc_dtype = torch.promote_types(weight.dtype, torch.float32)
        grad_weight = torch.zeros(weight.shape, dtype = acc_dtype, device = weight.device) if need_weight else None
        grad_bias = (
            torch.zeros(bias.shape, dtype = acc_dtype, device = bias.device)
            if (need_bias and bias is not None) else None
        )
        if grad_logps is None:
            grad_logps = torch.zeros(n_rows, dtype = acc_dtype, device = hidden.device)
        with torch.autocast(device_type = hidden.device.type, enabled = False):
            for start in range(0, n_rows, ctx.rows):
                end = min(start + ctx.rows, n_rows)
                chunk_hidden = hidden[start:end].to(device = weight.device, dtype = weight.dtype)
                logits = _chunk_logits(chunk_hidden, weight, bias, ctx.transforms)
                grad_logits = torch.softmax(logits, dim = -1)
                grad_logits.neg_()
                target = labels[start:end].to(logits.device).unsqueeze(-1)
                grad_logits.scatter_add_(-1, target, torch.ones_like(target, dtype = grad_logits.dtype))
                grad_logits.mul_(grad_logps[start:end].to(grad_logits).unsqueeze(-1))
                if softcapping != 0.0:
                    grad_logits.mul_(1.0 - (logits / softcapping).square())
                del logits
                if scale != 1.0:
                    grad_logits.mul_(scale)
                if need_bias and bias is not None:
                    grad_bias += grad_logits.sum(0)
                grad_logits = grad_logits.to(weight.dtype)
                if need_hidden:
                    grad_hidden[start:end] = (grad_logits @ weight).to(device = hidden.device, dtype = hidden.dtype)
                if need_weight:
                    # Opposite signed row chunks overflow an fp16 product before they cancel here.
                    grad_weight.add_(_mm_float32(grad_logits.t(), chunk_hidden) if weight.dtype in _HALF_DTYPES
                                     else (grad_logits.t() @ chunk_hidden).to(acc_dtype))
                del grad_logits
        if grad_weight is not None:
            grad_weight = grad_weight.to(weight.dtype)
        if grad_bias is not None:
            grad_bias = grad_bias.to(bias.dtype)
        return grad_hidden, grad_weight, grad_bias, None, None, None, None, None


def fused_linear_token_logps(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: Optional[torch.Tensor],
    labels: torch.Tensor,
    mask: torch.Tensor,
    logit_scale_multiply: float = 0.0,
    logit_scale_divide: float = 0.0,
    logit_softcapping: float = 0.0,
    chunk_size: Optional[int] = None,
    return_mean_logits: bool = False,
    return_log_sum_sq_probs: bool = False,
) -> dict:
    """Per-token ``log p(labels | hidden_states)`` through the output head, without full logits.

    Position ``t`` of ``hidden_states`` ``[..., hidden]`` predicts ``labels[t]``. Only ``mask``
    positions are projected; the rest are 0 in every output (float32, float64 for a float64 head).
    Optional no-grad outputs: ``mean_logits`` (vocab mean of the transformed logits) and
    ``log_sum_sq_probs`` (``log sum_v p_v^2``).
    """
    mask = mask.to(device = hidden_states.device, dtype = torch.bool)
    labels = labels.to(hidden_states.device)
    if _use_fused_linear(lm_head_weight):
        from unsloth_zoo.fused_losses.fused_linear import fused_linear_logprobs

        outs = fused_linear_logprobs(
            hidden_states, lm_head_weight, labels, bias = lm_head_bias, mask = mask,
            logit_scale_multiply = float(logit_scale_multiply), logit_scale_divide = float(logit_scale_divide),
            logit_softcapping = float(logit_softcapping), return_mean_logits = bool(return_mean_logits),
            return_log_sum_sq_probs = bool(return_log_sum_sq_probs),
            rows_per_chunk = chunk_size if chunk_size is not None and chunk_size > 0 else None,
        )
        outs = outs if isinstance(outs, tuple) else (outs,)
        output = {"per_token_logps": outs[0]}
        extra = iter(outs[1:])
        if return_mean_logits:
            output["mean_logits"] = next(extra)
        if return_log_sum_sq_probs:
            output["log_sum_sq_probs"] = next(extra)
        return output
    selected_hidden = hidden_states[mask]
    selected_labels = labels[mask]
    rows = _rows_per_chunk(selected_hidden.shape[0], lm_head_weight.shape[0], chunk_size)
    transforms = (float(logit_scale_multiply), float(logit_scale_divide), float(logit_softcapping))
    logps, mean_logits, log_sum_sq_probs = _ChunkedLinearTokenLogps.apply(
        selected_hidden, lm_head_weight, lm_head_bias, selected_labels,
        rows, transforms, bool(return_mean_logits), bool(return_log_sum_sq_probs),
    )
    # masked_scatter keeps the output attached to the graph even when nothing is selected.
    zeros = torch.zeros(mask.shape, dtype = logps.dtype, device = hidden_states.device)
    output = {"per_token_logps": zeros.masked_scatter(mask, logps)}
    if return_mean_logits:
        output["mean_logits"] = zeros.masked_scatter(mask, mean_logits)
    if return_log_sum_sq_probs:
        output["log_sum_sq_probs"] = zeros.masked_scatter(mask, log_sum_sq_probs)
    return output


@torch.no_grad()
def linear_mean_logits(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: Optional[torch.Tensor] = None,
    logit_scale_multiply: float = 0.0,
    logit_scale_divide: float = 0.0,
    logit_softcapping: float = 0.0,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Vocab mean of the transformed logits at every position.

    Without soft capping the head is affine, so the mean is ``h . mean(W) + mean(b)``; soft
    capping is not, and is projected in chunks.
    """
    shape = hidden_states.shape[:-1]
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    if logit_softcapping == 0.0:
        dtype = torch.promote_types(lm_head_weight.dtype, torch.float32)
        # Accumulate in float32 without a float32 copy of the whole head.
        weight_mean = lm_head_weight.sum(0, dtype = dtype) / lm_head_weight.shape[0]
        out = flat.to(device = weight_mean.device, dtype = dtype) @ weight_mean
        if lm_head_bias is not None:
            out = out + lm_head_bias.to(dtype).mean()
        out = _transform_logits(out, float(logit_scale_multiply), float(logit_scale_divide), 0.0)
        return out.to(hidden_states.device).reshape(shape)
    if _use_fused_linear(lm_head_weight):
        from unsloth_zoo.fused_losses.fused_linear import fused_linear_logprobs

        _, mean = fused_linear_logprobs(
            hidden_states, lm_head_weight, torch.zeros(shape, dtype = torch.long, device = hidden_states.device),
            bias = lm_head_bias, logit_scale_multiply = float(logit_scale_multiply),
            logit_scale_divide = float(logit_scale_divide), logit_softcapping = float(logit_softcapping),
            return_mean_logits = True, rows_per_chunk = chunk_size if chunk_size is not None and chunk_size > 0 else None,
        )
        return mean
    rows = _rows_per_chunk(flat.shape[0], lm_head_weight.shape[0], chunk_size)
    transforms = (float(logit_scale_multiply), float(logit_scale_divide), float(logit_softcapping))
    dtype = torch.promote_types(lm_head_weight.dtype, torch.float32)
    out = torch.empty(flat.shape[0], dtype = dtype, device = hidden_states.device)
    for start in range(0, flat.shape[0], rows):
        end = min(start + rows, flat.shape[0])
        out[start:end] = _chunk_logits(flat[start:end], lm_head_weight, lm_head_bias, transforms).mean(-1).to(out.device)
    return out.reshape(shape)
