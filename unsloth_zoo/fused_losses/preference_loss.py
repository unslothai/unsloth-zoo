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


def _linear_float32(hidden, weight, bias):
    # A half-precision head rounds its logits to bf16 / fp16 before the softmax; that rounding
    # dominates the logp error, so ask the GEMM for float32 output (torch >= 2.8) where it can.
    key = (hidden.device.type, weight.dtype)
    if weight.dtype in (torch.float16, torch.bfloat16) and _MM_OUT_DTYPE_OK.get(key, True):
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


def _chunk_logits(hidden, weight, bias, transforms):
    hidden = hidden.to(device = weight.device, dtype = weight.dtype)
    return _transform_logits(_linear_float32(hidden, weight, bias), *transforms)


class _ChunkedLinearTokenLogps(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, bias, labels, rows, transforms, want_mean, want_lssp):
        n_rows = hidden.shape[0]
        device = hidden.device
        dtype = torch.promote_types(weight.dtype, torch.float32)
        logps = torch.empty(n_rows, dtype = dtype, device = device)
        mean_logits = torch.empty(n_rows if want_mean else 0, dtype = dtype, device = device)
        log_sum_sq_probs = torch.empty(n_rows if want_lssp else 0, dtype = dtype, device = device)
        with torch.autocast(device_type = device.type, enabled = False):
            for start in range(0, n_rows, rows):
                end = min(start + rows, n_rows)
                logits = _chunk_logits(hidden[start:end], weight, bias, transforms)
                lse = logits.logsumexp(-1)
                target = labels[start:end].to(logits.device).unsqueeze(-1)
                logps[start:end] = (logits.gather(-1, target).squeeze(-1) - lse).to(device)
                if want_mean:
                    mean_logits[start:end] = logits.mean(-1).to(device)
                if want_lssp:
                    log_sum_sq_probs[start:end] = ((2.0 * logits).logsumexp(-1) - 2.0 * lse).to(device)
                del logits
        ctx.save_for_backward(hidden, weight, bias, labels)
        ctx.rows = rows
        ctx.transforms = transforms
        ctx.mark_non_differentiable(mean_logits, log_sum_sq_probs)
        return logps, mean_logits, log_sum_sq_probs

    @staticmethod
    def backward(ctx, grad_logps, grad_mean, grad_lssp):
        hidden, weight, bias, labels = ctx.saved_tensors
        need_hidden, need_weight, need_bias = ctx.needs_input_grad[:3]
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
                    grad_weight.add_((grad_logits.t() @ chunk_hidden).to(acc_dtype))
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
    rows = _rows_per_chunk(flat.shape[0], lm_head_weight.shape[0], chunk_size)
    transforms = (float(logit_scale_multiply), float(logit_scale_divide), float(logit_softcapping))
    dtype = torch.promote_types(lm_head_weight.dtype, torch.float32)
    out = torch.empty(flat.shape[0], dtype = dtype, device = hidden_states.device)
    for start in range(0, flat.shape[0], rows):
        end = min(start + rows, flat.shape[0])
        out[start:end] = _chunk_logits(flat[start:end], lm_head_weight, lm_head_bias, transforms).mean(-1).to(out.device)
    return out.reshape(shape)
