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

"""Chunked linear preference losses (DPO family, ORPO, CPO / SimPO).

``fused_linear_token_logps`` scores completion tokens from the final hidden states through
the output head one chunk of rows at a time and recomputes each chunk in backward, so the
``batch x seq_len x vocab`` logits are never materialised. The loss functions reproduce
TRL's formulas on sequence logps.
"""

__all__ = [
    "fused_linear_token_logps",
    "linear_mean_logits",
    "dpo_preference_loss",
    "orpo_preference_loss",
    "cpo_preference_loss",
    "fused_linear_preference_loss",
    "DPO_LOSS_TYPES",
    "CPO_LOSS_TYPES",
]

import math
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
        weight_mean = lm_head_weight.to(dtype).mean(0)
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


DPO_LOSS_TYPES = (
    "sigmoid", "robust", "exo_pair", "hinge", "ipo", "sppo_hard", "nca_pair",
    "aot", "aot_pair", "aot_unpaired", "apo_zero", "apo_down", "discopop",
)
CPO_LOSS_TYPES = ("sigmoid", "hinge", "ipo", "simpo")


def dpo_preference_loss(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    loss_type: str = "sigmoid",
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    reference_free: bool = False,
    discopop_tau: float = 0.05,
):
    """TRL ``DPOTrainer.dpo_loss`` (reverse KL): ``(losses, chosen_rewards, rejected_rewards)``.

    ``bco_pair`` (running mean across steps) and ``sft`` (chosen NLL) stay with the trainer.
    """
    chosen_logratios = chosen_logps - (not reference_free) * ref_chosen_logps
    rejected_logratios = rejected_logps - (not reference_free) * ref_rejected_logps
    logratios = chosen_logps - rejected_logps
    if reference_free:
        ref_logratios = torch.zeros(1, dtype = logratios.dtype, device = logratios.device)
    else:
        ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = logratios - ref_logratios

    if loss_type == "sigmoid":
        losses = (
            -F.logsigmoid(beta * logits) * (1 - label_smoothing)
            - F.logsigmoid(-beta * logits) * label_smoothing
        )
    elif loss_type == "robust":
        losses = (
            -F.logsigmoid(beta * logits) * (1 - label_smoothing)
            + F.logsigmoid(-beta * logits) * label_smoothing
        ) / (1 - 2 * label_smoothing)
    elif loss_type == "exo_pair":
        # TRL replaces a zero label_smoothing with 1e-3 for exo_pair (log(0) otherwise).
        eps = label_smoothing if label_smoothing != 0 else 1e-3
        losses = (beta * logits).sigmoid() * (
            F.logsigmoid(beta * logits) - math.log(1 - eps)
        ) + (-beta * logits).sigmoid() * (F.logsigmoid(-beta * logits) - math.log(eps))
    elif loss_type == "hinge":
        losses = torch.relu(1 - beta * logits)
    elif loss_type == "ipo":
        losses = (logits - 1 / (2 * beta)) ** 2
    elif loss_type == "sppo_hard":
        a = chosen_logps - ref_chosen_logps
        b = rejected_logps - ref_rejected_logps
        losses = (a - 0.5 / beta) ** 2 + (b + 0.5 / beta) ** 2
    elif loss_type == "nca_pair":
        chosen_rewards = (chosen_logps - ref_chosen_logps) * beta
        rejected_rewards = (rejected_logps - ref_rejected_logps) * beta
        losses = (
            -F.logsigmoid(chosen_rewards)
            - 0.5 * F.logsigmoid(-chosen_rewards)
            - 0.5 * F.logsigmoid(-rejected_rewards)
        )
    elif loss_type in ("aot_pair", "aot_unpaired", "aot"):
        # TRL 0.28 renamed aot_pair to aot_unpaired.
        if loss_type != "aot":
            first, _ = torch.sort(chosen_logps - ref_chosen_logps, dim = 0)
            second, _ = torch.sort(rejected_logps - ref_rejected_logps, dim = 0)
        else:
            first, _ = torch.sort(chosen_logps - rejected_logps, dim = 0)
            second, _ = torch.sort(ref_chosen_logps - ref_rejected_logps, dim = 0)
        delta = first - second
        losses = (
            -F.logsigmoid(beta * delta) * (1 - label_smoothing)
            - F.logsigmoid(-beta * delta) * label_smoothing
        )
    elif loss_type == "apo_zero":
        losses = (1 - torch.sigmoid(beta * chosen_logratios)) + torch.sigmoid(beta * rejected_logratios)
    elif loss_type == "apo_down":
        losses = torch.sigmoid(beta * chosen_logratios) + (
            1 - torch.sigmoid(beta * (chosen_logratios - rejected_logratios))
        )
    elif loss_type == "discopop":
        scaled = (logratios - (ref_chosen_logps - ref_rejected_logps)) * beta
        modulation = torch.sigmoid(scaled / discopop_tau)
        losses = -F.logsigmoid(scaled) * (1 - modulation) + torch.exp(-scaled) * modulation
    else:
        raise ValueError(f"Unsloth: unknown DPO loss_type {loss_type!r}; expected one of {DPO_LOSS_TYPES}.")

    chosen_rewards = beta * (chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (rejected_logps - ref_rejected_logps).detach()
    return losses, chosen_rewards, rejected_rewards


def _log1mexp(x):
    # log(1 - exp(x)) as TRL >= 0.29 computes it; log1p(-exp(x)) cancels near x = 0.
    return torch.where(x < -0.6931471805599453, torch.log1p(-torch.exp(x)), torch.log(-torch.expm1(x)))


def orpo_preference_loss(chosen_logps: torch.Tensor, rejected_logps: torch.Tensor, beta: float = 0.1):
    """TRL ``ORPOTrainer.odds_ratio_loss`` on length-averaged logps; minimise ``nll - losses.mean()``."""
    dtype = torch.promote_types(chosen_logps.dtype, torch.float32)
    chosen_logps, rejected_logps = chosen_logps.to(dtype), rejected_logps.to(dtype)
    log_odds = (chosen_logps - rejected_logps) - (_log1mexp(chosen_logps) - _log1mexp(rejected_logps))
    ratio = F.logsigmoid(log_odds)
    losses = beta * ratio
    chosen_rewards = beta * chosen_logps.detach()
    rejected_rewards = beta * rejected_logps.detach()
    return losses, chosen_rewards, rejected_rewards, torch.mean(ratio), torch.mean(log_odds)


def cpo_preference_loss(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    loss_type: str = "sigmoid",
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    simpo_gamma: float = 0.5,
    alpha: float = 0.0,
):
    """TRL ``CPOTrainer.cpo_loss`` (``loss_type="simpo"`` with ``cpo_alpha=0`` is SimPO)."""
    if alpha != 0.0:
        chosen_scores = (1 - torch.exp(chosen_logps).pow(-alpha)) / alpha
        rejected_scores = (1 - torch.exp(rejected_logps).pow(-alpha)) / alpha
    else:
        chosen_scores, rejected_scores = chosen_logps, rejected_logps
    logits = chosen_scores - rejected_scores
    if loss_type in ("simpo", "sigmoid"):
        if loss_type == "simpo":
            logits = logits - simpo_gamma / beta
        losses = (
            -F.logsigmoid(beta * logits) * (1 - label_smoothing)
            - F.logsigmoid(-beta * logits) * label_smoothing
        )
    elif loss_type == "hinge":
        losses = torch.relu(1 - beta * logits)
    elif loss_type == "ipo":
        losses = (logits - 1 / (2 * beta)) ** 2
    else:
        raise ValueError(f"Unsloth: unknown CPO loss_type {loss_type!r}; expected one of {CPO_LOSS_TYPES}.")
    chosen_rewards = beta * chosen_scores.detach()
    rejected_rewards = beta * rejected_scores.detach()
    return losses, chosen_rewards, rejected_rewards


def _sequence_logps(per_token_logps, mask, average):
    mask = mask.to(per_token_logps.dtype)
    total = (per_token_logps * mask).sum(-1)
    return total / mask.sum(-1) if average else total


def fused_linear_preference_loss(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: Optional[torch.Tensor],
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    method: str = "dpo",
    loss_type: str = "sigmoid",
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    ref_chosen_logps: Optional[torch.Tensor] = None,
    ref_rejected_logps: Optional[torch.Tensor] = None,
    ref_hidden_states: Optional[torch.Tensor] = None,
    ref_lm_head_weight: Optional[torch.Tensor] = None,
    ref_lm_head_bias: Optional[torch.Tensor] = None,
    reference_free: bool = False,
    simpo_gamma: float = 0.5,
    cpo_alpha: float = 1.0,
    nll_mask: Optional[torch.Tensor] = None,
    logit_scale_multiply: float = 0.0,
    logit_scale_divide: float = 0.0,
    logit_softcapping: float = 0.0,
    chunk_size: Optional[int] = None,
):
    """Preference loss from hidden states ``[2B, T, H]``, chosen rows first then rejected.

    ``method`` is ``"dpo"`` (``DPO_LOSS_TYPES``; ``ipo`` length-averages as TRL does), ``"orpo"`` or
    ``"cpo"`` (``simpo`` with ``cpo_alpha=0`` is SimPO). DPO reference logps are passed in
    (precomputed) or scored without grad from ``ref_hidden_states``. ``nll_mask`` picks the NLL
    tokens of ORPO / CPO (default ``loss_mask``). Returns ``(loss, metrics)`` with TRL's metric names.
    """
    transforms = dict(
        logit_scale_multiply = logit_scale_multiply,
        logit_scale_divide = logit_scale_divide,
        logit_softcapping = logit_softcapping,
        chunk_size = chunk_size,
    )
    n_pairs = hidden_states.shape[0] // 2
    loss_mask = loss_mask.bool()
    score_mask = loss_mask if nll_mask is None else (loss_mask | nll_mask.bool())
    per_token = fused_linear_token_logps(
        hidden_states, lm_head_weight, lm_head_bias, labels, score_mask, **transforms,
    )["per_token_logps"]

    average = method in ("orpo",) or (method == "dpo" and loss_type == "ipo") or (
        method == "cpo" and loss_type in ("ipo", "simpo")
    )
    logps = _sequence_logps(per_token, loss_mask, average)
    chosen_logps, rejected_logps = logps[:n_pairs], logps[n_pairs:]
    metrics = {}

    if method == "dpo":
        if ref_chosen_logps is None or ref_rejected_logps is None:
            if ref_hidden_states is None:
                raise ValueError("Unsloth: DPO needs ref_chosen_logps / ref_rejected_logps or ref_hidden_states.")
            with torch.no_grad():
                ref_per_token = fused_linear_token_logps(
                    ref_hidden_states,
                    lm_head_weight if ref_lm_head_weight is None else ref_lm_head_weight,
                    lm_head_bias if ref_lm_head_weight is None else ref_lm_head_bias,
                    labels, loss_mask, **transforms,
                )["per_token_logps"]
                ref_logps = _sequence_logps(ref_per_token, loss_mask, average)
            ref_chosen_logps, ref_rejected_logps = ref_logps[:n_pairs], ref_logps[n_pairs:]
        losses, chosen_rewards, rejected_rewards = dpo_preference_loss(
            chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps,
            loss_type = loss_type, beta = beta, label_smoothing = label_smoothing,
            reference_free = reference_free,
        )
        loss = losses.mean()
    else:
        nll_tokens = (loss_mask if nll_mask is None else nll_mask.bool())[:n_pairs]
        nll_loss = -(per_token[:n_pairs] * nll_tokens).sum() / nll_tokens.sum().clamp(min = 1)
        if method == "orpo":
            losses, chosen_rewards, rejected_rewards, log_odds_ratio, log_odds_chosen = orpo_preference_loss(
                chosen_logps, rejected_logps, beta = beta,
            )
            loss = nll_loss - losses.mean()
            metrics["log_odds_ratio"] = log_odds_ratio.detach()
            metrics["log_odds_chosen"] = log_odds_chosen.detach()
        elif method == "cpo":
            losses, chosen_rewards, rejected_rewards = cpo_preference_loss(
                chosen_logps, rejected_logps, loss_type = loss_type, beta = beta,
                label_smoothing = label_smoothing, simpo_gamma = simpo_gamma,
            )
            loss = losses.mean() + cpo_alpha * nll_loss
        else:
            raise ValueError(f"Unsloth: unknown preference method {method!r}; expected dpo, orpo or cpo.")
        metrics["nll_loss"] = nll_loss.detach()

    metrics["rewards/chosen"] = chosen_rewards.mean()
    metrics["rewards/rejected"] = rejected_rewards.mean()
    metrics["rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().mean()
    metrics["rewards/margins"] = (chosen_rewards - rejected_rewards).mean()
    metrics["logps/chosen"] = chosen_logps.detach().mean()
    metrics["logps/rejected"] = rejected_logps.detach().mean()
    return loss, metrics
