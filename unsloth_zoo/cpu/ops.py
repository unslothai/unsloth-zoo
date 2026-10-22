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

"""Pure-torch CPU replacements. No Triton, no CUDA."""

__all__ = [
    "chunked_linear_cross_entropy",
    "patch_chunked_causal_lm_loss",
]

import math
import os
import types

import torch
import torch.nn.functional as F


class _ChunkedLinearCrossEntropy(torch.autograd.Function):
    """sum CE(hidden @ weight.T, labels) / divisor without materialising the (rows, vocab) logits.

    Gradients are formed chunk by chunk inside forward (the loss is a scalar, so backward only
    rescales them): peak extra memory is one (chunk, vocab) fp32 block instead of the full logits
    plus their gradient, which for a 150k vocab is the largest tensor of a CPU training step.
    """

    @staticmethod
    def forward(ctx, hidden, weight, labels, divisor, chunk_rows, ignore_index):
        rows = hidden.shape[0]
        need_hidden_grad = hidden.requires_grad
        need_weight_grad = weight.requires_grad
        grad_hidden = torch.empty_like(hidden) if need_hidden_grad else None
        grad_weight = torch.zeros(weight.shape, dtype = torch.float32) if need_weight_grad else None
        total = torch.zeros((), dtype = torch.float32)
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            h = hidden[start:end]
            y = labels[start:end]
            logits = (h @ weight.t()).float()
            lse = torch.logsumexp(logits, dim = -1)
            valid = y != ignore_index
            safe_y = torch.where(valid, y, torch.zeros_like(y))
            target = logits.gather(-1, safe_y.unsqueeze(-1)).squeeze(-1)
            total += ((lse - target) * valid).sum()
            if need_hidden_grad or need_weight_grad:
                # d/dlogits of sum CE = softmax - onehot on valid rows, zero elsewhere.
                logits.sub_(lse.unsqueeze(-1)).exp_()
                logits.scatter_add_(-1, safe_y.unsqueeze(-1), -torch.ones_like(target).unsqueeze(-1))
                logits.mul_(valid.unsqueeze(-1))
                dlogits = logits.to(h.dtype)
                if need_hidden_grad:
                    grad_hidden[start:end] = dlogits @ weight
                if need_weight_grad:
                    grad_weight.addmm_(logits.t(), h.float())
        loss = total / divisor
        ctx.divisor = divisor
        ctx.save_for_backward(
            grad_hidden if grad_hidden is not None else torch.empty(0),
            grad_weight if grad_weight is not None else torch.empty(0),
        )
        ctx.has_hidden = need_hidden_grad
        ctx.has_weight = need_weight_grad
        ctx.weight_dtype = weight.dtype
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        grad_hidden, grad_weight = ctx.saved_tensors
        # Every label ignored: the loss is NaN like F.cross_entropy(mean), but its gradient is zero.
        scale = grad_output / ctx.divisor if not math.isnan(ctx.divisor) else torch.zeros_like(grad_output)
        dh = (grad_hidden * scale.to(grad_hidden.dtype)) if ctx.has_hidden else None
        dw = (grad_weight * scale).to(ctx.weight_dtype) if ctx.has_weight else None
        return dh, dw, None, None, None, None


def chunked_linear_cross_entropy(hidden, weight, labels, num_items_in_batch = None,
                                 ignore_index = -100, shift_labels = None, chunk_rows = None):
    """Causal-LM loss with the same shift and normalisation as transformers' ForCausalLMLoss."""
    if shift_labels is None:
        labels = F.pad(labels, (0, 1), value = ignore_index)
        shift_labels = labels[..., 1:]
    hidden = hidden.reshape(-1, hidden.shape[-1])
    shift_labels = shift_labels.reshape(-1).to(hidden.device)
    if num_items_in_batch is None:
        divisor = (shift_labels != ignore_index).sum().clamp(min = 1).item()
        # Matches F.cross_entropy(reduction="mean"): NaN when every label is ignored.
        if (shift_labels != ignore_index).sum().item() == 0:
            divisor = float("nan")
    else:
        divisor = num_items_in_batch.item() if torch.is_tensor(num_items_in_batch) else num_items_in_batch
    if chunk_rows is None:
        # ~64 MB of fp32 logits per chunk.
        chunk_rows = max(16, int(os.environ.get("UNSLOTH_CPU_CE_CHUNK_ELEMENTS", 16 * 1024 * 1024)) // weight.shape[0])
    return _ChunkedLinearCrossEntropy.apply(hidden, weight, shift_labels, float(divisor), chunk_rows, ignore_index)


def _plain_lm_head(model):
    """Only models whose logits are exactly lm_head(hidden): no softcap, scale or bias."""
    config = getattr(model, "config", None)
    text_config = getattr(config, "get_text_config", lambda: config)()
    for name in ("final_logit_softcapping", "logit_scale", "logits_scaling", "output_multiplier_scale"):
        value = getattr(text_config, name, None)
        if value not in (None, 1, 1.0):
            return False
    # MoE heads add a router load-balancing loss inside their own forward.
    if getattr(text_config, "output_router_logits", False) or getattr(text_config, "num_experts", None) \
            or getattr(text_config, "num_local_experts", None) or getattr(text_config, "n_routed_experts", None):
        return False
    lm_head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if not isinstance(lm_head, torch.nn.Linear) or lm_head.bias is not None:
        return False
    inner = getattr(model, getattr(model, "base_model_prefix", "model"), None)
    return isinstance(inner, torch.nn.Module)


def patch_chunked_causal_lm_loss(model):
    """Route training forwards with labels through chunked_linear_cross_entropy.

    Instance-level, so only this model changes. The returned logits are None in that case;
    unsloth_zoo.cpu.trainer makes TRL skip its logits-only metrics for it.
    Returns True when installed.
    """
    if os.environ.get("UNSLOTH_CPU_CHUNKED_CE", "1") == "0" or not _plain_lm_head(model):
        return False
    if getattr(model, "_unsloth_cpu_chunked_ce", False):
        return True
    from transformers.modeling_outputs import CausalLMOutputWithPast

    original_forward = model.forward
    inner_name = getattr(model, "base_model_prefix", "model")

    def forward(self, input_ids = None, attention_mask = None, position_ids = None,
                past_key_values = None, inputs_embeds = None, labels = None, use_cache = None,
                logits_to_keep = 0, **kwargs):
        shift_labels = kwargs.pop("shift_labels", None)
        if (labels is None and shift_labels is None) or not torch.is_grad_enabled() \
                or not getattr(self, "_unsloth_cpu_chunked_ce_active", True):
            if shift_labels is not None:
                kwargs["shift_labels"] = shift_labels
            return original_forward(
                input_ids = input_ids, attention_mask = attention_mask, position_ids = position_ids,
                past_key_values = past_key_values, inputs_embeds = inputs_embeds, labels = labels,
                use_cache = use_cache, logits_to_keep = logits_to_keep, **kwargs,
            )
        num_items_in_batch = kwargs.pop("num_items_in_batch", None)
        kwargs.pop("return_dict", None)
        outputs = getattr(self, inner_name)(
            input_ids = input_ids, attention_mask = attention_mask, position_ids = position_ids,
            past_key_values = past_key_values, inputs_embeds = inputs_embeds, use_cache = use_cache,
            **kwargs,
        )
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        loss = chunked_linear_cross_entropy(
            hidden, self.get_output_embeddings().weight, labels,
            num_items_in_batch = num_items_in_batch, shift_labels = shift_labels,
        )
        return CausalLMOutputWithPast(
            loss = loss, logits = None,
            past_key_values = getattr(outputs, "past_key_values", None),
            hidden_states = getattr(outputs, "hidden_states", None),
            attentions = getattr(outputs, "attentions", None),
        )

    model.forward = types.MethodType(forward, model)
    model._unsloth_cpu_chunked_ce = True
    return True
