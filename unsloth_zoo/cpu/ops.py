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

import functools
import math
import os
import types

import torch
import torch.nn.functional as F


def _cpu_fp16_autocast():
    try:
        return torch.is_autocast_enabled("cpu") and torch.get_autocast_dtype("cpu") == torch.float16
    except TypeError:
        # Older torch: the device-less CPU autocast API.
        return torch.is_autocast_cpu_enabled() and torch.get_autocast_cpu_dtype() == torch.float16


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
        # fp16 cannot hold these gradients before GradScaler multiplies them: keep them fp32 until backward.
        exact = hidden.dtype == torch.float16 or _cpu_fp16_autocast()
        grad_dtype = torch.float32 if exact else hidden.dtype
        grad_hidden = torch.empty(hidden.shape, dtype = grad_dtype) if need_hidden_grad else None
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
                if need_hidden_grad:
                    if exact:
                        with torch.autocast("cpu", enabled = False):
                            grad_hidden[start:end] = logits @ weight.float()
                    else:
                        grad_hidden[start:end] = logits.to(h.dtype) @ weight
                if need_weight_grad:
                    grad_weight.addmm_(logits.t(), h.float())
        # num_items_in_batch == 0 (a fully masked accumulation group): a finite 0, as TRL's clamped divisor gives.
        loss = total / (1.0 if divisor == 0 else divisor)
        ctx.divisor = divisor
        ctx.save_for_backward(
            grad_hidden if grad_hidden is not None else torch.empty(0),
            grad_weight if grad_weight is not None else torch.empty(0),
        )
        ctx.has_hidden = need_hidden_grad
        ctx.has_weight = need_weight_grad
        ctx.weight_dtype = weight.dtype
        ctx.hidden_dtype = hidden.dtype
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        grad_hidden, grad_weight = ctx.saved_tensors
        # Every label ignored: the loss is NaN like F.cross_entropy(mean), but its gradient is zero.
        # Also num_items_in_batch == 0 (a fully masked accumulation group): stock CE's gradient is zero there.
        empty = math.isnan(ctx.divisor) or ctx.divisor == 0
        scale = torch.zeros_like(grad_output) if empty else grad_output / ctx.divisor
        dh = (grad_hidden * scale.to(grad_hidden.dtype)).to(ctx.hidden_dtype) if ctx.has_hidden else None
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


# Architectures whose causal-LM loss is plain CE over lm_head(hidden). Others (soft caps, logit scales,
# z-loss, router losses, ...) keep their own forward.
_CHUNKED_CE_MODEL_TYPES = frozenset(("llama", "mistral", "qwen2", "qwen3", "phi3", "gemma", "olmo2", "smollm3"))


def _plain_lm_head(model):
    """Only models whose logits are exactly lm_head(hidden): no softcap, scale or bias."""
    config = getattr(model, "config", None)
    text_config = getattr(config, "get_text_config", lambda: config)()
    if getattr(text_config, "model_type", None) not in _CHUNKED_CE_MODEL_TYPES:
        return False
    for name in ("final_logit_softcapping", "logits_soft_cap", "logit_scale", "logits_scaling", "output_multiplier_scale"):
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


def _plain_head_weight(head):
    """The weight that alone produces this head's logits, or None (e.g. a LoRA-wrapped lm_head)."""
    if type(head) is torch.nn.Linear and head.bias is None:
        return head.weight
    if type(head).__name__ == "ModulesToSaveWrapper":
        adapter = getattr(head, "active_adapter", None)
        adapter = adapter[0] if isinstance(adapter, (list, tuple)) else adapter
        copies = getattr(head, "modules_to_save", None)
        active = copies[adapter] if copies is not None and adapter in copies else None
        return _plain_head_weight(active if active is not None else getattr(head, "original_module", None))
    return None


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
    # Call the class forward on `self`, not the bound method: deepcopy (TRL reference models,
    # merged copies) rebinds the outer method to the copy, and a bound original would still run
    # the source model's weights.
    original_func = original_forward.__func__ if getattr(original_forward, "__self__", None) is model else None
    inner_name = getattr(model, "base_model_prefix", "model")

    def forward(self, input_ids = None, attention_mask = None, position_ids = None,
                past_key_values = None, inputs_embeds = None, labels = None, use_cache = None,
                logits_to_keep = 0, **kwargs):
        shift_labels = kwargs.pop("shift_labels", None)
        head_weight = _plain_head_weight(self.get_output_embeddings())
        if (labels is None and shift_labels is None) or not torch.is_grad_enabled() \
                or not getattr(self, "_unsloth_cpu_chunked_ce_active", False) or head_weight is None:
            if shift_labels is not None:
                kwargs["shift_labels"] = shift_labels
            call = functools.partial(original_func, self) if original_func is not None else original_forward
            return call(
                input_ids = input_ids, attention_mask = attention_mask, position_ids = position_ids,
                past_key_values = past_key_values, inputs_embeds = inputs_embeds, labels = labels,
                use_cache = use_cache, logits_to_keep = logits_to_keep, **kwargs,
            )
        num_items_in_batch = kwargs.pop("num_items_in_batch", None)
        ignore_index = kwargs.pop("ignore_index", -100)
        kwargs.pop("return_dict", None)
        outputs = getattr(self, inner_name)(
            input_ids = input_ids, attention_mask = attention_mask, position_ids = position_ids,
            past_key_values = past_key_values, inputs_embeds = inputs_embeds, use_cache = use_cache,
            **kwargs,
        )
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        loss = chunked_linear_cross_entropy(
            hidden, head_weight, labels,
            num_items_in_batch = num_items_in_batch, shift_labels = shift_labels, ignore_index = ignore_index,
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
