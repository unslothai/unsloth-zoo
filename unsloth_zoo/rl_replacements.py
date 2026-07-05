# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = [
    "RL_REPLACEMENTS"
]

import torch
import inspect
import os
import math
import logging
import numpy as np
from typing import Union, Callable, Optional, List, Dict
from .device_type import DEVICE_TYPE, device_synchronize
from .temporary_patches.common import torch_compile_options
RL_REPLACEMENTS = dict()

# https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py#L1674
@torch.compile(dynamic = True, fullgraph = True, options = torch_compile_options,)
def selective_log_softmax(logits, index):
    logits = logits.to(torch.float32)
    selected_logits = torch.gather(logits, dim = -1, index = index.unsqueeze(-1)).squeeze(-1)
    logsumexp_values = torch.logsumexp(logits, dim = -1)
    per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    return per_token_logps
pass

# Memory-efficient chunked variant of the above on (bsz+qlen); exactly equivalent.
@torch.compile(dynamic = True, fullgraph = True, options = torch_compile_options,)
def chunked_selective_log_softmax(
    logits,
    index,
    temperature: float = 1.0,
    chunks: int = 4,
):
    chunked_logits = torch.chunk(logits.reshape(-1, logits.shape[-1]), chunks = chunks, dim = 0)
    chunked_index  = torch.chunk(index.reshape(-1), chunks = chunks, dim = 0)
    all_per_token_logps = []
    # Per-chunk selective_log_softmax.
    for chunk_logits, chunk_index in zip(chunked_logits, chunked_index):
        chunk_logits = chunk_logits.to(torch.float32)
        if temperature != 1.0:
            chunk_logits = chunk_logits / temperature
        selected_logits = torch.gather(chunk_logits, dim = -1, index = chunk_index.unsqueeze(-1)).squeeze(-1)
        logsumexp_values = torch.logsumexp(chunk_logits, dim = -1)
        per_token_logps = selected_logits - logsumexp_values
        all_per_token_logps.append(per_token_logps)
    pass
    all_per_token_logps = torch.concat(all_per_token_logps)
    all_per_token_logps = all_per_token_logps.reshape((logits.shape[0], logits.shape[1]))
    return all_per_token_logps
pass

RL_REPLACEMENTS["selective_log_softmax"] = chunked_selective_log_softmax

@torch.compile(dynamic = True, fullgraph = True, options = torch_compile_options,)
def chunked_hidden_states_selective_log_softmax(
    hidden_states: torch.Tensor,
    lm_head: torch.Tensor,
    index: torch.Tensor,
    chunks: int = 4,
    logit_scale_multiply: float = 0.0,
    logit_scale_divide: float = 0.0,
    logit_softcapping: float = 0.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    # All Unsloth Zoo code licensed under AGPL3
    flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_index = index.reshape(-1)

    chunked_hidden_states = torch.chunk(flat_hidden_states, chunks=chunks, dim=0)
    chunked_index = torch.chunk(flat_index, chunks=chunks, dim=0)

    all_per_token_logps = []

    for chunk_hidden_states, chunk_index in zip(chunked_hidden_states, chunked_index):
        chunk_logits = chunk_hidden_states.to(lm_head.dtype) @ lm_head.t()

        if logit_scale_multiply != 0.0:
            chunk_logits = chunk_logits * logit_scale_multiply
        if logit_scale_divide != 0.0:
            chunk_logits = chunk_logits / logit_scale_divide
        if logit_softcapping != 0.0:
            chunk_logits = logit_softcapping * torch.tanh(chunk_logits / logit_softcapping)

        chunk_logits = chunk_logits.to(torch.float32)

        if temperature != 1.0:
            chunk_logits = chunk_logits / temperature

        selected_logits = torch.gather(chunk_logits, dim=-1, index=chunk_index.unsqueeze(-1)).squeeze(-1)
        logsumexp_values = torch.logsumexp(chunk_logits, dim=-1)
        per_token_logps = selected_logits - logsumexp_values
        all_per_token_logps.append(per_token_logps)

    all_per_token_logps = torch.concat(all_per_token_logps)

    all_per_token_logps = all_per_token_logps.reshape((hidden_states.shape[0], hidden_states.shape[1]))
    return all_per_token_logps

RL_REPLACEMENTS["grpo_selective_log_softmax"] = chunked_hidden_states_selective_log_softmax

def calculate_pad_tokens_in_prompt(
    input_ids: torch.Tensor,
    logits_to_keep: int,
    pad_token_id: int
) -> torch.Tensor:
    """Count left-padded tokens per sequence, e.g. [pad, pad, pad, cat] -> 3."""
    if logits_to_keep >= input_ids.shape[1]:
        raise ValueError("logits_to_keep must be smaller than the sequence length.")

    prompt_section = input_ids[:, :-logits_to_keep]

    padding_mask = (prompt_section == pad_token_id)

    pad_token_counts = padding_mask.sum(dim=1)

    return pad_token_counts
pass
RL_REPLACEMENTS["calculate_pad_tokens_in_prompt"] = calculate_pad_tokens_in_prompt


def create_completion_attention_mask(
    completion_input_ids: torch.Tensor,
    left_pad_tokens_per_prompt: torch.Tensor,
    max_left_pad: int,
    pad_token_id: int
) -> torch.Tensor:
    """Build a completion mask that zeros leading prompt and trailing pad tokens.

    For [p,p,p,c,c,c,pad,pad,pad] (p=sliced prompt, c=completion, pad=padding)
    this returns [0,0,0,1,1,1,0,0,0].
    """
    batch_size, completion_len = completion_input_ids.shape
    device = completion_input_ids.device

    num_tokens_to_mask = max_left_pad - left_pad_tokens_per_prompt

    indices = torch.arange(completion_len, device=device).unsqueeze(0)
    shift_mask = indices >= num_tokens_to_mask.unsqueeze(1)

    non_padding_mask = (completion_input_ids != pad_token_id)

    final_mask = shift_mask & non_padding_mask

    return final_mask
pass
RL_REPLACEMENTS["create_completion_attention_mask"] = create_completion_attention_mask


# Rebuild Qwen-style mm_token_type_ids from full input_ids after GRPO generation changes sequence length.
# This is primarily towards VLMs that use MRoPE
def _unsloth_get_mm_token_id(processing_class, attr_name, token):
    tokenizer = getattr(processing_class, "tokenizer", processing_class)
    token_id = getattr(processing_class, attr_name, None)
    if token_id is None:
        token_id = getattr(tokenizer, attr_name, None)

    convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
    if token_id is None and convert_tokens_to_ids is not None:
        token_id = convert_tokens_to_ids(token)

    if type(token_id) is int and token_id >= 0:
        if token_id != getattr(tokenizer, "unk_token_id", None):
            return token_id
    return None
pass


def _unsloth_fix_mm_token_type_ids(
    processing_class, input_ids, mm_token_type_ids = None, completion_ids = None
):
    image_token_id = _unsloth_get_mm_token_id(
        processing_class, "image_token_id", "<|image_pad|>"
    )
    video_token_id = _unsloth_get_mm_token_id(
        processing_class, "video_token_id", "<|video_pad|>"
    )

    if image_token_id is not None or video_token_id is not None:
        rebuilt = input_ids.new_zeros(input_ids.shape)
        if image_token_id is not None:
            rebuilt = rebuilt.masked_fill(input_ids == image_token_id, 1)
        if video_token_id is not None:
            rebuilt = rebuilt.masked_fill(input_ids == video_token_id, 2)
        return rebuilt

    if (
        mm_token_type_ids is not None
        and completion_ids is not None
        and mm_token_type_ids.shape[0] == input_ids.shape[0]
        and mm_token_type_ids.shape[1] + completion_ids.shape[1] == input_ids.shape[1]
    ):
        return torch.cat(
            [mm_token_type_ids, mm_token_type_ids.new_zeros(completion_ids.shape)],
            dim = 1,
        )
    return mm_token_type_ids
pass

def left_pack_padding(tensor: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Move all padding tokens in each sequence to the right."""
    mask = (tensor != pad_id)
    # stable=True since the binary mask is unordered.
    sorted_indices = torch.argsort(mask, dim=1, descending=True, stable=True)
    packed_tensor = torch.gather(tensor, 1, sorted_indices)
    return packed_tensor
pass
RL_REPLACEMENTS["left_pack_padding"] = left_pack_padding

def align_logprobs_with_mask(
    logprob_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_value: float = 0.0
) -> torch.Tensor:
    """Align a log probability tensor with a given attention mask."""

    device = logprob_tensor.device
    batch_size, logprob_seq_len = logprob_tensor.shape
    mask_seq_len = attention_mask.shape[1]

    padded_logprobs = torch.full(
        attention_mask.shape,
        fill_value=pad_value,
        dtype=logprob_tensor.dtype,
        device=device
    )

    left_pad_counts = torch.argmax(attention_mask, dim=1)

    cols = torch.arange(logprob_seq_len, device=device)


    dest_indices = left_pad_counts.unsqueeze(1) + cols

    # Destination row indices, shape [batch_size, logprob_seq_len].
    row_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(dest_indices)

    # Keep only in-bounds destinations, then scatter via advanced indexing.
    valid_mask = dest_indices < mask_seq_len
    valid_rows = row_indices[valid_mask]
    valid_cols = dest_indices[valid_mask]
    valid_vals = logprob_tensor[valid_mask]
    padded_logprobs[valid_rows, valid_cols] = valid_vals

    return padded_logprobs

RL_REPLACEMENTS["align_logprobs_with_mask"] = align_logprobs_with_mask

def align_completion_tool_mask(
    tool_mask: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """Align a raw completion-length tool/env mask with Unsloth's repacked loss mask."""
    if tool_mask is None:
        return completion_mask
    if tool_mask.shape[0] != completion_mask.shape[0]:
        raise ValueError("tool_mask batch size must match completion_mask batch size.")

    tool_mask = tool_mask.to(device=completion_mask.device)
    if tool_mask.shape == completion_mask.shape:
        aligned_tool_mask = tool_mask
    else:
        aligned_tool_mask = align_logprobs_with_mask(
            tool_mask,
            completion_mask,
            pad_value=0,
        )
    return completion_mask * aligned_tool_mask.to(dtype=completion_mask.dtype)
pass
RL_REPLACEMENTS["align_completion_tool_mask"] = align_completion_tool_mask

def autotune_batch_and_chunks(
    total_input_rows,
    seq_len,
    hidden_size,
    vocab_size,
    dtype_bytes=16,
    multiplier=None
):
    if multiplier is None:
        final_m = max(4, seq_len // 4096)
    else:
        final_m = multiplier

    if torch.cuda.is_available():
        free_bytes, _ = torch.cuda.mem_get_info()
        limit_gb = (free_bytes / (1024**3))*.80
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        # XPU: estimate free memory as total - reserved.
        total_mem = torch.xpu.get_device_properties(0).total_memory
        reserved_mem = torch.xpu.memory_reserved()
        free_bytes = total_mem - reserved_mem
        limit_gb = (free_bytes / (1024**3)) * 0.80
    else:
        # Fallback: assume 8GB available.
        limit_gb = 8.0

    bytes_to_gb = 1024**3

    b_vals = torch.arange(total_input_rows, 0, -1, device='cpu', dtype=torch.float32)

    hidden_gb = (b_vals * seq_len * hidden_size * dtype_bytes) / bytes_to_gb

    base_logits = ((b_vals/total_input_rows) * b_vals * seq_len * vocab_size * dtype_bytes) / bytes_to_gb
    logits_gb = base_logits / final_m

    total_mem_gb = hidden_gb + logits_gb

    valid_mask = total_mem_gb <= limit_gb
    valid_indices = torch.nonzero(valid_mask, as_tuple=False)

    if valid_indices.shape[0] == 0:
        #This means your GPU will OOM
        return 4, final_m

    best_idx = valid_indices[0].item()
    final_b = int(b_vals[best_idx].item())

    return final_b, final_m

RL_REPLACEMENTS["grpo_autotune_batch_and_chunks"] = autotune_batch_and_chunks


def grpo_update_SamplingParams(SamplingParams, generation_kwargs, vllm_sampling_params = None):
    good_sampling_params_keys = inspect.signature(SamplingParams).parameters.keys()

    new_generation_kwargs = {}
    for key in generation_kwargs.keys():
        if key in good_sampling_params_keys:
            new_generation_kwargs[key] = generation_kwargs[key]
    generation_kwargs = new_generation_kwargs

    if vllm_sampling_params is not None:
        for key in good_sampling_params_keys:
            if hasattr(vllm_sampling_params, key):
                overwrited_key = getattr(vllm_sampling_params, key)
                if overwrited_key is not None and (type(overwrited_key) in (list, tuple,) and len(overwrited_key) != 0):
                    generation_kwargs[key] = overwrited_key
    return generation_kwargs
pass
RL_REPLACEMENTS["grpo_update_SamplingParams"] = grpo_update_SamplingParams


def sanitize_logprob(logprob):
    """Local port of trl.scripts.vllm_serve.sanitize_logprob.
    Filters NaN logprobs from vLLM outputs."""
    value = logprob.logprob
    if math.isnan(value):
        logging.getLogger(__name__).warning(
            f"Generated NaN logprob, token logprob '{logprob}' will be ignored"
        )
        return None
    return value

RL_REPLACEMENTS["sanitize_logprob"] = sanitize_logprob
# Custom compiled GRPO loss - creates 3 Triton kernels
def grpo_compute_loss(
    ref,
    new,
    old,
    sampling_per_token_logps,
    input_ids,
    mask,
    beta,
    advantages,
    **kwargs
):
    # All Unsloth Zoo code licensed under AGPL3
    # Optional argument defaults.
    loss_type = kwargs.get("loss_type", "grpo")
    epsilon_low = kwargs.get("epsilon_low", 0.2)
    epsilon_high = kwargs.get("epsilon_high", 0.2)
    max_completion_length = kwargs.get("max_completion_length", 8192)
    delta = kwargs.get("delta", None)
    importance_sampling_level = kwargs.get("importance_sampling_level", "token")
    num_items_in_batch = kwargs.get("num_items_in_batch", None)
    current_gradient_accumulation_steps = kwargs.get("current_gradient_accumulation_steps", 1)
    num_processes = kwargs.get("num_processes", 1)
    use_vllm = kwargs.get("use_vllm", False)
    vllm_importance_sampling_cap = kwargs.get("vllm_importance_sampling_cap", 2.0)
    get_sapo_token_loss = kwargs.get("get_sapo_token_loss", None)
    sapo_temperature_pos = kwargs.get("sapo_temperature_pos", 1.0)
    sapo_temperature_neg = kwargs.get("sapo_temperature_neg", 1.05)
    get_gamma_weights = kwargs.get("get_gamma_weights", None)
    vespo_k_pos = kwargs.get("vespo_k_pos", 2.0)
    vespo_lambda_pos = kwargs.get("vespo_lambda_pos", 3.0)
    vespo_k_neg = kwargs.get("vespo_k_neg", 3.0)
    vespo_lambda_neg = kwargs.get("vespo_lambda_neg", 2.0)
    get_off_policy_mask = kwargs.get("get_off_policy_mask", None)
    off_policy_mask_threshold  = kwargs.get("off_policy_mask_threshold", None)
    input_ids = input_ids.unsqueeze(-1)

    # exp(new - old) and exp(ref - new) below are taken before `mask` is applied. A sequence-packed
    # logp path leaves the masked (prompt/pad) columns at 0 while a padded one fills them with a real
    # logp, so when new and old/ref disagree there those ratios can overflow to inf and inf * 0 (the
    # masked-out loss) becomes nan. Force new/old/ref to share 0 on the masked columns so both ratios
    # are exp(0) = 1 there; every loss term below multiplies by `mask`, so this changes nothing.
    if mask is not None:
        _keep = mask.to(torch.bool)
        new = torch.where(_keep, new, 0.0)
        if old is not None: old = torch.where(_keep, old, 0.0)
        if ref is not None: ref = torch.where(_keep, ref, 0.0)

    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1)

    if off_policy_mask_threshold is not None:
        off_policy_mask = get_off_policy_mask(
            advantages=advantages,
            per_token_logps=new,
            old_per_token_logps=old,
            mask=mask,
            off_policy_threshold=off_policy_mask_threshold,
        )

    with torch.no_grad():
        if use_vllm and sampling_per_token_logps is not None:
            # Filter out extra leading prompt tokens after left-padding input_ids.
            importance_sampling_ratio = torch.exp((old * mask) - sampling_per_token_logps)
            importance_sampling_ratio = torch.clamp(
                importance_sampling_ratio, max=vllm_importance_sampling_cap
            )
    pass

    # Must detach when old is None: exp(new - new.detach()) == 1 but keeps grads correct.
    if old is not None:
        log_ratio = new - old
    else:
        log_ratio = new - new.detach()

    if importance_sampling_level == "token":
        log_importance_weights = log_ratio
    elif importance_sampling_level == "sequence":
        log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
        log_importance_weights = log_importance_weights.unsqueeze(-1)
    else:
        raise ValueError(
            f"Unknown importance sampling level: {importance_sampling_level}. Possible values are 'token' "
            "and 'sequence'."
        )

    coef_1 =  torch.exp(log_importance_weights)

    # Reverse KL: low-variance low-bias estimator as used in the GRPO paper.
    if beta != 0.0:
        kl_i = torch.exp(ref - new) - (ref - new) - 1.0

    else:
        # Zeros with the correct shape.
        if importance_sampling_level == "sequence":
            kl_i = new.new_zeros(new.size(0), 1)
        else:
            kl_i = torch.zeros_like(new)

    if loss_type == "cispo":
        clamped_ratios = torch.clamp(coef_1, max=epsilon_high).detach()
        loss_i = -clamped_ratios * advantages * new
    elif loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
        coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)

        if delta is not None:
            loss_1 = torch.clamp(coef_1, max=delta) * advantages
        else:
            loss_1 = coef_1 * advantages
        pass
        loss_2 = coef_2 * advantages
        loss_i = -torch.min(loss_1, loss_2)
    elif loss_type == "sapo":
        if get_sapo_token_loss is None:
            raise Exception(f"sapo is only available in TRL 0.26.0+")
        loss_i = torch.empty_like(coef_1)
        positive_advantages_mask = advantages.repeat([1, coef_1.shape[1]]) > 0
        # With n_chunks some tensors may be empty; guard the indexing.
        if coef_1[positive_advantages_mask].numel() != 0:
            loss_i[positive_advantages_mask] = get_sapo_token_loss(
                coef_1[positive_advantages_mask], sapo_temperature_pos
            )
        if coef_1[~positive_advantages_mask].numel() != 0:
            loss_i[~positive_advantages_mask] = get_sapo_token_loss(
                coef_1[~positive_advantages_mask], sapo_temperature_neg
            )
        loss_i = -loss_i * advantages
    elif loss_type == "vespo":
        if get_gamma_weights is None:
            raise Exception("vespo is only available in TRL 0.26.0+")
        phi_seq = get_gamma_weights(
            advantages=advantages,
            log_ratio_per_token=log_ratio,
            mask=mask,
            importance_sampling_ratio=kwargs.get("importance_sampling_ratio"),
            k_pos=vespo_k_pos,
            lambda_pos=vespo_lambda_pos,
            k_neg=vespo_k_neg,
            lambda_neg=vespo_lambda_neg,
        )
        loss_i = -phi_seq * advantages * new
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    if off_policy_mask_threshold is not None:
        loss_i = loss_i * off_policy_mask

    if use_vllm and sampling_per_token_logps is not None:
        loss_i = loss_i * importance_sampling_ratio
        # delta for the metric.
        with torch.no_grad():
            delta = torch.abs(old - sampling_per_token_logps)
            delta = delta * mask
            flat_is_ratio = importance_sampling_ratio * mask
    else:
        delta = torch.tensor([]).detach()
        flat_is_ratio = torch.tensor([]).detach()
    if beta != 0.0:
        loss_i = loss_i + beta * kl_i

    mask = mask.to(torch.float32)
    n_mask_per_reward = mask.sum(1)

    # https://github.com/huggingface/trl/blob/e8b8499f1f8d76838155b515e414ee98f757d6d5/trl/trainer/grpo_trainer.py#L1624
    if loss_type in ["grpo", "sapo"]:
        loss = ((loss_i * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
        loss = loss / current_gradient_accumulation_steps
    elif loss_type == "bnpo":
        loss = (loss_i * mask).sum() / mask.sum().clamp(min=1.0)
        loss = loss / current_gradient_accumulation_steps
    elif loss_type == "dr_grpo":
        loss = (loss_i * mask).sum() / (loss_i.size(0) * max_completion_length)
        loss = loss / current_gradient_accumulation_steps
    elif loss_type in ["cispo", "dapo", "vespo"]:
        normalizer = num_items_in_batch/ num_processes
        loss = (loss_i * mask).sum() / normalizer
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Folded metrics.
    def masked_batch_mean(x):
        with torch.inference_mode():
            completion_length = n_mask_per_reward.mean()
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return completion_length, x.mean()
            else:
                mean_kl_per_reward = (x * mask).sum(1) / n_mask_per_reward
                mean_kl = mean_kl_per_reward.mean()
                return completion_length, mean_kl
    completion_length, mean_kl = masked_batch_mean(kl_i)
    return loss, completion_length, mean_kl, delta, flat_is_ratio, coef_1, mask
pass
RL_REPLACEMENTS["grpo_compute_loss"]      = grpo_compute_loss
RL_REPLACEMENTS["grpo_compute_loss_slow"] = \
    f"@torch.compile(dynamic = True, fullgraph = True, options = torch_compile_options)\n"\
    f"{inspect.getsource(grpo_compute_loss)}"
RL_REPLACEMENTS["grpo_compute_loss_slow"] = \
    RL_REPLACEMENTS["grpo_compute_loss_slow"].replace(
        "def grpo_compute_loss",
        "def grpo_compute_loss_slow",
)

# Unsloth's memory efficient GRPO implementation
class UnslothEfficientGRPO(torch.autograd.Function):
    # All Unsloth Zoo code licensed under AGPL3
    @staticmethod
    def forward(ctx, _new_logps, _old_logps, _ref_logps, _sampling_per_token_logps, lm_head, _input_ids, _mask, _advantages, beta, scaler = None, n_chunks = 1, extra_kwargs=None):
        if extra_kwargs is None:
            extra_kwargs = {}
        def compute_loss(new_logps, old_logps, ref_logps, sampling_per_token_logps, input_ids, mask, advantages, scaling):
            loss, completion_length, mean_kl, delta, flat_is_ratio, coef_1, _mask  = grpo_compute_loss(
                ref_logps,
                new_logps,
                old_logps,
                sampling_per_token_logps,
                input_ids,
                mask,
                beta,
                advantages,
                **extra_kwargs,
            )

            # Scale for mixed precision; return loss.detach() or autograd uses 2x VRAM.
            scaled_loss = loss * scaling
            return scaled_loss, (loss.detach(), completion_length, mean_kl, delta, flat_is_ratio, coef_1)
        pass

        device =_new_logps.device
        grad_inputs = torch.empty_like(_new_logps)
        accumulated_loss              = torch.zeros(1, device = device)[0]
        accumulated_completion_length = torch.zeros(1, device = device)[0]
        accumulated_mean_kl           = torch.zeros(1, device = device)[0]
        accumulated_delta             = []
        accumulated_flat_is_ratio     = []
        accumulated_coef_1            = []

        def accumulate_chunk(
            new_logps_j,
            old_logps_j,
            ref_logps_j,
            sampling_per_token_logps_j,
            input_ids_j,
            mask_j,
            advantages_j,
            scaling,
            grad_inputs_j,
        ):
            (chunk_grad_input,), (chunk_loss, (unscaled_loss, chunk_completion_length, chunk_mean_kl, chunk_delta, chunk_flat_is_ratio, chunk_coef_1)) = torch.func.grad_and_value(
                compute_loss,
                argnums = (0,),
                has_aux = True,
            )(new_logps_j, old_logps_j, ref_logps_j, sampling_per_token_logps_j, input_ids_j, mask_j, advantages_j, scaling)
            accumulated_loss             .add_(unscaled_loss)
            accumulated_completion_length.add_(chunk_completion_length)
            accumulated_mean_kl          .add_(chunk_mean_kl)
            accumulated_delta            .append(chunk_delta)
            accumulated_flat_is_ratio    .append(chunk_flat_is_ratio)
            accumulated_coef_1           .append(chunk_coef_1)
            grad_inputs_j[:] = chunk_grad_input
        pass

        accumulate_chunk = torch.compile(
            accumulate_chunk,
            fullgraph = True,
            # [TODO] Dynamic marking causes torch.compile errors if sequence length is long
            dynamic = True,
            options = torch_compile_options,
        )

        grad_inputs_chunks = torch.chunk(grad_inputs,        chunks = n_chunks, dim = 0)
        new_logps  = torch.chunk(_new_logps, chunks = n_chunks, dim = 0)
        if _old_logps is not None:
            old_logps  = torch.chunk(_old_logps, chunks = n_chunks, dim = 0)
        else:
            old_logps = [None] * n_chunks
        if _ref_logps is not None:
            ref_logps  = torch.chunk(_ref_logps, chunks = n_chunks, dim = 0)
        else:
            ref_logps = [None] * n_chunks
        if _sampling_per_token_logps is not None:
            sampling_per_token_logps  = torch.chunk(_sampling_per_token_logps, chunks = n_chunks, dim = 0)
        else:
            sampling_per_token_logps = [None] * n_chunks
        input_ids          = torch.chunk(_input_ids,         chunks = n_chunks, dim = 0)
        mask               = torch.chunk(_mask,              chunks = n_chunks, dim = 0)
        advantages         = torch.chunk(_advantages,        chunks = n_chunks, dim = 0)

        # Mixed precision scaling if present.
        scaling = scaler.get_scale() if scaler is not None else 1.0

        for (grad_inputs_j, new_logps_j, old_logps_j, ref_logps_j, sampling_per_token_logps_j, input_ids_j, mask_j, advantages_j, ) in \
            zip(grad_inputs_chunks, new_logps, old_logps, ref_logps, sampling_per_token_logps, input_ids, mask, advantages):

            # [TODO] Dynamic marking causes torch.compile errors if sequence length is long

            # mark_dynamic(new_hidden_states_j)
            # mark_dynamic(ref_hidden_states_j)
            # if old_hidden_states_j is not None:
            #     mark_dynamic(old_hidden_states_j)
            # mark_dynamic(input_ids_j)
            # mark_dynamic(mask_j)
            accumulate_chunk(
                new_logps_j,
                old_logps_j,
                ref_logps_j,
                sampling_per_token_logps_j,
                input_ids_j,
                mask_j,
                advantages_j,
                scaling,
                grad_inputs_j,
            )
        pass

        grad_inputs                  .div_(n_chunks)
        accumulated_loss             .div_(n_chunks)
        accumulated_completion_length.div_(n_chunks)
        accumulated_mean_kl          .div_(n_chunks)

        if _sampling_per_token_logps is not None:
            accumulated_delta = torch.cat(accumulated_delta, dim=0)
            accumulated_flat_is_ratio = torch.cat(accumulated_flat_is_ratio, dim=0)
        else:
            accumulated_delta = None
            accumulated_flat_is_ratio = None
        accumulated_coef_1  = torch.cat(accumulated_coef_1, dim=0)
        ctx.save_for_backward(grad_inputs)
        return (
            accumulated_loss,
            accumulated_completion_length,
            accumulated_mean_kl,
            accumulated_delta,
            accumulated_flat_is_ratio,
            accumulated_coef_1
        )
    pass

    @staticmethod
    def backward(ctx, grad_output, dcompletion_length, dmean_kl, ddelta, ddflat_is_ratio, dcoef_1):
        (grad_input,) = ctx.saved_tensors
        return (grad_input, None, None, None, None, None, None, None, None, None, None, None)
    pass
pass
RL_REPLACEMENTS["UnslothEfficientGRPO"] = UnslothEfficientGRPO


def grpo_accumulated_loss(
    trainer,
    input_ids,
    attention_mask,
    logits_to_keep,
    completion_mask,
    advantages,
    old_logps,
    ref_logps,
    n_chunks = -1,
    tool_mask = None,
    **kwargs,
):
    # All Unsloth Zoo code licensed under AGPL3
    bsz, qlen = input_ids.shape

    pixel_values = kwargs.get('pixel_values',None)
    image_grid_thw = kwargs.get('image_grid_thw',None)
    pixel_attention_mask = kwargs.get('pixel_attention_mask',None)
    image_sizes = kwargs.get('image_sizes',None)
    num_images = kwargs.get('num_images',None)
    # Transformers 5.x requires token_type_ids/mm_token_type_ids for some vision models
    token_type_ids = kwargs.get('token_type_ids',None)
    mm_token_type_ids = kwargs.get('mm_token_type_ids',None)
    if mm_token_type_ids is not None or image_grid_thw is not None:
        mm_token_type_ids = _unsloth_fix_mm_token_type_ids(
            trainer.processing_class, input_ids, mm_token_type_ids
        )
    sampling_per_token_logps = kwargs.get("sampling_per_token_logps", None) if getattr(trainer, "vllm_importance_sampling_correction", False) else None
    temperature = kwargs.get("temperature", 1.0)
    logit_scale_multiply = kwargs.get("logit_scale_multiply", 0.0)
    logit_scale_divide   = kwargs.get("logit_scale_divide", 0.0)
    logit_softcapping    = kwargs.get("logit_softcapping", 0.0)
    prev_max_left_pad    = kwargs.get("max_left_pad", 0) # max_left_pad for LLM training, enabled by default.

    # Pop from kwargs to avoid downstream issues.
    _ = kwargs.pop("sampling_per_token_logps", None)
    kwargs["vllm_importance_sampling_cap"] = trainer.vllm_importance_sampling_cap if sampling_per_token_logps is not None else None
    kwargs["get_sapo_token_loss"] = trainer.get_sapo_token_loss if hasattr(trainer, "get_sapo_token_loss") else None
    kwargs["sapo_temperature_pos"] = trainer.args.sapo_temperature_pos if hasattr(trainer.args, "sapo_temperature_pos") else None
    kwargs["sapo_temperature_neg"] = trainer.args.sapo_temperature_neg if hasattr(trainer.args, "sapo_temperature_neg") else None
    kwargs["get_gamma_weights"] = trainer.get_gamma_weights if hasattr(trainer, "get_gamma_weights") else None
    kwargs["vespo_k_pos"] = trainer.args.vespo_k_pos if hasattr(trainer.args, "vespo_k_pos") else 2.0
    kwargs["vespo_k_neg"] = trainer.args.vespo_k_neg if hasattr(trainer.args, "vespo_k_neg") else 3.0
    kwargs["vespo_lambda_pos"] = trainer.args.vespo_lambda_pos if hasattr(trainer.args, "vespo_lambda_pos") else 3.0
    kwargs["vespo_lambda_neg"] = trainer.args.vespo_lambda_neg if hasattr(trainer.args, "vespo_lambda_neg") else 2.0
    kwargs["get_off_policy_mask"] = trainer.get_off_policy_mask if hasattr(trainer, "get_off_policy_mask") else None
    kwargs["off_policy_mask_threshold"] = trainer.args.off_policy_mask_threshold  if hasattr(trainer.args, "off_policy_mask_threshold") else None
    kwargs["use_vllm"] = trainer.use_vllm
    # Snap n_chunks to the closest divisor of bsz.
    factors = [i for i in range(1, bsz + 1) if bsz % i == 0]
    if n_chunks == -1: n_chunks = bsz
    n_chunks = factors[min(np.searchsorted(factors, n_chunks), len(factors)-1)]

    if not hasattr(trainer, '_autocast_dtype'):
        trainer._autocast_dtype = torch.float16 if os.environ.get('ACCELERATE_MIXED_PRECISION', 'fp16') == 'fp16' else torch.bfloat16
        if os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '1': trainer._autocast_dtype = None
    pass
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"

    lm_head = trainer.model.get_output_embeddings().weight
    dtype_bytes = 16 if trainer._autocast_dtype in [torch.float16, torch.bfloat16] else 32

    total_rows = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    hidden_dim = lm_head.shape[1]
    vocab_dim = lm_head.shape[0]

    if trainer.args.unsloth_grpo_mini_batch is None:
        if not hasattr(trainer, "_has_autotuned"):
            trainer._has_autotuned = True
            B, multiplier = autotune_batch_and_chunks(
                total_rows, seq_len, hidden_dim, vocab_dim, dtype_bytes, trainer.args.unsloth_logit_chunk_multiplier
            )
            trainer.args.unsloth_grpo_mini_batch = max(1, total_rows//B)
            trainer.args.unsloth_logit_chunk_multiplier = multiplier
            B = trainer.args.unsloth_grpo_mini_batch
            multiplier = trainer.args.unsloth_logit_chunk_multiplier
        elif trainer._step % trainer.current_gradient_accumulation_steps == 0:
            B = trainer.args.unsloth_grpo_mini_batch
            multiplier = trainer.args.unsloth_logit_chunk_multiplier
            del trainer._has_autotuned
            del trainer.args.unsloth_grpo_mini_batch
            del trainer.args.unsloth_logit_chunk_multiplier
        else:
            B = trainer.unsloth_grpo_mini_batch
            multiplier = trainer.args.unsloth_logit_chunk_multiplier
    else:
        if trainer.args.unsloth_grpo_mini_batch > total_rows:
            B = total_rows
        else:
            B = trainer.args.unsloth_grpo_mini_batch

        if trainer.args.unsloth_logit_chunk_multiplier is None:
            multiplier = max(4, seq_len // 4096)
        else:
            multiplier = trainer.args.unsloth_logit_chunk_multiplier

    if pixel_values is None:
        left_pad_tokens_per_prompt = calculate_pad_tokens_in_prompt(input_ids, logits_to_keep, trainer.processing_class.pad_token_id)

        # Determine max_left_pad from precomputed logprobs shape for consistency
        if old_logps is not None:
            max_left_pad = old_logps.shape[1] - logits_to_keep
        elif ref_logps is not None:
            max_left_pad = ref_logps.shape[1] - logits_to_keep
        else:
            max_left_pad = torch.max(left_pad_tokens_per_prompt).item()

        input_ids = left_pack_padding(input_ids, trainer.processing_class.pad_token_id)

        completion_input_ids = input_ids[:, -(logits_to_keep +max_left_pad):]
        completion_mask = create_completion_attention_mask(completion_input_ids, left_pad_tokens_per_prompt, max_left_pad, trainer.processing_class.pad_token_id).to(attention_mask.dtype)

        if trainer.use_vllm and sampling_per_token_logps is not None and getattr(trainer, "vllm_importance_sampling_correction", False):
            sampling_per_token_logps = align_logprobs_with_mask(sampling_per_token_logps, completion_mask)
        else:
            sampling_per_token_logps = None
        completion_mask = align_completion_tool_mask(tool_mask, completion_mask)
        attention_mask =  input_ids != trainer.processing_class.pad_token_id
        attention_mask = attention_mask.to(attention_mask.dtype)
    else:
        completion_input_ids = input_ids[:, -logits_to_keep:]
        completion_mask = align_completion_tool_mask(tool_mask, completion_mask)

    unwrapped_model = trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper = False)

    for module in unwrapped_model.modules():
        if hasattr(module, "_hf_hook") and hasattr(module._hf_hook, "io_same_decice"):
            module._hf_hook.io_same_decice = False
    pass

    all_logprobs_list = []

    def slice_sample_axis(value, start, end):
        if value is None:
            return None
        return value[start:end]

    import math
    total_samples = input_ids.shape[0]
    batch_size = math.ceil(total_samples / B)
    if isinstance(num_images, torch.Tensor):
        num_images = num_images.detach().cpu().reshape(-1).tolist()
    if image_grid_thw is not None and pixel_values is not None and num_images is not None:
        rows_per_image = image_grid_thw.prod(dim=-1)
        rows_per_sample = torch.split(rows_per_image, num_images)
        rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
        cum_rows = torch.cat(
            [
                torch.tensor([0], device=rows_per_sample.device),
                rows_per_sample.cumsum(0),
            ]
        )
        cum_imgs = torch.tensor([0] + num_images).cumsum(0)
    else:
        cum_rows = None
        cum_imgs = None

    input_ids_chunks = []
    attention_mask_chunks = []
    completion_ids_chunks = []
    pixel_values_chunks = []
    image_grid_thw_chunks = []
    pixel_attention_mask_chunks = []
    image_sizes_chunks = []
    token_type_ids_chunks = []
    mm_token_type_ids_chunks = []

    current_pixel_idx = 0
    #TRL 0.23.0 batching logic
    for start in range(0, total_samples, batch_size):
        end = min(start + batch_size, total_samples)

        input_ids_chunks.append(input_ids[start:end])
        attention_mask_chunks.append(attention_mask[start:end])
        completion_ids_chunks.append(completion_input_ids[start:end])
        image_sizes_chunks.append(slice_sample_axis(image_sizes, start, end))
        token_type_ids_chunks.append(slice_sample_axis(token_type_ids, start, end))
        mm_token_type_ids_chunks.append(
            slice_sample_axis(mm_token_type_ids, start, end)
        )

        if image_grid_thw is not None and pixel_values is not None:

            if num_images is None:
                grid_slice = image_grid_thw[start:end]
                batch_pixel_count = grid_slice.prod(dim=-1).sum().item()
                start_pixel_idx = current_pixel_idx
                end_pixel_idx = current_pixel_idx + batch_pixel_count
                current_pixel_idx = end_pixel_idx
            else:
                start_pixel_idx = cum_rows[start].item()
                end_pixel_idx = cum_rows[end].item()
                img_start, img_end = cum_imgs[start], cum_imgs[end]
                grid_slice = image_grid_thw[img_start:img_end]
            image_grid_thw_chunks.append(grid_slice)

            pixel_values_chunks.append(pixel_values[start_pixel_idx:end_pixel_idx])

            if pixel_attention_mask is not None:
                if pixel_attention_mask.shape[0] == pixel_values.shape[0]:
                    pixel_attention_mask_chunks.append(pixel_attention_mask[start_pixel_idx:end_pixel_idx])
                else:
                    pixel_attention_mask_chunks.append(pixel_attention_mask[start:end])
            else:
                pixel_attention_mask_chunks.append(None)

        else:
            pixel_values_chunks.append(None)
            image_grid_thw_chunks.append(None)
            pixel_attention_mask_chunks.append(None)

    zipped_inputs = zip(
        input_ids_chunks,
        attention_mask_chunks,
        pixel_values_chunks,
        image_grid_thw_chunks,
        pixel_attention_mask_chunks,
        image_sizes_chunks,
        token_type_ids_chunks,
        mm_token_type_ids_chunks,
        completion_ids_chunks
    )

    if trainer._autocast_dtype is None:
        autocaster = nullcontext()
    else:
        autocaster = torch.amp.autocast(device_type = trainer.model.device.type, dtype = trainer._autocast_dtype)

    # --- Consolidated lazy imports + env gate for the GRPO PrefixGrouper grad path ---
    # This function's SOURCE is inspect.getsource-copied verbatim into the generated
    # UnslothGRPOTrainer cache, whose namespace does NOT carry unsloth_zoo's module-level
    # imports, so these names must be bound inside the body (do NOT hoist to module scope).
    # The unsloth.utils.prefix_grouper import must also stay lazy + guarded: unsloth imports
    # unsloth_zoo at init (circular) and may not ship prefix_grouper at all, so a failed
    # import degrades to "PrefixGrouper off" and never raises.
    from unsloth_zoo.temporary_patches.common import UNSLOTH_ENABLE_LOGGING

    # One-time env gate + import, resolved once per process. Function attributes survive into
    # the cache (the function object is rebuilt there once per module and grpo_accumulated_loss
    # is a cache-module global), so memoize on grpo_accumulated_loss itself. The env gate is
    # checked FIRST, so UNSLOTH_GRPO_PREFIX_GROUPER=0 never imports or runs any PG code (that path
    # is byte-identical to before). () means gate off / unavailable / failed import -> PG stays off.
    _pg_funcs = getattr(grpo_accumulated_loss, "_pg_funcs", None)
    if _pg_funcs is None:
        _pg_funcs = ()
        if os.environ.get("UNSLOTH_GRPO_PREFIX_GROUPER", "1").lower() not in (
            "0", "false", "no", "off",
        ):
            try:
                from unsloth.utils.prefix_grouper import (
                    build_group_layout as _pg_build_layout,
                    prefix_grouper_enabled as _pg_enabled_fn,
                    verify_on as _pg_verify_on,
                    tol_ok as _pg_tol_ok,
                    TOL_KILL as _PG_TOL_KILL,
                )
                _pg_funcs = (
                    _pg_build_layout, _pg_enabled_fn, _pg_verify_on, _pg_tol_ok, _PG_TOL_KILL,
                )
            except Exception:
                _pg_funcs = ()
        grpo_accumulated_loss._pg_funcs = _pg_funcs
    # Skip PG when vLLM drives generation (fast_inference=True): the colocated rollout dominates
    # the step, so the shared-prefix forward saves little end-to-end and its first-use self-verify
    # (which runs the full-row path too) is net overhead. Keep the packed path instead.
    _pg_engage = bool(_pg_funcs) and not getattr(trainer, "use_vllm", False)

    # ---- PrefixGrouper (GRPO shared-prompt dedup; default ON, UNSLOTH_GRPO_PREFIX_GROUPER=0 disables) ----
    # In GRPO every prompt spawns G=num_generations completions that share the prompt prefix.
    # The full-row packed path below forwards that prefix G times; PrefixGrouper stores it ONCE
    # and concatenates only the G suffixes (FlexAttention shared-prefix mask), cutting the trunk
    # forward from G*(P+R) to P+G*R tokens. Gated behind UNSLOTH_GRPO_PREFIX_GROUPER (requires
    # seq-packing on). tok_r auto-gate + first-use self-verify vs the full-row packed new_logprobs
    # (fall back + mark-unsafe on mismatch). Loss/gradients flow through the shared-prefix stream
    # (the prefix contributes grad once = the sum of the G repeats, mathematically identical). When
    # off / grouping fails / not yet verified, the full-row packed path below runs as before.
    _pg_result = None
    _pg_use = False
    _pg_skip_pack = False
    _pg_num_gen = getattr(trainer, "num_generations", None)
    # Runtime gate (uses the memoized prefix_grouper callables); kept next to its use because it
    # reads unwrapped_model.config etc. Broad except -> engage False, matching the original
    # single import+gate try/except semantics exactly.
    if _pg_engage and _pg_funcs:
        try:
            _pg_build_layout, _pg_enabled_fn, _pg_verify_on, _pg_tol_ok, _PG_TOL_KILL = _pg_funcs
            # the FlexAttention kernel never applies attn_logit_softcapping, so skip PG entirely
            # for softcap models (e.g. gemma2) before building any layout. Hybrid SSM models
            # (FalconH1 etc.) are excluded too: only attention gets the shared-prefix isolation,
            # a Mamba branch would leak state across suffixes. MoE models (e.g. Qwen3-MoE) are
            # excluded for the same reason: they reuse LlamaModel_fast_forward but their decoder
            # does not thread prefix_seg_info to the attention, so the MoE branch runs plain causal
            # attention and would leak state across suffixes.
            _pg_cfg = getattr(unwrapped_model, "config", None)
            _pg_engage = (
                _pg_enabled_fn()
                and pixel_values is None
                and token_type_ids is None
                and mm_token_type_ids is None
                and _pg_num_gen is not None
                and _pg_num_gen >= 2
                and not getattr(_pg_cfg, "attn_logit_softcapping", None)
                and not any(
                    getattr(_pg_cfg, _pg_a, None) is not None
                    for _pg_a in ("mamba_d_ssm", "mamba_d_state", "mamba_expand")
                )
                and not any(
                    getattr(_pg_cfg, _pg_a, None) is not None
                    for _pg_a in (
                        "num_experts", "num_experts_per_tok", "num_local_experts",
                        "n_routed_experts", "moe_intermediate_size",
                    )
                )
            )
        except Exception:
            _pg_engage = False
    else:
        _pg_engage = False
    _pg_layout = None
    _pg_trusted = False   # signature already verified -> skip the full-row forward this step
    if _pg_engage:
        try:
            _pg_pad_id = trainer.processing_class.pad_token_id
            # Build the layout from the LEFT-PACKED input_ids with the ORIGINAL left-pad counts,
            # so PrefixGrouper's prefix/suffix split matches the packed path's prompt/completion
            # split (_pack_cstart) exactly and the verify is apples-to-apples.
            # sliding-window models lose the per-sequence window in the packed stream, so cap the
            # PG span (P+max(R)) at the window, mirroring the packed _pack_sw guard below.
            _pg_sw = getattr(getattr(unwrapped_model, "config", None), "sliding_window", None)
            if not (isinstance(_pg_sw, int) and _pg_sw > 0):
                _pg_sw = None
            _pg_layout = _pg_build_layout(
                input_ids, logits_to_keep, _pg_pad_id, _pg_num_gen, left_pad_tokens_per_prompt,
                max_segment_cap = _pg_sw,
            )
            _pg_unsafe = getattr(unwrapped_model, "_unsloth_prefix_grouper_grad_unsafe", None)
            if _pg_unsafe is None:
                _pg_unsafe = set()
            if _pg_layout is not None and _pg_layout.signature in _pg_unsafe:
                _pg_layout = None
            elif _pg_layout is not None:
                _pg_layout.W = logits_to_keep + max_left_pad
                _pg_verified = getattr(unwrapped_model, "_unsloth_prefix_grouper_grad_verified", None)
                # trust needs the verified envelope to cover this batch's lengths too
                # (re-verify when T or the longest segment grows, like the packed path)
                _pg_T = int(_pg_layout.flat_ids.shape[1])
                _pg_maxseg = int(_pg_layout.position_ids.max()) + 1
                _pg_env = (
                    _pg_verified.get(_pg_layout.signature)
                    if isinstance(_pg_verified, dict) else None
                )
                if (not _pg_verify_on()) or (
                    _pg_env is not None and _pg_T <= _pg_env[0] and _pg_maxseg <= _pg_env[1]
                ):
                    _pg_trusted = True
                    _pg_skip_pack = True   # trusted shape -> skip the full-row forward
        except Exception as _pg_err:
            _pg_layout = None
            _pg_trusted = False
            _pg_skip_pack = False
            if isinstance(_pg_err, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
            if UNSLOTH_ENABLE_LOGGING:
                print(f"[Unsloth] GRPO PrefixGrouper (grad) disabled (fell back to packed): {_pg_err!r}", flush = True)

    # ---- Sequence packing (default-on; disable with UNSLOTH_GRPO_SEQ_PACKING=0) ----
    # One varlen [1, sum L] block-diagonal forward replaces the padded [B, Lmax] loop: the exact per-row
    # result, and it fixes the padded path's left-pad RoPE error. Loss/gradients flow through it. Self-
    # verified against the per-row forward (shape/RoPE-aware, re-checked as T grows); falls back if a
    # backend ignores packed_seq_lengths. lm_head runs on completion positions only.
    new_logprobs = None
    _pack_result = None
    _pack_use = False
    _pack_enabled = os.environ.get("UNSLOTH_GRPO_SEQ_PACKING", "1").lower() not in ("0", "false", "no", "off")
    _pack_ok = getattr(unwrapped_model, "_unsloth_seq_packing_grad_ok", None)
    if (_pack_enabled and not _pg_skip_pack and pixel_values is None
            and token_type_ids is None and mm_token_type_ids is None and _pack_ok is not False):
        try:
            _pack_pad_id = trainer.processing_class.pad_token_id
            _pack_keep = input_ids != _pack_pad_id
            _pack_lengths = _pack_keep.sum(dim = 1)
            _pack_lengths_cpu = _pack_lengths.tolist()                 # single GPU->CPU sync, reused below
            _pack_nz_cpu = [_n for _n in _pack_lengths_cpu if _n > 0]
            _pack_flat_ids = input_ids[_pack_keep].unsqueeze(0)
            _pack_T = _pack_flat_ids.shape[1]
            _pack_L = input_ids.shape[1]
            _pack_W = logits_to_keep + max_left_pad
            _pack_maxseg = max(_pack_nz_cpu) if _pack_nz_cpu else 0
            # sliding-window models lose the per-sequence local window in a packed stream
            _pack_sw = getattr(getattr(unwrapped_model, "config", None), "sliding_window", None)
            _pack_sw_ok = not (isinstance(_pack_sw, int) and _pack_sw > 0 and _pack_maxseg > _pack_sw)
            _pack_active = int((completion_mask.sum(dim = 1) > 0).sum())
            _pack_unsafe = getattr(unwrapped_model, "_unsloth_seq_packing_grad_unsafe_T", None)
            # skip the whole packed forward for a known-unsafe length region (a prior moderate mismatch)
            if _pack_T >= 2 and len(_pack_nz_cpu) > 0 and _pack_sw_ok and (_pack_ok is True or _pack_active >= 2) \
                    and not (_pack_unsafe is not None and _pack_T >= _pack_unsafe):
                _pack_psl = torch.tensor(_pack_nz_cpu, dtype = torch.int32, device = input_ids.device)
                # reset 0-based position_ids per segment
                _pack_pos = (_pack_keep.cumsum(dim = 1) - 1)[_pack_keep].unsqueeze(0)
                _pack_chunks = max(1, total_rows * multiplier)
                _pack_nz_idx = _pack_keep.nonzero(as_tuple = False)            # [T, 2] = (row, col)
                _pack_within = _pack_nz_idx[1:, 0] == _pack_nz_idx[:-1, 0]     # [T-1]
                # completion start is per-row after left-packing: (L - logits_to_keep) minus that
                # row's left-pad (matches create_completion_attention_mask exactly)
                _pack_cstart = (_pack_L - logits_to_keep) - left_pad_tokens_per_prompt  # [rows]
                _pack_ctgt = (_pack_nz_idx[1:, 1] >= _pack_cstart[_pack_nz_idx[1:, 0]]) & _pack_within
                with autocaster:
                    # use_cache=False: a KV cache silently disables varlen packing
                    _pack_hidden = unwrapped_model(
                        input_ids = _pack_flat_ids,
                        position_ids = _pack_pos,
                        packed_seq_lengths = _pack_psl,
                        use_cache = False,
                    ).logits
                    _pack_sel = chunked_hidden_states_selective_log_softmax(
                        _pack_hidden[0, :-1, :][_pack_ctgt].unsqueeze(0), lm_head,
                        _pack_flat_ids[0, 1:][_pack_ctgt].unsqueeze(0), _pack_chunks,
                        logit_scale_multiply, logit_scale_divide, logit_softcapping, temperature,
                    )[0]
                # GPT-OSS offload race guard (matches the padded loop)
                device_synchronize()
                # scatter each completion logprob back to its (row, col) so [:, -_pack_W:] matches padded
                _pack_tgt = (_pack_nz_idx[1:, 0] * _pack_L + _pack_nz_idx[1:, 1])[_pack_ctgt]
                _pack_result = torch.zeros(
                    total_rows * _pack_L, dtype = torch.float32, device = input_ids.device,
                ).index_put((_pack_tgt,), _pack_sel.to(torch.float32)).view(total_rows, _pack_L)[:, -_pack_W:]
                # trust decision: re-verify when T or the longest segment grows past what was verified
                # (a LongRoPE cache switch can change the result)
                _pack_vT = int(getattr(unwrapped_model, "_unsloth_seq_packing_grad_verified_T", 0))
                _pack_vS = int(getattr(unwrapped_model, "_unsloth_seq_packing_grad_verified_seg", 0))
                _pack_force_verify = os.environ.get("UNSLOTH_GRPO_SEQ_PACKING_VERIFY", "0") == "1"
                if (not _pack_force_verify) and _pack_ok is True and _pack_T <= _pack_vT and _pack_maxseg <= _pack_vS:
                    _pack_use = True                                           # already verified for this shape
                else:
                    # verify against the per-row clean forward (exact ground truth; no grad, value check)
                    _pack_ref = torch.zeros_like(_pack_result)
                    with torch.no_grad(), autocaster:
                        for _pack_i in range(total_rows):
                            _pack_ni = _pack_lengths_cpu[_pack_i]
                            if _pack_ni < 2: continue
                            _pack_rmask = _pack_keep[_pack_i]
                            _pack_real = input_ids[_pack_i][_pack_rmask].unsqueeze(0)
                            _pack_rpos = torch.arange(_pack_ni, device = input_ids.device).unsqueeze(0)
                            _pack_rh = unwrapped_model(input_ids = _pack_real, position_ids = _pack_rpos, use_cache = False).logits
                            _pack_rsel = chunked_hidden_states_selective_log_softmax(
                                _pack_rh[:, :-1, :], lm_head, _pack_real[:, 1:], 1,
                                logit_scale_multiply, logit_scale_divide, logit_softcapping, temperature,
                            )[0]
                            _pack_rcols = _pack_rmask.nonzero(as_tuple = False).squeeze(1)[1:] - (_pack_L - _pack_W)
                            _pack_rkeep = _pack_rcols >= 0
                            _pack_ref[_pack_i, _pack_rcols[_pack_rkeep]] = _pack_rsel[_pack_rkeep].to(torch.float32)
                    device_synchronize()
                    # compare over the exact loss-mask region (same mask the loss uses; pure
                    # create_completion_attention_mask, before any tool_mask is applied)
                    _pack_cm = create_completion_attention_mask(
                        input_ids[:, -_pack_W:], left_pad_tokens_per_prompt, max_left_pad, _pack_pad_id
                    ).float()
                    _pack_diff = float(((_pack_result.detach() - _pack_ref).abs() * _pack_cm).max())
                    if UNSLOTH_ENABLE_LOGGING:
                        print(f"[Unsloth] GRPO seq-packing (grad) verify: T={_pack_T} maxseg={_pack_maxseg} packed-vs-perrow max|d|={_pack_diff:.4f}", flush = True)
                    # floor ~0.25 through different kernels; cross-sample contamination is >= 2.4
                    if _pack_diff < 7e-1:
                        unwrapped_model._unsloth_seq_packing_grad_ok = True
                        # only widen the trusted shape when >= 2 completion rows actually exercised
                        # cross-sample packing; a < 2 row pass proves nothing, so keep re-verifying
                        # larger shapes until a real multi-row batch clears them
                        if _pack_active >= 2:
                            unwrapped_model._unsloth_seq_packing_grad_verified_T = max(_pack_vT, _pack_T)
                            unwrapped_model._unsloth_seq_packing_grad_verified_seg = max(_pack_vS, _pack_maxseg)
                        _pack_ok = True
                        _pack_use = True
                    else:
                        _pack_use = False
                        if _pack_diff >= 1.5:
                            # large mismatch = contamination (attention ignores the packed mask, e.g.
                            # some MoE): disable packing for this model
                            unwrapped_model._unsloth_seq_packing_grad_ok = False
                        else:
                            # moderate mismatch -> likely a length boundary (LongRoPE): mark unsafe but
                            # keep packing for smaller shapes
                            unwrapped_model._unsloth_seq_packing_grad_unsafe_T = (
                                _pack_T if _pack_unsafe is None else min(_pack_unsafe, _pack_T)
                            )
                        if UNSLOTH_ENABLE_LOGGING:
                            print(f"[Unsloth] GRPO seq-packing (grad) fell back at T={_pack_T} (diff={_pack_diff:.3f})", flush = True)
        except Exception as _pack_err:
            # any failure -> drop intermediates, use the padded loop, do not retry
            _pack_hidden = None
            _pack_sel = None
            _pack_result = None
            _pack_use = False
            if isinstance(_pack_err, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            unwrapped_model._unsloth_seq_packing_grad_ok = False
            if UNSLOTH_ENABLE_LOGGING:
                print(f"[Unsloth] GRPO sequence-packing disabled (fell back to padded): {_pack_err!r}", flush = True)
    # ---- PrefixGrouper resolution + first-use self-verify (grad) ----
    # Verify (no_grad, value only) and the loss forward (grad) are DECOUPLED: the shared-prefix
    # verify forward runs under torch.no_grad() so its inference tensors never enter autograd,
    # then a SEPARATE grad forward builds new_logprobs for the loss. This mirrors the packed
    # path's verify/trust model and avoids "inference tensors saved for backward".
    def _pg_grad_forward():
        _pg_chunks = max(1, total_rows * multiplier)
        with autocaster:
            _h = unwrapped_model(
                input_ids = _pg_layout.flat_ids,
                position_ids = _pg_layout.position_ids,
                prefix_seg_info = _pg_layout.prefix_seg_info,
                use_cache = False,
            ).logits
            _pg_lp = _pg_layout.extract_logps(
                _h, lm_head, chunked_hidden_states_selective_log_softmax,
                _pg_chunks, logit_scale_multiply, logit_scale_divide,
                logit_softcapping, temperature,
            )  # [total_rows, W] with grad
            # GPT-OSS offload race guard (matches the packed and padded paths)
            device_synchronize()
            return _pg_lp

    if _pg_layout is not None:
        # A verify-phase OOM must not poison the signature. During first-use verify the full-row
        # packed graph is still co-resident (it is the comparison reference), so an OOM there does
        # not prove PG alone cannot fit; only an OOM in the trusted PG-alone forward below (after the
        # packed graph is freed) is a deterministic PG-alone failure worth marking unsafe.
        _pg_phase_verify = False
        try:
            if not _pg_trusted:
                # first use for this structure: verify vs the full-row packed new_logprobs
                # (itself self-verified vs per-row). PASS < tol_ok -> trust; >= TOL_KILL ->
                # mark unsafe forever; borderline -> fall back this shape.
                if _pack_use and _pack_result is not None:
                    _pg_phase_verify = True   # packed graph co-resident until it is freed below
                    with torch.no_grad():
                        _pg_ref = _pg_grad_forward()
                    _pg_W2 = logits_to_keep + max_left_pad
                    _pg_cm = create_completion_attention_mask(
                        input_ids[:, -_pg_W2:], left_pad_tokens_per_prompt, max_left_pad,
                        trainer.processing_class.pad_token_id,
                    ).float()
                    _pg_a = _pg_ref[:, -_pg_W2:].float()
                    _pg_b = _pack_result.detach()[:, -_pg_W2:].float()
                    _pg_diff = float(((_pg_a - _pg_b).abs() * _pg_cm).max())
                    if UNSLOTH_ENABLE_LOGGING:
                        print(
                            f"[Unsloth] GRPO PrefixGrouper (grad) verify: sig={_pg_layout.signature} "
                            f"shared-prefix vs full-row-packed max|d|={_pg_diff:.4f}", flush = True,
                        )
                    if _pg_diff < _pg_tol_ok():
                        _pg_v = getattr(unwrapped_model, "_unsloth_prefix_grouper_grad_verified", None)
                        if not isinstance(_pg_v, dict):
                            _pg_v = {}
                        _pg_vT = int(_pg_layout.flat_ids.shape[1])
                        _pg_vS = int(_pg_layout.position_ids.max()) + 1
                        _pg_old = _pg_v.get(_pg_layout.signature, (0, 0))
                        _pg_v[_pg_layout.signature] = (
                            max(_pg_vT, _pg_old[0]), max(_pg_vS, _pg_old[1]),
                        )
                        unwrapped_model._unsloth_prefix_grouper_grad_verified = _pg_v
                        _pg_trusted = True
                    else:
                        _pg_u = getattr(unwrapped_model, "_unsloth_prefix_grouper_grad_unsafe", None)
                        if _pg_u is None:
                            _pg_u = set()
                        if _pg_diff >= _PG_TOL_KILL:
                            _pg_u.add(_pg_layout.signature)
                            unwrapped_model._unsloth_prefix_grouper_grad_unsafe = _pg_u
                        _pg_trusted = False
                # else: no full-row reference (packing off/failed) -> cannot verify -> fall back.
            if _pg_trusted:
                # free the full-row packed graph BEFORE the grad forward (it was already
                # consumed above as the verify reference). Holding both graphs here can OOM
                # exactly when packed+PG exceed VRAM but PG alone would fit; on a PG failure
                # the padded loop recomputes new_logprobs, so packed is no longer needed.
                _pack_hidden = _pack_sel = _pack_result = None
                _pg_phase_verify = False   # packed freed: an OOM below is a genuine PG-alone failure
                _pg_result = _pg_grad_forward()   # grad forward for the loss
                _pg_use = True
        except Exception as _pg_err2:
            _pg_use = False
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
            # untrust this signature so the next batch runs the full-row packed path again
            # instead of skipping it and landing on the padded loop repeatedly
            _pg_v = getattr(unwrapped_model, "_unsloth_prefix_grouper_grad_verified", None)
            if isinstance(_pg_v, dict):
                _pg_v.pop(_pg_layout.signature, None)
            if isinstance(_pg_err2, torch.cuda.OutOfMemoryError):
                # OOM would recur deterministically at these lengths: mark unsafe for good, but only
                # when it happened in the PG-alone forward. A verify-phase OOM (packed graph still
                # co-resident) does not prove PG alone cannot fit, so leave the signature off the
                # unsafe set and just retry the full-row packed path next batch.
                if not _pg_phase_verify:
                    _pg_u = getattr(unwrapped_model, "_unsloth_prefix_grouper_grad_unsafe", None)
                    if _pg_u is None:
                        _pg_u = set()
                    _pg_u.add(_pg_layout.signature)
                    unwrapped_model._unsloth_prefix_grouper_grad_unsafe = _pg_u
                torch.cuda.empty_cache()
            if UNSLOTH_ENABLE_LOGGING:
                print(f"[Unsloth] GRPO PrefixGrouper (grad) forward failed -> packed/padded fallback: {_pg_err2!r}", flush = True)

    if _pg_use and _pg_result is not None:
        new_logprobs = _pg_result            # PrefixGrouper verified/trusted -> skip the loop
        zipped_inputs = []
    elif _pack_use and _pack_result is not None:
        new_logprobs = _pack_result          # verified -> skip the loop
        zipped_inputs = []
    else:
        # packing rejected/unused: drop the packed graph before the padded loop so both don't co-reside
        _pack_hidden = _pack_sel = _pack_result = None

    def to_device(tensor, device, non_blocking=True):
        if tensor is None: return None
        return tensor.to(device, non_blocking=non_blocking)

    class Unsloth_Offloaded_Log_Softmax(torch.autograd.Function):
        """Manual gradient checkpointing / CPU offloading for log softmax."""
        @staticmethod
        def forward(ctx, hidden_states, lm_head, index, chunks,
                    logit_scale_multiply, logit_scale_divide,
                    logit_softcapping, temperature):
            # Detach so we don't keep the graph (and extra memory) on CPU.
            ctx.saved_hidden_states = hidden_states.detach().contiguous().to("cpu", non_blocking=True)
            ctx.device = hidden_states.device
            ctx.dtype = hidden_states.dtype

            ctx.lm_head = lm_head
            ctx.lm_head_requires_grad = lm_head.requires_grad
            ctx.index = index
            ctx.args = (chunks, logit_scale_multiply, logit_scale_divide, logit_softcapping, temperature)

            with torch.no_grad():
                output = chunked_hidden_states_selective_log_softmax(
                    hidden_states, lm_head, index, *ctx.args
                )

            return output

        @staticmethod
        def backward(ctx, grad_output):
            hidden_states = to_device(ctx.saved_hidden_states, ctx.device)
            hidden_states = hidden_states.to(ctx.dtype)
            hidden_states.requires_grad_(True)

            lm_head = ctx.lm_head
            # #Possibly redundant lines
            # if ctx.lm_head_requires_grad:
            #     hidden_states.requires_grad_(True)
            # else:
            #     lm_head = lm_head.detach()

            index = ctx.index

            with torch.enable_grad():
                output = chunked_hidden_states_selective_log_softmax(
                    hidden_states, lm_head, index, *ctx.args
                )

            torch.autograd.backward(output, grad_output)

            return (
                hidden_states.grad,
                lm_head.grad if ctx.lm_head_requires_grad else None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

    def efficient_log_softmax(hidden_states, lm_head, index, chunks=32,
                            logit_scale_multiply=0.0, logit_scale_divide=0.0,
                            logit_softcapping=0.0, temperature=1, batch_size=8):
        if (index.shape[1] <= 1024 and batch_size <= 8) or batch_size==1:
            # Normal path is faster / saves a GB under these conditions.
            return chunked_hidden_states_selective_log_softmax(
                hidden_states,
                lm_head,
                index,
                chunks,
                logit_scale_multiply,
                logit_scale_divide,
                logit_softcapping,
                temperature
            )
        else:
            return Unsloth_Offloaded_Log_Softmax.apply(
                hidden_states, lm_head, index, chunks,
                logit_scale_multiply, logit_scale_divide,
                logit_softcapping, temperature
            )

    def compute_logprobs_chunk(new_hidden_states_chunk, completion_ids, input_ids_chunk):
        # Hidden states -> lm_head matmul path; raw logits -> skip matmul and
        # skip scale/softcap (model forward already applied them).
        chunks = input_ids_chunk.shape[0] * multiplier
        if new_hidden_states_chunk.shape[-1] == lm_head.shape[1]:
            return efficient_log_softmax(
                new_hidden_states_chunk,
                lm_head,
                completion_ids,
                chunks = chunks,
                logit_scale_multiply = logit_scale_multiply,
                logit_scale_divide = logit_scale_divide,
                logit_softcapping = logit_softcapping,
                temperature = temperature,
                batch_size = B,
            )
        return chunked_selective_log_softmax(
            new_hidden_states_chunk,
            completion_ids,
            temperature = temperature,
            chunks = chunks,
        )


    for (
        input_ids_chunk,
        attention_mask_chunk,
        pixel_values_chunk,
        image_grid_thw_chunk,
        pixel_attention_mask_chunk,
        image_sizes_chunk,
        token_type_ids_chunk,
        mm_token_type_ids_chunk,
        completion_ids
    ) in zipped_inputs:
            _extra_vision_kwargs = {}
            if token_type_ids_chunk is not None:
                _extra_vision_kwargs["token_type_ids"] = token_type_ids_chunk
            if mm_token_type_ids_chunk is not None:
                _extra_vision_kwargs["mm_token_type_ids"] = mm_token_type_ids_chunk
            with autocaster:
                if pixel_values is None:
                    new_hidden_states_chunk = unwrapped_model(
                        input_ids = input_ids_chunk,
                        attention_mask = attention_mask_chunk,
                        pixel_values = pixel_values_chunk,
                        image_grid_thw = image_grid_thw_chunk,
                        pixel_attention_mask = pixel_attention_mask_chunk,
                        image_sizes = image_sizes_chunk,
                        **_extra_vision_kwargs,
                    ).logits

                    new_hidden_states_chunk = new_hidden_states_chunk[:, -(logits_to_keep + max_left_pad + 1): , :]
                    new_hidden_states_chunk = new_hidden_states_chunk[:, :-1, :]
                    logprobs_chunk = compute_logprobs_chunk(new_hidden_states_chunk, completion_ids, input_ids_chunk)
                else:
                    new_hidden_states_chunk = unwrapped_model(
                        input_ids = input_ids_chunk,
                        attention_mask = attention_mask_chunk,
                        pixel_values = pixel_values_chunk,
                        image_grid_thw = image_grid_thw_chunk,
                        pixel_attention_mask = pixel_attention_mask_chunk,
                        image_sizes = image_sizes_chunk,
                        logits_to_keep = logits_to_keep + 1,
                        **_extra_vision_kwargs,
                    ).logits

                    new_hidden_states_chunk = new_hidden_states_chunk[:, :-1, :]
                    logprobs_chunk = compute_logprobs_chunk(new_hidden_states_chunk, completion_ids, input_ids_chunk)
                # Avoids race conditions with GPT OSS offload_embbed=True; no measurable slowdown.
                device_synchronize()
            all_logprobs_list.append(logprobs_chunk)

    if new_logprobs is None:
        # padded fallback (packing disabled / unsupported / not verified for this length)
        new_logprobs = torch.cat(all_logprobs_list, dim=0)

    with autocaster:
        loss, completion_length, mean_kl, delta, flat_is_ratio, coef_1 = UnslothEfficientGRPO.apply(
            new_logprobs,
            old_logps,
            ref_logps,
            sampling_per_token_logps,
            lm_head,
            completion_input_ids,
            completion_mask,
            advantages,
            trainer.beta,
            trainer.accelerator.scaler,
            1,
            kwargs
        )

    # Force logits (not hidden states) again or output is gibberish.
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"

    return loss, completion_length, mean_kl, delta, flat_is_ratio, coef_1, completion_mask
    # Old non-efficient code path (dead).
    new_logits = torch.matmul(new_hidden_states, lm_head.t())
    new_logits = new_logits[:, :-1, :] # exclude the last logit: it corresponds to the next token pred
    old_logits = torch.matmul(old_hidden_states, lm_head.t())
    old_logits = old_logits[:, :-1, :] # exclude the last logit: it corresponds to the next token pred
    loss, completion_length, mean_kl = grpo_compute_loss(
        old_logits,
        new_logits,
        completion_input_ids,
        completion_mask,
        trainer.beta,
        advantages,
    )
    return loss, completion_length, mean_kl
    pass
pass
RL_REPLACEMENTS["grpo_accumulated_loss"] = grpo_accumulated_loss

from .dataset_utils import sft_prepare_dataset
RL_REPLACEMENTS["sft_prepare_dataset"] = sft_prepare_dataset

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
