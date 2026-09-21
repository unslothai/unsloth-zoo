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
import sys
import math
import logging
from typing import Union, Callable, Optional, List, Dict
from .device_type import DEVICE_TYPE, device_synchronize
from unsloth_zoo.temporary_patches.common import (
    torch_compile_options,
    _maybe_compile,
)
from unsloth_zoo.log import logger



RL_REPLACEMENTS = dict()

# https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py#L1674
@_maybe_compile(dynamic = True, fullgraph = True, options = torch_compile_options,)
def selective_log_softmax(logits, index):
    logits = logits.to(torch.float32)
    selected_logits = torch.gather(logits, dim = -1, index = index.unsqueeze(-1)).squeeze(-1)
    logsumexp_values = torch.logsumexp(logits, dim = -1)
    per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    return per_token_logps
pass

# Memory-efficient chunked variant of the above on (bsz+qlen); exactly equivalent.
@_maybe_compile(dynamic = True, fullgraph = True, options = torch_compile_options,)
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

@_maybe_compile(dynamic = True, fullgraph = True, options = torch_compile_options,)
def chunked_hidden_states_selective_log_softmax(
    hidden_states: torch.Tensor,
    lm_head: torch.Tensor,
    index: torch.Tensor,
    chunks: int = 4,
    logit_scale_multiply: float = 0.0,
    logit_scale_divide: float = 0.0,
    logit_softcapping: float = 0.0,
    temperature: float = 1.0,
    # Rows per chunk cap. Read HERE, in the default, not in the body: the body
    # is traced with fullgraph = True, and `os.environ` is an unsupported op
    # there on torch 2.4 -- Dynamo raises
    #   torch._dynamo.exc.Unsupported: const method call bytes.decode
    # from os._Environ.__getitem__, which the eager fallback does not catch
    # (it only catches recompile-limit and disabled-hook breaks), so the very
    # first call would die even with the variable unset. A default is evaluated
    # once when the def runs, which is import time, outside any traced region.
    # It is also a plain int argument, so Dynamo guards on it instead of
    # constant-folding an unguarded read (2.7+ never notice a later change).
    # A non-numeric value is ignored rather than raised on, so a typo cannot
    # break the import. 0 keeps the previous chunk boundaries exactly.
    max_rows_per_chunk: int = (
        int(os.environ.get("UNSLOTH_GRPO_MAX_ROWS_PER_CHUNK", "0").strip())
        if os.environ.get("UNSLOTH_GRPO_MAX_ROWS_PER_CHUNK", "0").strip().isdigit()
        else 0
    ),
) -> torch.Tensor:
    # All Unsloth Zoo code licensed under AGPL3
    # Reshape on this tensor's own last dim: a no-op, so a wrong-width caller
    # cannot have its row count silently rewritten and instead fails at the
    # matmul below, which prints both operands. Do not swap in a bare
    # torch._check: it reports only "Expected cond to be True", naming neither
    # operand, and Dynamo rejects a message-carrying one. Callers dispatch on
    # the width first -- see `compute_logprobs_chunk`, the packed path and
    # `_pg_grad_forward`.
    flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_index = index.reshape(-1)

    # Each chunk materialises rows x vocab logits and then a float32 copy of
    # them, all on the device holding the output head. With a large vocabulary
    # and a fixed chunk count that grows with the batch, so the peak scales with
    # the batch rather than staying bounded. max_rows_per_chunk caps the rows
    # per chunk instead, which is pure loop splitting: more, smaller chunks,
    # same concatenated result. 0 (the default) keeps the previous chunk
    # boundaries exactly.
    if max_rows_per_chunk > 0:
        n_rows = flat_hidden_states.shape[0]
        chunks = max(chunks, -(-n_rows // max_rows_per_chunk))
        chunks = min(chunks, max(n_rows, 1))

    chunked_hidden_states = torch.chunk(flat_hidden_states, chunks=chunks, dim=0)
    chunked_index = torch.chunk(flat_index, chunks=chunks, dim=0)

    all_per_token_logps = []

    for chunk_hidden_states, chunk_index in zip(chunked_hidden_states, chunked_index):
        # When the model is dispatched over several devices, the output head can
        # sit on a different one from the hidden states, because accelerate
        # places the tail of the model on the last device it fills. Co-locate on
        # the head's device before the matmul, otherwise this raises
        #   Unhandled FakeTensor Device Propagation for aten.mm.default,
        #   found two different devices cuda:0, cuda:1
        # On a single device every .to() here is a no-op and the result is
        # bit-identical to before.
        chunk_hidden_states = chunk_hidden_states.to(device = lm_head.device, dtype = lm_head.dtype)
        chunk_index = chunk_index.to(lm_head.device)
        chunk_logits = chunk_hidden_states @ lm_head.t()

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
        # Return to the caller's device so the concatenation below and every
        # downstream consumer see the device they started on.
        all_per_token_logps.append(per_token_logps.to(hidden_states.device))

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
    # The off-policy mask uses vLLM sampling logprobs whenever the batch supplies them (matching TRL);
    # the vLLM importance-sampling ratio is applied to the loss only when this flag is on.
    vllm_importance_sampling_correction = kwargs.get("vllm_importance_sampling_correction", False)
    vllm_importance_sampling_mode = kwargs.get("vllm_importance_sampling_mode", "sequence_mask")
    vllm_importance_sampling_cap = kwargs.get("vllm_importance_sampling_cap", 2.0)
    vllm_importance_sampling_clip_min = kwargs.get("vllm_importance_sampling_clip_min", None)
    vllm_importance_sampling_clip_max = kwargs.get("vllm_importance_sampling_clip_max", 3.0)
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
    # Only direct callers see this fallback; the trainer always forwards an explicit value.
    use_bias_correction_kl = kwargs.get("use_bias_correction_kl", False)
    input_ids = input_ids.unsqueeze(-1)

    importance_sampling_ratio = None

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
        # DeepSeek-V3.2 off-policy mask. The mismatch logprobs are sampling_per_token_logps (vLLM
        # sampling logprobs) if present, else old, else new.detach() when both are absent
        # (num_iterations == 1 with no vLLM). This mirrors TRL, which defaults old_per_token_logps to
        # per_token_logps.detach() so get_off_policy_mask never receives None (it computes
        # mismatch - per_token_logps.detach(), so new.detach() yields a zero-KL keep-all mask). The
        # callable is a signature-stable adapter installed in grpo_accumulated_loss, so this stays
        # fixed across TRL versions with no signature introspection inside this compiled function.
        off_policy_mask = get_off_policy_mask(
            advantages=advantages,
            per_token_logps=new,
            sampling_per_token_logps=sampling_per_token_logps if sampling_per_token_logps is not None else (old if old is not None else new.detach()),
            mask=mask,
            off_policy_threshold=off_policy_mask_threshold,
        )

    with torch.no_grad():
        if use_vllm and sampling_per_token_logps is not None and vllm_importance_sampling_correction:
            # Filter out extra leading prompt tokens after left-padding input_ids.
            # Match TRL: aggregate log-ratios then exp (product), not sum of exp ratios.
            importance_sampling_ratio = (old - sampling_per_token_logps) * mask

            if vllm_importance_sampling_mode in ["sequence_mask", "sequence_truncate"]:
                importance_sampling_ratio = importance_sampling_ratio.sum(dim=-1, keepdim=True)

            importance_sampling_ratio = torch.exp(importance_sampling_ratio)

            if vllm_importance_sampling_mode in ["token_truncate", "sequence_truncate"]:
                importance_sampling_ratio = torch.clamp(
                    importance_sampling_ratio, 
                    min=vllm_importance_sampling_clip_min,
                    max=vllm_importance_sampling_clip_max
                )
            elif vllm_importance_sampling_mode in ["token_mask", "sequence_mask"]:
                min_val = (
                    vllm_importance_sampling_clip_min
                    if vllm_importance_sampling_clip_min is not None
                    else -math.inf
                )

                max_val = (
                    vllm_importance_sampling_clip_max
                    if vllm_importance_sampling_clip_max is not None
                    else math.inf
                )

                invalid_mis_mask = (importance_sampling_ratio < min_val) | (
                        importance_sampling_ratio > max_val
                )

                importance_sampling_ratio = importance_sampling_ratio.masked_fill(
                        invalid_mis_mask, value=0.0
                )
            else:
                raise ValueError(
                        f"Unknown vLLM importance sampling mode: {vllm_importance_sampling_mode}. Possible values are 'token_truncate', 'token_mask', 'sequence_truncate', and 'sequence_mask'."
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
        # TRL order: pre-clamp non-detached coef_1, before the loss_type dispatch.
        if use_bias_correction_kl:
            kl_i = kl_i * coef_1
    else:
        # Zeros with the correct shape.
        if importance_sampling_level == "sequence":
            kl_i = new.new_zeros(new.size(0), 1)
        else:
            kl_i = torch.zeros_like(new)

    if loss_type == "cispo":
        clamped_ratios = torch.clamp(coef_1, max=epsilon_high).detach()
        loss_i = -clamped_ratios * advantages * new
    elif loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
        coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)

        if delta is not None:
            loss_1 = torch.clamp(coef_1, max=delta) * advantages
        else:
            loss_1 = coef_1 * advantages
        pass
        loss_2 = coef_2 * advantages
        loss_i = -torch.min(loss_1, loss_2)
    elif loss_type == "sapo":
        temperatures = torch.where(advantages > 0, sapo_temperature_pos, sapo_temperature_neg)
        soft_coef_1 = torch.sigmoid(temperatures * (coef_1 - 1)) * 4 / temperatures
        loss_i = -soft_coef_1 * advantages
    elif loss_type == "vespo":
        if get_gamma_weights is None:
            raise Exception("vespo is only available in TRL 0.26.0+")
        phi_seq = get_gamma_weights(
            advantages=advantages,
            log_ratio_per_token=log_ratio,
            mask=mask,
            importance_sampling_ratio=importance_sampling_ratio,
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

    if use_vllm and sampling_per_token_logps is not None and vllm_importance_sampling_correction:
        # vespo applies the IS ratio inside get_gamma_weights, so skip it here.
        if loss_type != "vespo":
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
    elif loss_type == "luspo":
        loss = (loss_i * mask.sum(1, keepdim=True)).mean()
        normalizer = current_gradient_accumulation_steps
        loss = loss / normalizer
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
# Same eager fallback as every other fullgraph region: a bare decorator leaves
# cache exhaustion fatal under fullgraph.
RL_REPLACEMENTS["grpo_compute_loss_slow"] = \
    f"from unsloth_zoo.temporary_patches.utils import torch_compile_with_fallback\n"\
    f"@torch_compile_with_fallback(dynamic = True, fullgraph = True, options = torch_compile_options)\n"\
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

        from unsloth_zoo.temporary_patches.utils import torch_compile_with_fallback
        accumulate_chunk = torch_compile_with_fallback(
            fullgraph = True,
            # [TODO] Dynamic marking causes torch.compile errors if sequence length is long
            dynamic = True,
            options = torch_compile_options,
        )(accumulate_chunk)

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


def _warn_unsupported_grpo_options(trainer):
    """Warn once per trainer about TRL GRPOConfig options this path ignores, so setting
    them is not silently dropped. Only top_entropy_quantile < 1.0 (entropy masking) is
    unimplemented; its TRL default is 1.0 in 0.22.2 through 1.12.0, so only non-defaults
    warn. use_bias_correction_kl is supported and must never be listed here.
    """
    if getattr(trainer, "_unsloth_grpo_unsupported_warned", False):
        return
    args = getattr(trainer, "args", None)

    unsupported = []
    top_entropy_quantile = getattr(args, "top_entropy_quantile", 1.0)
    if top_entropy_quantile is not None and top_entropy_quantile < 1.0:
        unsupported.append(f"top_entropy_quantile={top_entropy_quantile}")

    if unsupported:
        message = (
            "Unsloth: GRPOConfig option(s) " + ", ".join(unsupported) + " are set but "
            "are not supported by Unsloth's optimized GRPO path and will be ignored "
            "(training proceeds as standard GRPO)."
        )
        try:
            logger.warning(message)
        except Exception:
            import warnings as _warnings
            _warnings.warn(message)
        # Only latch after warning, so a config changed mid-run still gets one warning.
        try:
            trainer._unsloth_grpo_unsupported_warned = True
        except Exception:
            pass
    return
pass
RL_REPLACEMENTS["_warn_unsupported_grpo_options"] = _warn_unsupported_grpo_options


_n_chunks_deprecation_warned = False

def _warn_deprecated_n_chunks(n_chunks):
    """Warn once per process when unsloth_num_chunks is set to a non-default value.
    The parameter stays because unsloth's generated trainer passes it, but the value
    never reaches UnslothEfficientGRPO.apply, which is always called with 1.
    """
    global _n_chunks_deprecation_warned
    if _n_chunks_deprecation_warned:
        return
    if n_chunks is None or n_chunks == -1 or n_chunks == 1:
        return
    _n_chunks_deprecation_warned = True
    message = (
        "Unsloth: unsloth_num_chunks is deprecated and is ignored; the GRPO loss now "
        "always runs as a single chunk, since memory is managed by "
        "unsloth_grpo_mini_batch and unsloth_logit_chunk_multiplier."
    )
    try:
        logger.warning(message)
    except Exception:
        import warnings as _warnings
        _warnings.warn(message)
    return
pass
RL_REPLACEMENTS["_warn_deprecated_n_chunks"] = _warn_deprecated_n_chunks


# The multimodal keys TRL's GRPO trainer puts in the inputs dict. Both Unsloth logprob
# paths read this one tuple: they must forward the same inputs, or the importance ratio
# compares two different policies. unslothai/unsloth#6960.
GRPO_VISION_KEYS = (
    "pixel_values",
    "image_grid_thw",
    "pixel_attention_mask",
    "image_sizes",
    "spatial_shapes",
    "num_tiles",
    # One Gemma 4 field under both TRL names: pixel_position_ids in 1.0.x, renamed in
    # 1.1.0. Only ever one of the two is present.
    "image_position_ids",
    "pixel_position_ids",
    "num_images",
    "token_type_ids",
    "mm_token_type_ids",
)


def grpo_get_vision_inputs(source):
    """Collect the GRPO multimodal inputs out of a kwargs or inputs mapping."""
    if source is None:
        return {}
    get = getattr(source, "get", None)
    if get is None:
        return {}
    return {key: get(key, None) for key in GRPO_VISION_KEYS}
pass
RL_REPLACEMENTS["grpo_get_vision_inputs"] = grpo_get_vision_inputs


# What every released unsloth up to 2026.9.4 forwards on the no-grad side, hard coded in its
# own `_get_per_token_logps_and_entropies` replacement. The conservative answer when the
# installed companion is present but cannot be read: forwarding MORE than this on the gradient
# side alone is what makes the two policies differ, so an unknown companion is assumed to be
# one of those rather than assumed to be current.
GRPO_RELEASED_VISION_KEYS = (
    "pixel_values",
    "image_grid_thw",
    "pixel_attention_mask",
    "image_sizes",
    "num_images",
    "token_type_ids",
    "mm_token_type_ids",
)

# The names whose presence in the installed unsloth's no-grad replacement means it reads THIS
# module's key tuple rather than a list of its own. Either one is enough: unslothai/unsloth#11031
# collects the keys in a module level helper and names only the chunker inside the replacement.
# One constant rather than two literals per site, because the integration test has to ask the
# same question the gate asks and a second copy of the answer drifts.
GRPO_SHARED_HELPER_MARKERS = ("grpo_get_vision_inputs", "grpo_vision_chunks")

# Memo for grpo_companion_vision_keys. Set once per process; tests reset it to None.
_GRPO_COMPANION_VISION_KEYS = None


def grpo_companion_vision_keys():
    """Which of GRPO_VISION_KEYS the installed unsloth also forwards on the no-grad side.

    unsloth ships separately from this package, and up to 2026.9.4 its
    ``_get_per_token_logps_and_entropies`` replacement, which computes the old and reference
    logprobs, carries its own hard coded list of seven keys. Forwarding more than that on the
    gradient side alone would let the current policy see image metadata the reference policy
    never saw, so the importance ratio and the KL term would compare two different policies.
    Agreeing on the smaller set is what makes the ratio meaningful, and it is still strictly
    better than before, since the shared chunker no longer drops pixel_values outright.

    Read out of the installed unsloth's source rather than its version, so an unsloth carrying
    the companion change lifts the restriction on its own: either shared name appearing in that
    replacement means the no-grad pass reads this module's tuple. Only sys.modules is consulted:
    at this point unsloth is what is driving the run, and importing it from here is circular.
    Only an answer READ off a patcher is memoized: a module that is not there yet is a call
    that came too early, not an unsloth without the companion.
    """
    global _GRPO_COMPANION_VISION_KEYS
    if _GRPO_COMPANION_VISION_KEYS is not None:
        return _GRPO_COMPANION_VISION_KEYS

    keys = GRPO_VISION_KEYS
    module = sys.modules.get("unsloth.models.rl_replacements")
    patcher = getattr(module, "grpo_trainer__get_per_token_logps_and_entropies", None)
    if patcher is None:
        # Nothing to classify yet, and an absence here is not a fact about the process: this
        # module can be imported before unsloth installs its RL replacements, and the first
        # caller would otherwise memoize "no companion" permanently, so the restriction could
        # never be applied to the run that actually needs it. Treat it like the unreadable
        # case and leave it unmemoized, so a later call gets to look again once the import has
        # happened. The full tuple is the right answer while nothing is known to restrict it.
        return GRPO_VISION_KEYS

    source = None
    unreadable = False
    try:
        source = inspect.getsource(patcher)
    except (OSError, TypeError):
        # Source stripped, frozen, or dynamically wrapped. The companion is THERE -- the
        # patcher exists -- and the only thing missing is the ability to see which keys it
        # forwards. Leaving the full tuple there was the unsafe half of the guess: the
        # gradient pass would forward spatial_shapes, num_tiles and the position ids that
        # every released companion omits, and the importance ratio and KL term would then
        # compare two different policies with nothing to show for it. Assume the released
        # set instead, which is the answer for every unsloth that does not carry the
        # companion change, and do not memoize it: this is a failure to look, not a fact
        # about the process, so a later call gets to look again.
        source = None
        unreadable = True
    shares = source is not None and any(
        marker in source for marker in GRPO_SHARED_HELPER_MARKERS
    )
    if source is not None and not shares:
        named = tuple(
            key for key in GRPO_VISION_KEYS
            if '"%s"' % key in source or "'%s'" % key in source
        )
        # A no-grad pass that names no pixel_values at all is one this reader does not
        # understand; leave the full set rather than silently turning vision off.
        if "pixel_values" in named and len(named) != len(GRPO_VISION_KEYS):
            keys = named
            logger.warning(
                "Unsloth: the installed unsloth computes GRPO reference logprobs without "
                f"{', '.join(k for k in GRPO_VISION_KEYS if k not in named)}. Holding the "
                "gradient pass to the same inputs so both policies match. Upgrade unsloth to "
                "forward every vision kwarg on both paths."
            )
    if unreadable:
        logger.warning(
            "Unsloth: the installed unsloth's GRPO reference-logprob pass cannot be read, so "
            "which vision kwargs it forwards is unknown. Holding the gradient pass to the "
            "keys every released unsloth forwards, so both policies still match."
        )
        return GRPO_RELEASED_VISION_KEYS
    _GRPO_COMPANION_VISION_KEYS = keys
    return keys
pass
RL_REPLACEMENTS["grpo_companion_vision_keys"] = grpo_companion_vision_keys


# Warned once per process, not per step: this is a property of the installed packages.
_GRPO_COMPANION_PIXELS_WARNED = False


def grpo_shared_vision_inputs(source):
    """grpo_get_vision_inputs, restricted to what both logprob passes actually forward.

    Two restrictions, not one. The key list is the first: a companion with its own hard coded
    tuple never sees the keys outside it. The second is a SHAPE, and intersecting names cannot
    express it -- a companion that does not share the chunker slices the pixels itself, and its
    loop drops ``pixel_values`` outright unless ``image_grid_thw`` is there to slice them by
    (``pixel_values_chunks.append(None)`` in its else branch). For a VLM that carries no grid,
    which is Gemma 3, InternVL and LFM2-VL, forwarding pixels on the gradient side alone would
    leave the current policy looking at images the reference policy never saw, and the
    importance ratio and the KL term would compare two different policies however carefully the
    key names were matched.

    That second restriction is NOT applied here, and the layer matters. ``pixel_values`` is a
    sentinel in the caller as well as a model input: ``grpo_accumulated_loss`` reads it to
    decide whether to left-pack the batch (recomputing ``max_left_pad``, repacking
    ``input_ids`` and rebuilding ``completion_mask``) and whether to take the sequence-packing
    path. The companion chooses those on its own ``pixel_values``, which is a real tensor --
    it appends None per chunk INSIDE its loop, after the branch is already chosen -- so
    blanking the key here would stop the two passes disagreeing about pixels and start them
    disagreeing about how the sequences are arranged, which is the worse comparison of the
    two. The suppression therefore happens one layer down, in ``grpo_vision_chunks``, on the
    chunk that is actually forwarded. See ``grpo_companion_drops_pixels``.
    """
    keys = grpo_companion_vision_keys()
    if len(keys) == len(GRPO_VISION_KEYS):
        return grpo_get_vision_inputs(source)
    return {key: value for key, value in grpo_get_vision_inputs(source).items() if key in keys}
pass
RL_REPLACEMENTS["grpo_shared_vision_inputs"] = grpo_shared_vision_inputs


def grpo_companion_drops_pixels(vision):
    """Whether the installed companion's no-grad loop forwards no pixels for THIS batch.

    Its else branch is ``pixel_values_chunks.append(None)``: with no ``image_grid_thw`` to
    slice the pixels by, that loop has nothing to forward, so for a VLM that carries no grid
    -- Gemma 3, InternVL, LFM2-VL -- the reference policy sees text where the gradient pass
    would see images. Matching it is what keeps the importance ratio and the KL term a
    comparison of one policy with itself.

    False whenever the companion shares this module's chunker, because then both passes slice
    with the same code and nothing needs to be given up.
    """
    if vision.get("pixel_values", None) is None:
        return False
    if vision.get("image_grid_thw", None) is not None:
        return False
    return len(grpo_companion_vision_keys()) != len(GRPO_VISION_KEYS)
pass
RL_REPLACEMENTS["grpo_companion_drops_pixels"] = grpo_companion_drops_pixels


def grpo_vision_chunks(vision, total_samples, batch_size):
    """Slice the GRPO multimodal inputs into per-chunk forward kwargs, one dict per chunk.

    One implementation for both logprob paths, so they cannot index the same tensors
    differently. The axis per family mirrors TRL's own ``_get_per_token_logps_and_entropies``:

    * ``image_grid_thw`` (Qwen2-VL): ``pixel_values`` by patch row, the grid by image.
    * ``image_position_ids`` (Gemma 4): both by image.
    * ``spatial_shapes`` (LFM2-VL): ``pixel_values``, ``pixel_attention_mask`` and
      ``spatial_shapes`` by tile, with ``num_tiles`` giving the tiles per sample.
    * ``num_tiles`` alone (InternVL): ``pixel_values`` by tile.
    * anything else: by image when there is one row per image, else by sample.

    An unrecognised model still gets its ``pixel_values``, as stock TRL does; dropping
    them silently recomputes the reference logprobs from the text alone.
    """
    def _as_int_list(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().reshape(-1).tolist()
        try:
            return [int(n) for n in value]
        except TypeError:
            return None

    def _first_dim_len(value):
        if value is None:
            return None
        if hasattr(value, "shape"):
            return value.shape[0]
        try:
            return len(value)
        except TypeError:
            return None

    pixel_values = vision.get("pixel_values", None)
    image_grid_thw = vision.get("image_grid_thw", None)
    pixel_attention_mask = vision.get("pixel_attention_mask", None)
    image_sizes = vision.get("image_sizes", None)
    spatial_shapes = vision.get("spatial_shapes", None)
    # One local for both spellings, but it must leave under the name it arrived with:
    # the model kwarg is named the same as the inputs key.
    image_position_ids = vision.get("image_position_ids", None)
    position_ids_key = "image_position_ids"
    if image_position_ids is None:
        image_position_ids = vision.get("pixel_position_ids", None)
        position_ids_key = "pixel_position_ids"
    token_type_ids = vision.get("token_type_ids", None)
    mm_token_type_ids = vision.get("mm_token_type_ids", None)
    num_images = _as_int_list(vision.get("num_images", None))
    num_tiles = _as_int_list(vision.get("num_tiles", None))

    # A count list is only a per-sample cumulative index when it has one entry per row.
    if num_images is not None and len(num_images) != total_samples:
        num_images = None
    if num_tiles is not None and len(num_tiles) != total_samples:
        num_tiles = None

    cum_imgs = None if num_images is None else torch.tensor([0] + num_images).cumsum(0)
    cum_tiles = None if num_tiles is None else torch.tensor([0] + num_tiles).cumsum(0)

    cum_rows = None
    if image_grid_thw is not None and pixel_values is not None and num_images is not None:
        rows_per_image = image_grid_thw.prod(dim = -1)
        rows_per_sample = torch.split(rows_per_image, num_images)
        rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
        # .item() in the loop below, so keep it on CPU or every chunk pays a sync.
        cum_rows = torch.cat(
            [
                torch.tensor([0], device = rows_per_sample.device),
                rows_per_sample.cumsum(0),
            ]
        ).cpu()

    total_images = None if num_images is None else sum(num_images)

    def _row_axis_is_images(value, flat_ndim):
        """Whether this tensor's first dimension counts IMAGES rather than samples.

        Equal lengths are not enough on their own. A model that pads its image tensors to the
        widest sample -- SmolVLM, Idefics -- keeps the sample axis first, and a batch whose
        image counts happen to sum to the number of samples (``num_images = [2, 0]`` over two
        samples) makes the two readings numerically identical. Slicing such a batch by image
        sends both of the first sample's rows to one chunk and an empty tensor to the next.

        The layouts differ in RANK, which the counts cannot express: the padded one carries an
        explicit per-sample image axis (``[B, max_images, C, H, W]`` for pixels,
        ``[B, max_images, 2]`` for image sizes), one dimension more than the flattened form
        this branch is for.
        """
        if value is None or total_images is None:
            return False
        if _first_dim_len(value) != total_images:
            return False
        if total_images != total_samples:
            # The length has already settled it. Rank can only add anything inside the region
            # where the image axis and the sample axis are the same length; outside it a
            # genuinely flattened tensor is free to outrank flat_ndim, and LLaVA-NeXT and
            # LLaVA-OneVision do exactly that -- they pad on the PATCH axis and store
            # [n_images, max_patches, C, H, W], one row per image with a rank of 5. Applying
            # the rank test there sent each sample the rows of whichever sample shares its
            # index, and dropped the surplus images entirely.
            return True
        ndim = getattr(value, "ndim", None)
        if isinstance(ndim, int) and ndim > flat_ndim:
            return False
        return True

    def _image_sizes_slice(start, end, img_start, img_end):
        if image_sizes is None:
            return None
        if img_start is not None and _row_axis_is_images(image_sizes, 2):
            return image_sizes[img_start:img_end]
        return image_sizes[start:end]

    # Asked once per call, not per chunk: it reads the installed companion's source.
    drop_pixels = grpo_companion_drops_pixels(vision)
    global _GRPO_COMPANION_PIXELS_WARNED
    if drop_pixels and not _GRPO_COMPANION_PIXELS_WARNED:
        _GRPO_COMPANION_PIXELS_WARNED = True
        logger.warning(
            "Unsloth: the installed unsloth computes GRPO reference logprobs with a chunk "
            "loop of its own, and that loop forwards no pixel_values for a model without "
            "image_grid_thw. Holding the gradient pass to the same inputs, so both policies "
            "match, which means this run trains on the text of these samples. Upgrade "
            "unsloth to train on the images."
        )

    chunks = []
    current_pixel_idx = 0
    for start in range(0, total_samples, batch_size):
        end = min(start + batch_size, total_samples)
        chunk = {}
        if token_type_ids is not None:
            chunk["token_type_ids"] = token_type_ids[start:end]
        if mm_token_type_ids is not None:
            chunk["mm_token_type_ids"] = mm_token_type_ids[start:end]

        img_start = img_end = None
        if cum_imgs is not None:
            img_start, img_end = int(cum_imgs[start]), int(cum_imgs[end])

        if pixel_values is None:
            if image_sizes is not None:
                chunk["image_sizes"] = _image_sizes_slice(start, end, img_start, img_end)
            chunks.append(chunk)
            continue

        if image_grid_thw is not None:
            if num_images is None:
                grid_slice = image_grid_thw[start:end]
                batch_pixel_count = grid_slice.prod(dim = -1).sum().item()
                start_pixel_idx = current_pixel_idx
                end_pixel_idx = current_pixel_idx + batch_pixel_count
                current_pixel_idx = end_pixel_idx
            else:
                start_pixel_idx = cum_rows[start].item()
                end_pixel_idx = cum_rows[end].item()
                grid_slice = image_grid_thw[img_start:img_end]
            chunk["image_grid_thw"] = grid_slice
            chunk["pixel_values"] = pixel_values[start_pixel_idx:end_pixel_idx]
            if pixel_attention_mask is not None:
                if img_start is not None and pixel_attention_mask.shape[0] == image_grid_thw.shape[0]:
                    chunk["pixel_attention_mask"] = pixel_attention_mask[img_start:img_end]
                elif (
                    pixel_attention_mask.shape[0] == pixel_values.shape[0]
                    and pixel_attention_mask.shape[0] != total_samples
                ):
                    chunk["pixel_attention_mask"] = pixel_attention_mask[start_pixel_idx:end_pixel_idx]
                else:
                    chunk["pixel_attention_mask"] = pixel_attention_mask[start:end]
            if image_sizes is not None:
                chunk["image_sizes"] = _image_sizes_slice(start, end, img_start, img_end)
        elif image_position_ids is not None:
            if img_start is None:
                chunk["pixel_values"] = pixel_values[start:end]
                chunk[position_ids_key] = image_position_ids[start:end]
            else:
                chunk["pixel_values"] = pixel_values[img_start:img_end]
                chunk[position_ids_key] = image_position_ids[img_start:img_end]
            if pixel_attention_mask is not None:
                chunk["pixel_attention_mask"] = pixel_attention_mask[start:end]
            if image_sizes is not None:
                chunk["image_sizes"] = _image_sizes_slice(start, end, img_start, img_end)
        elif spatial_shapes is not None:
            if cum_tiles is not None:
                tile_start, tile_end = int(cum_tiles[start]), int(cum_tiles[end])
            elif img_start is not None and _first_dim_len(spatial_shapes) == total_images:
                tile_start, tile_end = img_start, img_end
            else:
                tile_start, tile_end = start, end
            chunk["pixel_values"] = pixel_values[tile_start:tile_end]
            chunk["spatial_shapes"] = spatial_shapes[tile_start:tile_end]
            if pixel_attention_mask is not None:
                chunk["pixel_attention_mask"] = pixel_attention_mask[tile_start:tile_end]
            if image_sizes is not None:
                chunk["image_sizes"] = _image_sizes_slice(start, end, img_start, img_end)
        elif cum_tiles is not None:
            tile_start, tile_end = int(cum_tiles[start]), int(cum_tiles[end])
            chunk["pixel_values"] = pixel_values[tile_start:tile_end]
            if pixel_attention_mask is not None:
                chunk["pixel_attention_mask"] = pixel_attention_mask[start:end]
            if image_sizes is not None:
                chunk["image_sizes"] = _image_sizes_slice(start, end, img_start, img_end)
        else:
            # Not a bare length comparison: see _row_axis_is_images. A flattened image tensor
            # is one row per image, [N, C, H, W]; a padded one keeps the sample axis in front
            # of it and must be sliced by sample like everything else in this chunk.
            if img_start is not None and _row_axis_is_images(pixel_values, 4):
                chunk["pixel_values"] = pixel_values[img_start:img_end]
            else:
                chunk["pixel_values"] = pixel_values[start:end]
            if pixel_attention_mask is not None:
                chunk["pixel_attention_mask"] = pixel_attention_mask[start:end]
            if image_sizes is not None:
                chunk["image_sizes"] = _image_sizes_slice(start, end, img_start, img_end)
        if drop_pixels:
            # Here, on the forwarded chunk, rather than on the mapping the caller branches on.
            # The mask goes with the pixels: the companion appends None for it in the same
            # branch, and a mask with nothing to mask is not a shape any of these models take.
            chunk.pop("pixel_values", None)
            chunk.pop("pixel_attention_mask", None)
        chunks.append(chunk)
    return chunks
pass
RL_REPLACEMENTS["grpo_vision_chunks"] = grpo_vision_chunks


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
    # Body-local import so the copy inlined into the generated trainer cache resolves.
    try:
        from unsloth_zoo.rl_replacements import _warn_unsupported_grpo_options
        _warn_unsupported_grpo_options(trainer)
    except Exception:
        pass

    # Body-local: this source is copied into the generated trainer without its imports.
    from unsloth_zoo.rl_replacements import (
        grpo_shared_vision_inputs as _grpo_get_vision_inputs,
        grpo_vision_chunks as _grpo_vision_chunks,
    )
    vision_inputs = _grpo_get_vision_inputs(kwargs)
    pixel_values = vision_inputs.get('pixel_values', None)
    image_grid_thw = vision_inputs.get('image_grid_thw', None)
    # Released unsloth 2026.9.4 decides whether multi-image GRPO is supported by grepping
    # inspect.getsource(grpo_accumulated_loss) for "num_images", so moving the handling into
    # grpo_vision_chunks makes that probe answer no and raise "Please upgrade unsloth_zoo" at
    # the user who just did. The chunker reads num_images out of vision_inputs itself; this
    # binding is what the released probe looks for, and it keeps the name meaningful here.
    num_images = vision_inputs.get('num_images', None)
    # Transformers 5.x requires token_type_ids/mm_token_type_ids for some vision models
    token_type_ids = vision_inputs.get('token_type_ids', None)
    mm_token_type_ids = vision_inputs.get('mm_token_type_ids', None)
    if mm_token_type_ids is not None or image_grid_thw is not None:
        mm_token_type_ids = _unsloth_fix_mm_token_type_ids(
            trainer.processing_class, input_ids, mm_token_type_ids
        )
        vision_inputs['mm_token_type_ids'] = mm_token_type_ids
    # Thread vLLM sampling logprobs when something actually consumes them: the off-policy mask
    # (off_policy_mask_threshold) or the IS ratio (vllm_importance_sampling_correction). The mask
    # needs them regardless of IS correction (matching TRL, which feeds them to get_off_policy_mask
    # either way); the IS ratio stays gated on the correction flag inside grpo_compute_loss. On the
    # plain vLLM path (neither active) they are dropped so nothing pays for an unused aligned/compiled
    # input and grpo_compute_loss returns None (not empty) delta/flat_is_ratio.
    _sampling_logps_used = (
        getattr(trainer, "vllm_importance_sampling_correction", False)
        or getattr(trainer.args, "off_policy_mask_threshold", None) is not None
    )
    sampling_per_token_logps = kwargs.get("sampling_per_token_logps", None) if _sampling_logps_used else None
    temperature = kwargs.get("temperature", 1.0)
    logit_scale_multiply = kwargs.get("logit_scale_multiply", 0.0)
    logit_scale_divide   = kwargs.get("logit_scale_divide", 0.0)
    logit_softcapping    = kwargs.get("logit_softcapping", 0.0)
    prev_max_left_pad    = kwargs.get("max_left_pad", 0) # max_left_pad for LLM training, enabled by default.

    # Pop from kwargs to avoid downstream issues.
    _ = kwargs.pop("sampling_per_token_logps", None)
    kwargs["vllm_importance_sampling_cap"] = getattr(trainer.args, "vllm_importance_sampling_cap", None)
    # Older TRL lacks this arg; fall back to token_truncate (legacy clamp(max=cap) behavior).
    kwargs["vllm_importance_sampling_mode"] = getattr(trainer.args, "vllm_importance_sampling_mode", None) or "token_truncate"
    kwargs["vllm_importance_sampling_clip_min"] = getattr(trainer.args, "vllm_importance_sampling_clip_min", None)
    kwargs["vllm_importance_sampling_clip_max"] = getattr(trainer.args, "vllm_importance_sampling_clip_max", None)
    kwargs["get_sapo_token_loss"] = trainer.get_sapo_token_loss if hasattr(trainer, "get_sapo_token_loss") else None
    kwargs["sapo_temperature_pos"] = trainer.args.sapo_temperature_pos if hasattr(trainer.args, "sapo_temperature_pos") else None
    kwargs["sapo_temperature_neg"] = trainer.args.sapo_temperature_neg if hasattr(trainer.args, "sapo_temperature_neg") else None
    kwargs["get_gamma_weights"] = trainer.get_gamma_weights if hasattr(trainer, "get_gamma_weights") else None
    kwargs["vespo_k_pos"] = trainer.args.vespo_k_pos if hasattr(trainer.args, "vespo_k_pos") else 2.0
    kwargs["vespo_k_neg"] = trainer.args.vespo_k_neg if hasattr(trainer.args, "vespo_k_neg") else 3.0
    kwargs["vespo_lambda_pos"] = trainer.args.vespo_lambda_pos if hasattr(trainer.args, "vespo_lambda_pos") else 3.0
    kwargs["vespo_lambda_neg"] = trainer.args.vespo_lambda_neg if hasattr(trainer.args, "vespo_lambda_neg") else 2.0
    off_policy_mask_threshold = trainer.args.off_policy_mask_threshold if hasattr(trainer.args, "off_policy_mask_threshold") else None
    kwargs["off_policy_mask_threshold"] = off_policy_mask_threshold
    # get_off_policy_mask exists on TRL >= 0.27.0; its 3rd parameter was `old_per_token_logps` in 0.27.0
    # and renamed to `sampling_per_token_logps` in 0.27.1 (huggingface/trl#4857), so a fixed keyword call
    # crashes on one side of the rename. Wrap it in a signature-stable adapter here, outside the compiled
    # loss, so grpo_compute_loss always calls it with one keyword. Detect the real name once via inspect
    # and cache the adapter on the trainer; a fresh closure every step would re-trigger torch.compile.
    _off_policy_mask_fn = trainer.get_off_policy_mask if hasattr(trainer, "get_off_policy_mask") else None
    if _off_policy_mask_fn is None or off_policy_mask_threshold is None:
        kwargs["get_off_policy_mask"] = None
    else:
        _adapter = getattr(trainer, "_unsloth_off_policy_mask_adapter", None)
        # Compare by value, not identity: trainer.get_off_policy_mask returns a fresh bound-method
        # object on every access, so `is not` would always miss and rebuild the adapter each step
        # (re-triggering torch.compile). `!=` on bound methods compares __self__ and __func__, so the
        # cached adapter is reused; the first call still rebuilds since None != the bound method.
        if getattr(_adapter, "_unsloth_wrapped", None) != _off_policy_mask_fn:
            import inspect as _inspect
            try:
                _params = _inspect.signature(_off_policy_mask_fn).parameters
            except (TypeError, ValueError):
                _params = {}
            if "old_per_token_logps" in _params and "sampling_per_token_logps" not in _params:
                # TRL 0.27.0 named the mismatch-logprobs parameter old_per_token_logps.
                def _adapter(advantages, per_token_logps, sampling_per_token_logps, mask, off_policy_threshold):
                    return _off_policy_mask_fn(
                        advantages=advantages,
                        per_token_logps=per_token_logps,
                        old_per_token_logps=sampling_per_token_logps,
                        mask=mask,
                        off_policy_threshold=off_policy_threshold,
                    )
            else:
                # TRL >= 0.27.1 / 1.7.x use sampling_per_token_logps (also the default going forward).
                def _adapter(advantages, per_token_logps, sampling_per_token_logps, mask, off_policy_threshold):
                    return _off_policy_mask_fn(
                        advantages=advantages,
                        per_token_logps=per_token_logps,
                        sampling_per_token_logps=sampling_per_token_logps,
                        mask=mask,
                        off_policy_threshold=off_policy_threshold,
                    )
            _adapter._unsloth_wrapped = _off_policy_mask_fn
            trainer._unsloth_off_policy_mask_adapter = _adapter
        kwargs["get_off_policy_mask"] = _adapter
    # Read inside the compiled loss to gate the IS ratio, which the off-policy mask must not gate.
    kwargs["vllm_importance_sampling_correction"] = getattr(trainer, "vllm_importance_sampling_correction", False)
    # Follows TRL's own value; older TRL has no such field and False is correct there.
    kwargs["use_bias_correction_kl"] = getattr(trainer.args, "use_bias_correction_kl", False)
    kwargs["use_vllm"] = trainer.use_vllm
    # Generated trainers still pass unsloth_num_chunks; nothing downstream reads it.
    try:
        from unsloth_zoo.rl_replacements import _warn_deprecated_n_chunks
        _warn_deprecated_n_chunks(n_chunks)
    except Exception:
        pass

    if kwargs["vllm_importance_sampling_clip_max"] is None and kwargs["vllm_importance_sampling_cap"] is not None:
        kwargs["vllm_importance_sampling_clip_min"] = 0
        kwargs["vllm_importance_sampling_clip_max"] = kwargs["vllm_importance_sampling_cap"]

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

        if trainer.use_vllm and sampling_per_token_logps is not None:
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

    import math
    total_samples = input_ids.shape[0]
    batch_size = math.ceil(total_samples / B)
    input_ids_chunks = []
    attention_mask_chunks = []
    completion_ids_chunks = []
    for start in range(0, total_samples, batch_size):
        end = min(start + batch_size, total_samples)
        input_ids_chunks.append(input_ids[start:end])
        attention_mask_chunks.append(attention_mask[start:end])
        completion_ids_chunks.append(completion_input_ids[start:end])

    # Shared with the no-grad pass, so the two cannot slice the same tensors differently.
    vision_chunks = _grpo_vision_chunks(vision_inputs, total_samples, batch_size)

    zipped_inputs = zip(
        input_ids_chunks,
        attention_mask_chunks,
        vision_chunks,
        completion_ids_chunks,
    )

    # Bound in the body, not at module scope, for the reason spelled out just below: this
    # function's source is copied into the generated UnslothGRPOTrainer cache without
    # unsloth_zoo's module imports, so a module-level import reaches the import path and
    # not the one that actually runs in production.
    from contextlib import nullcontext

    if trainer._autocast_dtype is None:
        autocaster = nullcontext()
    else:
        autocaster = torch.amp.autocast(device_type = trainer.model.device.type, dtype = trainer._autocast_dtype)

    # PrefixGrouper grad path. This function's source is copied into the generated
    # UnslothGRPOTrainer cache without unsloth_zoo's module imports, so bind names
    # inside the body; the prefix_grouper import stays lazy + guarded (circular import,
    # may be absent) and a failed import just leaves PG off.
    from unsloth_zoo.temporary_patches.common import UNSLOTH_ENABLE_LOGGING

    # Memoize env gate + import once per process on the function object (which survives
    # into the cache). Env gate checked first so =0 never imports PG code; () = PG off.
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
    # Skip PG under vLLM (fast_inference=True): rollout dominates the step, so the
    # saving is small and the first-use self-verify is net overhead.
    _pg_engage = bool(_pg_funcs) and not getattr(trainer, "use_vllm", False)

    # ---- PrefixGrouper (GRPO shared-prompt dedup; UNSLOTH_GRPO_PREFIX_GROUPER=0 disables) ----
    # Each prompt's G completions share the prefix; PG forwards it once + the G suffixes
    # (FlexAttention shared-prefix mask), cutting G*(P+R) tokens to P+G*R. First-use
    # self-verify vs the full-row packed new_logprobs; grads flow through the shared stream
    # (prefix grad once = sum of G repeats, identical math). Off/failed/unverified ->
    # full-row packed path runs as before.
    _pg_result = None
    _pg_use = False
    _pg_skip_pack = False
    _pg_num_gen = getattr(trainer, "num_generations", None)
    # Runtime gate; broad except -> engage False.
    if _pg_engage and _pg_funcs:
        try:
            _pg_build_layout, _pg_enabled_fn, _pg_verify_on, _pg_tol_ok, _PG_TOL_KILL = _pg_funcs
            # Exclusions: softcap models (gemma2) - the FlexAttention kernel skips
            # attn_logit_softcapping; hybrid SSM (FalconH1) and MoE (Qwen3-MoE) - their
            # decoders do not thread prefix_seg_info, so state would leak across suffixes.
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
            # Build the layout from the left-packed input_ids with the original left-pad
            # counts so the prefix/suffix split matches the packed path (_pack_cstart) and
            # the verify is apples-to-apples. Cap the PG span at any sliding window,
            # mirroring the packed _pack_sw guard.
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
                # trust only if the verified envelope covers this batch's lengths
                # (re-verify when T or the longest segment grows)
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
                    # `.logits` carries hidden states only when the forward is the
                    # Unsloth generated one honouring UNSLOTH_RETURN_HIDDEN_STATES;
                    # otherwise it is real [T, vocab] logits and the lm_head matmul
                    # dies. Dispatch on width, as the padded path already does.
                    _pack_h   = _pack_hidden[0, :-1, :][_pack_ctgt].unsqueeze(0)
                    _pack_tid = _pack_flat_ids[0, 1:][_pack_ctgt].unsqueeze(0)
                    if _pack_h.shape[-1] == lm_head.shape[1]:
                        _pack_sel = chunked_hidden_states_selective_log_softmax(
                            _pack_h, lm_head, _pack_tid, _pack_chunks,
                            logit_scale_multiply, logit_scale_divide, logit_softcapping, temperature,
                        )[0]
                    else:
                        # Raw logits: the forward already applied scale/softcap.
                        _pack_sel = chunked_selective_log_softmax(
                            _pack_h, _pack_tid,
                            temperature = temperature, chunks = _pack_chunks,
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
                            # same width dispatch as the packed call above: this forward
                            # returns raw logits whenever that one did, and the first
                            # packed batch always lands here
                            if _pack_rh.shape[-1] == lm_head.shape[1]:
                                _pack_rsel = chunked_hidden_states_selective_log_softmax(
                                    _pack_rh[:, :-1, :], lm_head, _pack_real[:, 1:], 1,
                                    logit_scale_multiply, logit_scale_divide, logit_softcapping, temperature,
                                )[0]
                            else:
                                _pack_rsel = chunked_selective_log_softmax(
                                    _pack_rh[:, :-1, :], _pack_real[:, 1:],
                                    temperature = temperature, chunks = 1,
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
    # Verify runs under no_grad, then a separate grad forward builds new_logprobs,
    # so no inference tensors are saved for backward.
    def _pg_grad_forward():
        _pg_chunks = max(1, total_rows * multiplier)
        with autocaster:
            _h = unwrapped_model(
                input_ids = _pg_layout.flat_ids,
                position_ids = _pg_layout.position_ids,
                prefix_seg_info = _pg_layout.prefix_seg_info,
                use_cache = False,
            ).logits
            # Same width dispatch as the packed path and compute_logprobs_chunk.
            # `.logits` carries hidden states only when the forward is the Unsloth
            # generated one honouring UNSLOTH_RETURN_HIDDEN_STATES; otherwise it is
            # real [T, vocab] logits. extract_logps always calls its helper as
            # (hidden, lm_head, ids, chunks, ...), so pass a raw-logits helper with
            # that same signature, which skips the lm_head matmul and the scale /
            # softcap the forward already applied.
            _pg_fn = chunked_hidden_states_selective_log_softmax
            if _h.shape[-1] != lm_head.shape[1]:
                def _pg_fn(_pg_h, _pg_lm, _pg_ids, _pg_n, _pg_lsm, _pg_lsd, _pg_lsc, _pg_t):
                    return chunked_selective_log_softmax(
                        _pg_h, _pg_ids, temperature = _pg_t, chunks = _pg_n,
                    )
            _pg_lp = _pg_layout.extract_logps(
                _h, lm_head, _pg_fn,
                _pg_chunks, logit_scale_multiply, logit_scale_divide,
                logit_softcapping, temperature,
            )  # [total_rows, W] with grad
            # GPT-OSS offload race guard
            device_synchronize()
            return _pg_lp

    if _pg_layout is not None:
        # A verify-phase OOM (packed graph co-resident) does not prove PG alone cannot fit;
        # only an OOM after the packed graph is freed is worth marking unsafe.
        _pg_phase_verify = False
        try:
            if not _pg_trusted:
                # first use: verify vs the packed new_logprobs. < tol_ok -> trust;
                # >= TOL_KILL -> unsafe forever; borderline -> fall back this shape.
                if _pack_use and _pack_result is not None:
                    _pg_phase_verify = True   # packed graph still co-resident
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
                # else: no packed reference -> cannot verify -> fall back.
            if _pg_trusted:
                # free the packed graph BEFORE the grad forward: holding both can OOM when
                # PG alone would fit, and on PG failure the padded loop recomputes anyway.
                _pack_hidden = _pack_sel = _pack_result = None
                _pg_phase_verify = False   # packed freed: an OOM below is PG-alone
                _pg_result = _pg_grad_forward()
                _pg_use = True
        except Exception as _pg_err2:
            _pg_use = False
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
            # untrust this signature so the next batch runs the packed path again
            _pg_v = getattr(unwrapped_model, "_unsloth_prefix_grouper_grad_verified", None)
            if isinstance(_pg_v, dict):
                _pg_v.pop(_pg_layout.signature, None)
            if isinstance(_pg_err2, torch.cuda.OutOfMemoryError):
                # mark unsafe only for a PG-alone OOM (deterministic at these lengths);
                # a verify-phase OOM (packed co-resident) proves nothing, just retry.
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
        new_logprobs = _pg_result            # PrefixGrouper verified -> skip the loop
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

    def _offload_device_module(tensor_or_device):
        # Stream/Event module for the offload copy. torch.cuda is also the HIP
        # backend, so ROCm reports is_cuda and needs no branch of its own; XPU has
        # its own namespace, matching gradient_checkpointing.py. Anything else
        # (CPU, MPS, ...) returns None and takes the pageable copy.
        device = getattr(tensor_or_device, "device", tensor_or_device)
        if device.type == "cuda": return torch.cuda
        if device.type == "xpu": return getattr(torch, "xpu", None)
        return None

    class Unsloth_Offloaded_Log_Softmax(torch.autograd.Function):
        """Manual gradient checkpointing / CPU offloading for log softmax."""
        @staticmethod
        def forward(ctx, hidden_states, lm_head, index, chunks,
                    logit_scale_multiply, logit_scale_divide,
                    logit_softcapping, temperature):
            # Detach so we don't keep the graph (and extra memory) on CPU.
            detached_hidden_states = hidden_states.detach().contiguous()
            ctx.device = hidden_states.device
            ctx.copy_event = None

            # Always offload: this path only runs when the caller is already memory bound
            # (long completions / large batches), so the win is overlapping the copy.
            saved_hidden_states = None
            device_module = _offload_device_module(detached_hidden_states)
            if device_module is not None:
                # Async D2H on a side stream; backward MUST wait on copy_event before
                # the H2D reload or it races the copy.
                try:
                    pinned_buffer = torch.empty_like(detached_hidden_states, device = "cpu", pin_memory = True)
                    if pinned_buffer is not None:
                        current_stream = device_module.current_stream(detached_hidden_states.device)
                        copy_stream = device_module.Stream(device = detached_hidden_states.device)
                        copy_stream.wait_stream(current_stream)
                        with device_module.stream(copy_stream):
                            pinned_buffer.copy_(detached_hidden_states, non_blocking = True)
                        # Keeps the GPU storage alive until the side-stream copy finishes.
                        detached_hidden_states.record_stream(copy_stream)
                        copy_event = device_module.Event()
                        copy_event.record(copy_stream)
                        saved_hidden_states = pinned_buffer
                        ctx.copy_event = copy_event
                except (RuntimeError, OSError, AttributeError):
                    # Any accelerator that cannot do pinned side-stream copies falls
                    # back below; correctness never depends on this path.
                    saved_hidden_states = None
                    ctx.copy_event = None
            if saved_hidden_states is None:
                # No accelerator, or the async copy is unavailable: pageable copy.
                saved_hidden_states = detached_hidden_states.to("cpu", non_blocking = True)
            ctx.saved_hidden_states = saved_hidden_states
            # Drop the clone before the log-softmax below. hidden_states is usually a
            # [:, :-1, :] slice, so .contiguous() allocated a full copy; holding the
            # reference across the forward would keep it resident alongside the chunk
            # logits. record_stream still blocks reuse until the D2H lands, so the
            # allocator reclaims it mid-compute rather than at the end of forward.
            del detached_hidden_states

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
            if ctx.copy_event is not None:
                # The offload copy must land before the H2D reload.
                device_module = _offload_device_module(ctx.device)
                ctx.copy_event.wait(device_module.current_stream(ctx.device))
            hidden_states = to_device(ctx.saved_hidden_states, ctx.device)
            hidden_states.requires_grad_(True)

            lm_head = ctx.lm_head
            if ctx.lm_head_requires_grad:
                # Recompute against a private leaf. A Tensor.register_hook on the real
                # lm_head fires for tensors named in autograd.grad's inputs, so reusing
                # it here would run a user's grad mask / scaler once on this local
                # gradient and again when the returned gradient reaches lm_head.
                lm_head = lm_head.detach().requires_grad_(True)
            index = ctx.index

            with torch.enable_grad():
                output = chunked_hidden_states_selective_log_softmax(
                    hidden_states, lm_head, index, *ctx.args
                )

            # autograd.grad, not backward: backward writes into leaf .grad, which the
            # outer AccumulateGrad would then double-count.
            grad_inputs = torch.autograd.grad(
                output,
                (hidden_states, lm_head) if ctx.lm_head_requires_grad else (hidden_states,),
                grad_output,
            )

            return (
                grad_inputs[0],
                grad_inputs[1] if ctx.lm_head_requires_grad else None,
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
        vision_chunk,
        completion_ids
    ) in zipped_inputs:
            with autocaster:
                if pixel_values is None:
                    new_hidden_states_chunk = unwrapped_model(
                        input_ids = input_ids_chunk,
                        attention_mask = attention_mask_chunk,
                        **vision_chunk,
                    ).logits

                    new_hidden_states_chunk = new_hidden_states_chunk[:, -(logits_to_keep + max_left_pad + 1): , :]
                    new_hidden_states_chunk = new_hidden_states_chunk[:, :-1, :]
                    logprobs_chunk = compute_logprobs_chunk(new_hidden_states_chunk, completion_ids, input_ids_chunk)
                else:
                    new_hidden_states_chunk = unwrapped_model(
                        input_ids = input_ids_chunk,
                        attention_mask = attention_mask_chunk,
                        logits_to_keep = logits_to_keep + 1,
                        **vision_chunk,
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
pass
RL_REPLACEMENTS["grpo_accumulated_loss"] = grpo_accumulated_loss

from .dataset_utils import sft_prepare_dataset
RL_REPLACEMENTS["sft_prepare_dataset"] = sft_prepare_dataset


def _distillation_project_logits(
    hidden_states,
    lm_head,
    lm_head_bias = None,
    logit_scale: float = 1.0,
    logit_softcapping: float = 0.0,
):
    """``hidden_states @ lm_head.T`` plus the per model logit post processing.

    ``logit_scale`` is Cohere's multiplier (Muse Glimmer calls it
    ``output_multiplier``) and ``logit_softcapping`` is Gemma's; both are read off
    the model's own config so a chunk matches what that model's full forward would
    have produced. Co-locate on the head's device first: under accelerate the
    output head can sit on a different device from the hidden states.
    """
    hidden_states = hidden_states.to(device = lm_head.device, dtype = lm_head.dtype)
    logits = (hidden_states @ lm_head.t()).float()
    if lm_head_bias is not None:
        logits = logits + lm_head_bias.float()
    if logit_scale != 1.0:
        logits = logits * logit_scale
    if logit_softcapping is not None and logit_softcapping != 0.0:
        logits = logit_softcapping * torch.tanh(logits / logit_softcapping)
    return logits


def _distillation_generalized_jsd(student_log_probs, teacher_log_probs, beta: float):
    """``beta`` = 0 forward KL, 1 reverse KL, anything between generalized JSD.

    The endpoints are their own branches on purpose: substituting beta = 0 or 1
    into the interior expression collapses the mixture onto a single distribution
    and the divergence degenerates to exactly zero.

    ``F.kl_div(input, target)`` computes ``target * (log target - input)``, which
    is why the arguments look swapped against the KL as written in the paper.
    """
    if beta == 0.0:
        return torch.nn.functional.kl_div(
            student_log_probs, teacher_log_probs, reduction = "none", log_target = True,
        )
    if beta == 1.0:
        return torch.nn.functional.kl_div(
            teacher_log_probs, student_log_probs, reduction = "none", log_target = True,
        )
    beta_t = torch.tensor(beta, dtype = student_log_probs.dtype, device = student_log_probs.device)
    mixture_log_probs = torch.logsumexp(
        torch.stack([
            student_log_probs + torch.log1p(-beta_t),
            teacher_log_probs + torch.log(beta_t),
        ]),
        dim = 0,
    )
    kl_teacher = torch.nn.functional.kl_div(
        mixture_log_probs, teacher_log_probs, reduction = "none", log_target = True,
    )
    kl_student = torch.nn.functional.kl_div(
        mixture_log_probs, student_log_probs, reduction = "none", log_target = True,
    )
    return beta_t * kl_teacher + (1 - beta_t) * kl_student


def _distillation_jsd_chunk(
    student_hidden_states, student_lm_head, student_lm_head_bias,
    student_logit_scale, student_final_logit_softcapping,
    teacher_hidden_states, teacher_lm_head, teacher_lm_head_bias,
    teacher_logit_scale, teacher_final_logit_softcapping,
    beta, temperature, valid,
):
    """One chunk: project both models, score the divergence, return the sums.

    Called under gradient checkpointing, so only ``(chunk, hidden)`` survives into
    the backward, never ``(chunk, vocab)``.
    """
    student_logits = _distillation_project_logits(
        student_hidden_states, student_lm_head, student_lm_head_bias,
        student_logit_scale, student_final_logit_softcapping,
    )
    # The teacher is a fixed target: no autograd graph, and no teacher gradients
    # even when the caller forgot to freeze it.
    with torch.no_grad():
        teacher_logits = _distillation_project_logits(
            teacher_hidden_states, teacher_lm_head, teacher_lm_head_bias,
            teacher_logit_scale, teacher_final_logit_softcapping,
        )

    # Distillation temperature, applied after each model's own scaling and
    # softcapping. This is NOT the sampling temperature, and no T**2 gradient
    # rescale is applied: see the docstring of distillation_chunked_jsd.
    if temperature != 1.0:
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

    student_log_probs = torch.nn.functional.log_softmax(student_logits, dim = -1)
    teacher_log_probs = torch.nn.functional.log_softmax(teacher_logits, dim = -1)

    jsd = _distillation_generalized_jsd(student_log_probs, teacher_log_probs, beta)

    # The final chunk's tail holds positions packed out of the valid prefix.
    per_token_jsd = jsd.sum(dim = -1) * valid
    per_token_entropy = -(student_log_probs.exp() * student_log_probs).sum(dim = -1) * valid
    return per_token_jsd.sum(), per_token_entropy.sum()


def distillation_chunked_jsd(
    student_hidden_states,
    teacher_hidden_states,
    student_lm_head,
    teacher_lm_head,
    completion_mask,
    beta: float = 0.5,
    chunk_size: int = 256,
    num_items_in_batch = None,
    student_lm_head_bias = None,
    teacher_lm_head_bias = None,
    student_logit_scale: float = 1.0,
    teacher_logit_scale: float = 1.0,
    student_final_logit_softcapping: float = 0.0,
    teacher_final_logit_softcapping: float = 0.0,
    temperature: float = 1.0,
    use_checkpointing: bool = True,
):
    """Memory efficient generalized JSD between a student and a frozen teacher.

    Knowledge distillation compares two distributions over the whole vocabulary at
    every position, so the obvious implementation holds two ``(batch, seq, vocab)``
    logit tensors plus their float32 copies. At a 151936 token vocabulary that is
    about 1 GB per tensor for 1k positions, which is what makes distillation run
    out of memory long before the model does. Here the projections are done
    ``chunk_size`` positions at a time inside gradient checkpointing, so peak logit
    memory is ``2 * chunk_size * vocab`` rather than ``2 * batch * seq * vocab``
    and stops scaling with the batch at all.

    When the student's output head is frozen, which is the ordinary LoRA case,
    autograd allocates no dense ``(vocab, hidden)`` gradient for it either.

    Student and teacher must share a vocabulary but not a hidden width: each is
    projected through its own head.

    The objective is the one TRL's ``DistillationTrainer`` settled on: pure soft
    loss with no hard cross entropy term, no implicit ``T**2`` gradient rescale,
    and the mixture ``M = (1 - beta) * p_student + beta * p_teacher``. Other
    implementations differ on all three, so the tests check this against a dense
    reference rather than against any one of them.

    Args:
        student_hidden_states: ``(batch, seq, hidden)`` before the student's head.
        teacher_hidden_states: ``(batch, seq, hidden)`` for the same positions.
        student_lm_head: ``(vocab, hidden)`` student output head weight.
        teacher_lm_head: ``(vocab, hidden)`` teacher output head weight.
        completion_mask: ``(batch, seq)``, non zero where a position is trained on.
        beta: 0 forward KL, 1 reverse KL, in between generalized JSD.
        chunk_size: valid positions per chunk. Peak memory scales with this.
        num_items_in_batch: total valid tokens across the global batch. When given
            the reduction is ``sum / num_items_in_batch``, which is what makes
            gradient accumulation exact; when ``None`` it is the local mean.

    Returns:
        ``(loss, entropy_sum, n_valid_tokens)``. The last two are raw local sums so
        a distributed caller can reduce them itself.
    """
    flat_student = student_hidden_states.reshape(-1, student_hidden_states.shape[-1])
    flat_teacher = teacher_hidden_states.reshape(-1, teacher_hidden_states.shape[-1])
    valid = completion_mask.reshape(-1) != 0
    n_valid = valid.sum()

    # Pack the valid positions to the front so the masked ones form whole trailing
    # chunks that can be skipped. argsort on the mask is a static shape op, unlike
    # flat_student[valid], whose output shape is data dependent and upsets compile.
    order = valid.to(torch.int8).argsort(descending = True, stable = True)
    flat_student = flat_student[order]
    flat_teacher = flat_teacher[order]
    valid = valid[order]

    # At least one chunk always runs: a fully masked batch still has to reach every
    # trainable parameter, or backward and the gradient sync that follows it hang.
    n_padded = int(max(1, -(-int(n_valid) // chunk_size)) * chunk_size)

    loss = flat_student.new_zeros((), dtype = torch.float32)
    entropy_sum = flat_student.new_zeros((), dtype = torch.float32)

    for start in range(0, n_padded, chunk_size):
        stop = start + chunk_size
        arguments = (
            flat_student[start : stop], student_lm_head, student_lm_head_bias,
            student_logit_scale, student_final_logit_softcapping,
            flat_teacher[start : stop], teacher_lm_head, teacher_lm_head_bias,
            teacher_logit_scale, teacher_final_logit_softcapping,
            beta, temperature, valid[start : stop].float(),
        )
        # Recompute rather than keep each chunk's logits alive until the final
        # backward, which would put the whole sequence back in memory and undo the
        # point of chunking. Under no_grad there is nothing to recompute for.
        if use_checkpointing and torch.is_grad_enabled():
            chunk_loss, chunk_entropy = torch.utils.checkpoint.checkpoint(
                _distillation_jsd_chunk, *arguments, use_reentrant = False,
            )
        else:
            chunk_loss, chunk_entropy = _distillation_jsd_chunk(*arguments)
        loss = loss + chunk_loss
        entropy_sum = entropy_sum + chunk_entropy

    if num_items_in_batch is None:
        # Clamped for the same reason a chunk always runs: a fully masked batch has
        # to reduce to a finite zero rather than 0 / 0.
        loss = loss / n_valid.clamp(min = 1)
    else:
        if isinstance(num_items_in_batch, torch.Tensor):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / num_items_in_batch
    return loss, entropy_sum, n_valid
RL_REPLACEMENTS["distillation_chunked_jsd"] = distillation_chunked_jsd

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
