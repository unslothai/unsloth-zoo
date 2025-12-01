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
import numpy as np
from typing import Union, Callable, Optional, List, Dict
from .device_type import DEVICE_TYPE
from .temporary_patches.common import torch_compile_options
RL_REPLACEMENTS = dict()

# https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py#L1674
@torch.compile(dynamic = True, fullgraph = True, options = torch_compile_options,)
def selective_log_softmax(logits, index):
    logits = logits.to(torch.float32)
    selected_logits = torch.gather(logits, dim = -1, index = index.unsqueeze(-1)).squeeze(-1)
    # loop to reduce peak mem consumption
    # logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
    logsumexp_values = torch.logsumexp(logits, dim = -1)
    per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    return per_token_logps
pass

# More memory efficient by chunking on (bsz+qlen) dimension
# Exactly equivalent to the above
@torch.compile(dynamic = True, fullgraph = True, options = torch_compile_options,)
def chunked_selective_log_softmax(logits, index):
    # Split into 4 chunks only
    chunked_logits = torch.chunk(logits.reshape(-1, logits.shape[-1]), chunks = 4, dim = 0)
    chunked_index  = torch.chunk(index.reshape(-1), chunks = 4, dim = 0)
    all_per_token_logps = []
    # Below loop does the same as selective_log_softmax(chunk_logits, chunk_index)
    for chunk_logits, chunk_index in zip(chunked_logits, chunked_index):
        chunk_logits = chunk_logits.to(torch.float32)
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


def calculate_pad_tokens_in_prompt(
    input_ids: torch.Tensor,
    logits_to_keep: int,
    pad_token_id: int
) -> torch.Tensor:
    """
    Given prompt tensor, it returns all the left padded tokens in that sequence. so [pad, pad, pad, cat] = 3 tokens
    """
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
    """
    Given that we have a sequence, [p,p,p,c,c,c,pad,pad,pad]

    Where p are extra prompt tokens we got from slicing the torch tensor, c is completion tokens
    and pad are pad tokens, this function would make a completion mask that would 0 out the pad
    and p tokens. so in this example [0,0,0,1,1,1,0,0,0]
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


def left_pack_padding(tensor: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    Moves all padding tokens in each sequence of a batch to the right.
    """
    mask = (tensor != pad_id)
    # Must do stable=True since binary mark is unordered
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
    """
    Aligns a log probability tensor with a given attention mask.
    """

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

    # Create destination row indices
    # Shape: [batch_size, logprob_seq_len]
    row_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(dest_indices)

    # --- 4. Filter out-of-bounds indices and perform assignment ---
    # Create a mask to identify only the indices that are within the bounds
    # of the target tensor's sequence length.
    valid_mask = dest_indices < mask_seq_len

    # Use this mask to select only the valid row indices, column indices,
    # and the corresponding values from the logprob tensor.
    # This flattens the selected elements into 1D tensors.
    valid_rows = row_indices[valid_mask]
    valid_cols = dest_indices[valid_mask]
    valid_vals = logprob_tensor[valid_mask]

    # Place the valid values into their correct positions in the padded tensor
    # using a single, efficient advanced indexing operation.
    padded_logprobs[valid_rows, valid_cols] = valid_vals

    return padded_logprobs
pass
RL_REPLACEMENTS["align_logprobs_with_mask"] = align_logprobs_with_mask


def grpo_update_SamplingParams(SamplingParams, generation_kwargs, vllm_sampling_params = None):
    good_sampling_params_keys = inspect.signature(SamplingParams).parameters.keys()

    # Filter generation_kwargs
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


# Custom compiled GRPO loss - creates 3 Triton kernels
def grpo_compute_loss(
    ref_logits,
    new_logits,
    old_logits,
    sampling_per_token_logps,
    input_ids,
    mask,
    beta,
    advantages,
    **kwargs
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Set defaults for optional arguments
    loss_type = kwargs.get("loss_type", "grpo")
    epsilon_low = kwargs.get("epsilon_low", 0.2)
    epsilon_high = kwargs.get("epsilon_high", 0.2)
    max_completion_length = kwargs.get("max_completion_length", 8192)
    delta = kwargs.get("delta", None)
    temperature = kwargs.get("temperature", 1.0)
    logit_scale_multiply = kwargs.get("logit_scale_multiply", 0.0)
    logit_scale_divide   = kwargs.get("logit_scale_divide", 0.0)
    logit_softcapping    = kwargs.get("logit_softcapping", 0.0)
    importance_sampling_level = kwargs.get("importance_sampling_level", "token")
    num_items_in_batch = kwargs.get("num_items_in_batch", None)
    current_gradient_accumulation_steps = kwargs.get("current_gradient_accumulation_steps", 1)
    num_processes = kwargs.get("num_processes", 1)
    use_vllm = kwargs.get("use_vllm", False)
    vllm_importance_sampling_cap = kwargs.get("vllm_importance_sampling_cap", 2.0)
    input_ids = input_ids.unsqueeze(-1)

    # Optional logit softcapping and logit dividing
    if logit_scale_multiply != 0: new_logits = new_logits * logit_scale_multiply
    if logit_scale_divide   != 0: new_logits = new_logits / logit_scale_divide
    if logit_softcapping    != 0: new_logits = new_logits * torch.tanh(new_logits / logit_softcapping)

    new_logits = new_logits.to(torch.float32)
    # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
    if temperature != 1.0: new_logits = new_logits / temperature
    new_x = torch.gather(new_logits, dim = -1, index = input_ids).squeeze(-1)
    new = new_x - torch.logsumexp(new_logits, dim = -1)
    # x_i - logsumexp(x_i)
    with torch.no_grad():
        if beta != 0.0:
            assert ref_logits is not None, "ref_logits should not be None when beta != 0.0"

            # Optional logit softcapping and logit dividing
            if logit_scale_multiply != 0: ref_logits = ref_logits * logit_scale_multiply
            if logit_scale_divide   != 0: ref_logits = ref_logits / logit_scale_divide
            if logit_softcapping    != 0: ref_logits = ref_logits * torch.tanh(ref_logits / logit_softcapping)

            ref_logits = ref_logits.to(torch.float32)
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            if temperature != 1.0: ref_logits = ref_logits / temperature
            ref_x = torch.gather(ref_logits, dim = -1, index = input_ids).squeeze(-1)
            ref = ref_x - torch.logsumexp(ref_logits, dim = -1)
        pass

        if old_logits is not None:
            # Optional logit softcapping and logit dividing
            if logit_scale_multiply != 0: old_logits = old_logits * logit_scale_multiply
            if logit_scale_divide   != 0: old_logits = old_logits / logit_scale_divide
            if logit_softcapping    != 0: old_logits = old_logits * torch.tanh(old_logits / logit_softcapping)

            old_logits = old_logits.to(torch.float32)
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            if temperature != 1.0: old_logits = old_logits / temperature
            old_x = torch.gather(old_logits, dim = -1, index = input_ids).squeeze(-1)
            old = old_x - torch.logsumexp(old_logits, dim = -1)
        pass
        if use_vllm and sampling_per_token_logps is not None:
            #must filter out extra prompt tokens in begining after making input_ids left padded
            importance_sampling_ratio = torch.exp((old * mask) - sampling_per_token_logps)
            importance_sampling_ratio = torch.clamp(
                importance_sampling_ratio, max=vllm_importance_sampling_cap
            )
        pass
    pass

    # Reverse KL
    # Note that this is a low variance low bias estimator for the KL divergence as used in GRPO paper
    if beta != 0.0:
        kl_i = torch.exp(ref - new) - (ref - new) - 1.0

    else:
        # set kl_i to a tensor of zeros with the correct shape
        if importance_sampling_level == "sequence":
            kl_i = new.new_zeros(new.size(0), 1)
        else:
            kl_i = torch.zeros_like(new)
    # Full correct reverse KL divergence?? Missing term maybe?
    # kl_i = torch.exp(new) * kl_i

    # Below is forward KL (normal KL)
    # kl_i = torch.exp(old) * (old - new)
    if old_logits is not None:
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

    coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)

    if delta is not None:
        loss_1 = torch.clamp(coef_1, max=delta) * advantages.unsqueeze(1)
    else:
        loss_1 = coef_1 * advantages.unsqueeze(1)
    pass

    # Must detach - otherwise gradients are not propagated correctly!
    # exp(x - x) == 1
    # loss_i = torch.exp(new - new.detach()) * advantages.unsqueeze(1)

    loss_2 = coef_2 * advantages.unsqueeze(1)
    loss_i = -torch.min(loss_1, loss_2)

    if use_vllm and sampling_per_token_logps is not None:
        loss_i = loss_i * importance_sampling_ratio
        #delta for metric
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
    if loss_type == "grpo":
        loss = ((loss_i * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
        loss = loss / current_gradient_accumulation_steps
    elif loss_type == "bnpo":
        loss = (loss_i * mask).sum() / mask.sum().clamp(min=1.0)
        loss = loss / current_gradient_accumulation_steps
    elif loss_type == "dr_grpo":
        loss = (loss_i * mask).sum() / (loss_i.size(0) * max_completion_length)
        loss = loss / current_gradient_accumulation_steps
    elif loss_type == "dapo":
        normalizer = num_items_in_batch/ num_processes
        loss = (loss_i * mask).sum() / normalizer
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # loss = (loss_i * mask).sum() / mask.sum()

    # Get metrics as well which are folded
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
    return loss, completion_length, mean_kl, delta, flat_is_ratio
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
    # All Unsloth Zoo code licensed under LGPLv3
    @staticmethod
    def forward(ctx, _new_hidden_states, _old_hidden_states, _ref_hidden_states, _sampling_per_token_logps, lm_head, _input_ids, _mask, _advantages, beta, scaler = None, n_chunks = 1, extra_kwargs=None):
        if extra_kwargs is None:
            extra_kwargs = {}
        def compute_loss(new_hidden_states, old_hidden_states, ref_hidden_states, sampling_per_token_logps, input_ids, mask, advantages, scaling):
            new_logits = torch.matmul(new_hidden_states.to(lm_head.dtype), lm_head.t())
            new_logits = new_logits[:, :-1, :] # exclude the last logit: it corresponds to the next token pred
            with torch.no_grad():
                if beta != 0.0:
                    ref_logits = torch.matmul(ref_hidden_states.to(lm_head.dtype), lm_head.t())
                    ref_logits = ref_logits[:, :-1, :] # exclude the last logit: it corresponds to the next token pred
                else:
                    ref_logits = None
                if old_hidden_states is not None:
                    old_logits = torch.matmul(old_hidden_states.to(lm_head.dtype), lm_head.t())
                    old_logits = old_logits[:, :-1, :] # exclude the last logit: it corresponds to the next token pred
                else:
                    old_logits = None
            # if old_hidden_states is not None:
            #     old_logits = torch.matmul(old_hidden_states, lm_head.t()) #last logit already excluded
            #     old_logits = old_logits[:, :-1, :] # exclude the last logit: it corresponds to the next token pred
            # else:
            #     old_logits = None
            # unsloth_zoo/rl_replacements.py
            loss, completion_length, mean_kl, delta, flat_is_ratio = grpo_compute_loss(
                ref_logits,
                new_logits,
                old_logits,
                sampling_per_token_logps,
                input_ids,
                mask,
                beta,
                advantages,
                **extra_kwargs,
            )

            # Scale loss if needed for mixed precision training
            scaled_loss = loss * scaling
            # Must add .loss.detach otherwise autograd uses 2x VRAM
            return scaled_loss, (loss.detach(), completion_length, mean_kl, delta, flat_is_ratio)
        pass

        device =_new_hidden_states.device
        grad_inputs = torch.empty_like(_new_hidden_states)
        accumulated_loss              = torch.zeros(1, device = device)
        accumulated_completion_length = torch.zeros(1, device = device)
        accumulated_mean_kl           = torch.zeros(1, device = device)
        accumulated_delta             = []
        accumulated_flat_is_ratio     = []
        def accumulate_chunk(
            new_hidden_states_j,
            old_hidden_states_j,
            ref_hidden_states_j,
            sampling_per_token_logps_j,
            input_ids_j,
            mask_j,
            advantages_j,
            scaling,
            grad_inputs_j,
        ):
            (chunk_grad_input,), (chunk_loss, (unscaled_loss, chunk_completion_length, chunk_mean_kl, chunk_delta, chunk_flat_is_ratio)) = torch.func.grad_and_value(
                compute_loss,
                argnums = (0,),
                has_aux = True,
            )(new_hidden_states_j, old_hidden_states_j, ref_hidden_states_j, sampling_per_token_logps_j, input_ids_j, mask_j, advantages_j, scaling)
            accumulated_loss             .add_(unscaled_loss)
            accumulated_completion_length.add_(chunk_completion_length)
            accumulated_mean_kl          .add_(chunk_mean_kl)
            accumulated_delta            .append(chunk_delta)
            accumulated_flat_is_ratio    .append(chunk_flat_is_ratio)
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
        new_hidden_states  = torch.chunk(_new_hidden_states, chunks = n_chunks, dim = 0)
        if _old_hidden_states is not None:
            old_hidden_states  = torch.chunk(_old_hidden_states, chunks = n_chunks, dim = 0)
        else:
            old_hidden_states = [None] * n_chunks
        if _ref_hidden_states is not None:
            ref_hidden_states  = torch.chunk(_ref_hidden_states, chunks = n_chunks, dim = 0)
        else:
            ref_hidden_states = [None] * n_chunks
        if _sampling_per_token_logps is not None:
            sampling_per_token_logps  = torch.chunk(_sampling_per_token_logps, chunks = n_chunks, dim = 0)
        else:
            sampling_per_token_logps = [None] * n_chunks
        input_ids          = torch.chunk(_input_ids,         chunks = n_chunks, dim = 0)
        mask               = torch.chunk(_mask,              chunks = n_chunks, dim = 0)
        advantages         = torch.chunk(_advantages,        chunks = n_chunks, dim = 0)

        # Get mixed precision scaling if seen
        scaling = scaler.get_scale() if scaler is not None else 1.0

        # Force torch.compile to use dynamic shapes for seqlen dim
        # mark_dynamic = lambda x: torch._dynamo.mark_dynamic(x, 1)

        for (grad_inputs_j, new_hidden_states_j, old_hidden_states_j, ref_hidden_states_j, sampling_per_token_logps_j, input_ids_j, mask_j, advantages_j, ) in \
            zip(grad_inputs_chunks, new_hidden_states, old_hidden_states, ref_hidden_states, sampling_per_token_logps, input_ids, mask, advantages):

            # [TODO] Dynamic marking causes torch.compile errors if sequence length is long

            # mark_dynamic(new_hidden_states_j)
            # mark_dynamic(ref_hidden_states_j)
            # if old_hidden_states_j is not None:
            #     mark_dynamic(old_hidden_states_j)
            # mark_dynamic(input_ids_j)
            # mark_dynamic(mask_j)

            accumulate_chunk(
                new_hidden_states_j,
                old_hidden_states_j,
                ref_hidden_states_j,
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
        ctx.save_for_backward(grad_inputs)
        return (
            accumulated_loss,
            accumulated_completion_length,
            accumulated_mean_kl,
            accumulated_delta,
            accumulated_flat_is_ratio
        )
    pass

    @staticmethod
    def backward(ctx, grad_output, dcompletion_length, dmean_kl, ddelta, ddflat_is_ratio):
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
    old_hidden_states,
    ref_hidden_states,
    n_chunks = -1,
    **kwargs,
):
    # All Unsloth Zoo code licensed under LGPLv3
    bsz, qlen = input_ids.shape

    pixel_values = kwargs.get('pixel_values',None)
    image_grid_thw = kwargs.get('image_grid_thw',None)
    pixel_attention_mask = kwargs.get('pixel_attention_mask',None)
    image_sizes = kwargs.get('image_sizes',None)
    #delete this from kwargs so less issues
    sampling_per_token_logps = kwargs.pop("sampling_per_token_logps", None)
    kwargs["vllm_importance_sampling_cap"] = trainer.vllm_importance_sampling_cap if sampling_per_token_logps is not None else None
    kwargs["use_vllm"] = trainer.use_vllm
    # Find closest multiple
    factors = [i for i in range(1, bsz + 1) if bsz % i == 0]
    if n_chunks == -1: n_chunks = bsz
    n_chunks = factors[min(np.searchsorted(factors, n_chunks), len(factors)-1)]

    if not hasattr(trainer, '_autocast_dtype'):
        trainer._autocast_dtype = torch.float16 if os.environ.get('ACCELERATE_MIXED_PRECISION', 'fp16') == 'fp16' else torch.bfloat16
        if os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '1': trainer._autocast_dtype = None
    pass
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"

    lm_head = trainer.model.get_output_embeddings().weight

    if pixel_values is None:
        left_pad_tokens_per_prompt = calculate_pad_tokens_in_prompt(input_ids, logits_to_keep, trainer.processing_class.pad_token_id)

        max_left_pad = max(left_pad_tokens_per_prompt).item()

        input_ids = left_pack_padding(input_ids, trainer.processing_class.pad_token_id)

        completion_input_ids = input_ids[:, -(logits_to_keep +max_left_pad):]

        completion_mask = create_completion_attention_mask(completion_input_ids, left_pad_tokens_per_prompt, max_left_pad, trainer.processing_class.pad_token_id).to(attention_mask.dtype)
        #TODO given the completion mask here we need to, handle the left pad tokens so the sizes of completion
        #token or old logprobs are compatible with the importance sampling logprobs
        if trainer.use_vllm and sampling_per_token_logps is not None:
            sampling_per_token_logps = align_logprobs_with_mask(sampling_per_token_logps, completion_mask)
        attention_mask =  input_ids != trainer.processing_class.pad_token_id
        attention_mask = attention_mask.to(attention_mask.dtype)
    else:
        completion_input_ids = input_ids[:, -logits_to_keep:]

    unwrapped_model = trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper = False)

    # Do not move hidden_states from device 1 to device 0:
    for module in unwrapped_model.modules():
        if hasattr(module, "_hf_hook") and hasattr(module._hf_hook, "io_same_decice"):
            module._hf_hook.io_same_decice = False
    pass


    # Get autocaster
    if trainer._autocast_dtype is None:
        autocaster = nullcontext()
    else:
        autocaster = torch.amp.autocast(device_type = trainer.model.device.type, dtype = trainer._autocast_dtype)
    with autocaster:
        if pixel_values is None:
            new_hidden_states = unwrapped_model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                pixel_values = pixel_values,
                image_grid_thw = image_grid_thw,
                pixel_attention_mask = pixel_attention_mask,
                image_sizes = image_sizes,
                # logits_to_keep = logits_to_keep + 1,
            ).logits

            #keep extra logit as we generated a new token
            new_hidden_states = new_hidden_states[:, -(logits_to_keep +max_left_pad+1): , :]
            if ref_hidden_states is not None:
                ref_hidden_states = ref_hidden_states[:, -(logits_to_keep +max_left_pad+1): , :]
            if old_hidden_states is not None:
                old_hidden_states = old_hidden_states[:, -(logits_to_keep +max_left_pad+1): , :]
        else:
            new_hidden_states = unwrapped_model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                pixel_values = pixel_values,
                image_grid_thw = image_grid_thw,
                pixel_attention_mask = pixel_attention_mask,
                image_sizes = image_sizes,
                logits_to_keep = logits_to_keep + 1,
            ).logits
    loss, completion_length, mean_kl, delta, flat_is_ratio = UnslothEfficientGRPO.apply(
        new_hidden_states,
        old_hidden_states,
        ref_hidden_states,
        sampling_per_token_logps,
        lm_head,
        completion_input_ids,
        completion_mask,
        advantages,
        trainer.beta,
        trainer.accelerator.scaler,
        n_chunks,
        kwargs # pass kwargs as a dict
    )


    # Must force not returning hidden states but logits otherwise gibberish
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"

    return loss, completion_length, mean_kl, delta, flat_is_ratio
    # Old non efficient code path
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
