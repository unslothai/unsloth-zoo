"""
2026.4.7
2026.4.5
5.5.4
1.1.0
__UNSLOTH_VERSIONING__
"""

# Unsloth auto generated code
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

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import functional as F
from unsloth_zoo.temporary_patches.common import torch_compile
from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
from trl.trainer.dpo_trainer import (Any, AutoProcessor, Callable, DPOConfig, DPOTrainer, DataCollator, DataCollatorForPreference, DataCollatorForVisionPreference, DataLoader, Dataset, EvalPrediction, F, Hasher, IterableDataset, IterableDatasetDict, PartialState, Path, PeftConfig, PeftModel, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, SyncRefModelCallback, TrainerCallback, Version, _BaseTrainer, apply_chat_template, contextlib, create_model_from_path, dataclass, defaultdict, disable_dropout_in_model, disable_gradient_checkpointing, entropy_from_logits, extract_prompt, get_act_offloading_ctx_manager, get_config_model_id, get_peft_model, hash_module, is_conversational, is_liger_kernel_available, is_peft_available, is_peft_model, json, logger, np, os, pad, prepare_deepspeed, prepare_fsdp, prepare_multimodal_messages, selective_log_softmax, textwrap, torch, tqdm, transformers, use_adapter, AutoProcessor, Callable, DPOConfig, DPOTrainer, DataCollator, DataCollatorForPreference, DataCollatorForVisionPreference, Dataset, EvalPrediction, F, IterableDataset, IterableDatasetDict, PeftConfig, PeftModel, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, SyncRefModelCallback, TrainerCallback, Version, contextlib, create_model_from_path, defaultdict, disable_dropout_in_model, get_act_offloading_ctx_manager, get_config_model_id, get_peft_model, is_liger_kernel_available, is_peft_available, is_peft_model, logger, np, os, pad, prepare_deepspeed, prepare_fsdp, torch, transformers, F, PeftModel, PreTrainedModel, is_peft_available, logger, os, torch)


import os
import math
import logging
from typing import *
from dataclasses import dataclass, field
from packaging.version import Version
import torch
import numpy as np
from contextlib import nullcontext
from torch.nn import functional as F
import inspect
from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling as TransformersDataCollatorForLanguageModeling
from transformers.training_args import ParallelMode
from unsloth_zoo.device_type import DEVICE_TYPE, device_synchronize

# Wrap trainer with padding to right and enable training mode
import functools
from types import MethodType
try:
    from unsloth_zoo.gradient_checkpointing import reset_unsloth_gradient_checkpointing_buffers
except:
    def reset_unsloth_gradient_checkpointing_buffers(): pass
def prepare_for_training_mode(f):
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        # Finish the previous W&B run if this is a subsequent train() call.
        # We do this at the START of train() (not the end) so that
        # evaluate() / log() still work after train() completes.
        # HF's WandbCallback.setup() will call wandb.init() for the new run.
        # See: https://github.com/unslothai/unsloth/issues/3954
        if getattr(self, '_unsloth_training_completed', False):
            try:
                import wandb
                if wandb.run is not None:
                    wandb.finish()
                    # Reset HF's WandbCallback so it calls wandb.init() for the new run
                    for cb in self.callback_handler.callbacks:
                        if type(cb).__name__ == 'WandbCallback':
                            cb._initialized = False
                            break
            except:
                pass
        # Enable training mode
        _was_training = None
        # Get gradient checkpointing setting from training arguments
        use_gc = getattr(self.args, 'gradient_checkpointing', True)
        if hasattr(self, 'model') and hasattr(self.model, "training"):
            _was_training = self.model.training
        if hasattr(self, 'model') and hasattr(self.model, "for_training"):
            self.model.for_training(use_gradient_checkpointing=use_gc)
        output = f(self, *args, **kwargs)
        # Restore previous mode when possible
        if hasattr(self, 'model') and hasattr(self.model, "for_inference"):
            if _was_training is False:
                self.model.for_inference()
            elif _was_training is True and hasattr(self.model, "for_training"):
                self.model.for_training(use_gradient_checkpointing=use_gc)
        # Reset gradient checkpointing buffers to free memory while staying ready for next run
        try:
            reset_unsloth_gradient_checkpointing_buffers()
        except:
            pass
        # Mark that training completed so the next train() call can
        # finish this W&B run before starting a new one
        self._unsloth_training_completed = True
        return output
    return wrapper
pass

torch_compile_options = {
    "epilogue_fusion"   : True,
    "max_autotune"      : False,
    "shape_padding"     : True,
    "trace.enabled"     : False,
    "triton.cudagraphs" : False,
}

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
    logit_matmul_upcast: bool = False,
) -> torch.Tensor:
    # All Unsloth Zoo code licensed under AGPL3
    flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_index = index.reshape(-1)

    chunked_hidden_states = torch.chunk(flat_hidden_states, chunks=chunks, dim=0)
    chunked_index = torch.chunk(flat_index, chunks=chunks, dim=0)

    all_per_token_logps = []

    for chunk_hidden_states, chunk_index in zip(chunked_hidden_states, chunked_index):
        # logit_matmul_upcast: use float32 for the logit matmul to prevent
        # fp16 overflow on models like Gemma-4 whose hidden states can be large.
        if logit_matmul_upcast:
            chunk_logits = chunk_hidden_states.float() @ lm_head.float().t()
        else:
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

@torch.compile(dynamic = True, fullgraph = True, options = torch_compile_options,)
def chunked_selective_log_softmax(logits, index, temperature: float = 1.0):
    # Split into 4 chunks only
    chunked_logits = torch.chunk(logits.reshape(-1, logits.shape[-1]), chunks = 4, dim = 0)
    chunked_index  = torch.chunk(index.reshape(-1), chunks = 4, dim = 0)
    all_per_token_logps = []
    # Below loop does the same as selective_log_softmax(chunk_logits, chunk_index)
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

def left_pack_padding(tensor: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    Moves all padding tokens in each sequence of a batch to the right.
    """
    mask = (tensor != pad_id)
    # Must do stable=True since binary mark is unordered
    sorted_indices = torch.argsort(mask, dim=1, descending=True, stable=True)
    packed_tensor = torch.gather(tensor, 1, sorted_indices)
    return packed_tensor

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
        # For XPU: estimate free memory from total - reserved
        total_mem = torch.xpu.get_device_properties(0).total_memory
        reserved_mem = torch.xpu.memory_reserved()
        free_bytes = total_mem - reserved_mem
        limit_gb = (free_bytes / (1024**3)) * 0.80
    else:
        # Fallback: assume 8GB available
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
@dataclass
class UnslothDPOConfig(DPOConfig):
    """
    
Configuration class for the [`DPOTrainer`].

This class includes only the parameters that are specific to DPO training. For a full list of training arguments,
please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this class may
differ from those in [`~transformers.TrainingArguments`].

Using [`~transformers.HfArgumentParser`] we can turn this class into
[argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
command line.

Parameters:
    > Parameters that control the model

    model_init_kwargs (`dict[str, Any]`, *optional*):
        Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
        argument of the [`DPOTrainer`] is provided as a string.
    disable_dropout (`bool`, *optional*, defaults to `True`):
        Whether to disable dropout in the model and reference model.

    > Parameters that control the data preprocessing

    dataset_num_proc (`int`, *optional*):
        Number of processes to use for processing the dataset.
    max_length (`int` or `None`, *optional*, defaults to `1024`):
        Maximum length of the tokenized sequence. Sequences longer than `max_length` are truncated from the left or
        right depending on the `truncation_mode`. If `None`, no truncation is applied.
    truncation_mode (`str`, *optional*, defaults to `"keep_start"`):
        Truncation mode to use when the sequence exceeds `max_length`. The only supported value is
        `"keep_start"`. The `"keep_end"` value is deprecated and will be removed in v2.0.0.
    padding_free (`bool`, *optional*, defaults to `False`):
        Whether to perform forward passes without padding by flattening all sequences in the batch into a single
        continuous sequence. This reduces memory usage by eliminating padding overhead. Currently, this is only
        supported with the FlashAttention 2 or 3, which can efficiently handle the flattened batch structure.
    pad_to_multiple_of (`int`, *optional*):
        If set, the sequences will be padded to a multiple of this value.
    precompute_ref_log_probs (`bool`, *optional*, defaults to `False`):
        Whether to precompute the reference model log probabilities for the entire training dataset before
        training. This allows to save memory during training, as the reference model does not need to be kept in
        memory.
    precompute_ref_batch_size (`int`, *optional*):
        Batch size to use when precomputing reference model log probabilities. This can be set higher than the
        training batch size to speed up preprocessing. If `None`, defaults to `per_device_train_batch_size` for
        training and `per_device_eval_batch_size` for evaluation.

    > Parameters that control the training

    loss_type (`list[str]`, *optional*, defaults to `["sigmoid"]`):
        Type of loss to use. Possible values are: `'sigmoid'`, `'hinge'`, `'ipo'`, `'exo_pair'`, `'nca_pair'`,
        `'robust'`, `'bco_pair'`, `'sppo_hard'`, `'aot'`, `'aot_unpaired'`, `'apo_zero'`, `'apo_down'`,
        `'discopop'`, `'sft'`. If multiple loss types are provided, they will be combined using the weights
        specified in `loss_weights`.
    loss_weights (`list[float]`, *optional*):
        List of loss weights for multi-loss combinations. Used when combining multiple loss types. Example: `[0.8,
        0.2, 1.0]` for MPO. If not provided, defaults to equal weights (`1.0`) for all loss types.
    ld_alpha (`float`, *optional*):
        α parameter from the LD-DPO paper, which controls the weighting of the verbose token log-probabilities in
        responses. If `None`, no weighting is applied to the verbose part, and the loss is equivalent to the
        standard DPO loss. Must be in [0.0, 1.0]: `ld_alpha=1.0` applies no weighting, and `ld_alpha=0.0` masks
        tokens beyond shared lengths.
    f_divergence_type (`str`, *optional*, defaults to `"reverse_kl"`):
        f-divergence regularizer between policy and reference (f-DPO paper). Possible values are: `reverse_kl`
        (default), `forward_kl`, `js_divergence`, `alpha_divergence`.
    f_alpha_divergence_coef (`float`, *optional*, defaults to `0.5`):
        α coefficient for the α-divergence u^-α regularizer, used only when `f_divergence_type='alpha_divergence'`.
    label_smoothing (`float`, *optional*, defaults to `0.0`):
        Label smoothing parameter used in Robust DPO and EXO. In Robust DPO, it is interpreted as the probability
        that a preference label is flipped and must lie in [0.0, 0.5); a typical value recommended by the Robust
        DPO paper is 0.1. In EXO, it corresponds to the ε label smoothing parameter, for which the paper recommends
        a typical value of 1e-3.
    beta (`float`, *optional*, defaults to `0.1`):
        Parameter controlling the deviation from the reference model. Higher β means less deviation from the
        reference model. For the IPO loss (`loss_type='ipo'`), this value is the regularization parameter denoted
        by τ in the [paper](https://huggingface.co/papers/2310.12036).
    use_weighting (`bool`, *optional*, defaults to `False`):
        Whether to apply WPO-style weighting (https://huggingface.co/papers/2406.11827) to preference pairs using
        the policy's length-normalized sequence probabilities.
    discopop_tau (`float`, *optional*, defaults to `0.05`):
        τ/temperature parameter from the DiscoPOP paper, which controls the shape of the log-ratio modulated loss
        when using `loss_type='discopop'`. The paper recommends the default value `discopop_tau=0.05`.
    activation_offloading (`bool`, *optional*, defaults to `False`):
        Whether to offload the activations to the CPU.
    sync_ref_model (`bool`, *optional*, defaults to `False`):
        Whether to synchronize the reference model with the active model every `ref_model_sync_steps` steps, using
        the `ref_model_mixup_alpha` parameter. This synchronization originates from the
        [TR-DPO](https://huggingface.co/papers/2404.09656) paper. `sync_ref_model=True` is not yet compatible with
        PEFT or `precompute_ref_log_probs=True`.
    ref_model_mixup_alpha (`float`, *optional*, defaults to `0.6`):
        α parameter from the TR-DPO paper, which controls the mix between the current policy and the previous
        reference policy during updates. The reference policy is updated according to the equation: `π_ref = α *
        π_θ + (1 - α) * π_ref_prev`. To use this parameter, you must set `sync_ref_model=True`.
    ref_model_sync_steps (`int`, *optional*, defaults to `512`):
        τ parameter from the TR-DPO paper, which determines how frequently the current policy is synchronized with
        the reference policy. To use this parameter, you must set `sync_ref_model=True`.

    > Deprecated parameters

    pad_token:

        <Deprecated version="1.1.0">

        Parameter `pad_token` is deprecated and will be removed in version v2.0.0. Set `tokenizer.pad_token`
        directly and pass it as `processing_class` to the trainer instead.

        </Deprecated>

> [!NOTE]
> These parameters have default values different from [`~transformers.TrainingArguments`]:
> - `logging_steps`: Defaults to `10` instead of `500`.
> - `gradient_checkpointing`: Defaults to `True` instead of `False`.
> - `bf16`: Defaults to `True` if `fp16` is not set, instead of `False`.
> - `learning_rate`: Defaults to `1e-6` instead of `5e-5`.

    """
    vllm_sampling_params: Optional[Any] = field(
        default = None,
        metadata = {'help': 'vLLM SamplingParams'},
    )
    unsloth_num_chunks : Optional[int] = field(
        default = -1,
        metadata = {'help': 'Chunk size to reduce memory usage. -1 is most efficient.'},
    )
    unsloth_logit_chunk_multiplier : Optional[int] = field(
            default = None,
            metadata = {'help': 'Multiplier for chunked logit computations.'},
        )
    unsloth_grpo_mini_batch : Optional[int] = field(
        default = None,
        metadata = {'help': 'Mini batch size for GRPO hidden state accumulation. Default is None unless user defines it.'},
    )
    max_seq_length : Optional[int] = field(
        default = None,
        metadata = {'help': 'Maximum sequence length to truncate to.'},
    )
    def __init__(
        self,
        output_dir = None,
        per_device_train_batch_size = 4,
        num_train_epochs = 3.0,
        max_steps = -1,
        learning_rate = 5e-05,
        lr_scheduler_type = 'linear',
        lr_scheduler_kwargs = None,
        warmup_steps = 0.1,
        optim = 'adamw_8bit',
        optim_args = None,
        weight_decay = 0.01,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_epsilon = 1e-08,
        optim_target_modules = None,
        gradient_accumulation_steps = 2,
        average_tokens_across_devices = True,
        max_grad_norm = 1.0,
        label_smoothing_factor = 0.0,
        bf16 = False,
        fp16 = False,
        bf16_full_eval = False,
        fp16_full_eval = False,
        tf32 = None,
        gradient_checkpointing = True,
        gradient_checkpointing_kwargs = None,
        torch_compile = False,
        torch_compile_backend = None,
        torch_compile_mode = None,
        use_liger_kernel = False,
        liger_kernel_config = None,
        use_cache = False,
        neftune_noise_alpha = None,
        torch_empty_cache_steps = 250,
        auto_find_batch_size = False,
        logging_strategy = 'steps',
        logging_steps = 1,
        logging_first_step = False,
        log_on_each_node = True,
        logging_nan_inf_filter = False,
        include_num_input_tokens_seen = False,
        log_level = 'passive',
        log_level_replica = 'warning',
        disable_tqdm = None,
        report_to = 'none',
        run_name = None,
        project = 'huggingface',
        trackio_space_id = 'trackio',
        eval_strategy = 'no',
        eval_steps = None,
        eval_delay = 0,
        per_device_eval_batch_size = 4,
        prediction_loss_only = False,
        eval_on_start = False,
        eval_do_concat_batches = True,
        eval_use_gather_object = False,
        eval_accumulation_steps = 2,
        batch_eval_metrics = False,
        save_only_model = False,
        save_strategy = 'steps',
        save_steps = 500,
        save_on_each_node = False,
        save_total_limit = None,
        enable_jit_checkpoint = False,
        push_to_hub = False,
        hub_token = None,
        hub_private_repo = None,
        hub_model_id = None,
        hub_strategy = 'every_save',
        hub_always_push = False,
        hub_revision = None,
        load_best_model_at_end = False,
        metric_for_best_model = None,
        greater_is_better = None,
        ignore_data_skip = False,
        restore_callback_states_from_checkpoint = False,
        full_determinism = False,
        seed = 3407,
        data_seed = 3407,
        use_cpu = False,
        accelerator_config = None,
        parallelism_config = None,
        dataloader_drop_last = False,
        dataloader_num_workers = 0,
        dataloader_pin_memory = True,
        dataloader_persistent_workers = False,
        dataloader_prefetch_factor = None,
        remove_unused_columns = True,
        label_names = None,
        train_sampling_strategy = 'random',
        length_column_name = 'length',
        ddp_find_unused_parameters = None,
        ddp_bucket_cap_mb = None,
        ddp_broadcast_buffers = None,
        ddp_backend = None,
        ddp_timeout = 1800,
        fsdp = None,
        fsdp_config = None,
        deepspeed = None,
        debug = '',
        skip_memory_metrics = True,
        do_train = False,
        do_eval = False,
        do_predict = False,
        resume_from_checkpoint = None,
        warmup_ratio = None,
        logging_dir = None,
        local_rank = -1,
        model_init_kwargs = None,
        disable_dropout = True,
        dataset_num_proc = None,
        max_length = 1024,
        truncation_mode = 'keep_start',
        padding_free = None,
        pad_to_multiple_of = None,
        precompute_ref_log_probs = False,
        precompute_ref_batch_size = None,
        loss_weights = None,
        ld_alpha = None,
        f_divergence_type = 'reverse_kl',
        f_alpha_divergence_coef = 0.5,
        label_smoothing = 0.0,
        beta = 0.1,
        use_weighting = False,
        discopop_tau = 0.05,
        activation_offloading = False,
        sync_ref_model = False,
        ref_model_mixup_alpha = 0.6,
        ref_model_sync_steps = 512,
        pad_token = None,
        vllm_sampling_params = None,
        unsloth_num_chunks = -1,
        unsloth_logit_chunk_multiplier = None,
        unsloth_grpo_mini_batch = None,
        max_seq_length = None,
        **kwargs,
    ):
        if learning_rate < 1e-7: print(f'Unsloth: Your learning rate of `{learning_rate}` is too small and less than 1e-7! Consider increasing it, otherwise gradient updates will be close to 0!')
        if learning_rate > 1: print(f'Unsloth: Your learning rate of `{learning_rate}` is way too larger > 1! Consider decreasing it to 1e-1, otherwise gradient updates will explode!')
        if num_train_epochs is None:
            num_train_epochs = 3.0  # Default to 3 epochs if None, max_steps will override
        if output_dir is None and save_strategy == 'steps' and save_steps == 500:
            output_dir = 'unsloth_training_checkpoints'
            save_strategy = 'no'
        import multiprocessing as _mp
        if dataset_num_proc is None:
            if _mp.get_start_method() != 'fork':
                dataset_num_proc = None
            else:
                import psutil
                dataset_num_proc = min(max((psutil.cpu_count() or 1)+4, 2), 64)
                memory_gb_left = psutil.virtual_memory().available / (1024**3)
                if memory_gb_left <= 2: dataset_num_proc = 1
                else: dataset_num_proc = min(dataset_num_proc, int(memory_gb_left))
        if os.environ.get('UNSLOTH_ENABLE_FLEX_ATTENTION', '0') == '1':
            from unsloth_zoo.flex_attention import HAS_FLEX_ATTENTION
            if HAS_FLEX_ATTENTION and pad_to_multiple_of is None:
                from unsloth_zoo.flex_attention import FLEX_ATTENTION_BLOCK_SIZE
                pad_to_multiple_of = FLEX_ATTENTION_BLOCK_SIZE
        
        
        super().__init__(
            output_dir = output_dir,
            per_device_train_batch_size = per_device_train_batch_size,
            num_train_epochs = num_train_epochs,
            max_steps = max_steps,
            learning_rate = learning_rate,
            lr_scheduler_type = lr_scheduler_type,
            lr_scheduler_kwargs = lr_scheduler_kwargs,
            warmup_steps = warmup_steps,
            optim = optim,
            optim_args = optim_args,
            weight_decay = weight_decay,
            adam_beta1 = adam_beta1,
            adam_beta2 = adam_beta2,
            adam_epsilon = adam_epsilon,
            optim_target_modules = optim_target_modules,
            gradient_accumulation_steps = gradient_accumulation_steps,
            average_tokens_across_devices = average_tokens_across_devices,
            max_grad_norm = max_grad_norm,
            label_smoothing_factor = label_smoothing_factor,
            bf16 = bf16,
            fp16 = fp16,
            bf16_full_eval = bf16_full_eval,
            fp16_full_eval = fp16_full_eval,
            tf32 = tf32,
            gradient_checkpointing = gradient_checkpointing,
            gradient_checkpointing_kwargs = gradient_checkpointing_kwargs,
            torch_compile = torch_compile,
            torch_compile_backend = torch_compile_backend,
            torch_compile_mode = torch_compile_mode,
            use_liger_kernel = use_liger_kernel,
            liger_kernel_config = liger_kernel_config,
            use_cache = use_cache,
            neftune_noise_alpha = neftune_noise_alpha,
            torch_empty_cache_steps = torch_empty_cache_steps,
            auto_find_batch_size = auto_find_batch_size,
            logging_strategy = logging_strategy,
            logging_steps = logging_steps,
            logging_first_step = logging_first_step,
            log_on_each_node = log_on_each_node,
            logging_nan_inf_filter = logging_nan_inf_filter,
            include_num_input_tokens_seen = include_num_input_tokens_seen,
            log_level = log_level,
            log_level_replica = log_level_replica,
            disable_tqdm = disable_tqdm,
            report_to = report_to,
            run_name = run_name,
            project = project,
            trackio_space_id = trackio_space_id,
            eval_strategy = eval_strategy,
            eval_steps = eval_steps,
            eval_delay = eval_delay,
            per_device_eval_batch_size = per_device_eval_batch_size,
            prediction_loss_only = prediction_loss_only,
            eval_on_start = eval_on_start,
            eval_do_concat_batches = eval_do_concat_batches,
            eval_use_gather_object = eval_use_gather_object,
            eval_accumulation_steps = eval_accumulation_steps,
            batch_eval_metrics = batch_eval_metrics,
            save_only_model = save_only_model,
            save_strategy = save_strategy,
            save_steps = save_steps,
            save_on_each_node = save_on_each_node,
            save_total_limit = save_total_limit,
            enable_jit_checkpoint = enable_jit_checkpoint,
            push_to_hub = push_to_hub,
            hub_token = hub_token,
            hub_private_repo = hub_private_repo,
            hub_model_id = hub_model_id,
            hub_strategy = hub_strategy,
            hub_always_push = hub_always_push,
            hub_revision = hub_revision,
            load_best_model_at_end = load_best_model_at_end,
            metric_for_best_model = metric_for_best_model,
            greater_is_better = greater_is_better,
            ignore_data_skip = ignore_data_skip,
            restore_callback_states_from_checkpoint = restore_callback_states_from_checkpoint,
            full_determinism = full_determinism,
            seed = seed,
            data_seed = data_seed,
            use_cpu = use_cpu,
            accelerator_config = accelerator_config,
            parallelism_config = parallelism_config,
            dataloader_drop_last = dataloader_drop_last,
            dataloader_num_workers = dataloader_num_workers,
            dataloader_pin_memory = dataloader_pin_memory,
            dataloader_persistent_workers = dataloader_persistent_workers,
            dataloader_prefetch_factor = dataloader_prefetch_factor,
            remove_unused_columns = remove_unused_columns,
            label_names = label_names,
            train_sampling_strategy = train_sampling_strategy,
            length_column_name = length_column_name,
            ddp_find_unused_parameters = ddp_find_unused_parameters,
            ddp_bucket_cap_mb = ddp_bucket_cap_mb,
            ddp_broadcast_buffers = ddp_broadcast_buffers,
            ddp_backend = ddp_backend,
            ddp_timeout = ddp_timeout,
            fsdp = fsdp,
            fsdp_config = fsdp_config,
            deepspeed = deepspeed,
            debug = debug,
            skip_memory_metrics = skip_memory_metrics,
            do_train = do_train,
            do_eval = do_eval,
            do_predict = do_predict,
            resume_from_checkpoint = resume_from_checkpoint,
            warmup_ratio = warmup_ratio,
            logging_dir = logging_dir,
            local_rank = local_rank,
            model_init_kwargs = model_init_kwargs,
            disable_dropout = disable_dropout,
            dataset_num_proc = dataset_num_proc,
            max_length = max_length,
            truncation_mode = truncation_mode,
            padding_free = padding_free,
            pad_to_multiple_of = pad_to_multiple_of,
            precompute_ref_log_probs = precompute_ref_log_probs,
            precompute_ref_batch_size = precompute_ref_batch_size,
            loss_weights = loss_weights,
            ld_alpha = ld_alpha,
            f_divergence_type = f_divergence_type,
            f_alpha_divergence_coef = f_alpha_divergence_coef,
            label_smoothing = label_smoothing,
            beta = beta,
            use_weighting = use_weighting,
            discopop_tau = discopop_tau,
            activation_offloading = activation_offloading,
            sync_ref_model = sync_ref_model,
            ref_model_mixup_alpha = ref_model_mixup_alpha,
            ref_model_sync_steps = ref_model_sync_steps,
            pad_token = pad_token,**kwargs)
        self.vllm_sampling_params = vllm_sampling_params
        self.unsloth_num_chunks = unsloth_num_chunks
        if unsloth_grpo_mini_batch is not None:
            if self.generation_batch_size >= unsloth_grpo_mini_batch:
                self.unsloth_grpo_mini_batch = unsloth_grpo_mini_batch
            else:
                raise ValueError(
                    f"Unsloth GRPO mini batch size needs to be less than or equal to the effective generation batch size, "
                    f"which is self.per_device_train_batch_size * gradient_accumulation_steps."
                )
        self.unsloth_logit_chunk_multiplier = unsloth_logit_chunk_multiplier
        self.max_seq_length = max_seq_length
        # Unsloth: Remove use_reentrant=False forced by TRL 0.27.0+
        if getattr(self, 'gradient_checkpointing_kwargs', None) is not None:
            if 'use_reentrant' in self.gradient_checkpointing_kwargs:
                del self.gradient_checkpointing_kwargs['use_reentrant']

pass

class _UnslothDPOTrainer(_BaseTrainer):
    """
    Trainer for Direct Preference Optimization (DPO) method. This algorithm was initially proposed in the paper [Direct
    Preference Optimization: Your Language Model is Secretly a Reward Model](https://huggingface.co/papers/2305.18290).
    This class is a wrapper around the [`~transformers.Trainer`] class and inherits all of its attributes and methods.

    Example:

    ```python
    from trl import DPOTrainer
    from datasets import load_dataset

    dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

    trainer = DPOTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        train_dataset=dataset,
    )
    trainer.train()
    ```

    Args:
        model (`str` or [`~transformers.PreTrainedModel`] or [`~peft.PeftModel`]):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using `<ModelArchitecture>.from_pretrained` (where `<ModelArchitecture>` is derived from the model
              config) with the keyword arguments in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
            - A [`~peft.PeftModel`] object. Only causal language models are supported.
        ref_model (`PreTrainedModel`, *optional*):
            Reference model used to compute the reference log probabilities.

            - If provided, this model is used directly as the reference policy.
            - If `None`, the trainer will automatically use the initial policy corresponding to `model`, i.e. the model
              state before DPO training starts.
        args ([`DPOConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        data_collator ([`~transformers.DataCollator`], *optional*):
            Function to use to form a batch from a list of elements of the processed `train_dataset` or `eval_dataset`.
            Will default to [`~trainer.dpo_trainer.DataCollatorForPreference`] if the model is a language model and
            [`~trainer.dpo_trainer.DataCollatorForVisionPreference`] if the model is a vision-language model. Custom
            collators must truncate sequences before padding; the trainer does not apply post-collation truncation.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. This trainer supports both [language modeling](#language-modeling) type and
            [prompt-completion](#prompt-completion) type. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Dataset | IterableDataset]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoProcessor.from_pretrained`]. A
            padding token, `tokenizer.pad_token`, must be set. If the processing class has not set a padding token,
            `tokenizer.eos_token` will be used as the default.
        compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
            The function that will be used to compute metrics at evaluation. Must take a
            [`~transformers.EvalPrediction`] and return a dictionary string to metric values. When passing
            [`SFTConfig`] with `batch_eval_metrics` set to `True`, your `compute_metrics` function must take a boolean
            `compute_result` argument. This will be triggered after the last eval batch to signal that the function
            needs to calculate and return the global summary statistics rather than accumulating the batch-level
            statistics.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
            in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of `AdamW` on your
            model and a scheduler given by [`~transformers.get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "dpo"]
    _name = "DPO"
    _paper = {
        "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        "id": "2305.18290",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @inproceedings{rafailov2023direct,
                title        = {{Direct Preference Optimization: Your Language Model is Secretly a Reward Model}},
                author       = {Rafael Rafailov and Archit Sharma and Eric Mitchell and Christopher D. Manning and Stefano Ermon and Chelsea Finn},
                year         = 2023,
                booktitle    = {Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023},
                url          = {http://papers.nips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html},
                editor       = {Alice Oh and Tristan Naumann and Amir Globerson and Kate Saenko and Moritz Hardt and Sergey Levine},
            }"""),
    }

    def __init__(
        self,
        model: "str | PreTrainedModel | PeftModel",
        ref_model: PreTrainedModel | None = None,
        args: DPOConfig | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config: "PeftConfig | None" = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = DPOConfig(f"{model_name}-DPO")

        if train_dataset is None:
            raise ValueError("`train_dataset` is required")
        elif isinstance(train_dataset, IterableDataset):
            # IterableDataset requires dispatch_batches=False because Accelerate's dispatch mode may try to concatenate
            # batches from multiple processes, leading to mismatch errors.
            if args.accelerator_config.dispatch_batches is True:
                logger.warning(
                    "You are using an `IterableDataset` for training with `dispatch_batches=True`. `dispatch_batches` "
                    "is forced to `False` when using an `IterableDataset`. To remove this warning, unset "
                    "`dispatch_batches` in `DPOConfig` or set it to `False`."
                )
            args.accelerator_config.dispatch_batches = False

        # Model
        if isinstance(model, str):
            model_init_kwargs = args.model_init_kwargs or {}
            # Distributed training requires device_map=None ["auto" fails]
            if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            model = create_model_from_path(model, **model_init_kwargs)
        else:
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `DPOConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )
        if ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. In most cases you should omit `ref_model` and "
                "we'll initialize it to a copy of `model` for you."
            )

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(get_config_model_id(model.config))

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
            self._is_vlm = True
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
            self._is_vlm = False
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if is_peft_available() and is_peft_model(model) and peft_config is not None:
            raise ValueError(
                "You passed a `PeftModel` instance together with a `peft_config` to the trainer. Please first merge "
                "and unload the existing adapter, save the resulting base model, and then pass that base model along "
                "with the new `peft_config` to the trainer."
            )
        if is_peft_available() and is_peft_model(model) and ref_model is None:
            # If the model is a PEFT model with a pretrained adapter, we need to create a "ref" adapter that is a copy
            # of the "default" adapter, so that we can use it as the reference model during DPO training.
            model.add_adapter("ref", model.peft_config["default"])
            for name, param in model.named_parameters():
                if ".default." in name:
                    ref_name = name.replace(".default.", ".ref.")
                    ref_param = model.get_parameter(ref_name)
                    ref_param.data.copy_(param.data)

        # Create PEFT model
        if False:
            model = model

        # When using gradient checkpointing with PEFT, we need to enable input gradients. transformers.Trainer normally
        # handles this, but a bug currently prevents it; see https://github.com/huggingface/transformers/issues/42489
        if is_peft_available() and isinstance(model, PeftModel) and args.gradient_checkpointing:
            model.enable_input_require_grads()

        # When using QLoRA, the PEFT adapter weights are converted to bf16 to follow the recommendations from the
        # original paper [see https://huggingface.co/papers/2305.14314, paragraph 3]. Normally, this can be done by
        # passing `autocast_adapter_dtype=False` to `get_peft_model`, but this option is not yet supported for
        # quantized models. See: https://github.com/huggingface/peft/issues/2889
        # Non-quantized models do not have the `is_loaded_in_{8,4}bit` attributes, whereas quantized models do
        if False:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.data.to(torch.bfloat16)

        # Data collator
        self.padding_free = args.padding_free
        if self.padding_free:
            logger.warning(
                "`padding_free=True` is temporarily unavailable after a refactor and is currently disabled. Falling "
                "back to standard padding (`padding_free=False`). This feature is planned to return in a future "
                "update; for now, please set `padding_free=False` explicitly."
            )
            self.padding_free = False
        dataset_sample = next(iter(train_dataset))
        self._is_vision_dataset = "image" in dataset_sample or "images" in dataset_sample
        if self._is_vision_dataset and not self._is_vlm:
            raise ValueError(
                "The dataset appears to be vision-related (contains 'image' or 'images' keys), but the provided "
                "model does not seem to be a vision-language model. Please check your model and dataset."
            )
        if self._is_vision_dataset and args.max_length is not None and args.truncation_mode == "keep_end":
            raise ValueError(
                "truncation_mode='keep_end' is not supported for vision-language models. Image tokens reside "
                "inside the prompt portion of the sequence; depending on the example, keep_end may silently "
                "drop them, causing pixel_values to be forwarded to the model with no corresponding visual "
                "tokens in input_ids. Use truncation_mode='keep_start' (the default) or set max_length=None."
            )
        if data_collator is None and not self._is_vision_dataset:
            # Get the pad token: if not provided, use the one from the processing class or the eos token
            # if the processing class does not have a pad token.
            pad_token = args.pad_token or tokenizer.pad_token or tokenizer.eos_token
            if pad_token not in tokenizer.get_vocab():
                raise ValueError(
                    f"The specified `pad_token` ('{pad_token}') is not found in the vocabulary of the given "
                    f"`processing_class` ({processing_class.__class__.__name__}). Ensure that the `pad_token` exists "
                    "in the vocabulary before using it as a padding token."
                )
            tokenizer.pad_token = pad_token
            data_collator = DataCollatorForPreference(
                pad_token_id=tokenizer.pad_token_id,
                max_length=args.max_length,
                truncation_mode=args.truncation_mode,
                pad_to_multiple_of=args.pad_to_multiple_of,
            )
        elif data_collator is None and self._is_vision_dataset:
            data_collator = DataCollatorForVisionPreference(
                processor=processing_class,
                max_length=args.max_length,
                pad_to_multiple_of=args.pad_to_multiple_of,
            )

        # Training arguments
        self.beta = args.beta
        self.precompute_ref_logps = args.precompute_ref_log_probs
        self.loss_types = args.loss_type  # args.loss_type is already a list
        self.loss_weights = args.loss_weights or [1.0] * len(self.loss_types)
        self.ld_alpha = args.ld_alpha
        self.f_divergence_type = args.f_divergence_type
        self.f_alpha_divergence_coef = args.f_alpha_divergence_coef
        self.label_smoothing = args.label_smoothing
        self.use_weighting = args.use_weighting
        if self.use_weighting and any(loss_type in {"aot", "aot_unpaired"} for loss_type in self.loss_types):
            raise NotImplementedError(
                "WPO-style weighting is not implemented for 'aot' or 'aot_unpaired' because those losses sort "
                "samples, which would misalign per-pair weights."
            )
        if "robust" in self.loss_types and not (0.0 <= self.label_smoothing < 0.5):
            logger.warning(
                "The `label_smoothing` parameter should lie in [0.0, 0.5) for the 'robust' loss. You provided "
                f"{self.label_smoothing}."
            )
        if "exo_pair" in self.loss_types and self.label_smoothing == 0.0:
            raise ValueError(
                "Label smoothing must be greater than 0.0 when using 'exo_pair' loss. The EXO paper recommends a "
                "value of 1e-3."
            )
        self.use_liger_kernel = args.use_liger_kernel
        if args.use_liger_kernel:
            if not is_liger_kernel_available():
                raise ImportError(
                    "You set `use_liger_kernel=True` but the liger kernel is not available. "
                    "Please install liger-kernel first: `pip install liger-kernel`"
                )
            if len(self.loss_types) != 1:
                raise NotImplementedError(
                    "Multiple loss types are not yet supported when using Liger kernel. If you need this feature, "
                    "please open a feature request at https://github.com/huggingface/trl/issues."
                )
            self.liger_loss_fn = LigerFusedLinearDPOLoss(beta=args.beta, loss_type=self.loss_types[0])
            if compute_metrics is not None:
                raise ValueError(
                    "compute_metrics is not supported with the Liger kernel. compute_metrics requires to be able to "
                    "recover the logits from the forward pass, but Liger kernel does not materialize logits."
                )
            if self.precompute_ref_logps:
                raise ValueError(
                    "Liger DPO loss does not support precomputing reference log probabilities. Either disable "
                    "`precompute_ref_log_probs` or set `use_liger_kernel` to False."
                )

        # Dataset
        # Skip dataset preparation if it's a VLM, where preprocessing [e.g., image-to-pixel conversion] is too costly
        # and done on the fly instead.
        skip_prepare_dataset = self._is_vision_dataset
        if not skip_prepare_dataset:
            train_dataset = self._prepare_dataset(train_dataset, processing_class, args, "train")
            if eval_dataset is not None:
                if isinstance(eval_dataset, dict):
                    eval_dataset = {
                        key: self._prepare_dataset(dataset, processing_class, args, key)
                        for key, dataset in eval_dataset.items()
                    }
                else:
                    eval_dataset = self._prepare_dataset(eval_dataset, processing_class, args, "eval")

        # Transformers explicitly set use_reentrant=True in the past to silence a PyTorch warning, but the default was
        # never updated once PyTorch switched to recommending use_reentrant=False. Until that change lands upstream
        # [see https://github.com/huggingface/transformers/pull/43203] and is released [most likely in 5.0.0], we
        # default to the recommended non-reentrant behavior here, while preserving any user-provided value.
        if args.gradient_checkpointing and Version(transformers.__version__) < Version("5.0.0"):
            args.gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
            args.gradient_checkpointing_kwargs.setdefault("use_reentrant", False)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Initialize activation offloading context
        if self.args.activation_offloading:
            self.maybe_activation_offload_context = get_act_offloading_ctx_manager(model=self.model)
        else:
            self.maybe_activation_offload_context = contextlib.nullcontext()

        # Reference model
        if ref_model is None:
            if is_peft_model(self.model):
                # If PEFT is used, the reference model is not needed since the adapter can be disabled to revert to the
                # initial model.
                self.ref_model = None
            else:
                ref_model_init_kwargs = args.model_init_kwargs or {}
                # Distributed training requires device_map=None ["auto" fails]
                if self.args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                    ref_model_init_kwargs["device_map"] = None
                ref_model_path = get_config_model_id(self.model.config)
                self.ref_model = create_model_from_path(ref_model_path, **ref_model_init_kwargs)
        else:
            self.ref_model = ref_model

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            if self.ref_model is None:
                raise NotImplementedError(
                    "You passed `sync_ref_model=True` while using a PEFT model, which is currently not supported. "
                    "With PEFT, DPOTrainer does not keep a separate reference model in memory; instead, it recovers "
                    "reference behavior by temporarily disabling the adapter. As a result, there is no standalone "
                    "`ref_model` instance to synchronize. Use `sync_ref_model=False`, or opt for full fine-tuning if "
                    "you need a synced reference model. If you need `sync_ref_model` to work with PEFT, please open a "
                    "feature request at https://github.com/huggingface/trl/issues."
                )
            if args.precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `sync_ref_model=True` together with `precompute_ref_log_probs=True`. "
                    "`precompute_ref_log_probs=True` assumes a fixed reference model, but with `sync_ref_model=True` "
                    "the reference model is periodically updated during training, making any precomputed reference "
                    "log-probs stale. Set `precompute_ref_log_probs=False` or disable `sync_ref_model`."
                )
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        if args.precompute_ref_log_probs:
            if isinstance(self.train_dataset, IterableDataset) or isinstance(
                self.eval_dataset, (IterableDataset, IterableDatasetDict)
            ):
                raise ValueError(
                    "`precompute_ref_log_probs=True` is not supported with IterableDataset. Please use a map-style "
                    "Dataset or set `precompute_ref_log_probs=False`."
                )

            batch_size = self.args.precompute_ref_batch_size or self.args.per_device_train_batch_size
            self.train_dataset = self._precompute_ref_logps(self.train_dataset, "train", batch_size)
            if self.eval_dataset is not None:
                batch_size = self.args.precompute_ref_batch_size or self.args.per_device_eval_batch_size
                if isinstance(self.eval_dataset, dict):
                    self.eval_dataset = {
                        name: self._precompute_ref_logps(dataset, name, batch_size)
                        for name, dataset in self.eval_dataset.items()
                    }
                else:
                    self.eval_dataset = self._precompute_ref_logps(self.eval_dataset, "eval", batch_size)

    def _tokenize(
        self,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin,
        input: str | list,
        **kwargs,
    ) -> dict[str, list]:
        """Tokenize a single example for dataset preprocessing.

        Dispatches to `apply_chat_template` for conversational input (list of message dicts) and to `__call__` for
        non-conversational input (str). For VLMs, normalizes the batch dimension that processors emit even for single
        examples.

        Args:
            processing_class ([`~transformers.PreTrainedTokenizerBase`] or [`~transformers.ProcessorMixin`]):
                The tokenizer or processor to use.
            input (`str` or `list`):
                A string for non-conversational input, or a list of message dicts for conversational input.
            **kwargs:
                Forwarded to `apply_chat_template` (e.g. `add_generation_prompt`, `return_assistant_tokens_mask`).

        Returns:
            `dict` with at least an `"input_ids"` key mapping to a flat `list[int]`.
        """
        if isinstance(input, list):  # conversational: list of message dicts
            if self._is_vlm:
                input = prepare_multimodal_messages(input)
            result = processing_class.apply_chat_template(input, tokenize=True, return_dict=True, **kwargs)
        else:  # non-conversational: plain text string
            result = processing_class(text=input)
        # VLMs emit a batch dimension even for single examples; unwrap it
        if self._is_vlm:
            return {k: v[0] for k, v in result.items()}
        return result

    def _prepare_dataset(
        self,
        dataset: Dataset | IterableDataset,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin,
        args: DPOConfig,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = args.dataset_num_proc

        with PartialState().main_process_first():
            # Extract the prompt if needed
            first_example = next(iter(dataset))
            if "prompt" not in first_example:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Extracting prompt from {dataset_name} dataset"
                dataset = dataset.map(extract_prompt, **map_kwargs)

            # Apply the chat template if needed
            first_example = next(iter(dataset))
            if not is_conversational(first_example):
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Adding EOS to {dataset_name} dataset"

                def add_eos(example, eos_token):
                    if not example["chosen"].endswith(eos_token):
                        example["chosen"] = example["chosen"] + eos_token
                    if not example["rejected"].endswith(eos_token):
                        example["rejected"] = example["rejected"] + eos_token
                    return example

                eos_token = processing_class.tokenizer.eos_token if self._is_vlm else processing_class.eos_token
                dataset = dataset.map(add_eos, fn_kwargs={"eos_token": eos_token}, **map_kwargs)

            # Tokenize the dataset
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

            def tokenize_fn(example, processing_class):
                tools = example.get("tools")
                tools = json.loads(tools) if isinstance(tools, str) else tools
                output = {}
                if is_conversational(example):
                    prompt_ids = self._tokenize(
                        processing_class,
                        example["prompt"],
                        tools=tools,
                        add_generation_prompt=True,
                        **example.get("chat_template_kwargs", {}),
                    )["input_ids"]
                    prompt_chosen_ids = self._tokenize(
                        processing_class,
                        example["prompt"] + example["chosen"],
                        tools=tools,
                        **example.get("chat_template_kwargs", {}),
                    )["input_ids"]
                    prompt_rejected_ids = self._tokenize(
                        processing_class,
                        example["prompt"] + example["rejected"],
                        tools=tools,
                        **example.get("chat_template_kwargs", {}),
                    )["input_ids"]
                else:
                    prompt_ids = self._tokenize(processing_class, example["prompt"])["input_ids"]
                    prompt_chosen_ids = self._tokenize(processing_class, example["prompt"] + example["chosen"])[
                        "input_ids"
                    ]
                    prompt_rejected_ids = self._tokenize(processing_class, example["prompt"] + example["rejected"])[
                        "input_ids"
                    ]

                # Check if the tokenized prompt starts with the tokenized prompt+completion
                if not prompt_chosen_ids[: len(prompt_ids)] == prompt_ids:
                    logger.warning(
                        "Mismatch between tokenized prompt and the start of tokenized prompt+chosen. "
                        "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
                        "token handling. Verify that the tokenizer is processing text consistently."
                    )
                if not prompt_rejected_ids[: len(prompt_ids)] == prompt_ids:
                    logger.warning(
                        "Mismatch between tokenized prompt and the start of tokenized prompt+rejected. "
                        "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
                        "token handling. Verify that the tokenizer is processing text consistently."
                    )

                output["prompt_ids"] = prompt_ids
                output["chosen_ids"] = prompt_chosen_ids[len(prompt_ids) :]
                output["rejected_ids"] = prompt_rejected_ids[len(prompt_ids) :]
                return output

            dataset = dataset.map(tokenize_fn, fn_kwargs={"processing_class": processing_class}, **map_kwargs)

        return dataset

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs (usually, "input_ids"
        # and "attention_mask").
        if self._signature_columns is None:
            if self._is_vision_dataset:
                self._signature_columns = [
                    "prompt",
                    "chosen",
                    "rejected",
                    "image",
                    "images",
                    "tools",
                    "chat_template_kwargs",
                ]
            else:
                self._signature_columns = [
                    "prompt_ids",
                    "chosen_ids",
                    "rejected_ids",
                    "ref_chosen_logps",
                    "ref_rejected_logps",
                ]

    def _precompute_ref_logps(self, dataset: Dataset, name: str, batch_size: int) -> Dataset:
        model_hash = hash_module(self.ref_model or self.model)
        fingerprint = Hasher.hash((dataset._fingerprint, model_hash))
        cache_file = dataset._get_cache_file_path(fingerprint).removesuffix(".arrow") + ".npz"
        if os.path.exists(cache_file):
            loaded = np.load(cache_file)
            ref_chosen_logps = loaded["ref_chosen_logps"]
            ref_rejected_logps = loaded["ref_rejected_logps"]
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                shuffle=False,
            )
            data_loader = self.accelerator.prepare(dataloader)
            ref_chosen_logps = []
            ref_rejected_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc=f"Computing reference log probs for {name} dataset"):
                ref_chosen_logp, ref_rejected_logp = self.compute_ref_log_probs(padded_batch)
                ref_chosen_logp, ref_rejected_logp = self.accelerator.gather_for_metrics(
                    (ref_chosen_logp, ref_rejected_logp)
                )
                ref_chosen_logps.append(ref_chosen_logp.cpu())
                ref_rejected_logps.append(ref_rejected_logp.cpu())

            # Save the reference log probabilities to cache. We need .float() because bf16 is not supported by numpy
            ref_chosen_logps = torch.cat(ref_chosen_logps).float().numpy()
            ref_rejected_logps = torch.cat(ref_rejected_logps).float().numpy()
            if self.accelerator.is_main_process:
                np.savez_compressed(
                    cache_file, ref_chosen_logps=ref_chosen_logps, ref_rejected_logps=ref_rejected_logps
                )
            self.accelerator.wait_for_everyone()

        dataset = dataset.add_column(name="ref_chosen_logps", column=ref_chosen_logps)
        dataset = dataset.add_column(name="ref_rejected_logps", column=ref_rejected_logps, new_fingerprint=fingerprint)

        return dataset

    def compute_ref_log_probs(self, inputs):
        """Computes reference log probabilities for a single padded batch."""
        device = self.accelerator.device

        _non_model_keys = {"completion_mask", "ref_chosen_logps", "ref_rejected_logps"}
        model_kwargs = {k: v for k, v in inputs.items() if k not in _non_model_keys}
        model_kwargs["use_cache"] = False

        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            if is_peft_model(self.model) and self.ref_model is None:
                model = self.accelerator.unwrap_model(self.model)
                with use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None):
                    ref_outputs = self.model(**model_kwargs)
            else:
                ref_outputs = self.ref_model(**model_kwargs)

        input_ids = inputs["input_ids"]
        completion_mask = inputs["completion_mask"]
        shift_labels = input_ids[..., 1:].contiguous()
        shift_completion_mask = completion_mask[..., 1:].contiguous()
        ref_shift_logits = ref_outputs.logits[..., :-1, :].contiguous()
        ref_per_token_logps = selective_log_softmax(ref_shift_logits, shift_labels)
        ref_per_token_logps[shift_completion_mask == 0] = 0.0

        if self.ld_alpha is None:
            ref_logps = ref_per_token_logps.sum(dim=1)
        else:
            comp_pos = shift_completion_mask.cumsum(dim=1)
            comp_lens = shift_completion_mask.sum(dim=1).long()
            chosen_lens, rejected_lens = comp_lens.chunk(2, dim=0)
            shared_lens = torch.minimum(chosen_lens, rejected_lens)
            shared_lens = torch.cat([shared_lens, shared_lens], dim=0).to(device)
            shared_mask = (comp_pos > 0) & (comp_pos <= shared_lens.unsqueeze(1))
            tail_mask = comp_pos > shared_lens.unsqueeze(1)
            shared_logps = (ref_per_token_logps * shared_mask).sum(dim=1)
            tail_logps = (ref_per_token_logps * tail_mask).sum(dim=1)
            ref_logps = shared_logps + self.ld_alpha * tail_logps

        ref_chosen_logps, ref_rejected_logps = ref_logps.chunk(2, dim=0)
        return ref_chosen_logps, ref_rejected_logps

    def _compute_loss_liger(self, model, inputs, return_outputs):
        if return_outputs:
            raise RuntimeError(
                "return_outputs=True is not supported with the Liger DPO loss. The Liger loss computes the loss "
                "without materializing logits, so outputs cannot be returned."
            )

        mode = "train" if self.model.training else "eval"

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        completion_mask = inputs["completion_mask"]

        decoder = model.get_decoder()
        outputs = decoder(input_ids, attention_mask=attention_mask, use_cache=False)
        hidden_states = outputs.last_hidden_state[:, :-1].contiguous()
        lm_head = model.get_output_embeddings()
        weight = lm_head.weight
        bias = lm_head.bias

        if is_peft_model(model):
            raise NotImplementedError("Liger DPO loss is not implemented for PEFT models.")
        else:
            with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
                ref_decoder = self.ref_model.get_decoder()
                ref_outputs = ref_decoder(input_ids, attention_mask=attention_mask, use_cache=False)
                ref_lm_head = self.ref_model.get_output_embeddings()
                ref_hidden_states = ref_outputs.last_hidden_state[:, :-1].contiguous()
                ref_weight = ref_lm_head.weight
                ref_bias = ref_lm_head.bias

        shift_completion_mask = completion_mask[:, 1:].contiguous()
        labels = input_ids[:, 1:].clone()
        labels[shift_completion_mask == 0] = -100

        loss, metrics = self.liger_loss_fn(
            weight, hidden_states, labels, bias, ref_hidden_states, ref_weight, ref_bias
        )

        (
            chosen_logps,
            rejected_logps,
            chosen_logits_mean,
            rejected_logits_mean,
            nll_loss,
            chosen_rewards,
            rejected_rewards,
        ) = metrics

        if mode == "train":
            num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        avg_chosen_logits = self.accelerator.gather_for_metrics(chosen_logits_mean).mean().item()
        avg_rejected_logits = self.accelerator.gather_for_metrics(rejected_logits_mean).mean().item()
        self._metrics[mode]["logits/chosen"].append(avg_chosen_logits)
        self._metrics[mode]["logits/rejected"].append(avg_rejected_logits)

        agg_chosen_rewards = self.accelerator.gather(chosen_rewards)
        agg_rejected_rewards = self.accelerator.gather(rejected_rewards)
        self._metrics[mode]["rewards/chosen"].append(agg_chosen_rewards.mean().item())
        self._metrics[mode]["rewards/rejected"].append(agg_rejected_rewards.mean().item())

        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        agg_reward_accuracies = self.accelerator.gather(reward_accuracies)
        self._metrics[mode]["rewards/accuracies"].append(agg_reward_accuracies.mean().item())

        margins = chosen_rewards - rejected_rewards
        agg_margins = self.accelerator.gather(margins)
        self._metrics[mode]["rewards/margins"].append(agg_margins.mean().item())

        self._metrics[mode]["logps/chosen"].append(self.accelerator.gather(chosen_logps).mean().item())
        self._metrics[mode]["logps/rejected"].append(self.accelerator.gather(rejected_logps).mean().item())

        return loss

    def _compute_loss(self, model, inputs, return_outputs):
        mode = "train" if self.model.training else "eval"
        device = self.accelerator.device

        _non_model_keys = {"completion_mask", "ref_chosen_logps", "ref_rejected_logps"}
        model_kwargs = {k: v for k, v in inputs.items() if k not in _non_model_keys}
        model_kwargs["use_cache"] = False
        outputs = model(**model_kwargs)

        input_ids = inputs["input_ids"]
        completion_mask = inputs["completion_mask"]
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_completion_mask = completion_mask[..., 1:].contiguous()
        per_token_logps = selective_log_softmax(shift_logits, shift_labels)
        per_token_logps[shift_completion_mask == 0] = 0.0  # mask out non-completion tokens
        if self.ld_alpha is None:
            logps = per_token_logps.sum(dim=1)  # sum over sequence length
        else:
            comp_pos = shift_completion_mask.cumsum(dim=1)
            comp_lens = shift_completion_mask.sum(dim=1).long()
            chosen_lens, rejected_lens = comp_lens.chunk(2, dim=0)
            shared_lens = torch.minimum(chosen_lens, rejected_lens)
            shared_lens = torch.cat([shared_lens, shared_lens], dim=0).to(device)
            shared_mask = (comp_pos > 0) & (comp_pos <= shared_lens.unsqueeze(1))  # shared: 1 <= pos <= shared_len
            tail_mask = comp_pos > shared_lens.unsqueeze(1)  # tail: pos > shared_len
            shared_logps = (per_token_logps * shared_mask).sum(dim=1)
            tail_logps = (per_token_logps * tail_mask).sum(dim=1)
            logps = shared_logps + self.ld_alpha * tail_logps
        chosen_logps, rejected_logps = logps.chunk(2, dim=0)  # batch is [chosen, rejected]

        if self.precompute_ref_logps:
            ref_chosen_logps, ref_rejected_logps = inputs["ref_chosen_logps"], inputs["ref_rejected_logps"]
        else:
            # When gradient checkpointing is enabled with use_reentrant=True (default), calling the model inside a
            # torch.no_grad() block triggers a harmless PyTorch warning ("None of the inputs have requires_grad=True").
            # Temporarily disable checkpointing to avoid this warning during inference.
            with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
                if is_peft_model(model) and self.ref_model is None:
                    # When training a PEFT adapter, how we obtain the reference depends on the setup:
                    # - New adapter: disabling adapters yields the base model.
                    # - Re-training an existing adapter: an initial copy is loaded under the name "ref".
                    model = self.accelerator.unwrap_model(model)
                    with use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None):
                        ref_outputs = self.model(**model_kwargs)
                else:
                    ref_outputs = self.ref_model(**model_kwargs)

            ref_shift_logits = ref_outputs.logits[..., :-1, :].contiguous()
            ref_per_token_logps = selective_log_softmax(ref_shift_logits, shift_labels)
            ref_per_token_logps[shift_completion_mask == 0] = 0.0  # mask out non-completion tokens
            if self.ld_alpha is None:
                ref_logps = ref_per_token_logps.sum(dim=1)  # sum over sequence length
            else:
                # reuse comp_pos/shared_mask/tail_mask computed above (they depend only on completion_mask)
                ref_shared_logps = (ref_per_token_logps * shared_mask).sum(dim=1)
                ref_tail_logps = (ref_per_token_logps * tail_mask).sum(dim=1)
                ref_logps = ref_shared_logps + self.ld_alpha * ref_tail_logps
            ref_chosen_logps, ref_rejected_logps = ref_logps.chunk(2, dim=0)  # batch is [chosen, rejected]

        # Get the log ratios for the chosen and rejected responses
        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps

        if self.f_divergence_type == "reverse_kl":  # standard DPO
            chosen_scores = chosen_logratios
            rejected_scores = rejected_logratios
        elif self.f_divergence_type == "forward_kl":
            # f'(t) = 1 - 1/t  -> drop constant -> -exp(-logratio)
            chosen_scores = -torch.exp(-chosen_logratios)
            rejected_scores = -torch.exp(-rejected_logratios)
        elif self.f_divergence_type == "js_divergence":
            # f'(t) = log(2t/(t+1)) -> drop log 2
            chosen_scores = F.logsigmoid(chosen_logratios)
            rejected_scores = F.logsigmoid(rejected_logratios)
        elif self.f_divergence_type == "alpha_divergence":
            # alpha-divergence: f'(t) = (t^(α-1) - 1)/(α-1)
            if abs(self.f_alpha_divergence_coef - 1.0) < 1e-6:  # limit case f'(t) -> log(t), fall back to reverse_kl
                chosen_scores = chosen_logratios
                rejected_scores = rejected_logratios
            else:
                coef = 1.0 / (self.f_alpha_divergence_coef - 1.0)
                t_chosen = (self.f_alpha_divergence_coef - 1.0) * chosen_logratios
                t_rejected = (self.f_alpha_divergence_coef - 1.0) * rejected_logratios
                dtype = t_chosen.dtype
                # Clamp max so exp(.) stays representable after casting back
                clamp_max = {torch.float16: 11.0, torch.bfloat16: 80.0, torch.float32: 80.0}[dtype]
                t_chosen_float = torch.clamp(t_chosen.float(), max=clamp_max)
                t_rejected_float = torch.clamp(t_rejected.float(), max=clamp_max)
                chosen_scores = torch.exp(t_chosen_float).to(dtype) * coef
                rejected_scores = torch.exp(t_rejected_float).to(dtype) * coef
        else:
            raise ValueError(f"Unknown f_divergence_type: {self.f_divergence_type}")

        delta_score = chosen_scores - rejected_scores

        loss = 0.0
        for loss_type, loss_weight in zip(self.loss_types, self.loss_weights, strict=True):
            if loss_type == "sigmoid":
                per_sequence_loss = -F.logsigmoid(self.beta * delta_score)

            elif loss_type == "hinge":
                per_sequence_loss = torch.relu(1 - self.beta * delta_score)

            elif loss_type == "ipo":
                # IPO uses sequence-level log-prob differences; in code these are token-summed over the completion,
                # which makes the squared loss scale with completion length. We therefore normalize by the number of
                # completion tokens (average per token) to make β/loss comparable across variable lengths. This length
                # normalization is not explicitly discussed in the IPO paper; we confirmed this choice with the IPO
                # authors, and the results reported in the paper correspond to this normalized form.
                chosen_mask, rejected_mask = completion_mask.chunk(2, dim=0)
                chosen_avg_score = chosen_scores / chosen_mask.sum(dim=1).clamp(min=1.0)
                rejected_avg_score = rejected_scores / rejected_mask.sum(dim=1).clamp(min=1.0)
                ipo_delta = chosen_avg_score - rejected_avg_score
                # (Eq. 17) of the paper where beta is the regularization parameter for the IPO loss, denoted by τ.
                per_sequence_loss = (ipo_delta - 1 / (2 * self.beta)) ** 2

            elif loss_type == "exo_pair":
                # Implements EXO-pref from the paper https://huggingface.co/papers/2402.00856, (Eq. 16)
                # Minimize KL(p_fθ || p_rh) for K=2; p_fθ = softmax(βπ * (log πθ − log π_ref)) over {chosen, rejected}
                # p_rh = [(1−ε), ε]; expanded KL gives the weighted logsigmoid form below
                epsilon = torch.tensor(self.label_smoothing, device=device)
                qw = torch.sigmoid(self.beta * delta_score)
                log_qw = F.logsigmoid(self.beta * delta_score)
                log_pw = torch.log1p(-epsilon)
                ql = torch.sigmoid(-self.beta * delta_score)
                log_ql = F.logsigmoid(-self.beta * delta_score)
                log_pl = torch.log(epsilon)
                per_sequence_loss = qw * (log_qw - log_pw) + ql * (log_ql - log_pl)

            elif loss_type == "nca_pair":
                chosen_rewards = self.beta * chosen_scores
                rejected_rewards = self.beta * rejected_scores
                per_sequence_loss = (
                    -F.logsigmoid(chosen_rewards)
                    - 0.5 * F.logsigmoid(-chosen_rewards)
                    - 0.5 * F.logsigmoid(-rejected_rewards)
                )

            elif loss_type == "robust":
                clean_loss_term = -(1 - self.label_smoothing) * F.logsigmoid(self.beta * delta_score)
                flipped_loss_term = -self.label_smoothing * F.logsigmoid(-self.beta * delta_score)
                per_sequence_loss = (clean_loss_term - flipped_loss_term) / (1 - 2 * self.label_smoothing)

            elif loss_type == "bco_pair":
                chosen_rewards = self.beta * chosen_scores
                rejected_rewards = self.beta * rejected_scores
                per_sequence_loss = -F.logsigmoid(chosen_rewards) - F.logsigmoid(-rejected_rewards)

            elif loss_type == "sppo_hard":
                # In the paper (https://huggingface.co/papers/2405.00675), SPPO employs a soft probability approach,
                # estimated using the PairRM score. The probability calculation is conducted outside of the trainer
                # class. The version described here is the hard probability version, where P in Equation (4.7) of
                # Algorithm 1 is set to 1 for the winner and 0 for the loser.
                winner_margin_error = (chosen_scores - 0.5 / self.beta) ** 2
                loser_margin_error = (rejected_scores + 0.5 / self.beta) ** 2
                per_sequence_loss = winner_margin_error + loser_margin_error

            elif loss_type == "aot":
                logratios = chosen_logps - rejected_logps
                ref_logratios = ref_chosen_logps - ref_rejected_logps
                logratios_sorted, _ = torch.sort(logratios, dim=0)
                ref_logratios_sorted, _ = torch.sort(ref_logratios, dim=0)
                delta = logratios_sorted - ref_logratios_sorted
                per_sequence_loss = (
                    -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                    - F.logsigmoid(-self.beta * delta) * self.label_smoothing
                )

            elif loss_type == "aot_unpaired":
                chosen_logratios_sorted, _ = torch.sort(chosen_logratios, dim=0)
                rejected_logratios_sorted, _ = torch.sort(rejected_logratios, dim=0)
                delta = chosen_logratios_sorted - rejected_logratios_sorted
                per_sequence_loss = (
                    -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                    - F.logsigmoid(-self.beta * delta) * self.label_smoothing
                )

            elif loss_type == "apo_zero":
                # Eqn (7) of the APO paper (https://huggingface.co/papers/2408.06266)
                # Use this loss when you believe the chosen outputs are better than your model's default output
                # Increase chosen likelihood and decrease rejected likelihood
                losses_chosen = 1 - torch.sigmoid(self.beta * chosen_logratios)
                losses_rejected = torch.sigmoid(self.beta * rejected_logratios)
                per_sequence_loss = losses_chosen + losses_rejected

            elif loss_type == "apo_down":
                # Eqn (8) of the APO paper (https://huggingface.co/papers/2408.06266)
                # Use this loss when you believe the chosen outputs are worse than your model's default output.
                # Decrease chosen likelihood and decrease rejected likelihood more
                losses_chosen = torch.sigmoid(self.beta * chosen_logratios)
                losses_rejected = 1 - torch.sigmoid(self.beta * delta_score)
                per_sequence_loss = losses_chosen + losses_rejected

            elif loss_type == "discopop":
                # Eqn (5) of the DiscoPOP paper (https://huggingface.co/papers/2406.08414)
                logits = delta_score * self.beta
                # Modulate the mixing coefficient based on the log ratio magnitudes
                log_ratio_modulation = torch.sigmoid(logits / self.args.discopop_tau)
                logistic_component = -F.logsigmoid(logits)
                exp_component = torch.exp(-logits)
                # Blend between logistic and exponential component based on log ratio modulation
                per_sequence_loss = (
                    logistic_component * (1 - log_ratio_modulation) + exp_component * log_ratio_modulation
                )

            elif loss_type == "sft":
                chosen_logits, _ = shift_logits.chunk(2, dim=0)
                chosen_labels, _ = shift_labels.chunk(2, dim=0)
                chosen_mask, _ = shift_completion_mask.chunk(2, dim=0)
                batch_loss = F.cross_entropy(chosen_logits[chosen_mask.bool()], chosen_labels[chosen_mask.bool()])
                # Implementation convenience: expand the scalar SFT loss to a per-sequence tensor so it matches the
                # shape of other losses; only the mean is used, so this is a no-op numerically.
                per_sequence_loss = batch_loss.expand(chosen_logits.size(0))

            else:
                raise ValueError(
                    f"Unknown loss type: {loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'exo_pair', "
                    "'nca_pair', 'robust', 'bco_pair', 'sppo_hard', 'aot', 'aot_unpaired', 'apo_zero', 'apo_down', "
                    "'discopop', 'sft']"
                )

            if self.use_weighting:
                # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
                completion_lengths = shift_completion_mask.sum(dim=1).clamp_min(1)
                with torch.no_grad():
                    lse1 = torch.logsumexp(shift_logits, dim=-1)
                    lse2 = torch.logsumexp(2.0 * shift_logits, dim=-1)
                    log_denom = lse2 - 2.0 * lse1
                    aligned_logps = (per_token_logps - log_denom) * shift_completion_mask
                mean_logps = aligned_logps.sum(dim=1) / completion_lengths
                weights = torch.exp(mean_logps)
                chosen_weights, rejected_weights = weights.chunk(2, dim=0)
                per_sequence_loss *= chosen_weights * rejected_weights

            loss += per_sequence_loss.mean() * loss_weight

        # Log the metrics
        # Entropy
        per_token_entropy = entropy_from_logits(shift_logits.detach())
        entropy = per_token_entropy[shift_completion_mask.bool()].mean()
        entropy = self.accelerator.gather_for_metrics(entropy).mean().item()
        self._metrics[mode]["entropy"].append(entropy)

        # Number of tokens
        if mode == "train":
            num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # Average logits for chosen and rejected completions
        chosen_logits, rejected_logits = shift_logits.detach().chunk(2, dim=0)
        chosen_mask, rejected_mask = shift_completion_mask.chunk(2, dim=0)
        total_chosen_logits = chosen_logits[chosen_mask.bool()].mean(-1).sum()
        total_chosen_tokens = chosen_mask.sum()
        total_rejected_logits = rejected_logits[rejected_mask.bool()].mean(-1).sum()
        total_rejected_tokens = rejected_mask.sum()
        total_chosen_logits = self.accelerator.gather_for_metrics(total_chosen_logits).sum().item()
        total_chosen_tokens = self.accelerator.gather_for_metrics(total_chosen_tokens).sum().item()
        total_rejected_logits = self.accelerator.gather_for_metrics(total_rejected_logits).sum().item()
        total_rejected_tokens = self.accelerator.gather_for_metrics(total_rejected_tokens).sum().item()
        avg_chosen_logits = total_chosen_logits / total_chosen_tokens if total_chosen_tokens > 0 else 0.0
        avg_rejected_logits = total_rejected_logits / total_rejected_tokens if total_rejected_tokens > 0 else 0.0
        self._metrics[mode]["logits/chosen"].append(avg_chosen_logits)
        self._metrics[mode]["logits/rejected"].append(avg_rejected_logits)

        # Token accuracy for the chosen completions
        predictions = chosen_logits.argmax(dim=-1)
        chosen_mask = shift_completion_mask[: len(shift_completion_mask) // 2].bool()
        chosen_labels = shift_labels[: len(shift_labels) // 2]
        correct_predictions = (predictions == chosen_labels) & chosen_mask
        total_tokens = chosen_mask.sum()
        correct_tokens = correct_predictions.sum()
        correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
        total_tokens = self.accelerator.gather_for_metrics(total_tokens)
        total_sum = total_tokens.sum()
        accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
        self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        # Rewards for chosen and rejected completions
        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = self.beta * rejected_logratios.detach()
        agg_chosen_rewards = self.accelerator.gather(chosen_rewards)
        agg_rejected_rewards = self.accelerator.gather(rejected_rewards)
        self._metrics[mode]["rewards/chosen"].append(agg_chosen_rewards.mean().item())
        self._metrics[mode]["rewards/rejected"].append(agg_rejected_rewards.mean().item())

        # Reward accuracy
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        agg_reward_accuracies = self.accelerator.gather(reward_accuracies)
        self._metrics[mode]["rewards/accuracies"].append(agg_reward_accuracies.mean().item())

        # Reward margins
        margins = chosen_rewards - rejected_rewards
        agg_margins = self.accelerator.gather(margins)
        self._metrics[mode]["rewards/margins"].append(agg_margins.mean().item())

        # Average log probabilities for chosen and rejected completions
        self._metrics[mode]["logps/chosen"].append(self.accelerator.gather(chosen_logps).mean().item())
        self._metrics[mode]["logps/rejected"].append(self.accelerator.gather(rejected_logps).mean().item())

        return (loss, outputs) if return_outputs else loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.use_liger_kernel:
            return self._compute_loss_liger(model, inputs, return_outputs)
        else:
            return self._compute_loss(model, inputs, return_outputs)

    # Override training step to add activation offloading context.
    def training_step(self, *args, **kwargs):
        with self.maybe_activation_offload_context:
            return super().training_step(*args, **kwargs)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()

    # During eval, Trainer calls prediction_step. If no labels are present in the inputs, it only runs forward and
    # returns logits. We override prediction_step to force compute_loss, because this trainer doesn't involve labels.
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: list[str] | None = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            if prediction_loss_only:
                loss = self.compute_loss(model, inputs, return_outputs=False)  # logits aren't materialized with liger
                logits, labels = None, None
            else:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                logits, labels = outputs.logits, inputs["input_ids"]
        return loss, logits, labels

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)
class UnslothDPOTrainer(_UnslothDPOTrainer):
    """
    
Trainer for Direct Preference Optimization (DPO) method. This algorithm was initially proposed in the paper [Direct
Preference Optimization: Your Language Model is Secretly a Reward Model](https://huggingface.co/papers/2305.18290).
This class is a wrapper around the [`~transformers.Trainer`] class and inherits all of its attributes and methods.

Example:

```python
from trl import DPOTrainer
from datasets import load_dataset

dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

trainer = DPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    train_dataset=dataset,
)
trainer.train()
```

Args:
    model (`str` or [`~transformers.PreTrainedModel`] or [`~peft.PeftModel`]):
        Model to be trained. Can be either:

        - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
          path to a *directory* containing model weights saved using
          [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
          using `<ModelArchitecture>.from_pretrained` (where `<ModelArchitecture>` is derived from the model
          config) with the keyword arguments in `args.model_init_kwargs`.
        - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        - A [`~peft.PeftModel`] object. Only causal language models are supported.
    ref_model (`PreTrainedModel`, *optional*):
        Reference model used to compute the reference log probabilities.

        - If provided, this model is used directly as the reference policy.
        - If `None`, the trainer will automatically use the initial policy corresponding to `model`, i.e. the model
          state before DPO training starts.
    args ([`DPOConfig`], *optional*):
        Configuration for this trainer. If `None`, a default configuration is used.
    data_collator ([`~transformers.DataCollator`], *optional*):
        Function to use to form a batch from a list of elements of the processed `train_dataset` or `eval_dataset`.
        Will default to [`~trainer.dpo_trainer.DataCollatorForPreference`] if the model is a language model and
        [`~trainer.dpo_trainer.DataCollatorForVisionPreference`] if the model is a vision-language model. Custom
        collators must truncate sequences before padding; the trainer does not apply post-collation truncation.
    train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
        Dataset to use for training. This trainer supports both [language modeling](#language-modeling) type and
        [prompt-completion](#prompt-completion) type. The format of the samples can be either:

        - [Standard](dataset_formats#standard): Each sample contains plain text.
        - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
          and content).
    eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Dataset | IterableDataset]`):
        Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
    processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.ProcessorMixin`], *optional*):
        Processing class used to process the data. The padding side must be set to "left". If `None`, the
        processing class is loaded from the model's name with [`~transformers.AutoProcessor.from_pretrained`]. A
        padding token, `tokenizer.pad_token`, must be set. If the processing class has not set a padding token,
        `tokenizer.eos_token` will be used as the default.
    compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
        The function that will be used to compute metrics at evaluation. Must take a
        [`~transformers.EvalPrediction`] and return a dictionary string to metric values. When passing
        [`SFTConfig`] with `batch_eval_metrics` set to `True`, your `compute_metrics` function must take a boolean
        `compute_result` argument. This will be triggered after the last eval batch to signal that the function
        needs to calculate and return the global summary statistics rather than accumulating the batch-level
        statistics.
    callbacks (list of [`~transformers.TrainerCallback`], *optional*):
        List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
        in [here](https://huggingface.co/docs/transformers/main_classes/callback).

        If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
        method.
    optimizers (`tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None]`, *optional*, defaults to `(None, None)`):
        A tuple containing the optimizer and the scheduler to use. Will default to an instance of `AdamW` on your
        model and a scheduler given by [`~transformers.get_linear_schedule_with_warmup`] controlled by `args`.
    peft_config ([`~peft.PeftConfig`], *optional*):
        PEFT configuration used to wrap the model. If `None`, the model is not wrapped.

    """
    def __init__(
        self,
        model,
        ref_model = None,
        args = None,
        data_collator = None,
        train_dataset = None,
        eval_dataset = None,
        processing_class = None,
        compute_metrics = None,
        callbacks = None,
        peft_config = None,
        **kwargs
    ):
        if args is None: args = UnslothDPOConfig()
        use_bf16 = getattr(args, 'bf16', False)
        if type(use_bf16) is not bool: use_bf16 = False
        use_fp16 = getattr(args, 'fp16', False)
        if type(use_fp16) is not bool: use_fp16 = False
        force_float32 = False
        full_finetuning = os.environ.get('UNSLOTH_ENABLE_FULL_FINETUNING', '0') == '1'
        if not full_finetuning and (os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '1'):
            print('Unsloth: Switching to float32 training since model cannot work with float16')
            force_float32 = True
        mixed_precision_dtype = os.environ.get('UNSLOTH_MIXED_PRECISION', 'float32')
        dtype = getattr(model.config, 'dtype', None) or getattr(model.config, 'torch_dtype', None)
        if dtype is None: dtype = model.get_input_embeddings().weight.dtype
        from unsloth_zoo.utils import _get_dtype
        dtype = _get_dtype(dtype)
        float16 = dtype == torch.float16
        if not force_float32 and (float16 and use_bf16): raise TypeError('Unsloth: Model is in float16 precision but you want to use bfloat16 precision. Set fp16 to `True` and bf16 to `False`')
        if not force_float32 and (not float16 and use_fp16): raise TypeError('Unsloth: Model is in bfloat16 precision but you want to use float16 precision. Set fp16 to `False` and bf16 to `True`')
        if force_float32:
            # Forced float32 training
            args.fp16 = False
            args.bf16 = False
            os.environ['ACCELERATE_MIXED_PRECISION'] = 'no'
            if hasattr(args, 'mixed_precision'): args.mixed_precision = 'no'
            # args.mixed_precision is a new argument which needs to be set now
        elif (not use_bf16 and not use_fp16) and mixed_precision_dtype == 'float32':
            # Mixed precision training
            args.fp16 = float16
            args.bf16 = not float16
            os.environ['ACCELERATE_MIXED_PRECISION'] = 'fp16' if float16 else 'bf16'
            if hasattr(args, 'mixed_precision'): args.mixed_precision = 'fp16' if float16 else 'bf16'
            # args.mixed_precision is a new argument which needs to be set now
        elif mixed_precision_dtype == 'bfloat16':
            # Both False since bfloat16 full finetuning doesn't do any autocasting.
            args.fp16 = False
            args.bf16 = False
            os.environ['ACCELERATE_MIXED_PRECISION'] = 'no'
            if hasattr(args, 'mixed_precision'): args.mixed_precision = 'no'
            # args.mixed_precision is a new argument which needs to be set now
        
        if getattr(args, 'eval_dataset', None) is not None and getattr(args, 'eval_strategy', 'no') == 'no':
            args.eval_strategy = 'steps'
            if getattr(args, 'eval_steps', None) is None: args.eval_steps = 0.1
        ga_steps = getattr(args, 'gradient_accumulation_steps', None)
        if ga_steps is not None and ga_steps > 1:
            from transformers import __version__ as transformers_version
            if Version(transformers_version) <= Version('4.45.2'):
                print('**** Unsloth: Please use our fixed gradient_accumulation_steps by updating transformers, TRL and Unsloth!\n'
                      '`pip install --upgrade --no-cache-dir --force-reinstall --no-deps unsloth transformers trl unsloth_zoo`')
        if getattr(args, 'eval_strategy', 'no') != 'no':
            eval_bsz = getattr(args, 'per_device_eval_batch_size', 8)
            if eval_bsz == 8 and args.per_device_train_batch_size < eval_bsz: args.per_device_eval_batch_size = args.per_device_train_batch_size
            if getattr(args, 'eval_accumulation_steps', None) is None and ga_steps is not None: args.eval_accumulation_steps = ga_steps
        fp16_full_eval = getattr(args, 'fp16_full_eval', False)
        if type(fp16_full_eval) is not bool: fp16_full_eval = False
        bf16_full_eval = getattr(args, 'bf16_full_eval', False)
        if type(bf16_full_eval) is not bool: bf16_full_eval = False
        if args.fp16 and bf16_full_eval: args.bf16_full_eval = False; args.fp16_full_eval = True
        if args.bf16 and fp16_full_eval: args.bf16_full_eval = True; args.fp16_full_eval = False
        if force_float32:
            args.bf16_full_eval = False
            args.fp16_full_eval = False
        elif os.environ.get('UNSLOTH_MIXED_PRECISION', 'float32') == 'bfloat16':
            args.bf16_full_eval = True
            args.fp16_full_eval = False
        elif not bf16_full_eval and not fp16_full_eval:
            args.bf16_full_eval = args.bf16
            args.fp16_full_eval = args.fp16
        _output_logits = False
        if locals().get('compute_metrics', None) is not None: _output_logits = True
        if locals().get('preprocess_logits_for_metrics', None) is not None: _output_logits = True
        if _output_logits:
            os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
        if model is not None:
            _warnings_issued = getattr(model, 'warnings_issued', None)
            if _warnings_issued is None:
                model.warnings_issued = {}
            elif not isinstance(_warnings_issued, dict):
                try:
                    model.warnings_issued = dict(_warnings_issued)
                except Exception:
                    model.warnings_issued = {}
        if 'max_seq_length' not in locals() and not hasattr(args, 'max_seq_length'):
            pass
        else:
            model_max_seq_length = getattr(model, 'max_seq_length', None)
            args_max_seq_length  = getattr(args,  'max_seq_length', None)
            if args_max_seq_length is None and model_max_seq_length is not None:
                max_seq_length = model.max_seq_length
                if hasattr(args, 'max_seq_length'): args.max_seq_length = max_seq_length
            elif args_max_seq_length is not None and model_max_seq_length is not None:
                if args_max_seq_length > model_max_seq_length:
                    print('Unsloth: You set `max_seq_length` as ' + str(args_max_seq_length) + ' but '
                           'the maximum the model supports is ' + str(model_max_seq_length) + '. We shall reduce it.')
                    args.max_seq_length = model_max_seq_length
        if model is not None and hasattr(model, 'for_training'):
            model.for_training(use_gradient_checkpointing=getattr(args, 'gradient_checkpointing', True))
        if 'tokenizer' in locals() and hasattr(tokenizer, 'padding_side'): tokenizer.padding_side = 'right'
        if 'processing_class' in locals():
            if hasattr(processing_class, 'padding_side'): processing_class.padding_side = 'right'
            if hasattr(processing_class, 'tokenizer') and hasattr(processing_class.tokenizer, 'padding_side'): processing_class.tokenizer.padding_side = 'right'
        __tokenizer = processing_class if 'processing_class' in locals() else tokenizer
        from unsloth_zoo.vision_utils import UnslothVisionDataCollator
        if not isinstance(data_collator, UnslothVisionDataCollator):
            if isinstance(data_collator, DataCollatorForSeq2Seq) and 'labels' not in train_dataset.column_names:
                data_collator = TransformersDataCollatorForLanguageModeling(
                    __tokenizer,
                    mlm = False,
                    mlm_probability = 0.0,
                    pad_to_multiple_of = getattr(args, 'pad_to_multiple_of', None),
                )
            elif isinstance(data_collator, TransformersDataCollatorForLanguageModeling) and 'labels' in train_dataset.column_names:
                data_collator = DataCollatorForSeq2Seq(
                    __tokenizer,
                    pad_to_multiple_of = getattr(args, 'pad_to_multiple_of', None),
                )
        else:
            if hasattr(args, 'remove_unused_columns'): args.remove_unused_columns = False
            if hasattr(args, 'dataset_text_field'): args.dataset_text_field = ''
            if hasattr(args, 'dataset_kwargs'): args.dataset_kwargs = {'skip_prepare_dataset': True}
        if not isinstance(data_collator, UnslothVisionDataCollator):
            if not hasattr(__tokenizer, 'pad') and hasattr(__tokenizer, 'tokenizer'):
                if isinstance(data_collator, DataCollatorForSeq2Seq):
                    data_collator = DataCollatorForSeq2Seq(
                        __tokenizer.tokenizer,
                        pad_to_multiple_of = getattr(args, 'pad_to_multiple_of', None),
                    )
                else:
                    data_collator = TransformersDataCollatorForLanguageModeling(
                        __tokenizer.tokenizer,
                        mlm = False,
                        mlm_probability = 0.0,
                        pad_to_multiple_of = getattr(args, 'pad_to_multiple_of', None),
                    )
        other_metrics = []
        
        from unsloth_zoo.logging_utils import PatchRLStatistics
        PatchRLStatistics('dpo_trainer', other_metrics)
        if hasattr(train_dataset, 'column_names'):
            column_names = set(train_dataset.column_names)
            check = ['chosen', 'rejected', 'prompt', 'chosen_input_ids', 'chosen_attention_mask',
                     'chosen_labels', 'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels',
                     'prompt_input_ids', 'prompt_attention_mask']
            if all(x in column_names for x in check):
                train_dataset = train_dataset.remove_columns(['chosen', 'rejected', 'prompt'])
            del check, column_names
        
        # [TODO] Fix up DataParallel multiplying batch sizes
        # [TODO] DDP works, but DP seems to not work? [TODO]
        if getattr(args, "parallel_mode", None) == ParallelMode.NOT_DISTRIBUTED and args.n_gpu > 1:
            if getattr(args, "_n_gpu", 1) != 1:
                args._n_gpu = 1
        if "model" in locals() and hasattr(model, "for_training"):
            model.for_training(use_gradient_checkpointing=getattr(args, 'gradient_checkpointing', True))
        super().__init__(
            model = model,
            ref_model = ref_model,
            args = args,
            data_collator = data_collator,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            processing_class = processing_class,
            compute_metrics = compute_metrics,
            callbacks = callbacks,
            peft_config = peft_config,**kwargs)
        if "model" in locals() and hasattr(model, "for_inference"):
            model.for_inference()
        if hasattr(self, 'neftune_hook_handle'):
            self.neftune_hook_handle.remove()
            if hasattr(self, 'neftune_hook_handle'): del self.neftune_hook_handle
        if getattr(args, 'neftune_noise_alpha', None) is not None:
            model.get_input_embeddings().neftune_noise_alpha = self.neftune_noise_alpha
        pass
        if hasattr(self, 'accelerator'):
            scaler = self.accelerator.scaler
            current_model = model
            while hasattr(current_model, 'model'):
                current_model.accelerator_scaler = scaler
                current_model = current_model.model
            current_model.accelerator_scaler = scaler
        pass
        if hasattr(self, 'train'):
            self.train = MethodType(prepare_for_training_mode(self.__class__.train), self)
        pass
        if hasattr(self, 'llm') and self.llm is not None and hasattr(self.llm, 'get_tokenizer'):
            _vllm_tok = self.llm.get_tokenizer()
            _pc = getattr(self, 'processing_class', None) or getattr(self, 'tokenizer', None)
            if _vllm_tok is not None and _pc is not None and getattr(_pc, 'chat_template', None) is not None and getattr(_vllm_tok, 'chat_template', None) is None:
                _vllm_tok.chat_template = _pc.chat_template
        pass
        
pass


if hasattr(logger, "addFilter"):
    import logging
    class HideLoggingMessage(logging.Filter):
        def __init__(self, text): self.text = text
        def filter(self, x): return not (self.text in x.getMessage())
    pass
    logger.addFilter(HideLoggingMessage("`use_cache=True`"))

