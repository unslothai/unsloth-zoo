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

from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import sys
from .common import (
    TEMPORARY_PATCHES,
    torch_compile,
    _torch_compile,
    get_torch_compile_options,
    UNSLOTH_ENABLE_LOGGING,
)
from .utils import (
    patch_function,
    patch_function_past_key_values,
    dedent,
    KWARGS_TYPE,
    raise_error,
    logger,
    Cache,
    process_return,
)
from ..hf_utils import dtype_from_config
from .moe_utils import (
    patch_param_wrapper_for_moe,
)


def patch_deepseek_v3():
    """
    Patches DeepSeekV3 MoE to support Split LoRA using grouped GEMM.
    """
    # This Unsloth Zoo code section is licensed under AGPL3

    # Try to import the DeepSeekV3 MoE classes
    try:
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
            DeepseekV3NaiveMoe,
            DeepseekV3MoE,
            DeepseekV3TopkRouter,
            DeepseekV3Config,
        )
    except Exception as e:
        # DeepSeekV3 not available yet
        return

    # Check if already patched
    if hasattr(DeepseekV3NaiveMoe, "_unsloth_already_patched"):
        return

    # Patch PEFT ParamWrapper for separated LoRA weights
    patch_param_wrapper_for_moe()

    # ====================================================================
    # Define LoRA extraction function for DeepSeekV3 (Standard Format)
    # ====================================================================
    def _deepseek_v3_lora_extractor(wrapper, weight_A, weight_B, scaling, num_experts):
        """
        Custom LoRA extractor for DeepSeekV3.

        DeepSeekV3 expert weights are stored as (E, out_dim, in_dim) and PEFT's ParamWrapper
        treats dim1 as in_features and dim2 as out_features. For correct separated LoRA
        (X @ first @ second), we need to pick the weight that connects to the actual input dim.
        """
        total_rank = weight_A.shape[0]
        rank_per_expert = total_rank // num_experts
        dim_A = weight_A.shape[1]
        dim_B = weight_B.shape[0]

        input_dim = None
        if hasattr(wrapper, "parameter_name"):
            if wrapper.parameter_name == "gate_up_proj":
                base = wrapper.get_base_layer() if hasattr(wrapper, "get_base_layer") else None
                input_dim = getattr(base, "hidden_dim", None)
            elif wrapper.parameter_name == "down_proj":
                base = wrapper.get_base_layer() if hasattr(wrapper, "get_base_layer") else None
                input_dim = getattr(base, "intermediate_dim", None)

        if input_dim is None:
            base = wrapper.get_base_layer() if hasattr(wrapper, "get_base_layer") else None
            input_dim = getattr(base, "hidden_dim", None)

        # If lora_A connects to input_dim: standard (A then B)
        if input_dim is not None and dim_A == input_dim:
            first_weight = weight_A.view(num_experts, rank_per_expert, dim_A)
            first_weight = first_weight.permute(0, 2, 1).contiguous()  # (E, input_dim, R)
            second_weight = weight_B.view(dim_B, num_experts, rank_per_expert)
            second_weight = second_weight.permute(1, 2, 0).contiguous()  # (E, R, out_dim)
            return first_weight, second_weight, scaling, num_experts

        # If lora_B connects to input_dim: swapped (B then A)
        if input_dim is not None and dim_B == input_dim:
            first_weight = weight_B.view(dim_B, num_experts, rank_per_expert)
            first_weight = first_weight.permute(1, 0, 2).contiguous()  # (E, input_dim, R)
            second_weight = weight_A.view(num_experts, rank_per_expert, dim_A).contiguous()  # (E, R, out_dim)
            return first_weight, second_weight, scaling, num_experts

        # Fallback: standard (A then B)
        first_weight = weight_A.view(num_experts, rank_per_expert, dim_A)
        first_weight = first_weight.permute(0, 2, 1).contiguous()
        second_weight = weight_B.view(dim_B, num_experts, rank_per_expert)
        second_weight = second_weight.permute(1, 2, 0).contiguous()
        return first_weight, second_weight, scaling, num_experts

    # Register the extractor on the NaiveMoe class (avoid binding as instance method)
    DeepseekV3NaiveMoe._unsloth_lora_extractor_fn = staticmethod(_deepseek_v3_lora_extractor)
    # Also mark the model type for weight preprocessing
    DeepseekV3NaiveMoe._unsloth_model_type = "deepseek_v3"
    DeepseekV3NaiveMoe._unsloth_already_patched = True

    # ====================================================================
    # Patch DeepseekV3NaiveMoe.forward to use backend dispatch in moe_utils
    # ====================================================================

    @torch.compiler.disable
    def naive_moe_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Patched forward for Expert layer.
        Dispatches to moe_utils backend selection.
        """
        from unsloth_zoo.temporary_patches.moe_utils import forward_moe_backend
        return forward_moe_backend(self, hidden_states, top_k_index, top_k_weights)

    # Apply patch to DeepseekV3NaiveMoe
    DeepseekV3NaiveMoe.forward = naive_moe_forward
    patch_function(DeepseekV3NaiveMoe, "forward", naive_moe_forward)

    # ====================================================================
    # Patch DeepseekV3MoE.forward to mark model type
    # ====================================================================

    def patched_moe_forward(self, hidden_states):
        """
        Patched forward that adds model type marker for proper LoRA extraction.
        """
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        # Mark the experts module for proper LoRA extraction
        self.experts._unsloth_model_type = "deepseek_v3"

        hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(
            *orig_shape
        )
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states

    # Apply patch to DeepseekV3MoE
    DeepseekV3MoE.forward = patched_moe_forward
    patch_function(DeepseekV3MoE, "forward", patched_moe_forward)

    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Patched DeepSeekV3 MoE for Split LoRA support.")

    # ====================================================================
    # Patch DeepseekV3ForCausalLM.forward for GRPO training
    # When UNSLOTH_RETURN_HIDDEN_STATES=1, return hidden_states instead of logits
    # ====================================================================
    try:
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
            DeepseekV3ForCausalLM,
            CausalLMOutputWithPast,
        )

        _original_causal_lm_forward = DeepseekV3ForCausalLM.forward

        def _patched_causal_lm_forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_router_logits=None,
            cache_position=None,
            logits_to_keep=0,
            **kwargs,
        ):
            # This Unsloth Zoo code section is licensed under AGPL3

            RETURN_HIDDEN_STATES = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1"

            if not RETURN_HIDDEN_STATES:
                # Normal forward pass
                return _original_causal_lm_forward(
                    self,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_router_logits=output_router_logits,
                    cache_position=cache_position,
                    logits_to_keep=logits_to_keep,
                    **kwargs,
                )

            # RETURN_HIDDEN_STATES mode - return hidden_states instead of logits
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_router_logits=output_router_logits,
                cache_position=cache_position,
                **kwargs,
            )

            hidden_states = outputs.last_hidden_state

            # Apply slice_indices to hidden_states (same indexing as for logits)

            # DeepSeekV3 implementation of logits_to_keep handling:
            if logits_to_keep != 0:
                slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
                hidden_states = hidden_states[:, slice_indices, :]

            # Return hidden_states as "logits" for GRPO to use
            return CausalLMOutputWithPast(
                loss=None,
                logits=hidden_states, # Return hidden states here!
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                router_logits=outputs.router_logits,
            )

        # Preserve __qualname__ so _unsloth_get_batch_samples can detect
        # this is a CausalLM forward and compute num_items_in_batch properly.
        _patched_causal_lm_forward.__qualname__ = _original_causal_lm_forward.__qualname__
        DeepseekV3ForCausalLM.forward = _patched_causal_lm_forward
        patch_function(DeepseekV3ForCausalLM, "forward", _patched_causal_lm_forward)

        if UNSLOTH_ENABLE_LOGGING:
            logger.info("Unsloth: Patched DeepSeekV3ForCausalLM.forward for GRPO hidden states.")
    except Exception as e:
        if UNSLOTH_ENABLE_LOGGING:
            logger.warning(f"Unsloth: Could not patch DeepSeekV3ForCausalLM.forward: {e}")

    return True


# Register the patch - it will be called when unsloth is imported
TEMPORARY_PATCHES.append(patch_deepseek_v3)
