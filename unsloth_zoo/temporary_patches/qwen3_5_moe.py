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
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect

from .common import (
    TEMPORARY_PATCHES,
    torch_compile,
    UNSLOTH_ENABLE_LOGGING,
)
from .utils import (
    patch_function,
    raise_error,
    logger,
)


# ============================================================================
# Grouped GEMM kernel integration for MoE training acceleration
# ============================================================================

from .moe_utils import (
    _check_grouped_gemm_available,
    _TORCH_GROUPED_MM_AVAILABLE,
    forward_native_grouped_mm,
    forward_triton_grouped_gemm,
    forward_native_moe_loop,
    select_moe_backend,
    patch_param_wrapper_for_moe,
)


def patch_qwen3_5_moe():
    try:
        import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe

        transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeSparseMoeBlock
    except Exception as e:
        return raise_error(
            "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeSparseMoeBlock",
            e,
        )

    # Patch ParamWrapper.forward for MoE separated LoRA
    patch_param_wrapper_for_moe()

    # Robust LoRA Extractor for Qwen3.5-MoE
    def _qwen3_5_lora_extractor(self, wrapper, weight_A, weight_B, scaling, num_experts):
        """
        Robust extractor for Qwen3.5-MoE that handles PEFT's dimension layout.

        Expectation for grouped_mm:
        - first_weight:  (E, in_dim, R)   [Input projection to rank]
        - second_weight: (E, R, out_dim)  [Rank projection to output]
        """
        total_rank = weight_A.shape[0]
        rank_per_expert = total_rank // num_experts

        dim_A = weight_A.shape[1]
        dim_B = weight_B.shape[0]

        hidden_dim = None
        intermediate_dim = None
        current = wrapper
        while hasattr(current, "base_layer"):
            current = current.base_layer
            if hasattr(current, "hidden_dim"):
                hidden_dim = current.hidden_dim
            if hasattr(current, "intermediate_dim"):
                intermediate_dim = current.intermediate_dim
            if hasattr(current, "gate_up_proj") and hasattr(current.gate_up_proj, "shape"):
                shape = current.gate_up_proj.shape
                if len(shape) == 3:
                    hidden_dim = shape[2]
                    intermediate_dim = shape[1] // 2

        param_name = getattr(wrapper, "parameter_name", None)

        if param_name == "down_proj" and intermediate_dim is not None and hidden_dim is not None:
            first_weight = weight_B.view(dim_B, num_experts, rank_per_expert)
            first_weight = first_weight.permute(1, 0, 2).contiguous()
            second_weight = weight_A.view(num_experts, rank_per_expert, dim_A)
            return first_weight, second_weight, scaling, num_experts

        elif param_name == "gate_up_proj" and hidden_dim is not None:
            first_weight = weight_B.view(dim_B, num_experts, rank_per_expert)
            first_weight = first_weight.permute(1, 0, 2).contiguous()
            second_weight = weight_A.view(num_experts, rank_per_expert, dim_A)
            return first_weight, second_weight, scaling, num_experts

        if hidden_dim is not None:
            if dim_B == hidden_dim:
                first_weight = weight_B.view(dim_B, num_experts, rank_per_expert)
                first_weight = first_weight.permute(1, 0, 2).contiguous()
                second_weight = weight_A.view(num_experts, rank_per_expert, dim_A)
                return first_weight, second_weight, scaling, num_experts
            elif dim_A == hidden_dim:
                first_weight = weight_A.view(num_experts, rank_per_expert, dim_A)
                first_weight = first_weight.permute(0, 2, 1).contiguous()
                second_weight = weight_B.view(dim_B, num_experts, rank_per_expert)
                second_weight = second_weight.permute(1, 2, 0).contiguous()
                return first_weight, second_weight, scaling, num_experts

        first_weight = weight_A.view(num_experts, rank_per_expert, dim_A)
        first_weight = first_weight.permute(0, 2, 1).contiguous()
        second_weight = weight_B.view(dim_B, num_experts, rank_per_expert)
        second_weight = second_weight.permute(1, 2, 0).contiguous()
        return first_weight, second_weight, scaling, num_experts

    try:
        import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe

        transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeExperts._unsloth_lora_extractor_fn = _qwen3_5_lora_extractor
    except Exception as e:
        if UNSLOTH_ENABLE_LOGGING:
            logger.warning(
                f"Unsloth: Could not register Qwen3_5MoeExperts LoRA extractor: {e}"
            )

    backend = select_moe_backend()

    if backend == "grouped_mm":

        def forward(
            self,
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
        ) -> torch.Tensor:
            return forward_native_grouped_mm(self, hidden_states, top_k_index, top_k_weights)

    elif backend == "unsloth_triton":

        def forward(
            self,
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
        ) -> torch.Tensor:
            return forward_triton_grouped_gemm(self, hidden_states, top_k_index, top_k_weights)

    else:

        @torch.compiler.disable
        def forward(
            self,
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
        ) -> torch.Tensor:
            return forward_native_moe_loop(self, hidden_states, top_k_index, top_k_weights)

    patch_function(
        transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeExperts,
        "forward",
        forward,
    )

    @torch.compiler.disable
    def sparse_moe_block_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() == 3:
            batch_size, sequence_length, hidden_dim = hidden_states.shape
        else:
            total_tokens, hidden_dim = hidden_states.shape
            batch_size = 1
            sequence_length = total_tokens

        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        shared_expert_output = self.shared_expert(hidden_states_reshaped)

        gate_output = self.gate(hidden_states_reshaped)
        if isinstance(gate_output, tuple):
            _, routing_weights, selected_experts = gate_output
        else:
            router_logits = gate_output
            top_k = getattr(self.gate, "top_k", getattr(self, "top_k", 2))
            norm_topk_prob = getattr(self.gate, "norm_topk_prob", getattr(self, "norm_topk_prob", False))
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
            if norm_topk_prob:
                routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
            routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output
        final_hidden_states = final_hidden_states + shared_expert_output

        if hidden_states.dim() == 3:
            return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states

    patch_function(
        transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeSparseMoeBlock,
        "forward",
        sparse_moe_block_forward,
    )

    # Patch Qwen3_5MoeForCausalLM.forward for GRPO training
    try:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeForCausalLM,
            MoeCausalLMOutputWithPast,
        )

        _original_causal_lm_forward = Qwen3_5MoeForCausalLM.forward

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
            RETURN_HIDDEN_STATES = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1"

            if not RETURN_HIDDEN_STATES:
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

            if logits_to_keep != 0:
                slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
                hidden_states = hidden_states[:, slice_indices, :]

            return MoeCausalLMOutputWithPast(
                loss=None,
                aux_loss=None,
                logits=hidden_states,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                router_logits=outputs.router_logits,
            )

        _patched_causal_lm_forward.__qualname__ = _original_causal_lm_forward.__qualname__
        Qwen3_5MoeForCausalLM.forward = _patched_causal_lm_forward
        if UNSLOTH_ENABLE_LOGGING:
            logger.info(
                "Unsloth: Patched Qwen3_5MoeForCausalLM.forward for GRPO hidden states."
            )
    except Exception as e:
        if UNSLOTH_ENABLE_LOGGING:
            logger.warning(
                f"Unsloth: Could not patch Qwen3_5MoeForCausalLM.forward: {e}"
            )


pass
TEMPORARY_PATCHES.append(patch_qwen3_5_moe)
