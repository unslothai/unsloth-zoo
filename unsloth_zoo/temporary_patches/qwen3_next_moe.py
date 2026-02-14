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


def patch_qwen3_next_moe():
    try:
        import transformers.models.qwen3_next.modeling_qwen3_next

        transformers.models.qwen3_next.modeling_qwen3_next.Qwen3NextSparseMoeBlock
    except Exception as e:
        return raise_error(
            "transformers.models.qwen3_next.modeling_qwen3_next.Qwen3NextSparseMoeBlock",
            e,
        )

    # Patch ParamWrapper.forward for MoE separated LoRA
    patch_param_wrapper_for_moe()

    # Check if using new transformers (5.0+) with stacked expert weights
    new_transformers = True
    try:
        import transformers.models.qwen3_next.modeling_qwen3_next

        transformers.models.qwen3_next.modeling_qwen3_next.Qwen3NextExperts
    except Exception as e:
        new_transformers = False

    # Robust LoRA Extractor for Qwen3-Next MoE
    def _qwen3_next_lora_extractor(
        self, wrapper, weight_A, weight_B, scaling, num_experts
    ):
        """
        Robust extractor for Qwen3-Next MoE that handles PEFT's dimension layout.

        Expectation for grouped_mm:
        - first_weight:  (E, in_dim, R)   [Input projection to rank]
        - second_weight: (E, R, out_dim)  [Rank projection to output]

        PEFT wraps 3D parameters (E, out_dim, in_dim) with swapped in_features/out_features.
        This means the LoRA A/B weights have dimensions that don't match the actual
        computation flow. We need to detect the parameter type and reshape accordingly.

        For gate_up_proj: input=hidden_dim, output=2*intermediate_dim
        For down_proj: input=intermediate_dim, output=hidden_dim
        """
        total_rank = weight_A.shape[0]
        rank_per_expert = total_rank // num_experts

        dim_A = weight_A.shape[1]
        dim_B = weight_B.shape[0]

        # Get model dimensions from the experts module
        hidden_dim = None
        intermediate_dim = None
        current = wrapper
        while hasattr(current, "base_layer"):
            current = current.base_layer
            if hasattr(current, "hidden_dim"):
                hidden_dim = current.hidden_dim
            if hasattr(current, "intermediate_dim"):
                intermediate_dim = current.intermediate_dim
            # For Qwen3NextExperts, check for gate_up_proj shape
            if hasattr(current, "gate_up_proj") and hasattr(
                current.gate_up_proj, "shape"
            ):
                shape = current.gate_up_proj.shape
                if len(shape) == 3:
                    # gate_up_proj: (E, 2*intermediate_dim, hidden_dim)
                    hidden_dim = shape[2]
                    intermediate_dim = shape[1] // 2

        # Get parameter name to determine which projection we're handling
        param_name = getattr(wrapper, "parameter_name", None)

        # Handle based on parameter type
        if (
            param_name == "down_proj"
            and intermediate_dim is not None
            and hidden_dim is not None
        ):
            # down_proj: input=intermediate_dim, output=hidden_dim
            # PEFT set up LoRA backwards:
            #   lora_A: (E*R, hidden_dim)     - actually connects to output
            #   lora_B: (intermediate_dim, E*R) - actually connects to input

            # first_weight should be (E, intermediate_dim, R) for input
            # Use weight_B which has intermediate_dim
            first_weight = weight_B.view(dim_B, num_experts, rank_per_expert)
            first_weight = first_weight.permute(1, 0, 2).contiguous()

            # second_weight should be (E, R, hidden_dim) for output
            # Use weight_A which has hidden_dim
            second_weight = weight_A.view(num_experts, rank_per_expert, dim_A)

            return first_weight, second_weight, scaling, num_experts

        elif param_name == "gate_up_proj" and hidden_dim is not None:
            # gate_up_proj: input=hidden_dim, output=2*intermediate_dim
            # PEFT set up:
            #   lora_A: (E*R, 2*intermediate_dim) - connects to output
            #   lora_B: (hidden_dim, E*R)         - connects to input
            # Also swapped! Use B for first_weight, A for second_weight

            # first_weight should be (E, hidden_dim, R) for input
            # weight_B: (hidden_dim, E*R) -> view(H, E, R) -> permute(1,0,2) -> (E, H, R)
            first_weight = weight_B.view(dim_B, num_experts, rank_per_expert)
            first_weight = first_weight.permute(1, 0, 2).contiguous()

            # second_weight should be (E, R, 2*intermediate_dim) for output
            # weight_A: (E*R, 2*I) -> view(E, R, 2*I) -> (E, R, 2*I)
            second_weight = weight_A.view(num_experts, rank_per_expert, dim_A)

            return first_weight, second_weight, scaling, num_experts

        # Fallback: try dimension-based detection
        if hidden_dim is not None:
            # Check if B connects to hidden_dim (swapped case)
            if dim_B == hidden_dim:
                # first_weight from B
                first_weight = weight_B.view(dim_B, num_experts, rank_per_expert)
                first_weight = first_weight.permute(1, 0, 2).contiguous()

                # second_weight from A
                second_weight = weight_A.view(num_experts, rank_per_expert, dim_A)

                return first_weight, second_weight, scaling, num_experts

            # Check if A connects to hidden_dim (standard case)
            elif dim_A == hidden_dim:
                # first_weight from A
                first_weight = weight_A.view(num_experts, rank_per_expert, dim_A)
                first_weight = first_weight.permute(0, 2, 1).contiguous()

                # second_weight from B
                second_weight = weight_B.view(dim_B, num_experts, rank_per_expert)
                second_weight = second_weight.permute(1, 2, 0).contiguous()

                return first_weight, second_weight, scaling, num_experts

        # Final fallback: assume standard layout
        first_weight = weight_A.view(num_experts, rank_per_expert, dim_A)
        first_weight = first_weight.permute(0, 2, 1).contiguous()

        second_weight = weight_B.view(dim_B, num_experts, rank_per_expert)
        second_weight = second_weight.permute(1, 2, 0).contiguous()

        return first_weight, second_weight, scaling, num_experts

    # Register the LoRA extractor on Qwen3NextExperts
    try:
        import transformers.models.qwen3_next.modeling_qwen3_next

        transformers.models.qwen3_next.modeling_qwen3_next.Qwen3NextExperts._unsloth_lora_extractor_fn = _qwen3_next_lora_extractor
    except Exception as e:
        if UNSLOTH_ENABLE_LOGGING:
            logger.warning(
                f"Unsloth: Could not register Qwen3NextExperts LoRA extractor: {e}"
            )

    backend = select_moe_backend()

    if backend == "grouped_mm":

        def forward(
            self,
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
        ) -> torch.Tensor:
            """
            Native Pytorch grouped GEMM MoE forward pass for Qwen3NextExperts.
            """
            return forward_native_grouped_mm(
                self, hidden_states, top_k_index, top_k_weights
            )

    elif backend == "unsloth_triton":

        def forward(
            self,
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
        ) -> torch.Tensor:
            """
            Grouped GEMM MoE forward pass using Triton kernels.
            """
            return forward_triton_grouped_gemm(
                self, hidden_states, top_k_index, top_k_weights
            )

    else:

        @torch.compiler.disable
        def forward(
            self,
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
        ) -> torch.Tensor:
            """
            Loop-based MoE forward pass.
            """
            return forward_native_moe_loop(
                self, hidden_states, top_k_index, top_k_weights
            )

    # Patch Qwen3NextExperts.forward (the routed experts)
    patch_function(
        transformers.models.qwen3_next.modeling_qwen3_next.Qwen3NextExperts,
        "forward",
        forward,
    )

    @torch.compiler.disable
    def sparse_moe_block_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward for Qwen3NextSparseMoeBlock.

        In v5, self.gate is Qwen3NextTopKRouter which returns:
            (router_logits, router_scores, router_indices)
        where router_scores are already normalized if norm_topk_prob=True.

        Returns only the hidden states.
        """
        # This Unsloth Zoo code section is licensed under AGPL3

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
            # New transformers v5: (router_logits, router_scores, router_indices)
            _, routing_weights, selected_experts = gate_output
        else:
            # Fallback: old-style gate that returns just logits
            router_logits = gate_output
            top_k = getattr(self.gate, "top_k", getattr(self, "top_k", 2))
            norm_topk_prob = getattr(
                self.gate, "norm_topk_prob", getattr(self, "norm_topk_prob", False)
            )

            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(
                routing_weights, top_k, dim=-1
            )
            if norm_topk_prob:
                routing_weights = routing_weights / routing_weights.sum(
                    dim=-1, keepdim=True
                )
            routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = self.experts(
            hidden_states_reshaped, selected_experts, routing_weights
        )

        shared_expert_output = (
            F.sigmoid(self.shared_expert_gate(hidden_states_reshaped))
            * shared_expert_output
        )

        final_hidden_states = final_hidden_states + shared_expert_output

        if hidden_states.dim() == 3:
            return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states

    patch_function(
        transformers.models.qwen3_next.modeling_qwen3_next.Qwen3NextSparseMoeBlock,
        "forward",
        sparse_moe_block_forward,
    )

    # Patch Qwen3NextForCausalLM.forward for GRPO training
    try:
        from transformers.models.qwen3_next.modeling_qwen3_next import (
            Qwen3NextForCausalLM,
        )

        _original_causal_lm_forward = Qwen3NextForCausalLM.forward

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
            RETURN_HIDDEN_STATES = (
                os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1"
            )

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

            # Apply slice_indices to hidden_states
            if logits_to_keep != 0:
                slice_indices = (
                    slice(-logits_to_keep, None)
                    if isinstance(logits_to_keep, int)
                    else logits_to_keep
                )
                hidden_states = hidden_states[:, slice_indices, :]

            # Return hidden_states as "logits" for GRPO
            from transformers.modeling_outputs import CausalLMOutputWithPast

            return CausalLMOutputWithPast(
                loss=None,
                logits=hidden_states,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        _patched_causal_lm_forward.__qualname__ = (
            _original_causal_lm_forward.__qualname__
        )
        Qwen3NextForCausalLM.forward = _patched_causal_lm_forward
        if UNSLOTH_ENABLE_LOGGING:
            logger.info(
                "Unsloth: Patched Qwen3NextForCausalLM.forward for GRPO hidden states."
            )
    except Exception as e:
        if UNSLOTH_ENABLE_LOGGING:
            logger.warning(
                f"Unsloth: Could not patch Qwen3NextForCausalLM.forward: {e}"
            )


pass
TEMPORARY_PATCHES.append(patch_qwen3_next_moe)
