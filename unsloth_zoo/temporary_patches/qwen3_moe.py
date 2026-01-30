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




def patch_qwen3_moe():
    # https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L213
    # Transformers >= 5       uses self.gate_up_proj = nn.Parameter(...)
    # whilst old transformers uses self.experts = nn.ModuleList(...)

    # Patch ParamWrapper.forward for MoE separated LoRA
    patch_param_wrapper_for_moe()

    try:
        import transformers.models.qwen3_moe.modeling_qwen3_moe
        transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock
    except Exception as e:
        return raise_error("transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock", e)
    old_transformers = True
    try:
        import transformers.models.qwen3_moe.modeling_qwen3_moe
        transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeExperts # New transformers has this
        old_transformers = False
    except Exception as e:
        old_transformers = True

    if old_transformers:
        def old_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            """ """
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)
            # router_logits: (batch * sequence_length, n_experts)
            router_logits = self.gate(hidden_states)

            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            # we cast back to the input dtype
            routing_weights = routing_weights.to(hidden_states.dtype)

            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
            )

            # One hot encode the selected experts to create an expert mask
            # this will be used to easily index which expert is going to be sollicitated
            expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

            # Loop over all available experts in the model and perform the computation on each expert
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in expert_hit:
                expert_layer = self.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

                # Index the correct hidden states and compute the expert hidden state for
                # the current expert. We need to make sure to multiply the output hidden
                # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

                # However `index_add_` only support torch tensors for indexing so we'll use
                # the `top_x` tensor here.
                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
            final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
            return final_hidden_states, router_logits

        @torch_compile(dynamic = True, fullgraph = True)
        def router_forward(self, hidden_states):
            router_logits = self.gate(hidden_states)

            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
                routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
            # we cast back to the input dtype
            routing_weights = routing_weights.to(hidden_states.dtype)
            router_scores = torch.zeros_like(router_logits, dtype = hidden_states.dtype).scatter_(1, selected_experts, routing_weights)
            return router_scores, selected_experts, router_logits

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            """ """
            is_3d = hidden_states.dim() == 3
            if is_3d:
                batch_size, sequence_length, hidden_dim = hidden_states.shape
            else:
                total_tokens, hidden_dim = hidden_states.shape
                batch_size = 1
                sequence_length = total_tokens

            hidden_states = hidden_states.view(-1, hidden_dim)
            router_scores, selected_experts, router_logits = router_forward(self, hidden_states)
            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim), dtype=torch.float32, device=hidden_states.device
            )

            # one hot encode the selected experts to create an expert mask
            # this will be used to easily index which expert is going to be sollicitated
            # Loop over all available experts in the model and perform the computation on each expert
            for expert_idx in range(self.num_experts):
                expert_layer = self.experts[expert_idx]
                token_idx, _ = torch.where(selected_experts == expert_idx)

                # Index the correct hidden states and compute the expert hidden state for
                # the current expert. We need to make sure to multiply the output hidden
                # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
                current_state = hidden_states[token_idx].reshape(-1, hidden_dim)
                current_hidden_states = expert_layer(current_state) * router_scores[token_idx, expert_idx, None]

                # However `index_add_` only support torch tensors for indexing so we'll use
                # the `top_x` tensor here.
                final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(torch.float32))

            if is_3d:
                final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
            return final_hidden_states.to(hidden_states.dtype), router_logits
    else:
    # ====================================================================
        # New transformers (5.0+) with stacked expert weights
        # Uses Triton grouped GEMM kernels for high performance
        # ====================================================================

        # Robust LoRA Extractor for Qwen3-MoE
        def _qwen3_lora_extractor(wrapper, weight_A, weight_B, scaling, num_experts):
            """
            Robust extractor for Qwen3-MoE that detects dimension layouts.

            Expectation for grouped_mm:
            - first_weight:  (E, H, R)   [Input computation]
            - second_weight: (E, R, Out) [Output computation]

            Qwen3-MoE models may have different PEFT wrappings:
            1. Standard: lora_A (in->R) connects to H. Shape (E*R, H).
               Needs: View(E, R, H) -> Permute(0, 2, 1) -> (E, H, R).
            2. Swapped:  lora_B (R->in) connects to H. Shape (H, E*R).
               Needs: View(H, E, R) -> Permute(1, 0, 2) -> (E, H, R).

            This extractor dynamically detects 'H' matching dim to pick the correct path.
            """
            total_rank = weight_A.shape[0]
            rank_per_expert = total_rank // num_experts

            # Get dimensions
            # weight_A: (E*R, dim_A)
            # weight_B: (dim_B, E*R)
            dim_A = weight_A.shape[1]
            dim_B = weight_B.shape[0]

            # Try to get hidden_dim from the experts module
            hidden_dim = None
            experts_module = getattr(wrapper, "base_layer", None)
            # Traverse base_layer until we find the experts module which might have hidden_dim
            current = wrapper
            while hasattr(current, "base_layer"):
                current = current.base_layer
                if hasattr(current, "hidden_dim"):
                    hidden_dim = current.hidden_dim
                    break

            # If hidden_dim found, use it to disambiguate
            # gate_up input is hidden_dim.
            if hidden_dim is not None:
                # Check which weight connects to hidden_dim

                # Case 1: Standard/GLM4-like (lora_A connects to input/hidden_dim)
                # lora_A: (R, hidden_dim) -> weight_A: (E*R, hidden_dim)
                if dim_A == hidden_dim:
                    # first_weight from A
                    # weight_A (E*R, H) -> view(E, R, H) -> permute(0, 2, 1) -> (E, H, R)
                    first_weight = weight_A.view(num_experts, rank_per_expert, dim_A)
                    first_weight = first_weight.permute(0, 2, 1).contiguous()

                    # second_weight from B
                    # weight_B (2I, E*R) -> view(2I, E, R) -> permute(1, 2, 0) -> (E, R, 2I)
                    second_weight = weight_B.view(dim_B, num_experts, rank_per_expert)
                    second_weight = second_weight.permute(1, 2, 0).contiguous()

                    return first_weight, second_weight, scaling, num_experts

                # Case 2: Swapped/Old PEFT (lora_B connects to input/hidden_dim)
                # lora_B: (hidden_dim, R) -> weight_B: (hidden_dim, E*R)
                elif dim_B == hidden_dim:
                     # first_weight from B
                     # weight_B (H, E*R) -> view(H, E, R) -> permute(1, 0, 2) -> (E, H, R)
                     first_weight = weight_B.view(dim_B, num_experts, rank_per_expert)
                     first_weight = first_weight.permute(1, 0, 2).contiguous()

                     # second_weight from A
                     # weight_A (E*R, 2I) -> view(E, R, 2I) -> NO PERMUTE needed if strictly (E, R, out)?
                     # Wait, we need output (E, R, 2I).
                     # weight_A (E*R, 2I). view(E, R, 2I). This is already correct shape?
                     # Let's check dims.
                     second_weight = weight_A.view(num_experts, rank_per_expert, dim_A)
                     # Matches (E, R, out) directly without permute?
                     # Yes, if weight_A is (E*R, out).

                     return first_weight, second_weight, scaling, num_experts

            # Fallback if hidden_dim not found or no match (assume GLM4/Standard logic)
            # This matches the previous logic in moe_utils which works for GLM4
            first_weight = weight_A.view(num_experts, rank_per_expert, dim_A)
            first_weight = first_weight.permute(0, 2, 1).contiguous() # (E, dim_A, R)

            second_weight = weight_B.view(dim_B, num_experts, rank_per_expert)
            second_weight = second_weight.permute(1, 2, 0).contiguous() # (E, R, dim_B)

            return first_weight, second_weight, scaling, num_experts

        transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeExperts._unsloth_lora_extractor_fn = _qwen3_lora_extractor

        backend = select_moe_backend()

        if backend == "grouped_mm":

            def forward(
                self,
                hidden_states: torch.Tensor,
                top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor,
            ) -> torch.Tensor:
                """
                Native Pytorch grouped GEMM MoE forward pass.
                """
                return forward_native_grouped_mm(self, hidden_states, top_k_index, top_k_weights)

        elif backend == "unsloth_triton":
            # Import grouped GEMM interface
            from unsloth.kernels.moe.grouped_gemm.interface import grouped_gemm, supports_tma
            # Import autotune cache
            # from unsloth.kernels.moe.autotune_cache import get_or_autotune_moe_kernels

            def forward(
                self,
                hidden_states: torch.Tensor,
                top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor,
            ) -> torch.Tensor:
                """
                Grouped GEMM MoE forward pass using Triton kernels.

                Uses fused permutation (permute_x for first GEMM, permute_y for second GEMM)
                to minimize memory traffic and achieve high GPU utilization.

                Uses cached kernel configs (created once at start) for efficient operation.
                """
                return forward_triton_grouped_gemm(self, hidden_states, top_k_index, top_k_weights)

        else:
            # Fallback: Pure PyTorch loop-based implementation


            @torch.compiler.disable
            def forward(
                self,
                hidden_states: torch.Tensor,
                top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor,
            ) -> torch.Tensor:
                """
                Loop-based MoE forward pass. Loops over experts that have tokens routed to them.
                Uses @torch.compiler.disable because the loop is data-dependent.
                """
                return forward_native_moe_loop(self, hidden_states, top_k_index, top_k_weights)

        # SparseMoeBlock forward is disabled from compilation due to dynamic routing
        @torch.compiler.disable
        def sparse_moe_block_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

            """
            Forward for Qwen3MoeSparseMoeBlock in new transformers (v5+).

            In v5, self.gate is Qwen3MoeTopKRouter which returns:
                (router_logits, router_scores, router_indices)
            where router_scores are already normalized if norm_topk_prob=True.

            Returns only the hidden states (router_logits are recorded separately).
            """
            if hidden_states.dim() == 3:
                batch_size, sequence_length, hidden_dim = hidden_states.shape
            else:
                total_tokens, hidden_dim = hidden_states.shape
                batch_size = 1
                sequence_length = total_tokens

            hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

            # In v5, self.gate is Qwen3MoeTopKRouter
            # It returns (router_logits, router_scores, router_indices)
            # router_scores are already softmax'd and normalized if norm_topk_prob=True
            gate_output = self.gate(hidden_states_reshaped)

            if isinstance(gate_output, tuple):
                # New transformers v5: (router_logits, router_scores, router_indices)
                _, routing_weights, selected_experts = gate_output
            else:
                # Fallback: old-style gate that returns just logits
                router_logits = gate_output
                top_k = getattr(self.gate, 'top_k', getattr(self, 'top_k', 2))
                norm_topk_prob = getattr(self.gate, 'norm_topk_prob', getattr(self, 'norm_topk_prob', False))

                routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
                routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
                if norm_topk_prob:
                    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
                routing_weights = routing_weights.to(hidden_states.dtype)

            final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)

            if hidden_states.dim() == 3:
                return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
            return final_hidden_states

    # For old transformers, patch Qwen3MoeSparseMoeBlock
    # For new transformers, patch Qwen3MoeExperts (which has the expert loop)
    if old_transformers:
        patch_function(transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock, "forward", forward)
    else:
        patch_function(transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeExperts, "forward", forward)
        patch_function(transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock, "forward", sparse_moe_block_forward)

    # ====================================================================
    # Patch Qwen3MoeForCausalLM.forward for GRPO training
    # When UNSLOTH_RETURN_HIDDEN_STATES=1, return hidden_states instead of logits
    # ====================================================================
    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeForCausalLM,
            MoeCausalLMOutputWithPast,
        )

        _original_causal_lm_forward = Qwen3MoeForCausalLM.forward

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
            if logits_to_keep != 0:
                slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
                hidden_states = hidden_states[:, slice_indices, :]

            # Return hidden_states as "logits" for GRPO to use
            return MoeCausalLMOutputWithPast(
                loss=None,
                aux_loss=None,
                logits=hidden_states,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                router_logits=outputs.router_logits,
            )

        Qwen3MoeForCausalLM.forward = _patched_causal_lm_forward
        if UNSLOTH_ENABLE_LOGGING:
            logger.info("Unsloth: Patched Qwen3MoeForCausalLM.forward for GRPO hidden states.")
    except Exception as e:
        if UNSLOTH_ENABLE_LOGGING:
            logger.warning(f"Unsloth: Could not patch Qwen3MoeForCausalLM.forward: {e}")

pass
TEMPORARY_PATCHES.append(patch_qwen3_moe)
