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
from .moe_utils import (
    _check_grouped_gemm_available,
    _TORCH_GROUPED_MM_AVAILABLE,
    forward_native_grouped_mm,
    forward_triton_grouped_gemm,
    forward_native_moe_loop,
    select_moe_backend,
    patch_param_wrapper_for_moe,
)


def patch_qwen3_vl_moe():
    # All Unsloth Zoo code licensed under AGPL3

    # Patch ParamWrapper.forward for MoE separated LoRA
    patch_param_wrapper_for_moe()

    try:
        import transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe

        transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextSparseMoeBlock
    except Exception as e:
        return raise_error(
            "transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextSparseMoeBlock",
            e,
        )

    old_transformers = True
    try:
        import transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe

        transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextExperts
        old_transformers = False
    except Exception as e:
        old_transformers = True

    if old_transformers:
        # Fallback for older transformers if they exist (unlikely for Qwen3VL MoE but good for robustness)
        # Assuming typical sparse block structure if Experts class is missing
        @torch.compiler.disable
        def old_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            """ """
            if hidden_states.dim() == 3:
                batch_size, sequence_length, hidden_dim = hidden_states.shape
            else:
                total_tokens, hidden_dim = hidden_states.shape
                batch_size = 1
                sequence_length = total_tokens
            hidden_states = hidden_states.view(-1, hidden_dim)
            # router_logits: (batch * sequence_length, n_experts)
            router_logits = self.gate(hidden_states)

            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(
                routing_weights, self.top_k, dim=-1
            )
            if self.norm_topk_prob:
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            # we cast back to the input dtype
            routing_weights = routing_weights.to(hidden_states.dtype)

            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

            # One hot encode the selected experts to create an expert mask
            expert_mask = torch.nn.functional.one_hot(
                selected_experts, num_classes=self.num_experts
            ).permute(2, 1, 0)

            # Loop over all available experts in the model and perform the computation on each expert
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in expert_hit:
                expert_layer = self.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = (
                    expert_layer(current_state) * routing_weights[top_x, idx, None]
                )

                final_hidden_states.index_add_(
                    0, top_x, current_hidden_states.to(hidden_states.dtype)
                )
            final_hidden_states = final_hidden_states.reshape(
                batch_size, sequence_length, hidden_dim
            )
            return final_hidden_states, router_logits

        @torch_compile(dynamic=True, fullgraph=True)
        def router_forward(self, hidden_states):
            router_logits = self.gate(hidden_states)

            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
            routing_weights, selected_experts = torch.topk(
                routing_weights, self.top_k, dim=-1
            )
            if self.norm_topk_prob:
                routing_weights = routing_weights / routing_weights.sum(
                    dim=-1, keepdim=True
                )
            # we cast back to the input dtype
            routing_weights = routing_weights.to(hidden_states.dtype)
            router_scores = torch.zeros_like(
                router_logits, dtype=hidden_states.dtype
            ).scatter_(1, selected_experts, routing_weights)
            return router_scores, selected_experts, router_logits

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            """ """
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)

            router_scores, selected_experts, router_logits = router_forward(
                self, hidden_states
            )
            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim),
                dtype=torch.float32,
                device=hidden_states.device,
            )

            # Loop over all available experts
            for expert_idx in range(self.num_experts):
                expert_layer = self.experts[expert_idx]
                token_idx, _ = torch.where(selected_experts == expert_idx)

                current_state = hidden_states[token_idx].reshape(-1, hidden_dim)
                current_hidden_states = (
                    expert_layer(current_state)
                    * router_scores[token_idx, expert_idx, None]
                )

                final_hidden_states.index_add_(
                    0, token_idx, current_hidden_states.to(torch.float32)
                )
            final_hidden_states = final_hidden_states.reshape(
                batch_size, sequence_length, hidden_dim
            )
            return final_hidden_states.to(hidden_states.dtype), router_logits

    else:
        # ====================================================================
        # New transformers with stacked expert weights
        # ====================================================================

        # Qwen3-VL-MoE checkpoints store weights in grouped_mm format (transposed):
        #   gate_up_proj: (E, H, 2*I)  - ready for X @ W in grouped_mm
        #   down_proj:    (E, I, H)    - ready for X @ W in grouped_mm
        #
        # But transformers defines them in F.linear format:
        #   gate_up_proj: (E, 2*I, H)  - for F.linear: out = x @ weight.T
        #   down_proj:    (E, H, I)
        #
        # We patch __init__ to match checkpoint format so loading works correctly.
        # The forward() then uses these weights directly with grouped_mm.

        from transformers.activations import ACT2FN

        def patched_experts_init(self, config):
            """
            Patched __init__ for Qwen3VLMoeTextExperts.

            Creates weight parameters in grouped_mm format to match checkpoint:
            - gate_up_proj: (E, H, 2*I) instead of (E, 2*I, H)
            - down_proj: (E, I, H) instead of (E, H, I)

            This ensures checkpoint loading works correctly, and the forward
            can use weights directly with torch._grouped_mm without transposition.
            """
            # All Unsloth Zoo code licensed under AGPL3

            super(
                transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextExperts,
                self,
            ).__init__()

            self.num_experts = config.num_experts
            self.hidden_dim = config.hidden_size
            self.intermediate_dim = config.moe_intermediate_size

            # Weights in grouped_mm format (transposed from F.linear format)
            # gate_up_proj: (E, H, 2*I) for X @ W where X is (N, H)
            self.gate_up_proj = nn.Parameter(
                torch.empty(
                    self.num_experts, self.hidden_dim, 2 * self.intermediate_dim
                )
            )
            # down_proj: (E, I, H) for X @ W where X is (N, I)
            self.down_proj = nn.Parameter(
                torch.empty(self.num_experts, self.intermediate_dim, self.hidden_dim)
            )
            self.act_fn = ACT2FN[config.hidden_act]

            # Mark that weights are already in grouped_mm format (no transpose needed)
            self._unsloth_grouped_mm_format = True

        # Patch __init__ before any model instantiation
        patch_function(
            transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextExperts,
            "__init__",
            patched_experts_init,
            force=True,
        )

        # ====================================================================
        # Define LoRA extraction function for Qwen3-VL-MoE (Transposed Format)
        # ====================================================================
        def _qwen3_vl_lora_extractor(wrapper, weight_A, weight_B, scaling, num_experts):
            """
            Custom LoRA extractor for Qwen3-VL-MoE which uses Transposed Grouped MM format.
            Base weights are (E, in_dim, out_dim).

            Expectation for grouped_mm (Transposed):
            - first (Input):  (E, H, R)
            - second (Output): (E, R, Out)

            Qwen3-VL Weights:
            - weight_A (E*R, H): Input projection. Reshape(E, R, H) -> Permute(0, 2, 1) -> (E, H, R).
            - weight_B (Out, E*R): Output projection. Reshape(Out, E, R) -> Permute(1, 2, 0) -> (E, R, Out).
            """
            # All Unsloth Zoo code licensed under AGPL3

            total_rank = weight_A.shape[0]
            rank_per_expert = total_rank // num_experts
            dim1 = weight_A.shape[1]
            dim2 = weight_B.shape[0]

            # TRANSPOSED FORMAT logic from old moe_utils.py
            # first_weight: (E, in_dim, R) - from lora_A
            # second_weight: (E, R, out_dim) - from lora_B

            first_weight = weight_A.view(num_experts, rank_per_expert, dim1)
            first_weight = first_weight.permute(0, 2, 1).contiguous()  # (E, in_dim, R)

            second_weight = weight_B.view(dim2, num_experts, rank_per_expert)
            second_weight = second_weight.permute(1, 2, 0).contiguous()  # (E, R, out_dim)

            return first_weight, second_weight, scaling, num_experts

        # Register the extractor on the Experts class
        transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextExperts._unsloth_lora_extractor_fn = _qwen3_vl_lora_extractor


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
                Uses torch._grouped_mm which is significantly faster than loop and works without Triton dependencies.
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

                Uses fused permutation (permute_x for first GEMM, permute_y for second GEMM)
                to minimize memory traffic and achieve high GPU utilization.

                Uses cached kernel configs (created once at start) for efficient operation.
                """
                return forward_triton_grouped_gemm(
                    self, hidden_states, top_k_index, top_k_weights
                )

        else:
            # Fallback
            @torch.compiler.disable
            def forward(
                self,
                hidden_states: torch.Tensor,
                top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor,
            ) -> torch.Tensor:
                # Same loop fallback logic as Qwen3MoeExperts
                return forward_native_moe_loop(
                    self, hidden_states, top_k_index, top_k_weights
                )

        # SparseMoeBlock forward is disabled from compilation due to dynamic routing
        @torch.compiler.disable
        def sparse_moe_block_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            """
            Forward for Qwen3VLMoeTextSparseMoeBlock in new transformers (v5+).

            In v5, self.gate is Qwen3VLMoeTextTopKRouter which returns:
                (router_logits, router_scores, router_indices)
            where router_scores are already normalized.
            """
            # All Unsloth Zoo code licensed under AGPL3

            if hidden_states.dim() == 3:
                batch_size, sequence_length, hidden_dim = hidden_states.shape
            else:
                total_tokens, hidden_dim = hidden_states.shape
                batch_size = 1
                sequence_length = total_tokens

            hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

            # In v5, self.gate is Qwen3VLMoeTextTopKRouter
            # It returns (router_logits, router_scores, router_indices)
            gate_output = self.gate(hidden_states_reshaped)

            if isinstance(gate_output, tuple):
                # New transformers v5: (router_logits, router_scores, router_indices)
                _, routing_weights, selected_experts = gate_output
            else:
                # Fallback: old-style gate that returns just logits
                router_logits = gate_output
                top_k = getattr(self.gate, "top_k", getattr(self, "top_k", 2))
                norm_topk_prob = getattr(
                    self.gate, "norm_topk_prob", getattr(self, "norm_topk_prob", True)
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

            if hidden_states.dim() == 3:
                return final_hidden_states.reshape(
                    batch_size, sequence_length, hidden_dim
                )
            return final_hidden_states

    if old_transformers:
        patch_function(
            transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextSparseMoeBlock,
            "forward",
            forward,
        )
    else:
        # __init__ was patched above to create weights in grouped_mm format
        # matching the Qwen3-VL-MoE checkpoint format:
        #   gate_up_proj: (E, H, 2*I)  - for X @ W in grouped_mm
        #   down_proj:    (E, I, H)    - for X @ W in grouped_mm
        #
        # The forward() uses these weights directly with torch._grouped_mm.
        # LoRA extraction in moe_utils must account for this transposed layout.

        patch_function(
            transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextExperts,
            "forward",
            forward,
            force=True,
        )
        patch_function(
            transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextSparseMoeBlock,
            "forward",
            sparse_moe_block_forward,
            force=True,
        )

    # ====================================================================
    # Patch Qwen3VLMoeForConditionalGeneration.forward for GRPO training
    # When UNSLOTH_RETURN_HIDDEN_STATES=1, return hidden_states instead of logits
    # ====================================================================
    try:
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
            Qwen3VLMoeForConditionalGeneration,
            Qwen3VLMoeCausalLMOutputWithPast,
        )

        _original_causal_lm_forward = Qwen3VLMoeForConditionalGeneration.forward

        def _patched_causal_lm_forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            labels=None,
            pixel_values=None,
            pixel_values_videos=None,
            image_grid_thw=None,
            video_grid_thw=None,
            cache_position=None,
            logits_to_keep=0,
            **kwargs,
        ):
            RETURN_HIDDEN_STATES = (
                os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1"
            )

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
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    cache_position=cache_position,
                    logits_to_keep=logits_to_keep,
                    **kwargs,
                )

            # RETURN_HIDDEN_STATES mode - return hidden_states instead of logits
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                **kwargs,
            )

            hidden_states = outputs[0]

            # Apply slice_indices to hidden_states (same indexing as for logits)
            slice_indices = (
                slice(-logits_to_keep, None)
                if isinstance(logits_to_keep, int)
                else logits_to_keep
            )
            # Return hidden_states as "logits" for GRPO to use
            logits = hidden_states[:, slice_indices, :]

            return Qwen3VLMoeCausalLMOutputWithPast(
                loss=None,
                aux_loss=None,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                rope_deltas=outputs.rope_deltas,
            )

        Qwen3VLMoeForConditionalGeneration.forward = _patched_causal_lm_forward
        if UNSLOTH_ENABLE_LOGGING:
            logger.info(
                "Unsloth: Patched Qwen3VLMoeForConditionalGeneration.forward for GRPO hidden states."
            )
    except Exception as e:
        if UNSLOTH_ENABLE_LOGGING:
            logger.warning(
                f"Unsloth: Could not patch Qwen3VLMoeForConditionalGeneration.forward: {e}"
            )


TEMPORARY_PATCHES.append(patch_qwen3_vl_moe)
