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
    patch_param_wrapper_for_moe,
    forward_moe_backend,
)


def _make_qwen_moe_lora_extractor():
    def _qwen_moe_lora_extractor(self, wrapper, weight_A, weight_B, scaling, num_experts):
        """
        Robust extractor for Qwen-family MoE that handles PEFT's dimension layout.

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

    return _qwen_moe_lora_extractor


def _make_qwen_moe_experts_forward(module_name: Optional[str] = None):
    @torch.compiler.disable
    def qwen_moe_experts_forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        return forward_moe_backend(self, hidden_states, top_k_index, top_k_weights)
    forward = qwen_moe_experts_forward

    if module_name is not None:
        forward.__module__ = module_name
    return forward


def _make_qwen_moe_sparse_moe_block_forward(use_shared_expert: bool, module_name: Optional[str] = None):
    @torch.compiler.disable
    def sparse_moe_block_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        use_shared_expert = hasattr(self, "shared_expert") and hasattr(self, "shared_expert_gate")
        if hidden_states.dim() == 3:
            batch_size, sequence_length, hidden_dim = hidden_states.shape
        else:
            total_tokens, hidden_dim = hidden_states.shape
            batch_size = 1
            sequence_length = total_tokens

        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        shared_expert_output = None
        if use_shared_expert:
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

        if use_shared_expert:
            shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output
            final_hidden_states = final_hidden_states + shared_expert_output

        if hidden_states.dim() == 3:
            return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states

    if module_name is not None:
        sparse_moe_block_forward.__module__ = module_name
    return sparse_moe_block_forward


def _patch_causal_lm_forward_for_hidden_states(
    causal_lm_cls,
    output_cls,
    model_label: str,
    extra_output_kwargs_fn=None,
):
    _original_causal_lm_forward = causal_lm_cls.forward

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

        extra_output_kwargs = {}
        if extra_output_kwargs_fn is not None:
            extra_output_kwargs = extra_output_kwargs_fn(outputs) or {}

        return output_cls(
            loss=None,
            logits=hidden_states,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            **extra_output_kwargs,
        )

    _patched_causal_lm_forward.__qualname__ = _original_causal_lm_forward.__qualname__
    causal_lm_cls.forward = _patched_causal_lm_forward
    if UNSLOTH_ENABLE_LOGGING:
        logger.info(f"Unsloth: Patched {model_label}.forward for GRPO hidden states.")




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
            # This Unsloth Zoo code section is licensed under AGPL3

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

        _qwen3_lora_extractor = _make_qwen_moe_lora_extractor()

        transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeExperts._unsloth_lora_extractor_fn = _qwen3_lora_extractor

        forward = _make_qwen_moe_experts_forward(module_name=__name__)
        sparse_moe_block_forward = _make_qwen_moe_sparse_moe_block_forward(
            use_shared_expert=False,
            module_name=__name__,
        )

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
        _patch_causal_lm_forward_for_hidden_states(
            Qwen3MoeForCausalLM,
            MoeCausalLMOutputWithPast,
            "Qwen3MoeForCausalLM",
            extra_output_kwargs_fn=lambda outputs: {
                "aux_loss": None,
                "router_logits": outputs.router_logits,
            },
        )
    except Exception as e:
        if UNSLOTH_ENABLE_LOGGING:
            logger.warning(f"Unsloth: Could not patch Qwen3MoeForCausalLM.forward: {e}")

pass
TEMPORARY_PATCHES.append(patch_qwen3_moe)
