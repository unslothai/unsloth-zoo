
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
    select_moe_backend,
)

def patch_qwen3_vl_moe():
    try:
        import transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe
        transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextSparseMoeBlock
    except Exception as e:
        return raise_error("transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextSparseMoeBlock", e)

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
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)
            # router_logits: (batch * sequence_length, n_experts)
            router_logits = self.gate(hidden_states)

            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            if self.norm_topk_prob:
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            # we cast back to the input dtype
            routing_weights = routing_weights.to(hidden_states.dtype)

            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
            )

            # One hot encode the selected experts to create an expert mask
            expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

            # Loop over all available experts in the model and perform the computation on each expert
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in expert_hit:
                expert_layer = self.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
            final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
            return final_hidden_states, router_logits

        @torch_compile(dynamic = True, fullgraph = True)
        def router_forward(self, hidden_states):
            router_logits = self.gate(hidden_states)

            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            if self.norm_topk_prob:
                routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
            # we cast back to the input dtype
            routing_weights = routing_weights.to(hidden_states.dtype)
            router_scores = torch.zeros_like(router_logits, dtype = hidden_states.dtype).scatter_(1, selected_experts, routing_weights)
            return router_scores, selected_experts, router_logits

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            """ """
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)

            router_scores, selected_experts, router_logits = router_forward(self, hidden_states)
            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim), dtype=torch.float32, device=hidden_states.device
            )

            # Loop over all available experts
            for expert_idx in range(self.num_experts):
                expert_layer = self.experts[expert_idx]
                token_idx, _ = torch.where(selected_experts == expert_idx)

                current_state = hidden_states[token_idx].reshape(-1, hidden_dim)
                current_hidden_states = expert_layer(current_state) * router_scores[token_idx, expert_idx, None]

                final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(torch.float32))
            final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
            return final_hidden_states.to(hidden_states.dtype), router_logits

    else:
        # ====================================================================
        # New transformers with stacked expert weights
        # ====================================================================

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
                return forward_native_grouped_mm(self, hidden_states, top_k_index, top_k_weights)

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
                return forward_triton_grouped_gemm(self, hidden_states, top_k_index, top_k_weights)

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
                # For brevity, implementing basic correct loop
                final_hidden_states = torch.zeros_like(hidden_states)
                with torch.no_grad():
                    expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
                    # expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

                # simpler fallback
                for expert_idx in range(self.num_experts):
                    top_k_pos, token_idx = torch.where(expert_mask[expert_idx])

                    # Skip expert if no tokens routed to it
                    if token_idx.shape[0] == 0:
                        continue

                    current_state = hidden_states[token_idx]

                    # Assuming gate_up_proj and down_proj exist as stacked weights in Experts class
                    gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
                    current_hidden_states = self.act_fn(gate) * up
                    current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])

                    current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
                    final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

                return final_hidden_states

        # SparseMoeBlock forward
        # @torch.compiler.disable
        def sparse_moe_block_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            if hidden_states.dim() == 3:
                batch_size, sequence_length, hidden_dim = hidden_states.shape
            else:
                total_tokens, hidden_dim = hidden_states.shape
                batch_size = 1
                sequence_length = total_tokens

            hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

            # self.gate is nn.Linear - so it returns logits!
            router_logits = self.gate(hidden_states_reshaped)
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            routing_weights = routing_weights.to(hidden_states.dtype)

            final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)

            if hidden_states.dim() == 3:
                return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
            return final_hidden_states

    if old_transformers:
        patch_function(transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextSparseMoeBlock, "forward", forward)
    else:
        patch_function(transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextExperts, "forward", forward, force=True)
        patch_function(transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextSparseMoeBlock, "forward", sparse_moe_block_forward, force=True)

    transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.__UNSLOTH_PATCHED__ = True

TEMPORARY_PATCHES.append(patch_qwen3_vl_moe)
