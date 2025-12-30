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
)
from .utils import (
    patch_function,
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
    select_moe_backend,
)




def patch_qwen3_moe():
    # https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L213
    # Transformers >= 5       uses self.gate_up_proj = nn.Parameter(...)
    # whilst old transformers uses self.experts = nn.ModuleList(...)
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
            final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
            return final_hidden_states.to(hidden_states.dtype), router_logits
    else:
    # ====================================================================
        # New transformers (5.0+) with stacked expert weights
        # Uses Triton grouped GEMM kernels for high performance
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
            # Import grouped GEMM interface (sys.path was set by _check_grouped_gemm_available)
            from grouped_gemm.interface import grouped_gemm, supports_tma
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


                final_hidden_states = torch.zeros_like(hidden_states)

                # Create expert mask and find which experts have tokens
                with torch.no_grad():
                    expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
                    expert_mask = expert_mask.permute(2, 1, 0)  # (num_experts, top_k, n_tokens)
                    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

                # Only loop over experts that actually have tokens routed to them
                for expert_idx in expert_hit:
                    expert_idx = expert_idx[0]
                    if expert_idx == self.num_experts:
                        continue

                    # Find which tokens are routed to this expert
                    top_k_pos, token_idx = torch.where(expert_mask[expert_idx])

                    # Gather only the tokens for this expert
                    current_state = hidden_states[token_idx]

                    # Compute gate_up projection for this expert only
                    gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
                    current_hidden_states = self.act_fn(gate) * up

                    # Compute down projection for this expert only
                    current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])

                    # Apply routing weights
                    current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]

                    # Scatter back to final output
                    final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

                return final_hidden_states

        # SparseMoeBlock forward is disabled from compilation due to dynamic routing
        @torch.compiler.disable
        def sparse_moe_block_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
            _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
            final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
            return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

    # For old transformers, patch Qwen3MoeSparseMoeBlock
    # For new transformers, patch Qwen3MoeExperts (which has the expert loop)
    if old_transformers:
        patch_function(transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock, "forward", forward)
    else:
        patch_function(transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeExperts, "forward", forward)
        patch_function(transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock, "forward", sparse_moe_block_forward)

    transformers.models.qwen3_moe.modeling_qwen3_moe.__UNSLOTH_PATCHED__ = True
pass
TEMPORARY_PATCHES.append(patch_qwen3_moe)
