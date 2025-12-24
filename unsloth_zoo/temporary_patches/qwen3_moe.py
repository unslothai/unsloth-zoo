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


# ============================================================================
# Grouped GEMM kernel integration for MoE training acceleration
# ============================================================================

# Global flag to check if grouped GEMM is available
_GROUPED_GEMM_AVAILABLE = None
_GROUPED_GEMM_WARNED = False
_TRITON_ALLOCATOR_INITIALIZED = False
_PERSISTENT_BUFFER = None


def _init_triton_allocator():
    """
    Initialize a persistent Triton allocator to avoid memory allocation overhead per call.
    This significantly reduces GPU utilization fluctuation.
    """
    global _TRITON_ALLOCATOR_INITIALIZED, _PERSISTENT_BUFFER
    if _TRITON_ALLOCATOR_INITIALIZED:
        return

    try:
        import triton

        # Create a persistent buffer that grows as needed
        # This avoids allocating new memory on every kernel call
        _buffer_cache = {}

        def persistent_alloc_fn(size: int, alignment: int, stream):
            global _PERSISTENT_BUFFER
            # Round up size to avoid frequent reallocations
            rounded_size = ((size + 1024 * 1024 - 1) // (1024 * 1024)) * (1024 * 1024)

            if _PERSISTENT_BUFFER is None or _PERSISTENT_BUFFER.numel() < rounded_size:
                # Allocate with some headroom to reduce reallocations
                _PERSISTENT_BUFFER = torch.empty(
                    rounded_size * 2, device="cuda", dtype=torch.int8
                )
                _PERSISTENT_BUFFER.__hibernate__ = {"type": "ignore"}
            return _PERSISTENT_BUFFER

        triton.set_allocator(persistent_alloc_fn)
        _TRITON_ALLOCATOR_INITIALIZED = True
    except Exception:
        pass


def _check_grouped_gemm_available():
    """Check if Unsloth grouped GEMM kernels are available."""
    global _GROUPED_GEMM_AVAILABLE
    if _GROUPED_GEMM_AVAILABLE is None:
        try:
            # The grouped_gemm module uses relative imports like `from grouped_gemm.kernels...`
            # so we need to add its parent directory to sys.path
            import sys
            import unsloth
            unsloth_path = os.path.dirname(unsloth.__file__)
            moe_kernels_path = os.path.join(unsloth_path, "kernels", "moe")
            if moe_kernels_path not in sys.path:
                sys.path.insert(0, moe_kernels_path)
            from grouped_gemm.interface import grouped_gemm, supports_tma
            _GROUPED_GEMM_AVAILABLE = True
            # Initialize persistent allocator when grouped GEMM is available
            _init_triton_allocator()
        except (ImportError, ModuleNotFoundError) as e:
            _GROUPED_GEMM_AVAILABLE = False
    return _GROUPED_GEMM_AVAILABLE


def _supports_tma():
    """Check if TMA (Tensor Memory Accelerator) is supported (sm90+ / Hopper)."""
    return torch.cuda.get_device_capability()[0] >= 9


# Pre-allocated buffers for routing to avoid per-call allocations
_ROUTING_BUFFERS = {}


def _get_routing_indices_optimized(selected_experts, num_experts):
    """
    Optimized token→expert mapping for grouped GEMM.
    Uses bincount instead of histc to avoid float conversion overhead.
    Reuses buffers when possible to reduce memory allocation pressure.

    Returns:
        token_counts_by_expert: (num_experts,) token counts per expert
        gather_indices: (total_tokens,) indices for gathering tokens in expert order
    """
    flat_experts = selected_experts.view(-1)

    # bincount is faster than histc since it doesn't require float conversion
    token_counts_by_expert = torch.bincount(
        flat_experts,
        minlength=num_experts
    ).to(torch.int32)

    # argsort with stable=True preserves order within each expert
    gather_indices = flat_experts.argsort(stable=True)

    return token_counts_by_expert, gather_indices


@torch.no_grad()
def _get_routing_indices(selected_experts, num_experts):
    """
    Compute token→expert mapping for grouped GEMM.
    Wrapper that uses optimized implementation.

    Returns:
        token_counts_by_expert: (num_experts,) token counts per expert
        gather_indices: (total_tokens,) indices for gathering tokens in expert order
    """
    return _get_routing_indices_optimized(selected_experts, num_experts)


def _silu_and_mul(x):
    """Fused SiLU activation and element-wise multiply for gate/up projections."""
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


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
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)
            # router_logits: (batch * sequence_length, n_experts)
            # router_logits = self.gate(hidden_states)

            # routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
            # routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            # if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            #     routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            # # we cast back to the input dtype
            # routing_weights = routing_weights.to(hidden_states.dtype)
            # router_scores = torch.zeros_like(router_logits, dtype = hidden_states.dtype).scatter_(1, selected_experts, routing_weights)
            router_scores, selected_experts, router_logits = router_forward(self, hidden_states)
            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim), dtype=torch.float32, device=hidden_states.device
            )

            # One hot encode the selected experts to create an expert mask
            # this will be used to easily index which expert is going to be sollicitated
            # expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

            # Loop over all available experts in the model and perform the computation on each expert
            # expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            # for expert_idx in expert_hit:
            for expert_idx in range(self.num_experts):
                expert_layer = self.experts[expert_idx]
                # idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
                token_idx, _ = torch.where(selected_experts == expert_idx)

                # Index the correct hidden states and compute the expert hidden state for
                # the current expert. We need to make sure to multiply the output hidden
                # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
                # current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                # current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
                current_state = hidden_states[token_idx].reshape(-1, hidden_dim)
                current_hidden_states = expert_layer(current_state) * router_scores[token_idx, expert_idx, None]

                # However `index_add_` only support torch tensors for indexing so we'll use
                # the `top_x` tensor here.
                # final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
                final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(torch.float32))
            final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
            return final_hidden_states.to(hidden_states.dtype), router_logits
    else:
        # ====================================================================
        # New transformers (5.0+) with stacked expert weights
        # Uses Triton grouped GEMM kernels for high performance
        # ====================================================================

        use_grouped_gemm = _check_grouped_gemm_available()

        if use_grouped_gemm:
            # Import grouped GEMM interface (sys.path was set by _check_grouped_gemm_available)
            from grouped_gemm.interface import grouped_gemm, supports_tma
            # Import autotune cache
            from unsloth.kernels.moe.autotune_cache import get_or_autotune_moe_kernels

            # Cache for kernel configs - created once and reused
            _MODEL_DIMS_AND_CONFIGS = None

            def _get_use_tma():
                """Cache TMA support check to avoid repeated GPU queries."""
                # This should be graph-safe (returns bool constant during trace)
                return supports_tma()

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
                nonlocal _MODEL_DIMS_AND_CONFIGS

                num_tokens, hidden_dim = hidden_states.shape
                top_k = top_k_index.shape[1]

                # Cache model dimensions and kernel configs on first call
                # This block runs in eager mode or during tracing
                if _MODEL_DIMS_AND_CONFIGS is None:
                    intermediate_dim = self.gate_up_proj.shape[1] // 2

                    # Autotune first GEMM
                    gemm1_configs = get_or_autotune_moe_kernels(
                        num_experts=self.num_experts,
                        hidden_dim=hidden_dim,
                        intermediate_dim=intermediate_dim * 2,
                        top_k=top_k,
                        dtype=hidden_states.dtype,
                    )

                    # Autotune second GEMM
                    gemm2_configs = get_or_autotune_moe_kernels(
                        num_experts=self.num_experts,
                        hidden_dim=intermediate_dim,
                        intermediate_dim=hidden_dim, # Output dim for 2nd GEMM is hidden_dim
                        top_k=top_k,
                        dtype=hidden_states.dtype,
                    )

                    _MODEL_DIMS_AND_CONFIGS = (intermediate_dim, gemm1_configs, gemm2_configs)

                # Unpack cached configs
                intermediate_dim, gemm1_configs, gemm2_configs = _MODEL_DIMS_AND_CONFIGS

                # Unpack specific kernel configs
                fwd_config_1, bwd_dX_config_1, bwd_dW_config_1 = gemm1_configs
                fwd_config_2, bwd_dX_config_2, bwd_dW_config_2 = gemm2_configs

                # Compute routing indices for grouped GEMM
                token_counts_by_expert, gather_indices = _get_routing_indices(
                    top_k_index, self.num_experts
                )

                # First grouped GEMM: gate_up projection
                first_gemm_output = grouped_gemm(
                    X=hidden_states,
                    W=self.gate_up_proj,
                    m_sizes=token_counts_by_expert,
                    topk=top_k,
                    gather_indices=gather_indices,
                    permute_x=True,
                    permute_y=False,
                    autotune=False, # We use cached configs
                    kernel_config_fwd=fwd_config_1,
                    kernel_config_bwd_dX=bwd_dX_config_1,
                    kernel_config_bwd_dW=bwd_dW_config_1,
                    is_first_gemm=True,
                )

                # Apply SiLU activation and multiply gate with up
                intermediate = _silu_and_mul(first_gemm_output)

                # Second grouped GEMM: down projection
                second_gemm_output = grouped_gemm(
                    X=intermediate,
                    W=self.down_proj,
                    m_sizes=token_counts_by_expert,
                    topk=top_k,
                    gather_indices=gather_indices,
                    permute_x=False,
                    permute_y=True,
                    autotune=False, # We use cached configs
                    kernel_config_fwd=fwd_config_2,
                    kernel_config_bwd_dX=bwd_dX_config_2,
                    kernel_config_bwd_dW=bwd_dW_config_2,
                    is_first_gemm=False,
                )

                # Apply routing weights and sum across top_k experts
                # Output shape: (num_tokens, top_k, hidden_dim) -> (num_tokens, hidden_dim)
                final_hidden_states = (
                    second_gemm_output.view(num_tokens, top_k, hidden_dim)
                    * top_k_weights[..., None]
                )
                final_hidden_states = final_hidden_states.sum(dim=1)

                return final_hidden_states
        else:
            # Fallback: Pure PyTorch loop-based implementation
            global _GROUPED_GEMM_WARNED
            if not _GROUPED_GEMM_WARNED:
                logger.warning(
                    "Unsloth grouped GEMM kernels not available. "
                    "Falling back to PyTorch loop implementation. "
                    "For faster MoE training, install unsloth kernels."
                )
                _GROUPED_GEMM_WARNED = True

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
pass
TEMPORARY_PATCHES.append(patch_qwen3_moe)
