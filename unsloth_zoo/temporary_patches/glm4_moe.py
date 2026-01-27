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

import torch
import torch.nn.functional as F
from .common import TEMPORARY_PATCHES, torch_compile, UNSLOTH_ENABLE_LOGGING
from .utils import patch_function, raise_error, logger
from .moe_utils import (
    select_moe_backend,
    forward_native_grouped_mm,
    forward_triton_grouped_gemm,
    forward_native_moe_loop,
    patch_param_wrapper_for_moe
)
from . import moe_utils

def patch_glm4_moe():
    """
    Patches GLM4 MoE to support Split LoRA using grouped GEMM.
    """
    # Patch PEFT ParamWrapper for separated LoRA weights
    patch_param_wrapper_for_moe()

    # Try to import the GLM4 MoE classes
    try:
        from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
            Glm4MoeLiteMoE,
            Glm4MoeLiteNaiveMoe,
        )
    except ImportError:
        # If classes aren't available (e.g. older transformers), just return
        return

    # 1. Patch Glm4MoeLiteNaiveMoe (The Expert Layer)
    # This delegates to moe_utils which handles the Split LoRA logic

    def naive_moe_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Patched forward for Expert layer.
        Dispatches to appropriate backend (native torch grouped_mm, triton, or loop).
        """
        backend = select_moe_backend()

        if backend == "grouped_mm":
            return moe_utils.forward_native_grouped_mm(self, hidden_states, top_k_index, top_k_weights)
        elif backend == "unsloth_triton":
            return moe_utils.forward_triton_grouped_gemm(self, hidden_states, top_k_index, top_k_weights)
        else:
            return moe_utils.forward_native_moe_loop(self, hidden_states, top_k_index, top_k_weights)

    # 2. Patch Glm4MoeLiteMoE (The MoE Block)
    # This must be patched to delegate expert computation to naive_moe_forward instead of inlining it

    def moe_block_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Patched forward for MoE Block.
        Computes routing using GLM4 logic, then calls experts (NaiveMoe), then adds shared experts.
        """
        residuals = hidden_states
        orig_shape = hidden_states.shape

        # 1. Routing
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)

        # Cast routing weights to match hidden_states dtype if needed
        # (Sigmoid output is float32, hidden_states usually bf16/fp16)
        topk_weights = topk_weights.to(hidden_states.dtype)

        # 2. Expert Computation (delegated to Glm4MoeLiteNaiveMoe)
        # Flatten for experts input
        hidden_states_flat = hidden_states.view(-1, hidden_states.shape[-1])

        expert_output = self.experts(hidden_states_flat, topk_indices, topk_weights)

        # Reshape back
        hidden_states = expert_output.view(*orig_shape)

        # 3. Shared Experts
        shared_output = self.shared_experts(residuals)

        return hidden_states + shared_output

    # Apply patches
    patch_function(Glm4MoeLiteNaiveMoe, "forward", naive_moe_forward)
    patch_function(Glm4MoeLiteMoE,      "forward", moe_block_forward)

    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Patched GLM4 MoE for Split LoRA support.")

# Register the patch
TEMPORARY_PATCHES.append(patch_glm4_moe)
