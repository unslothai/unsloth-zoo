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

import torch
from .common import TEMPORARY_PATCHES, torch_compile, UNSLOTH_ENABLE_LOGGING
from .utils import patch_function, raise_error, logger
from .moe_utils import (
    patch_param_wrapper_for_moe,
    get_forward_moe_backend,
    extract_moe_lora_weights_for_grouped_mm,
)
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

    # ====================================================================
    # Define LoRA extraction function for GLM4-MoE (Standard Format)
    # ====================================================================
    def _glm4_lora_extractor(wrapper, weight_A, weight_B, scaling, num_experts):
        return extract_moe_lora_weights_for_grouped_mm(
            wrapper,
            weight_A,
            weight_B,
            scaling,
            num_experts,
            model_name="GLM4 MoE",
            enable_logging=UNSLOTH_ENABLE_LOGGING,
            logger_obj=logger,
        )

    Glm4MoeLiteNaiveMoe._unsloth_lora_extractor_fn = staticmethod(_glm4_lora_extractor)


    # 1. Patch Glm4MoeLiteNaiveMoe (The Expert Layer)
    # This delegates to moe_utils which handles the Split LoRA logic

    # 2. Patch Glm4MoeLiteMoE (The MoE Block)
    # This must be patched to delegate expert computation to naive_moe_forward instead of inlining it

    def moe_block_forward(self, hidden_states) -> torch.Tensor:
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
    patch_function(Glm4MoeLiteNaiveMoe, "forward", get_forward_moe_backend())
    patch_function(Glm4MoeLiteMoE,      "forward", moe_block_forward)

    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Patched GLM4 MoE Lite for Split LoRA support.")


def patch_glm4_moe_standard():
    """
    Patches standard (non-lite) GLM4 MoE (Glm4MoeNaiveMoe + Glm4MoeMoE).
    Uses the same _glm4_lora_extractor and grouped-MM forward as the lite variant.
    """
    patch_param_wrapper_for_moe()

    try:
        from transformers.models.glm4_moe.modeling_glm4_moe import (
            Glm4MoeMoE,
            Glm4MoeNaiveMoe,
        )
    except ImportError:
        return

    def _glm4_std_lora_extractor(wrapper, weight_A, weight_B, scaling, num_experts):
        total_rank = weight_A.shape[0]
        rank_per_expert = total_rank // num_experts
        dim1 = weight_A.shape[1]
        dim2 = weight_B.shape[0]

        if dim1 > dim2:
            first_weight = weight_B.view(dim2, num_experts, rank_per_expert)
            first_weight = first_weight.permute(1, 0, 2).contiguous()
            second_weight = weight_A.view(num_experts, rank_per_expert, dim1).contiguous()
        else:
            first_weight = weight_A.view(num_experts, rank_per_expert, dim1)
            first_weight = first_weight.permute(0, 2, 1).contiguous()
            second_weight = weight_B.view(dim2, num_experts, rank_per_expert)
            second_weight = second_weight.permute(1, 2, 0).contiguous()

        return first_weight, second_weight, scaling, num_experts

    Glm4MoeNaiveMoe._unsloth_lora_extractor_fn = staticmethod(_glm4_std_lora_extractor)

    patch_function(Glm4MoeNaiveMoe, "forward", get_forward_moe_backend())

    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Patched GLM4 MoE (standard) for Split LoRA support.")

# Register the patches
TEMPORARY_PATCHES.append(patch_glm4_moe)
TEMPORARY_PATCHES.append(patch_glm4_moe_standard)
