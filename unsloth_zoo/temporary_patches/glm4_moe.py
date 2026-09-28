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
    """Patch GLM4 MoE for Split LoRA using grouped GEMM."""
    patch_param_wrapper_for_moe()

    try:
        from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
            Glm4MoeLiteMoE,
            Glm4MoeLiteNaiveMoe,
        )
    except ImportError:
        # Classes absent on older transformers
        return

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

    def moe_block_forward(self, hidden_states) -> torch.Tensor:
        """MoE block forward: GLM4 routing, NaiveMoe experts, plus shared experts."""
        residuals = hidden_states
        orig_shape = hidden_states.shape

        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)

        # Sigmoid output is float32; cast to hidden_states dtype (bf16/fp16)
        topk_weights = topk_weights.to(hidden_states.dtype)

        # Experts (delegated to Glm4MoeLiteNaiveMoe)
        hidden_states_flat = hidden_states.view(-1, hidden_states.shape[-1])
        expert_output = self.experts(hidden_states_flat, topk_indices, topk_weights)
        hidden_states = expert_output.view(*orig_shape)

        shared_output = self.shared_experts(residuals)
        return hidden_states + shared_output

    patch_function(Glm4MoeLiteNaiveMoe, "forward", get_forward_moe_backend())
    patch_function(Glm4MoeLiteMoE,      "forward", moe_block_forward)

    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Patched GLM4 MoE Lite for Split LoRA support.")


def patch_glm4_moe_standard():
    """Patches standard (non-lite) GLM4 MoE for Split LoRA using grouped GEMM."""
    patch_param_wrapper_for_moe()

    try:
        from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeNaiveMoe
    except ImportError:
        return

    def _glm4_std_lora_extractor(wrapper, weight_A, weight_B, scaling, num_experts):
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

    Glm4MoeNaiveMoe._unsloth_lora_extractor_fn = staticmethod(_glm4_std_lora_extractor)
    patch_function(Glm4MoeNaiveMoe, "forward", get_forward_moe_backend())

    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Patched GLM4 MoE (standard) for Split LoRA support.")

TEMPORARY_PATCHES.append(patch_glm4_moe)
TEMPORARY_PATCHES.append(patch_glm4_moe_standard)
