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

import os
import torch
import torch.nn as nn
from .common import TEMPORARY_PATCHES, torch_compile, UNSLOTH_ENABLE_LOGGING
from .utils import patch_function, raise_error, logger
from .moe_utils import (
    patch_param_wrapper_for_moe,
    get_forward_moe_backend,
)


def maybe_patch_glm4_moe_expert_fp8_scales(
    model,
    model_name: str,
    token = None,
    revision = None,
):
    """
    GLM-4.7-Flash FP8 Dynamic stores routed expert weights as raw float8 tensors
    plus per-expert weight_scale tensors. Transformers currently leaves those
    scales as UNEXPECTED keys because Glm4MoeLiteNaiveMoe uses stacked
    nn.Parameters instead of Linear modules, so we patch the expert tensors here.
    We must preserve the FP8 parameters and attach the scale tensors for runtime
    dequantization only when a fallback path needs high-precision weights.
    """
    config = getattr(model, "config", None)
    if config is None or getattr(config, "model_type", None) != "glm4_moe_lite":
        return False

    quantization_config = getattr(config, "quantization_config", None)
    if isinstance(quantization_config, dict):
        quant_method = quantization_config.get("quant_method", None)
    else:
        quant_method = getattr(quantization_config, "quant_method", None)
    if quant_method != "compressed-tensors":
        return False

    inner_model = getattr(model, "model", None)
    if inner_model is None or not hasattr(inner_model, "layers"):
        return False

    routed_layers = []
    for layer_idx, layer in enumerate(inner_model.layers):
        experts = getattr(getattr(layer, "mlp", None), "experts", None)
        if experts is None or not hasattr(experts, "gate_up_proj"):
            continue
        if getattr(experts.gate_up_proj, "dtype", None) == torch.float8_e4m3fn:
            routed_layers.append((layer_idx, experts))
    if len(routed_layers) == 0:
        return False

    if os.path.isdir(model_name):
        safetensors_path = os.path.join(model_name, "model.safetensors")
    else:
        from huggingface_hub import hf_hub_download

        safetensors_path = hf_hub_download(
            repo_id = model_name,
            filename = "model.safetensors",
            token = token,
            revision = revision,
        )

    import safetensors.torch

    with safetensors.torch.safe_open(safetensors_path, framework = "pt") as file:
        for layer_idx, experts in routed_layers:
            device = experts.gate_up_proj.device
            num_experts = experts.gate_up_proj.shape[0]
            gate_up_rows = []
            down_rows = []
            gate_up_scales = []
            down_scales = []

            for expert_idx in range(num_experts):
                gate = file.get_tensor(
                    f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"
                )
                gate_scale = file.get_tensor(
                    f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight_scale"
                )
                up = file.get_tensor(
                    f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"
                )
                up_scale = file.get_tensor(
                    f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight_scale"
                )
                down = file.get_tensor(
                    f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"
                )
                down_scale = file.get_tensor(
                    f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight_scale"
                )

                gate_up_rows.append(torch.cat([gate, up], dim = 0))
                down_rows.append(down)
                gate_up_scales.append(torch.cat([gate_scale, up_scale], dim = 0))
                down_scales.append(down_scale)

            experts.gate_up_proj = nn.Parameter(
                torch.stack(gate_up_rows, dim = 0).to(device = device),
                requires_grad = experts.gate_up_proj.requires_grad,
            )
            experts.down_proj = nn.Parameter(
                torch.stack(down_rows, dim = 0).to(device = device),
                requires_grad = experts.down_proj.requires_grad,
            )
            experts.gate_up_proj_weight_scale = nn.Parameter(
                torch.stack(gate_up_scales, dim = 0).to(device = device),
                requires_grad = False,
            )
            experts.down_proj_weight_scale = nn.Parameter(
                torch.stack(down_scales, dim = 0).to(device = device),
                requires_grad = False,
            )

    return True


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
        """
        Custom LoRA extractor for GLM4.

        Expectation for grouped_mm (Standard):
        - first (Input):  (E, H, R)
        - second (Output): (E, R, Out)

        GLM4 Weights (Standard PEFT):
        - gate_up is (E, Out, In) or (Out, In).
        - lora_A (In->R) connects to H. Shape (E*R, H).
          Needs: View(E, R, H) -> Permute(0, 2, 1) -> (E, H, R).
        - lora_B (R->Out) connects to 2I. Shape (2I, E*R).
          Needs: View(2I, E, R) -> Permute(1, 2, 0) -> (E, R, 2I).
        """
        total_rank = weight_A.shape[0]
        rank_per_expert = total_rank // num_experts
        dim1 = weight_A.shape[1]
        dim2 = weight_B.shape[0]

        # GLM4 MoE sometimes stores weights transposed (E, in_dim, out_dim),
        # which flips LoRA A/B's input/output dimensions. Detect and handle both.
        if dim1 > dim2:
            # Transposed: weight_A is (E*R, out_dim), weight_B is (in_dim, E*R)
            # first_weight from B: (E, in_dim, R)
            first_weight = weight_B.view(dim2, num_experts, rank_per_expert)
            first_weight = first_weight.permute(1, 0, 2).contiguous()

            # second_weight from A: (E, R, out_dim)
            second_weight = weight_A.view(num_experts, rank_per_expert, dim1).contiguous()
        else:
            # Standard: weight_A is (E*R, in_dim), weight_B is (out_dim, E*R)
            first_weight = weight_A.view(num_experts, rank_per_expert, dim1)
            first_weight = first_weight.permute(0, 2, 1).contiguous()  # (E, in_dim, R)

            second_weight = weight_B.view(dim2, num_experts, rank_per_expert)
            second_weight = second_weight.permute(1, 2, 0).contiguous()  # (E, R, out_dim)

        return first_weight, second_weight, scaling, num_experts

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
        logger.info("Unsloth: Patched GLM4 MoE for Split LoRA support.")

# Register the patch
TEMPORARY_PATCHES.append(patch_glm4_moe)
