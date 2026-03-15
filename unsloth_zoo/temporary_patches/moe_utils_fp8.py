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


def _maybe_patch_glm4_stacked_moe_fp8_scales(
    model,
    model_name: str,
    token = None,
    revision = None,
):
    """
    Attach missing FP8 scale tensors to stacked routed-expert parameters.

    This currently handles GLM4-MoE Lite style experts where transformers loads
    the float8 expert weights but leaves the per-expert weight_scale tensors as
    unexpected keys because the experts are stacked nn.Parameters rather than
    Linear modules.
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


def maybe_patch_stacked_moe_expert_fp8_scales(
    model,
    model_name: str,
    token = None,
    revision = None,
):
    """
    Best-effort hook for prequantized FP8 MoE checkpoints that use stacked expert
    parameters and need extra runtime quant metadata attached after loading.

    This is intentionally generic at the callsite. Model-specific handlers can
    be added here as new stacked-FP8 MoE formats appear.
    """
    return _maybe_patch_glm4_stacked_moe_fp8_scales(
        model,
        model_name,
        token = token,
        revision = revision,
    )
