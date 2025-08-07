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
from .common import TEMPORARY_PATCHES, torch_compile_options
from .utils import (
    patch_function,
    KWARGS_TYPE,
    raise_error,
    logger,
)
torch_cuda_device = torch.cuda.device


def patch_gpt_oss():
    try:
        import triton_kernels
    except Exception as e:
        return raise_error("Please install triton_kernels", e)

    try:
        import transformers.quantizers.quantizer_mxfp4
        def is_kernels_available(): return True
        transformers.quantizers.quantizer_mxfp4.is_kernels_available = is_kernels_available
    except Exception as e:
        return raise_error("transformers.quantizers.quantizer_mxfp4.is_kernels_available", e)

    try:
        from triton_kernels import matmul_ogs, swiglu
        FnSpecs, FusedActivation, matmul_ogs = (
            matmul_ogs.FnSpecs,
            matmul_ogs.FusedActivation,
            matmul_ogs.matmul_ogs,
        )
        swiglu_fn = swiglu.swiglu_fn
    except Exception as e:
        return raise_error("triton_kernels", e)

    try:
        import transformers.integrations.mxfp4
    except Exception as e:
        return raise_error("transformers.integrations.mxfp4", e)

    def swizzle_mxfp4(w, w_scale):
        from triton_kernels import tensor, tensor_details
        FP4, convert_layout, wrap_torch_tensor = (
            tensor.FP4,
            tensor.convert_layout,
            tensor.wrap_torch_tensor,
        )
        layout = tensor_details.layout
        StridedLayout = tensor_details.layout.StridedLayout

        value_layout, value_layout_opts = layout.make_default_matmul_mxfp4_w_layout(mx_axis=1)
        w = convert_layout(wrap_torch_tensor(w, dtype=FP4), value_layout, **value_layout_opts)
        # TODO : add that when we are actually sure that it works on B200
        # if torch.cuda.get_device_capability()[0] == 10:
        #     constraints = {
        #         "is_persistent": True,
        #         "epilogue_subtile": 1,
        #     }
        #     opt_flags.update_opt_flags_constraints(constraints)
        # # transpose the tensor so that the quantization axis is on dim1

        # TODO: there is still an issue with the scales on hopper
        # scale_layout, scale_layout_opts = layout.make_default_matmul_mxfp4_w_scale_layout(mx_axis=1, num_warps=8)
        # w_scale = convert_layout(wrap_torch_tensor(w_scale), scale_layout, **scale_layout_opts)
        w_scale = convert_layout(wrap_torch_tensor(w_scale), StridedLayout)
        return w, w_scale
    patch_function(transformers.integrations.mxfp4, "swizzle_mxfp4", swizzle_mxfp4)

    class Mxfp4GptOssExperts(nn.Module):
        def __init__(self, config):
            super().__init__()

            self.num_experts = config.num_local_experts
            self.intermediate_size = config.intermediate_size
            self.hidden_size = config.hidden_size

            self.gate_up_proj_blocks = nn.Parameter(
                torch.zeros(self.num_experts, 2 * self.intermediate_size, self.hidden_size // 32, 16, dtype=torch.uint8),
                requires_grad=False,
            )
            self.gate_up_proj_scales = nn.Parameter(
                torch.zeros(self.num_experts, 2 * self.intermediate_size, self.hidden_size // 32, dtype=torch.uint8),
                requires_grad=False,
            )
            self.gate_up_proj_bias = nn.Parameter(
                torch.zeros(self.num_experts, 2 * self.intermediate_size, dtype=torch.float32), requires_grad=False
            )

            self.down_proj_blocks = nn.Parameter(
                torch.zeros((self.num_experts, self.hidden_size, self.intermediate_size // 32, 16), dtype=torch.uint8),
                requires_grad=False,
            )
            self.down_proj_scales = nn.Parameter(
                torch.zeros(self.num_experts, self.hidden_size, self.intermediate_size // 32, dtype=torch.uint8),
                requires_grad=False,
            )
            self.down_proj_bias = nn.Parameter(
                torch.zeros(self.num_experts, self.hidden_size, dtype=torch.float32), requires_grad=False
            )
            self.alpha = 1.702

            self.gate_up_proj_precision_config = None
            self.down_proj_precision_config = None

        def forward(self, hidden_states: torch.Tensor, routing_data, gather_idx, scatter_idx) -> torch.Tensor:
            with torch_cuda_device(hidden_states.device):
                if not hasattr(self, "act"):
                    self.act = FusedActivation(FnSpecs("swiglu", swiglu_fn, ("alpha", "limit")), (self.alpha, None), 2)
                intermediate_cache1 = matmul_ogs(
                    hidden_states.to(torch.bfloat16),
                    self.gate_up_proj,
                    self.gate_up_proj_bias,
                    routing_data,
                    gather_indx=gather_idx,
                    precision_config=self.gate_up_proj_precision_config,
                    gammas=None,
                    fused_activation=self.act,
                )
                intermediate_cache3 = matmul_ogs(
                    intermediate_cache1,
                    self.down_proj,
                    self.down_proj_bias,
                    routing_data,
                    scatter_indx=scatter_idx,
                    precision_config=self.down_proj_precision_config,
                    gammas=routing_data.gate_scal,
                )
        return intermediate_cache3
    patch_function(transformers.integrations.mxfp4, "Mxfp4GptOssExperts", Mxfp4GptOssExperts)

    try:
        routing = triton_kernels.routing.routing
        routing = torch.compiler.disable(routing)
    except Exception as e:
        return raise_error("triton_kernels.routing.routing", e)

    def mlp_forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.router.hidden_dim)
        router_logits = nn.functional.linear(hidden_states, self.router.weight, self.router.bias)

        with torch_cuda_device(router_logits.device):
            routing_data, gather_idx, scatter_idx = routing(router_logits, self.router.top_k)

        routed_out = self.experts(hidden_states, routing_data, gather_idx, scatter_idx)
        routed_out = routed_out.reshape(batch_size, -1, self.router.hidden_dim)
        return routed_out, router_logits
    patch_function(transformers.integrations.mxfp4, "mlp_forward", mlp_forward)

    try:
        PrecisionConfig, FlexCtx, InFlexData = (
            triton_kernels.matmul_ogs.PrecisionConfig,
            triton_kernels.matmul_ogs.FlexCtx,
            triton_kernels.matmul_ogs.InFlexData,
        )
    except Exception as e:
        return raise_error("triton_kernels.matmul_ogs", e)

    try:
        from transformers.integrations.tensor_parallel import shard_and_distribute_module
    except Exception as e:
        return raise_error("transformers.integrations.tensor_parallel.shard_and_distribute_module", e)

    def load_and_swizzle_mxfp4(module, param_name, param_value, target_device, **kwargs):
        model = kwargs.get("model", None)
        empty_param = kwargs.get("empty_param", None)
        casting_dtype = kwargs.get("casting_dtype", None)
        to_contiguous = kwargs.get("to_contiguous", None)
        rank = kwargs.get("rank", None)
        device_mesh = kwargs.get("device_mesh", None)

        for proj in ["gate_up_proj", "down_proj"]:
            if proj in param_name:
                if device_mesh is not None:
                    shard_and_distribute_module(
                        model, param_value, empty_param, param_name, casting_dtype, to_contiguous, rank, device_mesh
                    )
                else:
                    setattr(module, param_name.rsplit(".", 1)[1], torch.nn.Parameter(param_value, requires_grad=False))
                blocks_attr = f"{proj}_blocks"
                scales_attr = f"{proj}_scales"
                blocks = getattr(module, blocks_attr)
                scales = getattr(module, scales_attr)
                # Check if both blocks and scales both not on on meta device
                if blocks.device.type != "meta" and scales.device.type != "meta":
                    # need it for ep
                    local_experts = blocks.size(0)
                    if proj == "gate_up_proj":
                        blocks = blocks.view(local_experts, module.intermediate_size * 2, -1)
                    else:
                        blocks = blocks.view(local_experts, -1, module.intermediate_size // 2)
                    # TODO: we need to have the weights on cuda, refactor later
                    if getattr(target_device, "type", target_device) == "cpu":
                        target_device = "cuda"
                    # TODO: check why we still do move the tensors despite the context manager
                    blocks = blocks.to(target_device)
                    scales = scales.to(target_device)
                    with torch.cuda.device(target_device):
                        triton_weight_tensor, weight_scale = swizzle_mxfp4(
                            blocks.transpose(-2, -1), scales.transpose(-2, -1)
                        )

                    # need to overwrite the shapes for the kernels
                    if proj == "gate_up_proj":
                        triton_weight_tensor.shape = torch.Size(
                            [local_experts, module.hidden_size, module.intermediate_size * 2]
                        )
                    else:
                        triton_weight_tensor.shape = torch.Size(
                            [local_experts, module.intermediate_size, module.hidden_size]
                        )

                    # triton_weight_tensor is what needs to be passed in oai kernels. It stores the data, the shapes and any more objects. It is like a subtensor
                    setattr(module, proj, triton_weight_tensor)
                    setattr(
                        module,
                        f"{proj}_precision_config",
                        PrecisionConfig(weight_scale=weight_scale, flex_ctx=FlexCtx(rhs_data=InFlexData())),
                    )

                    # delete blocks and scales
                    delattr(module, scales_attr)
                    delattr(module, blocks_attr)
                    # setattr(module, blocks_attr, torch.nn.Parameter(triton_weight_tensor.storage.data, requires_grad=False))
                    del blocks
    pass
    patch_function(transformers.integrations.mxfp4, "load_and_swizzle_mxfp4", load_and_swizzle_mxfp4)

    try:
        from transformers.integrations.mxfp4 import _replace_with_mxfp4_linear
    except Exception as e:
        return raise_error("transformers.integrations.mxfp4._replace_with_mxfp4_linear", e)

    def replace_with_mxfp4_linear(
        model,
        modules_to_not_convert=None,
        current_key_name=None,
        quantization_config=None,
        config=None,
    ):
        if quantization_config.dequantize: return model
        modules_to_not_convert = ["lm_head"] if modules_to_not_convert is None else modules_to_not_convert
        if quantization_config.modules_to_not_convert is not None:
            modules_to_not_convert.extend(quantization_config.modules_to_not_convert)
        modules_to_not_convert = list(set(modules_to_not_convert))
        model, has_been_replaced = _replace_with_mxfp4_linear(
            model,
            modules_to_not_convert,
            current_key_name,
            quantization_config,
            config=config,
        )
        if not has_been_replaced:
            logger.warning_once(
                "You are loading your model using mixed-precision FP4 quantization but no linear modules were found in your model."
                " Please double check your model architecture, or submit an issue on github if you think this is"
                " a bug."
            )

        return model
    patch_function(transformers.integrations.mxfp4, "replace_with_mxfp4_linear", replace_with_mxfp4_linear)
pass
TEMPORARY_PATCHES.append(patch_gpt_oss)
