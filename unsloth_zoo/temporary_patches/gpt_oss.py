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
from importlib.metadata import version as importlib_version
from ..utils import Version
transformers_version = Version(importlib_version("transformers"))
has_static_cache = transformers_version >= Version("4.56.0.dev0")
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
from ..hf_utils import dtype_from_config
torch_cuda_device = torch.cuda.device


@torch_compile(dynamic = True, fullgraph = True)
def swiglu_torch_forward(a, alpha, limit, dtype = None):
    a_gelu = a[..., ::2].to(torch.float32)
    if limit is not None:
        a_gelu = a_gelu.clamp(max=limit)
    a_linear = a[..., 1::2].to(torch.float32)
    if limit is not None:
        a_linear = a_linear.clamp(min=-limit, max=limit)

    out_gelu = a_gelu * torch.sigmoid(alpha * a_gelu)
    out = out_gelu * (a_linear + 1)
    return out.to(a.dtype if dtype is None else dtype)
pass

@torch_compile(dynamic = True, fullgraph = True)
def swiglu_torch_backward(pre_act, alpha, limit, g1):
    g, l = pre_act[..., ::2].to(torch.float32), pre_act[..., 1::2].to(torch.float32)

    if limit is not None:
        mask_g = g <= limit
        mask_l = l.abs() <= limit
        ḡ = torch.where(mask_g, g, limit)
        l̄ = torch.where(mask_l, l, l.sign() * limit)
    else:                                            # no clipping
        mask_g = mask_l = torch.ones_like(g, dtype=bool)
        ḡ, l̄ = g, l

    σ   = torch.sigmoid(alpha * ḡ)
    dg  = (σ + alpha * ḡ * σ * (1 - σ)) * (l̄ + 1)
    dl  = ḡ * σ
    dg  = torch.where(mask_g, dg, 0.)                # clamp-grad
    dl  = torch.where(mask_l, dl, 0.)

    grad = torch.empty_like(pre_act)
    grad[..., ::2], grad[..., 1::2] = dg, dl
    return g1 * grad.to(g1.dtype)
pass


def patch_gpt_oss():
    try:
        import triton_kernels
    except Exception as e:
        return raise_error("Please install triton_kernels", e)

    try:
        import transformers.quantizers.quantizer_mxfp4
        def is_kernels_available(): return True
        transformers.quantizers.quantizer_mxfp4.is_kernels_available = is_kernels_available
        transformers.quantizers.quantizer_mxfp4.Mxfp4HfQuantizer.is_trainable = lambda *args, **kwargs: True
    except Exception as e:
        return raise_error("transformers.quantizers.quantizer_mxfp4.is_kernels_available", e)

    if hasattr(transformers.quantizers.quantizer_mxfp4.Mxfp4HfQuantizer, "_lazy_import_kernels"):
        transformers.quantizers.quantizer_mxfp4.Mxfp4HfQuantizer._lazy_import_kernels = lambda *args, **kwargs: triton_kernels

    try:
        transformers.quantizers.quantizer_mxfp4.Mxfp4HfQuantizer.is_trainable = lambda *args, **kwargs: True
    except Exception as e:
        return raise_error("transformers.quantizers.quantizer_mxfp4.Mxfp4HfQuantizer", e)

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

    def swizzle_mxfp4(w, w_scale, *args, **kwargs):
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
    patch_function(transformers.integrations.mxfp4, "swizzle_mxfp4", swizzle_mxfp4, match_level = "relaxed")

    class Mxfp4GptOssExperts_Training(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            hidden_states,
            self_class,
            routing_data,
            gather_idx,
            scatter_idx,
        ):
            pre_activation = matmul_ogs(
                hidden_states.to(torch.bfloat16), # tl.dot_scaled upcasts to BF16 for old hardware
                self_class.gate_up_proj,
                self_class.gate_up_proj_bias,
                routing_data,
                gather_indx=gather_idx,
                scatter_indx=None,
                precision_config=self_class.gate_up_proj_precision_config,
                gammas=None,
                fused_activation=None,
            )
            swiglu_output = swiglu_torch_forward(
                pre_activation,
                self_class.alpha,
                self_class.limit,
            )
            out = matmul_ogs(
                swiglu_output,
                self_class.down_proj,
                self_class.down_proj_bias,
                routing_data,
                gather_indx=None,
                scatter_indx=scatter_idx,
                precision_config=self_class.down_proj_precision_config,
                gammas=routing_data.gate_scal,
                fused_activation=None,
            )
            ctx.save_for_backward(
                pre_activation,
                routing_data.gate_scal,
                gather_idx.src_indx,
                gather_idx.dst_indx,
                scatter_idx.src_indx,
                scatter_idx.dst_indx,
            )
            ctx.self_class   = self_class
            ctx.gather_idx   = gather_idx
            ctx.scatter_idx  = scatter_idx
            ctx.routing_data = routing_data
            return out
        pass

        @staticmethod
        def backward(ctx, grad_token):
            raise NotImplementedError(
                "Backwards pass using MXFP4 is still under construction!\n"\
                "Instead, use `unsloth/gpt-oss-20b-BF16` for bfloat16 training which will work for LoRA.\n"\
                "Or, use `load_in_4bit = True` which allows finetuning."
            )
            (pre_act, gamma, gather_src, gather_dst, scatter_src, scatter_dst,) = ctx.saved_tensors
            self_class = ctx.self_class
            limit = self_class.limit
            alpha = self_class.alpha

            # 1) token ➜ expert (reverse of forward scatter)
            grad_exp = grad_token.index_select(0, scatter_src)
            grad_exp.mul_(gamma.unsqueeze(-1))
            # 2) grad_exp · Wdᵀ (reuse forward GEMM kernel)
            Wd_T = ctx.self_class.down_proj.data.swapaxes(1, 2).transpose(1, 2).contiguous().transpose(1, 2) # (E, d_model, d_ff)
            g1   = matmul_ogs(grad_exp, Wd_T, None, ctx.routing_data, gather_indx=ctx.scatter_idx)
            del Wd_T
            # 3) activation derivative
            g1 = swiglu_torch_backward(pre_act, alpha, limit, g1)
            # 4) g1 · Wuᵀ
            Wu_T = ctx.self_class.gate_up_proj.data.swapaxes(1, 2).transpose(1, 2).contiguous().transpose(1, 2) # (E, 2*d_ff, d_model)
            dx_exp = matmul_ogs(g1, Wu_T, None, ctx.routing_data, scatter_indx=ctx.gather_idx)
            del Wu_T

            # 5) expert ➜ token (reverse of forward gather)
            dx_token = torch.zeros_like(grad_token)
            dx_token.index_add_(0, gather_dst, dx_exp)
            return (dx_token, None, None, None, None,)
        pass
    pass

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
            self.limit = getattr(config, "swiglu_limit", 7.0)
            self.gate_up_proj_precision_config = None
            self.down_proj_precision_config = None

        def forward(self, hidden_states: torch.Tensor, routing_data, gather_idx, scatter_idx) -> torch.Tensor:
            with torch_cuda_device(hidden_states.device):
                if not hasattr(self, "act"):
                    self.act = FusedActivation(FnSpecs("swiglu", swiglu_fn, ("alpha", "limit")), (self.alpha, self.limit), 2)
                if not hidden_states.requires_grad:
                    intermediate_cache1 = matmul_ogs(
                        hidden_states.to(torch.bfloat16), # tl.dot_scaled upcasts to BF16 for old hardware
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
                        gammas=routing_data.gate_scal if routing_data else None,
                    )
                else:
                    intermediate_cache3 = Mxfp4GptOssExperts_Training.apply(
                        hidden_states,
                        self,
                        routing_data,
                        gather_idx,
                        scatter_idx,
                    )
            return intermediate_cache3
        pass
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

    def load_and_swizzle_mxfp4(module, param_name, param_value, target_device, *args, **kwargs):
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
    patch_function(transformers.integrations.mxfp4, "load_and_swizzle_mxfp4", load_and_swizzle_mxfp4, match_level = "relaxed")

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


class GptOssExperts(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = config.intermediate_size
        self.alpha = 1.702
        self.limit = getattr(config, "swiglu_limit", 7.0)
        self.dtype = dtype_from_config(config)

        self.gate_up_projs = nn.ModuleList([
            nn.Linear(self.hidden_size, 2 * self.expert_dim, dtype=self.dtype)
            for _ in range(self.num_experts)
        ])
        self.down_projs = nn.ModuleList([
            nn.Linear(self.expert_dim, self.hidden_size, dtype=self.dtype)
            for _ in range(self.num_experts)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_indices = None,
        routing_weights = None
    ) -> torch.Tensor:

        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        num_experts = routing_weights.shape[1]
        if self.training:
            next_states = torch.zeros_like(hidden_states, dtype=torch.float32, device=hidden_states.device)
            # with torch.no_grad():
                # expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=num_experts)
                # expert_mask = expert_mask.permute(2, 1, 0)
                # expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            # for expert_idx in expert_hitted[:]:
            for expert_idx in range(num_experts):
                with torch.no_grad():
                    # _, token_idx = torch.where(expert_mask[expert_idx[0]])
                    token_idx, _ = torch.where(router_indices == expert_idx)
                current_state = hidden_states[token_idx]
                gate_up = self.gate_up_projs[expert_idx](current_state)
                gated_output = swiglu_torch_forward(gate_up, self.alpha, self.limit)
                # gate, up = gate_up[..., ::2], gate_up[..., 1::2]
                # gate = gate.clamp(min=None, max=self.limit)
                # up = up.clamp(min=-self.limit, max=self.limit)
                # glu = gate * torch.sigmoid(gate * self.alpha)
                # gated_output = (up + 1) * glu
                out = self.down_projs[expert_idx](gated_output)
                weighted_output = out * routing_weights[token_idx, expert_idx, None].to(torch.float32)
                next_states.index_add_(0, token_idx, weighted_output)
            next_states = next_states.view(batch_size, -1, self.hidden_size)
            return next_states.to(hidden_states.dtype)
        else:
            X_rep = hidden_states.unsqueeze(0).expand(num_experts, -1, -1)
            gate_up_list = [up_l(X_rep[e]) for e, up_l in enumerate(self.gate_up_projs)]
            gate_up = torch.stack(gate_up_list, dim=0)
            fused = swiglu_torch_forward(gate_up, self.alpha, self.limit, dtype = X_rep.dtype)
            # gate = gate_up[..., ::2]
            # up_h = gate_up[..., 1::2]
            # gate = gate.clamp(max=self.limit)
            # up_h = up_h.clamp(min=-self.limit, max=self.limit)
            # glu = gate * torch.sigmoid(gate * self.alpha)
            # fused = (up_h + 1) * glu
            out_list = [down_l(fused[e]) for e, down_l in enumerate(self.down_projs)]
            outs = torch.stack(out_list, dim=0)
            rw = routing_weights.transpose(0, 1).unsqueeze(-1)
            mixed = (outs.to(torch.float32) * rw.to(torch.float32)).sum(dim=0)
            return mixed.view(batch_size, -1, self.hidden_size).to(hidden_states.dtype)
pass

class GptOssTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.linear = nn.Linear(self.hidden_dim, self.num_experts, dtype=dtype_from_config(config))

    @torch_compile(dynamic = True, fullgraph = True)
    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = self.linear(hidden_states.to(self.linear.weight.dtype)) # (batch_size * seq_len, num_experts)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1) # (seq_len, top_k)
        dtype = torch.float32 if router_logits.dtype == torch.float16 else router_logits.dtype
        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=torch.float32).to(dtype)
        router_scores = torch.zeros_like(router_logits, dtype = dtype).scatter_(1, router_indices, router_top_value)
        return router_scores, router_indices
pass


# Combo kernels uses too much VRAM for low memory GPUs
from ..device_type import DEVICE_TYPE
if DEVICE_TYPE == "xpu":
    device_memory = torch.xpu.memory.mem_get_info(0)[-1]
else:
    device_memory = torch.cuda.memory.mem_get_info(0)[-1]
use_combo_kernels = False if device_memory/1024/1024/1024 <= 40 else True
fused_torch_compile_options = get_torch_compile_options(
    epilogue_fusion = True,
    max_autotune = False, # Too slow
    shape_padding = True,
    cudagraphs = True,
    coordinate_descent_tuning = use_combo_kernels, # Very slow!
    combo_kernels = use_combo_kernels,
    memory_planning = True,
    multi_kernel = False, # Fails on torch 2.10 nightly
    use_block_ptr = True,
    logging = UNSLOTH_ENABLE_LOGGING,
)
no_combo_fused_torch_compile_options = get_torch_compile_options(
    epilogue_fusion = True,
    max_autotune = False, # Too slow
    shape_padding = True,
    cudagraphs = True,
    coordinate_descent_tuning = use_combo_kernels, # Very slow!
    combo_kernels = False, # Breaks on attention
    memory_planning = True,
    multi_kernel = False, # Fails on torch 2.10 nightly
    use_block_ptr = True,
    logging = UNSLOTH_ENABLE_LOGGING,
)

@_torch_compile(dynamic = None, fullgraph = True, options = fused_torch_compile_options)
def moe_forward_inference(self, hidden_states):
    """Torch compile for forward inference path only with CUDAGraphs"""
    # Router
    router_scores, router_indices = self.router(hidden_states)
    routing_weights = router_scores
    moe = self.experts
    batch_size = hidden_states.shape[0]
    hidden_states = hidden_states.reshape(-1, moe.hidden_size)

    num_experts = routing_weights.shape[1]
    X_rep = hidden_states.unsqueeze(0).expand(num_experts, -1, -1)

    # Gate up projection
    gate_up_list = [up_l(X_rep[e]) for e, up_l in enumerate(moe.gate_up_projs)]
    gate_up = torch.stack(gate_up_list, dim = 0)
    dtype = torch.float32 if hidden_states.dtype != torch.bfloat16 else hidden_states.dtype
    fused = swiglu_torch_forward(gate_up, moe.alpha, moe.limit, dtype = dtype)

    # Down projection must be done in float32 if not bfloat16 otherwise infinites
    fused = fused.to(dtype)
    device_type = fused.device.type if isinstance(fused.device.type, str) and fused.device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False): # Force float32
        out_list = [down_l(fused[e].to(dtype)) for e, down_l in enumerate(moe.down_projs)]
    outs = torch.stack(out_list, dim=0)

    rw = routing_weights.to(dtype).transpose(0, 1).unsqueeze(-1)
    mixed = (outs * rw).sum(dim=0)
    return mixed.view(batch_size, -1, moe.hidden_size).to(hidden_states.dtype)
pass

@torch_compile(dynamic = True, fullgraph = True)
def moe_router_forward(self, hidden_states):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states.to(self.weight.dtype), self.weight, self.bias)  # (seq_len, num_experts)
    router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
    dtype = torch.float32 if router_logits.dtype == torch.float16 else router_logits.dtype
    router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=torch.float32).to(dtype)
    router_scores = torch.zeros_like(router_logits, dtype = dtype).scatter_(1, router_indices, router_top_value)
    return router_scores, router_indices
pass

# Combo Kernels errors with InductorError: AttributeError: 'NullKernelHandler' object has no attribute 'index_to_str'
@_torch_compile(dynamic = None, fullgraph = True, options = no_combo_fused_torch_compile_options)
def moe_forward_inference_bf16(self, hidden_states):
    router_scores, router_indices = moe_router_forward(self.router, hidden_states)
    routing_weights = router_scores

    moe = self.experts
    batch_size = hidden_states.shape[0]
    hidden_states = hidden_states.reshape(-1, moe.hidden_size)
    num_experts = routing_weights.shape[1]
    hidden_states = hidden_states.repeat(num_experts, 1)
    hidden_states = hidden_states.view(num_experts, -1, moe.hidden_size)
    gate_up = torch.bmm(hidden_states, moe.gate_up_proj) + moe.gate_up_proj_bias[..., None, :]
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(min=None, max=moe.limit)
    up = up.clamp(min=-moe.limit, max=moe.limit)
    glu = gate * torch.sigmoid(gate.to(torch.float32) * moe.alpha).to(gate.dtype)
    next_states = torch.bmm(((up + 1) * glu), moe.down_proj)
    next_states = next_states + moe.down_proj_bias[..., None, :]
    next_states = next_states.view(num_experts, batch_size, -1, moe.hidden_size)
    next_states = next_states * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
    next_states = next_states.sum(dim=0)
    return next_states
pass


class GptOssMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = GptOssTopKRouter(config)
        self.experts = GptOssExperts(config)

    def forward(self, hidden_states):
        bsz, qlen, hd = hidden_states.shape
        if qlen == 1 and not self.training:
            return moe_forward_inference(self, hidden_states), None
        router_scores, router_indices = self.router(hidden_states)  # (num_experts, seq_len)
        routed_out = self.experts(hidden_states, router_indices=router_indices, routing_weights=router_scores)
        return routed_out, router_scores
pass

def patch_gpt_oss_linearized():
    if "gpt_oss" not in os.environ.get("UNSLOTH_MODEL_NAME", ""): return
    if "_load_in_4bit_" not in os.environ.get("UNSLOTH_MODEL_NAME", ""): return
    try:
        import transformers.models.gpt_oss.modeling_gpt_oss
    except Exception as e:
        return raise_error("transformers.models.gpt_oss.modeling_gpt_oss", e)

    # We find down_proj overflows in GPT OSS
    if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1":
        def forward(
            self,
            hidden_states: torch.Tensor,
            router_indices = None,
            routing_weights = None
        ) -> torch.Tensor:

            batch_size = hidden_states.shape[0]
            hidden_states = hidden_states.reshape(-1, self.hidden_size)
            num_experts = routing_weights.shape[1]
            if self.training:
                next_states = torch.zeros_like(hidden_states, dtype=torch.float32, device=hidden_states.device)
                # with torch.no_grad():
                #     expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=num_experts)
                #     expert_mask = expert_mask.permute(2, 1, 0)
                #     expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
                # for expert_idx in expert_hitted[:]:
                for expert_idx in range(num_experts):
                    with torch.no_grad():
                        # _, token_idx = torch.where(expert_mask[expert_idx[0]])
                        token_idx, _ = torch.where(router_indices == expert_idx)
                    current_state = hidden_states[token_idx]
                    gate_up = self.gate_up_projs[expert_idx](current_state)
                    down_proj = self.down_projs[expert_idx]
                    gated_output = swiglu_torch_forward(gate_up, self.alpha, self.limit, dtype = torch.float32)
                    # gate, up = gate_up[..., ::2], gate_up[..., 1::2]
                    # gate = gate.clamp(min=None, max=self.limit)
                    # up = up.clamp(min=-self.limit, max=self.limit)
                    # glu = gate * torch.sigmoid(gate * self.alpha)
                    # gated_output = (up + 1) * glu

                    # Force float32 matrix multiply on some down projection modules
                    gated_output = gated_output.to(torch.float32)
                    device_type = gated_output.device.type if isinstance(gated_output.device.type, str) and gated_output.device.type != "mps" else "cpu"
                    with torch.autocast(device_type=device_type, enabled=False): # Force float32
                        out = down_proj(gated_output)
                    weighted_output = out.to(torch.float32) * routing_weights[token_idx, expert_idx, None].to(torch.float32)
                    next_states.index_add_(0, token_idx, weighted_output)
                next_states = next_states.view(batch_size, -1, self.hidden_size)
                return next_states.to(torch.float32)
            else:
                X_rep = hidden_states.unsqueeze(0).expand(num_experts, -1, -1)
                gate_up_list = [up_l(X_rep[e]) for e, up_l in enumerate(self.gate_up_projs)]
                gate_up = torch.stack(gate_up_list, dim=0)
                dtype = torch.float32 if hidden_states.dtype != torch.bfloat16 else hidden_states.dtype
                fused = swiglu_torch_forward(gate_up, self.alpha, self.limit, dtype = dtype)
                # gate = gate_up[..., ::2]
                # up_h = gate_up[..., 1::2]
                # gate = gate.clamp(max=self.limit)
                # up_h = up_h.clamp(min=-self.limit, max=self.limit)
                # glu = gate * torch.sigmoid(gate * self.alpha)
                # fused = (up_h + 1) * glu

                # Force float32 matrix multiply on down projection only
                device_type = fused.device.type if isinstance(fused.device.type, str) and fused.device.type != "mps" else "cpu"
                with torch.autocast(device_type=device_type, enabled=False): # Force float32
                    out_list = [
                        down_l(fused[e].to(dtype))
                        for e, down_l in enumerate(self.down_projs)
                    ]
                outs = torch.stack(out_list, dim=0)
                rw = routing_weights.transpose(0, 1).unsqueeze(-1)
                mixed = (outs.to(dtype) * rw.to(dtype)).sum(dim=0)
                return mixed.view(batch_size, -1, self.hidden_size).to(hidden_states.dtype)
            pass
        pass
        GptOssExperts.forward = forward
    pass

    transformers.models.gpt_oss.modeling_gpt_oss.GptOssExperts = GptOssExperts
    transformers.models.gpt_oss.modeling_gpt_oss.GptOssTopKRouter = GptOssTopKRouter
    transformers.models.gpt_oss.modeling_gpt_oss.GptOssMLP = GptOssMLP
    return
pass
TEMPORARY_PATCHES.append(patch_gpt_oss_linearized)


def patch_GptOssAttention():
    if os.environ.get("UNSLOTH_ENABLE_FLEX_ATTENTION", "1") == "0": return
    if "gpt_oss" not in os.environ.get("UNSLOTH_MODEL_NAME", ""): return
    try:
        from ..flex_attention import (
            flex_attention_with_sink,
            is_flex_attention_decoding,
            flex_attention_with_sink_decoding,
            flex_attention_add_sinks,
        )
        assert flex_attention_with_sink is not None
    except Exception as e:
        return raise_error("flex_attention_with_sink", e)
    try:
        import transformers.models.gpt_oss.modeling_gpt_oss
        transformers.models.gpt_oss.modeling_gpt_oss.GptOssAttention
        from transformers.models.gpt_oss.modeling_gpt_oss import apply_rotary_pos_emb
    except Exception as e:
        return raise_error("transformers.models.gpt_oss.modeling_gpt_oss.GptOssAttention", e)

    torch._dynamo.config.cache_size_limit = 256
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    F_softmax = torch.nn.functional.softmax
    F_dropout = nn.functional.dropout
    matmul = torch.matmul
    def inplace_eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)

        bsz, n_heads, qlen, _  = query.shape
        bsz, n_heads, kvlen, _ = key_states.shape
        combined_logits = key_states.new_empty((bsz, n_heads, qlen, kvlen+1))

        attn_weights = matmul(query, key_states.transpose(2, 3), out = combined_logits[:,:,:,:kvlen])
        attn_weights *= scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights += causal_mask

        # sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
        # combined_logits = torch.cat([attn_weights, sinks], dim=-1)
        combined_logits[:, :, :, -1] = module.sinks.reshape(1, -1, 1)

        # This was not in the original implementation and slightly affect results; it prevents overflow in BF16/FP16
        # when training with bsz>1 we clamp max values.
        # combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        combined_logits[:] = F_softmax(combined_logits, dim=-1, dtype=torch.float32)
        probs = combined_logits
        scores = probs[..., :-1]  # we drop the sink here
        attn_weights = F_dropout(scores, p=dropout, training=module.training, inplace=True)
        attn_output = matmul(attn_weights, value_states, out = query)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, None
    pass

    def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
        attn_weights = matmul(query, key_states.transpose(2, 3))
        attn_weights *= scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights += causal_mask

        sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
        combined_logits = torch.cat([attn_weights, sinks], dim=-1)

        # This was not in the original implementation and slightly affect results; it prevents overflow in BF16/FP16
        # when training with bsz>1 we clamp max values.
        # combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        combined_logits[:] = F_softmax(combined_logits, dim=-1, dtype=torch.float32)
        probs = combined_logits
        scores = probs[..., :-1]  # we drop the sink here
        attn_weights = F_dropout(scores, p=dropout, training=module.training, inplace=True)
        attn_output = matmul(attn_weights, value_states, out = query)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, None
    pass

    apply_rotary_pos_emb = torch_compile(apply_rotary_pos_emb)
    if False: # Version(torch.__version__) >= Version("2.10.0"):
        eager_attention_forward = torch_compile(eager_attention_forward, dynamic = None, fullgraph = True)
    else:
        # Too many recompilation failures on 2.8.0, 2.9.0
        eager_attention_forward = inplace_eager_attention_forward

    def forward_function(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: KWARGS_TYPE,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states   = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_value is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # flex_attention_with_sink only works for training since KV cache is wrong
        # switch to flex_attention_with_sink which allows all to work
        # if is_flex_attention_decoding(self, query_states) and has_static_cache:
        #     attn_output, logsumexp = flex_attention_with_sink_decoding(
        #         self,
        #         query_states,
        #         key_states,
        #         value_states,
        #     )
        #     attn_output = flex_attention_add_sinks(
        #         self,
        #         attn_output,
        #         logsumexp,
        #     )
        # else:
        #     attn_output = flex_attention_with_sink(
        #         self,
        #         query_states,
        #         key_states,
        #         value_states,
        #         attention_mask,
        #         has_static_cache = has_static_cache,
        #     )
        # attn_weights = None
        if self.training:
            attn_output = flex_attention_with_sink(
                self,
                query_states,
                key_states,
                value_states,
            )
            attn_weights = None
        else:
            # Weirdly for inference, flex attention returns gibberish
            # Most likely due to left padding
            attn_output, attn_weights = eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                s_aux=self.sinks,  # diff with Llama
                **kwargs,
            )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    pass

    functions = []
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: KWARGS_TYPE,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        return forward_function(self, hidden_states, position_embeddings, attention_mask, past_key_value, cache_position, **kwargs)
    functions.append(forward)
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: KWARGS_TYPE,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        return forward_function(self, hidden_states, position_embeddings, attention_mask, past_key_values, cache_position, **kwargs)
    functions.append(forward)
    patch_function_past_key_values(transformers.models.gpt_oss.modeling_gpt_oss.GptOssAttention, "forward", functions)
    # Set env variable for padding purposes
    os.environ["UNSLOTH_ENABLE_FLEX_ATTENTION"] = "1"
pass
TEMPORARY_PATCHES.append(patch_GptOssAttention)


def patch_GptOssModel():
    if os.environ.get("UNSLOTH_ENABLE_FLEX_ATTENTION", "1") == "0": return
    if "gpt_oss" not in os.environ.get("UNSLOTH_MODEL_NAME", ""): return
    try:
        import transformers.models.gpt_oss.modeling_gpt_oss
        transformers.models.gpt_oss.modeling_gpt_oss.GptOssModel
        from transformers.models.gpt_oss.modeling_gpt_oss import MoeModelOutputWithPast
        from transformers.models.gpt_oss.modeling_gpt_oss import apply_rotary_pos_emb
    except Exception as e:
        return raise_error("transformers.models.gpt_oss.modeling_gpt_oss.GptOssModel", e)
    try:
        from transformers.models.gpt_oss.modeling_gpt_oss import DynamicCache
    except Exception as e:
        raise_error("transformers.models.gpt_oss.modeling_gpt_oss.GptOssModel", e)
        DynamicCache = lambda *args, **kwargs: None

    torch._dynamo.config.cache_size_limit = 256

    # Disable mask creations since we don't need them for GPT-OSS
    import transformers.masking_utils
    import transformers.generation.utils
    def wrap(f):
        def return_attention_mask(*args, **kwargs):
            if kwargs["input_embeds"].requires_grad:
                if "attention_mask" in kwargs:
                    return kwargs["attention_mask"]
                for arg in args:
                    if type(arg) is torch.Tensor and arg.dtype == torch.int32:
                        return arg
            else:
                # Eager
                return f(*args, **kwargs)
            pass
        return return_attention_mask
    pass
    create_causal_mask = getattr(
        transformers.masking_utils,
        "_old_create_causal_mask",
        getattr(transformers.masking_utils, "create_causal_mask", None),
    )
    create_sliding_window_causal_mask = getattr(
        transformers.masking_utils,
        "_old_create_sliding_window_causal_mask",
        getattr(transformers.masking_utils, "create_sliding_window_causal_mask", None),
    )
    if create_causal_mask is None:
        return raise_error("transformers.masking_utils.create_causal_mask")
    if create_sliding_window_causal_mask is None:
        return raise_error("transformers.masking_utils.create_sliding_window_causal_mask")
    if not hasattr(transformers.masking_utils, "__patched_causal_mask__"):
        transformers.masking_utils._old_create_causal_mask = _torch_compile(transformers.masking_utils.create_causal_mask, fullgraph = False, dynamic = True)
        transformers.masking_utils._old_create_sliding_window_causal_mask = _torch_compile(transformers.masking_utils.create_sliding_window_causal_mask, fullgraph = False, dynamic = True)
        transformers.masking_utils.create_causal_mask = wrap(create_causal_mask)
        transformers.masking_utils.create_sliding_window_causal_mask = wrap(create_sliding_window_causal_mask)
        transformers.models.gpt_oss.modeling_gpt_oss.create_causal_mask = transformers.masking_utils.create_causal_mask
        transformers.models.gpt_oss.modeling_gpt_oss.create_sliding_window_causal_mask = transformers.masking_utils.create_sliding_window_causal_mask
        transformers.masking_utils.create_masks_for_generate = wrap(transformers.masking_utils.create_masks_for_generate)
        transformers.generation.utils.create_masks_for_generate = wrap(transformers.generation.utils.create_masks_for_generate)
        transformers.masking_utils.__patched_causal_mask__ = True
    pass

    from ..flex_attention import (
        is_flex_attention_decoding,
        flex_attention_with_sink_decoding,
        flex_attention_add_sinks,
    )
    apply_rotary_pos_emb = torch_compile(apply_rotary_pos_emb)
    try:
        from transformers.integrations.mxfp4 import mlp_forward
    except:
        mlp_forward = None

    def pre_attention_decoding(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: KWARGS_TYPE,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states   = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        return query_states, key_states, value_states, input_shape
    pass
    # Do flex_attention_with_sink_decoding with cannot be compiled
    # attn_output, logsumexp = flex_attention_with_sink_decoding(
    #     self,
    #     query_states,
    #     key_states,
    #     value_states,
    # )
    def post_attention_decoding(self_attn, attn_output, logsumexp, input_shape):
        attn_output = flex_attention_add_sinks(self_attn, attn_output, logsumexp)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self_attn.o_proj(attn_output)
        return attn_output
    pass

    # RMSNorm forward
    def rms_layernorm_forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.square().mean(-1, keepdim=True)
        variance += self.variance_epsilon
        hidden_states *= torch.rsqrt_(variance)
        hidden_states *= self.weight.to(hidden_states.device).to(torch.float32)
        return hidden_states.to(input_dtype)  # main diff with Llama
    pass

    # Re-compiling for each new sequence length which is NOT ideal
    @_torch_compile(dynamic = True, fullgraph = False, mode = "reduce-overhead")
    def pre_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    ):
        hidden_states = rms_layernorm_forward(self.input_layernorm, hidden_states)
        # Self Attention
        query_states, key_states, value_states, input_shape = pre_attention_decoding(
            self=self.self_attn,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        return query_states, key_states, value_states, input_shape
    pass
    fused_torch_compile_options = get_torch_compile_options(
        epilogue_fusion = True,
        max_autotune = False, # Too slow
        shape_padding = True,
        cudagraphs = True,
        coordinate_descent_tuning = False,
        combo_kernels = False,
        memory_planning = True,
        multi_kernel = False, # Fails on torch 2.10 nightly
        use_block_ptr = True,
        logging = UNSLOTH_ENABLE_LOGGING,
    )
    @_torch_compile(dynamic = None, fullgraph = True, options = fused_torch_compile_options)
    def post_forward(
        self,
        residual: torch.Tensor,
        attn_output: torch.Tensor,
        logsumexp: torch.Tensor,
        input_shape,
    ):
        hidden_states = post_attention_decoding(self.self_attn, attn_output, logsumexp, input_shape)
        hidden_states += residual

        # Fully Connected
        residual = hidden_states.clone()
        hidden_states = rms_layernorm_forward(self.post_attention_layernorm, hidden_states)
        return hidden_states, residual
    pass

    def inference_forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_values,
        use_cache,
        cache_position,
        position_embeddings,
        **kwargs,
    ):
        residual = hidden_states.clone()
        hidden_states = rms_layernorm_forward(self.input_layernorm, hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states += residual.to(hidden_states.device)

        # Fully Connected
        residual = hidden_states.clone()
        hidden_states = rms_layernorm_forward(self.post_attention_layernorm, hidden_states)
        return hidden_states, residual
    pass
    # if has_static_cache and Version(torch.__version__) >= Version("2.10.0"):
    #     # torch 2.9.0 has excessive compilations
    #     inference_forward = _torch_compile(inference_forward, dynamic = None, fullgraph = True, options = fused_torch_compile_options)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: KWARGS_TYPE,
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            # Account for CPU offloaded embed_tokens
            embed_device = self.embed_tokens.weight.device
            inputs_embeds = self.embed_tokens(input_ids.to(embed_device, non_blocking = True)).to(input_ids.device)
        if not self.training:
            inputs_embeds.requires_grad_(False)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        try:
            torch._dynamo.mark_static (hidden_states, 0)
            torch._dynamo.mark_dynamic(hidden_states, 1)
            torch._dynamo.mark_static (hidden_states, 2)
        except:
            pass

        # It may already have been prepared by e.g. `generate`
        if not self.training and not isinstance(attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
            }
            attention_mask = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        # is_decoding = is_flex_attention_decoding(self.layers[0].self_attn, hidden_states)
        bsz, qlen, hd = hidden_states.shape
        if not self.training and qlen == 1 and isinstance(attention_mask, dict):
            # Add hack since residuals need to clone outside of the torch.compile region??
            # This forces it to free past residuals
            torch.compiler.cudagraph_mark_step_begin()
            for decoder_layer in self.layers:
                hidden_states, residual = inference_forward(
                    decoder_layer,
                    hidden_states,
                    attention_mask[decoder_layer.attention_type],
                    position_ids,
                    past_key_values,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    **kwargs,
                )
                if hasattr(decoder_layer.mlp.experts, "gate_up_projs"):
                    hidden_states = moe_forward_inference(decoder_layer.mlp, hidden_states)
                elif decoder_layer.mlp.experts.__class__.__name__ == "Mxfp4GptOssExperts":
                    if mlp_forward is None:
                        raise RuntimeError("Unsloth: MXFP4 forward is not found")
                    hidden_states, _ = mlp_forward(decoder_layer.mlp, hidden_states)
                else:
                    hidden_states = moe_forward_inference_bf16(decoder_layer.mlp, hidden_states)
                hidden_states += residual
            pass
            hidden_states = rms_layernorm_forward(self.norm, hidden_states)
        else:
            for decoder_layer in self.layers:
                mask = attention_mask[decoder_layer.attention_type] if isinstance(attention_mask, dict) else attention_mask
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            pass
            hidden_states = self.norm(hidden_states)
        # Fix float16 / float32 mismatching
        hidden_states = hidden_states.to(inputs_embeds.dtype)
        return process_return(MoeModelOutputWithPast, {
            "last_hidden_state" : hidden_states,
            "past_key_values" : past_key_values,
        })
    patch_function(transformers.models.gpt_oss.modeling_gpt_oss.GptOssModel, "forward", forward, match_level = "relaxed")
pass
TEMPORARY_PATCHES.append(patch_GptOssModel)

try:
    from openai_harmony import (
        Author,
        Conversation,
        DeveloperContent,
        HarmonyEncodingName,
        Message,
        Role,
        SystemContent,
        ToolDescription,
        load_harmony_encoding,
        ReasoningEffort
    )
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
except:
    pass
def encode_conversations_with_harmony(
    messages,
    reasoning_effort = "medium",
    add_generation_prompt = True,
    tool_calls = None,
    developer_instructions = None,
    model_identity = "You are ChatGPT, a large language model trained by OpenAI.",
):
    try:
        SystemContent
    except:
        raise ImportError("Please install openai_harmony via `pip install openai_harmony`")

    assert reasoning_effort in ("low", "medium", "high")

    match reasoning_effort:
        case "low":    harmony_reasoning = ReasoningEffort.LOW
        case "medium": harmony_reasoning = ReasoningEffort.MEDIUM
        case "high":   harmony_reasoning = ReasoningEffort.HIGH

    convos = []

    # Create system message
    import datetime
    today = datetime.datetime.today().strftime("%Y-%m-%d")
    system = Message.from_role_and_content(Role.SYSTEM,
        SystemContent.new()
            .with_model_identity(model_identity)
            .with_reasoning_effort(harmony_reasoning)
            .with_conversation_start_date(today)
            .with_knowledge_cutoff("2024-06")
            .with_required_channels(["analysis", "commentary", "final"])
    )
    convos.append(system)

    # Developer message and tool calling
    dev = DeveloperContent.new()
    if developer_instructions is not None: dev = dev.with_instructions(developer_instructions)
    if tool_calls is not None:
        new_tools = []
        for function in tool_calls:
            function = function["function"]
            name = function["name"]
            description = function["description"]
            parameters = function["parameters"]
            tool = ToolDescription.new(name, description, parameters)
            new_tools.append(tool)
        dev = dev.with_function_tools(new_tools)
    pass
    if developer_instructions is not None or tool_calls is not None:
        dev = Message.from_role_and_content(Role.DEVELOPER, dev)
        convos.append(dev)

    for message in messages:
        if message["role"] == "user":
            convos.append(
                Message.from_role_and_content(Role.USER, message["content"])
            )
        elif message["role"] == "assistant":
            if "thinking" in message:
                x = Message.from_role_and_content(Role.ASSISTANT, message["content"])
                x = x.with_channel("analysis")
            elif "tool_calls" in message:
                x = Message.from_role_and_content(Role.ASSISTANT, message['tool_calls'][0]["arguments"])
                x = x.with_channel("commentary")\
                     .with_recipient(f"functions.{message['tool_calls'][0]['name']}")\
                     .with_content_type("json")
            else:
                x = Message.from_role_and_content(Role.ASSISTANT, message["content"])
                x = x.with_channel("final")
            convos.append(x)
        elif message["role"] == "tool":
            x = Message.from_author_and_content(
                    Author.new(Role.TOOL, f"functions.{message['name']}"),
                    message["content"],
                ).with_recipient("assistant").with_channel("commentary")
            convos.append(x)
    pass

    # Create Harmony conversations
    convos = Conversation.from_messages(convos)
    if add_generation_prompt:
        harmony_input_ids = encoding.render_conversation_for_completion(convos, Role.ASSISTANT)
    else:
        harmony_input_ids = encoding.render_conversation(convos)
    harmony_decoded_text = encoding.decode(harmony_input_ids)
    return harmony_decoded_text, harmony_input_ids
pass


# Fix https://github.com/huggingface/transformers/pull/40474
# RuntimeError: Unsloth: Failed to load model. Both AutoConfig and PeftConfig loading failed.
# AutoConfig error: 'GptOssConfig' object has no attribute 'max_position_embeddings'
try:
    from transformers.configuration_utils import layer_type_validation
    try:
        from transformers.configuration_utils import PreTrainedConfig
        PretrainedConfig = PreTrainedConfig
    except:
        from transformers.configuration_utils import PretrainedConfig

    from transformers.modeling_rope_utils import rope_config_validation

    class Old_GptOssConfig(PretrainedConfig):
        r"""
        This will yield a configuration to that of the BERT
        [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) architecture.

        """

        model_type = "gpt_oss"
        base_model_pp_plan = {
            "embed_tokens": (["input_ids"], ["inputs_embeds"]),
            "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
            "norm": (["hidden_states"], ["hidden_states"]),
        }
        base_model_tp_plan = {
            "layers.*.self_attn.q_proj": "colwise",
            "layers.*.self_attn.k_proj": "colwise",
            "layers.*.self_attn.v_proj": "colwise",
            "layers.*.self_attn.o_proj": "rowwise",
            "layers.*.self_attn.sinks": "local_rowwise",
            "layers.*.mlp.experts": "gather",
            "layers.*.mlp.router": "ep_router",
            "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
            "layers.*.mlp.experts.gate_up_proj_bias": "grouped_gemm",
            "layers.*.mlp.experts.down_proj": "grouped_gemm",
            "layers.*.mlp.experts.down_proj_bias": "grouped_gemm",
        }

        def __init__(
            self,
            num_hidden_layers: int = 36,
            num_local_experts: int = 128,
            vocab_size: int = 201088,
            hidden_size: int = 2880,
            intermediate_size: int = 2880,
            head_dim: int = 64,
            num_attention_heads: int = 64,
            num_key_value_heads: int = 8,
            sliding_window: int = 128,
            rope_theta: float = 150000.0,
            tie_word_embeddings=False,
            hidden_act: str = "silu",
            initializer_range: float = 0.02,
            max_position_embeddings=131072,
            rms_norm_eps: float = 1e-5,
            rope_scaling={"rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0, "beta_slow": 1.0, "truncate": False},
            attention_dropout: float = 0.0,
            num_experts_per_tok=4,
            router_aux_loss_coef: float = 0.9,
            output_router_logits=False,
            use_cache=True,
            layer_types=None,
            **kwargs,
        ):
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.num_local_experts = num_local_experts
            self.sliding_window = sliding_window
            self.num_experts_per_tok = num_experts_per_tok
            # for backward compatibility
            if num_key_value_heads is None:
                num_key_value_heads = num_attention_heads

            self.num_key_value_heads = num_key_value_heads
            self.hidden_act = hidden_act
            self.initializer_range = initializer_range
            self.rms_norm_eps = rms_norm_eps
            self.rope_theta = rope_theta
            self.rope_scaling = rope_scaling
            self.attention_dropout = attention_dropout
            self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
            self.layer_types = layer_types
            if self.layer_types is None:
                self.layer_types = [
                    "sliding_attention" if bool((i + 1) % 2) else "full_attention" for i in range(self.num_hidden_layers)
                ]
            layer_type_validation(self.layer_types)

            # Validate the correctness of rotary position embeddings parameters
            # BC: if there is a 'type' field, copy it it to 'rope_type'.
            if self.rope_scaling is not None and "type" in self.rope_scaling:
                self.rope_scaling["rope_type"] = self.rope_scaling["type"]
            rope_config_validation(self)

            self.attention_bias = True
            self.max_position_embeddings = max_position_embeddings
            self.router_aux_loss_coef = router_aux_loss_coef
            self.output_router_logits = output_router_logits
            self.use_cache = use_cache
            super().__init__(
                tie_word_embeddings=tie_word_embeddings,
                **kwargs,
            )

    class GptOssConfig(PretrainedConfig):
        r"""
        This will yield a configuration to that of the BERT
        [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) architecture.

        """

        model_type = "gpt_oss"
        base_model_pp_plan = {
            "embed_tokens": (["input_ids"], ["inputs_embeds"]),
            "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
            "norm": (["hidden_states"], ["hidden_states"]),
        }
        base_model_tp_plan = {
            "layers.*.self_attn.q_proj": "colwise",
            "layers.*.self_attn.k_proj": "colwise",
            "layers.*.self_attn.v_proj": "colwise",
            "layers.*.self_attn.o_proj": "rowwise",
            "layers.*.self_attn.sinks": "local_rowwise",
            "layers.*.mlp.experts": "gather",
            "layers.*.mlp.router": "ep_router",
            "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
            "layers.*.mlp.experts.gate_up_proj_bias": "grouped_gemm",
            "layers.*.mlp.experts.down_proj": "grouped_gemm",
            "layers.*.mlp.experts.down_proj_bias": "grouped_gemm",
        }

        def __init__(
            self,
            num_hidden_layers: int = 36,
            num_local_experts: int = 128,
            vocab_size: int = 201088,
            hidden_size: int = 2880,
            intermediate_size: int = 2880,
            head_dim: int = 64,
            num_attention_heads: int = 64,
            num_key_value_heads: int = 8,
            sliding_window: int = 128,
            rope_theta: float = 150000.0,
            tie_word_embeddings=False,
            hidden_act: str = "silu",
            initializer_range: float = 0.02,
            max_position_embeddings=131072,
            rms_norm_eps: float = 1e-5,
            rope_scaling={"rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0, "beta_slow": 1.0, "truncate": False},
            attention_dropout: float = 0.0,
            num_experts_per_tok=4,
            router_aux_loss_coef: float = 0.9,
            output_router_logits=False,
            use_cache=True,
            layer_types=None,
            **kwargs,
        ):
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.num_local_experts = num_local_experts
            self.sliding_window = sliding_window
            self.num_experts_per_tok = num_experts_per_tok
            # for backward compatibility
            if num_key_value_heads is None:
                num_key_value_heads = num_attention_heads

            self.num_key_value_heads = num_key_value_heads
            self.hidden_act = hidden_act
            self.initializer_range = initializer_range
            self.rms_norm_eps = rms_norm_eps
            self.rope_theta = rope_theta
            self.rope_scaling = rope_scaling
            self.attention_dropout = attention_dropout
            self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
            self.layer_types = layer_types
            if self.layer_types is None:
                self.layer_types = [
                    "sliding_attention" if bool((i + 1) % 2) else "full_attention" for i in range(self.num_hidden_layers)
                ]
            layer_type_validation(self.layer_types)
            self.attention_bias = True
            self.max_position_embeddings = max_position_embeddings
            self.router_aux_loss_coef = router_aux_loss_coef
            self.output_router_logits = output_router_logits
            self.use_cache = use_cache

            # Validate the correctness of rotary position embeddings parameters
            # BC: if there is a 'type' field, copy it it to 'rope_type'.
            if self.rope_scaling is not None and "type" in self.rope_scaling:
                self.rope_scaling["rope_type"] = self.rope_scaling["type"]
            rope_config_validation(self)

            self.attention_bias = True
            self.max_position_embeddings = max_position_embeddings
            self.router_aux_loss_coef = router_aux_loss_coef
            self.output_router_logits = output_router_logits
            self.use_cache = use_cache
            super().__init__(
                tie_word_embeddings=tie_word_embeddings,
                **kwargs,
            )

    def patch_gpt_oss_config():
        try:
            import transformers.models.gpt_oss.configuration_gpt_oss
            transformers.models.gpt_oss.configuration_gpt_oss.GptOssConfig
        except Exception as e:
            return raise_error("transformers.models.gpt_oss.configuration_gpt_oss", e)

        try:
            current_class = dedent(inspect.getsource(transformers.models.gpt_oss.configuration_gpt_oss.GptOssConfig))
            new_class = dedent(inspect.getsource(Old_GptOssConfig))
            new_class = new_class.replace("Old_GptOssConfig", "GptOssConfig")
            if new_class == current_class:
                logger.info("Unsloth: Updating GPT OSS Config to fix missing `max_position_embeddings`")
                patch_function(transformers.models.gpt_oss.configuration_gpt_oss, "GptOssConfig", GptOssConfig)
        except Exception as e:
            return raise_error("transformers.models.gpt_oss.configuration_gpt_oss", e)
    pass
    TEMPORARY_PATCHES.append(patch_gpt_oss_config)
except Exception as e:
    raise_error("transformers.models.gpt_oss.configuration_gpt_oss.GptOssConfig", e)
