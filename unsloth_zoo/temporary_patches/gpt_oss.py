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
from .common import TEMPORARY_PATCHES, torch_compile
from .utils import (
    patch_function,
    KWARGS_TYPE,
    raise_error,
    logger,
)
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


class GptOssExperts(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = config.intermediate_size
        self.alpha = 1.702
        self.limit = getattr(config, "swiglu_limit", 7.0)
        self.dtype = config.torch_dtype

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
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in expert_hitted[:]:
                with torch.no_grad():
                    _, token_idx = torch.where(expert_mask[expert_idx[0]])
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
            return next_states
        else:
            X_rep = hidden_states.unsqueeze(0).expand(num_experts, -1, -1)
            gate_up_list = [up_l(X_rep[e]) for e, up_l in enumerate(self.gate_up_projs)]
            gate_up = torch.stack(gate_up_list, dim=0)
            fused = swiglu_torch_forward(gate_up, self.alpha, self.limit)
            # gate = gate_up[..., ::2]
            # up_h = gate_up[..., 1::2]
            # gate = gate.clamp(max=self.limit)
            # up_h = up_h.clamp(min=-self.limit, max=self.limit)
            # glu = gate * torch.sigmoid(gate * self.alpha)
            # fused = (up_h + 1) * glu
            out_list = [down_l(fused[e]) for e, down_l in enumerate(self.down_projs)]
            outs = torch.stack(out_list, dim=0)
            rw = routing_weights.transpose(0, 1).unsqueeze(-1)
            mixed = (outs * rw).sum(dim=0)
            return mixed.view(batch_size, -1, self.hidden_size)
pass

class GptOssTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.linear = nn.Linear(self.hidden_dim, self.num_experts, dtype=config.torch_dtype)

    @torch_compile(dynamic = True, fullgraph = True)
    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = self.linear(hidden_states.to(self.linear.weight.dtype))  # (batch_size * seq_len, num_experts)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
        dtype = torch.float32 if router_logits.dtype == torch.float16 else router_logits.dtype
        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=torch.float32).to(dtype)
        router_scores = torch.zeros_like(router_logits, dtype = dtype).scatter_(1, router_indices, router_top_value)
        return router_scores, router_indices
pass

class GptOssMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = GptOssTopKRouter(config)
        self.experts = GptOssExperts(config)

    def forward(self, hidden_states):
        router_scores, router_indices = self.router(hidden_states)  # (num_experts, seq_len)
        routed_out = self.experts(hidden_states, router_indices=router_indices, routing_weights=router_scores)
        return routed_out, router_scores
pass

def patch_gpt_oss_linearized():
    model_name = os.environ.get("UNSLOTH_MODEL_NAME", "")
    if not model_name.endswith("-bnb-4bit"): return
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
                with torch.no_grad():
                    expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=num_experts)
                    expert_mask = expert_mask.permute(2, 1, 0)
                    expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
                has_float32 = False
                for expert_idx in expert_hitted[:]:
                    with torch.no_grad():
                        _, token_idx = torch.where(expert_mask[expert_idx[0]])
                    current_state = hidden_states[token_idx]
                    gate_up = self.gate_up_projs[expert_idx](current_state)
                    down_proj = self.down_projs[expert_idx]
                    dtype = getattr(down_proj, "_pre_set_compute_dtype", torch.float16)
                    if dtype == torch.float32: has_float32 = True
                    gated_output = swiglu_torch_forward(gate_up, self.alpha, self.limit, dtype = dtype)
                    # gate, up = gate_up[..., ::2], gate_up[..., 1::2]
                    # gate = gate.clamp(min=None, max=self.limit)
                    # up = up.clamp(min=-self.limit, max=self.limit)
                    # glu = gate * torch.sigmoid(gate * self.alpha)
                    # gated_output = (up + 1) * glu

                    # Force float32 matrix multiply on some down projection modules
                    gated_output = gated_output.to(dtype)
                    device_type = gated_output.device.type if isinstance(gated_output.device.type, str) and gated_output.device.type != "mps" else "cpu"
                    with torch.autocast(device_type=device_type, enabled=False): # Force float32
                        out = down_proj(gated_output)
                    weighted_output = out.to(torch.float32) * routing_weights[token_idx, expert_idx, None].to(torch.float32)
                    next_states.index_add_(0, token_idx, weighted_output)
                next_states = next_states.view(batch_size, -1, self.hidden_size)
                return next_states.to(torch.float32 if has_float32 else torch.float16)
            else:
                X_rep = hidden_states.unsqueeze(0).expand(num_experts, -1, -1)
                gate_up_list = [up_l(X_rep[e]) for e, up_l in enumerate(self.gate_up_projs)]
                gate_up = torch.stack(gate_up_list, dim=0)
                fused = swiglu_torch_forward(gate_up, self.alpha, self.limit)
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
                        down_l(fused[e].to(
                            getattr(down_l, "_pre_set_compute_dtype", torch.float16)
                        ))
                        for e, down_l in enumerate(self.down_projs)
                    ]
                outs = torch.stack(out_list, dim=0)
                rw = routing_weights.transpose(0, 1).unsqueeze(-1)
                mixed = (outs.to(torch.float32) * rw.to(torch.float32)).sum(dim=0)
                return mixed.view(batch_size, -1, self.hidden_size).to(outs.dtype)
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
