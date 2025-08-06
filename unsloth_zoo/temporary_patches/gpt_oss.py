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
)

def patch_GptOssExperts_MXFP4():
    if "gpt-oss" not in os.environ.get("UNSLOTH_MODEL_NAME", ""):
        return
    if "-bnb-4bit" in os.environ.get("UNSLOTH_MODEL_NAME", ""):
        # We do NOT use MXFP4 to BF16 upcast
        return
    try:
        import transformers.models.gpt_oss.modeling_gpt_oss
        transformers.models.gpt_oss.modeling_gpt_oss.GptOssExperts
    except Exception as e:
        return raise_error("transformers.models.gpt_oss.modeling_gpt_oss.GptOssExperts", e)
    try:
        import transformers.integrations.mxfp4
        transformers.integrations.mxfp4.dequantize
    except Exception as e:
        return raise_error("transformers.integrations.mxfp4.dequantize", e)

    def convert_moe_packed_tensors(
        blocks,
        scales,
        dtype: torch.dtype = torch.bfloat16,
        rows_per_chunk: int = 32768 * 1024,
        target_device = "cuda",
    ) -> torch.Tensor:
        import math

        assert blocks.shape[:-1] == scales.shape, f"{blocks.shape=} does not match {scales.shape=}"

        # Check if blocks and scales are on CPU, and move to GPU if so
        if not blocks.is_cuda and torch.cuda.is_available():
            blocks = blocks.to("cuda", non_blocking = True)
            scales = scales.cuda()

        scales = scales.to(torch.int32) - 127

        FP4_VALUES = [
            +0.0,
            +0.5,
            +1.0,
            +1.5,
            +2.0,
            +3.0,
            +4.0,
            +6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ]
        lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

        *prefix_shape, G, B = blocks.shape
        rows_total = math.prod(prefix_shape) * G

        blocks = blocks.reshape(rows_total, B)
        scales = scales.reshape(rows_total, 1)

        out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

        for r0 in range(0, rows_total, rows_per_chunk):
            r1 = min(r0 + rows_per_chunk, rows_total)

            blk = blocks[r0:r1]
            exp = scales[r0:r1]

            # nibble indices -> int64
            idx_lo = (blk & 0x0F).to(torch.long)
            idx_hi = (blk >> 4).to(torch.long)

            sub = out[r0:r1]
            sub[:, 0::2] = lut[idx_lo]
            sub[:, 1::2] = lut[idx_hi]

            torch.ldexp(sub, exp, out=sub)
            del idx_lo, idx_hi, blk, exp, sub

        out = out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)

        # TODO: Delete after making sure this is not necessary! since we go back to cpu in the end in create_quantized_param using .to(target_device)
        # Move back to CPU if needed
        # if need_to_move_back:
        #     out = out.cpu()
        del blocks, scales, lut
        out = out.transpose(1, 2).contiguous().to(target_device)
        return out
    pass


    def delay_dequantize(module, param_name, param_value, target_device, dq_param_name, **kwargs):
        # print("Delaying dequantizing")
        # print(param_name, param_value.device, param_value.shape, target_device, dq_param_name)
        try:
            from transformers.integrations.tensor_parallel import shard_and_distribute_module
        except:
            shard_and_distribute_module = lambda *args, **kwargs: ""

        model = kwargs.get("model", None)
        empty_param = kwargs.get("empty_param", None)
        casting_dtype = kwargs.get("casting_dtype", None)
        to_contiguous = kwargs.get("to_contiguous", None)
        rank = kwargs.get("rank", None)
        device_mesh = kwargs.get("device_mesh", None)

        for proj in ["gate_up_proj", "down_proj"]:
            if proj in param_name:
                if device_mesh is not None:
                    param_value = shard_and_distribute_module(
                        model,
                        param_value,
                        empty_param,
                        dq_param_name,
                        casting_dtype,
                        to_contiguous,
                        rank,
                        device_mesh,
                        set_param=False,
                    )
                blocks_attr = f"{proj}_blocks"
                scales_attr = f"{proj}_scales"
                setattr(module, param_name.rsplit(".", 1)[1], param_value)
                if hasattr(module, blocks_attr) and hasattr(module, scales_attr):
                    # dequantized = convert_moe_packed_tensors(getattr(module, blocks_attr), getattr(module, scales_attr))
                    # dequantized = dequantized.transpose(1, 2).contiguous().to(target_device)
                    # dequantized = convert_moe_packed_tensors(getattr(module, blocks_attr), getattr(module, scales_attr), target_device = target_device)
                    # TODO: this is perhaps necessary since if target_device is cpu, and the param was on gpu
                    if target_device == "cpu" and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # setattr(module, proj, torch.nn.Parameter(dequantized))
                    setattr(module, proj, torch.nn.Parameter(torch.zeros(0), requires_grad = False))
                    # delattr(module, blocks_attr)
                    # delattr(module, scales_attr)
                    block = getattr(module, blocks_attr)
                    block.pin_memory()
                    setattr(module, blocks_attr, torch.nn.Parameter(block, requires_grad = False))
                    scale = getattr(module, scales_attr)
                    scale = scale.to("cuda", non_blocking = True)
                    setattr(module, scales_attr, torch.nn.Parameter(scale, requires_grad = False))
    pass
    patch_function(transformers.integrations.mxfp4, "dequantize", delay_dequantize, fullgraph = False)


    # transformers/models/gpt_oss/modeling_gpt_oss.py GptOssExperts
    # GptOssExperts.forward
    def GptOssExperts_forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
        """
        When training is is more efficient to just loop over the experts and compute the output for each expert
        as otherwise the memory would explode.

        For inference we can sacrifice some memory and compute the output for all experts at once. By repeating the inputs.

        Args:
            hidden_states (torch.Tensor): (batch_size, seq_len, hidden_size)
            selected_experts (torch.Tensor): (batch_size * token_num, top_k)
            routing_weights (torch.Tensor): (batch_size * token_num, num_experts)
        Returns:
            torch.Tensor
        """
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
        num_experts = routing_weights.shape[1]

        if hasattr(self, "down_proj_blocks"):
            down_proj_blocks = self.down_proj_blocks.to(hidden_states.device, non_blocking = True)
        if hasattr(self, "gate_up_proj_blocks"):
            gate_up_proj = convert_moe_packed_tensors(self.gate_up_proj_blocks, self.gate_up_proj_scales, dtype = hidden_states.dtype, target_device = hidden_states.device)
        else:
            gate_up_proj = self.gate_up_proj
        if self.training:
            next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                # we sum on the top_k and on the sequence lenght to get which experts
                # are hit this time around
                expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in expert_hitted[:]:
                with torch.no_grad():
                    _, token_idx = torch.where(expert_mask[expert_idx[0]])
                current_state = hidden_states[token_idx]
                gate_up = current_state @ gate_up_proj[expert_idx] + self.gate_up_proj_bias[expert_idx]
                gate, up = gate_up[..., ::2], gate_up[..., 1::2]
                gate = gate.clamp(min=None, max=self.limit)
                up = up.clamp(min=-self.limit, max=self.limit)
                glu = gate * torch.sigmoid(gate * self.alpha)
                gated_output = (up + 1) * glu

                if hasattr(self, "down_proj_blocks"):
                    down_proj = convert_moe_packed_tensors(self.down_proj_blocks, self.down_proj_scales, dtype = hidden_states.dtype, target_device = hidden_states.device)
                else:
                    down_proj = self.down_proj
                out = gated_output @ down_proj[expert_idx] + self.down_proj_bias[expert_idx]
                del down_proj
                weighted_output = out[0] * routing_weights[token_idx, expert_idx, None]
                next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
            next_states = next_states.view(batch_size, -1, self.hidden_size)
        else:
            hidden_states = hidden_states.repeat(num_experts, 1)
            hidden_states = hidden_states.view(num_experts, -1, self.hidden_size)
            gate_up = torch.bmm(hidden_states, gate_up_proj) + self.gate_up_proj_bias[..., None, :]
            del gate_up_proj

            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            glu = gate * torch.sigmoid(gate * self.alpha)

            if hasattr(self, "down_proj_blocks"):
                down_proj = convert_moe_packed_tensors(self.down_proj_blocks, self.down_proj_scales, dtype = hidden_states.dtype, target_device = hidden_states.device)
            else:
                down_proj = self.down_proj
            next_states = torch.bmm(((up + 1) * glu), down_proj)
            del down_proj

            next_states = next_states + self.down_proj_bias[..., None, :]
            next_states = next_states.view(num_experts, batch_size, -1, self.hidden_size)
            next_states = next_states * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
            next_states = next_states.sum(dim=0)
        return next_states
    pass
    patch_function(transformers.models.gpt_oss.modeling_gpt_oss.GptOssExperts, "forward", GptOssExperts_forward, fullgraph = True)
pass
TEMPORARY_PATCHES.append(patch_GptOssExperts_MXFP4)


def load_gpt_oss_MXFP4(
    model_name = "unsloth/gpt-oss-20b",
    torch_dtype = torch.float16,
):
    """ Loads GPT models with on the fly MXFP4->BF16 conversion"""
    assert "gpt-oss" in model_name
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name)
    num_hidden_layers = config.num_hidden_layers
    n_devices = torch.cuda.device_count()
    free_memory = [torch.cuda.mem_get_info(torch.cuda.device(i))[0] for i in range(n_devices)]
    free_memory = sum(free_memory) / 1024 / 1024 / 1024

    # Split balanced for now
    if free_memory <= 12:
        # Move down_proj to cpu
        down_proj_blocks_split = "cpu"
        # Offload all of gate_up_proj_blocks
        n_gate_up_proj_blocks_offload = num_hidden_layers
    elif free_memory <= 16:
        # Move down_proj to cpu
        down_proj_blocks_split = "cpu"
        # Offload 1/2 of gate_up_proj_blocks
        n_gate_up_proj_blocks_offload = int(num_hidden_layers * 0.7)
    else:
        down_proj_blocks_split = "x"
        n_gate_up_proj_blocks_offload = 0
    pass

    device_map = {}
    device_map[f"model.embed_tokens"] = 0
    device_map[f"model.rotary_emb"] = 0
    for i in range(num_hidden_layers):
        x = (i * n_devices) // num_hidden_layers
        device_map[f"model.layers.{i}.self_attn"] = x
        device_map[f"model.layers.{i}.input_layernorm"] = x
        device_map[f"model.layers.{i}.mlp.router"] = x
        device_map[f"model.layers.{i}.post_attention_layernorm"] = x
        device_map[f"model.layers.{i}.mlp.experts.gate_up_proj_scales"] = x
        device_map[f"model.layers.{i}.mlp.experts.down_proj_scales"] = x
        device_map[f"model.layers.{i}.mlp.experts.gate_up_proj_bias"] = x
        device_map[f"model.layers.{i}.mlp.experts.down_proj_bias"] = x
        device_map[f"model.layers.{i}.mlp.experts.gate_up_proj"] = x
        device_map[f"model.layers.{i}.mlp.experts.down_proj"] = x
        device_map[f"model.layers.{i}.mlp.experts.gate_up_proj_blocks"] = x

    # down_proj_blocks / gate_up_proj_blocks go to CPU if needed
    for i in range(num_hidden_layers):
        x = (i * n_devices) // num_hidden_layers

        device_map[f"model.layers.{i}.mlp.experts.down_proj_blocks"] = "cpu" if down_proj_blocks_split == "cpu" else x

        if i < n_gate_up_proj_blocks_offload:
            device_map[f"model.layers.{i}.mlp.experts.gate_up_proj_blocks"] = "cpu"
        else:
            device_map[f"model.layers.{i}.mlp.experts.gate_up_proj_blocks"] = x
    pass
    device_map[f"model.norm"] = n_devices-1
    device_map[f"lm_head"] = n_devices-1

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map = "cpu", # Use CPU to first make fake space
        torch_dtype = torch_dtype,
        use_kernels = False,
    )
    model = model.cuda()
    # Get downloaded model location - hack since SHA is used
    # so no downloading is done below!
    from huggingface_hub import snapshot_download
    checkpoint_location = snapshot_download(
        repo_id = model.config._name_or_path,
        revision = getattr(model.config, "revision", "main"),
        allow_patterns = ["model*safetensors"],
    )

    # Force move to CPU and not fake device
    preload_module_classes = set()
    for name, module in model.named_modules():
        if name in device_map and device_map[name] == "cpu":
            preload_module_classes.add(module.__class__.__name__)
    preload_module_classes = list(preload_module_classes)

    # Actually load the weights into CPU / CUDA
    from accelerate import load_checkpoint_and_dispatch
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint = checkpoint_location,
        device_map = device_map if n_devices <= 1 else "auto",
        preload_module_classes = preload_module_classes,
    )
    # Must bypass device_map check for training
    os.environ["ACCELERATE_BYPASS_DEVICE_MAP"] = "true"
    return model
pass

class GptOssExperts(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = config.intermediate_size
        self.alpha = 1.702
        self.limit = 7.0
        self.dtype = config.torch_dtype

        self.gate_up_projs = nn.ModuleList([
            nn.Linear(self.hidden_size, 2 * self.expert_dim, dtype=self.dtype)
            for _ in range(self.num_experts)
        ])
        self.down_projs = nn.ModuleList([
            nn.Linear(self.expert_dim, self.hidden_size, dtype=self.dtype)
            for _ in range(self.num_experts)
        ])

    def forward(self,
                hidden_states: torch.Tensor,
                router_indices = None,
                routing_weights = None
                ) -> torch.Tensor:

        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        num_experts = routing_weights.shape[1]

        if self.training:
            next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in expert_hitted[:]:
                with torch.no_grad():
                    _, token_idx = torch.where(expert_mask[expert_idx[0]])
                current_state = hidden_states[token_idx]
                gate_up = self.gate_up_projs[expert_idx](current_state)
                gate, up = gate_up[..., ::2], gate_up[..., 1::2]
                gate = gate.clamp(min=None, max=self.limit)
                up = up.clamp(min=-self.limit, max=self.limit)
                glu = gate * torch.sigmoid(gate * self.alpha)
                gated_output = (up + 1) * glu
                out = self.down_projs[expert_idx](gated_output)
                weighted_output = out * routing_weights[token_idx, expert_idx, None]
                next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
            next_states = next_states.view(batch_size, -1, self.hidden_size)
            return next_states

        else:
            X_rep = hidden_states.unsqueeze(0).expand(num_experts, -1, -1)
            gate_up_list = [up_l(X_rep[e]) for e, up_l in enumerate(self.gate_up_projs)]
            gate_up = torch.stack(gate_up_list, dim=0)
            gate = gate_up[..., ::2]
            up_h = gate_up[..., 1::2]
            gate = gate.clamp(max=self.limit)
            up_h = up_h.clamp(min=-self.limit, max=self.limit)
            glu = gate * torch.sigmoid(gate * self.alpha)
            fused = (up_h + 1) * glu
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

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = self.linear(hidden_states)  # (batch_size * seq_len, num_experts)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        return router_scores, router_indices
pass


def patch_GptOssExperts_bitsandbytes():
    if "gpt-oss" not in os.environ.get("UNSLOTH_MODEL_NAME", ""):
        return
    if "-bnb-4bit" not in os.environ.get("UNSLOTH_MODEL_NAME", ""):
        # We only use this for bnb-4bit models
        return
    try:
        import transformers.models.gpt_oss.modeling_gpt_oss
        transformers.models.gpt_oss.modeling_gpt_oss.GptOssExperts
    except Exception as e:
        return raise_error("transformers.models.gpt_oss.modeling_gpt_oss.GptOssExperts", e)

    try:
        import transformers.models.gpt_oss.modeling_gpt_oss
        transformers.models.gpt_oss.modeling_gpt_oss.GptOssTopKRouter
    except Exception as e:
        return raise_error("transformers.models.gpt_oss.modeling_gpt_oss.GptOssTopKRouter", e)
    try:
        import transformers.integrations.mxfp4
        transformers.integrations.mxfp4.dequantize
    except Exception as e:
        return raise_error("transformers.integrations.mxfp4.dequantize", e)



    transformers.models.gpt_oss.modeling_gpt_oss.GptOssExperts = GptOssExperts
    patch_function(transformers.models.gpt_oss.modeling_gpt_oss.GptOssExperts, "forward", GptOssExperts.forward, fullgraph = True)

    transformers.models.gpt_oss.modeling_gpt_oss.GptOssTopKRouter = GptOssTopKRouter
    patch_function(transformers.models.gpt_oss.modeling_gpt_oss.GptOssTopKRouter, "forward", GptOssTopKRouter.forward, fullgraph = True)
pass
TEMPORARY_PATCHES.append(patch_GptOssExperts_bitsandbytes)
