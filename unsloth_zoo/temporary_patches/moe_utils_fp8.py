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
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_TORCH_SCALED_GROUPED_MM_AVAILABLE = hasattr(torch, "_scaled_grouped_mm")
_TORCH_SCALED_GROUPED_MM_SUPPORTED = None


def _maybe_patch_glm4_stacked_moe_fp8_scales(
    model,
    model_name: str,
    token = None,
    revision = None,
):
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

    import safetensors.torch
    import json as _json

    # Collect all tensor keys we'll need so we can find the right shards
    needed_keys = set()
    for layer_idx, experts in routed_layers:
        num_experts = experts.gate_up_proj.shape[0]
        for expert_idx in range(num_experts):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                needed_keys.add(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj}.weight")
                needed_keys.add(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj}.weight_scale")

    # Resolve file path(s) -- single file or sharded
    shard_paths = {}  # tensor_key -> local file path
    if os.path.isdir(model_name):
        single_path = os.path.join(model_name, "model.safetensors")
        if os.path.exists(single_path):
            shard_paths = {k: single_path for k in needed_keys}
        else:
            index_path = os.path.join(model_name, "model.safetensors.index.json")
            if not os.path.exists(index_path):
                return False
            with open(index_path) as f:
                weight_map = _json.load(f).get("weight_map", {})
            for k in needed_keys:
                shard_file = weight_map.get(k)
                if shard_file is None:
                    return False
                shard_paths[k] = os.path.join(model_name, shard_file)
    else:
        from huggingface_hub import hf_hub_download

        try:
            single_path = hf_hub_download(
                repo_id = model_name,
                filename = "model.safetensors",
                token = token,
                revision = revision,
            )
            shard_paths = {k: single_path for k in needed_keys}
        except Exception:
            try:
                index_path = hf_hub_download(
                    repo_id = model_name,
                    filename = "model.safetensors.index.json",
                    token = token,
                    revision = revision,
                )
                with open(index_path) as f:
                    weight_map = _json.load(f).get("weight_map", {})
                # Download only the unique shards we need
                needed_shards = set()
                for k in needed_keys:
                    shard_file = weight_map.get(k)
                    if shard_file is None:
                        return False
                    needed_shards.add(shard_file)
                downloaded = {}
                for shard_file in needed_shards:
                    downloaded[shard_file] = hf_hub_download(
                        repo_id = model_name,
                        filename = shard_file,
                        token = token,
                        revision = revision,
                    )
                for k in needed_keys:
                    shard_paths[k] = downloaded[weight_map[k]]
            except Exception:
                return False

    if not shard_paths:
        return False

    # Open all unique shard files and build a multi-shard reader
    unique_paths = set(shard_paths.values())
    open_handles = {}
    try:
        for p in unique_paths:
            open_handles[p] = safetensors.torch.safe_open(p, framework = "pt")

        class _MultiShardFile:
            def get_tensor(self, key):
                return open_handles[shard_paths[key]].get_tensor(key)

        file = _MultiShardFile()
        _patch_result = _do_glm4_scale_patching(file, routed_layers)
    finally:
        open_handles.clear()

    return _patch_result


def _do_glm4_scale_patching(file, routed_layers):
    """Inner helper that reads tensors from file and patches routed_layers."""
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
        experts.gate_up_proj_scale = experts.gate_up_proj_weight_scale
        experts.down_proj_scale = experts.down_proj_weight_scale

    return True


def maybe_patch_stacked_moe_expert_fp8_scales(
    model,
    model_name: Optional[str] = None,
    token = None,
    revision = None,
):
    if model_name is None:
        config = getattr(model, "config", None)
        model_name = getattr(config, "_name_or_path", None)
    if not model_name:
        return False

    return _maybe_patch_glm4_stacked_moe_fp8_scales(
        model,
        model_name,
        token = token,
        revision = revision,
    )


def _is_float8_tensor(tensor: Optional[torch.Tensor]) -> bool:
    return tensor is not None and getattr(tensor, "dtype", None) == torch.float8_e4m3fn


def _get_fp8_dequant_target_dtype(hidden_states: torch.Tensor) -> torch.dtype:
    if hidden_states.dtype in (torch.float32, torch.float16, torch.bfloat16):
        return hidden_states.dtype
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _log_moe_fp8_backend_once(experts_module, message: str):
    from .common import logger
    from .moe_utils import _log_info

    if getattr(experts_module, "_unsloth_logged_fp8_backend", None) == message:
        return
    experts_module._unsloth_logged_fp8_backend = message
    logger.info(message)
    _log_info(message)


def _check_torch_scaled_grouped_mm_supported():
    global _TORCH_SCALED_GROUPED_MM_SUPPORTED
    if _TORCH_SCALED_GROUPED_MM_SUPPORTED is not None:
        return _TORCH_SCALED_GROUPED_MM_SUPPORTED

    if not _TORCH_SCALED_GROUPED_MM_AVAILABLE:
        _TORCH_SCALED_GROUPED_MM_SUPPORTED = False
        return False
    if not torch.cuda.is_available():
        _TORCH_SCALED_GROUPED_MM_SUPPORTED = False
        return False

    # The symbol can exist on unsupported GPUs, and the light probe can still
    # pass on SM100 while real MoE kernels later emit incompatible MMA
    # instructions and poison the CUDA context asynchronously. Restrict the
    # FP8 scaled_grouped_mm path to Hopper (SM 9.x) only for now.
    major, _minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    if major != 9:
        _TORCH_SCALED_GROUPED_MM_SUPPORTED = False
        return False

    try:
        device = torch.cuda.current_device()
        x = torch.randn((16, 16), device=device, dtype=torch.bfloat16)
        w_hp = torch.randn((1, 16, 16), device=device, dtype=torch.bfloat16)
        x_fp8, x_scale = _manual_fp8_rowwise_quantize(x)
        w_fp8, w_scale = _manual_fp8_rowwise_quantize(w_hp.view(-1, w_hp.shape[-1]))
        w_fp8 = w_fp8.view_as(w_hp)
        w_fp8 = _make_grouped_mm_rhs_column_major(w_fp8)
        w_scale = w_scale.view(w_hp.shape[0], w_hp.shape[1])
        offs = torch.tensor([16], device=device, dtype=torch.int32)
        torch._scaled_grouped_mm(
            x_fp8.contiguous(),
            w_fp8,
            x_scale.contiguous(),
            w_scale.contiguous(),
            offs=offs,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )
        _TORCH_SCALED_GROUPED_MM_SUPPORTED = True
        torch.cuda.synchronize(device)
    except Exception:
        _TORCH_SCALED_GROUPED_MM_SUPPORTED = False
    return _TORCH_SCALED_GROUPED_MM_SUPPORTED


def _slice_fp8_quant_state(weight: torch.Tensor, quant_state, expert_idx: int):
    from .moe_utils import _try_attach_block_size

    if quant_state is None or not isinstance(quant_state, torch.Tensor):
        return quant_state

    if quant_state.numel() == 1:
        sliced = quant_state
    elif quant_state.shape[0] == weight.shape[0]:
        sliced = quant_state[expert_idx]
    elif quant_state.shape[0] % weight.shape[0] == 0:
        chunk_size = quant_state.shape[0] // weight.shape[0]
        start = expert_idx * chunk_size
        end = start + chunk_size
        sliced = quant_state[start:end]
    else:
        return None

    block_size = getattr(weight, "block_size", None) or getattr(quant_state, "block_size", None)
    if block_size is not None:
        _try_attach_block_size(sliced, block_size)
    return sliced


def _ceil_div(a, b):
    return (a + b - 1) // b


def _dequantize_expert_slice(
    expert_weight: torch.Tensor,
    expert_quant_state,
    target_dtype: torch.dtype,
    quant_kind=None,
) -> Optional[torch.Tensor]:
    """Dequantize one expert's FP8 weight to target_dtype using pure PyTorch."""
    from .moe_utils import _try_attach_block_size

    if expert_weight.dtype != torch.float8_e4m3fn:
        return expert_weight.to(target_dtype)

    if expert_quant_state is None:
        return expert_weight.to(target_dtype)

    s = expert_quant_state
    if not isinstance(s, torch.Tensor):
        return expert_weight.to(target_dtype)

    if quant_kind == "weight_scale_inv":
        s = s.reciprocal()

    w = expert_weight.to(target_dtype)

    # Per-tensor scale
    if s.numel() == 1:
        return w * s.view(1, 1).to(target_dtype)

    # Reshape 1D to column vector for per-row handling
    if s.ndim == 1:
        s = s.view(-1, 1)

    # Per-row scale: (m, 1)
    if s.ndim == 2 and s.shape[1] == 1:
        # Per-sub-projection scalar scales (e.g. 2 scales for gate+up stacked weight).
        if (
            s.shape[0] > 1
            and s.shape[0] < w.shape[0]
            and w.shape[0] % s.shape[0] == 0
        ):
            repeat_factor = w.shape[0] // s.shape[0]
            s = s.repeat_interleave(repeat_factor, dim=0)

        if w.shape[0] == s.shape[0]:
            return w * s.to(target_dtype)
        elif w.shape[1] == s.shape[0]:
            return (w.t() * s.to(target_dtype)).t()
        return w * s.to(target_dtype)

    # Block scale: (ceil(m/bm), ceil(n/bn)) — expand to weight shape
    if s.ndim == 2:
        block_size = getattr(expert_weight, "block_size", None) or getattr(s, "block_size", None)
        M, N = w.shape
        p, q = s.shape

        if block_size is not None and len(block_size) == 2:
            bm, bn = block_size
            # Check if scale is transposed
            if _ceil_div(M, bm) != p or _ceil_div(N, bn) != q:
                if _ceil_div(M, bm) == q and _ceil_div(N, bn) == p:
                    s = s.T.contiguous()
                    p, q = s.shape
                else:
                    return expert_weight.to(target_dtype)
        else:
            # Infer block size from scale grid
            bm = _ceil_div(M, p)
            bn = _ceil_div(N, q)

        s_expanded = s.to(target_dtype).repeat_interleave(bm, dim=0)[:M].repeat_interleave(bn, dim=1)[:, :N]
        return w * s_expanded

    return expert_weight.to(target_dtype)


def _dequantize_full_expert_weights(weight: torch.Tensor, quant_state, target_dtype: torch.dtype, quant_kind=None):
    from .moe_utils import _try_attach_block_size

    if weight.ndim != 3:
        return None

    dequantized = []
    block_size = getattr(weight, "block_size", None)
    for expert_idx in range(weight.shape[0]):
        expert_weight = weight[expert_idx].contiguous()
        if block_size is not None:
            _try_attach_block_size(expert_weight, block_size)
        expert_quant_state = _slice_fp8_quant_state(weight, quant_state, expert_idx)
        expert_dequant = _dequantize_expert_slice(expert_weight, expert_quant_state, target_dtype, quant_kind=quant_kind)
        if expert_dequant is None:
            return None
        dequantized.append(expert_dequant)
    return torch.stack(dequantized, dim=0)


def _make_grouped_mm_rhs_column_major(weight: torch.Tensor) -> torch.Tensor:
    return weight.mT.contiguous()


def _get_moe_weight_and_quant_info(experts_module, param_name: str):
    from .moe_utils import _get_base_weight_and_quant_state, _try_attach_block_size

    param = getattr(experts_module, param_name)
    weight, quant_state = _get_base_weight_and_quant_state(param)
    quant_kind = "quant_state" if getattr(weight, "quant_state", None) is not None else None

    if quant_state is None:
        quant_state = getattr(experts_module, f"{param_name}_weight_scale_inv", None)
        if quant_state is not None:
            quant_kind = "weight_scale_inv"
        else:
            quant_state = getattr(experts_module, f"{param_name}_weight_scale", None)
            if quant_state is not None:
                quant_kind = "weight_scale"
        if quant_state is None:
            quant_state = getattr(experts_module, f"{param_name}_scale_inv", None)
            if quant_state is not None:
                quant_kind = "weight_scale_inv"
            else:
                quant_state = getattr(experts_module, f"{param_name}_scale", None)
                if quant_state is not None:
                    quant_kind = "weight_scale"

    block_size = getattr(param, "block_size", None)
    if block_size is None:
        block_size = getattr(experts_module, f"{param_name}_block_size", None)
    if block_size is None:
        # FP8Experts stores block_size on the module itself
        block_size = getattr(experts_module, "block_size", None)
    if block_size is not None:
        _try_attach_block_size(weight, block_size)
        if quant_state is not None:
            _try_attach_block_size(quant_state, block_size)
    return weight, quant_state, quant_kind


def _extract_scaled_grouped_mm_weight_scale(original_weight, processed_weight, quant_state, quant_kind):
    if quant_state is None or not isinstance(quant_state, torch.Tensor):
        return None
    if quant_kind == "quant_state":
        return None
    if getattr(original_weight, "block_size", None) is not None:
        return None
    if getattr(quant_state, "block_size", None) is not None:
        return None

    scale = quant_state
    if scale.ndim == 0:
        scale = scale.view(1, 1).expand(processed_weight.shape[0], processed_weight.shape[-1])
    elif scale.ndim == 1:
        if scale.shape[0] != processed_weight.shape[-1]:
            return None
        scale = scale.view(1, -1).expand(processed_weight.shape[0], -1)
    elif scale.ndim == 3:
        if scale.shape[1] == 1:
            scale = scale.squeeze(1)
        elif scale.shape[2] == 1:
            scale = scale.squeeze(2)
        else:
            return None
    elif scale.ndim != 2:
        return None

    if scale.ndim != 2:
        return None
    if scale.shape[0] != processed_weight.shape[0] or scale.shape[1] != processed_weight.shape[-1]:
        return None

    scale = scale.to(torch.float32)
    if quant_kind == "weight_scale_inv":
        scale = scale.reciprocal()
    return scale.contiguous()


def _prepare_scaled_grouped_mm_weight(experts_module, param_name: str, proj_type: str, hidden_dim: int, model_type=None):
    from .moe_utils import preprocess_weight

    weight, quant_state, quant_kind = _get_moe_weight_and_quant_info(experts_module, param_name)
    if not _is_float8_tensor(weight):
        return None
    processed_weight = preprocess_weight(weight, proj_type, hidden_dim, model_type)
    processed_weight = _make_grouped_mm_rhs_column_major(processed_weight)
    scale = _extract_scaled_grouped_mm_weight_scale(weight, processed_weight, quant_state, quant_kind)
    if scale is None:
        return None
    return processed_weight, scale


def _manual_fp8_rowwise_quantize(inputs: torch.Tensor):
    inputs_fp32 = inputs.to(torch.float32)
    max_fp8 = torch.finfo(torch.float8_e4m3fn).max
    amax = inputs_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    quant_scale = max_fp8 / amax
    quantized = (inputs_fp32 * quant_scale).to(torch.float8_e4m3fn)
    decode_scale = quant_scale.reciprocal().squeeze(-1).to(torch.float32)
    return quantized.contiguous(), decode_scale.contiguous()


def _forward_scaled_grouped_mm_fp8(self, hidden_states, top_k_index, top_k_weights):
    from .moe_utils import _get_grouped_lora, _apply_grouped_lora, _expand_grouped_bias

    if not _check_torch_scaled_grouped_mm_supported():
        return None
    if not hasattr(self, "gate_up_proj") or not hasattr(self, "down_proj"):
        return None

    is_2d_input = hidden_states.dim() == 2
    if is_2d_input:
        sequence_length, hidden_dim = hidden_states.shape
        batch_size = 1
    else:
        batch_size, sequence_length, hidden_dim = hidden_states.shape

    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.view(-1, hidden_dim)
    flat_top_k = top_k_index.view(-1)
    num_tokens_per_expert = torch.bincount(flat_top_k, minlength=self.num_experts).int()
    sorted_indices = torch.argsort(flat_top_k, stable=True)
    token_indices = sorted_indices // top_k_index.shape[-1]
    permuted_input = hidden_states[token_indices]
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    from .moe_utils import _should_use_separated_lora
    use_separated_lora = _should_use_separated_lora()
    model_type = getattr(self, "_unsloth_model_type", None)

    gate_up_prepared = _prepare_scaled_grouped_mm_weight(self, "gate_up_proj", "gate_up", hidden_dim, model_type)
    down_prepared = _prepare_scaled_grouped_mm_weight(self, "down_proj", "down", hidden_dim, model_type)
    if gate_up_prepared is None or down_prepared is None:
        return None
    gate_up_weight, gate_up_scale = gate_up_prepared
    down_weight, down_scale = down_prepared

    target_dtype = _get_fp8_dequant_target_dtype(permuted_input)
    permuted_input_fp8, input_scale = _manual_fp8_rowwise_quantize(permuted_input.to(target_dtype))
    mm1_out = torch._scaled_grouped_mm(
        permuted_input_fp8,
        gate_up_weight,
        input_scale,
        gate_up_scale,
        offs=offsets,
        out_dtype=target_dtype,
        use_fast_accum=True,
    )

    gate_up_lora = _get_grouped_lora(self, "gate_up_proj", "_unsloth_lora_gate_up_proj", use_separated_lora)
    if gate_up_lora is not None:
        mm1_out = mm1_out + _apply_grouped_lora(permuted_input, gate_up_lora, offsets, mm1_out.dtype)
    if hasattr(self, "gate_up_proj_bias") and self.gate_up_proj_bias is not None:
        bias_expanded = _expand_grouped_bias(self.gate_up_proj_bias, num_tokens_per_expert)
        mm1_out = mm1_out + bias_expanded.to(mm1_out.dtype)

    if "GptOssExperts" in self.__class__.__name__:
        gate = mm1_out[..., ::2]
        up = mm1_out[..., 1::2]
        limit = getattr(self, "limit", 7.0)
        alpha = getattr(self, "alpha", 1.702)
        gate = gate.clamp(min=None, max=limit)
        up = up.clamp(min=-limit, max=limit)
        inter = (up + 1.0) * (gate * torch.sigmoid(gate * alpha))
    else:
        gate, up = mm1_out.chunk(2, dim=-1)
        inter = F.silu(gate) * up

    inter_fp8, inter_scale = _manual_fp8_rowwise_quantize(inter)
    mm2_out = torch._scaled_grouped_mm(
        inter_fp8,
        down_weight,
        inter_scale,
        down_scale,
        offs=offsets,
        out_dtype=mm1_out.dtype,
        use_fast_accum=True,
    )

    down_lora = _get_grouped_lora(self, "down_proj", "_unsloth_lora_down_proj", use_separated_lora)
    if down_lora is not None:
        mm2_out = mm2_out + _apply_grouped_lora(inter, down_lora, offsets, inter.dtype)
    if hasattr(self, "down_proj_bias") and self.down_proj_bias is not None:
        bias_expanded = _expand_grouped_bias(self.down_proj_bias, num_tokens_per_expert).to(mm2_out.device)
        mm2_out = mm2_out + bias_expanded.to(mm2_out.dtype)

    flat_weights = top_k_weights.view(-1)
    permuted_weights = flat_weights[sorted_indices]
    mm2_out = mm2_out * permuted_weights.unsqueeze(-1)
    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=input_dtype,
        device=hidden_states.device,
    )
    final_hidden_states.index_add_(0, token_indices, mm2_out.to(input_dtype))
    if is_2d_input:
        return final_hidden_states
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


def _moe_uses_fp8_expert_weights(self) -> bool:
    from .moe_utils import _get_moe_weight_and_quant_state

    if not hasattr(self, "gate_up_proj") or not hasattr(self, "down_proj"):
        return False
    gate_param = getattr(self, "gate_up_proj", None)
    down_param = getattr(self, "down_proj", None)
    if _is_float8_tensor(gate_param) or _is_float8_tensor(down_param):
        return True
    gate_weight, _ = _get_moe_weight_and_quant_state(self, "gate_up_proj")
    down_weight, _ = _get_moe_weight_and_quant_state(self, "down_proj")
    return _is_float8_tensor(gate_weight) or _is_float8_tensor(down_weight)


def _call_with_temporary_moe_weights(experts_module, gate_up_proj, down_proj, forward_fn, *args):
    old_gate_up = getattr(experts_module, "gate_up_proj")
    old_down = getattr(experts_module, "down_proj")
    gate_up_param = nn.Parameter(gate_up_proj, requires_grad=old_gate_up.requires_grad)
    down_param = nn.Parameter(down_proj, requires_grad=old_down.requires_grad)
    setattr(experts_module, "gate_up_proj", gate_up_param)
    setattr(experts_module, "down_proj", down_param)
    try:
        return forward_fn(experts_module, *args)
    finally:
        setattr(experts_module, "gate_up_proj", old_gate_up)
        setattr(experts_module, "down_proj", old_down)


def _slice_fp8_linear_quant_state(experts_module, param_name: str, expert_idx: int):
    weight, quant_state, quant_kind = _get_moe_weight_and_quant_info(experts_module, param_name)
    expert_quant_state = _slice_fp8_quant_state(weight, quant_state, expert_idx)
    if not isinstance(expert_quant_state, torch.Tensor):
        return expert_quant_state
    if quant_kind == "weight_scale_inv":
        expert_quant_state = expert_quant_state.reciprocal()
    if expert_quant_state.ndim == 1:
        expert_quant_state = expert_quant_state.view(-1, 1)
    return expert_quant_state


def _forward_native_fp8_expert_loop(self, hidden_states, top_k_index, top_k_weights):
    try:
        from unsloth.kernels.fp8 import fp8_linear
    except ImportError:
        fp8_linear = None

    original_shape = hidden_states.shape
    hidden_dim = hidden_states.shape[-1]
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.view(-1, hidden_dim)
    top_k_index = top_k_index.view(-1, top_k_index.shape[-1])
    top_k_weights = top_k_weights.view(-1, top_k_weights.shape[-1])
    final_hidden_states = torch.zeros_like(hidden_states)

    gate_up_weight, _, _ = _get_moe_weight_and_quant_info(self, "gate_up_proj")
    down_weight, _, _ = _get_moe_weight_and_quant_info(self, "down_proj")

    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    gate_up_bias = getattr(self, "gate_up_proj_bias", None)
    down_bias = getattr(self, "down_proj_bias", None)

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.num_experts:
            continue

        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]

        expert_gate_up = gate_up_weight[expert_idx]
        gate_up_qstate = _slice_fp8_linear_quant_state(self, "gate_up_proj", expert_idx)
        gate_up_bias_expert = None if gate_up_bias is None else gate_up_bias[expert_idx]
        if _is_float8_tensor(expert_gate_up) and fp8_linear is not None:
            gate_up_out = fp8_linear(current_state, expert_gate_up, gate_up_qstate, gate_up_bias_expert)
        elif _is_float8_tensor(expert_gate_up):
            target_dtype = _get_fp8_dequant_target_dtype(current_state)
            expert_dequant = _dequantize_expert_slice(expert_gate_up, gate_up_qstate, target_dtype)
            gate_up_out = F.linear(current_state, expert_dequant, gate_up_bias_expert)
        else:
            gate_up_out = F.linear(current_state, expert_gate_up, gate_up_bias_expert)

        if "GptOssExperts" in self.__class__.__name__:
            gate = gate_up_out[..., ::2]
            up = gate_up_out[..., 1::2]
            limit = getattr(self, "limit", 7.0)
            alpha = getattr(self, "alpha", 1.702)
            gate = gate.clamp(min=None, max=limit)
            up = up.clamp(min=-limit, max=limit)
            current_hidden_states = (up + 1.0) * (gate * torch.sigmoid(gate * alpha))
        else:
            gate, up = gate_up_out.chunk(2, dim=-1)
            act_fn = getattr(self, "act_fn", None)
            if act_fn is None:
                act_fn = F.silu
            current_hidden_states = act_fn(gate) * up

        expert_down = down_weight[expert_idx]
        down_qstate = _slice_fp8_linear_quant_state(self, "down_proj", expert_idx)
        down_bias_expert = None if down_bias is None else down_bias[expert_idx]
        if _is_float8_tensor(expert_down) and fp8_linear is not None:
            current_hidden_states = fp8_linear(
                current_hidden_states,
                expert_down,
                down_qstate,
                down_bias_expert,
            )
        elif _is_float8_tensor(expert_down):
            target_dtype = _get_fp8_dequant_target_dtype(current_hidden_states)
            expert_dequant = _dequantize_expert_slice(expert_down, down_qstate, target_dtype)
            current_hidden_states = F.linear(current_hidden_states, expert_dequant, down_bias_expert)
        else:
            current_hidden_states = F.linear(current_hidden_states, expert_down, down_bias_expert)

        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(input_dtype))

    return final_hidden_states.view(original_shape)


@torch.compiler.disable
def forward_moe_backend_fp8(self, hidden_states, top_k_index, top_k_weights):
    from .moe_utils import (
        select_moe_backend,
        forward_native_grouped_mm,
        forward_triton_grouped_gemm,
        forward_native_moe_loop,
    )

    backend = select_moe_backend()

    # 1. Try _scaled_grouped_mm (fast FP8 path on Hopper/Blackwell)
    if backend == "grouped_mm" and _check_torch_scaled_grouped_mm_supported():
        scaled_grouped_mm_output = _forward_scaled_grouped_mm_fp8(
            self,
            hidden_states,
            top_k_index,
            top_k_weights,
        )
        if scaled_grouped_mm_output is not None:
            _log_moe_fp8_backend_once(
                self,
                "Unsloth: MoE FP8 is using _scaled_grouped_mm.",
            )
            return scaled_grouped_mm_output

    # 2. Dequant FP8 weights to bf16/fp16 and run through normal MoE forward
    target_dtype = _get_fp8_dequant_target_dtype(hidden_states)
    gate_up_base, gate_up_quant, gate_up_qkind = _get_moe_weight_and_quant_info(self, "gate_up_proj")
    down_base, down_quant, down_qkind = _get_moe_weight_and_quant_info(self, "down_proj")
    gate_up_weight = _dequantize_full_expert_weights(gate_up_base, gate_up_quant, target_dtype, quant_kind=gate_up_qkind)
    down_weight = _dequantize_full_expert_weights(down_base, down_quant, target_dtype, quant_kind=down_qkind)

    if gate_up_weight is not None and down_weight is not None:
        if backend == "grouped_mm":
            _log_moe_fp8_backend_once(self, "Unsloth: MoE FP8 is using dequantize-plus-grouped_mm.")
            forward_fn = forward_native_grouped_mm
        elif backend == "unsloth_triton":
            _log_moe_fp8_backend_once(self, "Unsloth: MoE FP8 is using dequantize-plus-Triton grouped GEMM.")
            forward_fn = forward_triton_grouped_gemm
        else:
            _log_moe_fp8_backend_once(self, "Unsloth: MoE FP8 is using dequantize-plus-native_torch loop.")
            forward_fn = forward_native_moe_loop

        return _call_with_temporary_moe_weights(
            self,
            gate_up_weight,
            down_weight,
            forward_fn,
            hidden_states.to(target_dtype),
            top_k_index,
            top_k_weights,
        )

    # 3. Last resort: per-expert fp8_linear loop
    _log_moe_fp8_backend_once(
        self,
        "Unsloth: MoE FP8 is using per-expert fp8_linear loop.",
    )
    return _forward_native_fp8_expert_loop(self, hidden_states, top_k_index, top_k_weights)
