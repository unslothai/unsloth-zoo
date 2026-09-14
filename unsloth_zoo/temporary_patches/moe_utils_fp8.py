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

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_TORCH_SCALED_GROUPED_MM_AVAILABLE = hasattr(torch, "_scaled_grouped_mm")
_TORCH_SCALED_GROUPED_MM_SUPPORTED = None


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
            if _ceil_div(M, bm) != p or _ceil_div(N, bn) != q:  # scale transposed?
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


def _dequantize_full_expert_weights_vectorized(weight: torch.Tensor, quant_state, target_dtype: torch.dtype,) -> Optional[torch.Tensor]:
    """Dequantize all experts in one batched op."""
    if weight.ndim != 3 or weight.dtype != torch.float8_e4m3fn:
        return None
    if quant_state is None or not isinstance(quant_state, torch.Tensor):
        return weight.to(target_dtype)

    E, M, N = weight.shape
    s = quant_state

    w = weight.to(target_dtype)

    # Per-tensor scalar
    if s.numel() == 1:
        return w * s.to(target_dtype).view(1, 1, 1)

    # 3D block scales: (E, ceil(M/bm), ceil(N/bn))
    if s.ndim == 3 and s.shape[0] == E:
        block_size = getattr(weight, "block_size", None) or getattr(s, "block_size", None)
        p, q = s.shape[1], s.shape[2]
        if block_size is not None and len(block_size) == 2:
            bm, bn = block_size
        else:
            bm = _ceil_div(M, p)
            bn = _ceil_div(N, q)
        # Expand block scales (E, p, q) -> (E, p*bm, q*bn), trim to (E, M, N)
        s_expanded = (
            s.to(target_dtype)
            .repeat_interleave(bm, dim=1)[:, :M, :]
            .repeat_interleave(bn, dim=2)[:, :, :N]
        )
        return w * s_expanded

    # 2D per-row scales: (E, M) or (E, 1)
    if s.ndim == 2 and s.shape[0] == E:
        if s.shape[1] == 1:
            return w * s.to(target_dtype).unsqueeze(-1)
        if s.shape[1] == M:
            return w * s.to(target_dtype).unsqueeze(-1)
        if s.shape[1] == N:
            return w * s.to(target_dtype).unsqueeze(1)

    # 1D: single per-row scale shared across experts
    if s.ndim == 1:
        if s.shape[0] == M:
            return w * s.to(target_dtype).view(1, -1, 1)
        if s.shape[0] == N:
            return w * s.to(target_dtype).view(1, 1, -1)

    return None


def _dequantize_full_expert_weights_unsloth(weight, scale, target_dtype):
    """
    Dequantize 3-D FP8 expert weights using unsloth.kernels.fp8.weight_dequant_block.

    Expects layout from compressed-tensors / finegrained_fp8:
      weight (E, M, N) float8_e4m3fn
      scale  (E, ceil(M/bm), ceil(N/bn)) bf16/fp32  -- "weight_scale_inv": dequant = q * s
    Returns (E, M, N) target_dtype, or None if shapes don't match the expected layout.
    """
    if weight.ndim != 3 or weight.dtype != torch.float8_e4m3fn:
        return None
    if not isinstance(scale, torch.Tensor) or scale.ndim != 3 or scale.shape[0] != weight.shape[0]:
        return None
    try:
        from unsloth.kernels.fp8 import weight_dequant_block
    except ImportError:
        return None

    E, M, N = weight.shape
    p, q = scale.shape[1], scale.shape[2]
    if p == 0 or q == 0:
        return None
    bm = _ceil_div(M, p)
    bn = _ceil_div(N, q)
    if bm != bn:
        # weight_dequant_block uses a single BLOCK_SIZE; fall through to caller.
        return None

    # Fast path: when M is exactly p*bm and both tensors are expert-major contiguous,
    # flatten (E, M, N) -> (E*M, N) and dequant in ONE kernel call. The 2-D dequant
    # kernel's row-major scale index r // bm equals e*p + (r%M)//bm for the flattened
    # layout, which matches scale.view(E*p, q). On B200 this is ~25x faster than the
    # per-expert loop because it removes 128 kernel launches.
    if M == p * bm and weight.is_contiguous() and scale.is_contiguous():
        w_flat = weight.view(E * M, N)
        s_flat = scale.view(E * p, q).to(torch.float32).contiguous()
        out_flat = weight_dequant_block(w_flat, s_flat, block_size=bm, dtype=target_dtype)
        return out_flat.view(E, M, N)

    out = torch.empty((E, M, N), dtype=target_dtype, device=weight.device)
    for e in range(E):
        out[e] = weight_dequant_block(
            weight[e].contiguous(),
            scale[e].contiguous().to(torch.float32),
            block_size=bm,
            dtype=target_dtype,
        )
    return out


def _dequantize_full_expert_weights(weight: torch.Tensor, quant_state, target_dtype: torch.dtype, quant_kind=None):
    if weight.ndim != 3:
        return None

    block_size = getattr(weight, "block_size", None)
    if block_size is not None and quant_state is not None:
        _try_attach_block_size(quant_state, block_size)
        _try_attach_block_size(weight, block_size)

    # Preferred: validated Triton block dequant kernel from unsloth.
    result = _dequantize_full_expert_weights_unsloth(weight, quant_state, target_dtype)
    if result is not None:
        return result

    # Fallback 1: vectorized PyTorch dequant.
    result = _dequantize_full_expert_weights_vectorized(weight, quant_state, target_dtype)
    if result is not None:
        return result

    # Fallback 2: per-expert loop. Only reachable when scale is 2-D with an
    # explicit transposed block_size annotation; vectorized + unsloth Triton
    # paths above cover every standard 3-D scale layout.
    if (
        not isinstance(quant_state, torch.Tensor)
        or quant_state.ndim != 2
        or block_size is None
    ):
        return None
    dequantized = []
    for expert_idx in range(weight.shape[0]):
        expert_weight = weight[expert_idx].contiguous()
        _try_attach_block_size(expert_weight, block_size)
        expert_quant_state = _slice_fp8_quant_state(weight, quant_state, expert_idx)
        expert_dequant = _dequantize_expert_slice(expert_weight, expert_quant_state, target_dtype, quant_kind=quant_kind)
        if expert_dequant is None:
            return None
        dequantized.append(expert_dequant)
    return torch.stack(dequantized, dim=0)


def _make_grouped_mm_rhs_column_major(weight: torch.Tensor) -> torch.Tensor:
    return weight.mT.contiguous()


def _try_attach_block_size(tensor, block_size):
    """Attach block_size, ignoring failures (e.g. read-only Tensor subclasses)."""
    try:
        setattr(tensor, "block_size", block_size)
    except Exception:
        pass


def _get_moe_weight_and_quant_info(experts_module, param_name: str):
    """Resolve the underlying expert weight tensor and (if present) its FP8
    block-quant scale tensor, walking through PEFT ParamWrapper if needed."""
    # FP8-aware unwrap returns the raw FP8 tensor, not a bf16-merged ParamWrapper value.
    weight = _unwrap_param_attr(experts_module, param_name)
    if weight is None:
        weight = getattr(experts_module, param_name, None)
    param = getattr(experts_module, param_name, None)
    quant_state = getattr(weight, "quant_state", None) if weight is not None else None
    quant_kind = "quant_state" if quant_state is not None else None

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
    processed_weight = preprocess_weight(weight, proj_type, hidden_dim, model_type, experts_module=experts_module)
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


def _moe_separated_lora_delta(lora_data, permuted_input, offsets, out_dtype):
    """Compute (X @ first) @ second * scaling for separated MoE LoRA — mirrors
    the inline path in moe_utils.forward_native_grouped_mm. Returns the delta
    to ADD to the base grouped-mm output, or None when no LoRA is active."""
    if lora_data is None:
        return None
    first_weight, second_weight, scaling = lora_data[:3]
    first_weight = first_weight.to(permuted_input.dtype).contiguous()
    second_weight = second_weight.to(permuted_input.dtype).contiguous()
    lora_out = torch._grouped_mm(permuted_input, first_weight, offs=offsets).contiguous()
    # grouped_mm requires the trailing dim to be a multiple of 8.
    if second_weight.shape[-1] % 8 != 0:
        pad_size = 8 - (second_weight.shape[-1] % 8)
        padded = F.pad(second_weight, (0, pad_size)).contiguous()
        lora_delta = torch._grouped_mm(lora_out, padded, offs=offsets)[:, :-pad_size]
    else:
        lora_delta = torch._grouped_mm(lora_out, second_weight, offs=offsets)
    return (lora_delta * scaling).to(out_dtype)


def _expand_grouped_bias(bias, num_tokens_per_expert):
    """Expand per-expert bias (E, out_dim) to (total_tokens, out_dim) by
    repeating row i exactly num_tokens_per_expert[i] times — matches the
    permuted-input layout used by grouped_mm."""
    num_repeats = num_tokens_per_expert.to(bias.device)
    return bias.repeat_interleave(num_repeats, dim=0)


def _forward_scaled_grouped_mm_fp8(self, hidden_states, top_k_index, top_k_weights):
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

    gate_up_lora = getattr(self, "_unsloth_lora_gate_up_proj", None) if use_separated_lora else None
    gate_up_delta = _moe_separated_lora_delta(gate_up_lora, permuted_input, offsets, mm1_out.dtype)
    if gate_up_delta is not None:
        mm1_out = mm1_out + gate_up_delta
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

    down_lora = getattr(self, "_unsloth_lora_down_proj", None) if use_separated_lora else None
    down_delta = _moe_separated_lora_delta(down_lora, inter, offsets, mm2_out.dtype)
    if down_delta is not None:
        mm2_out = mm2_out + down_delta
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
    if not hasattr(self, "gate_up_proj") or not hasattr(self, "down_proj"):
        return False
    for name in ("gate_up_proj", "down_proj"):
        if _is_float8_tensor(_unwrap_param_attr(self, name)):
            return True
    return False


def _unwrap_param_attr(module, name):
    """Resolve the underlying tensor for an experts-module attribute that
    may have been wrapped by PEFT ParamWrapper (or a chain of them)."""
    obj = getattr(module, name, None)
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj
    # Call get_param() BEFORE walking base_layer: base_layer points back to the
    # experts module, not the tensor.
    while hasattr(obj, "get_param") and callable(obj.get_param):
        try:
            inner = obj.get_param()
        except Exception:
            break
        if isinstance(inner, torch.Tensor):
            return inner
        if inner is obj:
            break
        obj = inner
    # Walk a base_layer chain ending in something with the named attribute
    # (handles nested ParamWrappers).
    seen = set()
    while hasattr(obj, "base_layer") and id(obj) not in seen:
        seen.add(id(obj))
        base = obj.base_layer
        if hasattr(base, name) and isinstance(getattr(base, name), torch.Tensor):
            return getattr(base, name)
        obj = base
    return None


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
    # This is the last-resort FP8 path (no scaled_grouped_mm support AND
    # _dequantize_full_expert_weights returned None for either projection).
    # It computes the FP8 forward but does NOT consume the separated LoRA
    # attributes that `patch_param_wrapper_for_moe` injects, so reaching this
    # path while LoRA is active would silently train without the adapter.
    # Refuse rather than corrupt: the user can disable LoRA or fix the
    # missing kernel before retrying.
    if (
        getattr(self, "_unsloth_lora_gate_up_proj", None) is not None
        or getattr(self, "_unsloth_lora_down_proj", None) is not None
    ):
        raise RuntimeError(
            "Unsloth: MoE FP8 fell through to the per-expert fp8_linear "
            "fallback while LoRA adapters are attached. This path does not "
            "apply the separated LoRA delta and would silently train without "
            "it. Install unsloth.kernels.fp8 (or a torch with "
            "_scaled_grouped_mm support, or a working FP8 dequant) so a "
            "LoRA-aware backend can be selected."
        )
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
        swap_moe_weights_for_call,
    )

    backend = select_moe_backend()

    # 1. _scaled_grouped_mm (fast FP8 path on Hopper/Blackwell)
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

    # 2. Dequant FP8 weights to bf16/fp16, run through normal MoE forward
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

        return swap_moe_weights_for_call(
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


# ============================================================================
# Save-side hooks: dequant base FP8 weights before LoRA merge, requant after.
# saving_utils.py consults `_MOE_QUANT_HANDLERS` for every expert weight key
# it reads, so the FP8 plumbing stays local to this file instead of polluting
# the generic save path.
# ============================================================================

# FP8 e4m3 max representable magnitude — matches transformers' Fp8Quantize.convert
# (finegrained_fp8.py:838-859) and compressed-tensors' fp8 encode path.
_FP8_E4M3_MAX = 448.0

# Sentinel returned by a handler when it positively identifies its quant kind on
# the key but cannot safely apply a LoRA merge (e.g. missing companion scale).
# saving_utils.py treats this as "skip this expert, record a fallback".
_MOE_QUANT_UNSAFE = object()


def _fp8_dequant_blockwise(W_fp8: torch.Tensor, scale_inv: torch.Tensor, block_size = None) -> torch.Tensor:
    """Dequant a 2-D float8_e4m3fn weight: W_real = decode_fp8(W) * scale_broadcast.

    Inverse of compressed-tensors / DeepSeek FP8. Handles every dense scale layout
    (per-tensor, 1-D per-row/col, 2-D per-channel, 2-D block). block_size (bm, bn) is
    the configured weight_block_size; without it bm/bn are inferred from the scale grid,
    which is only exact when rows/cols are block multiples.
    """
    rows, cols = W_fp8.shape
    out_dtype = scale_inv.dtype if scale_inv.dtype.is_floating_point else torch.float32
    W = W_fp8.to(out_dtype)
    if scale_inv.numel() == 1:
        return W * scale_inv.reshape(()).to(out_dtype)
    if scale_inv.ndim == 1:
        if scale_inv.shape[0] == rows:
            return W * scale_inv.view(-1, 1).to(out_dtype)
        if scale_inv.shape[0] == cols:
            return W * scale_inv.view(1, -1).to(out_dtype)
        raise RuntimeError(
            f"Unsloth: FP8 1-D scale length {scale_inv.shape[0]} matches neither "
            f"rows ({rows}) nor cols ({cols}); cannot dequantize."
        )
    srows, scols = scale_inv.shape
    bm = bn = None
    # Use the configured block size only when the scale grid tiles the weight by it;
    # per-channel 2-D scales (e.g. (rows, 1)) fall back to inference.
    if block_size is not None and len(block_size) == 2:
        cand_bm, cand_bn = block_size
        if srows == -(-rows // cand_bm) and scols == -(-cols // cand_bn):
            bm, bn = cand_bm, cand_bn
    if bm is None:
        bm = -(-rows // srows)  # ceil(rows / srows)
        bn = -(-cols // scols)  # ceil(cols / scols)
    scale = scale_inv.to(out_dtype)
    if bm > 1:
        scale = scale.repeat_interleave(bm, dim = 0)[:rows]
    if bn > 1:
        scale = scale.repeat_interleave(bn, dim = 1)[:, :cols]
    return W * scale


def _fp8_requant_blockwise(W: torch.Tensor, block_shape: tuple, scale_dtype: torch.dtype):
    """Block-wise FP8 e4m3 re-quantization. Returns (W_fp8, new_scale_inv).

    Per (bm, bn) block: scale_inv = max_abs / 448, W_fp8 = clamp(W / scale_inv, ±448).
    Zero-blocks clamp scale_inv to 1e-12 (avoids div-by-zero; the encoded fp8
    is 0 either way, so on reload `0 * 1e-12 == 0` round-trips cleanly).
    A partial final block (dim not a multiple of bm/bn) is zero-padded to the block
    grid and truncated back, so the scale grid matches the original ceil-tiled layout.
    """
    rows, cols = W.shape
    bm, bn = block_shape
    srows, scols = -(-rows // bm), -(-cols // bn)  # ceil
    Wf = W.to(torch.float32)
    if srows * bm != rows or scols * bn != cols:
        Wpad = Wf.new_zeros(srows * bm, scols * bn)
        Wpad[:rows, :cols] = Wf
        Wf = Wpad
    W_blocks = Wf.reshape(srows, bm, scols, bn)
    block_max = W_blocks.abs().amax(dim=(1, 3))
    scale_inv = (block_max / _FP8_E4M3_MAX).clamp_min(1e-12)
    W_scaled = (
        W_blocks / scale_inv.unsqueeze(-1).unsqueeze(1)
    ).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    W_fp8 = W_scaled.reshape(srows * bm, scols * bn)[:rows, :cols].to(torch.float8_e4m3fn)
    return W_fp8, scale_inv.to(scale_dtype)


def _fp8_save_handler(file, header_metadata, weight_key, block_size = None):
    """MoE-quant save handler for FP8 base weights.

    Returns:
      - None if `weight_key` is not FP8 (so other handlers / the plain
        `file.get_tensor(key)` path takes over).
      - `(_MOE_QUANT_UNSAFE, None)` if FP8 is detected but the companion
        `weight_scale_inv` / `weight_scale` cannot be located or its block
        layout does not divide the weight cleanly.
      - `(W_bf16, requant_fn)` otherwise, where `requant_fn(merged_W)` returns
        `(W_fp8, "extra_writes": [(scale_key, scale_tensor, scale_dtype)],
         storage_dtype)` so the caller can write the re-quantised data plus
        the new scale alongside it.
    """
    dtype_str = header_metadata.get(weight_key, {}).get("dtype")
    _e5m2 = getattr(torch, "float8_e5m2", None)
    # e5m2 is FP8 but this requant path only encodes e4m3; refuse (UNSAFE) so the merge is
    # skipped and logged rather than writing raw FP8 back with a stale scale.
    if dtype_str == "F8_E5M2":
        return _MOE_QUANT_UNSAFE, None
    # Cheap probe: skip non-FP8 keys without paying for a tensor read.
    if dtype_str not in ("F8_E4M3",):
        # Only FP8 keys are our concern; let the generic loader handle the rest.
        if dtype_str is None:
            # Unknown layout — fall back to full read to learn the dtype.
            W = file.get_tensor(weight_key)
            if _e5m2 is not None and W.dtype == _e5m2:
                return _MOE_QUANT_UNSAFE, None
            if W.dtype != torch.float8_e4m3fn:
                return None
        else:
            return None
    else:
        W = file.get_tensor(weight_key)
        if W.dtype != torch.float8_e4m3fn:
            return None

    base = weight_key[: -len(".weight")] if weight_key.endswith(".weight") else weight_key
    candidate_keys = (base + ".weight_scale_inv", base + ".weight_scale")
    scale_key = next((k for k in candidate_keys if k in header_metadata), None)
    if scale_key is None:
        return _MOE_QUANT_UNSAFE, None
    scale_inv = file.get_tensor(scale_key)
    if scale_inv.ndim != 2 or W.ndim != 2:
        return _MOE_QUANT_UNSAFE, None
    rows, cols = W.shape
    srows, scols = scale_inv.shape
    # Prefer the configured block size when its grid tiles the weight (handles partial
    # final blocks, e.g. 130x256 with 128x128 blocks). Otherwise fall back to exact-division
    # inference, which also covers per-channel scales like (rows, 1) that a global block size
    # does not tile.
    block_shape = None
    if block_size is not None and len(block_size) == 2:
        cand_bm, cand_bn = block_size
        if srows == -(-rows // cand_bm) and scols == -(-cols // cand_bn):
            block_shape = (cand_bm, cand_bn)
    if block_shape is None:
        if rows % srows != 0 or cols % scols != 0:
            return _MOE_QUANT_UNSAFE, None
        block_shape = (rows // srows, cols // scols)
    scale_dtype = scale_inv.dtype
    W_bf16 = _fp8_dequant_blockwise(W, scale_inv, block_size = block_shape)

    def _requant(merged_W):
        W_fp8, new_scale = _fp8_requant_blockwise(merged_W, block_shape, scale_dtype)
        return W_fp8, torch.float8_e4m3fn, [(scale_key, new_scale, scale_dtype)]

    return W_bf16, _requant


# Ordered list of save-side handlers. First non-None return wins.
_MOE_QUANT_HANDLERS = [_fp8_save_handler]


def apply_moe_quant_load(file, header_metadata, key, block_size = None):
    """Public entry point used by saving_utils._merge_moe_experts_file.

    Returns one of:
      - `(W_real, requant_fn_or_None)`
      - `(_MOE_QUANT_UNSAFE, None)`
    """
    for handler in _MOE_QUANT_HANDLERS:
        result = handler(file, header_metadata, key, block_size = block_size)
        if result is not None:
            return result
    return file.get_tensor(key), None


# ============================================================================
# Hook FP8 experts dispatch + relax HF Trainer FP8 guard
# ============================================================================
#
# transformers swaps a model's experts class with FP8Experts at load time when
# the checkpoint is FP8 (compressed-tensors / finegrained_fp8). The per-arch
# patch (patch_qwen3_moe etc.) targets the original Qwen3MoeExperts class, so
# the FP8 instance never sees our patched forward. Override the transformers
# FP8 experts registry directly so every FP8Experts.forward routes through
# forward_moe_backend_fp8.

from .common import (
    TEMPORARY_PATCHES,
    UNSLOTH_ENABLE_LOGGING,
    is_transformers_v5_moe_quantization_available,
)
from .utils import logger


def patch_fp8_experts_interface():
    try:
        from transformers.integrations.finegrained_fp8 import ALL_FP8_EXPERTS_FUNCTIONS
    except ImportError:
        return

    sentinel = "_unsloth_fp8_dispatcher"
    if getattr(ALL_FP8_EXPERTS_FUNCTIONS, sentinel, False):
        return

    def _unsloth_fp8_dispatch(self, hidden_states, top_k_index, top_k_weights, *args, **kwargs):
        return forward_moe_backend_fp8(self, hidden_states, top_k_index, top_k_weights)

    for key in ("grouped_mm", "batched_mm", "deepgemm"):
        try:
            ALL_FP8_EXPERTS_FUNCTIONS[key] = _unsloth_fp8_dispatch
        except Exception:
            pass

    setattr(ALL_FP8_EXPERTS_FUNCTIONS, sentinel, True)
    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: routed transformers ALL_FP8_EXPERTS_FUNCTIONS through forward_moe_backend_fp8")


def patch_fp8_validate_quantization_for_training():
    """
    HF Trainer rejects FP8 base models outright via validate_quantization_for_training.
    With LoRA the FP8 weights stay frozen and the adapter trains in bf16, which is the
    same contract the guard already accepts for bnb4bit / bnb8bit. Make the guard a
    no-op when the model is FP8-quantized; leave other quant kinds alone.
    """
    try:
        import transformers.trainer as _trainer_mod
        import transformers.trainer_utils as _trainer_utils_mod
    except ImportError:
        return

    _original = getattr(_trainer_utils_mod, "validate_quantization_for_training", None)
    if _original is None:
        return

    sentinel = "_unsloth_fp8_guard_patched"
    if getattr(_original, sentinel, False):
        return

    def _is_fp8_quantized(model):
        hq = getattr(model, "hf_quantizer", None)
        if hq is None:
            return False
        cfg = getattr(hq, "quantization_config", None)
        method = getattr(cfg, "quant_method", None)
        return method is not None and "fp8" in str(method).lower()

    def _patched(model):
        if _is_fp8_quantized(model):
            return None
        return _original(model)

    setattr(_patched, sentinel, True)
    _trainer_utils_mod.validate_quantization_for_training = _patched
    _trainer_mod.validate_quantization_for_training = _patched
    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: relaxed HF validate_quantization_for_training for FP8+LoRA")


def _register_transformers_v5_moe_fp8_patches():
    if not is_transformers_v5_moe_quantization_available():
        return
    TEMPORARY_PATCHES.append(patch_fp8_experts_interface)
    TEMPORARY_PATCHES.append(patch_fp8_validate_quantization_for_training)
pass
_register_transformers_v5_moe_fp8_patches()
