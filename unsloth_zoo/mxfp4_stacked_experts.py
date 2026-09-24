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

"""Remote-code MoE experts (Kimi-K3 compressed-tensors MXFP4 w1 / w2 / w3) kept as packed stacks."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mxfp4_dequant import Mxfp4ExpertParam, is_mxfp4_expert_param, mxfp4_dequantize

__all__ = [
    "Mxfp4StackedExperts",
    "dense_expert_modules",
    "mxfp4_grouped_linear",
    "stack_packed_experts",
]

_DEFAULT_CHUNK_BYTES = 1 << 31


def _chunk_bytes():
    try:
        return max(1, int(os.environ.get("UNSLOTH_MXFP4_EXPERT_CHUNK_MB", "")) << 20)
    except ValueError:
        return _DEFAULT_CHUNK_BYTES


def _experts_per_chunk(param, dtype):
    E, K, N = param._original_shape
    per_expert = K * N * torch.empty((), dtype = dtype).element_size()
    return max(1, min(E, _chunk_bytes() // per_expert))


def _grouped_mm(inputs, weight, offsets):
    # Imported late: moe_utils imports this package's mxfp4_dequant at module load.
    from .temporary_patches.moe_utils import _grouped_mm_with_backward_fix
    return _grouped_mm_with_backward_fix(inputs, weight, offsets)


def _dequantize_chunk(param, start, end, dtype, transpose, counts):
    scales = param.mxfp4_scales
    if scales.device != param.device:
        scales = param.mxfp4_scales = scales.to(param.device)
    return mxfp4_dequantize(
        param.data[start:end], scales[start:end], dtype = dtype, transpose = transpose,
        token_counts = counts[start:end],
    )


def _chunked_grouped_mm(inputs, param, counts, ends, dtype, transpose):
    E = len(ends)
    out_dim = param._original_shape[-1] if transpose else param._original_shape[-2]
    out = inputs.new_empty((inputs.shape[0], out_dim))
    step = _experts_per_chunk(param, dtype)
    ends_device = None
    for first in range(0, E, step):
        last = min(E, first + step)
        row_start = ends[first - 1] if first > 0 else 0
        row_end = ends[last - 1]
        if row_end == row_start:
            continue
        if ends_device is None:
            ends_device = torch.cumsum(counts, 0, dtype = torch.int32)
        offsets = ends_device[first:last] - row_start
        weight = _dequantize_chunk(param, first, last, dtype, transpose, counts)
        out[row_start:row_end] = _grouped_mm(inputs[row_start:row_end], weight, offsets)
        del weight
    return out


class _Mxfp4GroupedLinear(torch.autograd.Function):
    """Saves no weight: backward dequantizes W_e^T again."""

    @staticmethod
    def forward(ctx, inputs, param, counts, ends):
        ctx.param = param
        ctx.ends = ends
        ctx.save_for_backward(counts)
        with torch.no_grad():
            return _chunked_grouped_mm(inputs, param, counts, ends, inputs.dtype, True)

    @staticmethod
    def backward(ctx, grad_output):
        (counts,) = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            with torch.no_grad():
                grad_input = _chunked_grouped_mm(
                    grad_output.contiguous(), ctx.param, counts, ctx.ends, grad_output.dtype, False,
                )
        return grad_input, None, None, None


def mxfp4_grouped_linear(inputs, param, counts, ends):
    """Grouped ``inputs @ W_e`` over expert-sorted rows; ``ends`` is the host cumsum of ``counts``."""
    if not is_mxfp4_expert_param(param):
        offsets = torch.cumsum(counts, 0, dtype = torch.int32)
        return _grouped_mm(inputs.contiguous(), param.to(inputs.dtype), offsets)
    return _Mxfp4GroupedLinear.apply(inputs.contiguous(), param, counts, ends)


def _lora_delta(experts, name, inputs, ends_device):
    from .temporary_patches.moe_utils import _apply_lora_grouped_mm, take_moe_lora_stash
    lora = take_moe_lora_stash(experts, name)
    if lora is None:
        return None
    first, second, scaling = lora[0], lora[1], lora[2]
    return _apply_lora_grouped_mm(
        inputs.to(first.dtype), first, second, ends_device, scaling,
    ).to(inputs.dtype)


class _ExpertView:
    def __init__(self, experts, index):
        self.experts = experts
        self.index = index

    def _weight(self, param, dtype):
        if not is_mxfp4_expert_param(param):
            return param[self.index].t().to(dtype)
        start, end = self.index, self.index + 1
        return mxfp4_dequantize(
            param.data[start:end], param.mxfp4_scales.to(param.device)[start:end],
            dtype = dtype, transpose = False,
        )[0]

    def __call__(self, hidden_states):
        experts = self.experts
        dtype = hidden_states.dtype
        gate_up = F.linear(hidden_states, self._weight(experts.gate_up_proj, dtype))
        return F.linear(experts._activate(gate_up), self._weight(experts.down_proj, dtype))


class Mxfp4StackedExperts(nn.Module):
    """One MoE layer's experts as packed ``gate_up_proj`` / ``down_proj``; forward follows DeepSeek ``moe_infer``."""

    _unsloth_mxfp4_stacked_experts = True
    # Stacks are (E, in, out): how the MoE LoRA extraction orients PEFT's factors.
    _unsloth_grouped_mm_format = True

    def __init__(self, num_experts, hidden_size, intermediate_size, act_fn, fused_gate_up_act,
                 dtype = torch.bfloat16, device = "meta"):
        super().__init__()
        if hidden_size % 32 or intermediate_size % 32:
            raise ValueError("Unsloth: MXFP4 expert dims must be multiples of 32")
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.act_fn = act_fn
        # Kimi's SiTU takes the concatenated [gate, up] and returns act(gate) * up itself.
        self.fused_gate_up_act = bool(fused_gate_up_act)
        self.mxfp4_dtype = dtype
        E, H, I = num_experts, hidden_size, intermediate_size

        def _raw(*shape):
            return nn.Parameter(torch.empty(shape, dtype = torch.uint8, device = device), requires_grad = False)

        self.gate_up_blocks = _raw(E, 2 * I, H // 32, 16)
        self.gate_up_scales = _raw(E, 2 * I, H // 32)
        self.down_blocks = _raw(E, H, I // 32, 16)
        self.down_scales = _raw(E, H, I // 32)

    def __len__(self):
        return self.num_experts

    def __getitem__(self, index):
        self.finalize()
        if index < 0:
            index += self.num_experts
        if not 0 <= index < self.num_experts:
            raise IndexError(index)
        return _ExpertView(self, index)

    def finalize(self):
        if "gate_up_blocks" not in self._parameters:
            return self
        for name in ("gate_up", "down"):
            if any(self._parameters[f"{name}_{kind}"].is_meta for kind in ("blocks", "scales")):
                raise RuntimeError(f"Unsloth: MXFP4 expert stack `{name}` was never loaded")
        for name in ("gate_up", "down"):
            blocks = self._parameters.pop(f"{name}_blocks")
            scales = self._parameters.pop(f"{name}_scales")
            self.register_parameter(
                f"{name}_proj",
                Mxfp4ExpertParam(blocks.data, mxfp4_scales = scales.data, mxfp4_dtype = self.mxfp4_dtype),
            )
        return self

    def _activate(self, gate_up):
        if self.fused_gate_up_act:
            return self.act_fn(gate_up)
        gate, up = gate_up.chunk(2, dim = -1)
        return self.act_fn(gate) * up

    def forward(self, hidden_states, topk_idx, topk_weight):
        self.finalize()
        num_tokens, top_k = topk_idx.shape
        flat_idx = topk_idx.reshape(-1)
        order = torch.argsort(flat_idx, stable = True)
        rows = hidden_states[order // top_k]
        counts = torch.bincount(flat_idx, minlength = self.num_experts)
        ends_device = torch.cumsum(counts, 0, dtype = torch.int32)
        ends = ends_device.tolist()

        gate_up = mxfp4_grouped_linear(rows, self.gate_up_proj, counts, ends)
        delta = _lora_delta(self, "gate_up_proj", rows, ends_device)
        if delta is not None:
            gate_up = gate_up + delta
        hidden = self._activate(gate_up).to(rows.dtype)
        out = mxfp4_grouped_linear(hidden, self.down_proj, counts, ends)
        delta = _lora_delta(self, "down_proj", hidden, ends_device)
        if delta is not None:
            out = out + delta

        unsorted = torch.empty_like(out)
        unsorted[order] = out
        weighted = unsorted.view(num_tokens, top_k, out.shape[-1]).to(torch.float32) * topk_weight.to(
            torch.float32
        ).unsqueeze(-1)
        return weighted.sum(dim = 1).to(hidden_states.dtype)

    def extra_repr(self):
        return (
            f"num_experts={self.num_experts}, hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}, packed=mxfp4"
        )


def _dense_stack(param, dtype):
    if is_mxfp4_expert_param(param):
        return param.dequantize(dtype)
    return param.detach().to(dtype)


def dense_expert_modules(experts, device = "cpu"):
    """Per-expert ``w1`` / ``w2`` / ``w3`` Linears, so a full save writes keys the remote code reloads."""
    experts.finalize()
    dtype = experts.mxfp4_dtype
    I = experts.intermediate_size
    gate_up = _dense_stack(experts.gate_up_proj, dtype).to(device)
    down = _dense_stack(experts.down_proj, dtype).to(device)
    out = nn.ModuleList()
    for e in range(experts.num_experts):
        expert = nn.Module()
        for proj, weight in (("w1", gate_up[e, :, :I]), ("w3", gate_up[e, :, I:]), ("w2", down[e])):
            linear = nn.Linear(weight.shape[0], weight.shape[1], bias = False, device = "meta")
            linear.weight = nn.Parameter(weight.t().contiguous(), requires_grad = False)
            setattr(expert, proj, linear)
        out.append(expert)
    return out


def stack_packed_experts(per_expert_lists, blocks = True):
    """Per-projection lists of E (out, n) uint8 -> (E, sum(out), n); ``blocks`` reshapes to (..., in / 32, 16)."""
    stacks = [torch.stack(list(tensors), dim = 0) for tensors in per_expert_lists]
    stacked = stacks[0] if len(stacks) == 1 else torch.cat(stacks, dim = 1)
    if blocks:
        stacked = stacked.reshape(*stacked.shape[:2], -1, 16)
    return stacked.contiguous()
