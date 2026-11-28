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
from .mxfp4_gemm import _gemv_rows, mxfp4_gemm_available, mxfp4_grouped_gemv_available, mxfp4_grouped_matmul

try:
    # Bit-identical to the eager combine; returns None off CUDA or without Triton.
    from unsloth_zoo.temporary_patches.moe_triton_kernels import weighted_unpermute as _weighted_unpermute
except Exception:
    _weighted_unpermute = None

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


def _expert_bytes(param, dtype):
    E, K, N = param._original_shape
    return K * N * torch.empty((), dtype = dtype).element_size()


def _packed_rows_cols(param):
    """(N, K) of the packed (E, N, K) rows, whatever the logical orientation."""
    dims = getattr(param, "packed_dims", None)
    if dims is not None:
        _, N, K = dims()
        return N, K
    E, A, B = param._original_shape
    return (B, A) if param.mxfp4_transposed else (A, B)


def _experts_per_chunk(param, dtype, budget = None):
    per_expert = _expert_bytes(param, dtype)
    return max(1, min(param._original_shape[0], (_chunk_bytes() if budget is None else budget) // per_expert))


# Per-GPU fallback [chunk bytes, calls left before full-budget retry, smallest failed chunk bytes];
# in bytes, not experts, so smaller experts (down_proj is half gate_up) still probe below it.
_CHUNK_BUDGET = {}
_CHUNK_RETRY_CALLS = 1024


def _grouped_mm(inputs, weight, offsets):
    # Imported late: moe_utils imports this package's mxfp4_dequant at module load.
    from unsloth_zoo.temporary_patches.moe_utils import _grouped_mm_with_backward_fix
    return _grouped_mm_with_backward_fix(inputs, weight, offsets)


def _dequantize_chunk(param, start, end, dtype, transpose, counts):
    if hasattr(param, "dequantize_experts"):  # NVFP4 / MXFP8 stacks (packed_experts.py)
        from unsloth_zoo.packed_experts import PACKED_EXPERT_STATS
        if not torch.compiler.is_compiling():
            PACKED_EXPERT_STATS["chunks"] += 1
        # Contiguous rows used transposed: same GEMM layout as 16-bit path, agreeing bitwise.
        rows = param.dequantize_experts(start, end, dtype, False, counts[start:end])
        return rows.transpose(-2, -1) if transpose else rows
    scales = param.mxfp4_scales
    if scales.device != param.device:
        scales = param.mxfp4_scales = scales.to(param.device)
    return mxfp4_dequantize(
        param.data[start:end], scales[start:end], dtype = dtype, transpose = transpose,
        token_counts = counts[start:end],
    )


def _chunk_budget(device):
    state = _CHUNK_BUDGET.get(device.index)
    if state is None:
        return None
    state[1] -= 1
    if state[1] <= 0:
        _CHUNK_BUDGET.pop(device.index, None)
        return None
    return state


def _chunked_grouped_mm(inputs, param, counts, ends, dtype, transpose):
    """``ends`` = host cumsum of ``counts``; None derives it only when the experts need more than one chunk."""
    E = param._original_shape[0] if ends is None else len(ends)
    rows, cols = _packed_rows_cols(param)
    out_dim = rows if transpose else cols
    out = None   # allocated on the first multi-chunk write: the single-chunk path returns the GEMM output itself
    device = inputs.device
    adaptive = device.type == "cuda"
    state = _chunk_budget(device) if adaptive else None
    budget = None if state is None else state[0]
    if state is not None and _expert_bytes(param, dtype) >= state[2] and _fused_fallback_ok(inputs, param):
        # One expert of this size (or less) already failed to allocate: skip straight to the fused kernel.
        del out
        return mxfp4_grouped_matmul(inputs, param.data, _scales(param), counts, trans = not transpose)
    step = _experts_per_chunk(param, dtype, budget)
    ends_device = None
    first = 0
    while first < E:
        last = min(E, first + step)
        if ends is None and (first > 0 or last < E):
            if _no_host_sync(device):
                # A CUDA graph capture (or a decode step planned for one) cannot read the offsets on the host.
                return _chunked_grouped_mm_on_device(inputs, param, counts, dtype, transpose, step, out_dim)
            ends = torch.cumsum(counts, 0).tolist()   # several chunks slice rows: one host sync
        if ends is None:
            # One chunk covers every expert and every row: no host offsets needed.
            if inputs.shape[0] == 0:
                return inputs.new_empty((0, out_dim))
            ends_device = torch.cumsum(counts, 0, dtype = torch.int32)
            try:
                weight = _dequantize_chunk(param, 0, E, dtype, transpose, counts)
                if _no_host_sync(device) and not _grouped_mm_capture_safe(inputs, weight):
                    return _masked_grouped_mm(inputs, weight, ends_device, 0, inputs.new_zeros((inputs.shape[0], out_dim)))
                return _grouped_mm(inputs, weight, ends_device)
            except torch.cuda.OutOfMemoryError:
                if not adaptive or E == 1:
                    raise
                weight = None
            step = max(1, E // 2)
            _CHUNK_BUDGET[device.index] = [step * _expert_bytes(param, dtype), _CHUNK_RETRY_CALLS,
                                           E * _expert_bytes(param, dtype)]
            continue
        row_start = ends[first - 1] if first > 0 else 0
        row_end = ends[last - 1]
        if row_end == row_start:
            first = last
            continue
        if ends_device is None:
            ends_device = torch.cumsum(counts, 0, dtype = torch.int32)
        offsets = ends_device[first:last] - row_start
        if out is None:
            out = inputs.new_empty((inputs.shape[0], out_dim))
        try:
            weight = _dequantize_chunk(param, first, last, dtype, transpose, counts)
            out[row_start:row_end] = _grouped_mm(inputs[row_start:row_end], weight, offsets)
        except torch.cuda.OutOfMemoryError:
            if not adaptive:
                raise
            weight = None
        if weight is None:
            # A dequantized chunk (2 GiB default) that does not fit is halved and that size kept for later calls; below
            # one expert, use the fused kernel, which never builds a 16-bit expert.
            failed = step * _expert_bytes(param, dtype)
            if step > 1:
                step //= 2
                _CHUNK_BUDGET[device.index] = [step * _expert_bytes(param, dtype), _CHUNK_RETRY_CALLS, failed]
                continue
            if not _fused_fallback_ok(inputs, param):
                raise torch.cuda.OutOfMemoryError(
                    f"Unsloth: one dequantized MXFP4 expert ({_expert_bytes(param, dtype)} bytes) does not fit"
                )
            _CHUNK_BUDGET[device.index] = [0, _CHUNK_RETRY_CALLS, failed]
            del out
            return mxfp4_grouped_matmul(inputs, param.data, _scales(param), counts, trans = not transpose)
        del weight
        first = last
    return out if out is not None else inputs.new_empty((inputs.shape[0], out_dim))


def _no_host_sync(device):
    """Inside a CUDA graph capture, or a decode step Unsloth plans to capture (its eager warm-up must take the same
    kernels for the first replay to validate bitwise)."""
    if _tracing_or_capturing(device):
        return True
    try:
        from unsloth_zoo.decode_graph_state import in_graph_decode_step
    except Exception:
        return False
    return in_graph_decode_step()


# Capture-safety verdicts taken outside a capture (a graph decode step's eager warm-up), reused inside it.
_CAPTURE_SAFE = {}


def _grouped_mm_capture_safe(inputs, weight):
    """Whether the zoo's grouped GEMM reads no offsets on the host for these operands
    (`moe_utils.grouped_mm_backend_reads_host_offsets`). Inside a real capture nothing is probed: the verdict of the
    warm-up call of the same shapes, else False (the masked path)."""
    if inputs.device.type != "cuda":
        return False
    key = (inputs.device.index, inputs.dtype, weight.dtype, tuple(inputs.shape), tuple(weight.shape),
           tuple(weight.stride()))
    if _tracing_or_capturing(inputs.device):
        return _CAPTURE_SAFE.get(key, False)
    try:
        from unsloth_zoo.temporary_patches import moe_utils
        safe = not moe_utils.grouped_mm_backend_reads_host_offsets(inputs, weight)
    except Exception:
        safe = False
    if len(_CAPTURE_SAFE) > 4096:
        _CAPTURE_SAFE.clear()
    _CAPTURE_SAFE[key] = safe
    return safe


def _masked_grouped_mm(inputs, weight, ends, first, out):
    """Experts first.. of `weight` (C, K, N) over every row, each kept on its own rows: offsets stay on the device.
    `ends` are the absolute expert ends (int32, device); `out` holds the other experts' rows."""
    row = torch.arange(inputs.shape[0], device = inputs.device, dtype = torch.int32)[:, None]
    for i in range(weight.shape[0]):
        e = first + i
        lo = ends[e - 1] if e > 0 else ends.new_zeros(())
        keep = (row >= lo) & (row < ends[e])
        out = torch.where(keep, inputs @ weight[i].to(inputs.dtype), out)
    return out


def _chunked_grouped_mm_on_device(inputs, param, counts, dtype, transpose, step, out_dim):
    """`_chunked_grouped_mm` without host offsets: each chunk's grouped GEMM runs over every row with the absolute
    expert ends as offsets (its first expert's group also takes the rows before the chunk, rows after its last
    end are left out), and only the chunk's own rows are kept. Same chunk memory, more rows multiplied. Outside a
    real capture (a graph decode step's eager warm-up) a chunk that does not fit is halved and that size kept, as
    the host path does; the capture then starts from the kept size."""
    E = param._original_shape[0]
    M = inputs.shape[0]
    if M == 0:
        return inputs.new_empty((0, out_dim))
    device = inputs.device
    capturing = _tracing_or_capturing(device)
    ends = torch.cumsum(counts, 0, dtype = torch.int32)
    row = torch.arange(M, device = device, dtype = torch.int32)[:, None]
    out = inputs.new_zeros((M, out_dim))
    first = 0
    while first < E:
        last = min(E, first + step)
        weight = part = None
        try:
            weight = _dequantize_chunk(param, first, last, dtype, transpose, counts)
            if not _grouped_mm_capture_safe(inputs, weight):
                out = _masked_grouped_mm(inputs, weight, ends, first, out)
            else:
                part = _grouped_mm(inputs, weight, ends[first:last].contiguous())
                if part.shape[0] < M:
                    # The per-expert loop fallback returns rows only through the chunk's last end.
                    part = torch.nn.functional.pad(part, (0, 0, 0, M - part.shape[0]))
                lo = ends[first - 1] if first > 0 else ends.new_zeros(())
                keep = (row >= lo) & (row < ends[last - 1])
                out = torch.where(keep, part, out)
        except torch.cuda.OutOfMemoryError:
            if capturing:
                raise
            weight = part = None
            failed = step * _expert_bytes(param, dtype)
            if step == 1:
                if not _fused_fallback_ok(inputs, param):
                    raise
                if device.type == "cuda":
                    _CHUNK_BUDGET[device.index] = [0, _CHUNK_RETRY_CALLS, failed]
                return mxfp4_grouped_matmul(inputs, param.data, _scales(param), counts, trans = not transpose)
            step //= 2
            if device.type == "cuda":
                _CHUNK_BUDGET[device.index] = [step * _expert_bytes(param, dtype), _CHUNK_RETRY_CALLS, failed]
            continue
        del weight, part
        first = last
    return out


def _fused_fallback_ok(inputs, param):
    # Below sm_80 the verified tile kernel is slow but needs only its output: still better than an OOM.
    return (
        is_mxfp4_expert_param(param) and param.mxfp4_transposed
        and mxfp4_grouped_gemv_available(inputs.device, inputs.dtype)
    )


def _fused(inputs, param, transpose = True):
    if not (inputs.is_cuda and is_mxfp4_expert_param(param) and param.mxfp4_transposed):
        return False
    if mxfp4_gemm_available(inputs.device, inputs.dtype):
        return True
    # On sm_75 only decode rows take the GEMV, since the tile kernel loses to dequantize + grouped GEMM; under
    # torch.compile or CUDA graph capture all rows do, as the fallback reads host offsets.
    return (
        (_gemv_rows(inputs.shape[0], param.shape[0], transpose) or _tracing_or_capturing(inputs.device))
        and mxfp4_grouped_gemv_available(inputs.device, inputs.dtype)
    )


def _tracing_or_capturing(device = None):
    if torch.compiler.is_compiling():
        return True
    # A capture needs an initialized CUDA context; this also keeps the check free where CUDA never started.
    if not torch.cuda.is_initialized():
        return False
    if torch.cuda.is_current_stream_capturing():
        return True
    # The query only sees the current device; a split-device model may capture on another GPU.
    if device is None or device.type != "cuda" or device.index in (None, torch.cuda.current_device()):
        return False
    with torch.cuda.device(device):
        return torch.cuda.is_current_stream_capturing()


def _scales(param):
    scales = param.mxfp4_scales
    if scales.device != param.device:
        scales = param.mxfp4_scales = scales.to(param.device)
    return scales


def _grouped(inputs, param, counts, ends, transpose):
    if _fused(inputs, param, transpose):
        return mxfp4_grouped_matmul(inputs, param.data, _scales(param), counts, trans = not transpose)
    if ends is None:
        ends = torch.cumsum(counts, 0).tolist()
    return _chunked_grouped_mm(inputs, param, counts, ends, inputs.dtype, transpose)


class _Mxfp4GroupedLinear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, param, counts, ends):
        ctx.param = param
        ctx.ends = ends
        ctx.save_for_backward(counts)
        with torch.no_grad():
            return _grouped(inputs, param, counts, ends, True)

    @staticmethod
    def backward(ctx, grad_output):
        (counts,) = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            with torch.no_grad():
                grad_input = _grouped(grad_output.contiguous(), ctx.param, counts, ctx.ends, False)
        return grad_input, None, None, None


def mxfp4_grouped_linear(inputs, param, counts, ends = None):
    """Grouped ``inputs @ W_e`` over expert-sorted rows; ``ends`` = host cumsum of ``counts`` (None: derived if needed)."""
    if not is_mxfp4_expert_param(param):
        offsets = torch.cumsum(counts, 0, dtype = torch.int32)
        return _grouped_mm(inputs.contiguous(), param.to(inputs.dtype), offsets)
    inputs = inputs.contiguous()
    if not (torch.is_grad_enabled() and inputs.requires_grad):
        with torch.no_grad():
            return _grouped(inputs, param, counts, ends, True)
    return _Mxfp4GroupedLinear.apply(inputs, param, counts, ends)


def _lora_delta(experts, name, inputs, ends_device):
    from unsloth_zoo.temporary_patches.moe_utils import _apply_lora_grouped_mm, take_moe_lora_stash
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
        # scatter_add, not bincount: bincount reads max() back to the host, a sync per MoE layer.
        flat_idx = flat_idx.long()
        counts = torch.zeros(self.num_experts, dtype = torch.int64, device = flat_idx.device)
        counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx))
        ends_device = torch.cumsum(counts, 0, dtype = torch.int32)
        # The fused GEMM schedules on device; only the chunked fallback needs host offsets (one sync).
        ends = None if _fused(rows, self.gate_up_proj) or not is_mxfp4_expert_param(self.gate_up_proj) else ends_device.tolist()

        gate_up = mxfp4_grouped_linear(rows, self.gate_up_proj, counts, ends)
        delta = _lora_delta(self, "gate_up_proj", rows, ends_device)
        if delta is not None:
            gate_up = gate_up + delta
        hidden = self._activate(gate_up).to(rows.dtype)
        out = mxfp4_grouped_linear(hidden, self.down_proj, counts, ends)
        delta = _lora_delta(self, "down_proj", hidden, ends_device)
        if delta is not None:
            out = out + delta

        # Avoids a [T, top_k, H] fp32 product.
        if _weighted_unpermute is not None and out.is_cuda and num_tokens > 0:
            combined = _weighted_unpermute(
                out, order, topk_weight.to(torch.float32).reshape(-1)[order], num_tokens, top_k,
                out_dtype = hidden_states.dtype,
            )
            if combined is not None:
                return combined
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


def _dense_stack(param, dtype, device):
    if not is_mxfp4_expert_param(param):
        return param.detach().to(dtype).to(device)
    # Chunked into `device`: a whole bf16 stack per layer does not fit on the GPU.
    scales = param.mxfp4_scales
    if scales.device != param.device:
        scales = param.mxfp4_scales = scales.to(param.device)
    E, step, out = param.data.shape[0], _experts_per_chunk(param, dtype), None
    for start in range(0, E, step):
        end = min(E, start + step)
        piece = mxfp4_dequantize(
            param.data[start:end], scales[start:end], dtype = dtype, transpose = param.mxfp4_transposed,
        )
        if out is None:
            out = torch.empty((E, *piece.shape[1:]), dtype = dtype, device = device)
        out[start:end].copy_(piece)
    return out


def dense_expert_modules(experts, device = "cpu"):
    """Per-expert ``w1`` / ``w2`` / ``w3`` Linears, so a full save writes keys the remote code reloads."""
    experts.finalize()
    dtype = experts.mxfp4_dtype
    I = experts.intermediate_size
    gate_up = _dense_stack(experts.gate_up_proj, dtype, device)
    down = _dense_stack(experts.down_proj, dtype, device)
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
