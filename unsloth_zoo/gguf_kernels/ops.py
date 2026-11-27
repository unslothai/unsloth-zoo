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

"""Public GGUF kernel entry points.

``dequantize_gguf``            packed bytes -> dense tensor.
``gguf_linear``                x @ W^T (+ bias) against a packed weight, autograd gives dX only.
``gguf_linear_backward_input`` grad_out @ W.
``gguf_grouped_mm``            per-expert x @ W[e]^T (or x @ W[e]) on a packed (E, N, K) stack, rows sorted by
                               expert with ``counts`` (E,) like the other unsloth_zoo grouped MoE kernels.

CUDA (and ROCm) run Triton kernels: a fused GEMV for one row, a fused dequant-in-GEMM tile kernel for few rows
(and every grouped call), and Triton dequantize + cuBLAS for many rows. CPU, or a failed launch, falls back to the
torch reference. Kill switches: ``UNSLOTH_GGUF_TRITON=0`` (torch reference everywhere),
``UNSLOTH_GGUF_FUSED=0`` (dequantize + matmul, no fused GEMV / GEMM)."""

import os
from typing import List, Optional

import torch

from .formats import GGML_BLOCK, GGMLType as T, TABLE_OFFSETS, ggml_packed_bytes, ggml_row_bytes, ggml_type_id, table_values
from .reference import dequantize_reference

__all__ = [
    "dequantize_gguf",
    "gguf_linear",
    "gguf_linear_backward_input",
    "gguf_grouped_mm",
    "gguf_kernels_available",
]

try:
    import triton  # noqa: F401
    from . import triton_kernels as _tk
    _HAS_TRITON = True
except Exception:
    _tk = None
    _HAS_TRITON = False

_PASSTHROUGH = {T.F32: torch.float32, T.F16: torch.float16, T.BF16: torch.bfloat16}
# Formats whose 8-value group fields (u32 aux words, sign bytes, sub-block scales) dominate the decode: the tile GEMM
# indexes those per group (4-D tile); for the rest the cheaper reshape into a dot operand wins (L4 / A100 timings).
_SPLIT_MM = frozenset((T.IQ2_XXS, T.IQ3_XXS, T.IQ4_XS, T.Q3_K, T.Q6_K))
# The GEMV and dequantize kernels index group fields per group for every format.
_NO_SPLIT_GEMV = frozenset()
# Rows of x up to which the fused tile kernel beats Triton dequantize + cuBLAS (dense weights), from interleaved
# sweeps at 17408 x 5120: on HBM datacenter parts (A100 / H100 / B200: sm80, sm90, sm100) dequantize + cuBLAS is
# cheap, so 256-value formats leave the fused path above 8 rows and the rest above 64; elsewhere (T4 / L4 / consumer
# cards) fused wins up to 128. UNSLOTH_GGUF_FUSED_MAX_ROWS overrides.
_FUSED_MAX_ROWS_ENV = os.environ.get("UNSLOTH_GGUF_FUSED_MAX_ROWS")
_FUSED_MAX_ROWS = int(_FUSED_MAX_ROWS_ENV) if _FUSED_MAX_ROWS_ENV else None
_HBM_ARCHS = frozenset(((8, 0), (9, 0), (10, 0)))
_ROW_CAPS = {}


def _fused_max_rows(device, ggml_type):
    if _FUSED_MAX_ROWS is not None:
        return _FUSED_MAX_ROWS
    big_blocks = GGML_BLOCK[ggml_type][0] >= 256
    key = (device.index, big_blocks)
    cap = _ROW_CAPS.get(key)
    if cap is None:
        hbm = torch.version.hip is None and torch.cuda.get_device_capability(device) in _HBM_ARCHS
        cap = (8 if big_blocks else 64) if hbm else 128
        _ROW_CAPS[key] = cap
    return cap


def _env_on(name):
    return os.environ.get(name, "1").strip().lower() not in ("0", "false", "no", "off")


def _triton_ok(t):
    return _HAS_TRITON and t.is_cuda and _env_on("UNSLOTH_GGUF_TRITON")


def _old_triton():
    try:
        major, minor = (int(v) for v in triton.__version__.split(".")[:2])
    except Exception:
        return False
    return (major, minor) < (3, 3)


_OLD_TRITON = _HAS_TRITON and _old_triton()
_TABLES = {}


def _table(device):
    key = (device.type, device.index)
    tab = _TABLES.get(key)
    if tab is None:
        tab = torch.tensor(table_values(), dtype = torch.int8, device = device)
        _TABLES[key] = tab
    return tab


def _check(packed, ggml_type, shape, *others):
    if not isinstance(packed, torch.Tensor) or packed.dtype != torch.uint8:
        raise TypeError("Unsloth: packed GGUF weight must be a torch.uint8 tensor")
    need = ggml_packed_bytes(ggml_type, shape)
    if packed.numel() != need:
        raise ValueError(f"Unsloth: packed GGUF tensor has {packed.numel()} bytes, shape {tuple(shape)} needs {need}")
    for name, t in others:
        if t is not None and t.device != packed.device:
            raise ValueError(f"Unsloth: {name} is on {t.device} but the packed GGUF weight is on {packed.device}")


def _launch_ok(err):
    """Re-raise what a fallback must not hide: compile-time tracing, OOM."""
    if isinstance(err, torch.cuda.OutOfMemoryError):
        return False
    mod = type(err).__module__ or ""
    return not mod.startswith("torch._dynamo")


_WARNED = set()


def _warn_fallback(what, err):
    if what not in _WARNED:
        _WARNED.add(what)
        import warnings
        warnings.warn(f"Unsloth: GGUF Triton {what} failed ({type(err).__name__}: {err}); using the torch path.")


def _dequant_triton(packed, ggml_type, shape, dtype):
    qk, bs = GGML_BLOCK[ggml_type]
    K = shape[-1]
    rows = packed.numel() // ggml_row_bytes(ggml_type, K)
    # sm < 80 has no bf16 conversion instruction (Triton 3.2 cannot emit one): decode to float32, round in torch.
    native = dtype != torch.bfloat16 or torch.version.hip is not None or torch.cuda.get_device_capability(packed.device)[0] >= 8
    out = torch.empty(shape, dtype = dtype if native else torch.float32, device = packed.device)
    if out.numel() == 0:
        return out.to(dtype)
    NB = max(1, 256 // qk)
    BLOCK_R = 16
    grid = (-(-rows // BLOCK_R), -(-(K // qk) // NB))
    with torch.cuda.device(packed.device):
        _tk.dequantize_kernel[grid](
            packed, out, _table(packed.device), rows, K, ggml_row_bytes(ggml_type, K),
            TYPE = ggml_type, QK = qk, BS = bs, BLOCK_R = BLOCK_R, NB = NB, SPLIT = True,
            num_warps = 4, enable_fp_fusion = False,
        )
    return out if native else out.to(dtype)


def _dequantize_impl(packed, ggml_type, shape, dtype):
    if ggml_type in _PASSTHROUGH:
        out = dequantize_reference(packed, ggml_type, shape, dtype)
        # F32 -> float32 is a view of the input bytes; a custom op must return a fresh tensor
        return out.clone() if out.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr() else out
    if _triton_ok(packed):
        try:
            return _dequant_triton(packed.contiguous(), ggml_type, shape, dtype)
        except Exception as err:
            if not _launch_ok(err):
                raise
            _warn_fallback("dequantize", err)
    return dequantize_reference(packed, ggml_type, shape, dtype)


@torch.library.custom_op("unsloth_zoo::gguf_dequantize", mutates_args = ())
def _dequantize_op(packed: torch.Tensor, ggml_type: int, shape: List[int], dtype: torch.dtype) -> torch.Tensor:
    return _dequantize_impl(packed, ggml_type, tuple(shape), dtype)


@_dequantize_op.register_fake
def _dequantize_fake(packed, ggml_type, shape, dtype):
    return packed.new_empty(tuple(shape), dtype = dtype)


def dequantize_gguf(packed: torch.Tensor, ggml_type, shape, dtype = torch.bfloat16) -> torch.Tensor:
    """Dense ``shape`` tensor of ``dtype`` from packed GGUF bytes (``prod(shape[:-1])`` rows of blocks)."""
    ggml_type = ggml_type_id(ggml_type)
    shape = [int(s) for s in shape]
    _check(packed, ggml_type, shape)
    return _dequantize_op(packed, ggml_type, shape, dtype)


def _fused_ok(x, ggml_type):
    if not (_triton_ok(x) and _env_on("UNSLOTH_GGUF_FUSED")) or ggml_type in _PASSTHROUGH:
        return False
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if x.dtype == torch.bfloat16 and torch.version.hip is None:
        # tl.dot needs native bf16 (sm80+); T4 runs fp16.
        return torch.cuda.get_device_capability(x.device)[0] >= 8
    return True


def _mm_launch(x2, packed, ggml_type, out, bias, counts, M, N, K, E, R, trans_b, grouped):
    qk, bs = GGML_BLOCK[ggml_type]
    row_bytes = ggml_row_bytes(ggml_type, K if trans_b else N)
    rows_per_tile = M / E if grouped else M
    BLOCK_M = 16 if rows_per_tile <= 16 else (32 if rows_per_tile <= 32 else (64 if rows_per_tile <= 64 else 128))
    # Two tile configs from an L4 sweep (17408 x 5120, M = 16 / 64 / 128); the value axis holds whole blocks.
    if BLOCK_M <= 32:
        packed_rows, values, warps, stages = 32, 256, 4, (3 if qk <= 64 else 2)
    elif qk <= 64:
        packed_rows, values, warps, stages = 128, 128, 8, 3
    else:
        packed_rows, values, warps, stages = 32, 256, 8, 2
    BLOCK_N, BLOCK_K = (packed_rows, values) if trans_b else (values, packed_rows)
    tiles_m = (-(-M // BLOCK_M) + min(E, M)) if grouped else -(-M // BLOCK_M)
    grid = (tiles_m, -(-N // BLOCK_N))
    host_sched = grouped and _OLD_TRITON
    sched = None
    if host_sched:
        sched = torch.stack((torch.cumsum((counts + BLOCK_M - 1) // BLOCK_M, 0), torch.cumsum(counts, 0))).to(torch.int32)
    _tk.mm_kernel[grid](
        x2, packed, out, x2 if bias is None else bias, x2 if counts is None else counts,
        x2 if sched is None else sched, _table(x2.device),
        M, N, K, E, R, row_bytes, x2.stride(0), out.stride(0),
        TYPE = ggml_type, QK = qk, BS = bs, TRANS_B = trans_b, GROUPED = grouped,
        E_POW2 = 1 << max(E - 1, 0).bit_length(), HAS_BIAS = bias is not None, HOST_SCHED = host_sched,
        BLOCK_M = BLOCK_M, BLOCK_N = BLOCK_N, BLOCK_K = BLOCK_K, SPLIT = ggml_type in _SPLIT_MM,
        num_warps = warps, num_stages = stages,
    )
    return out


def _gemv_launch(x1, packed, ggml_type, out, bias, N, K):
    qk, bs = GGML_BLOCK[ggml_type]
    # 4 rows x 512 values per step: best or near-best for every format on A100 and L4 (interleaved sweep)
    BLOCK_N, NB = 4, max(1, 512 // qk)
    grid = (-(-N // BLOCK_N),)
    _tk.gemv_kernel[grid](
        x1, packed, out, x1 if bias is None else bias, _table(x1.device), N, K, ggml_row_bytes(ggml_type, K),
        TYPE = ggml_type, QK = qk, BS = bs, HAS_BIAS = bias is not None,
        BLOCK_N = BLOCK_N, NB = NB, SPLIT = ggml_type not in _NO_SPLIT_GEMV, num_warps = 4,
    )
    return out


def _linear_impl(x, packed, ggml_type, shape, bias, trans_b):
    """trans_b: y = x @ W^T (+ bias), W = (N, K). Else: y = x @ W."""
    N, K = shape
    inner, outer = (K, N) if trans_b else (N, K)
    lead = x.shape[:-1]
    x2 = x.reshape(-1, inner)
    M = x2.shape[0]
    gemv = M == 1 and trans_b
    # tl.dot rounds float32 inputs to tf32: float32 activations use the GEMV or dequantize + matmul only.
    if M > 0 and _fused_ok(x, ggml_type) and (gemv or (M <= _fused_max_rows(x.device, ggml_type) and x.dtype != torch.float32)):
        try:
            x2 = x2.contiguous()
            out = torch.empty((M, outer), dtype = x.dtype, device = x.device)
            with torch.cuda.device(x.device):
                if gemv:
                    _gemv_launch(x2, packed, ggml_type, out, bias, N, K)
                else:
                    _mm_launch(x2, packed, ggml_type, out, bias, None, M, outer, inner, 1, N, trans_b, False)
            return out.reshape(*lead, outer)
        except Exception as err:
            if not _launch_ok(err):
                raise
            _warn_fallback("fused GEMM", err)
    w = _dequantize_impl(packed, ggml_type, (N, K), x.dtype)
    if trans_b:
        y = torch.nn.functional.linear(x, w, bias)
    else:
        y = x @ w
    return y


@torch.library.custom_op("unsloth_zoo::gguf_linear", mutates_args = ())
def _linear_op(x: torch.Tensor, packed: torch.Tensor, ggml_type: int, shape: List[int], bias: Optional[torch.Tensor]) -> torch.Tensor:
    return _linear_impl(x, packed, ggml_type, tuple(shape), bias, True)


@_linear_op.register_fake
def _linear_fake(x, packed, ggml_type, shape, bias):
    return x.new_empty((*x.shape[:-1], shape[0]))


@torch.library.custom_op("unsloth_zoo::gguf_linear_backward_input", mutates_args = ())
def _linear_bwd_op(grad_out: torch.Tensor, packed: torch.Tensor, ggml_type: int, shape: List[int]) -> torch.Tensor:
    return _linear_impl(grad_out, packed, ggml_type, tuple(shape), None, False)


@_linear_bwd_op.register_fake
def _linear_bwd_fake(grad_out, packed, ggml_type, shape):
    return grad_out.new_empty((*grad_out.shape[:-1], shape[1]))


def _linear_setup(ctx, inputs, output):
    x, packed, ggml_type, shape, bias = inputs
    ctx.save_for_backward(packed)
    ctx.ggml_type = ggml_type
    ctx.shape = list(shape)
    ctx.bias_grad = bias is not None and bias.requires_grad


def _linear_backward(ctx, grad):
    (packed,) = ctx.saved_tensors
    grad_x = None
    if ctx.needs_input_grad[0]:
        grad_x = _linear_bwd_op(grad.contiguous(), packed, ctx.ggml_type, ctx.shape)
    grad_b = grad.reshape(-1, grad.shape[-1]).sum(0) if ctx.bias_grad else None
    return grad_x, None, None, None, grad_b


torch.library.register_autograd("unsloth_zoo::gguf_linear", _linear_backward, setup_context = _linear_setup)


def gguf_linear(x: torch.Tensor, packed: torch.Tensor, ggml_type, shape, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``x @ W^T + bias`` with ``W`` the packed GGUF weight of logical ``shape = (out_features, in_features)``.
    Differentiable in ``x`` (and ``bias``); the packed base is frozen."""
    ggml_type = ggml_type_id(ggml_type)
    shape = [int(s) for s in shape]
    if len(shape) != 2:
        raise ValueError(f"Unsloth: gguf_linear needs a 2-D weight shape, got {shape}")
    _check(packed, ggml_type, shape, ("x", x), ("bias", bias))
    if x.shape[-1] != shape[1]:
        raise ValueError(f"Unsloth: x has {x.shape[-1]} features, the GGUF weight expects {shape[1]}")
    return _linear_op(x, packed, ggml_type, shape, bias)


def gguf_linear_backward_input(grad_out: torch.Tensor, packed: torch.Tensor, ggml_type, shape) -> torch.Tensor:
    """``grad_out @ W`` (the input gradient of ``gguf_linear``)."""
    ggml_type = ggml_type_id(ggml_type)
    shape = [int(s) for s in shape]
    _check(packed, ggml_type, shape, ("grad_out", grad_out))
    if grad_out.shape[-1] != shape[0]:
        raise ValueError(f"Unsloth: grad_out has {grad_out.shape[-1]} features, the GGUF weight has {shape[0]} rows")
    return _linear_bwd_op(grad_out, packed, ggml_type, shape)


def _grouped_impl(x, packed, ggml_type, shape, counts, trans_b):
    E, N, K = shape
    inner, outer = (K, N) if trans_b else (N, K)
    M = x.shape[0]
    out = torch.empty((M, outer), dtype = x.dtype, device = x.device)
    if M == 0:
        return out
    if _fused_ok(x, ggml_type) and x.dtype != torch.float32:
        try:
            with torch.cuda.device(x.device):
                return _mm_launch(x.contiguous(), packed, ggml_type, out, None, counts.to(torch.int32).contiguous(),
                                  M, outer, inner, E, N, trans_b, True)
        except Exception as err:
            if not _launch_ok(err):
                raise
            _warn_fallback("grouped GEMM", err)
    w = _dequantize_impl(packed, ggml_type, (E, N, K), x.dtype)
    ends = torch.cumsum(counts.to(torch.int64), 0).tolist()
    start = 0
    for e, end in enumerate(ends):
        if end > start:
            we = w[e]
            out[start:end] = x[start:end] @ (we.t() if trans_b else we)
        start = end
    return out


@torch.library.custom_op("unsloth_zoo::gguf_grouped_mm", mutates_args = ())
def _grouped_op(x: torch.Tensor, packed: torch.Tensor, ggml_type: int, shape: List[int], counts: torch.Tensor, transpose_b: bool) -> torch.Tensor:
    return _grouped_impl(x, packed, ggml_type, tuple(shape), counts, transpose_b)


@_grouped_op.register_fake
def _grouped_fake(x, packed, ggml_type, shape, counts, transpose_b):
    return x.new_empty((x.shape[0], shape[1] if transpose_b else shape[2]))


def _grouped_setup(ctx, inputs, output):
    x, packed, ggml_type, shape, counts, transpose_b = inputs
    ctx.save_for_backward(packed, counts)
    ctx.ggml_type, ctx.shape, ctx.transpose_b = ggml_type, list(shape), transpose_b


def _grouped_backward(ctx, grad):
    packed, counts = ctx.saved_tensors
    grad_x = None
    if ctx.needs_input_grad[0]:
        grad_x = _grouped_op(grad.contiguous(), packed, ctx.ggml_type, ctx.shape, counts, not ctx.transpose_b)
    return grad_x, None, None, None, None, None


torch.library.register_autograd("unsloth_zoo::gguf_grouped_mm", _grouped_backward, setup_context = _grouped_setup)


def gguf_grouped_mm(x: torch.Tensor, packed: torch.Tensor, ggml_type, shape, counts: torch.Tensor, transpose_b: bool = True) -> torch.Tensor:
    """Grouped GEMM on a packed GGUF expert stack of logical ``shape = (E, N, K)`` (packed ``(E, N, row_bytes)``).
    Rows of ``x`` are sorted by expert, ``counts[e]`` rows each. ``transpose_b``: ``x (M, K) @ W[e]^T -> (M, N)``;
    else ``x (M, N) @ W[e] -> (M, K)``. Differentiable in ``x``."""
    ggml_type = ggml_type_id(ggml_type)
    shape = [int(s) for s in shape]
    if len(shape) != 3:
        raise ValueError(f"Unsloth: gguf_grouped_mm needs an (E, N, K) shape, got {shape}")
    _check(packed, ggml_type, shape, ("x", x), ("counts", counts))
    if x.dim() != 2 or x.shape[1] != (shape[2] if transpose_b else shape[1]):
        raise ValueError(f"Unsloth: x of shape {tuple(x.shape)} does not match the GGUF expert stack {shape}")
    if counts.dim() != 1 or counts.shape[0] != shape[0]:
        raise ValueError(f"Unsloth: counts must be ({shape[0]},), got {tuple(counts.shape)}")
    return _grouped_op(x, packed, ggml_type, shape, counts, transpose_b)


def gguf_kernels_available(device = None) -> bool:
    """Whether the Triton GGUF kernels can run on ``device`` (else the torch reference is used)."""
    if not _HAS_TRITON or not torch.cuda.is_available():
        return False
    device = torch.device(device) if device is not None else torch.device("cuda", torch.cuda.current_device())
    return device.type == "cuda"
