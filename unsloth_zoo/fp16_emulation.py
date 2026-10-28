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

"""Split float16 emulation of float32 matmul; NOT enabled anywhere (set UNSLOTH_FP16_EMULATION=1 or call fp16_split_mm);
needs power-of-two scaling, falls back to float32 mm for non-finite or wide-range operands; only pays on large GEMMs;
kept for hardware without bfloat16 tensor cores."""
import itertools
import math
import os

import torch

from .utils import device_guard

__all__ = [
    "fp16_emulation_enabled",
    "pow2_scale",
    "pow2_exponent",
    "split_terms",
    "fp16_split_mm",
    "fp16_split_matmul",
]


def fp16_emulation_enabled() -> bool:
    """Opt-in switch. Default off; nothing in Unsloth consults this today."""
    return os.environ.get("UNSLOTH_FP16_EMULATION", "0") == "1"


_HAS_OUT_DTYPE = None


def _has_out_dtype() -> bool:
    """torch.mm out_dtype needs torch 2.8+; probed on CUDA only, since aten::mm.dtype is CUDA-only and a CPU
    NotImplementedError, a RuntimeError, would be cached as False forever."""
    global _HAS_OUT_DTYPE
    if _HAS_OUT_DTYPE is None:
        if not torch.cuda.is_available():
            _HAS_OUT_DTYPE = False
        else:
            try:
                zero = torch.zeros(1, 1, dtype = torch.float16, device = "cuda")
                torch.mm(zero, zero, out_dtype = torch.float32)
                _HAS_OUT_DTYPE = True
            except (TypeError, RuntimeError):
                _HAS_OUT_DTYPE = False
    return _HAS_OUT_DTYPE


def _mm_float32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Plain float32 matmul autocast cannot downcast; .float() alone is not enough, since torch.mm is autocast-eligible and would be re-cast to float16."""
    with torch.autocast(device_type = a.device.type, enabled = False):
        return torch.mm(a.float(), b.float())


def _mm_f32_accumulate(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Low-precision inputs with float32 accumulate, as a tensor core does; the float32 fallback is equivalent since the operands are already exact in low precision."""
    # Device check first: probing allocates a CUDA tensor and initialises a CUDA context, which CPU-only calls must not
    # pay, least of all forked workers that cannot touch CUDA.
    if a.is_cuda and _has_out_dtype():
        return torch.mm(a, b, out_dtype = torch.float32)
    return _mm_float32(a, b)


def pow2_scale(x: torch.Tensor, target_exp: int = 14) -> float:
    """Power-of-two scale putting max|x| near 2**target_exp (14 is an octave below float16's 65504); exact in binary
    floating point, but .item() syncs, so use pow2_exponent under torch.compile; tests assert on this one."""
    m = x.abs().max().item()
    if m == 0 or not math.isfinite(m):
        return 1.0
    return 2.0 ** (target_exp - math.floor(math.log2(m)))


def pow2_exponent(x: torch.Tensor, target_exp: int = 14) -> torch.Tensor:
    """Exponent n of the scale 2**n, from frexp on device (no .item() sync, so fullgraph compile holds); apply with
    torch.ldexp, since a materialised 2**n scale can overflow to inf; no tensor-valued scale helper exists."""
    if x.numel() == 0:
        return torch.zeros((), device = x.device, dtype = torch.int32)
    m = x.abs().max()
    _, exp = torch.frexp(m)
    n = target_exp - (exp - 1)
    return torch.where(torch.isfinite(m) & (m > 0), n, torch.zeros_like(n))


# Rounding must survive Inductor, which otherwise fuses x.to(float16).to(float32) into a no-op (residual 0, split
# silently degrades to plain float16); an opaque custom op forces a real round trip.
_HAS_CUSTOM_OP = hasattr(torch.library, "custom_op")

if _HAS_CUSTOM_OP:
    @torch.library.custom_op("unsloth_zoo::fp16_emulation_round", mutates_args = ())
    def _round_to(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        # Clone when dtype already matches: .to() returns the input itself, and a custom op with no aliasing may not
        # return an input; otherwise split_terms(x, x.dtype, ...) raises.
        return x.clone() if x.dtype == dtype else x.to(dtype)

    @_round_to.register_fake
    def _(x, dtype):
        return torch.empty_like(x, dtype = dtype)
else:
    def _round_to(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return x.to(dtype)


def split_terms(x: torch.Tensor, dtype: torch.dtype, terms: int) -> list:
    """Split a float32 tensor into `terms` low-precision terms, residual by residual."""
    out, residual = [], x
    for _ in range(terms):
        head = _round_to(residual, dtype)
        out.append(head)
        residual = residual - head.float()
    return out


# _SPLIT_RANGE_LIMIT = 2**28 is a catastrophe guard, not a precision guard: 2**17 fires on ordinary tensors
# (randn(256)*0.02 is 2**23.9); outputs depending only on far-below-max entries can carry large error; per-row or
# per-block scaling is the fix (a tighter global limit is not).
_SPLIT_RANGE_LIMIT = 2.0 ** 28


def _split_can_represent(x: torch.Tensor) -> bool:
    """Whether one per-tensor scale can carry this operand: False for non-finite entries (residual inf - inf gives NaN)
    or a within-tensor range wider than the split window; always True under torch.compile, where host reads break the
    graph."""
    if torch.compiler.is_compiling():
        return True
    absx = x.abs()
    hi = absx.max()
    if not bool(torch.isfinite(hi)):
        return False
    lo = torch.where(absx == 0, torch.full_like(absx, float("inf")), absx).min()
    if not bool(torch.isfinite(lo)):
        return True         # all zeros: nothing to lose
    return bool(hi / lo <= _SPLIT_RANGE_LIMIT)


def fp16_split_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    dtype: torch.dtype = torch.float16,
    terms: int = 2,
    products: int = 3,
    scale: bool = True,
) -> torch.Tensor:
    """Split-term float32 matmul returning float32; products keeps the most significant cross terms (2-term split: 3
    products drop lo*lo); on float16, scale must stay True for tensors with magnitudes below 0.125 (nearly all LLM
    weights)."""
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"fp16_split_mm expects 2D operands, got {A.shape} and {B.shape}")
    if terms < 1 or products < 1:
        raise ValueError(f"terms and products must be positive, got terms={terms} products={products}")
    if products > terms * terms:
        raise ValueError(f"products={products} exceeds terms*terms={terms * terms}")

    A32, B32 = A.float(), B.float()
    # torch.mm accepts zero-sized dimensions but the max() reduction in the scale does not, and an empty MoE expert partition is an ordinary outcome.
    if A32.numel() == 0 or B32.numel() == 0:
        return _mm_float32(A32, B32)

    if not _split_can_represent(A32) or not _split_can_represent(B32):
        return _mm_float32(A32, B32)

    with device_guard(A32):
        zero = torch.zeros((), device = A32.device, dtype = torch.int32)
        eA = pow2_exponent(A32) if scale else zero
        eB = pow2_exponent(B32) if scale else zero
        a_terms = split_terms(torch.ldexp(A32, eA), dtype, terms)
        b_terms = split_terms(torch.ldexp(B32, eB), dtype, terms)

        pairs = sorted(itertools.product(range(terms), repeat = 2), key = lambda ij: ij[0] + ij[1])
        acc = None
        for i, j in pairs[:products]:
            part = _mm_f32_accumulate(a_terms[i], b_terms[j])
            acc = part if acc is None else acc + part
        # Single ldexp with the combined exponent: multiplying two scales overflows for small operands, and dividing per
        # scale passes through inf for a large-times-small pair.
        return torch.ldexp(acc, -(eA + eB))


class _FP16SplitMatmul(torch.autograd.Function):
    """Autograd wrapper so the emulation stands in for a float32 matmul in training; both input gradients are emulated
    the same way, keeping backward at forward accuracy."""
    @staticmethod
    def forward(ctx, A, B, dtype, terms, products, scale):
        ctx.save_for_backward(A, B)
        ctx.config = (dtype, terms, products, scale)
        return fp16_split_mm(A, B, dtype, terms, products, scale)

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        dtype, terms, products, scale = ctx.config
        grad_A = grad_B = None
        if ctx.needs_input_grad[0]:
            grad_A = fp16_split_mm(
                grad_output.contiguous(), B.t().contiguous(), dtype, terms, products, scale,
            ).to(A.dtype)
        if ctx.needs_input_grad[1]:
            grad_B = fp16_split_mm(
                A.t().contiguous(), grad_output.contiguous(), dtype, terms, products, scale,
            ).to(B.dtype)
        return grad_A, grad_B, None, None, None, None
    pass
pass


def fp16_split_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    dtype: torch.dtype = torch.float16,
    terms: int = 2,
    products: int = 3,
    scale: bool = True,
) -> torch.Tensor:
    """Differentiable form of `fp16_split_mm`."""
    return _FP16SplitMatmul.apply(A, B, dtype, terms, products, scale)
