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

"""Emulate float32 matmul accuracy using split float16 terms.

NOT ENABLED ANYWHERE. Nothing in Unsloth calls this module, and importing it changes no
behaviour. It is kept because it is correct and measured, so a future workload that needs
float32-accuracy matmuls on hardware without bfloat16 tensor cores can switch it on rather
than rediscover it. Enable explicitly with UNSLOTH_FP16_EMULATION=1, or just call
`fp16_split_mm` directly.

Background. pytorch/pytorch#195301 added `fp32_precision="bfx9"`: split each float32
operand into three bfloat16 terms and accumulate nine bfloat16 tensor-core products in
float32. That works because bfloat16 has 8 significand bits (3 terms covers float32's 24)
AND 8 exponent bits, the same range as float32, so the split is lossless across the whole
dynamic range.

Porting it to float16 is not a relabel, because float16 has 11 significand bits but only 5
exponent bits:

  - 2 terms give about 22 bits, 2 short of float32. 3 terms give 33, which is wasteful.
  - Normal float16 spans [6.104e-5, 65504]. In a 2-term split the low term is about 2^-11
    of the high term, so both stay normal only for |x| in [0.125, 65504], roughly 19 binary
    decades against float32's 277.

Typical LLM weights and activations live at 1e-3 to 1e-1, which is BELOW that floor, so the
low term underflows to subnormal or zero and the split silently degrades to plain float16
while still costing three matmuls. Power-of-two per-tensor scaling is therefore not an
optimisation here, it is load-bearing. `scale = False` exists so the failure can be shown
rather than asserted.

Measured, B200, median relative L2 against a float64 reference over a magnitude sweep:

    method                    1e-6      1e-3       0.1       1.0       1e3
    float32 IEEE            2.12e-7   2.11e-7   2.11e-7   2.11e-7   2.12e-7
    fp16x2 3-product scaled 2.16e-7   2.17e-7   2.17e-7   2.17e-7   2.17e-7
    fp16x2 3-product plain  2.44e-2   2.43e-5   3.22e-7   2.18e-7   2.17e-7
    float16 direct          2.44e-2   2.93e-4   2.94e-4   2.93e-4   2.94e-4

Scaled, it holds IEEE parity across six orders of magnitude, including on an outlier-heavy
distribution. Unscaled at 1e-6 it is bit-identical to plain float16: the low term has fully
underflowed and the split has ceased to exist. The rig was validated with a bf16x3 9-product
control at 0.9x IEEE, reproducing the relationship the upstream PR published.

Why it is off. On a real T4 (Kaggle T4x2, torch 2.10+cu128, sm75) the emulation only pays on
large square GEMMs, and loses badly on the small ones that dominate a QLoRA step:

    shape          float32    float16      fp16x2 3-product
    square-4096    33.54 ms   6.74 ms      27.13 ms   (1.24x vs float32)
    square-2048     4.40 ms   0.91 ms       4.69 ms   (0.94x)
    gemma3-QK^T     0.46 ms   0.19 ms       1.33 ms   (0.35x)
    gemma3-PV       0.41 ms   0.23 ms       1.73 ms   (0.24x)

and in a stock T4 QLoRA step every GEMM is already float16-in, so there is nothing to
accelerate: NF4 dequant plus base projection, the LoRA addmm, the lm_head, and attention.
The one genuine float32 target, Gemma3/Gemma4 attention under UNSLOTH_FORCE_FLOAT32, is
exactly the small-shape regime where this is 3-4x SLOWER than plain float32. Accuracy on the
T4 itself measured 9.06e-7 against IEEE 2.61e-7, i.e. 3.5x IEEE rather than the 1.0x seen on
B200, still 324x better than plain float16. That B200/T4 gap is unexplained; plausibly
different tensor-core accumulate behaviour, but that is a guess and is recorded as one.

torch.compile. `fp16_split_mm` compiles fullgraph and compiling it is worth 1.5-2.2x (B200,
torch 2.14), which turns square-4096 from 1.24x into 3.25x against float32. Two things had to
change to get there, both load-bearing:

  - the scale must not call `.item()`, which syncs and breaks fullgraph. `pow2_exponent`
    uses frexp instead and is applied with ldexp.
  - the float16 rounding must go through an opaque custom op. Inductor otherwise fuses
    `.to(float16).to(float32)` away, leaving a residual of exactly 0, so the split silently
    stops existing and lands at plain-float16 error while running faster. Guarded by
    test_compiled_keeps_float32_accuracy.

Degenerate operands fall back to plain float32 mm rather than returning something wrong: a
non-finite entry, whose residual would compute inf - inf and NaN the result, and a within-tensor
dynamic range wider than float16's normal window, which one per-tensor scale cannot hold and
which would round the small entries to zero. The check reads tensor values, so it is skipped
under torch.compile, where a host read breaks the graph and the operands are the ordinary finite
ones anyway. Extreme magnitudes need no fallback: the scaling is an exponent applied with ldexp,
so nothing is materialised that could overflow.

So: numerically sound, not currently worth enabling. Kept for the case where a large
float32 GEMM appears on hardware without bfloat16 tensor cores.
"""
from __future__ import annotations

import itertools
import math
import os

import torch

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
    """torch.mm(..., out_dtype = ) landed in torch 2.8. Probed once, not assumed.

    Probed on CUDA, because `aten::mm.dtype` is CUDA-only: a CPU probe raises
    NotImplementedError, which subclasses RuntimeError and so would be caught below and cache
    False forever, silently costing the tensor-core path this module exists for.
    """
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
    """Plain float32 matmul that autocast cannot downcast.

    `.float()` on the operands is not enough: torch.mm is autocast-eligible, so inside an
    autocast region it is re-cast to float16 and the result stops being float32-accumulated.
    """
    with torch.autocast(device_type = a.device.type, enabled = False):
        return torch.mm(a.float(), b.float())


def _mm_f32_accumulate(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Low-precision inputs with float32 accumulate, which is what a tensor core does.

    The fallback is numerically equivalent for this use because the operands are already
    exactly representable in the low precision, so widening them loses nothing.
    """
    # Device first: probing allocates a CUDA tensor and initialises a CUDA context, which a
    # CPU-only call must not pay for, least of all in a forked worker that cannot touch CUDA.
    if a.is_cuda and _has_out_dtype():
        return torch.mm(a, b, out_dtype = torch.float32)
    return _mm_float32(a, b)


def pow2_scale(x: torch.Tensor, target_exp: int = 14) -> float:
    """Power-of-two scale bringing max|x| to about 2**target_exp.

    A power of two is exact in binary floating point, so the scaling contributes no error of
    its own. That matters when the entire purpose is to measure error. target_exp = 14 puts
    the maximum an octave below the float16 ceiling of 65504.

    Returns a Python float, so it syncs. `pow2_exponent` is the torch.compile-safe form.
    This one is kept because it is the readable definition and is what the tests assert on.
    """
    m = x.abs().max().item()
    if m == 0 or not math.isfinite(m):
        return 1.0
    return 2.0 ** (target_exp - math.floor(math.log2(m)))


def pow2_exponent(x: torch.Tensor, target_exp: int = 14) -> torch.Tensor:
    """The exponent n such that the scale is 2**n.

    There is deliberately no tensor-valued *scale* helper. One existed and was removed: a 0-dim
    tensor follows scalar promotion, so multiplying a float32 operand by a float64 scale stays
    in float32, and at max|x| near 2**-114 the scale of 2**128 overflows to inf on the way in
    however the scale itself is stored. `torch.ldexp(x, pow2_exponent(x))` has no such edge and
    is what this module uses throughout: it cannot overflow on the way to a representable
    answer, in either direction. For a large-times-small pair like 2**114 @ 2**-113, which is
    just 2.0, dividing by one materialised scale before the other passes through inf.

    frexp gives the exponent on device, avoiding the `.item()` host read that would sync and
    break torch.compile(fullgraph = True). frexp returns m = mantissa * 2**exp with mantissa in
    [0.5, 1), so floor(log2(m)) == exp - 1.
    """
    if x.numel() == 0:
        return torch.zeros((), device = x.device, dtype = torch.int32)
    m = x.abs().max()
    _, exp = torch.frexp(m)
    n = target_exp - (exp - 1)
    return torch.where(torch.isfinite(m) & (m > 0), n, torch.zeros_like(n))


# Rounding to low precision has to survive Inductor, and by default it does not: it fuses
# `x.to(float16).to(float32)` into a no-op, keeping the value in a float32 register and never
# rounding. The residual is then exactly 0, the low term vanishes and the split silently
# degrades to a plain float16 matmul, measured at 2.93e-04 against eager's 4.86e-06 while
# looking 5.8x faster. An opaque custom op is the barrier that forces a real round trip.
_HAS_CUSTOM_OP = hasattr(torch.library, "custom_op")

if _HAS_CUSTOM_OP:
    @torch.library.custom_op("unsloth_zoo::fp16_emulation_round", mutates_args = ())
    def _round_to(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        # Cloned when the dtype already matches: .to() returns the input itself there, and a
        # custom op declaring no aliasing may not return one of its inputs. Without this,
        # split_terms(x, x.dtype, ...) raises instead of splitting.
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


# hi/lo window beyond which the smallest entries lose their second term entirely and the
# split degrades to plain float16 for them.
#
# This is deliberately a catastrophe guard, not a precision guard. Reserving the ~11 bits the
# residual needs would put the limit at 2**17, but ordinary tensors exceed that routinely --
# randn(256) * 0.02 measures 2**23.9, randn(4096) * 0.02 measures 2**24.4 -- because a matrix
# only needs one near-zero entry to blow up hi/lo. At 2**17 the guard fires on the exact
# workload this module is for and silently replaces the emulation with float32 mm.
#
# What is left is a real limitation of per-tensor scaling, recorded rather than papered over:
# an output element that depends only on entries far below the tensor maximum can carry much
# larger relative error than the aggregate figures in the docstring. diag([a, b*2**-27])
# against its reciprocal returns 0.999844 rather than 1.0. Per-row or per-block scaling is the
# fix if that case ever matters; a tighter global limit is not.
_SPLIT_RANGE_LIMIT = 2.0 ** 28


def _split_can_represent(x: torch.Tensor) -> bool:
    """Whether one per-tensor power-of-two scale can actually carry this operand.

    Two ways the split silently returns something wrong, both of which plain float32 mm gets
    right: a non-finite entry, whose residual computes inf - inf and poisons everything to NaN;
    and a within-tensor dynamic range wider than the window above, which leaves the small
    entries' residuals subnormal or zero. A single scale taken from the maximum cannot serve
    the second case at all.

    Reads tensor values, so it syncs and is skipped under torch.compile, where a host read
    breaks the graph. The compiled path is for the ordinary finite, normal-range operands the
    module is meant for.
    """
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
    """Split-term emulation of a float32 matmul. Returns float32.

    `products` keeps only the most significant cross terms: for a 2-term split, 3 products
    means hi*hi + hi*lo + lo*hi and drops the negligible lo*lo. Terms are ordered by
    term-index sum, which tracks magnitude.

    On float16, `scale` must stay True for any tensor whose magnitudes fall below 0.125,
    which is nearly all LLM weights and activations. See the module docstring.
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"fp16_split_mm expects 2D operands, got {A.shape} and {B.shape}")
    if terms < 1 or products < 1:
        raise ValueError(f"terms and products must be positive, got terms={terms} products={products}")
    if products > terms * terms:
        raise ValueError(f"products={products} exceeds terms*terms={terms * terms}")

    A32, B32 = A.float(), B.float()
    # torch.mm accepts a zero-sized dimension; the max() reduction in the scale does not, and
    # an empty expert partition is an ordinary outcome of MoE routing.
    if A32.numel() == 0 or B32.numel() == 0:
        return _mm_float32(A32, B32)

    if not _split_can_represent(A32) or not _split_can_represent(B32):
        return _mm_float32(A32, B32)

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
    # One ldexp with the combined exponent. Neither sA * sB nor a division per scale works: the
    # product overflows for two small operands, and dividing in either order passes through inf
    # for a large-times-small pair whose answer is perfectly representable.
    return torch.ldexp(acc, -(eA + eB))


class _FP16SplitMatmul(torch.autograd.Function):
    """Autograd wrapper so the emulation can stand in for a float32 matmul in training.

    Both input gradients are emulated the same way, keeping the backward at the same
    accuracy as the forward rather than silently dropping to plain float16.
    """
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
