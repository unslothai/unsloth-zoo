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

"""Tests for the dormant float16 matmul emulation.

The important test here is not that the scaled split is accurate, it is that the UNSCALED
split collapses at small magnitudes. float16's 5-bit exponent is the whole difficulty, and a
split implementation that quietly stopped splitting would still pass an accuracy test run at
magnitude 1.0. test_unscaled_underflows_at_small_magnitude fails if the cliff is absent.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unsloth_zoo.fp16_emulation import (  # noqa: E402
    fp16_emulation_enabled,
    fp16_split_matmul,
    fp16_split_mm,
    pow2_scale,
    pow2_exponent,
    split_terms,
)


def _rel_l2(actual, reference):
    diff = actual.double() - reference
    denom = torch.linalg.vector_norm(reference)
    return (torch.linalg.vector_norm(diff) / denom).item()


def _reference(A, B):
    return torch.mm(A.double(), B.double())


def _pair(m, k, n, mag = 0.02, device = "cuda"):
    """Operand pair at a magnitude typical of LLM weights, which is below float16's
    normal-range floor of 0.125 and therefore depends on the scaling to work at all."""
    torch.manual_seed(3407)
    return (torch.randn(m, k, device = device) * mag,
            torch.randn(k, n, device = device) * mag)


def _fp16_direct(A, B):
    """float16 operands with float32 accumulate, i.e. what a tensor core actually does.

    Not torch.mm(A.half(), B.half()), which also accumulates in float16 and is a different
    and much worse baseline. The question here is the precision of the stored operands.
    """
    return torch.mm(A.half().float(), B.half().float())


def test_disabled_by_default():
    """Nothing in Unsloth enables this, and importing it must not turn anything on."""
    os.environ.pop("UNSLOTH_FP16_EMULATION", None)
    assert fp16_emulation_enabled() is False
    os.environ["UNSLOTH_FP16_EMULATION"] = "1"
    try:
        assert fp16_emulation_enabled() is True
    finally:
        os.environ.pop("UNSLOTH_FP16_EMULATION", None)


def test_pow2_scale_is_exact():
    """The scale must be a power of two, otherwise it injects error into an error measurement."""
    for mag in (1e-6, 1e-3, 1.0, 1e3):
        x = torch.randn(64, 64) * mag
        s = pow2_scale(x)
        assert s > 0 and torch.tensor(s).log2().item().is_integer()
        # Scaling by a power of two is lossless, so scaling and unscaling round-trips exactly.
        assert torch.equal((x * s) / s, x)


def test_pow2_scale_handles_degenerate_input():
    assert pow2_scale(torch.zeros(8, 8)) == 1.0
    assert pow2_scale(torch.full((8, 8), float("inf"))) == 1.0


def test_split_terms_reconstructs():
    x = torch.randn(128, 128)
    terms = split_terms(x, torch.float16, 3)
    assert len(terms) == 3
    total = sum(t.float() for t in terms)
    # Three float16 terms carry ~33 significand bits, comfortably more than float32's 24.
    assert _rel_l2(total, x.double()) < 1e-6


@pytest.mark.parametrize("magnitude", [1e-6, 1e-3, 0.1, 1.0, 1e3])
def test_scaled_split_holds_ieee_parity(magnitude):
    """Scaled, the split should track IEEE float32 across the whole sweep."""
    torch.manual_seed(3407)
    A = torch.randn(256, 256) * magnitude
    B = torch.randn(256, 256) * magnitude
    ref = _reference(A, B)

    ieee = _rel_l2(torch.mm(A, B), ref)
    split = _rel_l2(fp16_split_mm(A, B, scale = True), ref)
    direct = _rel_l2(_fp16_direct(A, B), ref)

    # Within 2x of IEEE, and far better than plain float16.
    assert split < ieee * 2.0, f"scaled split {split:.3e} vs ieee {ieee:.3e}"
    assert split < direct / 10.0, f"scaled split {split:.3e} vs float16 {direct:.3e}"


def test_unscaled_underflows_at_small_magnitude():
    """The cliff below |x| = 0.125 must be reproducible.

    If this passes trivially, the split is not actually splitting. At 1e-6 the low term is
    entirely subnormal or zero, so the unscaled split should be no better than plain float16,
    while the scaled split stays at IEEE parity.
    """
    torch.manual_seed(3407)
    A = torch.randn(256, 256) * 1e-6
    B = torch.randn(256, 256) * 1e-6
    ref = _reference(A, B)

    unscaled = _rel_l2(fp16_split_mm(A, B, scale = False), ref)
    scaled = _rel_l2(fp16_split_mm(A, B, scale = True), ref)
    direct = _rel_l2(_fp16_direct(A, B), ref)

    assert unscaled > 1e-3, f"expected the unscaled split to collapse, got {unscaled:.3e}"
    assert unscaled == pytest.approx(direct, rel = 0.5), (
        f"unscaled {unscaled:.3e} should match plain float16 {direct:.3e}: "
        f"the low term has underflowed and the split no longer exists"
    )
    assert scaled < unscaled / 100.0


def test_outlier_distribution():
    """The LLM-realistic case: mostly small values with a few large ones."""
    torch.manual_seed(3407)
    def outlier(m, n):
        x = torch.randn(m, n) * 0.02
        return torch.where(torch.rand(m, n) < 0.001, x * 500.0, x)
    A, B = outlier(256, 256), outlier(256, 256)
    ref = _reference(A, B)
    assert _rel_l2(fp16_split_mm(A, B), ref) < _rel_l2(torch.mm(A, B), ref) * 3.0


def test_rejects_bad_shapes_and_products():
    with pytest.raises(ValueError):
        fp16_split_mm(torch.randn(2, 3, 4), torch.randn(4, 4))
    with pytest.raises(ValueError):
        fp16_split_mm(torch.randn(4, 4), torch.randn(4, 4), terms = 2, products = 5)


def test_autograd_matches_float32_matmul():
    """Gradients must be emulated too, not silently dropped to plain float16."""
    torch.manual_seed(3407)
    A = (torch.randn(64, 96) * 0.02).requires_grad_(True)
    B = (torch.randn(96, 32) * 0.02).requires_grad_(True)
    A_ref = A.detach().clone().requires_grad_(True)
    B_ref = B.detach().clone().requires_grad_(True)

    fp16_split_matmul(A, B).sum().backward()
    torch.mm(A_ref, B_ref).sum().backward()

    assert torch.allclose(A.grad, A_ref.grad, rtol = 1e-3, atol = 1e-6)
    assert torch.allclose(B.grad, B_ref.grad, rtol = 1e-3, atol = 1e-6)


def test_bf16_control_reproduces_upstream_relationship():
    """bf16x3 9-product needs no scaling, because bfloat16 has float32's exponent range.

    This is the control that validates the rig: it should track IEEE closely even unscaled,
    which is the property that makes the upstream bfloat16 scheme work and the float16 port
    require scaling.
    """
    torch.manual_seed(3407)
    A = torch.randn(256, 256) * 1e-6
    B = torch.randn(256, 256) * 1e-6
    ref = _reference(A, B)
    bf16x3 = _rel_l2(
        fp16_split_mm(A, B, dtype = torch.bfloat16, terms = 3, products = 9, scale = False), ref
    )
    assert bf16x3 < _rel_l2(torch.mm(A, B), ref) * 2.0


# ---------------------------------------------------------------- torch.compile

@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_compiles_fullgraph():
    """fullgraph is the real bar: it proves no host sync survives in the hot path."""
    A, B = _pair(512, 512, 512, mag = 0.02)
    out = torch.compile(fp16_split_mm, fullgraph = True)(A, B)
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_compiled_keeps_float32_accuracy():
    """The regression that matters, and the reason _round_to is a custom op.

    Without the opaque-op barrier Inductor fuses the float16 round trip away, the residual
    becomes exactly 0, and this lands at plain-float16 error (~2.9e-04) while running
    faster. Compiled output must stay at float32 parity, not merely be finite.
    """
    A, B = _pair(2048, 2048, 2048, mag = 0.02)
    ref = A.double() @ B.double()
    rel = lambda t: ((t.double() - ref).norm() / ref.norm()).item()
    fp32_err = rel(torch.mm(A, B))
    compiled_err = rel(torch.compile(fp16_split_mm)(A, B))
    fp16_err = rel(torch.mm(A.half(), B.half()).float())
    assert compiled_err < 10 * fp32_err, f"compiled {compiled_err:.2e} vs fp32 {fp32_err:.2e}"
    assert compiled_err < fp16_err / 50, f"compiled {compiled_err:.2e} near fp16 {fp16_err:.2e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_round_to_actually_rounds_under_compile():
    """Directly pin the mechanism: a non-zero residual is what the split IS."""
    def split(x):
        return split_terms(x, torch.float16, 2)[1]
    x = torch.randn(1024, 1024, device = "cuda") * 0.02
    head = split_terms(x, torch.float16, 2)[0]
    residual = x - head.float()
    assert residual.norm().item() > 0
    assert torch.compile(lambda t: t - split_terms(t, torch.float16, 2)[0].float())(x) \
        .norm().item() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_pow2_exponent_applies_cleanly_at_extremes():
    """A materialised scale is unusable here however it is stored: 2**128 overflows float32,
    and a 0-dim float64 scale still promotes down to float32 when applied. ldexp does not."""
    x = torch.tensor([2.0 ** -114])
    n = pow2_exponent(x)
    assert 2.0 ** n.item() == pow2_scale(x)
    assert torch.ldexp(x, n).item() == 2.0 ** 14


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_round_to_does_not_alias_when_dtype_matches():
    """.to(dtype) returns the input itself when the dtype already matches, and a custom op
    declaring no aliasing may not return one of its inputs."""
    A = torch.randn(8, 8, device = "cuda")
    terms = split_terms(A, torch.float32, 2)
    assert terms[0] is not A
    assert torch.equal(terms[0], A)
    fp16_split_mm(A, A, dtype = torch.float32)      # used to raise an aliasing error


def test_pow2_exponent_matches_pow2_scale():
    """The sync-free form must give the same number, not merely a similar one. Runs on CPU:
    it is device-independent, and the CPU gate executes this file."""
    for mag in (1e-6, 1e-3, 0.1, 1.0, 1e3):
        x = torch.randn(256, 256) * mag
        assert 2.0 ** pow2_exponent(x).item() == pow2_scale(x)
    zero = torch.zeros(8, 8)
    assert 2.0 ** pow2_exponent(zero).item() == pow2_scale(zero) == 1.0


# ---------------------------------------------------------------- degenerate operands
#
# Each of these returned something silently wrong before: a NaN, a zero, or an exception,
# where torch.mm returns the right answer. The emulation advertises itself as a float32
# matmul stand-in, so "wrong but fast" is the one outcome it must not have.

@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_out_dtype_probe_matches_a_direct_call():
    """aten::mm.dtype is CUDA-only. Probing it on CPU raises NotImplementedError, which
    subclasses RuntimeError, so a CPU probe would cache False and disable the tensor-core
    path the module exists for.

    Asserted against a direct attempt rather than against True: pyproject allows torch>=2.4,
    and out_dtype only landed in 2.8, so False is the correct answer on a supported older
    install and this must not fail there.
    """
    from unsloth_zoo.fp16_emulation import _has_out_dtype
    zero = torch.zeros(1, 1, dtype = torch.float16, device = "cuda")
    try:
        torch.mm(zero, zero, out_dtype = torch.float32)
        supported = True
    except (TypeError, RuntimeError):
        supported = False
    assert _has_out_dtype() is supported


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_infinite_operand_matches_torch_mm():
    A = torch.tensor([[float("inf")]], device = "cuda")
    B = torch.tensor([[1.0]], device = "cuda")
    assert torch.isinf(fp16_split_mm(A, B)).all()      # was NaN: inf - inf in the residual


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_scale_that_would_overflow_float32():
    """max|x| below about 2**-113 makes the scale itself larger than float32 can hold."""
    A = torch.full((2, 2), 2.0 ** -114, device = "cuda")
    B = torch.full((2, 2), 2.0 ** 114, device = "cuda")
    assert torch.allclose(fp16_split_mm(A, B).double(), A.double() @ B.double(), rtol = 1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_scale_product_overflows_but_operands_do_not():
    """Each scale is representable at 2**69; only sA * sB overflows, so the division is
    done one scale at a time."""
    A = torch.tensor([[2.0 ** -55]], device = "cuda")
    B = torch.tensor([[2.0 ** -55]], device = "cuda")
    assert torch.allclose(fp16_split_mm(A, B).double(), A.double() @ B.double(), rtol = 1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_large_times_small_does_not_overflow_intermediately():
    """2**114 @ 2**-113 is 2.0. Applying the inverse scales one at a time passes through inf
    in either order, so the combined exponent goes on in a single ldexp."""
    A = torch.tensor([[2.0 ** 114]], device = "cuda")
    B = torch.tensor([[2.0 ** -113]], device = "cuda")
    assert torch.allclose(fp16_split_mm(A, B).double(), A.double() @ B.double(), rtol = 1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_ordinary_tensors_are_not_diverted_to_the_fallback():
    """The dynamic-range guard is a catastrophe guard. A tighter bound would fire on ordinary
    randn operands, whose hi/lo runs past 2**23 on any near-zero entry, and would quietly
    replace the emulation with float32 mm on the workload it exists for."""
    from unsloth_zoo.fp16_emulation import _split_can_represent
    A, B = _pair(1024, 1024, 1024, mag = 0.02)
    assert _split_can_represent(A) and _split_can_represent(B)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_within_tensor_range_wider_than_float16():
    """One per-tensor scale cannot hold a range this wide, so the small entries would round
    to zero. Falls back to float32 rather than returning an all-zero matrix."""
    A = torch.diag(torch.tensor([1.0, 2.0 ** -39], device = "cuda"))
    B = torch.diag(torch.tensor([1.0, 2.0 ** 39], device = "cuda"))
    assert torch.allclose(fp16_split_mm(A, B).diagonal(), torch.ones(2, device = "cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_empty_dimension_matches_torch_mm():
    """MoE routing hands out empty expert partitions; torch.mm accepts them, max() does not."""
    A = torch.zeros(0, 8, device = "cuda")
    B = torch.zeros(8, 4, device = "cuda")
    assert fp16_split_mm(A, B).shape == (0, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_float32_accumulate_survives_autocast():
    """torch.mm is autocast-eligible, so .float() operands alone do not keep the fallback in
    float32 inside an autocast region."""
    A, B = _pair(512, 512, 512, mag = 0.02)
    ref = A.double() @ B.double()
    with torch.autocast(device_type = "cuda", dtype = torch.float16):
        out = fp16_split_mm(A, B)
    assert out.dtype == torch.float32
    rel = ((out.double() - ref).norm() / ref.norm()).item()
    assert rel < 1e-5, f"autocast degraded the emulation to {rel:.2e}"


@pytest.mark.parametrize("kwargs", [{"terms": 0}, {"products": 0}, {"products": -1}])
def test_non_positive_counts_rejected(kwargs):
    """products=-1 used to slice the full product list and silently run a different scheme."""
    A, B = _pair(4, 4, 4, device = "cpu")
    with pytest.raises(ValueError):
        fp16_split_mm(A, B, **kwargs)
