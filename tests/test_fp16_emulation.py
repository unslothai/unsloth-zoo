# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
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
    split_terms,
)


def _rel_l2(actual, reference):
    diff = actual.double() - reference
    denom = torch.linalg.vector_norm(reference)
    return (torch.linalg.vector_norm(diff) / denom).item()


def _reference(A, B):
    return torch.mm(A.double(), B.double())


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
