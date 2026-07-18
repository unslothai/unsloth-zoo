# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
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

"""Regression tests for preprocess_weight layout inference (unslothai/unsloth-zoo#849).

MoE gate_up experts are stored F.linear (E, 2I, H) and must become grouped_mm layout
(E, H, 2I); down is (E, H, I) -> (E, I, H). The layout was inferred from shape alone,
which is ambiguous when the two matmul dims are equal (2*moe_intermediate == hidden_size
for gate_up, or moe_intermediate == hidden_size for down): the tensor is square, the check
guessed "already correct", and every expert silently trained on transposed weights.

These tests drive that square branch and assert preprocess_weight recovers the correct
grouped_mm layout by voting with the guaranteed-non-square sibling projection. CPU-only,
no CUDA — the reporter proved UNSLOTH_MOE_BACKEND=native_torch is bit-exact on CPU, so an
exact-equality check against the reference grouped layout is a faithful oracle.
"""

import torch
import pytest

from unsloth_zoo.temporary_patches.moe_utils import (
    preprocess_weight,
    _orientation_needs_transpose,
    _logical_expert_shape,
)


class _FakeExperts:
    """Minimal stand-in for a transformers MoE experts module: exposes the two fused
    expert weights as plain tensors, which is all the sibling-vote needs to read a shape."""

    def __init__(self, gate_up_proj=None, down_proj=None, num_experts=4):
        if gate_up_proj is not None:
            self.gate_up_proj = gate_up_proj
        if down_proj is not None:
            self.down_proj = down_proj
        self.num_experts = num_experts


def _make_flinear_experts(num_experts, hidden, inter, seed=0):
    """Build reference F.linear-layout expert weights and their correct grouped_mm targets.

    F.linear:  gate_up (E, 2I, H), down (E, H, I)
    grouped_mm: gate_up (E, H, 2I), down (E, I, H)
    """
    g = torch.Generator().manual_seed(seed)
    gate_up_flinear = torch.randn(num_experts, 2 * inter, hidden, generator=g)
    down_flinear = torch.randn(num_experts, hidden, inter, generator=g)
    gate_up_grouped = gate_up_flinear.transpose(-2, -1).contiguous()
    down_grouped = down_flinear.transpose(-2, -1).contiguous()
    return gate_up_flinear, down_flinear, gate_up_grouped, down_grouped


# --- 1. Non-square control: fast path must be byte-identical to the historical behaviour ---

def test_nonsquare_gate_up_and_down_unchanged():
    E, H, I = 4, 64, 16  # 2I=32 != H=64, and I=16 != H=64 -> both non-square
    gu_f, dn_f, gu_g, dn_g = _make_flinear_experts(E, H, I)
    experts = _FakeExperts(gu_f, dn_f)

    out_gu = preprocess_weight(gu_f, "gate_up", H, experts_module=experts)
    out_dn = preprocess_weight(dn_f, "down", H, experts_module=experts)
    assert torch.equal(out_gu, gu_g)
    assert torch.equal(out_dn, dn_g)

    # Non-square inputs already in grouped layout are returned untouched (idempotent).
    assert torch.equal(preprocess_weight(gu_g, "gate_up", H, experts_module=experts), gu_g)
    assert torch.equal(preprocess_weight(dn_g, "down", H, experts_module=experts), dn_g)


# --- 2. Square gate_up (2I == H): the #849 bug. Sibling `down` is non-square and decides. ---

def test_square_gate_up_transposed_via_down_sibling():
    E, H, I = 4, 64, 32  # 2I == H == 64 -> gate_up is square; down (64, 32) is not
    gu_f, dn_f, gu_g, dn_g = _make_flinear_experts(E, H, I)
    assert gu_f.shape[1] == gu_f.shape[2]  # confirm we're driving the ambiguous branch
    experts = _FakeExperts(gu_f, dn_f)

    out_gu = preprocess_weight(gu_f, "gate_up", H, experts_module=experts)
    assert torch.equal(out_gu, gu_g), "square gate_up must be transposed to grouped layout"
    # And already-grouped square input stays put (sibling reports grouped orientation).
    experts_g = _FakeExperts(gu_g, dn_g)
    assert torch.equal(preprocess_weight(gu_g, "gate_up", H, experts_module=experts_g), gu_g)


# --- 3. Square down (I == H): sibling `gate_up` is non-square and decides. ---

def test_square_down_transposed_via_gate_up_sibling():
    E, H, I = 4, 64, 64  # I == H == 64 -> down is square; gate_up (128, 64) is not
    gu_f, dn_f, gu_g, dn_g = _make_flinear_experts(E, H, I)
    assert dn_f.shape[1] == dn_f.shape[2]  # confirm ambiguous branch
    experts = _FakeExperts(gu_f, dn_f)

    out_dn = preprocess_weight(dn_f, "down", H, experts_module=experts)
    assert torch.equal(out_dn, dn_g), "square down must be transposed to grouped layout"
    experts_g = _FakeExperts(gu_g, dn_g)
    assert torch.equal(preprocess_weight(dn_g, "down", H, experts_module=experts_g), dn_g)


# --- 4. Square input, no sibling reachable: warn + transpose (F.linear default). ---

def test_square_without_module_warns_and_transposes(capsys):
    E, H, I = 4, 64, 32
    gu_f, _dn_f, gu_g, _dn_g = _make_flinear_experts(E, H, I)

    out = preprocess_weight(gu_f, "gate_up", H)  # no experts_module
    assert torch.equal(out, gu_g)
    err = capsys.readouterr().err
    assert "#849" in err and "ambiguous" in err.lower()
    assert "register_weight_preprocessor" in err  # tells the user how to remove the guess


# --- 5. Escape hatch still wins: a registered preprocessor bypasses shape inference entirely. ---

def test_registered_preprocessor_takes_precedence():
    from unsloth_zoo.temporary_patches import moe_utils

    sentinel = torch.zeros(1)
    key = "unit_test_dummy_arch"
    moe_utils.register_weight_preprocessor(key, lambda w, p, h: sentinel)
    try:
        out = preprocess_weight(torch.randn(4, 64, 64), "gate_up", 64, model_type=key)
        assert out is sentinel
    finally:
        moe_utils._WEIGHT_PREPROCESSORS.pop(key, None)


# --- Unit coverage for the orientation primitive and the cheap shape reader. ---

def test_orientation_primitive_returns_none_on_square():
    assert _orientation_needs_transpose((4, 64, 64), "gate_up", 64) is None
    assert _orientation_needs_transpose((4, 64, 64), "down", 64) is None
    # F.linear (transpose) vs grouped (keep), non-square:
    assert _orientation_needs_transpose((4, 32, 64), "gate_up", 64) is True   # (E, 2I, H)
    assert _orientation_needs_transpose((4, 64, 32), "gate_up", 64) is False  # (E, H, 2I)
    assert _orientation_needs_transpose((4, 64, 16), "down", 64) is True      # (E, H, I)
    assert _orientation_needs_transpose((4, 16, 64), "down", 64) is False     # (E, I, H)


def test_logical_expert_shape_prefers_original_shape():
    class _PackedParam:
        # bnb Params4bit-like: .shape is packed, _original_shape is logical.
        shape = (2048, 1)
        _original_shape = (4, 128, 64)

    assert _logical_expert_shape(_PackedParam()) == (4, 128, 64)
    assert _logical_expert_shape(torch.randn(4, 128, 64)) == (4, 128, 64)
