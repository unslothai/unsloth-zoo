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

"""Regression tests for MoE expert weight layout inference (#849)."""

import torch
import pytest

from unsloth_zoo.temporary_patches.moe_utils import (
    preprocess_weight,
    _orientation_needs_transpose,
    _logical_expert_shape,
)


class _FakeExperts:
    """Minimal fused experts module."""

    def __init__(self, gate_up_proj=None, down_proj=None, num_experts=4):
        if gate_up_proj is not None:
            self.gate_up_proj = gate_up_proj
        if down_proj is not None:
            self.down_proj = down_proj
        self.num_experts = num_experts


def _make_flinear_experts(num_experts, hidden, inter, seed=0):
    """Return F.linear weights and grouped_mm transposes."""
    g = torch.Generator().manual_seed(seed)
    gate_up_flinear = torch.randn(num_experts, 2 * inter, hidden, generator=g)
    down_flinear = torch.randn(num_experts, hidden, inter, generator=g)
    gate_up_grouped = gate_up_flinear.transpose(-2, -1).contiguous()
    down_grouped = down_flinear.transpose(-2, -1).contiguous()
    return gate_up_flinear, down_flinear, gate_up_grouped, down_grouped


def test_nonsquare_gate_up_and_down_unchanged():
    E, H, I = 4, 64, 16
    gu_f, dn_f, gu_g, dn_g = _make_flinear_experts(E, H, I)
    experts = _FakeExperts(gu_f, dn_f)

    out_gu = preprocess_weight(gu_f, "gate_up", H, experts_module=experts)
    out_dn = preprocess_weight(dn_f, "down", H, experts_module=experts)
    assert torch.equal(out_gu, gu_g)
    assert torch.equal(out_dn, dn_g)

    assert torch.equal(preprocess_weight(gu_g, "gate_up", H, experts_module=experts), gu_g)
    assert torch.equal(preprocess_weight(dn_g, "down", H, experts_module=experts), dn_g)


def test_square_gate_up_transposed_via_down_sibling():
    E, H, I = 4, 64, 32
    gu_f, dn_f, gu_g, dn_g = _make_flinear_experts(E, H, I)
    assert gu_f.shape[1] == gu_f.shape[2]
    experts = _FakeExperts(gu_f, dn_f)

    out_gu = preprocess_weight(gu_f, "gate_up", H, experts_module=experts)
    assert torch.equal(out_gu, gu_g), "square gate_up must be transposed to grouped layout"
    experts_g = _FakeExperts(gu_g, dn_g)
    assert torch.equal(preprocess_weight(gu_g, "gate_up", H, experts_module=experts_g), gu_g)


def test_square_down_transposed_via_gate_up_sibling():
    E, H, I = 4, 64, 64
    gu_f, dn_f, gu_g, dn_g = _make_flinear_experts(E, H, I)
    assert dn_f.shape[1] == dn_f.shape[2]
    experts = _FakeExperts(gu_f, dn_f)

    out_dn = preprocess_weight(dn_f, "down", H, experts_module=experts)
    assert torch.equal(out_dn, dn_g), "square down must be transposed to grouped layout"
    experts_g = _FakeExperts(gu_g, dn_g)
    assert torch.equal(preprocess_weight(dn_g, "down", H, experts_module=experts_g), dn_g)


def test_square_without_module_warns_and_transposes(capsys):
    E, H, I = 4, 64, 32
    gu_f, _dn_f, gu_g, _dn_g = _make_flinear_experts(E, H, I)

    out = preprocess_weight(gu_f, "gate_up", H)
    assert torch.equal(out, gu_g)
    err = capsys.readouterr().err
    assert "#849" in err and "ambiguous" in err.lower()
    assert "register_weight_preprocessor" in err


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


def test_orientation_primitive_returns_none_on_square():
    assert _orientation_needs_transpose((4, 64, 64), "gate_up", 64) is None
    assert _orientation_needs_transpose((4, 64, 64), "down", 64) is None
    assert _orientation_needs_transpose((4, 32, 64), "gate_up", 64) is True
    assert _orientation_needs_transpose((4, 64, 32), "gate_up", 64) is False
    assert _orientation_needs_transpose((4, 64, 16), "down", 64) is True
    assert _orientation_needs_transpose((4, 16, 64), "down", 64) is False


def test_logical_expert_shape_prefers_original_shape_after_linear_unwrap():
    class _PackedParam:
        # Physical shape is packed; logical shape is recorded separately.
        shape = (2048, 1)
        _original_shape = (4, 128, 64)

    class _QuantLinear:
        # Linear4bit stores Params4bit on .weight.
        weight = _PackedParam()

    assert _logical_expert_shape(_PackedParam()) == (4, 128, 64)
    assert _logical_expert_shape(_QuantLinear()) == (4, 128, 64)
    assert _logical_expert_shape(torch.randn(4, 128, 64)) == (4, 128, 64)


def test_square_grouped_weight_preserved_via_paramwrapper_sibling():
    class _ParamWrapper:
        """PEFT-like target-parameter wrapper."""

        def __init__(self, base_layer, param):
            self.base_layer = base_layer
            self._param = param

        def get_param(self):
            return self._param

    E, H, I = 4, 64, 32
    _gu_f, _dn_f, gu_g, dn_g = _make_flinear_experts(E, H, I)
    experts = _FakeExperts(gu_g, dn_g)
    experts.down_proj = _ParamWrapper(experts, dn_g)

    assert _logical_expert_shape(experts.down_proj) == tuple(dn_g.shape)
    out = preprocess_weight(gu_g, "gate_up", H, experts_module=experts)
    assert torch.equal(out, gu_g), "a grouped square weight must not be transposed"


def test_logical_expert_shape_reads_shape_3d_without_materializing():
    class _Provider:
        # ParameterModule-like: shape_3d is recorded; get_param would build a full tensor.
        shape_3d = (4, 64, 128)

        def get_param(self):
            raise AssertionError("get_param must not run when shape_3d is recorded")

    assert _logical_expert_shape(_Provider()) == (4, 64, 128)
