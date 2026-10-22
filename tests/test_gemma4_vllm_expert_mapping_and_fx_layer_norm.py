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

"""Gemma 4 vLLM expert-mapping shim (vLLM 0.19-0.24 only) and FX-traceable F.layer_norm."""

from __future__ import annotations

import sys
import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


def _install_fake_vllm(monkeypatch, *, routed_has_mapping, routed_exists=True):
    calls = []

    class Experts:
        @classmethod
        def make_expert_params_mapping(cls, model, **kwargs):
            calls.append(kwargs)
            return [("experts.w13_weight", "experts.0.gate_proj.", 0, "w1")]

    if routed_has_mapping:
        Experts.get_expert_mapping = lambda self: []

    class Gemma4ForCausalLM:
        def __init__(self, num_experts):
            self.config = types.SimpleNamespace(num_experts = num_experts)

    mods = {
        "vllm": types.ModuleType("vllm"),
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.models": types.ModuleType("vllm.model_executor.models"),
        "vllm.model_executor.models.gemma4": types.ModuleType("vllm.model_executor.models.gemma4"),
        "vllm.model_executor.layers": types.ModuleType("vllm.model_executor.layers"),
        "vllm.model_executor.layers.fused_moe": types.ModuleType("vllm.model_executor.layers.fused_moe"),
        "vllm.model_executor.layers.fused_moe.layer": types.ModuleType("vllm.model_executor.layers.fused_moe.layer"),
    }
    mods["vllm.model_executor.models.gemma4"].Gemma4ForCausalLM = Gemma4ForCausalLM
    if routed_exists:
        routed = types.ModuleType("vllm.model_executor.layers.fused_moe.routed_experts")
        routed.RoutedExperts = Experts
        mods[routed.__name__] = routed
    else:
        # vLLM 0.19-0.23: the package-level FusedMoE may be a factory; the class is in .layer.
        mods["vllm.model_executor.layers.fused_moe"].FusedMoE = lambda *a, **k: None
        mods["vllm.model_executor.layers.fused_moe.layer"].FusedMoE = Experts
        monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.fused_moe.routed_experts", None)
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return Gemma4ForCausalLM, calls


@pytest.mark.parametrize("routed_exists", [True, False])
def test_shim_installed_when_vllm_lacks_expert_mapping(monkeypatch, routed_exists):
    from unsloth_zoo.empty_model import _patch_gemma4_vllm_expert_mapping

    cls, calls = _install_fake_vllm(monkeypatch, routed_has_mapping = False, routed_exists = routed_exists)
    _patch_gemma4_vllm_expert_mapping()
    assert cls(num_experts = 4).get_expert_mapping() == [("experts.w13_weight", "experts.0.gate_proj.", 0, "w1")]
    assert calls[0]["num_experts"] == 4
    assert (calls[0]["ckpt_gate_proj_name"], calls[0]["ckpt_up_proj_name"], calls[0]["ckpt_down_proj_name"]) == (
        "gate_proj", "up_proj", "down_proj",
    )
    assert cls(num_experts = None).get_expert_mapping() == []


def test_shim_skipped_when_vllm_has_native_mapping(monkeypatch):
    # A top-level method would shadow vLLM 0.25+ RoutedExperts.get_expert_mapping.
    from unsloth_zoo.empty_model import _patch_gemma4_vllm_expert_mapping

    cls, _ = _install_fake_vllm(monkeypatch, routed_has_mapping = True)
    _patch_gemma4_vllm_expert_mapping()
    assert not hasattr(cls, "get_expert_mapping")


def test_patched_layer_norm_is_fx_symbolically_traceable(monkeypatch):
    from unsloth_zoo import patch_torch_functions as ptf

    monkeypatch.setattr(F, "layer_norm", ptf.layer_norm)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(8)

        def forward(self, x):
            return F.layer_norm(x, (8,), self.ln.weight, self.ln.bias) * 2

    m = M()
    x = torch.randn(2, 8)
    traced = torch.fx.symbolic_trace(m)
    torch.testing.assert_close(traced(x), torch.layer_norm(x, (8,), m.ln.weight, m.ln.bias, 1e-5) * 2)
