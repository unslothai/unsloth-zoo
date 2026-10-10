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

"""A gate_up-only or down_proj-only stacked expert adapter gets the zero half vLLM's FusedMoE3DWithLoRA expects."""
import types

import pytest
import torch

import unsloth_zoo.vllm_utils as vu

E, R, H, I = 4, 2, 32, 16  # experts, rank, hidden, moe intermediate
P = "base_model.model.model.layers.0.mlp.experts"


def _module():
    return types.SimpleNamespace(
        w13_lora_a_stacked = [torch.zeros(1, 1, R * E, H)],
        w13_lora_b_stacked = [torch.zeros(1, 1, 2 * I, R * E)],
        w2_lora_a_stacked = [torch.zeros(1, 1, R * E, I)],
    )


@pytest.fixture
def live(monkeypatch):
    module = _module()
    manager = types.SimpleNamespace(modules = {"layer0": module})
    monkeypatch.setattr(vu, "_get_vllm_model_runner", lambda model: None)
    monkeypatch.setattr(vu, "_get_vllm_lora_manager", lambda model, runner = None: manager)
    monkeypatch.setattr(vu, "_get_vllm_lora_model", lambda model, runner = None: types.SimpleNamespace())
    monkeypatch.setattr(vu, "_resolve_lora_key_to_module", lambda key, mapper: "layer0")
    return module


def test_lone_prefixes():
    both = [f"{P}.base_layer.lora_A.weight", f"{P}.base_layer.lora_B.weight", f"{P}.lora_A.weight", f"{P}.lora_B.weight"]
    assert vu._moe_lone_expert_lora_prefixes(both) == []
    assert vu._moe_lone_expert_lora_prefixes(both[2:]) == [P]
    attn = ["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"]
    assert vu._moe_lone_expert_lora_prefixes(attn) == []


def test_gate_up_only_is_moved_to_base_layer_with_zero_down(live):
    A, B = torch.randn(R * E, H), torch.randn(2 * I, R * E)
    out = vu._pad_lone_moe_expert_lora(object(), {f"{P}.lora_A.weight": A, f"{P}.lora_B.weight": B})
    assert out[f"{P}.base_layer.lora_A.weight"] is A and out[f"{P}.base_layer.lora_B.weight"] is B
    assert out[f"{P}.lora_A.weight"].shape == (R * E, I) and not out[f"{P}.lora_A.weight"].any()
    assert out[f"{P}.lora_B.weight"].shape == (H, R * E) and not out[f"{P}.lora_B.weight"].any()


def test_down_only_gets_zero_gate_up(live):
    A, B = torch.randn(R * E, I), torch.randn(H, R * E)
    out = vu._pad_lone_moe_expert_lora(object(), {f"{P}.lora_A.weight": A, f"{P}.lora_B.weight": B})
    assert out[f"{P}.lora_A.weight"] is A and out[f"{P}.lora_B.weight"] is B
    assert out[f"{P}.base_layer.lora_A.weight"].shape == (R * E, H)
    assert out[f"{P}.base_layer.lora_B.weight"].shape == (2 * I, R * E)
    assert not out[f"{P}.base_layer.lora_A.weight"].any() and not out[f"{P}.base_layer.lora_B.weight"].any()


def test_both_halves_and_mismatched_shapes_are_untouched(live):
    both = {
        f"{P}.base_layer.lora_A.weight": torch.randn(R * E, H), f"{P}.base_layer.lora_B.weight": torch.randn(2 * I, R * E),
        f"{P}.lora_A.weight": torch.randn(R * E, I), f"{P}.lora_B.weight": torch.randn(H, R * E),
    }
    assert vu._pad_lone_moe_expert_lora(object(), both) is both
    odd = {f"{P}.lora_A.weight": torch.randn(R * E, 7), f"{P}.lora_B.weight": torch.randn(5, R * E)}
    assert set(vu._pad_lone_moe_expert_lora(object(), odd)) == set(odd)
