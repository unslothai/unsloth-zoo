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
"""Per-expert 2D MoE LoRA (`experts.<i>.<proj>`) is servable on a 2D vLLM model; engine lookups are stubbed."""
import types

import pytest

import unsloth_zoo.vllm_utils as vu


PER_EXPERT_KEYS = [
    "base_model.model.model.layers.0.mlp.experts.0.gate_proj.lora_A.weight",
    "base_model.model.model.layers.0.mlp.experts.0.up_proj.lora_B.weight",
    "base_model.model.model.layers.0.mlp.experts.7.down_proj.lora_A.weight",
    "base_model.model.model.layers.1.block_sparse_moe.experts.3.w1.lora_A.weight",
]
STACKED_KEYS = [
    "base_model.model.model.layers.0.mlp.experts.base_layer.lora_A.weight",
    "base_model.model.model.layers.0.mlp.experts.lora_B.weight",
    "base_model.model.model.layers.0.mlp.experts.gate_up_proj.lora_A.weight",
]
ATTENTION_KEYS = ["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"]


@pytest.fixture
def engine(monkeypatch):
    state = types.SimpleNamespace(is_3d = False, mixed = False, unmatched = [])

    def vllm_model(*a, **k):
        cls = type("FakeVllmMoe", (), {"is_3d_moe_weight": state.is_3d})
        return cls()

    monkeypatch.setattr(vu, "_get_vllm_model_runner", lambda model: object())
    monkeypatch.setattr(vu, "_get_vllm_lora_model", vllm_model)
    monkeypatch.setattr(
        vu,
        "_get_vllm_lora_manager",
        lambda model, runner: types.SimpleNamespace(_enable_mixed_moe_lora_format = state.mixed),
    )
    monkeypatch.setattr(vu, "_unmatched_lora_keys", lambda model, keys: list(state.unmatched))
    monkeypatch.delenv("UNSLOTH_DISABLE_LORA_NAME_CHECK", raising = False)
    return state


def _model(**attrs):
    return types.SimpleNamespace(config = types.SimpleNamespace(), **attrs)


@pytest.mark.parametrize("key", PER_EXPERT_KEYS)
def test_per_expert_keys_are_classified(key):
    assert vu._is_moe_expert_lora_key(key)
    assert vu._is_moe_per_expert_lora_key(key)


@pytest.mark.parametrize("key", STACKED_KEYS + ATTENTION_KEYS)
def test_stacked_and_dense_keys_are_not_per_expert(key):
    assert not vu._is_moe_per_expert_lora_key(key)


def test_per_expert_adapter_is_served_by_the_2d_wrapper(engine):
    vu._check_lora_is_servable(_model(), PER_EXPERT_KEYS + ATTENTION_KEYS, "the training model", {"r": 16})


def test_per_expert_adapter_on_a_3d_model_is_refused(engine):
    engine.is_3d = True
    with pytest.raises(NotImplementedError, match = "per-expert"):
        vu._check_lora_is_servable(_model(), PER_EXPERT_KEYS, "the training model", {"r": 16})
    engine.mixed = True
    vu._check_lora_is_servable(_model(), PER_EXPERT_KEYS, "the training model", {"r": 16})


def test_stacked_adapter_still_needs_the_3d_wrapper(engine):
    with pytest.raises(NotImplementedError, match = "is_3d_moe_weight"):
        vu._check_lora_is_servable(_model(), STACKED_KEYS, "the training model", {"r": 16})
    with pytest.raises(NotImplementedError, match = "stacked"):
        vu._check_lora_is_servable(
            _model(), PER_EXPERT_KEYS + STACKED_KEYS, "the training model", {"r": 16}
        )
    engine.is_3d = True
    vu._check_lora_is_servable(_model(), STACKED_KEYS, "the training model", {"r": 16})


def test_per_expert_adapter_keeps_the_rank_and_bitsandbytes_limits(engine):
    with pytest.raises(NotImplementedError, match = "128"):
        vu._check_lora_is_servable(_model(), PER_EXPERT_KEYS, "the training model", {"r": 256})
    with pytest.raises(NotImplementedError, match = "bitsandbytes"):
        vu._check_lora_is_servable(
            _model(is_loaded_in_4bit = True), PER_EXPERT_KEYS, "the training model", {"r": 16}
        )


def test_per_expert_adapter_still_goes_through_the_name_check(engine):
    engine.unmatched = [(PER_EXPERT_KEYS[0], "model.layers.0.mlp.experts.0.gate_proj")]
    with pytest.raises(RuntimeError, match = "does not have"):
        vu._check_lora_is_servable(_model(), PER_EXPERT_KEYS, "the training model", {"r": 16})
