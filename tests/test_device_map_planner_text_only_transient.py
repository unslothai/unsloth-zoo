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

"""A text_only load of a MoE VLM merges the parent's expert stacks, so the planner must keep room for them."""
import pytest
import torch

pytest.importorskip("transformers.models.minimax_m3_vl")
from transformers.conversion_mapping import get_checkpoint_conversion_mapping  # noqa: E402

from unsloth_zoo.device_map_planner import (  # noqa: E402
    _load_transient_by_unit,
    _merged_parameter_patterns,
    plan_device_map,
)

_H, _I, _E = 64, 32, 4


@pytest.fixture(autouse = True)
def _default_allocator(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising = False)
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising = False)


def _meta_minimax_m3_text(layers = 6):
    from transformers import AutoModelForCausalLM
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLTextConfig

    config = MiniMaxM3VLTextConfig(
        num_hidden_layers = layers, hidden_size = _H, intermediate_size = _I, num_local_experts = _E,
        vocab_size = 128, num_attention_heads = 4, num_key_value_heads = 1, head_dim = 16,
        dense_intermediate_size = 64, shared_intermediate_size = 32, bos_token_id = 1, eos_token_id = 2,
    )
    with torch.device("meta"):
        return AutoModelForCausalLM.from_config(config, dtype = torch.bfloat16)


def test_text_only_decoder_has_no_conversions_of_its_own():
    # The premise: the bare decoder's model_type registers nothing; the loader borrows the VLM's.
    model = _meta_minimax_m3_text()
    assert get_checkpoint_conversion_mapping(model.config.model_type) is None
    assert get_checkpoint_conversion_mapping("minimax_m3_vl")


def test_text_only_decoder_finds_the_parent_expert_merges():
    model = _meta_minimax_m3_text()
    patterns = _merged_parameter_patterns(model)
    name = "model.layers.3.mlp.experts.gate_up_proj"
    assert any(p.search(name) for p in patterns)
    units = [(f"model.layers.{i}", 0) for i in range(6)]
    transient = _load_transient_by_unit(model, units)
    assert transient["model.layers.3"] == _E * 2 * _I * _H * 2


def test_text_only_decoder_plan_keeps_room_to_merge():
    model = _meta_minimax_m3_text()
    plan = plan_device_map(model, max_memory = {0: 1024 ** 3, 1: 1024 ** 3}, activation_reserve_bytes = 0, safety_bytes = 0,
    )
    assert plan.load_transient_by_device
    assert any("load transient" in n for n in plan.notes)


def test_a_config_that_is_not_a_text_sub_config_gets_no_parent():
    from transformers import MixtralConfig, MixtralForCausalLM

    from unsloth_zoo.device_map_planner import _text_only_parent_model_types

    config = MixtralConfig(hidden_size = 32, intermediate_size = 32, num_hidden_layers = 1,
                           num_attention_heads = 4, num_key_value_heads = 2, vocab_size = 64)
    with torch.device("meta"):
        model = MixtralForCausalLM(config)
    assert _text_only_parent_model_types(model) == []
