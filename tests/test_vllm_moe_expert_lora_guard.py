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
"""`load_lora` must refuse MoE expert adapters rather than ship them to vLLM.

The state_dict filter in `load_lora` selects on ".lora_A." / ".lora_B." alone, so
stacked expert adapters pass it, and vLLM's own validation is bypassed for them. They
are then accepted and ignored, which means rollouts come from the base experts while
training keeps updating the adapters. Nothing raises. These tests pin the detector that
turns that into a refusal, and in particular pin the near-misses, since a substring
match on "experts" would wrongly reject a dense projection.
"""
import pytest

from unsloth_zoo.vllm_utils import _is_moe_expert_lora_key


EXPERT_KEYS = [
    # Qwen3 MoE / Qwen3.5 / Qwen3.6, the layout Unsloth logs as
    # "Enabling LoRA on MoE parameters: ['mlp.experts.gate_up_proj', 'mlp.experts.down_proj']"
    "base_model.model.model.layers.0.mlp.experts.gate_up_proj.lora_A.weight",
    "base_model.model.model.layers.3.mlp.experts.down_proj.lora_B.weight",
    # gpt-oss and Gemma 4 expose the stacked experts one level up
    "model.layers.0.experts.gate_up_proj.lora_A.weight",
    "model.layers.11.experts.down_proj.lora_B.weight",
]

NON_EXPERT_KEYS = [
    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
    "base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight",
    "base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight",
    "base_model.model.model.layers.0.mlp.up_proj.lora_A.weight",
    "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight",
    # Near-misses. A substring test on "experts" would reject both of these, and the
    # second one is a real module: Qwen MoE's shared expert is a DENSE mlp that vLLM
    # serves through the ordinary LoRA path, so rejecting it would break a working case.
    "base_model.model.model.layers.0.mlp.experts_gate.lora_A.weight",
    "base_model.model.model.layers.0.mlp.shared_expert.up_proj.lora_A.weight",
]


@pytest.mark.parametrize("key", EXPERT_KEYS)
def test_stacked_expert_adapters_are_detected(key):
    assert _is_moe_expert_lora_key(key), key


@pytest.mark.parametrize("key", NON_EXPERT_KEYS)
def test_dense_adapters_are_not_detected(key):
    assert not _is_moe_expert_lora_key(key), key


def test_detector_matches_dotted_segments_not_substrings():
    """"experts" has to be a whole dotted segment, otherwise the shared expert breaks."""
    assert _is_moe_expert_lora_key("a.experts.b.lora_A.weight")
    assert not _is_moe_expert_lora_key("a.my_experts_thing.b.lora_A.weight")
    assert not _is_moe_expert_lora_key("a.expertsb.lora_A.weight")
