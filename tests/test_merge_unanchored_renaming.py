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

"""LoRA keys of a natively loaded Ling 3.0 (bailing_hybrid) resolve to checkpoint keys per shard.

Transformers 5.20 renames `.attention.` to `.self_attn.` with an unanchored pattern, so a shard
without `model.word_embeddings` reads as new-format and its attention LoRAs were never merged
(unslothai/unsloth#13948: "# of LoRAs = 144 does not match # of saved modules = 92").
"""
from collections import defaultdict

import pytest

import unsloth_zoo.saving_utils as su

# The renaming half of transformers 5.20's `bailing_hybrid` conversion mapping, in order.
BAILING_HYBRID = {
    r"^model\.word_embeddings\.": "model.embed_tokens.",
    r"\.attention\.": ".self_attn.",
    r"self_attn\.f_proj\.": "self_attn.forget_gate.f_proj.",
    r"self_attn\.dt_bias": "self_attn.forget_gate.dt_bias",
    r"self_attn\.A_log": "self_attn.forget_gate.A_log",
    r"\.self_attn\.dense\.": ".self_attn.o_proj.",
    r"\.mlp\.gate\.expert_bias$": ".mlp.gate.e_score_correction_bias",
}


@pytest.fixture
def bailing_mapping(monkeypatch):
    monkeypatch.setattr(su, "_get_checkpoint_conversion_mapping", lambda name: dict(BAILING_HYBRID))


def _lora(keys):
    out = defaultdict(lambda: None)
    for k in keys:
        out[k] = object()
    return out


# Layer 2 is linear attention (KDA), layer 3 full attention (MLA, on-disk `attention.dense`).
SHARD = [
    "model.layers.2.attention.q_proj.weight",
    "model.layers.2.attention.k_proj.weight",
    "model.layers.2.attention.o_proj.weight",
    "model.layers.2.attention.f_proj.weight",
    "model.layers.2.mlp.shared_experts.up_proj.weight",
    "model.layers.3.attention.q_a_proj.weight",
    "model.layers.3.attention.dense.weight",
]
LORA = [
    "model.layers.2.self_attn.q_proj",
    "model.layers.2.self_attn.k_proj",
    "model.layers.2.self_attn.o_proj",
    "model.layers.2.mlp.shared_experts.up_proj",
    "model.layers.3.self_attn.o_proj",
]
EXPECTED = {
    "model.layers.2.self_attn.q_proj": "model.layers.2.attention.q_proj",
    "model.layers.2.self_attn.k_proj": "model.layers.2.attention.k_proj",
    "model.layers.2.self_attn.o_proj": "model.layers.2.attention.o_proj",
    "model.layers.2.mlp.shared_experts.up_proj": "model.layers.2.mlp.shared_experts.up_proj",
    "model.layers.3.self_attn.o_proj": "model.layers.3.attention.dense",
}


def _resolved(lora, shard):
    converted = su._convert_lora_keys_to_safetensor_format(lora, shard, "BailingHybridForCausalLM")
    by_stats = {id(v): k for k, v in converted.items()}
    return {k: by_stats[id(v)] for k, v in lora.items()}


@pytest.mark.parametrize("with_embeddings", [False, True], ids = ["inner_shard", "first_shard"])
def test_bailing_hybrid_lora_keys_reach_checkpoint_names(bailing_mapping, with_embeddings):
    shard = SHARD + (["model.word_embeddings.weight"] if with_embeddings else [])
    lora = _lora(LORA)
    assert _resolved(lora, shard) == EXPECTED
    backed = su._backed_lora_keys(
        su._convert_lora_keys_to_safetensor_format(lora, shard, "BailingHybridForCausalLM"),
        set(shard), False,
    )
    assert len(backed) == len(LORA)


def test_new_format_shard_is_left_alone(bailing_mapping):
    # A checkpoint already in transformers 5 names must not be reversed onto old names.
    shard = [k.replace(".attention.", ".self_attn.").replace("self_attn.dense", "self_attn.o_proj") for k in SHARD]
    lora = _lora(LORA)
    assert _resolved(lora, shard) == {k: k for k in LORA}


def test_lora_key_in_another_shard_stays_unbacked(bailing_mapping):
    lora = _lora(["model.layers.9.self_attn.q_proj"])
    converted = su._convert_lora_keys_to_safetensor_format(lora, SHARD, "BailingHybridForCausalLM")
    assert not su._backed_lora_keys(converted, set(SHARD), False)
