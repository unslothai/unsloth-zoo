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

`load_lora`'s ".lora_A." / ".lora_B." filter lets stacked expert adapters past vLLM's
validation, which then accepts and ignores them: rollouts come from the base experts while
training keeps updating the adapters, and nothing raises. These pin the detector that turns
that into a refusal, including the near-misses a substring match would wrongly reject.
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
    # Near-misses a substring test would reject. The second is real: Qwen MoE's shared
    # expert is a DENSE mlp that vLLM serves through the ordinary LoRA path.
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


# --- the on-disk default branch (load_tensors=False) ----------------------------------
# load_lora's default hands vLLM a PATH, so the in-memory check never runs; vLLM then skips
# the expert keys it cannot place, for the same silent base-experts outcome.
import os

import pytest
import torch

from unsloth_zoo.vllm_utils import _saved_adapter_expert_lora_keys


def _write_safetensors(path, keys):
    from safetensors.torch import save_file
    save_file({k: torch.zeros(2, 2) for k in keys}, path)


def test_reads_expert_keys_from_safetensors(tmp_path):
    _write_safetensors(str(tmp_path / "adapter_model.safetensors"), [
        "base_model.model.model.layers.0.mlp.experts.gate_up_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
    ])
    got = _saved_adapter_expert_lora_keys(str(tmp_path))
    assert len(got) == 1 and "experts" in got[0]


def test_dense_only_adapter_is_allowed(tmp_path):
    _write_safetensors(str(tmp_path / "adapter_model.safetensors"), [
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight",
        # the shared expert is a dense mlp and must NOT trip the guard
        "base_model.model.model.layers.0.mlp.shared_expert.up_proj.lora_A.weight",
    ])
    assert _saved_adapter_expert_lora_keys(str(tmp_path)) == []


def test_reads_expert_keys_from_bin(tmp_path):
    torch.save(
        {"base_model.model.model.layers.0.mlp.experts.down_proj.lora_B.weight": torch.zeros(2, 2)},
        str(tmp_path / "adapter_model.bin"),
    )
    assert len(_saved_adapter_expert_lora_keys(str(tmp_path))) == 1


def test_missing_or_unreadable_adapter_is_not_our_error(tmp_path):
    # No adapter at all: vLLM should raise its own error, we must not pre-empt it.
    assert _saved_adapter_expert_lora_keys(str(tmp_path)) == []
    (tmp_path / "adapter_model.safetensors").write_bytes(b"not a safetensors file")
    assert _saved_adapter_expert_lora_keys(str(tmp_path)) == []


def test_non_lora_expert_tensors_are_ignored(tmp_path):
    # Only adapter tensors matter; a stray expert base weight in the dir is not an adapter.
    _write_safetensors(str(tmp_path / "adapter_model.safetensors"), [
        "base_model.model.model.layers.0.mlp.experts.gate_up_proj.weight",
    ])
    assert _saved_adapter_expert_lora_keys(str(tmp_path)) == []


# --- the ".moe." spelling -------------------------------------------------------------
# saving_utils.py treats ".moe" as an expert prefix and remaps it for Gemma 4, so adapters
# from that layout carry ".moe." and are equally unservable. "experts" alone missed them.

MOE_SPELLING_KEYS = [
    "base_model.model.model.layers.0.moe.gate_up_proj.lora_A.weight",
    "base_model.model.model.layers.7.moe.down_proj.lora_B.weight",
    "model.layers.0.mlp.moe.gate_up_proj.lora_A.weight",
]


@pytest.mark.parametrize("key", MOE_SPELLING_KEYS)
def test_moe_spelling_is_detected(key):
    assert _is_moe_expert_lora_key(key), key


def test_moe_spelling_is_still_segment_matched():
    # "moe" must be a whole segment, so these dense names stay servable.
    assert not _is_moe_expert_lora_key("a.moecoder.b.lora_A.weight")
    assert not _is_moe_expert_lora_key("a.my_moe_thing.b.lora_A.weight")
    assert not _is_moe_expert_lora_key("base_model.model.model.layers.0.mlp.shared_expert.up_proj.lora_A.weight")


# --- parent-qualified expert names (GraniteMoE) ---------------------------------------
# On transformers 4.57 through 5.0 GraniteMoE's experts are `block_sparse_moe.input_linear` /
# `.output_linear`, which carry neither "experts" nor "moe" as a segment. They cannot be
# matched bare: the same pair is a DENSE nn.Linear under `shared_mlp` in granitemoeshared /
# _swa / hybrid, and a projector in granite_speech. Hence (parent, child) matching.

GRANITE_EXPERT_KEYS = [
    "base_model.model.model.layers.0.block_sparse_moe.input_linear.lora_A.default.weight",
    "base_model.model.model.layers.0.block_sparse_moe.output_linear.lora_B.default.weight",
    "model.layers.9.block_sparse_moe.input_linear.lora_A.weight",
]

GRANITE_DENSE_KEYS = [
    # dense shared MLP in granitemoeshared / _swa / hybrid: servable
    "base_model.model.model.layers.0.shared_mlp.input_linear.lora_A.weight",
    "base_model.model.model.layers.0.shared_mlp.output_linear.lora_B.weight",
    # granite_speech audio projector: servable
    "base_model.model.model.encoder.input_linear.lora_A.weight",
]


@pytest.mark.parametrize("key", GRANITE_EXPERT_KEYS)
def test_granitemoe_stacked_experts_are_detected(key):
    assert _is_moe_expert_lora_key(key), key


@pytest.mark.parametrize("key", GRANITE_DENSE_KEYS)
def test_granite_dense_linears_are_not_detected(key):
    assert not _is_moe_expert_lora_key(key), key


def test_qualified_match_requires_the_parent_segment():
    """input_linear alone must not match, or every granitemoeshared adapter breaks."""
    assert not _is_moe_expert_lora_key("a.b.input_linear.lora_A.weight")
    assert not _is_moe_expert_lora_key("input_linear.output_linear.lora_A.weight")
    # and the parent has to be the IMMEDIATE parent
    assert not _is_moe_expert_lora_key("a.block_sparse_moe.x.input_linear.lora_A.weight")


def test_granitemoe_new_layout_still_caught_by_bare_segment():
    # the 5.x refactor is covered by the "experts" segment, with no extra rule
    assert _is_moe_expert_lora_key(
        "base_model.model.model.layers.0.block_sparse_moe.experts.gate_up_proj.lora_A.weight"
    )


def test_granitemoe_old_layout_is_caught_on_disk(tmp_path):
    _write_safetensors(str(tmp_path / "adapter_model.safetensors"), [
        "base_model.model.model.layers.0.block_sparse_moe.input_linear.lora_A.default.weight",
        "base_model.model.model.layers.0.shared_mlp.input_linear.lora_A.default.weight",
    ])
    got = _saved_adapter_expert_lora_keys(str(tmp_path))
    assert len(got) == 1 and "block_sparse_moe" in got[0]


def test_moe_spelling_is_caught_on_disk(tmp_path):
    _write_safetensors(str(tmp_path / "adapter_model.safetensors"), [
        "base_model.model.model.layers.0.moe.gate_up_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
    ])
    got = _saved_adapter_expert_lora_keys(str(tmp_path))
    assert len(got) == 1 and ".moe." in got[0]
