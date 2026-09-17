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
"""`load_lora` must refuse only the adapters vLLM would accept and then silently ignore.

vLLM >= 0.12 serves stacked MoE expert LoRA through FusedMoE3DWithLoRA, so "this adapter
touches experts" is not by itself a reason to refuse. What is: the engine not picking the
3D wrapper, a rank past the fused kernel's ceiling, bitsandbytes weights, or any adapter
key that names a module the live engine does not have. These pin the detector, the
near-misses a substring match would wrongly reject, the capability gate, and the
name-resolution check.
"""
import pytest

from unsloth_zoo.vllm_utils import _is_moe_expert_lora_key


EXPERT_KEYS = [
    # Qwen3 MoE / Qwen3.5 / 3.6
    "base_model.model.model.layers.0.mlp.experts.gate_up_proj.lora_A.weight",
    "base_model.model.model.layers.3.mlp.experts.down_proj.lora_B.weight",
    # gpt-oss and Gemma 4: experts one level up
    "model.layers.0.experts.gate_up_proj.lora_A.weight",
    "model.layers.11.experts.down_proj.lora_B.weight",
]

NON_EXPERT_KEYS = [
    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
    "base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight",
    "base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight",
    "base_model.model.model.layers.0.mlp.up_proj.lora_A.weight",
    "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight",
    # Near-misses. The second is real: Qwen MoE's shared expert is a dense mlp vLLM serves.
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


# --- on-disk branch (load_tensors=False): vLLM gets a PATH, so the in-memory check never
# runs and vLLM silently skips the expert keys it cannot place. -------------------------
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
        # dense shared expert must NOT trip the guard
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
    # No adapter: vLLM raises its own error, we must not pre-empt it.
    assert _saved_adapter_expert_lora_keys(str(tmp_path)) == []
    (tmp_path / "adapter_model.safetensors").write_bytes(b"not a safetensors file")
    assert _saved_adapter_expert_lora_keys(str(tmp_path)) == []


def test_non_lora_expert_tensors_are_ignored(tmp_path):
    # A stray expert base weight is not an adapter.
    _write_safetensors(str(tmp_path / "adapter_model.safetensors"), [
        "base_model.model.model.layers.0.mlp.experts.gate_up_proj.weight",
    ])
    assert _saved_adapter_expert_lora_keys(str(tmp_path)) == []


# --- ".moe.", the Gemma 4 spelling saving_utils.py remaps to ".experts": equally
# unservable, and "experts" alone missed it. -------------------------------------------

MOE_SPELLING_KEYS = [
    "base_model.model.model.layers.0.moe.gate_up_proj.lora_A.weight",
    "base_model.model.model.layers.7.moe.down_proj.lora_B.weight",
    "model.layers.0.mlp.moe.gate_up_proj.lora_A.weight",
]


@pytest.mark.parametrize("key", MOE_SPELLING_KEYS)
def test_moe_spelling_is_detected(key):
    assert _is_moe_expert_lora_key(key), key


def test_moe_spelling_is_still_segment_matched():
    # whole segment only, so these dense names stay servable
    assert not _is_moe_expert_lora_key("a.moecoder.b.lora_A.weight")
    assert not _is_moe_expert_lora_key("a.my_moe_thing.b.lora_A.weight")
    assert not _is_moe_expert_lora_key("base_model.model.model.layers.0.mlp.shared_expert.up_proj.lora_A.weight")


# --- parent-qualified names (GraniteMoE) ----------------------------------------------
# On transformers 4.57 to 5.0 the experts are block_sparse_moe.input_linear / .output_linear,
# matching neither "experts" nor "moe". Matching them bare would reject the DENSE pair at
# shared_mlp.input_linear, which GraniteMoeShared carries in the SAME model. Hence
# (parent, child) matching. ------------------------------------------------------------

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


# --- the capability gate: MoE expert LoRA is servable when vLLM picks the 3D wrapper -----
# vLLM >= 0.12 serves PEFT's target_parameters layout directly through FusedMoE3DWithLoRA,
# verified end to end on Qwen3.6-35B-A3B (40 wrappers, output moved, and token-for-token
# identical to a hand-merged checkpoint). So detecting an expert key is not by itself a
# reason to refuse; the engine's own is_3d_moe_weight is. ---------------------------------

import types

import torch.nn as nn
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.fused_moe import MoERunner

from unsloth_zoo.vllm_utils import (
    _check_lora_is_servable,
    _moe_expert_lora_refusal_reason,
    _peft_max_rank,
    _resolve_lora_key_to_module,
    _saved_adapter_lora_keys,
    _unmatched_lora_keys,
    _vllm_lora_target_names,
)


class _FakeLinear(LinearBase):
    def __init__(self): nn.Module.__init__(self)


class _FakeExperts(MoERunner):
    def __init__(self): nn.Module.__init__(self)


class _Node(nn.Module):
    pass


def _build_vllm_model(paths, is_3d_moe_weight = True, lora_config = None):
    """A stand-in vLLM model whose named_modules() has exactly `paths`."""
    class _FakeVllmModel(nn.Module):
        packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    _FakeVllmModel.is_3d_moe_weight = is_3d_moe_weight

    root = _FakeVllmModel()
    if lora_config is not None:
        root.vllm_config = types.SimpleNamespace(lora_config = lora_config)
    for path, cls in paths.items():
        parts = path.split(".")
        node = root
        for part in parts[:-1]:
            if part not in node._modules: node.add_module(part, _Node())
            node = node._modules[part]
        node.add_module(parts[-1], cls())
    return root


def _fake_trainer(vllm_model, **attrs):
    """An HF-side model whose .vllm_engine reaches `vllm_model`, as load_lora expects."""
    runner = types.SimpleNamespace(model = vllm_model)
    worker = types.SimpleNamespace(model_runner = runner)
    executor = types.SimpleNamespace(driver_worker = worker)
    engine = types.SimpleNamespace(model_executor = executor)
    model = types.SimpleNamespace(vllm_engine = types.SimpleNamespace(llm_engine = engine))
    for k, v in attrs.items(): setattr(model, k, v)
    return model


QWEN_PATHS = {
    "model.layers.0.self_attn.qkv_proj": _FakeLinear,
    "model.layers.0.self_attn.o_proj": _FakeLinear,
    "model.layers.0.mlp.experts": _FakeExperts,
}
QWEN_EXPERT_KEYS = [
    "base_model.model.model.layers.0.mlp.experts.base_layer.lora_A.weight",
    "base_model.model.model.layers.0.mlp.experts.base_layer.lora_B.weight",
    "base_model.model.model.layers.0.mlp.experts.lora_A.weight",
    "base_model.model.model.layers.0.mlp.experts.lora_B.weight",
    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
    "base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight",
]


def test_target_names_include_packed_constituents():
    names, _ = _vllm_lora_target_names(_build_vllm_model(QWEN_PATHS))
    assert "model.layers.0.self_attn.qkv_proj" in names
    # q/k/v are one module in vLLM, so the adapter's own name must still resolve
    assert "model.layers.0.self_attn.q_proj" in names
    assert "model.layers.0.self_attn.v_proj" in names
    assert "model.layers.0.mlp.experts" in names


def test_three_d_moe_expert_lora_is_allowed():
    """The whole point of narrowing: Qwen3.5/3.6 stacked expert LoRA must load."""
    model = _fake_trainer(_build_vllm_model(QWEN_PATHS, is_3d_moe_weight = True))
    assert _moe_expert_lora_refusal_reason(model, {"r": 16}) is None
    _check_lora_is_servable(model, QWEN_EXPERT_KEYS, "the training model", {"r": 16})


def test_two_d_moe_expert_lora_is_refused():
    """Gemma 4 does not set is_3d_moe_weight, and its adapters are silently ignored."""
    model = _fake_trainer(_build_vllm_model(QWEN_PATHS, is_3d_moe_weight = False))
    reason = _moe_expert_lora_refusal_reason(model, {"r": 16})
    assert reason is not None and "is_3d_moe_weight" in reason
    with pytest.raises(NotImplementedError, match = "is_3d_moe_weight"):
        _check_lora_is_servable(model, QWEN_EXPERT_KEYS, "the training model", {"r": 16})


def test_rank_above_128_is_refused():
    """fused_moe_lora_op.py asserts rank <= 128, and max_lora_rank 256 passes config."""
    model = _fake_trainer(_build_vllm_model(QWEN_PATHS))
    assert "128" in _moe_expert_lora_refusal_reason(model, {"r": 256})
    assert _moe_expert_lora_refusal_reason(model, {"r": 128}) is None
    # rank_pattern can push an individual module past the ceiling on its own
    assert "128" in _moe_expert_lora_refusal_reason(model, {"r": 16, "rank_pattern": {"experts": 256}})


def test_bitsandbytes_is_refused():
    """vLLM does not support MoE LoRA on bnb weights, and Unsloth 4bit IS bnb."""
    model = _fake_trainer(_build_vllm_model(QWEN_PATHS), is_loaded_in_4bit = True)
    assert "bitsandbytes" in _moe_expert_lora_refusal_reason(model, {"r": 16})

    quantized = _fake_trainer(
        _build_vllm_model(QWEN_PATHS),
        config = types.SimpleNamespace(
            quantization_config = types.SimpleNamespace(quant_method = "bitsandbytes"),
        ),
    )
    assert "bitsandbytes" in _moe_expert_lora_refusal_reason(quantized, {"r": 16})


def test_mixed_moe_lora_format_is_refused():
    """It forces the 2D wrapper, which cannot read a stacked adapter."""
    vllm_model = _build_vllm_model(
        QWEN_PATHS, lora_config = types.SimpleNamespace(enable_mixed_moe_lora_format = True),
    )
    reason = _moe_expert_lora_refusal_reason(_fake_trainer(vllm_model), {"r": 16})
    assert reason is not None and "mixed_moe_lora_format" in reason


def test_unreachable_engine_stays_conservative():
    """We cannot confirm the 3D path, so we must not assume it."""
    reason = _moe_expert_lora_refusal_reason(types.SimpleNamespace(), {"r": 16})
    assert reason is not None and "could not be inspected" in reason


# --- the name-resolution check, which is what actually catches the silent cases ----------
# Unsloth's load_tensors path goes through from_lora_tensors, which runs no
# check_unexpected_modules, and activate_adapter then zeroes unmatched modules at debug
# level. vLLM's own check compares the LAST path component only (#34186), so Gemma 4's
# `...layers.N.experts` vs vLLM's `...layers.N.moe.experts` passes it. --------------------

GEMMA_PATHS = {
    "language_model.model.layers.0.self_attn.qkv_proj": _FakeLinear,
    "language_model.model.layers.0.moe.experts": _FakeExperts,
}


def test_parent_path_mismatch_is_caught_even_when_the_leaf_matches():
    model = _fake_trainer(_build_vllm_model(GEMMA_PATHS))
    # both end in "experts", so only the full name separates them
    unmatched = _unmatched_lora_keys(model, [
        "base_model.model.language_model.model.layers.0.experts.lora_A.weight",
    ])
    assert [n for _, n in unmatched] == ["language_model.model.layers.0.experts"]

    matched = _unmatched_lora_keys(model, [
        "base_model.model.language_model.model.layers.0.moe.experts.lora_A.weight",
        "base_model.model.language_model.model.layers.0.self_attn.q_proj.lora_A.weight",
    ])
    assert matched == []


def test_check_raises_on_an_unresolvable_key():
    model = _fake_trainer(_build_vllm_model(QWEN_PATHS))
    bad = "base_model.model.model.layers.0.self_attn.nonexistent_proj.lora_A.weight"
    with pytest.raises(RuntimeError, match = "silently ignore"):
        _check_lora_is_servable(model, [bad], "the training model", {"r": 16})


def test_dense_adapter_on_a_matching_engine_passes():
    model = _fake_trainer(_build_vllm_model(QWEN_PATHS))
    _check_lora_is_servable(model, [
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.k_proj.lora_B.weight",
        "base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight",
    ], "the training model", {"r": 16})


def test_name_check_is_skipped_when_the_engine_cannot_be_inspected():
    # No vLLM model to compare against: report "cannot tell", never a false refusal.
    assert _unmatched_lora_keys(types.SimpleNamespace(), ["a.b.lora_A.weight"]) is None


def test_unparseable_keys_are_left_to_vllm():
    model = _fake_trainer(_build_vllm_model(QWEN_PATHS))
    assert _unmatched_lora_keys(model, ["not_a_lora_tensor"]) == []
    assert _resolve_lora_key_to_module("not_a_lora_tensor", None) is None


def test_name_check_has_an_escape_hatch(monkeypatch):
    model = _fake_trainer(_build_vllm_model(QWEN_PATHS))
    bad = "base_model.model.model.layers.0.self_attn.nonexistent_proj.lora_A.weight"
    monkeypatch.setenv("UNSLOTH_DISABLE_LORA_NAME_CHECK", "1")
    _check_lora_is_servable(model, [bad], "the training model", {"r": 16})


def test_saved_adapter_lora_keys_returns_every_adapter_tensor(tmp_path):
    _write_safetensors(str(tmp_path / "adapter_model.safetensors"), [
        "base_model.model.model.layers.0.mlp.experts.gate_up_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight",
    ])
    assert sorted(_saved_adapter_lora_keys(str(tmp_path))) == [
        "base_model.model.model.layers.0.mlp.experts.gate_up_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
    ]


def test_peft_max_rank_handles_objects_and_missing_values():
    assert _peft_max_rank(types.SimpleNamespace(r = 32, rank_pattern = None)) == 32
    assert _peft_max_rank({"r": "not an int"}) is None
    assert _peft_max_rank(None) is None


# --- the wrapped layout, which is what a live enable_lora engine actually looks like -----
# _create_lora_modules replaces each target with a BaseLayerWithLoRA that HOLDS the
# LinearBase / MoERunner at `<module>.base_layer`. Walking for LinearBase alone therefore
# finds `...qkv_proj.base_layer` and never `...qkv_proj`, and every adapter key would be
# flagged. Both layouts have to resolve. ------------------------------------------------

WRAPPED_PATHS = {
    "model.layers.0.self_attn.qkv_proj.base_layer": _FakeLinear,
    "model.layers.0.self_attn.o_proj.base_layer": _FakeLinear,
    "model.layers.0.mlp.experts.base_layer": _FakeExperts,
}


def test_wrapped_modules_resolve_to_the_adapter_name():
    names, _ = _vllm_lora_target_names(_build_vllm_model(WRAPPED_PATHS))
    assert "model.layers.0.self_attn.qkv_proj" in names
    assert "model.layers.0.self_attn.q_proj" in names
    assert "model.layers.0.mlp.experts" in names

    model = _fake_trainer(_build_vllm_model(WRAPPED_PATHS))
    assert _unmatched_lora_keys(model, QWEN_EXPERT_KEYS) == []


def test_manager_modules_are_preferred_over_the_module_walk():
    """activate_adapter walks the manager's dict, so a module vLLM declined to wrap is
    correctly absent from it even though named_modules still reports it."""
    vllm_model = _build_vllm_model(WRAPPED_PATHS)
    manager = types.SimpleNamespace(
        modules = {"model.layers.0.self_attn.qkv_proj": object()},
        packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]},
    )
    names, _ = _vllm_lora_target_names(vllm_model, manager)
    assert names == {
        "model.layers.0.self_attn.qkv_proj",
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
    }
    assert "model.layers.0.mlp.experts" not in names


def test_embedding_modules_are_matched_by_leaf_name():
    """lm_head is a ParallelLMHead, not a LinearBase, and the model declares it instead."""
    vllm_model = _build_vllm_model(QWEN_PATHS)
    vllm_model.embedding_modules = {"embed_tokens": "input_embeddings", "lm_head": "output_embeddings"}
    model = _fake_trainer(vllm_model)
    assert _unmatched_lora_keys(model, [
        "base_model.model.lm_head.lora_A.weight",
        "base_model.model.model.embed_tokens.lora_B.weight",
    ]) == []
