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

"""Unit tests for the LoRA-save key remap and the Step-7 backed-module count in
unsloth_zoo/saving_utils.py.

Covers the VLM prefix reorder fix, its guards against misrouting a vision LoRA
onto a language tensor, and the backed-module count across model families
(normal, reordered VLM, Gemma4 ClippableLinear, fused/per-expert MoE, tied
embeddings, and a vision tower absent from the base safetensors). Pure-function,
CPU-only, no disk or model download.
"""

from __future__ import annotations

import collections

from unsloth_zoo.saving_utils import (
    LoraStats,
    _infer_prefix_and_remap,
    _count_backed_lora_modules,
)


def _lw(keys):
    """Build a lora_weights defaultdict; value is the key so we can trace remaps."""
    d = collections.defaultdict(lambda: LoraStats(None, None, None, 0))
    for k in keys:
        d[k] = k
    return d


# ---------------------------------------------------------------------------
# _infer_prefix_and_remap: reorder fix + misroute guards
# ---------------------------------------------------------------------------

def test_remap_reorders_ministral_language_keys():
    """model.language_model.* LoRA keys remap onto language_model.model.* tensors."""
    disk = [f"language_model.model.layers.0.self_attn.{p}.weight" for p in ("q_proj", "k_proj")]
    out = _infer_prefix_and_remap(
        _lw([f"model.language_model.layers.0.self_attn.{p}" for p in ("q_proj", "k_proj")]),
        disk,
    )
    assert out is not None
    for p in ("q_proj", "k_proj"):
        tgt = f"language_model.model.layers.0.self_attn.{p}"
        assert out.get(tgt) == f"model.language_model.layers.0.self_attn.{p}"


def test_remap_returns_none_when_already_aligned():
    disk = ["model.layers.0.self_attn.q_proj.weight"]
    assert _infer_prefix_and_remap(_lw(["model.layers.0.self_attn.q_proj"]), disk) is None


def test_remap_does_not_misroute_vision_onto_language_tensor():
    """A vision LoRA sharing a trailing path with a language tensor must not be
    rewritten onto that language tensor when only the language tensor exists."""
    disk = ["language_model.model.layers.0.self_attn.q_proj.weight"]
    out = _infer_prefix_and_remap(
        _lw([
            "model.language_model.layers.0.self_attn.q_proj",  # legitimate language target
            "model.vision_tower.layers.0.self_attn.q_proj",    # must NOT claim the language tensor
        ]),
        disk,
    )
    tgt = "language_model.model.layers.0.self_attn.q_proj"
    assert out.get(tgt) == "model.language_model.layers.0.self_attn.q_proj"


def test_remap_qwen35_extra_prefix_still_works():
    """Extra-prefix remap (model.* -> model.language_model.*) is unaffected."""
    disk = ["model.language_model.layers.0.self_attn.q_proj.weight"]
    out = _infer_prefix_and_remap(_lw(["model.layers.0.self_attn.q_proj"]), disk)
    assert out is not None
    assert out.get("model.language_model.layers.0.self_attn.q_proj") == "model.layers.0.self_attn.q_proj"


# ---------------------------------------------------------------------------
# _count_backed_lora_modules: count matches what the merge loop would write
# ---------------------------------------------------------------------------

def _count(keys, disk_keys, model_class_name="PreTrainedModel", tie=False):
    return _count_backed_lora_modules(_lw(keys), set(disk_keys), model_class_name, tie)


def test_count_normal_aligned():
    keys = [f"model.layers.0.self_attn.{p}" for p in ("q_proj", "k_proj", "v_proj")]
    disk = [k + ".weight" for k in keys]
    assert _count(keys, disk) == 3


def test_count_reordered_vlm():
    keys = [f"model.language_model.layers.0.self_attn.{p}" for p in ("q_proj", "k_proj", "v_proj")]
    disk = [f"language_model.model.layers.0.self_attn.{p}.weight" for p in ("q_proj", "k_proj", "v_proj")]
    assert _count(keys, disk) == 3


def test_count_gemma4_clippable_linear():
    keys = [f"model.layers.0.mlp.{p}" for p in ("gate_proj", "up_proj", "down_proj")]
    disk = [f"model.layers.0.mlp.{p}.linear.weight" for p in ("gate_proj", "up_proj", "down_proj")]
    assert _count(keys, disk) == 3


def test_count_moe_per_expert():
    keys = ["model.layers.0.mlp.experts.base_layer", "model.layers.0.mlp.experts"]
    disk = [f"model.layers.0.mlp.experts.{e}.{p}.weight" for e in (0, 1) for p in ("gate_proj", "up_proj", "down_proj")]
    assert _count(keys, disk) == 2


def test_count_vision_tower_unbacked_excluded():
    """A vision tower LoRA absent from the base safetensors is not counted."""
    keys = [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.vision_tower.transformer.layers.0.attention.q_proj",  # no backing tensor
    ]
    disk = ["language_model.model.layers.0.self_attn.q_proj.weight"]
    assert _count(keys, disk) == 1


def test_count_tied_embed_and_bare_lm_head():
    """Tied model, modules_to_save on model.embed_tokens and bare lm_head, only
    embed_tokens.weight on disk. lm_head resolves to embed_tokens via the model.
    add/strip, so both count; the Step-7 tied discount then makes it match the one
    merged tensor."""
    assert _count(["model.embed_tokens", "lm_head"], ["model.embed_tokens.weight"],
                  model_class_name="LlamaForCausalLM", tie=True) == 2


def test_count_tied_bare_lm_head_with_layer():
    """Bare lm_head + a normal layer; lm_head is backed via the tied embed tensor."""
    assert _count(["lm_head", "model.layers.0.self_attn.q_proj"],
                  ["model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight"],
                  model_class_name="LlamaForCausalLM", tie=True) == 2


def test_count_untied_lm_head_on_disk():
    """Untied: lm_head.weight present on disk, counted directly, no tie inflation."""
    assert _count(["lm_head", "model.layers.0.self_attn.q_proj"],
                  ["lm_head.weight", "model.layers.0.self_attn.q_proj.weight"],
                  model_class_name="LlamaForCausalLM", tie=False) == 2
