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


def test_remap_strips_extra_wrapper_prefix_onto_base_tensor():
    """A LoRA carrying an extra leading wrapper (model.vision_tower.*) strips onto the
    base tensor (vision_tower.*) when that tensor exists on disk."""
    disk = ["vision_tower.transformer.layers.0.attention.q_proj.weight"]
    out = _infer_prefix_and_remap(
        _lw(["model.vision_tower.transformer.layers.0.attention.q_proj"]), disk)
    assert out is not None
    tgt = "vision_tower.transformer.layers.0.attention.q_proj"
    assert out.get(tgt) == "model.vision_tower.transformer.layers.0.attention.q_proj"


def test_remap_mixed_language_reorder_and_vision_strip():
    """Composite VLM where the language half reorders (model.language_model.* ->
    language_model.model.*) and the vision half only drops the leading model.. Both
    namespaces resolve onto their own base tensors, neither onto a nonexistent key."""
    disk = (
        [f"language_model.model.layers.0.self_attn.{p}.weight" for p in ("q_proj", "k_proj")]
        + [f"vision_tower.transformer.layers.0.attention.{p}.weight" for p in ("q_proj", "k_proj")]
    )
    lora = (
        [f"model.language_model.layers.0.self_attn.{p}" for p in ("q_proj", "k_proj")]
        + [f"model.vision_tower.transformer.layers.0.attention.{p}" for p in ("q_proj", "k_proj")]
    )
    out = _infer_prefix_and_remap(_lw(lora), disk)
    assert out is not None
    for p in ("q_proj", "k_proj"):
        assert out.get(f"language_model.model.layers.0.self_attn.{p}") == \
            f"model.language_model.layers.0.self_attn.{p}"
        assert out.get(f"vision_tower.transformer.layers.0.attention.{p}") == \
            f"model.vision_tower.transformer.layers.0.attention.{p}"
    # No invented keys: every remapped target backs a real on-disk tensor.
    for tgt in out:
        if isinstance(tgt, str):
            assert (tgt + ".weight") in set(disk)


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
    embed_tokens.weight on disk. The merge writes the embed delta and drops lm_head
    (the tied bridge fires only when embed is not itself a target), so exactly one
    tensor is merged and the merge-accurate count is 1, with no separate discount."""
    assert _count(["model.embed_tokens", "lm_head"], ["model.embed_tokens.weight"],
                  model_class_name="LlamaForCausalLM", tie=True) == 1


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


def test_count_composite_tied_embed_and_bare_lm_head():
    """Composite VLM (deep model.language_model. prefix), tied, modules_to_save on the
    deep embed_tokens and a bare lm_head, only the deep embed tensor on disk. The merge
    writes the embed delta and cannot bridge the bare lm_head onto the deep-prefix embed,
    so it merges one tensor and the count must mirror that as 1. The previous
    discount-based scheme under-counted this to 0 and raised a false RuntimeError."""
    assert _count(["model.language_model.embed_tokens", "lm_head"],
                  ["model.language_model.embed_tokens.weight"],
                  model_class_name="PreTrainedModel", tie=True) == 1


def test_count_composite_tied_bare_lm_head_unbridgeable_dropped():
    """Composite VLM, tied, a bare lm_head whose embed lives only at a deep prefix. The
    merge cannot bridge a bare lm_head onto model.language_model.embed_tokens.weight, so
    it is dropped and contributes 0 (the standard model.embed_tokens case still bridges
    and counts, see test_count_tied_bare_lm_head_with_layer)."""
    assert _count(["lm_head"], ["model.language_model.embed_tokens.weight"],
                  model_class_name="PreTrainedModel", tie=True) == 0


def test_count_mxfp4_packed_experts():
    """MXFP4 base: fused experts are stored packed as <module>_blocks / _scales and the
    merge dequantizes them to <module> on save, so a LoRA on such a module has no .weight
    on disk. It must still be counted as backed (the merge writes and counts it)."""
    keys = ["model.layers.0.mlp.experts.gate_up_proj", "model.layers.0.mlp.experts.down_proj",
            "model.layers.0.self_attn.q_proj"]
    disk = ["model.layers.0.mlp.experts.gate_up_proj_blocks", "model.layers.0.mlp.experts.gate_up_proj_scales",
            "model.layers.0.mlp.experts.down_proj_blocks", "model.layers.0.mlp.experts.down_proj_scales",
            "model.layers.0.self_attn.q_proj.weight"]
    assert _count(keys, disk) == 3


def test_count_mxfp4_blocks_without_scales_not_counted():
    """A packed module missing its _scales tensor is skipped by the merge (it cannot be
    dequantized), so it must not be counted as backed."""
    keys = ["model.layers.0.mlp.experts.gate_up_proj", "model.layers.0.self_attn.q_proj"]
    disk = ["model.layers.0.mlp.experts.gate_up_proj_blocks",  # no matching _scales
            "model.layers.0.self_attn.q_proj.weight"]
    assert _count(keys, disk) == 1


def test_count_cross_namespace_same_trailing_path_union():
    """Sharded composite checkpoint where two namespaces (language reorder + vision strip)
    share the same trailing module path but live in different shards. Counting against the
    union of all shard keys must still resolve each LoRA onto its own tensor (count 2), not
    collapse them into an ambiguous unbacked pair."""
    keys = ["model.language_model.layers.0.self_attn.q_proj",
            "model.vision_tower.transformer.layers.0.self_attn.q_proj"]
    union = ["language_model.model.layers.0.self_attn.q_proj.weight",
             "vision_tower.transformer.layers.0.self_attn.q_proj.weight"]
    assert _count(keys, union) == 2


def test_remap_cross_namespace_union_resolves_both():
    """The remap, given the union of both shards' keys, routes the language key to its
    reordered tensor and the vision key to its stripped tensor, never crossing namespaces."""
    union = ["language_model.model.layers.0.self_attn.q_proj.weight",
             "vision_tower.transformer.layers.0.self_attn.q_proj.weight"]
    out = _infer_prefix_and_remap(
        _lw(["model.language_model.layers.0.self_attn.q_proj",
             "model.vision_tower.transformer.layers.0.self_attn.q_proj"]),
        union,
    )
    assert out is not None
    assert out.get("language_model.model.layers.0.self_attn.q_proj") == \
        "model.language_model.layers.0.self_attn.q_proj"
    assert out.get("vision_tower.transformer.layers.0.self_attn.q_proj") == \
        "model.vision_tower.transformer.layers.0.self_attn.q_proj"


def test_remap_common_prefix_still_applies_to_moe_expert_backing():
    """The common-prefix fallback must prefix MoE LoRA keys that are backed by descendant
    expert tensors (model.layers.0.mlp.experts.0.*.weight), not only keys with a direct
    <key>.weight. Otherwise the prefixed key is dropped and the MoE delta is skipped."""
    disk = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight",
    ]
    out = _infer_prefix_and_remap(
        _lw([
            "layers.0.self_attn.q_proj",
            "layers.0.mlp.experts.base_layer",
            "layers.0.mlp.experts",
        ]),
        disk,
    )
    assert out is not None
    assert out["model.layers.0.mlp.experts.base_layer"] == "layers.0.mlp.experts.base_layer"
    assert out["model.layers.0.mlp.experts"] == "layers.0.mlp.experts"


def test_count_fused_3d_experts_without_weight_suffix():
    """Fused 3D MoE (GPT-OSS / Gemma4) stores experts as <prefix>.gate_up_proj /
    <prefix>.down_proj with no .weight; _merge_moe_experts_file merges them, so they must
    be counted as backed."""
    keys = ["model.layers.0.experts.base_layer", "model.layers.0.experts"]
    disk = ["model.layers.0.experts.gate_up_proj", "model.layers.0.experts.down_proj"]
    assert _count(keys, disk) == 2


def test_count_moe_alias_resolves_to_fused_experts():
    """A LoRA key using .moe is aliased to .experts by the merge, so it must count against
    the fused expert tensors stored under .experts."""
    keys = ["model.layers.0.moe.base_layer", "model.layers.0.moe"]
    disk = ["model.layers.0.experts.gate_up_proj", "model.layers.0.experts.down_proj"]
    assert _count(keys, disk) == 2


def test_count_mxfp4_per_expert_packed():
    """Per-expert packed mxfp4 experts (descendant <prefix>.<e>.<proj>_blocks/_scales)
    count as backed."""
    keys = ["model.layers.0.mlp.experts.base_layer"]
    disk = ["model.layers.0.mlp.experts.0.gate_proj_blocks", "model.layers.0.mlp.experts.0.gate_proj_scales"]
    assert _count(keys, disk) == 1


def test_remap_reordered_clippable_linear():
    """A reordered composite VLM whose module is stored as Gemma4 ClippableLinear
    (<module>.linear.weight) must still learn the prefix substitution (the vote normalizes
    .linear.weight to its module key) and accept the .linear-backed target."""
    disk = ["language_model.model.layers.0.mlp.gate_proj.linear.weight"]
    out = _infer_prefix_and_remap(_lw(["model.language_model.layers.0.mlp.gate_proj"]), disk)
    assert out is not None
    assert out["language_model.model.layers.0.mlp.gate_proj"] == \
        "model.language_model.layers.0.mlp.gate_proj"
    assert _count(["model.language_model.layers.0.mlp.gate_proj"], disk) == 1


def test_remap_reordered_linear_applied_via_seeded_vote():
    """Vote seeded by a .weight attention key, then applied to a sibling .linear.weight
    module: the application must accept the .linear backing, not only direct .weight."""
    disk = ["language_model.model.layers.0.self_attn.q_proj.weight",
            "language_model.model.layers.0.mlp.gate_proj.linear.weight"]
    out = _infer_prefix_and_remap(
        _lw(["model.language_model.layers.0.self_attn.q_proj",
             "model.language_model.layers.0.mlp.gate_proj"]),
        disk,
    )
    assert out is not None
    assert out["language_model.model.layers.0.mlp.gate_proj"] == \
        "model.language_model.layers.0.mlp.gate_proj"


def test_remap_short_reordered_embed_tokens_alone():
    """A reordered short module (embed_tokens) on a shard that holds no longer layer keys
    must still learn its own reorder via a short-suffix vote, guarded so it only fires for a
    true component reordering. Otherwise the merge silently drops the embedding delta."""
    out = _infer_prefix_and_remap(_lw(["model.language_model.embed_tokens"]),
                                  ["language_model.model.embed_tokens.weight"])
    assert out is not None
    assert out["language_model.model.embed_tokens"] == "model.language_model.embed_tokens"


def test_remap_short_reordered_lm_head_alone():
    out = _infer_prefix_and_remap(_lw(["model.language_model.lm_head"]),
                                  ["language_model.model.lm_head.weight"])
    assert out is not None
    assert out["language_model.model.lm_head"] == "model.language_model.lm_head"


def test_remap_short_suffix_does_not_cross_namespaces():
    """A short suffix match that is NOT a component reordering must not vote (no misroute).
    model.embed_tokens vs language_model.model.embed_tokens is a prefix add, not a reorder;
    it resolves by the unique-prefix path, never by a cross-namespace substitution."""
    out = _infer_prefix_and_remap(
        _lw(["model.vision_tower.embed_tokens", "model.language_model.layers.0.self_attn.q_proj"]),
        ["language_model.model.layers.0.self_attn.q_proj.weight"])  # no vision embed tensor
    # the vision embed has no backing tensor and must be left unmapped, never pulled onto
    # the language tensor by a 1-component 'embed_tokens'/'q_proj' style vote.
    assert out.get("language_model.model.layers.0.self_attn.q_proj") == \
        "model.language_model.layers.0.self_attn.q_proj"
    assert "model.vision_tower.embed_tokens" not in out or \
        out["model.vision_tower.embed_tokens"] == "model.vision_tower.embed_tokens"


def test_remap_prefix_add_onto_linear_weight():
    """A prefix-add LoRA key whose only backing is a Gemma4 ClippableLinear .linear.weight
    tensor must still remap via the unique-prefix path (not only direct .weight)."""
    out = _infer_prefix_and_remap(_lw(["model.layers.0.mlp.gate_proj"]),
                                  ["model.language_model.layers.0.mlp.gate_proj.linear.weight"])
    assert out is not None
    assert out["model.language_model.layers.0.mlp.gate_proj"] == "model.layers.0.mlp.gate_proj"


def test_count_fused_named_gate_up_proj_backed_by_per_expert():
    """A fused-named LoRA key ...experts.gate_up_proj is backed by per-expert descendants
    ...experts.<e>.gate_proj.weight (the merge maps them onto it), so it must count."""
    keys = ["model.layers.0.mlp.experts.gate_up_proj"]
    disk = ["model.layers.0.mlp.experts.0.gate_proj.weight", "model.layers.0.mlp.experts.0.up_proj.weight",
            "model.layers.0.mlp.experts.1.gate_proj.weight", "model.layers.0.mlp.experts.1.up_proj.weight"]
    assert _count(keys, disk) == 1


def test_count_native_mxfp4_does_not_count_packed():
    """Native mxfp4 save (save_method='mxfp4') preserves _blocks/_scales without merging,
    so a LoRA on a packed expert is not written and must not be counted (count_packed_mxfp4
    =False). The dequantizing path (default True) does count it, since it merges."""
    keys = ["model.layers.0.mlp.experts.gate_up_proj"]
    disk = ["model.layers.0.mlp.experts.gate_up_proj_blocks", "model.layers.0.mlp.experts.gate_up_proj_scales"]
    assert _count_backed_lora_modules(_lw(keys), set(disk), "PreTrainedModel", False,
                                      count_packed_mxfp4=False) == 0
    assert _count_backed_lora_modules(_lw(keys), set(disk), "PreTrainedModel", False,
                                      count_packed_mxfp4=True) == 1
