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

"""Tests for the LoRA-save key remap and backed-module count in saving_utils.py.

Covers the VLM prefix reorder fix, vision-onto-language misroute guards, and the
backed-module count across model families (VLM, Gemma4 ClippableLinear, MoE, tied
embeddings, absent vision tower). Pure-function, CPU-only, no disk/download.
"""

from __future__ import annotations

import collections
import importlib.util
import sys
import types

# Stub bitsandbytes (imported at module scope) for CPU-only runs. Needs a real __spec__
# so find_spec() probes don't raise; built inline so package init can't pull the deps
# the stub avoids.
if importlib.util.find_spec("bitsandbytes") is None:
    from importlib.machinery import ModuleSpec
    _bnb = types.ModuleType("bitsandbytes")
    _bnb.__spec__ = ModuleSpec("bitsandbytes", loader=None, is_package=True)
    _bnb.__path__ = []
    _bnb_nn = types.ModuleType("bitsandbytes.nn")
    _bnb_nn.__spec__ = ModuleSpec("bitsandbytes.nn", loader=None)
    # Subclassable placeholders for older peft `class X(bnb.nn.Y)` import-time subclassing.
    for _cls in ("Linear8bitLt", "Linear4bit", "Int8Params", "Params4bit"):
        setattr(_bnb_nn, _cls, type(_cls, (object,), {}))
    _bnb.nn = _bnb_nn
    sys.modules["bitsandbytes"] = _bnb
    sys.modules["bitsandbytes.nn"] = _bnb_nn

from unsloth_zoo.saving_utils import (  # noqa: E402
    LoraStats,
    _infer_prefix_and_remap,
    _count_backed_lora_modules,
    _get_lora_scaling,
    create_lora_statistics,
)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


def _lw(keys):
    """Build a lora_weights defaultdict whose value is the key, to trace remaps."""
    d = collections.defaultdict(lambda: LoraStats(None, None, None, 0))
    for k in keys:
        d[k] = k
    return d


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
    """Vision LoRA sharing a trailing path with a language tensor isn't rewritten onto it."""
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
    """An extra leading wrapper (model.vision_tower.*) strips onto the base vision_tower.* tensor."""
    disk = ["vision_tower.transformer.layers.0.attention.q_proj.weight"]
    out = _infer_prefix_and_remap(
        _lw(["model.vision_tower.transformer.layers.0.attention.q_proj"]), disk)
    assert out is not None
    tgt = "vision_tower.transformer.layers.0.attention.q_proj"
    assert out.get(tgt) == "model.vision_tower.transformer.layers.0.attention.q_proj"


def test_remap_mixed_language_reorder_and_vision_strip():
    """Composite VLM: language half reorders, vision half strips leading model.; each resolves onto its own tensor."""
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
    # Every remapped target backs a real on-disk tensor; no invented keys.
    for tgt in out:
        if isinstance(tgt, str):
            assert (tgt + ".weight") in set(disk)


# _count_backed_lora_modules: count must match what the merge loop would write.

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
    """Tied embed + bare lm_head, only embed on disk: merge writes embed, drops lm_head, count 1."""
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
    """Composite VLM, tied, deep embed + bare lm_head: lm_head can't bridge the deep embed, count 1."""
    assert _count(["model.language_model.embed_tokens", "lm_head"],
                  ["model.language_model.embed_tokens.weight"],
                  model_class_name="PreTrainedModel", tie=True) == 1


def test_count_composite_tied_bare_lm_head_unbridgeable_dropped():
    """Composite VLM, tied: a bare lm_head can't bridge a deep-prefix embed, so it's dropped (count 0)."""
    assert _count(["lm_head"], ["model.language_model.embed_tokens.weight"],
                  model_class_name="PreTrainedModel", tie=True) == 0


def test_count_mxfp4_packed_experts():
    """MXFP4 fused experts (packed _blocks/_scales, no .weight) are dequantized on save, so counted."""
    keys = ["model.layers.0.mlp.experts.gate_up_proj", "model.layers.0.mlp.experts.down_proj",
            "model.layers.0.self_attn.q_proj"]
    disk = ["model.layers.0.mlp.experts.gate_up_proj_blocks", "model.layers.0.mlp.experts.gate_up_proj_scales",
            "model.layers.0.mlp.experts.down_proj_blocks", "model.layers.0.mlp.experts.down_proj_scales",
            "model.layers.0.self_attn.q_proj.weight"]
    assert _count(keys, disk) == 3


def test_count_mxfp4_blocks_without_scales_not_counted():
    """A packed module missing its _scales can't be dequantized, so it's not counted."""
    keys = ["model.layers.0.mlp.experts.gate_up_proj", "model.layers.0.self_attn.q_proj"]
    disk = ["model.layers.0.mlp.experts.gate_up_proj_blocks",  # no matching _scales
            "model.layers.0.self_attn.q_proj.weight"]
    assert _count(keys, disk) == 1


def test_count_cross_namespace_same_trailing_path_union():
    """Sharded VLM: two namespaces share a trailing path; against the key union each resolves (count 2)."""
    keys = ["model.language_model.layers.0.self_attn.q_proj",
            "model.vision_tower.transformer.layers.0.self_attn.q_proj"]
    union = ["language_model.model.layers.0.self_attn.q_proj.weight",
             "vision_tower.transformer.layers.0.self_attn.q_proj.weight"]
    assert _count(keys, union) == 2


def test_remap_cross_namespace_union_resolves_both():
    """Given both shards' key union, the remap routes language and vision keys without crossing namespaces."""
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
    """Common-prefix fallback must prefix MoE keys backed by descendant expert tensors, not only direct .weight."""
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
    """Fused 3D MoE (GPT-OSS/Gemma4) stores experts with no .weight; the merge handles them, so counted."""
    keys = ["model.layers.0.experts.base_layer", "model.layers.0.experts"]
    disk = ["model.layers.0.experts.gate_up_proj", "model.layers.0.experts.down_proj"]
    assert _count(keys, disk) == 2


def test_count_moe_alias_resolves_to_fused_experts():
    """A .moe LoRA key is aliased to .experts by the merge, so it counts against the fused .experts tensors."""
    keys = ["model.layers.0.moe.base_layer", "model.layers.0.moe"]
    disk = ["model.layers.0.experts.gate_up_proj", "model.layers.0.experts.down_proj"]
    assert _count(keys, disk) == 2


def test_count_mxfp4_per_expert_packed():
    """Per-expert packed mxfp4 experts (descendant <prefix>.<e>.<proj>_blocks/_scales) count as backed."""
    keys = ["model.layers.0.mlp.experts.base_layer"]
    disk = ["model.layers.0.mlp.experts.0.gate_proj_blocks", "model.layers.0.mlp.experts.0.gate_proj_scales"]
    assert _count(keys, disk) == 1


def test_remap_reordered_clippable_linear():
    """Reordered VLM stored as Gemma4 ClippableLinear (<module>.linear.weight) still learns the prefix and is backed."""
    disk = ["language_model.model.layers.0.mlp.gate_proj.linear.weight"]
    out = _infer_prefix_and_remap(_lw(["model.language_model.layers.0.mlp.gate_proj"]), disk)
    assert out is not None
    assert out["language_model.model.layers.0.mlp.gate_proj"] == \
        "model.language_model.layers.0.mlp.gate_proj"
    assert _count(["model.language_model.layers.0.mlp.gate_proj"], disk) == 1


def test_remap_reordered_linear_applied_via_seeded_vote():
    """A vote seeded by a .weight key, applied to a sibling .linear.weight module, accepts the .linear backing."""
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
    """A reordered short module (embed_tokens) alone still learns its reorder via a guarded short-suffix vote."""
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
    """A short-suffix match that is a prefix add, not a reorder, must not vote (no cross-namespace misroute)."""
    out = _infer_prefix_and_remap(
        _lw(["model.vision_tower.embed_tokens", "model.language_model.layers.0.self_attn.q_proj"]),
        ["language_model.model.layers.0.self_attn.q_proj.weight"])  # no vision embed tensor
    # The vision embed has no backing tensor; it must stay unmapped, not be pulled onto the language tensor.
    assert out.get("language_model.model.layers.0.self_attn.q_proj") == \
        "model.language_model.layers.0.self_attn.q_proj"
    assert "model.vision_tower.embed_tokens" not in out or \
        out["model.vision_tower.embed_tokens"] == "model.vision_tower.embed_tokens"


def test_remap_prefix_add_onto_linear_weight():
    """A prefix-add key backed only by a ClippableLinear .linear.weight still remaps via the unique-prefix path."""
    out = _infer_prefix_and_remap(_lw(["model.layers.0.mlp.gate_proj"]),
                                  ["model.language_model.layers.0.mlp.gate_proj.linear.weight"])
    assert out is not None
    assert out["model.language_model.layers.0.mlp.gate_proj"] == "model.layers.0.mlp.gate_proj"


def test_remap_strip_does_not_drop_semantic_namespace_to_bare_layers():
    """Wrapper-strip removes only generic wrappers, never a semantic namespace onto a bare language tensor."""
    out = _infer_prefix_and_remap(
        _lw(["model.vision_tower.layers.0.self_attn.q_proj"]),
        ["layers.0.self_attn.q_proj.weight"],
    )
    assert out is None or out.get("model.vision_tower.layers.0.self_attn.q_proj") == \
        "model.vision_tower.layers.0.self_attn.q_proj"
    assert not (out and out.get("layers.0.self_attn.q_proj") ==
                "model.vision_tower.layers.0.self_attn.q_proj")


def test_remap_strip_still_drops_leading_model_wrapper():
    """The legitimate case still works: model.vision_tower.* strips onto the real vision_tower.* tensor."""
    out = _infer_prefix_and_remap(
        _lw(["model.vision_tower.transformer.layers.0.attention.q_proj"]),
        ["vision_tower.transformer.layers.0.attention.q_proj.weight"])
    assert out is not None
    assert out["vision_tower.transformer.layers.0.attention.q_proj"] == \
        "model.vision_tower.transformer.layers.0.attention.q_proj"


def test_count_fused_named_gate_up_proj_backed_by_per_expert():
    """A fused-named key ...experts.gate_up_proj is backed by per-expert descendants, so it counts."""
    keys = ["model.layers.0.mlp.experts.gate_up_proj"]
    disk = ["model.layers.0.mlp.experts.0.gate_proj.weight", "model.layers.0.mlp.experts.0.up_proj.weight",
            "model.layers.0.mlp.experts.1.gate_proj.weight", "model.layers.0.mlp.experts.1.up_proj.weight"]
    assert _count(keys, disk) == 1


def test_count_native_mxfp4_does_not_count_packed():
    """Native mxfp4 save preserves packed experts without merging, so count_packed_mxfp4=False -> 0, True -> 1."""
    keys = ["model.layers.0.mlp.experts.gate_up_proj"]
    disk = ["model.layers.0.mlp.experts.gate_up_proj_blocks", "model.layers.0.mlp.experts.gate_up_proj_scales"]
    assert _count_backed_lora_modules(_lw(keys), set(disk), "PreTrainedModel", False,
                                      count_packed_mxfp4=False) == 0
    assert _count_backed_lora_modules(_lw(keys), set(disk), "PreTrainedModel", False,
                                      count_packed_mxfp4=True) == 1


# _get_lora_scaling + create_lora_statistics scaling-count alignment (#2966).

class _Scalar:
    def __getitem__(self, k): return 4.0

def test_get_lora_scaling_active_adapters_plural():
    m = types.SimpleNamespace(active_adapters=["default"], scaling=_Scalar())
    assert _get_lora_scaling(m) == 4.0

def test_get_lora_scaling_active_adapter_singular():
    """Older peft exposes only the singular active_adapter."""
    m = types.SimpleNamespace(active_adapter="default", scaling=_Scalar())
    assert _get_lora_scaling(m) == 4.0

def test_get_lora_scaling_unresolved_returns_zero():
    m = types.SimpleNamespace(active_adapter="missing", scaling={})
    assert _get_lora_scaling(m) == 0.0


class _FakeLoRALinear(nn.Module):
    """LoRA wrapper not surfaced by get_lora_layer_modules(); singular active_adapter only (#2966)."""
    def __init__(self, n=8, r=4, alpha=8):
        super().__init__()
        self.base_layer = nn.Linear(n, n, bias=False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(n, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, n, bias=False)})
        self.scaling = {"default": alpha / r}
        self.active_adapter = "default"  # singular only

class _Inner(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.layers = nn.ModuleList([_FakeLoRALinear() for _ in range(n)])

class _PeftLike(nn.Module):
    def __init__(self, inner):
        super().__init__()
        class _BM(nn.Module):
            def __init__(self, m):
                super().__init__(); self.model = m
        self.base_model = _BM(inner)

def test_create_lora_statistics_counts_align_for_unmatched_class():
    """Class-mismatched LoRA layers must capture scaling so the delta isn't merged with alpha = 0 (#2966)."""
    model = _PeftLike(_Inner(3))
    lora_weights, _ = create_lora_statistics(model, merge_into_original=True, return_state_dict=False)
    alphas = [v.alpha for v in lora_weights.values() if v.lora_A is not None]
    assert len(alphas) == 3
    assert all(abs(a - 2.0) < 1e-9 for a in alphas)


def test_non_lora_scaled_module_not_misclassified():
    """A module with `scaling` + singular active_adapter but no LoRA tensors
    (e.g. an attention scale) must not be captured as a LoRA wrapper (#806)."""
    inner = _Inner(2)  # two real LoRA layers (have lora_A/lora_B)
    non_lora = nn.Linear(8, 8, bias=False)
    non_lora.scaling = 0.125
    non_lora.active_adapter = "default"  # singular, but no lora_A/lora_B
    inner.add_module("attn_like", non_lora)
    model = _PeftLike(inner)

    lora_weights, _ = create_lora_statistics(model, merge_into_original=True, return_state_dict=False)
    # Real LoRA layers captured; non-LoRA scaled module not (its weight is not dropped).
    assert sum(1 for v in lora_weights.values() if v.lora_A is not None) == 2
    assert not any(k.endswith("attn_like") for k in lora_weights)
