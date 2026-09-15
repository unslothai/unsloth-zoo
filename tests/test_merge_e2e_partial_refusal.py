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

"""A composite vision-language merge must either merge every tower or refuse (#5290).

A Qwen2.5-VL checkpoint is written in the pre-5 flat layout (`model.layers.N...`,
`visual...`) while the runtime module paths PEFT adapts are the composite ones
(`model.language_model.layers.N...`, `model.visual...`). The bridge between the two
used to come from `<ModelClass>._checkpoint_conversion_mapping`, which Transformers 5
moved into a registry and removed from the class, and without it every language-tower
adapter resolved to nothing: the merge wrote the base weights back unchanged and
reported success, because a target that resolves to nothing is dropped from BOTH sides
of the merge's count check and the mismatch cancels out.

Two things are asserted here. The language tower merges, and when the bridge is
unavailable for any reason the merge refuses instead of writing a checkpoint that looks
trained and is not. A tiny (roughly 100k parameter) real Qwen2.5-VL is used, so the
whole file is CPU-only and needs no network.
"""

from __future__ import annotations

import os
import collections

import pytest
import torch

import _merge_e2e_helpers as H

from unsloth_zoo.saving_utils import (
    LoraStats,
    PartialLoraMergeError,
    _unresolved_lora_targets,
    _get_checkpoint_conversion_mapping,
)
import unsloth_zoo.saving_utils as SU


_VL_FAMILY = "qwen2_5_vl"
_VL_CLASS = "Qwen2_5_VLForConditionalGeneration"

_H = 32
_I = 64


def _skip_unless_vl_available():
    if not H.family_available(_VL_FAMILY):
        pytest.skip(f"{_VL_FAMILY} unavailable in this transformers")


def _build_tiny_vl(base_dir, dtype=torch.float32):
    """A tiny real Qwen2.5-VL, saved to `base_dir`. Returns the model.

    Whether `save_pretrained` writes the flat layout or the composite one is
    Transformers' business, not this test's: the point of the test is that the merge
    copes with whatever it wrote, so the layout is read back and asserted on rather
    than assumed.
    """
    import transformers as T

    cfg = T.AutoConfig.for_model(_VL_FAMILY)
    text_cfg = cfg.text_config
    vision_cfg = cfg.vision_config
    for obj, values in (
        (text_cfg, dict(hidden_size = _H, intermediate_size = _I, num_hidden_layers = 2,
                        num_attention_heads = 4, num_key_value_heads = 2, vocab_size = 64,
                        max_position_embeddings = 64, head_dim = 8,
                        tie_word_embeddings = False)),
        (vision_cfg, dict(hidden_size = _H, intermediate_size = _I, depth = 2,
                          num_heads = 4, out_hidden_size = _H)),
    ):
        for key, value in values.items():
            if hasattr(obj, key): setattr(obj, key, value)
    cfg.tie_word_embeddings = False

    torch.manual_seed(H.SEED)
    model = getattr(T, _VL_CLASS)._from_config(cfg).to(dtype)
    model.save_pretrained(base_dir, safe_serialization = True)
    model.config._name_or_path = base_dir
    return model


def _attach_vl_lora(model):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(H.SEED)
    peft_model = get_peft_model(model, LoraConfig(
        r = 8, lora_alpha = 16, lora_dropout = 0.0, bias = "none",
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
    ))
    H.seed_lora(peft_model)
    return peft_model


def _stage_tiny_vl(work_dir):
    """`(base_dir, out_dir, base_tensors, peft_model, adapted)` ready for a merge."""
    H.set_offline_cpu_env()
    base_dir = os.path.join(work_dir, "base")
    out_dir  = os.path.join(work_dir, "merged")
    model = _build_tiny_vl(base_dir)
    base_tensors = H.read_safetensors_dir(base_dir)
    peft_model = _attach_vl_lora(model)
    adapted = H.extract_adapted(peft_model)
    return base_dir, out_dir, base_tensors, peft_model, adapted


# The bridge this test asserts on, spelled out rather than rediscovered: Qwen2.5-VL
# adapts `model.language_model.*` and `model.visual.*` while the checkpoint is written
# flat. Stated here so the reference is independent of the resolution under test, and
# tried after the identity so a Transformers release that writes the composite layout
# on disk needs no change.
_EXPECTED_DISK_RENAMES = (
    ("model.language_model.", "model."),
    ("model.visual.",        "visual."),
)


def _resolve_composite_key(adapted_key, base_keys):
    """The on-disk key an adapted module path refers to, or None if there is none."""
    if adapted_key in base_keys: return adapted_key
    for lora_prefix, disk_prefix in _EXPECTED_DISK_RENAMES:
        if not adapted_key.startswith(lora_prefix): continue
        candidate = disk_prefix + adapted_key[len(lora_prefix) :]
        if candidate in base_keys: return candidate
    return None


def _assert_merge_values(base_tensors, merged, adapted):
    """Every adapted tensor equals base + scale * (B @ A); every other one is unchanged.

    An independent reference in float64, not a re-run of the merge's own arithmetic.
    Returns (adapted checked, pass-through checked).
    """
    base_keys = set(base_tensors)
    resolved = {}
    for adapted_key, record in adapted.items():
        disk_key = _resolve_composite_key(adapted_key, base_keys)
        assert disk_key is not None, (
            f"adapted key {adapted_key!r} does not resolve to one base key"
        )
        resolved[disk_key] = record

    n_adapted = n_passthrough = 0
    for key, tensor in merged.items():
        assert key in base_tensors, f"merged has unexpected key {key!r}"
        if key in resolved:
            record = resolved[key]
            reference = (
                base_tensors[key].to(torch.float64)
                + record.alpha * (record.lora_B.to(torch.float64) @ record.lora_A.to(torch.float64))
            )
            torch.testing.assert_close(
                tensor.to(torch.float64), reference, atol = 1e-5, rtol = 1e-5,
                msg = lambda built, key = key: f"{key} is not base + scale * (B @ A)\n{built}",
            )
            n_adapted += 1
        else:
            assert torch.equal(tensor, base_tensors[key]), (
                f"{key} carries no adapter and must be byte-identical to the base"
            )
            n_passthrough += 1
    return n_adapted, n_passthrough


def _language_weight_keys(base_tensors):
    return sorted(
        key for key in base_tensors
        if ".layers." in key and not key.startswith("visual.") and ".visual." not in key
        and key.endswith(".weight")
    )


# --------------------------------------------------------------------------------------
# The bridge itself


def test_conversion_mapping_available_for_composite_vl():
    """The disk-to-runtime renamings must be reachable however Transformers holds them.

    Transformers 4 kept them on the class, Transformers 5 in its conversion registry,
    keyed by model type in the early 5.x releases and by class name from 5.10. An empty
    mapping here is the root cause of #5290, so it is asserted directly.

    Only the language rename is required. The vision one is genuinely absent from the
    registry on some releases (5.5 ships the language rename alone), and the vision
    tower is reachable by prefix inference anyway; the language tower is not, which is
    why its absence was the silent half of the bug.
    """
    _skip_unless_vl_available()
    mapping = _get_checkpoint_conversion_mapping(_VL_CLASS)
    assert mapping, (
        f"no checkpoint conversion mapping for {_VL_CLASS}: the language tower cannot "
        f"be bridged to the flat on-disk layout"
    )
    targets = set(mapping.values())
    assert any(t.startswith("model.language_model") for t in targets), mapping


def test_partial_mapping_does_not_shadow_prefix_inference():
    """A mapping that reaches nothing must lose to the inference that reads the layout.

    Gemma 3 is the case: it is written as `language_model.model.layers...` while the
    renamings registered at its class only say `^language_model -> model.language_model`,
    because the remaining hop lives on a nested sub-model. Taking that mapping literally
    resolves the adapter to `language_model.layers...`, which is in no shard, so the
    conversion has to fall back rather than trust a mapping that lands nowhere.
    """
    if not H.family_available("gemma3"):
        pytest.skip("gemma3 unavailable in this transformers")
    lora_keys = [f"model.language_model.layers.{i}.self_attn.q_proj" for i in range(2)]
    disk_keys = [f"language_model.model.layers.{i}.self_attn.q_proj.weight" for i in range(2)]
    disk_keys += [f"language_model.model.layers.{i}.mlp.down_proj.weight" for i in range(2)]

    converted = SU._convert_lora_keys_to_safetensor_format(
        _fake_lora_weights(lora_keys, (32, 32)), disk_keys,
        model_class_name = "Gemma3ForConditionalGeneration",
    )
    for disk_key in disk_keys[:2]:
        module_key = disk_key[: -len(".weight")]
        assert module_key in converted, (
            f"{module_key} is on disk but no adapter resolved onto it: {sorted(converted)}"
        )


@pytest.mark.parametrize("pattern, literal", [
    # The Qwen2.5-VL renamings, in every shape Transformers has written them.
    (r"^visual",                                        "visual"),
    (r"^model(?!\.(language_model|visual))",            "model"),
    (r"(?<!_)model(?!\.(language_model|visual))",       "model"),
    # An escaped separator is a separator. Losing it rewrites
    # `model.encoder.layer.0...` to `backboneencoder.layer.0...`, which backs nothing.
    (r"^backbone\.",                                    "backbone."),
    (r"^language_model\.model\.",                       "language_model.model."),
    (r"^embed_out\.",                                   "embed_out."),
    (r"\.ln1\.",                                        ".ln1."),
    # A capture or a class ends the literal; it is not text to substitute.
    (r"blocks\.(\d+)\.",                               "blocks."),
    (r"^model\.(?:(?!language_model\.))(.+)$",           "model."),
    # Nothing literal at all.
    (r"(?!x)",                                          ""),
    (r"(?<!_)",                                         ""),
    ("",                                                ""),
])
def test_pattern_literal_prefix(pattern, literal):
    """A mapping entry is a regex on one side, so reversing it needs the literal head."""
    assert SU._pattern_literal_prefix(pattern) == literal


def test_escaped_separator_in_a_rename_still_resolves(monkeypatch):
    r"""`^backbone\.` to `model.` must reverse to `backbone.`, separator included.

    Truncating at the backslash yields `backbone` and concatenates the adapter's own
    next component onto it, producing a key no shard holds. That is #5290's failure mode
    reached by a different route, and it is reachable on more models now that these
    mappings resolve at all.
    """
    monkeypatch.setattr(SU, "_get_checkpoint_conversion_mapping",
                        lambda name: {r"^backbone\.": "model."})
    disk_keys = [f"backbone.encoder.layer.{i}.attention.query.weight" for i in range(2)]
    lora_keys = [f"model.encoder.layer.{i}.attention.query" for i in range(2)]
    converted = SU._convert_lora_keys_to_safetensor_format(
        _fake_lora_weights(lora_keys, (32, 32)), disk_keys,
        model_class_name = "Sapiens2ForSemanticSegmentation",
    )
    for disk_key in disk_keys:
        assert disk_key[: -len(".weight")] in converted, sorted(
            key for key in converted if isinstance(key, str)
        )


def test_two_renames_onto_one_target_pick_the_one_with_backing(monkeypatch):
    r"""A target reached by several sources must reverse onto the prefix the shards use.

    `hunyuan_vl` registers both `^model\.vit` and `^vit` onto `model.vision_tower`.
    Keying the reverse map by target keeps one of them, and taking the first match
    without asking whether it lands reverses the vision tower onto a prefix the
    checkpoint does not have, which merges nothing and says nothing.
    """
    monkeypatch.setattr(SU, "_get_checkpoint_conversion_mapping", lambda name: {
        r"^model(?!\.(language_model|vit|vision_tower))" : "model.language_model",
        r"^model\.vit"                                    : "model.vision_tower",
        r"^vit"                                           : "model.vision_tower",
    })
    disk_keys = ([f"model.vit.blocks.{i}.attn.qkv.weight" for i in range(2)]
                 + [f"model.layers.{i}.self_attn.q_proj.weight" for i in range(2)])
    lora_keys = ([f"model.vision_tower.blocks.{i}.attn.qkv" for i in range(2)]
                 + [f"model.language_model.layers.{i}.self_attn.q_proj" for i in range(2)])
    converted = SU._convert_lora_keys_to_safetensor_format(
        _fake_lora_weights(lora_keys, (32, 32)), disk_keys,
        model_class_name = "HunyuanVLForConditionalGeneration",
    )
    for disk_key in disk_keys:
        assert disk_key[: -len(".weight")] in converted, sorted(
            key for key in converted if isinstance(key, str)
        )


# --------------------------------------------------------------------------------------
# End to end: every tower merges


def test_vl_merge_covers_language_and_vision_towers(tmp_path):
    """The whole point of #5290: no tower is silently left at its base weights."""
    _skip_unless_vl_available()
    base_dir, out_dir, base_tensors, peft_model, adapted = _stage_tiny_vl(str(tmp_path))

    # The bug only exists when the checkpoint layout differs from the module paths.
    assert any(k.startswith("model.layers.") for k in base_tensors), sorted(base_tensors)[:5]
    assert any(k.startswith("model.language_model.") for k in adapted), sorted(adapted)[:5]

    H.run_merge(peft_model, base_dir, out_dir, save_dtype = torch.float32)
    merged = H.read_safetensors_dir(out_dir)

    language_keys = _language_weight_keys(base_tensors)
    assert language_keys, sorted(base_tensors)[:5]
    changed = [k for k in language_keys if not torch.equal(merged[k], base_tensors[k])]
    assert len(changed) >= len(language_keys) // 2, (
        f"language tower left at base weights: {len(changed)} of {len(language_keys)} "
        f"tensors changed. The adapters resolved to nothing and the merge said nothing."
    )

    vision_keys = sorted(k for k in base_tensors if k.startswith("visual.") and k.endswith(".weight"))
    if vision_keys:
        assert any(not torch.equal(merged[k], base_tensors[k]) for k in vision_keys)

    # Value check, not just "something changed": every adapted tensor equals
    # base + scale * (B @ A) and every other tensor is byte-identical.
    n_adapted, n_passthrough = _assert_merge_values(base_tensors, merged, adapted)
    assert n_adapted >= len(language_keys) // 2 and n_passthrough >= 1


# --------------------------------------------------------------------------------------
# End to end: refusal when the bridge is gone


def test_vl_merge_refuses_when_the_bridge_is_unavailable(tmp_path, monkeypatch):
    """With no conversion mapping the merge must refuse, not write base weights.

    The mapping is removed rather than mocked away at a lower level, because that is
    exactly the state a Transformers release can put Unsloth in: the whole defect is
    one lookup returning nothing.
    """
    _skip_unless_vl_available()
    base_dir, out_dir, base_tensors, peft_model, _ = _stage_tiny_vl(str(tmp_path))
    monkeypatch.setattr(SU, "_get_checkpoint_conversion_mapping", lambda name: {})
    monkeypatch.delenv("UNSLOTH_ALLOW_PARTIAL_LORA_MERGE", raising = False)

    with pytest.raises(PartialLoraMergeError) as excinfo:
        H.run_merge(peft_model, base_dir, out_dir, save_dtype = torch.float32)

    message = str(excinfo.value)
    assert "model.language_model." in message, message
    assert "Refusing to write a partially merged model" in message, message
    assert "UNSLOTH_ALLOW_PARTIAL_LORA_MERGE" in message, message

    # Refused BEFORE the first shard was rewritten: whatever is staged is still the
    # base checkpoint, byte for byte. A refusal that fired after the loop would leave
    # a half-merged export behind, which is the failure mode being fixed.
    if os.path.isdir(out_dir):
        staged = H.read_safetensors_dir(out_dir)
        for key, tensor in staged.items():
            assert key in base_tensors, key
            assert torch.equal(tensor, base_tensors[key]), (
                f"{key} was rewritten before the refusal"
            )


def test_partial_merge_escape_hatch_downgrades_the_refusal(tmp_path, monkeypatch):
    """A user who understands the consequence can still get the checkpoint out."""
    _skip_unless_vl_available()
    base_dir, out_dir, base_tensors, peft_model, _ = _stage_tiny_vl(str(tmp_path))
    monkeypatch.setattr(SU, "_get_checkpoint_conversion_mapping", lambda name: {})
    monkeypatch.setenv("UNSLOTH_ALLOW_PARTIAL_LORA_MERGE", "1")

    H.run_merge(peft_model, base_dir, out_dir, save_dtype = torch.float32)
    merged = H.read_safetensors_dir(out_dir)
    assert merged
    # Unmerged, as advertised by the message: this is the old behaviour, opted into.
    language_keys = _language_weight_keys(base_tensors)
    assert all(torch.equal(merged[k], base_tensors[k]) for k in language_keys)


# --------------------------------------------------------------------------------------
# The accounting's own edges: what it must NOT report


class _FakeLinear(torch.nn.Module):
    def __init__(self, out_features, in_features):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features


def _fake_lora_weights(keys, shape):
    weights = collections.defaultdict(lambda: LoraStats(None, None, None, 0))
    for key in keys:
        weights[key] = LoraStats(
            module = _FakeLinear(*shape),
            lora_A = torch.zeros(8, shape[1]),
            lora_B = torch.zeros(shape[0], 8),
            alpha  = 1.0,
        )
    return weights


_DISK_SHAPES = {f"model.layers.{i}.self_attn.q_proj": (2048, 2048) for i in range(4)}
_DISK_KEYS = {f"{module}.weight" for module in _DISK_SHAPES}


def _unresolved(lora_weights, model_class_name):
    return _unresolved_lora_targets(
        lora_weights, _DISK_KEYS, _DISK_SHAPES, model_class_name,
    )


def test_accounting_reports_a_missed_prefix_bridge(monkeypatch):
    monkeypatch.setattr(SU, "_get_checkpoint_conversion_mapping", lambda name: {})
    keys = [f"model.language_model.layers.{i}.self_attn.q_proj" for i in range(4)]
    unresolved = _unresolved(_fake_lora_weights(keys, (2048, 2048)), _VL_CLASS)
    assert list(unresolved) == [("model.language_model.", "model.")]
    assert len(unresolved[("model.language_model.", "model.")]) == 4


def test_accounting_ignores_a_tower_absent_from_the_export(monkeypatch):
    """A tower the export does not contain has nothing to merge onto, and its hidden
    size does not match the text tower, so there is no bridge to report."""
    monkeypatch.setattr(SU, "_get_checkpoint_conversion_mapping", lambda name: {})
    keys = [f"model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj"
            for i in range(4)]
    unresolved = _unresolved(
        _fake_lora_weights(keys, (1152, 1152)), "Gemma3ForConditionalGeneration",
    )
    assert unresolved == {}


def test_accounting_ignores_a_shape_coincidence_on_a_merged_tensor(monkeypatch):
    """Same shapes as the text tower, but those tensors are already merge targets of
    adapters that DID resolve, so the match is a coincidence rather than a miss."""
    monkeypatch.setattr(SU, "_get_checkpoint_conversion_mapping", lambda name: {})
    vision_keys = [f"model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj"
                   for i in range(4)]
    text_keys = [f"model.layers.{i}.self_attn.q_proj" for i in range(4)]
    weights = _fake_lora_weights(vision_keys + text_keys, (2048, 2048))
    unresolved = _unresolved(weights, "Gemma3ForConditionalGeneration")
    assert unresolved == {}


def test_accounting_ignores_modules_to_save_without_an_adapter(monkeypatch):
    """A trained head with no LoRA delta is the seeding pass's job, not a partial merge."""
    monkeypatch.setattr(SU, "_get_checkpoint_conversion_mapping", lambda name: {})
    weights = collections.defaultdict(lambda: LoraStats(None, None, None, 0))
    weights["model.language_model.layers.0.self_attn.q_proj"] = LoraStats(
        module = _FakeLinear(2048, 2048), lora_A = None, lora_B = None, alpha = 1.0,
    )
    assert _unresolved(weights, _VL_CLASS) == {}


def test_even_one_unplaced_module_refuses_and_states_its_share(tmp_path, monkeypatch):
    """Every report refuses, whatever its size, and the message says how big it is.

    A size threshold was tried and dropped. It would downgrade a genuine under-merge of a
    text encoder to a printed notice, and the false positive it guards against needs a
    component the export omits whose hidden size exactly equals the text tower's, which no
    shipping checkpoint has: the shape test is what rules those out. The env var is the
    escape hatch for the case that is nonetheless someone's model, and it is exercised
    separately above.
    """
    if not H.family_available("llama"):
        pytest.skip("llama unavailable in this transformers")
    H.set_offline_cpu_env()
    monkeypatch.delenv("UNSLOTH_ALLOW_PARTIAL_LORA_MERGE", raising = False)
    spec = H.make_spec("llama")
    base_dir = os.path.join(str(tmp_path), "base")
    model = H.build_and_save_base(spec, base_dir)
    peft_model = H.attach_lora(model, spec, "full")

    # One extra module beside an otherwise fully resolved adapter, named so that deleting
    # `extra.` lands on a real tensor of the same shape that nothing else claims.
    lora_weights, _ = SU.create_lora_statistics(peft_model, merge_into_original = True)
    placed = sum(1 for key, stats in lora_weights.items()
                 if isinstance(key, str) and stats.lora_A is not None)
    assert placed >= 3, placed
    target = "model.layers.0.self_attn.k_proj"
    hidden = spec.config.num_key_value_heads * (spec.config.hidden_size
                                                // spec.config.num_attention_heads)
    lora_weights["model.extra." + target] = LoraStats(
        module = _FakeLinear(hidden, spec.config.hidden_size),
        lora_A = torch.zeros(8, spec.config.hidden_size),
        lora_B = torch.zeros(hidden, 8),
        alpha  = 1.0,
    )
    # Free that tensor up so the extra module is the only claimant, which is what makes
    # the accounting report it at all.
    lora_weights.pop(target, None)

    with pytest.raises(PartialLoraMergeError) as excinfo:
        SU._check_lora_merge_is_complete(
            base_dir,
            sorted(name for name in os.listdir(base_dir) if name.endswith(".safetensors")),
            lora_weights, "LlamaForCausalLM",
        )
    message = str(excinfo.value)
    assert "model.extra." in message, message
    assert f"1 of {placed}" in message, message


def test_a_convolution_target_is_named_instead_of_crashing_the_matmul():
    """A LoRA on a 4-D weight must be reported by name, not by `mat1 must be a matrix`.

    Reachable because `proj` is both a vision patch-embedding Conv2d and an attention
    output projection on several composite models, so a name-based `target_modules`
    adapts both. The merge then folds a matrix delta into a 4-D tensor. The arithmetic
    cannot work either way; what changes is whether the caller learns which module to
    exclude.
    """
    stats = LoraStats(
        module = _FakeLinear(32, 3), lora_A = torch.zeros(8, 3),
        lora_B = torch.zeros(32, 8), alpha = 1.0,
    )
    with pytest.raises(ValueError, match = "patch_embed.proj"):
        SU._merge_lora(torch.zeros(32, 3, 8, 8), stats, "vision_tower.patch_embed.proj.weight")

    # The 2-D case is untouched: same call, exact arithmetic.
    weight = torch.arange(6, dtype = torch.float32).reshape(3, 2)
    stats_2d = LoraStats(
        module = _FakeLinear(3, 2), lora_A = torch.ones(1, 2),
        lora_B = torch.ones(3, 1), alpha = 2.0,
    )
    merged = SU._merge_lora(weight.clone(), stats_2d, "layers.0.self_attn.q_proj.weight")
    torch.testing.assert_close(merged.cpu().float(), weight + 2.0, atol = 0, rtol = 0)


# --------------------------------------------------------------------------------------
# save_method="lora" is not a merge


def test_save_method_lora_warns_that_it_is_not_an_adapter_save(tmp_path):
    """Asking the merge for an adapter must say so, and must not change what it writes.

    `save_method = "lora"` matches none of the `save_method ==` branches in the merge, so
    it falls through to a plain 16bit merge: the caller who asked for an adapter gets a
    full-size checkpoint with no adapter_config.json. Observed on a real
    `push_to_hub_merged(save_method = "lora")`, which uploaded 2.47 GB of merged weights.

    A warning, not a refusal, and the test pins that on purpose. The caller still both
    documents this value and defaults to it (`unsloth/save.py`'s `unsloth_generic_save`
    declares `save_method = "lora"`), so refusing here would turn a call that completes
    today into an uncaught error with no substitute shipped beside it.
    """
    if not H.family_available("llama"):
        pytest.skip("llama unavailable in this transformers")
    H.set_offline_cpu_env()
    spec = H.make_spec("llama")
    base_dir = os.path.join(str(tmp_path), "base")
    out_dir  = os.path.join(str(tmp_path), "merged")
    model = H.build_and_save_base(spec, base_dir)
    base_tensors = H.read_safetensors_dir(base_dir)
    peft_model = H.attach_lora(model, spec, "full")
    adapted = H.extract_adapted(peft_model)

    with pytest.warns(UserWarning, match = "adapter_config.json"):
        SU.merge_and_overwrite_lora(
            get_model_name = lambda *args, **kwargs: base_dir,
            model = peft_model,
            tokenizer = None,
            save_directory = out_dir,
            save_method = "lora",
            output_dtype = torch.float32,
            push_to_hub = False,
        )
    # Unchanged behaviour: still a merge, and still a correct one.
    merged = H.read_safetensors_dir(out_dir)
    assert merged
    n_adapted, n_passthrough = _assert_merge_values(base_tensors, merged, adapted)
    assert n_adapted >= 1 and n_passthrough >= 1


# --------------------------------------------------------------------------------------
# The exact-merge case must stay exactly as it was


@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_exact_merge_is_unchanged_by_the_accounting(family, tmp_path):
    """A merge that resolves every target must still be bit-exact and must not refuse.

    The accounting runs on every merge, so a false refusal or a perturbed write would
    show up here: untargeted tensors byte-identical, adapted tensors equal to
    base + scale * (B @ A).
    """
    if not H.family_available(family):
        pytest.skip(f"{family} unavailable in this transformers")
    n_adapted, n_passthrough = H.run_case(family, "full", str(tmp_path))
    assert n_adapted >= 1 and n_passthrough >= 1


# --------------------------------------------------------------------------------------
# Verification pass: shapes the guard must never refuse, and what the escape hatch takes


def _tiny_llama(base_dir, *, tie_word_embeddings = False):
    import transformers as T

    cfg = T.AutoConfig.for_model(
        "llama", hidden_size = _H, intermediate_size = _I, num_hidden_layers = 2,
        num_attention_heads = 4, num_key_value_heads = 2, vocab_size = 64,
        max_position_embeddings = 64, tie_word_embeddings = tie_word_embeddings,
    )
    torch.manual_seed(H.SEED)
    model = T.AutoModelForCausalLM.from_config(cfg).to(torch.float32)
    model.save_pretrained(base_dir, safe_serialization = True)
    model.config._name_or_path = base_dir
    return model


def _attach_text_lora(model, targets, **kwargs):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(H.SEED)
    peft_model = get_peft_model(model, LoraConfig(
        r = 8, lora_alpha = 16, lora_dropout = 0.0, bias = "none",
        target_modules = list(targets), **kwargs,
    ))
    H.seed_lora(peft_model)
    return peft_model


_ATTENTION_AND_MLP = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@pytest.mark.parametrize(
    "tie, targets, kwargs",
    [
        (False, _ATTENTION_AND_MLP, {}),
        (True,  _ATTENTION_AND_MLP, {}),
        (True,  _ATTENTION_AND_MLP, {"modules_to_save": ["lm_head"]}),
        (False, ("q_proj", "lm_head", "embed_tokens"), {}),
    ],
    ids = ["text_only", "tied_embeddings", "tied_plus_modules_to_save", "embedding_and_head"],
)
def test_a_text_only_merge_is_never_refused(tmp_path, tie, targets, kwargs):
    """The refusal exists for a composite model whose towers live under another prefix.
    A plain text model has no such prefix, so none of these shapes may raise, tied
    embeddings and an adapted head included."""
    H.set_offline_cpu_env()
    base_dir = str(tmp_path / "base")
    out_dir  = str(tmp_path / "merged")
    model = _tiny_llama(base_dir, tie_word_embeddings = tie)
    base_tensors = H.read_safetensors_dir(base_dir)
    if tie:
        assert "lm_head.weight" not in base_tensors, sorted(base_tensors)

    peft_model = _attach_text_lora(model, targets, **kwargs)
    H.run_merge(peft_model, base_dir, out_dir, save_dtype = torch.float32)

    merged = H.read_safetensors_dir(out_dir)
    assert merged
    if tie:
        assert "lm_head.weight" not in merged, sorted(merged)


def _one_unplaced_lora(hidden = _H):
    lora_weights = collections.defaultdict(lambda: LoraStats(None, None, None, 0))
    lora_weights["model.language_model.layers.0.self_attn.q_proj"] = LoraStats(
        module = torch.nn.Linear(hidden, hidden, bias = False),
        lora_A = torch.zeros(8, hidden), lora_B = torch.zeros(hidden, 8), alpha = 1.0,
    )
    return lora_weights


@pytest.mark.parametrize(
    "value, refuses",
    [("1", False), ("0", True), ("", True), ("true", True), ("yes", True)],
)
def test_only_an_exact_1_opens_the_escape_hatch(tmp_path, monkeypatch, value, refuses):
    """The env var is an exact `1`, like the other UNSLOTH_ALLOW_* switches. Anything
    else, including a truthy-looking word, must still refuse: a typo must not quietly
    hand back a checkpoint that is not trained."""
    from safetensors.torch import save_file

    shard = tmp_path / "model.safetensors"
    save_file({"model.layers.0.self_attn.q_proj.weight": torch.zeros(_H, _H)}, str(shard))

    monkeypatch.setenv("UNSLOTH_ALLOW_PARTIAL_LORA_MERGE", value)
    call = lambda: SU._check_lora_merge_is_complete(
        str(tmp_path), ["model.safetensors"], _one_unplaced_lora(), "LlamaForCausalLM",
    )
    if refuses:
        with pytest.raises(PartialLoraMergeError):
            call()
    else:
        assert call()


def test_the_guard_is_a_noop_when_no_shard_is_staged_yet(tmp_path):
    """Header reads skip a shard that is not on disk. A staging directory without its
    shards must therefore produce no report at all rather than refuse everything."""
    assert SU._check_lora_merge_is_complete(
        str(tmp_path), ["absent.safetensors"], _one_unplaced_lora(), "LlamaForCausalLM",
    ) == {}


def test_a_quantized_module_is_measured_by_its_feature_counts():
    """A 4-bit base layer stores a packed weight whose shape says nothing about the
    tensor the merge writes. The accounting must read the feature counts, or it silently
    stops working on every 4-bit merge."""
    class _Packed(torch.nn.Module):
        in_features, out_features = _H, 2 * _H
        def __init__(self):
            super().__init__()
            # What bitsandbytes stores: a flat uint8 buffer, not (out, in).
            self.weight = torch.nn.Parameter(
                torch.zeros(_H * 2 * _H // 2, 1, dtype = torch.uint8), requires_grad = False,
            )

    stats = LoraStats(module = _Packed(), lora_A = torch.zeros(8, _H),
                      lora_B = torch.zeros(2 * _H, 8), alpha = 1.0)
    assert SU._lora_target_logical_shape(stats) == (2 * _H, _H)

    lora_weights = collections.defaultdict(lambda: LoraStats(None, None, None, 0))
    lora_weights["model.language_model.layers.0.self_attn.q_proj"] = stats
    # The logical shape is what the export holds, so the missed bridge is still reported.
    assert _unresolved_lora_targets(
        lora_weights, {"model.layers.0.self_attn.q_proj.weight"},
        {"model.layers.0.self_attn.q_proj": (2 * _H, _H)}, "LlamaForCausalLM",
    )
    # The packed shape is not the module's shape, so it is not a match and not a report.
    assert _unresolved_lora_targets(
        lora_weights, {"model.layers.0.self_attn.q_proj.weight"},
        {"model.layers.0.self_attn.q_proj": (_H * 2 * _H // 2, 1)}, "LlamaForCausalLM",
    ) == {}


def test_the_guard_stays_cheap_on_a_large_adapter():
    """Every unplaced module walks its own prefix deletions, so the cost of the worst
    case (nothing resolves) is worth pinning: a 1000 module adapter against an 8000 key
    export must not turn a merge into a coffee break."""
    import time

    lora_weights = collections.defaultdict(lambda: LoraStats(None, None, None, 0))
    disk_keys, disk_shapes = set(), {}
    for layer in range(125):
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj", "extra_proj"):
            lora_weights[f"model.mystery_tower.layers.{layer}.self_attn.{projection}"] = LoraStats(
                module = torch.nn.Linear(_H, _H, bias = False),
                lora_A = torch.zeros(8, _H), lora_B = torch.zeros(_H, 8), alpha = 1.0,
            )
            key = f"model.layers.{layer}.self_attn.{projection}"
            disk_keys.add(key + ".weight")
            disk_shapes[key] = (_H, _H)
    for filler in range(6000):
        disk_keys.add(f"model.layers.{filler // 60}.mlp.filler{filler}.weight")

    started = time.time()
    unresolved = _unresolved_lora_targets(lora_weights, disk_keys, disk_shapes, "LlamaForCausalLM")
    elapsed = time.time() - started

    assert unresolved, "a wholly unplaced adapter must be reported"
    assert elapsed < 60, f"the guard took {elapsed:.1f}s on 1000 modules"
