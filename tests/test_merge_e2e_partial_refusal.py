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

"""A composite vision-language merge must either merge every tower or refuse (#5290).

A Qwen2.5-VL checkpoint is flat (`model.layers.N...`) while the paths PEFT adapts are
composite (`model.language_model.layers.N...`). The bridge came from
`<ModelClass>._checkpoint_conversion_mapping`, which Transformers 5 moved into a registry, and
without it every language adapter resolved to nothing and dropped out of both sides of the
merge's count check, so the base weights were written back and reported as success.
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
    """A tiny real Qwen2.5-VL. Which layout `save_pretrained` writes is Transformers'
    business: the tests read it back rather than assume it."""
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
    H.set_offline_cpu_env()
    base_dir = os.path.join(work_dir, "base")
    out_dir  = os.path.join(work_dir, "merged")
    model = _build_tiny_vl(base_dir)
    base_tensors = H.read_safetensors_dir(base_dir)
    peft_model = _attach_vl_lora(model)
    adapted = H.extract_adapted(peft_model)
    return base_dir, out_dir, base_tensors, peft_model, adapted


# Spelled out so the reference is independent of the resolution under test, and tried after
# the identity so a release that writes the composite layout on disk needs no change.
_EXPECTED_DISK_RENAMES = (
    ("model.language_model.", "model."),
    ("model.visual.",        "visual."),
)


def _resolve_composite_key(adapted_key, base_keys):
    if adapted_key in base_keys: return adapted_key
    for lora_prefix, disk_prefix in _EXPECTED_DISK_RENAMES:
        if not adapted_key.startswith(lora_prefix): continue
        candidate = disk_prefix + adapted_key[len(lora_prefix) :]
        if candidate in base_keys: return candidate
    return None


def _assert_merge_values(base_tensors, merged, adapted):
    """An independent float64 reference, not a re-run of the merge's own arithmetic."""
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


def test_conversion_mapping_available_for_composite_vl():
    """An empty mapping here is the root cause of #5290. Only the language rename is required:
    5.5 ships it alone, and prefix inference reaches the vision tower but not the language one."""
    _skip_unless_vl_available()
    mapping = _get_checkpoint_conversion_mapping(_VL_CLASS)
    assert mapping, (
        f"no checkpoint conversion mapping for {_VL_CLASS}: the language tower cannot "
        f"be bridged to the flat on-disk layout"
    )
    targets = set(mapping.values())
    assert any(t.startswith("model.language_model") for t in targets), mapping


def test_partial_mapping_does_not_shadow_prefix_inference():
    """Gemma 3 is written `language_model.model.layers...` while its class registers only
    `^language_model -> model.language_model`, the remaining hop living on a nested sub-model.
    Taken literally that resolves to `language_model.layers...`, which is in no shard."""
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
    # Losing the escaped separator rewrites `model.encoder...` to `backboneencoder...`.
    (r"^backbone\.",                                    "backbone."),
    (r"^language_model\.model\.",                       "language_model.model."),
    (r"^embed_out\.",                                   "embed_out."),
    (r"\.ln1\.",                                        ".ln1."),
    # A capture or a class ends the literal.
    (r"blocks\.(\d+)\.",                               "blocks."),
    (r"^model\.(?:(?!language_model\.))(.+)$",           "model."),
    (r"(?!x)",                                          ""),
    (r"(?<!_)",                                         ""),
    ("",                                                ""),
])
def test_pattern_literal_prefix(pattern, literal):
    """A mapping entry is a regex on one side, so reversing it needs the literal head."""
    assert SU._pattern_literal_prefix(pattern) == literal


def test_escaped_separator_in_a_rename_still_resolves(monkeypatch):
    r"""`^backbone\.` must reverse to `backbone.`, separator included: truncating at the
    backslash concatenates the adapter's next component onto it and holds no shard's key."""
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
    r"""`hunyuan_vl` registers both `^model\.vit` and `^vit` onto `model.vision_tower`, so
    keying the reverse map by target, or taking the first match blind, reverses the vision tower
    onto a prefix the checkpoint does not have."""
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

    n_adapted, n_passthrough = _assert_merge_values(base_tensors, merged, adapted)
    assert n_adapted >= len(language_keys) // 2 and n_passthrough >= 1


def test_vl_merge_refuses_when_the_bridge_is_unavailable(tmp_path, monkeypatch):
    """The mapping is removed, not mocked lower down: that is the state a Transformers release
    can put Unsloth in."""
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

    # Refused BEFORE the first shard was rewritten: a refusal after the loop leaves a
    # half-merged export behind.
    if os.path.isdir(out_dir):
        staged = H.read_safetensors_dir(out_dir)
        for key, tensor in staged.items():
            assert key in base_tensors, key
            assert torch.equal(tensor, base_tensors[key]), (
                f"{key} was rewritten before the refusal"
            )


def test_partial_merge_escape_hatch_downgrades_the_refusal(tmp_path, monkeypatch):
    _skip_unless_vl_available()
    base_dir, out_dir, base_tensors, peft_model, _ = _stage_tiny_vl(str(tmp_path))
    monkeypatch.setattr(SU, "_get_checkpoint_conversion_mapping", lambda name: {})
    monkeypatch.setenv("UNSLOTH_ALLOW_PARTIAL_LORA_MERGE", "1")

    H.run_merge(peft_model, base_dir, out_dir, save_dtype = torch.float32)
    merged = H.read_safetensors_dir(out_dir)
    assert merged
    # Unmerged, as the message advertises: the old behaviour, opted into.
    language_keys = _language_weight_keys(base_tensors)
    assert all(torch.equal(merged[k], base_tensors[k]) for k in language_keys)


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
    """A tower the export omits has nothing to merge onto, and its hidden size differs."""
    monkeypatch.setattr(SU, "_get_checkpoint_conversion_mapping", lambda name: {})
    keys = [f"model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj"
            for i in range(4)]
    unresolved = _unresolved(
        _fake_lora_weights(keys, (1152, 1152)), "Gemma3ForConditionalGeneration",
    )
    assert unresolved == {}


def test_accounting_ignores_a_shape_coincidence_on_a_merged_tensor(monkeypatch):
    """Those tensors are already merge targets of adapters that DID resolve."""
    monkeypatch.setattr(SU, "_get_checkpoint_conversion_mapping", lambda name: {})
    vision_keys = [f"model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj"
                   for i in range(4)]
    text_keys = [f"model.layers.{i}.self_attn.q_proj" for i in range(4)]
    weights = _fake_lora_weights(vision_keys + text_keys, (2048, 2048))
    unresolved = _unresolved(weights, "Gemma3ForConditionalGeneration")
    assert unresolved == {}


def test_accounting_ignores_modules_to_save_without_an_adapter(monkeypatch):
    monkeypatch.setattr(SU, "_get_checkpoint_conversion_mapping", lambda name: {})
    weights = collections.defaultdict(lambda: LoraStats(None, None, None, 0))
    weights["model.language_model.layers.0.self_attn.q_proj"] = LoraStats(
        module = _FakeLinear(2048, 2048), lora_A = None, lora_B = None, alpha = 1.0,
    )
    assert _unresolved(weights, _VL_CLASS) == {}


def test_even_one_unplaced_module_refuses_and_states_its_share(tmp_path, monkeypatch):
    """A size threshold would downgrade a genuine text-encoder under-merge to a notice."""
    if not H.family_available("llama"):
        pytest.skip("llama unavailable in this transformers")
    H.set_offline_cpu_env()
    monkeypatch.delenv("UNSLOTH_ALLOW_PARTIAL_LORA_MERGE", raising = False)
    spec = H.make_spec("llama")
    base_dir = os.path.join(str(tmp_path), "base")
    model = H.build_and_save_base(spec, base_dir)
    peft_model = H.attach_lora(model, spec, "full")

    # Named so deleting `extra.` lands on a real tensor of the same shape nothing else claims.
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
    # Free that tensor so the extra module is the only claimant.
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
    """Reachable because `proj` is both a vision patch-embedding Conv2d and an attention output
    projection, so a name-based `target_modules` adapts both. The arithmetic cannot work either
    way; what changes is whether the caller learns which module to exclude."""
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


def test_save_method_lora_warns_that_it_is_not_an_adapter_save(tmp_path):
    """`save_method = "lora"` matches no `save_method ==` branch and falls through to a plain
    16bit merge, so the caller gets a full-size checkpoint with no adapter_config.json. A
    warning, not a refusal: `unsloth/save.py`'s `unsloth_generic_save` still defaults to this
    value, so raising would break calls that complete today."""
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
    merged = H.read_safetensors_dir(out_dir)
    assert merged
    n_adapted, n_passthrough = _assert_merge_values(base_tensors, merged, adapted)
    assert n_adapted >= 1 and n_passthrough >= 1


@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_exact_merge_is_unchanged_by_the_accounting(family, tmp_path):
    """The accounting runs on every merge, so a false refusal shows up here."""
    if not H.family_available(family):
        pytest.skip(f"{family} unavailable in this transformers")
    n_adapted, n_passthrough = H.run_case(family, "full", str(tmp_path))
    assert n_adapted >= 1 and n_passthrough >= 1


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
    """The refusal exists for towers under another prefix; a plain text model has none."""
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
    """Exact `1`, like the other UNSLOTH_ALLOW_* switches: a truthy typo must still refuse."""
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
    """Header reads skip a shard that is not on disk, so no shards means no report."""
    assert SU._check_lora_merge_is_complete(
        str(tmp_path), ["absent.safetensors"], _one_unplaced_lora(), "LlamaForCausalLM",
    ) == {}


def test_a_quantized_module_is_measured_by_its_feature_counts():
    """A 4-bit layer's packed shape says nothing about the tensor the merge writes, so the
    accounting reads the feature counts or silently dies on every 4-bit merge."""
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
    # A packed shape is not the module's shape, so no match and no report.
    assert _unresolved_lora_targets(
        lora_weights, {"model.layers.0.self_attn.q_proj.weight"},
        {"model.layers.0.self_attn.q_proj": (_H * 2 * _H // 2, 1)}, "LlamaForCausalLM",
    ) == {}


def test_the_guard_stays_cheap_on_a_large_adapter():
    """Every unplaced module walks its own prefix deletions, so pin the worst case."""
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


def test_a_packed_target_gets_a_logical_shape_so_the_bridge_is_still_reported(tmp_path, monkeypatch):
    """An mxfp4 target is `<module>_blocks` + `<module>_scales` with no `.weight`, so it had no
    entry in `disk_module_shapes` and was dropped at `disk_shape is None` -- and the merge's own
    count cancels the same omission out of both its sides, the #5290 shape. `_blocks` is
    `(out, G, B)` at two 4-bit values per byte, so the width is `G * B * 2` = 2048 here."""
    from safetensors.torch import save_file

    monkeypatch.setattr(SU, "_get_checkpoint_conversion_mapping", lambda name: {})

    shard = "model-00001-of-00001.safetensors"
    tensors = {}
    for i in range(4):
        module = f"model.layers.{i}.self_attn.q_proj"
        tensors[f"{module}_blocks"] = torch.zeros(2048, 32, 32, dtype = torch.uint8)
        tensors[f"{module}_scales"] = torch.zeros(2048, 32, dtype = torch.uint8)
    save_file(tensors, os.path.join(str(tmp_path), shard))

    disk_keys, disk_shapes = SU._disk_module_shapes(str(tmp_path), [shard])

    # The raw names are still reported as written, which is what the backing test reads.
    assert "model.layers.0.self_attn.q_proj_blocks" in disk_keys
    assert "model.layers.0.self_attn.q_proj_scales" in disk_keys
    assert disk_shapes.get("model.layers.0.self_attn.q_proj") == (2048, 2048)

    # So a LoRA under a different prefix is reported rather than silently skipped.
    keys = [f"model.language_model.layers.{i}.self_attn.q_proj" for i in range(4)]
    unresolved = _unresolved_lora_targets(
        _fake_lora_weights(keys, (2048, 2048)), disk_keys, disk_shapes, _VL_CLASS,
    )
    assert list(unresolved) == [("model.language_model.", "model.")]
    assert len(unresolved[("model.language_model.", "model.")]) == 4


def test_a_packed_moe_stack_is_not_given_a_two_dimensional_shape(tmp_path):
    """`_lora_target_logical_shape` is always 2-D, so a 3D expert stack could never equal it."""
    from safetensors.torch import save_file

    shard = "model-00001-of-00001.safetensors"
    save_file(
        {
            "model.layers.0.mlp.experts.gate_up_proj_blocks":
                torch.zeros(4, 512, 16, 16, dtype = torch.uint8),
            "model.layers.0.mlp.experts.gate_up_proj_scales":
                torch.zeros(4, 512, 16, dtype = torch.uint8),
        },
        os.path.join(str(tmp_path), shard),
    )

    _, disk_shapes = SU._disk_module_shapes(str(tmp_path), [shard])
    assert "model.layers.0.mlp.experts.gate_up_proj" not in disk_shapes


def test_a_blocks_tensor_with_no_scales_partner_is_not_given_a_shape(tmp_path):
    from safetensors.torch import save_file

    shard = "model-00001-of-00001.safetensors"
    save_file(
        {"model.layers.0.self_attn.q_proj_blocks": torch.zeros(2048, 32, 32, dtype = torch.uint8)},
        os.path.join(str(tmp_path), shard),
    )

    _, disk_shapes = SU._disk_module_shapes(str(tmp_path), [shard])
    assert "model.layers.0.self_attn.q_proj" not in disk_shapes
