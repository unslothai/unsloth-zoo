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

"""The merged_16bit config must describe the weights the merge actually writes (#969).

`merge_and_overwrite_lora` takes its weights from the resolved base checkpoint but used to
save `model.config`, which `text_only = True` has already replaced with the nested text
config, so a `gemma3_text` config landed next to full VLM weights. Nothing raised: every
tensor was silently re-initialized on reload, so a finetune could be served untrained.

The fixture is the real shape -- a text-only decoder whose `_name_or_path` points at a VLM
checkpoint. Passing is not "no exception" (the bug threw none) but the saved architecture
matching the written tensors and a reload reporting no missing/unexpected/mismatched keys.
"""

from __future__ import annotations

import json
import os
import warnings

import pytest
import torch

import _merge_e2e_helpers as H

_LANG_QV = ["q_proj", "v_proj"]


def _gemma3_config():
    """Tiny but architecturally real Gemma3 VLM (same shape as the passthrough suite)."""
    import transformers as T
    text = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
                max_position_embeddings=64, head_dim=8)
    vision = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                  num_attention_heads=4, image_size=16, patch_size=8, num_channels=3)
    return T.Gemma3Config(text_config=text, vision_config=vision)


def _write_vlm_base(tmp_path):
    """Save a full VLM checkpoint and return (base_dir, parent_config)."""
    import transformers as T
    if not H.family_available("gemma3"):
        pytest.skip("gemma3 unavailable in this transformers")
    cfg = _gemma3_config()
    base_dir = os.path.join(str(tmp_path), "base")
    torch.manual_seed(H.SEED)
    try:
        vlm = T.AutoModelForImageTextToText.from_config(cfg).to(torch.float32)
    except Exception as e:  # tiny VLM config quirks vary by version
        pytest.skip(f"could not instantiate tiny gemma3: {type(e).__name__}: {e}")
    vlm.save_pretrained(base_dir, safe_serialization=True)
    return base_dir, cfg


def _text_only_model(cfg, base_dir):
    """What `text_only = True` leaves in memory: the nested text decoder, pointed at
    the VLM checkpoint. Mirrors `_get_text_only_config` in unsloth/models/_utils.py."""
    import transformers as T
    text_config = cfg.get_text_config()
    torch.manual_seed(H.SEED)
    model = T.AutoModelForCausalLM.from_config(text_config).to(torch.float32)
    model.config._name_or_path = base_dir
    assert model.config.model_type == "gemma3_text", "fixture is not the text-only shape"
    return model


def _attach_lora(model, modules_to_save=None):
    from peft import LoraConfig, get_peft_model
    pm = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
        target_modules=_LANG_QV, modules_to_save=modules_to_save))
    H.seed_lora(pm)
    return pm


def _saved_config(out_dir):
    with open(os.path.join(out_dir, "config.json")) as f:
        return json.load(f)


def _reload_info(out_dir):
    """Reload as the architecture the saved config declares and return loading info."""
    import transformers as T
    _, info = T.AutoModelForImageTextToText.from_pretrained(
        out_dir, output_loading_info=True, local_files_only=True)
    return info


def test_text_only_export_config_matches_written_weights(tmp_path):
    """#969: a text_only export must declare the VLM it actually wrote."""
    H.set_offline_cpu_env()
    base_dir, cfg = _write_vlm_base(tmp_path)
    out_dir = os.path.join(str(tmp_path), "merged")

    pm = _attach_lora(_text_only_model(cfg, base_dir))
    H.run_merge(pm, base_dir, out_dir, save_dtype=torch.float32)

    # transformers 5 flattens the VLM prefixes, so match on a substring, not the key name.
    written = list(H.read_safetensors_dir(out_dir))
    assert any("vision" in k for k in written), f"fixture wrote no vision weights: {written[:4]}"

    base, saved = _saved_config(base_dir), _saved_config(out_dir)
    assert saved.get("model_type") == base.get("model_type") != "gemma3_text", (
        f"saved config describes {saved.get('model_type')!r}, not the written weights")
    assert saved.get("architectures") == base.get("architectures") and saved.get("architectures"), (
        f"saved architectures {saved.get('architectures')!r} do not match the weights")

    info = _reload_info(out_dir)
    for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        assert not info.get(field), f"{field}: {info.get(field)[:6]}"


def test_text_only_export_preserves_resized_vocab(tmp_path):
    """Reading the base config must not drop vocab growth from training.

    Without the carry-over the export declares the base vocab beside larger embedding
    tensors, and the reload fails on a size mismatch instead of silently succeeding.
    """
    H.set_offline_cpu_env()
    base_dir, cfg = _write_vlm_base(tmp_path)
    out_dir = os.path.join(str(tmp_path), "merged")

    model = _text_only_model(cfg, base_dir)
    base_vocab = model.get_input_embeddings().weight.shape[0]
    new_vocab = base_vocab + 16
    model.resize_token_embeddings(new_vocab)
    pm = _attach_lora(model, modules_to_save=["embed_tokens", "lm_head"])

    H.run_merge(pm, base_dir, out_dir, save_dtype=torch.float32)

    merged = H.read_safetensors_dir(out_dir)
    embed = next(v for k, v in merged.items() if k.endswith("embed_tokens.weight"))
    assert embed.shape[0] == new_vocab, "the merge did not write the resized embedding"

    saved = _saved_config(out_dir)
    declared = saved.get("text_config", {}).get("vocab_size", saved.get("vocab_size"))
    assert declared == new_vocab, (
        f"config declares vocab {declared} beside {embed.shape[0]} embedding rows")

    info = _reload_info(out_dir)
    for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        assert not info.get(field), f"{field}: {info.get(field)[:6]}"


def test_base_config_read_failure_falls_back_with_a_warning(tmp_path, monkeypatch):
    """The fallback keeps a previously-working export working, and says so.

    It deliberately restores the #969 output, so it is an availability fallback and not
    a correct one: the warning is the only signal the user gets.
    """
    import transformers
    import unsloth_zoo.saving_utils as SU

    H.set_offline_cpu_env()
    base_dir, cfg = _write_vlm_base(tmp_path)
    out_dir = os.path.join(str(tmp_path), "merged")
    pm = _attach_lora(_text_only_model(cfg, base_dir))

    def _boom(*a, **k):
        raise OSError("forced base config read failure")

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", _boom)
    if getattr(SU, "AutoConfig", None) is not None:
        monkeypatch.setattr(SU.AutoConfig, "from_pretrained", _boom, raising=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        H.run_merge(pm, base_dir, out_dir, save_dtype=torch.float32)

    assert any("Could not read the base config" in str(w.message) for w in caught), \
        "the fallback took effect without warning the user"
    assert _saved_config(out_dir).get("model_type") == "gemma3_text"
    assert any(f.endswith(".safetensors") for f in os.listdir(out_dir)), \
        "the fallback must not lose the export it is protecting"


# vocab-size helpers, config-level only

_MISSING = object()


def _shape(top=_MISSING, nested=_MISSING):
    from types import SimpleNamespace as NS
    cfg = NS()
    if top is not _MISSING:
        cfg.vocab_size = top
    if nested is not _MISSING:
        cfg.text_config = NS(vocab_size=nested)
    return cfg


@pytest.mark.parametrize("top,nested,expected", [
    (64,       72,       72),    # PaliGemma: nested wins over a stale compat value
    (_MISSING, 72,       72),    # Gemma3 / Qwen-VL: nested only
    (72,       _MISSING, 72),    # plain LM: top level only
    (72,       None,     72),    # nested present but unset -> fall back
    (_MISSING, _MISSING, None),  # nothing to read
])
def test_config_vocab_size_precedence(top, nested, expected):
    from unsloth_zoo.saving_utils import _config_vocab_size
    assert _config_vocab_size(_shape(top, nested)) == expected


def test_carry_over_updates_both_levels_for_stale_top_level():
    """PaliGemma is the shape that makes precedence load-bearing: resize updates the
    nested vocab and leaves the top-level compat value behind."""
    from unsloth_zoo.saving_utils import _carry_over_vocab_size
    base = _shape(top=64, nested=64)
    _carry_over_vocab_size(base, _shape(top=64, nested=72))
    assert (base.vocab_size, base.text_config.vocab_size) == (72, 72)


def test_carry_over_nested_only_and_top_only():
    from unsloth_zoo.saving_utils import _carry_over_vocab_size
    nested_base = _shape(nested=64)
    _carry_over_vocab_size(nested_base, _shape(nested=72))
    assert nested_base.text_config.vocab_size == 72

    top_base = _shape(top=64)
    _carry_over_vocab_size(top_base, _shape(top=72))
    assert top_base.vocab_size == 72


def test_carry_over_leaves_a_distinct_top_level_vocabulary_alone():
    """A top level that already differs from the text vocab is a DIFFERENT vocabulary, not a
    stale copy, so it must not be overwritten."""
    from unsloth_zoo.saving_utils import _carry_over_vocab_size
    base = _shape(top=64, nested=72)
    _carry_over_vocab_size(base, _shape(nested=80))
    assert base.text_config.vocab_size == 80, "text vocab was not carried over"
    assert base.vocab_size == 64, "a distinct top-level vocabulary was clobbered"


def test_carry_over_preserves_ovis2_top_level_vocabulary():
    """Ovis2 sizes its lm_head from the top-level vocab_size and ships it deliberately
    different from text_config.vocab_size, so writing one value to both breaks the reload."""
    import transformers as T
    from unsloth_zoo.saving_utils import _carry_over_vocab_size

    if not H.family_available("ovis2"):
        pytest.skip("ovis2 unavailable in this transformers")
    base, trained = T.CONFIG_MAPPING["ovis2"](), T.CONFIG_MAPPING["ovis2"]()
    top_before = base.vocab_size
    assert top_before != base.text_config.vocab_size, "fixture no longer has distinct levels"

    _carry_over_vocab_size(base, trained)          # no resize at all
    assert base.vocab_size == top_before, "no-resize carry-over moved the top-level vocab"

    trained.text_config.vocab_size += 8
    _carry_over_vocab_size(base, trained)
    assert base.text_config.vocab_size == trained.text_config.vocab_size
    assert base.vocab_size == top_before, "resize moved the lm_head vocabulary too"


def test_carry_over_warns_instead_of_raising_when_vocab_is_read_only():
    """A config exposing vocab_size without a setter must not crash a working export."""
    from unsloth_zoo.saving_utils import _carry_over_vocab_size

    class ReadOnly:
        vocab_size = property(lambda self: 64)

    base = ReadOnly()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _carry_over_vocab_size(base, _shape(top=72))
    assert any("could not set vocab_size" in str(w.message) for w in caught)


def test_carry_over_shrink_is_carried_too():
    """Gemma3's tokenizer is shorter than its padded matrix, so shrink is a real case."""
    from unsloth_zoo.saving_utils import _carry_over_vocab_size
    base = _shape(top=262208, nested=262208)
    _carry_over_vocab_size(base, _shape(nested=262149))
    assert base.text_config.vocab_size == 262149


def test_carry_over_no_resize_and_no_trained_value_are_no_ops():
    from unsloth_zoo.saving_utils import _carry_over_vocab_size
    same = _shape(top=64, nested=64)
    _carry_over_vocab_size(same, _shape(top=64, nested=64))
    assert (same.vocab_size, same.text_config.vocab_size) == (64, 64)

    untouched = _shape(top=64, nested=64)
    _carry_over_vocab_size(untouched, _shape())
    assert (untouched.vocab_size, untouched.text_config.vocab_size) == (64, 64)


def test_carry_over_does_not_invent_fields():
    from unsloth_zoo.saving_utils import _carry_over_vocab_size
    bare = _shape()
    _carry_over_vocab_size(bare, _shape(top=72))
    assert not hasattr(bare, "vocab_size")
    assert not hasattr(bare, "text_config")


@pytest.mark.parametrize("family", ["qwen2_5_omni", "qwen3_omni_moe", "colqwen2", "t5gemma"])
def test_carry_over_reaches_text_configs_not_named_text_config(family):
    """Some composite configs nest the text section under another name, so `.text_config`
    misses it entirely. `get_text_config()` is the documented accessor."""
    import transformers as T
    from unsloth_zoo.saving_utils import _carry_over_vocab_size, _config_vocab_size

    if not H.family_available(family):
        pytest.skip(f"{family} unavailable in this transformers")
    try:
        base, trained = T.CONFIG_MAPPING[family](), T.CONFIG_MAPPING[family]()
    except Exception as e:
        pytest.skip(f"{family} will not instantiate bare: {type(e).__name__}")
    if getattr(base, "text_config", None) is not None:
        pytest.skip(f"{family} exposes .text_config on this version")

    target = _config_vocab_size(trained)
    if target is None:
        pytest.skip(f"{family} has no vocab_size to carry")
    target += 8
    trained.get_text_config().vocab_size = target

    assert _config_vocab_size(trained) == target, "nested vocab not read via get_text_config()"
    _carry_over_vocab_size(base, trained)
    assert base.get_text_config().vocab_size == target, "nested vocab not written"


def test_carry_over_on_real_transformers_configs():
    """Guard the precedence against config classes rather than stand-ins."""
    import transformers as T
    from unsloth_zoo.saving_utils import _carry_over_vocab_size, _config_vocab_size

    if not H.family_available("paligemma"):
        pytest.skip("paligemma unavailable in this transformers")
    base = T.PaliGemmaConfig(text_config={"model_type": "gemma", "vocab_size": 64})
    trained = T.PaliGemmaConfig(text_config={"model_type": "gemma", "vocab_size": 64})
    base.vocab_size = 64
    trained.vocab_size = 64
    trained.text_config.vocab_size = 72

    assert _config_vocab_size(trained) == 72, "stale top-level compat value won"
    _carry_over_vocab_size(base, trained)
    assert base.text_config.vocab_size == 72
    assert base.vocab_size == 72

    assert _config_vocab_size(T.LlamaConfig(vocab_size=72)) == 72
