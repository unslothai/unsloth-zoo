# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""The planner can plan from a resolved config instead of the repo's own.

A text_only load of a vision-language checkpoint builds the standalone decoder from
``text_config``: ``model.layers.0``, no vision tower. Rebuilt from the repo name the
planner describes the whole VLM (``model.language_model.layers.0``), so Unsloth had to
decline, and the "sequential" fallback fills GPU 0 to its whole budget. On a 7 card
Qwen3.5-397B-A17B-FP8 load that left no room for the per-layer FP8 expert merge and
the load died with an OOM on GPU 1.

Meta-device models only: no GPU, no weights, no network.
"""
import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import AutoConfig, Gemma3Config

import unsloth_zoo.device_map_planner as planner

_GiB = 2**30


def _tiny_vlm_config():
    return Gemma3Config(
        text_config = dict(
            vocab_size = 512,
            hidden_size = 64,
            intermediate_size = 128,
            num_hidden_layers = 4,
            num_attention_heads = 2,
            num_key_value_heads = 1,
            head_dim = 32,
        ),
        vision_config = dict(
            hidden_size = 64,
            intermediate_size = 128,
            num_hidden_layers = 2,
            num_attention_heads = 2,
            image_size = 32,
            patch_size = 16,
        ),
        mm_tokens_per_image = 4,
    )


@pytest.fixture
def no_hub(monkeypatch):
    """Planning from a resolved config must not look the repo up again."""

    def _refuse(*args, **kwargs):
        raise AssertionError("AutoConfig.from_pretrained called despite config=")

    monkeypatch.setattr(AutoConfig, "from_pretrained", _refuse)


def test_build_meta_model_uses_the_given_config(no_hub):
    text_config = _tiny_vlm_config().get_text_config()
    model, _quantizer, config = planner.build_meta_model(
        "org/unused", config = text_config, token = "t", revision = "r"
    )
    names = {name for name, _ in model.named_modules()}
    assert "model.layers.0" in names
    assert not any("language_model" in n or "vision" in n for n in names)
    assert type(config) is type(text_config)


def test_overrides_apply_to_a_copy(no_hub):
    text_config = _tiny_vlm_config().get_text_config()
    before = text_config.to_dict()
    _model, _q, config = planner.build_meta_model(
        "org/unused",
        config = text_config,
        dtype = torch.float16,
        max_position_embeddings = 64,
        cache_dir = "/nonexistent",
        local_files_only = True,
        trust_remote_code = False,
    )
    assert config is not text_config
    assert config.max_position_embeddings == 64
    assert text_config.to_dict() == before
    # Hub options are not config fields and never become attributes.
    assert not hasattr(config, "cache_dir")
    assert not hasattr(config, "local_files_only")


def test_the_plan_names_the_modules_the_text_only_load_builds(no_hub):
    text_config = _tiny_vlm_config().get_text_config()
    plan = planner.plan_device_map_for_pretrained(
        "org/unused",
        config = text_config,
        max_memory = {0: 8 * _GiB, 1: 8 * _GiB},
        activation_reserve_bytes = 0,
        headroom_bytes = 0,
    )
    keys = set(plan.device_map)
    assert "lm_head" in keys or "model.embed_tokens" in keys
    assert any(k.startswith("model.layers.") for k in keys)
    assert not any("language_model" in k or "vision" in k for k in keys)


def test_without_config_the_repo_config_is_planned(monkeypatch):
    """Unchanged default: the planner rebuilds the repo's own config from its name."""
    calls = []

    def _from_pretrained(name, **kwargs):
        calls.append(name)
        return _tiny_vlm_config()

    monkeypatch.setattr(AutoConfig, "from_pretrained", _from_pretrained)
    plan = planner.plan_device_map_for_pretrained(
        "org/vlm",
        max_memory = {0: 8 * _GiB, 1: 8 * _GiB},
        activation_reserve_bytes = 0,
        headroom_bytes = 0,
    )
    assert calls == ["org/vlm"]
    assert any("language_model" in k or "vision" in k for k in plan.device_map)


def test_dtype_wins_over_torch_dtype_in_either_order():
    config = _tiny_vlm_config().get_text_config()
    for overrides in (
        {"dtype": torch.float32, "torch_dtype": torch.float16},
        {"torch_dtype": torch.float16, "dtype": torch.float32},
    ):
        out = planner._apply_config_overrides(config, overrides)
        assert out.torch_dtype == torch.float32
    out = planner._apply_config_overrides(config, {"torch_dtype": torch.float16})
    assert out.torch_dtype == torch.float16


def test_a_per_module_dtype_mapping_resolves_to_its_main_dtype():
    # from_pretrained loads every module in the "" entry's dtype; a dict left on the
    # config breaks meta-model construction.
    config = _tiny_vlm_config().get_text_config()
    out = planner._apply_config_overrides(config, {"dtype": {"": torch.bfloat16, "text_config": torch.float16}})
    assert out.torch_dtype == torch.bfloat16
    out = planner._apply_config_overrides(config, {"dtype": {"": "float16"}})
    assert out.torch_dtype == torch.float16
    out = planner._apply_config_overrides(config, {"dtype": {"text_config": torch.float16}})
    assert out.torch_dtype == torch.get_default_dtype()


def test_a_partial_sub_config_dict_is_merged_not_substituted():
    config = _tiny_vlm_config()
    sub_type = type(config.text_config)
    before = config.text_config.to_dict()
    out = planner._apply_config_overrides(config, {"text_config": {"max_position_embeddings": 64}})
    assert isinstance(out.text_config, sub_type)
    assert out.text_config.max_position_embeddings == 64
    assert out.text_config.hidden_size == config.text_config.hidden_size
    assert config.text_config.to_dict() == before


def test_a_nested_sub_config_override_keeps_every_level_a_config():
    omni = pytest.importorskip("transformers.models.qwen2_5_omni.configuration_qwen2_5_omni")
    config = omni.Qwen2_5OmniConfig()
    thinker_type = type(config.thinker_config)
    text_type = type(config.thinker_config.text_config)
    out = planner._apply_config_overrides(
        config, {"thinker_config": {"text_config": {"max_position_embeddings": 64}}}
    )
    assert isinstance(out.thinker_config, thinker_type)
    assert isinstance(out.thinker_config.text_config, text_type)
    assert out.thinker_config.text_config.max_position_embeddings == 64
    assert config.thinker_config.text_config.max_position_embeddings != 64
