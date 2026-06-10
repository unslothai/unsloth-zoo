# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""CPU-pure behavioural tests for the `use_cache` handling in
`prepare_model_for_training` (unsloth_zoo/training_utils.py).

KV cache is unused under gradient checkpointing, so the prepare step
walks `model.config` and every nested transformers config (composite
VLM configs expose `text_config` / `vision_config` attributes) and
sets truthy `use_cache` flags to False. These tests pin the contract:

  - top-level `use_cache=True` flips to False for both
    `use_gradient_checkpointing=True` and `"unsloth"`;
  - nested sub-configs of composite configs flip too;
  - `use_cache=None` and `use_cache=False` are preserved (None means
    "defer to the model default" and must not become False);
  - nothing is touched when gradient checkpointing is disabled;
  - non-config attachments with a `use_cache` attribute are ignored;
  - self-referencing config graphs terminate (visited-id guard).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import AutoModelForCausalLM, LlamaConfig

try:
    from transformers import PreTrainedConfig
except ImportError:
    from transformers import PretrainedConfig as PreTrainedConfig

from unsloth_zoo.training_utils import (
    disable_use_cache,
    prepare_model_for_training,
    restore_use_cache,
)


def _tiny_llama(**config_overrides):
    config = LlamaConfig(
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 2,
        vocab_size = 64,
        max_position_embeddings = 32,
        **config_overrides,
    )
    return AutoModelForCausalLM.from_config(config)


class _ConfigCarrier(nn.Module):
    """Minimal module so the prepare step can run over an arbitrary config."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(4, 4)


@pytest.mark.parametrize("mode", [True, "unsloth"])
def test_top_level_use_cache_disabled(mode):
    model = _tiny_llama(use_cache = True)
    assert model.config.use_cache is True
    prepare_model_for_training(model, use_gradient_checkpointing = mode)
    assert model.config.use_cache is False


def test_no_gradient_checkpointing_leaves_use_cache():
    model = _tiny_llama(use_cache = True)
    prepare_model_for_training(model, use_gradient_checkpointing = False)
    assert model.config.use_cache is True


@pytest.mark.parametrize("initial", [None, False])
def test_falsy_use_cache_preserved(initial):
    # None means "defer to the model default": it must NOT be coerced to False.
    model = _tiny_llama(use_cache = initial)
    prepare_model_for_training(model, use_gradient_checkpointing = True)
    assert model.config.use_cache is initial


def test_nested_composite_config_disabled():
    Gemma3Config = pytest.importorskip("transformers").Gemma3Config
    config = Gemma3Config()
    assert isinstance(config.text_config, PreTrainedConfig)
    config.text_config.use_cache = True
    model = _ConfigCarrier(config)
    prepare_model_for_training(
        model, use_gradient_checkpointing = True, use_reentrant = False,
    )
    assert config.text_config.use_cache is False


def test_non_config_attachments_ignored():
    config = LlamaConfig(use_cache = True)
    bystander = SimpleNamespace(use_cache = True)
    config.bystander = bystander
    model = _ConfigCarrier(config)
    prepare_model_for_training(
        model, use_gradient_checkpointing = True, use_reentrant = False,
    )
    assert config.use_cache is False
    assert bystander.use_cache is True


def test_self_referencing_config_terminates():
    config = LlamaConfig(use_cache = True)
    config.self_loop = config
    model = _ConfigCarrier(config)
    prepare_model_for_training(
        model, use_gradient_checkpointing = True, use_reentrant = False,
    )
    assert config.use_cache is False


def test_restore_use_cache_after_training_prep():
    model = _tiny_llama(use_cache = True)
    prepare_model_for_training(model, use_gradient_checkpointing = True)
    assert model.config.use_cache is False
    restore_use_cache(model)
    assert model.config.use_cache is True


def test_restore_nested_composite_config():
    Gemma3Config = pytest.importorskip("transformers").Gemma3Config
    config = Gemma3Config()
    config.text_config.use_cache = True
    model = _ConfigCarrier(config)
    prepare_model_for_training(
        model, use_gradient_checkpointing = True, use_reentrant = False,
    )
    assert config.text_config.use_cache is False
    restore_use_cache(model)
    assert config.text_config.use_cache is True


def test_restore_without_prepare_is_noop():
    model = _tiny_llama(use_cache = True)
    restore_use_cache(model)
    assert model.config.use_cache is True


def test_restore_preserves_falsy_values():
    # Configs whose use_cache was None/False are never recorded, so a
    # restore after prepare must not invent True values for them.
    model = _tiny_llama(use_cache = None)
    prepare_model_for_training(model, use_gradient_checkpointing = True)
    restore_use_cache(model)
    assert model.config.use_cache is None


def test_disable_restore_cycle_keeps_originals():
    # for_inference -> for_training -> for_inference round trips.
    model = _tiny_llama(use_cache = True)
    prepare_model_for_training(model, use_gradient_checkpointing = True)
    restore_use_cache(model)
    disable_use_cache(model)
    assert model.config.use_cache is False
    restore_use_cache(model)
    assert model.config.use_cache is True


def test_double_prepare_keeps_first_originals():
    model = _tiny_llama(use_cache = True)
    prepare_model_for_training(model, use_gradient_checkpointing = True)
    prepare_model_for_training(model, use_gradient_checkpointing = True)
    restore_use_cache(model)
    assert model.config.use_cache is True
