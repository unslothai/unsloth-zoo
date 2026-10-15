# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""CPU-pure behavioural tests for `use_cache` handling in
`prepare_model_for_training` (unsloth_zoo/training_utils.py).

The prepare step walks `model.config` and every nested transformers config and
disables the KV cache, which gradient checkpointing makes dead weight. These
tests pin that contract, including its restore half.
"""

from __future__ import annotations

import copy
import pickle
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


def _none_use_cache_supported() -> bool:
    """v5 configs type use_cache as bool, so None is unrepresentable there."""
    try:
        LlamaConfig(use_cache = None)
    except Exception:
        return False
    return True


requires_none_use_cache = pytest.mark.skipif(
    not _none_use_cache_supported(),
    reason = "strict transformers configs type use_cache as bool; None is unrepresentable",
)


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


@pytest.mark.parametrize(
    "initial",
    [pytest.param(None, marks = requires_none_use_cache), False],
)
def test_falsy_use_cache_preserved(initial):
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


@requires_none_use_cache
def test_restore_preserves_falsy_values():
    # falsy values are never recorded, so restore must not invent True
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


# Configs that never declared use_cache: transformers sub-configs inherit no
# default, so a forward reading self.config.use_cache raised AttributeError.
# stepfun-ai/Step-3.7-Flash ships a text config of exactly this shape.


class _NoUseCacheConfig(PreTrainedConfig):
    model_type = "unsloth_no_use_cache_probe"


def _composite_without_use_cache():
    config = LlamaConfig(use_cache = True)
    config.text_config = _NoUseCacheConfig()
    assert not hasattr(config.text_config, "use_cache")
    return config


def test_absent_use_cache_is_set_so_forward_can_read_it():
    config = _composite_without_use_cache()
    model = _ConfigCarrier(config)
    prepare_model_for_training(
        model, use_gradient_checkpointing = True, use_reentrant = False,
    )
    # the attribute now exists, so the read returns False instead of raising
    assert config.text_config.use_cache is False


def test_restore_removes_an_invented_use_cache_rather_than_inventing_False():
    """Without _ABSENT, restore writes False back and disables the KV cache on
    a config the checkpoint never shipped one for."""
    config = _composite_without_use_cache()
    model = _ConfigCarrier(config)
    prepare_model_for_training(
        model, use_gradient_checkpointing = True, use_reentrant = False,
    )
    assert config.text_config.use_cache is False
    restore_use_cache(model)
    assert not hasattr(config.text_config, "use_cache")
    # the sibling that really had one is restored to its own value, not deleted
    assert config.use_cache is True


def test_absent_use_cache_survives_a_disable_restore_cycle():
    config = _composite_without_use_cache()
    model = _ConfigCarrier(config)
    prepare_model_for_training(
        model, use_gradient_checkpointing = True, use_reentrant = False,
    )
    restore_use_cache(model)
    disable_use_cache(model)
    assert config.text_config.use_cache is False
    restore_use_cache(model)
    assert not hasattr(config.text_config, "use_cache")


@pytest.mark.parametrize(
    "clone",
    [
        pytest.param(lambda m: copy.deepcopy(m), id = "deepcopy"),
        pytest.param(lambda m: pickle.loads(pickle.dumps(m)), id = "pickle"),
    ],
)
def test_absent_marker_survives_copying_the_model(clone):
    """The record is copied with the model, so an object() sentinel would lose
    identity and restore would write the unserializable sentinel into the
    config. TRL deepcopies a prepared model for its reference model."""
    config = _composite_without_use_cache()
    model = _ConfigCarrier(config)
    prepare_model_for_training(
        model, use_gradient_checkpointing = True, use_reentrant = False,
    )
    copied = clone(model)
    restore_use_cache(copied)
    assert not hasattr(copied.config.text_config, "use_cache")
    # and the config is still serializable, which the sentinel leak broke
    copied.config.text_config.to_json_string()


# A config first reached on a LATER disable_use_cache call: recording once
# meant it was disabled but never recorded, so restore could not undo it.

_ABSENT_PARAM = object()


@pytest.mark.parametrize(
    "initial",
    [pytest.param(True, id = "had_True"), pytest.param(_ABSENT_PARAM, id = "absent")],
)
def test_config_first_seen_after_the_first_disable_is_restorable(initial):
    model = _tiny_llama(use_cache = True)
    prepare_model_for_training(model, use_gradient_checkpointing = True)

    late = _NoUseCacheConfig()
    if initial is not _ABSENT_PARAM:
        late.use_cache = initial
    model.config.text_config = late      # attached while already prepared

    disable_use_cache(model)
    assert late.use_cache is False       # still disabled for training

    restore_use_cache(model)
    if initial is _ABSENT_PARAM:
        assert not hasattr(late, "use_cache")
    else:
        assert late.use_cache is initial


def test_late_config_does_not_disturb_the_first_baseline():
    # the config recorded on the first pass keeps its own original value
    model = _tiny_llama(use_cache = True)
    prepare_model_for_training(model, use_gradient_checkpointing = True)
    model.config.text_config = _NoUseCacheConfig()
    disable_use_cache(model)
    restore_use_cache(model)
    assert model.config.use_cache is True


class _EngineLike(nn.Module):
    """DeepSpeedEngine shape: its own dict config, the model on .module, other attributes forwarded."""

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.config = {"zero_optimization": {"stage": 2}}

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def test_disable_after_restore_through_a_deepspeed_style_wrapper():
    model = _tiny_llama(use_cache = True)
    disable_use_cache(model)
    engine = _EngineLike(model)
    restore_use_cache(engine)
    assert model.config.use_cache is True
    disable_use_cache(engine)
    assert model.config.use_cache is False
    assert engine.config == {"zero_optimization": {"stage": 2}}
