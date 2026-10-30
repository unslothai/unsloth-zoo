# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Pins get_transformers_model_type: an unresolved config must not come back as [] (consumers index or join it), a
single peft_config adapter unwraps whatever its key is, and ambiguous multi-adapter dicts raise; CPU-only,
network-free."""

import json
import os
from pathlib import Path

import pytest

from unsloth_zoo.hf_utils import get_transformers_model_type


_UNRESOLVED_MESSAGE = "Cannot determine model type for config file"


@pytest.fixture(autouse = True)
def _no_env_leak():
    """get_transformers_model_type sets `UNSLOTH_MODEL_NAME` in os.environ for every PeftConfig input (hf_utils.py), so
    restore it to keep temp paths out of other tests."""
    sentinel = object()
    previous = os.environ.get("UNSLOTH_MODEL_NAME", sentinel)
    try:
        yield
    finally:
        if previous is sentinel:
            os.environ.pop("UNSLOTH_MODEL_NAME", None)
        else:
            os.environ["UNSLOTH_MODEL_NAME"] = previous


@pytest.fixture(scope = "module")
def local_llama_base(tmp_path_factory):
    """A real on-disk base checkpoint dir AutoConfig can resolve without network."""
    base = tmp_path_factory.mktemp("llama_base")
    (base / "config.json").write_text(json.dumps({
        "model_type"          : "llama",
        "architectures"       : ["LlamaForCausalLM"],
        "hidden_size"         : 16,
        "intermediate_size"   : 32,
        "num_attention_heads" : 1,
        "num_key_value_heads" : 1,
        "num_hidden_layers"   : 1,
        "vocab_size"          : 32,
    }))
    return str(base)


def _lora_config(base_model_name_or_path):
    from peft import LoraConfig
    return LoraConfig(
        base_model_name_or_path = base_model_name_or_path,
        r                       = 8,
        lora_alpha              = 16,
        lora_dropout            = 0.0,
        target_modules          = ["q_proj", "v_proj"],
    )



def test_single_adapter_dict_with_non_default_name_resolves(local_llama_base):
    """A single adapter under a non-default name, as `get_peft_model` with `adapter_name` produces, is unambiguous and must resolve like the default key."""
    config = {"my_adapter": _lora_config(local_llama_base)}
    assert get_transformers_model_type(config) == ["llama"]


def test_multi_adapter_dict_without_default_raises(local_llama_base):
    """Adapters may carry different base models, so no winner may be guessed."""
    config = {
        "adapter_a": _lora_config(local_llama_base),
        "adapter_b": _lora_config(local_llama_base),
    }
    with pytest.raises(TypeError, match = _UNRESOLVED_MESSAGE):
        get_transformers_model_type(config)


def test_multi_adapter_dict_with_default_keeps_default_precedence(local_llama_base):
    config = {
        "default"  : _lora_config(local_llama_base),
        "adapter_b": _lora_config(local_llama_base),
    }
    assert get_transformers_model_type(config) == ["llama"]


def test_empty_dict_raises():
    with pytest.raises(TypeError, match = _UNRESOLVED_MESSAGE):
        get_transformers_model_type({})


# --- inputs that used to yield [] and then break the consumer -----------------

def test_config_object_whose_to_dict_lacks_model_type_raises():
    class RemoteCodeConfig:
        def to_dict(self):
            return {
                "architectures" : ["MyCustomModel"],
                "hidden_size"   : 16,
                "vision_config" : {"depth": 2},
            }

    with pytest.raises(TypeError, match = _UNRESOLVED_MESSAGE):
        get_transformers_model_type(RemoteCodeConfig())


def test_config_object_without_to_dict_raises():
    """The `lambda *args, **kwargs: {}` fallback yields an empty dict to walk."""
    class NoToDict:
        architectures = ["MyCustomModel"]

    with pytest.raises(TypeError, match = _UNRESOLVED_MESSAGE):
        get_transformers_model_type(NoToDict())



def test_plain_transformers_config_unchanged():
    from transformers import LlamaConfig
    config = LlamaConfig(
        hidden_size = 16, num_hidden_layers = 1, num_attention_heads = 1,
        intermediate_size = 32, vocab_size = 32,
    )
    assert get_transformers_model_type(config) == ["llama"]


def test_peft_config_dict_keyed_default_unchanged(local_llama_base):
    config = {"default": _lora_config(local_llama_base)}
    assert get_transformers_model_type(config) == ["llama"]


# --- the swallowing call site in hf_utils.py must behave identically ---------

def test_bare_except_call_site_still_swallows(local_llama_base):
    """Bare except in get_auto_processor catches both the old IndexError and the new TypeError, leaving model_type None
    so AutoTokenizer is used; the input is an ambiguous multi-adapter dict."""
    config = {
        "adapter_a": _lora_config(local_llama_base),
        "adapter_b": _lora_config(local_llama_base),
    }
    model_type = None
    try:
        model_type = get_transformers_model_type(config)[0]
    except:
        pass
    assert model_type is None


# --- the OTHER guard (the AutoConfig fallback) must be untouched -------------

def test_peft_config_with_unresolvable_base_still_raises(tmp_path):
    """If AutoConfig cannot resolve the base, `retry_config` stays False and `model_types` is genuinely None at the guard; that path is unchanged."""
    missing = str(tmp_path / "definitely-not-a-model")
    with pytest.raises(TypeError, match = _UNRESOLVED_MESSAGE):
        get_transformers_model_type(_lora_config(missing))


def test_peft_config_without_base_model_name_raises():
    """Distinct earlier guard; proves we did not shadow it."""
    with pytest.raises(TypeError, match = "base_model_name_or_path"):
        get_transformers_model_type(_lora_config(None))
