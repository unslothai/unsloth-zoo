# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression tests for get_transformers_model_type's config unwrapping and guard.

Two behaviours are pinned here:

1. An unresolved config must not come back as `[]`. `model_types` is assigned from
   `list(find(...))`, so it reaches the guard as `[]` and never as `None`; the guard
   used to test `is None`, so `[]` passed through and consumers broke on it --
   `unsloth/models/vision.py` indexes `model_types[0]`, `unsloth/models/loader.py`
   joins the list and silently matches no architecture.
2. `model.peft_config` maps adapter name -> config, and only the literal key
   "default" used to be unwrapped, so an adapter named anything else was unreadable.
   A single-adapter dict now unwraps whatever its key is; a multi-adapter dict
   without "default" stays ambiguous and must raise rather than pick a winner.

All cases are CPU-only and network-free: configs are built in a temp directory so
AutoConfig resolves from disk and never contacts the Hub.
"""

import json
import os
from pathlib import Path

import pytest

from unsloth_zoo.hf_utils import get_transformers_model_type


_UNRESOLVED_MESSAGE = "Cannot determine model type for config file"


@pytest.fixture(autouse = True)
def _no_env_leak():
    """get_transformers_model_type sets os.environ["UNSLOTH_MODEL_NAME"] as a side
    effect for every PeftConfig input (hf_utils.py). Other test modules read that
    variable, so restore it rather than leaking temp paths into the session."""
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


# --- adapter-name unwrapping -------------------------------------------------

def test_single_adapter_dict_with_non_default_name_resolves(local_llama_base):
    """`get_peft_model(..., adapter_name = "my_adapter")` produces a peft_config keyed
    by that name. One adapter is unambiguous, so it must resolve like "default" does."""
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


# --- paths that already worked must be byte-identical ------------------------

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
    """get_auto_processor wraps `get_transformers_model_type(peft_config)[0]` in a
    bare `except:`. That caught the old IndexError and catches the new TypeError
    alike, so `model_type` stays None and the caller falls through to
    AutoTokenizer. Uses an ambiguous multi-adapter dict, which still cannot
    resolve after the unwrap change."""
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
    """When AutoConfig cannot resolve the base, `retry_config` stays False, so
    `model_types` is still genuinely None at the guard. That path is unchanged."""
    missing = str(tmp_path / "definitely-not-a-model")
    with pytest.raises(TypeError, match = _UNRESOLVED_MESSAGE):
        get_transformers_model_type(_lora_config(missing))


def test_peft_config_without_base_model_name_raises():
    """Distinct earlier guard; proves we did not shadow it."""
    with pytest.raises(TypeError, match = "base_model_name_or_path"):
        get_transformers_model_type(_lora_config(None))
