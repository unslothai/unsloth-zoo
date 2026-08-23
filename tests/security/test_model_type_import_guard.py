# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`model_type` comes off a downloaded config.json and is interpolated into code.

`unsloth_compile_transformers` imports `transformers.models.{model_type}.modeling_{model_type}`,
and `create_new_function` names the compiled cache file from the same string, so a config
carrying `"model_type": "llama'); import os; os.system(...)"` used to reach an `exec`.
`get_transformers_model_type` is the single choke point every caller goes through.

CPU-only and network-free: a stub whose `to_dict` returns the payload is exactly how a
trust_remote_code config surfaces an arbitrary model_type.
"""

import re

import pytest

from unsloth_zoo.hf_utils import get_transformers_model_type


class _StubConfig:
    """Minimal stand-in for a (possibly remote-code) PretrainedConfig."""
    def __init__(self, model_type):
        self._model_type = model_type

    def to_dict(self):
        return {"model_type": self._model_type}


# --- rejection ---------------------------------------------------------------

@pytest.mark.parametrize("model_type", [
    "llama'); import os; os.system('touch /tmp/pwned",
    "llama import os",
    "llama\nimport os",
    "llama;os",
    "llama, os",
    "llama\x00",
    "llama)",
    "llama #",
    "lll as x; import os",
])
def test_injected_model_type_rejected(model_type):
    with pytest.raises(ValueError, match = "Invalid model_type"):
        get_transformers_model_type(_StubConfig(model_type))


def test_path_traversal_cannot_survive():
    """"/" and "." are rewritten to "_" before the guard, so traversal cannot reach
    either the import path or the compiled-cache filename."""
    for model_type in ("../../../etc/passwd", "a/../../b"):
        result = get_transformers_model_type(_StubConfig(model_type))
        assert all(re.fullmatch(r"[a-z0-9_]+", t) for t in result), result


# --- no legitimate model_type may be rejected --------------------------------

def test_every_shipped_model_type_accepted():
    """A false rejection here breaks loading a real model, so pin the whole set
    transformers ships plus the remote-code types unsloth is known to see."""
    import transformers.models
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    names = {n for n in CONFIG_MAPPING_NAMES} | {
        n for n in dir(transformers.models) if not n.startswith("_")
    }
    names |= {
        "nemotron_h", "nemotronh_nano_vl_v2", "chatglm", "minicpmv",
        "deepseek_v3", "MiniCPM-V-2_6", "Qwen2.5-VL", "gemma3_text",
    }

    rejected = []
    for name in names:
        try:
            get_transformers_model_type(_StubConfig(name))
        except ValueError as exception:
            rejected.append((name, str(exception)))
    assert not rejected, f"guard rejected legitimate model types: {rejected}"


@pytest.mark.parametrize("name", ["dbrx", "got_ocr2", "qwen3_omni_moe"])
def test_composite_config_with_empty_nested_sentinels(name):
    """`PretrainedConfig.model_type` defaults to "", so a nested sub-config that does not
    override it serialises as an empty string. The recursive walk collects those alongside
    the real top-level type, and they must not be mistaken for an injected module name."""
    from transformers import AutoConfig

    config = AutoConfig.for_model(name)
    assert name in get_transformers_model_type(config)


def test_all_empty_model_types_is_unresolved():
    """Nothing but sentinels means the architecture is still unknown, which is the
    existing "cannot determine" case rather than an injection."""
    class OnlySentinels:
        def to_dict(self):
            return {"model_type": "", "text_config": {"model_type": ""}}

    with pytest.raises(TypeError, match = "Cannot determine model type"):
        get_transformers_model_type(OnlySentinels())


def test_plain_config_unchanged():
    from transformers import LlamaConfig
    config = LlamaConfig(
        hidden_size = 16, num_hidden_layers = 1, num_attention_heads = 1,
        intermediate_size = 32, vocab_size = 32,
    )
    assert get_transformers_model_type(config) == ["llama"]
