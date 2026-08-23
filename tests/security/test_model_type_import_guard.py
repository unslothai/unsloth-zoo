# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`model_type` comes off a downloaded config.json and is interpolated into code.

`unsloth_compile_transformers` builds `transformers.models.{model_type}.modeling_{model_type}`
and imports it, and `create_new_function` builds the compiled-cache filename
`unsloth_compiled_module_{model_type}.py` from the same string. A config carrying
`"model_type": "llama'); import os; os.system(...)"` therefore used to reach an
`exec`. `get_transformers_model_type` now rejects anything that is not a plain
module name, which is the single choke point: every caller in unsloth
(`models/loader.py`) takes its `model_type` from this function.

CPU-only and network-free - configs are stubs whose `to_dict` returns the payload,
which is exactly how a trust_remote_code config surfaces an arbitrary model_type.
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
    "",
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


def test_plain_config_unchanged():
    from transformers import LlamaConfig
    config = LlamaConfig(
        hidden_size = 16, num_hidden_layers = 1, num_attention_heads = 1,
        intermediate_size = 32, vocab_size = 32,
    )
    assert get_transformers_model_type(config) == ["llama"]
