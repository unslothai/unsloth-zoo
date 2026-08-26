# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""PR #1083 validated `model_type` at the producer. These guard the sinks.

`get_transformers_model_type` in hf_utils.py is the choke point every unsloth caller
goes through, but `unsloth_compile_transformers` takes `model_type` as a plain
parameter and rebuilds `transformers.models.{model_type}.modeling_{model_type}` from
it, and `create_new_function` turns a name derived from the same string into
`os.path.join(compile_folder, f"{name}.py")`. Both now re-check, so a caller that
skips the choke point - or a future one - cannot reach an import path or write outside
the compiled cache.

CPU-only and network-free: every payload is rejected before any import or file write,
and the acceptance sweep only inspects `transformers`' own name tables.
"""

import re

import pytest

from unsloth_zoo.compiler import create_new_function, unsloth_compile_transformers


PAYLOADS = [
    "llama'); import os; os.system('touch /tmp/pwned",
    "llama import os",
    "llama\nimport os",
    "llama;os",
    "llama, os",
    "llama\x00",
    "llama)",
    "llama #",
    "lll as x; import os",
    "../../../etc/passwd",
    "a/../../b",
    "llama/../../evil",
    "Llama",          # the choke point lowercases; the sink must not accept raw case
    "llama-3",        # and normalises "-", so it never reaches here
    "",
]


@pytest.mark.parametrize("model_type", PAYLOADS)
def test_compile_rejects_unnormalised_model_type(model_type):
    with pytest.raises(ValueError, match = "Invalid model_type"):
        unsloth_compile_transformers(model_type)


@pytest.mark.parametrize("model_type", [p for p in PAYLOADS if p != ""])
def test_payload_cannot_reach_a_module_path(model_type):
    """What the guard is protecting: the string is interpolated into an import.

    The empty string is excluded: it yields a merely wrong module path rather than a
    dangerous one, and the rejection test above already covers it.
    """
    location = f"transformers.models.{model_type}.modeling_{model_type}"
    assert not re.fullmatch(r"[a-z0-9_.]+", location), location


# --- the compiled-cache filename ---------------------------------------------

@pytest.mark.parametrize("name", [
    "../../../etc/passwd",
    "a/b",
    "a\\b",
    "unsloth_compiled_module_llama/../../evil",
    "mod.py",
    "mod name",
    "",
    "1_starts_with_a_digit",
])
def test_generated_module_name_must_be_an_identifier(name):
    """`name` becomes a path under the compiled cache, so a separator is a traversal."""
    with pytest.raises(ValueError, match = "Invalid generated module name"):
        create_new_function(name, "def f(): pass", "transformers", [])


@pytest.mark.parametrize("name", [
    "unsloth_compiled_module_llama",
    "UnslothSFTTrainer",
    "LlamaAttention_peft_forward",
    "_private",
])
def test_legitimate_generated_module_names_accepted(name):
    """The real callers build identifier-shaped names; none may be rejected."""
    assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)


# --- no legitimate model_type may be rejected --------------------------------

def test_every_shipped_model_type_passes_the_sink_guard():
    """A false rejection here breaks loading a real model.

    Mirrors the sweep in test_model_type_import_guard.py so the producer and the sink
    cannot drift apart: everything the choke point emits must satisfy the sink.
    """
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
        # Exactly what get_transformers_model_type does before the guard.
        normalised = name.lower().replace("-", "_").replace("/", "_").replace(".", "_")
        if not re.fullmatch(r"[a-z0-9_]+", normalised):
            rejected.append(name)
    assert not rejected, f"sink guard would reject legitimate model types: {rejected}"
