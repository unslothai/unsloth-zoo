# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""PR 1083 validated model_type at the producer; unsloth_compile_transformers and create_new_function re-check it and
reject every payload before any import or file write, so a skipped or future choke point cannot reach an import or
write outside the compile cache; CPU-only, network-free"""

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
def test_compile_stops_on_an_unnormalised_model_type(model_type):
    """Returns None, not raises, for non-canonical names: raising would turn the silent skip into a hard failure for
    callers passing raw config.model_type (38 shipped names are hyphenated)."""
    assert unsloth_compile_transformers(model_type) is None


def _shipped_non_canonical_model_types():
    """Literal model_type values the installed transformers ships that fail the regex; derived, not hardcoded, since the
    set varies by version and a missing name fails for the wrong reason."""
    import re as _re
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    return sorted(
        m for m in CONFIG_MAPPING_NAMES if not _re.fullmatch(r"[a-z0-9_]+", m)
    )


def test_real_hyphenated_model_types_are_skipped_not_raised():
    """A direct call must return None, not raise, on hyphenated names: the producer rewrites - to _ first, so only direct calls reach the function with them."""
    shipped = _shipped_non_canonical_model_types()
    assert len(shipped) > 10, f"expected many hyphenated model types, got {shipped}"
    for model_type in shipped:
        assert unsloth_compile_transformers(model_type) is None, model_type


@pytest.mark.parametrize("model_type", [p for p in PAYLOADS if p != ""])
def test_payload_cannot_reach_a_module_path(model_type):
    """The string is interpolated into an import path; the empty string is excluded since it only yields a wrong path, and the rejection test already covers it."""
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
    """Everything the choke point emits must pass the sink guard, since a false rejection breaks loading a real model;
    mirrors test_model_type_import_guard.py so they cannot drift."""
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
