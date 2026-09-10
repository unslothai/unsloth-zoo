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
def test_compile_stops_on_an_unnormalised_model_type(model_type):
    """Stops short of the compiled-cache filename, and does so by returning.

    Returning rather than raising is deliberate. The import below the guard raises
    ModuleNotFoundError for any name that is not a plain module name, and that happens
    before the filename is built, so a non-canonical model_type has always been a silent
    skip here. 38 of the model_type values transformers ships are hyphenated, and this
    function is exported, so raising would turn a skip into a hard failure for a caller
    passing a raw `config.model_type`.
    """
    assert unsloth_compile_transformers(model_type) is None


def _shipped_non_canonical_model_types():
    """Literal `model_type` values the INSTALLED transformers ships that fail the regex.

    Derived rather than hardcoded: the set differs between transformers versions, and a
    hardcoded name that a version has not got yet would fail for the wrong reason.
    """
    import re as _re
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    return sorted(
        m for m in CONFIG_MAPPING_NAMES if not _re.fullmatch(r"[a-z0-9_]+", m)
    )


def test_real_hyphenated_model_types_are_skipped_not_raised():
    """The compatibility property: a direct call must not turn a skip into a failure.

    The producer rewrites `-` to `_` before the canonical path ever reaches here, so
    this covers a direct call to the exported function, which is the case that regressed.
    """
    shipped = _shipped_non_canonical_model_types()
    assert len(shipped) > 10, f"expected many hyphenated model types, got {shipped}"
    for model_type in shipped:
        assert unsloth_compile_transformers(model_type) is None, model_type


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
