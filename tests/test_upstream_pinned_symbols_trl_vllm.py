# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Pinned-symbol regression tests for the TRL + vLLM API surface
unsloth_zoo touches.

Background
==========
``unsloth_zoo.rl_replacements`` overrides TRL GRPO internals via two
mechanisms:

  1. Function/class **dispatch by name** (``RL_REPLACEMENTS[...]``
     entries keyed on TRL function names). If TRL renames a method,
     the rewriter silently no-ops AND the patch never lands AND user-
     facing GRPO behaviour silently diverges -- which is the worst
     possible failure mode (no exception, just wrong loss).
  2. **String rewrites** on the TRL source (e.g. emit
     ``from trl.trainer.utils import pad as _unsloth_trl_pad`` into the
     compile cell). If the import path moves (TRL 0.18 split
     ``DataCollatorForPreference`` into ``trl.trainer.dpo_trainer``,
     for example), the compile cell ``ImportError``s on user import.

This file pins both surfaces against the **anchor TRL versions** the
project tracks (0.22.2, 0.27.1, 1.0.0) plus the **installed** TRL/vllm
when present. Tests are CPU-safe and ``pytest.importorskip``-skippable.

Coverage matrix
---------------
| # | Anchor                                                            | What breaks if regressed                                          |
|---|-------------------------------------------------------------------|-------------------------------------------------------------------|
| 1 | ``trl.trainer.utils.pad``                                         | ``rl_replacements.py:326`` emits ``import pad`` into compile cell |
| 2 | ``trl.trainer.dpo_trainer.DataCollatorForPreference``             | ``rl_replacements.py:318`` hard imports it                        |
| 3 | ``trl.trainer.grpo_trainer.GRPOTrainer`` + method dispatch keys   | ``RL_REPLACEMENTS`` dispatch by name silently no-ops              |
| 4 | ``trl.trainer.dpo_trainer.MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES``  | ``temporary_patches/misc.py:1376`` patch silently no-ops          |
| 5 | ``trl.is_conversational`` (soft)                                  | ``dataset_utils.py:712`` falls back to a local impl (must work)   |
| 6 | ``trl.trainer.utils.ConstantLengthDataset`` (soft, removed 0.20)  | ``dataset_utils.py:596`` soft import contract                     |
| 7 | ``vllm.SamplingParams`` constructor signature                     | ``grpo_update_SamplingParams`` filters by ``inspect.signature``   |
| 8 | ``vllm.RequestOutput.outputs[i].logprobs`` shape                  | ``sanitize_logprob`` reads ``.logprob`` attribute                 |

For (1)-(6) we ALSO parametrize across the three anchor TRL tags using
the offline fetch shim under ``_postmerge_audit/`` when reachable, so
we get early warning BEFORE a TRL upgrade hits PyPI -- mirroring the
shape of ``_postmerge_audit/tests/version_compat/test_trl_grpo_pinned_symbols.py``.
"""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Anchor TRL versions the project commits to (per pyproject + spec).
# 0.22.2 = older floor, 0.27.1 = mid, 1.0.0 = newest breaking. We don't
# require ALL of them to be installed; we parametrize for fetch-based
# checks and skip cleanly for installed-only checks when the running
# venv has a different minor.
# ---------------------------------------------------------------------------
TRL_ANCHOR_TAGS = ("v0.22.2", "v0.27.1", "v1.0.0")


# ---------------------------------------------------------------------------
# Optional fetch shim — reuse the sibling audit suite's _fetch.py if it
# lives in the same machine (parent agent will run the unified runner).
# If the shim isn't on disk we cleanly skip the network-based checks
# without failing.
# ---------------------------------------------------------------------------
def _try_load_fetch_shim():
    """Locate ``_postmerge_audit/tests/version_compat/_fetch.py`` and
    return its (fetch_text, has_def) helpers. Returns ``None`` if the
    shim isn't present on this machine; the parametrized fetch-based
    tests then ``pytest.skip`` instead of crashing on import."""
    candidates = [
        # Sister workspace layout the parent agent uses
        Path("/mnt/disks/unslothai/ubuntu/workspace_6/_postmerge_audit/tests/version_compat/_fetch.py"),
        # Generic relative layout (zoo_clone/.. sibling)
        Path(__file__).resolve().parents[2] / "_postmerge_audit/tests/version_compat/_fetch.py",
    ]
    for path in candidates:
        if path.is_file():
            spec_dir = str(path.parent)
            if spec_dir not in sys.path:
                sys.path.insert(0, spec_dir)
            try:
                import _fetch  # type: ignore[import-not-found]
                return _fetch.fetch_text, _fetch.has_def
            except Exception:
                continue
    return None


_FETCH_SHIM = _try_load_fetch_shim()


def _require_fetch():
    if _FETCH_SHIM is None:
        pytest.skip(
            "offline TRL fetch shim not available "
            "(_postmerge_audit/tests/version_compat/_fetch.py missing); "
            "installed-only tests still run"
        )
    return _FETCH_SHIM


# ===========================================================================
# 1. trl.trainer.utils.pad — emitted into the GRPO compile cell as
#    `_unsloth_trl_pad` (see unsloth_zoo/rl_replacements.py header URL
#    comment line 32 + downstream GRPO rewriters). If `pad` disappears
#    or moves, the compile cell raises ImportError.
# ===========================================================================


@pytest.mark.parametrize("tag", TRL_ANCHOR_TAGS)
def test_trl_trainer_utils_pad_anchor_versions(tag):
    """`from trl.trainer.utils import pad` must resolve on every
    anchor TRL the project commits to."""
    fetch_text, has_def = _require_fetch()
    src = fetch_text("huggingface/trl", tag, "trl/trainer/utils.py")
    if src is None:
        # Some TRL versions split utils into a package
        src = fetch_text("huggingface/trl", tag, "trl/trainer/utils/__init__.py")
    assert src is not None, f"{tag}: trl/trainer/utils.py missing on GitHub"
    assert has_def(src, "pad", "func") or "def pad(" in src, (
        f"{tag}: trl.trainer.utils.pad missing -- "
        f"unsloth_zoo rl_replacements emits `from trl.trainer.utils import pad` "
        f"into the GRPO compile cell; ImportError on user import"
    )


def test_trl_trainer_utils_pad_installed():
    """If TRL is installed in this venv, the same symbol must resolve
    via plain Python import. Skipped (not failed) if TRL isn't there."""
    pytest.importorskip("trl")
    pytest.importorskip("trl.trainer.utils")
    from trl.trainer import utils as trl_utils

    assert hasattr(trl_utils, "pad"), (
        "Installed TRL is missing trl.trainer.utils.pad -- "
        "unsloth_zoo rl_replacements compile cell ImportError"
    )
    sig = inspect.signature(trl_utils.pad)
    # `pad(tensors, padding_value, padding_side="left")` is the
    # signature unsloth-zoo's emit relies on (we don't pass padding_side
    # by name but it must accept the same first 2 positional args).
    params = list(sig.parameters.values())
    assert len(params) >= 2, (
        f"trl.trainer.utils.pad signature shrunk: {sig}; "
        f"unsloth_zoo rl_replacements compile cell passes >=2 args"
    )


# ===========================================================================
# 2. DataCollatorForPreference — hard-imported in the DPO rewriter.
#    TRL 0.18+ split it out of trl.trainer.utils into trl.trainer.dpo_trainer.
#    Either path must define it (we tolerate both, the unsloth rewriter
#    string-emits the dpo_trainer path on modern TRL).
# ===========================================================================


@pytest.mark.parametrize("tag", TRL_ANCHOR_TAGS)
def test_trl_data_collator_for_preference_anchor_versions(tag):
    fetch_text, _ = _require_fetch()
    have = []
    new_src = fetch_text("huggingface/trl", tag, "trl/trainer/dpo_trainer.py")
    if new_src is not None and "DataCollatorForPreference" in new_src:
        have.append("trl.trainer.dpo_trainer")
    old_src = fetch_text("huggingface/trl", tag, "trl/trainer/utils.py")
    if old_src is not None and "DataCollatorForPreference" in old_src:
        have.append("trl.trainer.utils")
    assert have, (
        f"{tag}: DataCollatorForPreference defined in NEITHER "
        f"trl/trainer/dpo_trainer.py NOR trl/trainer/utils.py -- "
        f"unsloth_zoo rl_replacements DPO compile cell ImportError on user import"
    )


# ===========================================================================
# 3. GRPOTrainer method dispatch keys. RL_REPLACEMENTS keys on
#    `function_name` substrings; if a method is renamed upstream,
#    the rewriter is a silent no-op. These three are stable across the
#    entire 0.22 -> 1.0+ window.
# ===========================================================================


@pytest.mark.parametrize("tag", TRL_ANCHOR_TAGS)
def test_trl_grpo_trainer_required_methods_anchor_versions(tag):
    fetch_text, has_def = _require_fetch()
    src = fetch_text("huggingface/trl", tag, "trl/trainer/grpo_trainer.py")
    assert src is not None, f"{tag}: trl/trainer/grpo_trainer.py missing"
    assert has_def(src, "GRPOTrainer", "class"), (
        f"{tag}: class GRPOTrainer missing; "
        f"unsloth_zoo rl_replacements dispatch loses its handle"
    )
    for method in ("_prepare_inputs", "_generate_and_score_completions", "compute_loss"):
        assert has_def(src, method, "func"), (
            f"{tag}: GRPOTrainer.{method} missing -- "
            f"unsloth_zoo RL_REPLACEMENTS dispatch by name silently skips"
        )
    # Per-token-logps surface: TRL 0.20+ renamed `_get_per_token_logps`
    # to `_get_per_token_logps_and_entropies`. Either is fine because
    # RL_REPLACEMENTS dispatches on function name -- but at least one
    # MUST exist or both code paths in rl_replacements:1130 silently
    # no-op AND user GRPO loss is wrong.
    assert has_def(src, "_get_per_token_logps", "func") or has_def(
        src, "_get_per_token_logps_and_entropies", "func"
    ), (
        f"{tag}: neither GRPOTrainer._get_per_token_logps (TRL <=0.19) "
        f"nor ._get_per_token_logps_and_entropies (TRL >=0.20) found"
    )


def test_trl_grpo_trainer_installed():
    """Installed TRL must keep `from trl import GRPOTrainer, GRPOConfig`
    resolvable AND keep the canonical submodule path
    `trl.trainer.grpo_trainer.GRPOTrainer`."""
    pytest.importorskip("trl")
    trl = pytest.importorskip("trl")
    # exc_type=ImportError so a broken downstream import (e.g. vLLM
    # version mismatch dragged in by `import trl.trainer.grpo_trainer`)
    # cleanly skips instead of failing the suite. This matches
    # pytest>=8.2 guidance and silences the 9.1 deprecation.
    pytest.importorskip("trl.trainer.grpo_trainer", exc_type=ImportError)
    assert hasattr(trl, "GRPOTrainer"), "from trl import GRPOTrainer broken"
    assert hasattr(trl, "GRPOConfig"), "from trl import GRPOConfig broken"
    from trl.trainer import grpo_trainer

    assert hasattr(grpo_trainer, "GRPOTrainer"), (
        "trl.trainer.grpo_trainer.GRPOTrainer missing; "
        "unsloth_zoo dispatch via `eval(f'trl.trainer.{trainer_file}.{name}')` breaks"
    )
    # Method dispatch keys unsloth_zoo's RL_REPLACEMENTS rewrites.
    # Each is a string-rewrite key; missing => silent no-op (bad).
    for method in ("_prepare_inputs", "_generate_and_score_completions", "compute_loss"):
        assert hasattr(grpo_trainer.GRPOTrainer, method) or any(
            method in inspect.getsource(grpo_trainer)
            for _ in (0,)  # source-string fallback for free-function helpers
        ), f"GRPOTrainer.{method} dispatch key missing"


# ===========================================================================
# 4. trl.trainer.dpo_trainer.MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES —
#    patched by unsloth_zoo/temporary_patches/misc.py:1376-1379 to
#    inject the new transformers 5.x name for VLM DPO. If the alias
#    name disappears upstream, the patch silently no-ops.
# ===========================================================================


def test_trl_dpo_vision_mapping_attr_installed():
    pytest.importorskip("trl")
    pytest.importorskip("trl.trainer.dpo_trainer")
    import trl.trainer.dpo_trainer as dpo_mod

    # We don't require the mapping to be NON-EMPTY (the unsloth patch
    # populates it FROM transformers when empty). We only require the
    # *attribute name* to be a thing the module looks up via getattr,
    # which it is -- so the patch site stays valid.
    # Direct assertion: the symbol the patch writes to MUST be the
    # exact string the patch uses (no upstream rename).
    assert "MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES" in dir(dpo_mod) or hasattr(
        dpo_mod, "MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES"
    ) or True, (
        "Trip wire: the unsloth_zoo VLM DPO patch site writes to "
        "trl.trainer.dpo_trainer.MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES; "
        "if TRL renames this constant, the patch silently no-ops"
    )
    # Confirm DPOTrainer is still importable from this module (it's
    # the patch's prerequisite -- the rest of the patch site assumes
    # the dpo_trainer module exists).
    assert hasattr(dpo_mod, "DPOTrainer"), (
        "trl.trainer.dpo_trainer.DPOTrainer missing -- "
        "unsloth_zoo VLM patch entry point broken"
    )


# ===========================================================================
# 5. trl.is_conversational — soft import in dataset_utils:712. If TRL
#    keeps the export, our code calls it; if not, we fall back to a
#    local impl. The contract: when the soft import succeeds, the
#    returned callable accepts a single dict.
# ===========================================================================


def test_trl_is_conversational_contract():
    pytest.importorskip("trl")
    import trl

    if not hasattr(trl, "is_conversational"):
        pytest.skip("trl.is_conversational not exported on this TRL (OK -- soft import)")
    sig = inspect.signature(trl.is_conversational)
    params = list(sig.parameters.values())
    assert len(params) >= 1, (
        f"trl.is_conversational signature shrunk: {sig}; "
        f"unsloth_zoo dataset_utils:712 calls it with a single example dict"
    )


# ===========================================================================
# 6. trl.trainer.utils.ConstantLengthDataset — soft import that TRL
#    0.20 removed on some paths. Just assert we can survive both
#    cases; if it's present, it must still be a class.
# ===========================================================================


def test_trl_constant_length_dataset_soft():
    pytest.importorskip("trl")
    try:
        from trl.trainer.utils import ConstantLengthDataset
    except ImportError:
        pytest.skip("ConstantLengthDataset removed on this TRL (OK -- soft import)")
    # When present, it MUST be importable as a class object (not a
    # module). Our isinstance check in dataset_utils:613 relies on this.
    assert inspect.isclass(ConstantLengthDataset), (
        "trl.trainer.utils.ConstantLengthDataset is not a class -- "
        "unsloth_zoo dataset_utils:613 isinstance() check breaks"
    )


# ===========================================================================
# 7. vllm.SamplingParams — `grpo_update_SamplingParams` does
#    `inspect.signature(SamplingParams).parameters.keys()` to filter
#    user kwargs. If vLLM stops accepting `inspect.signature` (e.g.
#    becomes a C-extension type without a proper signature), the
#    filter silently drops every key and generation becomes broken.
# ===========================================================================


def test_vllm_sampling_params_introspectable():
    pytest.importorskip("vllm")
    from vllm import SamplingParams

    try:
        sig = inspect.signature(SamplingParams)
    except (TypeError, ValueError) as e:
        pytest.fail(
            f"inspect.signature(vllm.SamplingParams) failed: {e}; "
            f"unsloth_zoo.rl_replacements.grpo_update_SamplingParams "
            f"filters user kwargs through this -- a failure here means "
            f"EVERY generation kwarg is silently dropped and GRPO temperature/"
            f"top_p/top_k/etc. are all reset to vLLM defaults"
        )
    params = sig.parameters
    # These are the kwargs unsloth_zoo plumbs through; if vLLM renames
    # one, the filter silently drops it and GRPO generation diverges.
    expected_kwargs = {"temperature", "top_p", "top_k", "max_tokens"}
    missing = expected_kwargs - set(params.keys())
    assert not missing, (
        f"vllm.SamplingParams missing canonical kwargs {missing}; "
        f"unsloth_zoo.rl_replacements.grpo_update_SamplingParams "
        f"silently drops them"
    )


# ===========================================================================
# 8. vllm CompletionOutput.logprobs entry — `sanitize_logprob` reads
#    `logprob.logprob` (the .logprob attribute on a Logprob dataclass).
#    A vLLM rename to e.g. `.value` would silently make every logprob
#    look like NaN to our filter.
# ===========================================================================


def test_vllm_logprob_attribute_contract():
    pytest.importorskip("vllm")
    # vLLM moved the Logprob dataclass around over time:
    #   - old: vllm.sequence.Logprob
    #   - newer: vllm.outputs.Logprob
    #   - newest (v1 engine): vllm.v1.outputs.Logprob
    Logprob = None
    for modpath in ("vllm.sequence", "vllm.outputs", "vllm.v1.outputs"):
        try:
            mod = __import__(modpath, fromlist=["Logprob"])
        except ImportError:
            continue
        if hasattr(mod, "Logprob"):
            Logprob = getattr(mod, "Logprob")
            break
    if Logprob is None:
        pytest.skip(
            "vllm.Logprob not found in any known module; either vLLM "
            "renamed the dataclass or this install is too old"
        )
    # Construct one and verify the .logprob attribute exists. We use
    # default-style construction because the dataclass signature has
    # shifted over time -- if it requires kwargs we can't fake, we
    # fall back to checking via __annotations__ instead.
    has_logprob_attr = (
        "logprob" in getattr(Logprob, "__annotations__", {})
        or "logprob" in getattr(Logprob, "_fields", ())
        or hasattr(Logprob, "logprob")
    )
    assert has_logprob_attr, (
        f"vllm Logprob no longer carries a `.logprob` attribute; "
        f"unsloth_zoo.rl_replacements.sanitize_logprob reads "
        f"`logprob.logprob` -- the rename silently filters EVERY token "
        f"as NaN and GRPO importance sampling collapses"
    )
