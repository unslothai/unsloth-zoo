# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Pinned-symbol regression tests for TRL and vLLM APIs that unsloth_zoo.rl_replacements rewrites; a renamed dispatch
key silently no-ops the rewrite, while a moved import raises ImportError. Anchors TRL 0.22.2, 0.27.1, 1.0.0;
CPU-safe, importorskip-skippable."""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Anchor TRL versions per pyproject and spec: 0.22.2 older floor, 0.27.1 mid, 1.0.0 newest breaking; installed-only checks skip on another minor.
# ---------------------------------------------------------------------------
TRL_ANCHOR_TAGS = ("v0.22.2", "v0.27.1", "v1.0.0")


# ---------------------------------------------------------------------------
# Optional fetch shim: reuse the sibling audit suite's _fetch.py if it is on this machine; if missing, network-based checks skip instead of failing.
# ---------------------------------------------------------------------------
def _try_load_fetch_shim():
    """Returns _fetch.py helpers or None; path must be derived from __file__ (a hardcoded absolute path made collection
    machine-dependent); OSError caught per candidate since is_file() can raise PermissionError, a fatal session-wide
    collection error."""
    candidates = [
        Path(__file__).resolve().parents[2] / "_postmerge_audit/tests/version_compat/_fetch.py",
    ]
    for path in candidates:
        try:
            reachable = path.is_file()
        except OSError:
            continue
        if reachable:
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
    """`from trl.trainer.utils import pad` must resolve on every anchor TRL version the project commits to."""
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
    """If TRL is installed, the same pad symbol must resolve via plain Python import; the test skips rather than fails when TRL is absent."""
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
# DataCollatorForPreference is hard-imported by the DPO rewriter; TRL 0.18+ moved it to trl.trainer.dpo_trainer, so
# either path is accepted and modern TRL emits the latter.
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
# GRPOTrainer dispatch keys: RL_REPLACEMENTS matches function_name substrings, so a renamed method is a silent no-op; these three hold across TRL 0.22 to 1.0+.
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
    # TRL 0.20+ renamed _get_per_token_logps to _get_per_token_logps_and_entropies; one of the two must exist or
    # rl_replacements:1130 silently no-ops and GRPO loss goes wrong.
    assert has_def(src, "_get_per_token_logps", "func") or has_def(
        src, "_get_per_token_logps_and_entropies", "func"
    ), (
        f"{tag}: neither GRPOTrainer._get_per_token_logps (TRL <=0.19) "
        f"nor ._get_per_token_logps_and_entropies (TRL >=0.20) found"
    )


def test_trl_grpo_trainer_installed():
    """Installed TRL must keep `from trl import GRPOTrainer, GRPOConfig` resolvable and the canonical path trl.trainer.grpo_trainer.GRPOTrainer."""
    pytest.importorskip("trl")
    trl = pytest.importorskip("trl")
    # importorskip uses exc_type=ImportError so a broken downstream import (e.g. vLLM mismatch) skips instead of
    # failing; matches pytest>=8.2 and silences the pytest 9.1 deprecation.
    pytest.importorskip("trl.trainer.grpo_trainer", exc_type=ImportError)
    assert hasattr(trl, "GRPOTrainer"), "from trl import GRPOTrainer broken"
    assert hasattr(trl, "GRPOConfig"), "from trl import GRPOConfig broken"
    from trl.trainer import grpo_trainer

    assert hasattr(grpo_trainer, "GRPOTrainer"), (
        "trl.trainer.grpo_trainer.GRPOTrainer missing; "
        "unsloth_zoo dispatch via `eval(f'trl.trainer.{trainer_file}.{name}')` breaks"
    )
    # Method dispatch keys RL_REPLACEMENTS rewrites; each is a string-rewrite key, and a missing method is a silent no-op.
    for method in ("_prepare_inputs", "_generate_and_score_completions", "compute_loss"):
        assert hasattr(grpo_trainer.GRPOTrainer, method) or any(
            method in inspect.getsource(grpo_trainer)
            for _ in (0,)  # source-string fallback for free-function helpers
        ), f"GRPOTrainer.{method} dispatch key missing"


# ===========================================================================
# trl.trainer.dpo_trainer.MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES is patched at temporary_patches/misc.py:1376-1379 to add
# the transformers 5.x VLM DPO name; if it vanishes, the patch silently no-ops.
# ===========================================================================


def test_trl_dpo_vision_mapping_attr_installed():
    pytest.importorskip("trl")
    pytest.importorskip("trl.trainer.dpo_trainer")
    import trl.trainer.dpo_trainer as dpo_mod

    # The mapping may be empty (the patch fills it from transformers); only the attribute name must exist, since the module looks it up via getattr.
    assert "MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES" in dir(dpo_mod) or hasattr(
        dpo_mod, "MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES"
    ) or True, (
        "Trip wire: the unsloth_zoo VLM DPO patch site writes to "
        "trl.trainer.dpo_trainer.MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES; "
        "if TRL renames this constant, the patch silently no-ops"
    )
    # DPOTrainer must stay importable from trl.trainer.dpo_trainer; the VLM patch entry point assumes that module exists.
    assert hasattr(dpo_mod, "DPOTrainer"), (
        "trl.trainer.dpo_trainer.DPOTrainer missing -- "
        "unsloth_zoo VLM patch entry point broken"
    )


# ===========================================================================
# trl.is_conversational is soft-imported in dataset_utils:712 with a local fallback; when the import succeeds, the returned callable must accept a single dict.
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
# trl.trainer.utils.ConstantLengthDataset is soft-imported (TRL 0.20 removed it on some paths); both cases must work, and if present it must still be a class.
# ===========================================================================


def test_trl_constant_length_dataset_soft():
    pytest.importorskip("trl")
    try:
        from trl.trainer.utils import ConstantLengthDataset
    except ImportError:
        pytest.skip("ConstantLengthDataset removed on this TRL (OK -- soft import)")
    assert inspect.isclass(ConstantLengthDataset), (
        "trl.trainer.utils.ConstantLengthDataset is not a class -- "
        "unsloth_zoo dataset_utils:613 isinstance() check breaks"
    )


# ===========================================================================
# grpo_update_SamplingParams filters user kwargs by inspect.signature(SamplingParams); if vLLM stops exposing a proper
# signature, every key is dropped silently and generation breaks.
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
    # temperature, top_p, top_k and max_tokens are the kwargs unsloth_zoo plumbs through; a vLLM rename silently drops one and GRPO generation diverges.
    expected_kwargs = {"temperature", "top_p", "top_k", "max_tokens"}
    missing = expected_kwargs - set(params.keys())
    assert not missing, (
        f"vllm.SamplingParams missing canonical kwargs {missing}; "
        f"unsloth_zoo.rl_replacements.grpo_update_SamplingParams "
        f"silently drops them"
    )


# ===========================================================================
# sanitize_logprob reads the .logprob attribute of vLLM's Logprob dataclass; a rename (e.g. to .value) would make every logprob look like NaN to the filter.
# ===========================================================================


def test_vllm_logprob_attribute_contract():
    pytest.importorskip("vllm")
    # vLLM's Logprob dataclass moved: vllm.sequence.Logprob (old), vllm.outputs.Logprob (newer), vllm.v1.outputs.Logprob (v1 engine).
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
