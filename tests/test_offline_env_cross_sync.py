# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression tests for the import-time offline env cross-sync: setting any of HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE or
HF_DATASETS_OFFLINE sets all three; without HF_DATASETS_OFFLINE in the sync, load_dataset() hits the network and
fails with ConnectionError."""

from __future__ import annotations

import importlib
import os
import sys

import pytest


# Packages the reload disturbs; matched by prefix, so submodules like unsloth_zoo.vision_utils and bitsandbytes.functional travel with their parent.
_RELOAD_DISTURBS = ("unsloth_zoo", "bitsandbytes")


def _is_guarded(name: str) -> bool:
    return any(
        name == root or name.startswith(root + ".")
        for root in _RELOAD_DISTURBS
    )


@pytest.fixture
def reload_zoo(monkeypatch):
    """Strips the offline env vars, then reloads unsloth_zoo; teardown restores unsloth_zoo* and bitsandbytes* in
    sys.modules, since a reload swaps in a half-initialized package and the conftest guard flags bitsandbytes swaps;
    the contract is to leave sys.modules as found."""
    for v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.delenv(v, raising = False)

    saved = {
        name: module
        for name, module in sys.modules.items()
        if _is_guarded(name)
    }

    def _reload():
        sys.modules.pop("unsloth_zoo", None)
        return importlib.import_module("unsloth_zoo")

    yield _reload

    for name in [name for name in sys.modules if _is_guarded(name)]:
        if name not in saved:
            del sys.modules[name]
    sys.modules.update(saved)


class TestOfflineCrossSync:
    def test_unset_stays_unset(self, reload_zoo):
        reload_zoo()
        assert os.environ.get("HF_HUB_OFFLINE", "0") != "1"
        assert os.environ.get("TRANSFORMERS_OFFLINE", "0") != "1"
        assert os.environ.get("HF_DATASETS_OFFLINE", "0") != "1"

    def test_hf_hub_offline_implies_all_three(self, monkeypatch, reload_zoo):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        reload_zoo()
        assert os.environ.get("HF_HUB_OFFLINE") == "1"
        assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        assert os.environ.get("HF_DATASETS_OFFLINE") == "1"

    def test_transformers_offline_implies_all_three(self, monkeypatch, reload_zoo):
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
        reload_zoo()
        assert os.environ.get("HF_HUB_OFFLINE") == "1"
        assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        assert os.environ.get("HF_DATASETS_OFFLINE") == "1"

    def test_hf_datasets_offline_implies_all_three(self, monkeypatch, reload_zoo):
        monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
        reload_zoo()
        assert os.environ.get("HF_HUB_OFFLINE") == "1"
        assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        assert os.environ.get("HF_DATASETS_OFFLINE") == "1"

    def test_truthy_spellings_trigger_cross_sync_and_normalize(self, monkeypatch, reload_zoo):
        # Any documented truthy spelling (1, true, yes, on; case-insensitive) triggers the cross-sync and normalizes all
        # three vars to the literal 1 (see _OFFLINE_TRUE in unsloth_zoo/__init__.py).
        for spelling in ("true", "yes", "on", "TRUE", "Yes", "On"):
            monkeypatch.setenv("HF_HUB_OFFLINE", spelling)
            reload_zoo()
            assert os.environ.get("HF_HUB_OFFLINE") == "1", spelling
            assert os.environ.get("TRANSFORMERS_OFFLINE") == "1", spelling
            assert os.environ.get("HF_DATASETS_OFFLINE") == "1", spelling

    def test_non_truthy_value_leaves_env_unchanged(self, monkeypatch, reload_zoo):
        monkeypatch.setenv("HF_HUB_OFFLINE", "0")
        reload_zoo()
        assert os.environ.get("HF_HUB_OFFLINE") == "0"
        assert os.environ.get("TRANSFORMERS_OFFLINE", "0") != "1"
        assert os.environ.get("HF_DATASETS_OFFLINE", "0") != "1"
