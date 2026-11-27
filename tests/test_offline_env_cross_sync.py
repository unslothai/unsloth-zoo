# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Offline env cross-sync: any of HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE or HF_DATASETS_OFFLINE sets
all three; without the last, load_dataset() hits the network (ConnectionError)."""

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
    """Strips offline env vars and reloads unsloth_zoo; teardown restores sys.modules as found."""
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
        # Truthy spellings (1/true/yes/on, any case) set all three vars to 1 (_OFFLINE_TRUE).
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
