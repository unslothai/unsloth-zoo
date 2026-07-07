# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression tests for the offline env cross-sync at import time.

``unsloth_zoo/__init__.py`` cross-syncs three env vars so setting any one
implies all three: ``HF_HUB_OFFLINE`` (huggingface_hub),
``TRANSFORMERS_OFFLINE`` (transformers), ``HF_DATASETS_OFFLINE`` (datasets).
Without ``HF_DATASETS_OFFLINE`` in the sync, ``load_dataset()`` still hits the
network for metadata and fails with ``ConnectionError`` instead of using cache.

Tests reload ``unsloth_zoo`` with each env-var combination preset.
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest


@pytest.fixture
def reload_zoo(monkeypatch):
    """Strip the three offline env vars, then reload ``unsloth_zoo`` so its
    import-time cross-sync runs."""
    for v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.delenv(v, raising = False)

    def _reload():
        sys.modules.pop("unsloth_zoo", None)
        return importlib.import_module("unsloth_zoo")

    return _reload


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
        # Any documented truthy spelling ({"1","true","yes","on"},
        # case-insensitive) triggers the cross-sync and normalizes all three
        # vars to the literal "1" (see _OFFLINE_TRUE in unsloth_zoo/__init__.py).
        for spelling in ("true", "yes", "on", "TRUE", "Yes", "On"):
            monkeypatch.setenv("HF_HUB_OFFLINE", spelling)
            reload_zoo()
            assert os.environ.get("HF_HUB_OFFLINE") == "1", spelling
            assert os.environ.get("TRANSFORMERS_OFFLINE") == "1", spelling
            assert os.environ.get("HF_DATASETS_OFFLINE") == "1", spelling

    def test_non_truthy_value_leaves_env_unchanged(self, monkeypatch, reload_zoo):
        # Values outside _OFFLINE_TRUE are passed through untouched.
        monkeypatch.setenv("HF_HUB_OFFLINE", "0")
        reload_zoo()
        assert os.environ.get("HF_HUB_OFFLINE") == "0"
        assert os.environ.get("TRANSFORMERS_OFFLINE", "0") != "1"
        assert os.environ.get("HF_DATASETS_OFFLINE", "0") != "1"
