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


# The package the fixture reloads, plus what that reload drags in with it. A
# prefix match, so submodules (`unsloth_zoo.vision_utils`,
# `bitsandbytes.functional`) travel with their parent.
_RELOAD_DISTURBS = ("unsloth_zoo", "bitsandbytes")


def _is_guarded(name: str) -> bool:
    return any(
        name == root or name.startswith(root + ".")
        for root in _RELOAD_DISTURBS
    )


@pytest.fixture
def reload_zoo(monkeypatch):
    """Strip the three offline env vars, then reload ``unsloth_zoo`` so its
    import-time cross-sync runs.

    Reloading swaps a fresh, half-initialized ``unsloth_zoo`` object into
    ``sys.modules`` -- one that lacks submodule attributes (e.g.
    ``vision_utils``) the package only binds when they are first imported.
    Snapshot the original ``unsloth_zoo*`` modules and restore them on
    teardown so the reload does not leak that shell into the rest of the
    session and break later tests that resolve submodules via string paths
    (e.g. ``monkeypatch.setattr("unsloth_zoo.vision_utils.<attr>", ...)``).

    ``bitsandbytes*`` is snapshotted for the same reason and was missed the
    first time: importing ``unsloth_zoo`` pulls bitsandbytes in, so a reload
    can swap whatever was already in ``sys.modules`` under that name -- a stub
    an earlier test installed, or the real package -- for the other one. The
    conftest teardown guard catches exactly this and named this fixture:

        test_unset_stays_unset replaced ['bitsandbytes'] in sys.modules and
        did not put it back, so every later test in this process imports the
        substitute

    Restoring means "leave sys.modules as this fixture found it", which is the
    fixture's contract for unsloth_zoo already; bitsandbytes is not special.
    """
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
