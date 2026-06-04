# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for the read-only HF cache redirect in ``unsloth_zoo/hf_cache.py``.

The module is loaded directly via importlib so these tests do not import the
full ``unsloth_zoo`` package (which pulls in torch + GPU init).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

_HF_CACHE_PATH = Path(__file__).resolve().parents[1] / "unsloth_zoo" / "hf_cache.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "zoo_hf_cache_under_test", _HF_CACHE_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def hf_cache(monkeypatch):
    # Clear the cache env so each test fully controls it.
    for v in ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE"):
        monkeypatch.delenv(v, raising = False)
    return _load_module()


def test_is_writable(hf_cache, tmp_path):
    assert hf_cache._is_writable(tmp_path) is True
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    # Parent is a file, so mkdir can never succeed (reliable across uids).
    assert hf_cache._is_writable(blocker / "hub") is False


def test_writable_default_no_redirect(hf_cache, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert hf_cache.redirect_hf_cache_if_readonly() is None
    assert os.environ["HF_HOME"] == str(tmp_path)
    assert "HF_HUB_CACHE" not in os.environ


def test_explicit_writable_hub_honored(hf_cache, tmp_path, monkeypatch):
    hub = tmp_path / "myhub"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    assert hf_cache.redirect_hf_cache_if_readonly() is None
    assert os.environ["HF_HUB_CACHE"] == str(hub)


def test_readonly_hub_redirects(hf_cache, tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    monkeypatch.setenv("HF_HUB_CACHE", str(blocker / "hub"))  # not writable
    fallback = tmp_path / "fallback"
    monkeypatch.setattr(hf_cache, "_fallback_bases", lambda: [fallback])
    with pytest.warns(UserWarning):
        result = hf_cache.redirect_hf_cache_if_readonly()
    assert result == str(fallback)
    assert os.environ["HF_HOME"] == str(fallback)
    assert os.environ["HF_HUB_CACHE"] == str(fallback / "hub")
    assert os.environ["HF_XET_CACHE"] == str(fallback / "xet")
    assert hf_cache._is_writable(Path(os.environ["HF_HUB_CACHE"])) is True
