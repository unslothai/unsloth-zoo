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
    for v in ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "HF_TOKEN", "HF_TOKEN_PATH"):
        monkeypatch.delenv(v, raising = False)
    return _load_module()


def _readonly_hub(tmp_path, monkeypatch):
    # Point HF_HUB_CACHE under a file so mkdir can never succeed.
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    monkeypatch.setenv("HF_HUB_CACHE", str(blocker / "hub"))


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
    _readonly_hub(tmp_path, monkeypatch)
    fallback = tmp_path / "fallback"
    monkeypatch.setattr(hf_cache, "_fallback_bases", lambda: [fallback])
    with pytest.warns(UserWarning):
        result = hf_cache.redirect_hf_cache_if_readonly()
    assert result == str(fallback)
    assert os.environ["HF_HOME"] == str(fallback)
    assert os.environ["HF_HUB_CACHE"] == str(fallback / "hub")
    assert os.environ["HF_XET_CACHE"] == str(fallback / "xet")
    assert hf_cache._is_writable(Path(os.environ["HF_HUB_CACHE"])) is True


def test_home_resolution_failure_redirects(hf_cache, tmp_path, monkeypatch):
    # Unresolvable home (arbitrary-uid containers) must not crash the import
    # and must fall through to a writable fallback.
    def _no_home():
        raise RuntimeError("could not determine home directory")
    monkeypatch.setattr(hf_cache.Path, "home", staticmethod(_no_home))
    fallback = tmp_path / "fallback"
    monkeypatch.setattr(hf_cache, "_fallback_bases", lambda: [fallback])
    with pytest.warns(UserWarning):
        result = hf_cache.redirect_hf_cache_if_readonly()
    assert result == str(fallback)
    assert os.environ["HF_HUB_CACHE"] == str(fallback / "hub")


def test_symlinked_fallback_rejected(hf_cache, tmp_path, monkeypatch):
    _readonly_hub(tmp_path, monkeypatch)
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target)
    good = tmp_path / "good"
    monkeypatch.setattr(hf_cache, "_fallback_bases", lambda: [link, good])
    with pytest.warns(UserWarning):
        result = hf_cache.redirect_hf_cache_if_readonly()
    assert result == str(good)
    assert os.environ["HF_HOME"] == str(good)


@pytest.mark.skipif(os.name != "posix", reason = "POSIX permission bits")
def test_fallback_clamped_to_0700(hf_cache, tmp_path, monkeypatch):
    _readonly_hub(tmp_path, monkeypatch)
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    fallback.chmod(0o777)
    monkeypatch.setattr(hf_cache, "_fallback_bases", lambda: [fallback])
    with pytest.warns(UserWarning):
        assert hf_cache.redirect_hf_cache_if_readonly() == str(fallback)
    assert (fallback.stat().st_mode & 0o777) == 0o700


def test_blocked_xet_rejects_candidate(hf_cache, tmp_path, monkeypatch):
    _readonly_hub(tmp_path, monkeypatch)
    bad = tmp_path / "bad"
    (bad / "hub").mkdir(parents = True)
    (bad / "xet").write_text("")  # file blocks the xet dir
    good = tmp_path / "good"
    monkeypatch.setattr(hf_cache, "_fallback_bases", lambda: [bad, good])
    with pytest.warns(UserWarning):
        result = hf_cache.redirect_hf_cache_if_readonly()
    assert result == str(good)
    assert os.environ["HF_XET_CACHE"] == str(good / "xet")


def test_token_path_preserved_on_redirect(hf_cache, tmp_path, monkeypatch):
    old_home = tmp_path / "oldhome"
    old_home.mkdir()
    (old_home / "token").write_text("hf_dummy")
    (old_home / "hub").write_text("")  # file, so the hub cache is unwritable
    monkeypatch.setenv("HF_HOME", str(old_home))
    fallback = tmp_path / "fallback"
    monkeypatch.setattr(hf_cache, "_fallback_bases", lambda: [fallback])
    with pytest.warns(UserWarning):
        hf_cache.redirect_hf_cache_if_readonly()
    assert os.environ["HF_TOKEN_PATH"] == str(old_home / "token")


def test_explicit_token_env_not_overridden(hf_cache, tmp_path, monkeypatch):
    old_home = tmp_path / "oldhome"
    old_home.mkdir()
    (old_home / "token").write_text("hf_dummy")
    (old_home / "hub").write_text("")
    monkeypatch.setenv("HF_HOME", str(old_home))
    monkeypatch.setenv("HF_TOKEN", "hf_explicit")
    fallback = tmp_path / "fallback"
    monkeypatch.setattr(hf_cache, "_fallback_bases", lambda: [fallback])
    with pytest.warns(UserWarning):
        hf_cache.redirect_hf_cache_if_readonly()
    assert "HF_TOKEN_PATH" not in os.environ
