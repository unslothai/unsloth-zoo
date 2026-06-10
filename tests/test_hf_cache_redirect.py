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
    for v in (
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_XET_CACHE",
        "XDG_CACHE_HOME",
        "HF_TOKEN",
        "HF_TOKEN_PATH",
    ):
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
    monkeypatch.setenv("HF_XET_CACHE", str(tmp_path / "myxet"))
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


def test_xdg_cache_home_writable_no_redirect(hf_cache, tmp_path, monkeypatch):
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg))
    assert hf_cache.redirect_hf_cache_if_readonly() is None
    assert "HF_HUB_CACHE" not in os.environ
    # The probe must target where Hub will actually write.
    assert (xdg / "huggingface" / "hub").is_dir()


def test_xdg_cache_home_blocked_redirects(hf_cache, tmp_path, monkeypatch):
    xdg_blocker = tmp_path / "xdg_blocker"
    xdg_blocker.write_text("")  # file, so nothing can be created beneath it
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg_blocker))
    fallback = tmp_path / "fallback"
    monkeypatch.setattr(hf_cache, "_fallback_bases", lambda: [fallback])
    with pytest.warns(UserWarning):
        assert hf_cache.redirect_hf_cache_if_readonly() == str(fallback)
    assert os.environ["HF_HUB_CACHE"] == str(fallback / "hub")


def test_legacy_hub_cache_env_honored(hf_cache, tmp_path, monkeypatch):
    legacy = tmp_path / "legacy_hub"
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(legacy))
    monkeypatch.setenv("HF_XET_CACHE", str(tmp_path / "myxet"))
    assert hf_cache.redirect_hf_cache_if_readonly() is None
    assert "HF_HUB_CACHE" not in os.environ


def test_legacy_hub_cache_env_blocked_redirects(hf_cache, tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(blocker / "hub"))
    fallback = tmp_path / "fallback"
    monkeypatch.setattr(hf_cache, "_fallback_bases", lambda: [fallback])
    with pytest.warns(UserWarning):
        assert hf_cache.redirect_hf_cache_if_readonly() == str(fallback)
    assert os.environ["HF_HUB_CACHE"] == str(fallback / "hub")


def test_env_vars_in_hub_path_expanded(hf_cache, tmp_path, monkeypatch):
    root = tmp_path / "root"
    monkeypatch.setenv("MY_CACHE_ROOT", str(root))
    monkeypatch.setenv("HF_HUB_CACHE", "$MY_CACHE_ROOT/hub")
    monkeypatch.setenv("HF_XET_CACHE", str(root / "xet"))
    monkeypatch.chdir(tmp_path)
    assert hf_cache.redirect_hf_cache_if_readonly() is None
    # Probed the expanded path, not a literal "$MY_CACHE_ROOT" directory.
    assert (root / "hub").is_dir()
    assert not (tmp_path / "$MY_CACHE_ROOT").exists()


def test_xet_env_value_probed_literally(hf_cache, tmp_path, monkeypatch):
    # Hub does NOT expandvars/expanduser HF_XET_CACHE, so the probe must
    # target the same literal path Hub will use.
    root = tmp_path / "root"
    monkeypatch.setenv("MY_CACHE_ROOT", str(root))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hfhome"))
    monkeypatch.setenv("HF_XET_CACHE", "$MY_CACHE_ROOT/xet")
    monkeypatch.chdir(tmp_path)
    assert hf_cache.redirect_hf_cache_if_readonly() is None
    assert (tmp_path / "$MY_CACHE_ROOT" / "xet").is_dir()
    assert not (root / "xet").exists()


def test_explicit_symlinked_hub_cache_honored(hf_cache, tmp_path, monkeypatch):
    # Users symlink caches to large volumes; Hub writes through the link, so
    # a writable symlinked cache must not be redirected away.
    target = tmp_path / "real_hub"
    target.mkdir()
    link = tmp_path / "hub_link"
    link.symlink_to(target, target_is_directory = True)
    monkeypatch.setenv("HF_HUB_CACHE", str(link))
    monkeypatch.setenv("HF_XET_CACHE", str(tmp_path / "xet"))
    assert hf_cache.redirect_hf_cache_if_readonly() is None
    assert os.environ["HF_HUB_CACHE"] == str(link)


def test_explicit_hub_cache_survives_home_failure(hf_cache, tmp_path, monkeypatch):
    def _no_home():
        raise RuntimeError("could not determine home directory")
    monkeypatch.setattr(hf_cache.Path, "home", staticmethod(_no_home))
    hub = tmp_path / "myhub"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    fallback = tmp_path / "fallback"
    monkeypatch.setattr(hf_cache, "_fallback_bases", lambda: [fallback])
    # The hub cache stays put; only the unresolvable xet default moves.
    with pytest.warns(UserWarning):
        assert hf_cache.redirect_hf_cache_if_readonly() is None
    assert os.environ["HF_HUB_CACHE"] == str(hub)
    assert os.environ["HF_XET_CACHE"] == str(fallback / "xet")


def test_child_symlink_in_fallback_rejected(hf_cache, tmp_path, monkeypatch):
    _readonly_hub(tmp_path, monkeypatch)
    target = tmp_path / "target"
    target.mkdir()
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "hub").symlink_to(target)
    good = tmp_path / "good"
    monkeypatch.setattr(hf_cache, "_fallback_bases", lambda: [bad, good])
    with pytest.warns(UserWarning):
        assert hf_cache.redirect_hf_cache_if_readonly() == str(good)
    assert os.environ["HF_HUB_CACHE"] == str(good / "hub")


def test_explicit_writable_xet_kept_on_redirect(hf_cache, tmp_path, monkeypatch):
    _readonly_hub(tmp_path, monkeypatch)
    xet = tmp_path / "fast_xet"
    monkeypatch.setenv("HF_XET_CACHE", str(xet))
    fallback = tmp_path / "fallback"
    monkeypatch.setattr(hf_cache, "_fallback_bases", lambda: [fallback])
    with pytest.warns(UserWarning):
        assert hf_cache.redirect_hf_cache_if_readonly() == str(fallback)
    assert os.environ["HF_XET_CACHE"] == str(xet)


def test_xet_only_breakage_moves_only_xet(hf_cache, tmp_path, monkeypatch):
    home = tmp_path / "hfhome"
    (home / "hub").mkdir(parents = True)
    (home / "xet").write_text("")  # file blocks the xet dir
    monkeypatch.setenv("HF_HOME", str(home))
    fallback = tmp_path / "fallback"
    monkeypatch.setattr(hf_cache, "_fallback_bases", lambda: [fallback])
    with pytest.warns(UserWarning):
        assert hf_cache.redirect_hf_cache_if_readonly() is None
    assert os.environ["HF_HOME"] == str(home)
    assert "HF_HUB_CACHE" not in os.environ
    assert os.environ["HF_XET_CACHE"] == str(fallback / "xet")


def test_resolve_hf_home_none_on_failure(hf_cache, monkeypatch):
    def _no_home():
        raise RuntimeError("could not determine home directory")
    monkeypatch.setattr(hf_cache.Path, "home", staticmethod(_no_home))
    assert hf_cache._resolve_hf_home() is None


def test_safe_user_uid_fallback(hf_cache, monkeypatch):
    monkeypatch.delenv("USER", raising = False)
    monkeypatch.delenv("USERNAME", raising = False)
    user = hf_cache._safe_user()
    assert user
    if hasattr(os, "getuid"):
        assert user == str(os.getuid())


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
