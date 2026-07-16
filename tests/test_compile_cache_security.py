# SPDX-License-Identifier: AGPL-3.0-only

"""Security boundary tests for persistent torch.compile artifacts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest


_COMPILE_CACHE_PATH = (
    Path(__file__).resolve().parents[1] / "unsloth_zoo" / "compile_cache.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "zoo_compile_cache_under_test", _COMPILE_CACHE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed_bundle(root):
    key_dir = root / "deadbeef"
    key_dir.mkdir(parents = True)
    data = b"attacker-controlled artifact bundle"
    bundle_name = "megacache-seeded.bin"
    (key_dir / bundle_name).write_bytes(data)
    (key_dir / "manifest.json").write_text(
        json.dumps(
            {
                "bundle": bundle_name,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    )
    return key_dir, data


def _configure_load(monkeypatch, module, root, loaded):
    fake_torch = types.SimpleNamespace(
        compiler = types.SimpleNamespace(
            load_cache_artifacts = lambda bundle: loaded.append(bundle) or {},
            save_cache_artifacts = lambda: None,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("UNSLOTH_MEGA_CACHE", "1")
    monkeypatch.setenv("UNSLOTH_MEGA_CACHE_DIR", str(root))
    monkeypatch.setattr(module, "_bundle_key", lambda: "deadbeef")
    monkeypatch.setattr(module.atexit, "register", lambda function: None)


def test_megacache_is_opt_in(monkeypatch):
    module = _load_module()

    monkeypatch.delenv("UNSLOTH_MEGA_CACHE", raising = False)
    assert module.megacache_is_enabled() is False

    for value in ("1", "on", "true", "yes"):
        monkeypatch.setenv("UNSLOTH_MEGA_CACHE", value)
        assert module.megacache_is_enabled() is True

    for value in ("0", "off", "false", "no", "auto", ""):
        monkeypatch.setenv("UNSLOTH_MEGA_CACHE", value)
        assert module.megacache_is_enabled() is False


@pytest.mark.skipif(os.name != "posix", reason = "POSIX ownership and permissions")
@pytest.mark.parametrize("unsafe_component", ("root", "key"))
def test_megacache_does_not_load_from_group_writable_directory(
    monkeypatch, tmp_path, unsafe_component
):
    module = _load_module()
    root = tmp_path / "mega_cache"
    key_dir, _ = _seed_bundle(root)
    root.chmod(0o700)
    key_dir.chmod(0o700)
    (root if unsafe_component == "root" else key_dir).chmod(0o770)

    loaded = []
    _configure_load(monkeypatch, module, root, loaded)

    assert module.megacache_load("victim-model") is False
    assert loaded == []


@pytest.mark.skipif(os.name != "posix", reason = "POSIX ownership and permissions")
def test_megacache_loads_from_trusted_directory_when_enabled(monkeypatch, tmp_path):
    module = _load_module()
    root = tmp_path / "mega_cache"
    key_dir, data = _seed_bundle(root)
    root.chmod(0o700)
    key_dir.chmod(0o700)

    loaded = []
    _configure_load(monkeypatch, module, root, loaded)

    assert module.megacache_load("trusted-model") is True
    assert loaded == [data]


@pytest.mark.skipif(os.name != "posix", reason = "POSIX ownership and permissions")
def test_megacache_creates_owner_only_cache_directories(monkeypatch, tmp_path):
    module = _load_module()
    root = tmp_path / "mega_cache"
    key_dir = root / "deadbeef"
    data = b"locally generated artifact bundle"

    fake_torch = types.SimpleNamespace(
        compiler = types.SimpleNamespace(
            load_cache_artifacts = lambda bundle: {},
            save_cache_artifacts = lambda: (data, {}),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("UNSLOTH_MEGA_CACHE", "1")
    monkeypatch.setenv("UNSLOTH_MEGA_CACHE_DIR", str(root))
    module._STATE["armed"] = True
    module._STATE["loaded_key"] = "deadbeef"

    assert module.megacache_save() is True
    assert (root.stat().st_mode & 0o777) == 0o700
    assert (key_dir.stat().st_mode & 0o777) == 0o700
