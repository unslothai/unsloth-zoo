# SPDX-License-Identifier: AGPL-3.0-only

"""Security boundary tests for persistent torch.compile artifacts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
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


@pytest.fixture(autouse = True)
def _trust_ambient_tmp_ancestors(tmp_path_factory, monkeypatch):
    """Make dirs at/above pytest's basetemp look owner-only + sticky so the
    full-chain walk in ``_is_trusted_directory`` depends only on the dirs each
    test builds under ``tmp_path``. Without this the suite is non-hermetic: a
    group-writable, non-sticky temp root (common on shared CI runners) rejects
    every path and turns the accept-side tests red for the wrong reason."""
    if os.name != "posix":
        yield
        return
    ambient, current = set(), os.path.abspath(str(tmp_path_factory.getbasetemp()))
    while True:
        parent = os.path.dirname(current)
        if parent == current:
            break
        ambient.add(parent)
        current = parent
    real_lstat, euid = os.lstat, os.geteuid()

    def fake_lstat(path, *args, **kwargs):
        real = real_lstat(path, *args, **kwargs)
        if os.path.abspath(os.fspath(path)) not in ambient:
            return real
        return os.stat_result((
            (real.st_mode & ~0o777) | stat.S_ISVTX | 0o700,
            real.st_ino, real.st_dev, real.st_nlink, euid, real.st_gid,
            real.st_size, real.st_atime, real.st_mtime, real.st_ctime,
        ))

    monkeypatch.setattr(os, "lstat", fake_lstat)
    yield


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
    # A trusted bundle is owner-only; the tool writes 0600 and the reader
    # rejects group/other-writable files (umask 0002 would otherwise seed 0664).
    if os.name == "posix":
        (key_dir / bundle_name).chmod(0o600)
        (key_dir / "manifest.json").chmod(0o600)
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


@pytest.mark.skipif(os.name != "posix", reason = "POSIX symlink semantics")
@pytest.mark.parametrize("unsafe_component", ("root", "key"))
def test_megacache_does_not_load_through_directory_symlinks(
    monkeypatch, tmp_path, unsafe_component
):
    module = _load_module()
    real_root = tmp_path / "real-cache"
    key_dir, _ = _seed_bundle(real_root)
    real_root.chmod(0o700)
    key_dir.chmod(0o700)

    if unsafe_component == "root":
        configured_root = tmp_path / "cache-link"
        configured_root.symlink_to(real_root, target_is_directory = True)
    else:
        configured_root = tmp_path / "cache"
        configured_root.mkdir(mode = 0o700)
        (configured_root / "deadbeef").symlink_to(key_dir, target_is_directory = True)

    loaded = []
    _configure_load(monkeypatch, module, configured_root, loaded)

    assert module.megacache_load("victim-model") is False
    assert loaded == []


@pytest.mark.skipif(os.name != "posix", reason = "POSIX ownership")
def test_megacache_rejects_directory_owned_by_another_user(monkeypatch, tmp_path):
    module = _load_module()
    directory = tmp_path / "cache"
    directory.mkdir(mode = 0o700)
    real_lstat = module.os.lstat
    real_stat = real_lstat(directory)

    monkeypatch.setattr(
        module.os,
        "lstat",
        lambda path: types.SimpleNamespace(
            st_mode = real_stat.st_mode,
            st_uid = module.os.geteuid() + 1,
        )
        if Path(path) == directory
        else real_lstat(path),
    )

    assert module._is_trusted_directory(directory) is False


@pytest.mark.skipif(os.name != "posix", reason = "POSIX permissions")
@pytest.mark.parametrize("mode", (0o700, 0o750, 0o755))
def test_megacache_accepts_owned_nonwritable_directory_modes(monkeypatch, tmp_path, mode):
    module = _load_module()
    directory = tmp_path / "cache"
    directory.mkdir(mode = mode)
    directory.chmod(mode)

    assert module._is_trusted_directory(directory) is True


@pytest.mark.skipif(os.name != "posix", reason = "POSIX permissions")
def test_megacache_rejects_nonsticky_writable_ancestor(monkeypatch, tmp_path):
    module = _load_module()
    shared = tmp_path / "shared"
    directory = shared / "private-cache"
    directory.mkdir(parents = True, mode = 0o700)
    directory.chmod(0o700)
    shared.chmod(0o770)

    assert module._is_trusted_directory(directory) is False


@pytest.mark.skipif(os.name != "posix", reason = "POSIX sticky-directory semantics")
def test_megacache_accepts_private_cache_below_sticky_temp_parent(monkeypatch, tmp_path):
    module = _load_module()
    sticky = tmp_path / "sticky"
    directory = sticky / "private-cache"
    directory.mkdir(parents = True, mode = 0o700)
    directory.chmod(0o700)
    sticky.chmod(0o1777)

    assert module._is_trusted_directory(directory) is True


def test_non_posix_load_still_requires_explicit_opt_in(monkeypatch, tmp_path):
    module = _load_module()
    root = tmp_path / "mega_cache"
    _seed_bundle(root)
    loaded = []
    _configure_load(monkeypatch, module, root, loaded)
    monkeypatch.setattr(module.os, "name", "nt")

    monkeypatch.delenv("UNSLOTH_MEGA_CACHE")
    assert module.megacache_load("disabled-model") is False
    assert loaded == []

    monkeypatch.setenv("UNSLOTH_MEGA_CACHE", "1")
    assert module.megacache_load("trusted-model") is True
    assert loaded != []


def test_megacache_never_reads_a_missing_bundle_path(monkeypatch, tmp_path):
    module = _load_module()
    root = tmp_path / "missing-cache"
    loaded = []
    _configure_load(monkeypatch, module, root, loaded)

    def _unexpected_read(*_args, **_kwargs):
        pytest.fail("missing cache path reached the artifact reader")

    monkeypatch.setattr(module, "_read_disk_bundle", _unexpected_read)

    assert module.megacache_load("cache-miss") is False
    assert module._STATE["armed"] is True
    assert loaded == []


def test_megacache_save_secures_directories_before_existing_read(monkeypatch, tmp_path):
    module = _load_module()
    root = tmp_path / "new-cache"
    key_dir = root / "deadbeef"
    data = b"locally generated artifact bundle"
    read_observations = []

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

    def _observe_read(directory, manifest_path):
        read_observations.append((Path(directory).is_dir(), Path(manifest_path).parent.is_dir()))
        if os.name == "posix":
            assert Path(directory).stat().st_mode & 0o777 == 0o700
            assert Path(manifest_path).parent.stat().st_mode & 0o777 == 0o700
        return None, None

    monkeypatch.setattr(module, "_read_disk_bundle", _observe_read)

    assert module.megacache_save() is True
    assert read_observations == [(True, True)]
    assert key_dir.is_dir()


@pytest.mark.skipif(os.name != "posix", reason = "POSIX sticky-directory semantics")
def test_megacache_rejects_cache_below_foreign_owned_sticky_parent(monkeypatch, tmp_path):
    module = _load_module()
    parent = tmp_path / "shared"
    directory = parent / "cache"
    directory.mkdir(parents = True, mode = 0o700)
    directory.chmod(0o700)
    parent.chmod(0o1777)

    # Sticky bit is not enough: the parent's owner can rename our leaf, so a
    # foreign-owned sticky parent must be rejected.
    real_lstat = module.os.lstat

    def foreign_owner(path, *args, **kwargs):
        info = real_lstat(path, *args, **kwargs)
        if os.path.abspath(os.fspath(path)) == str(parent):
            return os.stat_result((
                info.st_mode, info.st_ino, info.st_dev, info.st_nlink,
                os.geteuid() + 1, info.st_gid, info.st_size,
                info.st_atime, info.st_mtime, info.st_ctime,
            ))
        return info

    monkeypatch.setattr(module.os, "lstat", foreign_owner)
    assert module._is_trusted_directory(directory) is False


@pytest.mark.skipif(os.name != "posix", reason = "POSIX permissions")
def test_megacache_does_not_load_a_group_writable_bundle_file(monkeypatch, tmp_path):
    module = _load_module()
    root = tmp_path / "mega_cache"
    key_dir, _ = _seed_bundle(root)
    root.chmod(0o700)
    key_dir.chmod(0o700)
    # Directories are trusted, but a group-writable bundle file can be rewritten
    # in place by a same-group attacker along with its checksum.
    (key_dir / "megacache-seeded.bin").chmod(0o664)

    loaded = []
    _configure_load(monkeypatch, module, root, loaded)

    assert module.megacache_load("victim-model") is False
    assert loaded == []


@pytest.mark.skipif(os.name != "posix", reason = "POSIX permissions")
def test_megacache_ignores_bundle_name_path_traversal(monkeypatch, tmp_path):
    module = _load_module()
    root = tmp_path / "mega_cache"
    key_dir = root / "deadbeef"
    key_dir.mkdir(parents = True)
    root.chmod(0o700)
    key_dir.chmod(0o700)
    payload = b"pwned"
    outside = root / "evil"
    outside.write_bytes(payload)
    outside.chmod(0o600)
    (key_dir / "manifest.json").write_text(
        json.dumps({"bundle": "../evil", "sha256": hashlib.sha256(payload).hexdigest()})
    )
    (key_dir / "manifest.json").chmod(0o600)

    loaded = []
    _configure_load(monkeypatch, module, root, loaded)

    # basename() keeps the read inside the key dir, so "../evil" misses.
    assert module.megacache_load("victim-model") is False
    assert loaded == []
