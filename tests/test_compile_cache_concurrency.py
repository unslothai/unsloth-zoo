# SPDX-License-Identifier: AGPL-3.0-only

"""Mega-cache key stability and concurrent-save safety, CPU-only with fake
torch artifacts (no torch import, no compile)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest


_TESTS = Path(__file__).resolve().parent
_COMPILE_CACHE_PATH = _TESTS.parents[0] / "unsloth_zoo" / "compile_cache.py"
_WRITER = _TESTS / "_megacache_writer.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "zoo_compile_cache_concurrency_under_test", _COMPILE_CACHE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cache(monkeypatch, tmp_path):
    """A module instance with a fake torch whose load records every bundle read."""
    module = _load_module()
    loaded = []
    fake_torch = types.SimpleNamespace(
        compiler = types.SimpleNamespace(
            load_cache_artifacts = lambda bundle: loaded.append(bundle) or {},
            save_cache_artifacts = lambda: None,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("UNSLOTH_MEGA_CACHE", "1")
    monkeypatch.setenv("UNSLOTH_MEGA_CACHE_DIR", str(tmp_path / "mega"))
    monkeypatch.setattr(module.atexit, "register", lambda function: None)
    monkeypatch.setattr(
        module, "_is_trusted_directory",
        lambda path, allow_missing = True: os.path.isdir(path) or allow_missing,
    )
    module.loaded = loaded
    return module


def _key_after(module, *models):
    module._STATE["model_keys"] = []
    for model in models:
        module.megacache_load(model, {"fullgraph": True}, {"max_autotune": False})
    return module._STATE["loaded_key"]


def test_key_ignores_repeats_and_load_order(cache):
    single = _key_after(cache, "llama")
    assert _key_after(cache, "llama", "llama") == single
    pair = _key_after(cache, "llama", "qwen2")
    assert pair != single
    assert _key_after(cache, "qwen2", "llama") == pair
    assert _key_after(cache, "qwen2", "llama", "qwen2", "llama") == pair


def test_key_still_separates_compile_configurations(cache):
    cache._STATE["model_keys"] = []
    cache.megacache_load("llama", {"fullgraph": True}, {})
    fullgraph = cache._STATE["loaded_key"]
    cache._STATE["model_keys"] = []
    cache.megacache_load("llama", {"fullgraph": False}, {})
    assert cache._STATE["loaded_key"] != fullgraph
    # Same model type, two configurations in one process: both are kept.
    cache.megacache_load("llama", {"fullgraph": True}, {})
    assert len(cache._STATE["model_keys"]) == 2


def test_fingerprint_covers_python_and_rocm_identity(cache):
    fingerprint = cache._environment_fingerprint()
    assert fingerprint["python"] == "%d.%d" % sys.version_info[:2]
    assert {"hip", "gpu_arch", "xpu_name"} <= set(fingerprint)


def _seed(module, artifacts):
    key = module._bundle_key()
    directory = Path(os.environ["UNSLOTH_MEGA_CACHE_DIR"]) / key
    directory.mkdir(parents = True, exist_ok = True)
    data = json.dumps(sorted(artifacts)).encode()
    name = "megacache-seed.bin"
    (directory / name).write_bytes(data)
    (directory / "manifest.json").write_text(
        json.dumps({"bundle": name, "sha256": hashlib.sha256(data).hexdigest()})
    )
    if os.name == "posix":
        os.chmod(directory / name, 0o600)
        os.chmod(directory / "manifest.json", 0o600)
    return directory, data


def test_reloading_the_same_model_repopulates_torch_caches(cache):
    # No "already loaded" shortcut: a cleared FX cache or a changed
    # TORCHINDUCTOR_CACHE_DIR must be refilled, not reported as a hit.
    cache._STATE["model_keys"] = []
    cache._register_model_key(cache._model_key("llama", {}, {}))
    _, data = _seed(cache, [("inductor", "a")])
    cache._STATE["model_keys"] = []
    assert cache.megacache_load("llama") is True
    assert cache.megacache_load("llama") is True
    assert cache.loaded == [data, data]


class AutotuneCacheArtifact(tuple):
    """Stand-in for torch's frozen (key, content) autotune CacheArtifact."""

    def __new__(cls, key, content):
        return super().__new__(cls, (key, content))

    key = property(lambda self: self[0])
    content = property(lambda self: self[1])


_Artifact = AutotuneCacheArtifact


class _FakeRecorder:
    """torch.compiler._cache.CacheArtifactManager semantics that matter here:
    record_artifact dedupes by value into _new_cache_artifacts/_seen_artifacts,
    serialize moves _new into an appending serializer and returns all of it."""

    def __init__(self):
        from collections import defaultdict

        self._new_cache_artifacts = defaultdict(list)
        self._seen_artifacts = set()
        self.serialized = []
        self.calls = 0

    def record(self, key, content):
        artifact = _Artifact(key, content)
        if artifact not in self._seen_artifacts:
            self._new_cache_artifacts["autotune"].append(artifact)
            self._seen_artifacts.add(artifact)

    @staticmethod
    def encode(artifacts):
        return json.dumps([[a.key, a.content] for a in artifacts]).encode()

    @staticmethod
    def deserialize(data):
        return {"autotune": [_Artifact(key, content) for key, content in json.loads(data)]}

    def save_cache_artifacts(self):
        self.calls += 1
        for items in self._new_cache_artifacts.values():
            self.serialized.extend(items)
        self._new_cache_artifacts.clear()
        return (self.encode(self.serialized), {}) if self.serialized else None


@pytest.fixture
def recorder(cache, monkeypatch):
    fake = _FakeRecorder()
    cache_module = types.SimpleNamespace(CacheArtifactManager = fake)
    compiler = types.SimpleNamespace(
        _cache = cache_module,
        load_cache_artifacts = lambda bundle: {},
        save_cache_artifacts = fake.save_cache_artifacts,
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(compiler = compiler))
    monkeypatch.setitem(sys.modules, "torch.compiler", compiler)
    monkeypatch.setitem(sys.modules, "torch.compiler._cache", cache_module)
    cache._STATE["model_keys"] = []
    cache._register_model_key(cache._model_key("llama", {}, {}))
    cache._STATE["armed"] = True
    cache._STATE["loaded_key"] = cache._bundle_key()
    return fake


def _seed_bundle(cache, data):
    directory = Path(os.environ["UNSLOTH_MEGA_CACHE_DIR"]) / cache._STATE["loaded_key"]
    directory.mkdir(parents = True, exist_ok = True)
    name = "megacache-seed.bin"
    (directory / name).write_bytes(data)
    (directory / "manifest.json").write_text(
        json.dumps({"bundle": name, "sha256": hashlib.sha256(data).hexdigest()})
    )
    if os.name == "posix":
        os.chmod(directory / name, 0o600)
        os.chmod(directory / "manifest.json", 0o600)
    return directory


def _published(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    return sorted(map(tuple, json.loads((directory / manifest["bundle"]).read_bytes())))


def test_warm_run_that_adds_nothing_still_serializes_but_does_not_rewrite(cache, recorder):
    directory = _seed_bundle(cache, _FakeRecorder.encode([_Artifact("a", "1"), _Artifact("b", "old")]))
    # This run re-hit both, b re-timed under the same key, in another order.
    recorder.record("b", "new")
    recorder.record("a", "1")
    before = sorted(os.listdir(directory))

    assert cache.megacache_save() is False
    assert recorder.calls == 1  # deferred precompile entries still get recorded
    assert sorted(set(os.listdir(directory)) - {".lock"}) == before

    recorder.record("c", "2")  # a genuinely new artifact is saved with the union
    assert cache.megacache_save() is True
    assert recorder.calls == 2
    assert _published(directory) == [("a", "1"), ("b", "new"), ("c", "2")]


def test_cold_save_merges_the_disk_union(cache, recorder):
    directory = _seed_bundle(cache, _FakeRecorder.encode([_Artifact("a", "1"), _Artifact("x", "9")]))
    recorder.record("b", "2")
    assert cache.megacache_save() is True
    assert _published(directory) == [("a", "1"), ("b", "2"), ("x", "9")]


def test_save_is_skipped_while_another_live_writer_holds_the_lock(cache, recorder, monkeypatch):
    directory = Path(os.environ["UNSLOTH_MEGA_CACHE_DIR"]) / cache._STATE["loaded_key"]
    directory.mkdir(parents = True)
    assert cache._prepare_lock_file(str(directory / ".lock"))
    holder = cache._platform_lock(str(directory / ".lock"))
    holder.acquire()
    monkeypatch.setattr(cache, "_LOCK_TIMEOUT", 0.2)
    recorder.record("a", "1")
    try:
        assert cache.megacache_save() is False
    finally:
        holder.release()
    assert not (directory / "manifest.json").exists()
    assert cache.megacache_save() is True


def test_unsupported_locking_keeps_the_unlocked_save(cache, recorder, monkeypatch):
    class _NoFlock:
        def acquire(self):
            raise NotImplementedError("FileSystem does not appear to support flock")

    monkeypatch.setattr(cache, "_platform_lock", lambda path: _NoFlock())
    recorder.record("a", "1")
    assert cache.megacache_save() is True



class _RetriesUntilTimeout:
    """filelock 3.x on a filesystem that refuses flock: every non-ENOSYS error is retried until the timeout."""

    def __init__(self, timeout):
        self.timeout = timeout

    def acquire(self):
        import filelock
        time.sleep(self.timeout)
        raise filelock.Timeout("lock")


@pytest.mark.skipif(os.name != "posix", reason = "flock")
@pytest.mark.parametrize("code, saved", [("ENOLCK", True), ("EOPNOTSUPP", True), ("EIO", False)])
def test_a_filesystem_refusing_flock_never_waits_for_the_timeout(cache, recorder, monkeypatch, code, saved):
    import errno
    import fcntl

    real_flock = fcntl.flock

    def refusing_flock(fd, operation):
        if operation & fcntl.LOCK_EX:
            raise OSError(getattr(errno, code), os.strerror(getattr(errno, code)))
        return real_flock(fd, operation)

    monkeypatch.setattr(fcntl, "flock", refusing_flock)
    monkeypatch.setattr(cache, "_LOCK_TIMEOUT", 5)
    monkeypatch.setattr(cache, "_platform_lock", lambda path: _RetriesUntilTimeout(cache._LOCK_TIMEOUT))
    recorder.record("a", "1")
    started = time.monotonic()
    # Unsupported locking saves unlocked as before the lock existed; any other error skips; neither waits.
    assert cache.megacache_save() is saved
    assert time.monotonic() - started < 2

def test_platform_lock_never_downgrades_to_a_soft_lock(cache):
    import filelock

    lock = cache._platform_lock(os.path.join(os.environ["UNSLOTH_MEGA_CACHE_DIR"], ".lock"))
    expected = filelock.WindowsFileLock if os.name == "nt" else filelock.UnixFileLock
    assert isinstance(lock, expected)
    assert not isinstance(lock, filelock.SoftFileLock)


@pytest.mark.skipif(os.name != "posix", reason = "POSIX symlink / hard-link semantics")
@pytest.mark.parametrize("kind", ("symlink", "hardlink"))
def test_save_refuses_a_redirected_lock_file(cache, recorder, tmp_path, kind):
    directory = Path(os.environ["UNSLOTH_MEGA_CACHE_DIR"]) / cache._STATE["loaded_key"]
    directory.mkdir(parents = True)
    target = tmp_path / "elsewhere"
    target.write_bytes(b"keep")
    target.chmod(0o600)
    (directory / ".lock").symlink_to(target) if kind == "symlink" else os.link(target, directory / ".lock")
    recorder.record("a", "1")
    assert cache.megacache_save() is False
    assert target.read_bytes() == b"keep"
    assert not (directory / "manifest.json").exists()


@pytest.mark.skipif(os.name != "posix", reason = "POSIX modes")
def test_lock_file_is_created_owner_only(cache, recorder):
    recorder.record("a", "1")
    assert cache.megacache_save() is True
    directory = Path(os.environ["UNSLOTH_MEGA_CACHE_DIR"]) / cache._STATE["loaded_key"]
    assert (directory / ".lock").stat().st_mode & 0o777 == 0o600


def _spawn(root, artifacts, pause, sync):
    return subprocess.Popen(
        [sys.executable, "-I", str(_WRITER), str(_COMPILE_CACHE_PATH), str(root), artifacts, pause, str(sync)],
        stdout = subprocess.PIPE, stderr = subprocess.PIPE, text = True,
    )


def _wait_for(path, timeout = 60):
    deadline = time.time() + timeout
    while not path.exists():
        assert time.time() < deadline, f"timed out waiting for {path}"
        time.sleep(0.02)


@pytest.mark.parametrize("pause", ("after_read", "before_manifest"))
def test_concurrent_saves_publish_a_loadable_union(tmp_path, pause):
    """Writer 1 stalls inside its save (after reading the old bundle, or after
    writing its bundle but before publishing the manifest) while writer 2 saves.
    Without a per-key lock writer 2's artifacts are dropped from the union, or
    writer 2 collects writer 1's bundle and writer 1's manifest then dangles."""
    root = tmp_path / "mega"
    root.mkdir()
    sync1, sync2 = tmp_path / "sync1", tmp_path / "sync2"
    sync1.mkdir()
    sync2.mkdir()

    first = _spawn(root, "a1,shared", pause, sync1)
    _wait_for(sync1 / "paused")
    second = _spawn(root, "a2,shared", "", sync2)
    deadline = time.time() + 60
    while not any(name.startswith("entered.") for name in os.listdir(sync2)):
        assert time.time() < deadline
        time.sleep(0.02)
    # Give writer 2 time to finish if nothing holds it back.
    try:
        second.wait(timeout = 2)
    except subprocess.TimeoutExpired:
        pass
    (sync1 / "go").touch()
    for process in (first, second):
        _, stderr = process.communicate(timeout = 120)
        assert process.returncode == 0, stderr

    directory = root / "deadbeef"
    manifest = json.loads((directory / "manifest.json").read_text())
    bundle = directory / manifest["bundle"]
    assert bundle.exists(), "manifest names a collected bundle"
    data = bundle.read_bytes()
    assert hashlib.sha256(data).hexdigest() == manifest["sha256"]
    assert sorted(key for _, key in json.loads(data)) == ["a1", "a2", "shared"]
    assert [name for name in os.listdir(directory) if name.endswith(".bin")] == [bundle.name]


def _fake_artifact_manager(monkeypatch, disk, seen, kind = "autotune"):
    """torch.compiler._cache stand-in: frozen (key, content) artifacts like torch's."""
    import collections

    manager = types.SimpleNamespace(
        deserialize = lambda data: {kind: list(disk)},
        _seen_artifacts = set(seen),
        _new_cache_artifacts = collections.defaultdict(list),
    )
    module = types.ModuleType("torch.compiler._cache")
    module.CacheArtifactManager = manager
    monkeypatch.setitem(sys.modules, "torch.compiler._cache", module)
    return manager


@pytest.fixture
def autotune_artifact():
    import dataclasses

    @dataclasses.dataclass(frozen = True)
    class AutotuneCacheArtifact:
        key: str
        content: bytes

    return AutotuneCacheArtifact


def test_retimed_autotune_entries_are_not_new_identities(monkeypatch, autotune_artifact):
    # Triton re-records best_config entries with new timings under the same key.
    module = _load_module()
    manager = _fake_artifact_manager(
        monkeypatch,
        disk = [autotune_artifact("k1", b"old"), autotune_artifact("k2", b"x")],
        seen = [autotune_artifact("k1", b"new")],
    )
    disk_ids = module._merge_recorded_artifacts(b"bundle")
    assert disk_ids == {("AutotuneCacheArtifact", "k1"), ("AutotuneCacheArtifact", "k2")}
    # The disk-only key is folded in; the re-timed one keeps this run's copy.
    assert [a.key for a in manager._new_cache_artifacts["autotune"]] == ["k2"]
    assert module._recorded_identities() <= disk_ids


def test_merge_keeps_one_copy_per_key_and_this_runs_copy_wins(monkeypatch, autotune_artifact):
    module = _load_module()
    mine = autotune_artifact("k1", b"new")
    manager = _fake_artifact_manager(
        monkeypatch,
        disk = [autotune_artifact("k1", b"old"), autotune_artifact("k3", b"a"), autotune_artifact("k3", b"b")],
        seen = [mine, autotune_artifact("k2", b"x")],
    )
    assert module._merge_recorded_artifacts(b"bundle") == {
        ("AutotuneCacheArtifact", "k1"), ("AutotuneCacheArtifact", "k3"),
    }
    merged = manager._new_cache_artifacts["autotune"]
    assert [a.key for a in merged] == ["k3"]
    assert mine in manager._seen_artifacts
    assert autotune_artifact("k1", b"old") not in manager._seen_artifacts


def test_guarded_inductor_variants_under_one_key_are_all_kept(monkeypatch):
    # Inductor stores several guarded graphs per key (key/sha256(content)).
    import dataclasses

    @dataclasses.dataclass(frozen = True)
    class InductorCacheArtifact:
        key: str
        content: bytes

    module = _load_module()
    manager = _fake_artifact_manager(
        monkeypatch,
        disk = [InductorCacheArtifact("g", b"guard_a")],
        seen = [InductorCacheArtifact("g", b"guard_b")],
        kind = "inductor",
    )
    disk_ids = module._merge_recorded_artifacts(b"bundle")
    assert [a.content for a in manager._new_cache_artifacts["inductor"]] == [b"guard_a"]
    # The new variant is not on disk yet, so the save must not be skipped.
    assert not module._recorded_identities() <= disk_ids


def test_updated_pgo_profile_replaces_the_disk_copy_and_is_saved(monkeypatch):
    # PGO / precompile keep one file per key, but a changed profile is a real update.
    import dataclasses

    @dataclasses.dataclass(frozen = True)
    class PGOCacheArtifact:
        key: str
        content: bytes

    module = _load_module()
    manager = _fake_artifact_manager(
        monkeypatch,
        disk = [PGOCacheArtifact("frame", b"static")],
        seen = [PGOCacheArtifact("frame", b"dynamic")],
        kind = "pgo",
    )
    disk_ids = module._merge_recorded_artifacts(b"bundle")
    assert manager._new_cache_artifacts["pgo"] == []  # stale disk copy not re-added
    assert not module._recorded_identities() <= disk_ids
