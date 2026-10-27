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


def test_reloading_the_same_model_does_not_reread_the_bundle(cache):
    cache._STATE["model_keys"] = []
    cache._register_model_key(cache._model_key("llama", {}, {}))
    _, data = _seed(cache, [("inductor", "a")])
    cache._STATE["model_keys"] = []
    assert cache.megacache_load("llama") is True
    assert cache.megacache_load("llama") is True
    assert cache.loaded == [data]


def test_warm_run_that_adds_nothing_does_not_rewrite_the_bundle(cache, monkeypatch):
    from collections import defaultdict

    seen = {("inductor", "b"), ("inductor", "a")}
    manager = types.SimpleNamespace(
        _seen_artifacts = seen,
        _new_cache_artifacts = defaultdict(list),
        deserialize = lambda data: {"inductor": [tuple(item) for item in json.loads(data)]},
    )
    serialized = []
    fake_cache_module = types.SimpleNamespace(CacheArtifactManager = manager)
    fake_compiler = types.SimpleNamespace(
        _cache = fake_cache_module,
        load_cache_artifacts = lambda bundle: {},
        # This run's recording order differs from the bundle's.
        save_cache_artifacts = lambda: serialized.append(1) or (json.dumps([["inductor", "b"], ["inductor", "a"]]).encode(), {}),
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(compiler = fake_compiler))
    monkeypatch.setitem(sys.modules, "torch.compiler", fake_compiler)
    monkeypatch.setitem(sys.modules, "torch.compiler._cache", fake_cache_module)
    cache._STATE["model_keys"] = []
    cache._register_model_key(cache._model_key("llama", {}, {}))
    directory, data = _seed(cache, seen)
    cache._STATE["armed"] = True
    cache._STATE["loaded_key"] = cache._bundle_key()
    before = sorted(os.listdir(directory))

    assert cache.megacache_save() is False
    assert serialized == []
    assert sorted(set(os.listdir(directory)) - {".lock"}) == before

    seen.add(("inductor", "c"))  # a genuinely new artifact is still saved
    assert cache.megacache_save() is True
    assert serialized == [1]


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


def _fake_artifact_manager(monkeypatch, disk, seen):
    """torch.compiler._cache stand-in: frozen (key, content) artifacts like torch's."""
    import collections

    manager = types.SimpleNamespace(
        deserialize = lambda data: {"autotune": list(disk)},
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


def test_warm_run_with_retimed_autotune_entries_skips_the_save(monkeypatch, autotune_artifact):
    # Triton re-records best_config entries with new timings under the same key.
    module = _load_module()
    manager = _fake_artifact_manager(
        monkeypatch,
        disk = [autotune_artifact("k1", b"old"), autotune_artifact("k2", b"x")],
        seen = [autotune_artifact("k1", b"new")],
    )
    assert module._merge_recorded_artifacts(b"bundle") is True
    assert not manager._new_cache_artifacts


def test_merge_keeps_one_copy_per_key_and_this_runs_copy_wins(monkeypatch, autotune_artifact):
    module = _load_module()
    mine = autotune_artifact("k1", b"new")
    manager = _fake_artifact_manager(
        monkeypatch,
        disk = [autotune_artifact("k1", b"old"), autotune_artifact("k3", b"a"), autotune_artifact("k3", b"b")],
        seen = [mine, autotune_artifact("k2", b"x")],
    )
    assert module._merge_recorded_artifacts(b"bundle") is False
    merged = manager._new_cache_artifacts["autotune"]
    assert [a.key for a in merged] == ["k3"]
    assert mine in manager._seen_artifacts
    assert autotune_artifact("k1", b"old") not in manager._seen_artifacts
