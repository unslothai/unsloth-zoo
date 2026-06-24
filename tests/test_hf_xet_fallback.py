# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for unsloth_zoo.hf_xet_fallback: the no-progress watchdog, the Xet->HTTP
transport policy, the per-file and whole-snapshot entrypoints, the UNSLOTH_DISABLE_XET
knob, and the HF_HUB_DISABLE_XET precondition the fallback rests on.

CPU-only, no network, no real subprocess (the per-attempt download seam is
monkeypatched). The two modules under test are loaded directly via importlib so the
tests do not import the full ``unsloth_zoo`` package (which pulls in torch + GPU init).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import threading
import time
import types as _types
from pathlib import Path

import pytest

import huggingface_hub
from huggingface_hub import constants as hf_constants

_ZOO_DIR = Path(__file__).resolve().parents[1] / "unsloth_zoo"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _ZOO_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# A package placeholder so ``from unsloth_zoo.hf_cache_state import ...`` inside
# hf_xet_fallback resolves to the file we load below, not the installed package.
if "unsloth_zoo" not in sys.modules:
    _pkg = _types.ModuleType("unsloth_zoo")
    _pkg.__path__ = [str(_ZOO_DIR)]
    sys.modules["unsloth_zoo"] = _pkg

_load("unsloth_zoo.hf_cache_state", "hf_cache_state.py")
xf = _load("unsloth_zoo.hf_xet_fallback", "hf_xet_fallback.py")


# --------------------------------------------------------------------------- #
# Watchdog: fires only on a constant-size .incomplete, sparse-aware byte total.
# --------------------------------------------------------------------------- #
REPO = "ztest/xet-watchdog"


@pytest.fixture
def hf_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
    return tmp_path


def _blobs_dir(root: Path, repo_id: str = REPO) -> Path:
    d = root / f"models--{repo_id.replace('/', '--')}" / "blobs"
    d.mkdir(parents = True, exist_ok = True)
    return d


def _wait(predicate, timeout: float = 2.0, step: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return predicate()


def test_constant_incomplete_fires_stall(hf_cache):
    blobs = _blobs_dir(hf_cache)
    (blobs / "deadbeef.incomplete").write_bytes(b"\0" * 1024)  # never grows

    calls: list[str] = []
    stop = xf.start_watchdog(
        repo_ids = [REPO], on_stall = calls.append, interval = 0.05, stall_timeout = 0.3
    )
    try:
        assert _wait(
            lambda: len(calls) >= 1, timeout = 3.0
        ), "watchdog never fired on a constant-size .incomplete"
    finally:
        stop.set()
    assert "stalled" in calls[0].lower()


def test_growing_incomplete_never_stalls(hf_cache):
    blobs = _blobs_dir(hf_cache)
    part = blobs / "growing.incomplete"
    part.write_bytes(b"\0" * 1024)

    grow_stop = threading.Event()

    def _grow():
        size = 1024
        while not grow_stop.wait(0.05):
            size += 4096
            part.write_bytes(b"\0" * size)

    grower = threading.Thread(target = _grow, daemon = True)
    grower.start()

    calls: list[str] = []
    stop = xf.start_watchdog(
        repo_ids = [REPO], on_stall = calls.append, interval = 0.05, stall_timeout = 0.3
    )
    try:
        time.sleep(1.0)  # well past stall_timeout, but bytes keep growing
        assert calls == [], "watchdog fired despite continuous progress"
    finally:
        stop.set()
        grow_stop.set()


def test_no_incomplete_never_stalls(hf_cache):
    blobs = _blobs_dir(hf_cache)
    (blobs / "finalized_blob").write_bytes(b"\0" * 4096)  # no .incomplete

    calls: list[str] = []
    stop = xf.start_watchdog(
        repo_ids = [REPO], on_stall = calls.append, interval = 0.05, stall_timeout = 0.3
    )
    try:
        time.sleep(0.8)
        assert calls == [], "watchdog fired with no active .incomplete"
    finally:
        stop.set()


def test_stall_fires_at_most_once(hf_cache):
    blobs = _blobs_dir(hf_cache)
    (blobs / "frozen.incomplete").write_bytes(b"\0" * 2048)

    calls: list[str] = []
    stop = xf.start_watchdog(
        repo_ids = [REPO], on_stall = calls.append, interval = 0.05, stall_timeout = 0.2
    )
    try:
        assert _wait(lambda: len(calls) >= 1, timeout = 3.0)
        time.sleep(0.6)  # keep ticking; must not fire again
        assert len(calls) == 1, f"on_stall fired {len(calls)} times, expected exactly 1"
    finally:
        stop.set()


def test_get_state_empty_cache(hf_cache):
    assert xf.get_hf_download_state([REPO]) == (0, False)


def test_get_state_absent_cache_root(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "no-such-cache"))
    assert xf.get_hf_download_state([REPO]) == (0, False)


def test_get_state_skips_local_paths(hf_cache):
    # Filesystem paths are not HF repo IDs and must be ignored without error.
    assert xf.get_hf_download_state(["/abs/path", "./rel", "~user", "c:\\x"]) == (0, False)


def test_get_state_sparse_aware(hf_cache):
    blobs = _blobs_dir(hf_cache)
    sparse = blobs / "sparse.incomplete"
    with open(sparse, "wb") as f:
        f.truncate(64 * 1024 * 1024)  # large apparent size, few allocated blocks
    st = sparse.stat()
    if getattr(st, "st_blocks", 0) == 0:
        pytest.skip("filesystem does not report st_blocks; sparse accounting unavailable")
    total, has_incomplete = xf.get_hf_download_state([REPO])
    assert has_incomplete is True
    assert total < st.st_size, "sparse partial counted at apparent size, not allocated blocks"


# --------------------------------------------------------------------------- #
# Transport policy: cached short-circuit, cancel, error propagation, the single
# Xet->HTTP fallback, the injected prepare seam, and the UNSLOTH_DISABLE_XET knob.
# _run_download_attempt is faked, so no real spawn.
# --------------------------------------------------------------------------- #
DL_REPO, FILE = "ztest/xet-dl", "model-Q4_K_XL.gguf"


@pytest.fixture(autouse = True)
def _no_real_cache_hit(monkeypatch):
    """Default: the file cached probe misses and the snapshot fast path misses, so
    tests exercise the download seam unless they override these."""
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)

    def _snap_miss(*a, **k):
        raise FileNotFoundError("not fully cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _snap_miss)
    # Neutralize the generic cache purge by default; tests that care record it.
    monkeypatch.setattr(xf, "_default_prepare_for_http", lambda *a, **k: None)
    # No env knob unless a test sets it.
    monkeypatch.delenv("UNSLOTH_DISABLE_XET", raising = False)
    monkeypatch.delenv("UNSLOTH_STABLE_DOWNLOADS", raising = False)
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising = False)


class _FakeAttempt:
    """Records calls to the download seam and returns scripted results.

    Matches unsloth_zoo.hf_xet_fallback._run_download_attempt's signature.
    """

    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    def __call__(
        self,
        repo_id,
        *,
        kind,
        params,
        token,
        repo_type,
        disable_xet,
        cancel_event,
        stall_timeout,
        interval,
        grace_period,
        on_status,
    ):
        self.calls.append(
            _types.SimpleNamespace(
                repo_id = repo_id,
                kind = kind,
                target = params.get("filename", repo_id),
                disable_xet = disable_xet,
                repo_type = repo_type,
            )
        )
        return self._results[len(self.calls) - 1]


def _install(monkeypatch, results):
    fake = _FakeAttempt(results)
    monkeypatch.setattr(xf, "_run_download_attempt", fake)
    return fake


def test_cached_file_short_circuits(monkeypatch, tmp_path):
    cached = tmp_path / "cached.gguf"
    cached.write_bytes(b"\0" * 8)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: str(cached))
    fake = _install(monkeypatch, [])  # must not be called

    out = xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None)
    assert out == str(cached)
    assert fake.calls == [], "spawned a download for an already-cached file"


def test_cancel_before_start_raises_no_attempt(monkeypatch):
    fake = _install(monkeypatch, [])
    ev = threading.Event()
    ev.set()
    with pytest.raises(RuntimeError, match = "Cancelled"):
        xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None, cancel_event = ev)
    assert fake.calls == []


def test_nonstall_error_propagates_without_fallback(monkeypatch):
    fake = _install(monkeypatch, [("error", "RepositoryNotFoundError: 404 not found")])
    with pytest.raises(RuntimeError, match = "RepositoryNotFoundError"):
        xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None)
    assert len(fake.calls) == 1, "deterministic error must not trigger an HTTP fallback"
    assert fake.calls[0].disable_xet is False


def test_immediate_success_uses_xet_only(monkeypatch):
    prepared = []
    monkeypatch.setattr(xf, "_default_prepare_for_http", lambda *a: prepared.append(a))
    fake = _install(monkeypatch, [("ok", "/cache/model.gguf")])
    out = xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None)
    assert out == "/cache/model.gguf"
    assert len(fake.calls) == 1 and fake.calls[0].disable_xet is False
    assert prepared == [], "no cache prep should run when Xet succeeds first try"


def test_stall_then_http_fallback_succeeds(monkeypatch):
    prepared = []
    monkeypatch.setattr(xf, "_default_prepare_for_http", lambda repo_type, repo_id: prepared.append((repo_type, repo_id)))
    fake = _install(monkeypatch, [("stall", None), ("ok", "/cache/model.gguf")])

    out = xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None)
    assert out == "/cache/model.gguf"
    assert len(fake.calls) == 2
    assert fake.calls[0].disable_xet is False  # Xet first
    assert fake.calls[1].disable_xet is True  # HTTP fallback
    assert prepared == [("model", DL_REPO)], "must prep cache for HTTP before the retry"


def test_injected_prepare_for_http_used(monkeypatch):
    """Studio injects its marker-aware prepare; the generic default must not run."""
    monkeypatch.setattr(
        xf, "_default_prepare_for_http", lambda *a: pytest.fail("generic prepare ran")
    )
    injected = []
    _install(monkeypatch, [("stall", None), ("ok", "/cache/model.gguf")])
    out = xf.hf_hub_download_with_xet_fallback(
        DL_REPO, FILE, None, prepare_for_http_fn = lambda rt, rid: injected.append((rt, rid))
    )
    assert out == "/cache/model.gguf"
    assert injected == [("model", DL_REPO)]


def test_second_stall_raises_download_stall_error(monkeypatch):
    fake = _install(monkeypatch, [("stall", None), ("stall", None)])
    with pytest.raises(xf.DownloadStallError):
        xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None)
    assert len(fake.calls) == 2


def test_cancelled_midattempt_raises_no_fallback(monkeypatch):
    fake = _install(monkeypatch, [("cancelled", None)])
    with pytest.raises(RuntimeError, match = "Cancelled"):
        xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None)
    assert len(fake.calls) == 1


def test_per_file_independent_fallback(monkeypatch):
    """A stalled shard falls back; a sibling shard that succeeds does not."""
    fake = _install(monkeypatch, [("ok", "/a"), ("stall", None), ("ok", "/b")])
    assert xf.hf_hub_download_with_xet_fallback(DL_REPO, "shardA.gguf", None) == "/a"
    assert xf.hf_hub_download_with_xet_fallback(DL_REPO, "shardB.gguf", None) == "/b"
    assert [c.disable_xet for c in fake.calls] == [False, False, True]


def test_unsloth_disable_xet_forces_http_first(monkeypatch):
    """UNSLOTH_DISABLE_XET=1 skips the Xet attempt: first (and only) attempt is HTTP."""
    monkeypatch.setenv("UNSLOTH_DISABLE_XET", "1")
    fake = _install(monkeypatch, [("ok", "/http/model.gguf")])
    out = xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None)
    assert out == "/http/model.gguf"
    assert len(fake.calls) == 1 and fake.calls[0].disable_xet is True


def test_unsloth_disable_xet_stall_raises_no_retry(monkeypatch):
    """With the knob set, a stall on the (already HTTP) attempt does not retry."""
    monkeypatch.setenv("UNSLOTH_DISABLE_XET", "1")
    fake = _install(monkeypatch, [("stall", None)])
    with pytest.raises(xf.DownloadStallError):
        xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None)
    assert len(fake.calls) == 1


# --------------------------------------------------------------------------- #
# Snapshot variant: in-process fast path on a warm cache, else watched download.
# --------------------------------------------------------------------------- #
def test_snapshot_fast_path_no_child(monkeypatch):
    """A fully cached repo resolves in-process via local_files_only -- no attempt."""
    seen = {}

    def _snap(*a, **k):
        seen["local_files_only"] = k.get("local_files_only")
        return "/cache/snap-dir"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _snap)
    fake = _install(monkeypatch, [])  # must not be called
    out = xf.snapshot_download_with_xet_fallback(DL_REPO, token = None)
    assert out == "/cache/snap-dir"
    assert seen["local_files_only"] is True
    assert fake.calls == [], "spawned a download for an already-cached snapshot"


def test_snapshot_stall_then_http(monkeypatch):
    prepared = []
    monkeypatch.setattr(xf, "_default_prepare_for_http", lambda rt, rid: prepared.append((rt, rid)))
    fake = _install(monkeypatch, [("stall", None), ("ok", "/cache/snap-dir")])
    out = xf.snapshot_download_with_xet_fallback(DL_REPO, token = None)
    assert out == "/cache/snap-dir"
    assert [c.kind for c in fake.calls] == ["snapshot", "snapshot"]
    assert [c.disable_xet for c in fake.calls] == [False, True]
    assert prepared == [("model", DL_REPO)]


# --------------------------------------------------------------------------- #
# Precondition: HF_HUB_DISABLE_XET is read at import time, so assert its effect
# in a FRESH interpreter (huggingface/huggingface_hub#3266 once ignored it).
# --------------------------------------------------------------------------- #
def _safe_path() -> str:
    import os

    return os.environ.get("PATH", "")


def test_disable_xet_constant_set_in_fresh_interpreter():
    code = (
        "from huggingface_hub import constants as c; "
        "import sys; sys.exit(0 if c.HF_HUB_DISABLE_XET is True else 17)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env = {"HF_HUB_DISABLE_XET": "1", "PATH": _safe_path()},
        capture_output = True,
        text = True,
    )
    assert proc.returncode == 0, (
        f"HF_HUB_DISABLE_XET=1 did not set constants.HF_HUB_DISABLE_XET=True "
        f"(rc={proc.returncode}): {proc.stderr}"
    )


def test_default_leaves_xet_enabled():
    code = (
        "from huggingface_hub import constants as c; "
        "import sys; sys.exit(0 if c.HF_HUB_DISABLE_XET is False else 17)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env = {"PATH": _safe_path()},  # no HF_HUB_DISABLE_XET
        capture_output = True,
        text = True,
    )
    assert proc.returncode == 0, (
        f"without the env var, constants.HF_HUB_DISABLE_XET was not False "
        f"(rc={proc.returncode}): {proc.stderr}"
    )
