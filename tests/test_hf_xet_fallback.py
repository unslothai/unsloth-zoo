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
import os
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

hcs = _load("unsloth_zoo.hf_cache_state", "hf_cache_state.py")
xf = _load("unsloth_zoo.hf_xet_fallback", "hf_xet_fallback.py")

# Real prep impl, captured before the autouse fixture stubs the module attribute.
_REAL_DEFAULT_PREPARE = xf._default_prepare_for_http


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
    assert xf.get_hf_download_state(
        ["/abs/path", "./rel", "~user", "c:\\x", "c:/x"]
    ) == (0, False)


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


def test_custom_cache_dir_is_watched_and_cleaned(tmp_path, monkeypatch):
    """A stall under a caller-supplied snapshot ``cache_dir`` (not HF_HUB_CACHE)
    must still be seen by the state probe, the watchdog, and the HTTP-prep purge."""
    default_cache = tmp_path / "default"
    custom_cache = tmp_path / "custom"
    default_cache.mkdir()
    custom_cache.mkdir()
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(default_cache))

    blobs = custom_cache / f"models--{REPO.replace('/', '--')}" / "blobs"
    blobs.mkdir(parents = True)
    partial = blobs / "stalled.incomplete"
    partial.write_bytes(b"partial-bytes")

    # Default cache sees nothing; the custom cache sees the active partial.
    assert xf.get_hf_download_state([REPO]) == (0, False)
    total, has_incomplete = xf.get_hf_download_state([REPO], cache_dir = str(custom_cache))
    assert has_incomplete is True and total > 0

    # The watchdog fires for the custom cache, not the (empty) default one.
    calls: list[str] = []
    stop = xf.start_watchdog(
        repo_ids = [REPO], on_stall = calls.append, cache_dir = str(custom_cache),
        interval = 0.05, stall_timeout = 0.3,
    )
    try:
        assert _wait(lambda: len(calls) >= 1, timeout = 3.0), "watchdog ignored the custom cache_dir"
    finally:
        stop.set()

    # The HTTP-prep purge removes the unsafe partial from the custom cache
    # (call the real impl; the autouse fixture stubs the module attribute).
    _REAL_DEFAULT_PREPARE("model", REPO, cache_dir = str(custom_cache))
    assert not partial.exists()


def test_prepare_for_http_clears_broken_snapshot_symlink(tmp_path):
    """A broken snapshot symlink is counted as active-incomplete state by the
    detector, so HTTP prep must clear it too or the retry re-trips the watchdog."""
    repo = "ztest/broken-symlink"
    repo_dir = tmp_path / f"models--{repo.replace('/', '--')}"
    snap = repo_dir / "snapshots" / "abc123"
    snap.mkdir(parents = True)
    link = snap / "model.safetensors"
    link.symlink_to(repo_dir / "blobs" / "missing-blob")  # dangling
    assert link.is_symlink() and not link.exists()

    # Detector treats the dangling link as active incomplete state.
    assert xf.get_hf_download_state([repo], cache_dir = str(tmp_path)) == (0, True)

    _REAL_DEFAULT_PREPARE("model", repo, cache_dir = str(tmp_path))

    assert not link.is_symlink(), "broken snapshot symlink not cleared by HTTP prep"
    assert xf.get_hf_download_state([repo], cache_dir = str(tmp_path)) == (0, False)


def test_snapshot_dir_has_broken_symlinks_unit(tmp_path):
    """The new per-snapshot primitive flags a dangling link and is clean otherwise."""
    snap = tmp_path / "snapshots" / "sha"
    snap.mkdir(parents = True)
    good = snap / "config.json"
    good.write_text("{}")
    assert hcs.snapshot_dir_has_broken_symlinks(snap) is False
    (snap / "model.safetensors").symlink_to(tmp_path / "blobs" / "missing")
    assert hcs.snapshot_dir_has_broken_symlinks(snap) is True


def test_broken_older_snapshot_detected_when_newer_is_clean(tmp_path):
    """Detector must inspect every snapshot, not just the newest by mtime: an older
    revision with a dangling symlink must read as incomplete even when a more
    recently landed snapshot is fully present."""
    repo = "ztest/two-snaps"
    repo_dir = tmp_path / f"models--{repo.replace('/', '--')}"
    old = repo_dir / "snapshots" / "oldsha"
    new = repo_dir / "snapshots" / "newsha"
    old.mkdir(parents = True)
    new.mkdir(parents = True)
    # Broken (older) revision; clean (newer) revision.
    (old / "model.safetensors").symlink_to(repo_dir / "blobs" / "missing")
    (new / "config.json").write_text("{}")
    # Make the clean snapshot the newest by mtime so a latest-only check would
    # report the repo healthy.
    os.utime(new, (time.time() + 10, time.time() + 10))
    assert xf.get_hf_download_state([repo], cache_dir = str(tmp_path)) == (0, True)


def test_snapshot_fast_path_rejects_broken_requested_revision(tmp_path, monkeypatch):
    """snapshot_download(local_files_only=True) can hand back an older requested
    revision whose snapshot is broken while the repo-wide scan is clean. The fast
    path must validate the EXACT returned dir and complete in the killable child
    rather than short-circuiting to a snapshot with missing files."""
    snap = tmp_path / "snapshots" / "oldsha"
    snap.mkdir(parents = True)
    (snap / "model.safetensors").symlink_to(tmp_path / "blobs" / "missing")  # dangling
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(snap))
    # Repo-wide incomplete-blob scan sees nothing (empty cache root), so only the
    # per-revision symlink check can catch the broken returned dir.
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "empty-cache"))
    fake = _install(monkeypatch, [("ok", "/cache/snap-fresh")])
    out = xf.snapshot_download_with_xet_fallback(DL_REPO, token = None)
    assert out == "/cache/snap-fresh", "fast path returned a broken requested revision"
    assert len(fake.calls) == 1


def test_prepare_for_http_clears_broken_symlink_in_older_snapshot(tmp_path):
    """HTTP prep must clear dangling links across all snapshots, not just the
    newest, so the incomplete detector reads clean afterwards."""
    repo = "ztest/old-broken"
    repo_dir = tmp_path / f"models--{repo.replace('/', '--')}"
    old = repo_dir / "snapshots" / "oldsha"
    new = repo_dir / "snapshots" / "newsha"
    old.mkdir(parents = True)
    new.mkdir(parents = True)
    link = old / "model.safetensors"
    link.symlink_to(repo_dir / "blobs" / "missing")  # dangling, older snapshot
    (new / "config.json").write_text("{}")
    os.utime(new, (time.time() + 10, time.time() + 10))  # newer snapshot is clean

    assert xf.get_hf_download_state([repo], cache_dir = str(tmp_path)) == (0, True)
    _REAL_DEFAULT_PREPARE("model", repo, cache_dir = str(tmp_path))
    assert not link.is_symlink(), "broken symlink in older snapshot not cleared"
    assert xf.get_hf_download_state([repo], cache_dir = str(tmp_path)) == (0, False)


def test_prepare_for_http_preserves_case_colliding_repo(tmp_path):
    """On a case-sensitive filesystem, preparing HTTP for ``Org/Repo`` must purge
    only its exact-case cache dir, never a case-colliding ``org/repo``."""
    upper = tmp_path / "models--Org--Repo" / "blobs"
    lower = tmp_path / "models--org--repo" / "blobs"
    upper.mkdir(parents = True)
    lower.mkdir(parents = True)
    if upper.parent.resolve() == lower.parent.resolve():
        pytest.skip("case-insensitive filesystem; cannot collide cache dirs")
    upper_partial = upper / "a.incomplete"
    lower_partial = lower / "b.incomplete"
    upper_partial.write_bytes(b"x")
    lower_partial.write_bytes(b"y")

    _REAL_DEFAULT_PREPARE("model", "Org/Repo", cache_dir = str(tmp_path))

    assert not upper_partial.exists(), "exact-case partial should be purged"
    assert lower_partial.exists(), "case-colliding repo's partial must be preserved"


def test_repo_type_none_resolves_model_cache(hf_cache):
    """A caller forwarding repo_type=None (HF's default model) must still see the
    real models--<id> partial, not look up a bogus Nones--<id> dir."""
    blobs = _blobs_dir(hf_cache)
    (blobs / "x.incomplete").write_bytes(b"abc")

    model_state = xf.get_hf_download_state([REPO], repo_type = "model")
    none_state = xf.get_hf_download_state([REPO], repo_type = None)
    assert model_state == none_state
    assert none_state[1] is True and none_state[0] > 0


def test_state_ignores_case_colliding_repo_partial(tmp_path, monkeypatch):
    """The read/watchdog path attributes a partial only to an exact-case repo dir,
    so a stale partial in a case-colliding repo cannot trip the watchdog."""
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
    exact = tmp_path / "models--Org--Repo" / "blobs"
    other = tmp_path / "models--org--repo" / "blobs"
    exact.mkdir(parents = True)
    other.mkdir(parents = True)
    if exact.parent.resolve() == other.parent.resolve():
        pytest.skip("case-insensitive filesystem; cannot collide cache dirs")
    (other / "stale.incomplete").write_bytes(b"x")  # only the lowercase repo

    # Org/Repo has no partial of its own; the lowercase repo's must not count.
    assert xf.get_hf_download_state(["Org/Repo"]) == (0, False)


def test_single_folded_match_rejected_on_case_sensitive_fs(tmp_path, monkeypatch):
    """A single folded-but-not-exact cache dir must not be attributed to a
    differently-cased repo on a case-sensitive filesystem -- it is a different
    repo, and charging its partial here could misread the watchdog or let HTTP-prep
    delete it. Only an exact-case dir (or a folded dir the FS resolves to the same
    entry on a case-insensitive FS) counts."""
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
    lower = tmp_path / "models--org--repo" / "blobs"
    lower.mkdir(parents = True)
    if (tmp_path / "models--Org--Repo").exists():
        pytest.skip("case-insensitive filesystem; the folded dir is the same entry")
    (lower / "stale.incomplete").write_bytes(b"x")  # only the lowercase repo exists
    # Request the exact-case repo, which has no dir of its own: the lowercase repo's
    # partial must not be attributed to it.
    assert xf.get_hf_download_state(["Org/Repo"]) == (0, False)


def test_cache_dir_is_expanded(tmp_path, monkeypatch):
    """A custom cache_dir with ~ must be expanded (as HF does on write), else the
    state probe scans the literal '~/...' path and misses the partial."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows home var
    blobs = tmp_path / "hfcache" / f"models--{REPO.replace('/', '--')}" / "blobs"
    blobs.mkdir(parents = True)
    (blobs / "p.incomplete").write_bytes(b"abc")

    total, has_incomplete = xf.get_hf_download_state([REPO], cache_dir = "~/hfcache")
    assert has_incomplete is True and total > 0


def test_status_callback_failure_does_not_kill_watchdog(hf_cache):
    """A raising on_heartbeat (e.g. a disconnected client) must not stop the
    daemon watchdog from detecting a real stall and firing on_stall."""
    blobs = _blobs_dir(hf_cache)
    (blobs / "x.incomplete").write_bytes(b"\0" * 1024)  # constant size -> stalls

    def boom(_message):
        raise RuntimeError("client disconnected")

    calls: list[str] = []
    stop = xf.start_watchdog(
        repo_ids = [REPO], on_stall = calls.append, on_heartbeat = boom,
        interval = 0.05, stall_timeout = 0.3,
    )
    try:
        assert _wait(
            lambda: len(calls) >= 1, timeout = 3.0
        ), "a raising on_heartbeat killed stall detection"
    finally:
        stop.set()


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
                cache_dir = params.get("cache_dir"),
                subfolder = params.get("subfolder"),
                force_download = params.get("force_download"),
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
    monkeypatch.setattr(xf, "_default_prepare_for_http", lambda repo_type, repo_id, cache_dir = None: prepared.append((repo_type, repo_id)))
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


def test_file_path_accepts_cache_dir(monkeypatch):
    """The single-file wrapper accepts cache_dir (no TypeError) and threads it through."""
    fake = _install(monkeypatch, [("ok", "/cache/model.gguf")])
    out = xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None, cache_dir = "/custom/cache")
    assert out == "/cache/model.gguf"
    assert fake.calls[0].cache_dir == "/custom/cache"


# --------------------------------------------------------------------------- #
# Spawn env-timing: the parent sets HF_HUB_DISABLE_XET before the child starts,
# so the child inherits it before re-importing huggingface_hub (whose constants
# cache the value at import). Uses a fake spawn context -- no real subprocess.
# --------------------------------------------------------------------------- #
class _FakeProc:
    def __init__(self, recorder):
        self._rec = recorder
        self.pid = 4242
        self.exitcode = 0

    def start(self):
        self._rec["disable_xet"] = os.environ.get("HF_HUB_DISABLE_XET")
        self._rec["hf_transfer"] = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER")
        self._rec["skip_gpu_init"] = os.environ.get("UNSLOTH_ZOO_DISABLE_GPU_INIT")
        self._rec["main_file"] = getattr(sys.modules.get("__main__"), "__file__", None)

    def is_alive(self):
        return False

    def join(self, timeout = None):
        pass


class _FakeQueue:
    def __init__(self, result):
        self._result = result

    def get(self, timeout = None):
        return self._result

    def get_nowait(self):
        return self._result

    def put(self, item):
        pass


class _FakeCtx:
    def __init__(self, recorder, result):
        self._rec = recorder
        self._result = result

    def Process(self, *, target = None, kwargs = None, daemon = None):
        return _FakeProc(self._rec)

    def Queue(self):
        return _FakeQueue(self._result)


def test_http_retry_sets_disable_xet_before_spawn(monkeypatch):
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising = False)
    monkeypatch.delenv("HF_HUB_ENABLE_HF_TRANSFER", raising = False)
    rec: dict = {}
    monkeypatch.setattr(xf, "_CTX", _FakeCtx(rec, {"ok": True, "path": "/cache/x"}))

    kind_result, payload = xf._run_download_attempt(
        DL_REPO, kind = "snapshot", params = {"repo_id": DL_REPO}, token = None,
        repo_type = "model", disable_xet = True, cancel_event = None,
        stall_timeout = 0.2, interval = 0.05, grace_period = 0.2, on_status = None,
    )
    assert (kind_result, payload) == ("ok", "/cache/x")
    # Child inherited HTTP transport env at spawn time.
    assert rec["disable_xet"] == "1"
    assert rec["hf_transfer"] == "0"
    # Parent env is restored afterwards (was unset).
    assert "HF_HUB_DISABLE_XET" not in os.environ


def test_xet_attempt_does_not_force_disable_before_spawn(monkeypatch):
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising = False)
    rec: dict = {}
    monkeypatch.setattr(xf, "_CTX", _FakeCtx(rec, {"ok": True, "path": "/cache/x"}))
    xf._run_download_attempt(
        DL_REPO, kind = "snapshot", params = {"repo_id": DL_REPO}, token = None,
        repo_type = "model", disable_xet = False, cancel_event = None,
        stall_timeout = 0.2, interval = 0.05, grace_period = 0.2, on_status = None,
    )
    # On the Xet-first attempt we must NOT force-disable Xet for the child.
    assert rec["disable_xet"] is None


def test_child_skips_gpu_init_env_set_before_spawn_and_restored(monkeypatch):
    """The download child inherits UNSLOTH_ZOO_DISABLE_GPU_INIT=1 at spawn (so its
    fresh unsloth_zoo import skips heavy torch/transformers init), and the parent's
    env is restored afterwards."""
    monkeypatch.delenv("UNSLOTH_ZOO_DISABLE_GPU_INIT", raising = False)
    rec: dict = {}
    monkeypatch.setattr(xf, "_CTX", _FakeCtx(rec, {"ok": True, "path": "/cache/x"}))

    xf._run_download_attempt(
        DL_REPO, kind = "snapshot", params = {"repo_id": DL_REPO}, token = None,
        repo_type = "model", disable_xet = False, cancel_event = None,
        stall_timeout = 0.2, interval = 0.05, grace_period = 0.2, on_status = None,
    )
    assert rec["skip_gpu_init"] == "1"  # set in the parent before proc.start()
    assert "UNSLOTH_ZOO_DISABLE_GPU_INIT" not in os.environ  # restored after


def test_spawn_repoints_main_file_and_restores(monkeypatch):
    """For an unguarded top-level caller script, the spawn child must import this
    side-effect-free module as __mp_main__ rather than re-execute the caller, so the
    parent repoints __main__.__file__ here at spawn and restores it afterwards."""
    main_mod = sys.modules["__main__"]
    monkeypatch.setattr(main_mod, "__file__", "/fake/user_script.py", raising = False)
    rec: dict = {}
    monkeypatch.setattr(xf, "_CTX", _FakeCtx(rec, {"ok": True, "path": "/cache/x"}))

    xf._run_download_attempt(
        DL_REPO, kind = "snapshot", params = {"repo_id": DL_REPO}, token = None,
        repo_type = "model", disable_xet = False, cancel_event = None,
        stall_timeout = 0.2, interval = 0.05, grace_period = 0.2, on_status = None,
    )
    assert rec["main_file"] == xf.__file__       # child imports the helper, not the script
    assert main_mod.__file__ == "/fake/user_script.py"  # restored in the parent


def test_scrub_secrets_handles_boolean_token():
    """token=True ("use the cached token") must not crash the child error scrubber."""
    out = xf._default_scrub_secrets("auth failed for hf_abcdefghij", hf_token = True)
    assert "hf_abcdefghij" not in out and "***" in out


def test_scrub_redacts_presigned_url():
    """A presigned S3/CAS blob URL in a child error carries temporary credentials in
    its query string; the default scrubber must redact the query before it is
    raised/logged in the parent, while leaving non-signed URLs intact."""
    url = (
        "https://cas-bridge.xethub.hf.co/xet-bridge-us/abc/def"
        "?X-Amz-Signature=deadbeefcafe&X-Amz-Credential=AKIAEXAMPLE123"
    )
    out = xf._default_scrub_secrets(f"403 Client Error for url: {url}")
    assert "X-Amz-Signature" not in out
    assert "deadbeefcafe" not in out and "AKIAEXAMPLE123" not in out
    assert "cas-bridge.xethub.hf.co/xet-bridge-us/abc/def?***" in out
    # A non-signed URL keeps its (harmless) query string.
    plain = xf._default_scrub_secrets("see https://huggingface.co/org/repo?download=true now")
    assert "download=true" in plain


def test_local_files_only_file_resolves_in_process(monkeypatch):
    """local_files_only resolves the single file from cache in-process and never
    spawns a network child (Hugging Face offline semantics)."""
    seen = {}

    def _dl(*a, **k):
        seen.update(k)
        return "/cache/file.gguf"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _dl)
    fake = _install(monkeypatch, [])  # the download seam must not be called
    out = xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None, local_files_only = True)
    assert out == "/cache/file.gguf"
    assert seen.get("local_files_only") is True
    assert fake.calls == [], "local_files_only must not spawn a download child"


def test_local_files_only_snapshot_resolves_in_process(monkeypatch):
    seen = {}

    def _snap(*a, **k):
        seen.update(k)
        return "/cache/snap"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _snap)
    fake = _install(monkeypatch, [])
    out = xf.snapshot_download_with_xet_fallback(DL_REPO, token = None, local_files_only = True)
    assert out == "/cache/snap"
    assert seen.get("local_files_only") is True
    assert fake.calls == [], "local_files_only must not spawn a download child"


def test_file_probe_uses_expanded_cache_dir(monkeypatch, tmp_path):
    """The single-file cache probe must use the expanded cache_dir (HF expands ~
    before writing), or a finalized file under ~/hf-cache is missed and a child is
    spawned for an already-cached file."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows home var
    seen = {}

    def _probe(repo_id, filename, *, repo_type, revision, cache_dir):
        seen["cache_dir"] = cache_dir
        return None  # not cached -> falls through to the (faked) download seam

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", _probe)
    fake = _install(monkeypatch, [("ok", "/cache/x")])
    xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None, cache_dir = "~/hfcache")
    assert seen["cache_dir"] == str(tmp_path / "hfcache")
    # The expanded cache_dir is also what the download attempt receives.
    assert fake.calls[0].cache_dir == str(tmp_path / "hfcache")


def test_pathlib_cache_dir_is_expanded(monkeypatch, tmp_path):
    """A pathlib.Path cache_dir with ~ must be normalized too (HF accepts Path), or
    the child writes under the literal '~/...' while the watchdog watches $HOME/...
    and the stall is never detected."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    fake = _install(monkeypatch, [("ok", "/cache/snap")])
    xf.snapshot_download_with_xet_fallback(
        DL_REPO, token = None, cache_dir = Path("~/hfcache")
    )
    # Normalized to an expanded string for the child attempt + probes.
    assert fake.calls[0].cache_dir == str(tmp_path / "hfcache")


def test_subfolder_forwarded_to_file_download(monkeypatch):
    """A single-file caller passing subfolder must not get a TypeError; subfolder
    is forwarded into the download params (and the cache probe uses the combined
    '<subfolder>/<filename>' path)."""
    probed = {}

    def _probe(repo_id, filename, *, repo_type, revision, cache_dir):
        probed["filename"] = filename
        return None  # not cached -> falls through to the faked attempt

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", _probe)
    fake = _install(monkeypatch, [("ok", "/cache/x")])
    out = xf.hf_hub_download_with_xet_fallback(
        DL_REPO, FILE, None, subfolder = "checkpoint-10"
    )
    assert out == "/cache/x"
    assert probed["filename"] == f"checkpoint-10/{FILE}"  # probe uses combined path
    assert fake.calls[0].subfolder == "checkpoint-10"  # forwarded to the child


def test_file_download_defaults_token_to_none(monkeypatch):
    """The single-file helper accepts no token (parity with hf_hub_download)."""
    fake = _install(monkeypatch, [("ok", "/cache/x")])
    out = xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE)  # no token arg
    assert out == "/cache/x" and len(fake.calls) == 1


def test_incomplete_cached_snapshot_not_short_circuited(hf_cache, monkeypatch):
    """A cached-but-incomplete snapshot (interrupted download) must not take the
    fast path; it must complete in the killable child instead."""
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: "/cache/snap")
    (_blobs_dir(hf_cache, DL_REPO) / "x.incomplete").write_bytes(b"abc")  # active partial
    fake = _install(monkeypatch, [("ok", "/cache/snap-fresh")])
    out = xf.snapshot_download_with_xet_fallback(DL_REPO, token = None)
    assert out == "/cache/snap-fresh" and len(fake.calls) == 1


def test_retry_status_failure_does_not_abort_fallback(monkeypatch):
    """A raising on_status during the Xet->HTTP retry must not abort the fallback."""
    fake = _install(monkeypatch, [("stall", None), ("ok", "/cache/x")])

    def boom(_message):
        raise RuntimeError("client gone")

    out = xf.snapshot_download_with_xet_fallback(DL_REPO, token = None, on_status = boom)
    assert out == "/cache/x"
    assert [c.disable_xet for c in fake.calls] == [False, True]


def test_unclearable_partial_forces_clean_redownload(hf_cache, monkeypatch):
    """When prep cannot clear an unsafe partial, the HTTP attempt forces a clean
    re-download instead of an unsafe resume over the sparse partial."""
    # The autouse fixture makes _default_prepare_for_http a no-op (simulates a
    # cleanup that left the partial in place).
    (_blobs_dir(hf_cache, DL_REPO) / "x.incomplete").write_bytes(b"abc")
    fake = _install(monkeypatch, [("stall", None), ("ok", "/cache/x")])
    out = xf.snapshot_download_with_xet_fallback(DL_REPO, token = None)
    assert out == "/cache/x"
    assert fake.calls[0].force_download is False   # Xet attempt: not forced
    assert fake.calls[1].force_download is True     # HTTP attempt: forced clean


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
    monkeypatch.setattr(xf, "_default_prepare_for_http", lambda rt, rid, cache_dir = None: prepared.append((rt, rid)))
    fake = _install(monkeypatch, [("stall", None), ("ok", "/cache/snap-dir")])
    out = xf.snapshot_download_with_xet_fallback(DL_REPO, token = None)
    assert out == "/cache/snap-dir"
    assert [c.kind for c in fake.calls] == ["snapshot", "snapshot"]
    assert [c.disable_xet for c in fake.calls] == [False, True]
    assert prepared == [("model", DL_REPO)]


def test_force_download_skips_fast_path_and_threads(monkeypatch):
    """force_download=True must bypass the warm-cache short-circuit and re-fetch in
    the killable child, forwarding force_download into the download params."""
    def _snap(*a, **k):
        pytest.fail("force_download must not take the local_files_only fast path")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _snap)
    fake = _install(monkeypatch, [("ok", "/cache/snap-dir")])
    out = xf.snapshot_download_with_xet_fallback(DL_REPO, token = None, force_download = True)
    assert out == "/cache/snap-dir"
    assert len(fake.calls) == 1 and fake.calls[0].force_download is True


def test_force_download_file_skips_cache_probe(monkeypatch, tmp_path):
    """The single-file path must also skip the cached-blob short-circuit and thread
    force_download through when force_download=True."""
    cached = tmp_path / "cached.gguf"
    cached.write_bytes(b"\0" * 8)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: str(cached))
    fake = _install(monkeypatch, [("ok", "/cache/x")])
    out = xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None, force_download = True)
    assert out == "/cache/x"
    assert len(fake.calls) == 1 and fake.calls[0].force_download is True


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
