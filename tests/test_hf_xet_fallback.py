# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for unsloth_zoo.hf_xet_fallback: watchdog, Xet->HTTP transport policy,
file/snapshot entrypoints, the UNSLOTH_DISABLE_XET knob, and HF_HUB_DISABLE_XET.

CPU-only, no network, no real subprocess (the download seam is monkeypatched).
The modules under test are loaded via importlib to avoid importing the full
``unsloth_zoo`` package (torch + GPU init).
"""

from __future__ import annotations

import errno
import importlib.util
import json
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


# Package placeholder so intra-package imports in hf_xet_fallback resolve to the
# files loaded below. Restored afterwards: a leftover would shadow the real
# unsloth_zoo; the loaded modules keep their own references and work regardless.
_saved_modules = {
    name: sys.modules.get(name)
    for name in ("unsloth_zoo", "unsloth_zoo.hf_cache_state", "unsloth_zoo.hf_xet_fallback")
}
if "unsloth_zoo" not in sys.modules:
    _pkg = _types.ModuleType("unsloth_zoo")
    _pkg.__path__ = [str(_ZOO_DIR)]
    sys.modules["unsloth_zoo"] = _pkg

hcs = _load("unsloth_zoo.hf_cache_state", "hf_cache_state.py")
xf = _load("unsloth_zoo.hf_xet_fallback", "hf_xet_fallback.py")

for _name, _mod in _saved_modules.items():
    if _mod is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _mod

# Real prep impl, captured before the autouse fixture stubs the module attribute.
_REAL_DEFAULT_PREPARE = xf._default_prepare_for_http


# Watchdog: fires only on a constant-size .incomplete, sparse-aware byte total.
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
        repo_ids = [REPO], on_stall = calls.append, interval = 0.05, stall_timeout = 0.5
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
        repo_ids = [REPO], on_stall = calls.append, interval = 0.05, stall_timeout = 0.5
    )
    try:
        time.sleep(0.8)
        assert calls == [], "watchdog fired with no active .incomplete"
    finally:
        stop.set()


def test_transient_unmeasurable_tick_is_progress(hf_cache, monkeypatch):
    """An unmeasurable tick (state -> None) counts as progress, but a later frozen state still stalls."""
    seq = {"n": 0}
    frozen = (2048, True)  # constant size + active .incomplete

    def fake_state(*args, **kwargs):
        seq["n"] += 1
        return None if seq["n"] <= 8 else frozen  # first ~8 ticks unmeasurable, then frozen

    monkeypatch.setattr(xf, "get_hf_download_state", fake_state)

    calls: list[str] = []
    stop = xf.start_watchdog(
        repo_ids = [REPO], on_stall = calls.append, interval = 0.05, stall_timeout = 0.3,
    )
    try:
        time.sleep(0.3)  # within the unmeasurable window
        assert calls == [], "watchdog fired during a transient-unmeasurable window"
        assert _wait(lambda: len(calls) >= 1, timeout = 3.0), "stall never fired after state recovered"
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


def test_file_watchdog_scopes_to_child_partial(hf_cache):
    """A single-file download follows only its own child's partials, so a growing baseline sibling does not mask its stalled child."""
    blobs = _blobs_dir(hf_cache)
    sibling = blobs / "sibling.incomplete"   # in flight -> in baseline
    sibling.write_bytes(b"\0" * 1024)
    baseline = {"sibling.incomplete"}

    grow_stop = threading.Event()

    def _grow():
        size = 1024
        while not grow_stop.wait(0.05):
            size += 4096
            sibling.write_bytes(b"\0" * size)   # healthy sibling progresses

    grower = threading.Thread(target = _grow, daemon = True)
    grower.start()

    # This download's child writes its own constant (stalled) partial, not in baseline.
    (blobs / "child.incomplete").write_bytes(b"\0" * 2048)

    calls: list[str] = []
    stop = xf.start_watchdog(
        repo_ids = [REPO], on_stall = calls.append, interval = 0.05, stall_timeout = 0.3,
        watch_new_partials_only = True, baseline_incomplete_blobs = baseline,
    )
    try:
        assert _wait(lambda: len(calls) >= 1, timeout = 3.0), (
            "file watchdog never fired: a growing sibling partial masked the stalled child"
        )
    finally:
        stop.set()
        grow_stop.set()


def test_repo_wide_watchdog_is_masked_by_sibling(hf_cache):
    """Contrast for file-scoping: the default repo-wide measure sums every blob, so a growing sibling resets the timer."""
    blobs = _blobs_dir(hf_cache)
    sibling = blobs / "sibling.incomplete"
    sibling.write_bytes(b"\0" * 1024)
    (blobs / "child.incomplete").write_bytes(b"\0" * 2048)   # constant

    grow_stop = threading.Event()

    def _grow():
        size = 1024
        while not grow_stop.wait(0.05):
            size += 4096
            sibling.write_bytes(b"\0" * size)

    grower = threading.Thread(target = _grow, daemon = True)
    grower.start()

    calls: list[str] = []
    stop = xf.start_watchdog(   # default: repo-wide (watch_new_partials_only = False)
        repo_ids = [REPO], on_stall = calls.append, interval = 0.05, stall_timeout = 0.5,
    )
    try:
        time.sleep(1.0)   # past stall_timeout, but repo-wide bytes keep growing
        assert calls == [], "repo-wide watchdog should be reset by the growing sibling"
    finally:
        stop.set()
        grow_stop.set()


def test_file_watchdog_ignores_baseline_only_partials(hf_cache):
    """When the only partial is a baseline sibling's, the file watchdog owns nothing and must not fire."""
    blobs = _blobs_dir(hf_cache)
    (blobs / "sibling.incomplete").write_bytes(b"\0" * 4096)   # constant baseline sibling

    calls: list[str] = []
    stop = xf.start_watchdog(
        repo_ids = [REPO], on_stall = calls.append, interval = 0.05, stall_timeout = 0.5,
        watch_new_partials_only = True, baseline_incomplete_blobs = {"sibling.incomplete"},
    )
    try:
        time.sleep(0.8)
        assert calls == [], "file watchdog fired on a baseline sibling partial it must ignore"
    finally:
        stop.set()


def _spawn_holding_open(path: Path) -> "subprocess.Popen":
    """Child process that opens *path* and holds it open without writing (a hung download); prints 'ok' when open."""
    code = (
        "import sys, time\n"
        "f = open(sys.argv[1], 'r+b')\n"
        "sys.stdout.write('ok'); sys.stdout.flush()\n"
        "time.sleep(30)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code, str(path)], stdout = subprocess.PIPE
    )
    assert proc.stdout.read(2) == b"ok"   # wait until the child holds the file open
    return proc


def test_file_watchdog_detects_resumed_baseline_partial(hf_cache):
    """A resumed download reuses a baseline .incomplete; pid-scoping (not name exclusion) still detects the hang."""
    blobs = _blobs_dir(hf_cache)
    partial = blobs / "resumed.incomplete"
    partial.write_bytes(b"\0" * 4096)        # leftover from a prior interrupted download
    baseline = {"resumed.incomplete"}        # present before the resuming child starts

    child = _spawn_holding_open(partial)      # hung resume: holds it open, never grows it
    try:
        calls: list[str] = []
        stop = xf.start_watchdog(
            repo_ids = [REPO], on_stall = calls.append, interval = 0.05, stall_timeout = 0.3,
            watch_new_partials_only = True, baseline_incomplete_blobs = baseline,
            child_pid = child.pid,
        )
        try:
            assert _wait(lambda: len(calls) >= 1, timeout = 3.0), (
                "watchdog did not fire on a hung resume of a baseline partial"
            )
        finally:
            stop.set()
    finally:
        child.terminate()
        child.wait(timeout = 5)


def test_file_watchdog_resumed_partial_fires_without_pid_ownership(hf_cache, monkeypatch):
    """When ownership is unknowable (no psutil/proc -> None), the repo-wide fallback still stalls a frozen resumed baseline partial."""
    blobs = _blobs_dir(hf_cache)
    (blobs / "resumed.incomplete").write_bytes(b"\0" * 4096)   # leftover resume, constant (hung)
    monkeypatch.setattr(xf, "_child_open_incomplete_blobs", lambda pid: None)  # no psutil / no /proc

    calls: list[str] = []
    stop = xf.start_watchdog(
        repo_ids = [REPO], on_stall = calls.append, interval = 0.05, stall_timeout = 0.3,
        watch_new_partials_only = True, baseline_incomplete_blobs = {"resumed.incomplete"},
        child_pid = 4242,   # non-None, but open-file inspection -> None -> repo-wide fallback
    )
    try:
        assert _wait(lambda: len(calls) >= 1, timeout = 3.0), (
            "watchdog never fired on a resumed baseline partial when pid ownership is unavailable"
        )
    finally:
        stop.set()


def test_file_watchdog_pid_scope_ignores_unowned_sibling(hf_cache):
    """With pid scoping, a growing sibling the child does not hold open is ignored; the child's own constant partial still stalls."""
    blobs = _blobs_dir(hf_cache)
    owned_partial = blobs / "owned.incomplete"
    owned_partial.write_bytes(b"\0" * 2048)   # child holds this open, constant (hung)
    sibling = blobs / "sibling.incomplete"
    sibling.write_bytes(b"\0" * 1024)

    grow_stop = threading.Event()

    def _grow():
        size = 1024
        while not grow_stop.wait(0.05):
            size += 4096
            sibling.write_bytes(b"\0" * size)   # unrelated sibling progressing

    grower = threading.Thread(target = _grow, daemon = True)
    grower.start()

    child = _spawn_holding_open(owned_partial)
    try:
        calls: list[str] = []
        stop = xf.start_watchdog(
            repo_ids = [REPO], on_stall = calls.append, interval = 0.05, stall_timeout = 0.3,
            watch_new_partials_only = True, baseline_incomplete_blobs = set(),
            child_pid = child.pid,
        )
        try:
            assert _wait(lambda: len(calls) >= 1, timeout = 3.0), (
                "pid-scoped watchdog never fired: an unowned growing sibling masked the stall"
            )
        finally:
            stop.set()
    finally:
        grow_stop.set()
        child.terminate()
        child.wait(timeout = 5)


def test_file_watchdog_empty_open_set_ignores_sibling(hf_cache, monkeypatch):
    """An EMPTY child open-set means the child owns no partial yet (connect/metadata phase), so a stalled sibling must not fire."""
    blobs = _blobs_dir(hf_cache)
    # Sibling partial created after baseline (not name-excluded), constant (stalled).
    (blobs / "sibling.incomplete").write_bytes(b"\0" * 4096)
    monkeypatch.setattr(xf, "_child_open_incomplete_blobs", lambda pid: set())  # child owns none

    calls: list[str] = []
    stop = xf.start_watchdog(
        repo_ids = [REPO], on_stall = calls.append, interval = 0.05, stall_timeout = 0.5,
        watch_new_partials_only = True, baseline_incomplete_blobs = set(),
        child_pid = 4242,   # non-None so the precise child-open path is taken
    )
    try:
        time.sleep(0.8)
        assert calls == [], "watchdog falsely fired on a sibling partial the child does not own"
    finally:
        stop.set()


def test_get_state_empty_cache(hf_cache):
    assert xf.get_hf_download_state([REPO]) == (0, False)


def test_get_state_absent_cache_root(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "no-such-cache"))
    assert xf.get_hf_download_state([REPO]) == (0, False)


def test_get_state_skips_local_paths(hf_cache):
    # Filesystem paths are not repo IDs and must be ignored without error.
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


def test_blob_bytes_present_zero_blocks_is_zero(tmp_path):
    """A fully-sparse .incomplete (st_size > 0, 0 allocated blocks) counts as 0 bytes present, not full size."""
    p = tmp_path / "sparse.incomplete"
    with open(p, "wb") as f:
        f.truncate(8 * 1024 * 1024)  # apparent 8 MiB, nothing actually written
    st = p.stat()
    if getattr(st, "st_blocks", None) is None:
        pytest.skip("st_blocks not reported on this platform")
    if st.st_blocks != 0:
        pytest.skip("filesystem pre-allocated blocks for the sparse file")
    assert hcs.blob_bytes_present(p) == 0


def test_custom_cache_dir_is_watched_and_cleaned(tmp_path, monkeypatch):
    """A stall under a caller-supplied cache_dir (not HF_HUB_CACHE) is seen by the probe, watchdog, and HTTP-prep purge."""
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

    # HTTP-prep purge removes the unsafe partial from the custom cache (real impl;
    # the autouse fixture stubs the attribute). Age past the grace so it reads stalled.
    old = time.time() - 600
    os.utime(partial, (old, old))
    _REAL_DEFAULT_PREPARE("model", REPO, cache_dir = str(custom_cache))
    assert not partial.exists()


def test_prepare_for_http_clears_broken_snapshot_symlink(tmp_path):
    """A broken snapshot symlink reads as incomplete, so HTTP prep must clear it too or the retry re-trips the watchdog."""
    repo = "ztest/broken-symlink"
    repo_dir = tmp_path / f"models--{repo.replace('/', '--')}"
    snap = repo_dir / "snapshots" / "abc123"
    snap.mkdir(parents = True)
    link = snap / "model.safetensors"
    link.symlink_to(repo_dir / "blobs" / "missing-blob")  # dangling
    assert link.is_symlink() and not link.exists()

    assert xf.get_hf_download_state([repo], cache_dir = str(tmp_path)) == (0, True)

    _REAL_DEFAULT_PREPARE("model", repo, cache_dir = str(tmp_path))

    assert not link.is_symlink(), "broken snapshot symlink not cleared by HTTP prep"
    assert xf.get_hf_download_state([repo], cache_dir = str(tmp_path)) == (0, False)


def test_prepare_for_http_spares_concurrent_sibling_active_symlink(tmp_path):
    """HTTP prep spares a sibling's dangling link while its .incomplete partner is being written, but clears our own stale link."""
    repo = "ztest/concurrent"
    repo_dir = tmp_path / f"models--{repo.replace('/', '--')}"
    blobs = repo_dir / "blobs"
    snap = repo_dir / "snapshots" / "sha"
    blobs.mkdir(parents = True)
    snap.mkdir(parents = True)

    # Sibling mid-download: dangling link to a blob with an active .incomplete partner.
    active_partner = blobs / "activehash.incomplete"
    active_partner.write_bytes(b"active")
    sibling_link = snap / "active.safetensors"
    sibling_link.symlink_to(blobs / "activehash")

    # Our own stale interrupted link: target blob has no .incomplete partner.
    stale_link = snap / "stale.safetensors"
    stale_link.symlink_to(blobs / "stalehash")

    _REAL_DEFAULT_PREPARE("model", repo, cache_dir = str(tmp_path), active_grace = 180)

    assert sibling_link.is_symlink(), "active sibling's dangling symlink must be preserved"
    assert not stale_link.is_symlink(), "our own stale dangling symlink must still be cleared"


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
    """The detector inspects every snapshot: an older broken revision reads incomplete even when a newer one is clean."""
    repo = "ztest/two-snaps"
    repo_dir = tmp_path / f"models--{repo.replace('/', '--')}"
    old = repo_dir / "snapshots" / "oldsha"
    new = repo_dir / "snapshots" / "newsha"
    old.mkdir(parents = True)
    new.mkdir(parents = True)
    (old / "model.safetensors").symlink_to(repo_dir / "blobs" / "missing")  # broken older
    (new / "config.json").write_text("{}")                                 # clean newer
    # Make the clean snapshot newest by mtime so a latest-only check would pass.
    os.utime(new, (time.time() + 10, time.time() + 10))
    assert xf.get_hf_download_state([repo], cache_dir = str(tmp_path)) == (0, True)


def test_snapshot_fast_path_rejects_broken_requested_revision(tmp_path, monkeypatch):
    """The fast path validates the EXACT returned dir, so a broken requested revision defers to the killable child."""
    snap = tmp_path / "snapshots" / "oldsha"
    snap.mkdir(parents = True)
    (snap / "model.safetensors").symlink_to(tmp_path / "blobs" / "missing")  # dangling
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(snap))
    # Repo-wide scan sees nothing (empty cache root); only the per-revision check can catch it.
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "empty-cache"))
    fake = _install(monkeypatch, [("ok", "/cache/snap-fresh")])
    out = xf.snapshot_download_with_xet_fallback(DL_REPO, token = None)
    assert out == "/cache/snap-fresh", "fast path returned a broken requested revision"
    assert len(fake.calls) == 1


def test_prepare_for_http_clears_broken_symlink_in_older_snapshot(tmp_path):
    """HTTP prep clears dangling links across all snapshots, not just the newest."""
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
    """On a case-sensitive FS, HTTP prep for Org/Repo purges only its exact-case dir, never a case-colliding org/repo."""
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
    # Age both past the grace; lower is spared by repo attribution, not mtime.
    old = time.time() - 600
    os.utime(upper_partial, (old, old))
    os.utime(lower_partial, (old, old))

    _REAL_DEFAULT_PREPARE("model", "Org/Repo", cache_dir = str(tmp_path))

    assert not upper_partial.exists(), "exact-case partial should be purged"
    assert lower_partial.exists(), "case-colliding repo's partial must be preserved"


def test_repo_type_none_resolves_model_cache(hf_cache):
    """repo_type=None (HF default model) resolves the models--<id> dir, not a bogus Nones--<id> dir."""
    blobs = _blobs_dir(hf_cache)
    (blobs / "x.incomplete").write_bytes(b"abc")

    model_state = xf.get_hf_download_state([REPO], repo_type = "model")
    none_state = xf.get_hf_download_state([REPO], repo_type = None)
    assert model_state == none_state
    assert none_state[1] is True and none_state[0] > 0


def test_state_ignores_case_colliding_repo_partial(tmp_path, monkeypatch):
    """The read/watchdog path attributes a partial only to an exact-case repo dir."""
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
    exact = tmp_path / "models--Org--Repo" / "blobs"
    other = tmp_path / "models--org--repo" / "blobs"
    exact.mkdir(parents = True)
    other.mkdir(parents = True)
    if exact.parent.resolve() == other.parent.resolve():
        pytest.skip("case-insensitive filesystem; cannot collide cache dirs")
    (other / "stale.incomplete").write_bytes(b"x")  # only the lowercase repo

    assert xf.get_hf_download_state(["Org/Repo"]) == (0, False)


def test_single_folded_match_rejected_on_case_sensitive_fs(tmp_path, monkeypatch):
    """On a case-sensitive FS a folded-but-not-exact dir is a different repo, so its partial is not attributed here."""
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
    lower = tmp_path / "models--org--repo" / "blobs"
    lower.mkdir(parents = True)
    if (tmp_path / "models--Org--Repo").exists():
        pytest.skip("case-insensitive filesystem; the folded dir is the same entry")
    (lower / "stale.incomplete").write_bytes(b"x")  # only the lowercase repo exists
    assert xf.get_hf_download_state(["Org/Repo"]) == (0, False)


def test_cache_dir_is_expanded(tmp_path, monkeypatch):
    """A cache_dir with ~ is expanded (as HF does on write), else the probe scans the literal '~/...' path."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows home var
    blobs = tmp_path / "hfcache" / f"models--{REPO.replace('/', '--')}" / "blobs"
    blobs.mkdir(parents = True)
    (blobs / "p.incomplete").write_bytes(b"abc")

    total, has_incomplete = xf.get_hf_download_state([REPO], cache_dir = "~/hfcache")
    assert has_incomplete is True and total > 0


def test_status_callback_failure_does_not_kill_watchdog(hf_cache):
    """A raising on_heartbeat must not stop the watchdog from detecting a stall and firing on_stall."""
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


# Transport policy: cached short-circuit, cancel, error propagation, the single
# Xet->HTTP fallback, injected prepare seam, and UNSLOTH_DISABLE_XET. The download
# seam (_run_download_attempt) is faked, so no real spawn.
DL_REPO, FILE = "ztest/xet-dl", "model-Q4_K_XL.gguf"


@pytest.fixture(autouse = True)
def _no_real_cache_hit(monkeypatch):
    """Default: cache probe and snapshot fast path miss, so tests exercise the download seam."""
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)

    def _snap_miss(*a, **k):
        raise FileNotFoundError("not fully cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _snap_miss)
    # Neutralize the generic cache purge; tests that care record it.
    monkeypatch.setattr(xf, "_default_prepare_for_http", lambda *a, **k: None)
    # No env knob unless a test sets it.
    monkeypatch.delenv("UNSLOTH_DISABLE_XET", raising = False)
    monkeypatch.delenv("UNSLOTH_STABLE_DOWNLOADS", raising = False)
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising = False)


class _FakeAttempt:
    """Records download-seam calls and returns scripted results (matches _run_download_attempt's signature)."""

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
    fake = _install(monkeypatch, [])

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


def test_cancel_honored_even_when_file_cached(monkeypatch, tmp_path):
    """A pre-set cancel_event raises even when the file is already cached (the warm-cache short-circuit honors cancel first)."""
    cached = tmp_path / "cached.gguf"
    cached.write_bytes(b"\0" * 8)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: str(cached))
    fake = _install(monkeypatch, [])
    ev = threading.Event()
    ev.set()
    with pytest.raises(RuntimeError, match = "Cancelled"):
        xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None, cancel_event = ev)
    assert fake.calls == []


def test_snapshot_cancel_honored_even_when_cached(monkeypatch, tmp_path):
    """The snapshot wrapper honors a pre-set cancel before its warm-cache short-circuit."""
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "model.safetensors").write_bytes(b"x")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(snap))
    fake = _install(monkeypatch, [])
    ev = threading.Event()
    ev.set()
    with pytest.raises(RuntimeError, match = "Cancelled"):
        xf.snapshot_download_with_xet_fallback(DL_REPO, token = None, cancel_event = ev)
    assert fake.calls == []


def test_nonstall_error_propagates_without_fallback(monkeypatch):
    fake = _install(monkeypatch, [("error", "RepositoryNotFoundError: 404 not found")])
    # Deterministic Hub error re-raised with its original type across the spawn boundary,
    # reconstructed from the child's "<Name>: ..." prefix.
    expected_cls = xf._resolve_exception_class("RepositoryNotFoundError")
    assert expected_cls is not None and expected_cls is not RuntimeError
    with pytest.raises(expected_cls, match = "RepositoryNotFoundError"):
        xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None)
    assert len(fake.calls) == 1, "deterministic error must not trigger an HTTP fallback"
    assert fake.calls[0].disable_xet is False


def test_crashed_child_retries_over_http(monkeypatch):
    """A silent process-level crash (child exits without a result) retries over HTTP; a clean second result is accepted."""
    fake = _install(monkeypatch, [("crashed", "exited without a result"), ("ok", "/cache/x")])
    out = xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None)
    assert out == "/cache/x"
    assert [c.disable_xet for c in fake.calls] == [False, True]


def test_crashed_child_on_both_transports_raises(monkeypatch):
    """If the child crashes on Xet AND on HTTP, surface a hard error after both attempts."""
    fake = _install(monkeypatch, [("crashed", "boom"), ("crashed", "boom")])
    with pytest.raises(RuntimeError, match = "boom"):
        xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None)
    assert [c.disable_xet for c in fake.calls] == [False, True]


def test_retryable_xet_error_retries_over_http(monkeypatch):
    """A transient Xet failure (CAS timeout / 5xx) retries over HTTP; a clean HTTP result is accepted."""
    fake = _install(monkeypatch, [("retryable_error", "CasClientError: request timed out"), ("ok", "/cache/x")])
    out = xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None)
    assert out == "/cache/x"
    assert [c.disable_xet for c in fake.calls] == [False, True]


def test_retryable_xet_error_on_both_transports_raises(monkeypatch):
    """A transient error on both transports surfaces after both attempts rather than looping."""
    fake = _install(monkeypatch, [("retryable_error", "503 Server Error"), ("retryable_error", "503 Server Error")])
    with pytest.raises(RuntimeError, match = "503"):
        xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None)
    assert [c.disable_xet for c in fake.calls] == [False, True]


def test_is_retryable_download_error_classification():
    """Transient transport failures (hf_xet/CAS, timeout, reset, 5xx/429) are retryable; deterministic Hub/OS and unknown errors are not."""
    f = xf._is_retryable_download_error

    # Transient transport failures -> retryable.
    assert f(Exception("hf_xet download failed: data processing error")) is True
    assert f(TimeoutError("connection reset by peer")) is True
    assert f(Exception("CasClientError: deadline exceeded")) is True

    class _Resp503(Exception):
        response = type("R", (), {"status_code": 503})()

    assert f(_Resp503("server error")) is True

    class _Status429(Exception):
        status_code = 429

    assert f(_Status429("Too Many Requests")) is True

    class _Status408(Exception):
        status_code = 408

    assert f(_Status408("Request Timeout")) is True  # 408 is transient

    # Deterministic Hub errors -> not retryable (class name or 4xx status).
    class _Status416(Exception):
        status_code = 416

    assert f(_Status416("Range Not Satisfiable")) is False
    class RepositoryNotFoundError(Exception):
        pass

    assert f(RepositoryNotFoundError("404 Client Error")) is False

    class _Resp404(Exception):
        response = type("R", (), {"status_code": 404})()

    assert f(_Resp404("not found")) is False
    assert f(OSError(errno.ENOSPC, "No space left on device")) is False
    assert f(ValueError("unexpected response payload")) is False  # unknown -> deterministic


def test_local_entry_not_found_transient_is_retryable():
    """A transient LocalEntryNotFoundError (HEAD connection error/timeout) is retryable; a genuine offline miss stays deterministic and type-preserved."""
    f = xf._is_retryable_download_error

    class LocalEntryNotFoundError(Exception):
        pass

    # Transient connection wrapper -> retryable.
    transient = LocalEntryNotFoundError(
        "An error happened while trying to locate the file on the Hub and we cannot find the "
        "requested files in the local cache. Please check your connection and try again."
    )
    assert f(transient) is True
    timed_out = LocalEntryNotFoundError("Read timed out while fetching metadata")
    assert f(timed_out) is True
    # Genuine offline miss (no transient hint) -> deterministic, type-preserved.
    offline = LocalEntryNotFoundError(
        "Cannot find the requested files in the disk cache and outgoing traffic has been disabled."
    )
    assert f(offline) is False
    assert "LocalEntryNotFoundError" in xf._DETERMINISTIC_ERROR_NAMES
    cls = xf._resolve_exception_class("LocalEntryNotFoundError")
    assert cls is not None and issubclass(cls, BaseException)


def test_immediate_success_uses_xet_only(monkeypatch):
    prepared = []
    monkeypatch.setattr(xf, "_default_prepare_for_http", lambda *a, **k: prepared.append(a))
    fake = _install(monkeypatch, [("ok", "/cache/model.gguf")])
    out = xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None)
    assert out == "/cache/model.gguf"
    assert len(fake.calls) == 1 and fake.calls[0].disable_xet is False
    assert prepared == [], "no cache prep should run when Xet succeeds first try"


def test_stall_then_http_fallback_succeeds(monkeypatch):
    prepared = []
    monkeypatch.setattr(xf, "_default_prepare_for_http", lambda repo_type, repo_id, cache_dir = None, **k: prepared.append((repo_type, repo_id)))
    fake = _install(monkeypatch, [("stall", None), ("ok", "/cache/model.gguf")])

    out = xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None)
    assert out == "/cache/model.gguf"
    assert len(fake.calls) == 2
    assert fake.calls[0].disable_xet is False  # Xet first
    assert fake.calls[1].disable_xet is True   # HTTP fallback
    assert prepared == [("model", DL_REPO)], "must prep cache for HTTP before the retry"


def test_injected_prepare_for_http_used(monkeypatch):
    """An injected prepare_for_http_fn is used; the generic default must not run."""
    monkeypatch.setattr(
        xf, "_default_prepare_for_http", lambda *a, **k: pytest.fail("generic prepare ran")
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
    """UNSLOTH_DISABLE_XET=1 skips Xet: the first (and only) attempt is HTTP."""
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
    """The single-file wrapper accepts cache_dir and threads it through."""
    fake = _install(monkeypatch, [("ok", "/cache/model.gguf")])
    out = xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None, cache_dir = "/custom/cache")
    assert out == "/cache/model.gguf"
    assert fake.calls[0].cache_dir == "/custom/cache"


# Spawn env-timing: the parent sets HF_HUB_DISABLE_XET before the child starts, so
# the child inherits it before re-importing huggingface_hub (constants cache it at
# import). Uses a fake spawn context -- no real subprocess.
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
    # Parent env restored afterwards (was unset).
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
    # On the Xet-first attempt, do not force-disable Xet for the child.
    assert rec["disable_xet"] is None


class _EmptyQueue:
    def get(self, timeout = None):
        import queue as _queue
        raise _queue.Empty

    def get_nowait(self):
        import queue as _queue
        raise _queue.Empty

    def put(self, item):
        pass


def test_run_attempt_no_result_is_crashed(monkeypatch):
    """A child that exits without enqueuing a result maps to 'crashed' (HTTP may recover), not a deterministic 'error'."""
    rec: dict = {}

    class _Ctx:
        def Process(self, *, target = None, kwargs = None, daemon = None):
            return _FakeProc(rec)

        def Queue(self):
            return _EmptyQueue()

    monkeypatch.setattr(xf, "_CTX", _Ctx())
    kind_result, _ = xf._run_download_attempt(
        DL_REPO, kind = "snapshot", params = {"repo_id": DL_REPO}, token = None,
        repo_type = "model", disable_xet = False, cancel_event = None,
        stall_timeout = 0.2, interval = 0.05, grace_period = 0.2, on_status = None,
    )
    assert kind_result == "crashed"


def test_child_skips_gpu_init_env_set_before_spawn_and_restored(monkeypatch):
    """The child inherits UNSLOTH_ZOO_DISABLE_GPU_INIT=1 at spawn (skips heavy torch init); the parent env is restored after."""
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
    """The parent repoints __main__.__file__ to this module at spawn (so an unguarded caller is not re-executed) and restores it."""
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
    """token=True must not crash the child error scrubber."""
    out = xf._default_scrub_secrets("auth failed for hf_abcdefghij", hf_token = True)
    assert "hf_abcdefghij" not in out and "***" in out


def test_scrub_redacts_presigned_url():
    """The scrubber redacts a presigned S3/CAS URL's credential query string, leaving non-signed URLs intact."""
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


def test_scrub_redaction_preserves_surrounding_delimiters():
    """Query redaction stops at the closing delimiter of an embedded signed URL (does not swallow the ``"}``)."""
    embedded = (
        '{"error": "403", "url": '
        '"https://cas-bridge.xethub.hf.co/x/y?X-Amz-Signature=deadbeef&X-Amz-Expires=3600"}'
    )
    out = xf._default_scrub_secrets(embedded)
    assert "deadbeef" not in out                       # signed query redacted
    assert "cas-bridge.xethub.hf.co/x/y?***" in out
    assert out.endswith('"}')                           # closing delimiters preserved
    # A signed URL wrapped in single quotes / parens keeps those delimiters too.
    wrapped = "(https://s3.amazonaws.com/b/k?X-Amz-Signature=abc123) tail"
    out2 = xf._default_scrub_secrets(wrapped)
    assert "abc123" not in out2 and "?***)" in out2 and out2.endswith(") tail")


def test_local_files_only_file_resolves_in_process(monkeypatch):
    """local_files_only resolves the file from cache in-process and never spawns a network child."""
    seen = {}

    def _dl(*a, **k):
        seen.update(k)
        return "/cache/file.gguf"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _dl)
    fake = _install(monkeypatch, [])
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
    """The single-file probe uses the expanded cache_dir, else a finalized file under ~/hf-cache is missed."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows home var
    seen = {}

    def _probe(repo_id, filename, *, repo_type, revision, cache_dir):
        seen["cache_dir"] = cache_dir
        return None  # not cached -> falls through to the faked seam

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", _probe)
    fake = _install(monkeypatch, [("ok", "/cache/x")])
    xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None, cache_dir = "~/hfcache")
    assert seen["cache_dir"] == str(tmp_path / "hfcache")
    assert fake.calls[0].cache_dir == str(tmp_path / "hfcache")


def test_pathlib_cache_dir_is_expanded(monkeypatch, tmp_path):
    """A pathlib.Path cache_dir with ~ is normalized too, else the child writes under '~/...' and the stall is undetected."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    fake = _install(monkeypatch, [("ok", "/cache/snap")])
    xf.snapshot_download_with_xet_fallback(
        DL_REPO, token = None, cache_dir = Path("~/hfcache")
    )
    assert fake.calls[0].cache_dir == str(tmp_path / "hfcache")


def test_subfolder_forwarded_to_file_download(monkeypatch):
    """subfolder is forwarded into the download params and the probe uses the combined '<subfolder>/<filename>' path."""
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


def test_unrelated_partial_does_not_block_clean_cached_snapshot(hf_cache, monkeypatch):
    """A clean requested snapshot short-circuits in-process even with a stale unrelated .incomplete: the fast path validates only the returned dir."""
    blobs = _blobs_dir(hf_cache, DL_REPO)
    repo_dir = blobs.parent
    snap = repo_dir / "snapshots" / "goodsha"
    snap.mkdir(parents = True)
    good = blobs / "good"
    good.write_bytes(b"weights")
    (snap / "model.safetensors").symlink_to(good)        # resolves -> snapshot clean
    (blobs / "other.incomplete").write_bytes(b"abc")     # unrelated stale partial
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(snap))
    fake = _install(monkeypatch, [])                     # must NOT spawn a child
    out = xf.snapshot_download_with_xet_fallback(DL_REPO, token = None)
    assert out == str(snap), "clean cached snapshot rejected by an unrelated partial"
    assert fake.calls == [], "spawned a download despite a clean requested snapshot"


def test_retry_status_failure_does_not_abort_fallback(monkeypatch):
    """A raising on_status during the Xet->HTTP retry must not abort the fallback."""
    fake = _install(monkeypatch, [("stall", None), ("ok", "/cache/x")])

    def boom(_message):
        raise RuntimeError("client gone")

    out = xf.snapshot_download_with_xet_fallback(DL_REPO, token = None, on_status = boom)
    assert out == "/cache/x"
    assert [c.disable_xet for c in fake.calls] == [False, True]


def test_unclearable_partial_forces_clean_redownload(hf_cache, monkeypatch):
    """When prep cannot clear an unsafe partial, the HTTP attempt forces a clean re-download rather than resume over it."""
    # The autouse fixture makes _default_prepare_for_http a no-op (partial left in place).
    (_blobs_dir(hf_cache, DL_REPO) / "x.incomplete").write_bytes(b"abc")
    fake = _install(monkeypatch, [("stall", None), ("ok", "/cache/x")])
    out = xf.snapshot_download_with_xet_fallback(DL_REPO, token = None)
    assert out == "/cache/x"
    assert fake.calls[0].force_download is False   # Xet attempt: not forced
    assert fake.calls[1].force_download is True    # HTTP attempt: forced clean


# Snapshot variant: in-process fast path on a warm cache, else watched download.
def test_snapshot_fast_path_no_child(hf_cache, monkeypatch):
    """A fully cached repo (weights present) resolves in-process via local_files_only, no child attempt."""
    blobs = _blobs_dir(hf_cache, DL_REPO)
    snap = blobs.parent / "snapshots" / "sha"
    snap.mkdir(parents = True)
    weight = blobs / "w"
    weight.write_bytes(b"\0" * 16)
    (snap / "model.safetensors").symlink_to(weight)   # weight present -> complete
    (snap / "config.json").write_text("{}")
    seen = {}

    def _snap(*a, **k):
        seen["local_files_only"] = k.get("local_files_only")
        return str(snap)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _snap)
    fake = _install(monkeypatch, [])  # must not be called
    out = xf.snapshot_download_with_xet_fallback(DL_REPO, token = None)
    assert out == str(snap)
    assert seen["local_files_only"] is True
    assert fake.calls == [], "spawned a download for an already-cached snapshot"


def test_snapshot_dir_is_complete_unit(tmp_path):
    """Weight presence drives completeness: config-only is incomplete, a resolvable weight is complete."""
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}")
    assert hcs.snapshot_dir_is_complete(snap) is False  # no weights
    blob = tmp_path / "blob"
    blob.write_bytes(b"weights")
    (snap / "model.safetensors").symlink_to(blob)
    assert hcs.snapshot_dir_is_complete(snap) is True


def test_snapshot_dir_is_complete_broken_symlink(tmp_path):
    """A dangling weight symlink reads as incomplete."""
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "model.safetensors").symlink_to(tmp_path / "missing")
    assert hcs.snapshot_dir_is_complete(snap) is False


def test_snapshot_dir_is_complete_missing_shard(tmp_path):
    """A shard index whose shards are not all on disk reads as incomplete until they are."""
    snap = tmp_path / "snap"
    snap.mkdir()
    blob = tmp_path / "blob"
    blob.write_bytes(b"x")
    (snap / "model-00001-of-00002.safetensors").symlink_to(blob)
    (snap / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    assert hcs.snapshot_dir_is_complete(snap) is False  # shard 2 missing
    (snap / "model-00002-of-00002.safetensors").symlink_to(blob)
    assert hcs.snapshot_dir_is_complete(snap) is True


def test_snapshot_dir_is_complete_missing_shard_without_index(tmp_path):
    """A multi-shard pull is incomplete until the index sidecar is present, even with every shard on disk (transformers needs the index to load)."""
    snap = tmp_path / "snap"
    snap.mkdir()
    blob = tmp_path / "blob"
    blob.write_bytes(b"x")
    (snap / "model-00001-of-00003.safetensors").symlink_to(blob)
    assert hcs.snapshot_dir_is_complete(snap) is False  # shards 2 and 3 missing
    (snap / "model-00002-of-00003.safetensors").symlink_to(blob)
    assert hcs.snapshot_dir_is_complete(snap) is False  # shard 3 still missing
    (snap / "model-00003-of-00003.safetensors").symlink_to(blob)
    assert hcs.snapshot_dir_is_complete(snap) is False  # all shards present, no index
    (snap / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00003.safetensors",
                    "b": "model-00002-of-00003.safetensors",
                    "c": "model-00003-of-00003.safetensors",
                }
            }
        )
    )
    assert hcs.snapshot_dir_is_complete(snap) is True   # index present -> loadable


def test_snapshot_dir_is_complete_ignores_trainer_artifacts(tmp_path):
    """Trainer/optimizer state files (.bin/.pt/.pth) are not loadable weights, so a cache holding only those reads incomplete."""
    snap = tmp_path / "snap"
    snap.mkdir()
    blob = tmp_path / "blob"
    blob.write_bytes(b"x")
    for junk in (
        "training_args.bin", "optimizer.pt", "scheduler.pt",
        "rng_state.pth", "rng_state_0.pth", "scaler.pt",
    ):
        (snap / junk).symlink_to(blob)
    (snap / "config.json").write_text("{}")
    assert hcs.snapshot_dir_is_complete(snap) is False  # only trainer state, no weights
    (snap / "model.safetensors").symlink_to(blob)
    assert hcs.snapshot_dir_is_complete(snap) is True   # real weight present


def test_snapshot_dir_is_complete_checkpoint_index_does_not_gate_root(tmp_path):
    """A per-checkpoint shard index with missing shards does not fail an unpatterned root warm (the root weights are what loads)."""
    snap = tmp_path / "snap"
    (snap / "checkpoint-7").mkdir(parents = True)
    blob = tmp_path / "blob"
    blob.write_bytes(b"x")
    (snap / "model.safetensors").symlink_to(blob)                    # complete root weight
    # Incomplete checkpoint shard index (shard 2 missing) under checkpoint-7/.
    (snap / "checkpoint-7" / "model-00001-of-00002.safetensors").symlink_to(blob)
    (snap / "checkpoint-7" / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors",
                                   "b": "model-00002-of-00002.safetensors"}})
    )
    # Root warm is complete: the checkpoint index is skipped, the root weight suffices.
    assert hcs.snapshot_dir_is_complete(snap) is True


def test_fast_path_rejects_config_only_snapshot(hf_cache, monkeypatch):
    """The fast path rejects a config-only snapshot HF's local_files_only may return, deferring to the killable child."""
    blobs = _blobs_dir(hf_cache, DL_REPO)
    snap = blobs.parent / "snapshots" / "sha"
    snap.mkdir(parents = True)
    cfg_blob = blobs / "cfg"
    cfg_blob.write_text("{}")
    (snap / "config.json").symlink_to(cfg_blob)   # only config, no weights
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(snap))
    fake = _install(monkeypatch, [("ok", "/cache/snap-fresh")])
    out = xf.snapshot_download_with_xet_fallback(DL_REPO, token = None)
    assert out == "/cache/snap-fresh" and len(fake.calls) == 1


def test_fast_path_requires_each_named_weight(hf_cache, monkeypatch):
    """The pre-download short-circuit rejects a cache holding only one of several explicitly named weights (base+adapter, only base cached)."""
    blobs = _blobs_dir(hf_cache, DL_REPO)
    snap = blobs.parent / "snapshots" / "sha"
    snap.mkdir(parents = True)
    base_blob = blobs / "w"
    base_blob.write_bytes(b"x")
    (snap / "model.safetensors").symlink_to(base_blob)   # base only; adapter missing
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(snap))
    fake = _install(monkeypatch, [("ok", "/cache/snap-fresh")])
    out = xf.snapshot_download_with_xet_fallback(
        DL_REPO, token = None,
        allow_patterns = ["model.safetensors", "adapter_model.safetensors"],
    )
    assert out == "/cache/snap-fresh" and len(fake.calls) == 1


def test_child_broken_snapshot_retries_over_http(monkeypatch, tmp_path):
    """A broken child snapshot (dangling symlinks) is rejected on Xet and retried over HTTP; a clean second result is accepted."""
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "model.safetensors").symlink_to(tmp_path / "missing")   # dangling
    clean = tmp_path / "clean"
    clean.mkdir()
    blob = tmp_path / "b"
    blob.write_bytes(b"x")
    (clean / "model.safetensors").symlink_to(blob)
    fake = _install(monkeypatch, [("ok", str(broken)), ("ok", str(clean))])
    out = xf.snapshot_download_with_xet_fallback(DL_REPO, token = None)
    assert out == str(clean)
    assert [c.disable_xet for c in fake.calls] == [False, True]


def test_child_broken_snapshot_after_http_raises(monkeypatch, tmp_path):
    """If even the HTTP attempt returns a broken snapshot, fail loudly rather than hand missing files to the load."""
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "model.safetensors").symlink_to(tmp_path / "missing")
    fake = _install(monkeypatch, [("ok", str(broken)), ("ok", str(broken))])
    with pytest.raises(xf.DownloadStallError, match = "incomplete snapshot"):
        xf.snapshot_download_with_xet_fallback(DL_REPO, token = None)
    assert [c.disable_xet for c in fake.calls] == [False, True]


def test_child_weight_incomplete_snapshot_retries_over_http(monkeypatch, tmp_path):
    """A weight-less child result (stale config-only snapshot) is rejected on Xet and retried; a complete second result is accepted."""
    cfg_only = tmp_path / "cfg"
    cfg_only.mkdir()
    (cfg_only / "config.json").write_text("{}")   # no weights
    complete = tmp_path / "complete"
    complete.mkdir()
    blob = tmp_path / "b"
    blob.write_bytes(b"x")
    (complete / "model.safetensors").symlink_to(blob)
    fake = _install(monkeypatch, [("ok", str(cfg_only)), ("ok", str(complete))])
    out = xf.snapshot_download_with_xet_fallback(DL_REPO, token = None)
    assert out == str(complete)
    assert [c.disable_xet for c in fake.calls] == [False, True]


def test_patterned_snapshot_without_weights_is_accepted(monkeypatch, tmp_path):
    """A patterned download that returns only the requested (weightless) files is accepted as-is, not retried for lacking weights."""
    cfg_only = tmp_path / "cfg"
    cfg_only.mkdir()
    (cfg_only / "config.json").write_text("{}")   # exactly what was requested
    fake = _install(monkeypatch, [("ok", str(cfg_only))])
    out = xf.snapshot_download_with_xet_fallback(
        DL_REPO, token = None, allow_patterns = ["config.json"]
    )
    assert out == str(cfg_only) and len(fake.calls) == 1


def test_dataset_snapshot_without_weights_is_accepted(monkeypatch, tmp_path):
    """A dataset snapshot has no weights by nature; its child result is accepted, not retried as 'incomplete'."""
    files = tmp_path / "ds"
    files.mkdir()
    (files / "data.json").write_text("[]")
    fake = _install(monkeypatch, [("ok", str(files))])
    out = xf.snapshot_download_with_xet_fallback(DL_REPO, token = None, repo_type = "dataset")
    assert out == str(files) and len(fake.calls) == 1


def test_model_snapshot_with_weights_excluded_is_accepted(monkeypatch, tmp_path):
    """A model repo whose ignore_patterns drop every weight format yields a weightless snapshot that is accepted, not retried."""
    cfg_only = tmp_path / "cfg"
    cfg_only.mkdir()
    (cfg_only / "config.json").write_text("{}")
    fake = _install(monkeypatch, [("ok", str(cfg_only))])
    out = xf.snapshot_download_with_xet_fallback(
        DL_REPO,
        token = None,
        ignore_patterns = [
            "*.safetensors", "*.bin", "*.h5", "*.msgpack", "*.gguf",
            "*.pt", "*.pth", "*.ckpt", "*.onnx", "*.pdparams", "*.index.json",
        ],
    )
    assert out == str(cfg_only) and len(fake.calls) == 1


def test_request_can_include_weights_unit():
    """Default prefetch ignores (onnx/h5/msgpack/gguf) still count as weight-including; excluding every weight format does not."""
    assert hcs.request_can_include_weights(None, None) is True
    assert hcs.request_can_include_weights(None, ["*.onnx", "*.h5", "*.msgpack", "*.gguf"]) is True
    assert hcs.request_can_include_weights(["config.json"], None) is False
    assert hcs.request_can_include_weights(
        None, ["*.safetensors", "*.bin", "*.h5", "*.msgpack", "*.gguf",
               "*.pt", "*.pth", "*.ckpt", "*.onnx", "*.pdparams", "*.index.json"]
    ) is False


def test_request_can_include_weights_index_json_only():
    """A request matching only shard *index* sidecars reads weightless (the index is JSON, not weights); a real weight pattern does not."""
    assert hcs.request_can_include_weights(["*.json"], None) is False
    assert hcs.request_can_include_weights(["*.index.json"], None) is False
    assert hcs.request_can_include_weights(
        ["model.safetensors.index.json", "pytorch_model.bin.index.json"], None
    ) is False
    assert hcs.request_can_include_weights(["*.safetensors"], None) is True


def test_request_can_include_weights_path_qualified():
    """Path-qualified allow_patterns resolve inside their directory, so a subfolder/checkpoint/shard weight request is not misread as weightless."""
    # Concrete subfolder globs: weights live under the directory.
    assert hcs.request_can_include_weights(["checkpoint-10/*"], None) is True
    assert hcs.request_can_include_weights(["checkpoint-10/*.safetensors"], None) is True
    assert hcs.request_can_include_weights(["models/*.bin"], None) is True
    # A specific (non-first) shard named verbatim.
    assert hcs.request_can_include_weights(["model-00002-of-00005.safetensors"], None) is True
    assert hcs.request_can_include_weights(["checkpoint-10/pytorch_model.bin"], None) is True
    # Globbed parent dir, weight-targeting basename.
    assert hcs.request_can_include_weights(["checkpoint-*/*.safetensors"], None) is True
    # Subfolder requests targeting only non-weight files stay weightless.
    assert hcs.request_can_include_weights(["checkpoint-10/config.json"], None) is False
    assert hcs.request_can_include_weights(["checkpoint-10/*.json"], None) is False
    assert hcs.request_can_include_weights(["checkpoint-*/tokenizer.json"], None) is False
    # The unsloth subfolder warmup shape: "<sub>/*" plus root aux files.
    assert hcs.request_can_include_weights(
        ["checkpoint-10/*", "config.json", "tokenizer.json"], None
    ) is True


def test_request_can_include_weights_path_qualified_custom_globs():
    """A path-qualified custom weight glob (checkpoint-10/lora_*.safetensors) reads weight-including via a concretized self-probe."""
    assert hcs.request_can_include_weights(["checkpoint-10/lora_*.safetensors"], None) is True
    assert hcs.request_can_include_weights(["checkpoint-*/lora_*.bin"], None) is True
    assert hcs.request_can_include_weights(["models/custom_*.pt"], None) is True
    assert hcs.request_can_include_weights(["checkpoint-10/model-[0-9].safetensors"], None) is True
    # A non-weight basename under a subfolder stays weightless.
    assert hcs.request_can_include_weights(["checkpoint-10/tokenizer.json"], None) is False


def test_request_can_include_weights_empty_allow_list(tmp_path):
    """allow_patterns=[] selects nothing (weightless, distinct from None); ignore_patterns=[] ignores nothing (weight-including)."""
    assert hcs.request_can_include_weights([], None) is False
    assert hcs.request_can_include_weights(None, None) is True
    assert hcs.request_can_include_weights(None, []) is True
    assert hcs.request_can_include_weights([], []) is False
    # snapshot_dir_is_complete agrees: allow=[] is a select-nothing request, so an unrelated weight is not complete for it.
    snap = tmp_path / "snap"
    snap.mkdir()
    blob = tmp_path / "blob"
    blob.write_bytes(b"x")
    (snap / "model.safetensors").symlink_to(blob)
    assert hcs.snapshot_dir_is_complete(snap, allow_patterns = []) is False
    assert hcs.snapshot_dir_is_complete(snap, allow_patterns = None) is True


def test_request_can_include_weights_string_form():
    """A bare-string allow/ignore pattern is treated as one pattern, not iterated char by char."""
    assert hcs.request_can_include_weights("checkpoint-10/*", None) is True
    assert hcs.request_can_include_weights("*.safetensors", None) is True
    assert hcs.request_can_include_weights("config.json", None) is False
    assert hcs.request_can_include_weights(None, "*.safetensors") is True   # ignore as str
    # A string ignore that drops every weight format leaves the request weightless.
    assert hcs.request_can_include_weights(
        "config.json", ["*.safetensors", "*.bin", "*.pt", "*.pth", "*.gguf",
                        "*.h5", "*.msgpack", "*.ckpt", "*.onnx", "*.pdparams"]
    ) is False


def test_prepare_for_http_spares_active_sibling_partial(hf_cache):
    """The HTTP-prep purge removes only stale (old-mtime) partials, so a concurrent sibling's active .incomplete keeps writing."""
    blobs = _blobs_dir(hf_cache, DL_REPO)
    stale = blobs / "stalled.incomplete"
    stale.write_bytes(b"\0" * 16)
    active = blobs / "sibling.incomplete"
    active.write_bytes(b"\0" * 16)
    # Age the stalled partial past the grace; leave the sibling current.
    old = time.time() - 600
    os.utime(stale, (old, old))
    _REAL_DEFAULT_PREPARE("model", DL_REPO, cache_dir = str(hf_cache))
    assert not stale.exists(), "stale partial should be purged for the HTTP resume"
    assert active.exists(), "an actively-written sibling partial must be preserved"


def test_snapshot_stall_then_http(monkeypatch):
    prepared = []
    monkeypatch.setattr(xf, "_default_prepare_for_http", lambda rt, rid, cache_dir = None, **k: prepared.append((rt, rid)))
    fake = _install(monkeypatch, [("stall", None), ("ok", "/cache/snap-dir")])
    out = xf.snapshot_download_with_xet_fallback(DL_REPO, token = None)
    assert out == "/cache/snap-dir"
    assert [c.kind for c in fake.calls] == ["snapshot", "snapshot"]
    assert [c.disable_xet for c in fake.calls] == [False, True]
    assert prepared == [("model", DL_REPO)]


def test_force_download_skips_fast_path_and_threads(monkeypatch):
    """force_download=True bypasses the warm-cache short-circuit and threads force_download into the download params."""
    def _snap(*a, **k):
        pytest.fail("force_download must not take the local_files_only fast path")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _snap)
    fake = _install(monkeypatch, [("ok", "/cache/snap-dir")])
    out = xf.snapshot_download_with_xet_fallback(DL_REPO, token = None, force_download = True)
    assert out == "/cache/snap-dir"
    assert len(fake.calls) == 1 and fake.calls[0].force_download is True


def test_force_download_file_skips_cache_probe(monkeypatch, tmp_path):
    """The single-file path also skips the cached-blob short-circuit and threads force_download through."""
    cached = tmp_path / "cached.gguf"
    cached.write_bytes(b"\0" * 8)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: str(cached))
    fake = _install(monkeypatch, [("ok", "/cache/x")])
    out = xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None, force_download = True)
    assert out == "/cache/x"
    assert len(fake.calls) == 1 and fake.calls[0].force_download is True


# Precondition: HF_HUB_DISABLE_XET is read at import time, so assert its effect in a
# FRESH interpreter (huggingface/huggingface_hub#3266 once ignored it).
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


# Exported Xet knobs + child-leak safety + malformed-index resilience.
def test_xet_availability_and_disable_helpers(monkeypatch):
    """child_should_disable_xet reads the per-worker flag; xet_force_disabled honors every env knob; is_hf_xet_available probes importability."""
    assert xf.child_should_disable_xet({"disable_xet": True}) is True
    assert xf.child_should_disable_xet({"disable_xet": False}) is False
    assert xf.child_should_disable_xet({}) is False

    for knob in ("UNSLOTH_DISABLE_XET", "UNSLOTH_STABLE_DOWNLOADS", "HF_HUB_DISABLE_XET"):
        for k in ("UNSLOTH_DISABLE_XET", "UNSLOTH_STABLE_DOWNLOADS", "HF_HUB_DISABLE_XET"):
            monkeypatch.delenv(k, raising = False)
        assert xf.xet_force_disabled() is False
        monkeypatch.setenv(knob, "1")
        assert xf.xet_force_disabled() is True, knob

    monkeypatch.setattr(xf.importlib.util, "find_spec", lambda name: object())
    assert xf.is_hf_xet_available() is True
    monkeypatch.setattr(xf.importlib.util, "find_spec", lambda name: None)
    assert xf.is_hf_xet_available() is False

    def _raise(name):
        raise ImportError("boom")

    monkeypatch.setattr(xf.importlib.util, "find_spec", _raise)
    assert xf.is_hf_xet_available() is False  # a probe exception -> unavailable


def test_run_attempt_terminates_child_if_watchdog_start_raises(monkeypatch):
    """If start_watchdog raises after the child spawned, the child is still reaped (no leak) and the error propagates."""
    rec = {"terminated": False}

    class _AliveProc:
        def __init__(self):
            self.pid = None  # None -> uses terminate(), not killpg
            self.exitcode = None
            self._alive = True

        def start(self):
            pass

        def is_alive(self):
            return self._alive

        def terminate(self):
            rec["terminated"] = True
            self._alive = False

        def kill(self):
            rec["terminated"] = True
            self._alive = False

        def join(self, timeout = None):
            pass

    class _Ctx:
        def Process(self, *, target = None, kwargs = None, daemon = None):
            return _AliveProc()

        def Queue(self):
            return _FakeQueue({"ok": True, "path": "/cache/x"})

    monkeypatch.setattr(xf, "_CTX", _Ctx())

    def _boom(*a, **k):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(xf, "start_watchdog", _boom)

    with pytest.raises(RuntimeError, match = "can't start new thread"):
        xf._run_download_attempt(
            DL_REPO, kind = "snapshot", params = {"repo_id": DL_REPO}, token = None,
            repo_type = "model", disable_xet = False, cancel_event = None,
            stall_timeout = 0.2, interval = 0.05, grace_period = 0.05, on_status = None,
        )
    assert rec["terminated"] is True  # child reaped despite the watchdog-start failure


# Codex review round: scoped completeness, weightless named files, type preservation.
def test_requested_named_files_present_exact_request(tmp_path):
    """An EXACT-named weightless request requires its named file on disk; a glob list or no allow_patterns is best-effort."""
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}")
    assert hcs.requested_named_files_present(snap, allow_patterns = ["tokenizer.json"]) is False
    (snap / "tokenizer.json").write_text("{}")
    assert hcs.requested_named_files_present(snap, allow_patterns = ["tokenizer.json"]) is True
    # A glob list is best-effort: a missing optional file does not fail it.
    assert hcs.requested_named_files_present(snap, allow_patterns = ["tokenizer*", "vocab.txt"]) is True
    # No allow_patterns -> trivially satisfied.
    assert hcs.requested_named_files_present(snap) is True
    # An ignore-filtered name is not requested, so its absence does not fail.
    assert hcs.requested_named_files_present(
        snap, allow_patterns = ["tokenizer.json", "absent.json"], ignore_patterns = ["absent.json"]
    ) is True


def test_deterministic_oserror_type_preserved(monkeypatch):
    """A deterministic disk-full OSError is re-raised as OSError across the spawn boundary, not flattened to RuntimeError."""
    fake = _install(monkeypatch, [("error", "OSError: [Errno 28] No space left on device")])
    with pytest.raises(OSError, match = "No space left"):
        xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None)
    assert len(fake.calls) == 1, "a deterministic error must not trigger an HTTP fallback"


def test_unknown_error_falls_back_to_runtimeerror(monkeypatch):
    """An unrecognized error class name surfaces as RuntimeError without a fallback; only known Hub/OS types are reconstructed."""
    fake = _install(monkeypatch, [("error", "SomeWeirdError: kaboom")])
    with pytest.raises(RuntimeError, match = "kaboom"):
        xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None)
    assert len(fake.calls) == 1


def test_resolve_exception_class_maps_known_names():
    """The map resolves known deterministic Hub error names + OSError, and returns None for an unknown name."""
    assert xf._resolve_exception_class("OSError") is OSError
    cls = xf._resolve_exception_class("RepositoryNotFoundError")
    assert cls is not None and issubclass(cls, BaseException)
    assert xf._resolve_exception_class("NotARealErrorType") is None


def test_error_type_preserved_when_constructor_needs_kwarg(monkeypatch):
    """A Hub error whose constructor rejects a lone positional string is still re-raised with its type via an __init__-bypassing reconstruction."""
    class PickyHubError(Exception):
        def __init__(self, message, *, response):  # response required + keyword-only
            super().__init__(message)
            self.response = response

    monkeypatch.setattr(
        xf, "_resolve_exception_class",
        lambda name: PickyHubError if name == "PickyHubError" else None,
    )
    fake = _install(monkeypatch, [("error", "PickyHubError: kaboom")])
    with pytest.raises(PickyHubError, match = "kaboom"):
        xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None)
    assert len(fake.calls) == 1, "a deterministic error must not trigger an HTTP fallback"


def test_instantiate_preserving_type_paths():
    """Layered reconstruction: a normal constructor is used when it accepts a string, else the __new__ bypass; both carry the message."""
    class Plain(Exception):
        pass

    class Picky(Exception):
        def __init__(self, message, *, response):
            super().__init__(message)

    for cls in (Plain, Picky):
        exc = xf._instantiate_preserving_type(cls, "the message")
        assert isinstance(exc, cls)
        assert "the message" in str(exc)


# Codex round: dir/ wildcard, logical-weight grouping post-download, errno preservation.
def test_parse_errno():
    assert xf._parse_errno("OSError: [Errno 28] No space left on device") == 28
    assert xf._parse_errno("OSError: [Errno 122] Disk quota exceeded") == 122
    assert xf._parse_errno("OSError: some message with no errno") is None


def test_oserror_errno_preserved(monkeypatch):
    """A disk-full child OSError keeps its errno (ENOSPC) across the spawn boundary, not errno=None."""
    import errno as _errno

    fake = _install(monkeypatch, [("error", "OSError: [Errno 28] No space left on device")])
    with pytest.raises(OSError) as excinfo:
        xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None)
    assert excinfo.value.errno == _errno.ENOSPC
    assert len(fake.calls) == 1, "a deterministic error must not trigger an HTTP fallback"


def test_oserror_subclass_errno_preserved(monkeypatch):
    """An OSError subclass (PermissionError) keeps both its type and errno across the spawn boundary; the message is not double-prefixed."""
    import errno as _errno

    fake = _install(monkeypatch, [("error", "PermissionError: [Errno 13] Permission denied")])
    with pytest.raises(PermissionError) as excinfo:
        xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None)
    assert excinfo.value.errno == _errno.EACCES
    assert "[Errno 13] [Errno 13]" not in str(excinfo.value)
    assert len(fake.calls) == 1, "a deterministic error must not trigger an HTTP fallback"


def test_raise_child_error_errno_only_for_builtin_oserror():
    """errno is preserved only for a BUILTIN OSError type; a non-builtin OSError subclass (e.g. HfHubHTTPError) with a bracketed [Errno N] gets no spurious errno."""
    # Builtin OSError subclass -> errno preserved.
    with pytest.raises(FileNotFoundError) as excinfo:
        xf._raise_child_error("FileNotFoundError: [Errno 2] No such file or directory")
    assert excinfo.value.errno == 2

    # Non-builtin OSError subclass whose message merely contains a bracketed [Errno N].
    class _FakeHubHTTPError(OSError):
        def __init__(self, message):  # single-arg, like hf_hub's error types
            super().__init__(message)

    orig = xf._resolve_exception_class
    try:
        xf._resolve_exception_class = (
            lambda name: _FakeHubHTTPError if name == "HfHubHTTPError" else orig(name)
        )
        with pytest.raises(_FakeHubHTTPError) as excinfo2:
            xf._raise_child_error("HfHubHTTPError: 500 Server Error [Errno 111] for url https://x")
        assert excinfo2.value.errno is None
    finally:
        xf._resolve_exception_class = orig


# Spawn-safety regressions: failed-spawn queue cleanup + disable-Xet env-race lock.
def test_failed_spawn_closes_result_queue(monkeypatch):
    """R2-2: if proc.start() raises, the result_queue's pipe fds (allocated before the spawn) are closed, not leaked."""
    closed = {"cancel_join": False, "close": False}

    class _FakeQueue:
        def cancel_join_thread(self):
            closed["cancel_join"] = True

        def close(self):
            closed["close"] = True

    class _FakeProc:
        def __init__(self, *a, **k):
            self.pid = None

        def start(self):
            raise OSError(errno.EAGAIN, "Resource temporarily unavailable")

    class _FakeCtx:
        def Queue(self):
            return _FakeQueue()

        def Process(self, *a, **k):
            return _FakeProc()

    monkeypatch.setattr(xf, "_CTX", _FakeCtx())
    with pytest.raises(OSError):
        xf._run_download_attempt(
            "owner/repo",
            kind = "snapshot",
            params = {},
            token = None,
            repo_type = "model",
            disable_xet = False,
            cancel_event = None,
            stall_timeout = 1.0,
            interval = 0.1,
            grace_period = 0.1,
            on_status = None,
        )
    assert closed["close"] is True
    assert closed["cancel_join"] is True


def test_disable_xet_read_under_spawn_lock(monkeypatch):
    """R2-1: xet_force_disabled() is read while holding _SPAWN_ENV_LOCK, so a concurrent spawn's transient HF_HUB_DISABLE_XET=1 is not observed."""
    seen = {}
    real = xf.xet_force_disabled

    def _spy():
        # A non-reentrant Lock's non-blocking acquire fails iff the read is inside `with _SPAWN_ENV_LOCK:`.
        got = xf._SPAWN_ENV_LOCK.acquire(blocking = False)
        if got:
            xf._SPAWN_ENV_LOCK.release()
        seen["held"] = not got
        return real()

    monkeypatch.setattr(xf, "xet_force_disabled", _spy)
    monkeypatch.setattr(xf, "_run_download_attempt", lambda *a, **k: ("ok", "/tmp/warm"))
    out = xf._download_with_xet_fallback(
        repo_id = "owner/repo",
        label = "test",
        kind = "snapshot",
        params = {},
        token = None,
        repo_type = "model",
        cancel_event = None,
        stall_timeout = 1.0,
        interval = 0.1,
        grace_period = 0.1,
        on_status = None,
        prepare_for_http_fn = None,
    )
    assert out == "/tmp/warm"
    assert seen.get("held") is True


# Conservative fast-path gate + pre/post-download acceptance split. The gate fast-paths
# ONLY the unambiguous canonical model cache, deferring everything else to the watched
# child. Pre-download (skip the child?) and post-download (accept the result?) are
# deliberately asymmetric: strict pre, lenient post.
def _mk_snapshot(tmp_path, name):
    blob = tmp_path / "_blob"
    if not blob.exists():
        blob.write_bytes(b"w")
    snap = tmp_path / name
    snap.mkdir()
    return snap, blob


def test_gate_fast_paths_canonical_single_file(tmp_path):
    """A complete, unpatterned single-file model cache is fast-path eligible."""
    snap, blob = _mk_snapshot(tmp_path, "single")
    (snap / "model.safetensors").symlink_to(blob)
    (snap / "config.json").write_text("{}")
    assert hcs.snapshot_dir_is_complete(snap) is True


def test_gate_fast_paths_canonical_sharded_with_index(tmp_path):
    """A complete sharded model with its index is fast-path eligible; without the index or with a listed shard missing, it is not."""
    snap, blob = _mk_snapshot(tmp_path, "shard")
    (snap / "model-00001-of-00002.safetensors").symlink_to(blob)
    (snap / "model-00002-of-00002.safetensors").symlink_to(blob)
    assert hcs.snapshot_dir_is_complete(snap) is False          # numbered shards, no index
    (snap / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors",
                                   "b": "model-00002-of-00002.safetensors"}}))
    assert hcs.snapshot_dir_is_complete(snap) is True
    snap2, _ = _mk_snapshot(tmp_path, "shard2")
    (snap2 / "model-00001-of-00002.safetensors").symlink_to(blob)  # shard 2 absent
    (snap2 / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors",
                                   "b": "model-00002-of-00002.safetensors"}}))
    assert hcs.snapshot_dir_is_complete(snap2) is False


def test_shard_index_with_non_string_value_is_incomplete(tmp_path):
    """A shard index mapping a tensor to a non-string value (null) is incomplete: fail closed and defer to the child."""
    snap, blob = _mk_snapshot(tmp_path, "badindex")
    (snap / "model-00001-of-00002.safetensors").symlink_to(blob)
    (snap / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors", "b": None}}))
    assert hcs._weight_shard_index_complete(snap / "model.safetensors.index.json") is False
    assert hcs.snapshot_dir_is_complete(snap) is False


def test_gate_defers_incomplete_preferred_index_masked_by_complete_bin(tmp_path):
    """An incomplete safetensors index (probed before the bin) is not masked by a complete bin; the gate defers unless safetensors is ignored."""
    snap, blob = _mk_snapshot(tmp_path, "prefidx")
    (snap / "model-00001-of-00002.safetensors").symlink_to(blob)  # ST shard 2 absent -> incomplete index
    (snap / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors",
                                   "b": "model-00002-of-00002.safetensors"}}))
    (snap / "pytorch_model.bin").symlink_to(blob)  # complete bin co-resident
    assert hcs.snapshot_dir_is_complete(snap) is False  # load prefers the incomplete safetensors
    # safetensors ignored -> the load reads the complete bin -> eligible.
    assert hcs.snapshot_dir_is_complete(snap, ignore_patterns = ["*.safetensors"]) is True
    # A complete safetensors index alongside the bin is eligible.
    (snap / "model-00002-of-00002.safetensors").symlink_to(blob)
    assert hcs.snapshot_dir_is_complete(snap) is True


def test_gate_rejects_sharded_adapter_only_root_cache(tmp_path):
    """A complete sharded adapter at root is not a canonical base model, so an adapter-only cache defers to the child."""
    assert hcs._is_canonical_weight_shard_index("adapter_model.safetensors.index.json") is False
    assert hcs._is_canonical_weight_shard_index("model.safetensors.index.json") is True
    assert hcs._is_canonical_weight_shard_index("pytorch_model.bin.index.json") is True
    snap, blob = _mk_snapshot(tmp_path, "adapteronly")
    (snap / "config.json").write_text("{}")
    (snap / "adapter_model-00001-of-00002.safetensors").symlink_to(blob)
    (snap / "adapter_model-00002-of-00002.safetensors").symlink_to(blob)
    (snap / "adapter_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "adapter_model-00001-of-00002.safetensors",
                                   "b": "adapter_model-00002-of-00002.safetensors"}}))
    assert hcs.snapshot_dir_is_complete(snap) is False
    assert xf._cache_can_skip_download(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None) is False


def test_gate_rejects_config_only(tmp_path):
    snap, _ = _mk_snapshot(tmp_path, "cfg")
    (snap / "config.json").write_text("{}")
    assert hcs.snapshot_dir_is_complete(snap) is False


def test_gate_rejects_diffusers_marker(tmp_path):
    """A diffusers pipeline (root model_index.json) is never fast-pathed, even with a root-level weight present."""
    snap, blob = _mk_snapshot(tmp_path, "diff")
    (snap / "model_index.json").write_text("{}")
    (snap / "model.safetensors").symlink_to(blob)
    assert hcs.snapshot_dir_is_complete(snap) is False


def test_gate_rejects_any_allow_pattern(tmp_path):
    """Any allow_patterns makes the request non-trivial, so no fast-path."""
    snap, blob = _mk_snapshot(tmp_path, "pat")
    (snap / "model.safetensors").symlink_to(blob)
    assert hcs.snapshot_dir_is_complete(snap, allow_patterns = ["*.safetensors"]) is False
    assert hcs.snapshot_dir_is_complete(snap, allow_patterns = ["model.safetensors"]) is False


def test_gate_eligible_under_ignore_patterns(tmp_path):
    """allow=None with any ignore patterns stays eligible: ignores that drop other formats cannot make an incomplete cache read complete."""
    snap, blob = _mk_snapshot(tmp_path, "ign")
    (snap / "model.safetensors").symlink_to(blob)
    assert hcs.snapshot_dir_is_complete(snap, ignore_patterns = ["*/*.safetensors", "*/*.bin"]) is True
    assert hcs.snapshot_dir_is_complete(snap, ignore_patterns = ["*.onnx"]) is True
    # The real unsloth bare-from_pretrained ignore list; the warm root model.safetensors must still fast-path.
    unsloth_ignore = [
        "*.onnx", "*.h5", "*.msgpack", "*.tflite", "*.mlmodel", "*.gguf", "*.pt", "*.pth",
        "*.ckpt", "optimizer.*", "scheduler.*", "rng_state*", "trainer_state.json",
        "events.out.tfevents*", "*.bin",
        "*/*.safetensors", "*/*.bin", "*/*.h5", "*/*.msgpack", "*/*.pt", "*/*.pth",
    ]
    assert hcs.snapshot_dir_is_complete(snap, ignore_patterns = unsloth_ignore) is True


def test_gate_rejects_broken_symlink(tmp_path):
    snap, _ = _mk_snapshot(tmp_path, "broken")
    (snap / "model.safetensors").symlink_to(tmp_path / "_missing")
    assert hcs.snapshot_dir_is_complete(snap) is False


def test_request_can_include_weights_trim_semantics():
    r = hcs.request_can_include_weights
    assert r(None, None) is True                              # bare unpatterned
    assert r(None, ["*/*.safetensors", "*/*.bin"]) is True    # subdir ignore (common bare)
    assert r(["*.safetensors"], None) is True                 # weight glob
    assert r(["model.fp16.safetensors"], None) is True        # variant exact weight
    assert r(["unet/*"], None) is True                        # subfolder weight glob
    assert r(["model.gguf"], None) is True                    # gguf is a weight
    assert r(["config.json", "tokenizer.json", "*.py"], None) is False  # tokenizer-only
    assert r(["adapter_model*", "adapter_config.json"], None) is True   # adapter
    assert r([], None) is False                               # empty allow selects nothing


def test_request_can_include_weights_partial_ignore_strip_is_weight_bearing():
    """An ignore-only request is weightless only when it strips EVERY weight format; a partial strip stays weight-bearing."""
    r = hcs.request_can_include_weights
    assert r(None, ["model.safetensors", "pytorch_model.bin"]) is True  # variant / other-format survives
    assert r(None, ["*.safetensors", "*.bin"]) is True                  # .pt / .gguf / .pth / ... survive
    # Only stripping EVERY weight format is weightless.
    all_formats = ["*.safetensors", "*.bin", "*.pt", "*.pth", "*.gguf", "*.ckpt",
                   "*.onnx", "*.msgpack", "*.h5", "*.pdparams"]
    assert r(None, all_formats) is False


def test_pre_download_skips_complete_model(tmp_path):
    snap, blob = _mk_snapshot(tmp_path, "m")
    (snap / "model.safetensors").symlink_to(blob)
    assert xf._cache_can_skip_download(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None) is True


def test_pre_download_defers_variant_on_canonical_cache(tmp_path):
    """A variant load reads model.<variant>.safetensors, so a canonical-only cache does not fast-path a variant request (but does with no variant)."""
    snap, blob = _mk_snapshot(tmp_path, "var")
    (snap / "model.safetensors").symlink_to(blob)
    assert xf._cache_can_skip_download(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None) is True
    assert xf._cache_can_skip_download(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None,
        variant = "fp16") is False


def test_pre_download_defers_bin_only_when_safetensors_preferred(tmp_path):
    """A default load probes model.safetensors before the bin, so the strict pre-gate defers a bin-only cache; the lenient post path accepts a finished bin-only download."""
    snap, blob = _mk_snapshot(tmp_path, "binonly")
    (snap / "config.json").write_text("{}")
    (snap / "pytorch_model.bin").symlink_to(blob)
    # PRE: safetensors preferred (not ignored) + bin-only -> defer.
    assert xf._cache_can_skip_download(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None) is False
    # PRE: safetensors ignored -> the bin cache fast-paths.
    assert xf._cache_can_skip_download(
        snap, repo_type = "model", allow_patterns = None,
        ignore_patterns = ["*.safetensors", "*.safetensors.index.json"]) is True
    # PRE: safetensors present -> fast-path.
    (snap / "model.safetensors").symlink_to(blob)
    assert xf._cache_can_skip_download(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None) is True
    # POST is lenient: a finished bin-only download is accepted.
    snap2, blob2 = _mk_snapshot(tmp_path, "binonly_post")
    (snap2 / "config.json").write_text("{}")
    (snap2 / "pytorch_model.bin").symlink_to(blob2)
    assert xf._download_result_usable(
        snap2, repo_type = "model", allow_patterns = None, ignore_patterns = None) is True
    # POST: a sharded bin-only repo is likewise accepted.
    snap3, blob3 = _mk_snapshot(tmp_path, "binonly_sharded_post")
    (snap3 / "pytorch_model-00001-of-00002.bin").symlink_to(blob3)
    (snap3 / "pytorch_model-00002-of-00002.bin").symlink_to(blob3)
    (snap3 / "pytorch_model.bin.index.json").write_text(json.dumps(
        {"weight_map": {"a": "pytorch_model-00001-of-00002.bin",
                        "b": "pytorch_model-00002-of-00002.bin"}}))
    assert xf._download_result_usable(
        snap3, repo_type = "model", allow_patterns = None, ignore_patterns = None) is True


def test_pre_download_does_not_skip_diffusers_but_post_accepts(tmp_path):
    """Pre/post asymmetry: a diffusers warm is not fast-pathed, but the same complete result is accepted post-download."""
    snap, blob = _mk_snapshot(tmp_path, "diff")
    (snap / "model_index.json").write_text("{}")
    comp = snap / "unet"
    comp.mkdir()
    (comp / "diffusion_pytorch_model.safetensors").symlink_to(blob)
    assert xf._cache_can_skip_download(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None) is False
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None) is True


def test_post_download_rejects_config_only_model(tmp_path):
    """A model warm returning no weight (stale config-only snapshot) is rejected post-download and retried."""
    snap, _ = _mk_snapshot(tmp_path, "cfg")
    (snap / "config.json").write_text("{}")
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None) is False


def test_post_download_rejects_ignored_only_format(tmp_path):
    """A safetensors load (ignore=['*.bin']) whose result kept only the ignored .bin is rejected (the weight check applies the ignore filter)."""
    snap, blob = _mk_snapshot(tmp_path, "ignfmt")
    (snap / "pytorch_model.bin").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = ["*.bin"]) is False
    # The requested safetensors present -> accepted.
    (snap / "model.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = ["*.bin"]) is True


def test_post_download_rejects_canonical_only_for_variant(tmp_path):
    """A variant load whose result kept only the canonical (non-variant) weight is rejected; a present variant weight is accepted."""
    snap, blob = _mk_snapshot(tmp_path, "varpost")
    (snap / "model.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None,
        variant = "fp16") is False
    (snap / "model.fp16.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None,
        variant = "fp16") is True
    # A complete sharded variant set (shards + index) is accepted.
    snap2, blob2 = _mk_snapshot(tmp_path, "varshard")
    (snap2 / "model.fp16-00001-of-00002.safetensors").symlink_to(blob2)
    (snap2 / "model.fp16-00002-of-00002.safetensors").symlink_to(blob2)
    (snap2 / "model.safetensors.index.fp16.json").write_text(json.dumps(
        {"weight_map": {"a": "model.fp16-00001-of-00002.safetensors",
                        "b": "model.fp16-00002-of-00002.safetensors"}}))
    assert xf._download_result_usable(
        snap2, repo_type = "model", allow_patterns = None, ignore_patterns = None,
        variant = "fp16") is True


def test_post_download_rejects_patterned_canonical_only_for_variant(tmp_path):
    """The variant check applies to the patterned branch too: a subfolder variant request kept only the canonical weight is rejected."""
    snap, blob = _mk_snapshot(tmp_path, "subvar")
    sub = snap / "weights"
    sub.mkdir()
    (sub / "model.safetensors").symlink_to(blob)  # canonical only, no variant
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = ["weights/*"], ignore_patterns = None,
        variant = "fp16") is False
    # The in-scope variant weight present -> accepted.
    (sub / "model.fp16.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = ["weights/*"], ignore_patterns = None,
        variant = "fp16") is True
    # A complete sharded in-scope variant weight (dash infix + variant index) is accepted.
    snap2, blob2 = _mk_snapshot(tmp_path, "subvarshard")
    sub2 = snap2 / "weights"
    sub2.mkdir()
    (sub2 / "model.fp16-00001-of-00002.safetensors").symlink_to(blob2)
    (sub2 / "model.fp16-00002-of-00002.safetensors").symlink_to(blob2)
    (sub2 / "model.safetensors.index.fp16.json").write_text(json.dumps(
        {"weight_map": {"a": "model.fp16-00001-of-00002.safetensors",
                        "b": "model.fp16-00002-of-00002.safetensors"}}))
    assert xf._download_result_usable(
        snap2, repo_type = "model", allow_patterns = ["weights/*"], ignore_patterns = None,
        variant = "fp16") is True
    # A lone variant shard with no index is an incomplete set -> rejected.
    snap2b, blob2b = _mk_snapshot(tmp_path, "subvarshard_lone")
    (snap2b / "weights").mkdir()
    (snap2b / "weights" / "model.fp16-00001-of-00002.safetensors").symlink_to(blob2b)
    assert xf._download_result_usable(
        snap2b, repo_type = "model", allow_patterns = ["weights/*"], ignore_patterns = None,
        variant = "fp16") is False
    # An out-of-scope variant weight does not satisfy an in-scope variant request.
    snap3, blob3 = _mk_snapshot(tmp_path, "subvaroos")
    (snap3 / "model.fp16.safetensors").symlink_to(blob3)  # at root, but request scopes to weights/
    (snap3 / "weights").mkdir()
    (snap3 / "weights" / "model.safetensors").symlink_to(blob3)
    assert xf._download_result_usable(
        snap3, repo_type = "model", allow_patterns = ["weights/*"], ignore_patterns = None,
        variant = "fp16") is False


def test_post_download_rejects_variant_only_diffusers_for_plain_load(tmp_path):
    """A plain diffusers warm (variant=None) whose result kept only variant component weights is rejected; complete plain / both-format pipelines are accepted."""
    def _mi(**comps):
        data = {"_class_name": "StableDiffusionPipeline", "_diffusers_version": "0.21.0"}
        data.update(comps)
        return json.dumps(data)

    snap, blob = _mk_snapshot(tmp_path, "plainvaronly")
    (snap / "model_index.json").write_text(
        _mi(unet = ["diffusers", "UNet2DConditionModel"], vae = ["diffusers", "AutoencoderKL"]))
    for comp in ("unet", "vae"):
        (snap / comp).mkdir()
        (snap / comp / "diffusion_pytorch_model.fp16.safetensors").symlink_to(blob)
    # plain load: variant-only components do not satisfy it -> retry.
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None, variant = None) is False
    # the same cache is a complete fp16 warm -> the variant load accepts it.
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None,
        variant = "fp16") is True
    # a complete plain pipeline (non-variant component weights) is accepted.
    snap2, blob2 = _mk_snapshot(tmp_path, "plaincomplete")
    (snap2 / "model_index.json").write_text(
        _mi(unet = ["diffusers", "UNet2DConditionModel"], vae = ["diffusers", "AutoencoderKL"]))
    for comp in ("unet", "vae"):
        (snap2 / comp).mkdir()
        (snap2 / comp / "diffusion_pytorch_model.safetensors").symlink_to(blob2)
    assert xf._download_result_usable(
        snap2, repo_type = "model", allow_patterns = None, ignore_patterns = None, variant = None) is True
    # a pipeline shipping both plain + fp16 in a component is accepted for a plain load.
    (snap2 / "unet" / "diffusion_pytorch_model.fp16.safetensors").symlink_to(blob2)
    assert xf._download_result_usable(
        snap2, repo_type = "model", allow_patterns = None, ignore_patterns = None, variant = None) is True


def test_post_download_rejects_incomplete_sharded_glob(tmp_path):
    """A globbed weight request (allow=['*.safetensors']) with a shard index missing a shard is rejected (globs get the same shard-completeness check)."""
    snap, blob = _mk_snapshot(tmp_path, "shardglob")
    (snap / "model-00001-of-00002.safetensors").symlink_to(blob)
    (snap / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors",
                                   "b": "model-00002-of-00002.safetensors"}}))
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = ["*.safetensors"], ignore_patterns = None) is False
    # The missing shard present -> complete -> accepted.
    (snap / "model-00002-of-00002.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = ["*.safetensors"], ignore_patterns = None) is True


def test_post_download_accepts_patterned_with_coresident_partial_canonical_shards(tmp_path):
    """A complete patterned download is accepted even with an unrelated partial canonical base shard set co-resident at root."""
    def _partial_base_shards(snap, blob):
        (snap / "model-00001-of-00002.safetensors").symlink_to(blob)             # shard 1 present
        (snap / "model-00002-of-00002.safetensors").symlink_to(snap / "MISSING")  # dangling shard 2
        (snap / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {"a": "model-00001-of-00002.safetensors",
                            "b": "model-00002-of-00002.safetensors"}}))
    # Adapter request completes; co-resident partial base shards must not reject it.
    snap, blob = _mk_snapshot(tmp_path, "adapter_coresident")
    (snap / "adapter_model.safetensors").symlink_to(blob)
    (snap / "adapter_config.json").write_text("{}")
    _partial_base_shards(snap, blob)
    assert xf._download_result_usable(
        snap, repo_type = "model",
        allow_patterns = ["adapter_model.safetensors", "adapter_config.json", "*.json"],
        ignore_patterns = None) is True
    # gguf request completes; same co-resident partial base shards.
    snap2, blob2 = _mk_snapshot(tmp_path, "gguf_coresident")
    (snap2 / "model.Q4_K_M.gguf").symlink_to(blob2)
    (snap2 / "config.json").write_text("{}")
    _partial_base_shards(snap2, blob2)
    assert xf._download_result_usable(
        snap2, repo_type = "model",
        allow_patterns = ["model.Q4_K_M.gguf", "config.json", "*.json"],
        ignore_patterns = None) is True
    # A globbed weight request that DOES select canonical root shards still gets the gate.
    snap3, blob3 = _mk_snapshot(tmp_path, "glob_still_gated")
    _partial_base_shards(snap3, blob3)
    assert xf._download_result_usable(
        snap3, repo_type = "model", allow_patterns = ["*.safetensors"], ignore_patterns = None) is False


def test_post_download_rejects_incomplete_ignored_format_shards(tmp_path):
    """A load ignoring safetensors (reads .bin) with a complete ST set but incomplete .bin set is rejected (the shard gate applies the ignore filter)."""
    snap, blob = _mk_snapshot(tmp_path, "ignored_format_shards")
    (snap / "model-00001-of-00002.safetensors").symlink_to(blob)
    (snap / "model-00002-of-00002.safetensors").symlink_to(blob)
    (snap / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a": "model-00001-of-00002.safetensors",
                        "b": "model-00002-of-00002.safetensors"}}))
    (snap / "pytorch_model-00001-of-00002.bin").symlink_to(blob)  # bin shard 1 only, no index/shard 2
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None,
        ignore_patterns = ["*.safetensors"]) is False
    # Ignoring the .bin instead (load reads the complete safetensors) -> accepted.
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = ["*.bin"]) is True


def test_post_download_rejects_incomplete_variant_shards(tmp_path):
    """A variant load with a variant shard index missing a listed shard is rejected; a complete set and a single-file variant are accepted."""
    snap, blob = _mk_snapshot(tmp_path, "variant_incomplete")
    (snap / "model.fp16-00001-of-00002.safetensors").symlink_to(blob)  # shard 1; shard 2 absent
    (snap / "model.safetensors.index.fp16.json").write_text(json.dumps(
        {"weight_map": {"a": "model.fp16-00001-of-00002.safetensors",
                        "b": "model.fp16-00002-of-00002.safetensors"}}))
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None,
        variant = "fp16") is False
    # A lone variant shard with no index is also incomplete.
    snap_noidx, blob_ni = _mk_snapshot(tmp_path, "variant_no_index")
    (snap_noidx / "model.fp16-00001-of-00002.safetensors").symlink_to(blob_ni)
    assert xf._download_result_usable(
        snap_noidx, repo_type = "model", allow_patterns = None, ignore_patterns = None,
        variant = "fp16") is False
    # The missing variant shard present -> complete set -> accepted.
    (snap / "model.fp16-00002-of-00002.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None,
        variant = "fp16") is True
    # A single-file variant (no index) is accepted.
    snap2, blob2 = _mk_snapshot(tmp_path, "variant_single")
    (snap2 / "model.fp16.safetensors").symlink_to(blob2)
    assert xf._download_result_usable(
        snap2, repo_type = "model", allow_patterns = None, ignore_patterns = None,
        variant = "fp16") is True


def test_post_download_accepts_exact_named_shard_subset(tmp_path):
    """An exact-named shard request is accepted once that file is present (the whole-checkpoint gate applies only to globbed warms); an absent named shard is rejected."""
    snap, blob = _mk_snapshot(tmp_path, "exact_shard_present")
    (snap / "model-00001-of-00002.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model",
        allow_patterns = ["model-00001-of-00002.safetensors"], ignore_patterns = None) is True
    # The exact-named shard absent -> rejected.
    snap2, _ = _mk_snapshot(tmp_path, "exact_shard_absent")
    (snap2 / "config.json").write_text("{}")
    assert xf._download_result_usable(
        snap2, repo_type = "model",
        allow_patterns = ["model-00001-of-00002.safetensors"], ignore_patterns = None) is False


def test_post_download_accepts_from_tf_flax_weights(tmp_path):
    """A from_tf/from_flax load (both PyTorch formats ignored) reading tf_model.h5 / flax_model.msgpack is accepted when complete."""
    ig = ["*.safetensors", "*.safetensors.index.json", "*.bin", "*.bin.index.json"]
    for wt in ("tf_model.h5", "flax_model.msgpack"):
        snap, blob = _mk_snapshot(tmp_path, f"tf_{wt}")
        (snap / wt).symlink_to(blob)
        (snap / "config.json").write_text("{}")
        assert xf._download_result_usable(
            snap, repo_type = "model", allow_patterns = None, ignore_patterns = ig) is True
    # Both PyTorch formats ignored but no h5/msgpack present -> still rejected.
    snap, _ = _mk_snapshot(tmp_path, "tf_none")
    (snap / "config.json").write_text("{}")
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = ig) is False
    # A normal load (PyTorch format not ignored): a stray leftover h5 does not count, so an h5-only repo is rejected.
    snap, blob = _mk_snapshot(tmp_path, "stray_h5")
    (snap / "tf_model.h5").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None) is False


def test_post_download_checks_sharded_tf_flax_completeness(tmp_path):
    """Sharded TF/Flax weights: a complete set (index + all shards) is accepted, an incomplete one (missing shard or lone shard, no index) rejected."""
    ig = ["*.safetensors", "*.safetensors.index.json", "*.bin", "*.bin.index.json"]
    for base, ext in (("tf_model", "h5"), ("flax_model", "msgpack")):
        idx = json.dumps({"weight_map": {"a": f"{base}-00001-of-00002.{ext}",
                                         "b": f"{base}-00002-of-00002.{ext}"}})
        # Complete sharded set -> accepted.
        snap, blob = _mk_snapshot(tmp_path, f"tfshard_ok_{base}")
        (snap / f"{base}-00001-of-00002.{ext}").symlink_to(blob)
        (snap / f"{base}-00002-of-00002.{ext}").symlink_to(blob)
        (snap / f"{base}.{ext}.index.json").write_text(idx)
        assert xf._download_result_usable(
            snap, repo_type = "model", allow_patterns = None, ignore_patterns = ig) is True
        # A shard listed by the index is missing -> rejected.
        snap2, blob2 = _mk_snapshot(tmp_path, f"tfshard_missing_{base}")
        (snap2 / f"{base}-00001-of-00002.{ext}").symlink_to(blob2)
        (snap2 / f"{base}.{ext}.index.json").write_text(idx)
        assert xf._download_result_usable(
            snap2, repo_type = "model", allow_patterns = None, ignore_patterns = ig) is False
        # A lone shard with no index -> rejected.
        snap3, blob3 = _mk_snapshot(tmp_path, f"tfshard_lone_{base}")
        (snap3 / f"{base}-00001-of-00002.{ext}").symlink_to(blob3)
        assert xf._download_result_usable(
            snap3, repo_type = "model", allow_patterns = None, ignore_patterns = ig) is False


def test_post_download_checks_explicit_checkpoint_shard_completeness(tmp_path):
    """An explicit checkpoint load (allow=['checkpoint-N/*']) reads its weights, so a lone shard there with no index is rejected; a complete set / untargeted leftover is fine."""
    # Lone checkpoint shard, no index, explicitly requested -> rejected.
    snap, blob = _mk_snapshot(tmp_path, "ckpt_lone")
    (snap / "checkpoint-7").mkdir()
    (snap / "checkpoint-7" / "model-00001-of-00002.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = ["checkpoint-7/*"], ignore_patterns = None) is False
    # Complete checkpoint shard set (index + all shards) -> accepted.
    snap2, blob2 = _mk_snapshot(tmp_path, "ckpt_complete")
    (snap2 / "checkpoint-7").mkdir()
    (snap2 / "checkpoint-7" / "model-00001-of-00002.safetensors").symlink_to(blob2)
    (snap2 / "checkpoint-7" / "model-00002-of-00002.safetensors").symlink_to(blob2)
    (snap2 / "checkpoint-7" / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a": "model-00001-of-00002.safetensors",
                        "b": "model-00002-of-00002.safetensors"}}))
    assert xf._download_result_usable(
        snap2, repo_type = "model", allow_patterns = ["checkpoint-7/*"], ignore_patterns = None) is True
    # A leftover checkpoint the request does not target (subfolder=unet) must not reject a complete in-scope download.
    snap3, blob3 = _mk_snapshot(tmp_path, "ckpt_leftover")
    (snap3 / "unet").mkdir()
    (snap3 / "unet" / "diffusion_pytorch_model.safetensors").symlink_to(blob3)
    (snap3 / "checkpoint-7").mkdir()
    (snap3 / "checkpoint-7" / "model-00001-of-00002.safetensors").symlink_to(blob3)  # lone, but not read
    assert xf._download_result_usable(
        snap3, repo_type = "model", allow_patterns = ["unet/*"], ignore_patterns = None) is True
    # A nested checkpoint the request explicitly targets (allow=['foo/checkpoint-7/*']) still rejects its lone shard
    # (the scope check matches a checkpoint dir at any leading segment).
    snap4, blob4 = _mk_snapshot(tmp_path, "ckpt_nested")
    (snap4 / "foo" / "checkpoint-7").mkdir(parents = True)
    (snap4 / "foo" / "checkpoint-7" / "model-00001-of-00002.safetensors").symlink_to(blob4)
    assert xf._download_result_usable(
        snap4, repo_type = "model",
        allow_patterns = ["foo/checkpoint-7/*"], ignore_patterns = None) is False


def test_post_download_accepts_exact_named_variant_shard_subset(tmp_path):
    """An exact variant shard request is accepted once present (index/sibling absent); the exact-name escape applies to the variant branch too."""
    snap, blob = _mk_snapshot(tmp_path, "exact_var_shard")
    (snap / "model.fp16-00001-of-00002.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model",
        allow_patterns = ["model.fp16-00001-of-00002.safetensors"], ignore_patterns = None,
        variant = "fp16") is True
    # The exact-named variant shard absent -> still rejected.
    snap2, _ = _mk_snapshot(tmp_path, "exact_var_absent")
    (snap2 / "config.json").write_text("{}")
    assert xf._download_result_usable(
        snap2, repo_type = "model",
        allow_patterns = ["model.fp16-00001-of-00002.safetensors"], ignore_patterns = None,
        variant = "fp16") is False


def test_post_download_rejects_patterned_incomplete_variant_shards(tmp_path):
    """A globbed variant request with a lone root variant shard (no index) is rejected too; a complete in-scope set is accepted."""
    snap, blob = _mk_snapshot(tmp_path, "pat_var_incomplete")
    (snap / "model.fp16-00001-of-00002.safetensors").symlink_to(blob)  # lone shard, no index
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = ["*.safetensors"], ignore_patterns = None,
        variant = "fp16") is False
    # Complete variant shard set -> accepted.
    (snap / "model.fp16-00002-of-00002.safetensors").symlink_to(blob)
    (snap / "model.safetensors.index.fp16.json").write_text(json.dumps(
        {"weight_map": {"a": "model.fp16-00001-of-00002.safetensors",
                        "b": "model.fp16-00002-of-00002.safetensors"}}))
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = ["*.safetensors"], ignore_patterns = None,
        variant = "fp16") is True


def test_post_download_applies_ignore_to_diffusers_components(tmp_path):
    """A diffusers warm ignoring a format is not satisfied by a component weight in that ignored format (only unet/*.bin under ignore=['*.bin'] is rejected)."""
    snap, blob = _mk_snapshot(tmp_path, "diff_ignore")
    (snap / "model_index.json").write_text("{}")
    (snap / "unet").mkdir()
    (snap / "unet" / "diffusion_pytorch_model.bin").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = ["*.bin"]) is False
    # The safetensors component present -> usable.
    (snap / "unet" / "diffusion_pytorch_model.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = ["*.bin"]) is True


def test_post_download_rejects_index_only_sharded_masked_by_bin(tmp_path):
    """A safetensors index with none of its shards (index-only), co-resident with a complete bin, is rejected (the index is probed before the bin)."""
    snap, blob = _mk_snapshot(tmp_path, "index_only")
    (snap / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a": "model-00001-of-00002.safetensors",
                        "b": "model-00002-of-00002.safetensors"}}))
    (snap / "pytorch_model.bin").symlink_to(blob)  # complete bin, no ST shards at all
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None) is False
    # safetensors explicitly ignored -> load reads the complete bin -> usable.
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = ["*.safetensors"]) is True


def test_post_download_patterned_shard_check_honors_ignore(tmp_path):
    """A patterned request ignoring safetensors selects the complete .bin; a co-resident incomplete ST set does not reject it (the check applies the ignore filter)."""
    snap, blob = _mk_snapshot(tmp_path, "pat_ignore")
    (snap / "model-00001-of-00002.safetensors").symlink_to(blob)  # incomplete ST (shard 2 absent)
    (snap / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a": "model-00001-of-00002.safetensors",
                        "b": "model-00002-of-00002.safetensors"}}))
    (snap / "pytorch_model.bin").symlink_to(blob)  # complete bin, the selected format
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = ["*"], ignore_patterns = ["*.safetensors"]) is True


def test_post_download_variant_root_shard_check_scoped_to_selection(tmp_path):
    """A subfolder variant request with a complete selected weight is accepted even with a stale out-of-scope root variant shard co-resident."""
    snap, blob = _mk_snapshot(tmp_path, "var_scope")
    (snap / "unet").mkdir()
    (snap / "unet" / "model.fp16.safetensors").symlink_to(blob)  # complete in-scope variant
    (snap / "model.fp16-00001-of-00002.safetensors").symlink_to(blob)  # stale ROOT variant shard, oos
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = ["unet/*"], ignore_patterns = None,
        variant = "fp16") is True
    # A GLOBBED root variant request DOES get the root variant-shard check.
    snap2, blob2 = _mk_snapshot(tmp_path, "var_scope_glob")
    (snap2 / "model.fp16-00001-of-00002.safetensors").symlink_to(blob2)  # lone root variant shard
    assert xf._download_result_usable(
        snap2, repo_type = "model", allow_patterns = ["*.safetensors"], ignore_patterns = None,
        variant = "fp16") is False


def test_post_download_root_variant_weight_honors_ignore(tmp_path):
    """A variant load ignoring .bin is not satisfied by a variant .bin (only model.fp16.bin under ignore=['*.bin'] is rejected)."""
    snap, blob = _mk_snapshot(tmp_path, "var_ignore")
    (snap / "model.fp16.bin").symlink_to(blob)  # only the ignored-format variant
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = ["*.bin"],
        variant = "fp16") is False
    # The safetensors variant present -> usable.
    (snap / "model.fp16.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = ["*.bin"],
        variant = "fp16") is True


def test_post_download_variant_shard_check_honors_ignore(tmp_path):
    """A variant load ignoring .bin judges the variant shard set for the read format only: a complete ST variant beside a stale ignored .bin shard is accepted."""
    snap, blob = _mk_snapshot(tmp_path, "var_shard_ignore")
    (snap / "model.fp16.safetensors").symlink_to(blob)                 # complete, the read format
    (snap / "model.fp16-00001-of-00002.bin").symlink_to(blob)         # stale ignored .bin shard, no index
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = ["*.bin"],
        variant = "fp16") is True
    # Without the ignore, the complete safetensors variant is preferred and read.
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None,
        variant = "fp16") is True  # the complete safetensors variant is preferred and read


def test_post_download_rejects_variant_index_only_masked_by_bin(tmp_path):
    """The variant analog of index-only: a variant ST index with none of its shards, beside a complete variant bin, is rejected."""
    snap, blob = _mk_snapshot(tmp_path, "var_index_only")
    (snap / "model.safetensors.index.fp16.json").write_text(json.dumps(
        {"weight_map": {"a": "model.fp16-00001-of-00002.safetensors",
                        "b": "model.fp16-00002-of-00002.safetensors"}}))
    (snap / "pytorch_model.fp16.bin").symlink_to(blob)  # complete bin, no ST variant shards at all
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None,
        variant = "fp16") is False
    # The variant safetensors ignored -> load reads the complete variant bin -> usable.
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = ["*.safetensors"],
        variant = "fp16") is True


def test_post_download_rejects_incomplete_sharded_adapter(tmp_path):
    """A PEFT adapter load whose partial has a sharded adapter index missing a shard is rejected (the selected-index check covers the non-model adapter index)."""
    snap, blob = _mk_snapshot(tmp_path, "adapter_incomplete")
    (snap / "adapter_config.json").write_text("{}")
    (snap / "adapter_model-00001-of-00002.safetensors").symlink_to(blob)  # shard 1; shard 2 absent
    (snap / "adapter_model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a": "adapter_model-00001-of-00002.safetensors",
                        "b": "adapter_model-00002-of-00002.safetensors"}}))
    allow = ["adapter_config.json", "adapter_model*"]
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = allow, ignore_patterns = None) is False
    # The missing adapter shard present -> complete set -> accepted.
    (snap / "adapter_model-00002-of-00002.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = allow, ignore_patterns = None) is True


def test_post_download_rejects_incomplete_component_subfolder_shards(tmp_path):
    """A subfolder request (allow=['unet/*']) whose component has a shard index missing a shard is rejected (the selected-index check covers component subfolders)."""
    snap, blob = _mk_snapshot(tmp_path, "component_incomplete")
    (snap / "unet").mkdir()
    (snap / "unet" / "diffusion_pytorch_model-00001-of-00002.safetensors").symlink_to(blob)
    (snap / "unet" / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a": "diffusion_pytorch_model-00001-of-00002.safetensors",
                        "b": "diffusion_pytorch_model-00002-of-00002.safetensors"}}))
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = ["unet/*"], ignore_patterns = None) is False
    (snap / "unet" / "diffusion_pytorch_model-00002-of-00002.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = ["unet/*"], ignore_patterns = None) is True


def test_post_download_rejects_gguf_only_default_load(tmp_path):
    """A default warm reads safetensors/bin, not GGUF, so a snapshot holding only a .gguf is rejected."""
    snap, blob = _mk_snapshot(tmp_path, "gguf_only")
    (snap / "model.Q4_K_M.gguf").symlink_to(blob)
    (snap / "config.json").write_text("{}")
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None) is False
    # The safetensors weight present -> the default warm accepts, even beside the gguf.
    (snap / "model.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None) is True


def test_post_download_rejects_adapter_variant_for_default_variant_load(tmp_path):
    """A variant warm reads the root model variant, not an adapter variant, so an adapter-variant-only snapshot is rejected; the base variant present is accepted."""
    snap, blob = _mk_snapshot(tmp_path, "adapter_variant_only")
    (snap / "adapter_model.fp16.safetensors").symlink_to(blob)
    (snap / "adapter_config.json").write_text("{}")
    (snap / "config.json").write_text("{}")
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None,
        variant = "fp16") is False
    (snap / "model.fp16.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None,
        variant = "fp16") is True


def test_post_download_accepts_complete_diffusers_variant(tmp_path):
    """A diffusers variant warm's weights are component-scoped, so a complete diffusers variant download is accepted; a non-variant-only pipeline does not satisfy a variant load."""
    snap, blob = _mk_snapshot(tmp_path, "diffusers_variant")
    (snap / "model_index.json").write_text("{}")
    (snap / "unet").mkdir()
    (snap / "unet" / "config.json").write_text("{}")
    (snap / "unet" / "diffusion_pytorch_model.fp16.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None,
        variant = "fp16") is True
    snap2, blob2 = _mk_snapshot(tmp_path, "diffusers_variant_missing")
    (snap2 / "model_index.json").write_text("{}")
    (snap2 / "unet").mkdir()
    (snap2 / "unet" / "diffusion_pytorch_model.safetensors").symlink_to(blob2)  # non-variant only
    assert xf._download_result_usable(
        snap2, repo_type = "model", allow_patterns = None, ignore_patterns = None,
        variant = "fp16") is False


def test_post_download_rejects_incomplete_diffusers_component_shards_unpatterned(tmp_path):
    """An unpatterned diffusers warm rejects a component shard index listing an absent shard; plain and variant indexes are covered, a complete set accepted."""
    snap, blob = _mk_snapshot(tmp_path, "diffusers_comp_incomplete")
    (snap / "model_index.json").write_text("{}")
    (snap / "unet").mkdir()
    (snap / "unet" / "config.json").write_text("{}")
    (snap / "unet" / "diffusion_pytorch_model-00001-of-00002.safetensors").symlink_to(blob)
    (snap / "unet" / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a": "diffusion_pytorch_model-00001-of-00002.safetensors",
                        "b": "diffusion_pytorch_model-00002-of-00002.safetensors"}}))
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None) is False
    (snap / "unet" / "diffusion_pytorch_model-00002-of-00002.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None) is True
    # Same for a variant component index (variant='fp16', unpatterned).
    snapv, blobv = _mk_snapshot(tmp_path, "diffusers_comp_variant_incomplete")
    (snapv / "model_index.json").write_text("{}")
    (snapv / "unet").mkdir()
    (snapv / "unet" / "diffusion_pytorch_model.fp16-00001-of-00002.safetensors").symlink_to(blobv)
    (snapv / "unet" / "diffusion_pytorch_model.safetensors.index.fp16.json").write_text(json.dumps(
        {"weight_map": {"a": "diffusion_pytorch_model.fp16-00001-of-00002.safetensors",
                        "b": "diffusion_pytorch_model.fp16-00002-of-00002.safetensors"}}))
    assert xf._download_result_usable(
        snapv, repo_type = "model", allow_patterns = None, ignore_patterns = None,
        variant = "fp16") is False
    (snapv / "unet" / "diffusion_pytorch_model.fp16-00002-of-00002.safetensors").symlink_to(blobv)
    assert xf._download_result_usable(
        snapv, repo_type = "model", allow_patterns = None, ignore_patterns = None,
        variant = "fp16") is True


def test_post_download_single_safetensors_beats_stale_index(tmp_path):
    """A complete single model.safetensors (probed before the index) beside a stale incomplete index is usable; a stale index with no single weight is breakage."""
    snap, blob = _mk_snapshot(tmp_path, "single_beats_index")
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").symlink_to(blob)
    (snap / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a": "model-00001-of-00002.safetensors",
                        "b": "model-00002-of-00002.safetensors"}}))  # shards absent
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None) is True
    assert hcs.snapshot_dir_is_complete(snap) is True  # the PRE gate agrees
    # No single weight, only the stale index -> incomplete.
    snap2, _ = _mk_snapshot(tmp_path, "stale_index_only")
    (snap2 / "config.json").write_text("{}")
    (snap2 / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a": "model-00001-of-00002.safetensors",
                        "b": "model-00002-of-00002.safetensors"}}))
    assert xf._download_result_usable(
        snap2, repo_type = "model", allow_patterns = None, ignore_patterns = None) is False


def test_post_download_rejects_noncanonical_root_weight_for_default_load(tmp_path):
    """A default load probes only canonical names, so a cache holding only a non-canonical root weight (consolidated.safetensors) is rejected; the canonical weight present is accepted."""
    snap, blob = _mk_snapshot(tmp_path, "noncanonical")
    (snap / "config.json").write_text("{}")
    (snap / "consolidated.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None) is False
    (snap / "model.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None) is True


def test_diffusers_component_check_scoped_to_declared_components(tmp_path):
    """The component shard check is scoped to declared components: a stale undeclared subtree does not reject a complete pipeline; an incomplete declared component still does."""
    snap, blob = _mk_snapshot(tmp_path, "declared_scope")
    (snap / "model_index.json").write_text(json.dumps(
        {"_class_name": "StableDiffusionPipeline",
         "unet": ["diffusers", "UNet2DConditionModel"], "vae": ["diffusers", "AutoencoderKL"]}))
    for comp in ("unet", "vae"):
        (snap / comp).mkdir()
        (snap / comp / "config.json").write_text("{}")
        (snap / comp / "diffusion_pytorch_model.safetensors").symlink_to(blob)
    (snap / "controlnet").mkdir()  # UNDECLARED leftover
    (snap / "controlnet" / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a": "diffusion_pytorch_model-00001-of-00002.safetensors",
                        "b": "diffusion_pytorch_model-00002-of-00002.safetensors"}}))
    (snap / "controlnet" / "diffusion_pytorch_model-00001-of-00002.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None) is True
    # An incomplete declared component (unet index missing a shard) is still caught.
    snap2, blob2 = _mk_snapshot(tmp_path, "declared_incomplete")
    (snap2 / "model_index.json").write_text(json.dumps(
        {"_class_name": "P", "unet": ["diffusers", "UNet2DConditionModel"]}))
    (snap2 / "unet").mkdir()
    (snap2 / "unet" / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a": "diffusion_pytorch_model-00001-of-00002.safetensors",
                        "b": "diffusion_pytorch_model-00002-of-00002.safetensors"}}))
    (snap2 / "unet" / "diffusion_pytorch_model-00001-of-00002.safetensors").symlink_to(blob2)
    assert xf._download_result_usable(
        snap2, repo_type = "model", allow_patterns = None, ignore_patterns = None) is False


def test_post_download_variant_presence_requires_canonical_name(tmp_path):
    """The variant presence check counts only a canonical variant name; a non-canonical sidecar or dot-infix shard does not satisfy the request."""
    snap, blob = _mk_snapshot(tmp_path, "var_noncanonical")
    (snap / "config.json").write_text("{}")
    (snap / "consolidated.fp16.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None,
        variant = "fp16") is False
    snap_dot, blob_dot = _mk_snapshot(tmp_path, "var_dotinfix")
    (snap_dot / "config.json").write_text("{}")
    (snap_dot / "model-00001-of-00001.fp16.safetensors").symlink_to(blob_dot)  # not a name tf reads
    assert xf._download_result_usable(
        snap_dot, repo_type = "model", allow_patterns = None, ignore_patterns = None,
        variant = "fp16") is False
    # The canonical single variant weight -> accepted.
    (snap / "model.fp16.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None,
        variant = "fp16") is True


def test_post_download_rejects_selected_shard_without_index(tmp_path):
    """A selected non-root numbered shard with no index is an incomplete set the load cannot enumerate, so it is rejected; a complete indexed set is accepted."""
    # A sharded adapter with a lone shard and no index.
    snap, blob = _mk_snapshot(tmp_path, "adapter_lone_shard")
    (snap / "config.json").write_text("{}")
    (snap / "adapter_config.json").write_text("{}")
    (snap / "adapter_model-00001-of-00002.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = ["adapter_model*", "adapter_config.json"],
        ignore_patterns = None) is False
    # Complete it with the second shard + index -> accepted.
    (snap / "adapter_model-00002-of-00002.safetensors").symlink_to(blob)
    (snap / "adapter_model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a": "adapter_model-00001-of-00002.safetensors",
                        "b": "adapter_model-00002-of-00002.safetensors"}}))
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = ["adapter_model*", "adapter_config.json"],
        ignore_patterns = None) is True
    # A component subfolder lone shard (allow=['unet/*']) is likewise rejected.
    snap2, blob2 = _mk_snapshot(tmp_path, "unet_lone_shard")
    (snap2 / "unet").mkdir()
    (snap2 / "unet" / "config.json").write_text("{}")
    (snap2 / "unet" / "diffusion_pytorch_model-00001-of-00002.safetensors").symlink_to(blob2)
    assert xf._download_result_usable(
        snap2, repo_type = "model", allow_patterns = ["unet/*"], ignore_patterns = None) is False


def test_post_download_diffusers_presence_scoped_to_declared(tmp_path):
    """A diffusers warm counts a component weight only for a declared component, so an undeclared-leftover-only cache is rejected; declared components present are accepted."""
    snap, blob = _mk_snapshot(tmp_path, "diffusers_undeclared_only")
    (snap / "model_index.json").write_text(json.dumps(
        {"_class_name": "P", "unet": ["diffusers", "U"], "vae": ["diffusers", "V"]}))
    (snap / "controlnet").mkdir()  # UNDECLARED
    (snap / "controlnet" / "diffusion_pytorch_model.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None) is False
    # The declared components present -> accepted.
    for comp in ("unet", "vae"):
        (snap / comp).mkdir()
        (snap / comp / "diffusion_pytorch_model.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None) is True


def test_post_download_diffusers_variant_presence_scoped_to_declared(tmp_path):
    """Variant twin of the declared-scope check: a diffusers variant warm counts a component variant weight only for a declared component."""
    snap, blob = _mk_snapshot(tmp_path, "diffusers_variant_undeclared_only")
    (snap / "model_index.json").write_text(json.dumps(
        {"_class_name": "P", "unet": ["diffusers", "U"], "vae": ["diffusers", "V"]}))
    (snap / "controlnet").mkdir()  # UNDECLARED
    (snap / "controlnet" / "diffusion_pytorch_model.fp16.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None,
        variant = "fp16") is False
    # The declared component variant weights present -> accepted.
    for comp in ("unet", "vae"):
        (snap / comp).mkdir()
        (snap / comp / "diffusion_pytorch_model.fp16.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None,
        variant = "fp16") is True


def test_post_download_rejects_diffusers_missing_declared_component(tmp_path):
    """A declared weight-bearing component absent (or holding only its config) is retried over HTTP, not
    accepted -- else the in-process pipeline load fetches the missing component over un-killable Xet. A
    ``[null, null]`` (disabled) component and weightless components (scheduler / tokenizer) are not
    required."""
    def _mi():
        return json.dumps({
            "_class_name": "StableDiffusionPipeline",
            "unet": ["diffusers", "UNet2DConditionModel"],
            "vae": ["diffusers", "AutoencoderKL"],
            "text_encoder": ["transformers", "CLIPTextModel"],
            "scheduler": ["diffusers", "PNDMScheduler"],
            "tokenizer": ["transformers", "CLIPTokenizer"],
            "safety_checker": [None, None],
        })

    def _model_comp(root, name, blob, *, weight = True, variant = None):
        d = root / name
        d.mkdir()
        (d / "config.json").write_text("{}")
        if weight:
            w = "diffusion_pytorch_model.safetensors" if variant is None \
                else f"diffusion_pytorch_model.{variant}.safetensors"
            (d / w).symlink_to(blob)

    def _weightless(root, name):
        d = root / name
        d.mkdir()
        # scheduler_config.json / tokenizer_config.json -- NOT a plain config.json, so no weight required
        (d / f"{name}_config.json").write_text("{}")

    # unet present, vae ABSENT (text_encoder present) -> reject.
    snap, blob = _mk_snapshot(tmp_path, "diff_missing_vae")
    (snap / "model_index.json").write_text(_mi())
    _model_comp(snap, "unet", blob)
    _model_comp(snap, "text_encoder", blob)
    _weightless(snap, "scheduler")
    _weightless(snap, "tokenizer")
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None) is False

    # vae present with config ONLY (weight missing) -> reject.
    snap2, blob2 = _mk_snapshot(tmp_path, "diff_vae_config_only")
    (snap2 / "model_index.json").write_text(_mi())
    _model_comp(snap2, "unet", blob2)
    _model_comp(snap2, "text_encoder", blob2)
    _model_comp(snap2, "vae", blob2, weight = False)  # config, no weight
    _weightless(snap2, "scheduler")
    _weightless(snap2, "tokenizer")
    assert xf._download_result_usable(
        snap2, repo_type = "model", allow_patterns = None, ignore_patterns = None) is False

    # Every weight-bearing component complete; safety_checker [null,null] absent -> accept (no false-reject).
    snap3, blob3 = _mk_snapshot(tmp_path, "diff_complete")
    (snap3 / "model_index.json").write_text(_mi())
    for c in ("unet", "vae", "text_encoder"):
        _model_comp(snap3, c, blob3)
    _weightless(snap3, "scheduler")
    _weightless(snap3, "tokenizer")
    assert xf._download_result_usable(
        snap3, repo_type = "model", allow_patterns = None, ignore_patterns = None) is True

    # Variant: vae absent -> reject; a mixed pipeline (unet fp16, vae/text_encoder canonical fallback) -> accept.
    snap4, blob4 = _mk_snapshot(tmp_path, "diff_variant_missing_vae")
    (snap4 / "model_index.json").write_text(_mi())
    _model_comp(snap4, "unet", blob4, variant = "fp16")
    _model_comp(snap4, "text_encoder", blob4, variant = "fp16")
    _weightless(snap4, "scheduler")
    _weightless(snap4, "tokenizer")
    assert xf._download_result_usable(
        snap4, repo_type = "model", allow_patterns = None, ignore_patterns = None,
        variant = "fp16") is False
    snap5, blob5 = _mk_snapshot(tmp_path, "diff_variant_mixed")
    (snap5 / "model_index.json").write_text(_mi())
    _model_comp(snap5, "unet", blob5, variant = "fp16")
    _model_comp(snap5, "vae", blob5)            # canonical fallback for this component
    _model_comp(snap5, "text_encoder", blob5)   # canonical fallback
    _weightless(snap5, "scheduler")
    _weightless(snap5, "tokenizer")
    assert xf._download_result_usable(
        snap5, repo_type = "model", allow_patterns = None, ignore_patterns = None,
        variant = "fp16") is True


def test_post_download_single_variant_beats_stale_variant_index(tmp_path):
    """Variant twin of single-beats-index: a complete single variant weight beside a stale variant index is usable (ST and bin); a stale index with no single weight is breakage."""
    snap, blob = _mk_snapshot(tmp_path, "single_variant_beats_index")
    (snap / "config.json").write_text("{}")
    (snap / "model.fp16.safetensors").symlink_to(blob)
    (snap / "model.safetensors.index.fp16.json").write_text(json.dumps(
        {"weight_map": {"a": "model.fp16-00001-of-00002.safetensors",
                        "b": "model.fp16-00002-of-00002.safetensors"}}))  # shards absent
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None,
        variant = "fp16") is True
    # A single .bin variant beside a stale .bin variant index (no ST) -> usable.
    snapb, blobb = _mk_snapshot(tmp_path, "single_bin_variant_beats_index")
    (snapb / "config.json").write_text("{}")
    (snapb / "pytorch_model.fp16.bin").symlink_to(blobb)
    (snapb / "pytorch_model.bin.index.fp16.json").write_text(json.dumps(
        {"weight_map": {"a": "pytorch_model.fp16-00001-of-00002.bin"}}))
    assert xf._download_result_usable(
        snapb, repo_type = "model", allow_patterns = None, ignore_patterns = None,
        variant = "fp16") is True
    # A stale variant index with no single variant weight -> incomplete.
    snap2, _ = _mk_snapshot(tmp_path, "stale_variant_index_only")
    (snap2 / "config.json").write_text("{}")
    (snap2 / "model.safetensors.index.fp16.json").write_text(json.dumps(
        {"weight_map": {"a": "model.fp16-00001-of-00002.safetensors",
                        "b": "model.fp16-00002-of-00002.safetensors"}}))
    assert xf._download_result_usable(
        snap2, repo_type = "model", allow_patterns = None, ignore_patterns = None,
        variant = "fp16") is False


def test_post_download_diffusers_skips_root_model_shard_checks(tmp_path):
    """A diffusers pipeline reads component subfolders, so a stale root model shard index (canonical or variant) is accepted; component completeness is still enforced."""
    # Plain: stale root model.safetensors.index.json alongside complete components.
    snap, blob = _mk_snapshot(tmp_path, "diffusers_stale_root_index_plain")
    (snap / "model_index.json").write_text(json.dumps(
        {"_class_name": "P", "unet": ["diffusers", "U"], "vae": ["diffusers", "V"]}))
    for comp in ("unet", "vae"):
        (snap / comp).mkdir()
        (snap / comp / "diffusion_pytorch_model.safetensors").symlink_to(blob)
    (snap / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a": "model-00001-of-00002.safetensors",
                        "b": "model-00002-of-00002.safetensors"}}))  # shards absent (stale)
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None) is True
    # Variant: stale root model.safetensors.index.fp16.json alongside complete variant components.
    snapv, blobv = _mk_snapshot(tmp_path, "diffusers_stale_root_index_variant")
    (snapv / "model_index.json").write_text(json.dumps(
        {"_class_name": "P", "unet": ["diffusers", "U"], "vae": ["diffusers", "V"]}))
    for comp in ("unet", "vae"):
        (snapv / comp).mkdir()
        (snapv / comp / "diffusion_pytorch_model.fp16.safetensors").symlink_to(blobv)
    (snapv / "model.safetensors.index.fp16.json").write_text(json.dumps(
        {"weight_map": {"a": "model.fp16-00001-of-00002.safetensors",
                        "b": "model.fp16-00002-of-00002.safetensors"}}))
    assert xf._download_result_usable(
        snapv, repo_type = "model", allow_patterns = None, ignore_patterns = None,
        variant = "fp16") is True
    # An incomplete declared component is still rejected.
    (snapv / "unet" / "diffusion_pytorch_model.fp16.safetensors").unlink()
    (snapv / "unet" / "diffusion_pytorch_model.fp16-00001-of-00002.safetensors").symlink_to(blobv)
    (snapv / "unet" / "diffusion_pytorch_model.safetensors.index.fp16.json").write_text(json.dumps(
        {"weight_map": {"a": "diffusion_pytorch_model.fp16-00001-of-00002.safetensors",
                        "b": "diffusion_pytorch_model.fp16-00002-of-00002.safetensors"}}))
    assert xf._download_result_usable(
        snapv, repo_type = "model", allow_patterns = None, ignore_patterns = None,
        variant = "fp16") is False


def test_post_download_out_of_scope_malformed_index_not_rejected(tmp_path):
    """A malformed shard index the request does not select is not read, so it does not reject a complete in-scope download; an in-scope malformed index is breakage."""
    snap, blob = _mk_snapshot(tmp_path, "malformed_out_of_scope")
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").symlink_to(blob)
    (snap / "adapter_model.safetensors.index.json").write_text("{ not valid json")  # malformed, unselected
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = ["model*"], ignore_patterns = None) is True
    # An in-scope malformed index (adapter warm selects adapter_model*) is still rejected.
    snap2, blob2 = _mk_snapshot(tmp_path, "malformed_in_scope")
    (snap2 / "config.json").write_text("{}")
    (snap2 / "adapter_config.json").write_text("{}")
    (snap2 / "adapter_model-00001-of-00002.safetensors").symlink_to(blob2)
    (snap2 / "adapter_model.safetensors.index.json").write_text("{ not valid json")
    assert xf._download_result_usable(
        snap2, repo_type = "model", allow_patterns = ["adapter_model*", "adapter_config.json"],
        ignore_patterns = None) is False


def test_selected_readable_weight_complete_entry_point(tmp_path):
    """The acceptance helper enforces (A) a readable weight present, (B) its shard set complete: exercise present+complete, absent, and incomplete-shard cases."""
    # Present + complete single weight -> True.
    snap, blob = _mk_snapshot(tmp_path, "srwc_ok")
    (snap / "model.safetensors").symlink_to(blob)
    assert xf._selected_readable_weight_complete(
        snap, allow_patterns = None, ignore_patterns = None, variant = None) is True
    # Invariant A fails: only an ignored-format weight present -> False.
    snap2, blob2 = _mk_snapshot(tmp_path, "srwc_ignored")
    (snap2 / "pytorch_model.bin").symlink_to(blob2)
    assert xf._selected_readable_weight_complete(
        snap2, allow_patterns = None, ignore_patterns = ["*.bin"], variant = None) is False
    # Invariant B fails: readable weight present but its shard set incomplete -> False.
    snap3, blob3 = _mk_snapshot(tmp_path, "srwc_incomplete")
    (snap3 / "model-00001-of-00002.safetensors").symlink_to(blob3)
    (snap3 / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a": "model-00001-of-00002.safetensors",
                        "b": "model-00002-of-00002.safetensors"}}))
    assert xf._selected_readable_weight_complete(
        snap3, allow_patterns = None, ignore_patterns = None, variant = None) is False


def test_post_download_accepts_dataset_without_weight(tmp_path):
    snap, blob = _mk_snapshot(tmp_path, "ds")
    (snap / "data.parquet").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "dataset", allow_patterns = None, ignore_patterns = None) is True


def test_post_download_accepts_either_format_single_present(tmp_path):
    """An either-format named request against a safetensors-only repo is accepted (not retried for the .bin the repo does not publish)."""
    snap, blob = _mk_snapshot(tmp_path, "either")
    (snap / "model.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model",
        allow_patterns = ["pytorch_model.bin", "model.safetensors"], ignore_patterns = None) is True


def test_pre_download_skips_intact_tokenizer_only(tmp_path):
    """A tokenizer-only (weightless) warm short-circuits offline: an intact requested subset is enough, no weight required."""
    snap, _ = _mk_snapshot(tmp_path, "tok")
    (snap / "tokenizer.json").write_text("{}")
    (snap / "config.json").write_text("{}")
    assert xf._cache_can_skip_download(
        snap, repo_type = "model",
        allow_patterns = ["tokenizer.json", "config.json"], ignore_patterns = None) is True


def test_pre_download_partial_ignore_does_not_skip_config_only(tmp_path):
    """An ignore-only request stripping only some weight formats on a config-only cache must not skip the child (a surviving weight format would hang over Xet)."""
    snap, _ = _mk_snapshot(tmp_path, "cfgign")
    (snap / "config.json").write_text("{}")
    assert xf._cache_can_skip_download(
        snap, repo_type = "model", allow_patterns = None,
        ignore_patterns = ["*.safetensors", "*.bin"]) is False


# Review-round regression guards (10-reviewer findings)
def test_gate_rejects_malformed_shard_index(tmp_path):
    """A truncated / non-dict / empty weight-shard index does not read as complete (_weight_shard_index_complete is fail-closed)."""
    snap, blob = _mk_snapshot(tmp_path, "malidx")
    (snap / "model-00001-of-00002.safetensors").symlink_to(blob)
    (snap / "model.safetensors.index.json").write_text("{not valid json")
    assert hcs.snapshot_dir_is_complete(snap) is False
    # Empty weight_map proves nothing.
    snap2, blob2 = _mk_snapshot(tmp_path, "emptyidx")
    (snap2 / "model-00001-of-00002.safetensors").symlink_to(blob2)
    (snap2 / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    assert hcs.snapshot_dir_is_complete(snap2) is False
    # weight_map not a dict.
    snap3, blob3 = _mk_snapshot(tmp_path, "listidx")
    (snap3 / "model.safetensors.index.json").write_text(json.dumps({"weight_map": ["a", "b"]}))
    assert hcs._weight_shard_index_complete(snap3 / "model.safetensors.index.json") is False


def test_shard_index_rejects_unsafe_path_refs(tmp_path):
    """An attacker-influenced shard value (absolute, drive-letter, UNC, parent-escaping) is rejected so ``base / shard`` cannot probe outside the snapshot."""
    # Unit: the helper flags every escape variant and keeps legit relative names.
    for bad in ["/etc/passwd", r"C:\evil.safetensors", "C:evil.safetensors", r"\\srv\share\x",
                "../../x.safetensors", r"..\x.safetensors", "a/../../b"]:
        assert hcs._is_unsafe_shard_ref(bad) is True, bad
    for ok in ["model-00001-of-00002.safetensors", "unet/diffusion_pytorch_model.safetensors",
               "model.fp16.safetensors"]:
        assert hcs._is_unsafe_shard_ref(ok) is False, ok
    # A crafted index listing a drive-letter shard is not "complete".
    snap, blob = _mk_snapshot(tmp_path, "unsafe_shard_idx")
    (snap / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a": r"C:\Windows\System32\x.safetensors",
                        "b": "model-00002-of-00002.safetensors"}}))
    assert hcs._weight_shard_index_complete(snap / "model.safetensors.index.json") is False
    # The enumerator returns None (defer) rather than a path escaping the snapshot.
    assert hcs._index_shard_rel_paths(snap / "model.safetensors.index.json", "") is None
    # A well-formed relative index still enumerates + validates.
    snap2, blob2 = _mk_snapshot(tmp_path, "safe_shard_idx")
    (snap2 / "model-00001-of-00002.safetensors").symlink_to(blob2)
    (snap2 / "model-00002-of-00002.safetensors").symlink_to(blob2)
    (snap2 / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a": "model-00001-of-00002.safetensors",
                        "b": "model-00002-of-00002.safetensors"}}))
    assert hcs._weight_shard_index_complete(snap2 / "model.safetensors.index.json") is True
    assert set(hcs._index_shard_rel_paths(snap2 / "model.safetensors.index.json", "")) == {
        "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}


def test_malformed_index_scope_honors_ignored_format(tmp_path):
    """A malformed index is judged by the weight the load reads, so a malformed index for an ignored format is skipped; a malformed index of the read format is breakage."""
    # Patterned subfolder warm reading safetensors: a co-resident malformed bin index is ignored.
    snap, blob = _mk_snapshot(tmp_path, "malformed_ignored_bin_idx")
    (snap / "unet").mkdir()
    (snap / "unet" / "config.json").write_text("{}")
    (snap / "unet" / "diffusion_pytorch_model.safetensors").symlink_to(blob)
    (snap / "unet" / "diffusion_pytorch_model.bin.index.json").write_text("{ truncated")
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = ["unet/*"], ignore_patterns = ["*.bin"]) is True
    # The malformed index of the READ format (safetensors, not ignored) is still breakage.
    snap2, _ = _mk_snapshot(tmp_path, "malformed_read_st_idx")
    (snap2 / "unet").mkdir()
    (snap2 / "unet" / "config.json").write_text("{}")
    (snap2 / "unet" / "diffusion_pytorch_model.safetensors.index.json").write_text("{ truncated")
    assert xf._download_result_usable(
        snap2, repo_type = "model", allow_patterns = ["unet/*"], ignore_patterns = None) is False
    # Diffusers: a malformed component bin index under ignore=['*.bin'] does not reject a complete ST pipeline.
    snap3, blob3 = _mk_snapshot(tmp_path, "malformed_diffusers_bin_idx")
    (snap3 / "model_index.json").write_text(json.dumps(
        {"_class_name": "P", "unet": ["diffusers", "U"], "vae": ["diffusers", "V"]}))
    for comp in ("unet", "vae"):
        (snap3 / comp).mkdir()
        (snap3 / comp / "diffusion_pytorch_model.safetensors").symlink_to(blob3)
    (snap3 / "unet" / "diffusion_pytorch_model.bin.index.json").write_text("{ truncated")
    assert xf._download_result_usable(
        snap3, repo_type = "model", allow_patterns = None, ignore_patterns = ["*.bin"]) is True


def test_gate_ignored_canonical_weight_does_not_prove_complete(tmp_path):
    """A canonical weight whose format the request ignores does not prove completeness (a bin-only cache under ignore=['*.bin'] defers to the child)."""
    snap, blob = _mk_snapshot(tmp_path, "ignbin")
    (snap / "config.json").write_text("{}")
    (snap / "pytorch_model.bin").symlink_to(blob)
    assert hcs.snapshot_dir_is_complete(snap, ignore_patterns = ["*.bin"]) is False
    # Without the ignore, the present .bin is what a default load reads -> complete.
    assert hcs.snapshot_dir_is_complete(snap) is True
    # A .bin shard index is also discarded when *.bin is ignored (the format probe catches its .json sidecar).
    snap2, blob2 = _mk_snapshot(tmp_path, "ignbinshard")
    (snap2 / "pytorch_model-00001-of-00001.bin").symlink_to(blob2)
    (snap2 / "pytorch_model.bin.index.json").write_text(
        json.dumps({"weight_map": {"a": "pytorch_model-00001-of-00001.bin"}}))
    assert hcs.snapshot_dir_is_complete(snap2, ignore_patterns = ["*.bin"]) is False
    assert hcs.snapshot_dir_is_complete(snap2) is True
    # A safetensors warm survives an *.bin ignore.
    snap3, blob3 = _mk_snapshot(tmp_path, "stignbin")
    (snap3 / "config.json").write_text("{}")
    (snap3 / "model.safetensors").symlink_to(blob3)
    assert hcs.snapshot_dir_is_complete(snap3, ignore_patterns = ["*.bin"]) is True


def test_post_download_accepts_weightless_patterned_result(tmp_path):
    """A genuinely weightless patterned result (allow=['tokenizer*']) is accepted post-download; the no-weight rejection stays for an unpatterned model warm."""
    snap, _ = _mk_snapshot(tmp_path, "tokglob")
    (snap / "tokenizer.json").write_text("{}")
    (snap / "tokenizer_config.json").write_text("{}")
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = ["tokenizer*"], ignore_patterns = None) is True
    # An unpatterned model warm with no weight is still rejected (stale config-only snapshot).
    cfg, _ = _mk_snapshot(tmp_path, "cfgonly")
    (cfg / "config.json").write_text("{}")
    assert xf._download_result_usable(
        cfg, repo_type = "model", allow_patterns = None, ignore_patterns = None) is False


def test_gate_rejects_variant_only_shard_index(tmp_path):
    """A variant-only shard index does not satisfy the canonical allow=None fast path (snapshot_dir_is_complete is variant-blind); a canonical index does."""
    snap, blob = _mk_snapshot(tmp_path, "variant")
    (snap / "config.json").write_text("{}")
    (snap / "model-00001-of-00001.fp16.safetensors").symlink_to(blob)
    (snap / "model.safetensors.index.fp16.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00001.fp16.safetensors"}}))
    assert hcs.snapshot_dir_is_complete(snap) is False
    # The canonical index for the same model still fast-paths.
    (snap / "model-00001-of-00001.safetensors").symlink_to(blob)
    (snap / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00001.safetensors"}}))
    assert hcs.snapshot_dir_is_complete(snap) is True


def test_generic_hub_http_error_type_preserved_but_status_drives_retry():
    """A bare HfHubHTTPError keeps its type across the spawn boundary while retry stays status-driven: a 5xx retries, a 4xx does not."""
    assert "HfHubHTTPError" not in xf._DETERMINISTIC_ERROR_NAMES   # status-driven, not name-driven
    cls = xf._resolve_exception_class("HfHubHTTPError")
    assert cls is not None and issubclass(cls, BaseException)

    class _Resp:
        def __init__(self, code): self.status_code = code
    e503 = xf._instantiate_preserving_type(cls, "HfHubHTTPError: 503 service unavailable")
    e503.response = _Resp(503)
    assert xf._is_retryable_download_error(e503) is True           # 5xx still retryable
    e403 = xf._instantiate_preserving_type(cls, "HfHubHTTPError: 403 forbidden")
    e403.response = _Resp(403)
    assert xf._is_retryable_download_error(e403) is False          # 4xx deterministic


def test_hfvalidationerror_type_preserved_across_spawn():
    """A malformed repo id fails identically over either transport, so HFValidationError is deterministic and its type is reconstructed across the spawn boundary."""
    assert "HFValidationError" in xf._DETERMINISTIC_ERROR_NAMES
    cls = xf._resolve_exception_class("HFValidationError")
    assert cls is not None and issubclass(cls, BaseException)
    inst = xf._instantiate_preserving_type(cls, "HFValidationError: bad repo id")
    assert type(inst).__name__ == "HFValidationError"
    assert xf._is_retryable_download_error(inst) is False


def test_oserror_subclass_type_preserved_across_spawn():
    """A builtin OSError subclass keeps its type across the spawn boundary; a non-OSError builtin is not spuriously resolved."""
    for name in ("PermissionError", "FileNotFoundError", "IsADirectoryError", "NotADirectoryError"):
        cls = xf._resolve_exception_class(name)
        assert cls is not None and issubclass(cls, OSError) and cls.__name__ == name
    # A deterministic PermissionError is reconstructed as a real PermissionError and not retried.
    perm = xf._instantiate_preserving_type(xf._resolve_exception_class("PermissionError"), "denied")
    assert isinstance(perm, PermissionError)
    assert xf._is_retryable_download_error(perm) is False
    # An unrelated builtin (not OSError, not a Hub error name) is not resolved.
    assert xf._resolve_exception_class("ValueError") is None


def test_weight_pattern_selector_handles_globs(tmp_path):
    """The selector reads tokenizer/config/json globs as weightless but standard weight names and ?/[] globs as weight-bearing."""
    weightless = ["tokenizer*", "*.json", "config.json", "tokenizer.model", "*.txt"]
    weighty = [
        "model.safetensors", "*.safetensors", "model.?afetensors", "model.[sp]afetensors",
        "checkpoint-*/model.?afetensors", "unet/*", "adapter_model*", "consolidated*", "model.gguf",
    ]
    for pat in weightless:
        assert hcs.request_can_include_weights([pat], None) is False, pat
    for pat in weighty:
        assert hcs.request_can_include_weights([pat], None) is True, pat


def test_post_download_rejects_config_only_for_explicit_weight_pattern(tmp_path):
    """An explicit weight request returning only config.json is a stale config-only snapshot: reject and retry."""
    snap, _ = _mk_snapshot(tmp_path, "patcfg")
    (snap / "config.json").write_text("{}")
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = ["model.safetensors"], ignore_patterns = None) is False
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = ["adapter_model.safetensors"],
        ignore_patterns = None) is False


def test_post_download_rejects_incomplete_canonical_root_shards(tmp_path):
    """An interrupted canonical sharded warm (loose shard, no index) is rejected; a complete set is accepted; a variant-only layout does not satisfy a default load."""
    snap, blob = _mk_snapshot(tmp_path, "incshard")
    (snap / "config.json").write_text("{}")
    (snap / "model-00001-of-00002.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None) is False
    # Complete the set with its index -> accepted.
    (snap / "model-00002-of-00002.safetensors").symlink_to(blob)
    (snap / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors",
                                   "b": "model-00002-of-00002.safetensors"}}))
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None) is True
    # A variant-named shard is not a canonical weight a default load reads, so a variant-only cache is rejected.
    vsnap, vblob = _mk_snapshot(tmp_path, "vshard")
    (vsnap / "config.json").write_text("{}")
    (vsnap / "model-00001-of-00001.fp16.safetensors").symlink_to(vblob)
    assert xf._download_result_usable(
        vsnap, repo_type = "model", allow_patterns = None, ignore_patterns = None) is False


def test_local_token_not_found_error_type_preserved():
    """A missing required token fails identically over either transport, so LocalTokenNotFoundError is deterministic and type-preserved."""
    assert "LocalTokenNotFoundError" in xf._DETERMINISTIC_ERROR_NAMES
    cls = xf._resolve_exception_class("LocalTokenNotFoundError")
    assert cls is not None and issubclass(cls, BaseException)
    assert xf._is_retryable_download_error(
        xf._instantiate_preserving_type(cls, "LocalTokenNotFoundError: token required")) is False


def test_metadata_directory_pattern_is_weightless(tmp_path):
    """A trailing-slash metadata dir pattern (allow=['tokenizer/']) reads weightless; component/checkpoint dir patterns stay weight-bearing."""
    assert hcs.request_can_include_weights(["tokenizer/"], None) is False
    assert hcs.request_can_include_weights(["processor/"], None) is False
    assert hcs.request_can_include_weights(["unet/"], None) is True
    assert hcs.request_can_include_weights(["checkpoint-10/"], None) is True
    snap, _ = _mk_snapshot(tmp_path, "tokdir")
    (snap / "tokenizer").mkdir()
    (snap / "tokenizer" / "tokenizer.json").write_text("{}")
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = ["tokenizer/"], ignore_patterns = None) is True


def test_metadata_directory_glob_is_weightless(tmp_path):
    """A metadata-dir glob (allow=['tokenizer/*']) reads weightless like its trailing-slash form; a component/checkpoint dir glob stays weight-bearing."""
    assert hcs.request_can_include_weights(["tokenizer/*"], None) is False
    assert hcs.request_can_include_weights(["tokenizer/*.json"], None) is False
    assert hcs.request_can_include_weights(["processor/*"], None) is False
    assert hcs.request_can_include_weights(["unet/*"], None) is True
    assert hcs.request_can_include_weights(["checkpoint-10/*"], None) is True
    snap, _ = _mk_snapshot(tmp_path, "tokglob")
    (snap / "tokenizer").mkdir()
    (snap / "tokenizer" / "tokenizer.json").write_text("{}")
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = ["tokenizer/*"], ignore_patterns = None) is True


def test_allow_star_with_all_weights_ignored_is_weightless(tmp_path):
    """An allow the ignore filter strips of every weight reads weightless (root allow=['*'] and subdir allow=['unet/*']); surviving weight suffixes stay weight-bearing."""
    all_weight_ignores = [
        "*.safetensors", "*.bin", "*.pt", "*.pth", "*.gguf",
        "*.ckpt", "*.onnx", "*.msgpack", "*.h5", "*.pdparams",
    ]
    assert hcs.request_can_include_weights(["*"], all_weight_ignores) is False
    assert hcs.request_can_include_weights(["*"], None) is True
    # A subdir allow that ignores every weight suffix is weightless too...
    assert hcs.request_can_include_weights(["unet/*"], all_weight_ignores) is False
    # ...but one whose weight suffixes survive stays weight-bearing.
    assert hcs.request_can_include_weights(["unet/*"], ["*.bin"]) is True
    assert hcs.request_can_include_weights(["*.safetensors"], ["*.bin"]) is True
    snap, _ = _mk_snapshot(tmp_path, "cfgonly")
    (snap / "config.json").write_text("{}")
    (snap / "tokenizer.json").write_text("{}")
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = ["*"], ignore_patterns = all_weight_ignores) is True


def test_post_download_rejects_checkpoint_only_root_model(tmp_path):
    """A snapshot whose only weight is under checkpoint-7/ is rejected for an unpatterned root warm, but accepted when explicitly scoped."""
    snap, blob = _mk_snapshot(tmp_path, "ckonly")
    (snap / "config.json").write_text("{}")
    (snap / "checkpoint-7").mkdir()
    (snap / "checkpoint-7" / "model.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None) is False
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = ["checkpoint-7/*"], ignore_patterns = None) is True
    # A diffusers pipeline's subfolder weights still count (model_index.json gates that).
    dsnap, dblob = _mk_snapshot(tmp_path, "diff")
    (dsnap / "model_index.json").write_text("{}")
    (dsnap / "unet").mkdir()
    (dsnap / "unet" / "diffusion_pytorch_model.safetensors").symlink_to(dblob)
    assert xf._download_result_usable(
        dsnap, repo_type = "model", allow_patterns = None, ignore_patterns = None) is True
    # A diffusers snapshot whose only weight is under checkpoint-N/ (not a pipeline component) is rejected.
    dck, dckb = _mk_snapshot(tmp_path, "diff_ckpt")
    (dck / "model_index.json").write_text("{}")
    (dck / "checkpoint-7").mkdir()
    (dck / "checkpoint-7" / "diffusion_pytorch_model.safetensors").symlink_to(dckb)
    assert xf._download_result_usable(
        dck, repo_type = "model", allow_patterns = None, ignore_patterns = None) is False


def test_post_download_rejects_adapter_only_for_default_load(tmp_path):
    """A default warm reads the base weight, not an adapter, so an adapter-only snapshot is rejected; an adapter-scoped request still accepts it."""
    snap, blob = _mk_snapshot(tmp_path, "adapter_only_default")
    (snap / "adapter_model.safetensors").symlink_to(blob)
    (snap / "adapter_config.json").write_text("{}")
    (snap / "config.json").write_text("{}")
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None) is False
    # An adapter load (patterned) reads the adapter and accepts it.
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = ["adapter_model*", "adapter_config.json"],
        ignore_patterns = None) is True
    # The base weight present -> the default warm accepts, even beside the adapter.
    (snap / "model.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None) is True


def test_post_download_variant_root_check_ignores_adapter_index(tmp_path):
    """A variant load reads the root model variant, so a complete model.fp16 beside a stale adapter variant index is accepted; an incomplete root variant index is still rejected."""
    snap, blob = _mk_snapshot(tmp_path, "var_adapter_idx")
    (snap / "model.fp16.safetensors").symlink_to(blob)  # complete root model variant (the read weight)
    (snap / "adapter_model.safetensors.index.fp16.json").write_text(json.dumps(
        {"weight_map": {"a": "adapter_model.fp16-00001-of-00002.safetensors",
                        "b": "adapter_model.fp16-00002-of-00002.safetensors"}}))  # stale, shards absent
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None,
        variant = "fp16") is True
    # An incomplete root model variant index is still caught.
    snap2, blob2 = _mk_snapshot(tmp_path, "var_root_idx_incomplete")
    (snap2 / "model.safetensors.index.fp16.json").write_text(json.dumps(
        {"weight_map": {"a": "model.fp16-00001-of-00002.safetensors",
                        "b": "model.fp16-00002-of-00002.safetensors"}}))
    (snap2 / "model.fp16-00001-of-00002.safetensors").symlink_to(blob2)  # shard 2 absent
    assert xf._download_result_usable(
        snap2, repo_type = "model", allow_patterns = None, ignore_patterns = None,
        variant = "fp16") is False


def test_post_download_rejects_variant_only_root_for_default_load(tmp_path):
    """A default (no-variant) load reads the canonical name, not a variant name, so a variant-only snapshot is rejected; canonical names still pass."""
    snap, blob = _mk_snapshot(tmp_path, "var_only")
    (snap / "model.fp16.safetensors").symlink_to(blob)          # variant-named only
    (snap / "config.json").write_text("{}")
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None) is False
    # A sharded variant name is likewise not a default weight.
    snap_sh, blob_sh = _mk_snapshot(tmp_path, "var_only_sharded")
    (snap_sh / "model.fp16-00001-of-00002.safetensors").symlink_to(blob_sh)
    (snap_sh / "config.json").write_text("{}")
    assert xf._download_result_usable(
        snap_sh, repo_type = "model", allow_patterns = None, ignore_patterns = None) is False
    # The canonical weight present -> accepted, even beside the variant.
    (snap / "model.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None) is True
    # A variant load DOES read the variant weight.
    assert xf._download_result_usable(
        snap_sh, repo_type = "model", allow_patterns = None, ignore_patterns = None,
        variant = "fp16") is False  # sharded variant with no index is still incomplete
    snap_v, blob_v = _mk_snapshot(tmp_path, "var_single")
    (snap_v / "model.fp16.safetensors").symlink_to(blob_v)
    (snap_v / "config.json").write_text("{}")
    assert xf._download_result_usable(
        snap_v, repo_type = "model", allow_patterns = None, ignore_patterns = None,
        variant = "fp16") is True


def test_post_download_variant_either_format_exact_alternatives(tmp_path):
    """An exact request listing both variant formats is an alternative (like the canonical either-format pair): a safetensors-variant-only repo is complete; distinct/base+adapter still requires each."""
    snap, blob = _mk_snapshot(tmp_path, "var_either")
    (snap / "model.fp16.safetensors").symlink_to(blob)  # only the safetensors variant present
    assert xf._download_result_usable(
        snap, repo_type = "model",
        allow_patterns = ["model.fp16.safetensors", "pytorch_model.fp16.bin"],
        ignore_patterns = None, variant = "fp16") is True
    # The canonical either-format pair keeps working.
    snap_c, blob_c = _mk_snapshot(tmp_path, "canon_either")
    (snap_c / "pytorch_model.bin").symlink_to(blob_c)
    assert xf._download_result_usable(
        snap_c, repo_type = "model",
        allow_patterns = ["model.safetensors", "pytorch_model.bin"], ignore_patterns = None) is True
    # Base and adapter are distinct groups: adapter present, base absent -> rejected.
    snap_d, blob_d = _mk_snapshot(tmp_path, "base_and_adapter")
    (snap_d / "adapter_model.safetensors").symlink_to(blob_d)
    assert xf._download_result_usable(
        snap_d, repo_type = "model",
        allow_patterns = ["model.safetensors", "adapter_model.safetensors"],
        ignore_patterns = None) is False


def test_post_download_validates_weightless_named_subset(tmp_path):
    """An exact weightless request missing its named file is rejected and retried; a glob allow list stays lenient."""
    snap, _ = _mk_snapshot(tmp_path, "noname")
    (snap / "config.json").write_text("{}")
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = ["tokenizer.json"], ignore_patterns = None) is False
    assert xf._download_result_usable(
        snap, repo_type = "dataset", allow_patterns = ["data.parquet"], ignore_patterns = None) is False
    # Present named file -> accepted; a glob stays lenient.
    (snap / "tokenizer.json").write_text("{}")
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = ["tokenizer.json"], ignore_patterns = None) is True
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = ["*.json"], ignore_patterns = None) is True


def test_post_download_rejects_missing_exact_weight_request(tmp_path):
    """An exact weight request whose file is missing is rejected even with a different weight present; the either-format pair stays satisfied by one."""
    base, blob = _mk_snapshot(tmp_path, "baseonly")
    (base / "model.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        base, repo_type = "model", allow_patterns = ["adapter_model.safetensors"],
        ignore_patterns = None) is False
    assert xf._download_result_usable(
        base, repo_type = "model",
        allow_patterns = ["model.safetensors", "adapter_model.safetensors"], ignore_patterns = None) is False
    assert xf._download_result_usable(
        base, repo_type = "model",
        allow_patterns = ["model.safetensors", "pytorch_model.bin"], ignore_patterns = None) is True
    # Both present -> accepted.
    (base / "adapter_model.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        base, repo_type = "model",
        allow_patterns = ["model.safetensors", "adapter_model.safetensors"], ignore_patterns = None) is True


def test_dataset_unpatterned_or_glob_partial_does_not_skip_child(tmp_path):
    """A dataset/space snapshot whose completeness cannot be proven locally (allow=None or a glob) defers to the child; an intact exact-named subset short-circuits."""
    snap, _ = _mk_snapshot(tmp_path, "dspart")
    (snap / "README.md").write_text("partial")
    assert xf._cache_can_skip_download(
        snap, repo_type = "dataset", allow_patterns = None, ignore_patterns = None) is False
    assert xf._cache_can_skip_download(
        snap, repo_type = "dataset", allow_patterns = ["*.parquet"], ignore_patterns = None) is False
    assert xf._cache_can_skip_download(
        snap, repo_type = "dataset", allow_patterns = ["README.md"], ignore_patterns = None) is True


def test_http_prep_scopes_blob_cleanup_to_owned_partials(tmp_path):
    """HTTP prep purges only the child's own partials: with ownership known a sibling's aged partial/link is spared; with ownership None the mtime guard purges both."""
    repo = "ztest/concurrent-blobs"
    repo_dir = tmp_path / f"models--{repo.replace('/', '--')}"
    blobs = repo_dir / "blobs"
    snap = repo_dir / "snapshots" / "sha"
    blobs.mkdir(parents = True)
    snap.mkdir(parents = True)
    old = time.time() - 600

    def _seed():
        owned = blobs / "ownedhash.incomplete"
        sibling = blobs / "siblinghash.incomplete"
        owned.write_bytes(b"o")
        sibling.write_bytes(b"s")
        os.utime(owned, (old, old))
        os.utime(sibling, (old, old))
        for name in list(snap.iterdir()):
            name.unlink()
        (snap / "our.safetensors").symlink_to(blobs / "ownedhash")
        (snap / "sib.safetensors").symlink_to(blobs / "siblinghash")
        return owned, sibling

    owned, sibling = _seed()
    _REAL_DEFAULT_PREPARE(
        "model", repo, cache_dir = str(tmp_path), active_grace = 180,
        owned_incomplete_blobs = {"ownedhash.incomplete"})
    assert not owned.exists(), "our own stalled partial must be purged"
    assert sibling.exists(), "a concurrent sibling's partial must be spared"
    assert not (snap / "our.safetensors").is_symlink()
    assert (snap / "sib.safetensors").is_symlink(), "sibling's dangling link must be spared"

    # No ownership info -> coarse mtime guard purges both aged partials.
    owned, sibling = _seed()
    _REAL_DEFAULT_PREPARE(
        "model", repo, cache_dir = str(tmp_path), active_grace = 180, owned_incomplete_blobs = None)
    assert not owned.exists() and not sibling.exists()
