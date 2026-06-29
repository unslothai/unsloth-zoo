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


# A package placeholder so ``from unsloth_zoo.hf_cache_state import ...`` inside hf_xet_fallback
# resolves to the file we load below, not the installed package. RESTORE sys.modules afterwards: a
# leftover placeholder would shadow the REAL unsloth_zoo (its __init__ never runs) and fail a later
# test that imports it. The two loaded modules keep their own bound references, so they work after
# the placeholder is removed.
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


def test_file_watchdog_scopes_to_child_partial(hf_cache):
    """A single-file download follows only its own child's partials. A concurrent sibling
    download of a different file in the same repo (its partial already in flight, so in the
    baseline) keeps growing, but must not keep resetting this file's stall timer -- the
    constant child partial still fires."""
    blobs = _blobs_dir(hf_cache)
    sibling = blobs / "sibling.incomplete"   # already in flight -> captured in baseline
    sibling.write_bytes(b"\0" * 1024)
    baseline = {"sibling.incomplete"}

    grow_stop = threading.Event()

    def _grow():
        size = 1024
        while not grow_stop.wait(0.05):
            size += 4096
            sibling.write_bytes(b"\0" * size)   # healthy sibling keeps making progress

    grower = threading.Thread(target = _grow, daemon = True)
    grower.start()

    # This download's child writes its own constant (stalled) partial, not in the baseline.
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
    """Contrast for the single-file scoping: the default repo-wide measurement sums every
    blob, so a growing sibling resets the timer and a constant partial never trips. This is
    correct for snapshots (all blobs are one pull) and is exactly what file-scoping avoids."""
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
        repo_ids = [REPO], on_stall = calls.append, interval = 0.05, stall_timeout = 0.3,
    )
    try:
        time.sleep(1.0)   # well past stall_timeout, but repo-wide bytes keep growing
        assert calls == [], "repo-wide watchdog should be reset by the growing sibling"
    finally:
        stop.set()
        grow_stop.set()


def test_file_watchdog_ignores_baseline_only_partials(hf_cache):
    """If the only active partial is a baseline sibling's (this child has not written one
    yet), the file watchdog sees no owned progress and must not fire: there is nothing of
    ours to stall on, so post-spawn metadata/connect time is never misread as our stall."""
    blobs = _blobs_dir(hf_cache)
    (blobs / "sibling.incomplete").write_bytes(b"\0" * 4096)   # constant baseline sibling

    calls: list[str] = []
    stop = xf.start_watchdog(
        repo_ids = [REPO], on_stall = calls.append, interval = 0.05, stall_timeout = 0.2,
        watch_new_partials_only = True, baseline_incomplete_blobs = {"sibling.incomplete"},
    )
    try:
        time.sleep(0.8)
        assert calls == [], "file watchdog fired on a baseline sibling partial it must ignore"
    finally:
        stop.set()


def _spawn_holding_open(path: Path) -> "subprocess.Popen":
    """A real child process that opens *path* and holds it open without writing, modelling a
    hung download. Prints 'ok' once the file is open so the caller can synchronize."""
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
    """A resumed single-file download reuses the prior blob-hash .incomplete, so it sits in
    the baseline. Name-based exclusion would never flag a hung resume; scoping to the
    partials the child process holds open detects it."""
    blobs = _blobs_dir(hf_cache)
    partial = blobs / "resumed.incomplete"
    partial.write_bytes(b"\0" * 4096)        # leftover from a prior interrupted download
    baseline = {"resumed.incomplete"}        # present before the (resuming) child starts

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


def test_file_watchdog_pid_scope_ignores_unowned_sibling(hf_cache):
    """With pid scoping, a sibling partial this child does NOT hold open is ignored even if
    it grows, so the child's own constant partial still trips the stall."""
    blobs = _blobs_dir(hf_cache)
    owned_partial = blobs / "owned.incomplete"
    owned_partial.write_bytes(b"\0" * 2048)   # the child holds this open, constant (hung)
    sibling = blobs / "sibling.incomplete"
    sibling.write_bytes(b"\0" * 1024)

    grow_stop = threading.Event()

    def _grow():
        size = 1024
        while not grow_stop.wait(0.05):
            size += 4096
            sibling.write_bytes(b"\0" * size)   # an unrelated sibling making progress

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
    """hf_xet writes in-process and holds its .incomplete fd continuously, so an EMPTY child
    open-set means the child owns no partial YET (the connect / metadata phase), NOT that a
    helper process owns one. A concurrent sibling's post-baseline partial must therefore NOT be
    attributed to a still-connecting child -- otherwise a stalled sibling would kill it and force
    a needless HTTP retry. The precise empty-set branch owns nothing, so no stall fires."""
    blobs = _blobs_dir(hf_cache)
    # A sibling partial created after baseline (not name-excluded), constant (stalled).
    (blobs / "sibling.incomplete").write_bytes(b"\0" * 4096)
    monkeypatch.setattr(xf, "_child_open_incomplete_blobs", lambda pid: set())  # child owns none

    calls: list[str] = []
    stop = xf.start_watchdog(
        repo_ids = [REPO], on_stall = calls.append, interval = 0.05, stall_timeout = 0.2,
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


def test_blob_bytes_present_zero_blocks_is_zero(tmp_path):
    """A freshly truncated, fully-sparse .incomplete reports st_size > 0 with 0
    allocated blocks; it must count as 0 bytes present, not full size (a > 0 guard
    would mis-read an empty partial as complete)."""
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
    # (call the real impl; the autouse fixture stubs the module attribute). Age it
    # past the active-partial grace so it reads as a stalled, not in-flight, blob.
    old = time.time() - 600
    os.utime(partial, (old, old))
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


def test_prepare_for_http_spares_concurrent_sibling_active_symlink(tmp_path):
    """HTTP prep must NOT delete a concurrent sibling's dangling snapshot symlink while that sibling is
    still writing the target blob (a fresh .incomplete partner exists). Our own stale interrupted link
    (no .incomplete partner) is still cleared in the same sweep."""
    repo = "ztest/concurrent"
    repo_dir = tmp_path / f"models--{repo.replace('/', '--')}"
    blobs = repo_dir / "blobs"
    snap = repo_dir / "snapshots" / "sha"
    blobs.mkdir(parents = True)
    snap.mkdir(parents = True)

    # Sibling mid-download: a dangling link to a blob whose .incomplete partner is being written now.
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
    # Age both past the active-partial grace so the purge is exercised on stalled blobs
    # (lower is preserved by repo attribution, not mtime).
    old = time.time() - 600
    os.utime(upper_partial, (old, old))
    os.utime(lower_partial, (old, old))

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


def test_cancel_honored_even_when_file_cached(monkeypatch, tmp_path):
    """A cancel_event set before the call must raise even when the file is ALREADY cached: the
    warm-cache short-circuit returns without reaching _download_with_xet_fallback (the other
    cancel check), so it must honor cancellation first rather than hand back the cached path
    (Codex #829)."""
    cached = tmp_path / "cached.gguf"
    cached.write_bytes(b"\0" * 8)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: str(cached))
    fake = _install(monkeypatch, [])  # the attempt must never run
    ev = threading.Event()
    ev.set()
    with pytest.raises(RuntimeError, match = "Cancelled"):
        xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None, cancel_event = ev)
    assert fake.calls == []


def test_snapshot_cancel_honored_even_when_cached(monkeypatch, tmp_path):
    """The snapshot wrapper must also honor a pre-set cancel before its warm-cache short-circuit,
    so a cancelled request does not resolve a cached snapshot (Codex #829)."""
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "model.safetensors").write_bytes(b"x")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(snap))
    fake = _install(monkeypatch, [])  # the attempt must never run
    ev = threading.Event()
    ev.set()
    with pytest.raises(RuntimeError, match = "Cancelled"):
        xf.snapshot_download_with_xet_fallback(DL_REPO, token = None, cancel_event = ev)
    assert fake.calls == []


def test_nonstall_error_propagates_without_fallback(monkeypatch):
    fake = _install(monkeypatch, [("error", "RepositoryNotFoundError: 404 not found")])
    # A deterministic Hub error is re-raised with its ORIGINAL type preserved across the spawn
    # boundary (not flattened to a bare RuntimeError), so a caller's typed except clause still
    # matches (Codex #829). The parent reconstructs the class from the child's "<Name>: ..." prefix.
    expected_cls = xf._resolve_exception_class("RepositoryNotFoundError")
    assert expected_cls is not None and expected_cls is not RuntimeError
    with pytest.raises(expected_cls, match = "RepositoryNotFoundError"):
        xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None)
    assert len(fake.calls) == 1, "deterministic error must not trigger an HTTP fallback"
    assert fake.calls[0].disable_xet is False


def test_crashed_child_retries_over_http(monkeypatch):
    """A silent process-level crash (child exits without a result, e.g. a native hf_xet
    abort) is not a deterministic error, so it retries over HTTP; a clean second result is
    accepted."""
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
    """A transient Xet transport failure (CAS timeout / 5xx) is not a deterministic Hub error,
    so it retries over HTTP; a clean HTTP result is accepted (Codex #829)."""
    fake = _install(monkeypatch, [("retryable_error", "CasClientError: request timed out"), ("ok", "/cache/x")])
    out = xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None)
    assert out == "/cache/x"
    assert [c.disable_xet for c in fake.calls] == [False, True]


def test_retryable_xet_error_on_both_transports_raises(monkeypatch):
    """A transient error on Xet AND on HTTP has no other transport left, so it surfaces after
    both attempts rather than looping (Codex #829)."""
    fake = _install(monkeypatch, [("retryable_error", "503 Server Error"), ("retryable_error", "503 Server Error")])
    with pytest.raises(RuntimeError, match = "503"):
        xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None)
    assert [c.disable_xet for c in fake.calls] == [False, True]


def test_is_retryable_download_error_classification():
    """Transient transport failures (hf_xet / CAS, timeout, reset, HTTP 5xx / 429) are
    retryable; deterministic Hub errors (not-found, gated, 4xx, disk full) and unknown errors
    are not, so a real repeatable failure is surfaced rather than looped (Codex #829)."""
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

    # Deterministic Hub errors -> not retryable (matched by class name or 4xx status).
    class _Status416(Exception):
        status_code = 416

    assert f(_Status416("Range Not Satisfiable")) is False  # a retry would fail identically
    class RepositoryNotFoundError(Exception):
        pass

    assert f(RepositoryNotFoundError("404 Client Error")) is False

    class _Resp404(Exception):
        response = type("R", (), {"status_code": 404})()

    assert f(_Resp404("not found")) is False
    assert f(OSError(errno.ENOSPC, "No space left on device")) is False
    # An unknown / generic error stays deterministic -> surfaced, not looped over transports.
    assert f(ValueError("unexpected response payload")) is False


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
    assert fake.calls[1].disable_xet is True  # HTTP fallback
    assert prepared == [("model", DL_REPO)], "must prep cache for HTTP before the retry"


def test_injected_prepare_for_http_used(monkeypatch):
    """Studio injects its marker-aware prepare; the generic default must not run."""
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
    """A child that exits without enqueuing a result maps to 'crashed' (a process-level
    crash that HTTP may still recover), not a deterministic 'error'."""
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


def test_unrelated_partial_does_not_block_clean_cached_snapshot(hf_cache, monkeypatch):
    """A clean requested snapshot must short-circuit in-process even when the same
    repo cache holds a stale .incomplete from another (unrelated) revision: the fast
    path validates only the returned snapshot dir, not the whole repo, so a sibling
    mid-download does not force a needless re-fetch of a snapshot that is complete."""
    blobs = _blobs_dir(hf_cache, DL_REPO)
    repo_dir = blobs.parent
    snap = repo_dir / "snapshots" / "goodsha"
    snap.mkdir(parents = True)
    good = blobs / "good"
    good.write_bytes(b"weights")
    (snap / "model.safetensors").symlink_to(good)        # resolves -> snapshot is clean
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
def test_snapshot_fast_path_no_child(hf_cache, monkeypatch):
    """A fully cached repo (weights present) resolves in-process via local_files_only
    -- no child attempt."""
    blobs = _blobs_dir(hf_cache, DL_REPO)
    snap = blobs.parent / "snapshots" / "sha"
    snap.mkdir(parents = True)
    weight = blobs / "w"
    weight.write_bytes(b"\0" * 16)
    (snap / "model.safetensors").symlink_to(weight)   # weights present -> complete
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
    """Weight presence drives completeness: a config-only snapshot is incomplete; one
    with a resolvable weight file is complete."""
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
    """An interrupted multi-shard pull with NO index sidecar reads as incomplete. While the shards
    are partial, the numbered shard name itself states the full set, so missing siblings are
    detectable without a manifest. But even with EVERY shard on disk, a full warm is still
    incomplete until model.safetensors.index.json is present: transformers' local from_pretrained
    resolves a directory by probing model.safetensors then model.safetensors.index.json (never by
    globbing model-*-of-*.safetensors), so a sharded checkpoint without its index raises rather than
    loads, and the missing index would otherwise be fetched in-process over Xet."""
    snap = tmp_path / "snap"
    snap.mkdir()
    blob = tmp_path / "blob"
    blob.write_bytes(b"x")
    (snap / "model-00001-of-00003.safetensors").symlink_to(blob)
    assert hcs.snapshot_dir_is_complete(snap) is False  # shards 2 and 3 missing
    (snap / "model-00002-of-00003.safetensors").symlink_to(blob)
    assert hcs.snapshot_dir_is_complete(snap) is False  # shard 3 still missing
    (snap / "model-00003-of-00003.safetensors").symlink_to(blob)
    assert hcs.snapshot_dir_is_complete(snap) is False  # all shards present but no index sidecar
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
    """Trainer / optimizer state files carry weight suffixes (.bin/.pt/.pth) but are not
    loadable model weights. A checkpoint cache holding only those must read as incomplete
    so the killable child still fetches the real weights."""
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
    """A per-checkpoint shard index with missing shards must not fail an unpatterned root warm:
    the root weights are what the load reads, so an incomplete checkpoint index is irrelevant to
    root completeness (and a complete root weight set is enough)."""
    snap = tmp_path / "snap"
    (snap / "checkpoint-7").mkdir(parents = True)
    blob = tmp_path / "blob"
    blob.write_bytes(b"x")
    (snap / "model.safetensors").symlink_to(blob)                    # complete root weight
    # An incomplete checkpoint shard index (shard 2 missing) lives under checkpoint-7/.
    (snap / "checkpoint-7" / "model-00001-of-00002.safetensors").symlink_to(blob)
    (snap / "checkpoint-7" / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors",
                                   "b": "model-00002-of-00002.safetensors"}})
    )
    # Root warm is complete: the checkpoint index is skipped, the root weight suffices.
    assert hcs.snapshot_dir_is_complete(snap) is True


def test_fast_path_rejects_config_only_snapshot(hf_cache, monkeypatch):
    """HF's local_files_only returns a config-only snapshot (e.g. left by an earlier
    AutoConfig fetch) without checking weights. The fast path must reject it and complete
    the download in the killable child rather than load with missing weights."""
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
    """The pre-download cache short-circuit must not accept a stale snapshot holding only one
    of several explicitly named weights: base + adapter requested, only the base cached -> the
    guarded child still runs to fetch the rest (Codex #829)."""
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
    """A real but broken child snapshot result (HF offline-fallback returning a dir with
    dangling symlinks) is rejected on the Xet attempt and retried over HTTP; a clean
    second result is accepted."""
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
    """If even the HTTP attempt returns a broken snapshot, fail loudly rather than hand
    missing files to the load."""
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "model.safetensors").symlink_to(tmp_path / "missing")
    fake = _install(monkeypatch, [("ok", str(broken)), ("ok", str(broken))])
    with pytest.raises(xf.DownloadStallError, match = "incomplete snapshot"):
        xf.snapshot_download_with_xet_fallback(DL_REPO, token = None)
    assert [c.disable_xet for c in fake.calls] == [False, True]


def test_child_weight_incomplete_snapshot_retries_over_http(monkeypatch, tmp_path):
    """A child result with no weight files (HF silently returning a stale config-only
    snapshot on an offline / timed-out request) is rejected on the Xet attempt and retried
    over HTTP; a complete second result is accepted. The helper warms model repos, so a
    weight-less result means the download did not finish, not that the repo is weightless."""
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
    """A patterned download (allow_patterns) legitimately returns only the requested files
    (e.g. config / tokenizer, no model weights). The child result must be accepted as-is,
    not rejected and retried for lacking weights."""
    cfg_only = tmp_path / "cfg"
    cfg_only.mkdir()
    (cfg_only / "config.json").write_text("{}")   # exactly what was requested, no weights
    fake = _install(monkeypatch, [("ok", str(cfg_only))])
    out = xf.snapshot_download_with_xet_fallback(
        DL_REPO, token = None, allow_patterns = ["config.json"]
    )
    assert out == str(cfg_only) and len(fake.calls) == 1


def test_dataset_snapshot_without_weights_is_accepted(monkeypatch, tmp_path):
    """A non-model snapshot (repo_type='dataset') has no model weights by nature; its
    child result must be accepted rather than retried/raised as 'incomplete'."""
    files = tmp_path / "ds"
    files.mkdir()
    (files / "data.json").write_text("[]")
    fake = _install(monkeypatch, [("ok", str(files))])
    out = xf.snapshot_download_with_xet_fallback(DL_REPO, token = None, repo_type = "dataset")
    assert out == str(files) and len(fake.calls) == 1


def test_model_snapshot_with_weights_excluded_is_accepted(monkeypatch, tmp_path):
    """A model repo fetched with ignore_patterns that drop every weight format (e.g. to
    warm only config / tokenizer files) legitimately yields a weightless snapshot; the
    result must be accepted, not rejected for lacking weights."""
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
    """Unsloth's default prefetch ignores (onnx/h5/msgpack/gguf, never safetensors) still
    count as including weights, so model warmups keep requiring them; excluding every
    weight format does not."""
    assert hcs.request_can_include_weights(None, None) is True
    assert hcs.request_can_include_weights(None, ["*.onnx", "*.h5", "*.msgpack", "*.gguf"]) is True
    assert hcs.request_can_include_weights(["config.json"], None) is False
    assert hcs.request_can_include_weights(
        None, ["*.safetensors", "*.bin", "*.h5", "*.msgpack", "*.gguf",
               "*.pt", "*.pth", "*.ckpt", "*.onnx", "*.pdparams", "*.index.json"]
    ) is False


def test_request_can_include_weights_index_json_only():
    """A metadata-only request that matches the shard *index* sidecars but no real weight
    file must read as weightless: the index is JSON, not weights, so a JSON-only warmup
    (allow_patterns=['*.json'] or ['*.index.json']) should not be forced to land shards."""
    assert hcs.request_can_include_weights(["*.json"], None) is False
    assert hcs.request_can_include_weights(["*.index.json"], None) is False
    assert hcs.request_can_include_weights(
        ["model.safetensors.index.json", "pytorch_model.bin.index.json"], None
    ) is False
    # A real weight pattern still counts as including weights.
    assert hcs.request_can_include_weights(["*.safetensors"], None) is True


def test_request_can_include_weights_path_qualified():
    """Path-qualified allow_patterns must be resolved inside their directory, and a bare
    non-first shard recognized, so a subfolder / checkpoint / specific-shard weight request
    is not misread as weightless (which would skip the killable child)."""
    # Concrete subfolder globs: weights live under the directory.
    assert hcs.request_can_include_weights(["checkpoint-10/*"], None) is True
    assert hcs.request_can_include_weights(["checkpoint-10/*.safetensors"], None) is True
    assert hcs.request_can_include_weights(["models/*.bin"], None) is True
    # A specific (non-first) shard named verbatim.
    assert hcs.request_can_include_weights(["model-00002-of-00005.safetensors"], None) is True
    assert hcs.request_can_include_weights(["checkpoint-10/pytorch_model.bin"], None) is True
    # Globbed parent dir, weight-targeting basename -> can include weights.
    assert hcs.request_can_include_weights(["checkpoint-*/*.safetensors"], None) is True
    # Subfolder requests that target only non-weight files stay weightless.
    assert hcs.request_can_include_weights(["checkpoint-10/config.json"], None) is False
    assert hcs.request_can_include_weights(["checkpoint-10/*.json"], None) is False
    assert hcs.request_can_include_weights(["checkpoint-*/tokenizer.json"], None) is False
    # The unsloth subfolder warmup shape: "<sub>/*" plus root aux files -> weights expected.
    assert hcs.request_can_include_weights(
        ["checkpoint-10/*", "config.json", "tokenizer.json"], None
    ) is True


def test_request_can_include_weights_path_qualified_custom_globs():
    """A path-qualified custom weight glob (checkpoint-10/lora_*.safetensors, with a globbed
    parent too) names a weight whose basename matches no canonical probe; it must read as
    weight-including via a concretized self-probe, not weightless (Codex #829)."""
    assert hcs.request_can_include_weights(["checkpoint-10/lora_*.safetensors"], None) is True
    assert hcs.request_can_include_weights(["checkpoint-*/lora_*.bin"], None) is True
    assert hcs.request_can_include_weights(["models/custom_*.pt"], None) is True
    assert hcs.request_can_include_weights(["checkpoint-10/model-[0-9].safetensors"], None) is True
    # A non-weight basename under a subfolder stays weightless.
    assert hcs.request_can_include_weights(["checkpoint-10/tokenizer.json"], None) is False


def test_request_can_include_weights_empty_allow_list(tmp_path):
    """allow_patterns=[] is a real filter that selects NO objects (Hugging Face semantics), so
    the request is weightless -- it must not collapse with None (an unfiltered warmup) and
    reject a legitimately empty snapshot (Codex #829). ignore_patterns=[] ignores nothing, so
    it stays weight-including."""
    assert hcs.request_can_include_weights([], None) is False
    assert hcs.request_can_include_weights(None, None) is True
    assert hcs.request_can_include_weights(None, []) is True
    assert hcs.request_can_include_weights([], []) is False
    # snapshot_dir_is_complete agrees: allow=[] is a scoped (select-nothing) request, not a full
    # warmup, so a snapshot carrying an unrelated weight is not read as complete for it.
    snap = tmp_path / "snap"
    snap.mkdir()
    blob = tmp_path / "blob"
    blob.write_bytes(b"x")
    (snap / "model.safetensors").symlink_to(blob)
    assert hcs.snapshot_dir_is_complete(snap, allow_patterns = []) is False
    assert hcs.snapshot_dir_is_complete(snap, allow_patterns = None) is True


def test_request_can_include_weights_string_form():
    """Hugging Face accepts allow / ignore patterns as a bare string; it must be treated as
    one pattern, not iterated character by character (which would misclassify a subfolder
    weight request as weightless)."""
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
    """The generic HTTP-prep purge must not unlink a concurrent download's still-active
    .incomplete temp file: only stale (old-mtime) partials are removed, so a sibling
    download of another file in the same repo keeps writing safely."""
    blobs = _blobs_dir(hf_cache, DL_REPO)
    stale = blobs / "stalled.incomplete"
    stale.write_bytes(b"\0" * 16)
    active = blobs / "sibling.incomplete"
    active.write_bytes(b"\0" * 16)
    # Age the stalled partial well past the active-partial grace; leave the sibling current.
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


# --------------------------------------------------------------------------- #
# Exported Xet knobs + child-leak safety + malformed-index resilience.
# --------------------------------------------------------------------------- #
def test_xet_availability_and_disable_helpers(monkeypatch):
    """The exported Xet knobs: child_should_disable_xet reads the per-worker config flag;
    xet_force_disabled honors every documented env knob; is_hf_xet_available reflects the
    importability of hf_xet and treats a probe error as 'unavailable'."""
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
    assert xf.is_hf_xet_available() is False  # a probe exception -> treated as unavailable


def test_run_attempt_terminates_child_if_watchdog_start_raises(monkeypatch):
    """If start_watchdog raises (e.g. thread/FD exhaustion: 'can't start new thread') AFTER the
    download child has already spawned, the child must STILL be reaped -- no leaked process. The
    error then propagates (a watchdog-start failure is not a transport fault to retry over HTTP)."""
    rec = {"terminated": False}

    class _AliveProc:
        def __init__(self):
            self.pid = None  # None -> _terminate_process_group skips killpg, uses terminate()
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


# --------------------------------------------------------------------------- #
# Codex review round: scoped completeness, weightless named files, type preservation.
# --------------------------------------------------------------------------- #


def test_requested_named_files_present_exact_request(tmp_path):
    """An EXACT-named weightless request (allow=['tokenizer.json'], no glob) requires its named file
    on disk; a config-only snapshot must not pass. A glob list or no allow_patterns is best-effort
    (Codex #829)."""
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
    # An ignore-filtered name is not actually requested, so its absence does not fail.
    assert hcs.requested_named_files_present(
        snap, allow_patterns = ["tokenizer.json", "absent.json"], ignore_patterns = ["absent.json"]
    ) is True


def test_deterministic_oserror_type_preserved(monkeypatch):
    """A deterministic disk-full OSError is re-raised as OSError (not flattened to RuntimeError), so a
    caller's `except OSError` cleanup still runs across the spawn boundary (Codex #829)."""
    fake = _install(monkeypatch, [("error", "OSError: [Errno 28] No space left on device")])
    with pytest.raises(OSError, match = "No space left"):
        xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None)
    assert len(fake.calls) == 1, "a deterministic error must not trigger an HTTP fallback"


def test_unknown_error_falls_back_to_runtimeerror(monkeypatch):
    """An unrecognized error class name still surfaces (as RuntimeError, the prior behavior) without
    a transport fallback -- only KNOWN deterministic Hub / OS types are reconstructed (Codex #829)."""
    fake = _install(monkeypatch, [("error", "SomeWeirdError: kaboom")])
    with pytest.raises(RuntimeError, match = "kaboom"):
        xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None)
    assert len(fake.calls) == 1


def test_resolve_exception_class_maps_known_names():
    """The reconstruction map resolves the documented deterministic Hub error names + OSError, and
    returns None (-> RuntimeError) for an unknown name (Codex #829)."""
    assert xf._resolve_exception_class("OSError") is OSError
    cls = xf._resolve_exception_class("RepositoryNotFoundError")
    assert cls is not None and issubclass(cls, BaseException)
    assert xf._resolve_exception_class("NotARealErrorType") is None


def test_error_type_preserved_when_constructor_needs_kwarg(monkeypatch):
    """A Hub error class whose constructor rejects a lone positional string (newer huggingface_hub
    makes HfHubHTTPError's `response` required / keyword-only) must STILL be re-raised with its type
    preserved -- via an __init__-bypassing reconstruction -- not silently downgraded to RuntimeError
    (Codex #829)."""
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
    """Direct coverage of the layered reconstruction: a normal constructor is used when it accepts a
    string; a keyword-only-required constructor falls through to the __new__ bypass; both yield an
    instance of the requested type carrying the message (Codex #829)."""
    class Plain(Exception):
        pass

    class Picky(Exception):
        def __init__(self, message, *, response):
            super().__init__(message)

    for cls in (Plain, Picky):
        exc = xf._instantiate_preserving_type(cls, "the message")
        assert isinstance(exc, cls)
        assert "the message" in str(exc)


# --------------------------------------------------------------------------- #
# Codex round: dir/ wildcard, logical-weight grouping post-download, errno preservation.
# --------------------------------------------------------------------------- #


def test_parse_errno():
    assert xf._parse_errno("OSError: [Errno 28] No space left on device") == 28
    assert xf._parse_errno("OSError: [Errno 122] Disk quota exceeded") == 122
    assert xf._parse_errno("OSError: some message with no errno") is None


def test_oserror_errno_preserved(monkeypatch):
    """A disk-full child OSError keeps its errno (ENOSPC) across the spawn boundary, so a caller's
    `except OSError` cleanup can still branch on exc.errno -- not see errno=None (Codex #829)."""
    import errno as _errno

    fake = _install(monkeypatch, [("error", "OSError: [Errno 28] No space left on device")])
    with pytest.raises(OSError) as excinfo:
        xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None)
    assert excinfo.value.errno == _errno.ENOSPC
    assert len(fake.calls) == 1, "a deterministic error must not trigger an HTTP fallback"


# --------------------------------------------------------------------------- #
# Spawn-safety regressions: failed-spawn queue cleanup + disable-Xet env-race lock.
# --------------------------------------------------------------------------- #


def test_failed_spawn_closes_result_queue(monkeypatch):
    """R2-2: if proc.start() raises (e.g. OSError "can't start new process" under fd / thread
    exhaustion), the result_queue's OS pipe fds -- allocated before the spawn -- must be closed
    rather than leaked. The lifecycle try/finally that closes them is only entered after a
    successful start, so a dedicated except around the spawn must close the queue and re-raise."""
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
    """R2-1: _download_with_xet_fallback must read xet_force_disabled() while holding
    _SPAWN_ENV_LOCK. A concurrent download briefly sets the child-only HF_HUB_DISABLE_XET=1 in the
    parent os.environ around its spawn (under the same lock); reading the live env outside the lock
    could observe that value and wrongly force THIS download onto HTTP from the first attempt."""
    seen = {}
    real = xf.xet_force_disabled

    def _spy():
        # A plain (non-reentrant) Lock cannot be re-acquired by its owner, so a non-blocking acquire
        # FAILS iff the read is happening inside `with _SPAWN_ENV_LOCK:`. If the read were outside the
        # lock, the acquire would succeed.
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


# --------------------------------------------------------------------------- #
# Conservative fast-path gate + pre/post-download acceptance split (PR #829 trim).
# The completeness gate is intentionally narrow: it fast-paths ONLY the unambiguous
# canonical model cache, deferring everything else to the watched snapshot_download
# child. The pre-download (skip the child?) and post-download (accept the result?)
# checks are deliberately asymmetric -- strict pre, lenient post.
# --------------------------------------------------------------------------- #
def _mk_snapshot(tmp_path, name):
    blob = tmp_path / "_blob"
    if not blob.exists():
        blob.write_bytes(b"w")
    snap = tmp_path / name
    snap.mkdir()
    return snap, blob


def test_gate_fast_paths_canonical_single_file(tmp_path):
    """A complete, unpatterned single-file model cache is fast-path eligible (skip the child)."""
    snap, blob = _mk_snapshot(tmp_path, "single")
    (snap / "model.safetensors").symlink_to(blob)
    (snap / "config.json").write_text("{}")
    assert hcs.snapshot_dir_is_complete(snap) is True


def test_gate_fast_paths_canonical_sharded_with_index(tmp_path):
    """A complete sharded model with its index sidecar is fast-path eligible; without the index, or
    with a listed shard missing, it is not (transformers cannot load a local sharded checkpoint
    without a complete index)."""
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


def test_gate_rejects_config_only(tmp_path):
    snap, _ = _mk_snapshot(tmp_path, "cfg")
    (snap / "config.json").write_text("{}")
    assert hcs.snapshot_dir_is_complete(snap) is False


def test_gate_rejects_diffusers_marker(tmp_path):
    """A diffusers pipeline (root model_index.json) is never fast-pathed -> defer to the child,
    even when a root-level weight happens to be present."""
    snap, blob = _mk_snapshot(tmp_path, "diff")
    (snap / "model_index.json").write_text("{}")
    (snap / "model.safetensors").symlink_to(blob)
    assert hcs.snapshot_dir_is_complete(snap) is False


def test_gate_rejects_any_allow_pattern(tmp_path):
    """Any allow_patterns makes the request non-trivial -> defer to the child (no fast-path)."""
    snap, blob = _mk_snapshot(tmp_path, "pat")
    (snap / "model.safetensors").symlink_to(blob)
    assert hcs.snapshot_dir_is_complete(snap, allow_patterns = ["*.safetensors"]) is False
    assert hcs.snapshot_dir_is_complete(snap, allow_patterns = ["model.safetensors"]) is False


def test_gate_eligible_under_ignore_patterns(tmp_path):
    """allow=None with ANY ignore patterns stays fast-path eligible: the canonical-weight presence
    check verifies the surviving root weight is on disk, so ignores that drop other formats cannot
    make an incomplete cache read complete. This covers the common bare from_pretrained warm, whose
    real ignore list mixes root-level format excludes (*.onnx, *.gguf, *.pt, *.bin) with subdir
    (*/*.safetensors) drops."""
    snap, blob = _mk_snapshot(tmp_path, "ign")
    (snap / "model.safetensors").symlink_to(blob)
    assert hcs.snapshot_dir_is_complete(snap, ignore_patterns = ["*/*.safetensors", "*/*.bin"]) is True
    assert hcs.snapshot_dir_is_complete(snap, ignore_patterns = ["*.onnx"]) is True
    # The actual unsloth bare-from_pretrained combined ignore list (root-level format excludes +
    # subdir-weight drops) -- the warm root model.safetensors must still fast-path.
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
    """An ignore-only request is weightless ONLY when it strips EVERY weight format. A partial strip
    -- only the canonical model.safetensors/pytorch_model.bin names while a variant survives, or only
    some suffixes while a .pt / .gguf weight survives -- must read as weight-bearing, so the fast path
    requires a real weight and never skips the protective child on a config-only cache (the Xet hang)."""
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


def test_pre_download_does_not_skip_diffusers_but_post_accepts(tmp_path):
    """The pre/post asymmetry: a diffusers warm is NOT fast-pathed (spawn the child), but the same
    complete diffusers result IS accepted post-download (it has component weights), so a good
    download is never re-looped into a stall error."""
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
    """A model warm that came back with NO weight (HF handed back a stale config-only snapshot) is
    rejected post-download and retried over HTTP."""
    snap, _ = _mk_snapshot(tmp_path, "cfg")
    (snap / "config.json").write_text("{}")
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = None, ignore_patterns = None) is False


def test_post_download_accepts_dataset_without_weight(tmp_path):
    snap, blob = _mk_snapshot(tmp_path, "ds")
    (snap / "data.parquet").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "dataset", allow_patterns = None, ignore_patterns = None) is True


def test_post_download_accepts_either_format_single_present(tmp_path):
    """An either-format named request (['pytorch_model.bin','model.safetensors']) against a repo that
    ships only safetensors: the finished download has a weight, so it is accepted -- not re-looped
    for the absent .bin the repo simply does not publish."""
    snap, blob = _mk_snapshot(tmp_path, "either")
    (snap / "model.safetensors").symlink_to(blob)
    assert xf._download_result_usable(
        snap, repo_type = "model",
        allow_patterns = ["pytorch_model.bin", "model.safetensors"], ignore_patterns = None) is True


def test_pre_download_skips_intact_tokenizer_only(tmp_path):
    """A tokenizer-only (weightless) warm short-circuits offline: an intact requested subset is
    enough, no weight required."""
    snap, _ = _mk_snapshot(tmp_path, "tok")
    (snap / "tokenizer.json").write_text("{}")
    (snap / "config.json").write_text("{}")
    assert xf._cache_can_skip_download(
        snap, repo_type = "model",
        allow_patterns = ["tokenizer.json", "config.json"], ignore_patterns = None) is True


def test_pre_download_partial_ignore_does_not_skip_config_only(tmp_path):
    """Over-accept guard (safety reviewer finding): an ignore-only request stripping only SOME weight
    formats (ignore=['*.safetensors','*.bin']) on a config-only cache must NOT skip the child -- a
    repo whose surviving weight is e.g. model.gguf / model.fp16.safetensors / a .pt checkpoint would
    otherwise be fetched in-process over un-killable Xet (the hang)."""
    snap, _ = _mk_snapshot(tmp_path, "cfgign")
    (snap / "config.json").write_text("{}")
    assert xf._cache_can_skip_download(
        snap, repo_type = "model", allow_patterns = None,
        ignore_patterns = ["*.safetensors", "*.bin"]) is False


# ---------------------------------------------------------------------------
# Review-round regression guards (10-reviewer findings)
# ---------------------------------------------------------------------------

def test_gate_rejects_malformed_shard_index(tmp_path):
    """Finding 2 (over-accept): a truncated / non-dict / empty weight-shard index must NOT read as
    complete. _weight_shard_index_complete is fail-CLOSED so the fast path defers a malformed index
    to the watched child rather than skipping it and failing the in-process load on the bad index."""
    snap, blob = _mk_snapshot(tmp_path, "malidx")
    (snap / "model-00001-of-00002.safetensors").symlink_to(blob)
    (snap / "model.safetensors.index.json").write_text("{not valid json")
    assert hcs.snapshot_dir_is_complete(snap) is False
    # Empty weight_map proves nothing.
    snap2, blob2 = _mk_snapshot(tmp_path, "emptyidx")
    (snap2 / "model-00001-of-00002.safetensors").symlink_to(blob2)
    (snap2 / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    assert hcs.snapshot_dir_is_complete(snap2) is False
    # weight_map present but not a dict.
    snap3, blob3 = _mk_snapshot(tmp_path, "listidx")
    (snap3 / "model.safetensors.index.json").write_text(json.dumps({"weight_map": ["a", "b"]}))
    assert hcs._weight_shard_index_complete(snap3 / "model.safetensors.index.json") is False


def test_gate_ignored_canonical_weight_does_not_prove_complete(tmp_path):
    """Finding 3 (over-accept): a stale canonical weight whose FORMAT the request ignores must not
    count as proof of completeness. ignore=['*.bin'] with only a pytorch_model.bin on disk (no
    safetensors) defers to the child, so a use_safetensors load cannot silently fetch over Xet."""
    snap, blob = _mk_snapshot(tmp_path, "ignbin")
    (snap / "config.json").write_text("{}")
    (snap / "pytorch_model.bin").symlink_to(blob)
    assert hcs.snapshot_dir_is_complete(snap, ignore_patterns = ["*.bin"]) is False
    # Without the ignore, the present .bin is what a default load reads -> complete.
    assert hcs.snapshot_dir_is_complete(snap) is True
    # A .bin shard index is also discarded when *.bin is ignored (its .json sidecar would slip the
    # raw name filter, but the format probe catches it).
    snap2, blob2 = _mk_snapshot(tmp_path, "ignbinshard")
    (snap2 / "pytorch_model-00001-of-00001.bin").symlink_to(blob2)
    (snap2 / "pytorch_model.bin.index.json").write_text(
        json.dumps({"weight_map": {"a": "pytorch_model-00001-of-00001.bin"}}))
    assert hcs.snapshot_dir_is_complete(snap2, ignore_patterns = ["*.bin"]) is False
    assert hcs.snapshot_dir_is_complete(snap2) is True
    # A safetensors warm survives an *.bin ignore (the common bare from_pretrained case).
    snap3, blob3 = _mk_snapshot(tmp_path, "stignbin")
    (snap3 / "config.json").write_text("{}")
    (snap3 / "model.safetensors").symlink_to(blob3)
    assert hcs.snapshot_dir_is_complete(snap3, ignore_patterns = ["*.bin"]) is True


def test_post_download_accepts_weightless_patterned_result(tmp_path):
    """Finding 1 (over-reject): a genuinely weightless PATTERNED result (e.g. allow=['tokenizer*'])
    must be accepted post-download -- the caller scoped it, so 'no weight' is intended, not a stale
    config-only snapshot. Rejecting it would loop into a spurious DownloadStallError on a good
    download. The no-weight rejection stays in force for an UNPATTERNED model warm."""
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
    """codex :269 (over-accept): a variant-only shard index (model.safetensors.index.fp16.json) must
    NOT satisfy the canonical allow=None fast path -- the fallback wrapper takes no variant param, so
    a default load probes the canonical index whose weights are absent. Only a canonical
    (non-variant) index counts; the variant layout defers to the watched child."""
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
    """codex :499: a deterministic 4xx surfaced as a bare HfHubHTTPError keeps its TYPE across the
    spawn boundary (so `except HfHubHTTPError` still works) WITHOUT joining the retry-deterministic
    name shortcut -- a transient 5xx bare HfHubHTTPError must still retry over HTTP via its status
    code."""
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
    """Finding 4: a malformed repo id fails identically over either transport, so HFValidationError is
    deterministic (not retried) and its TYPE is reconstructed across the spawn boundary instead of
    degrading to a generic RuntimeError."""
    assert "HFValidationError" in xf._DETERMINISTIC_ERROR_NAMES
    cls = xf._resolve_exception_class("HFValidationError")
    assert cls is not None and issubclass(cls, BaseException)
    inst = xf._instantiate_preserving_type(cls, "HFValidationError: bad repo id")
    assert type(inst).__name__ == "HFValidationError"
    assert xf._is_retryable_download_error(inst) is False


def test_weight_pattern_selector_handles_globs(tmp_path):
    """The weight-pattern selector reads tokenizer / config / json globs as weightless (keeps their
    offline short-circuit) but classifies standard weight names and ? / [] globs as weight-bearing."""
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
    """An explicit weight request (allow=['model.safetensors']) returning only config.json is a stale
    config-only snapshot: reject and retry over HTTP. A weightless patterned request stays accepted."""
    snap, _ = _mk_snapshot(tmp_path, "patcfg")
    (snap / "config.json").write_text("{}")
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = ["model.safetensors"], ignore_patterns = None) is False
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = ["adapter_model.safetensors"],
        ignore_patterns = None) is False


def test_post_download_rejects_incomplete_canonical_root_shards(tmp_path):
    """An interrupted canonical sharded warm (loose model-00001-of-00002.safetensors, no index) has a
    loadable file but a default load cannot read it and would fetch the rest over un-killable Xet, so
    it is rejected. A complete sharded set is accepted; a variant-only shard layout is not force-failed."""
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
    # A variant-only shard layout has no canonical shard -> not force-failed here.
    vsnap, vblob = _mk_snapshot(tmp_path, "vshard")
    (vsnap / "config.json").write_text("{}")
    (vsnap / "model-00001-of-00001.fp16.safetensors").symlink_to(vblob)
    assert xf._download_result_usable(
        vsnap, repo_type = "model", allow_patterns = None, ignore_patterns = None) is True


def test_local_token_not_found_error_type_preserved():
    """A missing required token fails identically over either transport, so LocalTokenNotFoundError is
    deterministic and its type is reconstructed across the spawn boundary."""
    assert "LocalTokenNotFoundError" in xf._DETERMINISTIC_ERROR_NAMES
    cls = xf._resolve_exception_class("LocalTokenNotFoundError")
    assert cls is not None and issubclass(cls, BaseException)
    assert xf._is_retryable_download_error(
        xf._instantiate_preserving_type(cls, "LocalTokenNotFoundError: token required")) is False


def test_metadata_directory_pattern_is_weightless(tmp_path):
    """A trailing-slash metadata dir pattern (allow=['tokenizer/']) reads weightless, so a complete
    tokenizer-only download is accepted. Component / checkpoint dir patterns stay weight-bearing."""
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
    """A metadata-dir GLOB (allow=['tokenizer/*'], 'processor/*.json') reads weightless like its
    trailing-slash form, so a complete tokenizer-only download is accepted instead of looped into a
    DownloadStallError. A component / checkpoint dir glob stays weight-bearing."""
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
    """A root-reachable allow that the ignore filter strips of every weight (allow=['*'] +
    ignore=[every weight suffix]) reads weightless, so a complete config-only download is accepted, not
    looped into a DownloadStallError. A subdir-scoped allow stays weight-bearing, and an allow whose
    weights survive the ignore stays weight-bearing."""
    all_weight_ignores = [
        "*.safetensors", "*.bin", "*.pt", "*.pth", "*.gguf",
        "*.ckpt", "*.onnx", "*.msgpack", "*.h5", "*.pdparams",
    ]
    assert hcs.request_can_include_weights(["*"], all_weight_ignores) is False
    assert hcs.request_can_include_weights(["*"], None) is True
    assert hcs.request_can_include_weights(["unet/*"], all_weight_ignores) is True
    assert hcs.request_can_include_weights(["*.safetensors"], ["*.bin"]) is True
    snap, _ = _mk_snapshot(tmp_path, "cfgonly")
    (snap / "config.json").write_text("{}")
    (snap / "tokenizer.json").write_text("{}")
    assert xf._download_result_usable(
        snap, repo_type = "model", allow_patterns = ["*"], ignore_patterns = all_weight_ignores) is True


def test_post_download_rejects_checkpoint_only_root_model(tmp_path):
    """A stale snapshot whose only weight is under checkpoint-7/ is rejected for an unpatterned root
    warm -- a default from_pretrained ignores checkpoint-*/ and would fetch the missing root weights
    over un-killable Xet. The same checkpoint is accepted when explicitly scoped."""
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


def test_post_download_validates_weightless_named_subset(tmp_path):
    """An exact weightless request (allow=['tokenizer.json'], or a dataset file) returning a stale
    snapshot missing the named file is rejected and retried. A glob allow list stays lenient."""
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
    """An exact weight request whose file is missing is rejected even when a different weight is present:
    allow=['adapter_model.safetensors'] is NOT satisfied by a stale base model.safetensors, and
    ['model.safetensors','adapter_model.safetensors'] needs both. The either-format
    ['model.safetensors','pytorch_model.bin'] pair stays satisfied by one (equivalence)."""
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
    """A dataset/space snapshot whose completeness cannot be proven from local files (allow=None or a
    glob) must defer to the watched child. An intact exact-named subset still short-circuits."""
    snap, _ = _mk_snapshot(tmp_path, "dspart")
    (snap / "README.md").write_text("partial")
    assert xf._cache_can_skip_download(
        snap, repo_type = "dataset", allow_patterns = None, ignore_patterns = None) is False
    assert xf._cache_can_skip_download(
        snap, repo_type = "dataset", allow_patterns = ["*.parquet"], ignore_patterns = None) is False
    assert xf._cache_can_skip_download(
        snap, repo_type = "dataset", allow_patterns = ["README.md"], ignore_patterns = None) is True


def test_http_prep_scopes_blob_cleanup_to_owned_partials(tmp_path):
    """HTTP prep must purge only the stalled child's OWN partials, never a concurrent same-repo
    sibling's blob (multi-rank). With ownership known, a sibling's aged partial and dangling link are
    spared; with ownership unknown (None), the coarser mtime guard purges both."""
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

    # No ownership info -> coarse mtime guard purges both aged partials (prior behavior).
    owned, sibling = _seed()
    _REAL_DEFAULT_PREPARE(
        "model", repo, cache_dir = str(tmp_path), active_grace = 180, owned_incomplete_blobs = None)
    assert not owned.exists() and not sibling.exists()
