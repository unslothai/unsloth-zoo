# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Xet -> HTTP transition scopes a forced re-download to the blob at fault (issue 9094); force_download is repo-wide and refetches all completed shards."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import sys
import threading
import time
import types as _types
from pathlib import Path

import pytest

from unsloth_zoo_downloads import hf_cache_state as hcs
from unsloth_zoo_downloads import hf_xet_fallback as xf


REPO = "Qwen/Qwen3.5-9B"
REV = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
REPO_DIR = "models--Qwen--Qwen3.5-9B"
INTACT = (
    "config.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
)
IN_FLIGHT = "model-00002-of-00002.safetensors"
STALL = "Download appears stalled (xet transport) -- no progress for 30s"


def _blob_name(filename: str) -> str:
    return hashlib.sha256(filename.encode()).hexdigest()


def _file_bytes(name: str) -> bytes:
    """Real content: the ladder's completeness judgement parses the shard index."""
    if name == "config.json":
        return json.dumps({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}).encode()
    if name == "model.safetensors.index.json":
        return json.dumps({
            "metadata": {"total_size": 2048},
            "weight_map": {
                "model.layers.0.weight": "model-00001-of-00002.safetensors",
                "model.layers.1.weight": "model-00002-of-00002.safetensors",
            },
        }).encode()
    return b"\xa5" * 1024


def _build_cache(root: Path, *, partial_age_s: float, extra_partial: str = None) -> Path:
    repo = root / REPO_DIR
    blobs, snap, refs = repo / "blobs", repo / "snapshots" / REV, repo / "refs"
    for directory in (blobs, snap, refs):
        directory.mkdir(parents = True, exist_ok = True)
    (refs / "main").write_text(REV)
    for name in INTACT:
        blob = blobs / _blob_name(name)
        blob.write_bytes(_file_bytes(name))
        (snap / name).symlink_to(os.path.relpath(blob, snap))
    partial = blobs / (_blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX)
    partial.write_bytes(b"\xa5" * 512)
    os.utime(partial, (time.time() - partial_age_s, time.time() - partial_age_s))
    # Dangles: the in-flight blob was never finalized.
    (snap / IN_FLIGHT).symlink_to(os.path.relpath(blobs / _blob_name(IN_FLIGHT), snap))
    if extra_partial is not None:
        other = blobs / (_blob_name(extra_partial) + xf.INCOMPLETE_SUFFIX)
        other.write_bytes(b"\x00" * 128)
        os.utime(other, (time.time() - 5.0, time.time() - 5.0))
    return snap


def _held(root: Path, name: str, repo_dir: str = REPO_DIR) -> str:
    """What the live-writer walk reports for a partial: where it is, not what it is called."""
    return xf._normalized_partial_key(root / repo_dir / "blobs" / name)


def _identity(root: Path) -> dict:
    blobs = root / REPO_DIR / "blobs"
    out = {}
    for name in INTACT:
        try:
            st = (blobs / _blob_name(name)).stat()
            out[name] = (st.st_ino, st.st_mtime_ns)
        except OSError:
            out[name] = None
    return out


class _Attempt:
    """Hub skips an already-resolvable pointer only while `force_download` is unset (huggingface_hub 1.31 `file_download.py` :1204 / :1248)."""

    def __init__(self, root: Path, results, owned = None):
        self.root = root
        self._results = list(results)
        self.owned = owned
        self.calls: list = []
        self.fetched: list = []

    def __call__(self, repo_id, *, kind, params, token, repo_type, disable_xet,
                 cancel_event, stall_timeout, interval, grace_period, on_status):
        self.calls.append(_types.SimpleNamespace(
            disable_xet = disable_xet, force_download = params.get("force_download"),
        ))
        kind_result, payload = self._results[len(self.calls) - 1]
        if self.owned is not None and kind_result == "stall":
            params["_owned_incomplete_blobs"] = set(self.owned)
        if kind_result == "ok":
            self._fetch(bool(params.get("force_download")))
        return (kind_result, payload)

    def _fetch(self, force_download: bool) -> None:
        blobs = self.root / REPO_DIR / "blobs"
        snap = self.root / REPO_DIR / "snapshots" / REV
        for name in INTACT + (IN_FLIGHT,):
            pointer = snap / name
            if pointer.exists() and not force_download:
                continue                     # follows the link: a dangling one is not cached
            blob = blobs / _blob_name(name)
            try:
                if blob.exists():
                    blob.unlink()
                blob.write_bytes(_file_bytes(name))
                if pointer.is_symlink():
                    pointer.unlink()
                pointer.symlink_to(os.path.relpath(blob, snap))
            except OSError:
                continue
            self.fetched.append(name)


@pytest.fixture(autouse = True)
def _hermetic(monkeypatch):
    monkeypatch.setenv("UNSLOTH_XET_ATTEMPTS", "1")
    monkeypatch.setenv("UNSLOTH_HTTP_RETRY_BACKOFF", "0")
    monkeypatch.delenv("UNSLOTH_DISABLE_XET", raising = False)
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising = False)
    monkeypatch.setattr(xf, "_record_xet_outcome", lambda ok, reason = "": None)
    monkeypatch.setattr(xf, "_xet_health_or_none", lambda: None)
    monkeypatch.setattr(xf, "_xet_failure_reason", lambda summary: summary)


def _run_ladder(
    monkeypatch,
    root: Path,
    *,
    owned = None,
    inject_hook: bool = False,
    caller_force: bool = False,
    results = None,
) -> _Attempt:
    snap = root / REPO_DIR / "snapshots" / REV
    attempt = _Attempt(
        root, results if results is not None else [("stall", STALL), ("ok", str(snap))],
        owned = owned,
    )
    monkeypatch.setattr(xf, "_run_download_attempt", attempt)
    # Stubs the walk as authoritative so ladder cases test the ladder, not what runs here; a real walk is not neutral:
    # non-dumpable processes like sd-pam raise the age bar and spare partials.
    monkeypatch.setattr(xf, "_partial_paths_with_a_live_writer", lambda: set())
    monkeypatch.setattr(xf, "_live_writer_walk_was_complete", lambda: True)
    monkeypatch.setattr(xf, "_live_writer_walk_missed_our_own_uid", lambda: False)
    hook_calls: list = []
    params = {
        "repo_id": REPO,
        "revision": REV,
        "cache_dir": str(root),
        "allow_patterns": None,
        "ignore_patterns": None,
        "force_download": caller_force,
    }
    out = xf._download_with_xet_fallback(
        repo_id = REPO,
        label = REPO,
        kind = "snapshot",
        params = params,
        token = None,
        repo_type = "model",
        cancel_event = None,
        stall_timeout = 30.0,
        interval = 0.05,
        grace_period = 0.1,
        on_status = None,
        prepare_for_http_fn = (
            (lambda rt, rid: hook_calls.append((rt, rid))) if inject_hook else None
        ),
    )
    attempt.returned = out
    attempt.hook_calls = hook_calls
    return attempt


def _http_force(attempt: _Attempt) -> bool:
    http = [c for c in attempt.calls if c.disable_xet]
    assert http, "the ladder never reached the HTTP rung"
    return bool(http[0].force_download)


def _partials(root: Path) -> list:
    blobs = root / REPO_DIR / "blobs"
    return sorted(p.name for p in blobs.iterdir() if p.name.endswith(xf.INCOMPLETE_SUFFIX))


def test_stale_partial_outside_the_ownership_set_does_not_force_the_whole_repo(
    monkeypatch, tmp_path
):
    """#9094: the purge skips blobs outside the ownership set at any age, so an earlier crash's partial forced a repo-wide re-download."""
    snap = _build_cache(tmp_path, partial_age_s = 1800.0, extra_partial = "vocab.json")
    before = _identity(tmp_path)
    attempt = _run_ladder(
        monkeypatch, tmp_path,
        owned = [_blob_name("vocab.json") + xf.INCOMPLETE_SUFFIX],
    )
    assert attempt.returned == str(snap)
    assert _http_force(attempt) is False, "a removable partial must not force a repo-wide re-download"
    assert _partials(tmp_path) == [], "the blob at fault is gone, so a resume cannot reach it"
    assert attempt.fetched == [IN_FLIGHT], "only the file whose blob was a partial"
    assert _identity(tmp_path) == before, "no completed blob may be rewritten"


def test_injected_prepare_hook_declining_still_scopes_the_force_to_the_blob(monkeypatch, tmp_path):
    """An injected ``prepare_for_http_fn`` (the Studio path) replaces the zoo purge and never sees the ownership set."""
    snap = _build_cache(tmp_path, partial_age_s = 20.0)
    before = _identity(tmp_path)
    attempt = _run_ladder(
        monkeypatch, tmp_path, inject_hook = True,
        owned = [_blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX],
    )
    assert attempt.hook_calls == [("model", REPO)], "the injected hook still owns the purge"
    assert attempt.returned == str(snap)
    assert _http_force(attempt) is False
    assert _partials(tmp_path) == []
    assert attempt.fetched == [IN_FLIGHT]
    assert _identity(tmp_path) == before


def test_a_partial_that_may_belong_to_a_live_sibling_still_forces(monkeypatch, tmp_path):
    """No ownership evidence, younger than the grace: deleting it could destroy a live sibling's blob."""
    _build_cache(tmp_path, partial_age_s = 10.0)
    attempt = _run_ladder(monkeypatch, tmp_path, owned = None)
    assert _http_force(attempt) is True
    assert _partials(tmp_path) == [_blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX], (
        "a possibly-live sibling's partial must survive"
    )


def test_an_unclearable_partial_still_forces(monkeypatch, tmp_path):
    _build_cache(tmp_path, partial_age_s = 1800.0)
    real_unlink = Path.unlink

    def _locked(self, *args, **kwargs):
        if self.name.endswith(xf.INCOMPLETE_SUFFIX):
            raise PermissionError(13, "The process cannot access the file")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", _locked)
    attempt = _run_ladder(
        monkeypatch, tmp_path, owned = [_blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX],
    )
    monkeypatch.undo()
    assert _http_force(attempt) is True
    assert _partials(tmp_path) == [_blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX]


def test_an_uninspectable_cache_after_the_clear_still_forces(monkeypatch, tmp_path):
    """``None`` means the cache could not be READ, never "no partials left"."""
    _build_cache(tmp_path, partial_age_s = 1800.0)
    monkeypatch.setattr(xf, "_incomplete_partial_names", lambda *a, **k: None)
    attempt = _run_ladder(monkeypatch, tmp_path, inject_hook = True, owned = None)
    assert _http_force(attempt) is True


def test_the_scoped_clear_keeps_a_caller_requested_force_download(monkeypatch, tmp_path):
    _build_cache(tmp_path, partial_age_s = 1800.0)
    attempt = _run_ladder(
        monkeypatch, tmp_path, caller_force = True,
        owned = [_blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX],
    )
    assert [c.force_download for c in attempt.calls] == [True, True]


def test_the_latch_still_holds_force_download_while_a_partial_survives(monkeypatch, tmp_path):
    """Only the partial's ABSENCE proves a resume is safe, so the force latches."""
    _build_cache(tmp_path, partial_age_s = 10.0)          # unremovable: possibly a live sibling's
    snap = tmp_path / REPO_DIR / "snapshots" / REV
    attempt = _run_ladder(
        monkeypatch, tmp_path, owned = None,
        results = [("stall", STALL), ("crashed", "died in HEAD"), ("ok", str(snap))],
    )
    assert [c.force_download for c in attempt.calls] == [False, True, True]


def test_a_cancel_at_the_stall_runs_no_clearance_at_all(monkeypatch, tmp_path):
    """Cancellation wins BEFORE the transition: an abandoned download buys no destructive purge."""
    _build_cache(tmp_path, partial_age_s = 1800.0)
    before = _identity(tmp_path)
    attempt = _Attempt(tmp_path, [("stall", STALL)])
    cancel = threading.Event()

    def _stall_then_cancel(*args, **kwargs):
        result = attempt(*args, **kwargs)
        cancel.set()
        return result

    monkeypatch.setattr(xf, "_run_download_attempt", _stall_then_cancel)
    with pytest.raises(RuntimeError, match = "Cancelled"):
        xf._download_with_xet_fallback(
            repo_id = REPO, label = REPO, kind = "snapshot",
            params = {
                "repo_id": REPO, "revision": REV, "cache_dir": str(tmp_path),
                "allow_patterns": None, "ignore_patterns": None, "force_download": False,
            },
            token = None, repo_type = "model", cancel_event = cancel, stall_timeout = 30.0,
            interval = 0.05, grace_period = 0.1, on_status = None, prepare_for_http_fn = None,
        )
    assert len(attempt.calls) == 1, "a cancel buys no further child"
    assert _partials(tmp_path) == [_blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX], (
        "nothing was purged on the way out"
    )
    assert _identity(tmp_path) == before



class _StalledQueue:
    """Never yields a result, so the attempt loop runs until the watchdog fires."""

    def get(self, timeout = None):
        raise queue.Empty

    def get_nowait(self):
        raise queue.Empty

    def put(self, item):
        pass

    def close(self):
        pass

    def cancel_join_thread(self):
        pass


class _StalledProc:
    def __init__(self, on_start):
        self._on_start = on_start
        self.pid = None          # open files uninspectable, exactly as on Windows
        self.exitcode = None

    def start(self):
        self._on_start()

    def is_alive(self):
        return True

    def join(self, timeout = None):
        pass


def test_a_stalled_snapshot_child_publishes_its_own_new_partial_as_owned(monkeypatch, tmp_path):
    """Snapshots used to claim no ownership at all, so the reporter's cached shards were re-downloaded (#9094)."""
    _build_cache(tmp_path, partial_age_s = 1800.0)
    blobs = tmp_path / REPO_DIR / "blobs"
    pre_existing = _blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX
    child_partial = _blob_name("model-00001-of-00002.safetensors") + ".child" + xf.INCOMPLETE_SUFFIX

    def _open_partial():
        (blobs / child_partial).write_bytes(b"\xa5" * 256)     # static: the watchdog will trip

    class _Ctx:
        def Process(self, *, target = None, kwargs = None, daemon = None):
            return _StalledProc(_open_partial)

        def Queue(self):
            return _StalledQueue()

    monkeypatch.setattr(xf, "_CTX", _Ctx())
    monkeypatch.setattr(xf, "_terminate_process_group", lambda proc, grace: None)
    # The stubbed walk is authoritative; with the real one this would assert the host, since sd-pam is non-dumpable on
    # every systemd login and the inference then declines.
    monkeypatch.setattr(xf, "_partial_paths_with_a_live_writer", lambda: set())
    monkeypatch.setattr(xf, "_live_writer_walk_was_complete", lambda: True)
    monkeypatch.setattr(xf, "_live_writer_walk_missed_our_own_uid", lambda: False)
    params = {"repo_id": REPO, "revision": REV, "cache_dir": str(tmp_path)}
    kind_result, _ = xf._run_download_attempt(
        REPO, kind = "snapshot", params = params, token = None, repo_type = "model",
        disable_xet = False, cancel_event = None, stall_timeout = 0.3, interval = 0.05,
        grace_period = 0.1, on_status = None,
    )
    assert kind_result == "stall"
    assert params.get("_owned_incomplete_blobs") == {child_partial}, (
        "a snapshot child must claim the partial it created, and only that one"
    )
    assert pre_existing in _partials(tmp_path), "the pre-existing partial is not ours to claim"


def test_the_snapshot_watchdog_still_measures_the_whole_repo(monkeypatch, tmp_path):
    """Narrowing what a SNAPSHOT measures would let a sibling's growth mask a wedged child."""
    _build_cache(tmp_path, partial_age_s = 10.0)
    seen: list = []

    def _record(**kwargs):
        seen.append(kwargs)
        return threading.Event()

    class _DoneProc:
        pid = 4242
        exitcode = 0

        def start(self):
            pass

        def is_alive(self):
            return False

        def join(self, timeout = None):
            pass

    class _OkQueue:
        def get(self, timeout = None):
            return {"ok": True, "path": "/cache/x"}

        def close(self):
            pass

        def cancel_join_thread(self):
            pass

    class _Ctx:
        def Process(self, *, target = None, kwargs = None, daemon = None):
            return _DoneProc()

        def Queue(self):
            return _OkQueue()

    monkeypatch.setattr(xf, "_CTX", _Ctx())
    monkeypatch.setattr(xf, "start_watchdog", _record)
    for kind in ("snapshot", "file"):
        params = {"repo_id": REPO, "cache_dir": str(tmp_path)}
        if kind == "file":
            params["filename"] = IN_FLIGHT
        xf._run_download_attempt(
            REPO, kind = kind, params = params, token = None, repo_type = "model",
            disable_xet = True, cancel_event = None, stall_timeout = 1.0, interval = 0.05,
            grace_period = 0.1, on_status = None,
        )
    snapshot_kwargs, file_kwargs = seen
    assert snapshot_kwargs["watch_new_partials_only"] is False
    assert snapshot_kwargs["baseline_incomplete_blobs"] is None, (
        "a snapshot keeps the repo-wide measurement"
    )
    assert file_kwargs["watch_new_partials_only"] is True
    assert file_kwargs["baseline_incomplete_blobs"] == {
        _blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX
    }


def test_an_aged_partial_a_live_process_still_holds_open_is_not_cleared(monkeypatch, tmp_path):
    """Age is not proof a writer exited: a paused sibling leaves a partial older than any grace and still finishes it, so the open-file table decides."""
    _build_cache(tmp_path, partial_age_s = 1800.0)
    stranger = _blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX
    monkeypatch.setattr(
        xf, "_partial_paths_with_a_live_writer", lambda: {_held(tmp_path, stranger)}
    )

    survivors = xf._clear_unsafe_partials_for_http(
        "model", REPO, cache_dir = str(tmp_path), active_grace = 180.0,
    )
    assert survivors == {stranger}, "an aged partial with a live writer was deleted"
    assert (tmp_path / REPO_DIR / "blobs" / stranger).exists()


def test_an_unreadable_open_file_table_declines_the_unscoped_pass(monkeypatch, tmp_path):
    """With the open-file table unreadable, the purge is not widened on age alone, or the least inspectable hosts become the most destructive."""
    _build_cache(tmp_path, partial_age_s = 1800.0)
    stranger = _blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX
    monkeypatch.setattr(xf, "_partial_paths_with_a_live_writer", lambda: None)

    survivors = xf._clear_unsafe_partials_for_http(
        "model", REPO, cache_dir = str(tmp_path), active_grace = 180.0,
    )
    assert survivors == {stranger}
    assert (tmp_path / REPO_DIR / "blobs" / stranger).exists()


def test_our_own_partial_is_cleared_even_with_the_table_unreadable(monkeypatch, tmp_path):
    """The ownership set is separate evidence: our own dead child wrote the partial, so no live writer needs protecting; this pass fixes issue 9094."""
    _build_cache(tmp_path, partial_age_s = 5.0)
    mine = _blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX
    monkeypatch.setattr(xf, "_partial_paths_with_a_live_writer", lambda: None)

    survivors = xf._clear_unsafe_partials_for_http(
        "model", REPO, cache_dir = str(tmp_path), active_grace = 180.0,
        owned_incomplete_blobs = {mine},
    )
    assert survivors == set()
    assert not (tmp_path / REPO_DIR / "blobs" / mine).exists()


def test_a_writer_busy_in_another_repo_does_not_shield_this_one(monkeypatch, tmp_path):
    """process_iter walks the whole host and hub names partials by etag, which identical files share across repos; a
    bare-name match lets another repo's live writer shield this repo's dead partial and force a repo-wide
    force_download."""
    _build_cache(tmp_path, partial_age_s = 1800.0)
    stale = _blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX
    # The same etag, in a different repo's cache, with a live writer on it.
    elsewhere = tmp_path / "models--acme--twin" / "blobs"
    elsewhere.mkdir(parents = True)
    (elsewhere / stale).write_bytes(b"\xa5" * 64)
    monkeypatch.setattr(
        xf,
        "_partial_paths_with_a_live_writer",
        lambda: {_held(tmp_path, stale, repo_dir = "models--acme--twin")},
    )

    survivors = xf._clear_unsafe_partials_for_http(
        "model", REPO, cache_dir = str(tmp_path), active_grace = 180.0,
    )
    assert survivors == set(), survivors
    assert not (tmp_path / REPO_DIR / "blobs" / stale).exists(), (
        "a stale partial was spared because an unrelated repo was writing the same etag"
    )
    assert (elsewhere / stale).exists(), "the other repo's live partial was touched"


def test_the_stall_paths_writer_set_is_projected_onto_this_repo(tmp_path):
    """Ownership still compares names, since the listing is this repo's own; the host-wide writer walk is projected onto
    this repo's blobs dirs first, so a writer elsewhere cannot drop an owned name."""
    _build_cache(tmp_path, partial_age_s = 5.0)
    name = _blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX
    here = _held(tmp_path, name)
    there = _held(tmp_path, name, repo_dir = "models--acme--twin")

    assert xf._live_writer_names_for_repo({here}, "model", REPO, str(tmp_path)) == {name}
    assert xf._live_writer_names_for_repo({there}, "model", REPO, str(tmp_path)) == set()


def test_the_live_writer_probe_finds_this_processs_own_open_partial(tmp_path):
    blob = tmp_path / ("deadbeef" + xf.INCOMPLETE_SUFFIX)
    blob.write_bytes(b"x")
    with blob.open("ab") as handle:
        handle.write(b"y")
        handle.flush()
        seen = xf._partial_paths_with_a_live_writer()
        if seen is None:
            pytest.skip("this host cannot read the open-file table")
        key = xf._normalized_partial_key(blob)
        assert key in seen, "an open partial was not seen as having a live writer"
        assert blob.name not in seen, "the walk reported a bare name, which two repos can share"
    after = xf._partial_paths_with_a_live_writer()
    assert after is not None and key not in after


def test_clear_unsafe_partials_reports_survivors_and_spares_a_fresh_stranger(monkeypatch, tmp_path):
    """Ours go at any age, a stranger only past the grace AND with no live writer; the return names what survived; the open-file table is pinned empty."""
    monkeypatch.setattr(xf, "_partial_paths_with_a_live_writer", lambda: set())
    _build_cache(tmp_path, partial_age_s = 5.0, extra_partial = "vocab.json")
    blobs = tmp_path / REPO_DIR / "blobs"
    mine = _blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX
    stranger = _blob_name("vocab.json") + xf.INCOMPLETE_SUFFIX
    survivors = xf._clear_unsafe_partials_for_http(
        "model", REPO, cache_dir = str(tmp_path), active_grace = 180.0,
        owned_incomplete_blobs = {mine},
    )
    assert survivors == {stranger}, "ours is cleared even fresh; a fresh stranger is spared"
    assert not (blobs / mine).exists()
    assert (blobs / stranger).exists()
    os.utime(blobs / stranger, (time.time() - 1800.0, time.time() - 1800.0))
    assert xf._clear_unsafe_partials_for_http(
        "model", REPO, cache_dir = str(tmp_path), active_grace = 180.0,
    ) == set()
    assert xf._clear_unsafe_partials_for_http(
        "model", REPO, cache_dir = str(tmp_path / "gone"), active_grace = 180.0,
    ) is None, "an unreadable cache reports None, never an empty set"


def test_a_baseline_scan_that_failed_is_unknown_rather_than_empty(tmp_path, monkeypatch):
    """Ownership is current minus baseline, and owned blobs skip the grace; a failed scan read as empty would unlink a live sibling's partial; None means unknown."""
    _build_cache(tmp_path, partial_age_s = 5.0)

    present = xf._baseline_incomplete_blob_names("model", REPO, cache_dir = str(tmp_path))
    assert isinstance(present, set) and present, "a readable cache reports the names it has"

    # Absent cache: a genuinely empty baseline, not a failure.
    assert xf._baseline_incomplete_blob_names(
        "model", REPO, cache_dir = str(tmp_path / "no-such-cache"),
    ) == set()

    # A raising scan must be UNKNOWN; patch _strict_repo_cache_dirs, not iter_active_repo_cache_dirs, which catches the
    # root's OSError and yields nothing, hiding a permission flap as empty.
    def _boom(*args, **kwargs):
        raise OSError("cache temporarily unreadable")

    monkeypatch.setattr(xf, "_strict_repo_cache_dirs", _boom)
    monkeypatch.setattr(xf, "iter_active_repo_cache_dirs", _boom)
    assert xf._baseline_incomplete_blob_names("model", REPO, cache_dir = str(tmp_path)) is None, (
        "a failed baseline scan must be unknown, not an empty ownership baseline"
    )
    # The sizes scan means bytes in flight, where unreadable is honestly zero.
    assert xf._active_incomplete_blob_sizes("model", REPO, cache_dir = str(tmp_path)) == {}
    monkeypatch.undo()

    # An unlistable cache root is unknown, but a root that is merely absent is the honestly empty baseline of a first download.
    unreadable = tmp_path / "locked"
    unreadable.mkdir()
    (unreadable / REPO_DIR).mkdir()
    os.chmod(unreadable, 0o000)
    try:
        if os.access(unreadable, os.R_OK):
            pytest.skip("this user can read a 0o000 directory (root), so the flap cannot be posed")
        assert xf._baseline_incomplete_blob_names(
            "model", REPO, cache_dir = str(unreadable),
        ) is None, "an unlistable cache root must be unknown, not empty"
    finally:
        os.chmod(unreadable, 0o755)


def test_a_link_that_stops_dangling_before_the_unlink_is_left_alone(monkeypatch, tmp_path):
    """Eligibility and deletion are separate moments: a link that starts resolving before the unlink must be kept, since
    an older revision's link is not recreated by the current retry, breaking a later offline load; a partial
    appearing beside the target means a download is in progress."""
    monkeypatch.setattr(xf, "_partial_paths_with_a_live_writer", lambda: set())
    snap = _build_cache(tmp_path, partial_age_s = 5.0)
    blobs = tmp_path / REPO_DIR / "blobs"
    (blobs / (_blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX)).unlink()
    orphan = snap / "tokenizer.model"
    target = blobs / _blob_name("tokenizer.model")
    orphan.symlink_to(os.path.relpath(target, snap))
    assert orphan.is_symlink() and not orphan.exists()

    scanned = xf._orphan_snapshot_links_safe_to_clear("model", REPO, str(tmp_path))
    assert "tokenizer.model" in [link.name for link in scanned]

    # Pins the link scan to what it found: a sibling may finish the blob in the window between the eligibility list and the unlink.
    monkeypatch.setattr(
        xf, "_orphan_snapshot_links_safe_to_clear", lambda *args, **kwargs: [orphan],
    )
    target.write_bytes(_file_bytes("tokenizer.model"))
    assert xf._clear_orphan_snapshot_links("model", REPO, str(tmp_path)) == set()
    assert orphan.is_symlink() and orphan.exists(), (
        "a pointer that had become valid was unlinked, so the cached blob lost its only name"
    )

    # A partial appearing beside the target between scan and unlink names a download in progress, so the link must be kept.
    target.unlink()
    (blobs / (_blob_name("tokenizer.model") + xf.INCOMPLETE_SUFFIX)).write_bytes(b"\xa5" * 32)
    assert xf._clear_orphan_snapshot_links("model", REPO, str(tmp_path)) == set()
    assert orphan.is_symlink(), "a link whose blob is being written was unlinked"


def test_a_nonce_suffixed_partial_still_says_the_blob_is_being_written(monkeypatch, tmp_path):
    """Hub writes a process-unique `<etag>.<nonce>.incomplete` then renames (huggingface_hub PR 4228: a shared name
    corrupts the cache where flock silently succeeds); the exact name alone misses live partials."""
    monkeypatch.setattr(xf, "_partial_paths_with_a_live_writer", lambda: set())
    snap = _build_cache(tmp_path, partial_age_s = 5.0)
    blobs = tmp_path / REPO_DIR / "blobs"
    (blobs / (_blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX)).unlink()
    orphan = snap / "tokenizer.model"
    target_name = _blob_name("tokenizer.model")
    orphan.symlink_to(os.path.relpath(blobs / target_name, snap))
    # What a sibling downloading that blob leaves on disk on this hub generation.
    (blobs / f"{target_name}.a1b2c3d4{xf.INCOMPLETE_SUFFIX}").write_bytes(b"\xa5" * 64)

    scanned = [link.name for link in xf._orphan_snapshot_links_safe_to_clear(
        "model", REPO, str(tmp_path),
    )]
    assert "tokenizer.model" not in scanned, scanned
    assert xf._clear_orphan_snapshot_links("model", REPO, str(tmp_path)) == set()
    assert orphan.is_symlink(), "a link whose blob a sibling was writing was unlinked"
    # And the same name is recognised as a fresh partner by the purge's own spare rule.
    assert xf._broken_link_has_active_partner(orphan, active_grace = 180.0) is True


def test_a_container_that_cannot_see_the_other_pods_processes_declines(monkeypatch, tmp_path):
    """process_iter walks only this PID namespace: a sibling container sharing the cache volume is absent with no error,
    so an aged partial it still writes must not be unlinked."""
    _build_cache(tmp_path, partial_age_s = 1800.0)
    blobs = tmp_path / REPO_DIR / "blobs"
    stale = _blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX
    monkeypatch.setattr(xf, "_partial_paths_with_a_live_writer", lambda: set())
    monkeypatch.setattr(xf, "_process_walk_sees_every_writer", lambda _cache_dir = None: False)

    assert xf._unowned_partials_safe_to_clear("model", REPO, str(tmp_path), 180.0, None) is None
    assert xf._clear_unsafe_partials_for_http(
        "model", REPO, cache_dir = str(tmp_path), active_grace = 180.0,
    ) == {stale}
    assert (blobs / stale).exists()


def test_the_namespace_test_reads_containerisation_and_the_caches_mount(monkeypatch, tmp_path):
    """Only a path the root mount itself carries counts as private, not st_dev: a host bind mount into LXC or nspawn
    shares st_dev yet siblings stay invisible to process_iter; topology is stubbed."""
    if os.name == "nt" or not os.path.isdir("/proc"):
        pytest.skip("POSIX containers only")

    monkeypatch.setattr(xf, "hf_cache_root", lambda cache_dir = None: Path("/cache"))
    ROOT_ONLY = "36 25 0:32 / / rw,relatime shared:1 - overlay overlay rw"
    BIND = ROOT_ONLY + "\n41 36 8:1 /srv/models /cache rw,relatime - ext4 /dev/sda1 rw"

    monkeypatch.setattr(xf, "_running_in_a_container", lambda: False)
    monkeypatch.setattr(xf, "_read_proc_text", lambda path: ROOT_ONLY)
    assert xf._process_walk_sees_every_writer(str(tmp_path)) is True, (
        "an ordinary host lost the purge it is entitled to"
    )

    # Containerised, cache carried by the container's own root mount: nobody outside is here.
    monkeypatch.setattr(xf, "_running_in_a_container", lambda: True)
    assert xf._process_walk_sees_every_writer(str(tmp_path)) is True

    # A bind mount from the host has the root's device even in a directory-backed container, yet a sibling can still write into it, so the walk is not complete.
    monkeypatch.setattr(xf, "_read_proc_text", lambda path: BIND)
    assert xf._process_walk_sees_every_writer(str(tmp_path)) is False

    # A mount table we cannot read is not proof of anything.
    monkeypatch.setattr(xf, "_read_proc_text", lambda path: None)
    assert xf._process_walk_sees_every_writer(str(tmp_path)) is False


def test_containerisation_is_not_read_off_optional_markers_alone(monkeypatch, tmp_path):
    """A cgroup-v2 container can leave no marker file and show `0::/` in /proc/1/cgroup; PID 1's own name decides (init
    system is host, entrypoint is container), and unreadable is not host."""
    if os.name == "nt" or not os.path.isdir("/proc"):
        pytest.skip("POSIX /proc only")

    files = {"/proc/1/cgroup": "0::/init.scope\n", "/proc/1/comm": "systemd\n"}
    monkeypatch.setattr(xf.os.path, "exists", lambda path: False)
    monkeypatch.setattr(xf, "_read_proc_text", lambda path: files.get(str(path)))

    assert xf._running_in_a_container() is False, "an ordinary systemd host read as a container"

    files["/proc/1/cgroup"] = "0::/\n"
    assert xf._running_in_a_container() is True, "a cgroup-v2 container read as the host"

    files["/proc/1/cgroup"] = "0::/init.scope\n"
    files["/proc/1/comm"] = "python3\n"
    assert xf._running_in_a_container() is True, "a container whose PID 1 is its entrypoint"

    files["/proc/1/cgroup"] = "12:pids:/kubepods/burstable/pod123\n"
    files["/proc/1/comm"] = "systemd\n"
    assert xf._running_in_a_container() is True

    del files["/proc/1/cgroup"]
    assert xf._running_in_a_container() is True, "unreadable is not evidence of the host"


def test_a_deletion_time_walk_that_could_not_read_every_process_declines(monkeypatch, tmp_path):
    """The deletion-time re-scan must apply the same gate as eligibility: a walk that could not read every process is
    only a lower bound, and an unreadable process on a shared cache may be a writer."""
    _build_cache(tmp_path, partial_age_s = 1800.0)
    blobs = tmp_path / REPO_DIR / "blobs"
    stale = _blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX
    seen = {"scans": 0}

    def _walk():
        seen["scans"] += 1
        if seen["scans"] == 1:
            xf._LIVE_WRITER_WALK.complete = True     # the eligibility scan read everything
        else:
            xf._LIVE_WRITER_WALK.complete = False    # a process it could not read, at deletion
        return set()

    monkeypatch.setattr(xf, "_partial_paths_with_a_live_writer", _walk)
    monkeypatch.setattr(
        xf, "_cache_is_private_to_this_user", lambda _cache_dir = None, **_kwargs: False,
    )

    survivors = xf._clear_unsafe_partials_for_http(
        "model", REPO, cache_dir = str(tmp_path), active_grace = 180.0,
    )
    assert seen["scans"] >= 2
    assert (blobs / stale).exists(), (
        "an unreadable process at deletion time did not stop the purge the scan had gated"
    )
    assert survivors == {stale}


def test_a_partial_a_sibling_reopens_after_the_scan_is_not_unlinked(monkeypatch, tmp_path):
    """Ownership exempts a blob from the age and live-writer guards, so only a partial our dead child wrote may be
    whitelisted; hub reuses a deterministic `<etag>.incomplete` path, so a scan-idle partial can be reopened by a
    sibling."""
    _build_cache(tmp_path, partial_age_s = 1800.0)
    blobs = tmp_path / REPO_DIR / "blobs"
    stale = _blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX

    seen = {"scans": 0}

    def _walk():
        # Nobody is writing when eligibility is decided, but a sibling has the partial open by the time the deletion runs.
        seen["scans"] += 1
        return set() if seen["scans"] == 1 else {_held(tmp_path, stale)}

    monkeypatch.setattr(xf, "_partial_paths_with_a_live_writer", _walk)

    survivors = xf._clear_unsafe_partials_for_http(
        "model", REPO, cache_dir = str(tmp_path), active_grace = 180.0,
    )
    assert seen["scans"] >= 2, "the deletion reused the eligibility scan's reading"
    assert (blobs / stale).exists(), "a partial a live sibling had reopened was unlinked"
    assert survivors == {stale}


def test_an_orphan_snapshot_link_is_cleared_instead_of_forcing_the_whole_repo(
    monkeypatch, tmp_path
):
    """Dangling snapshot links with no partial beside them must be cleared too; otherwise has_active_incomplete_blobs
    stays true and the caller forces a repo-wide force_download (issue 9094)."""
    monkeypatch.setattr(xf, "_partial_paths_with_a_live_writer", lambda: set())
    snap = _build_cache(tmp_path, partial_age_s = 5.0)
    blobs = tmp_path / REPO_DIR / "blobs"
    mine = _blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX
    (blobs / mine).unlink()
    # An older crash can leave a link whose blob was never written, with no partial left to show a download is in progress.
    orphan = snap / "tokenizer.model"
    orphan.symlink_to(os.path.relpath(blobs / _blob_name("tokenizer.model"), snap))
    assert orphan.is_symlink() and not orphan.exists()

    # Uses the module handle loaded hermetically, not a package import: unsloth_zoo/__init__ raises without the separate
    # unsloth install, which CI installs only best-effort.
    assert hcs.has_active_incomplete_blobs("model", REPO, cache_dir = str(tmp_path)) is True
    before = _identity(tmp_path)

    survivors = xf._clear_unsafe_partials_for_http(
        "model", REPO, cache_dir = str(tmp_path), active_grace = 180.0,
    )
    assert survivors == set(), survivors
    assert not orphan.is_symlink(), "the orphan link survived, so the caller still forces"
    assert hcs.has_active_incomplete_blobs("model", REPO, cache_dir = str(tmp_path)) is False
    # And nothing that was verified was touched: that is the whole point of staying blob-scoped.
    assert _identity(tmp_path) == before
    for name in INTACT:
        assert (snap / name).exists(), name


def test_a_dangling_link_whose_partial_is_still_being_written_is_kept(monkeypatch, tmp_path):
    """A dangling link whose `.incomplete` partner may still be running is kept: hub creates the link on finalise, so
    removing it loses a file a live writer is about to complete; while the partial survives, a repo-wide force is
    right anyway."""
    stranger = _blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX
    monkeypatch.setattr(
        xf, "_partial_paths_with_a_live_writer", lambda: {_held(tmp_path, stranger)}
    )
    snap = _build_cache(tmp_path, partial_age_s = 1800.0)
    link = snap / IN_FLIGHT
    assert link.is_symlink() and not link.exists()

    survivors = xf._clear_unsafe_partials_for_http(
        "model", REPO, cache_dir = str(tmp_path), active_grace = 180.0,
    )
    assert survivors == {stranger}
    assert link.is_symlink(), "a link whose blob is mid-download was removed"
    assert (tmp_path / REPO_DIR / "blobs" / stranger).exists()


def test_an_unremovable_orphan_link_is_reported_as_a_survivor(monkeypatch, tmp_path):
    """Absence is the only evidence the caller accepts, so a link this pass could not remove must be reported as a survivor rather than silently omitted."""
    monkeypatch.setattr(xf, "_partial_paths_with_a_live_writer", lambda: set())
    snap = _build_cache(tmp_path, partial_age_s = 5.0)
    blobs = tmp_path / REPO_DIR / "blobs"
    (blobs / (_blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX)).unlink()

    def _unlink(self):
        raise PermissionError("locked")

    monkeypatch.setattr(Path, "unlink", _unlink)
    survivors = xf._clear_unsafe_partials_for_http(
        "model", REPO, cache_dir = str(tmp_path), active_grace = 180.0,
    )
    assert survivors == {IN_FLIGHT}, survivors
    assert (snap / IN_FLIGHT).is_symlink()


def test_a_process_we_cannot_read_stops_the_unowned_purge_on_a_shared_cache(
    monkeypatch, tmp_path
):
    """The open-file walk is a lower bound, not proof of absence: another UID's writer raises AccessDenied on open_files(), and age cannot rule out a paused writer."""
    _build_cache(tmp_path, partial_age_s = 1800.0)
    stranger = _blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX

    class _Denied(Exception):
        pass

    class _Proc:
        def open_files(self):
            raise _Denied("not yours")

    class _FakePsutil:
        NoSuchProcess = type("NoSuchProcess", (Exception,), {})
        ZombieProcess = type("ZombieProcess", (Exception,), {})

        @staticmethod
        def process_iter():
            return [_Proc()]

    monkeypatch.setitem(sys.modules, "psutil", _FakePsutil)
    # Shared cache: another user can write into it, so a process we could not read may be a writer.
    monkeypatch.setattr(xf, "_cache_is_private_to_this_user", lambda _cache_dir = None, **_kwargs: False)
    assert xf._partial_paths_with_a_live_writer() == set()
    assert xf._live_writer_walk_was_complete() is False
    assert xf._unowned_partials_safe_to_clear(
        "model", REPO, str(tmp_path), 180.0, None,
    ) is None
    assert (tmp_path / REPO_DIR / "blobs" / stranger).exists()

    # Private cache: no other user can write here, so unreadable processes cannot be writing into it and the lower bound is exact for it.
    monkeypatch.setattr(xf, "_cache_is_private_to_this_user", lambda _cache_dir = None, **_kwargs: True)
    assert xf._unowned_partials_safe_to_clear(
        "model", REPO, str(tmp_path), 180.0, None,
    ) == {stranger}

    # A process that exited between listing and read holds nothing, so it does not make the walk incomplete.
    class _Gone:
        def open_files(self):
            raise _FakePsutil.NoSuchProcess("gone")

    monkeypatch.setattr(_FakePsutil, "process_iter", staticmethod(lambda: [_Gone()]))
    assert xf._partial_paths_with_a_live_writer() == set()
    assert xf._live_writer_walk_was_complete() is True


def test_the_private_cache_test_is_about_who_can_write_into_it(tmp_path):
    """The predicate itself, since the case above pins it to a fixed answer."""
    cache = tmp_path / "cache"
    cache.mkdir(mode = 0o700)
    if hasattr(os, "geteuid"):
        assert xf._cache_is_private_to_this_user(str(cache)) is True
        os.chmod(cache, 0o777)
        assert xf._cache_is_private_to_this_user(str(cache)) is False, (
            "a world-writable cache is one another user can put a partial in"
        )
        os.chmod(cache, 0o750)
        assert xf._cache_is_private_to_this_user(str(cache)) is True
        # Group-writable counts as shared only if others are in the group; under user-private groups (Fedora, RHEL,
        # umask 002) it is private, so the expectation follows that rule, not this host's groups; declining would switch
        # off the purge on those hosts.
        os.chmod(cache, 0o770)
        assert xf._cache_is_private_to_this_user(str(cache)) is (
            xf._group_is_private_to_this_user(os.getegid())
        )
    # A cache that is not there at all cannot be established as private.
    assert xf._cache_is_private_to_this_user(str(tmp_path / "no-such-cache")) is False


@pytest.mark.skipif(not hasattr(os, "geteuid"), reason = "POSIX permissions only")
def test_a_shared_directory_under_a_private_root_is_still_shared(monkeypatch, tmp_path):
    """Sharing is judged below the root: a 0700 root holding a group-writable repo directory is still shared, since
    partials live in `blobs`; the root alone says nothing."""
    monkeypatch.setattr(xf, "_group_is_private_to_this_user", lambda _gid: False)
    cache = tmp_path / "cache"
    _build_cache(cache, partial_age_s = 1800.0)
    os.chmod(cache, 0o700)
    repo = cache / REPO_DIR
    blobs = repo / "blobs"
    for directory in (repo, blobs):
        os.chmod(directory, 0o755)
    assert xf._cache_is_private_to_this_user(
        str(cache), repo_type = "model", repo_id = REPO,
    ) is True

    os.chmod(repo, 0o775)
    assert xf._cache_is_private_to_this_user(
        str(cache), repo_type = "model", repo_id = REPO,
    ) is False, "a group-writable repo directory under a private root read as private"

    os.chmod(repo, 0o755)
    os.chmod(blobs, 0o775)
    assert xf._cache_is_private_to_this_user(
        str(cache), repo_type = "model", repo_id = REPO,
    ) is False, "a group-writable blobs directory is where the partials actually go"
    os.chmod(blobs, 0o755)


def test_a_repo_directory_that_cannot_be_read_is_unknown_not_empty(tmp_path):
    """Path.is_dir() answers False for unreadable dirs too; only ENOENT and ENOTDIR mean absence, and 3.9-3.12 swallow other errors (3.13 propagates them)."""
    _build_cache(tmp_path, partial_age_s = 5.0)
    repo = tmp_path / REPO_DIR
    os.chmod(repo, 0o000)
    try:
        if os.access(repo, os.R_OK):
            pytest.skip("this user can read a 0o000 directory (root), so the flap cannot be posed")
        # On every supported version the helper answers absence and raises otherwise, so callers report None rather than an empty set for an unreadable repo.
        with pytest.raises(OSError):
            xf._blobs_dir_is_absent(repo / "blobs")
        assert xf._incomplete_partial_names("model", REPO, str(tmp_path)) is None, (
            "an unreadable repo directory reported as having no partials"
        )
        assert xf._baseline_incomplete_blob_names("model", REPO, str(tmp_path)) is None
    finally:
        os.chmod(repo, 0o755)
    assert xf._blobs_dir_is_absent(repo / "no-such-blobs") is True
    # And a repo that simply has no blobs directory yet is honestly empty, not unknown.
    (tmp_path / REPO_DIR / "blobs").rename(tmp_path / REPO_DIR / "blobs-moved")
    assert xf._incomplete_partial_names("model", REPO, str(tmp_path)) == set()


def test_a_baseline_entry_that_cannot_be_inspected_is_unknown(monkeypatch, tmp_path):
    """Ownership is current minus baseline, so a missing baseline name is credited to our child: exempt from age and
    live-writer guards and unlinked; an unreadable entry makes the baseline unknown."""
    _build_cache(tmp_path, partial_age_s = 5.0)
    assert xf._baseline_incomplete_blob_names("model", REPO, cache_dir = str(tmp_path))

    real_is_file = Path.is_file

    def _flaky(self):
        if self.name.endswith(xf.INCOMPLETE_SUFFIX):
            raise OSError("transport endpoint is not connected")
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", _flaky)
    assert xf._baseline_incomplete_blob_names("model", REPO, cache_dir = str(tmp_path)) is None


def test_a_siblings_new_partial_is_not_claimed_by_the_stalled_child(monkeypatch, tmp_path):
    """Subtraction is the fallback when the child's open files are unreadable; a same-repo sibling's new partial must
    not be claimed, or the purge skips its guards and deletes it mid-write."""
    _build_cache(tmp_path, partial_age_s = 1800.0)
    blobs = tmp_path / REPO_DIR / "blobs"
    mine = _blob_name("model-00001-of-00002.safetensors") + ".child" + xf.INCOMPLETE_SUFFIX
    theirs = _blob_name("tokenizer.model") + ".sibling" + xf.INCOMPLETE_SUFFIX

    def _open_partial():
        (blobs / mine).write_bytes(b"\xa5" * 256)
        (blobs / theirs).write_bytes(b"\xa5" * 256)

    class _Ctx:
        def Process(self, *, target = None, kwargs = None, daemon = None):
            return _StalledProc(_open_partial)

        def Queue(self):
            return _StalledQueue()

    monkeypatch.setattr(xf, "_CTX", _Ctx())
    monkeypatch.setattr(xf, "_terminate_process_group", lambda proc, grace: None)
    # The child's own table cannot be read -- Windows, and the case the subtraction exists for.
    monkeypatch.setattr(xf, "_child_open_incomplete_blobs", lambda _pid: None)
    monkeypatch.setattr(
        xf, "_partial_paths_with_a_live_writer", lambda: {_held(tmp_path, theirs)}
    )
    params = {"repo_id": REPO, "revision": REV, "cache_dir": str(tmp_path)}
    kind_result, _ = xf._run_download_attempt(
        REPO, kind = "snapshot", params = params, token = None, repo_type = "model",
        disable_xet = False, cancel_event = None, stall_timeout = 0.3, interval = 0.05,
        grace_period = 0.1, on_status = None,
    )
    assert kind_result == "stall"
    assert params.get("_owned_incomplete_blobs") == {mine}, (
        "a partial a live sibling holds open was claimed as this child's"
    )

    # With the writer walk unusable nothing is claimed by subtraction; the purge falls back to unscoped, where the age and live-writer guards still apply.
    monkeypatch.setattr(xf, "_partial_paths_with_a_live_writer", lambda: None)
    params = {"repo_id": REPO, "revision": REV, "cache_dir": str(tmp_path)}
    xf._run_download_attempt(
        REPO, kind = "snapshot", params = params, token = None, repo_type = "model",
        disable_xet = False, cancel_event = None, stall_timeout = 0.3, interval = 0.05,
        grace_period = 0.1, on_status = None,
    )
    assert params.get("_owned_incomplete_blobs") is None


def test_the_walk_completeness_is_per_thread(monkeypatch):
    """Walk completeness must be per-thread; a shared module global let one thread's complete walk overwrite another
    thread's unread incomplete verdict, and that thread would unlink an uninspectable partial."""
    import threading

    class _Denied(Exception):
        pass

    class _Blind:
        def open_files(self):
            raise _Denied("not yours")

    class _Open:
        def open_files(self):
            return []

    class _FakePsutil:
        NoSuchProcess = type("NoSuchProcess", (Exception,), {})
        ZombieProcess = type("ZombieProcess", (Exception,), {})
        visible = [_Blind()]

        @classmethod
        def process_iter(cls):
            return cls.visible

    monkeypatch.setitem(sys.modules, "psutil", _FakePsutil)
    xf._partial_paths_with_a_live_writer()
    assert xf._live_writer_walk_was_complete() is False

    seen = {}

    def _other_thread():
        _FakePsutil.visible = [_Open()]
        xf._partial_paths_with_a_live_writer()
        seen["theirs"] = xf._live_writer_walk_was_complete()

    thread = threading.Thread(target = _other_thread)
    thread.start()
    thread.join()

    assert seen["theirs"] is True, "the other thread's own walk was complete"
    assert xf._live_writer_walk_was_complete() is False, (
        "another thread's complete walk overwrote this thread's incomplete verdict"
    )


def _confirmation(monkeypatch, *, names, active):
    monkeypatch.setattr(xf, "_incomplete_partial_names", lambda *_a, **_k: names)
    monkeypatch.setattr(xf, "has_active_incomplete_blobs", lambda *_a, **_k: active)
    return xf._no_incomplete_blobs_confirmed("model", "Qwen/Qwen3.5-9B", None)


def test_a_cache_that_cannot_be_read_after_the_clear_does_not_release_the_force(monkeypatch):
    """has_active_incomplete_blobs swallows OSError, so an unreadable or unmounted cache answers False like a clean one
    and releases the force; HTTP then resumes the partial, finalizing it under its sha256."""
    assert _confirmation(monkeypatch, names = None, active = False) is False


def test_a_confirmed_empty_cache_still_releases_the_force(monkeypatch):
    assert _confirmation(monkeypatch, names = set(), active = False) is True


def test_a_surviving_partial_holds_the_force(monkeypatch):
    assert _confirmation(monkeypatch, names = {"abc.incomplete"}, active = False) is False


def test_a_dangling_snapshot_link_holds_the_force_with_no_partial_to_find(monkeypatch):
    """A broken link has no *.incomplete name, so the name probe alone would release on it."""
    assert _confirmation(monkeypatch, names = set(), active = True) is False


def test_the_release_after_the_scoped_clear_asks_the_confirmation_probe():
    """Placement: the probe above proves only what it does once reached."""
    import inspect

    source = inspect.getsource(xf)
    block = source.split("survivors = _clear_unsafe_partials_for_http(", 1)[1]
    block = block.split("forced_clean_redownload = True", 1)[0]
    assert "_no_incomplete_blobs_confirmed(" in block
    assert "not has_active_incomplete_blobs(" not in block, (
        "the release went back to the probe that answers False for an unreadable cache"
    )


def test_a_sibling_that_reopened_our_partials_name_is_not_unlinked(monkeypatch, tmp_path):
    """The older hub's deterministic `<etag>.incomplete` can be recreated by a sibling after our purge, so the captured
    basename may name someone else's open download."""
    _build_cache(tmp_path, partial_age_s = 5.0)
    mine = _blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX
    reopened = tmp_path / REPO_DIR / "blobs" / mine
    monkeypatch.setattr(
        xf, "_partial_paths_with_a_live_writer",
        lambda: {xf._normalized_partial_key(reopened)},
    )

    survivors = xf._clear_unsafe_partials_for_http(
        "model", REPO, cache_dir = str(tmp_path), active_grace = 180.0,
        owned_incomplete_blobs = {mine},
    )
    assert reopened.exists(), "a sibling's open download was unlinked on a name we used to own"
    assert mine in survivors, "and the survivor has to keep forcing"


def test_our_own_partial_still_goes_when_nobody_holds_it(monkeypatch, tmp_path):
    """The other half of the same guard: #9094 is this deletion, and it must still happen."""
    _build_cache(tmp_path, partial_age_s = 5.0)
    mine = _blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX
    monkeypatch.setattr(xf, "_partial_paths_with_a_live_writer", lambda: set())

    survivors = xf._clear_unsafe_partials_for_http(
        "model", REPO, cache_dir = str(tmp_path), active_grace = 180.0,
        owned_incomplete_blobs = {mine},
    )
    assert survivors == set()
    assert not (tmp_path / REPO_DIR / "blobs" / mine).exists()


def test_a_partner_scan_that_could_not_read_the_blobs_dir_spares_the_link(monkeypatch, tmp_path):
    """If _partial_partners_of swallows a read error it reports no partner, so a dangling link is judged an orphan and
    removed while a sibling finishes its blob; an older revision's link is not recreated."""
    _build_cache(tmp_path, partial_age_s = 5.0)
    snapshot = tmp_path / REPO_DIR / "snapshots" / REV
    orphan = snapshot / "orphan.safetensors"
    orphan.symlink_to(tmp_path / REPO_DIR / "blobs" / "deadbeef")

    class _DeniedStat:
        """Stubs only this module's os lookups, and only stat: patching the real os.stat breaks pytest, which stats the files it reads back for its own reporting."""

        def __getattr__(self, name):
            return getattr(os, name)

        def stat(self, path, *_a, **_k):
            if str(path).endswith(xf.INCOMPLETE_SUFFIX):
                raise PermissionError(13, "denied")
            return os.stat(path)

    monkeypatch.setattr(xf, "os", _DeniedStat())
    assert xf._partial_partners_of(tmp_path / REPO_DIR / "blobs" / "deadbeef") is None
    assert xf._broken_link_has_active_partner(orphan, active_grace = 180.0) is True


def test_a_partner_that_is_honestly_absent_is_still_absent(monkeypatch, tmp_path):
    """Absence and unreadability are different answers; only the second one spares."""
    _build_cache(tmp_path, partial_age_s = 5.0)
    assert xf._partial_partners_of(tmp_path / REPO_DIR / "blobs" / "deadbeef") == []


def test_a_blobs_dir_that_cannot_be_listed_is_unknown_not_partnerless(tmp_path):
    """Path.glob swallows a directory read error and returns an empty list; with execute without read, a sibling's
    nonce-spelled partial is invisible and its link gets unlinked as an orphan."""
    _build_cache(tmp_path, partial_age_s = 5.0)
    blobs = tmp_path / REPO_DIR / "blobs"
    target = blobs / _blob_name(IN_FLIGHT)
    mode = blobs.stat().st_mode
    os.chmod(blobs, 0o111)
    try:
        if os.access(blobs, os.R_OK):
            pytest.skip("this user can list a 0o111 directory (root), so the flap cannot be posed")
        assert xf._partial_partners_of(target) is None
    finally:
        os.chmod(blobs, mode)


def test_a_reopened_owned_name_is_spared_when_the_walk_cannot_see_every_writer(
    monkeypatch, tmp_path,
):
    """On a cache other UIDs can write, an empty lower bound from a walk that missed processes proves nothing; a sibling
    can reopen the older hub's `<etag>.incomplete` after our purge, and unlinking it deletes that live download."""
    _build_cache(tmp_path, partial_age_s = 5.0)
    mine = _blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX
    monkeypatch.setattr(xf, "_partial_paths_with_a_live_writer", lambda: set())
    monkeypatch.setattr(xf, "_live_writer_walk_was_complete", lambda: False)
    monkeypatch.setattr(xf, "_cache_is_private_to_this_user", lambda *a, **k: False)

    xf._clear_partials(
        "model", REPO, cache_dir = str(tmp_path), active_grace = 180.0,
        owned_incomplete_blobs = {mine}, owned_names_may_be_reopened = True,
    )
    assert mine in _partials(tmp_path), "a name another process may hold must not be unlinked"


def test_a_sibling_container_on_a_shared_volume_keeps_its_reopened_name(monkeypatch, tmp_path):
    """Privacy checks only UIDs; a sibling container shares our UID in another namespace, so a shared volume passes
    `_cache_is_private_to_this_user` yet the sibling is invisible, and an empty writer set is no evidence."""
    _build_cache(tmp_path, partial_age_s = 5.0)
    mine = _blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX
    monkeypatch.setattr(xf, "_partial_paths_with_a_live_writer", lambda: set())
    monkeypatch.setattr(xf, "_live_writer_walk_was_complete", lambda: False)
    monkeypatch.setattr(xf, "_live_writer_walk_missed_our_own_uid", lambda: False)
    monkeypatch.setattr(xf, "_cache_is_private_to_this_user", lambda *a, **k: True)
    monkeypatch.setattr(xf, "_process_walk_sees_every_writer", lambda cache_dir = None: False)

    xf._clear_partials(
        "model", REPO, cache_dir = str(tmp_path), active_grace = 180.0,
        owned_incomplete_blobs = {mine}, owned_names_may_be_reopened = True,
    )
    assert mine in _partials(tmp_path), (
        "a volume shared with another namespace is not answered by file ownership"
    )

    # Where the walk sees every writer, privacy still covers the cache, so the fix costs the single-user host of issue 9094 nothing.
    monkeypatch.setattr(xf, "_process_walk_sees_every_writer", lambda cache_dir = None: True)
    xf._clear_partials(
        "model", REPO, cache_dir = str(tmp_path), active_grace = 180.0,
        owned_incomplete_blobs = {mine}, owned_names_may_be_reopened = True,
    )
    assert mine not in _partials(tmp_path), "an ordinary private cache still clears"


def test_the_stalled_child_is_killed_before_the_live_writer_walk(monkeypatch, tmp_path):
    """Kill the stalled child before the live-writer walk: it holds its own partial open, and the per-pid probe answers
    nothing on Windows or for macOS children, causing the repo-wide re-download of issue 9094."""
    _build_cache(tmp_path, partial_age_s = 1800.0)
    blobs = tmp_path / REPO_DIR / "blobs"
    child_partial = _blob_name("model-00001-of-00002.safetensors") + ".child" + xf.INCOMPLETE_SUFFIX
    order = []

    def _open_partial():
        (blobs / child_partial).write_bytes(b"\xa5" * 256)

    class _Ctx:
        def Process(self, *, target = None, kwargs = None, daemon = None):
            return _StalledProc(_open_partial)

        def Queue(self):
            return _StalledQueue()

    def _walk():
        order.append("walk")
        return {_held(tmp_path, child_partial)} if "kill" not in order else set()

    monkeypatch.setattr(xf, "_CTX", _Ctx())
    monkeypatch.setattr(xf, "_child_open_incomplete_blobs", lambda pid: None)
    monkeypatch.setattr(xf, "_partial_paths_with_a_live_writer", _walk)
    monkeypatch.setattr(xf, "_terminate_process_group", lambda proc, grace: order.append("kill"))
    params = {"repo_id": REPO, "revision": REV, "cache_dir": str(tmp_path)}
    kind_result, _ = xf._run_download_attempt(
        REPO, kind = "snapshot", params = params, token = None, repo_type = "model",
        disable_xet = False, cancel_event = None, stall_timeout = 0.3, interval = 0.05,
        grace_period = 0.1, on_status = None,
    )
    assert kind_result == "stall"
    assert order[:2] == ["kill", "walk"], f"the walk must follow the kill, got {order}"
    assert params.get("_owned_incomplete_blobs") == {child_partial}


def test_a_process_of_our_own_that_nothing_can_describe_raises_the_age_bar(
    monkeypatch, tmp_path,
):
    """A process of ours nothing can describe (macOS TCC, non-dumpable Linux) may hold the partial; declining would
    leave issue 9094 unfixed for every systemd login, so the age bar is raised past 171s, the longest no-growth
    window a live Xet writer has shown."""
    _build_cache(tmp_path, partial_age_s = 200.0)
    partial = _blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX

    class _Denied(Exception):
        pass

    class _Proc:
        pid = 4242
        uid = None

        def open_files(self):
            raise _Denied("non-dumpable")

        def uids(self):
            return type("U", (), {"effective": self.uid})()

    ours = _Proc()
    ours.uid = os.geteuid() if hasattr(os, "geteuid") else 0
    theirs = _Proc()
    theirs.uid = -1

    class _FakePsutil:
        NoSuchProcess = type("NoSuchProcess", (Exception,), {})
        ZombieProcess = type("ZombieProcess", (Exception,), {})
        procs = [ours]

        @staticmethod
        def process_iter():
            return _FakePsutil.procs

    monkeypatch.setitem(sys.modules, "psutil", _FakePsutil)
    monkeypatch.setattr(xf, "_proc_fd_partial_paths", lambda pid: None)
    monkeypatch.setattr(xf, "_cache_is_private_to_this_user", lambda *a, **k: True)

    # Past the ordinary grace but well short of the raised bar, the partial stays uncleared while that undescribable process could be its writer.
    assert xf._unowned_partials_safe_to_clear("model", REPO, str(tmp_path), 180.0, None) == set()
    assert xf._live_writer_walk_missed_our_own_uid() is True

    # The same partial once it is genuinely stale clears, so the fix still works here.
    blobs = tmp_path / REPO_DIR / "blobs"
    old = time.time() - 4000.0
    os.utime(blobs / partial, (old, old))
    assert xf._unowned_partials_safe_to_clear(
        "model", REPO, str(tmp_path), 180.0, None,
    ) == {partial}

    # Privacy does answer for another user's unreadable process at the normal grace; where there is no uid to compare,
    # as on Windows, every unreadable process counts as possibly ours.
    if hasattr(os, "geteuid"):
        _FakePsutil.procs = [theirs]
        fresh = time.time() - 200.0
        os.utime(blobs / partial, (fresh, fresh))
        assert xf._unowned_partials_safe_to_clear(
            "model", REPO, str(tmp_path), 180.0, None,
        ) == {partial}
    else:
        assert xf._process_may_be_this_user(theirs, _FakePsutil) is True

    # And with an fd table that answers, our own denied process is not a miss at all.
    _FakePsutil.procs = [ours]
    fresh = time.time() - 200.0
    os.utime(blobs / partial, (fresh, fresh))
    monkeypatch.setattr(xf, "_proc_fd_partial_paths", lambda pid: set())
    assert xf._unowned_partials_safe_to_clear(
        "model", REPO, str(tmp_path), 180.0, None,
    ) == {partial}
    assert xf._live_writer_walk_missed_our_own_uid() is False


def test_the_studio_path_forces_when_a_writer_we_cannot_describe_could_hold_the_legacy_name(
    monkeypatch, tmp_path,
):
    """Hub 0.36's deterministic `<etag>.incomplete` can be recreated by an undescribable same-uid sibling, so it is
    spared and the force stands; current hub's nonce-spelled partial cannot be recreated, so the scoped re-download
    stays."""
    snap = _build_cache(tmp_path, partial_age_s = 20.0)
    legacy = _blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX
    monkeypatch.setattr(xf, "_partial_paths_with_a_live_writer", lambda: set())
    monkeypatch.setattr(xf, "_live_writer_walk_was_complete", lambda: False)
    monkeypatch.setattr(xf, "_live_writer_walk_missed_our_own_uid", lambda: True)
    monkeypatch.setattr(xf, "_cache_is_private_to_this_user", lambda *a, **k: True)

    survivors = xf._clear_unsafe_partials_for_http(
        "model", REPO, cache_dir = str(tmp_path), active_grace = 180.0,
        owned_incomplete_blobs = {legacy},
    )
    assert survivors == {legacy}, "a name that sibling could have reopened must survive"

    # The same situation with current hub's spelling: nobody can have recreated it.
    (tmp_path / REPO_DIR / "blobs" / legacy).unlink()
    nonce = _blob_name(IN_FLIGHT) + ".a1b2c3d4" + xf.INCOMPLETE_SUFFIX
    (tmp_path / REPO_DIR / "blobs" / nonce).write_bytes(b"\xa5" * 512)
    assert xf._clear_unsafe_partials_for_http(
        "model", REPO, cache_dir = str(tmp_path), active_grace = 180.0,
        owned_incomplete_blobs = {nonce},
    ) == set()
    assert snap.exists()


def test_a_link_whose_partial_was_kept_is_kept_too(monkeypatch, tmp_path):
    """A link whose partial was kept must be kept too; the snapshot sweep judges it on the same stale mtime, and a
    sibling for another revision will not recreate it, stranding the blob."""
    _build_cache(tmp_path, partial_age_s = 1800.0)
    blobs = tmp_path / REPO_DIR / "blobs"
    partial = _blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX
    # An older revision pointing at the same blob, dangling because the blob is not finalized.
    old_snapshot = tmp_path / REPO_DIR / "snapshots" / ("c" * 40)
    old_snapshot.mkdir(parents = True)
    link = old_snapshot / IN_FLIGHT
    link.symlink_to(os.path.relpath(blobs / _blob_name(IN_FLIGHT), old_snapshot))

    # The writer reappears after the scan whitelisted the partial.
    monkeypatch.setattr(
        xf, "_partial_paths_with_a_live_writer", lambda: {_held(tmp_path, partial)},
    )
    monkeypatch.setattr(xf, "_live_writer_walk_was_complete", lambda: True)
    monkeypatch.setattr(xf, "_live_writer_walk_missed_our_own_uid", lambda: False)

    xf._clear_partials(
        "model", REPO, cache_dir = str(tmp_path), active_grace = 180.0,
        owned_incomplete_blobs = {partial}, ownership_is_an_earlier_scan = True,
    )
    assert (blobs / partial).exists(), "the writer recheck must spare the blob"
    assert link.is_symlink(), "and the link to it must not be swept while the partial is there"


def test_a_link_whose_partial_really_went_is_still_cleared(monkeypatch, tmp_path):
    """The negative control: nothing above may stop the sweep doing its job."""
    _build_cache(tmp_path, partial_age_s = 1800.0)
    blobs = tmp_path / REPO_DIR / "blobs"
    partial = _blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX
    old_snapshot = tmp_path / REPO_DIR / "snapshots" / ("c" * 40)
    old_snapshot.mkdir(parents = True)
    link = old_snapshot / IN_FLIGHT
    link.symlink_to(os.path.relpath(blobs / _blob_name(IN_FLIGHT), old_snapshot))

    monkeypatch.setattr(xf, "_partial_paths_with_a_live_writer", lambda: set())
    monkeypatch.setattr(xf, "_live_writer_walk_was_complete", lambda: True)
    monkeypatch.setattr(xf, "_live_writer_walk_missed_our_own_uid", lambda: False)

    xf._clear_partials(
        "model", REPO, cache_dir = str(tmp_path), active_grace = 180.0,
        owned_incomplete_blobs = {partial}, ownership_is_an_earlier_scan = True,
    )
    assert not (blobs / partial).exists(), "nobody holds it, so it goes"
    assert not link.is_symlink(), "and its dangling link goes with it"


def test_a_snapshot_claims_nothing_when_a_process_of_ours_cannot_be_described(
    monkeypatch, tmp_path,
):
    """A wrong ownership claim exempts a live download from the age guard: if a process of ours is unreadable, a
    sibling's fresh partial looks owned, and privacy does not exclude it; claiming nothing costs only a re-download."""
    _build_cache(tmp_path, partial_age_s = 1800.0)
    blobs = tmp_path / REPO_DIR / "blobs"
    sibling = _blob_name("model-00001-of-00002.safetensors") + ".sibling" + xf.INCOMPLETE_SUFFIX

    def _open_partial():
        (blobs / sibling).write_bytes(b"\xa5" * 256)

    class _Ctx:
        def Process(self, *, target = None, kwargs = None, daemon = None):
            return _StalledProc(_open_partial)

        def Queue(self):
            return _StalledQueue()

    monkeypatch.setattr(xf, "_CTX", _Ctx())
    monkeypatch.setattr(xf, "_terminate_process_group", lambda proc, grace: None)
    monkeypatch.setattr(xf, "_partial_paths_with_a_live_writer", lambda: set())
    monkeypatch.setattr(xf, "_live_writer_walk_was_complete", lambda: False)
    monkeypatch.setattr(xf, "_live_writer_walk_missed_our_own_uid", lambda: True)
    monkeypatch.setattr(xf, "_cache_is_private_to_this_user", lambda *a, **k: True)

    params = {"repo_id": REPO, "revision": REV, "cache_dir": str(tmp_path)}
    kind_result, _ = xf._run_download_attempt(
        REPO, kind = "snapshot", params = params, token = None, repo_type = "model",
        disable_xet = False, cancel_event = None, stall_timeout = 0.3, interval = 0.05,
        grace_period = 0.1, on_status = None,
    )
    assert kind_result == "stall"
    assert params.get("_owned_incomplete_blobs") is None, (
        "a partial we cannot prove is ours must not be claimed"
    )
    assert (blobs / sibling).exists()


def test_an_unreadable_cache_at_the_transition_forces_rather_than_resuming(monkeypatch, tmp_path):
    """has_active_incomplete_blobs returns False for an unreadable cache, which skips clearance and confirmation and
    lets HTTP resume a sparse Xet partial; only a positive clean confirmation may skip the block."""
    _build_cache(tmp_path, partial_age_s = 20.0)
    monkeypatch.setattr(xf, "_partial_paths_with_a_live_writer", lambda: set())
    monkeypatch.setattr(xf, "_live_writer_walk_was_complete", lambda: True)
    monkeypatch.setattr(xf, "_live_writer_walk_missed_our_own_uid", lambda: False)
    # The shape of a permission flap: the walk yields nothing, the name probe cannot read.
    monkeypatch.setattr(xf, "has_active_incomplete_blobs", lambda *a, **k: False)
    monkeypatch.setattr(xf, "_incomplete_partial_names", lambda *a, **k: None)

    attempt = _run_ladder(monkeypatch, tmp_path, owned = None)
    assert _http_force(attempt) is True, (
        "a cache that cannot be read is not a cache that is clean"
    )


def test_a_cache_confirmed_clean_still_skips_the_clearance(monkeypatch, tmp_path):
    """The control: the common path must not start paying for the case above."""
    _build_cache(tmp_path, partial_age_s = 20.0)
    for name in _partials(tmp_path):
        (tmp_path / REPO_DIR / "blobs" / name).unlink()
    called: list = []
    monkeypatch.setattr(
        xf, "_clear_unsafe_partials_for_http",
        lambda *a, **k: called.append(a) or set(),
    )
    attempt = _run_ladder(monkeypatch, tmp_path, owned = None)
    assert called == [], "a confirmed-clean cache needs no clearance pass"
    assert _http_force(attempt) is False


class _NoMarkers:
    """Stubs os.path for the container gate with every runtime marker absent, so only its cgroup logic is tested, not the host running the suite."""

    _SIMULATED = frozenset({
        "/.dockerenv", "/run/.containerenv", "/run/systemd/container", "/run/host",
        "/proc/vz", "/proc/bc",
    })

    @classmethod
    def exists(cls, path):
        if path in cls._SIMULATED:
            return False
        return os.path.exists(path)

    @classmethod
    def isdir(cls, path):
        if path in cls._SIMULATED:
            return False
        return os.path.isdir(path)

    def __getattr__(self, name):
        return getattr(os.path, name)


def test_an_init_based_container_is_not_read_as_the_host(monkeypatch, tmp_path):
    """systemd-nspawn runs systemd as PID 1 in /init.scope with no runtime marker, so PID 1's name proves nothing; check
    the markers systemd writes, since answering host lets an aged partial be unlinked."""
    if os.name == "nt" or not os.path.isdir("/proc"):
        pytest.skip("POSIX containers only")

    proc_text = {"/proc/1/cgroup": "0::/init.scope\n", "/proc/1/comm": "systemd\n"}
    monkeypatch.setattr(xf, "_read_proc_text", lambda path: proc_text.get(path))

    present: set = set()
    # Every gate path is answered from `present` alone, since the real filesystem would expose the runner's own
    # /.dockerenv under Docker or Podman and fail the host arm.
    simulated = frozenset({
        "/.dockerenv", "/run/.containerenv", "/run/systemd/container", "/run/host",
        "/proc/vz", "/proc/bc",
    })

    class _WithMarkers:
        """Patches only this module's os lookups, not os.path itself, since pytest stats the files it is about to report on."""

        class path:
            @staticmethod
            def exists(path):
                if path in simulated:
                    return path in present
                return os.path.exists(path)

            @staticmethod
            def isdir(path):
                if path in simulated:
                    return path in present
                return os.path.isdir(path)

        def __getattr__(self, name):
            return getattr(os, name)

    monkeypatch.setattr(xf, "os", _WithMarkers())

    assert xf._running_in_a_container() is False, "an ordinary systemd host is not a container"

    for marker in ("/run/systemd/container", "/run/host"):
        present.clear()
        present.add(marker)
        assert xf._running_in_a_container() is True, f"{marker} names an init-based container"

    # And the gate then consults the mount table rather than returning True on PID 1's name.
    present.clear()
    present.add("/run/systemd/container")
    monkeypatch.setattr(xf, "hf_cache_root", lambda cache_dir = None: Path("/cache"))
    monkeypatch.setattr(
        xf, "_read_proc_text",
        lambda path: proc_text.get(path) if path in proc_text else
        "36 25 0:32 / / rw - overlay overlay rw\n41 36 8:1 /srv /cache rw - ext4 /dev/sda1 rw",
    )
    assert xf._process_walk_sees_every_writer(str(tmp_path)) is False


def test_a_root_cgroup_under_a_non_systemd_init_is_still_the_host(monkeypatch, tmp_path):
    """`0::/` is not container-specific: OpenRC, runit and busybox init leave PID 1 in the root cgroup on ordinary
    hosts, so reading it as a container declines the clearance repo-wide."""
    if os.name == "nt" or not os.path.isdir("/proc"):
        pytest.skip("POSIX containers only")

    proc_text = {"/proc/1/cgroup": "0::/\n", "/proc/1/comm": "openrc-init\n"}
    monkeypatch.setattr(xf, "_read_proc_text", lambda path: proc_text.get(path))

    links = {"/proc/1/ns/cgroup": "cgroup:[4026531835]"}

    class _NoMarkerOs:
        path = _NoMarkers()

        @staticmethod
        def readlink(path):
            return links[path]

        def __getattr__(self, name):
            return getattr(os, name)

    monkeypatch.setattr(xf, "os", _NoMarkerOs())
    assert xf._running_in_a_container() is False, "a non-systemd host is not a container"

    # An entrypoint that is not an init still reads as a container on the same line.
    proc_text["/proc/1/comm"] = "python3\n"
    assert xf._running_in_a_container() is True

    # A supported init with a non-initial cgroup namespace also reads as a container, as an unshared container looks when it runs one.
    proc_text["/proc/1/comm"] = "openrc-init\n"
    links["/proc/1/ns/cgroup"] = "cgroup:[4026533112]"
    assert xf._running_in_a_container() is True

    # An unreadable link is not evidence of the host either.
    links.clear()

    class _DeniedLink(_NoMarkerOs):
        @staticmethod
        def readlink(path):
            raise PermissionError(path)

    monkeypatch.setattr(xf, "os", _DeniedLink())
    assert xf._running_in_a_container() is True
