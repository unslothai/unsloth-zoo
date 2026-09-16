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

"""The Xet -> HTTP transition must scope a forced clean re-download to the blob at fault (issue #9094).

``force_download`` is repo-wide: huggingface_hub hands back an already-resolvable pointer only under
``not force_download``, so one partial surviving the pre-HTTP purge used to cost a re-download of
every completed shard in the repo. These tests run the REAL ladder against a REAL (temporary) HF
cache -- the detectors are not stubbed -- and measure both what the retry was told and what happened
to the blobs on disk.

The invariants that must survive the scoping are tested too: a partial that may belong to a live
sibling, one that cannot be removed at all, an uninspectable cache, and the caller's own
``force_download`` all still behave exactly as before.

CPU-only, no network, no real subprocess.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import queue
import sys
import threading
import time
import types as _types
from pathlib import Path

import pytest

_ZOO_DIR = Path(__file__).resolve().parents[1] / "unsloth_zoo"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _ZOO_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Package placeholder so intra-package imports in hf_xet_fallback resolve to the files loaded
# below. Restored afterwards: a leftover would shadow the real unsloth_zoo.
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
    """Three finalized files plus one in-flight partial, laid out as huggingface_hub does."""
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
    # The snapshot link for the in-flight file dangles: its blob was never finalized.
    (snap / IN_FLIGHT).symlink_to(os.path.relpath(blobs / _blob_name(IN_FLIGHT), snap))
    if extra_partial is not None:
        other = blobs / (_blob_name(extra_partial) + xf.INCOMPLETE_SUFFIX)
        other.write_bytes(b"\x00" * 128)
        os.utime(other, (time.time() - 5.0, time.time() - 5.0))
    return snap


def _identity(root: Path) -> dict:
    """``{file: (inode, mtime_ns)}`` for the finalized blobs, so a re-download shows up."""
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
    """Download seam stub: records the params of each attempt and writes what hub would write.

    Two behaviours are copied from the real ``_run_download_attempt``: a stalled child publishes the
    ``.incomplete`` basenames it held open before it is killed, and a child that runs FETCHES -- and
    hub skips an already-resolvable pointer only while ``force_download`` is unset (huggingface_hub
    1.31 ``file_download.py`` :1204 / :1248, 0.36.2 the same two checks).
    """

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
                continue                     # an unwritable cache: the retry fails, as it did live
            self.fetched.append(name)


@pytest.fixture(autouse = True)
def _hermetic(monkeypatch):
    """One Xet child then the transport changes, no health-state writes, no hf_xet log scan."""
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
    """One stalled Xet child, then one HTTP child, through the real ladder."""
    snap = root / REPO_DIR / "snapshots" / REV
    attempt = _Attempt(
        root, results if results is not None else [("stall", STALL), ("ok", str(snap))],
        owned = owned,
    )
    monkeypatch.setattr(xf, "_run_download_attempt", attempt)
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
    """Issue #9094. The purge drops the ownership SET over the grace and then skips every blob
    outside it whatever its age, so a partial left by an EARLIER crashed attempt survives, and the
    repo-wide force_download re-downloaded every completed shard. Clear that blob instead: the
    retry fetches only the file it belonged to."""
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
    """Unsloth injects its own marker-aware ``prepare_for_http_fn``, so the zoo purge never runs and
    a partial its markers decline to trust survives with the ownership set unused. The scoped pass
    is what reaches it: the reporter was on the Studio path, which always injects the hook."""
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
    """The invariant the scoping must not break. With no ownership evidence and a partial younger
    than the patient grace, deleting it could destroy a live sibling's in-flight blob, so it stays
    and the clean re-download still happens -- the resume hazard is real and unprovable either way."""
    _build_cache(tmp_path, partial_age_s = 10.0)
    attempt = _run_ladder(monkeypatch, tmp_path, owned = None)
    assert _http_force(attempt) is True
    assert _partials(tmp_path) == [_blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX], (
        "a possibly-live sibling's partial must survive"
    )


def test_an_unclearable_partial_still_forces(monkeypatch, tmp_path):
    """The reporter's literal "could not be cleared": a locked / denied blob (Windows AV, a handle a
    dead child never released). Removal fails, so force_download stands exactly as before."""
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
    """``None`` from ``_incomplete_partial_names`` means the cache could not be READ, which is not the
    same answer as "no partials left" and must never be read as proof of a clear -- a remount or a
    permission flap would otherwise hand the retry an unsafe resume. Isolated: this partial is old
    enough that the scoped pass does remove it, so only the missing proof forces."""
    _build_cache(tmp_path, partial_age_s = 1800.0)
    monkeypatch.setattr(xf, "_incomplete_partial_names", lambda *a, **k: None)
    attempt = _run_ladder(monkeypatch, tmp_path, inject_hook = True, owned = None)
    assert _http_force(attempt) is True


def test_the_scoped_clear_keeps_a_caller_requested_force_download(monkeypatch, tmp_path):
    """Scoping decides whether the LADDER adds a force, never whether the caller's own one survives:
    Unsloth's model-update path passes force_download=True to re-fetch a republished blob."""
    _build_cache(tmp_path, partial_age_s = 1800.0)
    attempt = _run_ladder(
        monkeypatch, tmp_path, caller_force = True,
        owned = [_blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX],
    )
    assert [c.force_download for c in attempt.calls] == [True, True]


def test_the_latch_still_holds_force_download_while_a_partial_survives(monkeypatch, tmp_path):
    """Once the ladder HAS forced, the latch rule is unchanged: a forced HTTP child that failed
    before replacing the partial leaves the next child forced too, because only the partial's
    ABSENCE proves a resume is safe."""
    _build_cache(tmp_path, partial_age_s = 10.0)          # unremovable: possibly a live sibling's
    snap = tmp_path / REPO_DIR / "snapshots" / REV
    attempt = _run_ladder(
        monkeypatch, tmp_path, owned = None,
        results = [("stall", STALL), ("crashed", "died in HEAD"), ("ok", str(snap))],
    )
    assert [c.force_download for c in attempt.calls] == [False, True, True]


def test_a_cancel_at_the_stall_runs_no_clearance_at_all(monkeypatch, tmp_path):
    """Cancellation still wins over every failure verdict, and it wins BEFORE the transition: the
    user's decision must not spend a destructive purge, scoped or not, on a download they abandoned."""
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


# ---------------------------------------------------------------------------------------------
# Ownership evidence for a SNAPSHOT child. Without it the transition above has nothing to scope
# to wherever open files cannot be inspected (no psutil and no /proc, i.e. Windows), because the
# killed child's own seconds-old partial is spared by the patient grace.
# ---------------------------------------------------------------------------------------------


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
    """A child that opens one new partial and then wedges."""

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
    """The partial that appeared AFTER the spawn is provably this child's, so the HTTP prep may
    clear it; the one that was there before is not claimed. A snapshot used to capture no ownership
    at all, which is why the reporter's cached shards were re-downloaded (issue #9094)."""
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
    """The baseline is ownership evidence only. Narrowing what a SNAPSHOT measures would let a
    sibling's growth mask a wedged child, and hub serialises same-file callers on an unbounded lock,
    so the repo-wide measurement stays."""
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
    """AGE IS NOT PROOF THAT A WRITER EXITED.

    A sibling downloader that is paused, blocked on a slow connection or waiting on a retry leaves a
    partial older than any grace and is still going to finish it. Deleting it wastes the transfer
    and makes its final rename fail, and on POSIX it can be deleted out from under an open handle
    without the writer noticing. The unscoped pass therefore asks the open-file table, and a blob
    that is open stays whatever its age, so the caller forces exactly as it did before.
    """
    _build_cache(tmp_path, partial_age_s = 1800.0)
    stranger = _blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX
    monkeypatch.setattr(xf, "_blobs_with_a_live_writer", lambda: {stranger})

    survivors = xf._clear_unsafe_partials_for_http(
        "model", REPO, cache_dir = str(tmp_path), active_grace = 180.0,
    )
    assert survivors == {stranger}, "an aged partial with a live writer was deleted"
    assert (tmp_path / REPO_DIR / "blobs" / stranger).exists()


def test_an_unreadable_open_file_table_declines_the_unscoped_pass(monkeypatch, tmp_path):
    """No psutil, or a platform that will not list open files. The question cannot be put, so the
    purge is not widened on age alone: the partial survives and the caller forces, which is what it
    did before this pass existed. Failing the other way would make the least inspectable hosts the
    most destructive ones."""
    _build_cache(tmp_path, partial_age_s = 1800.0)
    stranger = _blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX
    monkeypatch.setattr(xf, "_blobs_with_a_live_writer", lambda: None)

    survivors = xf._clear_unsafe_partials_for_http(
        "model", REPO, cache_dir = str(tmp_path), active_grace = 180.0,
    )
    assert survivors == {stranger}
    assert (tmp_path / REPO_DIR / "blobs" / stranger).exists()


def test_our_own_partial_is_cleared_even_with_the_table_unreadable(monkeypatch, tmp_path):
    """The ownership set is separate evidence and does not depend on the table: our own dead child
    wrote it, so there is no live writer to protect. #9094 is fixed by THIS pass, and the unscoped
    one only reaches an earlier crashed attempt's leftovers."""
    _build_cache(tmp_path, partial_age_s = 5.0)
    mine = _blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX
    monkeypatch.setattr(xf, "_blobs_with_a_live_writer", lambda: None)

    survivors = xf._clear_unsafe_partials_for_http(
        "model", REPO, cache_dir = str(tmp_path), active_grace = 180.0,
        owned_incomplete_blobs = {mine},
    )
    assert survivors == set()
    assert not (tmp_path / REPO_DIR / "blobs" / mine).exists()


def test_the_live_writer_probe_finds_this_processs_own_open_partial(tmp_path):
    """The probe itself, unfaked, against a real open file descriptor on this host.

    Skipped where the open-file table cannot be read, which is the same condition the caller
    declines on, so a host that skips this is a host the pass never runs on.
    """
    blob = tmp_path / ("deadbeef" + xf.INCOMPLETE_SUFFIX)
    blob.write_bytes(b"x")
    with blob.open("ab") as handle:
        handle.write(b"y")
        handle.flush()
        seen = xf._blobs_with_a_live_writer()
        if seen is None:
            pytest.skip("this host cannot read the open-file table")
        assert blob.name in seen, "an open partial was not seen as having a live writer"
    # closed: nothing holds it now
    after = xf._blobs_with_a_live_writer()
    assert after is not None and blob.name not in after


def test_clear_unsafe_partials_reports_survivors_and_spares_a_fresh_stranger(monkeypatch, tmp_path):
    """The helper's contract, directly: ours goes whatever its age, a stranger goes only once past
    the grace AND with no live writer, and the return value names what SURVIVED (``None`` for an
    unreadable cache). The open-file table is pinned to empty so the case is about the grace rather
    than about whatever else this host happens to be downloading."""
    monkeypatch.setattr(xf, "_blobs_with_a_live_writer", lambda: set())
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
    # Age the stranger past the grace, with no process holding it open: now it is clearable.
    os.utime(blobs / stranger, (time.time() - 1800.0, time.time() - 1800.0))
    assert xf._clear_unsafe_partials_for_http(
        "model", REPO, cache_dir = str(tmp_path), active_grace = 180.0,
    ) == set()
    assert xf._clear_unsafe_partials_for_http(
        "model", REPO, cache_dir = str(tmp_path / "gone"), active_grace = 180.0,
    ) is None, "an unreadable cache reports None, never an empty set"
