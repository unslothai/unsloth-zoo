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

"""The Xet -> HTTP transition must scope a forced re-download to the blob at fault (#9094):
``force_download`` is repo-wide, so one surviving partial cost every completed shard."""

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


# Package placeholder for hf_xet_fallback's imports; restored after, or it shadows the real one.
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
    """Hub skips an already-resolvable pointer only while ``force_download`` is unset
    (huggingface_hub 1.31 ``file_download.py`` :1204 / :1248)."""

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
    """AGE IS NOT PROOF THAT A WRITER EXITED: a paused or retrying sibling leaves a partial older
    than any grace and will still finish it, so the open-file table decides, not the mtime."""
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
    """Unanswerable, so the purge is not widened on age alone: failing the other way would make the
    least inspectable hosts the most destructive."""
    _build_cache(tmp_path, partial_age_s = 1800.0)
    stranger = _blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX
    monkeypatch.setattr(xf, "_partial_paths_with_a_live_writer", lambda: None)

    survivors = xf._clear_unsafe_partials_for_http(
        "model", REPO, cache_dir = str(tmp_path), active_grace = 180.0,
    )
    assert survivors == {stranger}
    assert (tmp_path / REPO_DIR / "blobs" / stranger).exists()


def test_our_own_partial_is_cleared_even_with_the_table_unreadable(monkeypatch, tmp_path):
    """The ownership set is separate evidence: our own dead child wrote it, so there is no live
    writer to protect. #9094 is fixed by THIS pass."""
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
    """`process_iter` walks the whole host, and hub names a partial after the file's etag, which
    identical files share across repositories. On a bare name, the partial repo A is downloading
    right now marked repo B's long-dead partial of the same file as live: the scoped clearance
    spared it, `has_active_incomplete_blobs` kept reading active, and repo B fell back to the
    repo-wide `force_download` that re-fetches every verified shard -- the exact outcome this
    whole path exists to avoid."""
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
    """The ownership inference still compares names, because the listing it subtracts from is
    this repo's own. So the host-wide walk is projected onto this repo's blobs directories
    first: a writer busy elsewhere must not remove a name from what the killed child owns."""
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
    """Ours goes at any age, a stranger only past the grace AND with no live writer; the return
    names what SURVIVED. The table is pinned empty so the case is about the grace."""
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
    """Ownership is ``current - baseline`` and an owned blob is exempt from the grace, so a failed
    scan reading as an empty baseline claims a live sibling's partial and unlinks it. ``None`` is
    the vocabulary for "unknown" at that site."""
    _build_cache(tmp_path, partial_age_s = 5.0)

    present = xf._baseline_incomplete_blob_names("model", REPO, cache_dir = str(tmp_path))
    assert isinstance(present, set) and present, "a readable cache reports the names it has"

    # Absent cache: a genuinely empty baseline, not a failure.
    assert xf._baseline_incomplete_blob_names(
        "model", REPO, cache_dir = str(tmp_path / "no-such-cache"),
    ) == set()

    # A scan that raises is UNKNOWN, the case the empty set used to swallow. Patched on the
    # strict resolver rather than on iter_active_repo_cache_dirs, because delegating to that
    # one was the defect: it catches the root's own OSError and yields nothing, so the raise
    # never reached this function and a permission flap read as an empty baseline.
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

    # And the real thing: a root that cannot be listed is unknown, while a root that is merely
    # not there yet is the honestly empty baseline of a first download.
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
    """Eligibility and deletion are two moments, and the cache moves in between.

    Another downloader finalizing the blob makes this pointer VALID: hub then leaves it alone
    because it already resolves, so unlinking it here deletes the only record of a file that
    is on disk -- and a link under an older revision is not recreated by the current retry,
    which breaks a later offline load with the blob cached. Same for a partial appearing
    beside the target, which names a download in progress.
    """
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

    # The scan is pinned to what it found, and the sibling finishes the blob after it: this is
    # the window between the eligibility list and the unlink that walks it.
    monkeypatch.setattr(
        xf, "_orphan_snapshot_links_safe_to_clear", lambda *args, **kwargs: [orphan],
    )
    target.write_bytes(_file_bytes("tokenizer.model"))
    assert xf._clear_orphan_snapshot_links("model", REPO, str(tmp_path)) == set()
    assert orphan.is_symlink() and orphan.exists(), (
        "a pointer that had become valid was unlinked, so the cached blob lost its only name"
    )

    # And the other way the moment can change: a partial appears beside the target, which
    # names a download in progress.
    target.unlink()
    (blobs / (_blob_name("tokenizer.model") + xf.INCOMPLETE_SUFFIX)).write_bytes(b"\xa5" * 32)
    assert xf._clear_orphan_snapshot_links("model", REPO, str(tmp_path)) == set()
    assert orphan.is_symlink(), "a link whose blob is being written was unlinked"


def test_a_nonce_suffixed_partial_still_says_the_blob_is_being_written(monkeypatch, tmp_path):
    """Current hub does not write `<etag>.incomplete` into the blobs directory.

    It downloads to a process-unique `<etag>.<nonce>.incomplete` and renames, because a shared
    name corrupts the cache wherever `flock` silently succeeds for every caller (huggingface_hub
    PR 4228). Checking only the exact name missed a live sibling's partial, so a dangling link
    whose blob was being downloaded right now was unlinked -- and a sibling targeting another
    revision never recreates this older pointer, so a later offline load fails with the blob
    cached.
    """
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
    """`process_iter` walks THIS PID namespace, not the host.

    A sibling container or pod sharing the cache volume under the same numeric UID is simply
    absent from the listing: no AccessDenied, no exception, nothing that makes the walk record
    itself as incomplete, and the private-cache test does not help because a volume shared
    only between containers running as the same UID looks owner-only to both. So the walk is
    not proof there, and an aged partial that sibling is still writing was whitelisted and
    unlinked, failing its eventual rename.
    """
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


def test_the_namespace_test_reads_containerisation_and_where_the_cache_lives(monkeypatch, tmp_path):
    """An ordinary host is unaffected; inside a container the cache's own mount decides.

    On the container's root filesystem nobody outside can be writing into it. A separate
    mount is the shared volume this case is about, and there the walk stops being proof.
    """
    if os.name == "nt" or not os.path.isdir("/proc"):
        pytest.skip("POSIX containers only")

    monkeypatch.setattr(xf, "_running_in_a_container", lambda: False)
    assert xf._process_walk_sees_every_writer(str(tmp_path)) is True, (
        "an ordinary host lost the purge it is entitled to"
    )

    monkeypatch.setattr(xf, "_running_in_a_container", lambda: True)
    # The workspace lives on its own mount here, so this IS the shared-volume shape.
    assert os.stat(tmp_path).st_dev != os.stat("/").st_dev, (
        "this host cannot pose the case: the temp dir is on the root filesystem"
    )
    assert xf._process_walk_sees_every_writer(str(tmp_path)) is False
    # And a cache on the container's own root filesystem still answers yes.
    monkeypatch.setattr(xf, "hf_cache_root", lambda cache_dir = None: Path("/"))
    assert xf._process_walk_sees_every_writer(str(tmp_path)) is True


def test_a_deletion_time_walk_that_could_not_read_every_process_declines(monkeypatch, tmp_path):
    """The re-scan at the deletion has to answer to the same gate as the eligibility scan.

    A walk that could not inspect every process is a lower bound, so "no writer holds this" is
    not a fact, and on a cache another UID can write into, the process it could not read is a
    likely writer. Without the gate, the deletion-time reading overturned the decision the
    eligibility scan had correctly declined to make.
    """
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
    """The whitelist is the result of an earlier scan, not a fact about our own dead child.

    Ownership exempts a blob from the age and live-writer guards, which is right for the
    partial a killed child was writing and wrong for one a scan merely judged idle: hub reuses
    a deterministic `<etag>.incomplete` path, so a sibling can open exactly that file in
    between, and the whitelist would then unlink an active download.
    """
    _build_cache(tmp_path, partial_age_s = 1800.0)
    blobs = tmp_path / REPO_DIR / "blobs"
    stale = _blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX

    seen = {"scans": 0}

    def _walk():
        # Nobody is writing when eligibility is decided; a sibling has it open by the time
        # the deletion runs.
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
    """The guard reads dangling links too, and an older interrupted download leaves one with no
    partial beside it at all.

    Nothing in either blob-scoped pass could see it -- there is no `.incomplete` to delete -- so
    `has_active_incomplete_blobs` stayed true, the clearance reported failure, and the caller
    fell back to the repo-wide `force_download` that re-fetches every verified shard (#9094),
    for a symlink pointing at nothing.
    """
    monkeypatch.setattr(xf, "_partial_paths_with_a_live_writer", lambda: set())
    snap = _build_cache(tmp_path, partial_age_s = 5.0)
    blobs = tmp_path / REPO_DIR / "blobs"
    mine = _blob_name(IN_FLIGHT) + xf.INCOMPLETE_SUFFIX
    (blobs / mine).unlink()
    # What an older crash leaves: the link was written, the blob never was, and no partial
    # remains to say a download is in progress.
    orphan = snap / "tokenizer.model"
    orphan.symlink_to(os.path.relpath(blobs / _blob_name("tokenizer.model"), snap))
    assert orphan.is_symlink() and not orphan.exists()

    # The module handle this file already loaded hermetically, not a package import: the real
    # `unsloth_zoo/__init__` raises when the separate `unsloth` install is missing, and the CI
    # job that runs this suite standalone installs it best-effort.
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
    """A link with an `.incomplete` partner names a download that may still be running.

    Hub creates the link when it finalises the blob, so removing it would delete the record of a
    file a live writer is about to complete -- and while that partial survives, the repo-wide
    force is the right answer anyway.
    """
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
    """Absence is the only evidence the caller accepts, so a link this pass could not remove has
    to be named rather than silently omitted."""
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
    """The open-file walk is a LOWER bound, and it was being read as proof of absence.

    A sibling downloader under another UID raises AccessDenied on `open_files()`, so its
    partial looks writer-less; past the grace it was unlinked mid-write. Age cannot rule that
    out -- a paused or retrying writer touches nothing for minutes -- which is why the probe
    exists in the first place.
    """
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
    # Shared: another user can write into this cache, so the process we could not read is a
    # possible writer here.
    monkeypatch.setattr(xf, "_cache_is_private_to_this_user", lambda _cache_dir = None, **_kwargs: False)
    assert xf._partial_paths_with_a_live_writer() == set()
    assert xf._live_writer_walk_was_complete() is False
    assert xf._unowned_partials_safe_to_clear(
        "model", REPO, str(tmp_path), 180.0, None,
    ) is None
    assert (tmp_path / REPO_DIR / "blobs" / stranger).exists()

    # Private: nobody else can write here, so the processes we could not read cannot be
    # writing into THIS cache and the lower bound is exact for it.
    monkeypatch.setattr(xf, "_cache_is_private_to_this_user", lambda _cache_dir = None, **_kwargs: True)
    assert xf._unowned_partials_safe_to_clear(
        "model", REPO, str(tmp_path), 180.0, None,
    ) == {stranger}

    # A process that simply exited between the listing and the read holds nothing, so it does
    # not make the walk incomplete.
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
        # Group-writable is shared only when somebody else is in the group. Under the
        # user-private-group scheme (Fedora, RHEL, any host at umask 002) the group is this
        # user alone, and declining there would turn the purge off on all of those hosts --
        # so the expectation is read from the same rule rather than from this host's groups.
        os.chmod(cache, 0o770)
        assert xf._cache_is_private_to_this_user(str(cache)) is (
            xf._group_is_private_to_this_user(os.getegid())
        )
    # A cache that is not there at all cannot be established as private.
    assert xf._cache_is_private_to_this_user(str(tmp_path / "no-such-cache")) is False


@pytest.mark.skipif(not hasattr(os, "geteuid"), reason = "POSIX permissions only")
def test_a_shared_directory_under_a_private_root_is_still_shared(monkeypatch, tmp_path):
    """The root says nothing about what is underneath it.

    A 0700 root owned by this user with a group-writable repo directory inside it is a cache
    another UID can put a partial in, and `blobs` is where the partials actually live. Read
    off the root alone, an incomplete process walk was called trustworthy there, and an aged
    partial that sibling was still writing could be unlinked mid-write.
    """
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
    """`Path.is_dir()` answers False for "not allowed to look" as well as for "not there".

    A permission or FUSE flap on the repo directory therefore reported NO partials, which is
    the empty answer that releases the guard -- and an unforced HTTP child then resumes the
    sparse Xet partial onto a finalized blob. Only ENOENT and ENOTDIR are absence.

    Python 3.13 began propagating that error from `is_dir()`, so on THIS interpreter the old
    code already answered None. The scans are read by 3.9 through 3.12 as well, where it is
    swallowed, which is why the helper is asserted directly below as well as through them.
    """
    _build_cache(tmp_path, partial_age_s = 5.0)
    repo = tmp_path / REPO_DIR
    os.chmod(repo, 0o000)
    try:
        if os.access(repo, os.R_OK):
            pytest.skip("this user can read a 0o000 directory (root), so the flap cannot be posed")
        # The helper itself, on every supported version: absence is answered, and anything
        # else is raised so the caller reports None rather than an empty set.
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
    """Per-blob tolerance is wrong on THIS scan.

    Ownership is `current - baseline`, so a name the baseline misses is a name credited to our
    own child: exempt from the age and live-writer guards, and unlinked. A transient failure on
    one entry that clears before the stall-time scan is exactly that, so an entry that cannot be
    read makes the whole baseline unknown.
    """
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
    """Appearing after the baseline says WHEN, not WHO.

    The subtraction is the fallback used wherever the child's own open files cannot be read,
    and a same-repo sibling downloading beside this one creates partials in the same window.
    Claiming one hands it to a purge scoped by ownership, which skips the age and live-writer
    guards entirely -- so the sibling's active download is deleted mid-write.
    """
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

    # And with the writer walk unusable, nothing is claimed by subtraction at all: the purge
    # falls back to unscoped, where the age and live-writer guards still apply.
    monkeypatch.setattr(xf, "_partial_paths_with_a_live_writer", lambda: None)
    params = {"repo_id": REPO, "revision": REV, "cache_dir": str(tmp_path)}
    xf._run_download_attempt(
        REPO, kind = "snapshot", params = params, token = None, repo_type = "model",
        disable_xet = False, cancel_event = None, stall_timeout = 0.3, interval = 0.05,
        grace_period = 0.1, on_status = None,
    )
    assert params.get("_owned_incomplete_blobs") is None


def test_the_walk_completeness_is_per_thread(monkeypatch):
    """Two downloads run this clearance concurrently.

    Communicating the completeness through a module global let a complete walk in one thread
    overwrite the incomplete verdict another thread had not read yet -- and that thread is
    exactly the one that would then unlink a partial held by a process it could not inspect.
    """
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
