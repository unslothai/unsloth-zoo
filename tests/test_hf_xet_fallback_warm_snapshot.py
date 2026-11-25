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

"""Warm snapshot short-circuit from one Hub metadata call (offline, mock-based, plus one local-socket exit test).

A glob allow list (tokenizer / adapter pre-download) used to spawn the download child even when every file was
cached. The Hub's file list for the commit is now checked in-process first; only a provable "nothing to fetch" skips
the child, every other outcome spawns it as before."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
import time
import types
from pathlib import Path

import pytest

import huggingface_hub
from huggingface_hub import constants as hf_constants
from huggingface_hub.hf_api import RepoFile

from unsloth_zoo_downloads import hf_xet_fallback as xf

REPO = "ztest/dry-run"
SHA = "a" * 40
PATTERNS = ["config.json", "tokenizer*", "*.py"]


class _Attempt:
    """Stands in for the spawn child: records calls, returns a sentinel path."""

    def __init__(self):
        self.calls = []

    def __call__(self, repo_id, **kwargs):
        self.calls.append(kwargs)
        return "ok", "/child/snapshot"


def _cache(tmp_path: Path, files = ("config.json", "tokenizer.json"), ref = SHA) -> Path:
    storage = tmp_path / ("models--" + REPO.replace("/", "--"))
    snap = storage / "snapshots" / SHA
    snap.mkdir(parents = True)
    (storage / "blobs").mkdir()
    for i, name in enumerate(files):
        blob = storage / "blobs" / f"blob{i}"
        blob.write_text("{}")
        (snap / name).parent.mkdir(parents = True, exist_ok = True)
        (snap / name).symlink_to(blob)
    (storage / "refs").mkdir()
    (storage / "refs" / "main").write_text(ref)
    return snap


def _hub(monkeypatch, snap: Path, files, *, sha = SHA, info_result = None, tree = None):
    """local_files_only returns the snapshot; HfApi.repo_info reports sha + siblings (or info_result raises / hangs)."""
    seen = {"repo_info": [], "list_repo_tree": [], "snapshot_download": []}

    def _snapshot_download(**kwargs):
        seen["snapshot_download"].append(kwargs)
        assert kwargs.get("local_files_only") is True, "the parent must never download in-process"
        return str(snap)

    class _Api:
        def __init__(self, token = None, **kwargs):
            self.token = token

        def repo_info(self, repo_id, *, revision = None, repo_type = None, timeout = None, **kwargs):
            seen["repo_info"].append(dict(repo_id = repo_id, revision = revision, repo_type = repo_type,
                                          timeout = timeout, token = self.token))
            if info_result is not None:
                result = info_result()
                if isinstance(result, BaseException):
                    raise result
            return types.SimpleNamespace(
                sha = sha, siblings = [types.SimpleNamespace(rfilename = f) for f in files] if files is not None else None,
            )

        def list_repo_tree(self, repo_id, *, recursive = False, revision = None, repo_type = None, **kwargs):
            seen["list_repo_tree"].append(dict(repo_id = repo_id, revision = revision, recursive = recursive))
            return [RepoFile(path = f, size = 2, oid = "0" * 40, type = "file") for f in (tree or [])]

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _snapshot_download)
    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    return seen


@pytest.fixture(autouse = True)
def _isolated(monkeypatch):
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", False)
    # Default: a hub without the tree listing cache; tests that need one install it.
    monkeypatch.setattr(xf, "_hub_tree_cache_module", lambda: None)
    monkeypatch.setattr(xf, "_default_prepare_for_http", lambda *a, **k: None)
    monkeypatch.setenv("UNSLOTH_HTTP_RETRY_BACKOFF", "0")
    attempt = _Attempt()
    monkeypatch.setattr(xf, "_run_download_attempt", attempt)
    return attempt


def _call(tmp_path, **kwargs):
    kwargs.setdefault("allow_patterns", PATTERNS)
    return xf.snapshot_download_with_xet_fallback(REPO, token = "hf_tok", cache_dir = str(tmp_path), **kwargs)


def test_nothing_to_fetch_returns_snapshot_without_child(tmp_path, monkeypatch, _isolated):
    snap = _cache(tmp_path)
    seen = _hub(monkeypatch, snap, ["config.json", "tokenizer.json", "model.safetensors", "README.md"])
    assert _call(tmp_path) == str(snap)
    assert _isolated.calls == []
    # One bounded call, with the child's token / revision / repo type.
    assert seen["repo_info"] == [dict(repo_id = REPO, revision = "main", repo_type = "model",
                                      timeout = xf.DEFAULT_METADATA_TIMEOUT, token = "hf_tok")]
    assert seen["list_repo_tree"] == []


def test_nested_file_and_commit_revision(tmp_path, monkeypatch, _isolated):
    snap = _cache(tmp_path, files = ("config.json", "additional_chat_templates/a.jinja"))
    seen = _hub(monkeypatch, snap, ["config.json", "additional_chat_templates/a.jinja"])
    assert _call(tmp_path, revision = SHA, allow_patterns = ["config.json", "additional_chat_templates/*.jinja"]) == str(snap)
    assert _isolated.calls == []
    assert seen["repo_info"][0]["revision"] == SHA


def test_ignore_patterns_follow_hub_filter(tmp_path, monkeypatch, _isolated):
    snap = _cache(tmp_path, files = ("adapter_config.json", "adapter_model.safetensors"))
    _hub(monkeypatch, snap, ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"])
    out = _call(tmp_path, allow_patterns = ["adapter_config.json", "adapter_model*"],
                ignore_patterns = ["adapter_model*.bin"])
    assert out == str(snap) and _isolated.calls == []


def test_file_to_fetch_spawns_child(tmp_path, monkeypatch, _isolated):
    snap = _cache(tmp_path)
    _hub(monkeypatch, snap, ["config.json", "tokenizer.json", "tokenizer.model"])
    assert _call(tmp_path) == "/child/snapshot"
    assert len(_isolated.calls) == 1


def test_dangling_pointer_spawns_child(tmp_path, monkeypatch, _isolated):
    snap = _cache(tmp_path)
    (snap / "tokenizer.json").resolve().unlink()  # blob gone, pointer dangles
    _hub(monkeypatch, snap, ["config.json", "tokenizer.json"])
    assert _call(tmp_path) == "/child/snapshot"
    assert len(_isolated.calls) == 1


def test_newer_remote_commit_spawns_child(tmp_path, monkeypatch, _isolated):
    snap = _cache(tmp_path)
    _hub(monkeypatch, snap, ["config.json", "tokenizer.json"], sha = "b" * 40)
    assert _call(tmp_path) == "/child/snapshot"
    assert len(_isolated.calls) == 1


@pytest.mark.parametrize("info_result", [
    lambda: RuntimeError("401 Unauthorized"),
    lambda: OSError("network down"),
])
def test_metadata_error_spawns_child(tmp_path, monkeypatch, _isolated, info_result):
    snap = _cache(tmp_path)
    _hub(monkeypatch, snap, ["config.json"], info_result = info_result)
    assert _call(tmp_path) == "/child/snapshot"
    assert len(_isolated.calls) == 1


def test_nothing_matched_spawns_child(tmp_path, monkeypatch, _isolated):
    snap = _cache(tmp_path)
    _hub(monkeypatch, snap, ["README.md"])
    assert _call(tmp_path) == "/child/snapshot"
    assert len(_isolated.calls) == 1


@pytest.mark.parametrize("siblings", [None, [], [f"f{i}.bin" for i in range(1001)]])
def test_unreliable_siblings_use_the_full_tree_listing(tmp_path, monkeypatch, _isolated, siblings):
    """hub lists the tree when siblings are absent, empty or very many; so do we (a superset is the safe side)."""
    snap = _cache(tmp_path)
    seen = _hub(monkeypatch, snap, siblings, tree = ["config.json", "tokenizer.json"])
    assert _call(tmp_path) == str(snap)
    assert seen["list_repo_tree"] == [dict(repo_id = REPO, revision = SHA, recursive = True)]
    seen = _hub(monkeypatch, snap, siblings, tree = ["config.json", "tokenizer.json", "tokenizer.model"])
    assert _call(tmp_path) == "/child/snapshot"


def test_offline_skips_metadata(tmp_path, monkeypatch, _isolated):
    snap = _cache(tmp_path)
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", True)
    seen = _hub(monkeypatch, snap, ["config.json", "tokenizer.json"])
    assert _call(tmp_path) == "/child/snapshot"
    assert seen["repo_info"] == []


def test_force_download_skips_metadata(tmp_path, monkeypatch, _isolated):
    snap = _cache(tmp_path)
    seen = _hub(monkeypatch, snap, ["config.json", "tokenizer.json"])
    assert _call(tmp_path, force_download = True) == "/child/snapshot"
    assert seen["repo_info"] == [] and seen["snapshot_download"] == []
    assert _isolated.calls[0]["params"]["force_download"] is True


def test_uncached_revision_skips_metadata(tmp_path, monkeypatch, _isolated):
    """No local snapshot at all: a download is certain, so no metadata round trip is spent before the child."""
    seen = _hub(monkeypatch, tmp_path / "unused", ["config.json"])

    def _miss(**kwargs):
        raise FileNotFoundError("not cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _miss)
    assert _call(tmp_path) == "/child/snapshot"
    assert seen["repo_info"] == []


def test_stale_branch_ref_spawns_child(tmp_path, monkeypatch, _isolated):
    snap = _cache(tmp_path)
    (snap.parent.parent / "refs" / "dev").write_text("c" * 40)
    _hub(monkeypatch, snap, ["config.json", "tokenizer.json"])
    assert _call(tmp_path, revision = "dev") == "/child/snapshot"
    assert len(_isolated.calls) == 1


def test_tree_cache_missing_spawns_child_present_uses_it(tmp_path, monkeypatch, _isolated):
    """On a hub that keeps trees/<commit>.json the child writes it when absent; defer then, and once it exists use its
    listing (what the child filters) without any listing request; a commit revision then needs no request at all."""
    snap = _cache(tmp_path)
    storage = snap.parent.parent
    tree_file = storage / "trees" / f"{SHA}.json"
    listing = {"config.json": object(), "tokenizer.json": object()}
    fake_tree = types.SimpleNamespace(
        _tree_cache_path = lambda folder, commit: str(Path(folder) / "trees" / f"{commit}.json"),
        read_tree_cache = lambda folder, commit: listing if tree_file.is_file() else None,
    )
    monkeypatch.setattr(xf, "_hub_tree_cache_module", lambda: fake_tree)
    seen = _hub(monkeypatch, snap, ["config.json", "tokenizer.json"])
    assert _call(tmp_path) == "/child/snapshot"
    assert len(_isolated.calls) == 1
    tree_file.parent.mkdir()
    tree_file.write_text("{}")
    assert _call(tmp_path) == str(snap)
    assert len(_isolated.calls) == 1
    before = len(seen["repo_info"])
    assert _call(tmp_path, revision = SHA) == str(snap)
    assert len(seen["repo_info"]) == before and seen["list_repo_tree"] == []
    listing["tokenizer.model"] = object()  # the cached listing names a file not on disk
    assert _call(tmp_path) == "/child/snapshot"


def test_hung_metadata_hands_off_to_child(tmp_path, monkeypatch, _isolated):
    snap = _cache(tmp_path)
    release = threading.Event()
    monkeypatch.setattr(xf, "DEFAULT_METADATA_TIMEOUT", 0.3)
    _hub(monkeypatch, snap, ["config.json"], info_result = lambda: release.wait(30) and None)
    started = time.monotonic()
    try:
        assert _call(tmp_path) == "/child/snapshot"
        assert time.monotonic() - started < 5
        assert len(_isolated.calls) == 1
    finally:
        release.set()


def test_cancel_during_metadata_raises_without_child(tmp_path, monkeypatch, _isolated):
    snap = _cache(tmp_path)
    release = threading.Event()
    cancel = threading.Event()

    def _hang():
        cancel.set()
        release.wait(30)

    _hub(monkeypatch, snap, ["config.json"], info_result = _hang)
    try:
        with pytest.raises(RuntimeError, match = "Cancelled"):
            _call(tmp_path, cancel_event = cancel)
        assert _isolated.calls == []
    finally:
        release.set()


def test_incomplete_result_still_goes_to_child(tmp_path, monkeypatch, _isolated):
    """The child's post-download gate also guards this answer: a weight request with no readable weight spawns."""
    snap = _cache(tmp_path)
    _hub(monkeypatch, snap, ["config.json", "tokenizer.json"])
    out = xf.snapshot_download_with_xet_fallback(
        REPO, token = None, cache_dir = str(tmp_path), allow_patterns = ["*.json", "*.safetensors"]
    )
    assert out == "/child/snapshot"
    assert len(_isolated.calls) == 1


_EXIT_SCRIPT = textwrap.dedent('''
    import http.server, json, os, socketserver, sys, threading, time
    cache_dir, mode = sys.argv[1], sys.argv[2]
    sha = "a" * 40
    storage = os.path.join(cache_dir, "models--ztest--exit")
    snap = os.path.join(storage, "snapshots", sha)
    os.makedirs(snap); os.makedirs(os.path.join(storage, "blobs")); os.makedirs(os.path.join(storage, "refs"))
    open(os.path.join(storage, "refs", "main"), "w").write(sha)
    open(os.path.join(storage, "blobs", "b0"), "w").write("{}")
    os.symlink(os.path.join(storage, "blobs", "b0"), os.path.join(snap, "config.json"))
    siblings = [{"rfilename": "config.json"}, {"rfilename": "tokenizer.json"}]  # tokenizer.json is not cached

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass
        def do_HEAD(self):
            time.sleep(600)
        def do_GET(self):
            if mode == "stall":
                time.sleep(600)
            body = json.dumps({"id": "ztest/exit", "sha": sha, "siblings": siblings}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    srv = S(("127.0.0.1", 0), H)
    threading.Thread(target = srv.serve_forever, daemon = True).start()
    os.environ["HF_ENDPOINT"] = "http://127.0.0.1:%d" % srv.server_address[1]
    from unsloth_zoo_downloads import hf_xet_fallback as xf
    xf.DEFAULT_METADATA_TIMEOUT = 0.5
    xf._run_download_attempt = lambda *a, **k: ("ok", "/child/snapshot")
    out = xf.snapshot_download_with_xet_fallback("ztest/exit", cache_dir = cache_dir, allow_patterns = ["*.json"])
    print(json.dumps({"out": out, "threads": [t.name for t in threading.enumerate()
                                              if not t.daemon and t is not threading.main_thread()]}), flush = True)
''')


@pytest.mark.parametrize("mode", ["stall", "partial"])
def test_abandoned_metadata_call_does_not_delay_exit(tmp_path, mode):
    """A stalled Hub (repo_info never answers) or a partial cache with stalling per-file HEADs must not leave any
    non-daemon thread behind: the process exits as soon as the child hand-off returns."""
    root = Path(xf.__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(root), os.environ.get("PYTHONPATH", "")])}
    env.pop("HF_HUB_OFFLINE", None)
    started = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-c", _EXIT_SCRIPT, str(tmp_path / "cache"), mode],
        capture_output = True, text = True, env = env, timeout = 60,
    )
    elapsed = time.monotonic() - started
    assert proc.returncode == 0, proc.stderr[-2000:]
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert report == {"out": "/child/snapshot", "threads": []}
    assert elapsed < 20, f"interpreter exit took {elapsed:.1f}s"
