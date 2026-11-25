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

"""Warm snapshot short-circuit via hub's metadata-only dry run (offline, mock-based).

A glob allow list (tokenizer / adapter pre-download) used to spawn the download child even when every file was
cached. snapshot_download(dry_run=True) now answers in-process first; only a provable "nothing to fetch" skips
the child, every other outcome spawns it as before."""

from __future__ import annotations

import threading
import time
import types
from pathlib import Path

import pytest

import huggingface_hub
from huggingface_hub import constants as hf_constants

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
        (snap / name).symlink_to(blob)
    (storage / "refs").mkdir()
    (storage / "refs" / "main").write_text(ref)
    return snap


def _info(snap: Path, name: str, *, cached = True, will_download = False, commit = SHA):
    return types.SimpleNamespace(
        commit_hash = commit, file_size = 2, filename = name, local_path = str(snap / name),
        is_cached = cached, will_download = will_download,
    )


def _hub(monkeypatch, snap: Path, dry_result, *, seen = None, has_dry_run = True):
    """Fake snapshot_download: local_files_only returns the snapshot, dry_run returns / raises dry_result."""
    seen = seen if seen is not None else []

    def _real_signature(*, repo_id, repo_type = None, revision = None, token = None, cache_dir = None,
                        allow_patterns = None, ignore_patterns = None, local_files_only = False,
                        dry_run = False, tqdm_class = None):
        seen.append(dict(repo_id = repo_id, repo_type = repo_type, revision = revision, token = token,
                         cache_dir = cache_dir, allow_patterns = allow_patterns,
                         ignore_patterns = ignore_patterns, local_files_only = local_files_only,
                         dry_run = dry_run, tqdm_class = tqdm_class))
        if local_files_only:
            return str(snap)
        assert dry_run, "the parent must never run a real download in-process"
        result = dry_result() if callable(dry_result) else dry_result
        if isinstance(result, BaseException):
            raise result
        return result

    if has_dry_run:
        fake = _real_signature
    else:
        def fake(*, repo_id, repo_type = None, revision = None, token = None, cache_dir = None,
                 allow_patterns = None, ignore_patterns = None, local_files_only = False):
            return _real_signature(repo_id = repo_id, repo_type = repo_type, revision = revision, token = token,
                                   cache_dir = cache_dir, allow_patterns = allow_patterns,
                                   ignore_patterns = ignore_patterns, local_files_only = local_files_only)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    return seen


@pytest.fixture(autouse = True)
def _isolated(monkeypatch):
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", False)
    # Default: a hub without the tree listing cache; tests that need one install it.
    monkeypatch.setattr(huggingface_hub, "_tree_cache", None, raising = False)
    monkeypatch.setattr(xf, "_default_prepare_for_http", lambda *a, **k: None)
    monkeypatch.setenv("UNSLOTH_HTTP_RETRY_BACKOFF", "0")
    attempt = _Attempt()
    monkeypatch.setattr(xf, "_run_download_attempt", attempt)
    return attempt


def _call(tmp_path, **kwargs):
    return xf.snapshot_download_with_xet_fallback(
        REPO, token = "hf_tok", cache_dir = str(tmp_path), allow_patterns = PATTERNS, **kwargs
    )


def test_nothing_to_fetch_returns_snapshot_without_child(tmp_path, monkeypatch, _isolated):
    snap = _cache(tmp_path)
    seen = _hub(monkeypatch, snap, [_info(snap, "config.json"), _info(snap, "tokenizer.json")])
    assert _call(tmp_path) == str(snap)
    assert _isolated.calls == []
    dry = [c for c in seen if c["dry_run"]]
    assert len(dry) == 1
    # Same request the child would make.
    assert dry[0]["repo_id"] == REPO and dry[0]["token"] == "hf_tok"
    assert dry[0]["cache_dir"] == str(tmp_path) and dry[0]["allow_patterns"] == PATTERNS
    assert dry[0]["repo_type"] == "model" and dry[0]["revision"] is None
    assert dry[0]["tqdm_class"] is not None  # silent bars: the child prints none either


def test_nested_file_and_commit_revision(tmp_path, monkeypatch, _isolated):
    snap = _cache(tmp_path, files = ("config.json",))
    (snap / "additional_chat_templates").mkdir()
    blob = snap.parent.parent / "blobs" / "tmpl"
    blob.write_text("x")
    (snap / "additional_chat_templates" / "a.jinja").symlink_to(blob)
    _hub(monkeypatch, snap, [_info(snap, "config.json"), _info(snap, "additional_chat_templates/a.jinja")])
    assert _call(tmp_path, revision = SHA) == str(snap)
    assert _isolated.calls == []


def test_file_to_fetch_spawns_child(tmp_path, monkeypatch, _isolated):
    snap = _cache(tmp_path)
    _hub(monkeypatch, snap, [_info(snap, "config.json"),
                             _info(snap, "tokenizer.model", cached = False, will_download = True)])
    assert _call(tmp_path) == "/child/snapshot"
    assert len(_isolated.calls) == 1


def test_dangling_pointer_spawns_child(tmp_path, monkeypatch, _isolated):
    snap = _cache(tmp_path)
    (snap / "tokenizer.json").resolve().unlink()  # blob gone, pointer dangles
    _hub(monkeypatch, snap, [_info(snap, "config.json"), _info(snap, "tokenizer.json")])
    assert _call(tmp_path) == "/child/snapshot"
    assert len(_isolated.calls) == 1


def test_pointer_missing_but_blob_present_spawns_child(tmp_path, monkeypatch, _isolated):
    """hub's dry run counts a present blob as cached; the child would still create the pointer, so defer."""
    snap = _cache(tmp_path)
    (snap / "tokenizer.json").unlink()
    _hub(monkeypatch, snap, [_info(snap, "config.json"), _info(snap, "tokenizer.json")])
    assert _call(tmp_path) == "/child/snapshot"
    assert len(_isolated.calls) == 1


@pytest.mark.parametrize("result", [
    RuntimeError("dry run cannot reach the Hub"),
    [],  # nothing matched
    "not-a-list",
])
def test_dry_run_error_or_unusable_answer_spawns_child(tmp_path, monkeypatch, _isolated, result):
    snap = _cache(tmp_path)
    _hub(monkeypatch, snap, result)
    assert _call(tmp_path) == "/child/snapshot"
    assert len(_isolated.calls) == 1


def test_mixed_commits_spawn_child(tmp_path, monkeypatch, _isolated):
    snap = _cache(tmp_path)
    _hub(monkeypatch, snap, [_info(snap, "config.json"), _info(snap, "tokenizer.json", commit = "b" * 40)])
    assert _call(tmp_path) == "/child/snapshot"
    assert len(_isolated.calls) == 1


def test_hub_without_dry_run_spawns_child(tmp_path, monkeypatch, _isolated):
    snap = _cache(tmp_path)
    seen = _hub(monkeypatch, snap, None, has_dry_run = False)
    assert _call(tmp_path) == "/child/snapshot"
    assert len(_isolated.calls) == 1
    assert all(c["local_files_only"] for c in seen)


def test_offline_skips_dry_run(tmp_path, monkeypatch, _isolated):
    snap = _cache(tmp_path)
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", True)
    seen = _hub(monkeypatch, snap, [_info(snap, "config.json"), _info(snap, "tokenizer.json")])
    assert _call(tmp_path) == "/child/snapshot"
    assert not any(c["dry_run"] for c in seen)


def test_force_download_skips_dry_run(tmp_path, monkeypatch, _isolated):
    snap = _cache(tmp_path)
    seen = _hub(monkeypatch, snap, [_info(snap, "config.json"), _info(snap, "tokenizer.json")])
    assert _call(tmp_path, force_download = True) == "/child/snapshot"
    assert seen == []
    assert _isolated.calls[0]["params"]["force_download"] is True


def test_uncached_revision_skips_dry_run(tmp_path, monkeypatch, _isolated):
    """No local snapshot at all: a download is certain, so no metadata round trip is spent before the child."""
    calls = []

    def _miss(**kwargs):
        calls.append(kwargs)
        raise FileNotFoundError("not cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _miss)
    assert _call(tmp_path) == "/child/snapshot"
    assert [c.get("local_files_only") for c in calls] == [True]


def test_stale_branch_ref_spawns_child(tmp_path, monkeypatch, _isolated):
    snap = _cache(tmp_path, ref = "c" * 40)
    _hub(monkeypatch, snap, [_info(snap, "config.json"), _info(snap, "tokenizer.json")])
    assert _call(tmp_path, revision = "main") == "/child/snapshot"
    assert len(_isolated.calls) == 1


def test_tree_cache_missing_spawns_child_present_skips(tmp_path, monkeypatch, _isolated):
    """On a hub that keeps trees/<commit>.json the child writes it when absent; defer then, skip once it exists."""
    snap = _cache(tmp_path)
    storage = snap.parent.parent
    tree_file = storage / "trees" / f"{SHA}.json"
    fake_tree = types.SimpleNamespace(
        _tree_cache_path = lambda folder, commit: str(Path(folder) / "trees" / f"{commit}.json"),
        read_tree_cache = lambda folder, commit: {"config.json": object()} if tree_file.is_file() else None,
    )
    monkeypatch.setattr(huggingface_hub, "_tree_cache", fake_tree, raising = False)
    _hub(monkeypatch, snap, lambda: [_info(snap, "config.json"), _info(snap, "tokenizer.json")])
    assert _call(tmp_path) == "/child/snapshot"
    assert len(_isolated.calls) == 1
    tree_file.parent.mkdir()
    tree_file.write_text("{}")
    assert _call(tmp_path) == str(snap)
    assert len(_isolated.calls) == 1


def test_hung_dry_run_hands_off_to_child(tmp_path, monkeypatch, _isolated):
    """hub's repo_info has no timeout of its own; a stalled answer must reach the killable child, not hold the load."""
    snap = _cache(tmp_path)
    release = threading.Event()
    monkeypatch.setattr(xf, "DEFAULT_DRY_RUN_TIMEOUT", 0.3)

    def _hang():
        release.wait(30)
        return [_info(snap, "config.json")]

    _hub(monkeypatch, snap, _hang)
    started = time.monotonic()
    try:
        assert _call(tmp_path) == "/child/snapshot"
        assert time.monotonic() - started < 5
        assert len(_isolated.calls) == 1
    finally:
        release.set()


def test_cancel_during_dry_run_raises_without_child(tmp_path, monkeypatch, _isolated):
    snap = _cache(tmp_path)
    release = threading.Event()
    cancel = threading.Event()

    def _hang():
        cancel.set()
        release.wait(30)
        return [_info(snap, "config.json")]

    _hub(monkeypatch, snap, _hang)
    try:
        with pytest.raises(RuntimeError, match = "Cancelled"):
            _call(tmp_path, cancel_event = cancel)
        assert _isolated.calls == []
    finally:
        release.set()


def test_incomplete_result_still_goes_to_child(tmp_path, monkeypatch, _isolated):
    """The child's post-download gate also guards the dry-run answer: a weight request with no readable weight spawns."""
    snap = _cache(tmp_path)
    _hub(monkeypatch, snap, [_info(snap, "config.json"), _info(snap, "tokenizer.json")])
    out = xf.snapshot_download_with_xet_fallback(
        REPO, token = None, cache_dir = str(tmp_path), allow_patterns = ["*.json", "*.safetensors"]
    )
    assert out == "/child/snapshot"
    assert len(_isolated.calls) == 1
