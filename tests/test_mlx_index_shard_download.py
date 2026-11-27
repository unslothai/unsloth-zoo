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

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_mlx_shim():
    from mlx_simulation import simulate_mlx_on_torch

    simulate_mlx_on_torch()


def _snapshot(tmp_path, weight_map, present):
    root = tmp_path / "snap"
    root.mkdir()
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    for name in present:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(b"")
    return root


def _recorder(root):
    calls = []

    def download(repo, revision=None, allow_patterns=None):
        calls.append((repo, revision, allow_patterns))
        for name in allow_patterns:
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_bytes(b"")
        return root

    return calls, download


def test_subfolder_shard_named_by_the_index_is_fetched(tmp_path):
    # gemma-4-e2b-it-OptiQ-4bit keeps its vision tower in optiq/optiq_vision.safetensors.
    from unsloth_zoo.mlx.loader import _download_missing_index_shards

    root = _snapshot(tmp_path, {"language_model.w": "model.safetensors",
                                "vision_tower.a": "optiq/optiq_vision.safetensors",
                                "vision_tower.b": "optiq/optiq_vision.safetensors"},
                     present=["model.safetensors"])
    calls, download = _recorder(root)
    out = _download_missing_index_shards("org/repo", str(root), "abc123", download)
    assert calls == [("org/repo", "abc123", ["optiq/optiq_vision.safetensors"])]
    assert out == str(root) and (root / "optiq/optiq_vision.safetensors").exists()


def test_complete_snapshots_local_dirs_and_indexless_repos_download_nothing(tmp_path):
    from unsloth_zoo.mlx.loader import _download_missing_index_shards

    complete = _snapshot(tmp_path, {"w": "model-00001-of-00001.safetensors"},
                         present=["model-00001-of-00001.safetensors"])
    calls, download = _recorder(complete)
    assert _download_missing_index_shards("org/repo", str(complete), None, download) == str(complete)

    local = tmp_path / "local"
    local.mkdir()
    (local / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "gone.safetensors"}}))
    assert _download_missing_index_shards(str(local), str(local), None, download) == str(local)

    bare = tmp_path / "bare"
    bare.mkdir()
    assert _download_missing_index_shards("org/other", str(bare), None, download) == str(bare)
    assert calls == []


def test_follow_up_download_pins_the_fetched_snapshot_commit(tmp_path):
    from unsloth_zoo.mlx.loader import _download_missing_index_shards

    sha = "a" * 40
    snap = tmp_path / "snapshots" / sha
    snap.mkdir(parents=True)
    (snap / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"v": "optiq/optiq_vision.safetensors"}}))
    calls, download = _recorder(snap)
    _download_missing_index_shards("org/repo", str(snap), "main", download)
    assert calls == [("org/repo", sha, ["optiq/optiq_vision.safetensors"])]


def _offline_hub(monkeypatch, tmp_path):
    # No trees/ listing: huggingface_hub 1.x would then list the remote tree for a pinned commit.
    from huggingface_hub import constants
    import mlx_lm.utils as lm_utils

    commit = "c" * 40
    snap = tmp_path / "hub" / "models--org--repo" / "snapshots" / commit
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({"model_type": "llama"}))
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    loaded = []

    def load_model(path, **_):
        loaded.append(path)
        return object(), {}

    monkeypatch.setattr(lm_utils, "load_model", load_model)
    monkeypatch.setattr(lm_utils, "load_tokenizer", lambda *a, **k: "tok")
    return commit, snap, loaded


def _skip_unless_hub_refuses_pinned_lookup(commit):
    # Older Hubs resolve a pinned commit from the cache themselves; only newer ones need the fallback.
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import LocalEntryNotFoundError, OfflineModeIsEnabled

    try:
        snapshot_download("org/repo", revision=commit)
    except OfflineModeIsEnabled:
        return
    except LocalEntryNotFoundError:
        pass
    pytest.skip("this huggingface_hub resolves pinned commits offline without listing the tree")


def test_offline_load_pinned_to_a_commit_resolves_the_cached_snapshot(monkeypatch, tmp_path):
    from unsloth_zoo.mlx.loader import _load_mlx_lm_with_strict_fallback

    commit, snap, loaded = _offline_hub(monkeypatch, tmp_path)
    _load_mlx_lm_with_strict_fallback("org/repo", "llama", None, {"revision": commit})
    assert [str(path) for path in loaded] == [str(snap)]


def test_offline_load_of_an_uncached_commit_names_the_offline_setting(monkeypatch, tmp_path):
    # text_only=False routes to mlx-vlm, which would otherwise retry the repo id and raise OfflineModeIsEnabled.
    from huggingface_hub.utils import LocalEntryNotFoundError
    from unsloth_zoo.mlx.loader import FastMLXModel

    _offline_hub(monkeypatch, tmp_path)
    _skip_unless_hub_refuses_pinned_lookup("d" * 40)
    with pytest.raises(LocalEntryNotFoundError, match="HF_HUB_OFFLINE"):
        FastMLXModel.from_pretrained("org/repo", revision="d" * 40, text_only=False)


@pytest.mark.parametrize("offline", [False, True])
def test_a_working_mlx_lm_download_is_returned_unchanged(monkeypatch, tmp_path, offline):
    # Offline too: with a cached tree listing the Hub's own path can still rebuild snapshot links from blobs.
    import mlx_lm.utils as lm_utils
    from huggingface_hub import constants
    from unsloth_zoo.mlx.loader import _mlx_download

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", offline)
    calls, download = _recorder(tmp_path)
    monkeypatch.setattr(lm_utils, "_download", download)
    assert _mlx_download("org/repo", revision="c" * 40, allow_patterns=["a.json"]) == tmp_path
    assert calls == [("org/repo", "c" * 40, ["a.json"])]


def test_offline_named_shard_missing_from_the_cached_snapshot_is_refused(monkeypatch, tmp_path):
    from huggingface_hub.utils import LocalEntryNotFoundError
    from unsloth_zoo.mlx.loader import _mlx_download

    commit, snap, _ = _offline_hub(monkeypatch, tmp_path)
    _skip_unless_hub_refuses_pinned_lookup(commit)
    (snap / "model-1.safetensors").write_bytes(b"")
    # Optional named metadata (tokenizer.model) may be absent; a named shard may not.
    assert Path(_mlx_download("org/repo", commit, ["model-1.safetensors", "tokenizer.model", "*.json"])) == snap
    with pytest.raises(LocalEntryNotFoundError, match="cached without model-2.safetensors"):
        _mlx_download("org/repo", commit, ["model-1.safetensors", "model-2.safetensors"])


def test_offline_entry_shard_miss_still_reaches_the_backend_load(monkeypatch, tmp_path):
    # The entry swallows a shard miss as before: a backend may not need it (mlx-lm drops a stitched mtp shard).
    import mlx_lm.utils as lm_utils
    from unsloth_zoo.mlx.loader import FastMLXModel

    commit, snap, _ = _offline_hub(monkeypatch, tmp_path)
    (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"mtp.w": "model-mtp.safetensors"}}))
    (tmp_path / "hub" / "models--org--repo" / "refs").mkdir()
    (tmp_path / "hub" / "models--org--repo" / "refs" / "main").write_text(commit)

    class Reached(Exception):
        pass

    def load_model(path, **_):
        raise Reached(str(path))

    monkeypatch.setattr(lm_utils, "load_model", load_model)
    with pytest.raises(Reached):
        FastMLXModel.from_pretrained("org/repo")


def test_offline_distributed_vlm_resolves_the_cached_snapshot(monkeypatch, tmp_path):
    import sys
    import types

    from huggingface_hub import snapshot_download
    import unsloth_zoo.mlx.loader as loader

    commit, snap, _ = _offline_hub(monkeypatch, tmp_path)
    _skip_unless_hub_refuses_pinned_lookup(commit)
    seen = []

    def sharded_load(path, **_):
        seen.append(path)
        return object(), object()

    vlm_utils = types.ModuleType("mlx_vlm.utils")
    # mlx-vlm's get_model_path is snapshot_download with its own patterns.
    vlm_utils.get_model_path = lambda repo, revision=None: Path(snapshot_download(repo, revision=revision))
    vlm_utils.sharded_load = sharded_load
    monkeypatch.setitem(sys.modules, "mlx_vlm", types.ModuleType("mlx_vlm"))
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", vlm_utils)
    monkeypatch.setattr(loader, "_bind_mlx_vlm_processor_loader", lambda sharded, **_: sharded)
    group = types.SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    loader._load_mlx_vlm_distributed("org/repo", "llama", tensor_group=group, revision=commit)
    assert [str(path) for path in seen] == [str(snap)]
