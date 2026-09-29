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
    # mlx-community/gemma-4-e2b-it-OptiQ-4bit: 1411 vision tensors live in a subfolder shard
    # mlx-lm's `model*.safetensors` default never downloads; mlx-vlm then drops them silently.
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
    # A branch revision re-resolved after a push would return a snapshot holding only the shard.
    from unsloth_zoo.mlx.loader import _download_missing_index_shards

    sha = "a" * 40
    snap = tmp_path / "snapshots" / sha
    snap.mkdir(parents=True)
    (snap / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"v": "optiq/optiq_vision.safetensors"}}))
    calls, download = _recorder(snap)
    _download_missing_index_shards("org/repo", str(snap), "main", download)
    assert calls == [("org/repo", sha, ["optiq/optiq_vision.safetensors"])]
