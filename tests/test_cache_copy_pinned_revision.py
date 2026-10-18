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

"""An export pinned to the Hub's current commit still copies shards from a cached older `main`
snapshot when the file is unchanged at that commit (a README-only push), instead of downloading."""

import os
import types

import pytest

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

try:
    from unsloth_zoo import saving_utils as S
except ImportError as _e:
    pytest.skip(f"unsloth_zoo unavailable: {_e}", allow_module_level = True)

SHA1, SHA2, ETAG = "1" * 40, "2" * 40, "a" * 64


def _cache(tmp_path):
    repo = tmp_path / "cache" / "models--org--m"
    (repo / "blobs").mkdir(parents = True)
    (repo / "refs").mkdir()
    (repo / "snapshots" / SHA1).mkdir(parents = True)
    (repo / "blobs" / ETAG).write_bytes(b"weights")
    (repo / "refs" / "main").write_text(SHA1)
    os.symlink(f"../../blobs/{ETAG}", repo / "snapshots" / SHA1 / "model.safetensors")
    return str(tmp_path / "cache")


def _hub_reports(monkeypatch, sha256, xet_hash = None):
    import huggingface_hub
    info = types.SimpleNamespace(
        path = "model.safetensors", lfs = types.SimpleNamespace(sha256 = sha256), blob_id = "b" * 40, xet_hash = xet_hash,
    )
    monkeypatch.setattr(huggingface_hub.HfApi, "get_paths_info", lambda self, repo, paths, revision = None, **k: [info])


def test_pinned_commit_reuses_unchanged_cached_main_shard(tmp_path, monkeypatch):
    cache = _cache(tmp_path)
    _hub_reports(monkeypatch, ETAG)
    out = tmp_path / "out"
    assert S._try_copy_all_from_cache("org/m", ["model.safetensors"], str(out), cache, None, revision = SHA2)
    assert (out / "model.safetensors").read_bytes() == b"weights"


def test_pinned_commit_matches_a_xet_named_blob(tmp_path, monkeypatch):
    # Caches that name blobs by xet hash instead of the LFS sha256.
    cache = _cache(tmp_path)
    _hub_reports(monkeypatch, "d" * 64, xet_hash = ETAG)
    assert S._try_copy_all_from_cache("org/m", ["model.safetensors"], str(tmp_path / "out"), cache, None, revision = SHA2)


def test_pinned_commit_with_changed_shard_still_downloads(tmp_path, monkeypatch):
    cache = _cache(tmp_path)
    _hub_reports(monkeypatch, "c" * 64)
    out = tmp_path / "out"
    assert not S._try_copy_all_from_cache("org/m", ["model.safetensors"], str(out), cache, None, revision = SHA2)
    assert not (out / "model.safetensors").exists()


def test_pinned_commit_offline_still_misses(tmp_path, monkeypatch):
    import huggingface_hub
    cache = _cache(tmp_path)

    def offline(self, *a, **k):
        raise OSError("offline")

    monkeypatch.setattr(huggingface_hub.HfApi, "get_paths_info", offline)
    assert not S._try_copy_all_from_cache("org/m", ["model.safetensors"], str(tmp_path / "out"), cache, None, revision = SHA2)
