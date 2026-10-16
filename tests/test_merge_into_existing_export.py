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

"""A merged_16bit export into a directory that already holds one must merge onto the base, not onto it."""

from __future__ import annotations

import os

import pytest
import torch

import _merge_e2e_helpers as H

TARGETS = ["q_proj", "v_proj", "down_proj"]


def _build(tmp_path, shard_size = None):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    H.set_offline_cpu_env()
    if not H.family_available("llama"):
        pytest.skip("llama unavailable")
    base = str(tmp_path / "base")
    torch.manual_seed(H.SEED)
    kwargs = {"max_shard_size": shard_size} if shard_size else {}
    AutoModelForCausalLM.from_config(H.make_spec("llama").config).to(torch.float32).save_pretrained(
        base, safe_serialization = True, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(base, dtype = torch.float32)
    model.config._name_or_path = base
    peft_model = get_peft_model(model, LoraConfig(
        r = 4, lora_alpha = 8, lora_dropout = 0.0, bias = "none", target_modules = TARGETS))
    H.seed_lora(peft_model)
    return base, peft_model


@pytest.mark.parametrize("shard_size", [None, "200KB"])
@pytest.mark.parametrize("low_disk", [False, True])
def test_second_merge_into_same_directory_is_identical(tmp_path, shard_size, low_disk):
    base, peft_model = _build(tmp_path, shard_size)
    fresh = str(tmp_path / "fresh")
    H.run_merge(peft_model, base, fresh, save_dtype = torch.float32, low_disk_space_usage = low_disk)
    expected = H.read_safetensors_dir(fresh)

    out = str(tmp_path / "out")
    H.run_merge(peft_model, base, out, save_dtype = torch.float32, low_disk_space_usage = low_disk)
    H.run_merge(peft_model, base, out, save_dtype = torch.float32, low_disk_space_usage = low_disk)
    got = H.read_safetensors_dir(out)

    assert sorted(got) == sorted(expected)
    differ = [k for k in expected if not torch.equal(got[k], expected[k])]
    assert differ == []
    # The fresh export really carries the delta, so equality above is not base == base.
    base_tensors = H.read_safetensors_dir(base)
    assert any(not torch.equal(expected[k], base_tensors[k]) for k in expected if "q_proj" in k)


def test_in_place_export_still_merges_once(tmp_path):
    base, peft_model = _build(tmp_path)
    fresh = str(tmp_path / "fresh")
    H.run_merge(peft_model, base, fresh, save_dtype = torch.float32)
    H.run_merge(peft_model, base, base, save_dtype = torch.float32)
    got, expected = H.read_safetensors_dir(base), H.read_safetensors_dir(fresh)
    assert [k for k in expected if not torch.equal(got[k], expected[k])] == []


def test_previous_export_shards_skips_in_place_and_linked_parents(tmp_path):
    from unsloth_zoo.saving_utils import _previous_export_shards

    out = tmp_path / "out"
    out.mkdir()
    target = tmp_path / "outside.safetensors"
    target.write_bytes(b"keep")
    (out / "model.safetensors").symlink_to(target)
    (out / "model-00001-of-00002.safetensors").write_bytes(b"stale")
    linked = tmp_path / "linked"
    linked.mkdir()
    (out / "weights").symlink_to(linked, target_is_directory = True)
    (linked / "model.safetensors").write_bytes(b"keep")
    names = ["model.safetensors", "model-00001-of-00002.safetensors",
             os.path.join("weights", "model.safetensors"), "missing.safetensors"]

    assert _previous_export_shards(names, str(out)) == names[:2]
    # In place: the output shard is the source itself.
    assert _previous_export_shards(names[1:2], str(out), str(out)) == []
    assert target.read_bytes() == b"keep" and (linked / "model.safetensors").read_bytes() == b"keep"


def _fake_hub(monkeypatch, base, fail):
    import shutil
    from unsloth_zoo import saving_utils as S

    shards = sorted(f for f in os.listdir(base) if f.endswith(".safetensors"))
    revisions = []
    _fake_hub.revisions = revisions

    class _FS:
        def __init__(self, *a, **k):
            pass
        def ls(self, name, detail = True, **k):
            revisions.append(k.get("revision"))
            return [{"name": f"{name}/{f}", "size": os.path.getsize(os.path.join(base, f))} for f in shards]

    fetched = []
    _fake_hub.fetched = fetched

    def _fetch(local_dir, names):
        # fail: True fails the first shard fetch, an int N fails once N shards landed.
        for f in names:
            if f in shards:
                if fail is True or (fail is not False and len(fetched) >= fail):
                    raise OSError("simulated network failure")
                fetched.append(f)
            if os.path.exists(os.path.join(base, f)):
                shutil.copyfile(os.path.join(base, f), os.path.join(local_dir, f))

    def _snapshot(repo_id, local_dir = None, allow_patterns = (), **k):
        revisions.append(k.get("revision"))
        _fetch(local_dir, list(allow_patterns))
        return local_dir

    def _hf_hub(repo_id, filename, local_dir = None, **k):
        revisions.append(k.get("revision"))
        if local_dir is None:
            raise OSError("not cached")
        _fetch(local_dir, [filename])
        return os.path.join(local_dir, filename)

    monkeypatch.setattr(S, "determine_base_model_source", lambda *a, **k: ("fake/repo", False, "hub", False, None))
    monkeypatch.setattr(S, "HfFileSystem", _FS)
    monkeypatch.setattr(S, "snapshot_download", _snapshot)
    monkeypatch.setattr(S, "hf_hub_download", _hf_hub)
    monkeypatch.setattr(S, "_try_copy_all_from_cache", lambda *a, **k: False)


def _shards(directory):
    return sorted(f for f in os.listdir(directory) if f.endswith(".safetensors"))


def _no_staging_left(directory):
    return not [f for f in os.listdir(directory) if f.startswith(".unsloth-")]


@pytest.mark.parametrize("low_disk", [False, True])
def test_failed_first_hub_fetch_keeps_previous_export(tmp_path, monkeypatch, low_disk):
    base, peft_model = _build(tmp_path, "20KB")
    out = str(tmp_path / "out")
    _fake_hub(monkeypatch, base, fail = False)
    H.run_merge(peft_model, base, out, save_dtype = torch.float32, low_disk_space_usage = low_disk)
    expected = H.read_safetensors_dir(out)
    assert len(_shards(out)) > 1

    _fake_hub(monkeypatch, base, fail = True)
    with pytest.raises(OSError, match = "simulated"):
        H.run_merge(peft_model, base, out, save_dtype = torch.float32, low_disk_space_usage = low_disk)
    got = H.read_safetensors_dir(out)
    assert sorted(got) == sorted(expected)
    assert [k for k in expected if not torch.equal(got[k], expected[k])] == []
    assert _no_staging_left(out)

    # A later successful export still merges onto the base, not onto the kept shards.
    _fake_hub(monkeypatch, base, fail = False)
    H.run_merge(peft_model, base, out, save_dtype = torch.float32, low_disk_space_usage = low_disk)
    again = H.read_safetensors_dir(out)
    assert [k for k in expected if not torch.equal(again[k], expected[k])] == []


@pytest.mark.parametrize("low_disk", [False, True])
def test_failed_later_hub_fetch_keeps_unfetched_shards(tmp_path, monkeypatch, low_disk):
    # Shards are replaced one at a time (no second copy of the checkpoint), so a network
    # failure part way only loses the shards already refreshed, and says so.
    base, peft_model = _build(tmp_path, "20KB")
    out = str(tmp_path / "out")
    _fake_hub(monkeypatch, base, fail = False)
    H.run_merge(peft_model, base, out, save_dtype = torch.float32, low_disk_space_usage = low_disk)
    before = {f : open(os.path.join(out, f), "rb").read() for f in _shards(out)}

    _fake_hub(monkeypatch, base, fail = 1)
    with pytest.raises(OSError, match = "simulated"):
        H.run_merge(peft_model, base, out, save_dtype = torch.float32, low_disk_space_usage = low_disk)
    replaced = _fake_hub.fetched
    assert len(replaced) == 1 and _shards(out) == sorted(before)
    for f in before:
        now = open(os.path.join(out, f), "rb").read()
        assert now == (open(os.path.join(base, f), "rb").read() if f in replaced else before[f]), f
    assert _no_staging_left(out)

    _fake_hub(monkeypatch, base, fail = False)
    H.run_merge(peft_model, base, out, save_dtype = torch.float32, low_disk_space_usage = low_disk)
    assert {f : open(os.path.join(out, f), "rb").read() for f in _shards(out)} == before


def _dir_bytes(directory):
    total = 0
    for root, _, files in os.walk(directory):
        for f in files:
            total += os.lstat(os.path.join(root, f)).st_size
    return total


@pytest.mark.parametrize("source", ["local", "hub", "hub_low_disk"])
def test_reexport_keeps_no_second_copy_of_the_checkpoint(tmp_path, monkeypatch, source):
    from unsloth_zoo import saving_utils as S

    base, peft_model = _build(tmp_path, "20KB")
    out = str(tmp_path / "out")
    low_disk = source == "hub_low_disk"
    if source != "local":
        _fake_hub(monkeypatch, base, fail = False)
    H.run_merge(peft_model, base, out, save_dtype = torch.float32, low_disk_space_usage = low_disk)
    expected = H.read_safetensors_dir(out)
    size_before = _dir_bytes(out)
    largest = max(os.path.getsize(os.path.join(base, f)) for f in _shards(base))

    peak = [0]
    if source == "local":
        real_copy = S._copy_file_from_source
        def _copy(src, target_dir, filename):
            real_copy(src, target_dir, filename)
            peak[0] = max(peak[0], _dir_bytes(out))
        monkeypatch.setattr(S, "_copy_file_from_source", _copy)
    else:
        real_hf = S.hf_hub_download
        def _hf(*a, **k):
            path = real_hf(*a, **k)
            peak[0] = max(peak[0], _dir_bytes(out))
            return path
        monkeypatch.setattr(S, "hf_hub_download", _hf)
    H.run_merge(peft_model, base, out, save_dtype = torch.float32, low_disk_space_usage = low_disk)
    assert peak[0] > 0
    assert peak[0] <= size_before + largest, (peak[0], size_before, largest)
    got = H.read_safetensors_dir(out)
    assert [k for k in expected if not torch.equal(got[k], expected[k])] == []
    assert _no_staging_left(out)


@pytest.mark.parametrize("source", ["local", "hub_low_disk"])
def test_reexport_refused_before_touching_when_disk_is_short(tmp_path, monkeypatch, source):
    # Per-shard refresh needs at most one shard more than the export, which prepare_saving's
    # free space check already covers, so a short disk is refused before any shard changes.
    from unsloth_zoo import saving_utils as S

    base, peft_model = _build(tmp_path, "20KB")
    out = str(tmp_path / "out")
    low_disk = source == "hub_low_disk"
    if source != "local":
        _fake_hub(monkeypatch, base, fail = False)
    H.run_merge(peft_model, base, out, save_dtype = torch.float32, low_disk_space_usage = low_disk)
    before = {f : open(os.path.join(out, f), "rb").read() for f in _shards(out)}

    disk = S.shutil.disk_usage(out)
    monkeypatch.setattr(S.shutil, "disk_usage", lambda _: type(disk)(disk.total, disk.total - 4096, 4096))
    if source != "local":
        _fake_hub(monkeypatch, base, fail = False)
    with pytest.raises(RuntimeError, match = "no disk space"):
        H.run_merge(peft_model, base, out, save_dtype = torch.float32, low_disk_space_usage = low_disk)
    assert _fake_hub.fetched == [] if source != "local" else True
    assert {f : open(os.path.join(out, f), "rb").read() for f in _shards(out)} == before
    assert _no_staging_left(out)


def test_reexport_with_longest_valid_shard_name(tmp_path):
    import json
    from pathlib import Path

    base, peft_model = _build(tmp_path, "20KB")
    index = json.loads((Path(base) / "model.safetensors.index.json").read_text())
    old_name = sorted(set(index["weight_map"].values()))[-1]
    long_name = "x" * (240 - len(".safetensors")) + ".safetensors"
    (Path(base) / old_name).rename(Path(base) / long_name)
    index["weight_map"] = {k : long_name if v == old_name else v for k, v in index["weight_map"].items()}
    (Path(base) / "model.safetensors.index.json").write_text(json.dumps(index))

    fresh = str(tmp_path / "fresh")
    H.run_merge(peft_model, base, fresh, save_dtype = torch.float32)
    expected = H.read_safetensors_dir(fresh)
    out = str(tmp_path / "out")
    H.run_merge(peft_model, base, out, save_dtype = torch.float32)
    H.run_merge(peft_model, base, out, save_dtype = torch.float32)
    got = H.read_safetensors_dir(out)
    assert sorted(got) == sorted(expected)
    assert [k for k in expected if not torch.equal(got[k], expected[k])] == []
    assert long_name in _shards(out) and _no_staging_left(out)


def test_failed_local_copy_keeps_uncopied_shards(tmp_path, monkeypatch):
    from unsloth_zoo import saving_utils as S

    base, peft_model = _build(tmp_path, "20KB")
    out = str(tmp_path / "out")
    H.run_merge(peft_model, base, out, save_dtype = torch.float32)
    before = {f : open(os.path.join(out, f), "rb").read() for f in _shards(out)}
    assert len(before) > 2

    real_copy, copied = S._copy_file_from_source, []
    def _copy_then_fail(src, target_dir, filename):
        if str(src).endswith(".safetensors"):
            if len(copied) >= 1:
                raise OSError("simulated I/O error")
            copied.append(filename)
        return real_copy(src, target_dir, filename)
    monkeypatch.setattr(S, "_copy_file_from_source", _copy_then_fail)
    with pytest.raises(OSError, match = "simulated"):
        H.run_merge(peft_model, base, out, save_dtype = torch.float32)
    assert len(copied) == 1 and _shards(out) == sorted(before)
    for f in before:
        now = open(os.path.join(out, f), "rb").read()
        assert now == (open(os.path.join(base, f), "rb").read() if f in copied else before[f]), f
    assert _no_staging_left(out)


class _ModelInfo:
    sha = "0123456789abcdef0123456789abcdef01234567"


@pytest.mark.parametrize("low_disk", [False, True])
def test_hub_fetches_pin_one_commit(tmp_path, monkeypatch, low_disk):
    # Per-shard downloads must not mix commits if the repo is pushed to mid-export.
    import huggingface_hub

    base, peft_model = _build(tmp_path, "20KB")
    out = str(tmp_path / "out")
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", lambda self, repo_id, **k: _ModelInfo())
    _fake_hub(monkeypatch, base, fail = False)
    H.run_merge(peft_model, base, out, save_dtype = torch.float32, low_disk_space_usage = low_disk)
    _fake_hub(monkeypatch, base, fail = False)
    H.run_merge(peft_model, base, out, save_dtype = torch.float32, low_disk_space_usage = low_disk)
    assert _fake_hub.revisions and set(_fake_hub.revisions) == {_ModelInfo.sha}


def test_hub_fetches_unpinned_when_commit_unknown(tmp_path, monkeypatch):
    import huggingface_hub

    def _offline(self, repo_id, **k):
        raise OSError("offline")
    base, peft_model = _build(tmp_path, "20KB")
    out = str(tmp_path / "out")
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", _offline)
    _fake_hub(monkeypatch, base, fail = False)
    H.run_merge(peft_model, base, out, save_dtype = torch.float32, low_disk_space_usage = True)
    assert _fake_hub.revisions and set(_fake_hub.revisions) == {None}


def test_partial_cache_copy_is_reported_when_a_later_fetch_fails(tmp_path, monkeypatch, capsys):
    from unsloth_zoo import saving_utils as S

    base, peft_model = _build(tmp_path, "20KB")
    out = str(tmp_path / "out")
    _fake_hub(monkeypatch, base, fail = False)
    H.run_merge(peft_model, base, out, save_dtype = torch.float32, low_disk_space_usage = True)
    first = _shards(out)[0]

    def _partial_cache_copy(repo_id, filenames_to_check, target_dir_str, hf_cache_dir, token, **k):
        # One shard copies from the cache, then the copy gives up.
        if first not in filenames_to_check:
            return False
        S._copy_file_from_source(os.path.join(base, first), target_dir_str, first)
        if k.get("copied") is not None:
            k["copied"].append(first)
        return False
    _fake_hub(monkeypatch, base, fail = True)
    monkeypatch.setattr(S, "_get_hf_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(S, "_try_copy_all_from_cache", _partial_cache_copy)
    capsys.readouterr()
    with pytest.raises(OSError, match = "simulated"):
        H.run_merge(peft_model, base, out, save_dtype = torch.float32, low_disk_space_usage = True)
    printed = capsys.readouterr().out
    assert "is incomplete" in printed and first in printed
    assert open(os.path.join(out, first), "rb").read() == open(os.path.join(base, first), "rb").read()


def test_hub_listing_pins_the_same_commit_as_the_fetches(tmp_path, monkeypatch):
    # Names and sizes come from the listing; an unpinned one can describe an older commit.
    import huggingface_hub
    from unsloth_zoo import saving_utils as S

    base, peft_model = _build(tmp_path, "20KB")
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", lambda self, repo_id, **k: _ModelInfo())
    _fake_hub(monkeypatch, base, fail = False)
    listed = []
    real_fs = S.HfFileSystem
    class _RecordingFS(real_fs):
        def ls(self, name, detail = True, **k):
            listed.append(k.get("revision"))
            return super().ls(name, detail = detail, **k)
    monkeypatch.setattr(S, "HfFileSystem", _RecordingFS)
    H.run_merge(peft_model, base, str(tmp_path / "out"), save_dtype = torch.float32, low_disk_space_usage = True)
    assert listed and set(listed) == {_ModelInfo.sha}
