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


def _short_disk_for_backups(monkeypatch, base):
    # Room for the per-shard refresh (the largest shard plus the margin) but not for keeping
    # every previous shard beside the new ones.
    from unsloth_zoo import saving_utils as S

    sizes = [os.path.getsize(os.path.join(base, f)) for f in _shards(base)]
    assert len(sizes) > 2
    free = S._REFRESH_DISK_MARGIN + 2 * max(sizes)
    assert free < S._REFRESH_DISK_MARGIN + sum(sizes)
    disk = S.shutil.disk_usage(base)
    monkeypatch.setattr(S.shutil, "disk_usage", lambda _: type(disk)(disk.total, disk.total - free, free))


@pytest.mark.parametrize("mode", ["low_disk", "short_disk"])
def test_failed_later_hub_fetch_keeps_unfetched_shards(tmp_path, monkeypatch, mode):
    # In low disk mode, or when the disk cannot hold the previous shards beside the new ones,
    # shards are replaced one at a time (no second copy of the checkpoint), so a network
    # failure part way only loses the shards already refreshed, and says so.
    low_disk = mode == "low_disk"
    base, peft_model = _build(tmp_path, "20KB")
    out = str(tmp_path / "out")
    _fake_hub(monkeypatch, base, fail = False)
    H.run_merge(peft_model, base, out, save_dtype = torch.float32, low_disk_space_usage = low_disk)
    before = {f : open(os.path.join(out, f), "rb").read() for f in _shards(out)}

    if mode == "short_disk":
        _short_disk_for_backups(monkeypatch, base)
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


def test_failed_later_hub_fetch_restores_previous_export(tmp_path, monkeypatch):
    # Outside low disk mode, with room for both, a failure part way puts every previous
    # shard back instead of leaving some on the base weights and the rest still merged.
    base, peft_model = _build(tmp_path, "20KB")
    out = str(tmp_path / "out")
    _fake_hub(monkeypatch, base, fail = False)
    H.run_merge(peft_model, base, out, save_dtype = torch.float32)
    before = {f : open(os.path.join(out, f), "rb").read() for f in _shards(out)}
    assert len(before) > 2

    _fake_hub(monkeypatch, base, fail = 1)
    with pytest.raises(OSError, match = "simulated"):
        H.run_merge(peft_model, base, out, save_dtype = torch.float32)
    assert len(_fake_hub.fetched) == 1
    # The refetched shard held merged weights, so keeping it is a real rollback.
    replaced = _fake_hub.fetched[0]
    assert open(os.path.join(base, replaced), "rb").read() != before[replaced]
    assert {f : open(os.path.join(out, f), "rb").read() for f in _shards(out)} == before
    assert _no_staging_left(out)

    _fake_hub(monkeypatch, base, fail = False)
    H.run_merge(peft_model, base, out, save_dtype = torch.float32)
    assert {f : open(os.path.join(out, f), "rb").read() for f in _shards(out)} == before
    assert _no_staging_left(out)


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
    if not low_disk:
        # With room to spare the previous shards are kept for a rollback; without it, never.
        _short_disk_for_backups(monkeypatch, base)

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


@pytest.mark.parametrize("short_disk", [True, False])
def test_failed_local_copy_keeps_uncopied_shards(tmp_path, monkeypatch, short_disk):
    from unsloth_zoo import saving_utils as S

    base, peft_model = _build(tmp_path, "20KB")
    out = str(tmp_path / "out")
    H.run_merge(peft_model, base, out, save_dtype = torch.float32)
    before = {f : open(os.path.join(out, f), "rb").read() for f in _shards(out)}
    assert len(before) > 2
    if short_disk:
        _short_disk_for_backups(monkeypatch, base)

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
        # Restored when the disk could hold the previous shards, else only the copied one is lost.
        assert now == (open(os.path.join(base, f), "rb").read() if short_disk and f in copied else before[f]), f
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


@pytest.mark.parametrize("source", ["local", "hub_low_disk"])
def test_refresh_refused_when_free_space_is_below_the_largest_shard(tmp_path, monkeypatch, source):
    # A 4bit model with a narrow adapter has an in-memory save size below the largest base
    # shard, so prepare_saving passes; the refresh itself must refuse before touching anything.
    # On-disk dtype: a float32 upcast has its own peak check.
    import errno
    from unsloth_zoo import saving_utils as S

    base, peft_model = _build(tmp_path, "20KB")
    out = str(tmp_path / "out")
    low_disk = source == "hub_low_disk"
    if source != "local":
        _fake_hub(monkeypatch, base, fail = False)
    H.run_merge(peft_model, base, out, save_dtype = None, low_disk_space_usage = low_disk)
    before = {f : open(os.path.join(out, f), "rb").read() for f in _shards(out)}
    sizes = sorted(len(v) for v in before.values())
    assert sizes[0] < sizes[-1]
    free = sizes[-1] - 1

    monkeypatch.setattr(S, "get_torch_storage_size_new", lambda x, element_size: 1)
    disk = S.shutil.disk_usage(out)
    monkeypatch.setattr(S.shutil, "disk_usage", lambda _: type(disk)(disk.total, disk.total - free, free))
    def _stage(path):
        if os.path.getsize(path) > free:
            raise OSError(errno.ENOSPC, "simulated full disk")
    if source == "local":
        real_copy = S._copy_file_from_source
        def _copy(src, target_dir, filename):
            if str(src).endswith(".safetensors"):
                _stage(src)
            return real_copy(src, target_dir, filename)
        monkeypatch.setattr(S, "_copy_file_from_source", _copy)
    else:
        _fake_hub(monkeypatch, base, fail = False)
        real_hf = S.hf_hub_download
        def _hf(repo_id, filename, local_dir = None, **k):
            if local_dir is not None and filename.endswith(".safetensors"):
                _stage(os.path.join(base, filename))
            return real_hf(repo_id, filename, local_dir = local_dir, **k)
        monkeypatch.setattr(S, "hf_hub_download", _hf)

    # The message says nothing was changed, so the metadata written before the shards stays too.
    _metadata = [f for f in ("config.json", "generation_config.json", "model.safetensors.index.json")
                 if os.path.exists(os.path.join(out, f))]
    assert "config.json" in _metadata and "model.safetensors.index.json" in _metadata
    for f in _metadata:
        with open(os.path.join(out, f), "ab") as handle:
            handle.write(b"\n")
    metadata_before = {f : open(os.path.join(out, f), "rb").read() for f in _metadata}

    with pytest.raises(RuntimeError, match = "Nothing was changed"):
        H.run_merge(peft_model, base, out, save_dtype = None, low_disk_space_usage = low_disk)
    assert {f : open(os.path.join(out, f), "rb").read() for f in _shards(out)} == before
    assert {f : open(os.path.join(out, f), "rb").read() for f in _metadata} == metadata_before
    assert _no_staging_left(out)


@pytest.mark.parametrize("source", ["local", "hub", "hub_low_disk"])
def test_dangling_shard_link_out_of_the_output_is_replaced(tmp_path, source):
    # A shard entry left as a dangling link out of the output (a deleted cache blob) is a stale
    # entry the refresh replaces with os.replace, which never follows the link: not a refusal.
    monkeypatch = pytest.MonkeyPatch()
    try:
        base, peft_model = _build(tmp_path)
        fresh = str(tmp_path / "fresh")
        H.run_merge(peft_model, base, fresh, save_dtype = torch.float32)
        expected = H.read_safetensors_dir(fresh)
        if source != "local":
            _fake_hub(monkeypatch, base, fail = False)
        out = tmp_path / "out"
        out.mkdir()
        missing = tmp_path / "deleted_outside.safetensors"
        (out / "model.safetensors").symlink_to(missing)
        H.run_merge(peft_model, base, str(out), save_dtype = torch.float32,
                    low_disk_space_usage = source == "hub_low_disk")
        assert not os.path.islink(out / "model.safetensors") and not missing.exists()
        got = H.read_safetensors_dir(str(out))
        assert sorted(got) == sorted(expected)
        assert [k for k in expected if not torch.equal(got[k], expected[k])] == []
        assert _no_staging_left(str(out))
    finally:
        monkeypatch.undo()


def test_live_shard_link_out_survives_a_failed_hub_fetch(tmp_path, monkeypatch):
    # The fetch replaces the entry, so the link is not copied first: a failed fetch leaves it as it was.
    base, peft_model = _build(tmp_path)
    _fake_hub(monkeypatch, base, fail = True)
    out = tmp_path / "out"
    out.mkdir()
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"external-payload")
    (out / "model.safetensors").symlink_to(outside)
    with pytest.raises(OSError, match = "simulated"):
        H.run_merge(peft_model, base, str(out), save_dtype = torch.float32)
    assert os.path.islink(out / "model.safetensors")
    assert outside.read_bytes() == b"external-payload"
    assert _no_staging_left(str(out))


def test_shard_link_to_the_local_base_shard_is_copied_once(tmp_path, monkeypatch):
    from unsloth_zoo import saving_utils as S

    base, peft_model = _build(tmp_path)
    fresh = str(tmp_path / "fresh")
    H.run_merge(peft_model, base, fresh, save_dtype = torch.float32)
    expected = H.read_safetensors_dir(fresh)
    base_before = open(os.path.join(base, "model.safetensors"), "rb").read()

    out = tmp_path / "out"
    out.mkdir()
    (out / "model.safetensors").symlink_to(os.path.join(base, "model.safetensors"))
    copies = []
    real_copy, real_materialize = S._copy_file_from_source, S._materialize_shard_that_resolves_outside
    def _copy(src, target_dir, filename):
        if str(filename).endswith(".safetensors"):
            copies.append("copy_from_source")
        return real_copy(src, target_dir, filename)
    def _materialize(file_path, save_directory):
        was_link = os.path.islink(file_path)
        real_materialize(file_path, save_directory)
        if was_link and not os.path.islink(file_path):
            copies.append("materialize")
    monkeypatch.setattr(S, "_copy_file_from_source", _copy)
    monkeypatch.setattr(S, "_materialize_shard_that_resolves_outside", _materialize)
    H.run_merge(peft_model, base, str(out), save_dtype = torch.float32)

    assert len(copies) == 1, copies
    assert open(os.path.join(base, "model.safetensors"), "rb").read() == base_before
    got = H.read_safetensors_dir(str(out))
    assert [k for k in expected if not torch.equal(got[k], expected[k])] == []


def test_failed_hub_fetch_restores_previous_export_metadata(tmp_path, monkeypatch):
    # The refresh rewrites config.json and the index before the shard fetch; a rollback of the
    # shards alone would pair the previous shards with the new metadata.
    import json

    base, peft_model = _build(tmp_path, "20KB")
    out = str(tmp_path / "out")
    _fake_hub(monkeypatch, base, fail = False)
    H.run_merge(peft_model, base, out, save_dtype = torch.float32)
    for name in ("config.json", "model.safetensors.index.json"):
        path = os.path.join(out, name)
        with open(path) as f:
            data = json.load(f)
        data["previous_export_marker"] = name
        with open(path, "w") as f:
            json.dump(data, f)
    before = {
        f : open(os.path.join(out, f), "rb").read() for f in os.listdir(out)
        if os.path.isfile(os.path.join(out, f))
    }

    _fake_hub(monkeypatch, base, fail = 1)
    with pytest.raises(OSError, match = "simulated"):
        H.run_merge(peft_model, base, out, save_dtype = torch.float32)
    after = {
        f : open(os.path.join(out, f), "rb").read() for f in os.listdir(out)
        if os.path.isfile(os.path.join(out, f))
    }
    assert sorted(after) == sorted(before)
    assert [f for f in before if after[f] != before[f]] == []
    assert _no_staging_left(out)


def test_previous_export_backup_names_cannot_collide_with_shards(tmp_path):
    from unsloth_zoo import saving_utils as S

    out = tmp_path / "out"
    out.mkdir()
    names = ["foo.safetensors", S._PREVIOUS_EXPORT_PREFIX + "foo.safetensors"]
    for name in names:
        (out / name).write_bytes(name.encode())
    kept = S._keep_previous_export_shards(sorted(names), str(out), {n : 1 for n in names})
    assert kept
    for name in names:
        assert (out / name).read_bytes() == name.encode()
    S._restore_previous_export_shards(kept)
    assert sorted(os.listdir(out)) == sorted(names)
    for name in names:
        assert (out / name).read_bytes() == name.encode()
    kept = S._keep_previous_export_shards(sorted(names), str(out), {n : 1 for n in names})
    S._drop_previous_export_shards(kept)
    assert sorted(os.listdir(out)) == sorted(names)
    for name in names:
        assert (out / name).read_bytes() == name.encode()


@pytest.mark.parametrize("refresh", ["keeps_link", "replaces_link", "removes_link"])
def test_rollback_keeps_previous_export_metadata_links(tmp_path, refresh):
    # A top level metadata link the previous export held (tokenizer.model shared with another
    # folder) is not a file the failed refresh added: the rollback must leave it as it was.
    from unsloth_zoo import saving_utils as S

    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "tokenizer.model").write_bytes(b"shared tokenizer")
    out = tmp_path / "out"
    out.mkdir()
    (out / "model.safetensors").write_bytes(b"previous shard")
    (out / "config.json").write_bytes(b"previous config")
    target = os.path.join("..", "shared", "tokenizer.model")
    os.symlink(target, out / "tokenizer.model")

    metadata = S._snapshot_previous_export_metadata(str(out))
    kept = S._keep_previous_export_shards(
        ["model.safetensors"], str(out), {"model.safetensors" : 1}, metadata)
    assert kept
    # The refresh rewrites the metadata, adds a file and starts on the shard.
    (out / "config.json").write_bytes(b"new config")
    (out / "added.json").write_bytes(b"new file")
    if refresh != "keeps_link":
        os.remove(out / "tokenizer.model")
    if refresh == "replaces_link":
        (out / "tokenizer.model").write_bytes(b"new tokenizer")
    os.replace(out / "added.json", out / "model.safetensors.tmp")
    os.replace(out / "model.safetensors.tmp", out / "model.safetensors")

    S._restore_previous_export_shards(kept)
    assert sorted(os.listdir(out)) == ["config.json", "model.safetensors", "tokenizer.model"]
    assert os.path.islink(out / "tokenizer.model")
    assert os.readlink(out / "tokenizer.model") == target
    assert (out / "tokenizer.model").read_bytes() == b"shared tokenizer"
    assert (out / "config.json").read_bytes() == b"previous config"
    assert (out / "model.safetensors").read_bytes() == b"previous shard"
    assert (shared / "tokenizer.model").read_bytes() == b"shared tokenizer"


def test_rollback_never_writes_through_a_link_named_like_a_restore_file(tmp_path):
    # The previous export may hold any name as a link, including the one a rollback would stage
    # a restored file under: the restore must replace entries, never follow a link.
    from unsloth_zoo import saving_utils as S

    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"outside file")
    out = tmp_path / "out"
    out.mkdir()
    (out / "model.safetensors").write_bytes(b"previous shard")
    (out / "config.json").write_bytes(b"previous config")
    os.chmod(out / "config.json", 0o640)
    os.symlink(victim, out / "config.json.unsloth-restore")

    metadata = S._snapshot_previous_export_metadata(str(out))
    kept = S._keep_previous_export_shards(
        ["model.safetensors"], str(out), {"model.safetensors" : 1}, metadata)
    assert kept
    (out / "config.json").write_bytes(b"new config")
    S._restore_previous_export_shards(kept)

    assert victim.read_bytes() == b"outside file"
    assert not os.path.islink(out / "config.json")
    assert (out / "config.json").read_bytes() == b"previous config"
    assert os.stat(out / "config.json").st_mode & 0o777 == 0o640
    assert os.readlink(out / "config.json.unsloth-restore") == str(victim)
    assert sorted(os.listdir(out)) == ["config.json", "config.json.unsloth-restore", "model.safetensors"]


@pytest.mark.parametrize("relative", [False, True])
def test_rollback_removes_shards_the_refresh_added(tmp_path, monkeypatch, relative):
    # A shard name the previous export did not hold, fetched or split before a later step
    # failed, would otherwise survive the rollback beside the restored shards. A relative
    # output path must not make every previous shard look new.
    from unsloth_zoo import saving_utils as S

    out = tmp_path / "out"
    out.mkdir()
    if relative:
        monkeypatch.chdir(tmp_path)
        out = type(out)("out")
    (out / "model.safetensors").write_bytes(b"previous shard")
    (out / "kept-unrelated.safetensors").write_bytes(b"not refreshed")
    (out / "config.json").write_bytes(b"previous config")
    before = {f : (out / f).read_bytes() for f in os.listdir(out)}

    metadata = S._snapshot_previous_export_metadata(str(out))
    kept = S._keep_previous_export_shards(
        ["model.safetensors"], str(out), {"model.safetensors" : 1}, metadata)
    assert kept
    (out / "model-00002-of-00002.safetensors").write_bytes(b"new shard")
    (out / "temp_split_abcd_000.safetensors").write_bytes(b"split piece")
    (out / "sub").mkdir()
    (out / "sub" / "nested.safetensors").write_bytes(b"new nested shard")
    S._restore_previous_export_shards(kept)

    assert not (out / "sub" / "nested.safetensors").exists()
    os.rmdir(out / "sub")
    assert {f : (out / f).read_bytes() for f in os.listdir(out)} == before


def test_failed_local_refresh_restores_the_previous_tokenizer_model(tmp_path, monkeypatch):
    # For a local base, tokenizer.model was copied over the previous one before the snapshot, so a
    # rollback paired the previous shards with the new tokenizer.
    from unsloth_zoo import saving_utils as S

    base, peft_model = _build(tmp_path)
    out = str(tmp_path / "out")
    H.run_merge(peft_model, base, out, save_dtype = torch.float32)
    with open(os.path.join(out, "tokenizer.model"), "wb") as f:
        f.write(b"previous tokenizer")
    with open(os.path.join(base, "tokenizer.model"), "wb") as f:
        f.write(b"new tokenizer")
    before = {
        f : open(os.path.join(out, f), "rb").read() for f in os.listdir(out)
        if os.path.isfile(os.path.join(out, f))
    }

    real_copy = S._copy_file_from_source
    def _copy(src, target_dir, filename):
        if str(filename).endswith(".safetensors"):
            raise IOError("simulated copy failure")
        return real_copy(src, target_dir, filename)
    monkeypatch.setattr(S, "_copy_file_from_source", _copy)
    with pytest.raises(OSError, match = "simulated"):
        H.run_merge(peft_model, base, out, save_dtype = torch.float32)

    after = {
        f : open(os.path.join(out, f), "rb").read() for f in os.listdir(out)
        if os.path.isfile(os.path.join(out, f))
    }
    assert after["tokenizer.model"] == b"previous tokenizer"
    assert sorted(after) == sorted(before)
    assert [f for f in before if after[f] != before[f]] == []
