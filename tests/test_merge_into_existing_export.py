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


def test_set_aside_moves_links_not_targets_and_restores(tmp_path):
    from unsloth_zoo.saving_utils import (
        _set_aside_stale_output_shards, _restore_stale_output_shards, _discard_stale_output_shard,
    )

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
    paths = [str(out / n) for n in ("model.safetensors", "model-00001-of-00002.safetensors",
                                    os.path.join("weights", "model.safetensors"))]

    aside = _set_aside_stale_output_shards(paths, str(out))
    assert not os.path.lexists(out / "model.safetensors")
    assert not os.path.exists(out / "model-00001-of-00002.safetensors")
    assert target.read_bytes() == b"keep"
    assert (linked / "model.safetensors").read_bytes() == b"keep"
    assert paths[2] not in aside

    (out / "model-00001-of-00002.safetensors").write_bytes(b"partial")
    _restore_stale_output_shards(aside)
    assert os.path.islink(out / "model.safetensors")
    assert (out / "model-00001-of-00002.safetensors").read_bytes() == b"stale"
    assert sorted(os.listdir(out)) == ["model-00001-of-00002.safetensors", "model.safetensors", "weights"]

    aside = _set_aside_stale_output_shards(paths[1:2], str(out))
    (out / "model-00001-of-00002.safetensors").write_bytes(b"base")
    _discard_stale_output_shard(aside, paths[1])
    assert (out / "model-00001-of-00002.safetensors").read_bytes() == b"base"
    assert aside == {} and sorted(os.listdir(out)) == ["model-00001-of-00002.safetensors", "model.safetensors", "weights"]


def _fake_hub(monkeypatch, base, fail):
    import shutil
    from unsloth_zoo import saving_utils as S

    shards = sorted(f for f in os.listdir(base) if f.endswith(".safetensors"))

    class _FS:
        def __init__(self, *a, **k):
            pass
        def ls(self, name, detail = True):
            return [{"name": f"{name}/{f}", "size": os.path.getsize(os.path.join(base, f))} for f in shards]

    def _fetch(local_dir, names):
        if fail and any(f in shards for f in names):
            raise OSError("simulated network failure")
        for f in names:
            if os.path.exists(os.path.join(base, f)):
                shutil.copyfile(os.path.join(base, f), os.path.join(local_dir, f))

    def _snapshot(repo_id, local_dir = None, allow_patterns = (), **k):
        _fetch(local_dir, list(allow_patterns))
        return local_dir

    def _hf_hub(repo_id, filename, local_dir = None, **k):
        if local_dir is None:
            raise OSError("not cached")
        _fetch(local_dir, [filename])
        return os.path.join(local_dir, filename)

    monkeypatch.setattr(S, "determine_base_model_source", lambda *a, **k: ("fake/repo", False, "hub", False, None))
    monkeypatch.setattr(S, "HfFileSystem", _FS)
    monkeypatch.setattr(S, "snapshot_download", _snapshot)
    monkeypatch.setattr(S, "hf_hub_download", _hf_hub)
    monkeypatch.setattr(S, "_try_copy_all_from_cache", lambda *a, **k: False)


@pytest.mark.parametrize("low_disk", [False, True])
def test_failed_hub_fetch_keeps_previous_export(tmp_path, monkeypatch, low_disk):
    base, peft_model = _build(tmp_path, "20KB")
    out = str(tmp_path / "out")
    _fake_hub(monkeypatch, base, fail = False)
    H.run_merge(peft_model, base, out, save_dtype = torch.float32, low_disk_space_usage = low_disk)
    expected = H.read_safetensors_dir(out)
    assert len([f for f in os.listdir(out) if f.endswith(".safetensors")]) > 1

    _fake_hub(monkeypatch, base, fail = True)
    with pytest.raises(OSError, match = "simulated"):
        H.run_merge(peft_model, base, out, save_dtype = torch.float32, low_disk_space_usage = low_disk)
    got = H.read_safetensors_dir(out)
    assert sorted(got) == sorted(expected)
    assert [k for k in expected if not torch.equal(got[k], expected[k])] == []
    assert not [f for f in os.listdir(out) if f.endswith(".unsloth-previous-export")]

    # A later successful export still merges onto the base, not onto the restored shards.
    _fake_hub(monkeypatch, base, fail = False)
    H.run_merge(peft_model, base, out, save_dtype = torch.float32, low_disk_space_usage = low_disk)
    again = H.read_safetensors_dir(out)
    assert [k for k in expected if not torch.equal(again[k], expected[k])] == []
