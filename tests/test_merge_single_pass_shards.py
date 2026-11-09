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

"""A plain merged_16bit export writes each shard once from its source, byte-identical to copy-then-overwrite."""

from __future__ import annotations

import hashlib
import os
import shutil

import pytest
import torch

import _merge_e2e_helpers as H

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _build(tmp_path, shard_size = "5GB", tied = False):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    H.set_offline_cpu_env()
    if not H.family_available("llama"):
        pytest.skip("llama unavailable")
    base = str(tmp_path / "base")
    config = H.make_spec("llama").config
    config.tie_word_embeddings = tied
    torch.manual_seed(H.SEED)
    AutoModelForCausalLM.from_config(config).to(torch.bfloat16).save_pretrained(
        base, safe_serialization = True, max_shard_size = shard_size)
    model = AutoModelForCausalLM.from_pretrained(base, dtype = torch.float32)
    model.config._name_or_path = base
    peft_model = get_peft_model(model, LoraConfig(
        r = 4, lora_alpha = 8, lora_dropout = 0.0, bias = "none", target_modules = TARGETS))
    H.seed_lora(peft_model)
    return base, peft_model


def _digests(directory):
    out = {}
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            with open(path, "rb") as f:
                out[name] = hashlib.sha256(f.read()).hexdigest()
    return out


def _count_shard_copies(monkeypatch):
    from unsloth_zoo import saving_utils as S

    copies = []
    real_copy = S._copy_file_from_source
    def _copy(src, target_dir, filename):
        if str(filename).endswith(".safetensors"):
            copies.append(filename)
        return real_copy(src, target_dir, filename)
    monkeypatch.setattr(S, "_copy_file_from_source", _copy)
    return copies


def _fake_cached_hub(monkeypatch, tmp_path, base):
    # A hub-cache snapshot (links into blobs/); only the Hub calls are faked.
    from huggingface_hub.errors import LocalEntryNotFoundError
    from unsloth_zoo import saving_utils as S

    repo = tmp_path / "hub" / "models--fake--repo"
    snapshot = repo / "snapshots" / ("0" * 40)
    (repo / "blobs").mkdir(parents = True)
    snapshot.mkdir(parents = True)
    names = sorted(os.listdir(base))
    for name in names:
        data = open(os.path.join(base, name), "rb").read()
        blob = repo / "blobs" / hashlib.sha256(data).hexdigest()
        blob.write_bytes(data)
        os.chmod(blob, 0o444)
        (snapshot / name).symlink_to(os.path.join("..", "..", "blobs", blob.name))
    shards = [n for n in names if n.endswith(".safetensors")]

    class _FS:
        def __init__(self, *a, **k):
            pass
        def ls(self, name, detail = True, **k):
            return [{"name": f"{name}/{f}", "size": os.path.getsize(snapshot / f)} for f in shards]

    def _hf_hub(repo_id, filename, local_dir = None, local_files_only = False, **k):
        path = snapshot / filename
        if local_files_only and local_dir is None and path.exists():
            return str(path)
        raise LocalEntryNotFoundError("not cached")

    monkeypatch.setattr(S, "determine_base_model_source", lambda *a, **k: ("fake/repo", False, "hub", False, None))
    monkeypatch.setattr(S, "HfFileSystem", _FS)
    monkeypatch.setattr(S, "hf_hub_download", _hf_hub)
    monkeypatch.setattr(S, "snapshot_download", lambda repo_id, local_dir = None, **k: local_dir)
    monkeypatch.setattr(S, "_get_hf_cache_dir", lambda: tmp_path / "hub")
    return "fake/repo"


@pytest.mark.parametrize("source", ["local", "cache"])
@pytest.mark.parametrize("shard_size", ["5GB", "20KB"])
@pytest.mark.parametrize("tied", [False, True])
def test_shards_are_written_once_and_match_the_copy_path(tmp_path, monkeypatch, source, shard_size, tied):
    base, peft_model = _build(tmp_path, shard_size, tied)
    model_name = base if source == "local" else _fake_cached_hub(monkeypatch, tmp_path, base)
    n_shards = len([f for f in os.listdir(base) if f.endswith(".safetensors")])

    monkeypatch.setenv("UNSLOTH_MERGE_SINGLE_PASS", "0")
    copied_path = str(tmp_path / "copied")
    copies = _count_shard_copies(monkeypatch)
    H.run_merge(peft_model, model_name, copied_path, save_dtype = None)
    assert len(copies) == n_shards

    monkeypatch.setenv("UNSLOTH_MERGE_SINGLE_PASS", "1")
    single = str(tmp_path / "single")
    copies.clear()
    H.run_merge(peft_model, model_name, single, save_dtype = None)
    # No full copy of a base shard lands in the export before the merge rewrites it.
    assert copies == []
    assert _digests(single) == _digests(copied_path)
    base_tensors, merged = H.read_safetensors_dir(base), H.read_safetensors_dir(single)
    assert any(not torch.equal(merged[k], base_tensors[k]) for k in merged if "q_proj" in k)
    assert not [f for f in os.listdir(single) if f.startswith(".unsloth-")]
    if source == "cache":
        # The cache blobs are read, never written.
        for blob in (tmp_path / "hub" / "models--fake--repo" / "blobs").iterdir():
            assert os.stat(blob).st_mode & 0o222 == 0


def test_reexport_over_a_previous_export_is_identical(tmp_path, monkeypatch):
    base, peft_model = _build(tmp_path, "20KB")
    fresh, out = str(tmp_path / "fresh"), str(tmp_path / "out")
    H.run_merge(peft_model, base, fresh, save_dtype = None)
    H.run_merge(peft_model, base, out, save_dtype = None)
    copies = _count_shard_copies(monkeypatch)
    H.run_merge(peft_model, base, out, save_dtype = None)
    assert copies == []
    assert _digests(out) == _digests(fresh)


def test_failed_merge_keeps_the_previous_export_shard(tmp_path, monkeypatch):
    from unsloth_zoo import saving_utils as S

    base, peft_model = _build(tmp_path, "20KB")
    out = str(tmp_path / "out")
    H.run_merge(peft_model, base, out, save_dtype = None)
    before = _digests(out)

    calls = []
    real = S._write_tensor_direct_torch
    def _write_then_fail(*a, **k):
        calls.append(1)
        if len(calls) > 3:
            raise KeyboardInterrupt
        return real(*a, **k)
    monkeypatch.setattr(S, "_write_tensor_direct_torch", _write_then_fail)
    with pytest.raises(KeyboardInterrupt):
        H.run_merge(peft_model, base, out, save_dtype = None)
    # The shard being written is staged: the previous export stays whole, nothing is left behind.
    assert _digests(out) == before
    assert not [f for f in os.listdir(out) if f.startswith(".unsloth-")]


def _shard_bytes(tmp_path, tensors):
    from safetensors.torch import save_file

    path = str(tmp_path / "src.safetensors")
    save_file(tensors, path, metadata = {"format": "pt"})
    return path


@pytest.mark.parametrize("kernel_copy", [
    pytest.param(True, marks = pytest.mark.skipif(not hasattr(os, "copy_file_range"), reason = "os.copy_file_range is Linux only")),
    False,
])
def test_writer_matches_copy_then_overwrite_in_any_write_order(tmp_path, kernel_copy):
    import json
    import mmap
    from unsloth_zoo.saving_utils import _SinglePassShardWriter

    torch.manual_seed(0)
    tensors = {f"t{i}": torch.randn(3 + i, 5, dtype = torch.bfloat16) for i in range(6)}
    tensors["norm"] = torch.ones(7, dtype = torch.float32)
    src = _shard_bytes(tmp_path, tensors)
    with open(src, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    replace = {"t4": -tensors["t4"], "t1": tensors["t1"] * 2, "t5": tensors["t5"] + 1}

    expected = str(tmp_path / "expected.safetensors")
    shutil.copyfile(src, expected)
    with open(expected, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)
        for key, value in replace.items():
            a, b = header[key]["data_offsets"]
            mm[8 + n + a: 8 + n + b] = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        mm.flush(); mm.close()

    out = str(tmp_path / "out.safetensors")
    writer = _SinglePassShardWriter(src, out)
    writer._kernel_copy = kernel_copy
    for key, value in replace.items():  # t4 before t1: one write lands behind the cursor
        a, b = header[key]["data_offsets"]
        writer[8 + n + a: 8 + n + b] = memoryview(value.reshape(-1).view(torch.uint8).numpy())
    with pytest.raises(IndexError):
        writer[8 + n: 8 + n + 1] = memoryview(b"\x00\x00")
    writer.commit()
    writer.close()
    assert open(out, "rb").read() == open(expected, "rb").read()
    assert os.stat(out).st_mode & 0o200
    assert sorted(os.listdir(tmp_path)) == ["expected.safetensors", "out.safetensors", "src.safetensors"]


def test_writer_close_without_commit_leaves_destination(tmp_path):
    from unsloth_zoo.saving_utils import _SinglePassShardWriter

    src = _shard_bytes(tmp_path, {"a": torch.zeros(4)})
    out = tmp_path / "out.safetensors"
    out.write_bytes(b"previous")
    writer = _SinglePassShardWriter(src, str(out))
    writer.close()
    assert out.read_bytes() == b"previous"
    assert sorted(os.listdir(tmp_path)) == ["out.safetensors", "src.safetensors"]
