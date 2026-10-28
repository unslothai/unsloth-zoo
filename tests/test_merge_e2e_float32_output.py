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

"""float32 export (issue 5186): fp32(base)+scale*(B@A) unrounded; output_dtype None keeps base dtype."""

from __future__ import annotations

import json
import os
import struct

import pytest
import torch
from safetensors.torch import save_file

import _merge_e2e_helpers as H


def _header_dtypes(out_dir):
    dtypes = {}
    for f in sorted(os.listdir(out_dir)):
        if not f.endswith(".safetensors"):
            continue
        with open(os.path.join(out_dir, f), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        dtypes.update({k: v["dtype"] for k, v in header.items() if k != "__metadata__"})
    return dtypes


def _build(family, work_dir, max_shard_size = "5GB"):
    H.set_offline_cpu_env()
    spec = H.make_spec(family)
    base_dir = os.path.join(work_dir, "base")
    model = H.build_and_save_base(spec, base_dir, dtype = torch.bfloat16, max_shard_size = max_shard_size)
    base = H.read_safetensors_dir(base_dir)
    peft_model = H.attach_lora(model, spec, "full")
    return base_dir, base, peft_model, H.extract_adapted(peft_model)


@pytest.mark.parametrize("output_dtype", [torch.float32, "float32"])
@pytest.mark.parametrize("max_shard_size", ["5GB", "40KB"])
def test_bf16_base_float32_merge(tmp_path, output_dtype, max_shard_size):
    base_dir, base, peft_model, adapted = _build("llama", str(tmp_path), max_shard_size)
    out_dir = str(tmp_path / "merged")
    H.run_merge(peft_model, base_dir, out_dir, save_dtype = output_dtype)

    dtypes = _header_dtypes(out_dir)
    assert dtypes and set(dtypes.values()) == {"F32"}, set(dtypes.values())
    merged = H.read_safetensors_dir(out_dir)
    assert set(merged) == set(base)

    n_adapted = n_lossy = 0
    for key, got in merged.items():
        if key in adapted:
            ref = H._ref_dense(base[key], adapted[key])
            torch.testing.assert_close(got.double(), ref, atol = 1e-5, rtol = 1e-5)
            # The bf16 merge would have rounded these away.
            n_lossy += int((got != ref.to(torch.bfloat16).float()).sum())
            n_adapted += 1
        else:
            assert torch.equal(got, base[key].float()), key
    assert n_adapted >= 1 and n_lossy > 0

    with open(os.path.join(out_dir, "config.json"), encoding = "utf-8") as f:
        config = json.load(f)
    assert config.get("torch_dtype", config.get("dtype")) == "float32"

    index = H.read_index(out_dir)
    if index is not None:
        for key, shard in index["weight_map"].items():
            assert os.path.exists(os.path.join(out_dir, shard)) and key in merged

    from transformers import AutoModelForCausalLM
    reloaded = AutoModelForCausalLM.from_pretrained(out_dir, dtype = "auto")
    assert {p.dtype for p in reloaded.parameters()} == {torch.float32}


@pytest.mark.parametrize("output_dtype", [torch.float32, None])
def test_sharded_index_total_size_matches_the_written_shards(tmp_path, output_dtype):
    """The copied index describes the 16-bit base; a float32 export must record its real size."""
    base_dir, base, peft_model, adapted = _build("llama", str(tmp_path), "40KB")
    out_dir = str(tmp_path / "merged")
    H.run_merge(peft_model, base_dir, out_dir, save_dtype = output_dtype)

    index = H.read_index(out_dir)
    assert index is not None and len(set(index["weight_map"].values())) > 1
    merged = H.read_safetensors_dir(out_dir)
    actual = sum(t.numel() * t.element_size() for t in merged.values())
    assert index["metadata"]["total_size"] == actual
    if output_dtype is torch.float32:
        assert H.read_index(base_dir)["metadata"]["total_size"] * 2 == actual


def test_default_dtype_keeps_bf16(tmp_path):
    base_dir, base, peft_model, adapted = _build("llama", str(tmp_path))
    out_dir = str(tmp_path / "merged")
    H.run_merge(peft_model, base_dir, out_dir, save_dtype = None)
    assert set(_header_dtypes(out_dir).values()) == {"BF16"}
    with open(os.path.join(out_dir, "config.json"), encoding = "utf-8") as f:
        config = json.load(f)
    assert config.get("torch_dtype", config.get("dtype")) == "bfloat16"


def test_unknown_dtype_string_rejected():
    from unsloth_zoo.saving_utils import _normalize_output_dtype
    assert _normalize_output_dtype("torch.float32") is torch.float32
    assert _normalize_output_dtype(None) is None
    with pytest.raises(ValueError):
        _normalize_output_dtype("float8")


def test_upcast_shard_casts_only_16bit_floats(tmp_path):
    from unsloth_zoo.saving_utils import _upcast_shard_to_float32
    tensors = {
        "a.weight": torch.randn(3, 5).to(torch.bfloat16),
        "b.weight": torch.randn(4).to(torch.float16),
        "c.weight": torch.randn(2, 2),
        "ids": torch.arange(7, dtype = torch.int64),
        "scalar": torch.tensor(1.5, dtype = torch.bfloat16),
        # Larger than the 64 MB copy chunk, so the cast crosses a chunk boundary.
        "big.weight": torch.randn(33 * 1024 * 1024 + 5).to(torch.bfloat16),
    }
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path), metadata = {"format": "pt"})
    _upcast_shard_to_float32(str(tmp_path), "model.safetensors")
    out = H.read_safetensors_dir(str(tmp_path))
    for key, t in tensors.items():
        want = t.float() if t.dtype in (torch.bfloat16, torch.float16) else t
        assert out[key].dtype == want.dtype and torch.equal(out[key], want), key
    # Idempotent: a second pass leaves the file untouched.
    before = path.read_bytes()
    _upcast_shard_to_float32(str(tmp_path), "model.safetensors")
    assert path.read_bytes() == before


def test_float32_refuses_before_writing_when_disk_cannot_hold_the_rewrite(tmp_path, monkeypatch):
    # Widening a shard holds it and its float32 copy at once: final size + one 16-bit shard.
    import shutil
    base_dir, base, peft_model, adapted = _build("llama", str(tmp_path))
    shard = sum(os.path.getsize(os.path.join(base_dir, f)) for f in os.listdir(base_dir) if f.endswith(".safetensors"))
    final = sum(t.numel() * 4 for t in base.values())
    free = int((final + shard // 2) / 0.95)
    real = shutil.disk_usage
    monkeypatch.setattr(shutil, "disk_usage", lambda p: real(p)._replace(free = free))
    out_dir = tmp_path / "merged"
    with pytest.raises(RuntimeError, match = "free disk"):
        H.run_merge(peft_model, base_dir, str(out_dir), save_dtype = torch.float32)
    assert not any(out_dir.glob("*.safetensors"))


def test_float32_peak_counts_every_16bit_tensor_and_the_rewrite():
    from unsloth_zoo.saving_utils import _float32_merge_peak_bytes
    # Float32 peak: 16-bit tensors widened, plus a resized rewrite's second copy of the largest shard.
    assert _float32_merge_peak_bytes(10, 4) == 28
    # Streamed to the Hub: the 16-bit copies plus the largest shard's float32 copy and temp.
    assert _float32_merge_peak_bytes(10, 4, uploads_as_it_goes = True) == 22


def test_index_total_size_keeps_a_cp1252_index_byte_for_byte(tmp_path):
    """An index the loader reads in the locale encoding is fixed in place, not skipped or re-encoded."""
    from unsloth_zoo.saving_utils import _set_index_total_size
    raw = (
        b'{\n  "metadata": {"total_size": 100},\n'
        b'  "weight_map": {"caf\xe9.weight": "model-00001-of-00002.safetensors"}\n}\n'
    )
    path = tmp_path / "model.safetensors.index.json"
    path.write_bytes(raw)
    assert _set_index_total_size(str(tmp_path), 200)
    assert path.read_bytes() == raw.replace(b'"total_size": 100', b'"total_size": 200')
    assert not _set_index_total_size(str(tmp_path), 200)
