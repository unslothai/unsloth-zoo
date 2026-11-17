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

"""mxfp4 spool writes skip CRC32, restore the global option (even on error), and stay byte-identical."""

import hashlib
import os
from collections import defaultdict

import pytest
import torch
from safetensors.torch import save_file

import unsloth_zoo.saving_utils as su

_needs_crc_api = pytest.mark.skipif(
    not hasattr(torch.serialization, "set_crc32_options"),
    reason = "torch < 2.6 has no CRC32 option",
)

FNAME = "model-00001-of-00001.safetensors"


def _write_shard(directory):
    g = torch.Generator().manual_seed(0)
    E, D, G, B, H = 2, 8, 4, 16, 16
    kb = "model.layers.0.mlp.experts.gate_up_proj"
    tensors = {
        kb + "_blocks": torch.randint(0, 256, (E, D, G, B), dtype = torch.uint8, generator = g),
        kb + "_scales": torch.randint(120, 135, (E, D, G), dtype = torch.uint8, generator = g),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(H, H, generator = g).bfloat16(),
        "model.layers.0.input_layernorm.weight": torch.randn(H, generator = g).bfloat16(),
    }
    os.makedirs(directory, exist_ok = True)
    save_file(tensors, os.path.join(directory, FNAME), metadata = {"format": "pt"})


def _lora():
    g = torch.Generator().manual_seed(1)
    lw = defaultdict(lambda: su.LoraStats(None, None, None, 0))
    lw["model.layers.0.self_attn.q_proj"] = su.LoraStats(
        None, torch.randn(4, 16, generator = g), torch.randn(16, 4, generator = g), 8.0,
    )
    return lw


def _merge(directory):
    _write_shard(directory)
    su._merge_and_overwrite_lora_mxfp4(
        save_directory = directory, filename = FNAME, lora_weights = _lora(),
        output_dtype = torch.bfloat16, model_class_name = "GptOssForCausalLM",
        base_model_is_quantized = True, quant_type = "mxfp4",
    )
    with open(os.path.join(directory, FNAME), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@_needs_crc_api
def test_spool_writes_skip_crc32_and_restore_the_option(tmp_path, monkeypatch):
    seen = []
    real_save = torch.save

    def spy(*args, **kwargs):
        seen.append(torch.serialization.get_crc32_options())
        return real_save(*args, **kwargs)

    monkeypatch.setattr(torch, "save", spy)
    assert torch.serialization.get_crc32_options() is True
    _merge(str(tmp_path / "out"))
    assert len(seen) == 3, seen  # dequantized experts, merged q_proj, norm
    assert seen == [False] * len(seen)
    assert torch.serialization.get_crc32_options() is True


@_needs_crc_api
def test_merged_shard_is_byte_identical_to_a_crc32_spool(tmp_path, monkeypatch):
    fast = _merge(str(tmp_path / "fast"))
    real_save = torch.save

    def save_with_crc(tensor, path, pickle_module, pickle_protocol):
        assert torch.serialization.get_crc32_options() is True
        return real_save(tensor, path, pickle_module = pickle_module, pickle_protocol = pickle_protocol)

    monkeypatch.setattr(su, "_torch_save_spool", save_with_crc)
    assert _merge(str(tmp_path / "crc")) == fast


@_needs_crc_api
def test_crc32_option_restored_when_the_spool_write_fails(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        assert torch.serialization.get_crc32_options() is False
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", boom)
    import pickle
    with pytest.raises(OSError, match = "disk full"):
        su._torch_save_spool(torch.ones(2), str(tmp_path / "x.pt"), pickle, pickle.HIGHEST_PROTOCOL)
    assert torch.serialization.get_crc32_options() is True


def test_torch_without_crc32_option_keeps_plain_save(tmp_path, monkeypatch):
    calls = []
    monkeypatch.delattr(torch.serialization, "get_crc32_options", raising = False)
    monkeypatch.delattr(torch.serialization, "set_crc32_options", raising = False)
    monkeypatch.setattr(torch, "save", lambda *a, **k: calls.append(a[1]))
    import pickle
    su._torch_save_spool(torch.ones(2), str(tmp_path / "x.pt"), pickle, pickle.HIGHEST_PROTOCOL)
    assert calls == [str(tmp_path / "x.pt")]
