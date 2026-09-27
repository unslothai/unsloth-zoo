# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""A 16bit merge of an fp8 base reads the block size from the base's own config.json.

A 4bit (NF4) load of an fp8 checkpoint holds a bitsandbytes quantization_config in memory, so
the merge found no `weight_block_size` there and inferred it from the scale grid. A ragged dim
that the grid still divides evenly (192 rows over 2 scale rows) then dequantized with block 96
instead of 128 plus a partial block, and the merged weights came out wrong.
"""

import json

import pytest
import torch

if not hasattr(torch, "float8_e4m3fn"):
    pytest.skip("float8_e4m3fn unavailable", allow_module_level = True)

from safetensors import safe_open
from safetensors.torch import save_file

from unsloth_zoo import saving_utils

_BLOCK = 128


def _quantize(weight):
    rows, cols = weight.shape
    grid_r, grid_c = -(-rows // _BLOCK), -(-cols // _BLOCK)
    padded = torch.zeros(grid_r * _BLOCK, grid_c * _BLOCK)
    padded[:rows, :cols] = weight
    blocks = padded.view(grid_r, _BLOCK, grid_c, _BLOCK)
    scale = blocks.abs().amax(dim = (1, 3)).clamp(min = 1e-12) / 448.0
    quant = (blocks / scale[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return quant.view(grid_r * _BLOCK, grid_c * _BLOCK)[:rows, :cols].contiguous(), scale


def _reference(quant, scale):
    rows, cols = quant.shape
    grid = scale.repeat_interleave(_BLOCK, 0)[:rows].repeat_interleave(_BLOCK, 1)[:, :cols]
    return (quant.float() * grid).to(torch.bfloat16)


def _write(tmp_path, quantization_config):
    torch.manual_seed(0)
    quant, scale = _quantize(torch.randn(192, 256) * 0.02)
    save_file(
        {"layer.weight": quant, "layer.weight_scale_inv": scale},
        str(tmp_path / "model.safetensors"),
        metadata = {"format": "pt"},
    )
    config = {"model_type": "llama"}
    if quantization_config is not None:
        config["quantization_config"] = quantization_config
    (tmp_path / "config.json").write_text(json.dumps(config))
    return quant, scale


def test_block_size_comes_from_the_fp8_base_config(tmp_path):
    _write(tmp_path, {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]})
    assert saving_utils._fp8_block_size_on_disk(str(tmp_path)) == (128, 128)


@pytest.mark.parametrize(
    "quantization_config",
    [None, {"quant_method": "bitsandbytes", "load_in_4bit": True, "bnb_4bit_quant_type": "nf4"}],
)
def test_no_block_size_for_non_fp8_bases(tmp_path, quantization_config):
    _write(tmp_path, quantization_config)
    assert saving_utils._fp8_block_size_on_disk(str(tmp_path)) is None


def test_on_disk_block_size_dequantizes_an_evenly_divided_ragged_dim(tmp_path):
    quant, scale = _write(tmp_path, {"quant_method": "fp8", "weight_block_size": [128, 128]})
    path = str(tmp_path / "model.safetensors")
    with safe_open(path, framework = "pt") as handle:
        header = {key: None for key in handle.keys()}
        block = saving_utils._fp8_block_size_on_disk(str(tmp_path))
        got, _ = saving_utils._fp8_dequantize_weight(handle, header, "layer.weight", weight_block_size = block)
        guessed, _ = saving_utils._fp8_dequantize_weight(handle, header, "layer.weight", weight_block_size = None)
    expected = _reference(quant, scale)
    assert torch.equal(got.to(torch.bfloat16), expected)
    # What the merge produced before: 192 / 2 scale rows read as a 96-row block.
    assert not torch.equal(guessed.to(torch.bfloat16), expected)
