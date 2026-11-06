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

"""The pre-merge "no adapter placed" refusal must accept every quantized backing the merge itself
dequantizes and merges into: NVFP4 weight_packed, FP8 block weight_scale_inv, mxfp4 _blocks/_scales
and compressed-tensors int / mxfp4 weight_packed (through their disk views)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import pytest
import torch
from safetensors.torch import save_file

import unsloth_zoo.saving_utils as SU

if not hasattr(torch, "float8_e4m3fn"):
    pytest.skip("float8_e4m3fn unavailable", allow_module_level = True)

OUT, IN = 32, 64
PREFIX = "model.layers.0.self_attn."
PROJS = ("q_proj", "v_proj")


@dataclass
class _FakeLoraStats:
    lora_A: torch.Tensor = None
    lora_B: torch.Tensor = None
    alpha: float = 1.0
    module: object = None


def _lora():
    return defaultdict(lambda: None, {
        PREFIX + p: _FakeLoraStats(lora_A = torch.ones(2, IN), lora_B = torch.ones(OUT, 2), module = torch.nn.Linear(IN, OUT))
        for p in PROJS
    })


def _nvfp4(base):
    return {
        base + ".weight_packed": torch.zeros(OUT, IN // 2, dtype = torch.uint8),
        base + ".weight_scale": torch.ones(OUT, IN // 16).to(torch.float8_e4m3fn),
        base + ".weight_global_scale": torch.tensor([1.0]),
        base + ".input_global_scale": torch.tensor([1.0]),
    }


def _fp8_block(base):
    return {
        base + ".weight": torch.zeros(OUT, IN).to(torch.float8_e4m3fn),
        base + ".weight_scale_inv": torch.ones(1, 1),
    }


def _mxfp4(base):
    return {
        base + "_blocks": torch.zeros(OUT, IN // 32, 16, dtype = torch.uint8),
        base + "_scales": torch.zeros(OUT, IN // 32, dtype = torch.uint8),
    }


def _int_packed(base):
    return {
        base + ".weight_packed": torch.zeros(OUT, IN // 8, dtype = torch.int32),
        base + ".weight_scale": torch.ones(OUT, 1),
        base + ".weight_shape": torch.tensor([OUT, IN]),
    }


def _stage(tmp_path, build, prefix = PREFIX):
    tensors = {"model.embed_tokens.weight": torch.zeros(8, IN)}
    for p in PROJS:
        tensors.update(build(prefix + p))
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return ["model.safetensors"]


def _check(tmp_path, files, disk = None):
    return SU._check_lora_merge_is_complete(
        str(tmp_path), files, _lora(), "Qwen3ForCausalLM", disk = disk,
    )


@pytest.fixture(autouse = True)
def _no_override(monkeypatch):
    monkeypatch.delenv("UNSLOTH_ALLOW_PARTIAL_LORA_MERGE", raising = False)


@pytest.mark.parametrize("build", [_nvfp4, _fp8_block, _mxfp4], ids = ["nvfp4", "fp8_block", "mxfp4"])
def test_every_target_quantized_is_not_refused(tmp_path, build):
    assert _check(tmp_path, _stage(tmp_path, build)) == {}


def test_compressed_int_view_is_not_refused(tmp_path):
    files = _stage(tmp_path, _int_packed)
    plan = {"logical": {PREFIX + p: (OUT, IN) for p in PROJS}}
    assert _check(tmp_path, files, disk = SU._compressed_int_disk_view(str(tmp_path), files, plan)) == {}


def test_compressed_mxfp4_view_is_not_refused(tmp_path):
    files = _stage(tmp_path, lambda b: {
        b + ".weight_packed": torch.zeros(OUT, IN // 2, dtype = torch.uint8),
        b + ".weight_scale": torch.zeros(OUT, IN // 32, dtype = torch.uint8),
    })
    plan = {
        "packed_bases": {PREFIX + p for p in PROJS},
        "shapes": {PREFIX + p + ".weight_packed": (OUT, IN // 2) for p in PROJS},
    }
    assert _check(tmp_path, files, disk = SU._compressed_mxfp4_disk_view(str(tmp_path), files, plan)) == {}


@pytest.mark.parametrize("build", [_nvfp4, _fp8_block, lambda b: {b + ".weight": torch.zeros(OUT, IN)}],
                         ids = ["nvfp4", "fp8_block", "bf16"])
def test_adapters_with_no_checkpoint_tensor_are_still_refused(tmp_path, build):
    # Weights under a layout no rule maps: no adapter lands, so the export would be the base model.
    files = _stage(tmp_path, build, prefix = "thinker.unmapped.self_attn.")
    with pytest.raises(SU.PartialLoraMergeError, match = "None of the"):
        _check(tmp_path, files)
