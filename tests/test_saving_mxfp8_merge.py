# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
"""16-bit merges of compressed-tensors MXFP8 checkpoints decode the e8m0 1x32 scales (they were read as raw bytes)."""

import pytest
import torch

from unsloth_zoo.saving_utils import _fp8_dequantize_weight, _is_mxfp8_compressed_tensors_config
from unsloth_zoo.temporary_patches.moe_utils_fp8 import _fp8_dequant_blockwise


def _mxfp8(rows = 8, cols = 64, seed = 0):
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 256, (rows, cols), dtype = torch.uint8, generator = g)
    codes[(codes & 0x7F) == 0x7F] = 0
    scales = torch.randint(115, 135, (rows, cols // 32), dtype = torch.uint8, generator = g)
    return codes.view(torch.float8_e4m3fn), scales


def test_e8m0_scales_decode_as_powers_of_two():
    w, s = _mxfp8()
    want = w.float() * torch.exp2(s.float() - 127).repeat_interleave(32, dim = 1)
    assert torch.equal(_fp8_dequant_blockwise(w, s), want)


def test_dense_fp8_scales_unchanged():
    w, _ = _mxfp8()
    s = torch.rand(8, 1) + 0.5
    assert torch.equal(_fp8_dequant_blockwise(w, s), w.float() * s)


def test_mxfp8_config_detected():
    groups = {"group_0": {"format": "mxfp8-quantized", "weights": {"num_bits": 8, "type": "float", "group_size": 32}}}
    assert _is_mxfp8_compressed_tensors_config({"quant_method": "compressed-tensors", "format": "mxfp8-quantized"})
    assert _is_mxfp8_compressed_tensors_config({"quant_method": "compressed-tensors", "config_groups": groups})
    assert not _is_mxfp8_compressed_tensors_config({"quant_method": "compressed-tensors", "format": "float-quantized"})
    assert not _is_mxfp8_compressed_tensors_config({"quant_method": "fp8"})


def test_merge_rewrite_reads_the_checkpoint_scales(tmp_path):
    from safetensors import safe_open
    from safetensors.torch import save_file
    w, s = _mxfp8()
    path = str(tmp_path / "m.safetensors")
    save_file({"a.weight": w, "a.weight_scale": s}, path)
    with safe_open(path, framework = "pt") as f:
        header = {k: None for k in f.keys()}
        W, dropped = _fp8_dequantize_weight(f, header, "a.weight")
    assert dropped == ["a.weight_scale"]
    assert torch.equal(W, w.float() * torch.exp2(s.float() - 127).repeat_interleave(32, dim = 1))
