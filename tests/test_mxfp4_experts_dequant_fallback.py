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

"""Mxfp4GptOssExperts.gate_up_proj / down_proj decode the packed blocks when nothing swizzled
them at load, which is every transformers 5 load: load_and_swizzle_mxfp4 is only called by the
4.x quantizer. The old code imported transformers.integrations.mxfp4.dequantize, which 5.16.0
removed and which in every earlier release is a loader hook taking (module, param_name, ...),
not (blocks, scales), so that path could not work on any version.

_dequantize_mxfp4_experts must return GPT-OSS's (E, in, out) stack whichever
convert_moe_packed_tensors is installed (stock self-transposing, or Unsloth's un-transposed
replacement), must not call a loader-hook dequantize, and must decode on its own when
transformers has neither. gpt-oss's down_proj is square, so orientation cannot be read off
the output shape; the square case is covered explicitly."""
import itertools
import types

import pytest
import torch

import transformers.integrations.mxfp4 as mxfp4_integration
from unsloth_zoo.temporary_patches import gpt_oss
from unsloth_zoo.temporary_patches.gpt_oss import (
    _dequantize_mxfp4_experts,
    _mxfp4_dequantize_experts_torch,
)

_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _packed(E, rows, in_features, seed = 0):
    g = torch.Generator().manual_seed(seed)
    blocks = torch.randint(0, 256, (E, rows, in_features // 32, 16), dtype = torch.uint8, generator = g)
    # Exponents around the bias plus the extremes a checkpoint can hold.
    scales = torch.randint(118, 136, (E, rows, in_features // 32), dtype = torch.uint8, generator = g)
    scales[0, 0, 0] = 0
    scales[-1, -1, -1] = 254
    return blocks, scales


def _reference(blocks, scales):
    """Spec decode, element by element: weight[e, k, r] is input feature k of output row r."""
    E, R, G, B = blocks.shape
    out = torch.empty(E, G * B * 2, R, dtype = torch.float64)
    for e, r, gi, b in itertools.product(range(E), range(R), range(G), range(B)):
        byte = int(blocks[e, r, gi, b])
        scale = 2.0 ** (int(scales[e, r, gi]) - 127)
        k = gi * 32 + 2 * b
        out[e, k, r] = _E2M1[byte & 0x0F] * scale
        out[e, k + 1, r] = _E2M1[byte >> 4] * scale
    return out.to(torch.bfloat16)


def _stock_convert(blocks, scales, *, dtype = torch.bfloat16, rows_per_chunk = 0):
    """Self-transposing convention (stock transformers >= 4.56)."""
    return _untransposed_convert(blocks, scales, dtype = dtype).transpose(1, 2).contiguous()


def _untransposed_convert(blocks, scales, *, dtype = torch.bfloat16, rows_per_chunk = 0):
    """Unsloth's replacement convention (temporary_patches/mxfp4.py): (E, out, in)."""
    lut = torch.tensor(_E2M1, dtype = torch.float32)
    *prefix, G, B = blocks.shape
    out = torch.empty(*prefix, G, B * 2, dtype = torch.float32)
    out[..., 0::2] = lut[(blocks & 0x0F).long()]
    out[..., 1::2] = lut[(blocks >> 4).long()]
    out = torch.ldexp(out, (scales.to(torch.int32) - 127).unsqueeze(-1))
    return out.reshape(*prefix, G * B * 2).to(dtype)


@pytest.fixture(autouse = True)
def _fresh_probe_cache():
    gpt_oss._CONVERT_TRANSPOSES.clear()
    yield
    gpt_oss._CONVERT_TRANSPOSES.clear()


# (E, out rows, in features): gate_up-like (rows = 2 * in), down-like square, and wide.
SHAPES = [(2, 128, 64), (3, 64, 64), (2, 32, 96)]


@pytest.mark.parametrize("shape", SHAPES)
def test_local_decode_matches_spec(shape):
    blocks, scales = _packed(*shape)
    got = _mxfp4_dequantize_experts_torch(blocks, scales)
    assert got.dtype == torch.bfloat16
    assert tuple(got.shape) == (shape[0], shape[2], shape[1])
    assert torch.equal(got, _reference(blocks, scales))


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("convert", [_stock_convert, _untransposed_convert])
def test_either_convert_convention_gives_gpt_oss_layout(monkeypatch, shape, convert):
    blocks, scales = _packed(*shape, seed = 1)
    monkeypatch.setattr(mxfp4_integration, "convert_moe_packed_tensors", convert, raising = False)
    monkeypatch.delattr(mxfp4_integration, "dequantize", raising = False)
    assert torch.equal(_dequantize_mxfp4_experts(blocks, scales), _reference(blocks, scales))


def test_loader_hook_dequantize_is_not_called(monkeypatch):
    """4.x / <= 5.15 ship dequantize(module, param_name, param_value, target_device, dq_param_name)."""
    calls = []

    def dequantize(module, param_name, param_value, target_device, dq_param_name, **kwargs):
        calls.append(1)
        raise AssertionError("loader hook called with (blocks, scales)")

    blocks, scales = _packed(2, 64, 64, seed = 2)
    monkeypatch.setattr(mxfp4_integration, "dequantize", dequantize, raising = False)
    monkeypatch.setattr(mxfp4_integration, "convert_moe_packed_tensors", _stock_convert, raising = False)
    assert torch.equal(_dequantize_mxfp4_experts(blocks, scales), _reference(blocks, scales))
    assert calls == []


def test_blocks_scales_dequantize_is_used(monkeypatch):
    blocks, scales = _packed(2, 64, 32, seed = 3)
    calls = []

    def dequantize(blocks, scales):
        calls.append(1)
        return _stock_convert(blocks, scales)

    monkeypatch.setattr(mxfp4_integration, "dequantize", dequantize, raising = False)
    assert torch.equal(_dequantize_mxfp4_experts(blocks, scales), _reference(blocks, scales))
    assert calls == [1]


def test_falls_back_to_local_decode(monkeypatch):
    blocks, scales = _packed(2, 64, 64, seed = 4)
    monkeypatch.delattr(mxfp4_integration, "dequantize", raising = False)
    monkeypatch.delattr(mxfp4_integration, "convert_moe_packed_tensors", raising = False)
    assert torch.equal(_dequantize_mxfp4_experts(blocks, scales), _reference(blocks, scales))

    def broken(blocks, scales, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(mxfp4_integration, "convert_moe_packed_tensors", broken, raising = False)
    assert torch.equal(_dequantize_mxfp4_experts(blocks, scales), _reference(blocks, scales))


def test_installed_transformers_convert(monkeypatch):
    """Whatever convert_moe_packed_tensors this transformers ships (or Unsloth installed)."""
    if not hasattr(mxfp4_integration, "convert_moe_packed_tensors"):
        pytest.skip("transformers has no convert_moe_packed_tensors")
    blocks, scales = _packed(3, 64, 64, seed = 5)
    got = _dequantize_mxfp4_experts(blocks, scales)
    assert got.device == blocks.device
    assert torch.equal(got, _reference(blocks, scales))


def test_mismatched_scales_rejected():
    blocks, scales = _packed(2, 64, 64)
    with pytest.raises(ValueError):
        _mxfp4_dequantize_experts_torch(blocks, scales[:, :, :1])
