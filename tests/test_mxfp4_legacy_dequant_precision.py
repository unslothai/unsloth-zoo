# SPDX-License-Identifier: AGPL-3.0-only
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
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""torch < 2.12 decodes MXFP4 through fp32 (its ldexp forms 2^e in the output dtype). That detour must not narrow a
float64 output, and the converter's generic loop (non 16-byte blocks, CPU, other dtypes) must take it too."""

import pytest
import torch

import unsloth_zoo.mxfp4_dequant as mxd
from unsloth_zoo.mxfp4_dequant import mxfp4_dequantize_torch

_FP4 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _every_byte_and_scale():
    blocks = torch.arange(256, dtype = torch.uint8).reshape(16, 1, 16).repeat(256, 1, 1)
    scales = torch.arange(256, dtype = torch.uint8).repeat_interleave(16).reshape(4096, 1)
    lut = torch.tensor(_FP4, dtype = torch.float64)
    want = torch.empty(4096, 1, 32, dtype = torch.float64)
    want[..., 0::2] = lut[(blocks & 0x0F).long()]
    want[..., 1::2] = lut[(blocks >> 4).long()]
    return blocks, scales, (want * torch.exp2(scales.double() - 127).unsqueeze(-1)).reshape(4096, 32)


_DEVICES = ["cpu", pytest.param("cuda", marks = pytest.mark.skipif(not torch.cuda.is_available(), reason = "no CUDA"))]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("old_torch_path", [False, True])
def test_float64_output_is_exact_on_every_torch(monkeypatch, old_torch_path, device):
    if old_torch_path:
        monkeypatch.setattr(mxd, "_LDEXP_EXACT", False)
    blocks, scales, want = _every_byte_and_scale()
    # Old CUDA float64 ldexp rounds (0x11 at scale 14 gave 4.8148e-35, not 2^-114).
    got = mxfp4_dequantize_torch(blocks.to(device), scales.to(device), dtype = torch.float64).cpu()
    assert int((got != want).sum()) == 0
    # 6 * 2^128: finite in float64, inf if it passes through fp32.
    top = mxfp4_dequantize_torch(torch.full((1, 1, 16), 0x77, dtype = torch.uint8, device = device), torch.full((1, 1), 255, dtype = torch.uint8, device = device), dtype = torch.float64)
    assert torch.equal(top.cpu(), torch.full((1, 32), 6 * 2.0 ** 128, dtype = torch.float64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "no CUDA")
@pytest.mark.parametrize("old_torch_path", [False, True])
def test_float64_temporaries_stay_within_the_chunk_budget(monkeypatch, old_torch_path):
    if old_torch_path:
        monkeypatch.setattr(mxd, "_LDEXP_EXACT", False)
    blocks = torch.full((8192, 128, 16), 0x77, dtype = torch.uint8, device = "cuda")
    scales = torch.full((8192, 128), 127, dtype = torch.uint8, device = "cuda")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = mxfp4_dequantize_torch(blocks, scales, dtype = torch.float64)
    temp = torch.cuda.max_memory_allocated() - base - out.numel() * out.element_size()
    # Merge base peaks at 516 MiB here on every torch; the old float64 ldexp + where path reached 773 MiB.
    assert temp <= 2 * mxd._TORCH_CHUNK_BYTES + (8 << 20), temp / 2**20


@pytest.mark.parametrize("block_bytes", [8, 16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_converter_generic_loop_is_exact_at_extreme_scales(dtype, block_bytes):
    M = pytest.importorskip("transformers.integrations.mxfp4")
    from unsloth_zoo.temporary_patches.mxfp4 import patch_convert_moe_packed_tensors
    patch_convert_moe_packed_tensors()
    convert = getattr(M, "convert_moe_packed_tensors", None)
    if convert is None:
        pytest.skip("no convert_moe_packed_tensors on this transformers")
    lut = torch.tensor(_FP4, dtype = torch.float64)
    cases = [(0x11, 143), (0x00, 255), (0x77, 127), (0x7F, 0)]  # 0.5 * 2^16 (fp16 inf before), 0 * 2^128 (NaN)
    for byte, scale in cases:
        blocks = torch.full((2, 3, 4, block_bytes), byte, dtype = torch.uint8)
        scales = torch.full((2, 3, 4), scale, dtype = torch.uint8)
        got = convert(blocks, scales, dtype = dtype).cpu()
        want = torch.empty(2, 3, 4, block_bytes * 2, dtype = torch.float64)
        want[..., 0::2] = lut[byte & 0x0F]
        want[..., 1::2] = lut[byte >> 4]
        want = (want * 2.0 ** (scale - 127)).reshape(2, 3, 4 * block_bytes * 2).to(dtype)
        assert got.shape == want.shape and got.dtype == dtype
        assert torch.equal(got, want), (byte, scale, got.flatten()[:4], want.flatten()[:4])
