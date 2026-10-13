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

"""PyTorch FP8 expert dequant fallbacks: an FP16 target rounds q * s once, never the scale first (subnormal scales flushed to 0)."""

import pytest
import torch

from unsloth_zoo.temporary_patches import moe_utils_fp8 as F

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
E, M, N = 2, 8, 16
SMALL_SCALES = (1e-8, 1e-7, 3e-5)


def _fp8(shape, device):
    g = torch.Generator().manual_seed(0)
    q = (torch.rand(shape, generator = g) * 2 - 1) * 448
    return q.to(torch.float8_e4m3fn).to(device)


def _reference(q, s_broadcast, dtype):
    return (q.to(torch.float32) * s_broadcast.to(torch.float32)).to(dtype)


def _layouts(device, scale):
    full = lambda *shape: torch.full(shape, scale, dtype = torch.float32, device = device)
    return {
        "per_tensor": (full(1), lambda s: s.view(1, 1, 1)),
        "per_expert": (full(E, 1, 1), lambda s: s),
        "block": (full(E, 2, 2), lambda s: s.repeat_interleave(M // 2, 1).repeat_interleave(N // 2, 2)),
        "rows_EM": (full(E, M), lambda s: s.unsqueeze(-1)),
        "cols_EN": (full(E, N), lambda s: s.unsqueeze(1)),
        "expert_E1": (full(E, 1), lambda s: s.unsqueeze(-1)),
        "rows_M": (full(M), lambda s: s.view(1, -1, 1)),
        "cols_N": (full(N), lambda s: s.view(1, 1, -1)),
    }


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("scale", SMALL_SCALES)
@pytest.mark.parametrize("layout", list(_layouts("cpu", 1.0)))
def test_vectorized_fp16_rounds_once(device, scale, layout):
    q = _fp8((E, M, N), device)
    s, expand = _layouts(device, scale)[layout]
    out = F._dequantize_full_expert_weights_vectorized(q, s, torch.float16)
    assert out.dtype == torch.float16
    assert torch.equal(out, _reference(q, expand(s), torch.float16))
    if scale == 1e-8:
        assert out.abs().amax() > 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("layout", list(_layouts("cpu", 1.0)))
def test_vectorized_bf16_unchanged(device, layout):
    q = _fp8((E, M, N), device)
    s, expand = _layouts(device, 3e-5)[layout]
    out = F._dequantize_full_expert_weights_vectorized(q, s, torch.bfloat16)
    # bf16 keeps its scale-in-target-dtype product, bit for bit.
    assert torch.equal(out, q.to(torch.bfloat16) * expand(s).to(torch.bfloat16))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("scale", SMALL_SCALES)
def test_expert_slice_fp16_rounds_once(device, scale):
    q = _fp8((M, N), device)
    cases = [
        (torch.full((1,), scale, device = device), lambda s: s.view(1, 1)),
        (torch.full((M,), scale, device = device), lambda s: s.view(-1, 1)),
        (torch.full((N,), scale, device = device), lambda s: s.view(1, -1)),
        (torch.full((2, 2), scale, device = device), lambda s: s.repeat_interleave(M // 2, 0).repeat_interleave(N // 2, 1)),
    ]
    for s, expand in cases:
        out = F._dequantize_expert_slice(q, s, torch.float16)
        assert torch.equal(out, _reference(q, expand(s), torch.float16))


@pytest.mark.parametrize("device", DEVICES)
def test_fp8_source_untouched(device):
    q = _fp8((E, M, N), device)
    before = q.view(torch.uint8).clone()
    F._dequantize_full_expert_weights_vectorized(q, torch.full((E, 1, 1), 1e-7, device = device), torch.float16)
    F._dequantize_expert_slice(q[0], torch.full((1,), 1e-7, device = device), torch.float16)
    assert torch.equal(q.view(torch.uint8), before)
