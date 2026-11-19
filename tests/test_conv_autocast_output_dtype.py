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

"""Unsloth's conv dtype wrapper must return what stock conv returns under autocast (autocast's dtype), not a float32
copy of a 16-bit result: float32 input_features (Whisper, Qwen3-ASR) otherwise doubled the frozen conv stem's memory."""

import pytest
import torch

from unsloth_zoo.compiler import _dtype_safe_forward

_CONVS = [
    (torch.nn.Conv1d, (2, 3, 16)),
    (torch.nn.Conv2d, (2, 3, 8, 8)),
    (torch.nn.Conv3d, (2, 3, 4, 4, 4)),
    (torch.nn.ConvTranspose1d, (2, 3, 16)),
    (torch.nn.ConvTranspose2d, (2, 3, 8, 8)),
]


def _wrapped(cls, weight_dtype):
    torch.manual_seed(0)
    conv = cls(3, 4, 3).to(weight_dtype)
    stock = cls.forward
    forward = _dtype_safe_forward(stock, True, False)
    return conv, stock, forward


@pytest.mark.parametrize("cls,shape", _CONVS)
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_autocast_output_matches_stock_dtype_and_values(cls, shape, weight_dtype):
    conv, stock, forward = _wrapped(cls, weight_dtype)
    x = torch.randn(*shape)
    with torch.no_grad(), torch.autocast("cpu", dtype = torch.bfloat16):
        expected = stock(conv, x)
        got = forward(conv, x)
    assert expected.dtype == torch.bfloat16
    assert got.dtype == torch.bfloat16
    assert torch.equal(got, expected)


@pytest.mark.parametrize("cls,shape", _CONVS)
def test_without_autocast_output_follows_input_dtype(cls, shape):
    # bf16 weights, float32 input and no autocast: stock raises, the wrapper computes in bf16 and returns float32
    conv, _, forward = _wrapped(cls, torch.bfloat16)
    with torch.no_grad():
        assert forward(conv, torch.randn(*shape)).dtype == torch.float32
        assert forward(conv, torch.randn(*shape, dtype = torch.bfloat16)).dtype == torch.bfloat16


def test_fp16_autocast_with_bf16_weight_returns_fp16():
    # PR 552's case: a bf16 weight under fp16 autocast must not crash and stays fp16
    conv, _, forward = _wrapped(torch.nn.Conv1d, torch.bfloat16)
    with torch.no_grad(), torch.autocast("cpu", dtype = torch.float16):
        assert forward(conv, torch.randn(2, 3, 16, dtype = torch.float16)).dtype == torch.float16
