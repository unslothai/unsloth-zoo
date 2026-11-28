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


"""CPU checks of the GGUF kernel routes ROCm takes (measured on gfx1151, Linux and Windows): fp16 dequantize decodes
to float32 in the kernel and rounds in torch (the kernel's fp16 store writes -0.0 as +0.0 there), and the one-row
Q4_1 GEMV uses the flat layout (the split one is wrong there). NVIDIA keeps both routes as they were."""

import contextlib
import types

import pytest
import torch

from unsloth_zoo import rocm_build
from unsloth_zoo.gguf_kernels import GGMLType as T
from unsloth_zoo.gguf_kernels import ops as O
from unsloth_zoo.gguf_kernels.testing import random_packed


@pytest.fixture(params = [True, False], ids = ["rocm", "cuda"])
def rocm(request, monkeypatch):
    monkeypatch.setattr(rocm_build, "is_rocm_torch", lambda torch_mod = None: request.param)
    return request.param


def test_q4_1_gemv_layout(rocm):
    assert O._gemv_split(T.Q4_1) is (not rocm)
    for t in (T.Q4_0, T.Q5_0, T.Q5_1, T.Q8_0, T.Q4_K, T.Q6_K, T.IQ4_XS):
        assert O._gemv_split(t) is True, t


def test_no_split_list_still_wins(rocm, monkeypatch):
    monkeypatch.setattr(O, "_NO_SPLIT_GEMV", frozenset((T.Q4_0, T.Q4_1)))
    assert O._gemv_split(T.Q4_0) is False and O._gemv_split(T.Q4_1) is False


class _Launch:
    def __init__(self):
        self.out_dtypes = []

    def __getitem__(self, grid):
        def run(packed, out, *args, **kwargs):
            self.out_dtypes.append(out.dtype)
            out.fill_(-0.0)
        return run


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32], ids = str)
def test_fp16_dequant_store_dtype(rocm, dtype, monkeypatch):
    launch = _Launch()
    monkeypatch.setattr(O, "_tk", types.SimpleNamespace(dequantize_kernel = launch))
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    packed = random_packed(T.Q4_0, 4, 64)
    out = O._dequant_triton(packed, T.Q4_0, (4, 64), dtype)
    assert out.dtype == dtype and out.shape == (4, 64)
    assert launch.out_dtypes == [torch.float32 if (rocm and dtype == torch.float16) else dtype]
    # torch's rounding keeps the sign of zero
    assert torch.signbit(out).all()
