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

"""_ogs_dense_chunks halves its expert chunk when the MXFP4 decode runs out of memory (T4 gpt-oss-20b LoRA OOM)."""

import pytest
import torch

import unsloth_zoo.mxfp4_dequant as mxfp4_dequant
import unsloth_zoo.temporary_patches.gpt_oss as gpt_oss


class _Packed:
    """Stands in for a swizzled triton_kernels weight: only its (E, K, N) shape is read."""

    def __init__(self, shape):
        self.shape = shape


class _Backend:
    def unswizzle(self, x):
        return x

    def weight_scale(self, precision_config):
        return precision_config


def _setup(monkeypatch, max_experts):
    E, N, K = 8, 4, 64
    gen = torch.Generator().manual_seed(0)
    blocks = torch.randint(0, 256, (E, N, K // 2), dtype = torch.uint8, generator = gen)
    scales = torch.randint(120, 130, (E, N, K // 32), dtype = torch.uint8, generator = gen)
    weight = _Packed((E, K, N))
    monkeypatch.setattr(gpt_oss, "_ogs_backend_for", lambda w: _Backend())
    monkeypatch.setattr(gpt_oss, "_ogs_chunk_bytes", lambda: 1 << 30)
    calls = []

    def decode(b, s, dtype, transpose):
        calls.append(b.shape[0])
        if b.shape[0] > max_experts:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory.")
        return mxfp4_dequant.mxfp4_dequantize_torch(b, s, dtype = dtype, transpose = transpose)

    monkeypatch.setattr(mxfp4_dequant, "mxfp4_dequantize", decode)
    weight_t, scales_t = blocks.transpose(-2, -1), scales.transpose(-2, -1)
    return weight, weight_t, scales_t, blocks, scales, calls


def test_chunk_shrinks_on_oom(monkeypatch):
    weight, weight_t, scales_t, blocks, scales, calls = _setup(monkeypatch, max_experts = 3)
    monkeypatch.setattr(_Backend, "unswizzle", lambda self, x: weight_t if x is weight else x)
    chunks = list(gpt_oss._ogs_dense_chunks(weight, scales_t, torch.float32))
    assert [(a, b) for a, b, _ in chunks] == [(0, 2), (2, 4), (4, 6), (6, 8)]
    dense = torch.cat([d for _, _, d in chunks])
    E, N, Kh = blocks.shape
    ref = mxfp4_dequant.mxfp4_dequantize_torch(
        blocks.view(E, N, Kh // 16, 16), scales, dtype = torch.float32, transpose = True,
    )
    torch.testing.assert_close(dense, ref, rtol = 0, atol = 0)
    assert calls[:3] == [8, 4, 2]


def test_single_expert_oom_still_raises(monkeypatch):
    weight, weight_t, scales_t, *_ = _setup(monkeypatch, max_experts = 0)
    monkeypatch.setattr(_Backend, "unswizzle", lambda self, x: weight_t if x is weight else x)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        list(gpt_oss._ogs_dense_chunks(weight, scales_t, torch.float32))
