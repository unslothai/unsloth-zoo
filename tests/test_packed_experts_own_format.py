# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
"""A packed expert stack of a format this module does not know (a subclass with its own `_decode`, as Unsloth's
GGUF stacks are) runs forward and dX through the chunked dequantize + grouped GEMM, bitwise vs its dense value."""

import pytest
import torch

import unsloth_zoo.packed_experts as pe
from unsloth_zoo.mxfp4_stacked_experts import _grouped_mm

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class _ScaledInt8ExpertParam(pe._PackedExpertParam):
    """Toy format: int8 codes (E, N, K) times one fp32 scale per row, held in `packed_scales` (E, N)."""

    packed_format = "toy_int8"

    def packed_dims(self):
        return tuple(self.data.shape)

    def _decode(self, codes, scales, glob, dtype, transpose, token_counts, out = None):
        dense = (codes.view(torch.int8).to(torch.float32) * scales[..., None]).to(dtype)
        return dense.transpose(-2, -1).contiguous() if transpose else dense


def _toy(E, N, K, device):
    codes = torch.randint(-127, 128, (E, N, K), dtype = torch.int8, device = device).view(torch.uint8)
    scales = torch.rand(E, N, device = device) * 0.01
    return _ScaledInt8ExpertParam(codes, packed_scales = scales)


def test_flag_is_published():
    assert pe.PACKED_EXPERT_OWN_GROUPED_LINEAR is True


@pytest.mark.parametrize("transpose_b", [True, False])
@pytest.mark.parametrize("device", DEVICES)
def test_own_format_forward_and_dx_bitwise_vs_dense(transpose_b, device):
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    E, N, K = 4, 64, 96
    param = _toy(E, N, K, device)
    dense = param.dequantize(dtype, transpose = False)
    weight = dense.transpose(-1, -2) if transpose_b else dense
    counts = torch.tensor([5, 0, 7, 3], device = device)
    x = torch.randn((int(counts.sum()), weight.shape[1]), device = device, dtype = dtype, requires_grad = True)
    x2 = x.detach().clone().requires_grad_(True)
    before = pe.PACKED_EXPERT_STATS["calls"]
    out = pe.packed_expert_grouped_linear(x, param, counts, transpose_b)
    assert pe.PACKED_EXPERT_STATS["calls"] == before + 1
    ref = _grouped_mm(x2, weight, torch.cumsum(counts, 0, dtype = torch.int32))
    assert torch.equal(out, ref)
    g = torch.randn_like(out)
    out.backward(g)
    ref.backward(g)
    assert torch.equal(x.grad, x2.grad)
    assert param.grad is None


def test_own_format_multi_chunk_matches_single_chunk(monkeypatch):
    import unsloth_zoo.mxfp4_stacked_experts as mse

    param = _toy(4, 32, 64, "cpu")
    counts = torch.tensor([2, 3, 0, 4])
    x = torch.randn(9, 64)
    one = pe.packed_expert_grouped_linear(x, param, counts, True)
    monkeypatch.setattr(mse, "_chunk_bytes", lambda: 32 * 64 * 4)  # one expert per chunk
    many = pe.packed_expert_grouped_linear(x, param, counts, True)
    assert torch.equal(one, many)
