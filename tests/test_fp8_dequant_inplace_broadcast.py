# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
"""In-place tiled _fp8_dequant_blockwise is bitwise identical to repeat_interleave and never mutates input."""

import pytest
import torch

from unsloth_zoo.temporary_patches.moe_utils_fp8 import _fp8_dequant_blockwise


def _reference(W_fp8, scale_inv, block_size = None):
    rows, cols = W_fp8.shape
    if scale_inv.dtype == torch.uint8:
        scale_inv = (scale_inv.to(torch.float32) - 127.0).exp2()
    out_dtype = scale_inv.dtype if scale_inv.dtype.is_floating_point else torch.float32
    W = W_fp8.to(out_dtype)
    if scale_inv.numel() == 1:
        return W * scale_inv.reshape(()).to(out_dtype)
    if scale_inv.ndim == 1:
        return W * (scale_inv.view(-1, 1) if scale_inv.shape[0] == rows else scale_inv.view(1, -1)).to(out_dtype)
    srows, scols = scale_inv.shape
    bm = bn = None
    if block_size is not None and len(block_size) == 2:
        if srows == -(-rows // block_size[0]) and scols == -(-cols // block_size[1]):
            bm, bn = block_size
    if bm is None:
        bm, bn = -(-rows // srows), -(-cols // scols)
    scale = scale_inv.to(out_dtype)
    if bm > 1:
        scale = scale.repeat_interleave(bm, dim = 0)[:rows]
    if bn > 1:
        scale = scale.repeat_interleave(bn, dim = 1)[:, :cols]
    return W * scale


def _weight(rows, cols, fp8_dtype, seed = 0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(rows, cols, generator = g) * 50).clamp(-400, 400).to(fp8_dtype)


def _scale(shape, kind, seed = 1):
    g = torch.Generator().manual_seed(seed)
    if kind == "e8m0":
        return torch.randint(110, 140, shape, dtype = torch.uint8, generator = g)
    if kind == "int32":
        return torch.randint(1, 4, shape, dtype = torch.int32, generator = g)
    return (torch.rand(shape, generator = g) * 0.01 + 1e-4).to(kind)


def _same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)


_SCALES = [torch.float32, torch.bfloat16, torch.float16, "e8m0", "int32"]
_GRIDS = [
    ((256, 512), (128, 128)),  # exact
    ((256, 512), (64, 128)),   # exact, non-square block
    ((8, 512), (1, 128)),      # bm == 1
    ((256, 6), (128, 1)),      # bn == 1
    ((64, 96), (1, 32)),       # MXFP8 layout
    ((130, 256), (128, 128)),  # partial row block
    ((256, 257), (128, 128)),  # partial col block
]


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("kind", _SCALES)
@pytest.mark.parametrize("shape,block", _GRIDS)
@pytest.mark.parametrize("pass_block", [True, False])
def test_block_scale_bitwise_matches_reference(fp8_dtype, kind, shape, block, pass_block):
    rows, cols = shape
    W = _weight(rows, cols, fp8_dtype)
    S = _scale((-(-rows // block[0]), -(-cols // block[1])), kind)
    bs = block if pass_block else None
    W_bytes = W.view(torch.uint8).clone()
    assert _same(_fp8_dequant_blockwise(W, S, bs), _reference(W, S, bs))
    assert torch.equal(W.view(torch.uint8), W_bytes)
    Wt = W.t().contiguous().t()
    assert _same(_fp8_dequant_blockwise(Wt, S, bs), _reference(Wt, S, bs))


@pytest.mark.parametrize("kind", [torch.float32, torch.bfloat16, "e8m0"])
def test_dense_and_per_channel_scales(kind):
    W = _weight(96, 64, torch.float8_e4m3fn)
    for shape in [(96,), (64,), (1,), (1, 1), (96, 1), (1, 64), (96, 64)]:
        S = _scale(shape, kind)
        for bs in (None, (128, 128), (1, 64)):
            assert _same(_fp8_dequant_blockwise(W, S, bs), _reference(W, S, bs))


def test_input_already_out_dtype_not_mutated():
    W = torch.randn(256, 256)
    W0 = W.clone()
    S = torch.rand(2, 2)
    out = _fp8_dequant_blockwise(W, S, (128, 128))
    assert out is not W
    assert torch.equal(W, W0)
    assert _same(out, _reference(W0, S, (128, 128)))


def test_fused_expert_slices():
    W3 = (torch.randn(3, 256, 384) * 50).to(torch.float8_e4m3fn)
    S3 = torch.rand(3, 2, 3)
    for e in range(3):
        assert _same(_fp8_dequant_blockwise(W3[e], S3[e], (128, 128)), _reference(W3[e], S3[e], (128, 128)))


def test_special_values_propagate():
    W = torch.tensor([[float("nan"), -0.0, float("inf"), 1.0]] * 2).to(torch.float8_e5m2)
    S = torch.tensor([[0.0, float("inf")]])
    a, b = _fp8_dequant_blockwise(W, S, (2, 2)), _reference(W, S, (2, 2))
    assert torch.equal(a.isnan(), b.isnan())
    assert torch.equal(a.signbit(), b.signbit())
    assert torch.equal(torch.nan_to_num(a), torch.nan_to_num(b))


def test_exact_tiling_skips_full_size_scale(monkeypatch):
    calls = []
    orig = torch.Tensor.repeat_interleave
    monkeypatch.setattr(torch.Tensor, "repeat_interleave", lambda self, *a, **k: calls.append(1) or orig(self, *a, **k))
    _fp8_dequant_blockwise(_weight(256, 256, torch.float8_e4m3fn), torch.rand(2, 2), (128, 128))
    assert calls == []
    _fp8_dequant_blockwise(_weight(130, 256, torch.float8_e4m3fn), torch.rand(2, 2), (128, 128))
    assert calls
