# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""Fused MXFP4 GEMM: decode bits match mxfp4_dequant, products match an fp32 reference, experts stay packed."""

import pytest
import torch
import torch.nn as nn

import unsloth_zoo.mxfp4_gemm as mg
from unsloth_zoo.mxfp4_dequant import mxfp4_dequantize_torch
from unsloth_zoo.mxfp4_gemm import mxfp4_gemm_available, mxfp4_grouped_matmul, mxfp4_matmul

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and mxfp4_gemm_available(torch.device("cuda", 0))),
    reason = "needs a CUDA device with Triton",
)
DEV = "cuda"


def _stack(E, N, G, low = 118, high = 126, seed = 0):
    g = torch.Generator(device = DEV).manual_seed(seed)
    blocks = torch.randint(0, 256, (E, N, G, 16), dtype = torch.uint8, device = DEV, generator = g)
    scales = torch.randint(low, high, (E, N, G), dtype = torch.uint8, device = DEV, generator = g)
    return blocks, scales


def _small_nibbles_where_large_scale(blocks, scales):
    # Keep every decoded value finite: rows scaled by >= 2^127 hold only 0 / +-0.5.
    small = torch.tensor([0, 1, 8, 9], dtype = torch.uint8, device = DEV)
    pick = torch.randint(0, 4, (2, *blocks.shape), device = DEV)
    return torch.where(scales[..., None] > 126, small[pick[0]] | (small[pick[1]] << 4), blocks)


@pytest.mark.parametrize("scale_edges", [(0, 1, 127, 128), (0, 1, 127, 200, 254, 255)])
@pytest.mark.parametrize("asm", [True, False])
def test_decode_bits_match_the_dequant_reference(scale_edges, asm, monkeypatch):
    """Identity input: every output is one exact product, so the fused decode must equal mxfp4_dequant bit for bit."""
    if not asm:
        monkeypatch.setattr(mg, "_DEVICE_INFO", {k: (v[0], v[1], False) for k, v in mg._DEVICE_INFO.items()})
    blocks, scales = _stack(1, 96, 5, low = 0, high = max(scale_edges) + 1)
    scales[0, : len(scale_edges), 0] = torch.tensor(scale_edges, dtype = torch.uint8, device = DEV)
    blocks = _small_nibbles_where_large_scale(blocks, scales)
    want = mxfp4_dequantize_torch(blocks[0], scales[0], dtype = torch.bfloat16).float()
    fwd = mxfp4_matmul(torch.eye(160, device = DEV, dtype = torch.bfloat16), blocks[0], scales[0])
    bwd = mxfp4_matmul(torch.eye(96, device = DEV, dtype = torch.bfloat16), blocks[0], scales[0], trans = True)
    assert torch.equal(fwd.float().t(), want)
    assert torch.equal(bwd.float(), want)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows", [1, 7, 16, 33, 64, 130])
@pytest.mark.parametrize("trans", [False, True])
def test_dense_matmul_close_to_fp32(rows, trans, dtype):
    blocks, scales = _stack(1, 160, 6)
    w = mxfp4_dequantize_torch(blocks[0], scales[0], dtype = torch.float32)
    x = torch.randn(rows, 160 if trans else 192, device = DEV).to(dtype)
    bias = None if trans else torch.randn(160, device = DEV).to(dtype)
    got = mxfp4_matmul(x, blocks[0], scales[0], trans = trans, bias = bias).float()
    want = x.float() @ (w if trans else w.t())
    if bias is not None:
        want = want + bias.float()
    torch.testing.assert_close(got, want, atol = 2e-2 * want.abs().max().item(), rtol = 2e-2)
    out = torch.empty(rows, got.shape[-1], device = DEV, dtype = dtype)
    same = mxfp4_matmul(x, blocks[0], scales[0], trans = trans, bias = bias, out = out)
    assert same.data_ptr() == out.data_ptr() and torch.equal(same.float(), got)


@pytest.mark.parametrize("counts", [[5, 0, 37, 1, 0, 200], [0, 0, 0, 3, 0, 0], [1, 1, 1, 1, 1, 1], [0] * 6])
@pytest.mark.parametrize("trans", [False, True])
def test_grouped_matmul_per_expert_close_to_fp32(counts, trans):
    blocks, scales = _stack(6, 64, 3, seed = 1)
    w = mxfp4_dequantize_torch(blocks, scales, dtype = torch.float32)
    x = torch.randn(sum(counts), 64 if trans else 96, device = DEV).to(torch.bfloat16)
    got = mxfp4_grouped_matmul(x, blocks, scales, torch.tensor(counts, device = DEV), trans = trans).float()
    assert got.shape == (sum(counts), 96 if trans else 64)
    start = 0
    for e, c in enumerate(counts):
        if c == 0:
            continue
        want = x[start : start + c].float() @ (w[e] if trans else w[e].t())
        torch.testing.assert_close(got[start : start + c], want, atol = 2e-2 * max(1.0, want.abs().max().item()), rtol = 2e-2)
        start += c


def test_stacked_experts_fused_matches_the_chunked_decode_path(monkeypatch):
    from unsloth_zoo.mxfp4_stacked_experts import Mxfp4StackedExperts

    E, H, I, T, k = 12, 128, 96, 40, 3
    experts = Mxfp4StackedExperts(E, H, I, nn.SiLU(), False, device = DEV)
    for name in ("gate_up", "down"):
        b, s = _stack(E, *getattr(experts, f"{name}_scales").shape[1:], seed = 3 if name == "down" else 2)
        getattr(experts, f"{name}_blocks").data = b
        getattr(experts, f"{name}_scales").data = s
    experts.finalize()
    x = torch.randn(T, H, device = DEV, dtype = torch.bfloat16)
    idx = torch.stack([torch.randperm(E - 2, device = DEV)[:k] for _ in range(T)])  # two experts never routed
    w = torch.softmax(torch.randn(T, k, device = DEV), -1)
    results = []
    for enabled in (True, False):
        monkeypatch.setattr(mg, "ENABLED", enabled)
        xa = x.clone().requires_grad_()
        y = experts(xa, idx, w)
        y.float().square().sum().backward()
        results.append((y.float(), xa.grad.float()))
    (yf, gf), (yc, gc) = results
    torch.testing.assert_close(yf, yc, atol = 2e-2 * yc.abs().max().item(), rtol = 2e-2)
    torch.testing.assert_close(gf, gc, atol = 2e-2 * gc.abs().max().item(), rtol = 2e-2)
