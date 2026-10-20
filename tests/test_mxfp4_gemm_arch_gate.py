# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""The fused MXFP4 tile GEMM is sm_80+ only; below (T4) only decode rows take the grouped GEMV."""

import types

import pytest
import torch

import unsloth_zoo.mxfp4_gemm as mg
import unsloth_zoo.mxfp4_stacked_experts as mse
from unsloth_zoo.mxfp4_dequant import Mxfp4ExpertParam


@pytest.fixture
def gpu(monkeypatch):
    """A verified CUDA device 0 of a chosen capability, no GPU needed."""
    def make(capability, hip = None):
        monkeypatch.setattr(mg, "_TILES_FAST", {})
        monkeypatch.setattr(mg, "_AVAILABLE_DTYPE", {
            (str(torch.device("cuda", 0)), dtype): True for dtype in (torch.float16, torch.bfloat16)
        })
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: capability)
        monkeypatch.setattr(torch.version, "hip", hip)
        monkeypatch.delenv("UNSLOTH_MXFP4_FUSED_GEMM", raising = False)
        return torch.device("cuda", 0)
    return make


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("capability,fused", [((7, 5), False), ((8, 0), True), ((8, 6), True), ((8, 9), True),
                                              ((9, 0), True), ((10, 0), True), ((12, 0), True)])
def test_tile_gemm_needs_sm80(gpu, capability, fused, dtype):
    device = gpu(capability)
    assert mg.mxfp4_gemm_available(device, dtype) is fused
    assert mg.mxfp4_grouped_gemv_available(device, dtype)


def test_sm75_opt_in_and_off_switch(gpu, monkeypatch):
    device = gpu((7, 5))
    monkeypatch.setenv("UNSLOTH_MXFP4_FUSED_GEMM", "1")
    assert mg.mxfp4_gemm_available(device, torch.float16)
    for mode in ("0", "dequant"):
        monkeypatch.setenv("UNSLOTH_MXFP4_FUSED_GEMM", mode)
        assert not mg.mxfp4_gemm_available(device, torch.float16)
        assert not mg.mxfp4_grouped_gemv_available(device, torch.float16)


def test_rocm_is_not_gated_by_cuda_capability(gpu):
    # ROCm reports gfx arch numbers as the capability; the CUDA mma.sync floor does not apply there.
    assert mg.mxfp4_gemm_available(gpu((7, 5), hip = "6.4"), torch.bfloat16)


def _expert_param(E = 32, R = 64, G = 4):
    return Mxfp4ExpertParam(
        torch.zeros(E, R, G, 16, dtype = torch.uint8), mxfp4_scales = torch.zeros(E, R, G, dtype = torch.uint8),
    )


def _rows(n, device, dtype = torch.float16):
    return types.SimpleNamespace(is_cuda = True, device = device, dtype = dtype, shape = (n, 128))


@pytest.mark.parametrize("rows", [1, 4, 16, 17, 64, 2048])
def test_stacked_experts_on_sm75_fuse_only_decode_rows(gpu, rows):
    device = gpu((7, 5))
    param = _expert_param(E = 32)
    # Forward (transpose=True): the GEMV covers 2 * rows <= experts; everything else dequantizes.
    assert mse._fused(_rows(rows, device), param, True) is (2 * rows <= 32)
    # dX (transpose=False) is always the tile kernel.
    assert not mse._fused(_rows(rows, device), param, False)


@pytest.mark.parametrize("rows", [1, 17, 2048])
def test_stacked_experts_on_sm80_fuse_every_shape(gpu, rows):
    device = gpu((8, 0))
    param = _expert_param(E = 32)
    assert mse._fused(_rows(rows, device), param, True)
    assert mse._fused(_rows(rows, device), param, False)
