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


@pytest.fixture(autouse = True)
def _fresh_trace_answers(monkeypatch):
    # A traced test on a real GPU caches a passing in-trace probe that a later test's eager call would reuse.
    if hasattr(mg, "_TRACE_AVAILABLE"):
        monkeypatch.setattr(mg, "_TRACE_AVAILABLE", {})


@pytest.fixture
def gpu(monkeypatch):
    """A verified CUDA device 0 of a chosen capability, no GPU needed."""
    def make(capability, hip = None):
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


@pytest.mark.parametrize("device", [None, "cuda"])
def test_unindexed_device_resolves_the_current_gpu(gpu, monkeypatch, device):
    # Mixed GPUs: the arch answer for cuda:0 (sm80) must not be reused once cuda:1 (sm75) is current.
    gpu((8, 0))
    caps = {0: (8, 0), 1: (7, 5)}
    current = [0]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current[0])
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d = None: caps[torch.device(d).index])
    monkeypatch.setitem(mg._AVAILABLE_DTYPE, (str(torch.device("cuda", 1)), torch.float16), True)
    assert mg.mxfp4_gemm_available(device, torch.float16)
    current[0] = 1
    assert not mg.mxfp4_gemm_available(device, torch.float16)


@pytest.mark.parametrize("device", [None, "cuda"])
def test_no_cuda_answers_false_instead_of_raising(monkeypatch, device):
    monkeypatch.setattr(mg, "_AVAILABLE_DTYPE", {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: pytest.fail("resolved a GPU without CUDA"))
    monkeypatch.setattr(mg, "mxfp4_grouped_mm_available", lambda device = None: False)
    monkeypatch.delenv("UNSLOTH_MXFP4_FUSED_GEMM", raising = False)
    assert not mg.mxfp4_gemm_available(device, torch.bfloat16)
    assert not mg.mxfp4_grouped_gemv_available(device, torch.bfloat16)


@pytest.mark.parametrize("device", [None, "cuda"])
def test_unindexed_device_resolves_the_current_gpu_for_the_gemv_probe(monkeypatch, device):
    # The self-check result of cuda:0 must not answer for cuda:1.
    monkeypatch.setattr(mg, "_AVAILABLE_DTYPE", {
        (str(torch.device("cuda", 0)), torch.float16): True,
        (str(torch.device("cuda", 1)), torch.float16): False,
    })
    current = [0]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current[0])
    monkeypatch.delenv("UNSLOTH_MXFP4_FUSED_GEMM", raising = False)
    assert mg.mxfp4_grouped_gemv_available(device, torch.float16)
    current[0] = 1
    assert not mg.mxfp4_grouped_gemv_available(device, torch.float16)


@pytest.mark.parametrize("mode", ["compiling", "capturing"])
@pytest.mark.parametrize("rows", [17, 2048])
def test_stacked_experts_on_sm75_keep_the_device_schedule_when_traced(gpu, monkeypatch, mode, rows):
    # The dequantize fallback reads routing offsets on the host (.tolist()): a graph break, a fullgraph
    # error and a CUDA graph capture error, so compiled and captured calls keep the device-scheduled kernel.
    device = gpu((7, 5))
    param = _expert_param(E = 32)
    assert not mse._fused(_rows(rows, device), param, True)
    assert not mse._fused(_rows(rows, device), param, False)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: mode == "compiling")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: mode == "capturing")
    assert mse._fused(_rows(rows, device), param, True)
    assert mse._fused(_rows(rows, device), param, False)


def test_stacked_experts_traced_on_sm75_still_honour_the_off_switch(gpu, monkeypatch):
    device = gpu((7, 5))
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setenv("UNSLOTH_MXFP4_FUSED_GEMM", "0")
    assert not mse._fused(_rows(2048, device), _expert_param(E = 32), False)


@pytest.mark.parametrize("capability", [(8, 0), (9, 0), (10, 0), (12, 0)])
@pytest.mark.parametrize("traced", [False, True])
@pytest.mark.parametrize("rows", [1, 17, 2048])
def test_sm80_and_newer_route_the_same_traced_or_not(gpu, monkeypatch, capability, traced, rows):
    device = gpu(capability)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: traced)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: traced)
    for transpose in (True, False):
        assert mse._fused(_rows(rows, device), _expert_param(E = 32), transpose)


@pytest.mark.parametrize("operand,current", [(1, 0), (0, 1), (1, 1)])
def test_stacked_experts_on_sm75_see_a_capture_on_the_operand_gpu(gpu, monkeypatch, operand, current):
    gpu((7, 5))
    monkeypatch.setattr(mg, "_AVAILABLE_DTYPE", {
        (str(torch.device("cuda", i)), dtype): True for i in (0, 1) for dtype in (torch.float16, torch.bfloat16)
    })
    state = {"current": current, "capturing": {operand}}

    class _device:
        def __init__(self, device):
            self.index = torch.device(device).index

        def __enter__(self):
            self.prev, state["current"] = state["current"], self.index

        def __exit__(self, *exc):
            state["current"] = self.prev

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", _device)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: state["current"])
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: state["current"] in state["capturing"])
    param = _expert_param(E = 32)
    inputs = _rows(2048, torch.device("cuda", operand))
    assert mse._fused(inputs, param, True)
    assert mse._fused(inputs, param, False)
    assert state["current"] == current
    state["capturing"] = set()
    assert not mse._fused(inputs, param, True)
    assert not mse._fused(inputs, param, False)
