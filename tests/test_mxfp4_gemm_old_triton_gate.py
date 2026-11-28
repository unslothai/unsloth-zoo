# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
"""CPU checks: the fused MXFP4 GEMM stays off sm_100 / sm_120 below Triton 3.3, where compiling tl.dot aborts the
process, and its availability check never reaches the probe there."""

import pytest
import torch

from unsloth_zoo import mxfp4_gemm as mg


@pytest.mark.parametrize("capability, triton_version, ok", [
    ((8, 0), "3.2.0", True), ((8, 9), "3.2.0", True), ((9, 0), "3.2.0", True),
    ((10, 0), "3.2.0", False), ((12, 0), "3.2.0", False), ((10, 0), "3.1.0", False),
    ((10, 0), "3.3.0", True), ((12, 0), "3.6.0", True), ((12, 0), "3.9.0+git1234", True), ((10, 0), "bad", False),
])
def test_triton_lowers_dot(capability, triton_version, ok):
    assert mg._triton_lowers_dot(capability, triton_version) is ok


@pytest.mark.skipif(not mg._HAS_TRITON, reason = "needs triton importable")
@pytest.mark.parametrize("capability, triton_version, probed", [((10, 0), "3.2.0", False), ((12, 0), "3.2.0", False),
                                                                ((10, 0), "3.3.0", True), ((8, 9), "3.2.0", True)])
def test_availability_skips_the_probe_on_old_triton_blackwell(monkeypatch, capability, triton_version, probed):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    monkeypatch.setattr(torch.version, "hip", None)
    # A ROCm wheel still reads as ROCm through its +rocm version tag: fake the CUDA card all the way.
    monkeypatch.setattr(mg, "is_rocm_torch", lambda torch_mod = None: False)
    monkeypatch.setattr(mg.triton, "__version__", triton_version)
    monkeypatch.setattr(mg, "_AVAILABLE", {})
    monkeypatch.setattr(mg, "_probe", lambda device: calls.append(device) or True)
    assert mg.mxfp4_grouped_mm_available(torch.device("cuda", 0)) is probed
    assert bool(calls) is probed
