# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""A once-per-process fast-path probe that hits an OOM must not cache "unavailable": the OOM is transient,
and caching it kept the slow path for the rest of the run. CPU only: the probes are driven by stand-ins."""

import contextlib

import pytest
import torch

import unsloth_zoo.fp16_emulation as fe
import unsloth_zoo.mxfp4_gemm as mg
import unsloth_zoo.temporary_patches.moe_fp8_grouped as mfg


def _oom_once(result):
    calls = []

    def probe(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 MiB")
        return result

    return probe, calls


def test_fp16_out_dtype_probe_oom_is_not_cached(monkeypatch):
    monkeypatch.setattr(fe, "_HAS_OUT_DTYPE", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    real_zeros = torch.zeros
    monkeypatch.setattr(torch, "zeros", lambda *a, device = None, **k: real_zeros(*a, **k))
    mm, calls = _oom_once(None)
    monkeypatch.setattr(torch, "mm", mm)
    assert fe._has_out_dtype() is False
    assert fe._has_out_dtype() is True
    assert fe._has_out_dtype() is True and len(calls) == 2


def test_fp16_out_dtype_real_refusal_is_cached(monkeypatch):
    monkeypatch.setattr(fe, "_HAS_OUT_DTYPE", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    real_zeros = torch.zeros
    monkeypatch.setattr(torch, "zeros", lambda *a, device = None, **k: real_zeros(*a, **k))

    def mm(*a, **k):
        raise TypeError("mm() got an unexpected keyword argument 'out_dtype'")

    monkeypatch.setattr(torch, "mm", mm)
    assert fe._has_out_dtype() is False and fe._HAS_OUT_DTYPE is False


def test_mxfp4_probe_oom_is_not_cached(monkeypatch):
    monkeypatch.setattr(mg, "_AVAILABLE", {})
    monkeypatch.setattr(mg, "_HAS_TRITON", True)
    monkeypatch.setattr(mg, "_triton_too_old", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: contextlib.nullcontext())
    probe, calls = _oom_once(True)
    monkeypatch.setattr(mg, "_probe", probe)
    device = torch.device("cuda", 0)
    assert mg.mxfp4_grouped_mm_available(device) is False
    assert str(device) not in mg._AVAILABLE
    assert mg.mxfp4_grouped_mm_available(device) is True
    assert mg.mxfp4_grouped_mm_available(device) is True and len(calls) == 2


def test_fp8_w8a16_probe_oom_is_not_cached(monkeypatch):
    if mfg.triton is None or mfg._FP8_DTYPE is None:
        pytest.skip("needs Triton and float8_e4m3fn")
    monkeypatch.setattr(mfg, "_AVAILABLE", {})
    monkeypatch.setattr(mfg, "fp8_fused_mode", lambda: "1")
    monkeypatch.setattr(mfg, "_capability", lambda device: (9, 0))
    monkeypatch.setattr(mfg, "_triton_too_old", lambda capability: False)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: contextlib.nullcontext())
    probe, calls = _oom_once(True)
    monkeypatch.setattr(mfg, "_probe", probe)
    device = torch.device("cuda", 0)
    assert mfg.fp8_fused_enabled(device) is False
    assert str(device) not in mfg._AVAILABLE
    assert mfg.fp8_fused_enabled(device) is True
    assert mfg.fp8_fused_enabled(device) is True and len(calls) == 2


def test_fp8_w8a16_launch_pins_fp_fusion(monkeypatch):
    # Triton 3.10 warns when a launch leaves enable_fp_fusion unset; every other Unsloth launch pins it.
    seen = {}

    class _Launch:
        def __getitem__(self, grid):
            def run(*args, **kwargs):
                seen.update(kwargs)
            return run

    E, N, K, M = 2, 128, 128, 4
    w = torch.zeros(E, N, K, dtype = torch.uint8)
    s = torch.ones(E, 1, 1)
    x = torch.zeros(M, K)
    out = torch.zeros(M, N)
    counts = torch.tensor([2, 2], dtype = torch.int32)
    monkeypatch.setattr(mfg, "_config", lambda *args: (64, 64, 128, 4, 3, 0))
    mfg._launch(x, w, s, counts, out, True, (128, 128), lambda kernel: _Launch())
    assert seen.get("enable_fp_fusion") is True
