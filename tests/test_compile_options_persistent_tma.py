# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""`triton.enable_persistent_tma_matmul` is only requested on torch < 2.12.

From torch 2.12 Inductor guards every GEMM size when it considers its TMA templates, so a compiled
torch._scaled_mm (rowwise FP8 linears) recompiled for each new token count on sm90+.
"""

import subprocess
import sys
import textwrap

import pytest
import torch

from unsloth_zoo.temporary_patches import common

KEY = "triton.enable_persistent_tma_matmul"


@pytest.mark.parametrize(
    "version, ok",
    [
        ("2.10.0+cu128", True),
        ("2.11.0", True),
        ("2.11.1+rocm7.1", True),
        ("2.12.0.dev20260401+cu130", False),
        ("2.12.0", False),
        ("2.14.1+cu130", False),
        ("3.0.0", False),
        ("garbage", False),
    ],
)
def test_gate_follows_the_torch_version(monkeypatch, version, ok):
    monkeypatch.setattr(torch, "__version__", version)
    assert common._persistent_tma_matmul_ok() is ok


def test_options_carry_the_gate():
    options = common.get_torch_compile_options()
    if not all(part in common.inductor_config_source for part in KEY.split(".")):
        pytest.skip("this Inductor has no persistent TMA option")
    assert options[KEY] is common._persistent_tma_matmul_ok()


def _has_tma_scaled_mm():
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    return torch.cuda.get_device_capability() >= (9, 0) and hasattr(torch, "float8_e4m3fn")


@pytest.mark.skipif(not _has_tma_scaled_mm(), reason = "needs an sm90+ NVIDIA GPU")
def test_compiled_scaled_mm_does_not_recompile_per_token_count():
    # Fresh interpreter: Dynamo caches and counters are process-wide.
    code = textwrap.dedent(
        """
        import torch
        from torch._dynamo.utils import counters
        from unsloth_zoo.temporary_patches.common import torch_compile

        w = (torch.randn(1024, 512, device = "cuda") * 0.1).to(torch.float8_e4m3fn)
        w_scale = torch.rand(1024, 1, device = "cuda") + 0.5

        @torch_compile
        def f(x):
            scale = x.abs().amax(-1, keepdim = True).float().clamp(min = 1e-12) / 448.0
            xq = (x / scale).to(torch.float8_e4m3fn)
            return torch._scaled_mm(xq, w.t(), scale_a = scale, scale_b = w_scale.t(), out_dtype = torch.bfloat16)

        for m in (512, 768, 640, 896, 384, 1000, 272):
            f(torch.randn(m, 512, device = "cuda", dtype = torch.bfloat16))
        torch.cuda.synchronize()
        print("GRAPHS", counters["stats"]["unique_graphs"])
        """
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output = True, text = True, timeout = 900)
    line = [l for l in out.stdout.splitlines() if l.startswith("GRAPHS ")]
    assert out.returncode == 0 and line, out.stdout[-2000:] + out.stderr[-4000:]
    # One static graph, then one dynamic one: not a graph per token count (0 = compile fell back to eager).
    assert 1 <= int(line[-1].split()[1]) <= 2, line[-1]
