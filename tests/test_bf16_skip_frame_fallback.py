# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""bf16 on a device without native bf16 must run eager, not abort the call.

Inductor raises SkipFrame("BF16 is not supported") for any bf16 GPU graph input on
sm < 8 (`_check_triton_bf16_support`). Under `fullgraph = True`, torch >= 2.13
turns the skipped frame into "torch.compile with fullgraph=True found no compiled
frames" after the body ran. Hit on a T4 by Gemma4's vision-tower RMSNorm inside
vLLM (bf16 norm weights from the checkpoint's sub-config, float16 engine).

CPU stand-in: Inductor's own check is made to refuse bf16 regardless of device,
and the device probe answers "lacks bf16" for CPU.
"""
from __future__ import annotations

import pytest
import torch

from unsloth_zoo.temporary_patches import utils as patch_utils


def _kernel(x, w):
    return (x.to(torch.float32) * w.to(torch.float32)).to(torch.float16)


@pytest.fixture
def bf16_refusing_inductor(monkeypatch):
    from torch._dynamo.exc import SkipFrame
    import torch._inductor.compile_fx as compile_fx

    def refuse(graph):
        from torch._inductor.ir import IRNode
        for node in (*graph.graph_inputs.values(), *graph.graph_outputs):
            if isinstance(node, IRNode) and node.get_dtype() == torch.bfloat16:
                raise SkipFrame("BF16 is not supported")

    monkeypatch.setattr(compile_fx, "_check_triton_bf16_support", refuse)
    monkeypatch.setattr(
        patch_utils, "_device_lacks_native_bf16", lambda device: device.type == "cpu")
    monkeypatch.setattr(patch_utils, "_ANY_CUDA_LACKS_BF16", True)
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def test_bf16_input_runs_eager_instead_of_raising(bf16_refusing_inductor):
    fn = patch_utils.compile_with_eager_fallback(_kernel, "test bf16 skip")
    x = torch.randn(4, 8, dtype = torch.float16)
    w = torch.randn(8, dtype = torch.bfloat16)
    out = fn(x, w)
    assert out.dtype is torch.float16
    torch.testing.assert_close(out, _kernel(x, w), rtol = 0, atol = 0)
    # Not latched: a float16 call still goes through the compiled region.
    assert patch_utils.eager_fallback_state()["test bf16 skip"] is False
    out = fn(x, w.to(torch.float16))
    torch.testing.assert_close(out, _kernel(x, w.to(torch.float16)), rtol = 0, atol = 0)


def test_probe_mirrors_capability(monkeypatch):
    patch_utils._CUDA_NATIVE_BF16.clear()
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (7, 5))
    assert patch_utils._device_lacks_native_bf16(torch.device("cuda", 0))
    patch_utils._CUDA_NATIVE_BF16.clear()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (8, 0))
    assert not patch_utils._device_lacks_native_bf16(torch.device("cuda", 0))
    assert not patch_utils._device_lacks_native_bf16(torch.device("cpu"))
    patch_utils._CUDA_NATIVE_BF16.clear()


def test_bf16_capable_hosts_skip_the_scan(monkeypatch):
    from unsloth_zoo.temporary_patches import utils as patch_utils
    monkeypatch.setattr(patch_utils, "_ANY_CUDA_LACKS_BF16", False)
    calls = []
    monkeypatch.setattr(patch_utils, "_device_lacks_native_bf16", lambda d: calls.append(d) or True)
    assert patch_utils._inductor_would_skip_bf16((torch.zeros(2, dtype = torch.bfloat16),), {}) is False
    assert calls == []
