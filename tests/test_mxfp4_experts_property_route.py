# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""Mxfp4GptOssExperts.gate_up_proj / down_proj properties themselves; needs triton_kernels."""
import os, sys, importlib.util
import pytest, torch

if importlib.util.find_spec("triton_kernels") is None:
    try:
        import vllm, os as _os
        sys.path.append(_os.path.join(_os.path.dirname(vllm.__file__), "third_party"))
    except Exception:
        pass
if importlib.util.find_spec("triton_kernels") is None:
    pytest.skip("no triton_kernels", allow_module_level = True)

import transformers.integrations.mxfp4 as M
from transformers import GptOssConfig
from unsloth_zoo.temporary_patches import gpt_oss as G

_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _ref(blocks, scales):
    lut = torch.tensor(_E2M1, dtype = torch.float64)
    E, R, Gn, B = blocks.shape
    v = torch.empty(E, R, Gn, B * 2, dtype = torch.float64)
    v[..., 0::2] = lut[(blocks & 15).long()]; v[..., 1::2] = lut[(blocks >> 4).long()]
    v = v * torch.pow(2.0, scales.double() - 127).unsqueeze(-1)
    return v.reshape(E, R, Gn * B * 2).transpose(1, 2).to(torch.bfloat16)


@pytest.fixture(scope = "module")
def experts_cls():
    G.patch_gpt_oss()
    cls = M.Mxfp4GptOssExperts
    if cls.__module__ != G.__name__:
        pytest.skip("patch_gpt_oss did not install its Mxfp4GptOssExperts")
    return cls


def _make(cls, seed = 0):
    cfg = GptOssConfig(num_local_experts = 2, intermediate_size = 64, hidden_size = 64, num_hidden_layers = 1)
    ex = cls(cfg)
    g = torch.Generator().manual_seed(seed)
    for p in ("gate_up_proj", "down_proj"):
        b = getattr(ex, p + "_blocks"); s = getattr(ex, p + "_scales")
        b.data = torch.randint(0, 256, b.shape, dtype = torch.uint8, generator = g)
        s.data = torch.randint(120, 134, s.shape, dtype = torch.uint8, generator = g)
    return ex


def test_packed_blocks_decode_through_the_property(experts_cls):
    ex = _make(experts_cls)
    assert torch.equal(ex.gate_up_proj, _ref(ex.gate_up_proj_blocks, ex.gate_up_proj_scales))
    assert torch.equal(ex.down_proj, _ref(ex.down_proj_blocks, ex.down_proj_scales))
    # Not cached: the packed bytes stay the only copy.
    assert "_gate_up_proj" not in ex.__dict__ and "_down_proj" not in ex.__dict__


def test_swizzled_value_set_at_load_wins(experts_cls):
    ex = _make(experts_cls)
    sentinel = torch.zeros(1)
    ex.gate_up_proj = sentinel
    assert ex.gate_up_proj is sentinel


def test_all_zero_blocks_still_raise_attribute_error(experts_cls):
    cfg = GptOssConfig(num_local_experts = 2, intermediate_size = 64, hidden_size = 64, num_hidden_layers = 1)
    ex = experts_cls(cfg)
    assert not hasattr(ex, "gate_up_proj")
    assert not hasattr(ex, "down_proj")


@pytest.mark.parametrize("native", [True, False])
def test_forward_returns_the_input_dtype(experts_cls, monkeypatch, native):
    """The native kernels run in bf16; an fp16 residual (T4) that got bf16 back was promoted to fp32."""
    ex = _make(experts_cls)
    x = torch.randn(3, 64, dtype = torch.float16)
    monkeypatch.setattr(G, "torch_cuda_device", lambda device: __import__("contextlib").nullcontext())
    monkeypatch.setattr(G, "_mxfp4_experts_routing", lambda self, h, *a: (h, None, None, None, None))
    monkeypatch.setattr(G, "_ogs_native_inference_ok", lambda device, backend: native)
    rows = lambda *a, **k: torch.ones(3, 64, dtype = torch.bfloat16)
    monkeypatch.setattr(G, "mxfp4_ogs_experts_forward", rows)
    backend = G._top_level_triton_kernels_backend()
    monkeypatch.setattr(type(backend), "gate_up", lambda self, *a, **k: rows())
    monkeypatch.setattr(type(backend), "down", lambda self, *a, **k: rows())
    out = ex(x, routing_data = None, gather_idx = None, scatter_idx = None)
    assert out.dtype == torch.float16
