# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""Mxfp4StackedExperts combines expert outputs with the Triton weighted_unpermute: same bits for the output and the
input / LoRA grads as the eager [T, top_k, H] fp32 product, a routing-weight grad, and no fp32 [T, top_k, H] tensor
saved for backward."""

import pytest
import torch

import unsloth_zoo.mxfp4_stacked_experts as mse
from unsloth_zoo.temporary_patches import moe_triton_kernels as K
from unsloth_zoo.temporary_patches import moe_utils as mu

from test_mxfp4_stacked_experts import E, H, I, _dense, _experts, _grouped_reference, _routing

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and K.moe_triton_kernels_available(torch.device("cuda"))),
    reason = "the Triton combine runs on CUDA / ROCm only",
)


def _count_calls(monkeypatch):
    calls = []
    real = getattr(mse, "_weighted_unpermute", None)

    def counting(*args, **kwargs):
        out = real(*args, **kwargs)
        calls.append(out is not None)
        return out

    # raising=False: without the fused combine the counter is never called, so the caller fails.
    monkeypatch.setattr(mse, "_weighted_unpermute", counting, raising = False)
    return calls


def _lora(r = 8, seed = 5):
    g = torch.Generator().manual_seed(seed)
    first = (torch.randn(E, H, r, generator = g) * 0.1).to("cuda", torch.bfloat16).requires_grad_(True)
    second = (torch.randn(E, r, 2 * I, generator = g) * 0.1).to("cuda", torch.bfloat16).requires_grad_(True)
    return first, second


@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("lora", [False, True])
def test_fused_combine_matches_eager_and_trains_the_router(monkeypatch, weight_dtype, lora):
    module, _ = _experts()
    x, idx, weight = _routing(tokens = 64, top_k = 4)
    weight = weight.to(weight_dtype)
    calls = _count_calls(monkeypatch)

    leaves = lambda: (x.detach().clone().requires_grad_(True), weight.detach().clone().requires_grad_(True))
    x1, w1 = leaves()
    adapters = _lora() if lora else None
    if lora:
        setattr(module, mu.moe_lora_stash_name("gate_up_proj"), (*adapters, 2.0, E))
    got = module(x1, idx, w1)
    assert calls == [True], calls
    dy = torch.randn(got.shape, generator = torch.Generator().manual_seed(3)).to("cuda", got.dtype)
    got_grads = torch.autograd.grad(got, [x1, w1, *(adapters or ())], dy)

    x2, w2 = leaves()
    ref_adapters = tuple(a.detach().clone().requires_grad_(True) for a in adapters) if lora else None
    want = _grouped_reference(
        module, x2, idx, w2, *_dense(module), lora = (*ref_adapters, 2.0) if lora else None,
    )
    want_grads = torch.autograd.grad(want, [x2, w2, *(ref_adapters or ())], dy)

    assert torch.equal(got, want)
    assert torch.equal(got_grads[0], want_grads[0])
    for a, b in zip(got_grads[2:], want_grads[2:]):
        assert torch.equal(a, b)
    assert got_grads[1].dtype == weight_dtype and got_grads[1].abs().sum() > 0
    torch.testing.assert_close(got_grads[1].float(), want_grads[1].float(), rtol = 2e-5, atol = 2e-5)


def test_no_fp32_token_by_expert_product_is_saved_for_backward():
    module, _ = _experts()
    tokens, top_k = 64, 4
    x, idx, weight = _routing(tokens = tokens, top_k = top_k)
    x.requires_grad_(True)
    weight.requires_grad_(True)
    saved = []

    def pack(t):
        saved.append((t.dtype, t.numel()))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = module(x, idx, weight)
    out.float().sum().backward()
    big_fp32 = [n for dtype, n in saved if dtype == torch.float32 and n >= tokens * top_k * H]
    assert not big_fp32, saved
    assert weight.grad is not None and x.grad is not None


def test_compiled_forward_matches_eager_and_keeps_the_router_grad():
    module, _ = _experts()
    x, idx, weight = _routing(tokens = 64, top_k = 4)
    w_eager = weight.clone().requires_grad_(True)
    eager = module(x, idx, w_eager)
    eager.float().square().sum().backward()

    torch._dynamo.reset()
    w_comp = weight.clone().requires_grad_(True)
    compiled = torch.compile(module)(x, idx, w_comp)
    compiled.float().square().sum().backward()
    assert torch.equal(compiled, eager)
    torch.testing.assert_close(w_comp.grad, w_eager.grad, rtol = 2e-5, atol = 2e-5)


def test_an_empty_batch_keeps_the_eager_combine(monkeypatch):
    module, _ = _experts()
    x, idx, weight = _routing()
    calls = _count_calls(monkeypatch)
    out = module(x[:0], idx[:0], weight[:0])
    assert out.shape == (0, H) and calls == []
    # A failed launch would disable the kernels process wide.
    assert K._DISABLED_REASON is None
