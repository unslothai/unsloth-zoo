# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""MXFP4 gpt-oss experts return float32 when training on float16 (T4), like GptOssExpertsBnb4bit, so the experts
output and the residual adds after it no longer overflow float16's 65504; bfloat16 and eval are unchanged. CPU only:
the exact dequant path (sm75's) with dense expert stacks and the zoo's own routing tensors."""

import contextlib
import types

import pytest
import torch

from unsloth_zoo.temporary_patches import gpt_oss as go

FP16_MAX = 65504.0
E, H, I, T, K = 4, 64, 32, 24, 2


class _Experts(torch.nn.Module):
    def __init__(self, gen):
        super().__init__()
        self.num_experts, self.alpha, self.limit = E, 1.702, 7.0
        # Dense stacks take _ogs_dense_chunks' tensor branch: the same bf16 x fp32-accumulate math as a decoded stack.
        self.gate_up_proj = (torch.randn(E, H, 2 * I, generator = gen) * 0.3).to(torch.bfloat16)
        # Large down weights: some tokens' expert output lands past float16's range (gpt-oss-20b reaches ~1e5).
        self.down_proj = (torch.randn(E, I, H, generator = gen) * 600.0).to(torch.bfloat16)
        self.gate_up_proj_bias = torch.nn.Parameter(torch.randn(E, 2 * I, generator = gen) * 0.1)
        self.down_proj_bias = torch.nn.Parameter(torch.randn(E, H, generator = gen) * 0.1)
        self.gate_up_proj_precision_config = self.down_proj_precision_config = None


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(go, "is_triton_kernels_available", lambda: True)
    monkeypatch.setattr(go, "_ogs_backend_for", lambda *a, **k: None)
    monkeypatch.setattr(go, "_ogs_module_backend", lambda module: None)
    monkeypatch.setattr(go, "_ogs_native_training_ok", lambda device, backend: False)
    monkeypatch.setattr(go, "torch_cuda_device", lambda device: contextlib.nullcontext())
    monkeypatch.delenv("UNSLOTH_GPT_OSS_MXFP4_FP32_RESIDUAL", raising = False)
    monkeypatch.setattr(go, "_GPT_OSS_FLEX_SINK_ATTENTION_INSTALLED", True, raising = False)
    gen = torch.Generator().manual_seed(0)
    module = _Experts(gen)
    x = torch.randn(T, H, generator = gen) * 4.0
    values, experts = torch.topk(torch.randn(T, E, generator = gen), K, dim = -1)
    weights = torch.softmax(values, -1)
    gate_scal, hist, combine, dispatch = go.topk_to_routing_tensors(experts, weights, E)
    routing = types.SimpleNamespace(n_expts_act = K, gate_scal = gate_scal, expt_hist = hist)
    gather = types.SimpleNamespace(src_indx = combine, dst_indx = dispatch)
    scatter = types.SimpleNamespace(src_indx = dispatch, dst_indx = combine)
    return module, x, experts, weights, (routing, gather, scatter)


def _reference(module, x, experts, weights):
    """float64 experts forward on the bf16 roundings the dequant path makes (input, pre-activation, SwiGLU output)."""
    xb = x.to(torch.bfloat16).double()
    out = torch.zeros(T, H, dtype = torch.float64)
    for slot in range(K):
        e = experts[:, slot]
        pre = torch.bmm(xb.unsqueeze(1), module.gate_up_proj[e].double()).squeeze(1) + module.gate_up_proj_bias[e].double()
        h = go.swiglu_torch_forward(pre.to(torch.bfloat16), module.alpha, module.limit).double()
        y = torch.bmm(h.unsqueeze(1), module.down_proj[e].double()).squeeze(1) + module.down_proj_bias[e].double()
        out += weights[:, slot : slot + 1].double() * y
    return out


def _close(out, ref):
    # float32 sums of bf16 products: well inside 1e-3 of the output scale.
    err = (out.double() - ref).abs().max() / ref.abs().max()
    assert err < 1e-3, float(err)


def _forward(module, x, routes, dtype, training = True):
    module.train(training)
    xin = x.to(dtype).requires_grad_(training)
    return go.forward_mxfp4_gpt_oss_with_lora(module, xin, *routes), xin


def test_fp16_training_returns_finite_fp32_past_the_fp16_range(setup):
    module, x, experts, weights, routes = setup
    out, xin = _forward(module, x, routes, torch.float16)
    ref = _reference(module, xin.detach(), experts, weights)
    assert ref.abs().max() > 1.5 * FP16_MAX  # the case under test: the expert output itself leaves float16's range
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    _close(out, ref)
    # The residual add the decoder layer does next stays finite (float16 + float32 promotes).
    residual = torch.full((T, H), 40000.0, dtype = torch.float16)
    assert torch.isfinite(residual + out).all()
    out.backward(torch.full_like(out, 1e-4))  # a loss-sized upstream gradient
    assert xin.grad.dtype == torch.float16 and torch.isfinite(xin.grad).all()


def test_kill_switch_keeps_the_fp16_output(setup, monkeypatch):
    monkeypatch.setenv("UNSLOTH_GPT_OSS_MXFP4_FP32_RESIDUAL", "0")
    module, x, experts, weights, routes = setup
    out, _ = _forward(module, x, routes, torch.float16)
    assert out.dtype == torch.float16
    assert not torch.isfinite(out).all()  # what main returned: the overflow this change removes


def test_stock_attention_keeps_the_fp16_output(setup, monkeypatch):
    # UNSLOTH_COMPILE_DISABLE=1 / UNSLOTH_ENABLE_FLEX_ATTENTION=0: no attention input cast, so no float32 stream.
    monkeypatch.setattr(go, "_GPT_OSS_FLEX_SINK_ATTENTION_INSTALLED", False)
    module, x, experts, weights, routes = setup
    out, _ = _forward(module, x, routes, torch.float16)
    assert out.dtype == torch.float16


def test_fp32_input_after_the_first_layer_stays_fp32(setup):
    module, x, experts, weights, routes = setup
    out, xin = _forward(module, x, routes, torch.float32)
    assert out.dtype == torch.float32 and torch.isfinite(out).all()
    _close(out, _reference(module, xin.detach(), experts, weights))


def test_bf16_training_is_unchanged(setup, monkeypatch):
    module, x, experts, weights, routes = setup
    out, _ = _forward(module, x, routes, torch.bfloat16)
    monkeypatch.setenv("UNSLOTH_GPT_OSS_MXFP4_FP32_RESIDUAL", "0")
    off, _ = _forward(module, x, routes, torch.bfloat16)
    assert out.dtype == off.dtype == torch.bfloat16
    assert torch.equal(out, off)


def test_fp16_eval_keeps_the_activation_dtype(setup):
    module, x, experts, weights, routes = setup
    with torch.no_grad():
        out, _ = _forward(module, x, routes, torch.float16, training = False)
    assert out.dtype == torch.float16


def test_router_logits_take_the_router_dtype():
    router = torch.nn.Linear(H, E).to(torch.float16)
    hidden = torch.randn(T, H, dtype = torch.float32)
    logits = go._mxfp4_router_logits(router, hidden)
    assert logits.dtype == torch.float16
    torch.testing.assert_close(logits, router(hidden.to(torch.float16)), rtol = 0, atol = 0)


def test_attention_input_meets_a_dense_fp16_projection():
    import types as _types
    hidden = torch.randn(T, H, dtype = torch.float32)
    fp16 = _types.SimpleNamespace(q_proj = torch.nn.Linear(H, H).to(torch.float16))
    assert go._gpt_oss_attention_input(fp16, hidden).dtype == torch.float16
    # A LoRA wrapper is looked through to its base layer.
    wrapped = _types.SimpleNamespace(q_proj = _types.SimpleNamespace(base_layer = fp16.q_proj))
    assert go._gpt_oss_attention_input(wrapped, hidden).dtype == torch.float16
    # bf16 / float32 dense and quantized (uint8) projections, and same-dtype input, pass through.
    for dtype in (torch.bfloat16, torch.float32, torch.uint8):
        proj = _types.SimpleNamespace(q_proj = _types.SimpleNamespace(weight = torch.zeros(1, dtype = dtype)))
        assert go._gpt_oss_attention_input(proj, hidden) is hidden
    half = hidden.to(torch.float16)
    assert go._gpt_oss_attention_input(fp16, half) is half


@pytest.mark.parametrize("positional", [False, True])
def test_tensor_routed_call_keeps_fp16(setup, monkeypatch, positional):
    # Stock GptOssMLP (keywords, transformers 4.x) and transformers 5.x Mxfp4GptOssMLP (positional) pass
    # (router_indices, routing_weights) and run their own float16 router: a float32 output would reach it next layer.
    module, x, experts, weights, routes = setup
    monkeypatch.setattr(go, "_mxfp4_routing_from_topk", lambda module, indices, weights: routes)
    module.train()
    xin = x.to(torch.float16)
    if positional:
        out = go.forward_mxfp4_gpt_oss_with_lora(module, xin, experts, weights)
    else:
        out = go.forward_mxfp4_gpt_oss_with_lora(module, xin, router_indices = experts, routing_weights = weights)
    assert out.dtype == torch.float16
    router = torch.nn.Linear(H, E).to(torch.float16)
    router(xin + out)  # the next layer's stock router, no autocast: no dtype error
