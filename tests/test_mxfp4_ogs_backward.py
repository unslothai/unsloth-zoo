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

"""Native MXFP4 gpt-oss experts (triton_kernels matmul_ogs) train: backward vs an fp32 reference on the
exact dequantized weights."""

import importlib
import types

import pytest
import torch

from unsloth_zoo.temporary_patches import gpt_oss as go

ALPHA, LIMIT = 1.702, 7.0


def _swiglu_reference(pre):
    gate, linear = pre[..., ::2].clamp(max = LIMIT), pre[..., 1::2].clamp(-LIMIT, LIMIT)
    return gate * torch.sigmoid(ALPHA * gate) * (linear + 1)


def test_swiglu_backward_matches_autograd_including_the_clamps():
    torch.manual_seed(0)
    pre = (torch.randn(64, 32) * 6).requires_grad_()
    grad = torch.randn(64, 16)
    (_swiglu_reference(pre) * grad).sum().backward()
    x = pre.detach().clone().requires_grad_()
    (go._OgsSwiglu.apply(x, ALPHA, LIMIT) * grad).sum().backward()
    assert torch.allclose(x.grad, pre.grad, rtol = 1e-5, atol = 1e-6)
    assert (pre.detach()[..., ::2] > LIMIT).any() and (pre.detach()[..., 1::2].abs() > LIMIT).any()


def _triton_kernels():
    if not torch.cuda.is_available():
        pytest.skip("matmul_ogs needs CUDA")
    from unsloth_zoo.triton_kernels_compat import get_triton_kernels

    tk = get_triton_kernels()
    if tk is None:
        pytest.skip("no triton_kernels (top-level or vLLM's copy)")
    return tk


def _tk(name):
    return importlib.import_module(f"{_triton_kernels().__name__}.{name}")


def _module(E, H, I, gen):
    mo, tt, L = _tk("matmul_ogs"), _tk("tensor"), _tk("tensor_details.layout")
    from unsloth_zoo.mxfp4_dequant import mxfp4_dequantize

    def packed(N, K):
        blocks = torch.randint(0, 256, (E, N, K // 32, 16), dtype = torch.uint8, device = "cuda", generator = gen)
        scales = torch.randint(119, 124, (E, N, K // 32), dtype = torch.uint8, device = "cuda", generator = gen)
        return blocks, scales

    def swizzled(blocks, scales, N, K):
        # As transformers' swizzle_mxfp4_convertops, with the zoo's layout choice.
        flat = blocks.reshape(E, N, -1)
        value_layout, opts, strided = go._mxfp4_layout_arguments(L, flat)
        weight = tt.convert_layout(tt.wrap_torch_tensor(flat.transpose(-2, -1), dtype = tt.FP4), value_layout, **opts)
        scale = tt.convert_layout(tt.wrap_torch_tensor(scales.transpose(-2, -1)), strided)
        weight.shape = torch.Size([E, K, N])
        return weight, mo.PrecisionConfig(weight_scale = scale, flex_ctx = mo.FlexCtx(rhs_data = mo.InFlexData()))

    gate_up, down = packed(2 * I, H), packed(H, I)
    module = types.SimpleNamespace(alpha = ALPHA, limit = LIMIT, num_experts = E)
    module.gate_up_proj, module.gate_up_proj_precision_config = swizzled(*gate_up, 2 * I, H)
    module.down_proj, module.down_proj_precision_config = swizzled(*down, H, I)
    module.gate_up_proj_bias = torch.nn.Parameter(torch.randn(E, 2 * I, device = "cuda", generator = gen) * 0.1)
    module.down_proj_bias = torch.nn.Parameter(torch.randn(E, H, device = "cuda", generator = gen) * 0.1)
    dense = [mxfp4_dequantize(b, s, dtype = torch.bfloat16, transpose = True).float() for b, s in (gate_up, down)]
    return module, dense


def _lora(E, n_in, n_out, rank, gen):
    first = (torch.randn(E, n_in, rank, device = "cuda", generator = gen) * 0.02).to(torch.bfloat16).requires_grad_()
    second = (torch.randn(E, rank, n_out, device = "cuda", generator = gen) * 0.02).to(torch.bfloat16).requires_grad_()
    return first, second


def _reference(x, router, module, dense, k, loras):
    gate_up, down = dense
    logits = (x.to(torch.bfloat16) @ router.to(torch.bfloat16).t()).float()
    values, experts = torch.topk(logits, k, dim = -1)
    weights = torch.softmax(values, -1)
    out = 0
    for slot in range(k):
        e = experts[:, slot]
        pre = torch.bmm(x.unsqueeze(1), gate_up[e]).squeeze(1) + module.gate_up_proj_bias[e].float()
        if loras is not None:
            (a, b), _ = loras
            pre = pre + torch.bmm(torch.bmm(x.unsqueeze(1), a[e].float()), b[e].float()).squeeze(1) * 2.0
        h = _swiglu_reference(pre)
        y = torch.bmm(h.unsqueeze(1), down[e]).squeeze(1) + module.down_proj_bias[e].float()
        if loras is not None:
            _, (a, b) = loras
            y = y + torch.bmm(torch.bmm(h.unsqueeze(1), a[e].float()), b[e].float()).squeeze(1) * 2.0
        out = out + weights[:, slot : slot + 1] * y
    return out


def _run(E, H, I, T, k, seed, lora_rank = None):
    rt = _tk("routing")

    gen = torch.Generator(device = "cuda").manual_seed(seed)
    module, dense = _module(E, H, I, gen)
    x0 = torch.randn(T, H, device = "cuda", generator = gen).to(torch.bfloat16)
    router0 = torch.randn(E, H, device = "cuda", generator = gen) * 0.05
    grad = torch.randn(T, H, device = "cuda", generator = gen)
    loras = None
    if lora_rank:
        loras = (_lora(E, H, 2 * I, lora_rank, gen), _lora(E, I, H, lora_rank, gen))

    x, router = x0.clone().requires_grad_(), router0.clone().requires_grad_()
    logits = torch.nn.functional.linear(x, router.to(torch.bfloat16))
    routing_data, gather_idx, scatter_idx = rt.routing(logits, k)
    kwargs = {}
    if loras is not None:
        kwargs = dict(gate_up_lora = (*loras[0], 2.0, E), down_lora = (*loras[1], 2.0, E))
    out = go.mxfp4_ogs_experts_forward(module, x, routing_data, gather_idx, scatter_idx, **kwargs)
    (out.float() * grad).sum().backward()
    ours = {
        "out": out.detach(), "x": x.grad, "router": router.grad,
        "gate_up_bias": module.gate_up_proj_bias.grad, "down_bias": module.down_proj_bias.grad,
    }
    if loras is not None:
        ours.update({f"lora{i}{j}": t.grad for i, pair in enumerate(loras) for j, t in enumerate(pair)})
        for pair in loras:
            for t in pair:
                t.grad = None
    module.gate_up_proj_bias.grad = module.down_proj_bias.grad = None

    x_ref, router_ref = x0.float().requires_grad_(), router0.clone().requires_grad_()
    ref = _reference(x_ref, router_ref, module, dense, k, loras)
    (ref * grad).sum().backward()
    theirs = {
        "out": ref.detach(), "x": x_ref.grad, "router": router_ref.grad,
        "gate_up_bias": module.gate_up_proj_bias.grad, "down_bias": module.down_proj_bias.grad,
    }
    if loras is not None:
        theirs.update({f"lora{i}{j}": t.grad for i, pair in enumerate(loras) for j, t in enumerate(pair)})
    return {key: ((ours[key].float() - theirs[key].float()).norm() / theirs[key].float().norm()).item() for key in theirs}


# bf16 GEMMs against an fp32 reference: relative L2 error bounds.
TOLERANCE = {"out": 1e-2, "x": 5e-2, "router": 2e-2, "gate_up_bias": 5e-2, "down_bias": 1e-2}


@pytest.mark.parametrize("E, H, I, T, k", [(4, 256, 256, 64, 2), (8, 512, 256, 200, 4)])
def test_native_mxfp4_experts_backward_matches_the_dequantized_reference(E, H, I, T, k):
    _triton_kernels()
    errors = _run(E, H, I, T, k, seed = 1)
    assert all(errors[key] < TOLERANCE[key] for key in TOLERANCE), errors


def test_backward_is_the_same_decoded_one_expert_at_a_time(monkeypatch):
    _triton_kernels()
    whole = _run(8, 512, 256, 200, 4, seed = 2)
    monkeypatch.setenv("UNSLOTH_MXFP4_EXPERT_CHUNK_MB", "1")
    chunked = _run(8, 512, 256, 200, 4, seed = 2)
    assert whole == chunked


def test_expert_lora_trains_on_the_native_path():
    _triton_kernels()
    errors = _run(8, 512, 256, 200, 4, seed = 3, lora_rank = 8)
    assert all(errors[key] < TOLERANCE[key] for key in TOLERANCE), errors
    assert all(errors[key] < 5e-2 for key in errors if key.startswith("lora")), errors
