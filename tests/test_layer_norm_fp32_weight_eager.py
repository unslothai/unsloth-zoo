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

"""Eager layer_norm with float32 weights and bfloat16 activations (Gemma 3 SigLIP train-after-eval crash)."""

import pytest
import torch
import torch._dynamo

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA layer_norm rejects mixed dtypes; CPU accepts them")


def _inputs(requires_grad):
    g = torch.Generator(device = "cuda").manual_seed(0)
    x = torch.randn(2, 16, 64, device = "cuda", generator = g).to(torch.bfloat16).requires_grad_(requires_grad)
    w = (1 + 0.1 * torch.randn(64, device = "cuda", generator = g)).requires_grad_(requires_grad)
    b = (0.1 * torch.randn(64, device = "cuda", generator = g)).requires_grad_(requires_grad)
    return x, w, b


def _reference(x, w, b):
    return torch.nn.functional.layer_norm(x.detach().float(), (64,), w.detach(), b.detach(), 1e-6)


def test_eager_layer_norm_fp32_weight_bf16_input():
    from unsloth_zoo.patch_torch_functions import _layer_norm_eager
    x, w, b = _inputs(True)
    y = _layer_norm_eager(x, (64,), w, b, 1e-6)
    assert y.dtype == torch.bfloat16
    torch.testing.assert_close(y.float(), _reference(x, w, b).to(torch.bfloat16).float())
    y.float().sum().backward()
    assert x.grad.dtype == torch.bfloat16 and w.grad.dtype == torch.float32
    assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()


def test_eager_layer_norm_fp32_input_bf16_weight_is_not_downcast():
    from unsloth_zoo.patch_torch_functions import _layer_norm_eager
    g = torch.Generator(device = "cuda").manual_seed(1)
    x = torch.randn(2, 16, 64, device = "cuda", generator = g)
    w = (1 + 0.1 * torch.randn(64, device = "cuda", generator = g)).to(torch.bfloat16)
    b = (0.1 * torch.randn(64, device = "cuda", generator = g)).to(torch.bfloat16)
    y = _layer_norm_eager(x, (64,), w, b, 1e-6)
    assert y.dtype == torch.float32
    ref = torch.nn.functional.layer_norm(x, (64,), w.float(), b.float(), 1e-6)
    torch.testing.assert_close(y, ref, rtol = 0, atol = 1e-6)


def test_eager_layer_norm_wider_bias_is_not_downcast():
    from unsloth_zoo.patch_torch_functions import _layer_norm_eager
    g = torch.Generator(device = "cuda").manual_seed(2)
    x = torch.randn(2, 16, 64, device = "cuda", generator = g)
    w = (1 + 0.1 * torch.randn(64, device = "cuda", generator = g)).to(torch.bfloat16)
    b = (0.1 * torch.randn(64, device = "cuda", generator = g)).double()
    y = _layer_norm_eager(x, (64,), w, b, 1e-6)
    assert y.dtype == torch.float32
    ref = torch.nn.functional.layer_norm(x.double(), (64,), w.double(), b, 1e-6).float()
    assert torch.equal(y, ref)


def test_patched_layer_norm_train_after_eval_under_eager_on_recompile():
    if not hasattr(torch.compiler, "set_stance"):
        pytest.skip(reason = "torch.compiler.set_stance was added in torch 2.6; torch 2.4/2.5 have no eager_on_recompile stance")
    from unsloth_zoo.patch_torch_functions import layer_norm
    torch._dynamo.reset()
    try:
        x, w, b = _inputs(False)
        with torch.no_grad():
            layer_norm(x, (64,), w, b, 1e-6)
        torch.compiler.set_stance("eager_on_recompile")
        x, w, b = _inputs(True)
        y = layer_norm(x, (64,), w, b, 1e-6)
        assert y.dtype == torch.bfloat16
        torch.testing.assert_close(y.float(), _reference(x, w, b).to(torch.bfloat16).float())
        y.float().sum().backward()
        assert torch.isfinite(x.grad).all()
    finally:
        torch.compiler.set_stance("default")
        torch._dynamo.reset()
