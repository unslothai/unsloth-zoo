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

"""Unsloth keeps vision LayerNorm weights in float32 while activations stay bfloat16.

The compiled ``F.layer_norm`` replacement handles that mix, but its eager body runs whenever
Dynamo does not compile: FX tracing, ``UNSLOTH_COMPILE_DISABLE``, the recompile limit, or the
``eager_on_recompile`` stance the compiled CausalLM forward sets after two inference forwards.
CUDA ``torch.layer_norm`` rejects a bfloat16 input with float32 weights, so a training forward
after two eval forwards crashed in the SigLIP encoder of Gemma 3.
"""

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


def test_patched_layer_norm_train_after_eval_under_eager_on_recompile():
    if not hasattr(torch.compiler, "set_stance"):
        pytest.skip("torch.compiler.set_stance unavailable")
    from unsloth_zoo.patch_torch_functions import layer_norm
    torch._dynamo.reset()
    try:
        # Eval forward compiles under no_grad, then the compiled CausalLM forward flips the stance
        x, w, b = _inputs(False)
        with torch.no_grad():
            layer_norm(x, (64,), w, b, 1e-6)
        torch.compiler.set_stance("eager_on_recompile")
        # Training forward: grad mode guard fails, so the call recompiles, i.e. runs eagerly
        x, w, b = _inputs(True)
        y = layer_norm(x, (64,), w, b, 1e-6)
        assert y.dtype == torch.bfloat16
        torch.testing.assert_close(y.float(), _reference(x, w, b).to(torch.bfloat16).float())
        y.float().sum().backward()
        assert torch.isfinite(x.grad).all()
    finally:
        torch.compiler.set_stance("default")
        torch._dynamo.reset()
