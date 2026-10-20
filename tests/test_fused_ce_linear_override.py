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

"""The fused CE head gradient survives DeepSpeed ZeRO-3's process-wide F.linear override.

`deepspeed.initialize` with ZeRO-3 (memory_efficient_linear, on by default) replaces
torch.nn.functional.linear with an autograd.Function and keeps it replaced. Inside the compiled
torch.func.grad_and_value chunk loop that Function produced a zero lm_head gradient, so a ZeRO-3
full finetune never trained an untied head (grad norm 0 for lm_head.weight, the rest matching one
GPU). The stand-in below is deepspeed/runtime/zero/linear.py's LinearFunctionForZeroStage3 reduced
to its math.
"""

import pytest
import torch

from unsloth_zoo.fused_losses import cross_entropy_loss as ce


class _ZeroStage3Linear(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(x, w, b = None):
        out = x.matmul(w.t())
        return out if b is None else out + b

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[0], inputs[1], inputs[2] if len(inputs) > 2 else None)

    @staticmethod
    def backward(ctx, g):
        x, w, b = ctx.saved_tensors
        gx = g.matmul(w) if ctx.needs_input_grad[0] else None
        gw = g.reshape(-1, g.shape[-1]).t().matmul(x.reshape(-1, x.shape[-1])) if ctx.needs_input_grad[1] else None
        gb = g.reshape(-1, g.shape[-1]).sum(0) if b is not None and ctx.needs_input_grad[2] else None
        return gx, gw, gb


def _zero3_linear(x, w, b = None):
    return _ZeroStage3Linear.apply(x, w, b)


def _fused(device, compile_):
    torch.manual_seed(0)
    V, H = 512, 32
    ids = torch.randint(0, V, (2, 16), device = device)
    w = torch.nn.Parameter(torch.randn(V, H, device = device) * 0.05)
    x = torch.randn(2, 16, H, device = device, requires_grad = True)
    loss = ce.unsloth_fused_ce_loss(
        trainer = None, hidden_states = x, lm_head_weight = w, lm_head_bias = None, labels = ids,
        mask = None, n_items = None, scaling = None, target_gb = None, torch_compile = compile_,
    )
    loss.backward()
    return loss.detach(), w.grad.detach().clone(), x.grad.detach().clone()


def test_fused_ce_logits_do_not_go_through_a_replaced_functional_linear(monkeypatch):
    calls = []

    def replaced(*args, **kwargs):
        calls.append(1)
        return _zero3_linear(*args, **kwargs)

    torch.manual_seed(0)
    x = torch.randn(2, 8, 16)
    w = torch.randn(64, 16)
    labels = torch.randint(0, 64, (2, 8))
    want = ce.compute_fused_ce_loss(x, w, None, labels)
    monkeypatch.setattr(torch.nn.functional, "linear", replaced)
    got = ce.compute_fused_ce_loss(x, w, None, labels)
    assert calls == []
    torch.testing.assert_close(got, want, rtol = 0, atol = 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "the compiled chunk loop runs on CUDA")
def test_compiled_fused_ce_head_gradient_under_zero3_linear(monkeypatch):
    want = _fused("cuda", compile_ = True)
    monkeypatch.setattr(torch.nn.functional, "linear", _zero3_linear)
    got = _fused("cuda", compile_ = True)
    assert float(got[1].norm()) > 0
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, rtol = 1e-5, atol = 1e-6)
