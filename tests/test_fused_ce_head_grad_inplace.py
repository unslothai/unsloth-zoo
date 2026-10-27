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

"""A trainable head's gradient, stored by the fused CE forward, becomes .grad without a second [V, H] copy.

The backward scales the stored gradient in place (eager, no retain_graph), so autograd adopts that buffer as
weight.grad. Gradient accumulation, retain_graph, hooks, loss scaling and torch.compile keep their results."""

import pytest
import torch
import torch.nn.functional as F
from torch._dynamo.utils import counters

from unsloth_zoo.fused_losses import fused_linear as fl

V, H = 97, 16


def _inputs(seed, n = 40, dtype = torch.bfloat16):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(n, H, generator = g).to(dtype)
    weight = (torch.randn(V, H, generator = g) * H ** -0.5).to(dtype)
    targets = torch.randint(0, V, (n,), generator = g)
    targets[::7] = -100
    return hidden, weight, targets


def _loss(h, w, t, scaling = None):
    return fl.fused_linear_cross_entropy(h, w, None, t, divisor = torch.tensor(float((t != -100).sum())),
                                         scaling = scaling)


@pytest.fixture
def stored_head_grad(monkeypatch):
    """data_ptr of every head gradient the CE forward stores."""
    ptrs = []
    impl = fl._OPS["ce_impl"]

    def spy(*args, **kwargs):
        out = impl(*args, **kwargs)
        if out[2].numel():
            ptrs.append(out[2].data_ptr())
        return out

    monkeypatch.setitem(fl._OPS, "ce_impl", spy)
    return ptrs


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_head_grad_buffer_becomes_param_grad(stored_head_grad, dtype):
    hidden, weight, targets = _inputs(0, dtype = dtype)
    h = hidden.clone().requires_grad_()
    w = weight.clone().requires_grad_()
    (_loss(h, w, targets) * 0.75).backward()
    assert len(stored_head_grad) == 1
    # Before: the backward wrote a fresh [V, H] tensor, so the stored one and .grad coexisted.
    assert w.grad.data_ptr() == stored_head_grad[0]


def test_jsd_student_head_grad_buffer_becomes_param_grad(monkeypatch):
    ptrs = []
    impl = fl._OPS["jsd_impl"]

    def spy(*args, **kwargs):
        out = impl(*args, **kwargs)
        ptrs.append(out[2].data_ptr())
        return out

    monkeypatch.setitem(fl._OPS, "jsd_impl", spy)
    hidden, weight, _ = _inputs(7)
    t_hidden, t_weight, _ = _inputs(8)
    ws = []
    for retain in (False, True):
        wi = weight.clone().requires_grad_()
        (fl.fused_linear_jsd(hidden.clone().requires_grad_(), wi, t_hidden, t_weight) * 0.5).backward(
            retain_graph = retain)
        ws.append(wi)
    assert ws[0].grad.data_ptr() == ptrs[0]
    assert torch.equal(ws[0].grad, ws[1].grad)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("factor", [1.0, 0.375, 3.0])
def test_inplace_matches_out_of_place_bitwise(dtype, factor):
    hidden, weight, targets = _inputs(1, dtype = dtype)
    grads = []
    for retain in (False, True):  # retain_graph keeps the out-of-place scale
        h = hidden.clone().requires_grad_()
        w = weight.clone().requires_grad_()
        (_loss(h, w, targets) * factor).backward(retain_graph = retain)
        grads.append((h.grad, w.grad))
    assert torch.equal(grads[0][0], grads[1][0])
    assert torch.equal(grads[0][1], grads[1][1])


def test_gradient_accumulation_adds_into_existing_grad():
    w = _inputs(0)[1].clone().requires_grad_()
    ref = w.detach().double().clone().requires_grad_()
    singles = []
    for step in range(3):
        hidden, _, targets = _inputs(10 + step)
        h = hidden.clone().requires_grad_()
        (_loss(h, w, targets) / 3).backward()
        w1 = w.detach().clone().requires_grad_()
        (_loss(hidden.clone(), w1, targets) / 3).backward(retain_graph = True)
        singles.append(w1.grad)
        logits = hidden.double() @ ref.t()
        (F.cross_entropy(logits, targets, ignore_index = -100) / 3).backward()
    expected = singles[0].clone()
    for g in singles[1:]:
        expected += g
    # Same summation order as autograd's AccumulateGrad (adopt the first, add the rest).
    assert torch.equal(w.grad, expected)
    rel = float((w.grad.double() - ref.grad).norm() / ref.grad.norm())
    assert rel < 1e-2, rel


def test_retain_graph_second_backward_is_not_double_scaled():
    hidden, weight, targets = _inputs(2)
    h = hidden.clone().requires_grad_()
    w = weight.clone().requires_grad_()
    loss = _loss(h, w, targets) * 0.5
    loss.backward(retain_graph = True)
    first = w.grad.clone()
    loss.backward()
    w2 = weight.clone().requires_grad_()
    w2.grad = first.clone()
    w2.grad += first
    assert torch.equal(w.grad, w2.grad)


def test_grad_scaler_scaling_is_divided_out():
    hidden, weight, targets = _inputs(3)
    out = []
    for scaling in (None, 1024.0):
        h = hidden.clone().requires_grad_()
        w = weight.clone().requires_grad_()
        loss = _loss(h, w, targets, scaling = scaling)
        (loss * (scaling or 1.0)).backward()
        out.append(w.grad.float() / (scaling or 1.0))
    torch.testing.assert_close(out[0], out[1], rtol = 1e-2, atol = 1e-4)


def test_hooks_see_final_grad():
    hidden, weight, targets = _inputs(4)
    seen = {}
    w = weight.clone().requires_grad_()
    w.register_hook(lambda g: seen.__setitem__("pre", g.clone()))
    w.register_post_accumulate_grad_hook(lambda p: seen.__setitem__("post", p.grad.clone()))
    (_loss(hidden.clone().requires_grad_(), w, targets) * 0.25).backward()
    assert torch.equal(seen["pre"], w.grad) and torch.equal(seen["post"], w.grad)


def test_autograd_grad_returns_scaled():
    hidden, weight, targets = _inputs(5)
    w = weight.clone().requires_grad_()
    gw, = torch.autograd.grad(_loss(hidden, w, targets) * 2.0, w)
    w2 = weight.clone().requires_grad_()
    (_loss(hidden, w2, targets) * 2.0).backward(retain_graph = True)
    assert torch.equal(gw, w2.grad)


def test_compiled_no_graph_break_and_same_grads():
    hidden, weight, targets = _inputs(6)
    divisor = torch.tensor(float((targets != -100).sum()))

    def f(h, w, t):
        return fl.fused_linear_cross_entropy(h, w, None, t, divisor = divisor) * 0.5

    torch._dynamo.reset()
    counters.clear()
    w_c = weight.clone().requires_grad_()
    torch.compile(f, backend = "aot_eager", fullgraph = True)(hidden.clone().requires_grad_(), w_c, targets).backward()
    assert not counters["graph_break"]
    w_e = weight.clone().requires_grad_()
    f(hidden.clone().requires_grad_(), w_e, targets).backward()
    assert torch.equal(w_c.grad, w_e.grad)
