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

"""CPU chunked CE: the head gradient buffer formed in forward becomes .grad (no zero fill, no scaled copy)."""

import pytest
import torch
import torch.nn.functional as F

from unsloth_zoo.cpu import ops

V, H = 211, 24

needs_keep_graph = pytest.mark.skipif(
    getattr(torch._C._autograd, "_get_current_graph_task_keep_graph", None) is None,
    reason = "torch without _get_current_graph_task_keep_graph scales out of place",
)


def _inputs(seed, n = 50, dtype = torch.float32):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(n, H, generator = g).to(dtype)
    weight = (torch.randn(V, H, generator = g) * H ** -0.5).to(dtype)
    labels = torch.randint(0, V, (n,), generator = g)
    labels[::7] = -100
    return hidden, weight, labels


def _loss(h, w, labels, chunk_rows = 8):
    return ops.chunked_linear_cross_entropy(h, w, None, shift_labels = labels, chunk_rows = chunk_rows)


@pytest.fixture
def stored(monkeypatch):
    ptrs = []
    impl = ops._chunked_ce_forward_impl

    def spy(*args, **kwargs):
        out = impl(*args, **kwargs)
        if out[2].numel():
            ptrs.append(out[2].data_ptr())
        return out

    monkeypatch.setattr(ops, "_chunked_ce_forward_impl", spy)
    return ptrs


@needs_keep_graph
@pytest.mark.parametrize("factor", [1.0, 0.75])
def test_head_grad_buffer_becomes_param_grad(stored, factor):
    hidden, weight, labels = _inputs(0)
    w = weight.clone().requires_grad_()
    (_loss(hidden.clone().requires_grad_(), w, labels) * factor).backward()
    assert len(stored) == 1
    assert w.grad.data_ptr() == stored[0]


@needs_keep_graph
def test_tied_embedding_and_head_take_the_buffer(stored):
    hidden, weight, labels = _inputs(1)
    ids = torch.randint(0, V, (hidden.shape[0],), generator = torch.Generator().manual_seed(1))
    w = weight.clone().requires_grad_()
    x = F.embedding(ids, w) + hidden
    _loss(x, w, labels).backward()
    ref_w = weight.double().clone().requires_grad_()
    ref = F.cross_entropy((F.embedding(ids, ref_w) + hidden.double()) @ ref_w.t(), labels, ignore_index = -100)
    ref.backward()
    torch.testing.assert_close(w.grad.double(), ref_w.grad, rtol = 1e-5, atol = 1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("factor", [1.0, 0.375, 3.0])
@pytest.mark.parametrize("chunk_rows", [8, 1000])
def test_in_place_matches_out_of_place_bitwise(dtype, factor, chunk_rows):
    hidden, weight, labels = _inputs(2, dtype = dtype)
    grads = []
    for retain in (False, True):  # retain_graph keeps the out-of-place scale
        h = hidden.clone().requires_grad_()
        w = weight.clone().requires_grad_()
        (_loss(h, w, labels, chunk_rows) * factor).backward(retain_graph = retain)
        grads.append((h.grad, w.grad))
    assert torch.equal(grads[0][0], grads[1][0])
    assert torch.equal(grads[0][1], grads[1][1])
    ref_w = weight.double().clone().requires_grad_()
    (F.cross_entropy(hidden.double() @ ref_w.t(), labels, ignore_index = -100) * factor).backward()
    tol = 1e-5 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(grads[0][1].double(), ref_w.grad, rtol = tol, atol = tol)


def test_retain_graph_second_backward_is_not_double_scaled():
    hidden, weight, labels = _inputs(3)
    w = weight.clone().requires_grad_()
    loss = _loss(hidden.clone().requires_grad_(), w, labels) * 0.5
    loss.backward(retain_graph = True)
    first = w.grad.clone()
    loss.backward()
    assert torch.equal(w.grad, first + first)


def test_gradient_accumulation_adds_into_existing_grad():
    w = _inputs(0)[1].clone().requires_grad_()
    expected = None
    for step in range(3):
        hidden, _, labels = _inputs(10 + step)
        (_loss(hidden.clone().requires_grad_(), w, labels) / 3).backward()
        w1 = w.detach().clone().requires_grad_()
        (_loss(hidden.clone(), w1, labels) / 3).backward(retain_graph = True)
        expected = w1.grad.clone() if expected is None else expected + w1.grad
    assert torch.equal(w.grad, expected)


def test_autograd_grad_and_hooks_see_the_scaled_grad():
    hidden, weight, labels = _inputs(4)
    w = weight.clone().requires_grad_()
    gw, = torch.autograd.grad(_loss(hidden, w, labels) * 2.0, w)
    seen = {}
    w2 = weight.clone().requires_grad_()
    w2.register_hook(lambda g: seen.__setitem__("pre", g.clone()))
    (_loss(hidden, w2, labels) * 2.0).backward()
    assert torch.equal(gw, w2.grad) and torch.equal(seen["pre"], w2.grad)


def test_all_labels_ignored_gives_zero_head_grad():
    hidden, weight, labels = _inputs(5)
    labels[:] = -100
    w = weight.clone().requires_grad_()
    _loss(hidden.clone().requires_grad_(), w, labels).backward()
    assert torch.equal(w.grad, torch.zeros_like(w))


def test_forward_does_not_zero_fill_the_head_grad():
    hidden, weight, labels = _inputs(6, n = 64)
    labels[:] = labels.clamp_min(0)
    w = weight.clone().requires_grad_()
    with torch.profiler.profile(activities = [torch.profiler.ProfilerActivity.CPU], record_shapes = True) as prof:
        _loss(hidden.clone().requires_grad_(), w, labels, chunk_rows = 16).backward()
    head = str([V, H])
    fills = [e for e in prof.key_averages(group_by_input_shape = True)
             if e.key in ("aten::fill_", "aten::zero_") and str(e.input_shapes).startswith("[" + head)]
    assert not fills, [(e.key, e.input_shapes, e.count) for e in fills]
