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

"""Memory checks: fused CE and JSD ops return stored gradients as extra outputs, so backward must not materialize zero
[N, H] or trainable-head [V, H] gradients on each backward."""

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

import unsloth_zoo.fused_losses.fused_linear as fl

N, H, V = 64, 32, 300


def _inputs(device, trainable):
    g = torch.Generator().manual_seed(0)
    w = (torch.randn(V, H, generator = g) / H**0.5).to(device).requires_grad_(trainable)
    h = torch.randn(N, H, generator = g).to(device).requires_grad_(True)
    th = torch.randn(N, H, generator = g).to(device)
    t = torch.randint(0, V, (N,), generator = g).to(device)
    return w, h, th, t


def _loss(kind, w, h, th, t):
    if kind == "ce":
        return fl.fused_linear_cross_entropy(h, w, None, t, divisor = torch.tensor(float(N), device = h.device))
    return fl.fused_linear_jsd(h, w, th, w.detach(), labels = t)


def _zero_fills(kind, trainable):
    w, h, th, t = _inputs("cpu", trainable)
    loss = _loss(kind, w, h, th, t)
    with profile(activities = [ProfilerActivity.CPU], record_shapes = True) as p:
        loss.backward()
    big = {(N, H), (V, H)}
    # A materialized gradient is an aten::zeros, seen as the aten::zero_ of its [N, H] / [V, H] buffer.
    return [e.name for e in p.events() if e.name == "aten::zero_"
            and any(tuple(s) in big for s in (e.input_shapes or []) if s)]


@pytest.mark.parametrize("trainable", [False, True])
@pytest.mark.parametrize("kind", ["ce", "jsd"])
def test_backward_does_not_materialize_stored_gradients(kind, trainable):
    assert _zero_fills(kind, trainable) == []


@pytest.mark.parametrize("trainable", [False, True])
@pytest.mark.parametrize("kind", ["ce", "jsd"])
def test_backward_memory_is_one_gradient_copy(kind, trainable):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    n, hdim, v = 4096, 1024, 32000
    w = (torch.randn(v, hdim, device = "cuda") / hdim**0.5).bfloat16().requires_grad_(trainable)
    h = torch.randn(n, hdim, device = "cuda", dtype = torch.bfloat16).requires_grad_(True)
    th = torch.randn(n, hdim, device = "cuda", dtype = torch.bfloat16)
    t = torch.randint(0, v, (n,), device = "cuda")
    for _ in range(2):  # the first call compiles / JITs
        h.grad = w.grad = None
        if kind == "ce":
            loss = fl.fused_linear_cross_entropy(h, w, None, t, divisor = torch.tensor(float(n), device = "cuda"))
        else:
            loss = fl.fused_linear_jsd(h, w, th, w.detach(), labels = t)
        torch.cuda.synchronize()
        a0 = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        loss.backward()
        torch.cuda.synchronize()
        extra = torch.cuda.max_memory_allocated() - a0
        del loss
    # The scaled gradients the op returns (one copy of what it stored), plus allocator rounding.
    stored = h.numel() * h.element_size() + (w.numel() * w.element_size() if trainable else 0)
    assert extra <= 1.1 * stored, (extra, stored)
