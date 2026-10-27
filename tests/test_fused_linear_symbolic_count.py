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

"""A shape-derived count (``labels.numel()``, a SymInt under a dynamic compile) passed as the fused CE divisor or the
fused JSD ``num_items_in_batch`` must stay symbolic: one graph for every row count, no specialization error under
``mark_dynamic``, and the same loss and gradients as eager."""

import pytest
import torch
from torch._dynamo.testing import CompileCounter

import unsloth_zoo.fused_losses.fused_linear as fl

ROWS = (17, 33, 49, 65)
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _loss(kind, h, t, teacher, w, wt, count):
    if kind == "ce":
        return fl.fused_linear_cross_entropy(h, w, None, t, divisor = count)
    return fl.fused_linear_jsd(h, w, teacher, wt, labels = t, num_items_in_batch = count)


def _inputs(n, device):
    g = torch.Generator().manual_seed(n)
    h = torch.randn(n, 16, generator = g).to(device).requires_grad_()
    t = torch.randint(0, 64, (n,), generator = g).to(device)
    t[::5] = -100
    teacher = torch.randn(n, 24, generator = g).to(device)
    return h, t, teacher


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["dynamic", "mark_dynamic"])
@pytest.mark.parametrize("kind", ["ce", "jsd"])
def test_shape_derived_count_compiles_once(kind, mode, device):
    torch._dynamo.reset()
    g = torch.Generator().manual_seed(0)
    w = torch.randn(64, 16, generator = g).to(device)
    wt = torch.randn(64, 24, generator = g).to(device)

    def body(h, t, teacher):
        return _loss(kind, h, t, teacher, w, wt, t.numel())

    counter = CompileCounter()
    compiled = torch.compile(body, backend = counter, fullgraph = True, dynamic = True if mode == "dynamic" else None)
    for n in ROWS:
        h, t, teacher = _inputs(n, device)
        h_ref = h.detach().clone().requires_grad_()
        ref = _loss(kind, h_ref, t, teacher, w, wt, n)
        ref.backward()
        if mode == "mark_dynamic":
            for x in (h, t, teacher):
                torch._dynamo.mark_dynamic(x, 0)
        out = compiled(h, t, teacher)
        out.backward()
        torch.testing.assert_close(out, ref, rtol = 1e-5, atol = 1e-6)
        torch.testing.assert_close(h.grad, h_ref.grad, rtol = 1e-5, atol = 1e-6)
    assert counter.frame_count == 1, f"{counter.frame_count} graphs for {len(ROWS)} row counts"


@pytest.mark.parametrize("kind", ["ce", "jsd"])
def test_python_and_tensor_counts_give_the_same_loss(kind):
    h, t, teacher = _inputs(33, "cpu")
    g = torch.Generator().manual_seed(1)
    w = torch.randn(64, 16, generator = g)
    wt = torch.randn(64, 24, generator = g)
    kept = int((t != -100).sum())
    losses = [_loss(kind, h, t, teacher, w, wt, c) for c in (kept, float(kept), torch.tensor(kept), torch.tensor([kept]))]
    for loss in losses[1:]:
        assert torch.equal(loss, losses[0])
    zero = _loss(kind, h, t, teacher, w, wt, 0)
    assert torch.equal(zero, _loss(kind, h, t, teacher, w, wt, 1))
