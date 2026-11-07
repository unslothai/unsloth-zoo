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

"""The compiled CPU chunked cross entropy accepts num_items_in_batch as a Python int that changes per step."""

import pytest
import torch
from torch._dynamo.utils import counters


def _loss(hidden, weight, labels, count):
    from unsloth_zoo.cpu.ops import chunked_linear_cross_entropy
    return chunked_linear_cross_entropy(hidden, weight, labels, num_items_in_batch = count, chunk_rows = 4)


def _inputs(seed):
    torch.manual_seed(seed)
    hidden = torch.randn(2, 12, 8, requires_grad = True)
    weight = (torch.randn(13, 8) * 0.3).requires_grad_()
    labels = torch.randint(0, 13, (2, 12))
    labels[:, :6] = -100
    return hidden, weight, labels


@pytest.mark.parametrize("kind", ["int", "tensor"])
def test_chunked_ce_fullgraph_dynamic_count_matches_eager(kind):
    torch._dynamo.reset()
    counters.clear()
    fn = torch.compile(_loss, fullgraph = True, dynamic = True, backend = "aot_eager")
    wrap = (lambda c: c) if kind == "int" else (lambda c: torch.tensor(c))
    tol = dict(atol = 1e-5, rtol = 1e-5)
    for count in [7, 9, 11, 0]:
        hidden, weight, labels = _inputs(count)
        out = fn(hidden, weight, labels, wrap(count))
        out.backward()
        grads = (hidden.grad.clone(), weight.grad.clone())
        hidden.grad = weight.grad = None
        ref = _loss(hidden, weight, labels, count)
        ref.backward()
        torch.testing.assert_close(out, ref, **tol)
        torch.testing.assert_close(grads[0], hidden.grad, **tol)
        torch.testing.assert_close(grads[1], weight.grad, **tol)
    hidden = torch.randn(1, 5, 8, requires_grad = True)
    out = fn(hidden, torch.randn(13, 8), torch.full((1, 5), -100), wrap(0))
    out.backward()
    assert out.item() == 0.0
    assert torch.equal(hidden.grad, torch.zeros_like(hidden.grad))
    assert sum(counters["graph_break"].values()) == 0, dict(counters["graph_break"])
