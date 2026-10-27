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

"""The CPU chunked cross entropy stays traceable: masked-row skipping must not break torch.compile."""

import pytest
import torch
import torch.nn.functional as F
from torch._dynamo.utils import counters


def _inputs(n, masked, seed = 0):
    torch.manual_seed(seed)
    hidden = torch.randn(2, n, 8, requires_grad = True)
    weight = (torch.randn(13, 8) * 0.3).requires_grad_()
    labels = torch.randint(0, 13, (2, n))
    if masked:
        labels[:, : n // 2] = -100      # prompt tokens
        labels[1, -1] = -100
    return hidden, weight, labels


def _fp64_reference(hidden, weight, labels, divisor):
    logits = hidden.double() @ weight.double().t()
    shift = F.pad(labels, (0, 1), value = -100)[..., 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), shift.reshape(-1), ignore_index = -100, reduction = "sum")
    return loss / divisor


def _loss(hidden, weight, labels):
    from unsloth_zoo.cpu.ops import chunked_linear_cross_entropy
    return chunked_linear_cross_entropy(hidden, weight, labels, num_items_in_batch = 8, chunk_rows = 4)


def _check_against_fp64(out, hidden, weight, labels):
    h = hidden.detach().double().requires_grad_()
    w = weight.detach().double().requires_grad_()
    ref = _fp64_reference(h, w, labels, 8)
    ref.backward()
    tol = dict(atol = 1e-5, rtol = 1e-5)
    torch.testing.assert_close(out.double(), ref, **tol)
    torch.testing.assert_close(hidden.grad.double(), h.grad, **tol)
    torch.testing.assert_close(weight.grad.double(), w.grad, **tol)


@pytest.mark.parametrize("masked", [True, False])
@pytest.mark.parametrize("dynamic", [False, True])
def test_chunked_ce_compiles_fullgraph_without_breaks(masked, dynamic):
    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(_loss, fullgraph = True, dynamic = dynamic, backend = "aot_eager")
    lengths = [7, 9, 7] if dynamic else [7]
    for n in lengths:
        hidden, weight, labels = _inputs(n, masked, seed = n)
        if dynamic:
            torch._dynamo.mark_dynamic(hidden, 1)
            torch._dynamo.mark_dynamic(labels, 1)
        out = compiled(hidden, weight, labels)
        out.backward()
        _check_against_fp64(out, hidden, weight, labels)
    assert sum(counters["graph_break"].values()) == 0, dict(counters["graph_break"])
    # Chunk loop specialises on row count: at most one graph per length.
    assert counters["stats"]["unique_graphs"] <= len(set(lengths))


def test_eager_still_skips_masked_rows(monkeypatch):
    from unsloth_zoo.cpu import ops
    seen = []
    apply = ops._ChunkedLinearCrossEntropy.apply

    def spy(hidden, *args):
        seen.append(hidden.shape[0])
        return apply(hidden, *args)

    monkeypatch.setattr(ops._ChunkedLinearCrossEntropy, "apply", spy)
    hidden, weight, labels = _inputs(9, masked = True)
    out = _loss(hidden, weight, labels)
    out.backward()
    shift = F.pad(labels, (0, 1), value = -100)[..., 1:]
    assert seen == [int((shift != -100).sum())] and seen[0] < shift.numel()
    _check_against_fp64(out, hidden, weight, labels)
