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


def _overflow_loss(hidden, weight, labels):
    from unsloth_zoo.cpu.ops import chunked_linear_cross_entropy
    return chunked_linear_cross_entropy(hidden, weight, labels, shift_labels = labels, num_items_in_batch = 1, chunk_rows = 4)


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_ignored_fp16_row_cannot_overflow_compiled_loss(backend):
    if backend == "inductor":
        from unsloth_zoo.cpu.loader import _cpu_compile_unavailable
        reason = _cpu_compile_unavailable()
        if reason is not None:
            pytest.skip(f"Inductor cannot build CPU kernels here: {reason}")
    # Row 0 is ignored but its fp16 logits (2.4e5) overflow: traced code keeps the row, so it must be zeroed first.
    def run(fn):
        hidden = torch.tensor([[60000.0, 60000.0], [0.1, 0.2]], dtype = torch.float16, requires_grad = True)
        weight = torch.tensor([[2.0, 2.0], [1.0, 2.0], [2.0, 1.0]], dtype = torch.float16, requires_grad = True)
        loss = fn(hidden, weight, torch.tensor([-100, 1]))
        loss.backward()
        return loss.detach(), hidden.grad, weight.grad

    eager = run(_overflow_loss)
    torch._dynamo.reset()
    counters.clear()
    compiled = run(torch.compile(_overflow_loss, fullgraph = True, backend = backend))
    assert sum(counters["graph_break"].values()) == 0, dict(counters["graph_break"])
    assert torch.isfinite(eager[0]) and abs(eager[0].item() - 1.1017700) < 1e-5
    for e, c in zip(eager, compiled):
        assert torch.isfinite(c).all(), c
        torch.testing.assert_close(c, e, atol = 1e-3, rtol = 1e-3)
    assert torch.equal(compiled[1][0], torch.zeros_like(compiled[1][0]))


def _loss_tensor_count(hidden, weight, labels, count):
    from unsloth_zoo.cpu.ops import chunked_linear_cross_entropy
    return chunked_linear_cross_entropy(hidden, weight, labels, num_items_in_batch = count, chunk_rows = 4)


def _loss_no_count(hidden, weight, labels):
    from unsloth_zoo.cpu.ops import chunked_linear_cross_entropy
    return chunked_linear_cross_entropy(hidden, weight, labels, chunk_rows = 4)


@pytest.mark.parametrize("count", ["tensor", "none"])
def test_chunked_ce_fullgraph_with_tensor_or_omitted_count(count):
    # The HF Trainer passes num_items_in_batch as a tensor: no .item() may run inside the graph.
    torch._dynamo.reset()
    counters.clear()
    for n in [7, 9]:
        hidden, weight, labels = _inputs(n, masked = True, seed = n)
        shift = F.pad(labels, (0, 1), value = -100)[..., 1:]
        items = int((shift != -100).sum())
        if count == "tensor":
            fn = torch.compile(_loss_tensor_count, fullgraph = True, backend = "aot_eager")
            out = fn(hidden, weight, labels, torch.tensor(items))
        else:
            fn = torch.compile(_loss_no_count, fullgraph = True, backend = "aot_eager")
            out = fn(hidden, weight, labels)
        out.backward()
        h = hidden.detach().double().requires_grad_()
        w = weight.detach().double().requires_grad_()
        ref = _fp64_reference(h, w, labels, items)
        ref.backward()
        tol = dict(atol = 1e-5, rtol = 1e-5)
        torch.testing.assert_close(out.double(), ref, **tol)
        torch.testing.assert_close(hidden.grad.double(), h.grad, **tol)
        torch.testing.assert_close(weight.grad.double(), w.grad, **tol)
    assert sum(counters["graph_break"].values()) == 0, dict(counters["graph_break"])


@pytest.mark.parametrize("compiled", [False, True])
def test_fully_ignored_batch_edge_cases_with_tensor_divisors(compiled):
    # Omitted count: NaN loss like mean CE, zero gradient. Zero tensor count: loss 0, zero gradient.
    wrap = (lambda f: torch.compile(f, fullgraph = True, backend = "aot_eager")) if compiled else (lambda f: f)
    torch._dynamo.reset()
    labels = torch.full((1, 5), -100)
    hidden = torch.randn(1, 5, 8, requires_grad = True)
    weight = torch.randn(13, 8)
    loss = wrap(_loss_no_count)(hidden, weight, labels)
    assert torch.isnan(loss)
    loss.backward()
    assert torch.equal(hidden.grad, torch.zeros_like(hidden.grad))
    hidden.grad = None
    loss = wrap(_loss_tensor_count)(hidden, weight, labels, torch.tensor(0))
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(hidden.grad, torch.zeros_like(hidden.grad))


def test_fully_ignored_fp16_overflow_stays_finite_in_eager():
    # Eager skips compaction when no label is valid (a masked accumulation micro-batch), so the chunk loop sees the row.
    hidden = torch.tensor([[60000.0, 60000.0], [0.1, 0.2]], dtype = torch.float16, requires_grad = True)
    weight = torch.tensor([[2.0, 2.0], [1.0, 2.0], [2.0, 1.0]], dtype = torch.float16)
    labels = torch.tensor([-100, -100])
    from unsloth_zoo.cpu.ops import chunked_linear_cross_entropy
    loss = chunked_linear_cross_entropy(hidden, weight, labels, shift_labels = labels, num_items_in_batch = 4)
    loss.backward()
    assert loss.item() == 0.0
    assert torch.equal(hidden.grad, torch.zeros_like(hidden.grad))
