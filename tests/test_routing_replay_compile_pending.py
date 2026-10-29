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


"""A cold fullgraph layer traces, and the pending forward count is not a Dynamo guard. CPU only."""

import pytest
import torch
import torch._dynamo.utils as dynamo_utils
from torch.utils.checkpoint import checkpoint

import unsloth_zoo.routing_replay as rr


@pytest.fixture
def replay(monkeypatch):
    monkeypatch.setenv(rr.ROUTING_REPLAY_ENV, "1")
    rr.reset_routing_replay()
    rr._SIGNATURE_PROJECTIONS.clear()
    torch._dynamo.reset()
    yield rr
    torch._dynamo.reset()
    rr.reset_routing_replay()


class Qwen3MoeTopKRouter(torch.nn.Module):
    norm_topk_prob = False

    def forward(self, x):
        weights, ids = x.softmax(-1).topk(2, -1)
        return x, weights, ids


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.router = Qwen3MoeTopKRouter()

    def forward(self, x):
        _, weights, ids = self.router(x)
        return x.sin() * weights.sum(-1, keepdim = True), ids


class _Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(16, 8)
        self.layers = torch.nn.ModuleList([_Layer()])


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Decoder()


def _layer(replay, formula = "softmax"):
    model = _Model()
    assert replay.install_routing_replay(model) == 1
    layer = model.model.layers[0]
    layer.router._unsloth_replay_formula = formula
    return layer


def _forward(replay, fn, x, forced, record = True):
    replay._STATE.in_forward = True
    replay._STATE.current = None if forced is None else torch.tensor(forced).unsqueeze(1)
    if replay._STATE.recomputed:
        replay._STATE.pending.clear()
        replay._STATE.recomputed = False
    replay._STATE.record = {}
    replay._STATE.pending.append(replay._STATE.record)
    replay._STATE.has_pending = True
    out, ids = checkpoint(fn, x, use_reentrant = True)
    replay._STATE.in_forward = False
    replay._STATE.record = None
    return out, ids


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_cold_fullgraph_layer_traces_without_eager_warm_up(replay, backend):
    layer = _layer(replay)
    assert replay._SIGNATURE_PROJECTIONS == {}
    fn = torch.compile(lambda x: layer(x), backend = backend, fullgraph = True)
    x = torch.randn(4, 8, requires_grad = True)
    out, ids = _forward(replay, fn, x, [[0, 1]] * 4)
    assert ids.sort(-1).values.tolist() == [[0, 1]] * 4
    out.sum().backward()
    assert torch.isfinite(x.grad).all()
    generator = torch.Generator().manual_seed(0x5EED)
    expected = torch.randn(8, 2, generator = generator, dtype = torch.float32)
    assert torch.equal(replay._SIGNATURE_PROJECTIONS[(8, "cpu")], expected)


def _steps(replay, fn, counts):
    grads, graphs = [], []
    for count in counts:
        before = dynamo_utils.counters["stats"]["unique_graphs"]
        xs = [torch.arange(1.0, 9.0).reshape(1, 8).requires_grad_() for _ in range(count)]
        losses = []
        for k, x in enumerate(xs):
            forced = [[k % 8, (k + 3) % 8]]
            out, ids = _forward(replay, fn, x, forced)
            assert ids.sort(-1).values.tolist() == [sorted(forced[0])]
            losses.append((out * torch.arange(1.0, 9.0)).sum())
        replay._STATE.current = None
        sum(losses).backward()
        grads.append(torch.stack([x.grad for x in xs]))
        graphs.append(dynamo_utils.counters["stats"]["unique_graphs"] - before)
    return grads, graphs


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_pending_forward_count_is_not_a_guard(replay, backend):
    counts = [1, 2, 3, 4, 5, 6, 9, 2]
    reference, _ = _steps(replay, _layer(replay), counts)
    replay.reset_routing_replay()
    torch._dynamo.reset()
    dynamo_utils.counters.clear()
    layer = _layer(replay)
    fn = torch.compile(lambda x: layer(x), backend = backend, fullgraph = True)
    grads, graphs = _steps(replay, fn, counts)
    # Forward + recompute graphs, and one more once _STATE.changed holds a tensor; none after.
    assert sum(graphs) <= 3 and graphs[2:] == [0] * (len(counts) - 2), graphs
    exact = dict(rtol = 0, atol = 0) if backend == "aot_eager" else {}
    for got, want in zip(grads, reference):
        torch.testing.assert_close(got, want, **exact)
    assert not torch.equal(reference[1][0], reference[1][1])


def test_compiled_recompute_of_a_layer_that_never_replayed_keeps_natural_routing(replay):
    # No formula yet (no forward replayed this layer): the fullgraph recompute must not probe it.
    layer = _layer(replay, formula = None)
    fn = torch.compile(lambda x: layer(x), backend = "aot_eager", fullgraph = True)
    x = torch.randn(3, 8, requires_grad = True)
    out, ids = _forward(replay, fn, x, None)
    out.sum().backward()
    y = x.detach().clone().requires_grad_()
    want_out, want_ids = layer(y)
    want_out.sum().backward()
    assert torch.equal(ids, want_ids)
    torch.testing.assert_close(x.grad, y.grad, rtol = 0, atol = 0)
    assert layer.router._unsloth_replay_formula is None


@pytest.mark.parametrize("forwards", [
    [(2, True), (3, False)], [(3, False), (2, True)], [(2, True), (2, False)], [(2, True), (2, True)],
])
def test_non_reentrant_compiled_recompute_matches_eager(replay, forwards):
    def run(compiled, use_reentrant):
        replay.reset_routing_replay()
        torch._dynamo.reset()
        layer = _layer(replay)
        fn = torch.compile(lambda x: layer(x), backend = "aot_eager", fullgraph = True) if compiled else layer
        xs, losses = [], []
        for k, (rows, replayed) in enumerate(forwards):
            x = torch.randn(rows, 8, generator = torch.Generator().manual_seed(k)).requires_grad_()
            xs.append(x)
            replay._STATE.in_forward = True
            replay._STATE.current = torch.tensor([[k, k + 3]] * rows).unsqueeze(1) if replayed else None
            replay._STATE.record = {}
            replay._STATE.pending.append(replay._STATE.record)
            replay._STATE.has_pending = True
            out, _ = checkpoint(fn, x, use_reentrant = use_reentrant)
            replay._STATE.in_forward = False
            replay._STATE.record = None
            losses.append((out * torch.arange(1.0, 9.0)).sum())
        replay._STATE.current = None
        sum(losses).backward()
        return [x.grad for x in xs]

    reference = run(compiled = False, use_reentrant = True)
    for got, want in zip(run(compiled = True, use_reentrant = False), reference):
        torch.testing.assert_close(got, want, rtol = 0, atol = 0)


@pytest.mark.parametrize("natural_first", [False, True])
def test_non_reentrant_natural_forward_with_identical_inputs_keeps_its_own_routing(replay, natural_first):
    # Identical values, one replayed and one natural: only input addresses tell them apart.
    values = torch.arange(8.0).reshape(1, 8)  # natural top-2: experts [7, 6]
    order = [None, [[0, 1]]] if natural_first else [[[0, 1]], None]

    def run(compiled):
        replay.reset_routing_replay()
        torch._dynamo.reset()
        layer = _layer(replay)
        fn = torch.compile(lambda x: layer(x), backend = "aot_eager", fullgraph = True) if compiled else layer
        xs, ids_out, losses = [], [], []
        for forced in order:
            x = values.clone().requires_grad_()
            xs.append(x)
            replay._STATE.in_forward = True
            replay._STATE.current = None if forced is None else torch.tensor(forced).unsqueeze(1)
            if compiled:
                replay._STATE.record = {}
                replay._STATE.pending.append(replay._STATE.record)
                replay._STATE.has_pending = True
                out, ids = checkpoint(fn, x, use_reentrant = False)
            else:
                out, ids = fn(x)  # no checkpoint: the true gradients
            replay._STATE.in_forward = False
            replay._STATE.record = None
            ids_out.append(ids.sort(-1).values.tolist())
            losses.append((out * torch.arange(1.0, 9.0)).sum())
        replay._STATE.current = None
        sum(losses).backward()
        return ids_out, [x.grad for x in xs]

    want_ids, want = run(compiled = False)
    assert sorted(map(tuple, (i[0] for i in want_ids))) == [(0, 1), (6, 7)]
    got_ids, got = run(compiled = True)
    assert got_ids == want_ids
    assert not torch.equal(want[0], want[1])
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol = 0, atol = 0)
