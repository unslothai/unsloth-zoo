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

"""A compiled hooked decoder layer traces, records real input addresses and replays. CPU only."""

import pytest
import torch

import unsloth_zoo.routing_replay as rr


@pytest.fixture
def replay(monkeypatch):
    monkeypatch.setenv(rr.ROUTING_REPLAY_ENV, "1")
    rr.reset_routing_replay()
    torch._dynamo.reset()
    yield rr
    torch._dynamo.reset()
    rr.reset_routing_replay()


class Qwen3MoeTopKRouter(torch.nn.Module):
    norm_topk_prob = False

    def forward(self, x):
        probs = x.softmax(-1)
        weights, ids = probs.topk(2, -1)
        return x, weights, ids


class _Layer(torch.nn.Module):
    def __init__(self, graph_break = False):
        super().__init__()
        self.router = Qwen3MoeTopKRouter()
        self.graph_break = graph_break

    def forward(self, x):
        if self.graph_break:
            # Unsloth MoE blocks run eagerly: router hook outside the pre hook's graph.
            torch._dynamo.graph_break()
        _, weights, ids = self.router(x)
        return x.sin() * weights.sum(-1, keepdim = True), ids


class _Decoder(torch.nn.Module):
    def __init__(self, graph_break = False):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(16, 8)
        self.layers = torch.nn.ModuleList([_Layer(graph_break)])

    def forward(self, input_ids):
        return self.layers[0](self.embed_tokens(input_ids))


class _Model(torch.nn.Module):
    def __init__(self, graph_break = False):
        super().__init__()
        self.model = _Decoder(graph_break)

    def forward(self, input_ids):
        return self.model(input_ids)


def _addresses(layer_input):
    # The recorded identity: nonzero input addresses (an int64 tensor; a set before it).
    if layer_input is None:
        return None
    values = layer_input.tolist() if torch.is_tensor(layer_input) else layer_input
    return tuple(sorted(a for a in values if a != 0))


@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
@pytest.mark.parametrize("fullgraph, dynamic, graph_break", [
    (False, None, False), (False, None, True), (False, True, True), (True, False, False), (True, True, False),
])
def test_compiled_layer_with_pre_hook_traces_and_never_sees_old_inputs(replay, monkeypatch, backend, fullgraph, dynamic, graph_break):
    seen = []

    def recomputed_routes(layer_index, ids2, signature = None, layer_input = None):
        seen.append(layer_input)
        return None

    monkeypatch.setattr(replay, "_recomputed_routes", recomputed_routes)
    monkeypatch.setattr(replay, "_input_signature", lambda args, tokens: None)
    model = _Model(graph_break)
    assert replay.install_routing_replay(model) == 1
    replay._STATE.pending = [{}]
    replay._STATE.in_forward = False
    layer = model.model.layers[0]
    fn = torch.compile(lambda x: layer(x), backend = backend, fullgraph = fullgraph, dynamic = dynamic)
    xs = []  # kept alive so no two calls share an address
    for n in (3, 5, 7, 3):
        x = torch.randn(n, 8, requires_grad = True)
        if dynamic:
            torch._dynamo.mark_dynamic(x, 0)
        xs.append(x)
        out, _ = fn(x)
        out.sum().backward()
    assert len(seen) == len(xs)
    for x, addresses in zip(xs, seen):
        # With or without a graph break: this call's runtime addresses, never None or a stale call's.
        assert _addresses(addresses) == (x.data_ptr(),)


@pytest.mark.parametrize("fullgraph", [False, True])
def test_compiled_layer_still_replays_forced_experts(replay, fullgraph):
    model = _Model()
    assert replay.install_routing_replay(model) == 1
    layer = model.model.layers[0]
    x = torch.randn(4, 8)
    _, natural = layer(x)
    forced = (natural + 3) % 8
    replay._STATE.in_forward = True
    replay._STATE.current = forced.unsqueeze(1)  # [tokens, layers, top_k]
    _, ids = layer(x)  # eager call caches the weight formula
    assert torch.equal(ids.sort(-1).values, forced.sort(-1).values)
    fn = torch.compile(lambda x: layer(x), backend = "eager", fullgraph = fullgraph)
    _, ids = fn(x)
    assert torch.equal(ids.sort(-1).values, forced.sort(-1).values)


def _hooked_layer(replay, graph_break = False):
    model = _Model(graph_break)
    assert replay.install_routing_replay(model) == 1
    layer = model.model.layers[0]
    layer.router._unsloth_replay_formula = "softmax"
    # Built eagerly, as the first eager forward does: a fullgraph region cannot create a Generator.
    replay._signature_projection(8, torch.device("cpu"))
    return layer


@pytest.mark.parametrize("backend", ["eager", "aot_eager", "inductor"])
@pytest.mark.parametrize("graph_break", [False, True])
def test_compiled_layer_records_this_calls_input_addresses(replay, backend, graph_break):
    # The forward record must carry the runtime input addresses: traced together with the router
    # hook (one graph) and on Inductor, which drops an op whose output nothing uses.
    layer = _hooked_layer(replay, graph_break)
    fn = torch.compile(lambda x: layer(x), backend = backend, fullgraph = not graph_break)
    replay._STATE.in_forward = True
    xs = []
    for n in (3, 3, 5):
        x = torch.randn(n, 8, requires_grad = True)
        xs.append(x)
        replay._STATE.current = torch.zeros(n, 1, 2, dtype = torch.long)
        replay._STATE.record = {}
        replay._STATE.pending.append(replay._STATE.record)
        out, _ = fn(x)
        out.sum().backward()
        assert _addresses(replay._STATE.record[0][4]) == (x.data_ptr(),)


@pytest.mark.parametrize("backend", [None, "eager", "aot_eager", "inductor"])
@pytest.mark.parametrize("graph_break", [False, True])
def test_reentrant_recompute_of_identical_inputs_keeps_forced_experts(replay, backend, graph_break):
    # Two separately allocated, numerically identical inputs forced to different experts: only
    # their addresses tell the recomputes apart. Natural routing there changes the gradients.
    from torch.utils.checkpoint import checkpoint

    def run(checkpointing):
        replay.reset_routing_replay()
        torch._dynamo.reset()
        layer = _hooked_layer(replay, graph_break)
        fn = layer if backend is None else torch.compile(
            lambda x: layer(x), backend = backend, fullgraph = not graph_break,
        )
        xs = [torch.arange(1.0, 9.0).reshape(1, 8).requires_grad_() for _ in range(2)]
        losses = []
        for x, forced in zip(xs, ([[0, 1]], [[2, 3]])):
            replay._STATE.in_forward = True
            replay._STATE.current = torch.tensor(forced).unsqueeze(1)
            replay._STATE.record = {}
            replay._STATE.pending.append(replay._STATE.record)
            out, ids = checkpoint(fn, x, use_reentrant = True) if checkpointing else fn(x)
            assert ids.sort(-1).values.tolist() == forced
            losses.append((out * torch.arange(1.0, 9.0)).sum())
        replay._STATE.current = None
        replay._STATE.record = None
        replay._STATE.in_forward = False
        sum(losses).backward()
        return torch.stack([x.grad for x in xs])

    reference = run(checkpointing = False)
    assert not torch.allclose(reference[0], reference[1])
    torch.testing.assert_close(run(checkpointing = True), reference, rtol = 0, atol = 0)
