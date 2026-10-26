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

"""MoE routing replay: a compiled region holding a hooked decoder layer (default, fullgraph,
dynamic) traces, still records the layer's real input addresses and still replays. CPU only."""

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
            # Unsloth's MoE blocks run eagerly (compiler disabled): the router hook is not in the
            # graph that holds the layer pre hook.
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


@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
@pytest.mark.parametrize("fullgraph, dynamic, graph_break", [
    (False, None, False), (False, None, True), (False, True, True), (True, False, False), (True, True, False),
])
def test_compiled_layer_with_pre_hook_traces_and_never_sees_old_inputs(replay, monkeypatch, backend, fullgraph, dynamic, graph_break):
    seen = []

    def recomputed_routes(layer_index, ids2, signature = None, layer_input = None):
        seen.append(None if layer_input is None else tuple(sorted(layer_input)))
        return None

    # A checkpoint recompute: the router hook hands the layer input addresses to the matcher.
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
        if graph_break:
            # The pre hook ran before the router's graph: the matcher gets this call's input.
            assert addresses == (x.data_ptr(),)
        else:
            # One graph: the pre hook's op writes at run time, after the router hook was traced.
            assert addresses in (None, (x.data_ptr(),))


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
    _, ids = layer(x)  # the eager call picks and caches the weight formula
    assert torch.equal(ids.sort(-1).values, forced.sort(-1).values)
    fn = torch.compile(lambda x: layer(x), backend = "eager", fullgraph = fullgraph)
    _, ids = fn(x)
    assert torch.equal(ids.sort(-1).values, forced.sort(-1).values)
