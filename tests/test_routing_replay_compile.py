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

"""MoE routing replay under torch.compile: router hooks, and probability routers whose forced experts underflow; runs on CPU only."""

import pytest
import torch

import unsloth_zoo.routing_replay as rr


@pytest.fixture
def replay(monkeypatch):
    monkeypatch.setenv(rr.ROUTING_REPLAY_ENV, "1")
    rr.reset_routing_replay()
    yield rr
    rr.reset_routing_replay()


class _LogitsRouter(torch.nn.Module):
    norm_topk_prob = True

    def forward(self, x):
        ids = x.topk(2, -1).indices
        return x, x.gather(-1, ids).softmax(-1), ids


class _ProbsRouter(torch.nn.Module):
    # transformers 5.2 - 5.7 Qwen MoE routers: the "logits" output holds softmax probabilities.
    norm_topk_prob = True

    def forward(self, x):
        probs = x.float().softmax(-1)
        weights, ids = probs.topk(2, -1)
        return probs, weights / weights.sum(-1, keepdim = True), ids


def test_replayed_router_does_not_recompile_every_call(replay):
    replay._STATE.in_forward = True
    replay._STATE.current = torch.tensor([[[2, 3]]])
    router = _LogitsRouter()
    router.register_forward_hook(replay._router_hook(0))
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    router(x)  # eager call picks and caches the weight formula
    torch._dynamo.reset()
    compiled = torch.compile(router, backend = "eager")
    frames = []
    for _ in range(4):
        before = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        _, weights, ids = compiled(x)
        frames.append(torch._dynamo.utils.counters["stats"]["unique_graphs"] - before)
        assert ids.tolist() == [[2, 3]]
        torch.testing.assert_close(weights, x[:, [2, 3]].softmax(-1))
    torch._dynamo.reset()
    # Mutable replay counters must not become Dynamo guards: only the first call may compile.
    assert frames[1:] == [0, 0, 0], frames


def test_underflowed_forced_probabilities_keep_natural_routing(replay):
    replay._STATE.in_forward = True
    # Forced experts 2 and 3 sit ~200 below the natural top-2: their fp32 probabilities are 0.
    replay._STATE.current = torch.tensor([[[2, 3]], [[1, 2]]])
    router = _ProbsRouter()
    router.register_forward_hook(replay._router_hook(0))
    x = torch.tensor([[0.0, -1.0, -200.0, -201.0], [0.0, -1.0, -1.5, -9.0]], requires_grad = True)
    probs, weights, ids = router(x)
    assert router._unsloth_replay_formula == "probs_renorm"
    assert torch.isfinite(weights).all(), weights
    # Row 0 cannot be renormalised (0 / 0): it keeps its natural routing and weights.
    assert ids[0].tolist() == [0, 1]
    natural = x[0].detach().softmax(-1)[[0, 1]]
    torch.testing.assert_close(weights[0], natural / natural.sum())
    # Row 1 still replays.
    assert ids[1].tolist() == [1, 2]
    forced = x[1].detach().softmax(-1)[[1, 2]]
    torch.testing.assert_close(weights[1], forced / forced.sum())
    weights.square().sum().backward()
    assert torch.isfinite(x.grad).all(), x.grad


def test_hooked_decoder_inside_compiled_model_does_not_recompile(replay):
    # An outer torch.compile also traces the decoder pre hook: its forward counter must not become a guard either.
    class _Decoder(torch.nn.Module):
        def forward(self, inputs_embeds):
            return inputs_embeds * 2

    decoder = _Decoder()
    decoder.register_forward_pre_hook(replay._decoder_pre_hook, with_kwargs = True)
    decoder.register_forward_hook(replay._decoder_post_hook)
    x = torch.ones(1, 3, 4)
    torch._dynamo.reset()
    compiled = torch.compile(decoder, backend = "eager")
    frames = []
    for _ in range(4):
        before = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        torch.testing.assert_close(compiled(inputs_embeds = x), x * 2)
        frames.append(torch._dynamo.utils.counters["stats"]["unique_graphs"] - before)
    torch._dynamo.reset()
    assert frames[1:] == [0, 0, 0], frames


@pytest.mark.parametrize("far", [90.0, 200.0])
def test_subnormal_forced_probabilities_keep_natural_routing(replay, far):
    # Logits ~90 below the natural top-2 give subnormal fp32 probabilities; renormalising gives finite weights but a
    # backward of ~1 / total that overflows to NaN.
    replay._STATE.in_forward = True
    replay._STATE.current = torch.tensor([[[2, 3]]])
    router = _ProbsRouter()
    router.register_forward_hook(replay._router_hook(0))
    x = torch.tensor([[0.0, -1.0, -far, -far - 1.0]], requires_grad = True)
    _, weights, ids = router(x)
    assert ids[0].tolist() == [0, 1]
    natural = x[0].detach().softmax(-1)[[0, 1]]
    torch.testing.assert_close(weights[0], natural / natural.sum())
    weights.square().sum().backward()
    assert torch.isfinite(x.grad).all(), x.grad
