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

"""MoE routing replay: checkpoint recompute identity, router normalisation, sparse MoE layers. CPU only."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
try:
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter
except Exception:
    pytest.skip("needs transformers >= 5 router modules", allow_module_level = True)

import unsloth_zoo.routing_replay as rr

PAD = 0
NUM_EXPERTS = 8
TOP_K = 2


@pytest.fixture
def replay(monkeypatch):
    monkeypatch.setenv(rr.ROUTING_REPLAY_ENV, "1")
    rr.reset_routing_replay()
    yield rr
    rr.reset_routing_replay()


def _model(num_layers = 2, mlp_only_layers = None, seed = 0):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(seed)
    config = Qwen3MoeConfig(
        vocab_size = 64, hidden_size = 32, intermediate_size = 64, moe_intermediate_size = 16,
        num_hidden_layers = num_layers, num_attention_heads = 4, num_key_value_heads = 2, head_dim = 8,
        num_experts = NUM_EXPERTS, num_experts_per_tok = TOP_K, norm_topk_prob = True,
        pad_token_id = PAD, mlp_only_layers = list(mlp_only_layers or []),
    )
    model = Qwen3MoeForCausalLM(config).float().eval()
    for layer in model.model.layers:
        gate = getattr(layer.mlp, "gate", None)
        if isinstance(gate, Qwen3MoeTopKRouter):
            gate.weight.data.normal_(0, 0.5)
    return model


def _request(prompt, completion, routed):
    # vLLM returns routing for every forwarded token: all but the last sampled one.
    return SimpleNamespace(
        prompt_token_ids = list(prompt),
        outputs = [SimpleNamespace(token_ids = list(completion), routed_experts = np.asarray(routed, dtype = np.uint8)[:-1])],
    )


def _install(model, routers):
    assert rr.install_routing_replay(model, None, SimpleNamespace(pad_token_id = PAD)) == routers


def _grads(model):
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def test_recompute_replays_forwards_with_equal_natural_routes_but_different_replays(replay):
    # Two different sequences whose first layer routes every token identically, replayed to
    # different experts: the recompute must replay each forward's own routes, so checkpointed
    # gradients equal the plain ones.
    model = _model()
    batch = torch.randint(1, 64, (1024, 2), generator = torch.Generator().manual_seed(749))
    captured = {}
    handle = model.model.layers[0].mlp.gate.register_forward_hook(
        lambda m, a, out: captured.__setitem__("ids", out[2].detach().clone())
    )
    with torch.no_grad():
        model(input_ids = batch)
    handle.remove()
    routes = captured["ids"].reshape(1024, 2, TOP_K)
    seen, pair = {}, None
    for i, route in enumerate(routes):
        key = tuple(route.sort(-1).values.flatten().tolist())
        if key in seen and not torch.equal(batch[i], batch[seen[key]]):
            pair = (seen[key], i)
            break
        seen[key] = i
    assert pair is not None
    sequences = [batch[i].tolist() for i in pair]
    forced_rows = [
        torch.tensor([[[4, 5], [4, 5]], [[4, 5], [4, 5]]]),
        torch.tensor([[[6, 7], [6, 7]], [[6, 7], [6, 7]]]),
    ]

    def run(checkpointing):
        m = _model()
        m.train()
        if checkpointing:
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": True})
        rr.reset_routing_replay()
        _install(m, 2)
        rr.record_routed_experts([_request(s[:1], s[1:], f.numpy()) for s, f in zip(sequences, forced_rows)])
        losses = [m(input_ids = torch.tensor([s])).logits.float().square().mean() for s in sequences]
        if checkpointing:
            a, b = rr._STATE.pending
            assert torch.equal(a[0][0], b[0][0]) and not torch.equal(a[0][1], b[0][1])
        sum(losses).backward()
        return torch.stack([l.detach() for l in losses]), _grads(m)

    losses_plain, grads_plain = run(False)
    losses_gc, grads_gc = run(True)
    torch.testing.assert_close(losses_gc, losses_plain, rtol = 1e-5, atol = 1e-6)
    assert grads_gc.keys() == grads_plain.keys()
    for name in grads_plain:
        torch.testing.assert_close(grads_gc[name], grads_plain[name], rtol = 1e-4, atol = 1e-6, msg = name)
