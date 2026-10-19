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


def test_nonreentrant_recompute_of_a_natural_forward_adds_no_ops(replay):
    # No rollout matches, so the forward keeps its natural routing. The non-reentrant checkpoint
    # recompute must run the same ops, or torch raises a CheckpointError.
    model = _model()
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
    _install(model, 2)
    rr.record_routed_experts([_request([5, 6], [7, 8], np.zeros((4, 2, TOP_K)) + [[0, 1]])])
    model(input_ids = torch.tensor([[11, 12, 13, 14]])).logits.float().square().mean().backward()
    assert model.model.layers[0].mlp.gate.weight.grad is not None


def test_unnormalised_router_is_never_read_as_renormalised(replay):
    # norm_topk_prob=False, but the natural top-k mass is ~1: the weights of forced experts must
    # stay plain softmax probabilities, not be renormalised to sum to one.
    from transformers import Qwen3MoeConfig

    config = Qwen3MoeConfig(hidden_size = 4, num_experts = 4, num_experts_per_tok = 2, norm_topk_prob = False)
    router = Qwen3MoeTopKRouter(config)
    with torch.no_grad():
        router.weight.copy_(torch.eye(4))
    router.register_forward_hook(rr._router_hook(0))
    rr._STATE.current = torch.tensor([[[2, 3]]])
    rr._STATE.in_forward = True
    logits, weights, ids = router(torch.tensor([[8.0, 7.0, 0.0, -1.0]]))
    assert ids.tolist() == [[2, 3]]
    expected = logits.double().softmax(-1).gather(-1, ids)
    torch.testing.assert_close(weights.double(), expected, rtol = 1e-5, atol = 1e-7)
    assert router._unsloth_replay_formula == "softmax"


def test_compact_moe_rows_map_to_their_decoder_layers(replay):
    # vLLM 0.21 numbers routed experts by MoE layer only (moe_layer_id), so a model with a dense
    # first layer returns [tokens, 2, k] for decoder layers 1 and 2.
    model = _model(num_layers = 3, mlp_only_layers = [0])
    prompt, completion = [5, 6, 7], [9, 10, 11]
    ids = prompt + completion
    captured = {}
    hooks = [
        model.model.layers[i].mlp.gate.register_forward_hook(
            lambda m, a, out, i = i: captured.__setitem__(i, out[2].detach().clone())
        )
        for i in (1, 2)
    ]
    with torch.no_grad():
        model(input_ids = torch.tensor([ids]))
    for h in hooks:
        h.remove()
    natural = torch.stack([captured[1], captured[2]], dim = 1)  # [T, 2 MoE layers, k]
    forced = (natural + 3) % NUM_EXPERTS
    forced[-1] = natural[-1]

    _install(model, 2)
    rr.record_routed_experts([_request(prompt, completion, forced.numpy())])
    seen = {}
    hooks = [
        model.model.layers[i].mlp.gate.register_forward_hook(
            lambda m, a, out, i = i: seen.__setitem__(i, out[2].detach().clone())
        )
        for i in (1, 2)
    ]
    with torch.no_grad():
        model(input_ids = torch.tensor([ids]))
    for h in hooks:
        h.remove()
    assert torch.equal(seen[1][:-1].sort(-1).values, forced[:-1, 0].long().sort(-1).values)
    assert torch.equal(seen[2][:-1].sort(-1).values, forced[:-1, 1].long().sort(-1).values)
    assert rr.routing_replay_stats()["router_calls_natural"] == 0


def test_input_signature_tells_apart_inputs_with_equal_token_sums():
    # Zero-centred router inputs (LayerNorm) give every token the same sum, so a sum cannot tell
    # two forwards apart; the projections must, and must repeat bit for bit on identical inputs.
    torch.manual_seed(0)
    a = torch.randn(5, 32)
    a = a - a.mean(-1, keepdim = True)
    b = a[:, torch.randperm(32)]
    b = b.flip(-1)
    assert torch.allclose(a.sum(-1), b.sum(-1), atol = 1e-5)
    sig_a = rr._input_signature((a,), 5)
    assert sig_a.shape == (5, 2) and sig_a.dtype == torch.float32
    assert torch.equal(sig_a, rr._input_signature((a.clone(),), 5))
    assert not torch.equal(sig_a, rr._input_signature((b,), 5))
    assert torch.equal(rr._input_signature((a.to(torch.bfloat16),), 5), rr._input_signature((a.to(torch.bfloat16),), 5))


def test_input_signature_stays_fp32_under_autocast():
    torch.manual_seed(1)
    a = torch.randn(4, 32)
    plain = rr._input_signature((a,), 4)
    with torch.autocast(device_type = "cpu", dtype = torch.bfloat16):
        cast = rr._input_signature((a,), 4)
    assert cast.dtype == torch.float32
    assert torch.equal(cast, plain)


def test_softmax_renorm_survives_underflowed_forced_experts():
    # Forced experts with logits ~200 below the natural top-k underflow in a full-vocab softmax:
    # gather-then-renormalise gave 0 / 0 = NaN. Softmax over the gathered logits is the same function.
    logits = torch.tensor([[200.0, 199.0, 0.0, -1.0], [1.0, 2.0, 0.5, -0.3]], requires_grad = True)
    ids = torch.tensor([[2, 3], [1, 2]])
    got = rr._formulas()["softmax_renorm"](logits, ids)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got[0], torch.softmax(torch.tensor([0.0, -1.0]), -1))
    ref = torch.softmax(logits.detach().double(), -1)[1].gather(-1, ids[1])
    torch.testing.assert_close(got[1].double(), ref / ref.sum(), rtol = 1e-6, atol = 1e-7)
    got.sum().backward()
    assert torch.isfinite(logits.grad).all()
