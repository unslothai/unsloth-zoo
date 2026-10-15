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

"""MoE routing replay (UNSLOTH_GRPO_ROUTING_REPLAY): forcing the rollout's expert ids. CPU only."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

transformers = pytest.importorskip("transformers")
try:
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter  # noqa: F401
except Exception:
    pytest.skip("needs transformers >= 5 router modules", allow_module_level = True)

import unsloth_zoo.routing_replay as rr

PAD = 0
NUM_LAYERS = 2
NUM_EXPERTS = 8
TOP_K = 2


def _model(seed = 0):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(seed)
    config = Qwen3MoeConfig(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 64,
        moe_intermediate_size = 16,
        num_hidden_layers = NUM_LAYERS,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 8,
        num_experts = NUM_EXPERTS,
        num_experts_per_tok = TOP_K,
        norm_topk_prob = True,
        pad_token_id = PAD,
    )
    model = Qwen3MoeForCausalLM(config).float().eval()
    # Spread the router logits so forcing other experts visibly changes the output.
    for layer in model.model.layers:
        layer.mlp.gate.weight.data.normal_(0, 0.5)
    return model


def _natural_ids(model, ids):
    captured = {}
    handles = []
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.mlp.gate.register_forward_hook(
            lambda m, a, out, i = i: captured.__setitem__(i, out[2].detach().clone())
        ))
    with torch.no_grad():
        model(input_ids = torch.tensor([ids]))
    for h in handles:
        h.remove()
    return torch.stack([captured[i] for i in range(NUM_LAYERS)], dim = 1)  # [T, L, k]


def _request(prompt, completion, routed):
    # vLLM returns routing for every forwarded token: all but the last sampled one.
    return SimpleNamespace(
        prompt_token_ids = list(prompt),
        outputs = [SimpleNamespace(token_ids = list(completion), routed_experts = np.asarray(routed, dtype = np.uint8)[:-1])],
    )


@pytest.fixture
def replay(monkeypatch):
    monkeypatch.setenv(rr.ROUTING_REPLAY_ENV, "1")
    rr.reset_routing_replay()
    yield rr
    rr.reset_routing_replay()


def _install(model):
    tokenizer = SimpleNamespace(pad_token_id = PAD)
    assert rr.install_routing_replay(model, None, tokenizer) == NUM_LAYERS


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(rr.ROUTING_REPLAY_ENV, raising = False)
    model = _model()
    assert rr.install_routing_replay(model, None, SimpleNamespace(pad_token_id = PAD)) == 0
    assert not model.model.layers[0].mlp.gate._forward_hooks
    assert rr.routing_replay_engine_args(model.config) == {}


def test_engine_args_need_moe(replay, monkeypatch):
    monkeypatch.setattr(rr, "vllm_supports_routed_experts", lambda: True)
    assert rr.routing_replay_engine_args(_model().config) == {"enable_return_routed_experts": True}
    assert rr.routing_replay_engine_args(SimpleNamespace(num_experts = None)) == {}


def test_matching_ids_reproduce_natural_forward_and_grad(replay):
    model = _model()
    prompt, completion = [5, 6, 7, 8], [9, 10, 11, 12, 13]
    ids = prompt + completion
    natural = _natural_ids(model, ids)

    model.zero_grad()
    ref = model(input_ids = torch.tensor([ids])).logits
    ref.sum().backward()
    ref_grad = model.model.layers[0].mlp.gate.weight.grad.clone()

    _install(model)
    rr.record_routed_experts([_request(prompt, completion, natural.numpy())])
    model.zero_grad()
    out = model(input_ids = torch.tensor([ids])).logits
    out.sum().backward()

    torch.testing.assert_close(out, ref, rtol = 1e-5, atol = 1e-6)
    torch.testing.assert_close(model.model.layers[0].mlp.gate.weight.grad, ref_grad, rtol = 1e-4, atol = 1e-6)
    stats = rr.routing_replay_stats()
    assert stats["replayed_tokens"] == len(ids)
    assert stats["router_calls_replayed"] == NUM_LAYERS
    assert stats["changed_selections"] == 0


def _reference_forced_forward(model, ids, forced):
    """Independent reference: each router takes `forced` ids and softmax-renorm weights over them."""
    originals = []
    for li, layer in enumerate(model.model.layers):
        gate = layer.mlp.gate
        originals.append(gate.forward)

        def forward(hidden_states, gate = gate, li = li):
            hidden_states = hidden_states.reshape(-1, gate.hidden_dim)
            logits = F.linear(hidden_states, gate.weight)
            probs = F.softmax(logits, dtype = torch.float, dim = -1)
            idx = forced[:, li, :].long()
            w = probs.gather(-1, idx)
            w = w / w.sum(-1, keepdim = True)
            return logits, w.to(logits.dtype), idx

        gate.forward = forward
    try:
        return model(input_ids = torch.tensor([ids])).logits
    finally:
        for layer, original in zip(model.model.layers, originals):
            layer.mlp.gate.forward = original


def test_different_ids_change_output_only_through_ids(replay):
    model = _model()
    prompt, completion = [5, 6, 7], [20, 21, 22, 23]
    ids = prompt + completion
    natural = _natural_ids(model, ids)
    forced = natural.clone()
    forced[2:5] = (forced[2:5] + 3) % NUM_EXPERTS  # new expert sets on three tokens, all layers
    forced[-1] = natural[-1]  # vLLM has no routing for the last token, so it stays natural

    with torch.no_grad():
        natural_logits = model(input_ids = torch.tensor([ids])).logits
        expected = _reference_forced_forward(model, ids, forced)

    _install(model)
    rr.record_routed_experts([_request(prompt, completion, forced.numpy())])
    with torch.no_grad():
        out = model(input_ids = torch.tensor([ids])).logits

    torch.testing.assert_close(out, expected, rtol = 1e-5, atol = 1e-6)
    assert not torch.allclose(out, natural_logits, atol = 1e-4)
    # Causal: tokens before the first forced change are untouched.
    torch.testing.assert_close(out[:, :2], natural_logits[:, :2], rtol = 1e-5, atol = 1e-6)
    assert rr.routing_replay_stats()["changed_selections"] == 3 * NUM_LAYERS


def _forced_rows(model, sequences, shift):
    rows = []
    for prompt, completion in sequences:
        natural = _natural_ids(model, prompt + completion)
        row = (natural + shift) % NUM_EXPERTS
        row[-1] = natural[-1]
        rows.append(row)
    return rows


def test_padded_packed_and_reordered_layouts_find_their_rollout(replay):
    model = _model()
    sequences = [([5, 6, 7], [30, 31, 32, 33]), ([8, 9], [40, 41, 42])]
    forced = _forced_rows(model, sequences, shift = 1)
    singles = [
        _reference_forced_forward(model, p + c, f).detach() for (p, c), f in zip(sequences, forced)
    ]

    _install(model)
    rr.record_routed_experts([_request(p, c, f.numpy()) for (p, c), f in zip(sequences, forced)])

    # Padded, reversed order: left pad on the prompt, right pad after the completion.
    a = sequences[0][0] + sequences[0][1]
    b = sequences[1][0] + sequences[1][1]
    padded = torch.tensor([[PAD] + b + [PAD], a])
    mask = (padded != PAD).long()
    position_ids = (mask.cumsum(-1) - 1).clamp(min = 0)
    with torch.no_grad():
        out = model(input_ids = padded, attention_mask = mask, position_ids = position_ids).logits
    torch.testing.assert_close(out[0, 1:1 + len(b)], singles[1][0], rtol = 1e-4, atol = 1e-5)
    torch.testing.assert_close(out[1], singles[0][0], rtol = 1e-4, atol = 1e-5)

    # Packed [1, sum L] with per-segment positions, as the GRPO sequence packing path feeds it.
    packed = torch.tensor([a + b])
    position_ids = torch.tensor([list(range(len(a))) + list(range(len(b)))])
    replayed = rr._build_replay(packed, position_ids)
    expected = torch.cat([forced[0], forced[1]]).long()
    expected[len(a) - 1] = -1
    expected[-1] = -1
    assert torch.equal(replayed, expected)


def test_unmatched_rows_keep_natural_routing(replay):
    model = _model()
    _install(model)
    rr.record_routed_experts([_request([5, 6], [7, 8], np.zeros((4, NUM_LAYERS, TOP_K)))])
    ids = [11, 12, 13, 14]
    with torch.no_grad():
        out = model(input_ids = torch.tensor([ids])).logits
    rr.reset_routing_replay()
    with torch.no_grad():
        ref = model(input_ids = torch.tensor([ids])).logits
    torch.testing.assert_close(out, ref)


def test_gradient_checkpointing_recompute_uses_replay(replay):
    sequences = [([5, 6, 7], [30, 31, 32, 33])]
    model = _model()
    forced = _forced_rows(model, sequences, shift = 2)
    ids = sequences[0][0] + sequences[0][1]

    def run(checkpointing):
        m = _model()
        m.train()
        if checkpointing:
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": True})
        m.model.embed_tokens.weight.requires_grad_(True)
        rr.reset_routing_replay()
        _install(m)
        rr.record_routed_experts([_request(*sequences[0], forced[0].numpy())])
        logits = m(input_ids = torch.tensor([ids])).logits
        logits.float().pow(2).mean().backward()
        return logits.detach(), m.model.layers[1].mlp.gate.weight.grad.clone()

    logits_plain, grad_plain = run(False)
    logits_gc, grad_gc = run(True)
    torch.testing.assert_close(logits_gc, logits_plain, rtol = 1e-5, atol = 1e-6)
    torch.testing.assert_close(grad_gc, grad_plain, rtol = 1e-4, atol = 1e-6)


def test_topk_softmax_router_is_covered_by_softmax_renorm(replay):
    class GptOssLikeRouter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(NUM_EXPERTS, 16))

        def forward(self, hidden_states):
            logits = F.linear(hidden_states, self.weight)
            top, idx = torch.topk(logits, TOP_K, dim = -1)
            return logits, F.softmax(top, dim = 1, dtype = top.dtype), idx

    router = GptOssLikeRouter()
    hidden = torch.randn(5, 16)
    forced = torch.tensor([[0, 1]] * 5)
    rr._STATE.current = forced[:, None, :].expand(5, 1, TOP_K).clone()
    logits, weights, ids = rr._router_hook(0)(router, (hidden,), router(hidden))
    assert router._unsloth_replay_formula == "softmax_renorm"
    assert torch.equal(ids, forced)
    torch.testing.assert_close(weights, F.softmax(logits.gather(-1, forced), dim = -1))


def test_record_drops_pad_and_marks_last_token(replay):
    rr._STATE.pad_token_id = PAD
    routed = np.arange(6 * NUM_LAYERS * TOP_K).reshape(6, NUM_LAYERS, TOP_K) % NUM_EXPERTS
    # pad == eos style: the final sampled token is the pad id, so the key drops it.
    rr.record_routed_experts([_request([3, 4], [5, 6, 7, PAD], routed)])
    rows = rr._lookup((3, 4, 5, 6, 7))
    assert rows.shape == (5, NUM_LAYERS, TOP_K)
    assert torch.equal(rows.long(), torch.as_tensor(routed[:5]).long())


def test_wrap_vllm_generate_records_and_is_idempotent(replay):
    natural = np.zeros((3, NUM_LAYERS, TOP_K))
    llm = SimpleNamespace(generate = lambda *a, **k: [_request([1, 2], [3], natural)])
    rr.wrap_vllm_generate(llm)
    first = llm.generate
    rr.wrap_vllm_generate(llm)
    assert llm.generate is first
    llm.generate([])
    assert rr.routing_replay_stats()["recorded"] == 1
