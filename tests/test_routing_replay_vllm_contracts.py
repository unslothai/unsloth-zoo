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

"""MoE routing replay against vLLM's output contracts (0.21 split routes, prompt start offset,
async scheduling), duplicate completions and checkpoint recompute identity. CPU only."""

import ast
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import unsloth_zoo.routing_replay as rr

PAD = 0
LAYERS = 2
TOP_K = 2


@pytest.fixture
def replay(monkeypatch):
    monkeypatch.setenv(rr.ROUTING_REPLAY_ENV, "1")
    rr.reset_routing_replay()
    yield rr
    rr.reset_routing_replay()


def _routes(n, seed):
    # n tokens of distinct top-2 expert pairs per layer.
    g = torch.Generator().manual_seed(seed)
    return np.stack([
        np.stack([torch.randperm(8, generator = g)[:TOP_K].numpy() for _ in range(LAYERS)]) for _ in range(n)
    ]).astype(np.int16)


@pytest.mark.parametrize("version, expected", [
    ("0.20.2", None), ("0.21.0", False), ("0.21.1rc0", False), ("0.22.0", None), ("0.32.0", None),
])
def test_vllm_021_turns_async_scheduling_off(replay, monkeypatch, version, expected):
    # vLLM 0.21 _validate_return_routed_experts refuses async scheduling, which it enables by default.
    import importlib.metadata
    monkeypatch.setattr(rr, "_engine_arg_names", lambda: {"enable_return_routed_experts", "async_scheduling"})
    monkeypatch.setattr(importlib.metadata, "version", lambda name: version)
    args = rr.routing_replay_engine_args(SimpleNamespace(num_experts = 8))
    assert args.get("async_scheduling") is expected
    assert args["enable_return_routed_experts"] is True


def test_vllm_021_prompt_routes_on_the_request_are_placed_first(replay):
    # vLLM 0.21: RequestOutput.prompt_routed_experts [prompt_len], CompletionOutput.routed_experts
    # only the generated tokens (split_routed_experts in routed_experts_capturer.py).
    prompt, completion = [11, 12, 13], [21, 22, 23]
    full = _routes(6, 0)
    request = SimpleNamespace(
        prompt_token_ids = prompt,
        prompt_routed_experts = full[:3],
        outputs = [SimpleNamespace(token_ids = completion, routed_experts = full[3:5])],
    )
    rr.record_routed_experts([request])
    rows = rr._lookup(tuple(prompt + completion), exact = True)
    assert torch.equal(rows[:5].long(), torch.as_tensor(full[:5]).long())
    assert bool((rows[5] == -1).all())

    # No prompt routes returned: the prompt keeps its natural routing, the completion still lines up.
    rr.reset_routing_replay()
    request.prompt_routed_experts = None
    rr.record_routed_experts([request])
    rows = rr._lookup(tuple(prompt + completion), exact = True)
    assert bool((rows[:3] == -1).all())
    assert torch.equal(rows[3:5].long(), torch.as_tensor(full[3:5]).long())


@pytest.mark.parametrize("per_prompt", [False, True])
def test_routed_experts_prompt_start_offsets_the_rows(replay, per_prompt):
    # vLLM >= 0.22 SamplingParams.routed_experts_prompt_start = N: returned row 0 is position N.
    prompt, completion = [1, 2, 3], [4, 5]
    full = _routes(5, 1)
    params = SimpleNamespace(routed_experts_prompt_start = 2)
    llm = SimpleNamespace(generate = lambda *a, **k: [SimpleNamespace(
        prompt_token_ids = prompt, outputs = [SimpleNamespace(token_ids = completion, routed_experts = full[2:4])],
    )])
    rr.wrap_vllm_generate(llm)
    llm.generate(["x"], sampling_params = [params] if per_prompt else params)
    rows = rr._lookup(tuple(prompt + completion), exact = True)
    assert bool((rows[:2] == -1).all())
    assert torch.equal(rows[2:4].long(), torch.as_tensor(full[2:4]).long())
    assert bool((rows[4] == -1).all())


def test_duplicate_completions_replay_only_their_common_routes(replay):
    tokens = [1, 2, 3]
    a = _routes(3, 2)
    b = a.copy()
    b[1, 0] = (a[1, 0] + 1) % 8  # token 1, layer 0: another expert set in the second copy

    def req(routed):
        return SimpleNamespace(prompt_token_ids = tokens[:1], outputs = [SimpleNamespace(token_ids = tokens[1:], routed_experts = routed)])

    for split_calls in (False, True):
        rr.reset_routing_replay()
        if split_calls:
            rr.record_routed_experts([req(a)])
            rr.record_routed_experts([req(b)])
        else:
            rr.record_routed_experts([req(a), req(b)])
        rows = rr._lookup(tuple(tokens), exact = True)
        assert bool((rows[1, 0] == -1).all()), split_calls
        keep = torch.ones(3, LAYERS, dtype = torch.bool)
        keep[1, 0] = False
        assert torch.equal(rows[keep].long(), torch.as_tensor(a)[keep].long())


transformers = pytest.importorskip("transformers")


def _moe(seed = 0):
    try:
        from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter  # noqa: F401
    except Exception:
        pytest.skip("needs transformers >= 5 router modules")
    torch.manual_seed(seed)
    config = Qwen3MoeConfig(
        vocab_size = 64, hidden_size = 32, intermediate_size = 64, moe_intermediate_size = 16,
        num_hidden_layers = LAYERS, num_attention_heads = 4, num_key_value_heads = 2, head_dim = 8,
        num_experts = 8, num_experts_per_tok = TOP_K, norm_topk_prob = True, pad_token_id = PAD,
    )
    model = Qwen3MoeForCausalLM(config).float()
    for layer in model.model.layers:
        layer.mlp.gate.weight.data.normal_(0, 0.5)
    return model


@pytest.mark.parametrize("use_reentrant", [True, False])
def test_recompute_of_identical_inputs_with_different_replays_keeps_gradients(replay, use_reentrant):
    # Record / forward one rollout, record / forward another for the same tokens, then one backward.
    # Both forwards feed every router the same bits with the same natural routes, so only the
    # checkpointed layer's input tensor can tell their recomputes apart.
    ids = [5, 6, 7, 30, 31, 32]
    natural = None
    with torch.no_grad():
        m = _moe()
        captured = {}
        for i, layer in enumerate(m.model.layers):
            layer.mlp.gate.register_forward_hook(lambda mod, a, out, i = i: captured.__setitem__(i, out[2].clone()))
        m(input_ids = torch.tensor([ids]))
        natural = torch.stack([captured[i] for i in range(LAYERS)], dim = 1).numpy()
    shifted = [((natural + s) % 8).astype(np.int16) for s in (2, 5)]

    def run(checkpointing):
        m = _moe()
        m.train()
        if checkpointing:
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": use_reentrant})
        m.model.embed_tokens.weight.requires_grad_(True)
        rr.reset_routing_replay()
        assert rr.install_routing_replay(m, None, SimpleNamespace(pad_token_id = PAD)) == LAYERS
        losses = []
        for routed in shifted:
            rr.record_routed_experts([SimpleNamespace(
                prompt_token_ids = ids[:3], outputs = [SimpleNamespace(token_ids = ids[3:], routed_experts = routed[:-1])],
            )])
            losses.append(m(input_ids = torch.tensor([ids])).logits.float().square().mean())
        sum(losses).backward()
        return torch.stack([l.detach() for l in losses]), {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}

    losses_plain, grads_plain = run(False)
    losses_gc, grads_gc = run(True)
    assert not torch.allclose(losses_plain[0], losses_plain[1])
    torch.testing.assert_close(losses_gc, losses_plain, rtol = 1e-5, atol = 1e-6)
    assert grads_gc.keys() == grads_plain.keys()
    for name in grads_plain:
        torch.testing.assert_close(grads_gc[name], grads_plain[name], rtol = 1e-4, atol = 1e-6, msg = name)


def test_replayssm_yields_to_routing_replay_on_vllm_031():
    # vLLM 0.31 _verify_aux_output_compatibility: routed experts capture needs Model Runner V2,
    # while load_vllm pins V1 for ReplaySSM. The skip must come before use_replayssm is set.
    path = pathlib.Path(rr.__file__).with_name("vllm_utils.py")
    src = path.read_text(encoding = "utf-8-sig")
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "load_vllm")
    body = ast.get_source_segment(src, fn)
    guard = body.find('engine_args.get("enable_return_routed_experts")')
    assert guard != -1 and "_vllm_at_least(0, 31)" in body[guard : guard + 600]
    assert guard < body.find('engine_args["use_replayssm"] = True')
