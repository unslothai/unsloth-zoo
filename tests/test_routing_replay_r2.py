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

"""Old-policy routing replay (R2, UNSLOTH_GRPO_ROUTING_REPLAY_R2): after a rollout, the first forward
of a row (pi_old's) records its experts and later forwards (num_iterations > 1) replay them. CPU only."""

from types import SimpleNamespace

import pytest
import torch

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


@pytest.fixture
def r2(monkeypatch):
    monkeypatch.delenv(rr.ROUTING_REPLAY_ENV, raising = False)
    monkeypatch.setenv(rr.ROUTING_REPLAY_R2_ENV, "1")
    rr.reset_routing_replay()
    yield rr
    rr.reset_routing_replay()


def _model(seed = 0):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(seed)
    config = Qwen3MoeConfig(
        vocab_size = 64, hidden_size = 32, intermediate_size = 64, moe_intermediate_size = 16,
        num_hidden_layers = NUM_LAYERS, num_attention_heads = 4, num_key_value_heads = 2, head_dim = 8,
        num_experts = NUM_EXPERTS, num_experts_per_tok = TOP_K, norm_topk_prob = True, pad_token_id = PAD,
    )
    model = Qwen3MoeForCausalLM(config).float()
    for layer in model.model.layers:
        layer.mlp.gate.weight.data.normal_(0, 0.5)
    return model


class _Routes:
    """Expert ids every router call of a forward used, [rows * T, L, k]."""

    def __init__(self, model):
        self.calls = {}
        self.handles = [
            layer.mlp.gate.register_forward_hook(
                lambda m, a, out, i = i: self.calls.__setitem__(i, out[2].detach().clone())
            )
            for i, layer in enumerate(model.model.layers)
        ]
        # Hooks registered before install run before the replay hook: re-register them last.
        self.model = model

    def ids(self):
        return torch.stack([self.calls[i] for i in range(NUM_LAYERS)], dim = 1).sort(-1).values

    def remove(self):
        for h in self.handles:
            h.remove()


def _install(model):
    assert rr.install_routing_replay(model, None, SimpleNamespace(pad_token_id = PAD)) == NUM_LAYERS
    return _Routes(model)  # after install: sees the replayed ids


def _rollout(model):
    # HF generation: the decoder gets a KV cache, which starts a new rollout for R2.
    with torch.no_grad():
        model.generate(input_ids = torch.tensor([[3, 4]]), max_new_tokens = 2, do_sample = False, pad_token_id = PAD)


def _policy_update(model, seed = 3):
    # Stands in for an optimizer step between iterations: enough to move the natural routing.
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.gate.weight.add_(torch.randn(layer.mlp.gate.weight.shape, generator = generator) * 0.5)


def _batch():
    # Left padded GRPO style batch with an attention mask.
    input_ids = torch.tensor([[PAD, PAD, 5, 6, 7, 30, 31, 32], [8, 9, 10, 11, 40, 41, 42, 43]])
    return input_ids, (input_ids != PAD).long()


def _agreement(a, b, mask):
    keep = mask.reshape(-1).bool()
    return float((a[keep] == b[keep]).all(-1).all(-1).float().mean())


@pytest.mark.parametrize("old_mode", ["no_grad", "inference_mode"])
def test_second_pass_replays_the_old_forward_experts(r2, old_mode):
    model = _model()
    routes = _install(model)
    _rollout(model)
    input_ids, mask = _batch()
    with (torch.no_grad() if old_mode == "no_grad" else torch.inference_mode()):
        model(input_ids = input_ids, attention_mask = mask)
    old = routes.ids()
    assert rr.routing_replay_stats()["r2_recorded"] == 2

    _policy_update(model)
    model.train()
    out = model(input_ids = input_ids, attention_mask = mask).logits
    out.float().pow(2).mean().backward()
    assert _agreement(routes.ids(), old, mask) == 1.0
    assert rr.routing_replay_stats()["r2_replayed_tokens"] == int(mask.sum())
    assert model.model.layers[0].mlp.gate.weight.grad.abs().sum() > 0


def test_pi_old_taken_from_the_first_policy_forward_is_replayed(r2):
    # pi_old from the first forward with grad (no separate no-grad forward): iteration 2 replays it.
    model = _model()
    model.train()
    routes = _install(model)
    _rollout(model)
    input_ids, mask = _batch()
    model(input_ids = input_ids, attention_mask = mask).logits.float().pow(2).mean().backward()
    old = routes.ids()
    _policy_update(model)
    model(input_ids = input_ids, attention_mask = mask).logits.float().pow(2).mean().backward()
    assert _agreement(routes.ids(), old, mask) == 1.0
    stats = rr.routing_replay_stats()
    assert stats["r2_recorded"] == 2 and stats["r2_replayed_tokens"] == int(mask.sum())


def test_no_rollout_seen_keeps_natural_routing(r2):
    # Without a rollout boundary R2 cannot tell pi_old's forward apart: it records nothing.
    model = _model()
    routes = _install(model)
    input_ids, mask = _batch()
    with torch.no_grad():
        model(input_ids = input_ids, attention_mask = mask)
    first = routes.ids()
    _policy_update(model)
    model(input_ids = input_ids, attention_mask = mask)
    assert _agreement(routes.ids(), first, mask) < 0.9
    assert rr.routing_replay_stats()["r2_recorded"] == 0


def test_vllm_generate_starts_a_rollout(r2):
    model = _model()
    llm = SimpleNamespace(generate = lambda *a, **k: [])
    assert rr.install_routing_replay(model, llm, SimpleNamespace(pad_token_id = PAD)) == NUM_LAYERS
    assert rr._STATE.r2_rollout == 0
    llm.generate(["x"])
    assert rr._STATE.r2_rollout == 1


def test_without_r2_the_second_pass_routes_differently(monkeypatch):
    # Base arm: the policy update really moves routes, so 100% agreement above is R2's doing.
    monkeypatch.delenv("UNSLOTH_GRPO_ROUTING_REPLAY_R2", raising = False)
    model = _model()
    routes = _Routes(model)
    input_ids, mask = _batch()
    with torch.no_grad():
        model(input_ids = input_ids, attention_mask = mask)
    old = routes.ids()
    _policy_update(model)
    model(input_ids = input_ids, attention_mask = mask)
    assert _agreement(routes.ids(), old, mask) < 0.9


def test_reordered_and_repadded_minibatch_finds_its_old_routes(r2):
    model = _model()
    routes = _install(model)
    _rollout(model)
    input_ids, mask = _batch()
    with torch.no_grad():
        model(input_ids = input_ids, attention_mask = mask)
    old = routes.ids().reshape(2, 8, NUM_LAYERS, TOP_K)
    _policy_update(model)
    # Second row alone, then the first one right padded.
    out_ids = torch.tensor([[5, 6, 7, 30, 31, 32, PAD, PAD]])
    model(input_ids = input_ids[1:], attention_mask = mask[1:])
    assert torch.equal(routes.ids().reshape(1, 8, NUM_LAYERS, TOP_K)[0], old[1])
    model(input_ids = out_ids, attention_mask = (out_ids != PAD).long())
    assert torch.equal(routes.ids().reshape(1, 8, NUM_LAYERS, TOP_K)[0, :6], old[0, 2:])


def test_the_next_rollout_refreshes_the_old_routes(r2):
    # The next rollout's pi_old forward scores the updated policy: it must not replay stale routes,
    # even for a row identical to one of the previous rollout.
    model = _model()
    routes = _install(model)
    _rollout(model)
    input_ids, mask = _batch()
    with torch.no_grad():
        model(input_ids = input_ids, attention_mask = mask)
    first = routes.ids()
    _policy_update(model)
    _rollout(model)
    with torch.no_grad():
        model(input_ids = input_ids, attention_mask = mask)
    second = routes.ids()
    assert _agreement(second, first, mask) < 0.9
    model(input_ids = input_ids, attention_mask = mask)
    assert _agreement(routes.ids(), second, mask) == 1.0


@pytest.mark.parametrize("use_reentrant", [True, False])
def test_checkpoint_recompute_replays_r2_routes(r2, use_reentrant):
    input_ids, mask = _batch()

    def run(checkpointing):
        rr.reset_routing_replay()
        m = _model()
        _install(m)
        _rollout(m)
        with torch.no_grad():
            m(input_ids = input_ids, attention_mask = mask)
        _policy_update(m)
        m.train()
        if checkpointing:
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": use_reentrant})
        m.model.embed_tokens.weight.requires_grad_(True)
        logits = m(input_ids = input_ids, attention_mask = mask).logits
        logits.float().pow(2).mean().backward()
        return logits.detach(), m.model.layers[1].mlp.gate.weight.grad.clone()

    logits_plain, grad_plain = run(False)
    logits_gc, grad_gc = run(True)
    torch.testing.assert_close(logits_gc, logits_plain, rtol = 1e-5, atol = 1e-6)
    torch.testing.assert_close(grad_gc, grad_plain, rtol = 1e-4, atol = 1e-6)


def test_r2_store_is_bounded(r2, monkeypatch):
    monkeypatch.setattr(rr, "_MAX_R2_ROWS", 3)
    model = _model()
    _install(model)
    _rollout(model)
    with torch.no_grad():
        for start in range(5):
            ids = torch.tensor([[start + 1, start + 2, start + 3]])
            model(input_ids = ids, attention_mask = torch.ones_like(ids))
    assert len(rr._STATE.r2_exact) == 3 and len(rr._STATE.r2_unpadded) == 3
    assert (5, 6, 7) in rr._STATE.r2_exact and (1, 2, 3) not in rr._STATE.r2_exact


def test_r2_off_by_default(monkeypatch):
    monkeypatch.delenv(rr.ROUTING_REPLAY_ENV, raising = False)
    monkeypatch.delenv(rr.ROUTING_REPLAY_R2_ENV, raising = False)
    rr.reset_routing_replay()
    assert rr.install_routing_replay(_model(), None, SimpleNamespace(pad_token_id = PAD)) == 0


def test_no_pad_id_and_no_mask_still_replays(r2):
    model = _model()
    assert rr.install_routing_replay(model, None, None) == NUM_LAYERS
    routes = _Routes(model)
    _rollout(model)
    ids = torch.tensor([[5, 6, 7, 30, 31, 32]])
    with torch.no_grad():
        model(input_ids = ids)
    old = routes.ids()
    _policy_update(model)
    model(input_ids = ids)
    assert torch.equal(routes.ids(), old)
    assert rr.routing_replay_stats()["r2_replayed_tokens"] == ids.numel()


@pytest.mark.parametrize("env", ["UNSLOTH_GRPO_ROUTING_REPLAY", "UNSLOTH_GRPO_ROUTING_REPLAY_R2"])
def test_looped_model_refuses_routing_replay(monkeypatch, env):
    # Inkling-2 style loop: every pass reuses the same layers and routers, so per-layer routes collide.
    monkeypatch.delenv("UNSLOTH_GRPO_ROUTING_REPLAY", raising = False)
    monkeypatch.delenv("UNSLOTH_GRPO_ROUTING_REPLAY_R2", raising = False)
    monkeypatch.setenv(env, "1")
    rr.reset_routing_replay()
    plain = _model()
    assert rr.install_routing_replay(plain, None, SimpleNamespace(pad_token_id = PAD)) == NUM_LAYERS
    looped = _model()
    looped.config.loop_num_passes = 3
    with pytest.warns(UserWarning, match = "looped model"):
        assert rr.install_routing_replay(looped, None, SimpleNamespace(pad_token_id = PAD)) == 0
    assert all(not layer.mlp.gate._forward_hooks for layer in looped.model.layers)
    if env == "UNSLOTH_GRPO_ROUTING_REPLAY":
        monkeypatch.setattr(rr, "_engine_arg_names", lambda: {"enable_return_routed_experts"})
        with pytest.warns(UserWarning, match = "looped model"):
            assert rr.routing_replay_engine_args(looped.config) == {}
        assert rr.routing_replay_engine_args(plain.config) != {}
    rr.reset_routing_replay()
