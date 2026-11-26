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

"""Routing replay of the rollout's fp32 routing weights (UNSLOTH_GRPO_ROUTING_REPLAY_WEIGHTS). CPU only."""

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

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
def replay(monkeypatch):
    monkeypatch.setenv(rr.ROUTING_REPLAY_ENV, "1")
    monkeypatch.delenv(rr.ROUTING_REPLAY_R2_ENV, raising = False)
    # As after the vLLM patch ran in this process for a top-k 2 engine.
    monkeypatch.setattr(rr, "_VLLM_WEIGHTS_PACKED", True)
    monkeypatch.setattr(rr, "_PACKED_TOP_K", TOP_K)
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
    model = Qwen3MoeForCausalLM(config).float().eval()
    for layer in model.model.layers:
        layer.mlp.gate.weight.data.normal_(0, 0.5)
    return model


def _natural(model, ids):
    """Natural [T, L, k] ids and fp32 weights."""
    captured = {}
    handles = [
        layer.mlp.gate.register_forward_hook(
            lambda m, a, out, i = i: captured.__setitem__(i, (out[2].detach().clone(), out[1].detach().float().clone()))
        )
        for i, layer in enumerate(model.model.layers)
    ]
    with torch.no_grad():
        model(input_ids = torch.tensor([ids]))
    for h in handles:
        h.remove()
    ids_ = torch.stack([captured[i][0] for i in range(NUM_LAYERS)], dim = 1)
    weights = torch.stack([captured[i][1] for i in range(NUM_LAYERS)], dim = 1)
    return ids_, weights


def _perturbed(weights, seed = 1, scale = 0.03):
    # Rollout weights as another engine computes them: close to the trainer's, still normalised.
    generator = torch.Generator().manual_seed(seed)
    w = (weights + scale * torch.rand(weights.shape, generator = generator)).clamp_min(1e-4)
    return w / w.sum(-1, keepdim = True)


def _packed(ids, weights):
    return rr._pack_routed_weights(ids.long(), weights.float()).numpy().astype(np.uint16)


def _request(prompt, completion, routed):
    return SimpleNamespace(
        prompt_token_ids = list(prompt),
        outputs = [SimpleNamespace(token_ids = list(completion), routed_experts = np.asarray(routed)[:-1])],
    )


def _install(model):
    assert rr.install_routing_replay(model, None, SimpleNamespace(pad_token_id = PAD)) == NUM_LAYERS


def _reference(model, ids, forced, rollout, straight_through = True):
    """Each router takes `forced` ids. Value = `rollout` weights where given (finite), else the
    trainer's softmax renorm weights; gradient = the trainer's weights."""
    originals = []
    for li, layer in enumerate(model.model.layers):
        gate = layer.mlp.gate
        originals.append(gate.forward)

        def forward(hidden_states, gate = gate, li = li):
            hidden_states = hidden_states.reshape(-1, gate.hidden_dim)
            logits = F.linear(hidden_states, gate.weight)
            idx = forced[:, li, :].long().contiguous()
            w = F.softmax(logits.float().gather(-1, idx), dim = -1)
            r = rollout[:, li, :]
            use = torch.isfinite(r).all(-1, keepdim = True)
            target = torch.where(use, r, w.detach())
            w = w + (target - w).detach() if straight_through else target
            return logits, w.to(logits.dtype), idx

        gate.forward = forward
    try:
        return model(input_ids = torch.tensor([ids])).logits
    finally:
        for layer, original in zip(model.model.layers, originals):
            layer.mlp.gate.forward = original


def test_pack_round_trips_fp32_bits_exactly():
    generator = torch.Generator().manual_seed(0)
    weights = torch.rand(64, 3, 4, generator = generator)
    weights[0, 0] = torch.tensor([0.0, 1.0, 1e-40, 0.5])  # zero, one, a subnormal
    ids = torch.randint(0, 300, (64, 3, 4), generator = generator)
    packed = rr._pack_routed_weights(ids, weights)
    assert packed.dtype == torch.int32 and packed.shape == (64, 3, 12)
    assert int(packed.min()) >= 0 and int(packed.max()) <= 0xFFFF  # fits vLLM's uint16 snapshot
    got_ids, got_weights = rr._unpack_routed_weights(packed.numpy().astype(np.uint16), 4)
    assert torch.equal(got_ids, ids)
    assert torch.equal(got_weights.view(torch.int32), weights.view(torch.int32))


def test_rollout_weights_replace_trainer_values_with_trainer_gradient(replay):
    model = _model()
    prompt, completion = [5, 6, 7, 8], [9, 10, 11, 12, 13]
    ids = prompt + completion
    natural, weights = _natural(model, ids)
    rollout = _perturbed(weights)
    rollout_rows = rollout.clone()
    rollout_rows[-1] = float("nan")  # vLLM never forwards the last token

    model.zero_grad()
    expected = _reference(model, ids, natural, rollout_rows)
    expected.float().pow(2).mean().backward()
    want_grads = [layer.mlp.gate.weight.grad.clone() for layer in model.model.layers]
    want_embed = model.model.embed_tokens.weight.grad.clone()

    model.zero_grad()
    ids_only = _reference(model, ids, natural, torch.full_like(rollout_rows, float("nan")))

    _install(model)
    rr.record_routed_experts([_request(prompt, completion, _packed(natural, rollout))])
    model.zero_grad()
    out = model(input_ids = torch.tensor([ids])).logits
    out.float().pow(2).mean().backward()

    torch.testing.assert_close(out, expected, rtol = 1e-5, atol = 1e-6)
    assert not torch.allclose(out, ids_only, atol = 1e-4)  # the weights really changed the forward
    for layer, want in zip(model.model.layers, want_grads):
        torch.testing.assert_close(layer.mlp.gate.weight.grad, want, rtol = 1e-4, atol = 1e-6)
        assert layer.mlp.gate.weight.grad.abs().sum() > 0  # the router keeps its gradient
    torch.testing.assert_close(model.model.embed_tokens.weight.grad, want_embed, rtol = 1e-4, atol = 1e-7)
    stats = rr.routing_replay_stats()
    assert stats["recorded_weights"] == 1
    assert stats["replayed_weight_rows"] == (len(ids) - 1) * NUM_LAYERS
    assert stats["changed_selections"] == 0


def test_rollout_weights_off_without_the_vllm_patch(replay, monkeypatch):
    # An engine whose routes keep k columns (out of process, or vLLM < 0.28) is read as ids only.
    monkeypatch.setattr(rr, "_PACKED_TOP_K", None)
    monkeypatch.setattr(rr, "_VLLM_WEIGHTS_PACKED", False)
    model = _model()
    prompt, completion = [5, 6, 7], [20, 21, 22]
    ids = prompt + completion
    natural, weights = _natural(model, ids)
    with torch.no_grad():
        natural_logits = model(input_ids = torch.tensor([ids])).logits
    _install(model)
    rr.record_routed_experts([_request(prompt, completion, natural.numpy().astype(np.uint8))])
    with torch.no_grad():
        out = model(input_ids = torch.tensor([ids])).logits
    torch.testing.assert_close(out, natural_logits, rtol = 1e-5, atol = 1e-6)
    assert rr.routing_replay_stats()["replayed_weight_rows"] == 0


def test_unusable_rollout_weights_keep_the_trainer_weights(replay):
    model = _model()
    prompt, completion = [5, 6, 7], [20, 21, 22, 23]
    ids = prompt + completion
    natural, weights = _natural(model, ids)
    rollout = _perturbed(weights)
    bad = rollout.clone()
    bad[0, 0] = torch.tensor([float("nan"), 0.5])  # not finite
    bad[1, 0] = torch.tensor([0.9, 0.9])  # sums past 1 (a scaled router)
    bad[2, 0] = torch.tensor([0.0, 0.0])  # an ids-only capture
    bad[3, 0] = (weights[3, 0] + torch.tensor([0.3, -0.3])).clamp(0, 1)  # far from the trainer's
    used = torch.isfinite(bad).all(-1)
    used[:4, 0] = False
    used[-1] = False
    reference_rollout = torch.where(used.unsqueeze(-1), bad, torch.full_like(bad, float("nan")))
    with torch.no_grad():
        expected = _reference(model, ids, natural, reference_rollout)
    _install(model)
    rr.record_routed_experts([_request(prompt, completion, _packed(natural, bad))])
    with torch.no_grad():
        out = model(input_ids = torch.tensor([ids])).logits
    torch.testing.assert_close(out, expected, rtol = 1e-5, atol = 1e-6)
    assert rr.routing_replay_stats()["replayed_weight_rows"] == int(used.sum())


def test_ids_only_rows_of_a_widened_capture_keep_the_ids(replay):
    model = _model()
    prompt, completion = [5, 6, 7], [20, 21, 22]
    ids = prompt + completion
    natural, weights = _natural(model, ids)
    forced = natural.clone()
    forced[1:3] = (forced[1:3] + 3) % NUM_EXPERTS
    packed = _packed(forced, torch.zeros_like(weights))  # a monolithic kernel's zero padded capture
    with torch.no_grad():
        expected = _reference(model, ids, forced, torch.full_like(weights, float("nan")))
    _install(model)
    rr.record_routed_experts([_request(prompt, completion, packed)])
    with torch.no_grad():
        out = model(input_ids = torch.tensor([ids])).logits
    torch.testing.assert_close(out, expected, rtol = 1e-5, atol = 1e-6)
    stats = rr.routing_replay_stats()
    assert stats["replayed_weight_rows"] == 0 and stats["changed_selections"] == 2 * NUM_LAYERS


@pytest.mark.parametrize("use_reentrant", [True, False])
def test_gradient_checkpointing_recompute_replays_the_same_weights(replay, use_reentrant):
    prompt, completion = [5, 6, 7], [30, 31, 32, 33]
    ids = prompt + completion
    natural, weights = _natural(_model(), ids)
    packed = _packed(natural, _perturbed(weights))

    def run(checkpointing):
        m = _model()
        m.train()
        if checkpointing:
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": use_reentrant})
        m.model.embed_tokens.weight.requires_grad_(True)
        rr.reset_routing_replay()
        _install(m)
        rr.record_routed_experts([_request(prompt, completion, packed)])
        logits = m(input_ids = torch.tensor([ids])).logits
        logits.float().pow(2).mean().backward()
        return logits.detach(), m.model.layers[0].mlp.gate.weight.grad.clone(), rr.routing_replay_stats()

    logits_plain, grad_plain, stats_plain = run(False)
    logits_gc, grad_gc, stats_gc = run(True)
    torch.testing.assert_close(logits_gc, logits_plain, rtol = 1e-5, atol = 1e-6)
    torch.testing.assert_close(grad_gc, grad_plain, rtol = 1e-4, atol = 1e-6)
    assert stats_plain["replayed_weight_rows"] > 0
    assert stats_gc["router_calls_recomputed"] == NUM_LAYERS
    # The recompute applied the rollout weights again.
    assert stats_gc["replayed_weight_rows"] == 2 * stats_plain["replayed_weight_rows"]


class _Router(torch.nn.Module):
    # Named so that _is_router finds it.
    norm_topk_prob = True

    def forward(self, x):
        weights, ids = x.softmax(-1).topk(2, -1)
        return x, weights / weights.sum(-1, keepdim = True), ids


_Router.__name__ = "Qwen3MoeTopKRouter"


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.router = _Router()

    def forward(self, x):
        _, weights, ids = self.router(x)
        picked = x.gather(-1, ids)
        return (picked * weights).sum(-1, keepdim = True) * x.sin(), ids


class _Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(16, 8)
        self.layers = torch.nn.ModuleList([_Layer()])


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Decoder()


def _compiled_step(fn, x, forced, rollout):
    rr._STATE.in_forward = True
    rr._STATE.current = forced.unsqueeze(1)
    rr._STATE.current_weights = rollout.unsqueeze(1)
    rr._STATE.record = {}
    rr._STATE.pending.append(rr._STATE.record)
    rr._STATE.has_pending = True
    out, ids = checkpoint(fn, x, use_reentrant = True)
    rr._STATE.in_forward = False
    rr._STATE.record = None
    rr._STATE.current = rr._STATE.current_weights = None
    (out * torch.arange(1.0, 9.0)).sum().backward()
    return out.detach(), ids


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_compiled_fullgraph_layer_replays_weights_like_eager(replay, backend):
    torch._dynamo.reset()
    forced = torch.tensor([[1, 4], [0, 7], [5, 2]])
    trainer = torch.linspace(-1, 1, 24).reshape(3, 8).gather(-1, forced).softmax(-1)
    # Within the 0.1 tolerance of the trainer's weights, last row without rollout weights.
    rollout = trainer + torch.tensor([[0.04, -0.04], [-0.03, 0.03], [0.0, 0.0]])
    rollout[2] = float("nan")
    results = []
    for compiled in (False, True):
        rr.reset_routing_replay()
        model = _Model()
        assert rr.install_routing_replay(model) == 1
        layer = model.model.layers[0]
        layer.router._unsloth_replay_formula = "softmax_renorm"
        fn = torch.compile(lambda x: layer(x), backend = backend, fullgraph = True) if compiled else layer
        x = torch.linspace(-1, 1, 24).reshape(3, 8).requires_grad_()
        out, ids = _compiled_step(fn, x, forced, rollout)
        results.append((out, ids, x.grad.clone(), rr.routing_replay_stats()))
    (out_e, ids_e, grad_e, stats_e), (out_c, ids_c, grad_c, stats_c) = results
    assert torch.equal(ids_c, ids_e) and torch.equal(ids_e, forced)
    torch.testing.assert_close(out_c, out_e, rtol = 1e-5, atol = 1e-6)
    torch.testing.assert_close(grad_c, grad_e, rtol = 1e-5, atol = 1e-6)
    # Forward value = rollout weights on the two rows that have them.
    x = torch.linspace(-1, 1, 24).reshape(3, 8)
    want = (x.gather(-1, forced) * rollout).sum(-1, keepdim = True) * x.sin()
    torch.testing.assert_close(out_e[:2], want[:2], rtol = 1e-5, atol = 1e-6)
    assert stats_e["replayed_weight_rows"] == stats_c["replayed_weight_rows"] == 4  # forward + recompute
    torch._dynamo.reset()


def _fake_vllm(monkeypatch, num_experts = 8, top_k = 2):
    """Just the vLLM pieces the patch touches, shaped like vLLM 0.28 - 0.34."""
    capturer_module = types.ModuleType("vllm.model_executor.layers.fused_moe.routed_experts_capturer")

    def _get_routed_experts_shape(vllm_config):
        return 3, num_experts, top_k

    class RoutedExpertsCapturer:
        def __init__(self, max_tokens):
            layers, self.num_experts, k = capturer_module._get_routed_experts_shape(None)
            self.device_buffer = torch.zeros((max_tokens, layers, k), dtype = torch.int32)

        def capture(self, layer_id, topk_ids):
            self.device_buffer[: len(topk_ids), layer_id] = topk_ids

    capturer_module._get_routed_experts_shape = _get_routed_experts_shape
    capturer_module.RoutedExpertsCapturer = RoutedExpertsCapturer
    router_module = types.ModuleType("vllm.model_executor.layers.fused_moe.router.base_router")

    class BaseRouter:
        capture_fn = None

        def _select_experts(self, hidden_states, router_logits, topk_indices_dtype = None, **kwargs):
            topk_weights, topk_ids = router_logits.softmax(-1).topk(top_k, -1)
            topk_weights = topk_weights / topk_weights.sum(-1, keepdim = True)
            if self.capture_fn is not None:
                self.capture_fn(topk_ids)
            # EPLB maps logical to physical ids after the capture.
            return topk_weights, topk_ids + 100

    router_module.BaseRouter = BaseRouter
    names = [
        "vllm", "vllm.model_executor", "vllm.model_executor.layers", "vllm.model_executor.layers.fused_moe",
        "vllm.model_executor.layers.fused_moe.router",
    ]
    for name in names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, capturer_module.__name__, capturer_module)
    monkeypatch.setitem(sys.modules, router_module.__name__, router_module)
    monkeypatch.setattr(sys.modules["vllm.model_executor.layers.fused_moe"], "routed_experts_capturer", capturer_module, raising = False)
    return capturer_module, BaseRouter


def test_vllm_patch_widens_capture_and_packs_weights(monkeypatch):
    monkeypatch.setattr(rr, "_VLLM_WEIGHTS_PACKED", False)
    monkeypatch.setattr(rr, "_PACKED_TOP_K", None)
    capturer_module, BaseRouter = _fake_vllm(monkeypatch)
    original_select = BaseRouter._select_experts
    assert rr._patch_vllm_routed_weights()
    assert rr._patch_vllm_routed_weights()  # idempotent
    assert capturer_module._get_routed_experts_shape(None) == (3, 257, 6)  # 257 experts: uint16 snapshot
    assert rr._PACKED_TOP_K == 2

    capturer = capturer_module.RoutedExpertsCapturer(max_tokens = 5)
    router = BaseRouter()
    router.capture_fn = lambda ids: capturer.capture(1, ids)
    logits = torch.randn(4, 8)
    weights, ids = router._select_experts(None, logits)
    want_weights, want_ids = original_select(BaseRouter(), None, logits)
    assert torch.equal(weights, want_weights) and torch.equal(ids, want_ids)  # vLLM's own outputs unchanged
    got_ids, got_weights = rr._unpack_routed_weights(capturer.device_buffer[:4, 1], 2)
    assert torch.equal(got_ids, want_ids - 100)  # logical ids, captured before the EPLB map
    assert torch.equal(got_weights, want_weights)

    capturer.capture(2, torch.tensor([[3, 5]]))  # an ids-only capture source
    assert capturer.device_buffer[0, 2].tolist() == [3, 5, 0, 0, 0, 0]
    router.capture_fn = None
    assert torch.equal(router._select_experts(None, logits)[1], want_ids)


def test_vllm_patch_refuses_an_unknown_capture_layout(monkeypatch):
    monkeypatch.setattr(rr, "_VLLM_WEIGHTS_PACKED", False)
    capturer_module, BaseRouter = _fake_vllm(monkeypatch)
    del capturer_module._get_routed_experts_shape  # vLLM < 0.28
    assert not rr._patch_vllm_routed_weights()
    assert rr._VLLM_WEIGHTS_PACKED is False


def test_engine_args_warn_when_weights_cannot_be_returned(replay, monkeypatch):
    monkeypatch.setattr(rr, "_engine_arg_names", lambda: {"enable_return_routed_experts"})
    monkeypatch.setattr(rr, "_patch_vllm_routed_weights", lambda: False)
    with pytest.warns(UserWarning, match = "0.28 or newer"):
        assert rr.routing_replay_engine_args(_model().config) == {"enable_return_routed_experts": True}
    monkeypatch.setenv(rr.ROUTING_REPLAY_WEIGHTS_ENV, "0")
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        assert rr.routing_replay_engine_args(_model().config) == {"enable_return_routed_experts": True}


def test_prompt_start_prefix_reuse_keeps_the_earlier_turn_weights(replay):
    # Multi-turn: turn 2 skips the rows turn 1 returned; the reused prefix keeps turn 1's weights too.
    generator = torch.Generator().manual_seed(0)
    ids1 = torch.stack([torch.randperm(NUM_EXPERTS, generator = generator)[:TOP_K] for _ in range(5 * NUM_LAYERS)]).reshape(5, NUM_LAYERS, TOP_K)
    w1 = torch.rand(5, NUM_LAYERS, TOP_K, generator = generator)
    w1 = w1 / w1.sum(-1, keepdim = True)
    turn1 = SimpleNamespace(prompt_token_ids = [3, 4, 5], outputs = [SimpleNamespace(token_ids = [6, 7, 8], routed_experts = _packed(ids1, w1))])
    rr.record_routed_experts([turn1])
    ids2 = torch.stack([torch.randperm(NUM_EXPERTS, generator = generator)[:TOP_K] for _ in range(2 * NUM_LAYERS)]).reshape(2, NUM_LAYERS, TOP_K)
    w2 = torch.rand(2, NUM_LAYERS, TOP_K, generator = generator)
    w2 = w2 / w2.sum(-1, keepdim = True)
    prompt2 = [3, 4, 5, 6, 7, 8, 9]
    turn2 = SimpleNamespace(prompt_token_ids = prompt2, outputs = [SimpleNamespace(token_ids = [11, 12], routed_experts = _packed(ids2, w2))])
    rr.record_routed_experts([turn2], prompt_starts = [6])
    rows, weights = rr._lookup_routes(tuple(prompt2 + [11, 12]))
    assert torch.equal(rows[:5].long(), ids1.long()) and torch.equal(rows[6:8].long(), ids2.long())
    assert torch.equal(weights[:5], w1) and torch.equal(weights[6:8], w2)
    # Turn 1 never forwarded its last sampled token, nor turn 2 its own: no weights there.
    assert bool(torch.isnan(weights[5]).all()) and bool(torch.isnan(weights[8]).all())
