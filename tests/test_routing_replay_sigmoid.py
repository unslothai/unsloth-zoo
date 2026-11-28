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

"""Routing replay for sigmoid routers with a score correction bias (DeepSeek-V3, GLM-4 MoE): the weights over
forced experts are the sigmoid scores (not the biased choice scores), renormalised, times routed_scaling_factor.
CPU only."""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
try:
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3TopkRouter
except Exception:
    pytest.skip("needs the DeepSeek-V3 router module", allow_module_level = True)

import unsloth_zoo.routing_replay as rr

PAD = 0
NUM_LAYERS = 2
NUM_EXPERTS = 8
TOP_K = 2


@pytest.fixture
def replay(monkeypatch):
    monkeypatch.setenv(rr.ROUTING_REPLAY_ENV, "1")
    rr.reset_routing_replay()
    yield rr
    rr.reset_routing_replay()


def _config(norm = True, scale = 2.5):
    from transformers import DeepseekV3Config

    return DeepseekV3Config(
        vocab_size = 64, hidden_size = 32, intermediate_size = 64, moe_intermediate_size = 16,
        num_hidden_layers = NUM_LAYERS, first_k_dense_replace = 0, num_attention_heads = 4,
        num_key_value_heads = 4, n_routed_experts = NUM_EXPERTS, n_shared_experts = 1,
        num_experts_per_tok = TOP_K, n_group = 2, topk_group = 1, norm_topk_prob = norm,
        routed_scaling_factor = scale, kv_lora_rank = 16, q_lora_rank = None, qk_rope_head_dim = 8,
        qk_nope_head_dim = 8, v_head_dim = 8, pad_token_id = PAD,
    )


def _router(norm = True, scale = 2.5, seed = 0):
    torch.manual_seed(seed)
    router = DeepseekV3TopkRouter(_config(norm, scale))
    if not hasattr(router, "e_score_correction_bias"):
        pytest.skip("this transformers' DeepSeek-V3 router has no score correction bias")
    router.weight.data.normal_(0, 0.5)
    # A bias large enough to change which experts win: the weights must still come from the unbiased scores.
    router.e_score_correction_bias.data.copy_(torch.linspace(-1.0, 1.0, NUM_EXPERTS))
    output = router(torch.randn(3, 32))
    if not (isinstance(output, tuple) and len(output) == 3):
        pytest.skip("this transformers' DeepSeek-V3 router does not return (logits, weights, ids)")
    return router


def _expected(logits, ids, norm, scale):
    w = logits.float().sigmoid().gather(-1, ids)
    if norm:
        w = w / (w.sum(-1, keepdim = True) + 1e-20)
    return w * scale


@pytest.mark.parametrize("norm, scale", [(True, 2.5), (False, 1.0), (True, 1.0)])
def test_sigmoid_bias_router_replays_forced_experts(replay, norm, scale):
    router = _router(norm, scale)
    hidden = torch.randn(5, 32)
    natural = router(hidden)
    forced = torch.tensor([[0, 7], [1, 2], [3, 4], [5, 6], [6, 0]])
    rr._STATE.current = forced[:, None, :].expand(5, 1, TOP_K).clone()
    out = rr._router_hook(0)(router, (hidden,), natural)
    assert out is not None, "the sigmoid router kept its natural routing"
    logits, weights, ids = out
    assert router._unsloth_replay_formula in ("sigmoid_renorm", "sigmoid")
    assert torch.equal(ids, forced)
    torch.testing.assert_close(weights, _expected(logits, forced, norm, scale))
    assert rr.routing_replay_stats()["router_calls_replayed"] == 1


def test_natural_ids_reproduce_the_router_weights_exactly(replay):
    router = _router()
    hidden = torch.randn(6, 32)
    natural = router(hidden)
    rr._STATE.current = natural[2][:, None, :].clone()
    logits, weights, ids = rr._router_hook(0)(router, (hidden,), natural)
    assert torch.equal(ids, natural[2])
    torch.testing.assert_close(weights, natural[1], rtol = 0, atol = 0)


def test_scaled_sigmoid_weights_are_not_read_as_softmax(replay):
    # Near-zero logits make every renormalised formula give about 1/k: the scale and the bias tell them apart.
    router = _router()
    router.weight.data.mul_(1e-3)
    hidden = torch.randn(4, 32)
    forced = torch.tensor([[0, 1]] * 4)
    rr._STATE.current = forced[:, None, :].clone()
    rr._router_hook(0)(router, (hidden,), router(hidden))
    assert router._unsloth_replay_formula == "sigmoid_renorm"


def _model(seed = 0):
    from transformers import DeepseekV3ForCausalLM

    torch.manual_seed(seed)
    model = DeepseekV3ForCausalLM(_config()).float().eval()
    for layer in model.model.layers:
        layer.mlp.gate.weight.data.normal_(0, 0.5)
        layer.mlp.gate.e_score_correction_bias.data.copy_(torch.linspace(-1.0, 1.0, NUM_EXPERTS))
    return model


def _natural_ids(model, ids):
    captured = {}
    handles = [
        layer.mlp.gate.register_forward_hook(lambda m, a, out, i = i: captured.__setitem__(i, out[2].detach().clone()))
        for i, layer in enumerate(model.model.layers)
    ]
    with torch.no_grad():
        model(input_ids = torch.tensor([ids]))
    for h in handles:
        h.remove()
    return torch.stack([captured[i] for i in range(NUM_LAYERS)], dim = 1)


def _request(prompt, completion, routed):
    return SimpleNamespace(
        prompt_token_ids = list(prompt),
        outputs = [SimpleNamespace(token_ids = list(completion), routed_experts = np.asarray(routed, dtype = np.uint8)[:-1])],
    )


def test_deepseek_v3_model_replays_rollout_routes_with_grad(replay):
    _router()  # skips on a transformers whose router has another layout
    model = _model()
    prompt, completion = [5, 6, 7, 8], [9, 10, 11, 12, 13]
    ids = prompt + completion
    natural = _natural_ids(model, ids)
    model.zero_grad()
    ref = model(input_ids = torch.tensor([ids])).logits
    ref.sum().backward()
    ref_grad = model.model.layers[0].mlp.gate.weight.grad.clone()

    assert rr.install_routing_replay(model, None, SimpleNamespace(pad_token_id = PAD)) == NUM_LAYERS
    # Same routes as the natural forward: identical logits and router gradient.
    rr.record_routed_experts([_request(prompt, completion, natural.numpy())])
    model.zero_grad()
    out = model(input_ids = torch.tensor([ids])).logits
    out.sum().backward()
    torch.testing.assert_close(out, ref, rtol = 1e-5, atol = 1e-6)
    torch.testing.assert_close(model.model.layers[0].mlp.gate.weight.grad, ref_grad, rtol = 1e-4, atol = 1e-6)
    stats = rr.routing_replay_stats()
    assert stats["router_calls_replayed"] == NUM_LAYERS and stats["changed_selections"] == 0

    # Other routes: the forced experts are used and the router still gets a gradient.
    rr.reset_routing_replay()
    other = (natural + 1) % NUM_EXPERTS
    rr.record_routed_experts([_request(prompt, completion, other.numpy())])
    model.zero_grad()
    forced_out = model(input_ids = torch.tensor([ids])).logits
    forced_out.sum().backward()
    assert not torch.allclose(forced_out, ref)
    assert model.model.layers[0].mlp.gate.weight.grad.abs().sum() > 0
    assert rr.routing_replay_stats()["changed_selections"] > 0


def _bias_router_model(model_type):
    from transformers import AutoConfig, AutoModelForCausalLM

    extra = {
        "minimax_m2": dict(num_local_experts = NUM_EXPERTS),
        "hy_v3": dict(num_local_experts = NUM_EXPERTS, router_scaling_factor = 2.0),
        "lfm2_moe": dict(
            num_experts = NUM_EXPERTS, use_expert_bias = True, num_dense_layers = 0,
            layer_types = ["full_attention"] * NUM_LAYERS, routed_scaling_factor = 1.5, norm_topk_prob = True,
        ),
        "glm4_moe": dict(
            n_routed_experts = NUM_EXPERTS, n_shared_experts = 1, first_k_dense_replace = 0, n_group = 2,
            topk_group = 1, routed_scaling_factor = 2.5, norm_topk_prob = True,
        ),
    }[model_type]
    try:
        config = AutoConfig.for_model(
            model_type, vocab_size = 64, hidden_size = 32, intermediate_size = 64, moe_intermediate_size = 16,
            num_hidden_layers = NUM_LAYERS, num_attention_heads = 4, num_key_value_heads = 2, head_dim = 8,
            num_experts_per_tok = TOP_K, pad_token_id = PAD, **extra,
        )
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(config).float().eval()
    except Exception as error:
        pytest.skip(f"this transformers has no {model_type} ({type(error).__name__})")
    routers = [m for m in model.modules() if "TopkRouter" in type(m).__name__ or "TopKRouter" in type(m).__name__]
    if not routers:
        pytest.skip(f"{model_type} has no top-k router module in this transformers")
    return model, routers


def _bias_of(model, router):
    for module in model.modules():
        for child in module.children():
            if child is router:
                for name in ("e_score_correction_bias", "expert_bias"):
                    value = getattr(module, name, None)
                    if value is None:
                        value = getattr(router, name, None)
                    if torch.is_tensor(value):
                        return name, value
    return None, None


@pytest.mark.parametrize("model_type", ["minimax_m2", "hy_v3", "lfm2_moe", "glm4_moe"])
def test_bias_routers_of_other_families_replay(replay, model_type):
    model, routers = _bias_router_model(model_type)
    router = routers[0]
    router.weight.data.normal_(0, 0.5)
    name, bias = _bias_of(model, router)
    extra = ()
    if bias is not None:
        bias.data.copy_(torch.linspace(-1.0, 1.0, bias.numel()))
        if name in inspect.signature(type(router).forward).parameters:
            extra = (bias,)
    hidden = torch.randn(5, 32)
    natural = router(hidden, *extra)
    if not (isinstance(natural, tuple) and len(natural) == 3):
        pytest.skip(f"{model_type}'s router does not return (logits, weights, ids) here")
    rr._STATE.current = natural[2][:, None, :].clone()
    out = rr._router_hook(0)(router, (hidden, *extra), natural)
    assert out is not None and router._unsloth_replay_formula.startswith("sigmoid")
    torch.testing.assert_close(out[1], natural[1], rtol = 1e-5, atol = 1e-6)

    forced = torch.tensor([[0, 7], [1, 2], [3, 4], [5, 6], [6, 0]])
    rr._STATE.current = forced[:, None, :].clone()
    logits, weights, ids = rr._router_hook(0)(router, (hidden, *extra), router(hidden, *extra))
    assert torch.equal(ids.long(), forced)
    renorm = router._unsloth_replay_formula == "sigmoid_renorm"
    torch.testing.assert_close(
        weights.float(), _expected(logits, forced, renorm, rr._replay_scale(router)), rtol = 1e-4, atol = 1e-6,
    )


def test_lfm2_forced_low_score_experts_keep_the_router_epsilon(replay):
    # LFM2-MoE renormalises with sum + 1e-6. On forced experts whose sigmoid scores are ~1e-7 that epsilon decides
    # the weights; replay must give what the router itself gives for that selection.
    model, routers = _bias_router_model("lfm2_moe")
    router = routers[0]
    name, bias = _bias_of(model, router)
    if bias is None or name not in inspect.signature(type(router).forward).parameters:
        pytest.skip("this transformers' LFM2-MoE router takes no expert_bias")
    torch.manual_seed(0)
    router.weight.data.normal_(0, 0.5)
    hidden = torch.ones(4, 32)
    router.weight.data[6:8] = -16.0 / 32  # logits of -16 for experts 6 and 7
    zero = torch.zeros_like(bias)
    natural = router(hidden, zero)
    forced = torch.tensor([[6, 7]] * 4)
    assert not torch.equal(natural[2].sort(-1).values, forced)
    # The router's own weights for that selection: a bias that makes it pick experts 6 and 7.
    pick = zero.clone()
    pick[6:8] = 10.0
    own = router(hidden, pick)
    assert torch.equal(own[2].sort(-1).values, forced)
    # Prime the formula on a natural call, then replay the forced selection.
    rr._STATE.current = natural[2][:, None, :].clone()
    rr._router_hook(0)(router, (hidden, zero), natural)
    assert router._unsloth_replay_formula == "sigmoid_renorm"
    rr._STATE.current = forced[:, None, :].clone()
    _, weights, ids = rr._router_hook(0)(router, (hidden, zero), router(hidden, zero))
    assert torch.equal(ids.long().sort(-1).values, forced)
    replayed = weights.float().gather(-1, ids.long().argsort(-1))
    own_weights = own[1].float().gather(-1, own[2].argsort(-1))
    torch.testing.assert_close(replayed, own_weights, rtol = 1e-4, atol = 1e-6)
    assert float(own_weights.max()) < 0.5  # the epsilon matters here: 1e-20 would give 0.75 each


def test_router_epsilon_comes_from_the_family_not_the_forward_source(monkeypatch):
    # A compiled, wrapped or patched forward has no readable source (or other text): the epsilon must not change.
    def no_source(obj):
        raise AssertionError("read the router's source")

    monkeypatch.setattr(rr.inspect, "getsource", no_source)

    def wrapped_forward(self, *args, **kwargs):
        return None

    Lfm2 = type("Lfm2MoeTopKRouter", (torch.nn.Module,), {"forward": wrapped_forward})
    Lfm2.__module__ = "transformers.models.lfm2_moe.modeling_lfm2_moe"
    Hy = type("HYV3TopKRouter", (torch.nn.Module,), {"forward": wrapped_forward})
    Hy.__module__ = "transformers.models.hy_v3.modeling_hy_v3"
    Sub = type("MyRouter", (Lfm2,), {})
    Sub.__module__ = "my_package.router"
    assert rr._renorm_eps(Lfm2()) == 1e-6
    assert rr._renorm_eps(Sub()) == 1e-6
    assert rr._renorm_eps(Hy()) == rr._DEFAULT_RENORM_EPS == 1e-20
    stated = Hy()
    stated._unsloth_replay_eps = 1e-9
    assert rr._renorm_eps(stated) == 1e-9


def test_scaled_rollout_weights_reach_a_scaled_router(replay, monkeypatch):
    # DeepSeek-V3 weights sum to routed_scaling_factor (2.5): rollout weights on that scale, close to the trainer's,
    # must be ingested and used, not dropped for summing past 1.
    _router()  # skips on a transformers whose router has another layout
    monkeypatch.setattr(rr, "_VLLM_WEIGHTS_PACKED", True)
    monkeypatch.setattr(rr, "_PACKED_TOP_K", TOP_K)
    model = _model()
    prompt, completion = [5, 6, 7, 8], [9, 10, 11, 12, 13]
    ids = prompt + completion
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
    routes = torch.stack([captured[i][0] for i in range(NUM_LAYERS)], dim = 1)
    weights = torch.stack([captured[i][1] for i in range(NUM_LAYERS)], dim = 1)
    assert float(weights.sum(-1).mean()) == pytest.approx(2.5, rel = 1e-4)
    generator = torch.Generator().manual_seed(1)
    rollout = (weights + 0.02 * torch.rand(weights.shape, generator = generator)).clamp_min(1e-4)
    rollout = rollout / rollout.sum(-1, keepdim = True) * 2.5
    packed = rr._pack_routed_weights(routes.long(), rollout.float()).numpy().astype(np.uint16)
    assert rr.install_routing_replay(model, None, SimpleNamespace(pad_token_id = PAD)) == NUM_LAYERS
    rr.record_routed_experts([
        SimpleNamespace(
            prompt_token_ids = list(prompt),
            outputs = [SimpleNamespace(token_ids = list(completion), routed_experts = packed[:-1])],
        )
    ])
    with torch.no_grad():
        model(input_ids = torch.tensor([ids]))
    assert rr.routing_replay_stats()["replayed_weight_rows"] == (len(ids) - 1) * NUM_LAYERS
