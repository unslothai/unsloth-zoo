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

import json
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen3_5")

from safetensors.torch import save_file  # noqa: E402

from unsloth_zoo.speculative.mtp import (  # noqa: E402
    Qwen35MTPHead,
    build_mtp_head,
    find_mtp_tensor_names,
    load_mtp_state_dict,
)

VOCAB = 97
HIDDEN = 64


def _config():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    return Qwen3_5TextConfig(
        vocab_size = VOCAB,
        hidden_size = HIDDEN,
        intermediate_size = 128,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
        num_hidden_layers = 2,
        layer_types = ["linear_attention", "full_attention"],
        mtp_num_hidden_layers = 1,
    )


def _random_state_dict(config, seed = 0):
    gen = torch.Generator().manual_seed(seed)
    head = build_mtp_head(config)
    sd = {}
    for name, tensor in head.state_dict().items():
        # Norm weights are (1 + w); keep them away from zero so a dropped norm shows.
        scale = 0.3 if "norm" in name else 1.0 / math.sqrt(tensor.shape[-1])
        sd[name] = torch.randn(tensor.shape, generator = gen) * scale
    return sd


def _write_checkpoint(tmp_path, config, sd, prefix = "mtp.", with_mtp = True):
    tensors = {"model.language_model.embed_tokens.weight": torch.randn(VOCAB, HIDDEN)}
    if with_mtp:
        tensors.update({prefix + k: v.contiguous() for k, v in sd.items()})
    save_file(tensors, str(tmp_path / "model-00001-of-00001.safetensors"))
    index = {"metadata": {}, "weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"mtp_num_hidden_layers": 1}}))
    return tmp_path


@pytest.fixture
def setup(tmp_path):
    torch.manual_seed(0)
    config = _config()
    sd = _random_state_dict(config)
    ckpt = _write_checkpoint(tmp_path, config, sd)
    head = build_mtp_head(config, state_dict = load_mtp_state_dict(ckpt)[0], dtype = torch.float32)
    embed = torch.nn.Embedding(VOCAB, HIDDEN)
    return SimpleNamespace(config = config, sd = sd, ckpt = ckpt, head = head.eval(), embed = embed)


def _rms(x, w, eps = 1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + eps) * (1.0 + w)


def _reference_catch_up(sd, config, h_prev, emb, positions):
    """llama.cpp graph_mtp written out with plain ops over a causal sequence (one batch row)."""
    S = emb.shape[0]
    D = config.head_dim
    Hq, Hkv = config.num_attention_heads, config.num_key_value_heads
    p = "layers.0."
    e = _rms(emb, sd["pre_fc_norm_embedding.weight"])
    h = _rms(h_prev, sd["pre_fc_norm_hidden.weight"])
    cur = torch.cat([e, h], -1) @ sd["fc.weight"].T
    inp_sa = cur
    x = _rms(cur, sd[p + "input_layernorm.weight"])
    q_full = (x @ sd[p + "self_attn.q_proj.weight"].T).view(S, Hq, 2 * D)
    q = _rms(q_full[..., :D], sd[p + "self_attn.q_norm.weight"])
    gate = q_full[..., D:].reshape(S, Hq * D)
    k = _rms((x @ sd[p + "self_attn.k_proj.weight"].T).view(S, Hkv, D), sd[p + "self_attn.k_norm.weight"])
    v = (x @ sd[p + "self_attn.v_proj.weight"].T).view(S, Hkv, D)
    rot = int(D * config.rope_parameters["partial_rotary_factor"])
    inv = 1.0 / (config.rope_parameters["rope_theta"] ** (torch.arange(0, rot, 2).float() / rot))
    ang = positions.float()[:, None] * inv[None]
    cos, sin = torch.cat([ang, ang], -1).cos(), torch.cat([ang, ang], -1).sin()

    def rope(t):
        a, b = t[..., :rot], t[..., rot:]
        half = torch.cat([-a[..., rot // 2 :], a[..., : rot // 2]], -1)
        return torch.cat([a * cos[:, None] + half * sin[:, None], b], -1)

    q, k = rope(q), rope(k)
    k = k.repeat_interleave(Hq // Hkv, dim = 1)
    v = v.repeat_interleave(Hq // Hkv, dim = 1)
    scores = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(D)
    scores = scores.masked_fill(torch.ones(S, S).triu(1).bool(), float("-inf"))
    attn = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v).reshape(S, Hq * D)
    attn = (attn * torch.sigmoid(gate)) @ sd[p + "self_attn.o_proj.weight"].T
    cur = attn + inp_sa
    ffn_in = _rms(cur, sd[p + "post_attention_layernorm.weight"])
    cur = cur + (_reference_moe(sd, config, ffn_in) if p + "mlp.gate.weight" in sd else _swiglu(ffn_in, sd, p + "mlp."))
    return _rms(cur, sd["norm.weight"])


def _swiglu(x, sd, p):
    act = torch.nn.functional.silu(x @ sd[p + "gate_proj.weight"].T) * (x @ sd[p + "up_proj.weight"].T)
    return act @ sd[p + "down_proj.weight"].T


def _reference_moe(sd, config, x):
    """Token-by-token reference of the llama.cpp qwen35moe graph_mtp FFN: softmax router, top-k with renormalised
    weights, SiLU experts (gate_up_proj = [gate; up] on dim -2), plus sigmoid-gated shared expert."""
    p = "layers.0.mlp."
    n_ff = config.moe_intermediate_size
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        xt = x[t]
        probs = (xt @ sd[p + "gate.weight"].T).softmax(-1)
        w, idx = probs.topk(config.num_experts_per_tok)
        w = w / w.sum()
        for weight, e in zip(w, idx):
            gu = sd[p + "experts.gate_up_proj"][e]
            act = torch.nn.functional.silu(gu[:n_ff] @ xt) * (gu[n_ff:] @ xt)
            out[t] += weight * (sd[p + "experts.down_proj"][e] @ act)
        shared = _swiglu(xt[None], sd, p + "shared_expert.")[0]
        out[t] += torch.sigmoid(xt @ sd[p + "shared_expert_gate.weight"][0]) * shared
    return out


def test_load_round_trip_and_names(setup):
    sd, info = load_mtp_state_dict(setup.ckpt)
    assert info["prefix"] == "mtp."
    assert info["num_layers"] == 1
    assert set(sd) == set(setup.sd)
    assert set(setup.head.state_dict()) == set(setup.sd)
    for name, tensor in setup.head.state_dict().items():
        torch.testing.assert_close(tensor, setup.sd[name])
    names = find_mtp_tensor_names(setup.ckpt)
    assert all(n.startswith("mtp.") for n in names) and len(names) == len(setup.sd)


def test_nested_prefix(tmp_path, setup):
    ckpt = _write_checkpoint(tmp_path, setup.config, setup.sd, prefix = "model.language_model.mtp.")
    sd, info = load_mtp_state_dict(ckpt)
    assert info["prefix"] == "model.language_model.mtp."
    assert set(sd) == set(setup.sd)


def test_missing_mtp_tensors_raises(tmp_path, setup):
    ckpt = _write_checkpoint(tmp_path, setup.config, setup.sd, with_mtp = False)
    with pytest.raises(ValueError, match = "declares"):
        load_mtp_state_dict(ckpt)


def test_other_types_refused():
    with pytest.raises(NotImplementedError):
        Qwen35MTPHead(SimpleNamespace(model_type = "llama"))


def _moe_config():
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    return Qwen3_5MoeTextConfig(
        vocab_size = VOCAB,
        hidden_size = HIDDEN,
        moe_intermediate_size = 24,
        shared_expert_intermediate_size = 32,
        num_experts = 6,
        num_experts_per_tok = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
        num_hidden_layers = 2,
        layer_types = ["linear_attention", "full_attention"],
        mtp_num_hidden_layers = 1,
    )


@pytest.fixture
def moe_setup(tmp_path):
    pytest.importorskip("transformers.models.qwen3_5_moe")
    torch.manual_seed(0)
    config = _moe_config()
    sd = _random_state_dict(config)
    ckpt = _write_checkpoint(tmp_path, config, sd)
    head = build_mtp_head(config, state_dict = load_mtp_state_dict(ckpt)[0], dtype = torch.float32)
    return SimpleNamespace(config = config, sd = sd, ckpt = ckpt, head = head.eval(), embed = torch.nn.Embedding(VOCAB, HIDDEN))


def test_moe_names_and_shapes_match_the_checkpoint_layout(moe_setup):
    names = set(moe_setup.head.state_dict())
    for extra in ("mlp.gate.weight", "mlp.experts.gate_up_proj", "mlp.experts.down_proj",
                  "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight",
                  "mlp.shared_expert.down_proj.weight", "mlp.shared_expert_gate.weight"):
        assert "layers.0." + extra in names
    sd = moe_setup.head.state_dict()
    c = moe_setup.config
    # Same orientation as the shipped Qwen3.6-35B-A3B checkpoint: [E, 2I, H] and [E, H, I].
    assert tuple(sd["layers.0.mlp.experts.gate_up_proj"].shape) == (c.num_experts, 2 * c.moe_intermediate_size, HIDDEN)
    assert tuple(sd["layers.0.mlp.experts.down_proj"].shape) == (c.num_experts, HIDDEN, c.moe_intermediate_size)
    assert set(load_mtp_state_dict(moe_setup.ckpt)[0]) == names
    for name, tensor in sd.items():
        torch.testing.assert_close(tensor, moe_setup.sd[name])


@torch.no_grad()
def test_moe_matches_independent_reference(moe_setup):
    S = 9
    ids = torch.randint(0, VOCAB, (1, S))
    hidden = torch.randn(1, S, HIDDEN)
    step0 = moe_setup.head.forward_train(hidden, ids, moe_setup.embed, num_steps = 1)[0]
    h_prev = torch.cat([torch.zeros(1, HIDDEN), hidden[0, :-1]], 0)
    ref = _reference_catch_up(moe_setup.sd, moe_setup.config, h_prev, moe_setup.embed(ids[0]), torch.arange(S))
    torch.testing.assert_close(step0["hidden"][0], ref, atol = 1e-5, rtol = 1e-4)


def test_moe_gradients_reach_router_experts_and_shared_expert(moe_setup):
    head = moe_setup.head.train()
    moe_setup.embed.weight.requires_grad_(False)
    ids = torch.randint(0, VOCAB, (2, 8))
    steps = head.forward_train(torch.randn(2, 8, HIDDEN), ids, moe_setup.embed, num_steps = 2)
    loss = sum(
        torch.nn.functional.cross_entropy((s["hidden"] @ moe_setup.embed.weight.T).flatten(0, 1), s["labels"].flatten(), ignore_index = -100)
        for s in steps
    )
    loss.backward()
    for name, param in head.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
    mlp = head.layers[0].mlp
    assert mlp.gate.weight.grad.abs().sum() > 0
    assert mlp.experts.gate_up_proj.grad.abs().sum() > 0
    assert mlp.shared_expert_gate.weight.grad.abs().sum() > 0


def test_moe_random_init_is_finite():
    head = build_mtp_head(_moe_config())
    for name, param in head.named_parameters():
        assert torch.isfinite(param).all(), name
    assert head.layers[0].mlp.experts.gate_up_proj.abs().sum() > 0


@torch.no_grad()
def test_matches_independent_reference(setup):
    B, S = 2, 11
    ids = torch.randint(0, VOCAB, (B, S))
    hidden = torch.randn(B, S, HIDDEN)
    step0 = setup.head.forward_train(hidden, ids, setup.embed, num_steps = 1)[0]
    for b in range(B):
        h_prev = torch.cat([torch.zeros(1, HIDDEN), hidden[b, :-1]], 0)
        ref = _reference_catch_up(setup.sd, setup.config, h_prev, setup.embed(ids[b]), torch.arange(S))
        torch.testing.assert_close(step0["hidden"][b], ref, atol = 1e-5, rtol = 1e-4)


@torch.no_grad()
def test_sentinel_alignment(setup):
    S = 9
    ids = torch.arange(10, 10 + S)[None]
    steps = setup.head.forward_train(torch.randn(1, S, HIDDEN), ids, setup.embed, num_steps = 3)
    for k, step in enumerate(steps):
        for p in range(S):
            want = 10 + p + k + 1 if (p >= 1 and p + k + 1 < S) else -100
            assert step["labels"][0, p].item() == want, (k, p)
            assert step["positions"][0, p].item() == p + k


@torch.no_grad()
def test_padding_masks_labels(setup):
    S = 8
    ids = torch.randint(0, VOCAB, (1, S))
    mask = torch.ones(1, S, dtype = torch.long)
    mask[0, 6:] = 0
    steps = setup.head.forward_train(torch.randn(1, S, HIDDEN), ids, setup.embed, attention_mask = mask, num_steps = 2)
    assert (steps[0]["labels"][0, 5:] == -100).all()
    assert (steps[1]["labels"][0, 4:] == -100).all()
    assert torch.isfinite(steps[1]["hidden"]).all()


@torch.no_grad()
def test_causality(setup):
    S, p = 12, 5
    ids = torch.randint(0, VOCAB, (1, S))
    hidden = torch.randn(1, S, HIDDEN)
    base = setup.head.forward_train(hidden, ids, setup.embed, num_steps = 2)
    ids2, hidden2 = ids.clone(), hidden.clone()
    # Step k at p reads tokens <= p + k and target h <= p - 1.
    ids2[:, p + 2 :] = (ids2[:, p + 2 :] + 1) % VOCAB
    hidden2[:, p:] = torch.randn_like(hidden2[:, p:])
    pert = setup.head.forward_train(hidden2, ids2, setup.embed, num_steps = 2)
    torch.testing.assert_close(base[0]["hidden"][:, : p + 1], pert[0]["hidden"][:, : p + 1])
    torch.testing.assert_close(base[1]["hidden"][:, : p + 1], pert[1]["hidden"][:, : p + 1])
    assert not torch.allclose(base[0]["hidden"][:, p + 2 :], pert[0]["hidden"][:, p + 2 :])


@torch.no_grad()
def test_full_sequence_equals_incremental_draft(setup):
    """Training step k == llama.cpp: catch-up over the prefix, then k recursive draft steps."""
    S, num_steps = 10, 3
    ids = torch.randint(0, VOCAB, (1, S))
    hidden = torch.randn(1, S, HIDDEN)
    steps = setup.head.forward_train(hidden, ids, setup.embed, num_steps = num_steps)
    h_prev = torch.cat([torch.zeros(1, 1, HIDDEN), hidden[:, :-1]], 1)
    for p0 in (1, 4, 6):
        cache = Qwen35MTPHead.new_cache()
        out = setup.head.draft(h_prev[:, : p0 + 1], ids[:, : p0 + 1], torch.arange(p0 + 1)[None], setup.embed, cache)
        torch.testing.assert_close(out[:, -1], steps[0]["hidden"][:, p0], atol = 1e-5, rtol = 1e-4)
        h = out[:, -1:]
        for k in range(1, num_steps):
            if p0 + k >= S:
                break
            h = setup.head.draft(h, ids[:, p0 + k : p0 + k + 1], torch.tensor([[p0 + k]]), setup.embed, cache)
            torch.testing.assert_close(h[:, 0], steps[k]["hidden"][:, p0], atol = 1e-5, rtol = 1e-4)


def test_gradients_reach_head_only(setup):
    head = setup.head.train()
    for p in setup.embed.parameters():
        p.requires_grad_(False)
    S = 7
    ids = torch.randint(0, VOCAB, (2, S))
    steps = head.forward_train(torch.randn(2, S, HIDDEN), ids, setup.embed, num_steps = 2)
    lm_head = setup.embed.weight
    loss = sum(
        torch.nn.functional.cross_entropy((s["hidden"] @ lm_head.T).flatten(0, 1), s["labels"].flatten(), ignore_index = -100)
        for s in steps
    )
    loss.backward()
    assert setup.embed.weight.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
