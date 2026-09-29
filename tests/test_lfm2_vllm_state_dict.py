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

import sys, os, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch


def _linear(weight, output_sizes = None):
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(weight.clone(), requires_grad = False)
    module.bias = None
    if output_sizes is not None:
        module.output_sizes = output_sizes
    return module


def _norm(weight):
    return _linear(weight)


def _vllm_moe_from_hf(sd, p, runner, backend):
    """Mirrors vllm/model_executor/models/lfm2_moe.py Lfm2MoeSparseMoeBlock."""
    ff = torch.nn.Module()
    ff.gate = _linear(sd[f"{p}.feed_forward.gate.weight"])
    ff.gate.e_score_correction_bias = torch.nn.Parameter(sd[f"{p}.feed_forward.expert_bias"].clone(), requires_grad = False)
    routed = torch.nn.Module()
    routed.w13_weight = torch.nn.Parameter(sd[f"{p}.feed_forward.experts.gate_up_proj"].clone(), requires_grad = False)
    routed.w2_weight = torch.nn.Parameter(sd[f"{p}.feed_forward.experts.down_proj"].clone(), requires_grad = False)
    routed.quant_method = types.SimpleNamespace(unquantized_backend = types.SimpleNamespace(name = backend))
    if runner:
        # vLLM >= 0.30: FusedMoEFactory returns a MoERunner holding the weights on routed_experts
        ff.experts = torch.nn.Module()
        ff.experts.routed_experts = routed
    else:
        ff.experts = routed
    return ff


def _vllm_lfm2_from_hf(hf, fused_ff_name = "w13", runner = True, backend = "TRITON"):
    """Mirrors vllm/model_executor/models/lfm2.py module names and fusions."""
    sd = hf.state_dict()
    config = hf.config
    root = torch.nn.Module()
    model = torch.nn.Module()
    model.embed_tokens = _linear(sd["model.embed_tokens.weight"])
    model.embedding_norm = _norm(sd["model.embedding_norm.weight"])
    layers = torch.nn.ModuleList()
    for kk, layer_type in enumerate(config.layer_types):
        p = f"model.layers.{kk}"
        layer = torch.nn.Module()
        if layer_type == "full_attention":
            attn = torch.nn.Module()
            q, k, v = (sd[f"{p}.self_attn.{x}_proj.weight"] for x in "qkv")
            attn.qkv_proj = _linear(torch.cat([q, k, v]), [q.shape[0], k.shape[0], v.shape[0]])
            attn.out_proj = _linear(sd[f"{p}.self_attn.out_proj.weight"])
            attn.q_layernorm = _norm(sd[f"{p}.self_attn.q_layernorm.weight"])
            attn.k_layernorm = _norm(sd[f"{p}.self_attn.k_layernorm.weight"])
            layer.self_attn = attn
        else:
            conv = torch.nn.Module()
            in_proj = sd[f"{p}.conv.in_proj.weight"]
            conv.in_proj = _linear(in_proj, [in_proj.shape[0] // 3] * 3)
            conv.out_proj = _linear(sd[f"{p}.conv.out_proj.weight"])
            conv.conv = _linear(sd[f"{p}.conv.conv.weight"])
            layer.short_conv = conv
        if f"{p}.feed_forward.gate.weight" in sd:
            layer.feed_forward = _vllm_moe_from_hf(sd, p, runner, backend)
        else:
            ff = torch.nn.Module()
            w1, w3 = sd[f"{p}.feed_forward.w1.weight"], sd[f"{p}.feed_forward.w3.weight"]
            setattr(ff, fused_ff_name, _linear(torch.cat([w1, w3]), [w1.shape[0], w3.shape[0]]))
            ff.w2 = _linear(sd[f"{p}.feed_forward.w2.weight"])
            layer.feed_forward = ff
        layer.operator_norm = _norm(sd[f"{p}.operator_norm.weight"])
        layer.ffn_norm = _norm(sd[f"{p}.ffn_norm.weight"])
        layers.append(layer)
    model.layers = layers
    root.model = model
    root.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"], "w13": ["w1", "w3"], "in_proj": ["in_proj"]}
    engine = types.SimpleNamespace(model_executor = types.SimpleNamespace(
        driver_worker = types.SimpleNamespace(model_runner = types.SimpleNamespace(model = root))))
    return types.SimpleNamespace(llm_engine = engine)


# vLLM calls the fused gate/up w13; older releases called it w1
@pytest.mark.parametrize("fused_ff_name", ["w13", "w1"])
def test_lfm2_vllm_state_dict_matches_hf(monkeypatch, fused_ff_name):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Lfm2Config"):
        pytest.skip("transformers has no LFM2")
    from unsloth_zoo import vllm_utils
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 0))

    config = transformers.Lfm2Config(
        vocab_size = 64, hidden_size = 16, intermediate_size = 48, num_hidden_layers = 3,
        num_attention_heads = 4, num_key_value_heads = 2, conv_L_cache = 3,
        layer_types = ["conv", "full_attention", "conv"], tie_word_embeddings = True,
    )
    torch.manual_seed(0)
    hf = transformers.Lfm2ForCausalLM(config)
    for param in hf.parameters():
        torch.nn.init.normal_(param)

    state_dict, _ = vllm_utils._get_vllm_state_dict(
        _vllm_lfm2_from_hf(hf, fused_ff_name), return_state_dict = True, config = config,
    )
    expected = hf.state_dict()
    assert set(expected) - set(state_dict) == set()
    for key, value in expected.items():
        assert torch.equal(state_dict[key], value), key


def _tiny_lfm2_moe(monkeypatch):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Lfm2MoeConfig"):
        pytest.skip("transformers has no LFM2-MoE")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 0))
    config = transformers.Lfm2MoeConfig(
        vocab_size = 64, hidden_size = 16, intermediate_size = 48, moe_intermediate_size = 8,
        num_hidden_layers = 3, num_attention_heads = 4, num_key_value_heads = 2, conv_L_cache = 3,
        layer_types = ["conv", "full_attention", "conv"], num_dense_layers = 1, num_experts = 4,
        num_experts_per_tok = 2, tie_word_embeddings = True,
    )
    torch.manual_seed(0)
    hf = transformers.Lfm2MoeForCausalLM(config)
    for param in hf.parameters():
        torch.nn.init.normal_(param)
    for name, buffer in hf.named_buffers():
        if name.endswith("expert_bias"):
            torch.nn.init.normal_(buffer)
    return hf, config


@pytest.mark.parametrize("runner", [True, False])
def test_lfm2_moe_vllm_state_dict_matches_hf(monkeypatch, runner):
    from unsloth_zoo import vllm_utils
    hf, config = _tiny_lfm2_moe(monkeypatch)
    state_dict, _ = vllm_utils._get_vllm_state_dict(
        _vllm_lfm2_from_hf(hf, runner = runner), return_state_dict = True, config = config,
    )
    expected = hf.state_dict()
    assert set(expected) - set(state_dict) == set()
    for key, value in expected.items():
        assert torch.equal(state_dict[key], value), key


def test_lfm2_moe_reordered_expert_layout_raises(monkeypatch):
    from unsloth_zoo import vllm_utils
    hf, config = _tiny_lfm2_moe(monkeypatch)
    with pytest.raises(NotImplementedError, match = "FLASHINFER_CUTLASS"):
        vllm_utils._get_vllm_state_dict(
            _vllm_lfm2_from_hf(hf, backend = "FLASHINFER_CUTLASS"), return_state_dict = True, config = config,
        )


def test_unknown_feed_forward_raises_instead_of_skipping(monkeypatch):
    from unsloth_zoo import vllm_utils
    hf, config = _tiny_lfm2_moe(monkeypatch)
    llm = _vllm_lfm2_from_hf(hf)
    layers = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
    layers[1].feed_forward = torch.nn.Module()
    with pytest.raises(NotImplementedError, match = "layer 1"):
        vllm_utils._get_vllm_state_dict(llm, return_state_dict = True, config = config)


def test_lfm2_moe_blocked_trtllm_layout_raises(monkeypatch):
    # FusedMoEModularMethod wraps the real method; TRTLLM stores w13 as 4D blocks
    from unsloth_zoo import vllm_utils
    hf, config = _tiny_lfm2_moe(monkeypatch)
    llm = _vllm_lfm2_from_hf(hf)
    layers = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
    routed = layers[1].feed_forward.experts.routed_experts
    routed.quant_method = types.SimpleNamespace(old_quant_method = routed.quant_method)
    E, two_i, H = routed.w13_weight.shape
    routed.w13_weight = torch.nn.Parameter(routed.w13_weight.data.reshape(E, H // 8, two_i, 8), requires_grad = False)
    with pytest.raises(NotImplementedError, match = "w13"):
        vllm_utils._get_vllm_state_dict(llm, return_state_dict = True, config = config)
