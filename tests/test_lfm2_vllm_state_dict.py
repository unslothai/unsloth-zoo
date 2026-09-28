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


def _vllm_lfm2_from_hf(hf, fused_ff_name):
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
