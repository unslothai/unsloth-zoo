"""CPU checks for DeepSeek-V4.1 fast_inference: every vLLM tensor is handed to training as a view under its native
name, and anything that cannot be a view is refused."""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from torch import nn

from unsloth_zoo.vllm_deepseek_v41 import (
    deepseek_v41_attention_backend,
    deepseek_v41_engine_args,
    deepseek_v41_fast_inference_refusal,
    get_deepseek_v41_vllm_state_dict,
    is_deepseek_v41_config,
)

H, QR, HD, NH, PAD_H, I, E, V, ROWS = 16, 8, 32, 2, 64, 8, 4, 50, 11


def _text_config(**kw):
    base = dict(
        model_type = "deepseek_v41_text", hidden_size = H, q_lora_rank = QR, head_dim = HD, num_attention_heads = NH,
        vocab_size = V, n_routed_experts = E, moe_intermediate_size = I,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _config(quantization_config = None):
    return SimpleNamespace(model_type = "deepseek_v41", text_config = _text_config(), quantization_config = quantization_config)


class Linear(nn.Module):
    def __init__(self, out_features, in_features, output_sizes = None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype = torch.bfloat16), requires_grad = False)
        if output_sizes is not None:
            self.output_sizes = list(output_sizes)


class Norm(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n, dtype = torch.bfloat16), requires_grad = False)


def _p(*shape, dtype = torch.float32):
    return nn.Parameter(torch.randn(*shape).to(dtype), requires_grad = False)


class Layer(nn.Module):
    def __init__(self, compress = None, indexer = False, engram = False):
        super().__init__()
        attn = nn.Module()
        attn.fused_wqa_wkv = Linear(QR + HD, H, (QR, HD))
        attn.q_norm, attn.kv_norm = Norm(QR), Norm(HD)
        attn.wq_b = Linear(NH * HD, QR)
        attn.wo_a = Linear(2 * 4, NH * HD // 2)
        attn.wo_b = Linear(H, 8)
        attn.attn_sink = _p(PAD_H)
        attn.compressor = None
        attn.indexer = None
        if compress is not None:
            comp = nn.Module()
            comp.fused_wkv_wgate = Linear(sum(compress), H, compress)
            comp.norm = Norm(HD)
            attn.compressor = comp
        if indexer:
            idx = nn.Module()
            idx.wq_b, idx.weights_proj, idx.wk, idx.k_norm = Linear(16, QR), Linear(2, H), Linear(8, HD), Norm(8)
            attn.indexer = idx
        self.attn = attn
        ffn = nn.Module()
        gate = Linear(E, H)
        gate.e_score_correction_bias = _p(E)
        gate.bias_vl = _p(E)
        ffn.gate = gate
        shared = nn.Module()
        shared.gate_up_proj = Linear(2 * I, H, (I, I))
        shared.down_proj = Linear(H, I)
        ffn.shared_experts = shared
        experts = nn.Module()
        routed = nn.Module()
        routed.w13_weight = _p(E, 2 * I, H, dtype = torch.bfloat16)
        routed.w2_weight = _p(E, H, I, dtype = torch.bfloat16)
        experts.routed_experts = routed
        ffn.experts = experts
        self.ffn = ffn
        for hc in ("attn", "ffn"):
            setattr(self, f"hc_{hc}_fn", _p(24, 4 * H))
            setattr(self, f"hc_{hc}_base", _p(24))
            setattr(self, f"hc_{hc}_scale", _p(3))
        self.attn_norm, self.ffn_norm = Norm(H), Norm(H)
        self.engram = None
        if engram:
            eg = nn.Module()
            eg.wkv = Linear(4 * H, 32)
            eg.q_weight, eg.k_weight = _p(4, H, dtype = torch.bfloat16), _p(4, H, dtype = torch.bfloat16)
            table = nn.Module()
            table.weight = nn.Parameter(torch.randn(ROWS, 32).to(torch.float8_e4m3fn), requires_grad = False)
            table.weight_scale_inv = nn.Parameter(torch.randint(120, 130, (ROWS, 1), dtype = torch.uint8), requires_grad = False)
            table.num_embeddings = ROWS
            eg.embed_tokens = table
            self.engram = eg


def _vllm_model(**layer_kw):
    text = nn.Module()
    text.embed_tokens = Linear(V + 14, H)  # vLLM pads the vocab
    text.layers = nn.ModuleList([Layer(engram = True), Layer(compress = (HD, HD), indexer = True), Layer(compress = (HD,), **layer_kw)])
    text.norm = Norm(H)
    hash_state = nn.Module()
    hash_state.register_buffer("token_map", torch.arange(V, dtype = torch.int32))
    text.engram_hash = hash_state
    lm = nn.Module()
    lm.model = text
    lm.lm_head = Linear(V + 14, H)
    outer = nn.Module()
    outer.language_model = lm
    return outer


def _same_storage(a, b):
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def test_every_vllm_tensor_is_a_view_under_its_native_name():
    vllm = _vllm_model()
    sd = get_deepseek_v41_vllm_state_dict(vllm, _config())
    layers = vllm.language_model.model.layers
    fused = layers[0].attn.fused_wqa_wkv.weight
    assert torch.equal(sd["model.layers.0.self_attn.q_a_proj.weight"], fused[:QR])
    assert torch.equal(sd["model.layers.0.self_attn.kv_proj.weight"], fused[QR:])
    assert _same_storage(sd["model.layers.0.self_attn.kv_proj.weight"], fused)
    assert sd["model.layers.0.self_attn.sinks"].shape == (NH,)
    assert _same_storage(sd["model.layers.0.self_attn.sinks"], layers[0].attn.attn_sink)
    assert torch.equal(sd["model.layers.1.self_attn.compressor.gate_proj.weight"], layers[1].attn.compressor.fused_wkv_wgate.weight[HD:])
    assert "model.layers.2.self_attn.compressor.gate_proj.weight" not in sd  # a gate-less compressor
    assert _same_storage(sd["model.layers.1.self_attn.indexer.k_proj.weight"], layers[1].attn.indexer.wk.weight)
    assert _same_storage(sd["model.layers.2.attn_hc.fn"], layers[2].hc_attn_fn)
    assert _same_storage(sd["model.layers.2.mlp.gate.e_score_correction_bias_vl"], layers[2].ffn.gate.bias_vl)
    assert _same_storage(sd["model.layers.2.mlp.experts.gate_up_proj"], layers[2].ffn.experts.routed_experts.w13_weight)
    up = sd["model.layers.2.mlp.shared_experts.up_proj.weight"]
    assert torch.equal(up, layers[2].ffn.shared_experts.gate_up_proj.weight[I:])
    # engram rows stay fp8; vLLM's uint8 e8m0 bytes are reinterpreted, not converted
    table = layers[0].engram.embed_tokens
    assert sd["model.engram_tables.0.weight"].dtype == torch.float8_e4m3fn
    scale = sd["model.engram_tables.0.weight_scale_inv"]
    assert scale.dtype == torch.float8_e8m0fnu and _same_storage(scale, table.weight_scale_inv)
    assert torch.equal(scale.view(torch.uint8), table.weight_scale_inv)
    assert sd["model.embed_tokens.weight"].shape[0] == V and sd["lm_head.weight"].shape[0] == V
    assert sd._unsloth_dsv41_token_map is vllm.language_model.model.engram_hash.token_map
    # every vLLM parameter backs some training tensor
    used = {t.untyped_storage().data_ptr() for t in sd.values()}
    assert all(p.untyped_storage().data_ptr() in used for p in vllm.parameters())


def test_lora_wrapped_vllm_modules_are_unwrapped():
    vllm = _vllm_model()
    attn = vllm.language_model.model.layers[0].attn
    wrapper = nn.Module()
    wrapper.base_layer = attn.wq_b
    attn.wq_b = wrapper
    sd = get_deepseek_v41_vllm_state_dict(vllm, _config())
    assert _same_storage(sd["model.layers.0.self_attn.q_b_proj.weight"], wrapper.base_layer.weight)


def test_quantized_vllm_linear_is_refused():
    vllm = _vllm_model()
    layer = vllm.language_model.model.layers[1].attn.wo_b
    layer.weight_scale = nn.Parameter(torch.ones(1), requires_grad = False)
    with pytest.raises(NotImplementedError, match = "attn.wo_b is quantized"):
        get_deepseek_v41_vllm_state_dict(vllm, _config())


def test_unmapped_vllm_tensor_is_refused():
    vllm = _vllm_model()
    vllm.language_model.model.layers[2].attn.compressor.ape = _p(2, HD)
    with pytest.raises(NotImplementedError, match = "compressor.ape"):
        get_deepseek_v41_vllm_state_dict(vllm, _config())
    vllm = _vllm_model()
    vllm.language_model.model.layers[0].new_tensor = _p(3)
    with pytest.raises(NotImplementedError, match = "no native DeepSeek-V4.1 name"):
        get_deepseek_v41_vllm_state_dict(vllm, _config())


def test_non_fp8_engram_table_is_refused():
    vllm = _vllm_model()
    table = vllm.language_model.model.layers[0].engram.embed_tokens
    table.weight = nn.Parameter(torch.randn(ROWS, 32), requires_grad = False)
    with pytest.raises(NotImplementedError, match = "engram table"):
        get_deepseek_v41_vllm_state_dict(vllm, _config())


def test_released_fp8_fp4_checkpoint_and_bitsandbytes_are_refused_before_vllm():
    release = {"quant_method": "fp8", "weight_block_size": [32, 32], "scale_fmt": "ue8m0", "expert_dtype": "fp4"}
    reason = deepseek_v41_fast_inference_refusal(_config(release))
    assert reason is not None and "MXFP4" in reason and "expert_dtype fp4" in reason
    assert "bitsandbytes" in deepseek_v41_fast_inference_refusal(_config(), load_in_4bit = True)
    assert deepseek_v41_fast_inference_refusal(_config()) is None
    assert deepseek_v41_fast_inference_refusal(SimpleNamespace(model_type = "deepseek_v3")) is None


def test_engine_args_build_vllm_text_only(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DSV41_VLLM_ATTENTION_BACKEND", "vllm")
    assert is_deepseek_v41_config(_config()) and is_deepseek_v41_config(_text_config())
    assert not is_deepseek_v41_config(SimpleNamespace(model_type = "deepseek_v4"))
    assert deepseek_v41_engine_args(_config()) == {"language_model_only": True, "limit_mm_per_prompt": {"image": 0}}
    with pytest.raises(NotImplementedError, match = "text model"):
        deepseek_v41_engine_args(_config(), is_vision_model = True)


def test_vllm_utils_routes_deepseek_v41_to_the_dedicated_path():
    import inspect
    from unsloth_zoo import vllm_utils
    assert "get_deepseek_v41_vllm_state_dict(vllm_internals, config)" in inspect.getsource(vllm_utils._get_vllm_state_dict)
    assert "convert_deepseek_v41_vllm_to_huggingface" in inspect.getsource(vllm_utils.convert_vllm_to_huggingface)
    assert vllm_utils._get_multimodal_engine_args(_config(), False)["language_model_only"] is True


def test_load_vllm_refuses_an_unshareable_deepseek_v41_before_the_fp8_moe_check():
    import inspect
    from unsloth_zoo import vllm_utils
    source = inspect.getsource(vllm_utils.load_vllm)
    assert source.index("deepseek_v41_fast_inference_refusal(config") < source.index("fp8_moe_checkpoint_scheme(config)")


def test_bfloat16_kv_backend_is_picked_on_sm100_only(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DSV41_VLLM_ATTENTION_BACKEND", raising = False)
    assert deepseek_v41_attention_backend((10, 0)) == "FLASHINFER_MLA_SPARSE_DSV41"
    for capability in ((9, 0), (12, 0), (8, 0)):
        assert deepseek_v41_attention_backend(capability) is None
    monkeypatch.setenv("UNSLOTH_DSV41_VLLM_ATTENTION_BACKEND", "vllm")
    assert deepseek_v41_attention_backend((10, 0)) is None
    monkeypatch.setenv("UNSLOTH_DSV41_VLLM_ATTENTION_BACKEND", "FLASHMLA_SPARSE_DSV41")
    assert deepseek_v41_attention_backend((10, 0)) == "FLASHMLA_SPARSE_DSV41"


def test_vllm_lora_fallback_is_refused_for_deepseek_v41():
    from unsloth_zoo import vllm_utils
    model = SimpleNamespace(config = _config(), _unsloth_merged_rollout_state = SimpleNamespace(reason = "o_a_proj uses DoRA"))
    with pytest.raises(NotImplementedError, match = "merged-weight rollout.*o_a_proj uses DoRA"):
        vllm_utils._check_lora_is_servable(model, ["base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight"], "x", None)


def test_automatic_merged_rollout_folds_deepseek_v41_experts():
    import inspect
    from unsloth_zoo import vllm_utils
    source = inspect.getsource(vllm_utils._prepare_merged_rollout_locked)
    gate = source.index("the automatic default does not fold MoE expert LoRA")
    assert "not is_deepseek_v41_config(getattr(model, \"config\", None))" in source[:gate]
