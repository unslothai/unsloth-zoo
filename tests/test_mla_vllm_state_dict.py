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

"""fast_inference on DeepSeek-V2 / V3 multi-head latent attention: vLLM's DeepseekV2MLAAttention and
DeepseekV2MoE module tree (vllm/model_executor/models/deepseek_v2.py) maps back onto the HF names, every
tensor aliasing vLLM's storage, and the rebuilt HF model computes what the HF reference computes."""

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


def _gated_mlp(sd, p):
    mlp = torch.nn.Module()
    g, u = sd[f"{p}.gate_proj.weight"], sd[f"{p}.up_proj.weight"]
    mlp.gate_up_proj = _linear(torch.cat([g, u]), [g.shape[0], u.shape[0]])
    mlp.down_proj = _linear(sd[f"{p}.down_proj.weight"])
    return mlp


def _stacked_experts(sd, p, n_experts):
    """vLLM's w13 [E, 2I, H] / w2 [E, H, I]; transformers < 5 keeps one gated MLP per expert."""
    if f"{p}.gate_up_proj" in sd:
        return sd[f"{p}.gate_up_proj"].clone(), sd[f"{p}.down_proj"].clone()
    w13 = torch.stack([torch.cat([sd[f"{p}.{e}.gate_proj.weight"], sd[f"{p}.{e}.up_proj.weight"]]) for e in range(n_experts)])
    w2 = torch.stack([sd[f"{p}.{e}.down_proj.weight"] for e in range(n_experts)])
    return w13, w2


def _stacked_hf(hf):
    return hasattr(hf.model.layers[1].mlp.experts, "gate_up_proj")


def _vllm_deepseek_from_hf(hf, fused_qkv_a = True, fused_shared_experts = False, indexer = False):
    """Mirrors vLLM's DeepseekV2ForCausalLM: fused_qkv_a_proj = [q_a_proj | kv_a_proj_with_mqa] when
    q_lora_rank is set, a MoE block of gate (+ e_score_correction_bias), shared_experts and routed experts."""
    sd = hf.state_dict()
    config = hf.config
    root = torch.nn.Module()
    model = torch.nn.Module()
    model.embed_tokens = _linear(sd["model.embed_tokens.weight"])
    model.norm = _linear(sd["model.norm.weight"])
    layers = torch.nn.ModuleList()
    for kk in range(config.num_hidden_layers):
        p = f"model.layers.{kk}"
        a = f"{p}.self_attn"
        layer = torch.nn.Module()
        attn = torch.nn.Module()
        if config.q_lora_rank is not None:
            q_a, kv_a = sd[f"{a}.q_a_proj.weight"], sd[f"{a}.kv_a_proj_with_mqa.weight"]
            if fused_qkv_a:
                attn.fused_qkv_a_proj = _linear(torch.cat([q_a, kv_a]), [q_a.shape[0], kv_a.shape[0]])
            else:
                attn.q_a_proj = _linear(q_a)
                attn.kv_a_proj_with_mqa = _linear(kv_a)
            attn.q_a_layernorm = _linear(sd[f"{a}.q_a_layernorm.weight"])
            attn.q_b_proj = _linear(sd[f"{a}.q_b_proj.weight"])
        else:
            attn.kv_a_proj_with_mqa = _linear(sd[f"{a}.kv_a_proj_with_mqa.weight"])
            attn.q_proj = _linear(sd[f"{a}.q_proj.weight"])
        attn.kv_a_layernorm = _linear(sd[f"{a}.kv_a_layernorm.weight"])
        attn.kv_b_proj = _linear(sd[f"{a}.kv_b_proj.weight"])
        attn.o_proj = _linear(sd[f"{a}.o_proj.weight"])
        attn.indexer = torch.nn.Module() if indexer else None
        attn.mla_attn = types.SimpleNamespace(is_sparse = indexer)
        layer.self_attn = attn
        if f"{p}.mlp.gate.weight" in sd:
            moe = torch.nn.Module()
            moe.gate = _linear(sd[f"{p}.mlp.gate.weight"])
            bias = sd.get(f"{p}.mlp.gate.e_score_correction_bias")
            moe.gate.e_score_correction_bias = None if bias is None else \
                torch.nn.Parameter(bias.float().clone(), requires_grad = False)
            moe.is_fused_shared_expert_enabled = fused_shared_experts
            moe.shared_experts = None if fused_shared_experts else _gated_mlp(sd, f"{p}.mlp.shared_experts")
            routed = torch.nn.Module()
            w13, w2 = _stacked_experts(sd, f"{p}.mlp.experts", config.n_routed_experts)
            routed.w13_weight = torch.nn.Parameter(w13, requires_grad = False)
            routed.w2_weight = torch.nn.Parameter(w2, requires_grad = False)
            routed.quant_method = types.SimpleNamespace(unquantized_backend = types.SimpleNamespace(name = "TRITON"))
            moe.experts = torch.nn.Module()
            moe.experts.routed_experts = routed
            layer.mlp = moe
        else:
            layer.mlp = _gated_mlp(sd, f"{p}.mlp")
        layer.input_layernorm = _linear(sd[f"{p}.input_layernorm.weight"])
        layer.post_attention_layernorm = _linear(sd[f"{p}.post_attention_layernorm.weight"])
        layers.append(layer)
    model.layers = layers
    root.model = model
    root.lm_head = _linear(sd["lm_head.weight"])
    root.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    if config.q_lora_rank is not None:
        root.packed_modules_mapping["fused_qkv_a_proj"] = ["q_a_proj", "kv_a_proj_with_mqa"]
    engine = types.SimpleNamespace(model_executor = types.SimpleNamespace(
        driver_worker = types.SimpleNamespace(model_runner = types.SimpleNamespace(model = root))))
    return types.SimpleNamespace(llm_engine = engine)


def _tiny(monkeypatch, version, q_lora_rank):
    transformers = pytest.importorskip("transformers")
    cls_config = getattr(transformers, f"DeepseekV{version}Config", None)
    cls_model = getattr(transformers, f"DeepseekV{version}ForCausalLM", None)
    if cls_config is None or cls_model is None:
        pytest.skip(f"transformers has no DeepSeek-V{version}")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 0))
    kwargs = dict(
        vocab_size = 64, hidden_size = 32, intermediate_size = 48, moe_intermediate_size = 8,
        num_hidden_layers = 2, num_attention_heads = 2, num_key_value_heads = 2, q_lora_rank = q_lora_rank,
        kv_lora_rank = 16, qk_rope_head_dim = 8, qk_nope_head_dim = 8, v_head_dim = 8, first_k_dense_replace = 1,
        n_routed_experts = 4, num_experts_per_tok = 2, n_shared_experts = 1, tie_word_embeddings = False,
    )
    if version == 3:
        kwargs.update(n_group = 1, topk_group = 1)
    else:
        kwargs.update(topk_method = "greedy")
    config = cls_config(**kwargs)
    torch.manual_seed(0)
    hf = cls_model(config).float().eval()
    with torch.no_grad():
        for param in hf.parameters():
            param.normal_(0, 0.2)
        for name, buffer in hf.named_buffers():
            if name.endswith("e_score_correction_bias"):
                buffer.normal_()
    return hf, config


CASES = [(3, 16), (3, None), (2, None), (2, 16)]


@pytest.mark.parametrize("version,q_lora_rank", CASES)
def test_mla_state_dict_matches_hf_and_aliases_vllm(monkeypatch, version, q_lora_rank):
    from unsloth_zoo import vllm_utils
    hf, config = _tiny(monkeypatch, version, q_lora_rank)
    if not _stacked_hf(hf): pytest.skip("transformers < 5 keeps the experts per expert; see the logits test")
    llm = _vllm_deepseek_from_hf(hf)
    vllm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    state_dict, quant_state_dict = vllm_utils._get_vllm_state_dict(llm, return_state_dict = True, config = config)
    expected = hf.state_dict()
    assert set(expected) - set(state_dict) == set()
    for key, value in expected.items():
        assert torch.equal(state_dict[key].to(value.dtype), value), key
    # Shared, not copied: every tensor lives in vLLM's storage (fused_qkv_a_proj rows included).
    storages = {t.untyped_storage().data_ptr() for t in vllm_model.parameters()}
    for key in expected:
        assert quant_state_dict[key].untyped_storage().data_ptr() in storages, key
    if q_lora_rank is not None:
        fused = vllm_model.model.layers[0].self_attn.fused_qkv_a_proj.weight
        assert quant_state_dict["model.layers.0.self_attn.q_a_proj.weight"].data_ptr() == fused.data_ptr()


def test_mla_unfused_qkv_a_still_maps(monkeypatch):
    # vLLM's DeepseekV2Attention (VLLM_MLA_DISABLE) keeps q_a_proj and kv_a_proj_with_mqa apart.
    from unsloth_zoo import vllm_utils
    hf, config = _tiny(monkeypatch, 3, 16)
    if not _stacked_hf(hf): pytest.skip("transformers < 5 keeps the experts per expert")
    state_dict, _ = vllm_utils._get_vllm_state_dict(
        _vllm_deepseek_from_hf(hf, fused_qkv_a = False), return_state_dict = True, config = config,
    )
    for key, value in hf.state_dict().items():
        assert torch.equal(state_dict[key].to(value.dtype), value), key


@pytest.mark.parametrize("version,q_lora_rank", CASES)
def test_mla_rebuilt_hf_model_matches_reference_logits(monkeypatch, version, q_lora_rank):
    from unsloth_zoo import vllm_utils
    hf, config = _tiny(monkeypatch, version, q_lora_rank)
    monkeypatch.setattr(vllm_utils, "get_target_device", lambda *a, **k: "cpu", raising = False)
    _, quant_state_dict = vllm_utils._get_vllm_state_dict(_vllm_deepseek_from_hf(hf), config = config)
    rebuilt = vllm_utils.convert_vllm_to_huggingface(quant_state_dict, config, dtype = torch.float32)
    ids = torch.randint(0, config.vocab_size, (1, 12), generator = torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = hf(input_ids = ids).logits
        got = rebuilt(input_ids = ids).logits
    torch.testing.assert_close(got, ref, rtol = 1e-5, atol = 1e-5)
    if not _stacked_hf(hf):
        # Per-expert Linears are views of vLLM's stacked experts, not copies.
        w13 = quant_state_dict["model.layers.1.mlp.experts.gate_up_proj"]
        gate = rebuilt.model.layers[1].mlp.experts[1].gate_proj.weight
        assert gate.untyped_storage().data_ptr() == w13.untyped_storage().data_ptr()
    if version == 3:
        # The noaux_tc router bias is the vLLM tensor, not the meta template's zeros.
        bias = rebuilt.model.layers[1].mlp.gate.e_score_correction_bias
        assert bias.data_ptr() == quant_state_dict["model.layers.1.mlp.gate.e_score_correction_bias"].data_ptr()


def test_mla_sparse_indexer_refused(monkeypatch):
    from unsloth_zoo import vllm_utils
    hf, config = _tiny(monkeypatch, 3, 16)
    with pytest.raises(NotImplementedError, match = "sparse attention indexer"):
        vllm_utils._get_vllm_state_dict(_vllm_deepseek_from_hf(hf, indexer = True), config = config)


def test_mla_fused_shared_experts_refused(monkeypatch):
    from unsloth_zoo import vllm_utils
    hf, config = _tiny(monkeypatch, 3, 16)
    with pytest.raises(NotImplementedError, match = "shared experts"):
        vllm_utils._get_vllm_state_dict(_vllm_deepseek_from_hf(hf, fused_shared_experts = True), config = config)


@pytest.mark.parametrize("version", [2, 3])
def test_mla_yarn_softmax_scale_matches_vllm(monkeypatch, version):
    # vLLM (and DeepSeek's reference) scale MLA logits by qk_head_dim ** -0.5 * mscale(mscale_all_dim) ** 2 under
    # YaRN; transformers < 5 DeepseekV2Attention drops the mscale ** 2, so the rebuild sets it.
    import math
    from unsloth_zoo import vllm_utils
    transformers = pytest.importorskip("transformers")
    cls_config = getattr(transformers, f"DeepseekV{version}Config")
    rope = dict(type = "yarn", rope_type = "yarn", factor = 40.0, mscale = 0.707, mscale_all_dim = 0.707,
                original_max_position_embeddings = 64, beta_fast = 32, beta_slow = 1)
    key = "rope_parameters" if hasattr(cls_config(), "rope_parameters") else "rope_scaling"
    config = cls_config(
        vocab_size = 64, hidden_size = 32, intermediate_size = 48, num_hidden_layers = 2, num_attention_heads = 2,
        num_key_value_heads = 2, q_lora_rank = None, kv_lora_rank = 16, qk_rope_head_dim = 8, qk_nope_head_dim = 8,
        v_head_dim = 8, first_k_dense_replace = 2, n_routed_experts = 4, max_position_embeddings = 2560,
        **{key: dict(rope, rope_theta = 10000.0) if key == "rope_parameters" else rope},
    )
    model = getattr(transformers, f"DeepseekV{version}ForCausalLM")(config)
    for layer in model.model.layers:
        layer.self_attn.scaling = 16 ** -0.5      # what transformers < 5 DeepseekV2Attention sets
    mscale = 0.1 * 0.707 * math.log(40.0) + 1.0
    assert vllm_utils._align_mla_softmax_scale(model, config) == 2
    for layer in model.model.layers:
        assert math.isclose(layer.self_attn.scaling, 16 ** -0.5 * mscale * mscale, rel_tol = 1e-9)
    assert vllm_utils._align_mla_softmax_scale(model, config) == 0


class _FakeWithLoRA(torch.nn.Module):
    def __init__(self, base_layer):
        super().__init__()
        self.base_layer = base_layer


def _fake_vllm_mla_lora(separate_wrappers):
    """vLLM's MLA layer under enable_lora: 0.11.2 gives attn.X and attn.mla_attn.X separate wrappers (the adapter
    lands in attn.X's, the forward calls mla_attn.X's); 0.34 wraps once and both names hold it."""
    root = torch.nn.Module()
    attn = torch.nn.Module()
    attn.kv_b_proj = _FakeWithLoRA(_linear(torch.zeros(4, 4)))
    attn.mla_attn = torch.nn.Module()
    for proj in ("fused_qkv_a_proj", "q_b_proj", "o_proj"):
        base = _linear(torch.zeros(4, 4))
        setattr(attn, proj, _FakeWithLoRA(base))
        setattr(attn.mla_attn, proj, _FakeWithLoRA(base) if separate_wrappers else getattr(attn, proj))
    root.self_attn = attn
    return root


@pytest.mark.parametrize("separate_wrappers", [True, False])
def test_mla_attention_lora_refused_when_vllm_forward_reads_another_wrapper(monkeypatch, separate_wrappers):
    from unsloth_zoo import vllm_utils
    monkeypatch.delenv("UNSLOTH_DISABLE_LORA_NAME_CHECK", raising = False)
    vllm_model = _fake_vllm_mla_lora(separate_wrappers)
    monkeypatch.setattr(vllm_utils, "_get_vllm_lora_model", lambda model, *a: vllm_model)
    monkeypatch.setattr(vllm_utils, "_unmatched_lora_keys", lambda model, keys: [])
    p = "base_model.model.model.layers.0.self_attn"
    keys = [f"{p}.q_a_proj.lora_A.weight", f"{p}.q_a_proj.lora_B.weight", f"{p}.o_proj.lora_A.weight"]
    assert vllm_utils._vllm_mla_unread_lora_projections(vllm_model) == \
        ({"fused_qkv_a_proj", "q_b_proj", "o_proj"} if separate_wrappers else set())
    check = vllm_utils._check_lora_is_servable
    mlp_keys = ["base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight"]
    check(object(), mlp_keys, "the training model", {"r": 8})
    if not separate_wrappers:
        check(object(), keys, "the training model", {"r": 8})
        return
    with pytest.raises(NotImplementedError, match = "fused_qkv_a_proj"):
        check(object(), keys, "the training model", {"r": 8})
    monkeypatch.setenv("UNSLOTH_DISABLE_LORA_NAME_CHECK", "1")
    check(object(), keys, "the training model", {"r": 8})


def test_vllm_lora_model_unwraps_cuda_graph_wrapper(monkeypatch):
    # vLLM 0.11 keeps runner.model as a CUDAGraphWrapper (not an nn.Module) that forwards attributes to .runnable.
    from unsloth_zoo import vllm_utils
    inner = _fake_vllm_mla_lora(separate_wrappers = True)
    class CUDAGraphWrapper:
        def __init__(self, runnable): self.runnable = runnable
        def __getattr__(self, key): return getattr(self.runnable, key)
    runner = types.SimpleNamespace(model = CUDAGraphWrapper(inner))
    assert vllm_utils._get_vllm_lora_model(None, runner) is inner
    assert vllm_utils._vllm_mla_unread_lora_projections(vllm_utils._get_vllm_lora_model(None, runner))
    plain = types.SimpleNamespace(model = inner)
    assert vllm_utils._get_vllm_lora_model(None, plain) is inner
