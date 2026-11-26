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

"""fast_inference on Qwen3.8-Flash-Next (qwen4_exp): vLLM's Qwen4Exp module tree (vllm/models/qwen4_exp) maps back
onto the transformers names, every tensor aliasing vLLM's storage (hyper-connections, PLE, QSA indexer, GDN, MoE),
the rebuilt model computes what the transformers reference computes, and what cannot be shared is refused."""

import sys, os, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch


def _linear(weight, output_sizes = None):
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(weight.clone(), requires_grad = False)
    module.bias = None
    if output_sizes is not None:
        module.output_sizes = list(output_sizes)
    return module


def _norm(weight):
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(weight.clone(), requires_grad = False)
    return module


def _gated_residual(sd, p, config, use_combine = True, pad = 12):
    """vLLM's GatedResidual: [input_mix_weight_down | block_inject_weight | pad] in one merged linear."""
    hc = torch.nn.Module()
    hc.hc_lowrank, hc.hc_count = config.hc_lowrank, config.hc_count
    hc.hc_norm = _norm(sd[f"{p}.hc_norm.weight"])
    down = sd[f"{p}.input_mix_weight_down.weight"]
    if use_combine:
        inject = sd[f"{p}.block_inject_weight.weight"]
        garbage = torch.full((pad, down.shape[1]), float("nan"))
        hc.input_mix_weight_down_block_inject = _linear(torch.cat([down, inject, garbage]), [down.shape[0], inject.shape[0], pad])
    else:
        hc.input_mix_weight_down = _linear(down)
    hc.input_mix_weight_up = _linear(sd[f"{p}.input_mix_weight_up.weight"])
    return hc


def _vllm_qwen4_exp_from_hf(hf, extra_table_rows = 64, table_dtype = None, fuse_indexer = False, tp_table = 1):
    """Mirrors vLLM 0.34's Qwen4ExpForCausalLM at TP = 1 (enable_lora keeps index_qk_proj apart)."""
    sd = hf.state_dict()
    config = hf.config
    root = torch.nn.Module()
    model = torch.nn.Module()
    model.embed_tokens = _linear(sd["model.embed_tokens.weight"])
    layers = torch.nn.ModuleList()
    for kk in range(config.num_hidden_layers):
        p = f"model.layers.{kk}"
        layer = torch.nn.Module()
        if config.layer_types[kk] == "linear_attention":
            a = f"{p}.linear_attn"
            gdn = torch.nn.Module()
            qkv, z = sd[f"{a}.in_proj_qkv.weight"], sd[f"{a}.in_proj_z.weight"]
            key_dim = config.linear_num_key_heads * config.linear_key_head_dim
            value_dim = config.linear_num_value_heads * config.linear_value_head_dim
            gdn.in_proj_qkvz = _linear(torch.cat([qkv, z]), [key_dim, key_dim, value_dim, value_dim])
            b, a_ = sd[f"{a}.in_proj_b.weight"], sd[f"{a}.in_proj_a.weight"]
            gdn.in_proj_ba = _linear(torch.cat([b, a_]), [b.shape[0], a_.shape[0]])
            gdn.conv1d = _linear(sd[f"{a}.conv1d.weight"])
            gdn.dt_bias = torch.nn.Parameter(sd[f"{a}.dt_bias"].clone(), requires_grad = False)
            # vLLM keeps A_log in float32 (mamba_ssm_dtype)
            gdn.A_log = torch.nn.Parameter(sd[f"{a}.A_log"].float().clone(), requires_grad = False)
            gdn.norm = _norm(sd[f"{a}.norm.weight"])
            gdn.out_proj = _linear(sd[f"{a}.out_proj.weight"])
            layer.linear_attn = gdn
        else:
            a = f"{p}.self_attn"
            attn = torch.nn.Module()
            q, k, v = sd[f"{a}.q_proj.weight"], sd[f"{a}.k_proj.weight"], sd[f"{a}.v_proj.weight"]
            index_qk = sd[f"{a}.indexer.index_qk_proj.weight"]
            attn.indexer = torch.nn.Module()
            if fuse_indexer:
                attn.qkv_proj = _linear(torch.cat([q, k, v, index_qk]), [q.shape[0], k.shape[0], v.shape[0], index_qk.shape[0]])
            else:
                attn.qkv_proj = _linear(torch.cat([q, k, v]), [q.shape[0], k.shape[0], v.shape[0]])
                attn.indexer.index_qk_proj = _linear(index_qk)
            attn.indexer.q_layernorm = _norm(sd[f"{a}.indexer.q_layernorm.weight"])
            attn.indexer.k_layernorm = _norm(sd[f"{a}.indexer.k_layernorm.weight"])
            attn.o_proj = _linear(sd[f"{a}.o_proj.weight"])
            attn.q_norm = _norm(sd[f"{a}.q_norm.weight"])
            attn.k_norm = _norm(sd[f"{a}.k_norm.weight"])
            layer.self_attn = attn
        moe = torch.nn.Module()
        m = f"{p}.mlp"
        moe.gate = _linear(sd[f"{m}.gate.weight"])
        g, u = sd[f"{m}.shared_expert.gate_proj.weight"], sd[f"{m}.shared_expert.up_proj.weight"]
        moe.shared_expert = torch.nn.Module()
        moe.shared_expert.gate_up_proj = _linear(torch.cat([g, u]), [g.shape[0], u.shape[0]])
        moe.shared_expert.down_proj = _linear(sd[f"{m}.shared_expert.down_proj.weight"])
        moe.shared_expert_gate = _linear(sd[f"{m}.shared_expert_gate.weight"])
        routed = torch.nn.Module()
        routed.w13_weight = torch.nn.Parameter(sd[f"{m}.experts.gate_up_proj"].clone(), requires_grad = False)
        routed.w2_weight = torch.nn.Parameter(sd[f"{m}.experts.down_proj"].clone(), requires_grad = False)
        routed.quant_method = types.SimpleNamespace(unquantized_backend = types.SimpleNamespace(name = "TRITON"))
        moe.experts = torch.nn.Module()
        moe.experts.routed_experts = routed
        layer.mlp = moe
        layer.attn_hyper_connection = _gated_residual(sd, f"{p}.attn_hyper_connection", config)
        layer.mlp_hyper_connection = _gated_residual(sd, f"{p}.mlp_hyper_connection", config)
        if f"{p}.ple.key_proj.weight" in sd:
            e = f"{p}.ple"
            ple = torch.nn.Module()
            key, value = sd[f"{e}.key_proj.weight"], sd[f"{e}.value_proj.weight"]
            ple.kv_proj = _linear(torch.cat([key, value]), [key.shape[0], value.shape[0]])
            for name in ("norm_key", "norm_query", "norm_conv"):
                setattr(ple, name, _norm(sd[f"{e}.{name}.weight"]))
            ple.conv1d = _norm(sd[f"{e}.conv1d.weight"])
            table = sd[f"{e}.ple_embedding.ngram_embedding.weight"]
            # vLLM pads the table rows (vocab-parallel embedding)
            table = torch.cat([table, torch.zeros(extra_table_rows, table.shape[1], dtype = table.dtype)])
            ple.ple_embedding = torch.nn.Module()
            ple.ple_embedding.ngram_embedding = _norm(table.to(table_dtype or table.dtype))
            ple.ple_embedding.ngram_embedding.tp_size = tp_table
            for name in ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"):
                ple.ple_embedding.register_buffer(name, sd[f"{e}.ple_embedding.{name}"].clone())
            layer.ple = ple
        layers.append(layer)
    model.layers = layers
    model.hyper_connection_mixer = _gated_residual(sd, "model.hyper_connection_mixer", config, use_combine = False)
    root.model = model
    root.lm_head = _linear(sd["lm_head.weight"])
    root.packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"], "gate_up_proj": ["gate_proj", "up_proj"],
        "kv_proj": ["key_proj", "value_proj"], "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
        "input_mix_weight_down_block_inject": ["input_mix_weight_down", "block_inject_weight", "_input_mix_padding"],
    }
    engine = types.SimpleNamespace(model_executor = types.SimpleNamespace(
        driver_worker = types.SimpleNamespace(model_runner = types.SimpleNamespace(model = root))))
    return types.SimpleNamespace(llm_engine = engine), root


def _tiny(monkeypatch):
    transformers = pytest.importorskip("transformers")
    cls_config = getattr(transformers, "Qwen4ExpTextConfig", None)
    cls_model = getattr(transformers, "Qwen4ExpForCausalLM", None)
    if cls_config is None or cls_model is None:
        pytest.skip("transformers has no Qwen4Exp")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 0))
    config = cls_config(
        vocab_size = 96, hidden_size = 32, num_hidden_layers = 4, num_attention_heads = 2, num_key_value_heads = 1,
        head_dim = 64, layer_types = ["linear_attention"] * 3 + ["full_attention"],
        linear_num_key_heads = 2, linear_num_value_heads = 4, linear_key_head_dim = 8, linear_value_head_dim = 8,
        num_experts = 4, num_experts_per_tok = 2, moe_intermediate_size = 16, shared_expert_intermediate_size = 16,
        hc_count = 4, hc_lowrank = 8, indexer_n_heads = 2, indexer_kv_heads = 1, indexer_head_dim = 16,
        indexer_budget = 8, indexer_compress_ratio = 2, ple_embed_dim = 32, ple_layer_ids = [2],
        ngram_vocab_size_base = 97, heads_per_ngram = 8, ngram_size = 3, tie_word_embeddings = False,
        max_position_embeddings = 256, eos_token_id = 0, pad_token_id = 0,
        rope_parameters = dict(rope_type = "default", rope_theta = 10000.0, partial_rotary_factor = 0.25,
                               mrope_section = [3, 3, 2], mrope_interleaved = True),
    )
    torch.manual_seed(0)
    hf = cls_model(config).float().eval()
    with torch.no_grad():
        # Random norms / A_log / dt_bias / PLE table: an unshared or misplaced tensor changes the logits.
        for param in hf.parameters():
            param.normal_(0, 0.2)
    return hf, config


def test_qwen4_exp_state_dict_matches_hf_and_aliases_vllm(monkeypatch):
    from unsloth_zoo import vllm_utils
    hf, config = _tiny(monkeypatch)
    llm, root = _vllm_qwen4_exp_from_hf(hf)
    state_dict, quant_state_dict = vllm_utils._get_vllm_state_dict(llm, return_state_dict = True, config = config)
    expected = {k: v for k, v in hf.state_dict().items() if "rotary_emb" not in k}
    assert set(expected) - set(state_dict) == set()
    storages = {t.untyped_storage().data_ptr() for t in list(root.parameters()) + list(root.buffers())}
    for key, value in expected.items():
        got = state_dict[key]
        if key.endswith("ngram_embedding.weight"):
            got = got[: value.shape[0]]
        assert torch.equal(got.to(value.dtype), value), key
        assert quant_state_dict[key].untyped_storage().data_ptr() in storages, key
    # The pad rows of the hyper-connection GEMM are never handed out.
    fused = root.model.layers[0].attn_hyper_connection.input_mix_weight_down_block_inject.weight
    inject = quant_state_dict["model.layers.0.attn_hyper_connection.block_inject_weight.weight"]
    assert inject.data_ptr() == fused[config.hc_lowrank:].data_ptr()
    assert not torch.isnan(inject).any()


def test_qwen4_exp_fused_indexer_projection_maps(monkeypatch):
    # Without LoRA vLLM folds indexer.index_qk_proj into qkv_proj as a fourth shard.
    from unsloth_zoo import vllm_utils
    hf, config = _tiny(monkeypatch)
    llm, _ = _vllm_qwen4_exp_from_hf(hf, fuse_indexer = True)
    state_dict, _ = vllm_utils._get_vllm_state_dict(llm, return_state_dict = True, config = config)
    for name in ("q_proj", "k_proj", "v_proj", "indexer.index_qk_proj"):
        key = f"model.layers.3.self_attn.{name}.weight"
        assert torch.equal(state_dict[key], hf.state_dict()[key]), key


def test_qwen4_exp_rebuilt_model_matches_reference_and_shares(monkeypatch):
    from unsloth_zoo import vllm_utils
    hf, config = _tiny(monkeypatch)
    monkeypatch.setattr(vllm_utils, "get_target_device", lambda *a, **k: "cpu", raising = False)
    llm, root = _vllm_qwen4_exp_from_hf(hf)
    _, quant_state_dict = vllm_utils._get_vllm_state_dict(llm, config = config)
    rebuilt = vllm_utils.convert_vllm_to_huggingface(quant_state_dict, config, dtype = torch.float32)
    # Every language-model tensor is vLLM's storage: A_log stays float32, the PLE table is vLLM's (row-sliced).
    storages = {t.untyped_storage().data_ptr() for t in list(root.parameters()) + list(root.buffers())}
    for name, param in rebuilt.named_parameters():
        assert param.untyped_storage().data_ptr() in storages, name
    ids = torch.randint(1, config.vocab_size, (1, 24), generator = torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = hf(input_ids = ids).logits
        got = rebuilt(input_ids = ids).logits
    torch.testing.assert_close(got, ref, rtol = 1e-5, atol = 1e-5)


def test_qwen4_exp_rebuild_detects_a_wrong_ple_table(monkeypatch):
    # Positive control for the parity test above: a stale PLE table moves the logits.
    from unsloth_zoo import vllm_utils
    hf, config = _tiny(monkeypatch)
    monkeypatch.setattr(vllm_utils, "get_target_device", lambda *a, **k: "cpu", raising = False)
    llm, root = _vllm_qwen4_exp_from_hf(hf)
    _, quant_state_dict = vllm_utils._get_vllm_state_dict(llm, config = config)
    rebuilt = vllm_utils.convert_vllm_to_huggingface(quant_state_dict, config, dtype = torch.float32)
    with torch.no_grad():
        root.model.layers[1].ple.ple_embedding.ngram_embedding.weight.mul_(-1.0)
    ids = torch.randint(1, config.vocab_size, (1, 24), generator = torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert not torch.allclose(rebuilt(input_ids = ids).logits, hf(input_ids = ids).logits, atol = 1e-3)


@pytest.mark.parametrize("case", ["fp8_table", "tp_table", "no_mixer", "bad_hc_split"])
def test_qwen4_exp_unshareable_layouts_refused(monkeypatch, case):
    from unsloth_zoo import vllm_utils
    hf, config = _tiny(monkeypatch)
    if case == "fp8_table" and not hasattr(torch, "float8_e4m3fn"): pytest.skip("no float8")
    llm, root = _vllm_qwen4_exp_from_hf(
        hf, table_dtype = getattr(torch, "float8_e4m3fn", None) if case == "fp8_table" else None,
        tp_table = 2 if case == "tp_table" else 1,
    )
    if case == "no_mixer":
        root.model.hyper_connection_mixer = None
    if case == "bad_hc_split":
        root.model.layers[0].attn_hyper_connection.input_mix_weight_down_block_inject.output_sizes = [config.hc_lowrank + config.hc_count, 12]
    with pytest.raises(NotImplementedError, match = "fast_inference = False"):
        vllm_utils._get_vllm_state_dict(llm, config = config)


def test_qwen4_exp_unshared_parameter_refused(monkeypatch):
    from unsloth_zoo import vllm_utils
    from unsloth_zoo.vllm_qwen4_exp import install_qwen4_exp_shared_tensors
    hf, config = _tiny(monkeypatch)
    monkeypatch.setattr(vllm_utils, "get_target_device", lambda *a, **k: "cpu", raising = False)
    llm, _ = _vllm_qwen4_exp_from_hf(hf)
    _, quant_state_dict = vllm_utils._get_vllm_state_dict(llm, config = config)
    rebuilt = vllm_utils.convert_vllm_to_huggingface(quant_state_dict, config, dtype = torch.float32)
    norm = rebuilt.model.layers[0].attn_hyper_connection.hc_norm
    norm.weight = torch.nn.Parameter(norm.weight.detach().clone(), requires_grad = False)
    with pytest.raises(NotImplementedError, match = "not views of vLLM"):
        install_qwen4_exp_shared_tensors(rebuilt, quant_state_dict, config)


def test_qwen4_exp_hash_buffer_mismatch_refused(monkeypatch):
    from unsloth_zoo import vllm_utils
    hf, config = _tiny(monkeypatch)
    monkeypatch.setattr(vllm_utils, "get_target_device", lambda *a, **k: "cpu", raising = False)
    llm, root = _vllm_qwen4_exp_from_hf(hf)
    root.model.layers[1].ple.ple_embedding.layer_multipliers.add_(2)
    _, quant_state_dict = vllm_utils._get_vllm_state_dict(llm, config = config)
    with pytest.raises(NotImplementedError, match = "hash"):
        vllm_utils.convert_vllm_to_huggingface(quant_state_dict, config, dtype = torch.float32)


class _FakeMerged:
    def __init__(self, pad):
        self.shard_ids = (0, 1, 2) if pad else (0, 1)
        self.weight = torch.full((8 + 4 + pad, 16), float("nan"))


def _fake_gated_residual(pad, upstream_padding = False):
    class GatedResidual:
        def __init__(self):
            self.pad_size = pad
            self.input_mix_weight_down_block_inject = _FakeMerged(0 if upstream_padding else pad)
            if upstream_padding:
                self.input_mix_weight_down_block_inject.padding_size = pad
    return GatedResidual


def test_hc_padding_patch_drops_the_pad_shard_and_zeroes_it():
    from unsloth_zoo.vllm_qwen4_exp import _patch_gated_residual_padding
    cls = _fake_gated_residual(12)
    # Unpatched: the pad is a third shard no checkpoint fills (vLLM 0.34.0's weight tracking then fails).
    assert cls().input_mix_weight_down_block_inject.shard_ids == (0, 1, 2)
    assert _patch_gated_residual_padding(cls) is True
    assert _patch_gated_residual_padding(cls) is False  # idempotent
    linear = cls().input_mix_weight_down_block_inject
    assert linear.shard_ids == (0, 1)
    assert torch.count_nonzero(linear.weight[-12:]) == 0
    assert torch.isnan(linear.weight[:-12]).all()


def test_hc_padding_patch_leaves_fixed_vllm_alone():
    from unsloth_zoo.vllm_qwen4_exp import _patch_gated_residual_padding
    cls = _fake_gated_residual(12, upstream_padding = True)
    _patch_gated_residual_padding(cls)
    linear = cls().input_mix_weight_down_block_inject
    assert linear.shard_ids == (0, 1) and torch.isnan(linear.weight).all()


def test_qwen4_exp_templates_cover_both_text_prefixes():
    from unsloth_zoo.empty_model import get_model_layer_config
    templates = get_model_layer_config()
    for prefix in ("model.language_model.layers.{kk}.", "model.layers.{kk}."):
        assert prefix + "attn_hyper_connection.input_mix_weight_down" in templates["standard_layers"]
        assert prefix + "ple.conv1d.weight" in templates["standard_layers"]
        assert prefix + "self_attn.indexer.q_layernorm" in templates["layernorms"]


def test_vllm_without_qwen4_exp_is_refused_before_loading():
    from unsloth_zoo.vllm_qwen4_exp import check_vllm_supports_qwen4_exp
    config = types.SimpleNamespace(architectures = ["Qwen4ExpForConditionalGeneration"])
    with pytest.raises(NotImplementedError, match = "vLLM 0.34"):
        check_vllm_supports_qwen4_exp(config, supported = {"Qwen3NextForCausalLM"})
    check_vllm_supports_qwen4_exp(config, supported = {"Qwen4ExpForConditionalGeneration"})


def test_qwen4_exp_marked_as_3d_moe_lora_model():
    # transformers stacks Qwen4Exp's experts, so vLLM must serve the adapter with FusedMoE3DWithLoRA.
    from unsloth_zoo.vllm_qwen4_exp import _mark_stacked_moe_lora
    module = types.ModuleType("fake_vllm_qwen4_exp")
    module.Qwen4ExpForCausalLM = type("Qwen4ExpForCausalLM", (), {})
    module.Qwen4ExpForConditionalGeneration = type("Qwen4ExpForConditionalGeneration", (), {})
    assert not hasattr(module.Qwen4ExpForConditionalGeneration, "is_3d_moe_weight")
    assert len(_mark_stacked_moe_lora(module)) == 2
    assert module.Qwen4ExpForConditionalGeneration.is_3d_moe_weight is True
    assert _mark_stacked_moe_lora(module) == []


@pytest.mark.parametrize("vision_type", ["qwen4_exp", "qwen4_exp_vision"])
def test_qwen4_exp_vision_tower_keeps_fused_qkv(vision_type):
    # HF's Qwen4Exp vision attention keeps blocks.N.attn.qkv merged, like Qwen3-VL; splitting it into three shards
    # under one name keeps only the last shard.
    from unsloth_zoo.empty_model import QWEN_VL_MERGED_QKV_TYPES, get_model_type
    config = types.SimpleNamespace(model_type = "qwen4_exp", vision_config = types.SimpleNamespace(model_type = vision_type))
    assert get_model_type(config) in QWEN_VL_MERGED_QKV_TYPES
