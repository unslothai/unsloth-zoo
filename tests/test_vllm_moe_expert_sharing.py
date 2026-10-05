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
"""fast_inference on sparse MoE: the training model must ALIAS vLLM's expert weights.

Loading and generating prove nothing here, because generation runs inside vLLM and never
reads the HF module. These pin storage identity, the layouts that must be refused instead
of aliased, the router staying a router, and the Gemma 4 adapter rename.
"""
import sys, os, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from unsloth_zoo.empty_model import extract_moe_layers, vllm_moe_expert_weights, get_model_layer_config

E, I, H = 4, 3, 5


def _routed(backend = "TRITON", dtype = torch.bfloat16, w13_shape = (E, 2 * I, H), w2_shape = (E, H, I), **flags):
    routed = torch.nn.Module()
    routed.w13_weight = torch.nn.Parameter(torch.randn(w13_shape).to(dtype), requires_grad = False)
    routed.w2_weight = torch.nn.Parameter(torch.randn(w2_shape).to(dtype), requires_grad = False)
    qm = types.SimpleNamespace(**flags)
    if backend is not None:
        qm.unquantized_backend = types.SimpleNamespace(name = backend)
    routed.quant_method = qm
    return routed


def _linear(out_f, in_f):
    m = torch.nn.Module()
    m.weight = torch.nn.Parameter(torch.randn(out_f, in_f), requires_grad = False)
    return m


def _record_get_state_dict(calls):
    def get_state_dict(prefix, kk, state_dict, proj, slice_weights = True, slice_index = -1):
        calls.append((prefix, kk, slice_weights))
        state_dict[f"{prefix}.weight"] = proj.weight.data
    return get_state_dict


@pytest.mark.parametrize("runner", [False, True])
def test_qwen_style_block_aliases_experts_and_keeps_router_weight(runner):
    block = torch.nn.Module()
    block.gate = _linear(E, H)
    routed = _routed()
    if runner:  # vLLM >= 0.24: MoERunner holds the weights on routed_experts
        block.experts = torch.nn.Module()
        block.experts.routed_experts = routed
    else:
        block.experts = routed
    sd, qsd, calls = {}, {}, []
    extract_moe_layers(block, "model.layers.0.mlp", sd, qsd, _record_get_state_dict(calls),
                       config = types.SimpleNamespace(num_experts = E, moe_intermediate_size = I, hidden_size = H))
    assert sd["model.layers.0.mlp.experts.gate_up_proj"].data_ptr() == routed.w13_weight.data_ptr()
    assert sd["model.layers.0.mlp.experts.down_proj"].data_ptr() == routed.w2_weight.data_ptr()
    # The router is stored as its WEIGHT so the assignment loop keeps the HF router class.
    assert "model.layers.0.mlp.gate.weight" in sd and "model.layers.0.mlp.gate" not in sd
    assert sd["model.layers.0.mlp.gate.weight"].data_ptr() == block.gate.weight.data_ptr()


@pytest.mark.parametrize("wrapped_in_moe", [True, False])
def test_gemma4_style_router_and_per_expert_scale(wrapped_in_moe):
    # vLLM 0.19 - 0.30: layer.moe holds experts + per_expert_scale; 0.31: both on the layer / router.
    router = torch.nn.Module()
    router.proj = _linear(E, H)
    router.scale = torch.nn.Parameter(torch.randn(H), requires_grad = False)
    block = torch.nn.Module()
    block.experts = _routed()
    per_expert_scale = torch.nn.Parameter(torch.randn(E), requires_grad = False)
    if wrapped_in_moe:
        block.per_expert_scale = per_expert_scale
    else:
        router.per_expert_scale = per_expert_scale
    sd, qsd, calls = {}, {}, []
    extract_moe_layers(block, "model.language_model.layers.2", sd, qsd, _record_get_state_dict(calls), router = router)
    p = "model.language_model.layers.2"
    assert sd[f"{p}.router.per_expert_scale"].data_ptr() == per_expert_scale.data_ptr()
    assert sd[f"{p}.router.scale"].data_ptr() == router.scale.data_ptr()
    assert sd[f"{p}.router.proj.weight"].data_ptr() == router.proj.weight.data_ptr()
    assert sd[f"{p}.experts.gate_up_proj"].data_ptr() == block.experts.w13_weight.data_ptr()
    for name in ("experts.gate_up_proj", "experts.down_proj", "router.proj", "router.scale", "router.per_expert_scale"):
        assert f"{p.replace('.2', '.{kk}')}.{name}" in get_model_layer_config()["standard_layers"]


@pytest.mark.parametrize("backend", ["FLASHINFER_TRTLLM", "FLASHINFER_CUTLASS", "AITER", "MOONEP"])
def test_rewriting_backends_are_refused(backend):
    with pytest.raises(NotImplementedError, match = backend):
        vllm_moe_expert_weights(_routed(backend = backend), "layer 0")


@pytest.mark.parametrize("flag,name", [("rocm_aiter_moe_enabled", "AITER"), ("flashinfer_cutlass_moe_enabled", "FLASHINFER_CUTLASS")])
def test_pre_backend_field_vllm_flags_are_refused(flag, name):
    # vLLM < 0.15 has no unquantized_backend; these flags rewrite the experts while keeping 3-D.
    with pytest.raises(NotImplementedError, match = name):
        vllm_moe_expert_weights(_routed(backend = None, **{flag: True}), "layer 0")
    w13, _ = vllm_moe_expert_weights(_routed(backend = None), "layer 0")
    assert w13.dim() == 3


def test_tiled_and_padded_layouts_are_refused():
    with pytest.raises(NotImplementedError, match = "w13"):
        vllm_moe_expert_weights(_routed(w13_shape = (E, H, 2 * I, 1)), "layer 0")
    # Padded intermediate keeps the 2:1 ratio; only the config comparison catches it.
    padded = _routed(w13_shape = (E, 2 * (I + 1), H), w2_shape = (E, H, I + 1))
    vllm_moe_expert_weights(padded, "layer 0")
    with pytest.raises(NotImplementedError, match = "w13"):
        vllm_moe_expert_weights(padded, "layer 0", types.SimpleNamespace(num_experts = E, moe_intermediate_size = I, hidden_size = H))


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float8_e4m3fn])
def test_quantized_experts_are_refused(dtype):
    with pytest.raises(NotImplementedError, match = "quantized MoE"):
        vllm_moe_expert_weights(_routed(dtype = dtype), "layer 0")


def test_config_get_survives_per_layer_attributes():
    from unsloth_zoo.vllm_utils import _config_get

    class Ambiguous(Exception):
        pass

    class Config:
        per_layer_config = [types.SimpleNamespace(head_dim = 256), types.SimpleNamespace(head_dim = 512)]

        @property
        def head_dim(self):
            raise Ambiguous("varies per layer")

    assert _config_get(Config(), "head_dim") == 512
    assert _config_get(Config(), "missing", 7) == 7


def test_memory_estimate_accepts_moe_config_without_intermediate_size(monkeypatch):
    import unsloth_zoo.vllm_utils as vu
    monkeypatch.setattr(vu, "get_mem_info", lambda: (80 * 1024**3, 80 * 1024**3))
    config = types.SimpleNamespace(
        vocab_size = 1000, hidden_size = 64, max_position_embeddings = 4096, num_hidden_layers = 2,
        num_key_value_heads = 2, num_attention_heads = 4, num_experts = 8, moe_intermediate_size = 16,
        shared_expert_intermediate_size = 16, tie_word_embeddings = True,
    )
    vu.approximate_vllm_memory_usage(config, max_seq_length = 512)


def test_gemma4_lora_keys_are_renamed_onto_moe_experts(monkeypatch):
    import unsloth_zoo.vllm_utils as vu
    names = {"language_model.model.layers.0.moe.experts", "language_model.model.layers.0.self_attn.qkv_proj"}
    monkeypatch.setattr(vu, "_get_vllm_model_runner", lambda model: None)
    monkeypatch.setattr(vu, "_get_vllm_lora_model", lambda model, runner = None: types.SimpleNamespace())
    monkeypatch.setattr(vu, "_get_vllm_lora_manager", lambda model, runner = None: None)
    monkeypatch.setattr(vu, "_vllm_lora_target_names", lambda vllm_model, manager = None: (names, set()))

    def resolve(key, mapper):
        key = key.replace("base_model.model.model.language_model.", "language_model.model.")
        module = key.rsplit(".lora_", 1)[0]
        return module[:-len(".base_layer")] if module.endswith(".base_layer") else module
    monkeypatch.setattr(vu, "_resolve_lora_key_to_module", resolve)

    sd = {
        "base_model.model.model.language_model.layers.0.experts.base_layer.lora_A.weight": 1,
        "base_model.model.model.language_model.layers.0.experts.lora_B.weight": 2,
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight": 3,
    }
    out = vu._remap_moe_expert_lora_keys(object(), sd)
    assert set(out) == {
        "base_model.model.model.language_model.layers.0.moe.experts.base_layer.lora_A.weight",
        "base_model.model.model.language_model.layers.0.moe.experts.lora_B.weight",
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight",
    }
    # vLLM 0.31 names the module layers.N.experts itself: nothing to rename.
    names.clear(); names.update({"language_model.model.layers.0.experts"})
    assert vu._remap_moe_expert_lora_keys(object(), sd) == sd


def _fake_vllm_targets(monkeypatch, vu, names):
    monkeypatch.setattr(vu, "_get_vllm_model_runner", lambda model: None)
    monkeypatch.setattr(vu, "_get_vllm_lora_model", lambda model, runner = None: types.SimpleNamespace())
    monkeypatch.setattr(vu, "_get_vllm_lora_manager", lambda model, runner = None: None)
    monkeypatch.setattr(vu, "_vllm_lora_target_names", lambda vllm_model, manager = None: (names, set()))

    def resolve(key, mapper):
        key = key.replace("base_model.model.model.language_model.", "language_model.model.")
        module = key.rsplit(".lora_", 1)[0]
        return module[:-len(".base_layer")] if module.endswith(".base_layer") else module
    monkeypatch.setattr(vu, "_resolve_lora_key_to_module", resolve)


def _load_lora_from_disk(monkeypatch, tmp_path, names):
    """Call load_lora's default path branch on a real safetensors adapter, capturing the
    LoRARequest it builds and the keys the servability check saw."""
    import unsloth_zoo.vllm_utils as vu
    from safetensors.torch import save_file
    _fake_vllm_targets(monkeypatch, vu, names)
    save_file({
        "base_model.model.model.language_model.layers.0.experts.base_layer.lora_A.weight": torch.ones(2, 3),
        "base_model.model.model.language_model.layers.0.experts.lora_B.weight": torch.full((4, 2), 2.0),
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight": torch.full((2, 3), 3.0),
    }, str(tmp_path / "adapter_model.safetensors"))
    checked = {}
    monkeypatch.setattr(vu, "_check_lora_is_servable",
                        lambda model, keys, source, peft_config: checked.setdefault("keys", list(keys)))
    monkeypatch.setattr(vu, "get_peft_config", lambda save_directory: "PEFT_CONFIG")

    class FakeLoRARequest:
        def __init__(self, name, int_id, path = None, lora_tensors = None, lora_config = None):
            self.path, self.lora_tensors, self.lora_config = path, lora_tensors, lora_config
    fake = types.ModuleType("vllm.lora.request")
    fake.LoRARequest = FakeLoRARequest
    monkeypatch.setitem(sys.modules, "vllm.lora.request", fake)
    monkeypatch.setattr(vu, "LORA_REQUEST_ID", 7, raising = False)  # created on first load_lora
    return vu.load_lora(object(), str(tmp_path)), checked["keys"]


def test_a_saved_gemma4_adapter_is_remapped_on_the_default_path(monkeypatch, tmp_path):
    # load_lora defaults to load_tensors = False, which is what `model.load_lora(dir)` after
    # `model.save_lora(dir)` uses. vLLM would read the path with HF's `layers.N.experts` names.
    names = {"language_model.model.layers.0.moe.experts", "language_model.model.layers.0.self_attn.q_proj"}
    request, checked = _load_lora_from_disk(monkeypatch, tmp_path, names)
    expected = {
        "base_model.model.model.language_model.layers.0.moe.experts.base_layer.lora_A.weight",
        "base_model.model.model.language_model.layers.0.moe.experts.lora_B.weight",
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight",
    }
    assert request.path is None and request.lora_config == "PEFT_CONFIG"
    assert set(request.lora_tensors) == expected
    assert set(checked) == expected
    assert torch.equal(
        request.lora_tensors["base_model.model.model.language_model.layers.0.moe.experts.lora_B.weight"],
        torch.full((4, 2), 2.0),
    )


def test_an_adapter_needing_no_rename_still_goes_by_path(monkeypatch, tmp_path):
    names = {"language_model.model.layers.0.experts", "language_model.model.layers.0.self_attn.q_proj"}
    request, checked = _load_lora_from_disk(monkeypatch, tmp_path, names)
    assert request.path == str(tmp_path) and request.lora_tensors is None
    assert all(".moe.experts" not in k for k in checked)


def test_gemma4_config_proxy_forwards_writes():
    from unsloth_zoo.temporary_patches.gemma4 import _Gemma4KVSharedSafeProxy
    real = types.SimpleNamespace(tie_word_embeddings = True, num_kv_shared_layers = 0)
    proxy = _Gemma4KVSharedSafeProxy(real)
    proxy.tie_word_embeddings = False
    assert real.tie_word_embeddings is False
    assert not hasattr(proxy, "num_kv_shared_layers")


def test_qwen3_5_moe_vision_qkv_stays_merged():
    # HF Qwen3.5 MoE keeps model.visual.blocks.N.attn.qkv fused (3 * hidden rows); splitting it
    # left only the v rows in the training model's vision tower.
    from unsloth_zoo.empty_model import QWEN_VL_MERGED_QKV_TYPES
    assert "qwen3_5_moe" in QWEN_VL_MERGED_QKV_TYPES


def test_the_untiled_moe_patch_is_unconditional():
    # vllm_moe_expert_weights refuses the tiled TRT-LLM layout, so an opt-out would only abort fast_inference.
    import ast, inspect
    import unsloth_zoo.vllm_utils as vu
    tree = ast.parse(inspect.getsource(vu.patch_vllm))
    function = tree.body[0]
    calls = [
        node for node in function.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "id", None) == "patch_vllm_untiled_moe_experts"
    ]
    assert len(calls) == 1
    assert "UNSLOTH_VLLM_TILED_MOE" not in inspect.getsource(vu.patch_vllm)
