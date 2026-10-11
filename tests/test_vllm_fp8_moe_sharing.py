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
"""fast_inference on block FP8 MoE: training aliases vLLM's float8 experts AND their scales.

Mock vLLM layers pin the one shareable runtime format (Triton FP8 experts, 128x128 fp32 block
scales, dynamic activations, TP = 1) and the refusal of every other FP8 runtime.
"""
import sys, os, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from unsloth_zoo.empty_model import (
    extract_moe_layers, vllm_moe_expert_descriptor, vllm_moe_expert_weights,
    fp8_moe_checkpoint_scheme, get_model_layer_config,
)

E, I, H = 4, 256, 384


class Fp8MoEMethod:  # name-matched like vLLM's class
    pass


class TritonExperts:
    pass


class DeepGemmExperts:
    pass


def _fp8_routed(
    backend = "TRITON", experts_cls = TritonExperts, block = (128, 128), moe_block = None, refine = None,
    activation = "dynamic", block_quant = True, scale_dtype = torch.float32, tp = 1, ep = 1, use_ep = False,
    inter = I, scale_shape = None, method_cls = Fp8MoEMethod,
):
    routed = torch.nn.Module()
    routed.w13_weight = torch.nn.Parameter(torch.randn(E, 2 * inter, H).to(torch.float8_e4m3fn), requires_grad = False)
    routed.w2_weight = torch.nn.Parameter(torch.randn(E, H, inter).to(torch.float8_e4m3fn), requires_grad = False)
    routed.w13_weight_scale_inv = torch.nn.Parameter(
        torch.rand(scale_shape or (E, 2 * inter // 128, H // 128)).to(scale_dtype), requires_grad = False)
    routed.w2_weight_scale_inv = torch.nn.Parameter(
        torch.rand(E, H // 128, -(-inter // 128)).to(scale_dtype), requires_grad = False)
    qm = method_cls()
    qm.block_quant = block_quant
    qm.weight_block_size = list(block) if block is not None else None
    qm.moe_block_shape = list(moe_block or block) if block is not None else None
    qm.weight_scale_refine = refine
    qm.quant_config = types.SimpleNamespace(activation_scheme = activation)
    qm.fp8_backend = types.SimpleNamespace(name = backend)
    qm.experts_cls = experts_cls
    qm.moe = types.SimpleNamespace(moe_parallel_config = types.SimpleNamespace(tp_size = tp, ep_size = ep, use_ep = use_ep))
    routed.quant_method = qm
    return routed


def _cfg(inter = I):
    return types.SimpleNamespace(num_experts = E, moe_intermediate_size = inter, hidden_size = H)


def test_triton_block_fp8_is_shared_weights_and_scales():
    routed = _fp8_routed()
    found = vllm_moe_expert_descriptor(routed, "layer 0", _cfg())
    assert found["scheme"] == "fp8_block" and found["block_size"] == [128, 128]
    assert found["w13"] is routed.w13_weight and found["w2"] is routed.w2_weight
    assert found["w13_scale_inv"] is routed.w13_weight_scale_inv
    assert found["w2_scale_inv"] is routed.w2_weight_scale_inv
    # The 16-bit API keeps refusing float8.
    with pytest.raises(NotImplementedError, match = "quantized MoE"):
        vllm_moe_expert_weights(routed, "layer 0")


def test_extract_aliases_scales_beside_weights():
    block = torch.nn.Module()
    block.experts = torch.nn.Module()
    block.experts.base_layer = torch.nn.Module()  # FusedMoEWithLoRA
    block.experts.base_layer.routed_experts = routed = _fp8_routed()
    sd, qsd = {}, {}
    extract_moe_layers(block, "model.layers.0.mlp", sd, qsd, None, config = _cfg())
    p = "model.layers.0.mlp.experts"
    for hf, v in (("gate_up_proj", routed.w13_weight), ("down_proj", routed.w2_weight),
                  ("gate_up_proj_scale_inv", routed.w13_weight_scale_inv), ("down_proj_scale_inv", routed.w2_weight_scale_inv)):
        t = sd[f"{p}.{hf}"]
        assert t.untyped_storage().data_ptr() == v.untyped_storage().data_ptr()
        assert (t.storage_offset(), t.stride(), t.shape, t.dtype) == (v.storage_offset(), v.stride(), v.shape, v.dtype)
        assert qsd[f"{p}.{hf}"] is t
        assert f"model.layers.{{kk}}.mlp.experts.{hf}" in get_model_layer_config()["standard_layers"]


def test_16bit_descriptor_unchanged():
    routed = torch.nn.Module()
    routed.w13_weight = torch.nn.Parameter(torch.randn(E, 2 * I, H, dtype = torch.bfloat16), requires_grad = False)
    routed.w2_weight = torch.nn.Parameter(torch.randn(E, H, I, dtype = torch.bfloat16), requires_grad = False)
    routed.quant_method = types.SimpleNamespace(unquantized_backend = types.SimpleNamespace(name = "TRITON"))
    found = vllm_moe_expert_descriptor(routed, "layer 0", _cfg())
    assert found["scheme"] == "16bit" and found["w13_scale_inv"] is None
    sd, qsd = {}, {}
    block = torch.nn.Module(); block.experts = routed
    extract_moe_layers(block, "model.layers.0.mlp", sd, qsd, None, config = _cfg())
    assert sorted(sd) == ["model.layers.0.mlp.experts.down_proj", "model.layers.0.mlp.experts.gate_up_proj"]


@pytest.mark.parametrize("kwargs,match", [
    (dict(backend = "FLASHINFER_CUTLASS"), "FLASHINFER_CUTLASS"),
    (dict(backend = "DEEPGEMM", experts_cls = DeepGemmExperts), "DEEPGEMM"),
    (dict(backend = "MARLIN"), "MARLIN"),
    (dict(backend = "FLASHINFER_TRTLLM"), "FLASHINFER_TRTLLM"),
    (dict(backend = "BATCHED_TRITON"), "BATCHED_TRITON"),
    (dict(block_quant = False, block = None), "per-tensor"),
    (dict(block = (64, 64)), "weight_block_size"),
    (dict(moe_block = (64, 128), refine = (2, 1)), "refined"),
    (dict(activation = "static"), "activation_scheme"),
    (dict(scale_dtype = torch.float8_e8m0fnu), "E8M0"),
    (dict(tp = 2), "tp_size = 2"),
    (dict(ep = 2, use_ep = True), "ep_size = 2"),
    (dict(scale_shape = (E, 1, H // 128)), "scale grid"),
    (dict(method_cls = type("CompressedTensorsW8A8Fp8MoEMethod", (), {})), "CompressedTensorsW8A8Fp8MoEMethod"),
])
def test_unshareable_fp8_runtimes_are_refused(kwargs, match):
    with pytest.raises(NotImplementedError, match = match):
        vllm_moe_expert_descriptor(_fp8_routed(**kwargs), "layer 0", _cfg())


def test_intermediate_not_block_aligned_is_refused():
    with pytest.raises(NotImplementedError, match = "not a multiple of 128"):
        vllm_moe_expert_descriptor(_fp8_routed(inter = 192), "layer 0", _cfg(192))


def test_fp8_experts_with_biases_are_refused():
    routed = _fp8_routed()
    routed.w13_bias = torch.nn.Parameter(torch.zeros(E, 2 * I), requires_grad = False)
    with pytest.raises(NotImplementedError, match = "biases"):
        vllm_moe_expert_descriptor(routed, "layer 0", _cfg())


def test_fnuz_and_int8_experts_are_refused():
    routed = _fp8_routed()
    routed.w13_weight = torch.nn.Parameter(torch.zeros(E, 2 * I, H, dtype = torch.float8_e4m3fnuz), requires_grad = False)
    with pytest.raises(NotImplementedError, match = "float8_e4m3fnuz"):
        vllm_moe_expert_descriptor(routed, "layer 0", _cfg())


def _qcfg(**quant):
    q = dict(quant_method = "fp8", weight_block_size = [128, 128], activation_scheme = "dynamic")
    q.update(quant)
    return types.SimpleNamespace(num_experts = 8, moe_intermediate_size = 768, quantization_config = q)


@pytest.mark.parametrize("quant,ok", [
    (dict(), True),
    (dict(scale_fmt = "float"), True),
    (dict(weight_block_size = None), False),
    (dict(weight_block_size = [64, 64]), False),
    (dict(activation_scheme = "static"), False),
    (dict(scale_fmt = "ue8m0"), False),
    (dict(quant_method = "fbgemm_fp8"), False),
])
def test_checkpoint_scheme_from_config(quant, ok):
    assert fp8_moe_checkpoint_scheme(_qcfg(**quant))[0] is ok


def test_checkpoint_scheme_ignores_dense_and_unquantized():
    dense = types.SimpleNamespace(quantization_config = dict(quant_method = "fp8", weight_block_size = [128, 128]))
    assert fp8_moe_checkpoint_scheme(dense) is None
    assert fp8_moe_checkpoint_scheme(types.SimpleNamespace(num_experts = 8)) is None
    bnb = types.SimpleNamespace(num_experts = 8, quantization_config = dict(quant_method = "bitsandbytes"))
    assert fp8_moe_checkpoint_scheme(bnb) is None
    ct = types.SimpleNamespace(num_local_experts = 8, quantization_config = dict(
        quant_method = "compressed-tensors", config_groups = {"g": {"weights": {"num_bits": 8, "type": "float"}}}))
    assert fp8_moe_checkpoint_scheme(ct)[0] is False
    odd = _qcfg(); odd.moe_intermediate_size = 192
    assert fp8_moe_checkpoint_scheme(odd)[0] is False


def test_fp8_experts_swap_binds_without_allocating():
    pytest.importorskip("transformers.integrations.finegrained_fp8", reason = "transformers without FP8")
    from transformers.integrations import finegrained_fp8
    if not hasattr(finegrained_fp8, "FP8Experts"):
        pytest.skip("transformers without FP8Experts")
    from transformers import Qwen3MoeConfig
    from unsloth_zoo.vllm_utils import _swap_in_fp8_experts
    cfg = Qwen3MoeConfig(hidden_size = H, moe_intermediate_size = I, num_experts = E, num_hidden_layers = 1,
                         num_attention_heads = 2, num_key_value_heads = 1, vocab_size = 32)
    from transformers import Qwen3MoeForCausalLM
    with torch.device("meta"):
        model = Qwen3MoeForCausalLM(cfg)
    routed = _fp8_routed()
    qsd = {
        "model.layers.0.mlp.experts.gate_up_proj_scale_inv": routed.w13_weight_scale_inv.data,
    }
    _swap_in_fp8_experts(model, qsd, cfg)
    experts = model.model.layers[0].mlp.experts
    assert isinstance(experts, finegrained_fp8.FP8Experts)
    assert all(p.device.type == "meta" for p in experts.parameters())
    assert not any(p.requires_grad for p in experts.parameters())
    assert experts.block_size == (128, 128)


def test_fp8_experts_swap_keeps_step3p7_swiglu_limits():
    pytest.importorskip("transformers.integrations.finegrained_fp8", reason = "transformers without FP8")
    from transformers.integrations import finegrained_fp8
    if not hasattr(finegrained_fp8, "FP8Experts"):
        pytest.skip("transformers without FP8Experts")
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    from unsloth_zoo.vllm_utils import _swap_in_fp8_experts
    cfg = Qwen3MoeConfig(hidden_size = H, moe_intermediate_size = I, num_experts = E, num_hidden_layers = 2,
                         num_attention_heads = 2, num_key_value_heads = 1, vocab_size = 32)
    with torch.device("meta"):
        model = Qwen3MoeForCausalLM(cfg)
    Step3p7Experts = type("Step3p7Experts", (torch.nn.Module,), {})
    for layer, limit in zip(model.model.layers, (7.0, float("inf"))):
        layer.mlp.experts = Step3p7Experts()
        layer.mlp.experts.limit = limit
    scale = _fp8_routed().w13_weight_scale_inv.data
    qsd = {f"model.layers.{i}.mlp.experts.gate_up_proj_scale_inv": scale for i in range(2)}
    _swap_in_fp8_experts(model, qsd, cfg)
    clamped, unclamped = (layer.mlp.experts for layer in model.model.layers)
    assert isinstance(clamped, finegrained_fp8.FP8Experts) and clamped.limit == 7.0
    assert clamped._apply_gate.__func__.__name__ == "_step3p7_apply_gate"
    assert getattr(unclamped, "_unsloth_own_apply_gate", False) is False


def test_fp8_experts_swap_builds_lfm2_moe_with_its_implicit_silu():
    pytest.importorskip("transformers.integrations.finegrained_fp8", reason = "transformers without FP8")
    from transformers.integrations import finegrained_fp8
    if not hasattr(finegrained_fp8, "FP8Experts"):
        pytest.skip("transformers without FP8Experts")
    lfm2 = pytest.importorskip("transformers.models.lfm2_moe.modeling_lfm2_moe", reason = "transformers without LFM2-MoE")
    from unsloth_zoo.vllm_utils import _swap_in_fp8_experts
    cfg = lfm2.Lfm2MoeConfig(hidden_size = H, moe_intermediate_size = I, intermediate_size = I, num_experts = E,
                             num_experts_per_tok = 2, num_hidden_layers = 1, num_dense_layers = 0,
                             num_attention_heads = 2, num_key_value_heads = 1, vocab_size = 32, layer_types = ["full_attention"])
    assert getattr(cfg, "hidden_act", None) is None and getattr(cfg, "hidden_activation", None) is None
    with torch.device("meta"):
        model = lfm2.Lfm2MoeForCausalLM(cfg)
    qsd = {"model.layers.0.feed_forward.experts.gate_up_proj_scale_inv": _fp8_routed().w13_weight_scale_inv.data}
    _swap_in_fp8_experts(model, qsd, cfg)
    experts = model.model.layers[0].feed_forward.experts
    assert isinstance(experts, finegrained_fp8.FP8Experts)
    x = torch.randn(4)
    assert torch.equal(experts.act_fn(x), torch.nn.functional.silu(x))
    assert getattr(cfg, "hidden_act", None) is None  # the model config is not modified


def test_missing_fp8_experts_is_refused_cleanly(monkeypatch):
    import unsloth_zoo.vllm_utils as vu
    monkeypatch.setattr(vu, "_fp8_experts_class", lambda: None)
    with pytest.raises(NotImplementedError, match = "FP8Experts"):
        vu._swap_in_fp8_experts(torch.nn.Module(), {"model.layers.0.mlp.experts.gate_up_proj_scale_inv": None}, None)


@pytest.mark.parametrize("skip", [
    dict(modules_to_not_convert = ["lm_head", "model.layers.0.mlp.experts"]),
    dict(modules_to_not_convert = ["mlp.experts.3.down_proj"]),
    dict(ignore = ["re:.*experts.*"]),
])
def test_unquantized_experts_are_left_to_the_16bit_path(skip):
    # FP8 attention with 16-bit experts: vLLM keeps the experts 16-bit, which are shared as before.
    assert fp8_moe_checkpoint_scheme(_qcfg(weight_block_size = None, **skip)) is None
    # A shareable block checkpoint keeps its verdict (Unsloth still pins the Triton FP8 MoE kernel).
    assert fp8_moe_checkpoint_scheme(_qcfg(**skip)) == (True, None)


def test_shared_experts_skip_is_not_routed_experts():
    cfg = _qcfg(weight_block_size = None, modules_to_not_convert = ["model.layers.0.mlp.shared_experts"])
    assert fp8_moe_checkpoint_scheme(cfg)[0] is False


def test_16bit_refusal_names_a_working_alternative():
    with pytest.raises(NotImplementedError) as err:
        vllm_moe_expert_weights(_fp8_routed(), "layer 0")
    assert "BF16" in str(err.value) and "fast_inference = False" in str(err.value)
    assert "(load_in_4bit = False)" not in str(err.value)
