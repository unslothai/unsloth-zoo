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

"""Every card holding an FP8 MoE layer keeps that layer's bf16 dequant free (Mistral-Large-3 peaked 28 GiB over)."""
import pytest
import torch
import torch.nn as nn

from unsloth_zoo.device_map_planner import DeviceMapInfeasible, plan_device_map

pytestmark = pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason = "no float8 dtype")

_E, _H, _I, _LAYERS, _VOCAB = 8, 64, 128, 4, 64
_FP8_LAYER = _E * 2 * _I * _H + _E * _H * _I
_BF16_DEQUANT = 2 * _FP8_LAYER


class Experts(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(_E, 2 * _I, _H, dtype = dtype), requires_grad = False)
        self.down_proj = nn.Parameter(torch.empty(_E, _H, _I, dtype = dtype), requires_grad = False)


class Block(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.norm = nn.LayerNorm(_H, dtype = torch.bfloat16)
        self.experts = Experts(dtype)


class Tiny(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.embed_tokens = nn.Embedding(_VOCAB, _H, dtype = torch.bfloat16)
        self.layers = nn.ModuleList([Block(dtype) for _ in range(_LAYERS)])
        self.lm_head = nn.Linear(_H, _VOCAB, bias = False, dtype = torch.bfloat16)


def _meta(dtype = torch.float8_e4m3fn):
    with torch.device("meta"):
        return Tiny(dtype)


def _plan(model, budgets, **kw):
    return plan_device_map(
        model, max_memory = dict(enumerate(budgets)), headroom_bytes = 0,
        no_split_module_classes = ["Block"], reserve_load_transient = False, **kw,
    )


def _left(plan):
    return {d: plan.raw_budgets[d] - plan.weight_bytes.get(d, 0) for d in plan.raw_budgets}


def _moe_devices(plan):
    return {dev for name, dev in plan.device_map.items() if name.startswith("layers.")}


def test_every_card_with_an_fp8_moe_layer_keeps_its_bf16_dequant_free():
    # Unequal cards: the balanced reserve alone packs a layer where its dequant does not fit.
    budgets = [int(_FP8_LAYER * 1.5), _FP8_LAYER * 4, int(_FP8_LAYER * 5.25)]
    plan = _plan(_meta(), budgets)
    left = _left(plan)
    for dev in _moe_devices(plan):
        assert left[dev] >= _BF16_DEQUANT, (dev, left, plan.device_map)


def test_refuses_rather_than_plans_an_oom():
    budgets = [_FP8_LAYER * 2 + _FP8_LAYER // 2] * 2
    with pytest.raises(DeviceMapInfeasible, match = "FP8 expert dequant"):
        _plan(_meta(), budgets)


def test_explicit_reserve_is_left_to_the_caller():
    budgets = [_FP8_LAYER * 2 + _FP8_LAYER // 2] * 2
    plan = _plan(_meta(), budgets, activation_reserve_bytes = 0)
    assert plan is not None and _moe_devices(plan)


def test_bf16_experts_are_not_floored():
    budgets = [2 * _FP8_LAYER * 2 + 4096, 2 * _FP8_LAYER * 6]
    plan = _plan(_meta(torch.bfloat16), budgets)
    assert plan is not None
    assert "FP8" not in "\n".join(plan.notes)


def test_a_real_fine_grained_fp8_mixtral_is_detected(monkeypatch, tmp_path):
    # Without a GPU the quantiser would dequantize on load; skip that to see the FP8Experts a GPU host gets.
    qf = pytest.importorskip("transformers.quantizers.quantizer_finegrained_fp8")
    from transformers import MixtralConfig
    from unsloth_zoo.device_map_planner import (
        _compute_module_sizes, _moe_dequant_transient_by_unit, _split_units, build_meta_model,
        resolve_no_split_classes,
    )
    monkeypatch.setattr(qf.FineGrainedFP8HfQuantizer, "validate_environment", lambda self, *a, **k: None)
    config = MixtralConfig(
        hidden_size = 256, intermediate_size = 512, num_local_experts = 8, num_hidden_layers = 2,
        num_attention_heads = 4, num_key_value_heads = 2, vocab_size = 512, dtype = "bfloat16",
    )
    config.quantization_config = {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [128, 128]}
    config.save_pretrained(tmp_path)
    model, quantizer, _ = build_meta_model(str(tmp_path))
    # transformers 4.x builds per-expert FP8Linear modules, which dequantize one expert at a
    # time and hold no fused stack; only a 3D FP8 stack is what the planner budgets for.
    if not any(p.dim() == 3 and p.dtype == torch.float8_e4m3fn for p in model.parameters()):
        pytest.skip("this transformers does not build FP8 fused experts")
    units = _split_units(model, resolve_no_split_classes(model), _compute_module_sizes(model, quantizer))
    need = _moe_dequant_transient_by_unit(model, units)
    assert need == dict.fromkeys(("model.layers.0", "model.layers.1"), 2 * (8 * 1024 * 256 + 8 * 256 * 512))


def test_a_packing_that_breaks_the_floor_is_repacked_not_refused():
    # cuda:0 fits a layer's weights but not its dequant; the in-order walk puts one there first.
    budgets = [_FP8_LAYER * 10 // 4, _FP8_LAYER * 25 // 4]
    plan = _plan(_meta(), budgets)
    assert _moe_devices(plan) == {1}
    assert _left(plan)[1] >= _BF16_DEQUANT


@pytest.mark.parametrize("gate_up_scale, down_scale, copies", [
    ((_E, 4, 1), (_E, 1, 2), 1),   # 64 x 64 blocks: Triton block kernel, one bf16 copy per stack
    ((_E, 1, 1), (_E, 1, 1), 2),   # per-expert scale: vectorized `weight.to(bf16) * scale`
    ((_E, 4, 4), (_E, 4, 4), 3),   # non-square blocks: vectorized, scale expanded to full size too
])
def test_dequant_peak_follows_the_scale_layout(gate_up_scale, down_scale, copies):
    from unsloth_zoo.device_map_planner import _moe_dequant_transient_by_unit
    model = _meta()
    with torch.device("meta"):
        for block in model.layers:
            block.experts.gate_up_proj_scale_inv = nn.Parameter(torch.empty(gate_up_scale), requires_grad = False)
            block.experts.down_proj_scale_inv = nn.Parameter(torch.empty(down_scale), requires_grad = False)
    gate_up, down = 2 * _E * 2 * _I * _H, 2 * _E * _H * _I
    units = [(f"layers.{i}", 0) for i in range(_LAYERS)]
    expected = max(copies * gate_up, gate_up + copies * down)
    assert _moe_dequant_transient_by_unit(model, units) == dict.fromkeys((u for u, _ in units), expected)


def test_a_float32_load_sizes_a_float32_dequant():
    from unsloth_zoo.device_map_planner import _moe_dequant_transient_by_unit
    model = _meta()
    model.config = type("Config", (), {"dtype": torch.float32})()
    units = [(f"layers.{i}", 0) for i in range(_LAYERS)]
    assert _moe_dequant_transient_by_unit(model, units) == dict.fromkeys((u for u, _ in units), 2 * _BF16_DEQUANT)


def test_ungated_experts_keep_transformers_kernels_and_no_floor():
    from unsloth_zoo.device_map_planner import _moe_dequant_transient_by_unit
    model = _meta()
    with torch.device("meta"):
        for block in model.layers:
            del block.experts.gate_up_proj
            block.experts.up_proj = nn.Parameter(torch.empty(_E, _I, _H, dtype = torch.float8_e4m3fn), requires_grad = False)
    assert _moe_dequant_transient_by_unit(model, [(f"layers.{i}", 0) for i in range(_LAYERS)]) == {}


def test_a_partial_scale_block_adds_one_expert_for_the_per_expert_triton_loop():
    from unsloth_zoo.device_map_planner import _moe_dequant_transient_by_unit
    model = _meta()
    with torch.device("meta"):
        for block in model.layers:
            # down: 64 x 128 in 22 x 22 blocks, 3 x 6 of them, the last row of blocks partial (66 > 64).
            block.experts.gate_up_proj_scale_inv = nn.Parameter(torch.empty(_E, 4, 1), requires_grad = False)
            block.experts.down_proj_scale_inv = nn.Parameter(torch.empty(_E, 3, 6), requires_grad = False)
    gate_up, down = 2 * _E * 2 * _I * _H, 2 * _E * _H * _I
    assert set(_moe_dequant_transient_by_unit(model, [(f"layers.{i}", 0) for i in range(_LAYERS)]).values()) == {
        gate_up + down + down // _E
    }


def test_an_fp8_layer_pinned_with_the_head_keeps_its_dequant_free():
    # An MTP-style last block owning the head: the whole block is pinned to the head's card.
    class HeadBlock(Block):
        def __init__(self, dtype):
            super().__init__(dtype)
            self.lm_head = nn.Linear(_H, _VOCAB, bias = False, dtype = torch.bfloat16)

    class WithHeadBlock(nn.Module):
        def __init__(self, dtype):
            super().__init__()
            self.embed_tokens = nn.Embedding(_VOCAB, _H, dtype = torch.bfloat16)
            self.layers = nn.ModuleList([Block(dtype) for _ in range(_LAYERS - 1)] + [HeadBlock(dtype)])

        def get_output_embeddings(self):
            return self.layers[-1].lm_head

    with torch.device("meta"):
        model = WithHeadBlock(torch.float8_e4m3fn)
    # cuda:3, tried first as the head's card, fits the pinned block's weights but not its dequant.
    budgets = [_FP8_LAYER * 16 // 5] * 3 + [_FP8_LAYER * 3 // 2]
    try:
        plan = plan_device_map(
            model, max_memory = dict(enumerate(budgets)), headroom_bytes = 0,
            no_split_module_classes = ["Block", "HeadBlock"], reserve_load_transient = False,
        )
    except DeviceMapInfeasible:
        return
    head_card = plan.device_map["layers.3"]
    assert _left(plan)[head_card] >= _BF16_DEQUANT, (plan.device_map, _left(plan))
