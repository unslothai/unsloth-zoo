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

"""FP8 fused experts are dequantized whole to bf16 on every forward (moe_utils_fp8.forward_moe_backend_fp8),
so every card holding such a layer must keep that much free beyond its weights. Mistral-Large-3 on 8 B200s
was planned with a 0.000 GiB reserve on a card holding FP8 MoE layers and peaked 28 GiB over its budget."""
import pytest
import torch
import torch.nn as nn

from unsloth_zoo.device_map_planner import DeviceMapInfeasible, plan_device_map

pytestmark = pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason = "no float8 dtype")

_E, _H, _I, _LAYERS, _VOCAB = 8, 64, 128, 4, 64
_FP8_LAYER = _E * 2 * _I * _H + _E * _H * _I          # bytes of one layer's fp8 expert stacks
_BF16_DEQUANT = 2 * _FP8_LAYER                         # both stacks dequantized to bf16 together


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
    # Unequal cards: the balanced reserve packs a layer onto a card with less than its dequant free,
    # although placing layers only where the dequant fits exists (1 + 2 on cuda:1, 3 + 2 on cuda:2 ... ).
    budgets = [int(_FP8_LAYER * 1.5), _FP8_LAYER * 4, int(_FP8_LAYER * 5.25)]
    plan = _plan(_meta(), budgets)
    left = _left(plan)
    for dev in _moe_devices(plan):
        assert left[dev] >= _BF16_DEQUANT, (dev, left, plan.device_map)


def test_refuses_rather_than_plans_an_oom():
    # Room for the weights, never for weights plus one layer's dequant on any card.
    budgets = [_FP8_LAYER * 2 + _FP8_LAYER // 2] * 2
    with pytest.raises(DeviceMapInfeasible, match = "FP8 expert dequant"):
        _plan(_meta(), budgets)


def test_explicit_reserve_is_left_to_the_caller():
    budgets = [_FP8_LAYER * 2 + _FP8_LAYER // 2] * 2
    plan = _plan(_meta(), budgets, activation_reserve_bytes = 0)
    assert plan is not None and _moe_devices(plan)


def test_bf16_experts_are_not_floored():
    # bf16 experts are used in place, nothing is dequantized: same plan as before this floor existed.
    budgets = [2 * _FP8_LAYER * 2 + 4096, 2 * _FP8_LAYER * 6]
    plan = _plan(_meta(torch.bfloat16), budgets)
    assert plan is not None
    assert "FP8" not in "\n".join(plan.notes)


def test_a_real_fine_grained_fp8_mixtral_is_detected(monkeypatch, tmp_path):
    # What the planner sees for a pre-quantized FP8 MoE repo on a GPU host: FineGrainedFP8 swaps the
    # experts to FP8Experts with float8 stacks on meta. Without a GPU the quantiser falls back to
    # dequantizing, so skip that fallback here to size the GPU load.
    qf = pytest.importorskip("transformers.quantizers.quantizer_finegrained_fp8")
    from transformers import MixtralConfig
    from unsloth_zoo.device_map_planner import (
        _compute_module_sizes, _moe_dequant_transient_by_unit, _split_units, build_meta_model,
        resolve_no_split_classes,
    )
    monkeypatch.setattr(qf.FineGrainedFP8HfQuantizer, "validate_environment", lambda self, *a, **k: None)
    config = MixtralConfig(
        hidden_size = 256, intermediate_size = 512, num_local_experts = 8, num_hidden_layers = 2,
        num_attention_heads = 4, num_key_value_heads = 2, vocab_size = 512,
    )
    config.quantization_config = {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [128, 128]}
    config.save_pretrained(tmp_path)
    model, quantizer, _ = build_meta_model(str(tmp_path))
    if not any(p.dtype == torch.float8_e4m3fn for p in model.parameters()):
        pytest.skip("this transformers does not build FP8 fused experts")
    units = _split_units(model, resolve_no_split_classes(model), _compute_module_sizes(model, quantizer))
    need = _moe_dequant_transient_by_unit(model, units)
    assert need == dict.fromkeys(("model.layers.0", "model.layers.1"), 2 * (8 * 1024 * 256 + 8 * 256 * 512))
