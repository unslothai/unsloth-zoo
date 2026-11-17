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

"""Dense (unquantized) experts in a prequantized bnb-4bit checkpoint must load in the requested model dtype.

The expert twins carry `hf_quantizer` so transformers >= 5.21 keeps fp32 quant metadata in its checkpoint dtype
(`HfQuantizer.get_new_dtype` returns None), but that also skips the loader cast for an unquantized expert layer
(dynamic-quant skip layers, mixed checkpoints). Runs the real `convert_and_load_state_dict_in_model` on CPU."""

from __future__ import annotations

import pytest
import torch

_core_model_loading = pytest.importorskip(
    "transformers.core_model_loading",
    reason="requires transformers v5 core_model_loading",
)
pytest.importorskip("bitsandbytes", reason="requires bitsandbytes")
if not hasattr(_core_model_loading.WeightConverter, "target_patterns"):
    pytest.skip("transformers.core_model_loading is a transformers<5 compat stub", allow_module_level=True)

from transformers import BitsAndBytesConfig
from transformers.core_model_loading import (
    MergeModulelist,
    Transpose,
    WeightConverter,
    convert_and_load_state_dict_in_model,
)
from transformers.modeling_utils import LoadStateDictConfig
from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer

Qwen3MoeConfig = pytest.importorskip("transformers.models.qwen3_moe.configuration_qwen3_moe").Qwen3MoeConfig
Qwen3MoeExperts = getattr(
    pytest.importorskip("transformers.models.qwen3_moe.modeling_qwen3_moe"), "Qwen3MoeExperts", None
)
if Qwen3MoeExperts is None:
    pytest.skip("transformers has no fused Qwen3MoeExperts", allow_module_level=True)

from unsloth_zoo.temporary_patches.moe_utils_bnb4bit import (
    _bnb4bit_per_expert_conversions,
    _bnb4bit_split_fused_expert_conversions,
)

NUM_EXPERTS, HIDDEN, INTER = 2, 8, 2


class _Model(torch.nn.Module):
    base_model_prefix = ""

    def __init__(self, dtype):
        super().__init__()
        self.config = Qwen3MoeConfig(
            hidden_size = HIDDEN, moe_intermediate_size = INTER,
            num_experts = NUM_EXPERTS, num_experts_per_tok = 1,
        )
        self.config._experts_implementation = "eager"
        self.experts = Qwen3MoeExperts(self.config).to(dtype)
        with torch.no_grad():
            self.experts.gate_up_proj.zero_()
            self.experts.down_proj.fill_(0.125)


def _layout(kind, quantizer):
    """(weight_mapping, fp32 checkpoint) for an unquantized gate_up_proj in the given checkpoint layout."""
    if kind == "per_expert":
        conv = WeightConverter("experts.*.gate_proj.weight", "experts.gate_up_proj", [MergeModulelist(dim = 0)])
        twin = _bnb4bit_per_expert_conversions([conv], quantizer)[0]
        sd = {
            f"experts.{e}.gate_proj.weight": torch.full((2 * INTER, HIDDEN), 0.25 * (e + 1), dtype = torch.float32)
            for e in range(NUM_EXPERTS)
        }
        return [twin, conv], sd
    if kind == "fused":
        conv = WeightConverter("experts.gate_up_proj", "experts.gate_up_proj", [Transpose(1, 2)])
        twin = _bnb4bit_per_expert_conversions([conv], quantizer)[0]
        sd = {"experts.gate_up_proj": torch.ones(NUM_EXPERTS, HIDDEN, 2 * INTER, dtype = torch.float32)}
        return [twin, conv], sd
    twin = _bnb4bit_split_fused_expert_conversions(quantizer)[0]
    sd = {
        f"experts.{p}.weight": torch.ones(NUM_EXPERTS * INTER, HIDDEN, dtype = torch.float32)
        for p in ("gate_proj", "up_proj")
    }
    return [twin], sd


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("kind", ["per_expert", "fused", "split_fused"])
def test_dense_expert_fallback_loads_in_model_dtype(kind, dtype):
    quantizer = Bnb4BitHfQuantizer(BitsAndBytesConfig(load_in_4bit = True), pre_quantized = True)
    model = _Model(dtype)
    mapping, sd = _layout(kind, quantizer)
    config = LoadStateDictConfig(
        device_map = {"": "cpu"}, hf_quantizer = quantizer, dtype = dtype, weight_mapping = mapping,
    )
    info, _ = convert_and_load_state_dict_in_model(model, sd, config)

    assert not info.conversion_errors, info.conversion_errors
    assert "experts.gate_up_proj" not in info.missing_keys
    loaded = model.experts.gate_up_proj
    assert loaded.dtype == dtype, f"{kind}: dense expert loaded as {loaded.dtype}, model dtype is {dtype}"
    assert tuple(loaded.shape) == (NUM_EXPERTS, 2 * INTER, HIDDEN)
    # Values survive the cast (the fallbacks still replicate the model's own ops).
    if kind == "per_expert":
        assert torch.equal(loaded[1].float(), torch.full((2 * INTER, HIDDEN), 0.5))
    else:
        assert torch.equal(loaded.float(), torch.ones(NUM_EXPERTS, 2 * INTER, HIDDEN))

    if dtype is torch.bfloat16:  # CPU fp16 bmm coverage varies across torch builds; bf16 is the reported failure
        out = model.experts(
            torch.ones(NUM_EXPERTS, HIDDEN, dtype = dtype),
            torch.tensor([[0], [1]]),
            torch.ones(NUM_EXPERTS, 1, dtype = dtype),
        )
        assert out.dtype == dtype


def test_quantized_fused_expert_keeps_fp32_quant_metadata():
    """The fix must not touch the quantized path: absmax / quant_map stay fp32 under a bf16 load."""
    import bitsandbytes as bnb

    weight = torch.randn(NUM_EXPERTS, 2 * INTER, HIDDEN * 8, dtype = torch.float32)
    try:
        packed, qs = bnb.functional.quantize_4bit(weight, blocksize = 64, quant_type = "nf4")
    except Exception as e:
        pytest.skip(f"bitsandbytes cannot quantize on this host: {e}")

    from unsloth_zoo.temporary_patches.moe_utils_bnb4bit import patch_bnb4bit_quantizer_param_needs_quantization

    # As a real load leaves it: process_model_before_weight_loading swaps the fused expert for an unquantized
    # Params4bit placeholder, which the patched param_needs_quantization claims. transformers >= 5.23 asks it
    # before accepting a packed tensor whose shape differs from the model's.
    patch_bnb4bit_quantizer_param_needs_quantization()
    bnb_config = BitsAndBytesConfig(load_in_4bit = True)

    class _QModel(torch.nn.Module):
        base_model_prefix = ""

        def __init__(self):
            super().__init__()
            self.config = Qwen3MoeConfig(hidden_size = HIDDEN * 8, num_experts = NUM_EXPERTS)
            self.experts = torch.nn.Module()
            self.experts.gate_up_proj = bnb.nn.Params4bit(
                torch.zeros(NUM_EXPERTS, 2 * INTER, HIDDEN * 8), requires_grad = False,
                compress_statistics = bnb_config.bnb_4bit_use_double_quant,
                quant_type = bnb_config.bnb_4bit_quant_type, quant_storage = bnb_config.bnb_4bit_quant_storage,
            )

    quantizer = Bnb4BitHfQuantizer(bnb_config, pre_quantized = True)
    conv = WeightConverter("experts.gate_up_proj", "experts.gate_up_proj", [Transpose(1, 2)])
    twin = _bnb4bit_per_expert_conversions([conv], quantizer)[0]
    sd = {"experts.gate_up_proj": packed}
    sd.update({f"experts.gate_up_proj.{k}": v for k, v in qs.as_dict(packed = True).items()})
    model = _QModel()
    config = LoadStateDictConfig(
        device_map = {"": "cpu"}, hf_quantizer = quantizer, dtype = torch.bfloat16, weight_mapping = [twin, conv],
    )
    info, _ = convert_and_load_state_dict_in_model(model, sd, config)
    assert not info.conversion_errors, info.conversion_errors
    loaded = model.experts.gate_up_proj
    assert isinstance(loaded, bnb.nn.Params4bit)
    assert loaded.dtype == torch.uint8
    assert loaded.quant_state.absmax.dtype == torch.float32
    assert loaded.quant_state.code.dtype == torch.float32
    assert torch.equal(loaded.quant_state.absmax, qs.absmax)
