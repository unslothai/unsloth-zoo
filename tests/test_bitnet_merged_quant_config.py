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

"""unslothai/unsloth#2390: merged_16bit must keep BitNet's online quantization_config.

microsoft/bitnet-b1.58-2B-4T-bf16 ships bf16 master weights that AutoBitLinear ternarizes at
runtime only while config.json carries `quant_method: bitnet, quantization_mode: online`.
Stripping it (as for bnb) reloads the merge as a dense model: eval loss 9.64 vs 1.70 for base.
"""
import json

import pytest

from unsloth_zoo.saving_utils import _remove_quantization_config

BITNET_ONLINE = {"quant_method": "bitnet", "linear_class": "autobitlinear", "quantization_mode": "online"}
BNB_4BIT = {"quant_method": "bitsandbytes", "load_in_4bit": True, "bnb_4bit_quant_type": "nf4"}


def _roundtrip(tmp_path, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding = "utf-8")
    _remove_quantization_config(path)
    return json.loads(path.read_text(encoding = "utf-8"))


def test_bitnet_online_quantization_config_is_kept(tmp_path):
    out = _roundtrip(tmp_path, {"model_type": "bitnet", "quantization_config": dict(BITNET_ONLINE)})
    assert out["quantization_config"] == BITNET_ONLINE


@pytest.mark.parametrize("quant", [
    BNB_4BIT,
    {"quant_method": "bitsandbytes", "load_in_8bit": True},
    {"quant_method": "fp8", "weight_block_size": [128, 128]},
    {"quant_method": "compressed-tensors", "format": "pack-quantized"},
    {"quant_method": "mxfp4"},
    {"quant_method": "torchao", "quant_type": {"default": {}}},
    # Packed BitNet weights are unpacked on merge, so the offline config must still go.
    {"quant_method": "bitnet", "linear_class": "autobitlinear", "quantization_mode": "offline"},
    {"quant_method": "bitnet", "linear_class": "bitlinear"},
])
def test_other_quantization_configs_are_still_stripped(tmp_path, quant):
    out = _roundtrip(tmp_path, {
        "model_type": "x",
        "quantization_config": dict(quant),
        "text_config": {"quantization_config": dict(quant)},
        "sub_configs": [{"quantization_config": dict(quant)}],
    })
    assert "quantization_config" not in out
    assert "quantization_config" not in out["text_config"]
    assert "quantization_config" not in out["sub_configs"][0]


def test_nested_bnb_stripped_next_to_kept_bitnet(tmp_path):
    out = _roundtrip(tmp_path, {
        "quantization_config": dict(BITNET_ONLINE),
        "vision_config": {"quantization_config": dict(BNB_4BIT)},
    })
    assert out["quantization_config"] == BITNET_ONLINE
    assert "quantization_config" not in out["vision_config"]


def test_merged_bitnet_reloads_with_autobitlinear(tmp_path):
    from transformers import AutoModelForCausalLM, BitNetConfig
    from transformers.integrations.bitnet import AutoBitLinear

    config = BitNetConfig(hidden_size = 32, intermediate_size = 64, num_hidden_layers = 1,
                          num_attention_heads = 2, num_key_value_heads = 1, vocab_size = 64)
    AutoModelForCausalLM.from_config(config).save_pretrained(tmp_path)
    path = tmp_path / "config.json"
    saved = json.loads(path.read_text(encoding = "utf-8"))
    saved["quantization_config"] = dict(BITNET_ONLINE)
    path.write_text(json.dumps(saved), encoding = "utf-8")

    _remove_quantization_config(path)
    model = AutoModelForCausalLM.from_pretrained(tmp_path)
    layers = [m for m in model.modules() if isinstance(m, AutoBitLinear)]
    assert len(layers) == 7 and all(m.online_quant for m in layers)
