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

"""Pack-quantized compressed-tensors checkpoints are planned at packed bytes, not bf16."""

import json
import os
import tempfile

import pytest
import torch

pytest.importorskip("compressed_tensors")
pytest.importorskip("accelerate")

from unsloth_zoo.device_map_planner import build_meta_model, _compute_module_sizes


_HIDDEN = 256
_LAYERS = 2
_VOCAB = 1024
_INTER = 512


def _write_config(path, quantized):
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": _HIDDEN,
        "intermediate_size": _INTER,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "num_hidden_layers": _LAYERS,
        "vocab_size": _VOCAB,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
    }
    if quantized:
        config["quantization_config"] = {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "quantization_status": "compressed",
            "ignore": ["lm_head"],
            "kv_cache_scheme": None,
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "input_activations": None,
                    "output_activations": None,
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "group",
                        "group_size": 32,
                        "dynamic": False,
                        "observer": "minmax",
                        "observer_kwargs": {},
                        "actorder": None,
                        "block_structure": None,
                    },
                }
            },
        }
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f)


def _total_bytes(path):
    model, hf_quantizer, _ = build_meta_model(path, dtype = torch.bfloat16)
    sizes = _compute_module_sizes(model, hf_quantizer)
    return sizes[""], model, hf_quantizer


def test_pack_quantized_int4_is_sized_at_its_packed_bytes():
    with tempfile.TemporaryDirectory() as d:
        _write_config(d, quantized = False)
        plain, _, _ = _total_bytes(d)
    with tempfile.TemporaryDirectory() as d:
        _write_config(d, quantized = True)
        packed, model, hf_quantizer = _total_bytes(d)
    assert type(hf_quantizer).__name__ == "CompressedTensorsHfQuantizer"
    q_proj = model.model.layers[0].self_attn.q_proj
    names = dict(q_proj.named_parameters(recurse = False))
    assert "weight_packed" in names, sorted(names)
    assert names["weight_packed"].dtype == torch.int32
    linear_params = _LAYERS * (4 * _HIDDEN * _HIDDEN + 3 * _HIDDEN * _INTER)
    bf16_linears = linear_params * 2
    assert plain - packed > bf16_linears * 0.6, (plain, packed, bf16_linears)
    assert packed < plain
