# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The planner sizes a compressed-tensors pack-quantized checkpoint at its packed bytes.

transformers runs the quantiser's `validate_environment` before `preprocess_model`, and the
compressed-tensors quantiser resolves `use_fp8_kernel` in the former; the planner used to skip
straight to the latter, which raised, was swallowed, and left every Linear at full precision.
A 4-bit INT4 checkpoint was then sized at bf16 (2.9x), and unsloth/Kimi-K2.7-Code (595 GB
packed) was refused on 8 x 183 GB cards with "slack after weights: -558 GiB". Config only, no
weights, meta device."""

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
    # The packed layout really was installed: int32 `weight_packed` on the quantized Linears.
    q_proj = model.model.layers[0].self_attn.q_proj
    names = dict(q_proj.named_parameters(recurse = False))
    assert "weight_packed" in names, sorted(names)
    assert names["weight_packed"].dtype == torch.int32
    # The linears are 4 bits per weight plus a bf16 group scale; the embedding and the ignored head stay bf16, so the whole model lands well under half the bf16 size.
    linear_params = _LAYERS * (4 * _HIDDEN * _HIDDEN + 3 * _HIDDEN * _INTER)
    bf16_linears = linear_params * 2
    assert plain - packed > bf16_linears * 0.6, (plain, packed, bf16_linears)
    assert packed < plain
