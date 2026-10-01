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

"""load_vllm sizes vLLM's KV cache from a weight estimate. The dense per-layer
formula ignores MoE experts, so it put Qwen3-30B-A3B at 5 GiB instead of 57 GiB."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from transformers import Gemma3Config, Qwen3MoeConfig, AutoModelForCausalLM

from unsloth_zoo import vllm_utils


def _tiny_moe(**kwargs):
    return Qwen3MoeConfig(
        vocab_size = 512, hidden_size = 64, intermediate_size = 128,
        moe_intermediate_size = 32, num_experts = 16, num_experts_per_tok = 2,
        num_hidden_layers = 2, num_attention_heads = 4, num_key_value_heads = 2,
        head_dim = 16, **kwargs,
    )


def _real_bytes(config):
    model = AutoModelForCausalLM.from_config(config)
    return sum(p.numel() for p in model.parameters()) * 2


@pytest.mark.parametrize("tie", [True, False])
def test_counts_every_parameter_once(tie):
    config = _tiny_moe(tie_word_embeddings = tie)
    assert vllm_utils.vllm_weights_memory_usage(config) == _real_bytes(config)


def test_moe_experts_change_the_kv_estimate(monkeypatch):
    config = _tiny_moe()
    monkeypatch.setattr(vllm_utils, "get_mem_info", lambda: (2**20, 2**20))
    kwargs = dict(max_seq_length = 64, gpu_memory_utilization = 1.0,
                  enable_lora = False, account_for_gradients = False,
                  cuda_graph_overhead = False)
    dense = vllm_utils.approximate_vllm_memory_usage(config, **kwargs)
    counted = vllm_utils.approximate_vllm_memory_usage(
        config, weight_bytes = vllm_utils.vllm_weights_memory_usage(config), **kwargs,
    )
    assert counted[3] < dense[3]
    free = 2**20 - _real_bytes(config)
    assert counted[3] == pytest.approx(free / 2**30)


def test_skip_modules_match_checkpoint_key_spelling():
    config = Gemma3Config(
        text_config = dict(
            vocab_size = 512, hidden_size = 64, intermediate_size = 128,
            num_hidden_layers = 2, num_attention_heads = 4, num_key_value_heads = 2,
            head_dim = 16,
        ),
        vision_config = dict(
            hidden_size = 32, intermediate_size = 64, num_hidden_layers = 1,
            num_attention_heads = 2, image_size = 28, patch_size = 14,
        ),
        mm_tokens_per_image = 4,
    )
    quantized = vllm_utils.vllm_weights_memory_usage(config, load_in_4bit = True)
    # bnb checkpoints list "language_model.model.layers.0.mlp"; the module is
    # "model.language_model.layers.0.mlp".
    config.quantization_config = {"llm_int8_skip_modules": ["language_model.model.layers.0.mlp"]}
    skipped = vllm_utils.vllm_weights_memory_usage(config, load_in_4bit = True)
    mlp_elements = 3 * 64 * 128
    assert skipped - quantized == pytest.approx(mlp_elements * (2 - 2 / (16/5)))


def test_unbuildable_config_falls_back(monkeypatch):
    config = _tiny_moe()
    import transformers
    def boom(*args, **kwargs): raise ValueError("unknown architecture")
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_config", boom)
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_config", boom)
    assert vllm_utils.vllm_weights_memory_usage(config) is None
