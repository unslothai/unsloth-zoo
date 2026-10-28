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

"""load_vllm sizing: explicit max_num_seqs / max_num_batched_tokens reach vLLM instead of the memory tier; the KV
estimate uses the real per-layer cache (head_dim, sliding window, MLA, linear attention, KV sharing), not hidden_size
/ heads."""

import os
import sys
from types import SimpleNamespace
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from unsloth_zoo import vllm_utils

GiB = 1024 ** 3


def _config(**kwargs):
    base = dict(
        vocab_size = 151936, hidden_size = 2560, max_position_embeddings = 40960,
        num_hidden_layers = 36, num_attention_heads = 32, num_key_value_heads = 8,
        intermediate_size = 9728, tie_word_embeddings = True, model_type = "qwen3",
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def _estimate(monkeypatch, config, **kwargs):
    monkeypatch.setattr(vllm_utils, "get_mem_info", lambda: (80 * GiB, 80 * GiB))
    return vllm_utils.approximate_vllm_memory_usage(
        config, max_seq_length = 2048, weight_bytes = 8 * GiB, **kwargs,
    )


def _tokens(kv_gb, bytes_per_token):
    return int(0.95 * kv_gb * GiB / bytes_per_token) // 256 * 256


def test_kv_estimate_uses_config_head_dim(monkeypatch):
    tokens, _, _, kv_gb = _estimate(monkeypatch, _config(head_dim = 128))
    assert tokens == _tokens(kv_gb, 36 * 2 * 8 * 128 * 2)


def test_kv_estimate_separate_value_head_dim():
    # MiMo-V2-Flash style: K is 192 wide, V 128; vLLM caches (head_size + head_size_v) per KV head.
    config = _config(head_dim = 192, v_head_dim = 128, num_key_value_heads = 4)
    assert vllm_utils._kv_cache_elements_per_token(config) == (36 * (192 + 128) * 4, [])


def test_kv_estimate_without_head_dim_is_unchanged(monkeypatch):
    tokens, _, _, kv_gb = _estimate(monkeypatch, _config())
    assert tokens == _tokens(kv_gb, 36 * 2 * 8 * 80 * 2)


def test_kv_estimate_float8(monkeypatch):
    tokens, _, _, kv_gb = _estimate(monkeypatch, _config(head_dim = 128), float8_kv_cache = True)
    assert tokens == _tokens(kv_gb, 36 * 2 * 8 * 128 * 1.25)


def test_mla_caches_one_latent(monkeypatch):
    config = _config(hidden_size = 2048, num_hidden_layers = 27, num_attention_heads = 16,
                     num_key_value_heads = 16, kv_lora_rank = 512, qk_rope_head_dim = 64)
    tokens, _, _, kv_gb = _estimate(monkeypatch, config)
    assert tokens == _tokens(kv_gb, 27 * 576 * 2)


def test_mla_rope_head_dim_is_not_the_query_width(monkeypatch):
    # DeepSeek-V2 head_dim is the rope width only; it must not shrink projection / LoRA shapes.
    def mla(**kw):
        return _config(kv_lora_rank = 512, qk_rope_head_dim = 64, num_attention_heads = 16,
                       num_key_value_heads = 16, hidden_size = 2048, **kw)
    monkeypatch.setattr(vllm_utils, "get_mem_info", lambda: (16 * GiB, 16 * GiB))
    run = lambda c: vllm_utils.approximate_vllm_memory_usage(
        c, max_seq_length = 32768, weight_bytes = 8 * GiB, gpu_memory_utilization = 0.9,
    )
    assert run(mla(head_dim = 64)) == run(mla())


def test_head_dim_leaves_the_vllm_memory_share_alone(monkeypatch):
    # The activation reserve sets vLLM's gpu_memory_utilization; Qwen3's real head_dim there failed Qwen3-4B 53248 on L4.
    monkeypatch.setattr(vllm_utils, "get_mem_info", lambda: (22 * GiB, 22 * GiB))
    run = lambda c: vllm_utils.approximate_vllm_memory_usage(
        c, max_seq_length = 53248, weight_bytes = 8 * GiB, gpu_memory_utilization = 0.9,
    )[2]
    assert run(_config(head_dim = 128)) == run(_config())


def test_sliding_layers_hold_the_whole_prefill_budget(monkeypatch):
    # Vision prefill budget max(8192, max_seq_length): at 32768 every sliding layer spans the sequence.
    layer_types = (["sliding_attention"] * 5 + ["full_attention"]) * 5 + ["sliding_attention"] * 4
    config = _config(num_hidden_layers = 34, num_attention_heads = 8, num_key_value_heads = 4,
                     head_dim = 256, sliding_window = 1024, layer_types = layer_types)
    monkeypatch.setattr(vllm_utils, "get_mem_info", lambda: (16 * GiB, 16 * GiB))
    tokens, _, _, kv_gb = vllm_utils.approximate_vllm_memory_usage(
        config, max_seq_length = 32768, weight_bytes = 8 * GiB, max_num_batched_tokens = 32768,
    )
    assert tokens < 1024 + 2 * 32768
    assert tokens == _tokens(kv_gb, 34 * 2 * 4 * 256 * 2)


def test_linear_attention_layers_hold_no_per_token_cache(monkeypatch):
    layer_types = ["linear_attention"] * 3 + ["full_attention"]
    config = _config(num_hidden_layers = 32, head_dim = 256, num_key_value_heads = 4,
                     layer_types = layer_types * 8)
    tokens, _, _, kv_gb = _estimate(monkeypatch, config)
    assert tokens == _tokens(kv_gb, 8 * 2 * 4 * 256 * 2)


def test_kv_shared_layers_hold_no_cache(monkeypatch):
    config = _config(head_dim = 128, num_kv_shared_layers = 6)
    tokens, _, _, kv_gb = _estimate(monkeypatch, config)
    assert tokens == _tokens(kv_gb, 30 * 2 * 8 * 128 * 2)


def test_sliding_layers_stop_growing_at_their_window(monkeypatch):
    layer_types = (["sliding_attention"] * 5 + ["full_attention"]) * 5 + ["sliding_attention"] * 4
    config = _config(num_hidden_layers = 34, num_attention_heads = 8, num_key_value_heads = 4,
                     head_dim = 256, sliding_window = 1024, layer_types = layer_types)
    tokens, _, _, kv_gb = _estimate(monkeypatch, config)
    layer = 2 * 4 * 256 * 2
    cap = 1024 + 2 * 8192
    memory = 0.95 * kv_gb * GiB
    expected = int(cap + (memory - cap * 34 * layer) / (5 * layer)) // 256 * 256
    assert tokens == expected
    assert tokens >= _tokens(kv_gb, 34 * layer)


def test_kv_cache_tokens_short_memory_counts_every_layer():
    assert vllm_utils._kv_cache_tokens(1000 * 10, 4, [(1024, 6)], 4096) == 1000
    assert vllm_utils._kv_cache_tokens(0, 4, [(1024, 6)], 4096) == 0
    # Only sliding layers: past the cap nothing grows, so the context length is the limit.
    assert vllm_utils._kv_cache_tokens(10 ** 12, 0, [(128, 6)], 131072) == 131072


def test_per_layer_config_shapes(monkeypatch):
    class Ambiguous(AttributeError): pass
    sliding = SimpleNamespace(head_dim = 256, num_key_value_heads = 16, sliding_window = 1024)
    full    = SimpleNamespace(head_dim = 512, num_key_value_heads = 4, sliding_window = 1024)
    class Config:
        hidden_size = 5376
        num_hidden_layers = 6
        num_attention_heads = 32
        sliding_window = 1024
        layer_types = ["sliding_attention"] * 5 + ["full_attention"]
        per_layer_config = [sliding] * 5 + [full]
        @property
        def head_dim(self): raise Ambiguous("per layer")
        @property
        def num_key_value_heads(self): raise Ambiguous("per layer")
    full_elements, sliding_elements = vllm_utils._kv_cache_elements_per_token(Config())
    assert full_elements == 2 * 512 * 4
    assert sliding_elements == [(1024, 5 * 2 * 256 * 16)]


def _load_args(monkeypatch, **kwargs):
    monkeypatch.setattr(vllm_utils, "get_mem_info", lambda: (80 * GiB, 80 * GiB))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    if "vllm_version" not in vars(vllm_utils):
        # load_vllm imports and patches vLLM's modules, so these engine-arg checks need vLLM installed.
        pytest.skip(reason = "load_vllm needs vLLM installed: it imports and patches vLLM modules")
    config = _config(head_dim = 128, architectures = ["Qwen3ForCausalLM"], torch_dtype = "bfloat16")
    return vllm_utils.load_vllm(
        "unsloth/Qwen3-4B", config = config, max_seq_length = 2048, use_bitsandbytes = False,
        return_args = True, **kwargs,
    )


def test_default_max_num_seqs_keeps_memory_tier(monkeypatch):
    args = _load_args(monkeypatch)
    assert args["max_num_seqs"] == 128
    assert args["max_num_batched_tokens"] == 8192


@pytest.mark.parametrize("max_num_seqs", [4, 512])
def test_explicit_max_num_seqs_is_honoured(monkeypatch, max_num_seqs):
    args = _load_args(monkeypatch, max_num_seqs = max_num_seqs)
    assert args["max_num_seqs"] == max_num_seqs
    assert args["max_num_batched_tokens"] >= max_num_seqs


def test_explicit_max_num_batched_tokens_is_honoured(monkeypatch):
    args = _load_args(monkeypatch, max_num_seqs = 4, max_num_batched_tokens = 16384)
    assert args["max_num_seqs"] == 4
    assert args["max_num_batched_tokens"] == 16384


def test_explicit_budget_reaches_the_vision_seq_cap(monkeypatch):
    # vLLM 0.11 pads each profiled image to the scheduler budget, so the vision cap must see the requested one.
    seen = []
    real = vllm_utils.vision_max_num_seqs

    def spy(approx, max_num_seqs, config, budget, *args, **kwargs):
        seen.append(budget)
        return real(approx, max_num_seqs, config, budget, *args, **kwargs)

    monkeypatch.setattr(vllm_utils, "vision_max_num_seqs", spy)
    args = _load_args(monkeypatch, is_vision_model = True, max_num_batched_tokens = 32768)
    assert seen == [32768]
    assert args["max_num_batched_tokens"] == 32768


def test_explicit_small_budget_lowers_auto_max_num_seqs(monkeypatch):
    # vLLM refuses max_num_batched_tokens < max_num_seqs; only the auto-picked count may move.
    args = _load_args(monkeypatch, max_num_batched_tokens = 64)
    assert args["max_num_batched_tokens"] == 64
    assert args["max_num_seqs"] == 64


def test_mla_disabled_caches_full_k_and_v(monkeypatch):
    # VLLM_MLA_DISABLE=1: vLLM stores K (nope + rope) and V per head instead of the latent.
    monkeypatch.setenv("VLLM_MLA_DISABLE", "1")
    config = _config(
        num_hidden_layers = 27, num_attention_heads = 16, num_key_value_heads = 16, head_dim = 64,
        kv_lora_rank = 512, qk_rope_head_dim = 64, qk_nope_head_dim = 128, v_head_dim = 128,
    )
    assert vllm_utils._kv_cache_elements_per_token(config) == (27 * (192 + 128) * 16, [])
