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

"""Full finetuning rollout: float32 RoPE key + half value must not crash the KV cache write."""
import pytest
import torch
import transformers.cache_utils as cache_utils

from unsloth_zoo.temporary_patches.misc import patch_kv_cache_compute_dtype


@pytest.fixture(scope = "module", autouse = True)
def patched():
    patch_kv_cache_compute_dtype()


def _kv(k_dtype, v_dtype, seq = 3):
    return torch.randn(1, 2, seq, 4, dtype = k_dtype), torch.randn(1, 2, seq, 4, dtype = v_dtype)


def test_static_layer_mixed_dtypes_allocates_value_dtype():
    layer = cache_utils.StaticLayer(max_cache_len = 8)
    k, v = _kv(torch.float32, torch.bfloat16)
    keys, values = layer.update(k, v, {"cache_position": torch.arange(3)})
    assert keys.dtype == values.dtype == torch.bfloat16
    k, v = _kv(torch.float32, torch.bfloat16, seq = 1)
    keys, values = layer.update(k, v, {"cache_position": torch.tensor([3])})
    assert keys.dtype == values.dtype == torch.bfloat16
    torch.testing.assert_close(keys[:, :, 3:4], k.to(torch.bfloat16))


def test_dynamic_layer_mixed_dtypes_stay_aligned():
    layer = cache_utils.DynamicLayer()
    keys, values = layer.update(*_kv(torch.float32, torch.float16))
    assert keys.dtype == values.dtype == torch.float16
    keys, values = layer.update(*_kv(torch.float32, torch.float32, seq = 1))
    assert keys.dtype == values.dtype == torch.float16 and keys.shape[2] == 4


def test_matching_dtypes_untouched():
    layer = cache_utils.DynamicLayer()
    k, v = _kv(torch.float32, torch.float32)
    keys, values = layer.update(k, v)
    assert keys.dtype == values.dtype == torch.float32
    torch.testing.assert_close(keys, k)


def test_patch_idempotent():
    before = cache_utils.StaticLayer.update
    patch_kv_cache_compute_dtype()
    assert cache_utils.StaticLayer.update is before


def test_float32_model_under_autocast_generates_with_static_cache():
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig(
        vocab_size = 64, hidden_size = 32, intermediate_size = 64, num_hidden_layers = 2,
        num_attention_heads = 4, num_key_value_heads = 2, max_position_embeddings = 64,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).float().eval()
    input_ids = torch.randint(0, 64, (1, 5))
    with torch.autocast("cpu", dtype = torch.bfloat16):
        out = model.generate(
            input_ids, max_new_tokens = 4, do_sample = False,
            cache_implementation = "static", pad_token_id = 0,
        )
    assert out.shape == (1, 9)
