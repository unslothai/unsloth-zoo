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

"""fast_inference vLLM -> HF extraction for Qwen3.5, on CPU.

1. vLLM's in-flight bitsandbytes loader packs in_proj_qkvz / in_proj_ba per LOADED shard
   (HF stores in_proj_qkv + in_proj_z, in_proj_b + in_proj_a), so bnb_quant_state has one
   entry per checkpoint tensor and bnb_shard_offsets index the packed rows, not output_sizes.
2. transformers 5 gives Qwen3.5's vision_config model_type "qwen3_5_vision", whose merged
   vision attn.qkv must be kept whole.
"""

import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch


def _bnb_functional():
    bnb_functional = pytest.importorskip("bitsandbytes.functional")
    try:
        bnb_functional.quantize_4bit(torch.randn(64, 32, dtype = torch.bfloat16), quant_type = "nf4")
    except Exception as e:
        pytest.skip(f"bitsandbytes 4-bit unavailable on CPU: {e}")
    return bnb_functional


def _packed_linear(shards, output_sizes):
    """Mirrors vLLM's in-flight BnB packing: each loaded shard quantized alone, packed rows concatenated."""
    bnb_functional = _bnb_functional()
    packed, states, offsets, reference = [], {}, [0], []
    for i, shard in enumerate(shards):
        q, state = bnb_functional.quantize_4bit(shard, quant_type = "nf4")
        packed.append(q)
        states[i] = state
        offsets.append(offsets[-1] + q.shape[0])
        reference.append(bnb_functional.dequantize_4bit(q, quant_state = state))
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(torch.cat(packed), requires_grad = False)
    module.weight.bnb_quant_state = states
    module.weight.bnb_shard_offsets = offsets
    module.output_sizes = list(output_sizes)
    return module, torch.cat(reference)


def _bnb_gdn(hidden = 32, num_k_heads = 2, num_v_heads = 4, head_k_dim = 4, head_v_dim = 4):
    torch.manual_seed(0)
    key_dim, value_dim = num_k_heads * head_k_dim, num_v_heads * head_v_dim
    gdn = torch.nn.Module()
    gdn.key_dim, gdn.value_dim = key_dim, value_dim
    qkv = torch.randn(2 * key_dim + value_dim, hidden, dtype = torch.bfloat16)
    z = torch.randn(value_dim, hidden, dtype = torch.bfloat16)
    gdn.in_proj_qkvz, qkvz_ref = _packed_linear([qkv, z], [key_dim, key_dim, value_dim, value_dim])
    b = torch.randn(num_v_heads, hidden, dtype = torch.bfloat16)
    a = torch.randn(num_v_heads, hidden, dtype = torch.bfloat16)
    gdn.in_proj_ba, ba_ref = _packed_linear([b, a], [num_v_heads, num_v_heads])
    gdn.conv1d = torch.nn.Module()
    gdn.conv1d.weight = torch.nn.Parameter(torch.randn(2 * key_dim + value_dim, 1, 4), requires_grad = False)
    gdn.dt_bias = torch.nn.Parameter(torch.randn(num_v_heads), requires_grad = False)
    gdn.A_log = torch.nn.Parameter(torch.randn(num_v_heads), requires_grad = False)
    gdn.norm = torch.nn.Module()
    gdn.norm.weight = torch.nn.Parameter(torch.randn(head_v_dim), requires_grad = False)
    gdn.out_proj = torch.nn.Module()
    gdn.out_proj.weight = torch.nn.Parameter(torch.randn(hidden, value_dim), requires_grad = False)
    return gdn, qkvz_ref, ba_ref


def _get_state_dict(prefix, kk, state_dict, module, slice_weights = True):
    weight = module.weight.data
    if slice_weights and getattr(module, "output_sizes", None) is not None:
        start = sum(module.output_sizes[:kk])
        weight = weight[start:start + module.output_sizes[kk]]
    state_dict[f"{prefix}.weight"] = weight


def _extract_gdn(gdn):
    from unsloth_zoo.empty_model import extract_gdn_layers
    prefix = "model.language_model.layers.0.linear_attn"
    state_dict, quant_state_dict = {}, {}
    # vLLM runs extraction under inference_mode
    with torch.inference_mode():
        extract_gdn_layers(gdn, prefix, state_dict, quant_state_dict, _get_state_dict)
    return prefix, state_dict, quant_state_dict


def test_gdn_in_proj_qkvz_per_loaded_shard_bnb_dequantizes():
    # Pre-fix: offsets from output_sizes sliced packed rows -> "Sizes of tensors must match".
    gdn, qkvz_ref, _ = _bnb_gdn()
    prefix, state_dict, quant_state_dict = _extract_gdn(gdn)
    split = 2 * gdn.key_dim + gdn.value_dim
    qkv = state_dict[f"{prefix}.in_proj_qkv.weight"]
    z = state_dict[f"{prefix}.in_proj_z.weight"]
    assert qkv.shape == (split, 32) and z.shape == (gdn.value_dim, 32)
    assert torch.equal(qkv, qkvz_ref[:split])
    assert torch.equal(z, qkvz_ref[split:])
    assert quant_state_dict[f"{prefix}.in_proj_qkv.weight"] is qkv
    assert f"{prefix}.in_proj_qkv.weight.quant_state" not in quant_state_dict


def test_gdn_in_proj_ba_per_loaded_shard_bnb_dequantizes():
    # Pre-fix: two quant states -> raw packed uint8 rows stored as in_proj_b / in_proj_a.
    gdn, _, ba_ref = _bnb_gdn()
    prefix, state_dict, _ = _extract_gdn(gdn)
    b = state_dict[f"{prefix}.in_proj_b.weight"]
    a = state_dict[f"{prefix}.in_proj_a.weight"]
    assert b.dtype == torch.bfloat16 and b.shape == (4, 32) and a.shape == (4, 32)
    assert torch.equal(b, ba_ref[:4])
    assert torch.equal(a, ba_ref[4:])


def test_gdn_bnb_dequantized_weights_are_trainable_outside_inference_mode():
    gdn, _, _ = _bnb_gdn()
    prefix, state_dict, _ = _extract_gdn(gdn)
    for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
        weight = state_dict[f"{prefix}.{name}.weight"]
        assert not weight.is_inference(), name
        x = torch.randn(2, weight.shape[1], dtype = weight.dtype, requires_grad = True)
        (x @ weight.t()).sum().backward()
        assert x.grad is not None


def test_dequantize_bnb_shards_declines_other_layouts():
    from unsloth_zoo.empty_model import _dequantize_bnb_shards
    gdn, qkvz_ref, _ = _bnb_gdn()
    raw = gdn.in_proj_qkvz.weight
    states = raw.bnb_quant_state
    assert torch.equal(_dequantize_bnb_shards(raw, raw.data, states, qkvz_ref.shape[0]), qkvz_ref)
    assert _dequantize_bnb_shards(raw, raw.data, states, qkvz_ref.shape[0] + 1) is None
    assert _dequantize_bnb_shards(raw, raw.data, None, None) is None
    assert _dequantize_bnb_shards(raw, raw.data, {0: states[0]}, None) is None
    plain = torch.nn.Parameter(torch.randn(4, 4), requires_grad = False)
    assert _dequantize_bnb_shards(plain, plain.data, {0: states[0]}, None) is None


# ---------------------------------------------------------------- Qwen3.5 vision qkv

def _qwen3_5_config(moe = False):
    transformers = pytest.importorskip("transformers")
    cls = getattr(transformers, "Qwen3_5MoeConfig" if moe else "Qwen3_5Config", None)
    if cls is None:
        pytest.skip("transformers has no Qwen3.5 config")
    return cls(
        text_config = dict(num_hidden_layers = 3, hidden_size = 32),
        vision_config = dict(depth = 2, hidden_size = 16, num_heads = 2),
    )


@pytest.mark.parametrize("moe", [False, True])
def test_qwen3_5_vision_model_type_keeps_merged_qkv(moe):
    from unsloth_zoo.empty_model import QWEN_VL_MERGED_QKV_TYPES, get_model_layer_counts, get_model_type
    config = _qwen3_5_config(moe)
    assert config.vision_config.model_type == ("qwen3_5_moe_vision" if moe else "qwen3_5_vision")
    assert get_model_type(config) in QWEN_VL_MERGED_QKV_TYPES
    assert get_model_layer_counts(config) == {"text_layers": 3, "vision_layers": 2}


def test_qwen3_5_extract_vision_layers_stores_full_qkv():
    # Pre-fix: q/k/v were written to the same attn.qkv key, so only v survived.
    from unsloth_zoo.empty_model import extract_vision_layers
    config = _qwen3_5_config()
    hidden = config.vision_config.hidden_size
    torch.manual_seed(0)
    blocks = torch.nn.ModuleList()
    for _ in range(config.vision_config.depth):
        block = torch.nn.Module()
        block.attn = torch.nn.Module()
        block.attn.qkv = torch.nn.Module()
        block.attn.qkv.weight = torch.nn.Parameter(torch.randn(3 * hidden, hidden), requires_grad = False)
        block.attn.qkv.output_sizes = [hidden, hidden, hidden]
        blocks.append(block)
    internals = types.SimpleNamespace(
        config = config, model = types.SimpleNamespace(visual = types.SimpleNamespace(blocks = blocks)),
    )
    state_dict, quant_state_dict = {}, {}
    extract_vision_layers(internals, state_dict, quant_state_dict, _get_state_dict)
    for kk, block in enumerate(blocks):
        weight = state_dict[f"model.visual.blocks.{kk}.attn.qkv.weight"]
        assert weight.shape == (3 * hidden, hidden)
        assert torch.equal(weight, block.attn.qkv.weight.data)
    assert not any(".attn.q_proj" in k or ".attn.v_proj" in k for k in state_dict)
