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

"""One-token decode on fused gpt-oss experts quantized to bnb 4-bit (packed 2D Params4bit)
must take the module forward, not the dense bmm kernel."""

import os

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

transformers = pytest.importorskip("transformers")
from transformers.models.gpt_oss import modeling_gpt_oss

from unsloth_zoo.temporary_patches import gpt_oss as zoo_gpt_oss


def _tiny_mlp():
    config = modeling_gpt_oss.GptOssConfig(
        hidden_size = 8,
        intermediate_size = 16,
        num_local_experts = 4,
        num_experts_per_tok = 2,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 4,
        vocab_size = 32,
    )
    torch.manual_seed(0)
    mlp = modeling_gpt_oss.GptOssMLP(config).eval()
    with torch.no_grad():
        for p in mlp.parameters():
            p.normal_(0.0, 0.1)
    return mlp


class _PackedParam(torch.nn.Parameter):
    pass


def _pack(param):
    # The layout bitsandbytes leaves on a fused expert: uint8 [numel // 2, 1] plus quant_state.
    packed = _PackedParam(
        torch.zeros(param.numel() // 2, 1, dtype = torch.uint8), requires_grad = False
    )
    packed.quant_state = object()
    return packed


def test_packed_bnb4bit_experts_decode_through_module_forward():
    mlp = _tiny_mlp()
    x = torch.randn(1, 1, 8)
    with torch.no_grad():
        expected = mlp(x)
    expected = expected[0] if isinstance(expected, tuple) else expected

    calls = []

    def fake_forward(hidden_states):
        calls.append(hidden_states.shape)
        return expected, None

    mlp.experts.gate_up_proj = _pack(mlp.experts.gate_up_proj)
    mlp.experts.down_proj = _pack(mlp.experts.down_proj)
    mlp.forward = fake_forward

    with torch.no_grad():
        out = zoo_gpt_oss.moe_forward_inference_bf16(mlp, x)
    assert calls == [x.shape]
    torch.testing.assert_close(out, expected)
    assert zoo_gpt_oss._has_packed_bnb4bit_experts(mlp.experts)


def test_dense_experts_keep_the_fused_kernel():
    mlp = _tiny_mlp()
    assert not zoo_gpt_oss._has_packed_bnb4bit_experts(mlp.experts)
