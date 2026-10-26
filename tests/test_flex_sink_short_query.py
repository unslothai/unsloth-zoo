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

"""Sink attention on a short query (under 128 rows) with GQA must compile without Inductor's flex_decoding kernel.

AUTO picks flex_decoding there and its BLOCK_M = next_pow2(q_len * GQA group) can exceed the 128-row sparse block,
leaving no valid config. Dynamo then skips flex_attention for the rest of the process, so a gradient-checkpointed
GPT-OSS layer recomputes through another kernel than its forward used and backward raises CheckpointError.
"""

import types

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "flex attention compiles for CUDA here")


def _attn(heads_q, heads_kv):
    from unsloth_zoo.flex_attention import HAS_FLEX_ATTENTION

    if not HAS_FLEX_ATTENTION:
        pytest.skip("no flex attention in this torch")
    self_attn = torch.nn.Module()
    self_attn.sinks = torch.nn.Parameter(torch.randn(heads_q, device = "cuda"))
    self_attn.num_key_value_groups = heads_q // heads_kv
    self_attn.scaling = 0.125
    self_attn.sliding_window = None
    self_attn.train()
    return self_attn


@pytest.mark.parametrize("q_len", [70, 100])
def test_short_query_sink_attention_compiles(q_len, monkeypatch):
    from unsloth_zoo.flex_attention import flex_attention_with_sink

    torch._dynamo.reset()
    # A lowering failure raises here instead of silently falling back for the rest of the process.
    monkeypatch.setattr(torch._dynamo.config, "suppress_errors", False)
    gen = torch.Generator(device = "cuda").manual_seed(0)
    self_attn = _attn(4, 2)
    q = torch.randn(1, 4, q_len, 64, device = "cuda", dtype = torch.bfloat16, generator = gen)
    k = torch.randn(1, 2, q_len, 64, device = "cuda", dtype = torch.bfloat16, generator = gen)
    v = torch.randn(1, 2, q_len, 64, device = "cuda", dtype = torch.bfloat16, generator = gen)
    with torch.no_grad():
        got = flex_attention_with_sink(self_attn, q, k, v)
        want = flex_attention_with_sink(self_attn, q, k, v, compile = False)
    torch.testing.assert_close(got.float(), want.float(), atol = 2e-2, rtol = 2e-2)
