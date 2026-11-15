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

"""torch <= 2.14 flex_decoding read the query's strides before Inductor froze its layout (pytorch/pytorch#198543, fixed
in 2.15): a query made by a pointwise op on a permuted [B, S, H, D] view (RoPE, scaling) inside the same compiled
graph is read at the wrong offsets once 1 < q_len < 128. zoo stays clear of it: one-row decode reads no query stride
but the head / batch ones (identical under both layouts), and longer short queries take the main kernel
(PREFILL_KERNEL_OPTIONS). Both are checked here inlined into an enclosing compile, against float32 SDPA."""

import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("flex_decoding is a CUDA kernel", allow_module_level = True)
flex = pytest.importorskip("torch.nn.attention.flex_attention")
from unsloth_zoo.flex_attention import utils as ZU

if not ZU.HAS_FLEX_ATTENTION:
    pytest.skip("zoo flex attention unavailable", allow_module_level = True)


def _reference(q, k, v, mask = None):
    g = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
    return F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask = mask)


@pytest.fixture(autouse = True)
def _no_tf32():
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch._dynamo.reset()
    yield
    torch.backends.cuda.matmul.allow_tf32 = old
    torch._dynamo.reset()


def test_short_queries_inlined_take_the_main_kernel():
    B, S, Hq, Hkv, D, T = 2, 32, 8, 2, 64, 256
    g = torch.Generator(device = "cpu").manual_seed(0)
    x = torch.randn(B, S, 128, generator = g).cuda()
    w = (torch.randn(128, Hq * D, generator = g) / 11).cuda()
    k, v = (torch.randn(B, Hkv, T, D, generator = g).cuda() for _ in range(2))

    def query(x, w):
        return (x @ w).view(B, S, Hq, D).transpose(1, 2) * 0.5

    def attend(x, w, k, v):
        return ZU.flex_attention(query(x, w), k, v, enable_gqa = True, kernel_options = dict(ZU.PREFILL_KERNEL_OPTIONS))

    out = torch.compile(attend, fullgraph = True)(x, w, k, v)
    torch.testing.assert_close(out.float(), _reference(query(x, w), k, v), atol = 1e-4, rtol = 1e-4)


@pytest.mark.parametrize("batch", [1, 3])
def test_one_row_decode_inlined_with_rope_query(batch):
    from torch.nn.attention.flex_attention import create_block_mask
    from unsloth_zoo.flex_attention.decode import flex_decode_attention

    Hq, Hkv, D, T, pos = 8, 2, 128, 256, 200
    g = torch.Generator(device = "cpu").manual_seed(1)
    x = torch.randn(batch, 1, 256, generator = g).cuda()
    w = (torch.randn(256, Hq * D, generator = g) / 16).cuda()
    cos, sin = (torch.randn(batch, 1, D, generator = g).cuda() for _ in range(2))
    k, v = (torch.randn(batch, Hkv, T, D, generator = g).cuda() for _ in range(2))
    block_mask = create_block_mask(lambda b, h, q, kv: kv <= q + pos, batch, None, 1, T, device = "cuda")

    def query(x, w, cos, sin):
        q = (x @ w).view(batch, 1, Hq, D).transpose(1, 2)
        rotated = torch.cat((-q[..., D // 2:], q[..., : D // 2]), -1)
        return q * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)

    def step(x, w, cos, sin, k, v):
        return flex_decode_attention(query(x, w, cos, sin), k, v, block_mask, 1.0 / D ** 0.5)

    out = torch.compile(step, fullgraph = True)(x, w, cos, sin, k, v)
    visible = (torch.arange(T, device = "cuda") <= pos).view(1, 1, 1, T)
    torch.testing.assert_close(out.float(), _reference(query(x, w, cos, sin), k, v, visible), atol = 1e-4, rtol = 1e-4)
