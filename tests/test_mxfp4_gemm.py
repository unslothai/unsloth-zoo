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

"""Fused MXFP4 grouped GEMM: in-register decode is bit exact, products match dequantize + matmul."""

import pytest
import torch

from unsloth_zoo.mxfp4_dequant import mxfp4_dequantize_torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")


def _stack(E, R, C, seed, lo = 118, hi = 130):
    g = torch.Generator(device = "cuda").manual_seed(seed)
    blocks = torch.randint(0, 256, (E, R, C // 32, 16), dtype = torch.uint8, device = "cuda", generator = g)
    scales = torch.randint(lo, hi, (E, R, C // 32), dtype = torch.uint8, device = "cuda", generator = g)
    return blocks, scales


def _reference(x, dense, counts, transpose_b):
    rows = torch.repeat_interleave(torch.arange(dense.shape[0], device = x.device), counts.long())
    w = dense.float()[rows]
    w = w.transpose(1, 2) if transpose_b else w
    return torch.bmm(x.float().unsqueeze(1), w).squeeze(1)


def test_in_register_decode_is_bit_exact_for_every_byte_and_scale():
    from unsloth_zoo.mxfp4_gemm import _decode_for_test
    # 16 rows of all 256 bytes per scale value, every e8m0 value incl. 0 (subnormal), 254 and 255 (2^128 -> inf).
    blocks = torch.arange(256, dtype = torch.uint8, device = "cuda").reshape(16, 1, 16).repeat(256, 1, 1)
    scales = torch.arange(256, dtype = torch.uint8, device = "cuda").repeat_interleave(16).reshape(4096, 1)
    got = _decode_for_test(blocks, scales)
    want = mxfp4_dequantize_torch(blocks, scales)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


@pytest.mark.parametrize("transpose_b", [True, False])
@pytest.mark.parametrize("counts", [[0, 0, 0, 0], [1, 0, 2, 1], [5, 0, 19, 40], [300, 7, 0, 150]])
def test_grouped_mm_matches_dequantize_then_matmul(transpose_b, counts):
    from unsloth_zoo.mxfp4_gemm import mxfp4_grouped_mm
    E, R, C = 4, 192, 256
    blocks, scales = _stack(E, R, C, seed = sum(counts) + transpose_b)
    counts = torch.tensor(counts, dtype = torch.int32, device = "cuda")
    M = int(counts.sum())
    x = torch.randn(M, C if transpose_b else R, dtype = torch.bfloat16, device = "cuda")
    got = mxfp4_grouped_mm(x, blocks, scales, counts, transpose_b = transpose_b)
    assert got.shape == (M, R if transpose_b else C) and got.dtype == torch.bfloat16
    if M:
        want = _reference(x, mxfp4_dequantize_torch(blocks, scales), counts, transpose_b)
        torch.testing.assert_close(got.float(), want, rtol = 1e-2, atol = 1e-2 * want.abs().max().item())


@pytest.mark.parametrize("M_per_expert", [1, 12, 40, 150, 300])
def test_every_tile_config_matches(M_per_expert):
    # _pick_config switches tiles by rows per expert; each branch must stay correct.
    from unsloth_zoo.mxfp4_gemm import mxfp4_grouped_mm
    E, R, C = 4, 320, 192
    blocks, scales = _stack(E, R, C, seed = M_per_expert)
    counts = torch.tensor([M_per_expert * 2, 0, M_per_expert, M_per_expert], dtype = torch.int32, device = "cuda")
    dense = mxfp4_dequantize_torch(blocks, scales)
    for transpose_b in (True, False):
        x = torch.randn(int(counts.sum()), C if transpose_b else R, dtype = torch.bfloat16, device = "cuda")
        got = mxfp4_grouped_mm(x, blocks, scales, counts, transpose_b = transpose_b).float()
        want = _reference(x, dense, counts, transpose_b)
        torch.testing.assert_close(got, want, rtol = 1e-2, atol = 1e-2 * want.abs().max().item())


def test_autograd_gives_dx_of_the_dense_product():
    from unsloth_zoo.mxfp4_gemm import mxfp4_expert_grouped_mm
    E, R, C = 3, 128, 96
    blocks, scales = _stack(E, R, C, seed = 3)
    counts = torch.tensor([7, 0, 21], dtype = torch.int32, device = "cuda")
    x = torch.randn(28, C, dtype = torch.bfloat16, device = "cuda", requires_grad = True)
    y = mxfp4_expert_grouped_mm(x, blocks, scales, counts, transpose_b = True)
    dy = torch.randn_like(y)
    y.backward(dy)
    dense = mxfp4_dequantize_torch(blocks, scales)   # (E, R, C)
    xr = x.detach().float().requires_grad_(True)
    rows = torch.repeat_interleave(torch.arange(E, device = "cuda"), counts.long())
    torch.bmm(xr.unsqueeze(1), dense.float()[rows].transpose(1, 2)).squeeze(1).backward(dy.float())
    torch.testing.assert_close(x.grad.float(), xr.grad, rtol = 2e-2, atol = 2e-2 * xr.grad.abs().max().item())


def test_custom_op_compiles_without_graph_breaks():
    from unsloth_zoo.mxfp4_gemm import mxfp4_grouped_mm, mxfp4_grouped_mm_op
    if mxfp4_grouped_mm_op is None:
        pytest.skip("torch.library.triton_op unavailable")
    E, R, C = 4, 128, 64
    blocks, scales = _stack(E, R, C, seed = 9)
    counts = torch.tensor([2, 1, 0, 1], dtype = torch.int32, device = "cuda")
    x = torch.randn(4, C, dtype = torch.bfloat16, device = "cuda")
    compiled = torch.compile(lambda t: mxfp4_grouped_mm_op(t * 2, blocks, scales, counts, True), fullgraph = True)
    assert torch.equal(compiled(x), mxfp4_grouped_mm(x * 2, blocks, scales, counts))
