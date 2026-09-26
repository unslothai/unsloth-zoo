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


def test_stacked_experts_calling_convention():
    from unsloth_zoo.mxfp4_gemm import mxfp4_gemm_available, mxfp4_grouped_matmul, mxfp4_grouped_mm
    assert mxfp4_gemm_available(torch.device("cuda")) and not mxfp4_gemm_available(dtype = torch.float32)
    blocks, scales = _stack(3, 128, 64, seed = 4)
    counts = torch.tensor([3, 0, 5], dtype = torch.int64, device = "cuda")
    x = torch.randn(8, 64, dtype = torch.bfloat16, device = "cuda")
    assert torch.equal(mxfp4_grouped_matmul(x, blocks, scales, counts), mxfp4_grouped_mm(x, blocks, scales, counts.int()))
    g = torch.randn(8, 128, dtype = torch.bfloat16, device = "cuda")
    assert torch.equal(
        mxfp4_grouped_matmul(g, blocks, scales, counts, trans = True),
        mxfp4_grouped_mm(g, blocks, scales, counts.int(), transpose_b = False),
    )


def _single(N, K, seed, lo = 118, hi = 130):
    blocks, scales = _stack(1, N, K, seed, lo, hi)
    return blocks[0], scales[0]


@pytest.mark.parametrize("asm", [True, False])
def test_single_weight_decode_is_bit_exact_for_every_scale(asm, monkeypatch):
    """Identity input makes every output one exact product: both decoders must equal mxfp4_dequant bit for bit."""
    import unsloth_zoo.mxfp4_gemm as mg
    monkeypatch.setattr(mg, "_ASM_OK", {k: asm for k in (None, 0, torch.cuda.current_device())})
    # Rows scaled by >= 2^127 keep only |v| <= 0.5 so every value stays finite.
    small = torch.tensor([0, 1, 8, 9], dtype = torch.uint8, device = "cuda")
    blocks = torch.randint(0, 256, (256, 4, 16), dtype = torch.uint8, device = "cuda")
    scales = torch.arange(256, dtype = torch.uint8, device = "cuda")[:, None].repeat(1, 4)
    pick = torch.randint(0, 4, (2, 256, 4, 16), device = "cuda")
    blocks = torch.where(scales[..., None] > 126, small[pick[0]] | (small[pick[1]] << 4), blocks)
    want = mxfp4_dequantize_torch(blocks, scales, dtype = torch.bfloat16)
    fwd = mg.mxfp4_matmul(torch.eye(128, dtype = torch.bfloat16, device = "cuda"), blocks, scales)
    bwd = mg.mxfp4_matmul(torch.eye(256, dtype = torch.bfloat16, device = "cuda"), blocks, scales, trans = True)
    # Values, not bits: the fp32 accumulator turns a decoded -0 into +0.
    assert torch.equal(fwd.float().t(), want.float())
    assert torch.equal(bwd.float(), want.float())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows", [1, 8, 33, 64, 130])
@pytest.mark.parametrize("trans", [False, True])
def test_single_weight_matmul_matches_dequantize_then_matmul(rows, trans, dtype):
    from unsloth_zoo.mxfp4_gemm import mxfp4_gemm_available, mxfp4_matmul
    assert mxfp4_gemm_available(torch.device("cuda"), dtype)
    N, K = 320, 256
    blocks, scales = _single(N, K, seed = rows + trans)
    w = mxfp4_dequantize_torch(blocks, scales, dtype = torch.float32)
    x = torch.randn(rows, N if trans else K, device = "cuda").to(dtype)
    bias = torch.randn(K if trans else N, device = "cuda").to(dtype)
    want = x.float() @ (w if trans else w.t()) + bias.float()
    got = mxfp4_matmul(x, blocks.reshape(N, K // 2), scales, trans = trans, bias = bias)
    assert got.dtype == dtype and got.shape == want.shape
    torch.testing.assert_close(got.float(), want, atol = 2e-2 * want.abs().max().item(), rtol = 2e-2)
    out = torch.empty_like(got)
    same = mxfp4_matmul(x, blocks, scales, trans = trans, bias = bias, out = out)
    assert same.data_ptr() == out.data_ptr() and torch.equal(same, got)


def test_single_weight_matmul_keeps_leading_dims_and_strided_input():
    from unsloth_zoo.mxfp4_gemm import mxfp4_matmul
    blocks, scales = _single(96, 128, seed = 7)
    x = torch.randn(2, 3, 256, dtype = torch.bfloat16, device = "cuda")[..., ::2]
    got = mxfp4_matmul(x, blocks, scales)
    assert got.shape == (2, 3, 96)
    assert torch.equal(got, mxfp4_matmul(x.contiguous(), blocks, scales))
    assert mxfp4_matmul(x[:, :0], blocks, scales).shape == (2, 0, 96)


def test_fused_gemm_off_switch(monkeypatch):
    from unsloth_zoo.mxfp4_gemm import mxfp4_gemm_available
    monkeypatch.setenv("UNSLOTH_MXFP4_FUSED_GEMM", "0")
    assert not mxfp4_gemm_available(torch.device("cuda"), torch.bfloat16)
