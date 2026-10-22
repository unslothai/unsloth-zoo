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
# Inductor skips bf16 graphs on GPUs without native bf16 (T4), so a bf16 fullgraph compile finds no frames there.
needs_native_bf16 = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation = False),
    reason = "needs native bf16 for a bf16 torch.compile",
)


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


def _cvt_supported():
    import unsloth_zoo.mxfp4_gemm as mg
    import os
    old = os.environ.get("UNSLOTH_MXFP4_CVT")
    os.environ["UNSLOTH_MXFP4_CVT"] = "1"
    try:
        return bool(mg._cvt_ok(torch.cuda.current_device()))
    finally:
        os.environ.pop("UNSLOTH_MXFP4_CVT") if old is None else os.environ.__setitem__("UNSLOTH_MXFP4_CVT", old)


def _need_cvt(asm):
    if asm and (torch.version.hip is not None or torch.cuda.get_device_capability() < (8, 0)):
        pytest.skip("the inline-PTX decodes are CUDA sm_80+ only; _asm_ok keeps the plain decode here")
    if asm == 2 and not _cvt_supported():
        pytest.skip("hardware e2m1 convert needs sm_100+ / sm_120 and ptxas with PTX 9.2")


def test_cvt_decode_is_bit_exact_for_every_byte_and_scale():
    import unsloth_zoo.mxfp4_gemm as mg
    _need_cvt(2)
    rows = torch.arange(65536, device = "cuda")
    blocks = (rows % 256).to(torch.uint8)[:, None].expand(65536, 16).contiguous()
    scales = (rows // 256).to(torch.uint8)
    got = torch.empty(65536, 32, dtype = torch.bfloat16, device = "cuda")
    mg._decode_probe_kernel[(1024,)](blocks, scales, got, CVT = True)
    want = mxfp4_dequantize_torch(blocks.view(65536, 1, 16), scales.view(65536, 1)).view(65536, 32)
    # Every e8m0 incl. 255 (NaN to the cvt): bits of the reference, inf / 2^127 at 255, signed zeros kept.
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


def test_decode_mode_dispatch_and_off_switch(monkeypatch):
    import unsloth_zoo.mxfp4_gemm as mg
    index = torch.cuda.current_device()
    major = torch.cuda.get_device_capability(index)[0]
    for env, want in (("0", False), ("1", _cvt_supported()), ("auto", major == 12 and _cvt_supported())):
        monkeypatch.setenv("UNSLOTH_MXFP4_CVT", env)
        monkeypatch.setattr(mg, "_ASM_OK", {})
        mode = mg._asm_ok(torch.device("cuda", index))
        # sm_75 and HIP keep the plain Triton decode (False); sm_80+ CUDA never drops below the fma decode.
        fma_eligible = torch.version.hip is None and torch.cuda.get_device_capability(index) >= (8, 0)
        assert (mode == 2) == want and bool(mode) == fma_eligible
    monkeypatch.setenv("UNSLOTH_MXFP4_ASM", "0")
    monkeypatch.setattr(mg, "_ASM_OK", {})
    assert mg._asm_ok(torch.device("cuda", index)) is False


def test_first_call_inside_cuda_graph_capture_does_not_probe(monkeypatch):
    import unsloth_zoo.mxfp4_gemm as mg
    from unsloth_zoo.mxfp4_gemm import mxfp4_grouped_mm
    _need_cvt(2)
    monkeypatch.setenv("UNSLOTH_MXFP4_CVT", "1")
    E, R, C = 4, 128, 256
    blocks, scales = _stack(E, R, C, seed = 5)
    counts = torch.full((E,), 16, dtype = torch.int32, device = "cuda")
    x = torch.randn(64, C, dtype = torch.bfloat16, device = "cuda")
    monkeypatch.setattr(mg, "_ASM_OK", {k: True for k in (None, 0, torch.cuda.current_device())})
    want = mxfp4_grouped_mm(x, blocks, scales, counts, transpose_b = True)
    monkeypatch.setattr(mg, "_ASM_OK", {})
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = mxfp4_grouped_mm(x, blocks, scales, counts, transpose_b = True)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(got, want)
    assert mg._ASM_OK == {}  # decided on the next eager call, not during capture
    assert mg._asm_ok(torch.device("cuda", torch.cuda.current_device())) == 2


def test_capture_on_a_non_current_operand_device_does_not_probe(monkeypatch):
    # Capture check must ask GPU 1, else the probe syncs mid-capture and caches the fma fallback.
    import unsloth_zoo.mxfp4_gemm as mg
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs")
    _need_cvt(2)
    monkeypatch.setenv("UNSLOTH_MXFP4_CVT", "1")
    for index in (0, 1):
        # _need_cvt only asked the current GPU: a mixed host may lack the convert on GPU 0 or 1.
        if mg._cvt_ok(index, True) is not True:  # also warms the probe kernels outside the capture
            pytest.skip(f"GPU {index} has no hardware e2m1 convert")
    monkeypatch.setattr(mg, "_ASM_OK", {})
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    stream = torch.cuda.Stream(device = 1)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.device(1):
        with torch.cuda.graph(graph, stream = stream):
            torch.empty(1, device = "cuda:1")
            with torch.cuda.device(0):
                mode = mg._resolve_decode_mode(1)
    assert mode is True and 1 not in mg._ASM_OK  # deferred, not decided inside the capture
    assert mg._asm_ok(1) == 2


def test_capture_on_either_device_defers_the_probe(monkeypatch):
    # Capturing on the current GPU or on the operand's GPU: both must defer (the probe syncs).
    import contextlib
    import unsloth_zoo.mxfp4_gemm as mg
    monkeypatch.setenv("UNSLOTH_MXFP4_CVT", "1")
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: (10, 0))
    current = {"index": 0}

    @contextlib.contextmanager
    def device(index):
        previous, current["index"] = current["index"], index
        try:
            yield
        finally:
            current["index"] = previous

    monkeypatch.setattr(torch.cuda, "device", device)
    for capturing in (0, 1):
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda c = capturing: current["index"] == c)
        assert mg._cvt_ok(1) is None, capturing


def test_cold_compiled_region_bakes_in_the_probed_decode_mode(monkeypatch):
    # The op body is traced on FakeTensors and its ASM constexpr is frozen into the graph: probe before it.
    import unsloth_zoo.mxfp4_gemm as mg
    _need_cvt(2)
    if mg.mxfp4_grouped_mm_op is None:
        pytest.skip("needs torch.library.triton_op")
    monkeypatch.setenv("UNSLOTH_MXFP4_CVT", "1")
    monkeypatch.setattr(mg, "_ASM_OK", {})
    traced, real_asm_ok = [], mg._asm_ok
    monkeypatch.setattr(mg, "_asm_ok", lambda device, *a, **k: traced.append(real_asm_ok(device, *a, **k)) or traced[-1])
    E, R, C = 4, 256, 256
    blocks, scales = _stack(E, R, C, seed = 3)
    counts = torch.full((E,), 16, dtype = torch.int32, device = "cuda")
    x = torch.randn(64, C, dtype = torch.bfloat16, device = "cuda")
    torch._dynamo.reset()
    got = torch.compile(lambda t: mg.mxfp4_grouped_mm_compiled(t, blocks, scales, counts, True), fullgraph = True)(x)
    torch._dynamo.reset()
    assert traced and all(mode == 2 for mode in traced), traced
    want = _reference(x, mxfp4_dequantize_torch(blocks, scales), counts, True)
    torch.testing.assert_close(got.float(), want, rtol = 1e-2, atol = 1e-2 * want.abs().max().item())


@pytest.mark.parametrize("big_tiles", [True, False])
@pytest.mark.parametrize("asm", [2, True, False])
@pytest.mark.parametrize("E,M_per_expert", [(4, 1), (4, 12), (4, 40), (4, 80), (4, 150), (4, 300), (300, 1), (300, 3)])
def test_every_tile_config_matches(E, M_per_expert, asm, big_tiles, monkeypatch):
    import unsloth_zoo.mxfp4_gemm as mg
    from unsloth_zoo.mxfp4_gemm import mxfp4_grouped_mm
    if big_tiles and not mg._big_tiles(torch.device("cuda", torch.cuda.current_device())):
        pytest.skip("the B200 tile table only runs on sm_100 (256x256 overflows smaller shared memory)")
    _need_cvt(asm)
    monkeypatch.setattr(mg, "_ASM_OK", {k: asm for k in (None, 0, torch.cuda.current_device())})
    monkeypatch.setattr(mg, "_BIG_TILES", {k: big_tiles for k in (None, 0, torch.cuda.current_device())})
    R, C = (320, 192) if E < 256 else (64, 64)
    blocks, scales = _stack(E, R, C, seed = M_per_expert)
    counts = torch.full((E,), M_per_expert, dtype = torch.int32, device = "cuda")
    counts[0] = M_per_expert * 2
    counts[1] = 0
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
    dense = mxfp4_dequantize_torch(blocks, scales)
    xr = x.detach().float().requires_grad_(True)
    rows = torch.repeat_interleave(torch.arange(E, device = "cuda"), counts.long())
    torch.bmm(xr.unsqueeze(1), dense.float()[rows].transpose(1, 2)).squeeze(1).backward(dy.float())
    torch.testing.assert_close(x.grad.float(), xr.grad, rtol = 2e-2, atol = 2e-2 * xr.grad.abs().max().item())


@needs_native_bf16
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
    from unsloth_zoo.mxfp4_gemm import mxfp4_gemm_available, mxfp4_grouped_gemv_available, mxfp4_grouped_matmul, mxfp4_grouped_mm
    # sm_75 verifies the kernel but leaves it to decode rows (tile kernel too slow there).
    assert mxfp4_grouped_gemv_available(torch.device("cuda")) and not mxfp4_gemm_available(dtype = torch.float32)
    assert mxfp4_gemm_available(torch.device("cuda")) is (torch.cuda.get_device_capability() >= (8, 0))
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


@pytest.mark.parametrize("asm", [2, True, False])
def test_single_weight_decode_is_bit_exact_for_every_scale(asm, monkeypatch):
    """Identity input makes every output one exact product: every decoder must equal mxfp4_dequant bit for bit."""
    import unsloth_zoo.mxfp4_gemm as mg
    _need_cvt(asm)
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
    from unsloth_zoo.mxfp4_gemm import mxfp4_grouped_gemv_available, mxfp4_matmul
    assert mxfp4_grouped_gemv_available(torch.device("cuda"), dtype)
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


@pytest.mark.parametrize("asm", [2, True, False])
@pytest.mark.parametrize("E,rows", [(8, 1), (8, 4), (32, 16), (300, 12)])
def test_decode_rows_take_the_grouped_gemv(E, rows, asm, monkeypatch):
    import unsloth_zoo.mxfp4_gemm as mg
    _need_cvt(asm)
    from unsloth_zoo.mxfp4_gemm import mxfp4_grouped_mm
    monkeypatch.setattr(mg, "_ASM_OK", {k: asm for k in (None, 0, torch.cuda.current_device())})
    launched = []
    real = mg._mxfp4_grouped_gemv_kernel
    monkeypatch.setattr(mg, "_mxfp4_grouped_gemv_kernel", type("K", (), {"__getitem__": lambda self, grid: launched.append(grid) or real[grid]})())
    R, C = 200, 320
    blocks, scales = _stack(E, R, C, seed = rows)
    g = torch.Generator().manual_seed(rows)
    counts = torch.bincount(torch.randperm(E, generator = g)[:rows] if rows <= E else torch.randint(0, E, (rows,), generator = g), minlength = E)
    counts = counts.to(torch.int32).cuda()
    x = torch.randn(rows, C, dtype = torch.bfloat16, device = "cuda")
    got = mxfp4_grouped_mm(x, blocks, scales, counts, transpose_b = True).float()
    want = _reference(x, mxfp4_dequantize_torch(blocks, scales), counts, True)
    assert launched
    torch.testing.assert_close(got, want, rtol = 1e-2, atol = 1e-2 * want.abs().max().item())


@needs_native_bf16
def test_compiled_region_switches_between_gemv_and_tile_kernel_by_row_count():
    # The choice has to be a dynamo guard: made inside the op, the first traced size would fix it for every size.
    import unsloth_zoo.mxfp4_gemm as mg
    if mg.mxfp4_grouped_gemv_op is None:
        pytest.skip("needs torch.library.triton_op")
    E, R, C = 8, 128, 256
    blocks, scales = _stack(E, R, C, seed = 7)
    dense = mxfp4_dequantize_torch(blocks, scales)

    @torch.compile(fullgraph = True, dynamic = True)
    def f(x, counts):
        return mg.mxfp4_grouped_mm_compiled(x, blocks, scales, counts, True)

    for rows in (64, 3, 40, 2):
        counts = torch.bincount(torch.randint(0, E, (rows,), generator = torch.Generator().manual_seed(rows)), minlength = E)
        counts = counts.to(torch.int32).cuda()
        x = torch.randn(rows, C, dtype = torch.bfloat16, device = "cuda")
        want = _reference(x, dense, counts, True)
        with torch.profiler.profile(activities = [torch.profiler.ProfilerActivity.CUDA]) as prof:
            got = f(x, counts).float()
            torch.cuda.synchronize()
        names = {e.name for e in prof.events()}
        gemv = 2 * rows <= E
        assert any("_mxfp4_grouped_gemv_kernel" in n for n in names) == gemv
        assert any("_mxfp4_grouped_mm_kernel" in n for n in names) == (not gemv)
        torch.testing.assert_close(got, want, rtol = 1e-2, atol = 1e-2 * want.abs().max().item())
