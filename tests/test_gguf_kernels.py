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

"""GGUF kernels: torch reference bitwise vs gguf-py, Triton dequant bitwise vs the reference, fused GEMV / GEMM /
grouped GEMM vs an fp64 oracle, compile with no graph breaks, operand checks, kill switches."""

import os

import pytest
import torch

from unsloth_zoo.gguf_kernels import (
    GGML_BLOCK,
    GGML_TYPE_NAMES,
    GGMLType as T,
    dequantize_gguf,
    dequantize_reference,
    gguf_grouped_mm,
    gguf_linear,
    gguf_linear_backward_input,
)
from unsloth_zoo.gguf_kernels import ops as gguf_ops
from unsloth_zoo.gguf_kernels.testing import random_packed

ALL_TYPES = sorted(GGML_BLOCK)
QUANT_TYPES = [t for t in ALL_TYPES if t not in (T.F32, T.F16, T.BF16)]
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")


def _half():
    # what Unsloth computes in: bf16 on sm80+, fp16 on T4 (is_bf16_supported() is True there through emulation)
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def _id(t):
    return GGML_TYPE_NAMES[t]


def _row_length(t, blocks = 3):
    return GGML_BLOCK[t][0] * blocks if GGML_BLOCK[t][0] > 1 else 96


def _bits_equal(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    same = (a.view(torch.int32) == b.view(torch.int32)) | (torch.isnan(a) & torch.isnan(b))
    return int((~same).sum())


# ------------------------------------------------------------------------------------------------- CPU oracle

@pytest.mark.parametrize("ggml_type", ALL_TYPES, ids = _id)
def test_reference_matches_gguf_py(ggml_type):
    gguf = pytest.importorskip("gguf")
    import numpy as np
    from gguf.quants import dequantize
    try:
        qtype = gguf.GGMLQuantizationType(ggml_type)
    except ValueError:
        pytest.skip("installed gguf does not know this type")
    for rows, blocks, seed in ((7, 1, 0), (33, 4, 1)):
        K = _row_length(ggml_type, blocks)
        packed = random_packed(ggml_type, rows, K, seed = seed)
        try:
            want = dequantize(packed.numpy(), qtype).reshape(rows, K).astype(np.float32)
        except NotImplementedError:
            pytest.skip("installed gguf cannot dequantize this type")
        got = dequantize_reference(packed, ggml_type, (rows, K))
        assert _bits_equal(got, torch.from_numpy(want)) == 0


def test_cpu_linear_uses_reference():
    packed = random_packed(T.Q4_K, 5, 512)
    x = torch.randn(3, 512)
    w = dequantize_reference(packed, T.Q4_K, (5, 512))
    assert torch.equal(gguf_linear(x, packed, T.Q4_K, (5, 512)), x @ w.t())
    assert torch.equal(gguf_linear_backward_input(torch.ones(3, 5), packed, "Q4_K", (5, 512)), torch.ones(3, 5) @ w)


def test_shape_and_type_checks():
    packed = random_packed(T.Q8_0, 4, 64)
    with pytest.raises(ValueError):
        dequantize_gguf(packed, T.Q8_0, (4, 96))
    with pytest.raises(ValueError):
        dequantize_gguf(packed[:, :10], T.Q8_0, (4, 64))
    with pytest.raises(TypeError):
        dequantize_gguf(packed.to(torch.int8), T.Q8_0, (4, 64))
    with pytest.raises(NotImplementedError):
        dequantize_gguf(packed, 15, (4, 64))  # Q8_K is an activation format, never stored
    with pytest.raises(ValueError):
        gguf_linear(torch.randn(2, 32), packed, T.Q8_0, (4, 64))


# ------------------------------------------------------------------------------------------------- CUDA

def _fp64_bound_ok(got, x, w32, stock):
    """fused output vs fp64: mean error <= 1.5x the dequant + cuBLAS path's, max <= its max + 1 output ulp."""
    ref = (x.double() @ w32.double().t())
    e_got = (got.double() - ref).abs()
    e_stock = (stock.double() - ref).abs()
    ulp = torch.finfo(got.dtype).eps * ref.abs().max().item()
    return (e_got.mean() <= 1.5 * e_stock.mean() + 1e-12) and (e_got.max() <= e_stock.max() + ulp), (e_got, e_stock)


@cuda
@pytest.mark.parametrize("ggml_type", ALL_TYPES, ids = _id)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16], ids = str)
def test_triton_dequant_bitwise(ggml_type, dtype):
    for rows, blocks, seed in ((1, 1, 3), (7, 3, 4), (65, 8, 5)):
        K = _row_length(ggml_type, blocks)
        packed = random_packed(ggml_type, rows, K, seed = seed)
        want = dequantize_reference(packed, ggml_type, (rows, K), dtype)
        got = dequantize_gguf(packed.cuda(), ggml_type, (rows, K), dtype)
        assert got.dtype == dtype and got.shape == (rows, K)
        assert _bits_equal(got, want) == 0


@cuda
@pytest.mark.parametrize("ggml_type", QUANT_TYPES, ids = _id)
@pytest.mark.parametrize("M", [1, 5, 16, 40])
def test_fused_linear_vs_fp64(ggml_type, M):
    dtype = _half()
    N, K = 72, _row_length(ggml_type, 4 if GGML_BLOCK[ggml_type][0] < 256 else 2)
    packed = random_packed(ggml_type, N, K, seed = M, special = False).cuda()
    w32 = dequantize_reference(packed, ggml_type, (N, K))
    g = torch.Generator(device = "cuda").manual_seed(M)
    x = torch.randn(M, K, device = "cuda", dtype = dtype, generator = g)
    bias = torch.randn(N, device = "cuda", dtype = dtype, generator = g)
    got = gguf_linear(x, packed, ggml_type, (N, K))
    stock = torch.nn.functional.linear(x, w32.to(dtype))
    ok, _ = _fp64_bound_ok(got, x, w32, stock)
    assert ok
    # negative control: a 2^-7 relative error must fail the same bound
    bad_ok, _ = _fp64_bound_ok(got * (1 + 2 ** -7), x, w32, stock)
    assert not bad_ok
    # bias: fp64 bound too (two cuBLAS calls differ by more than an absolute tolerance on A100: bf16 split-K partials)
    got_b = gguf_linear(x, packed, ggml_type, (N, K), bias)
    ref_b = x.double() @ w32.double().t() + bias.double()
    e_got = (got_b.double() - ref_b).abs()
    e_stock = (torch.nn.functional.linear(x, w32.to(dtype), bias).double() - ref_b).abs()
    assert e_got.mean() <= 1.5 * e_stock.mean() + 1e-12
    assert e_got.max() <= e_stock.max() + torch.finfo(dtype).eps * ref_b.abs().max().item()
    gy = torch.randn(M, N, device = "cuda", dtype = dtype, generator = g)
    dx = gguf_linear_backward_input(gy, packed, ggml_type, (N, K))
    ok_b, _ = _fp64_bound_ok(dx, gy, w32.t().contiguous(), gy @ w32.to(dtype))
    assert ok_b


@cuda
def test_linear_autograd_and_3d_input():
    dtype = _half()
    packed = random_packed(T.Q4_K, 48, 512, special = False).cuda()
    w = dequantize_reference(packed, T.Q4_K, (48, 512)).cuda()
    x = torch.randn(2, 3, 512, device = "cuda", dtype = dtype, requires_grad = True)
    y = gguf_linear(x, packed, T.Q4_K, (48, 512))
    assert y.shape == (2, 3, 48)
    gy = torch.randn_like(y)
    (dx,) = torch.autograd.grad(y, x, gy)
    gy2 = gy.reshape(-1, 48)
    ok, _ = _fp64_bound_ok(dx.reshape(-1, 512), gy2, w.t().contiguous(), gy2 @ w.to(dtype))
    assert ok


@cuda
@pytest.mark.parametrize("ggml_type", [T.Q4_K, T.Q6_K, T.Q8_0, T.IQ2_XXS, T.IQ3_S, T.MXFP4, T.NVFP4, T.Q2_0, T.IQ4_NL], ids = _id)
def test_grouped_vs_fp64(ggml_type):
    dtype = _half()
    E, N = 5, 40
    K = _row_length(ggml_type, 2 if GGML_BLOCK[ggml_type][0] == 256 else 4)
    packed = random_packed(ggml_type, E * N, K, seed = 7, special = False).reshape(E, N, -1).cuda()
    w32 = dequantize_reference(packed, ggml_type, (E, N, K))
    counts = torch.tensor([3, 0, 17, 1, 9], dtype = torch.int32, device = "cuda")
    M = int(counts.sum())
    x = torch.randn(M, K, device = "cuda", dtype = dtype, requires_grad = True)
    y = gguf_grouped_mm(x, packed, ggml_type, (E, N, K), counts)
    rows = torch.repeat_interleave(torch.arange(E, device = "cuda"), counts.long())
    ref = torch.bmm(x.double().unsqueeze(1), w32.double()[rows].transpose(1, 2)).squeeze(1)
    stock = torch.bmm(x.unsqueeze(1), w32.to(dtype)[rows].transpose(1, 2)).squeeze(1)
    assert (y.double() - ref).abs().mean() <= 1.5 * (stock.double() - ref).abs().mean() + 1e-12
    gy = torch.randn_like(y)
    (dx,) = torch.autograd.grad(y, x, gy)
    ref_dx = torch.bmm(gy.double().unsqueeze(1), w32.double()[rows]).squeeze(1)
    stock_dx = torch.bmm(gy.unsqueeze(1), w32.to(dtype)[rows]).squeeze(1)
    assert (dx.double() - ref_dx).abs().mean() <= 1.5 * (stock_dx.double() - ref_dx).abs().mean() + 1e-12


@cuda
def test_compile_fullgraph_no_breaks():
    dtype = _half()
    packed = random_packed(T.Q5_K, 64, 512, special = False).cuda()
    stack = random_packed(T.IQ4_XS, 3 * 32, 256, special = False).reshape(3, 32, -1).cuda()
    counts = torch.tensor([2, 0, 4], dtype = torch.int32, device = "cuda")

    def fn(x, xs):
        y = gguf_linear(x, packed, T.Q5_K, (64, 512)) * 2
        z = gguf_grouped_mm(xs, stack, T.IQ4_XS, (3, 32, 256), counts)
        w = dequantize_gguf(packed, T.Q5_K, (64, 512), x.dtype)
        return y, z, w

    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph = True)
    # small inputs: float16 outputs of random K-quant weights (|w| up to ~60) must not overflow
    x = (torch.randn(4, 512, device = "cuda") * 1e-2).to(dtype).requires_grad_()
    xs = (torch.randn(6, 256, device = "cuda") * 1e-2).to(dtype).requires_grad_()
    got = compiled(x, xs)
    want = fn(x, xs)
    for name, g, w in zip(("linear", "grouped", "dequant"), got, want):
        assert torch.isfinite(w).all(), f"eager {name} not finite"
        assert torch.isfinite(g).all(), f"compiled {name} not finite"
        assert torch.equal(g, w), f"compiled {name} differs from eager"
    (got[0].float().sum() + got[1].float().sum()).backward()
    assert x.grad is not None and xs.grad is not None and torch.isfinite(x.grad).all() and torch.isfinite(xs.grad).all()


@cuda
@pytest.mark.skipif(not gguf_ops._TRACED, reason = "traced dequantize is off (torch < 2.14 or UNSLOTH_GGUF_TRACED=0)")
@pytest.mark.parametrize("ggml_type", [T.Q4_K, T.Q6_K, T.Q8_0, T.IQ4_XS], ids = _id)
def test_compiled_graph_launches_the_dequant_kernel(ggml_type):
    # Under torch.compile (symbolic row count) dequantize / linear / dX launch the Triton dequantize kernel from the
    # graph (no custom-op call per use): bitwise dequantize + aten matmul at every row count, the dense product.
    # A single row keeps the fused GEMV custom op, bitwise its eager run.
    dtype = _half()
    N, K = 48, _row_length(ggml_type, 4)
    packed = random_packed(ggml_type, N, K, special = False).cuda()
    targets = []

    def backend(gm, example_inputs):
        targets.extend(str(n.target) for n in gm.graph.nodes if n.op == "call_function")
        from torch._inductor.compile_fx import compile_fx
        return compile_fx(gm, example_inputs)

    def fn(x, g):
        y = gguf_linear(x, packed, ggml_type, (N, K))
        dx = gguf_linear_backward_input(g, packed, ggml_type, (N, K))
        w = dequantize_gguf(packed, ggml_type, (N, K), x.dtype)
        return y, dx, w

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend = backend, fullgraph = True, dynamic = True)
    w = dequantize_gguf(packed, ggml_type, (N, K), dtype)
    for M in (5, 37, 130):
        x = (torch.randn(M, K, device = "cuda") * 1e-2).to(dtype)
        g = (torch.randn(M, N, device = "cuda") * 1e-2).to(dtype)
        got, want = compiled(x, g), (torch.nn.functional.linear(x, w), g @ w, w)
        for name, a, b in zip(("linear", "dX", "dequant"), got, want):
            assert torch.isfinite(b).all(), name
            assert torch.equal(a, b), f"compiled {name} differs from dequantize + matmul at M = {M}"
    assert not [t for t in targets if "unsloth_zoo" in t], targets
    assert any("triton_kernel_wrapper" in t for t in targets), targets
    targets.clear()
    torch._dynamo.reset()
    x1 = (torch.randn(1, K, device = "cuda") * 1e-2).to(dtype)
    one = torch.compile(lambda x: gguf_linear(x, packed, ggml_type, (N, K)), backend = backend, fullgraph = True, dynamic = False)
    assert torch.equal(one(x1), gguf_linear(x1, packed, ggml_type, (N, K)))
    assert any("unsloth_zoo.gguf_linear" in t for t in targets), targets
    torch._dynamo.reset()


@cuda
@pytest.mark.skipif(not gguf_ops._TRACED, reason = "traced dequantize is off (torch < 2.14 or UNSLOTH_GGUF_TRACED=0)")
def test_compiled_differentiable_linear_keeps_the_packed_only_backward():
    # x requiring grad: the custom op (its autograd saves only the packed bytes), never an aten matmul that would keep
    # the 16-bit weight alive until backward.
    dtype = _half()
    packed = random_packed(T.Q4_K, 64, 512, special = False).cuda()
    targets = []

    def backend(gm, example_inputs):
        targets.extend(str(n.target) for n in gm.graph.nodes if n.op == "call_function")
        return gm.forward

    torch._dynamo.reset()
    fn = torch.compile(lambda x: gguf_linear(x, packed, T.Q4_K, (64, 512)), backend = backend, fullgraph = True)
    x = (torch.randn(300, 512, device = "cuda") * 1e-2).to(dtype).requires_grad_()
    x2 = x.detach().clone().requires_grad_()
    y, y2 = fn(x), gguf_linear(x2, packed, T.Q4_K, (64, 512))
    assert torch.equal(y, y2)
    y.float().sum().backward()
    y2.float().sum().backward()
    assert torch.equal(x.grad, x2.grad)
    assert any("unsloth_zoo.gguf_linear" in t for t in targets), targets
    assert not any("triton_kernel_wrapper" in t for t in targets), targets
    torch._dynamo.reset()


def test_traced_dequantize_setting(monkeypatch):
    # Inductor re-emits a user Triton kernel with fp fusion on before torch 2.14: the traced launch is off there.
    has_triton = gguf_ops._HAS_TRITON
    monkeypatch.delenv("UNSLOTH_GGUF_TRACED", raising = False)
    monkeypatch.setattr(torch, "__version__", "2.13.1+cu130")
    assert gguf_ops._traced_setting() is False
    monkeypatch.setattr(torch, "__version__", "2.14.0+cu130")
    assert gguf_ops._traced_setting() is has_triton
    monkeypatch.setenv("UNSLOTH_GGUF_TRACED", "0")
    assert gguf_ops._traced_setting() is False
    monkeypatch.setattr(torch, "__version__", "2.6.0")
    monkeypatch.setenv("UNSLOTH_GGUF_TRACED", "1")
    assert gguf_ops._traced_setting() is has_triton


@cuda
def test_traced_kill_switch_keeps_the_custom_ops(monkeypatch):
    monkeypatch.setattr(gguf_ops, "_TRACED", False)
    packed = random_packed(T.Q4_K, 32, 512, special = False).cuda()
    targets = []

    def backend(gm, example_inputs):
        targets.extend(str(n.target) for n in gm.graph.nodes if n.op == "call_function")
        return gm.forward

    torch._dynamo.reset()
    fn = torch.compile(lambda x: gguf_linear(x, packed, T.Q4_K, (32, 512)), backend = backend, fullgraph = True)
    x = (torch.randn(300, 512, device = "cuda") * 1e-2).to(_half())
    assert torch.equal(fn(x), gguf_linear(x, packed, T.Q4_K, (32, 512)))
    assert any("unsloth_zoo.gguf_linear" in t for t in targets), targets
    assert not any("triton_kernel_wrapper" in t for t in targets), targets
    torch._dynamo.reset()


def test_ggml_type_id_accepts_ints_names_and_enums():
    import enum

    class Plain(enum.Enum):
        Q4_K = int(T.Q4_K)

    assert gguf_ops.ggml_type_id(int(T.Q4_K)) == T.Q4_K
    assert gguf_ops.ggml_type_id("q4_k") == T.Q4_K
    assert gguf_ops.ggml_type_id(Plain.Q4_K) == T.Q4_K
    with pytest.raises(NotImplementedError):
        gguf_ops.ggml_type_id(9999)


@cuda
def test_device_mismatch_raises():
    packed = random_packed(T.Q8_0, 4, 64)
    with pytest.raises(ValueError):
        gguf_linear(torch.randn(2, 64, device = "cuda"), packed, T.Q8_0, (4, 64))
    with pytest.raises(ValueError):
        gguf_grouped_mm(torch.randn(2, 64, device = "cuda"), packed.reshape(2, 2, -1).cuda(), T.Q8_0, (2, 2, 64), torch.tensor([1, 1]))


@cuda
def test_kill_switches(monkeypatch):
    packed = random_packed(T.Q4_0, 16, 64, special = False).cuda()
    x = torch.randn(2, 64, device = "cuda", dtype = torch.float16)
    calls = {"ref": 0, "mm": 0}
    real_ref, real_mm = gguf_ops.dequantize_reference, gguf_ops._mm_launch

    def ref(*a, **k):
        calls["ref"] += 1
        return real_ref(*a, **k)

    def mm(*a, **k):
        calls["mm"] += 1
        return real_mm(*a, **k)

    monkeypatch.setattr(gguf_ops, "dequantize_reference", ref)
    monkeypatch.setattr(gguf_ops, "_mm_launch", mm)
    gguf_linear(x, packed, T.Q4_0, (16, 64))
    assert calls == {"ref": 0, "mm": 1}
    monkeypatch.setenv("UNSLOTH_GGUF_FUSED", "0")
    gguf_linear(x, packed, T.Q4_0, (16, 64))
    assert calls == {"ref": 0, "mm": 1}
    monkeypatch.setenv("UNSLOTH_GGUF_TRITON", "0")
    gguf_linear(x, packed, T.Q4_0, (16, 64))
    assert calls == {"ref": 1, "mm": 1}


@cuda
@pytest.mark.skipif(os.environ.get("UNSLOTH_GGUF_BIG_TEST") != "1", reason = "needs ~8 GB; set UNSLOTH_GGUF_BIG_TEST=1")
def test_int64_offsets():
    # packed bytes past 2^31 and dequantized elements past 2^31
    rows, K = (1 << 31) // 2048 + 1024, 2048  # 2.28 GB of Q8_0 blocks, 2^31 + 2M values
    packed = torch.randint(0, 127, (rows, 2176), dtype = torch.uint8, device = "cuda")
    packed.view(-1, 34)[:, 1] = 0x3C  # Q8_0 scales near 1.0
    last = packed[-3:].cpu()
    want = dequantize_reference(last, T.Q8_0, (3, K))
    x = torch.randn(1, K, device = "cuda", dtype = torch.bfloat16)
    y = gguf_linear(x, packed, T.Q8_0, (rows, K))
    ref = x.double().cpu() @ want.double().t()
    assert torch.allclose(y[0, -3:].double().cpu(), ref[0], rtol = 2e-2, atol = 0.5)
    y16 = gguf_linear(torch.randn(16, K, device = "cuda", dtype = torch.bfloat16), packed, T.Q8_0, (rows, K))
    assert torch.isfinite(y16[:, -3:]).all()
    del y16
    small = packed[: (1 << 31) // K + 8]
    dense = dequantize_gguf(small, T.Q8_0, (small.shape[0], K), torch.bfloat16)
    assert dense.numel() > (1 << 31)
    assert _bits_equal(dense[-3:].cpu(), dequantize_reference(small[-3:].cpu(), T.Q8_0, (3, K), torch.bfloat16)) == 0


@cuda
@pytest.mark.parametrize("M", [1, 5])
def test_fp32_activations_keep_fp32_accuracy(M):
    # tl.dot would round float32 inputs to tf32 (~1e-3 relative); float32 activations must keep float32 accuracy
    packed = random_packed(T.Q6_K, 40, 512, special = False).cuda()
    w = dequantize_reference(packed, T.Q6_K, (40, 512)).cuda()
    x = torch.randn(M, 512, device = "cuda")
    got = gguf_linear(x, packed, T.Q6_K, (40, 512))
    ref = x.double() @ w.double().t()
    assert ((got.double() - ref).norm() / ref.norm()).item() < 1e-5


@cuda
@pytest.mark.parametrize("ggml_type", QUANT_TYPES, ids = _id)
@pytest.mark.parametrize("split", [True, False], ids = ["split8", "flat"])
def test_group_split_both_layouts(ggml_type, split, monkeypatch):
    # the per-format layout choice is a speed knob: both layouts must decode the same weights
    monkeypatch.setattr(gguf_ops, "_SPLIT_MM", frozenset(QUANT_TYPES) if split else frozenset())
    monkeypatch.setattr(gguf_ops, "_NO_SPLIT_GEMV", frozenset() if split else frozenset(QUANT_TYPES))
    dtype = _half()
    N, K = 40, _row_length(ggml_type, 2 if GGML_BLOCK[ggml_type][0] >= 256 else 4)
    packed = random_packed(ggml_type, N, K, seed = 11, special = False).cuda()
    w32 = dequantize_reference(packed, ggml_type, (N, K))
    for M in (1, 6):
        x = torch.randn(M, K, device = "cuda", dtype = dtype)
        ok, _ = _fp64_bound_ok(gguf_linear(x, packed, ggml_type, (N, K)), x, w32, torch.nn.functional.linear(x, w32.to(dtype)))
        assert ok
        gy = torch.randn(M, N, device = "cuda", dtype = dtype)
        ok, _ = _fp64_bound_ok(gguf_linear_backward_input(gy, packed, ggml_type, (N, K)), gy, w32.t().contiguous(), gy @ w32.to(dtype))
        assert ok


@cuda
@pytest.mark.parametrize("host_sched", [True, False], ids = ["host_sched", "kernel_sched"])
@pytest.mark.parametrize("ggml_type", [T.IQ4_XS, T.Q4_0, T.NVFP4], ids = _id)
def test_grouped_schedules(ggml_type, host_sched, monkeypatch):
    # the expert schedule (tile / row prefix sums) is computed in the kernel, or on device by torch on Triton < 3.3
    if not host_sched and gguf_ops._old_triton():
        pytest.skip("Triton < 3.3 scans a replicated layout wrongly: the in-kernel schedule is never used there")
    monkeypatch.setattr(gguf_ops, "_OLD_TRITON", host_sched)
    dtype = _half()
    E, N = 7, 32
    K = _row_length(ggml_type, 1 if GGML_BLOCK[ggml_type][0] == 256 else 8)
    packed = random_packed(ggml_type, E * N, K, seed = 3, special = False).reshape(E, N, -1).cuda()
    w32 = dequantize_reference(packed, ggml_type, (E, N, K))
    for counts_l in ([2, 0, 4, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 3], [40, 1, 0, 17, 0, 0, 2]):
        counts = torch.tensor(counts_l, dtype = torch.int32, device = "cuda")
        x = (torch.randn(sum(counts_l), K, device = "cuda") * 1e-2).to(dtype)
        guard = torch.full((4096,), 7.0, device = "cuda")
        y = gguf_grouped_mm(x, packed, ggml_type, (E, N, K), counts)
        rows = torch.repeat_interleave(torch.arange(E, device = "cuda"), counts.long())
        ref = torch.bmm(x.double().unsqueeze(1), w32.double()[rows].transpose(1, 2)).squeeze(1)
        stock = torch.bmm(x.unsqueeze(1), w32.to(dtype)[rows].transpose(1, 2)).squeeze(1)
        assert torch.isfinite(y).all() and (guard == 7.0).all()
        assert (y.double() - ref).abs().mean() <= 1.5 * (stock.double() - ref).abs().mean() + 1e-12


@cuda
@pytest.mark.parametrize("ggml_type", [T.Q4_0, T.Q8_0, T.Q4_K, T.Q6_K, T.IQ2_XXS, T.IQ3_S, T.NVFP4, T.Q1_0], ids = _id)
@pytest.mark.parametrize("M", [40, 130])
def test_fused_large_tiles(ggml_type, M, monkeypatch):
    # the per-GPU row cap may route these through dequantize + cuBLAS; force the 64 / 128-row tile configs
    monkeypatch.setattr(gguf_ops, "_FUSED_MAX_ROWS", 1 << 30)
    dtype = _half()
    N, K = 96, _row_length(ggml_type, 2 if GGML_BLOCK[ggml_type][0] >= 128 else 8)
    packed = random_packed(ggml_type, N, K, seed = M, special = False).cuda()
    w32 = dequantize_reference(packed, ggml_type, (N, K))
    calls = {"mm": 0}
    real = gguf_ops._mm_launch

    def mm(*a, **k):
        calls["mm"] += 1
        return real(*a, **k)

    monkeypatch.setattr(gguf_ops, "_mm_launch", mm)
    x = torch.randn(M, K, device = "cuda", dtype = dtype)
    ok, _ = _fp64_bound_ok(gguf_linear(x, packed, ggml_type, (N, K)), x, w32, torch.nn.functional.linear(x, w32.to(dtype)))
    assert ok
    gy = torch.randn(M, N, device = "cuda", dtype = dtype)
    ok, _ = _fp64_bound_ok(gguf_linear_backward_input(gy, packed, ggml_type, (N, K)), gy, w32.t().contiguous(), gy @ w32.to(dtype))
    assert ok and calls["mm"] == 2
