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
"""Fused MoE SwiGLU / gpt-oss clamped SwiGLU (moe_fused_swiglu) against the eager expression it replaces.

GPU only; runs from the zoo tree alone (pytest tests/test_moe_fused_swiglu.py). On a T4 (sm75) the bf16 cases
skip and fp16 / fp32 run. Bitwise = torch.equal plus equal sign bits (catches -0 vs +0)."""

import importlib
import os
import warnings

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA required")

if torch.cuda.is_available():
    sw = importlib.import_module("unsloth_zoo.temporary_patches.moe_fused_swiglu")

DEV = "cuda"
ACTS = ("silu", "gptoss")
LAYOUTS = ("halves", "interleaved")


def _dtypes():
    out = [torch.float16, torch.float32]
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        out.insert(0, torch.bfloat16)
    return out


DTYPES = _dtypes() if torch.cuda.is_available() else []


def _act(name):
    return sw.ACT_GPTOSS if name == "gptoss" else sw.ACT_SILU


def eager_expr(gate_up, act, interleaved, alpha = 1.702, limit = 7.0):
    """Verbatim the zoo eager expressions (moe_utils.forward_native_grouped_mm / _silu_and_mul)."""
    if interleaved:
        gate = gate_up[..., ::2]
        up = gate_up[..., 1::2]
    else:
        gate, up = gate_up.chunk(2, dim = -1)
    if act == "gptoss":
        gate = gate.clamp(min = None, max = limit)
        up = up.clamp(min = -limit, max = limit)
        glu = gate * torch.sigmoid(gate * alpha)
        return (up + 1.0) * glu
    return F.silu(gate) * up


def bitwise(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if not torch.equal(torch.nan_to_num(a, nan = 0.0), torch.nan_to_num(b, nan = 0.0)):
        return False
    if not torch.equal(torch.isnan(a), torch.isnan(b)):
        return False
    return torch.equal(torch.signbit(a), torch.signbit(b))


def make_gate_up(M, N, dtype, scale = 4.0, seed = 0):
    g = torch.Generator(device = DEV).manual_seed(seed)
    return (torch.randn(M, 2 * N, device = DEV, generator = g, dtype = torch.float32) * scale).to(dtype)


def run_both(gate_up, grad, act, interleaved, eager_rounding = True):
    x_e = gate_up.detach().clone().requires_grad_()
    y_e = eager_expr(x_e, act, interleaved)
    y_e.backward(grad)
    x_f = gate_up.detach().clone().requires_grad_()
    y_f = sw.moe_fused_swiglu(x_f, _act(act), interleaved, eager_rounding = eager_rounding)
    y_f.backward(grad)
    return y_e.detach(), x_e.grad, y_f.detach(), x_f.grad


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("act", ACTS)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("M,N", [(1, 1), (7, 13), (33, 768), (129, 2880), (5, 1031)])
def test_bitwise_vs_eager(dtype, act, layout, M, N):
    interleaved = layout == "interleaved"
    gu = make_gate_up(M, N, dtype, seed = M * 7 + N)
    grad = make_gate_up(M, N, dtype, scale = 1.0, seed = M + N)[:, :N].contiguous()
    calls = list(sw.FUSED_SWIGLU_CALLS)
    y_e, g_e, y_f, g_f = run_both(gu, grad, act, interleaved)
    assert sw.FUSED_SWIGLU_CALLS[0] == calls[0] + 1 and sw.FUSED_SWIGLU_CALLS[1] == calls[1] + 1
    assert bitwise(y_f, y_e), (y_f - y_e).abs().max()
    assert bitwise(g_f, g_e), (g_f.float() - g_e.float()).abs().max()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("act", ACTS)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_special_values(dtype, act, layout):
    interleaved = layout == "interleaved"
    fi = torch.finfo(dtype)
    vals = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, 7.0, -7.0, 7.0001, -7.0001, 6.999, 20.0, -20.0, 88.0, -88.0, 100.0, -100.0,
         fi.max, -fi.max, fi.tiny, -fi.tiny, fi.tiny / 4, -fi.tiny / 4, float("inf"), float("-inf"), float("nan")],
        dtype = torch.float32,
    ).to(dtype)
    n = vals.numel()
    gate = vals.repeat_interleave(n)
    up = vals.repeat(n)
    if interleaved:
        gu = torch.stack((gate, up), -1).flatten().view(1, -1).to(DEV)
    else:
        gu = torch.cat((gate, up)).view(1, -1).to(DEV)
    N = n * n
    grad_vals = torch.tensor([1.0, -1.0, 0.0, -0.0, 1e-3, fi.tiny, 3.0], dtype = torch.float32).to(dtype)
    grad = grad_vals.repeat(N // grad_vals.numel() + 1)[:N].view(1, N).to(DEV)
    y_e, g_e, y_f, g_f = run_both(gu, grad, act, interleaved)
    assert bitwise(y_f, y_e)
    assert bitwise(g_f, g_e)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("act", ACTS)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_strided_and_nd(dtype, act, layout):
    interleaved = layout == "interleaved"
    M, N = 37, 96
    big = make_gate_up(2 * M, 3 * N, dtype, seed = 3)
    cases = [
        big[::2, : 2 * N],                              # row stride 6I, sliced
        make_gate_up(2 * N, M, dtype, seed = 4).t(),    # column stride != 1
        make_gate_up(M * 2, N, dtype, seed = 5).view(2, M, 2 * N),  # 3-D
    ]
    for gu in cases:
        lead = gu.shape[:-1]
        grad = torch.randn(*lead, N, device = DEV, dtype = torch.float32).to(dtype)
        for g in (grad, grad.transpose(-1, -2).contiguous().transpose(-1, -2), torch.ones((), device = DEV, dtype = dtype).expand(*lead, N)):
            y_e, g_e, y_f, g_f = run_both(gu, g, act, interleaved)
            assert bitwise(y_f, y_e)
            assert bitwise(g_f, g_e)


@pytest.mark.parametrize("act", ACTS)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_empty_rows(act, layout):
    gu = torch.empty(0, 64, device = DEV, dtype = torch.float16, requires_grad = True)
    y = sw.moe_fused_swiglu(gu, _act(act), layout == "interleaved")
    assert y.shape == (0, 32)
    y.sum().backward()
    assert gu.grad.shape == (0, 64)


@pytest.mark.parametrize("act", ACTS)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_large_m(act, layout):
    dtype = DTYPES[0]
    M, N = 70000, 768
    gu = make_gate_up(M, N, dtype, seed = 11)
    grad = make_gate_up(M, N // 2, dtype, scale = 1.0, seed = 12)
    y_e, g_e, y_f, g_f = run_both(gu, grad, act, layout == "interleaved")
    assert bitwise(y_f, y_e)
    assert bitwise(g_f, g_e)


def _fp64_err(dtype, act, interleaved, eager_rounding, M = 512, N = 1024):
    gu = make_gate_up(M, N, dtype, seed = 21)
    grad = make_gate_up(M, N // 2, dtype, scale = 1.0, seed = 22)
    x64 = gu.double().requires_grad_()
    y64 = eager_expr(x64, act, interleaved)
    y64.backward(grad.double())
    x = gu.clone().requires_grad_()
    y = sw.moe_fused_swiglu(x, _act(act), interleaved, eager_rounding = eager_rounding)
    y.backward(grad)
    return (y.double() - y64).abs().max().item(), (x.grad.double() - x64.grad).abs().max().item(), y64.abs().max().item(), x64.grad.abs().max().item()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("act", ACTS)
def test_fp64_reference(dtype, act):
    # eager_rounding=True is the eager expression (bitwise above); round-once must be no worse than it.
    for interleaved in (False, True):
        fe, be, fmax, bmax = _fp64_err(dtype, act, interleaved, True)
        fo, bo, _, _ = _fp64_err(dtype, act, interleaved, False)
        eps = torch.finfo(dtype).eps
        assert fo <= fe + 1e-12 and fe <= 4 * eps * max(fmax, 1.0)
        assert bo <= be + 2 * eps * max(bmax, 1.0) and be <= 16 * eps * max(bmax, 1.0)


def test_comparator_negative_control():
    gu = make_gate_up(64, 64, DTYPES[0])
    y = sw.moe_fused_swiglu(gu, sw.ACT_SILU, False)
    assert not bitwise(y * (1 + 2 ** -7), y)
    assert not bitwise(-torch.zeros(4, device = DEV), torch.zeros(4, device = DEV))


@pytest.mark.parametrize("act", ACTS)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_saves_only_gate_up(act, layout):
    dtype = DTYPES[0]
    M, N = 256, 512
    gu0 = make_gate_up(M, N, dtype)

    def saved_bytes(fn):
        seen = {}

        def pack(t):
            seen[(t.untyped_storage().data_ptr())] = t.untyped_storage().nbytes()
            return t

        x = gu0.clone().requires_grad_()
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            fn(x)
        return sum(seen.values())

    fused = saved_bytes(lambda x: sw.moe_fused_swiglu(x, _act(act), layout == "interleaved"))
    eager = saved_bytes(lambda x: eager_expr(x, act, layout == "interleaved"))
    assert fused == gu0.numel() * gu0.element_size()
    assert eager > fused


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("act", ACTS)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_compile_fullgraph_no_breaks(dynamic, act, layout):
    from torch._dynamo.utils import counters
    torch._dynamo.reset()
    counters.clear()
    dtype = DTYPES[0]
    interleaved = layout == "interleaved"
    w = (torch.randn(64, 2 * 96, device = DEV) * 0.3).to(dtype).requires_grad_()

    def f(x, w):
        return sw.moe_fused_swiglu(x @ w, _act(act), interleaved)

    cf = torch.compile(f, fullgraph = True, dynamic = dynamic)
    calls = sw.FUSED_SWIGLU_CALLS[0]
    for M in (40, 57, 1):
        x = torch.randn(M, 64, device = DEV).to(dtype).requires_grad_()
        go = torch.randn(M, 96, device = DEV).to(dtype)
        y_c = cf(x, w)
        gx_c, gw_c = torch.autograd.grad(y_c, (x, w), go)
        y_e = f(x, w)
        gx_e, gw_e = torch.autograd.grad(y_e, (x, w), go)
        assert bitwise(y_c, y_e)
        assert bitwise(gx_c, gx_e) and bitwise(gw_c, gw_e)
    # the compiled graph ran the custom op (impl bumps the counter), not a traced eager expression
    assert sw.FUSED_SWIGLU_CALLS[0] >= calls + 6
    assert sum(counters["graph_break"].values()) == 0, dict(counters["graph_break"])


@pytest.mark.skipif(os.environ.get("UNSLOTH_MOE_FUSED_SWIGLU", "1") == "0", reason = "kill switch preset")
def test_kill_switch(monkeypatch):
    gu = make_gate_up(8, 16, DTYPES[0])
    assert sw.fused_swiglu_supported(gu)
    monkeypatch.setenv("UNSLOTH_MOE_FUSED_SWIGLU", "0")
    sw.refresh_fused_swiglu_enabled()
    try:
        assert not sw.fused_swiglu_supported(gu)
    finally:
        monkeypatch.delenv("UNSLOTH_MOE_FUSED_SWIGLU")
        sw.refresh_fused_swiglu_enabled()
    assert sw.fused_swiglu_supported(gu)
    assert not sw.fused_swiglu_supported(gu.cpu())


@pytest.mark.parametrize("act", ACTS)
def test_launch_failure_falls_back(monkeypatch, act):
    dtype = DTYPES[0]
    gu = make_gate_up(33, 40, dtype)
    grad = make_gate_up(33, 20, dtype, seed = 2)

    def boom(*a, **k):
        raise RuntimeError("simulated launch failure")

    monkeypatch.setattr(sw, "_launch_fwd", boom)
    monkeypatch.setattr(sw, "_launch_bwd", boom)
    try:
        with warnings.catch_warnings(record = True):
            warnings.simplefilter("always")
            y_e, g_e, y_f, g_f = run_both(gu, grad, act, True)
        assert sw._STATE["failed"]
        assert not sw.fused_swiglu_supported(gu)
        assert bitwise(y_f, y_e) and bitwise(g_f, g_e)
    finally:
        sw._STATE["failed"] = False


def test_double_backward():
    dtype = torch.float32
    gu = make_gate_up(9, 12, dtype).requires_grad_()
    for act in ACTS:
        y = sw.moe_fused_swiglu(gu, _act(act), False)
        (g,) = torch.autograd.grad(y.sum(), gu, create_graph = True)
        (gg,) = torch.autograd.grad(g.square().sum(), gu)
        ye = eager_expr(gu, act, False)
        (ge,) = torch.autograd.grad(ye.sum(), gu, create_graph = True)
        (gge,) = torch.autograd.grad(ge.square().sum(), gu)
        assert torch.allclose(gg, gge, rtol = 1e-5, atol = 1e-6)


@pytest.mark.parametrize("act", ACTS)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_self_check(monkeypatch, act, layout):
    dtype = DTYPES[0]
    interleaved = layout == "interleaved"
    assert sw.fused_swiglu_self_check(torch.device(DEV, torch.cuda.current_device()), dtype, _act(act), interleaved)
    # A kernel that drifts by one rounding step must fail the probe (negative control), uncached key.
    real = sw._launch_fwd
    monkeypatch.setattr(sw, "_launch_fwd", lambda *a: real(*a) * (1 + 2 ** -7))
    with warnings.catch_warnings(record = True):
        warnings.simplefilter("always")
        assert not sw.fused_swiglu_self_check(torch.device(DEV, torch.cuda.current_device()), dtype, _act(act), interleaved, 1.5, 3.0)


def test_clamp_boundary_probe_matches_autograd():
    x = torch.tensor([7.0, -7.0, 6.0], device = DEV, requires_grad = True)
    x.clamp(min = -7.0, max = 7.0).sum().backward()
    _, u_max, u_min = sw._clamp_grad_boundaries()
    assert (x.grad[0] != 0) == u_max and (x.grad[1] != 0) == u_min and x.grad[2] == 1
