# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
"""Fused decode-in-GEMM path for NVFP4 / MXFP8 packed expert stacks (mxfp4_gemm.packed_grouped_mm): the decoded tile
is bitwise the dequantized weight, products match an fp64 reference as closely as dequantize + grouped GEMM, the
auto row cutoff and the kill switch pick the path, autograd gives dX, and a compiled caller sees one opaque op."""

import pytest
import torch

import unsloth_zoo.packed_experts as pe
from unsloth_zoo import mxfp4_gemm as mg
from unsloth_zoo.packed_experts import (
    Nvfp4ExpertParam,
    packed_expert_grouped_linear,
    quantize_experts_mxfp8,
    quantize_experts_nvfp4,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None or torch.cuda.get_device_capability() < (8, 0),
    reason = "fused NVFP4 / MXFP8 grouped GEMM needs CUDA sm_80+",
)

_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _param(fmt, E, N, K, dtype = torch.bfloat16, parts = 1, multiply = False, seed = 0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn((E, N, K), generator = g).cuda()  # unit scale: no * 0.02, so a decode slip is not hidden
    if fmt == "mxfp8":
        return quantize_experts_mxfp8(w, dtype)
    p = quantize_experts_nvfp4(w, dtype)
    if parts == 1 and not multiply:
        return p
    scales, glob = p._companions()
    glob = glob.repeat(1, parts) * torch.linspace(1.0, 1.5, parts, device = "cuda")
    if multiply:
        glob = 1.0 / glob
    return Nvfp4ExpertParam(p.data, packed_scales = scales, packed_global = glob, packed_dtype = dtype,
                            packed_global_mode = "multiply" if multiply else "divide")


def _ref64(fmt, param):
    """Independent fp64 decode of the packed rows (E, N, K): low nibble = even column."""
    scales, glob = param._companions()
    codes = param.data.cpu()
    E, N = codes.shape[:2]
    if fmt == "nvfp4":
        lut = torch.tensor(_E2M1 + [-v for v in _E2M1], dtype = torch.float64)
        vals = torch.stack((lut[(codes & 15).long()], lut[(codes >> 4).long()]), -1).reshape(E, N, -1)
        s = scales.cpu().view(torch.float8_e4m3fn).to(torch.float64).repeat_interleave(16, -1)
        parts = glob.reshape(E, -1).shape[1]
        gl = glob.cpu().to(torch.float64).reshape(E, parts).repeat_interleave(N // parts, 1)[:, :, None]
        return vals * (s * gl if param.packed_global_mode == "multiply" else s / gl)
    q = codes.view(torch.float8_e4m3fn).to(torch.float64)
    return q * torch.exp2(scales.cpu().to(torch.float64) - 127).repeat_interleave(32, -1)


def _call(param, x, counts, transpose_b):
    scales, glob = param._companions()
    return mg.packed_grouped_mm(
        x, param.data, scales, glob, counts, param.packed_format, transpose_b, param.packed_global_mode == "multiply",
    )


def _route(counts, in_dim, dtype, seed = 1):
    g = torch.Generator().manual_seed(seed)
    counts = torch.tensor(counts, dtype = torch.int32)
    x = torch.randn((int(counts.sum()), in_dim), generator = g).to(dtype).cuda()
    rows = torch.repeat_interleave(torch.arange(len(counts)), counts.long())
    return x, counts.cuda(), rows


CASES = [("nvfp4", 1, False), ("nvfp4", 2, False), ("nvfp4", 1, True), ("nvfp4", 2, True), ("mxfp8", 1, False)]


@pytest.mark.parametrize("fmt,parts,multiply", CASES)
@pytest.mark.parametrize("transpose_b", [True, False])
def test_decoded_tile_is_the_dequantized_weight(fmt, parts, multiply, transpose_b):
    # One-hot rows read single weight entries through the GEMM (1 * w + exact zeros), so any decode slip (nibble
    # order, scale group, global divide vs multiply, rounding) shows as a bit difference against dequantize.
    E, N, K = 3, 80, 96  # 80 / 96: partial N and K tiles
    param = _param(fmt, E, N, K, parts = parts, multiply = multiply)
    dense = param.dequantize(torch.bfloat16, transpose = False)  # (E, N, K)
    in_dim = K if transpose_b else N
    for e in range(E):
        counts = [0] * E
        counts[e] = in_dim
        x = torch.eye(in_dim, dtype = torch.bfloat16, device = "cuda")
        got = _call(param, x, torch.tensor(counts, dtype = torch.int32, device = "cuda"), transpose_b)
        want = dense[e].t() if transpose_b else dense[e]  # row i of eye @ W = W row i
        assert torch.equal(got.view(torch.int16), want.contiguous().view(torch.int16)), (fmt, e, transpose_b)


@pytest.mark.parametrize("fmt,parts,multiply", CASES)
@pytest.mark.parametrize("transpose_b", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("counts", [[5, 0, 19, 1], [64, 64, 3, 0], [1, 1, 1, 1]])
def test_products_as_close_to_fp64_as_dequantize(fmt, parts, multiply, transpose_b, dtype, counts):
    E, N, K = 4, 128, 160
    param = _param(fmt, E, N, K, dtype, parts = parts, multiply = multiply)
    x, c, rows = _route(counts, K if transpose_b else N, dtype)
    got = _call(param, x, c, transpose_b)
    W64 = _ref64(fmt, param)[rows]
    ref = torch.bmm(x.cpu().double().unsqueeze(1), W64.transpose(1, 2) if transpose_b else W64).squeeze(1)
    dense = param.dequantize(dtype, transpose = False).float()[rows.cuda()]
    base = torch.bmm(x.float().unsqueeze(1), dense.transpose(1, 2) if transpose_b else dense).squeeze(1).to(dtype)
    err = (got.cpu().double() - ref).norm() / ref.norm()
    err_base = (base.cpu().double() - ref).norm() / ref.norm()
    assert torch.isfinite(got).all()
    assert err <= 1.5 * err_base + 1e-3, (err.item(), err_base.item())


def test_wrong_decode_is_caught():
    # Negative control for the one-hot test: a perturbed weight must not pass the bitwise comparison.
    param = _param("nvfp4", 2, 64, 64)
    dense = param.dequantize(torch.bfloat16, transpose = False)
    x = torch.eye(64, dtype = torch.bfloat16, device = "cuda")
    got = _call(param, x, torch.tensor([64, 0], dtype = torch.int32, device = "cuda"), True)
    assert torch.equal(got, dense[0].t().contiguous())
    assert not torch.equal(got * (1 + 2 ** -7), dense[0].t().contiguous())


def _counters():
    return dict(pe.PACKED_EXPERT_STATS)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
def test_auto_cutoff_and_kill_switch_pick_the_path(monkeypatch, fmt):
    E, N, K = 4, 64, 96
    param = _param(fmt, E, N, K)
    few = torch.tensor([3, 0, 5, 2], device = "cuda")
    x = torch.randn((10, K), dtype = torch.bfloat16, device = "cuda")
    monkeypatch.delenv("UNSLOTH_PACKED_EXPERTS_FUSED", raising = False)
    before = _counters()
    fused = packed_expert_grouped_linear(x, param, few, True)
    assert pe.PACKED_EXPERT_STATS["fused"] - before["fused"] == 1
    assert pe.PACKED_EXPERT_STATS["chunks"] == before["chunks"]
    monkeypatch.setenv("UNSLOTH_PACKED_EXPERTS_FUSED", "0")
    before = _counters()
    dequant = packed_expert_grouped_linear(x, param, few, True)
    assert pe.PACKED_EXPERT_STATS["fused"] == before["fused"]
    assert pe.PACKED_EXPERT_STATS["chunks"] > before["chunks"]
    assert (fused.float() - dequant.float()).abs().max() <= 2e-2 * dequant.float().abs().max()
    # Past the per-expert row cutoff auto dequantizes; =1 fuses at any size.
    monkeypatch.setenv("UNSLOTH_PACKED_EXPERTS_FUSED", "auto")
    monkeypatch.setenv("UNSLOTH_PACKED_EXPERTS_FUSED_MAX_ROWS", "2")
    before = _counters()
    packed_expert_grouped_linear(x, param, few, True)
    assert pe.PACKED_EXPERT_STATS["fused"] == before["fused"]
    monkeypatch.setenv("UNSLOTH_PACKED_EXPERTS_FUSED", "1")
    before = _counters()
    packed_expert_grouped_linear(x, param, few, True)
    assert pe.PACKED_EXPERT_STATS["fused"] - before["fused"] == 1


def test_fp32_inputs_keep_the_dequantize_path(monkeypatch):
    monkeypatch.setenv("UNSLOTH_PACKED_EXPERTS_FUSED", "1")
    param = _param("nvfp4", 2, 64, 64, torch.float32)
    x = torch.randn((4, 64), device = "cuda")
    before = _counters()
    packed_expert_grouped_linear(x, param, torch.tensor([2, 2], device = "cuda"), True)
    assert pe.PACKED_EXPERT_STATS["fused"] == before["fused"]


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("transpose_b", [True, False])
def test_autograd_dx_matches_the_dequantize_path(monkeypatch, fmt, transpose_b):
    E, N, K = 4, 96, 128
    param = _param(fmt, E, N, K)
    counts = torch.tensor([7, 0, 12, 4], device = "cuda")
    in_dim = K if transpose_b else N
    x = torch.randn((23, in_dim), dtype = torch.bfloat16, device = "cuda")
    g = torch.randn((23, N if transpose_b else K), dtype = torch.bfloat16, device = "cuda")
    grads = {}
    for mode in ("1", "0"):
        monkeypatch.setenv("UNSLOTH_PACKED_EXPERTS_FUSED", mode)
        before = _counters()
        inp = x.clone().requires_grad_(True)
        out = packed_expert_grouped_linear(inp, param, counts, transpose_b)
        out.backward(g)
        fused_calls = pe.PACKED_EXPERT_STATS["fused"] - before["fused"]
        assert fused_calls == (2 if mode == "1" else 0)  # forward and dX
        grads[mode] = (out.detach().float(), inp.grad.float())
    for a, b in zip(grads["1"], grads["0"]):
        assert torch.isfinite(a).all()
        assert (a - b).abs().max() <= 2e-2 * b.abs().max()
    assert param.grad is None


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
def test_compiled_caller_runs_the_fused_op(monkeypatch, fmt):
    monkeypatch.setenv("UNSLOTH_PACKED_EXPERTS_FUSED", "1")
    torch._dynamo.reset()
    param = _param(fmt, 4, 64, 96)
    counts = torch.tensor([5, 0, 7, 3], device = "cuda")
    x = torch.randn((15, 96), dtype = torch.bfloat16, device = "cuda")

    def fwd(inp):
        return packed_expert_grouped_linear(inp, param, counts, True)

    inp = x.clone().requires_grad_(True)
    want = fwd(inp)
    want.float().square().sum().backward()
    want_dx = inp.grad.clone()
    from torch._dynamo.utils import counters
    counters.clear()
    before = _counters()
    compiled = torch.compile(fwd, fullgraph = True)
    inp = x.clone().requires_grad_(True)
    out = compiled(inp)
    out.float().square().sum().backward()
    assert torch.equal(out.detach(), want.detach()) and torch.equal(inp.grad, want_dx)
    assert pe.PACKED_EXPERT_STATS["fused"] - before["fused"] >= 2
    assert not counters["graph_break"]


def test_large_expert_count_and_row_offsets():
    # 128 experts (Qwen3-30B-A3B), skewed routing: the per-tile expert lookup and int64 row offsets.
    E, N, K = 128, 64, 128
    param = _param("nvfp4", E, N, K)
    g = torch.Generator().manual_seed(3)
    counts = torch.randint(0, 9, (E,), generator = g)
    counts[::7] = 0
    x, c, rows = _route(counts.tolist(), K, torch.bfloat16)
    got = _call(param, x, c, True)
    dense = param.dequantize(torch.bfloat16, transpose = False).float()[rows.cuda()]
    want = torch.bmm(x.float().unsqueeze(1), dense.transpose(1, 2)).squeeze(1)
    assert (got.float() - want).abs().max() <= 1e-2 * want.abs().max()


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
def test_strided_counts_match_contiguous(fmt):
    # A counts view with a stride (every other element of a larger buffer) must route rows as its values say.
    E, N, K = 4, 64, 96
    param = _param(fmt, E, N, K)
    x, counts, _ = _route([3, 0, 5, 2], K, torch.bfloat16)
    storage = torch.zeros(2 * E, dtype = torch.int32, device = "cuda")
    storage[::2] = counts
    strided = storage[::2]
    assert not strided.is_contiguous()
    assert torch.equal(_call(param, x, strided, True), _call(param, x, counts, True))
