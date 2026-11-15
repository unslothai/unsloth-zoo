# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
"""NVFP4 / MXFP8 packed routed-expert stacks: decode, RTN quantizers, grouped forward + dX bitwise vs the dense
dequantized stack, the MoE route, and the param's device / copy / pickle contract."""

import copy
import pickle

import pytest
import torch

import unsloth_zoo.packed_experts as pe
from unsloth_zoo.packed_experts import (
    Mxfp8ExpertParam,
    Nvfp4ExpertParam,
    mxfp8_experts_dequantize,
    mxfp8_experts_dequantize_torch,
    nvfp4_experts_dequantize,
    nvfp4_experts_dequantize_torch,
    packed_expert_grouped_linear,
    quantize_experts_mxfp4,
    quantize_experts_mxfp8,
    quantize_experts_nvfp4,
)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _ref_nvfp4(codes, scales, glob, dtype, multiply = False):
    # Independent per-element reference: e2m1 * (scale / global) in fp32, one cast; global per (expert, part).
    codes, scales, glob = codes.cpu(), scales.cpu(), glob.cpu()
    E, N, half = codes.shape
    lut = torch.tensor(_E2M1 + [-v for v in _E2M1])
    vals = torch.empty(E, N, half * 2)
    vals[..., 0::2] = lut[(codes & 15).long()]
    vals[..., 1::2] = lut[(codes >> 4).long()]
    s = scales.to(torch.float32).repeat_interleave(16, dim = -1)
    parts = glob.reshape(E, -1).shape[1]
    g = glob.reshape(E, parts).repeat_interleave(N // parts, dim = 1)[:, :, None]
    return (vals * (s * g if multiply else s / g)).to(dtype)


def _ref_mxfp8(codes, scales, dtype):
    q = codes.cpu().view(torch.float8_e4m3fn).to(torch.float32)
    s = (scales.cpu().to(torch.float32) - 127.0).exp2().repeat_interleave(32, dim = -1)
    return (q * s).to(dtype)


def _random_nvfp4(E, N, K, parts = 1, seed = 0, device = "cpu"):
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 256, (E, N, K // 2), dtype = torch.uint8, generator = g)
    # Finite e4m3 scales only (0x7F / 0xFF are NaN).
    sbytes = torch.randint(0, 0x7E, (E, N, K // 16), dtype = torch.uint8, generator = g)
    sbytes |= (torch.rand((E, N, K // 16), generator = g) < 0.1).to(torch.uint8) << 7
    scales = sbytes.view(torch.float8_e4m3fn)
    glob = torch.rand((E, parts), generator = g) * 3000 + 1.0
    return codes.to(device), scales.to(device), glob.to(device)


def _random_mxfp8(E, N, K, seed = 0, device = "cpu"):
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 256, (E, N, K), dtype = torch.uint8, generator = g)
    codes[(codes & 0x7F) == 0x7F] = 0  # no NaN codes
    scales = torch.randint(110, 135, (E, N, K // 32), dtype = torch.uint8, generator = g)
    return codes.to(device), scales.to(device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("parts", [1, 2])
@pytest.mark.parametrize("multiply", [False, True])
def test_nvfp4_decode_bitwise(device, dtype, parts, multiply):
    codes, scales, glob = _random_nvfp4(3, 48, 64, parts = parts, device = device)
    if multiply:
        glob = 1.0 / glob
    ref = _ref_nvfp4(codes, scales, glob, dtype, multiply)
    for transpose in (False, True):
        out = nvfp4_experts_dequantize(codes, scales, glob, dtype, transpose = transpose, multiply = multiply).cpu()
        want = ref.transpose(-1, -2) if transpose else ref
        assert out.shape == want.shape
        assert torch.equal(out.view(torch.int16 if dtype != torch.float32 else torch.int32),
                           want.contiguous().view(torch.int16 if dtype != torch.float32 else torch.int32))


@needs_cuda
def test_nvfp4_triton_matches_unsloth_2d_reference():
    # One expert, one global scale: the 2-D compressed-tensors decode of unsloth (if importable) on the same bytes.
    nvfp4 = pytest.importorskip("unsloth.kernels.nvfp4")
    codes, scales, glob = _random_nvfp4(4, 64, 128, device = "cuda")
    stack = nvfp4_experts_dequantize(codes, scales, glob, torch.bfloat16)
    for e in range(4):
        two_d = nvfp4.nvfp4_dequantize(codes[e], scales[e], glob[e].reshape(1), torch.bfloat16)
        assert torch.equal(stack[e].view(torch.int16), two_d.view(torch.int16))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mxfp8_decode_bitwise(device, dtype):
    codes, scales = _random_mxfp8(3, 40, 96, device = device)
    ref = _ref_mxfp8(codes, scales, dtype)
    for transpose in (False, True):
        out = mxfp8_experts_dequantize(codes, scales, dtype, transpose = transpose).cpu()
        want = ref.transpose(-1, -2).contiguous() if transpose else ref
        assert torch.equal(out.view(torch.int16), want.view(torch.int16))


@needs_cuda
def test_triton_equals_torch_reference_with_token_counts():
    codes, scales, glob = _random_nvfp4(4, 32, 64, device = "cuda")
    counts = torch.tensor([3, 0, 1, 0], device = "cuda")
    out = nvfp4_experts_dequantize(codes, scales, glob, torch.bfloat16, token_counts = counts)
    ref = nvfp4_experts_dequantize_torch(codes, scales, glob, torch.bfloat16)
    assert torch.equal(out[0], ref[0]) and torch.equal(out[2], ref[2])
    c8, s8 = _random_mxfp8(2, 16, 64, device = "cuda")
    assert torch.equal(mxfp8_experts_dequantize(c8, s8, torch.bfloat16), mxfp8_experts_dequantize_torch(c8, s8, torch.bfloat16))


def _rtn_stack(E = 4, N = 64, K = 96, seed = 0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn((E, N, K), generator = g) * 0.02


@pytest.mark.parametrize("fmt,tol", [("nvfp4", 0.12), ("mxfp4", 0.16), ("mxfp8", 0.04)])
def test_rtn_quantizers_round_trip(fmt, tol):
    w = _rtn_stack()
    q = {"nvfp4": quantize_experts_nvfp4, "mxfp4": quantize_experts_mxfp4, "mxfp8": quantize_experts_mxfp8}[fmt](w, torch.float32)
    deq = q.dequantize(torch.float32)
    assert tuple(q._original_shape) == tuple(w.shape) == tuple(deq.shape)
    rel = float((deq - w).norm() / w.norm())
    assert rel < tol, rel
    # Re-quantizing the decoded stack is a fixed point (codes land exactly on the grid).
    if fmt == "mxfp8":
        assert torch.equal(quantize_experts_mxfp8(deq).data, q.data)


def test_nvfp4_rtn_picks_nearest_code():
    # A single block whose values sit on the grid after scaling: RTN must reproduce them exactly.
    vals = torch.tensor(_E2M1 + [-v for v in _E2M1[1:]] + [0.0], dtype = torch.float32)  # 16 values, max 6
    w = vals.view(1, 1, 16).repeat(2, 3, 2) * 0.125
    q = quantize_experts_nvfp4(w, torch.float32)
    torch.testing.assert_close(q.dequantize(torch.float32), w, rtol = 0, atol = 0)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("transpose_b", [True, False])
@pytest.mark.parametrize("device", DEVICES)
def test_grouped_forward_and_dx_bitwise_vs_dense(fmt, transpose_b, device):
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    E, N, K = 4, 64, 96
    w = _rtn_stack(E, N, K).to(device)
    param = (quantize_experts_nvfp4 if fmt == "nvfp4" else quantize_experts_mxfp8)(w, dtype)
    dense = param.dequantize(dtype, transpose = False)  # (E, N, K) packed rows
    weight = dense.transpose(-1, -2) if transpose_b else dense  # what x @ W multiplies
    in_dim = weight.shape[1]
    counts = torch.tensor([5, 0, 7, 3], device = device)
    x = torch.randn((int(counts.sum()), in_dim), device = device, dtype = dtype, requires_grad = True)
    x2 = x.detach().clone().requires_grad_(True)
    out = packed_expert_grouped_linear(x, param, counts, transpose_b)
    from unsloth_zoo.mxfp4_stacked_experts import _grouped_mm
    offsets = torch.cumsum(counts, 0, dtype = torch.int32)
    # The 16-bit path multiplies the module's (E, N, K) stack through a transposed view: same layout here.
    ref = _grouped_mm(x2, weight, offsets)
    assert torch.equal(out, ref)
    g = torch.randn_like(out)
    out.backward(g)
    ref.backward(g)
    assert torch.equal(x.grad, x2.grad)


def test_grouped_multi_chunk_matches_single_chunk(monkeypatch):
    E, N, K = 6, 32, 64
    param = quantize_experts_nvfp4(_rtn_stack(E, N, K), torch.float32)
    counts = torch.tensor([2, 0, 3, 1, 0, 4])
    x = torch.randn((10, K))
    one = packed_expert_grouped_linear(x, param, counts, True)
    import unsloth_zoo.mxfp4_stacked_experts as mse
    monkeypatch.setenv("UNSLOTH_MXFP4_EXPERT_CHUNK_MB", "0")
    monkeypatch.setattr(mse, "_chunk_bytes", lambda: N * K * 4 * 2)  # two experts per chunk
    before = pe.PACKED_EXPERT_STATS["chunks"]
    many = packed_expert_grouped_linear(x, param, counts, True)
    assert pe.PACKED_EXPERT_STATS["chunks"] - before >= 2
    assert torch.equal(one, many)


def test_param_contract_device_copy_pickle():
    param = quantize_experts_nvfp4(_rtn_stack(2, 32, 32), torch.bfloat16, transposed = True)
    assert tuple(param._original_shape) == (2, 32, 32) and param.packed_transposed
    assert param.to(torch.float16) is param  # a dtype cast never touches the packed bytes
    clone = copy.deepcopy(param)
    assert type(clone) is Nvfp4ExpertParam and torch.equal(clone.data, param.data)
    assert clone.packed_scales.data_ptr() != param.packed_scales.data_ptr()
    back = pickle.loads(pickle.dumps(param))
    assert torch.equal(back.dequantize(), param.dequantize())
    # accelerate rebuild: cls(data, requires_grad, **__dict__)
    rebuilt = Nvfp4ExpertParam(param.data, requires_grad = False, **param.__dict__)
    assert torch.equal(rebuilt.dequantize(), param.dequantize())
    module = torch.nn.Module()
    module.w = param
    module.half()
    assert type(module.w) is Nvfp4ExpertParam and module.w.dtype == torch.uint8
    with pytest.raises(RuntimeError):
        param.copy_(torch.zeros_like(param.data))
    fresh = quantize_experts_nvfp4(_rtn_stack(2, 32, 32, seed = 1), torch.bfloat16, transposed = True)
    param.copy_(fresh)
    assert torch.equal(param.dequantize(), fresh.dequantize())
    m8 = quantize_experts_mxfp8(_rtn_stack(2, 32, 64))
    assert type(copy.deepcopy(m8)) is Mxfp8ExpertParam
    assert type(m8.detach()) is Mxfp8ExpertParam


def test_bad_shapes_refused():
    codes, scales, glob = _random_nvfp4(2, 16, 32)
    with pytest.raises(ValueError):
        Nvfp4ExpertParam(codes, packed_scales = scales[:, :, :1], packed_global = glob)
    with pytest.raises(ValueError):
        Nvfp4ExpertParam(codes, packed_scales = scales, packed_global = glob.double())
    with pytest.raises(ValueError):
        Nvfp4ExpertParam(codes, packed_scales = scales, packed_global = torch.ones(2, 3))  # 16 rows do not split in 3
    c8, s8 = _random_mxfp8(2, 16, 64)
    with pytest.raises(ValueError):
        Mxfp8ExpertParam(c8, packed_scales = s8.view(torch.int8))


@needs_cuda
@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
def test_compiled_caller_traces_one_opaque_op(fmt):
    torch._dynamo.reset()
    quantize = quantize_experts_nvfp4 if fmt == "nvfp4" else quantize_experts_mxfp8
    param = quantize(_rtn_stack(4, 64, 96).cuda(), torch.bfloat16)
    counts = torch.tensor([5, 0, 7, 3], device = "cuda")
    x = torch.randn((15, 96), device = "cuda", dtype = torch.bfloat16)

    def step(inp):
        inp = inp.detach().requires_grad_(True)
        out = packed_expert_grouped_linear(inp, param, counts, True)
        out.float().square().sum().backward()
        return out.detach(), inp.grad

    def fwd(inp):
        return packed_expert_grouped_linear(inp, param, counts, True)

    want, want_dx = step(x)
    from torch._dynamo.utils import counters
    counters.clear()
    compiled = torch.compile(fwd, fullgraph = True)
    inp = x.detach().requires_grad_(True)
    out = compiled(inp)
    out.float().square().sum().backward()
    assert torch.equal(out.detach(), want) and torch.equal(inp.grad, want_dx)
    assert not counters["graph_break"]


def test_accelerate_style_rebuild_keeps_the_scales():
    # accelerate's set_module_tensor_to_device: param_cls(value.to(device), requires_grad = old.requires_grad)
    for quantize in (quantize_experts_nvfp4, quantize_experts_mxfp8):
        param = quantize(_rtn_stack(2, 32, 64), torch.bfloat16)
        rebuilt = type(param)(param.to("cpu"), requires_grad = False)
        assert type(rebuilt) is type(param) and torch.equal(rebuilt.dequantize(), param.dequantize())
    accelerate = pytest.importorskip("accelerate.utils")
    module = torch.nn.Module()
    module.w = quantize_experts_nvfp4(_rtn_stack(2, 32, 64), torch.bfloat16)
    want = module.w.dequantize()
    accelerate.set_module_tensor_to_device(module, "w", "cpu")
    assert type(module.w) is Nvfp4ExpertParam and torch.equal(module.w.dequantize(), want)
