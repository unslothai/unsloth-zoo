# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""Below sm_80 the stacked-expert fallback dequantizes chunks of experts (2 GiB by default) where the tile kernel
ran before: a chunk that does not fit must shrink, and below one expert use the fused kernel, never OOM. An OOM
inside the dequant kernel must not turn the kernel off for the process."""

import contextlib
import gc

import pytest
import torch

import unsloth_zoo.mxfp4_dequant as md
import unsloth_zoo.mxfp4_gemm as mg
import unsloth_zoo.mxfp4_stacked_experts as mse
from unsloth_zoo.mxfp4_dequant import Mxfp4ExpertParam, mxfp4_dequantize, mxfp4_dequantize_torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")


@pytest.fixture
def sm75(monkeypatch):
    """The sm_75 policy on whatever GPU runs this: tile GEMM off, GEMV (decode rows) on, real kernels."""
    device = torch.device("cuda", torch.cuda.current_device())
    monkeypatch.delenv("UNSLOTH_MXFP4_FUSED_GEMM", raising = False)
    monkeypatch.delenv("UNSLOTH_MXFP4_EXPERT_CHUNK_MB", raising = False)
    for dtype in (torch.float16, torch.bfloat16):
        if not mg.mxfp4_grouped_gemv_available(device, dtype):
            pytest.skip("the fused MXFP4 kernel does not self-check on this GPU")
        monkeypatch.setitem(mg._AVAILABLE_DTYPE, (str(device), dtype, "tiles"), False)
    monkeypatch.setattr(mse, "_CHUNK_BUDGET", {}, raising = False)
    # An OOM must not leak a disabled dequant kernel into later tests (main sets these on OOM).
    monkeypatch.setattr(md, "_KERNEL_FAILED", md._KERNEL_FAILED)
    monkeypatch.setattr(md, "_KERNEL_VERIFIED", dict(md._KERNEL_VERIFIED))
    return device


@contextlib.contextmanager
def _headroom(mib, device):
    """Let the caching allocator hand out at most ``mib`` MiB more than is allocated now."""
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    total = torch.cuda.get_device_properties(device).total_memory
    cap = torch.cuda.memory_allocated(device) + mib * 2**20
    torch.cuda.set_per_process_memory_fraction(cap / total, device)
    try:
        yield
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0, device)
        torch.cuda.empty_cache()


def _stack(E, K, N, device, seed = 0):
    # Logical (E, K, N) for ``inputs @ W_e``, stored transposed as blocks (E, N, K / 32, 16).
    g = torch.Generator(device = device).manual_seed(seed)
    blocks = torch.randint(0, 256, (E, N, K // 32, 16), dtype = torch.uint8, device = device, generator = g)
    scales = torch.randint(118, 128, (E, N, K // 32), dtype = torch.uint8, device = device, generator = g)
    return Mxfp4ExpertParam(blocks, mxfp4_scales = scales, mxfp4_dtype = torch.float16)


def _routed(E, rows, K, device, dtype, seed = 1):
    g = torch.Generator(device = device).manual_seed(seed)
    counts = torch.full((E,), rows // E, dtype = torch.int64, device = device)
    counts[: rows % E] += 1
    x = torch.randn(rows, K, dtype = dtype, device = device, generator = g)
    return x, counts, torch.cumsum(counts, 0).tolist()


def _reference(x, param, counts):
    w = mxfp4_dequantize_torch(param.data, param.mxfp4_scales, dtype = torch.float32, transpose = True)
    out, start = [], 0
    for e, c in enumerate(counts.tolist()):
        out.append(x[start : start + c].float() @ w[e])
        start += c
    return torch.cat(out)


E, K, N = 16, 2048, 2048  # 8 MiB fp16 per expert, the whole 128 MiB stack inside the 2 GiB default chunk


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_t4_expert_chunk_shrinks_to_fit_64mib_headroom(sm75, dtype):
    device = sm75
    param = _stack(E, K, N, device)
    x, counts, ends = _routed(E, 64, K, device, dtype)
    assert not mse._fused(x, param)  # 64 rows over 16 experts: the dequantize fallback on sm_75
    want = mse.mxfp4_grouped_linear(x, param, counts, ends)  # unconstrained: one 2 GiB-budget chunk
    with _headroom(64, device):
        got = mse.mxfp4_grouped_linear(x, param, counts, ends)
        torch.cuda.synchronize(device)
    assert torch.equal(got, want)  # per-expert products do not depend on the chunking
    ref = _reference(x, param, counts)
    torch.testing.assert_close(got.float(), ref, rtol = 2e-2, atol = 2e-2 * ref.abs().max().item())
    # The shrunk size is kept: the next call does not fail an allocation again.
    ooms = torch.cuda.memory_stats(device).get("num_ooms", 0)
    with _headroom(64, device):
        mse.mxfp4_grouped_linear(x, param, counts, ends)
    assert torch.cuda.memory_stats(device).get("num_ooms", 0) == ooms


def test_t4_no_room_for_one_expert_takes_the_fused_kernel(sm75):
    device = sm75
    param = _stack(E, K, N, device, seed = 2)
    x, counts, ends = _routed(E, 64, K, device, torch.float16)
    ref = _reference(x, param, counts)
    with _headroom(4, device):  # one expert decodes to 8 MiB
        got = mse.mxfp4_grouped_linear(x, param, counts, ends)
        torch.cuda.synchronize(device)
    torch.testing.assert_close(got.float(), ref, rtol = 2e-2, atol = 2e-2 * ref.abs().max().item())


def test_backward_chunks_shrink_too(sm75):
    device = sm75
    param = _stack(E, K, N, device, seed = 3)
    x, counts, ends = _routed(E, 64, K, device, torch.float16)
    x.requires_grad_(True)
    gy = torch.randn(64, N, dtype = torch.float16, device = device)
    mse.mxfp4_grouped_linear(x, param, counts, ends).backward(gy)
    want = x.grad.clone()
    x.grad = None
    out = mse.mxfp4_grouped_linear(x, param, counts, ends)
    with _headroom(64, device):
        out.backward(gy)
        torch.cuda.synchronize(device)
    assert torch.equal(x.grad, want)


def test_oom_in_the_dequant_kernel_keeps_the_kernel(monkeypatch):
    device = torch.device("cuda", torch.cuda.current_device())
    monkeypatch.setattr(md, "_KERNEL_FAILED", False)
    monkeypatch.setattr(md, "_KERNEL_VERIFIED", dict(md._KERNEL_VERIFIED))
    blocks = torch.zeros(8, 4096, 128, 16, dtype = torch.uint8, device = device)  # 256 MiB of bf16 out
    scales = torch.full((8, 4096, 128), 127, dtype = torch.uint8, device = device)
    mxfp4_dequantize(blocks[:1], scales[:1], dtype = torch.bfloat16)  # verified + compiled before the cap
    if md._KERNEL_FAILED or not md._KERNEL_VERIFIED.get((str(device), torch.bfloat16, False)):
        pytest.skip("the fused dequant kernel does not run on this GPU")
    with _headroom(16, device), pytest.raises(torch.cuda.OutOfMemoryError):
        mxfp4_dequantize(blocks, scales, dtype = torch.bfloat16)
    assert not md._KERNEL_FAILED
