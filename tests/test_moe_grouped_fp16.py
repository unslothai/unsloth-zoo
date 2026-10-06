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

"""moe_grouped_fp16 Triton grouped GEMMs against an fp64 oracle.

* Forward over ragged expert groups (empty experts, masked tails at 2880 and odd sizes),
  fp16 / fp32 operands, fp16 / fp32 outputs, the fp32-x cast and hi + lo split modes.
* An fp32 output keeps sums above 65504 finite; an fp16 output of the same sums does not
  (negative control). A transposed weight is caught (negative control).
* Backward dX from an fp32 dY against fp16 weights: per-row power-of-two scaling keeps
  tiny unscaled gradients that a plain fp16 cast flushes (negative control).
* Per-expert weight gradients (LoRA ranks 1 / 7 / 16 / 64), expert windows, no host sync.
* Every test runs on both backends: Triton, and the per-expert cuBLAS path used below sm80.
"""
import os

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level = True)
pytest.importorskip("triton")

from unsloth_zoo.temporary_patches import moe_grouped_fp16 as mg

DEV = torch.device("cuda", torch.cuda.current_device())
if not mg.fp16_grouped_available(DEV):
    pytest.skip(f"fp16 grouped GEMM unavailable: {mg.unavailable_reason()}", allow_module_level = True)

F16, F32, F64 = torch.float16, torch.float32, torch.float64


@pytest.fixture(autouse = True, params = ["triton", "cublas"])
def backend(request, monkeypatch):
    # cuBLAS is the default below sm80 (Triton's tl.dot is FMA-only on sm75); test both everywhere.
    monkeypatch.setenv("UNSLOTH_GPTOSS_FP16_GEMM", request.param)
    return request.param


def _counts(E, M, seed, empty = (1,)):
    g = torch.Generator().manual_seed(seed)
    w = torch.rand(E, generator = g) + 0.2
    for e in empty:
        w[e] = 0
    c = torch.floor(w / w.sum() * M).long()
    c[int(torch.argmax(w))] += M - int(c.sum())
    return c.to(DEV, torch.int32)


def _expert_of(counts):
    return torch.repeat_interleave(torch.arange(counts.numel(), device = DEV), counts.long())


def _oracle(x, w, counts, b_trans = True):
    # Per expert, in fp64 (gathering [M, N, K] weights would take tens of GB).
    x = x.to(F64)
    outs, start = [], 0
    for e, c in enumerate(counts.tolist()):
        W = w[e].to(F64)
        outs.append(x[start:start + c] @ (W.T if b_trans else W))
        start += c
    return torch.cat(outs)


def _rel(a, b):
    return float((a.to(F64) - b.to(F64)).norm() / (b.to(F64).norm() + 1e-300))


SHAPES = [(8, 2880, 2880, 300), (8, 53, 37, 97), (32, 2880, 64, 2048), (5, 16, 16, 3)]


@pytest.mark.parametrize("E,N,K,M", SHAPES)
@pytest.mark.parametrize("case", ["f16", "f32x_cast", "f32x_split", "f32"])
def test_forward_matches_fp64(E, N, K, M, case):
    g = torch.Generator(device = DEV).manual_seed(0)
    counts = _counts(E, M, 1, empty = (1,) if E > 2 else ())
    x = torch.randn(M, K, device = DEV, generator = g)
    w = torch.randn(E, N, K, device = DEV, generator = g) / K ** 0.5
    xd, wd, od, mode = {"f16": (F16, F16, F16, None), "f32x_cast": (F32, F16, F32, mg.A_CAST),
                        "f32x_split": (F32, F16, F32, mg.A_SPLIT), "f32": (F32, F32, F32, None)}[case]
    xq, wq = x.to(xd), w.to(wd)
    y = mg.grouped_gemm(xq, wq, counts, od, b_trans = True, a_mode = mode)
    assert y.dtype == od and y.shape == (M, N)
    # Against the oracle of the operands the kernel actually multiplies: fp32 accumulation error only.
    x_eff = xq.to(F16) if mode == mg.A_CAST else xq
    ref = _oracle(x_eff, wq, counts)
    out_eps = {F16: 2 ** -10, F32: 0}[od]
    # fp32 accumulation over K = 2880: ~sqrt(K) * 2^-24 relative.
    assert _rel(y, ref) < 1e-5 + out_eps, _rel(y, ref)
    if case == "f32x_split":
        # hi + lo carries x to ~2^-22: the remaining error is the fp16 weight's.
        assert _rel(y, _oracle(x, wq, counts)) < 1e-5
    if case == "f32x_cast":
        assert _rel(y, _oracle(x, wq, counts)) < 2e-3


def test_wrong_transpose_is_caught():
    # Negative control: a square stack read untransposed must fail the oracle comparison above.
    E, N, M = 4, 64, 128
    counts = _counts(E, M, 2, empty = ())
    x = torch.randn(M, N, device = DEV)
    w = torch.randn(E, N, N, device = DEV) / N ** 0.5
    y = mg.grouped_gemm(x.half(), w.half(), counts, F32, b_trans = False)
    assert _rel(y, _oracle(x.half(), w.half(), counts)) > 0.5
    assert _rel(y, _oracle(x.half(), w.half(), counts, b_trans = False)) < 1e-5


def test_fp32_output_keeps_overflowing_sums_finite():
    # gpt-oss down: |gated| <= ~56, K = 2880. Sums of fp16 products above 65504 stay finite in fp32.
    E, N, K, M = 4, 256, 2880, 512
    counts = _counts(E, M, 3, empty = ())
    x = (torch.rand(M, K, device = DEV) * 50 + 6)
    w = (torch.rand(E, N, K, device = DEV) + 0.5)
    ref = _oracle(x, w.half(), counts)
    assert float(ref.abs().min()) > 65504
    y = mg.grouped_gemm(x, w.half(), counts, F32, b_trans = True, a_mode = mg.A_SPLIT)
    # All-positive sums: tensor-core fp32 accumulation truncates (not RNE), a ~2e-5 bias here.
    assert bool(torch.isfinite(y).all()) and _rel(y, ref) < 1e-4
    # Negative control: the same GEMM stored in fp16 overflows.
    y16 = mg.grouped_gemm(x, w.half(), counts, F16, b_trans = True, a_mode = mg.A_SPLIT)
    assert not bool(torch.isfinite(y16).all())


@pytest.mark.parametrize("dy_mode", ["scale", "split"])
@pytest.mark.parametrize("magnitude", [1e-9, 1e-3, 3e4])
def test_backward_dx_from_fp32_dy(dy_mode, magnitude):
    # dY of the fp32 down output arrives unscaled (no GradScaler under the forced-fp32 rule).
    E, N, K, M = 8, 2880, 2880, 512
    counts = _counts(E, M, 4)
    w = (torch.randn(E, N, K, device = DEV) * 0.02).half()
    dy = torch.randn(M, N, device = DEV) * magnitude
    dy[::7] *= 1e-3   # rows at another scale: per-row scaling, not one global factor
    ref = _oracle(dy, w, counts, b_trans = False)
    dx = mg._bwd_dx(dy, w, counts, F32, dy_mode)
    assert dx.dtype == F32 and bool(torch.isfinite(dx).all())
    tol = 1e-5 if dy_mode == "split" else 2e-3
    assert _rel(dx, ref) < tol, _rel(dx, ref)
    # Rows at the small scale are as accurate as the others.
    assert _rel(dx[::7], ref[::7]) < tol
    if magnitude == 1e-9:
        # Negative control: a plain fp16 cast of dY flushes it to (nearly) zero.
        blind = mg.grouped_gemm(dy.half(), w, counts, F32, b_trans = False)
        assert _rel(blind, ref) > 0.5


@pytest.mark.parametrize("r", [1, 7, 16, 64])
@pytest.mark.parametrize("dtype", [F16, F32])
def test_wgrad_matches_fp64(r, dtype):
    E, M, K = 8, 700, 2880
    counts = _counts(E, M, 5, empty = (1, 6))
    gr = torch.randn(M, r, device = DEV).to(dtype)
    x = torch.randn(M, K, device = DEV).to(dtype)
    dw = mg.grouped_wgrad(gr, x, counts, F32)
    e = _expert_of(counts)
    ref = torch.stack([gr.to(F64)[e == i].T @ x.to(F64)[e == i] for i in range(E)])
    assert dw.shape == (E, r, K)
    assert torch.count_nonzero(dw[1]) == 0 and torch.count_nonzero(dw[6]) == 0
    assert _rel(dw, ref) < (1e-5 if dtype == F32 else 1e-5), _rel(dw, ref)
    # LoRA B orientation: [E, out, r] from grad [M, out] and xA [M, r].
    gb = torch.randn(M, 2880, device = DEV).to(dtype)
    xa = torch.randn(M, r, device = DEV).to(dtype)
    dB = mg.grouped_wgrad(gb, xa, counts, F32)
    refB = torch.stack([gb.to(F64)[e == i].T @ xa.to(F64)[e == i] for i in range(E)])
    assert dB.shape == (E, 2880, r) and _rel(dB, refB) < 1e-5


@pytest.mark.parametrize("r", [1, 7, 16, 64])
@pytest.mark.parametrize("dtype", [F16, F32])
def test_grouped_linear_autograd(r, dtype):
    E, M, K = 8, 333, 2880
    counts = _counts(E, M, 6)
    x = torch.randn(M, K, device = DEV).to(dtype).requires_grad_(True)
    A = (torch.randn(E, r, K, device = DEV) * 0.02).to(dtype).requires_grad_(True)
    y = mg.grouped_linear(x, A, counts)
    gy = torch.randn_like(y)
    y.backward(gy)
    x64, A64 = x.detach().to(F64).requires_grad_(True), A.detach().to(F64).requires_grad_(True)
    e = _expert_of(counts)
    y64 = torch.einsum("mk,mrk->mr", x64, A64[e])
    y64.backward(gy.to(F64))
    tol = 1e-5 if dtype == F32 else 2e-3
    assert _rel(y, y64) < tol and _rel(x.grad, x64.grad) < tol and _rel(A.grad, A64.grad) < tol
    assert x.grad.dtype == dtype and A.grad.dtype == dtype


class _Provider:
    def __init__(self, w, chunk):
        self.w, self.chunk, self.pin = w, chunk, False
        self.num_experts, self.N, self.K = w.shape
        self.dtype = w.dtype
        self.calls = 0

    def windows(self):
        return [(lo, min(lo + self.chunk, self.num_experts)) for lo in range(0, self.num_experts, self.chunk)]

    def __call__(self, lo, hi):
        self.calls += 1
        return self.w[lo:hi].clone()


@pytest.mark.parametrize("chunk", [8, 3, 1])
def test_frozen_linear_windows_and_bias(chunk):
    E, N, K, M = 8, 2880, 2880, 640
    counts = _counts(E, M, 7, empty = (2,))
    w = (torch.randn(E, N, K, device = DEV) * 0.02).half()
    bias = torch.randn(E, N, device = DEV)
    x = (torch.randn(M, K, device = DEV) * 10).requires_grad_(True)
    p = _Provider(w, chunk)
    y = mg.grouped_frozen_linear(x, counts, p, bias = bias, out_dtype = F32, x_mode = "split", dy_mode = "split")
    gy = torch.randn_like(y) * 1e-6
    y.backward(gy)
    e = _expert_of(counts)
    ref = _oracle(x.detach(), w, counts) + bias.to(F64)[e]
    assert _rel(y, ref) < 1e-5
    assert _rel(x.grad, _oracle(gy, w, counts, b_trans = False)) < 1e-5
    assert p.calls == 2 * len(p.windows())   # forward + backward rebuild per window
    if chunk != 8:
        p8 = _Provider(w, 8)
        x8 = x.detach().clone().requires_grad_(True)
        y8 = mg.grouped_frozen_linear(x8, counts, p8, bias = bias, out_dtype = F32, x_mode = "split", dy_mode = "split")
        y8.backward(gy)
        assert torch.equal(y, y8) and torch.equal(x.grad, x8.grad)


def test_no_host_sync(backend):
    if backend == "cublas":
        pytest.skip("the cuBLAS backend reads the row counts on the host once per layer")
    E, N, K, M, r = 8, 512, 256, 300, 16
    counts = _counts(E, M, 8)
    w = (torch.randn(E, N, K, device = DEV) * 0.05).half()
    x = torch.randn(M, K, device = DEV).requires_grad_(True)
    A = (torch.randn(E, r, K, device = DEV) * 0.05).half().requires_grad_(True)
    B = (torch.randn(E, N, r, device = DEV) * 0.05).float().requires_grad_(True)
    p = _Provider(w, 3)

    def step():
        y = mg.grouped_frozen_linear(x, counts, p, out_dtype = F32)
        y = y + mg.grouped_linear(mg.grouped_linear(x.half(), A, counts).float(), B, counts)
        y.sum().backward()

    step()
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        step()
    finally:
        torch.cuda.set_sync_debug_mode("default")
