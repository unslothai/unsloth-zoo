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

"""Fused linear CE / log-probs (torch.library ops, Triton row kernels on GPU) against an fp64 oracle, the chunked CE
it replaces, and torch.compile(fullgraph=True)."""

import os

import numpy
import pytest
import torch

import unsloth_zoo.fused_losses.fused_linear as fl
from unsloth_zoo.fused_losses.fused_linear import fused_linear_cross_entropy, fused_linear_logprobs

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
# Tests of the Triton kernels themselves (AMD's Windows wheels, for one, ship no Triton: the torch path runs there).
TRITON = pytest.mark.skipif(not (torch.cuda.is_available() and fl._HAS_TRITON), reason = "needs CUDA and Triton")


@pytest.fixture(autouse = True)
def _triton_never_fell_back():
    # A silent torch fallback would pass every accuracy test; no test may leave a device marked broken.
    yield
    broken = set(fl._TRITON_BROKEN)
    fl._TRITON_BROKEN.clear()
    assert not broken, f"Triton fell back to torch on {broken}"


def _half_ok(device):
    return device == "cuda" and torch.cuda.get_device_capability()[0] >= 8


def _dtypes(device):
    out = [torch.float32]
    if device == "cuda":
        # fp16 everywhere (the T4 path), bf16 where the GPU has it.
        out.append(torch.float16)
        if _half_ok(device):
            out.append(torch.bfloat16)
    return out


def _tol(dtype):
    return {torch.float64: 1e-9, torch.float32: 2e-4, torch.bfloat16: 3e-2, torch.float16: 1e-2}[dtype]


def _transform(z, mul, div, cap):
    if mul:
        z = z * mul
    if div:
        z = z / div
    if cap:
        z = cap * torch.tanh(z / cap)
    return z


def _close(a, b, tol, what):
    a, b = a.double(), b.double()
    scale = max(float(b.abs().max()), 1.0)
    err = float((a - b).abs().max())
    assert err <= tol * scale, f"{what}: max err {err:.3g} > {tol * scale:.3g}"


def _make(device, dtype, N, H, V, seed = 0, train_head = False, with_bias = False):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(N, H, generator = g).to(device, dtype).requires_grad_(True)
    w = (torch.randn(V, H, generator = g) * H**-0.5 * 2).to(device, dtype).requires_grad_(train_head)
    b = (torch.randn(V, generator = g) * 0.1).to(device, dtype).requires_grad_(train_head) if with_bias else None
    t = torch.randint(0, V, (N,), generator = g).to(device)
    return h, w, b, t


def _leaves(*xs):
    return [x for x in xs if x is not None and x.requires_grad]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("train_head,with_bias", [(False, False), (True, True), (False, True)])
@pytest.mark.parametrize("mul,div,cap,eps", [(0.0, 0.0, 0.0, 0.0), (1.5, 0.0, 30.0, 0.0), (0.0, 2.0, 0.0, 0.1)])
def test_ce_matches_fp64_oracle(device, train_head, with_bias, mul, div, cap, eps):
    for dtype in _dtypes(device):
        N, H, V = 61, 32, 4099 if device == "cuda" else 257
        h, w, b, t = _make(device, dtype, N, H, V, train_head = train_head, with_bias = with_bias)
        t[::4] = -100
        lb = torch.randn(V, device = device) * 0.05 if eps else None
        loss = fused_linear_cross_entropy(
            h, w, b, t, divisor = 17.0, scaling = 4.0, logit_scale_multiply = mul, logit_scale_divide = div,
            logit_softcapping = cap, label_smoothing = eps, logits_bias = lb, rows_per_chunk = 7,
        )
        leaves = _leaves(h, w, b)
        grads = torch.autograd.grad(loss * 3.0, leaves)
        hd, wd = h.detach().double().requires_grad_(True), w.detach().double().requires_grad_(train_head)
        bd = b.detach().double().requires_grad_(b.requires_grad) if b is not None else None
        z = hd @ wd.t() + (bd if bd is not None else 0) + (lb.double() if lb is not None else 0)
        ref = torch.nn.functional.cross_entropy(
            _transform(z, mul, div, cap), t, ignore_index = -100, reduction = "sum", label_smoothing = eps,
        ) / 17.0
        refs = torch.autograd.grad(ref * 3.0, _leaves(hd, wd, bd))
        _close(loss, ref, _tol(dtype), f"loss {dtype}")
        for x, y in zip(grads, refs):
            assert x.dtype == dtype
            _close(x, y, _tol(dtype), f"grad {dtype}")


def test_ce_matches_chunked_ce():
    from unsloth_zoo.fused_losses.chunked_logprob import unsloth_chunked_ce_loss

    h, w, b, t = _make("cpu", torch.float32, 50, 16, 97, with_bias = True, train_head = True)
    t[:10] = -100
    a = fused_linear_cross_entropy(h, w, b, t, divisor = torch.tensor(23.0), logit_softcapping = 5.0)
    ga = torch.autograd.grad(a, (h, w, b))
    c = unsloth_chunked_ce_loss(h, w, b, t, n_items = torch.tensor(23.0), shift_labels = False, logit_softcapping = 5.0)
    gc = torch.autograd.grad(c, (h, w, b))
    _close(a, c, 1e-5, "loss")
    for x, y in zip(ga, gc):
        _close(x, y, 1e-5, "grad")


@pytest.mark.parametrize("device", DEVICES)
def test_ce_incoming_gradient_scales_in_fp32(device):
    # The incoming gradient multiplies in fp32: half * 0-dim fp32 rounds the factor to half (0.2% off).
    for dtype in _dtypes(device):
        h, w, b, t = _make(device, dtype, 50, 32, 131, train_head = True, with_bias = True)
        loss = fused_linear_cross_entropy(h, w, b, t, divisor = 50)
        base = torch.autograd.grad(loss, (h, w, b))
        loss = fused_linear_cross_entropy(h, w, b, t, divisor = 50)
        go = torch.tensor(0.37, device = device)
        scaled = torch.autograd.grad(loss, (h, w, b), go)
        g32 = torch.tensor(0.37, dtype = torch.float32)
        for x, y in zip(scaled, base):
            want = (y.float() * g32.to(device)).to(dtype)
            # ROCm rounds the exact product once (a tie of the fp32 product can land 1 ulp the other way); numpy
            # rounds float64 -> float16 directly, torch goes through float32.
            exact = y.double().cpu() * g32.double()
            once = torch.from_numpy(exact.numpy().astype(numpy.float16)).to(device = device).to(dtype) \
                if dtype == torch.float16 else exact.to(dtype).to(device)
            assert ((x == want) | (x == once)).all(), f"{dtype}: incoming gradient not applied in fp32"


def test_ce_all_masked_and_no_grad():
    h, w, b, t = _make("cpu", torch.float32, 9, 8, 33)
    t[:] = -100
    loss = fused_linear_cross_entropy(h, w, None, t, divisor = 0)
    assert float(loss) == 0.0
    (g,) = torch.autograd.grad(loss, (h,))
    assert torch.count_nonzero(g) == 0
    t[:] = 3
    w.requires_grad_(True)
    with torch.no_grad():
        before = dict(fl.CALLS)
        loss = fused_linear_cross_entropy(h, w, None, t, divisor = 9)
    assert loss.grad_fn is None and fl.CALLS["ce"] == before["ce"] + 1


def _lp_oracle(h, w, b, t, mask, T, mul, div, cap):
    z = h.double() @ w.double().t() + (b.double() if b is not None else 0)
    z = _transform(z, mul, div, cap)
    lp = (z / T).log_softmax(-1)
    tc = t.clamp(0, w.shape[0] - 1)
    m = mask.double()
    return (
        lp.gather(-1, tc[:, None])[:, 0] * m,
        -(lp.exp() * lp).sum(-1) * m,
        z.mean(-1) * m,
        torch.logsumexp(2 * lp, -1) * m,
        ((z / T).argmax(-1) == tc) & mask,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("train_head,with_bias,ent_grad", [
    (False, False, False), (False, False, True), (True, False, False), (False, True, False), (True, True, True),
])
@pytest.mark.parametrize("T,mul,div,cap", [(1.0, 0.0, 0.0, 0.0), (0.7, 0.0, 0.0, 2.0), (1.3, 2.0, 3.0, 0.0)])
def test_logprobs_match_fp64_oracle(device, train_head, with_bias, ent_grad, T, mul, div, cap):
    for dtype in _dtypes(device):
        N, H, V = 45, 32, 4103 if device == "cuda" else 131
        h, w, b, t = _make(device, dtype, N, H, V, train_head = train_head, with_bias = with_bias)
        mask = torch.ones(N, dtype = torch.bool, device = device)
        mask[::3] = False
        t[::3] = -100  # never read under the mask
        out = fused_linear_logprobs(
            h, w, t, bias = b, mask = mask, temperature = T, logit_scale_multiply = mul, logit_scale_divide = div,
            logit_softcapping = cap, return_entropy = True, return_mean_logits = True,
            return_log_sum_sq_probs = True, return_is_top1 = True, rows_per_chunk = 8,
            entropy_requires_grad = ent_grad,
        )
        hd = h.detach().double().requires_grad_(True)
        wd = w.detach().double().requires_grad_(train_head)
        bd = b.detach().double().requires_grad_(b.requires_grad) if b is not None else None
        ref = _lp_oracle(hd, wd, bd, t, mask, T, mul, div, cap)
        tol = _tol(dtype)
        for name, x, y in zip(("logp", "entropy", "mean", "lssp"), out[:4], ref[:4]):
            assert x.dtype == torch.float32 and x.shape == (N,)
            _close(x, y, tol, f"{name} {dtype}")
        if dtype == torch.float32:
            assert torch.equal(out[4], ref[4])
        assert out[2].grad_fn is None and out[4].dtype == torch.bool
        g = torch.randn(N, device = device, dtype = torch.float64)
        ge = torch.randn(N, device = device, dtype = torch.float64)
        obj = (out[0].double() * g).sum() + ((out[1].double() * ge).sum() if ent_grad else 0)
        robj = (ref[0] * g).sum() + ((ref[1] * ge).sum() if ent_grad else 0)
        grads = torch.autograd.grad(obj, _leaves(h, w, b))
        refs = torch.autograd.grad(robj, _leaves(hd, wd, bd))
        for x, y in zip(grads, refs):
            assert x.dtype == dtype
            _close(x, y, tol, f"grad {dtype}")


@pytest.mark.parametrize("device", DEVICES)
def test_logprobs_trainable_head_bias_over_vocab_slices(monkeypatch, device):
    # A small budget splits the recompute backward into several vocab slices; each slice reads its own bias columns.
    N, H, V = 37, 32, 4103 if device == "cuda" else 131
    slice_v = 1000 if device == "cuda" else 40
    monkeypatch.setattr(fl, "_budget_bytes", lambda device: slice_v * 16 * H)
    for dtype in _dtypes(device):
        h, w, b, t = _make(device, dtype, N, H, V, train_head = True, with_bias = True)
        torch_calls = fl.CALLS["torch"]
        lp = fused_linear_logprobs(h, w, t, bias = b, temperature = 0.9, logit_softcapping = 3.0)
        hd = h.detach().double().requires_grad_(True)
        wd = w.detach().double().requires_grad_(True)
        bd = b.detach().double().requires_grad_(True)
        ref = _lp_oracle(hd, wd, bd, t, torch.ones(N, dtype = torch.bool, device = device), 0.9, 0.0, 0.0, 3.0)[0]
        g = torch.randn(N, device = device, dtype = torch.float64)
        grads = torch.autograd.grad((lp.double() * g).sum(), (h, w, b))
        refs = torch.autograd.grad((ref * g).sum(), (hd, wd, bd))
        for name, x, y in zip(("hidden", "weight", "bias"), grads, refs):
            _close(x, y, _tol(dtype), f"{name} grad {dtype}")
        if device == "cuda" and fl._HAS_TRITON:
            assert fl.CALLS["torch"] == torch_calls


@TRITON
@pytest.mark.parametrize("split_cols", [0, 512, 1000])
@pytest.mark.parametrize("mode", ["1", "2"])
def test_split_statistics_match_oracle(monkeypatch, split_cols, mode):
    # Split edges off the block and a top-1 tie across a split edge (first index wins, as torch.argmax).
    monkeypatch.setattr(fl, "_SPLIT_COLS", split_cols)
    monkeypatch.setattr(fl, "_ACCURACY", mode)
    N, H, V = 45, 32, 4103
    for dtype in _dtypes("cuda"):
        h, w, b, t = _make("cuda", dtype, N, H, V, with_bias = True)
        with torch.no_grad():
            w[1023] = w[1024] = 0.5  # rows 0 and 1 (hidden all ones) tie at columns 1023 / 1024
            if b is not None:
                b[1023] = b[1024] = 0.0
        h.data[0] = 1.0
        t[0] = 1024  # the later of the tied columns: not the top-1
        t[1] = 1023
        h.data[1] = 1.0
        mask = torch.ones(N, dtype = torch.bool, device = "cuda")
        before = fl.CALLS["torch"]
        out = fused_linear_logprobs(
            h, w, t, bias = b, mask = mask, temperature = 0.8, logit_softcapping = 0.0,
            return_entropy = True, return_mean_logits = True, return_log_sum_sq_probs = True, return_is_top1 = True,
            rows_per_chunk = 16,
        )
        assert fl.CALLS["torch"] == before
        ref = _lp_oracle(h.detach().double(), w.detach().double(), b.detach().double() if b is not None else None,
                         t, mask, 0.8, 0.0, 0.0, 0.0)
        tol = _tol(dtype)
        for name, x, y in zip(("logp", "entropy", "mean", "lssp"), out[:4], ref[:4]):
            _close(x, y, tol, f"{name} {dtype} split {split_cols}")
        assert not bool(out[4][0]) and bool(out[4][1]), "a tie must go to the first column"
        if dtype == torch.float32:
            assert torch.equal(out[4], ref[4])
        # CE with label smoothing reads the merged lse and logit sum.
        loss = fused_linear_cross_entropy(h, w, b, t, divisor = N, label_smoothing = 0.1, rows_per_chunk = 16)
        z = h.detach().double() @ w.detach().double().t() + (b.detach().double() if b is not None else 0)
        want = torch.nn.functional.cross_entropy(z, t, label_smoothing = 0.1)
        _close(loss, want, tol, f"ce {dtype} split {split_cols}")


@TRITON
@pytest.mark.parametrize("train_head,with_bias,ent_grad", [(False, False, True), (True, True, False), (True, True, True)])
@pytest.mark.parametrize("mode,recompute_fp32", [("1", False), ("1", True), ("2", False)])
def test_recompute_forward_tiles_match_oracle(monkeypatch, train_head, with_bias, ent_grad, mode, recompute_fp32):
    # Small tiles / splits / chunks: ragged last tile, compacted rows, top-1 tie across two tiles.
    monkeypatch.setattr(fl, "_SPLIT_COLS", 256)
    monkeypatch.setattr(fl, "_BWD_ROWS", 16)
    monkeypatch.setattr(fl, "_BWD_TILE_BYTES", 16 * 600 * 4)
    monkeypatch.setattr(fl, "_ACCURACY", mode)
    monkeypatch.setattr(fl, "_RECOMPUTE_FP32", recompute_fp32)
    N, H, V = 45, 32, 4103
    for dtype in _dtypes("cuda"):
        h, w, b, t = _make("cuda", dtype, N, H, V, train_head = train_head, with_bias = with_bias)
        with torch.no_grad():
            w[1023] = w[1024] = 0.5  # tiles of 512 (fp32 logits) or 1024 columns: the tie straddles two tiles
            if b is not None:
                b[1023] = b[1024] = 0.0
            h[0] = h[1] = 1.0
        t[0], t[1] = 1024, 1023
        mask = torch.ones(N, dtype = torch.bool, device = "cuda")
        mask[5::4] = False
        before = dict(fl.CALLS)
        out = fused_linear_logprobs(
            h, w, t, bias = b, mask = mask, temperature = 0.8, logit_softcapping = 4.0,
            return_entropy = True, return_mean_logits = True, return_log_sum_sq_probs = True, return_is_top1 = True,
            entropy_requires_grad = ent_grad,
        )
        assert fl.CALLS["tiled"] == before["tiled"] + 1 and fl.CALLS["torch"] == before["torch"]
        hd = h.detach().double().requires_grad_(True)
        wd = w.detach().double().requires_grad_(train_head)
        bd = b.detach().double().requires_grad_(train_head) if b is not None else None
        ref = _lp_oracle(hd, wd, bd, t, mask, 0.8, 0.0, 0.0, 4.0)
        tol = _tol(dtype)
        for name, x, y in zip(("logp", "entropy", "mean", "lssp"), out[:4], ref[:4]):
            _close(x, y, tol, f"{name} {dtype}")
        assert not bool(out[4][0]) and bool(out[4][1]), "a tie must go to the first column"
        if dtype == torch.float32:
            assert torch.equal(out[4], ref[4])
        g = torch.randn(N, device = "cuda", dtype = torch.float64)
        ge = torch.randn(N, device = "cuda", dtype = torch.float64)
        obj = (out[0].double() * g).sum() + ((out[1].double() * ge).sum() if ent_grad else 0)
        robj = (ref[0] * g).sum() + ((ref[1] * ge).sum() if ent_grad else 0)
        for x, y in zip(torch.autograd.grad(obj, _leaves(h, w, b)), torch.autograd.grad(robj, _leaves(hd, wd, bd))):
            _close(x, y, tol, f"grad {dtype}")


@pytest.mark.parametrize("device", DEVICES)
def test_logprobs_bit_identical_with_and_without_grad(device):
    # Policy and reference on the same weights must agree exactly (on-policy ratio starts at 1).
    for dtype in _dtypes(device):
        V = 4103 if device == "cuda" else 131
        h, w, b, t = _make(device, dtype, 300, 64, V)
        for kwargs in ({}, dict(return_entropy = True, entropy_requires_grad = False), dict(rows_per_chunk = 128)):
            with torch.no_grad():
                ref = fused_linear_logprobs(h, w, t, **kwargs)
            out = fused_linear_logprobs(h, w, t, **kwargs)
            ref = ref if isinstance(ref, tuple) else (ref,)
            out = out if isinstance(out, tuple) else (out,)
            assert out[0].requires_grad
            for x, y in zip(out, ref):
                assert torch.equal(x.detach(), y), f"{dtype} {kwargs}: grad / no-grad differ"


def test_logits_mode_per_backend(monkeypatch):
    # fp32 logits only where fp32-output GEMM is fast (not ROCm / torch < 2.8); exact-target mode is opt-in.
    dev = torch.device("cuda", 0)
    monkeypatch.setattr(fl, "_fp32_out_ok", lambda device, dtype = None: True)
    for hip, acc, want in ((False, "1", (True, False)), (True, "1", (False, False)), (True, "2", (False, True)),
                           (False, "2", (False, True)), (False, "0", (False, False))):
        monkeypatch.setattr(fl, "_IS_HIP", hip)
        monkeypatch.setattr(fl, "_ACCURACY", acc)
        assert fl._logits_mode(dev) == want, (hip, acc)
    monkeypatch.setattr(fl, "_IS_HIP", False)
    monkeypatch.setattr(fl, "_ACCURACY", "1")
    monkeypatch.setattr(fl, "_fp32_out_ok", lambda device, dtype = None: False)
    assert fl._logits_mode(dev) == (False, False)
    assert fl._logits_mode(torch.device("cpu")) == (False, False)


def test_default_rows_follow_the_vocabulary(monkeypatch):
    # >= 512 rows (fewer starved the K = vocab GEMM); fixed 2 GB budget so a shared card's free memory is irrelevant.
    monkeypatch.setattr(fl, "_budget_bytes", lambda device = None: 2 * 1024**3)
    for V, want in ((32001, 2048), (151936, 512), (262144, 512)):
        assert fl.fused_linear_rows(8192, 2560, V) == want, V


def test_max_rows_caps_and_rows_sets_the_chunk():
    auto = fl._chunk_rows(0, 100000, 4096, 32000, 4, None)
    assert auto > 8
    assert fl._chunk_rows(-8, 100000, 4096, 32000, 4, None) == 8
    assert fl._chunk_rows(-10**9, 100000, 4096, 32000, 4, None) == auto  # a cap never raises the chunk
    assert fl._chunk_rows(10**6, 100000, 4096, 32000, 4, None) == 10**6


def test_entropy_as_mask_keeps_the_no_recompute_backward():
    # GRPO's top_entropy_quantile mask reads entropies without a gradient: the frozen head keeps the u path.
    h, w, b, t = _make("cpu", torch.float32, 23, 16, 97)
    before = fl.CALLS["logprobs_bwd"]
    lp, ent = fused_linear_logprobs(h, w, t, return_entropy = True, entropy_requires_grad = False)
    assert not ent.requires_grad
    lp.sum().backward()
    assert fl.CALLS["logprobs_bwd"] == before and h.grad is not None


def test_logprobs_frozen_head_skips_recompute():
    h, w, b, t = _make("cpu", torch.float32, 20, 8, 40)
    before = dict(fl.CALLS)
    lp = fused_linear_logprobs(h, w, t)
    lp.sum().backward()
    assert fl.CALLS["logprobs"] == before["logprobs"] + 1
    assert fl.CALLS["logprobs_bwd"] == before["logprobs_bwd"]
    w.requires_grad_(True)
    fused_linear_logprobs(h, w, t).sum().backward()
    assert fl.CALLS["logprobs_bwd"] == before["logprobs_bwd"] + 1


def test_logprobs_all_masked_connected_zero():
    h, w, b, t = _make("cpu", torch.float32, 6, 8, 40)
    mask = torch.zeros(6, dtype = torch.bool)
    lp = fused_linear_logprobs(h, w, t, mask = mask)
    assert torch.count_nonzero(lp) == 0 and lp.requires_grad
    (g,) = torch.autograd.grad(lp.sum(), (h,))
    assert torch.count_nonzero(g) == 0


def test_logprobs_lead_shape_and_out_of_range_targets():
    h = torch.randn(2, 5, 8, requires_grad = True)
    w = torch.randn(30, 8)
    t = torch.randint(0, 30, (2, 5))
    t[0, 0] = -100
    t[1, 1] = 30
    lp = fused_linear_logprobs(h, w, t)
    assert lp.shape == (2, 5) and lp[0, 0] == 0 and lp[1, 1] == 0
    ref = (h @ w.t()).log_softmax(-1).gather(-1, t.clamp(0, 29)[..., None])[..., 0]
    keep = (t >= 0) & (t < 30)
    _close(lp[keep], ref[keep], 1e-5, "logp")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["ce", "lp_frozen", "lp_trainable"])
def test_fullgraph_compile_no_breaks_no_recompile(kind, device):
    if device == "cuda" and not fl._HAS_TRITON:
        pytest.skip("Inductor compiles GPU code through Triton")
    import torch._dynamo
    from torch._dynamo.testing import CompileCounterWithBackend
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()
    cnt = CompileCounterWithBackend("inductor" if device == "cuda" else "aot_eager")
    N, H, V = 24, 8, 50
    w = torch.randn(V, H, device = device, requires_grad = kind == "lp_trainable")

    def step(h, t, mask):
        if kind == "ce":
            loss = fused_linear_cross_entropy(h, w, None, t, divisor = (t != -100).sum(), logit_softcapping = 4.0)
        else:
            lp, ent = fused_linear_logprobs(h, w, t, mask = mask, temperature = 0.9, return_entropy = True,
                                            entropy_requires_grad = False)
            loss = -(lp * mask).sum() / mask.sum().clamp(min = 1) + 0.0 * ent.sum()
        return loss

    compiled = torch.compile(step, backend = cnt, fullgraph = True)

    def fn(h, t, mask):
        # Forward compiled with fullgraph=True; backward runs the AOT-compiled backward graph.
        loss = compiled(h, t, mask)
        loss.backward()
        return loss.detach()
    results = []
    for n_valid in (17, 5):
        h = torch.randn(N, H, device = device, requires_grad = True)
        t = torch.randint(0, V, (N,), device = device)
        mask = torch.arange(N, device = device) < n_valid
        t[~mask] = -100
        loss = fn(h, t, mask)
        eager_h = h.detach().clone().requires_grad_(True)
        if w.grad is not None:
            compiled_wg = w.grad.clone()
            w.grad = None
        eager = step(eager_h, t, mask)
        eager.backward()
        _close(loss, eager, 1e-5, "compiled loss")
        _close(h.grad, eager_h.grad, 1e-5, "compiled grad")
        if kind == "lp_trainable":
            _close(compiled_wg, w.grad, 1e-5, "compiled head grad")
            w.grad = None
        results.append(loss)
    assert sum(counters["graph_break"].values()) == 0, dict(counters["graph_break"])
    assert cnt.frame_count == 1, f"recompiled: {cnt.frame_count} frames"


def test_opcheck():
    h, w, b, t = _make("cpu", torch.float32, 10, 8, 33, with_bias = True, train_head = True)
    t[0] = -100
    f32 = torch.ones(())
    torch.library.opcheck(
        torch.ops.unsloth_zoo.fused_linear_ce.default,
        (h, w, b, t, f32 * 9, f32, -100, 0.0, 0.0, 3.0, 0.0, None, torch.float32, True, True, True, False, 4),
    )
    mask = t != -100
    torch.library.opcheck(
        getattr(torch.ops.unsloth_zoo, fl._LP_OP).default,
        (h, w, b, t.clamp(min = 0), mask, 0.8, 0.0, 0.0, 0.0, torch.float32, True, True, True, True, False,
         True, True, True, 4, False),
    )
    # Frozen head with a lazy entropy: forward-gradient outputs, entropy gradient recomputed in backward.
    hf = h.detach().clone().requires_grad_(True)
    torch.library.opcheck(
        getattr(torch.ops.unsloth_zoo, fl._LP_OP).default,
        (hf, w.detach(), None, t.clamp(min = 0), mask, 0.8, 0.0, 0.0, 0.0, torch.float32, True, False, False, False,
         True, True, False, False, 4, True),
    )


@TRITON
@pytest.mark.parametrize("V", [4096, 4096 * 3 + 5, 151936])
@pytest.mark.parametrize("mode", ["0", "1", "2"])
def test_gpu_accuracy_modes_and_inplace(monkeypatch, V, mode):
    """Every accuracy mode (fp32 logits + in-place half gradient, half logits + exact target, plain half), with
    several vocab blocks per row so the in-place gradient overlaps logits other warps still read."""
    monkeypatch.setattr(fl, "_ACCURACY", mode)
    dtype = torch.bfloat16 if _half_ok("cuda") else torch.float16
    N, H = 300, 64
    h, w, b, t = _make("cuda", dtype, N, H, V, seed = 1)
    before = fl.CALLS["triton"]
    loss = fused_linear_cross_entropy(h, w, None, t, divisor = N, rows_per_chunk = 128)
    (gh,) = torch.autograd.grad(loss, (h,))
    lp = fused_linear_logprobs(h, w, t, rows_per_chunk = 128)
    (gl,) = torch.autograd.grad(lp.sum(), (h,))
    assert fl.CALLS["triton"] > before
    hd = h.detach().double().requires_grad_(True)
    z = hd @ w.double().t()
    ref = torch.nn.functional.cross_entropy(z, t, reduction = "sum") / N
    (rgh,) = torch.autograd.grad(ref, (hd,))
    _close(loss, ref, 2e-2, "loss")
    _close(gh, rgh, 3e-2, "ce grad")
    _close(-lp.double().sum() / N, ref, 2e-2, "logp")
    _close(-gl.double() / N, rgh, 3e-2, "logp grad")


@CUDA
def test_gpu_exact_target_keeps_half_gradients(monkeypatch):
    # Exact-target mode changes only values; gradients equal plain half mode bit for bit.
    dtype = torch.bfloat16 if _half_ok("cuda") else torch.float16
    N, H, V = 300, 64, 4103
    h, w, b, t = _make("cuda", dtype, N, H, V, seed = 2)
    up = torch.randn(N, device = "cuda")
    got = {}
    for mode in ("0", "2"):
        monkeypatch.setattr(fl, "_ACCURACY", mode)
        lp = fused_linear_logprobs(h, w, t, rows_per_chunk = 128)
        ce = fused_linear_cross_entropy(h, w, None, t, divisor = N, rows_per_chunk = 128)
        got[mode] = (lp.detach(), torch.autograd.grad((lp * up).sum(), h)[0], torch.autograd.grad(ce, h)[0])
    assert torch.equal(got["0"][1], got["2"][1]) and torch.equal(got["0"][2], got["2"][2])
    ref = (h.detach().double() @ w.double().t()).log_softmax(-1).gather(-1, t[:, None])[:, 0]
    assert (got["2"][0].double() - ref).abs().mean() <= (got["0"][0].double() - ref).abs().mean()


@CUDA
def test_gpu_torch_fallback_matches_triton(monkeypatch):
    dtype = torch.bfloat16 if _half_ok("cuda") else torch.float16
    h, w, b, t = _make("cuda", dtype, 200, 64, 5001, with_bias = True)
    outs = []
    for on in (True, False):
        monkeypatch.setattr(fl, "_TRITON", on)
        lp, ent = fused_linear_logprobs(h, w, t, bias = b, temperature = 0.8, logit_softcapping = 3.0,
                                        return_entropy = True, rows_per_chunk = 64)
        (g,) = torch.autograd.grad((lp + 0.1 * ent).sum(), (h,))
        outs.append((lp, ent, g))
    for x, y in zip(*outs):
        _close(x, y, 2e-2, "triton vs torch")


@CUDA
def test_gpu_grad_on_other_device_returns_home():
    if torch.cuda.device_count() < 1:
        pytest.skip("needs CUDA")
    h, w, b, t = _make("cpu", torch.float32, 12, 8, 40)
    w = w.cuda()
    lp = fused_linear_logprobs(h, w, t.cuda())
    assert lp.device == h.device
    (g,) = torch.autograd.grad(lp.sum(), (h,))
    assert g.device == h.device


@TRITON
def test_one_transient_triton_failure_is_retried(monkeypatch):
    # One launch failure is retried; a persistent one still falls back.
    real = fl._launch
    fails = {"left": 1}

    def flaky(device, fn):
        if fails["left"] > 0:
            fails["left"] -= 1
            raise RuntimeError("simulated transient Triton failure")
        return real(device, fn)

    monkeypatch.setattr(fl, "_launch", flaky)
    h, w, b, t = _make("cuda", torch.float32, 32, 16, 257)
    before = fl.CALLS["torch"]
    lp = fused_linear_logprobs(h, w, t)
    assert not fl._TRITON_BROKEN and fl.CALLS["torch"] == before
    ref = fused_linear_logprobs(h.detach(), w, t)
    torch.testing.assert_close(lp.detach(), ref)
    fails["left"] = 10**9
    with pytest.warns(UserWarning, match = "Triton kernel failed"):
        lp2 = fused_linear_logprobs(h.detach(), w, t)
    assert h.device in fl._TRITON_BROKEN or torch.device("cuda", torch.cuda.current_device()) in fl._TRITON_BROKEN
    torch.testing.assert_close(lp2, ref, rtol = 1e-4, atol = 1e-5)
    fl._TRITON_BROKEN.clear()


@CUDA
def test_bf16_on_pre_ampere_keeps_triton(monkeypatch):
    # T4 lacks bf16-in / fp32-out GEMM: the probe is per dtype, so bf16 takes half logits instead of raising.
    real = torch.mm

    def mm(a, b, *args, out_dtype = None, **kw):
        if out_dtype is not None and a.dtype == torch.bfloat16:
            raise RuntimeError("gemm input type at::BFloat16 and output type float is only supported for CUDA devices "
                               "with compute capability 8.0 or higher")
        return real(a, b, *args, **kw) if out_dtype is None else real(a, b, *args, out_dtype = out_dtype, **kw)

    monkeypatch.setattr(torch, "mm", mm)
    fl._fp32_out_ok.cache_clear()
    try:
        h, w, b, t = _make("cuda", torch.bfloat16, 64, 32, 1031)
        lp = fused_linear_logprobs(h, w, t)
        lp.sum().backward()
        assert torch.isfinite(lp).all() and torch.isfinite(h.grad).all()
        assert not fl._TRITON_BROKEN and fl._logits_mode(h.device, torch.bfloat16) == (False, False)
        # fp16 keeps fp32 logits where torch has the fp32-output GEMM at all (2.8+); ROCm takes plain half logits.
        has_out_dtype = tuple(int(x) for x in torch.__version__.split(".")[:2]) >= (2, 8)
        assert fl._logits_mode(h.device, torch.float16) == (has_out_dtype and not fl._IS_HIP, False)
    finally:
        fl._fp32_out_ok.cache_clear()


def test_opcheck_with_disabled_outputs():
    # Disabled optional outputs are distinct empty tensors in the real op, so the fakes must not alias them either.
    h, w, b, t = _make("cpu", torch.float32, 10, 8, 33, with_bias = True)
    f32 = torch.ones(())
    torch.library.opcheck(
        torch.ops.unsloth_zoo.fused_linear_ce.default,
        (h, w, b, t, f32 * 9, f32, -100, 0.0, 0.0, 0.0, 0.0, None, torch.float32, True, False, False, False, 4),
        test_utils = ("test_faketensor",),
    )
    torch.library.opcheck(
        getattr(torch.ops.unsloth_zoo, fl._LP_OP).default,
        (h, w, b, t, None, 1.0, 0.0, 0.0, 0.0, torch.float32, False, False, False, False, True,
         True, False, False, 4, False),
        test_utils = ("test_faketensor",),
    )
    n = h.shape[0]
    torch.library.opcheck(
        getattr(torch.ops.unsloth_zoo, fl._LP_BWD_OP).default,
        (h, w, b, t, None, torch.zeros(n), torch.empty(0), torch.ones(n), torch.empty(0), torch.empty(0), 1.0, 0.0,
         0.0, 0.0, torch.float32, True, False, False, 4),
        test_utils = ("test_faketensor",),
    )


@pytest.mark.parametrize("device", DEVICES)
def test_two_dim_logits_bias(device):
    # Bart / Marian keep final_logits_bias as [1, vocab]; a direct call must not need it flattened first.
    h, w, _, t = _make(device, torch.float32, 12, 16, 37)
    lb = torch.randn(1, 37).to(device)
    ref = fused_linear_cross_entropy(h, w, None, t, divisor = 12, logits_bias = lb.reshape(-1))
    out = fused_linear_cross_entropy(h, w, None, t, divisor = 12, logits_bias = lb)
    torch.testing.assert_close(out, ref)


def test_detached_entropy_skips_the_entropy_backward(monkeypatch):
    # A detached entropy must not arrive as a zero gradient that runs the entropy term.
    seen = []
    real = fl._OPS["lp_bwd_impl"]
    monkeypatch.setitem(fl._OPS, "lp_bwd_impl", lambda *a: seen.append(a[8].numel()) or real(*a))
    h, w, _, t = _make("cpu", torch.float32, 10, 8, 33, train_head = True)
    lp, ent = fused_linear_logprobs(h, w, t, return_entropy = True, entropy_requires_grad = False)
    grads = torch.autograd.grad(lp.sum(), [h, w])
    ref = torch.autograd.grad(fused_linear_logprobs(h, w, t).sum(), [h, w])
    assert seen and seen[0] == 0, seen
    for a, b in zip(grads, ref):
        torch.testing.assert_close(a, b)


def test_backward_honours_max_rows_per_chunk(monkeypatch):
    # max_rows_per_chunk (UNSLOTH_GRPO_MAX_ROWS_PER_CHUNK) caps the recompute backward's row tiles as well.
    rows = []
    real = fl._Buffers
    monkeypatch.setattr(fl, "_Buffers", lambda R, *a, **k: rows.append(R) or real(R, *a, **k))
    h, w, _, t = _make("cpu", torch.float32, 40, 8, 33, train_head = True)
    lp = fused_linear_logprobs(h, w, t, max_rows_per_chunk = 8)
    rows.clear()
    lp.sum().backward()
    assert rows and max(rows) <= 8, rows


def test_input_error_does_not_disable_triton(monkeypatch):
    # An error the torch body raises too belongs to the inputs: re-raise it without marking Triton broken.
    monkeypatch.setattr(fl, "_use_triton", lambda device: device not in fl._TRITON_BROKEN)

    @fl._guarded
    def body(hidden, weight):
        raise ValueError("bad input")

    with pytest.raises(ValueError, match = "bad input"):
        body(torch.zeros(2, 2), torch.zeros(2, 2))
    assert not fl._TRITON_BROKEN
    # A failure only on the Triton path still falls back once and is remembered.
    def flaky(hidden, weight):
        if weight.device not in fl._TRITON_BROKEN:
            raise RuntimeError("triton")
        return hidden

    with pytest.warns(UserWarning, match = "Triton kernel failed"):
        fl._guarded(flaky)(torch.zeros(2, 2), torch.zeros(2, 2))
    assert torch.device("cpu") in fl._TRITON_BROKEN
    fl._TRITON_BROKEN.clear()


@CUDA
@pytest.mark.parametrize("backend", ["no_fp32_out", "hip"])
def test_fp16_tile_partials_do_not_overflow_without_fp32_out_gemm(monkeypatch, backend):
    # No fp16-in / fp32-out GEMM: one tile's partial passes 65504 while the tiles cancel, so fp16 output gave inf.
    if backend == "hip":
        monkeypatch.setattr(fl, "_IS_HIP", True)
    else:
        monkeypatch.setattr(fl, "_fp32_out_ok", lambda device, dtype = None: False)
    monkeypatch.setattr(fl, "_BWD_TILE_BYTES", 1)  # 256-column tiles
    h = torch.zeros((1, 128), device = "cuda", dtype = torch.float16, requires_grad = True)
    w = torch.full((4096, 128), 4.0, device = "cuda", dtype = torch.float16, requires_grad = True)
    t = torch.zeros(1, dtype = torch.long, device = "cuda")
    lp = fused_linear_logprobs(h, w, t)
    (lp * 32768.0).sum().backward()
    assert h.grad.isfinite().all() and w.grad.isfinite().all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bad", [-1, 64, 1000])
def test_ce_rejects_out_of_range_targets(device, bad):
    # Out-of-vocab targets raise as F.cross_entropy (else they stay in the divisor); ignore_index stays legal.
    h, w, _, t = _make(device, torch.float32, 8, 16, 64)
    t[0] = -100
    fused_linear_cross_entropy(h, w, None, t, divisor = 7.0)
    t[3] = bad
    with pytest.raises(IndexError, match = "out of bounds"):
        fused_linear_cross_entropy(h, w, None, t, divisor = 7.0)
    assert not fl._TRITON_BROKEN


def test_ce_rejects_invalid_label_smoothing():
    h, w, _, t = _make("cpu", torch.float32, 8, 16, 64)
    for eps in (1.1, -0.1):
        with pytest.raises(ValueError, match = "label_smoothing"):
            fused_linear_cross_entropy(h, w, None, t, divisor = 8.0, label_smoothing = eps)


@pytest.mark.parametrize("device", DEVICES)
def test_head_bias_shape_is_checked_and_broadcast_bias_works(device):
    N, H, V = 9, 16, 40
    h, w, _, t = _make(device, torch.float32, N, H, V, train_head = True)
    short = torch.zeros(V // 2, device = device)
    with pytest.raises(ValueError, match = "bias"):
        fused_linear_logprobs(h, w, t, bias = short)
    with pytest.raises(ValueError, match = "bias"):
        fused_linear_cross_entropy(h, w, short, t, divisor = 1.0)
    # A [1, V] bias: values and gradients (in the caller's shape) as a 1-D one.
    b2 = (torch.randn(1, V) * 0.1).to(device).requires_grad_(True)
    for fn in ("ce", "lp"):
        b1 = b2.detach().reshape(-1).requires_grad_(True)
        outs = []
        for b in (b2, b1):
            if fn == "ce":
                y = fused_linear_cross_entropy(h, w, b, t, divisor = 3.0)
            else:
                y = (fused_linear_logprobs(h, w, t, bias = b) * torch.arange(N, device = device)).sum()
            outs.append((y,) + torch.autograd.grad(y, (h, w, b)))
        (y2, gh2, gw2, gb2), (y1, gh1, gw1, gb1) = outs
        assert gb2.shape == b2.shape
        torch.testing.assert_close(y2, y1)
        torch.testing.assert_close(gb2.reshape(-1), gb1)
        torch.testing.assert_close(gh2, gh1)
        torch.testing.assert_close(gw2, gw1)


@CUDA
def test_bias_on_another_device_follows_the_head():
    h, w, _, t = _make("cuda", torch.float32, 6, 16, 32, train_head = True)
    b = torch.zeros(32, requires_grad = True)  # CPU bias, CUDA head
    lp = fused_linear_logprobs(h, w, t, bias = b)
    lp.sum().backward()
    assert b.grad is not None and b.grad.device.type == "cpu"
    loss = fused_linear_cross_entropy(h, w, b, t, divisor = 1.0)
    loss.backward()


def _record_buffers(monkeypatch):
    sizes = []
    real = fl._Buffers

    class Rec(real):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            sizes.append(self.buf.numel() * self.buf.element_size())

    monkeypatch.setattr(fl, "_Buffers", Rec)
    return sizes


@pytest.mark.parametrize("kind", ["ce", "lp", "lp_trainable"])
def test_fp32_head_chunks_stay_within_the_budget(monkeypatch, kind):
    # An fp32 head's logits were sized as 2 bytes, so chunks took twice the budget.
    budget = 64 * 1024
    monkeypatch.setattr(fl, "_budget_bytes", lambda device = None: budget)
    sizes = _record_buffers(monkeypatch)
    h, w, _, t = _make("cpu", torch.float32, 1024, 16, 64, train_head = kind == "lp_trainable")
    if kind == "ce":
        fused_linear_cross_entropy(h, w, None, t, divisor = 1.0).backward()
    else:
        fused_linear_logprobs(h, w, t).sum().backward()
    assert sizes and max(sizes) <= budget, (sizes, budget)


def test_recompute_backward_tiles_follow_the_budget(monkeypatch):
    # The recompute backward sized its tiles from the static 256 MB allowance alone, ignoring the device budget.
    N, H, V = 37, 32, 136
    budget = 20 * 16 * H
    monkeypatch.setattr(fl, "_budget_bytes", lambda device = None: budget)
    sizes = _record_buffers(monkeypatch)
    cols = []
    real = fl._row_backward

    def spy(device, x, *a, **k):
        cols.append(a[-2])
        return real(device, x, *a, **k)

    monkeypatch.setattr(fl, "_row_backward", spy)
    h, w, b, t = _make("cpu", torch.float32, N, H, V, train_head = True, with_bias = True)
    fused_linear_logprobs(h, w, t, bias = b).sum().backward()
    assert max(sizes) <= budget, (sizes, budget)
    assert len(set(cols)) >= 2, cols  # several vocab slices


@CUDA
def test_tiled_forward_follows_the_budget(monkeypatch):
    if not fl._HAS_TRITON:
        pytest.skip("needs Triton")
    budget = 4 * 16384 * 4
    monkeypatch.setattr(fl, "_budget_bytes", lambda device = None: budget)
    sizes = _record_buffers(monkeypatch)
    h, w, _, t = _make("cuda", torch.float32, 300, 32, 40000, train_head = True)  # > 2 splits: tiled
    tiled = fl.CALLS["tiled"]
    lp = fused_linear_logprobs(h, w, t)
    assert fl.CALLS["tiled"] == tiled + 1
    assert max(sizes) <= budget, (sizes, budget)
    ref = fused_linear_logprobs(h.detach(), w.detach(), t)
    torch.testing.assert_close(lp.detach(), ref, rtol = 1e-5, atol = 1e-5)


@CUDA
@pytest.mark.parametrize("backend", ["no_fp32_out", "hip"])
def test_fp16_odd_vocab_hidden_gradient_does_not_overflow(monkeypatch, backend):
    # Aligned columns' fp16 partial overflowed before the left-over column cancelled it.
    if backend == "hip":
        monkeypatch.setattr(fl, "_IS_HIP", True)
    else:
        monkeypatch.setattr(fl, "_fp32_out_ok", lambda device, dtype = None: False)
    dz = torch.tensor([[8.0] * 8 + [-56.0]], device = "cuda", dtype = torch.float16)
    w = torch.full((9, 1), 10000.0, device = "cuda", dtype = torch.float16)
    w[8] = 11428.0
    out = fl._dz_w(dz, w)
    ref = dz.double() @ w.double()
    assert out.isfinite().all()
    torch.testing.assert_close(out.double(), ref, rtol = 0, atol = 64.0)


@CUDA
def test_fp16_frozen_head_keeps_small_unit_gradients():
    # fp16: a confident token's p ~ 4e-8 went subnormal and ~1e-7 u entries lost bits; fp32 math resolves both.
    H, V = 4, 4096
    w = torch.zeros(V, H)
    w[1:, 0] = -17.0
    w[1:, 1] = 0.001
    h = torch.zeros(1, H)
    h[0, 0] = 1.0
    t = torch.zeros(1, dtype = torch.long)
    w16 = w.to("cuda", torch.float16)
    hd = h.double().requires_grad_(True)
    ref = torch.autograd.grad((hd @ w16.double().cpu().t()).log_softmax(-1)[0, 0] * 32768.0, hd)[0]
    hh = h.to("cuda", torch.float16).requires_grad_(True)
    lp = fused_linear_logprobs(hh, w16, t.cuda())
    (gh,) = torch.autograd.grad(lp.sum() * 32768.0, hh)
    assert ref[0, 1].abs() > 1e-3
    torch.testing.assert_close(gh.double().cpu(), ref, rtol = 2e-2, atol = 1e-6)


@CUDA
def test_scale_rows_retries_one_transient_failure(monkeypatch):
    real = fl._launch
    fails = {"left": 1}

    def flaky(device, fn):
        if fails["left"] > 0:
            fails["left"] -= 1
            raise RuntimeError("simulated transient Triton failure")
        return real(device, fn)

    monkeypatch.setattr(fl, "_launch", flaky)
    u = torch.randn(64, 32, device = "cuda")
    g = torch.randn(64, device = "cuda")
    out = torch.ops.unsloth_zoo.fused_linear_scale_rows(u, g, torch.float32)
    assert not fl._TRITON_BROKEN
    torch.testing.assert_close(out, u * g[:, None])


@pytest.mark.parametrize("device", DEVICES)
def test_trainable_head_long_context_backward(device):
    # Past `vocab` rows backward runs row-major; both orders match the oracle.
    N, H, V = 300, 16, 40
    h, w, b, t = _make(device, torch.float32, N, H, V, train_head = True, with_bias = True)
    lp = fused_linear_logprobs(h, w, t, bias = b, rows_per_chunk = 64)
    g = torch.randn(N, device = device, dtype = torch.float64)
    grads = torch.autograd.grad((lp.double() * g).sum(), (h, w, b))
    hd, wd, bd = (x.detach().double().requires_grad_(True) for x in (h, w, b))
    ref = _lp_oracle(hd, wd, bd, t, torch.ones(N, dtype = torch.bool, device = device), 1.0, 0.0, 0.0, 0.0)[0]
    refs = torch.autograd.grad((ref * g).sum(), (hd, wd, bd))
    for name, x, y in zip(("hidden", "weight", "bias"), grads, refs):
        _close(x, y, _tol(torch.float32), name)


@TRITON
def test_trainable_head_long_context_skips_the_rows_accumulator():
    N, H, V = 16384, 256, 128
    h, w, _, t = _make("cuda", torch.float32, N, H, V, train_head = True)
    fused_linear_logprobs(h, w, t).sum().backward()  # warm up: GEMM workspaces are not the accumulator
    h.grad = w.grad = None
    lp = fused_linear_logprobs(h, w, t)
    loss = lp.sum()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    loss.backward()
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - base
    # Vocab-major added an fp32 [N, H] accumulator on top of grad_h.
    assert extra < N * H * 4 * 1.75, extra / 2**20


def _peaked(device, dtype, N, H, V, seed = 0, scale = 13.7, cluster = 0):
    # Unit-norm head rows, h = scale (w_a + 0.95 w_b) + noise: two likely tokens (p ~ 0.6 / 0.3), the regime where
    # rounding the top logits to half dominates the gradient error. cluster = k: k likely tokens among the first
    # 2048 ids (frequent tokens share low ids), scale 20.
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(V, H, generator = g)
    w = w / w.norm(dim = 1, keepdim = True)
    if cluster:
        idx = torch.randint(0, 2048, (N, cluster), generator = g)
        c = torch.linspace(1.0, 0.8, cluster)
        h = 20.0 * (w[idx] * c[None, :, None]).sum(1) + 0.5 * torch.randn(N, H, generator = g) / H**0.5
        t = idx[torch.arange(N), torch.randint(0, cluster, (N,), generator = g)]
        return h.to(device, dtype), w.to(device, dtype), t.to(device)
    a = torch.randint(0, V, (N,), generator = g)
    b = (a + 1 + torch.randint(0, V - 1, (N,), generator = g)) % V
    h = scale * (w[a] + 0.95 * w[b]) + 0.5 * torch.randn(N, H, generator = g) / H**0.5
    t = torch.where(torch.rand(N, generator = g) < 0.7, a, b)
    return h.to(device, dtype), w.to(device, dtype), t.to(device)


def _rel(x, y):
    return float((x.double() - y).norm() / y.norm())


def _peaked_grads(monkeypatch, dtype, exact_top, recompute_fp32, ent_grad, N = 192, H = 512, V = 40000, cluster = 0):
    monkeypatch.setattr(fl, "_BWD_EXACT_TOP", exact_top)
    monkeypatch.setattr(fl, "_RECOMPUTE_FP32", recompute_fp32)
    h, w, t = _peaked("cuda", dtype, N, H, V, cluster = cluster)
    h.requires_grad_(True)
    w.requires_grad_(True)
    mask = torch.ones(N, dtype = torch.bool, device = "cuda")
    mask[::5] = False
    gen = torch.Generator().manual_seed(1)
    g = torch.randn(N, generator = gen).cuda().double()
    ge = (0.1 * torch.randn(N, generator = gen)).cuda().double()
    before = dict(fl.CALLS)
    lp, ent = fused_linear_logprobs(h, w, t, mask = mask, return_entropy = True, entropy_requires_grad = ent_grad)
    obj = (lp.double() * g).sum() + ((ent.double() * ge).sum() if ent_grad else 0)
    gh, gw = torch.autograd.grad(obj, (h, w))
    engaged = fl.CALLS["exact_top"] - before["exact_top"]
    hd = h.detach().double().requires_grad_(True)
    wd = w.detach().double().requires_grad_(True)
    ref = _lp_oracle(hd, wd, None, t, mask, 1.0, 0.0, 0.0, 0.0)
    robj = (ref[0] * g).sum() + ((ref[1] * ge).sum() if ent_grad else 0)
    rh, rw = torch.autograd.grad(robj, (hd, wd))
    return _rel(gh, rh), _rel(gw, rw), engaged, lp.detach(), ent.detach()


@CUDA
@pytest.mark.parametrize("ent_grad", [False, True])
# random likely pair; heavy tail (H 64); 2 / 8 likely tokens in the first 2048 ids
@pytest.mark.parametrize("shape", [(192, 512, 40000), (256, 64, 40000), (192, 512, 40000, 2), (192, 512, 40000, 8)])
def test_exact_top_backward_keeps_fp32_gradient_accuracy(monkeypatch, ent_grad, shape):
    # Half recompute logits with each block's top logit swapped back to its fp32 value, under the fp32 lse: the
    # gradient stays at the fp32 recompute's error. Plain half logits under the fp32 lse were 3x worse than TRL.
    if not fl._HAS_TRITON:
        pytest.skip("needs Triton")
    for dtype in [torch.float16] + ([torch.bfloat16] if _half_ok("cuda") else []):
        if not fl._logits_mode(torch.device("cuda"), dtype)[0]:
            continue  # no fp32-output GEMM: the half path is the only one
        eh, ew, engaged, lp, ent = _peaked_grads(monkeypatch, dtype, True, True, ent_grad, *shape)
        fh, fw, off, lp_f, ent_f = _peaked_grads(monkeypatch, dtype, False, True, ent_grad, *shape)
        hh, hw, _, _, _ = _peaked_grads(monkeypatch, dtype, False, False, ent_grad, *shape)
        assert engaged > 0 and off == 0, (engaged, off)
        assert torch.equal(lp, lp_f) and torch.equal(ent, ent_f)
        # B200: within 1-5% of fp32, 0.2-0.3x all-half (top-1 only per 2048 block: 4-5x fp32).
        assert eh <= 1.1 * fh and ew <= 1.1 * fw, (dtype, eh, fh, ew, fw)
        if hh > 2 * fh:  # where half logits cost accuracy at all (not fp16 with a heavy tail)
            assert eh < 0.4 * hh and ew < 0.4 * hw, (dtype, eh, hh, ew, hw)


@CUDA
@pytest.mark.parametrize("V", [1003, 20001, 40000])
def test_trainable_head_grad_and_no_grad_passes_are_bit_identical(monkeypatch, V):
    # The exact-top forward (several vocab splits, compacted rows) must not move the statistics a no-grad pass sees.
    # Up to two splits the no-grad pass runs one row kernel; the tiled (split) forward differed there (oracle
    # COMPILED_DRIFT under no_grad), so those vocabularies stay chunked.
    if not fl._HAS_TRITON:
        pytest.skip("needs Triton")
    for dtype in _dtypes("cuda"):
        h, w, t = _peaked("cuda", dtype, 96, 64, V)
        mask = torch.ones(96, dtype = torch.bool, device = "cuda")
        mask[3::7] = False
        with torch.no_grad():
            ref = fused_linear_logprobs(h, w, t, mask = mask, return_entropy = True)
        w.requires_grad_(True)
        h.requires_grad_(True)
        tiled = fl.CALLS["tiled"]
        out = fused_linear_logprobs(h, w, t, mask = mask, return_entropy = True)
        assert fl.CALLS["tiled"] == tiled + (V > 2 * fl._SPLIT_COLS)
        for x, y in zip(out, ref):
            assert torch.equal(x.detach(), y), (dtype, V)


@pytest.mark.parametrize("device", DEVICES)
def test_trainable_head_single_row_chunk_backward(monkeypatch, device):
    if fl._HIP_TILES:
        pytest.skip("ROCm keeps its small backward tiles (no single row chunk)")
    # Every kept row in one chunk: the head-gradient tile is one GEMM straight into grad_w (no fp32 accumulator
    # re-read per row chunk); ragged last tile, odd vocabulary, dropped rows.
    N, H, V = 70, 24, 4103 if device == "cuda" else 523
    monkeypatch.setattr(fl, "_SINGLE_MIN_COLS", 512)
    monkeypatch.setattr(fl, "_budget_bytes", lambda device = None: 64 * 4 * 1200)
    for dtype in _dtypes(device):
        h, w, b, t = _make(device, dtype, N, H, V, train_head = True, with_bias = True)
        mask = torch.ones(N, dtype = torch.bool, device = device)
        mask[::6] = False
        g = torch.randn(N, device = device, dtype = torch.float64)

        def grads():
            lp, ent = fused_linear_logprobs(h, w, t, bias = b, mask = mask, return_entropy = True)
            return torch.autograd.grad((lp.double() * g).sum() + 0.1 * ent.double().sum(), (h, w, b))

        single = fl.CALLS["bwd_single"]
        got = grads()
        assert fl.CALLS["bwd_single"] == single + 1
        monkeypatch.setattr(fl, "_SINGLE_MIN_COLS", 0)
        chunked = grads()
        assert fl.CALLS["bwd_single"] == single + 1
        monkeypatch.setattr(fl, "_SINGLE_MIN_COLS", 512)
        hd, wd, bd = (x.detach().double().requires_grad_(True) for x in (h, w, b))
        lp, ent = _lp_oracle(hd, wd, bd, t, mask, 1.0, 0.0, 0.0, 0.0)[:2]
        refs = torch.autograd.grad((lp * g).sum() + 0.1 * ent.sum(), (hd, wd, bd))
        for name, x, y, z in zip(("hidden", "weight", "bias"), got, chunked, refs):
            assert x.dtype == y.dtype and x.shape == y.shape
            _close(x, z, _tol(dtype), f"{name} {dtype}")
            # The single chunk rounds each head-gradient entry once; never worse than the chunked accumulator.
            assert _rel(x, z) <= 1.05 * _rel(y, z) + 1e-12, (name, dtype, _rel(x, z), _rel(y, z))
        assert (got[0][~mask] == 0).all()


def test_mm_into_restores_the_reduction_flags():
    flags = torch.backends.cuda.matmul
    before = (flags.allow_bf16_reduced_precision_reduction, flags.allow_fp16_reduced_precision_reduction)
    a, b = torch.randn(8, 16), torch.randn(16, 4)
    out = torch.empty(8, 4)
    fl._mm_into(out, a, b)
    torch.testing.assert_close(out, a @ b)
    if torch.cuda.is_available():
        for dtype in _dtypes("cuda")[1:]:
            ac, bc = a.cuda().to(dtype), b.cuda().to(dtype)
            outc = torch.empty(8, 4, device = "cuda", dtype = dtype)
            fl._mm_into(outc, ac, bc)
            torch.testing.assert_close(outc.float(), (ac.float() @ bc.float()).to(dtype).float())
    assert (flags.allow_bf16_reduced_precision_reduction, flags.allow_fp16_reduced_precision_reduction) == before


@CUDA
def test_opcheck_exact_top_outputs():
    # The block-top output's shape comes from the fake as well (compile / AOT), not only the real op.
    if not fl._HAS_TRITON or fl._IS_HIP:
        pytest.skip("exact-top backward: Triton on NVIDIA only")
    dtype = torch.bfloat16 if _half_ok("cuda") else torch.float16
    h, w, b, t = _make("cuda", dtype, 40, 16, 40000, train_head = True, with_bias = True)
    out = getattr(torch.ops.unsloth_zoo, fl._LP_OP).default(
        h, w, b, t, None, 1.0, 0.0, 0.0, 0.0, dtype, True, False, False, False, False, True, True, True, 0, False)
    assert out[7].shape == fl._block_top_shape(40, 40000, h.device, dtype, True) and out[7].numel() > 0
    torch.library.opcheck(
        getattr(torch.ops.unsloth_zoo, fl._LP_OP).default,
        (h, w, b, t, None, 1.0, 0.0, 0.0, 0.0, dtype, True, False, False, False, False, True, True, True, 0, False),
        test_utils = ("test_faketensor",),
    )


@CUDA
def test_frozen_head_entropy_gradient_keeps_fp32_recompute():
    # Exact-top is for trainable heads; the frozen entropy-gradient backward stays on fp32 recompute logits.
    if not fl._HAS_TRITON or fl._IS_HIP:
        pytest.skip("exact-top backward: Triton on NVIDIA only")
    dtype = torch.bfloat16 if _half_ok("cuda") else torch.float16
    h, w, t = _peaked("cuda", dtype, 64, 64, 40000)
    h.requires_grad_(True)
    tiled, exact = fl.CALLS["tiled"], fl.CALLS["exact_top"]
    lp, ent = fused_linear_logprobs(h, w, t, return_entropy = True)
    (lp.sum() + ent.sum()).backward()
    assert fl.CALLS["tiled"] == tiled + 1 and fl.CALLS["exact_top"] == exact
    w.requires_grad_(True)
    lp, ent = fused_linear_logprobs(h, w, t, return_entropy = True)
    (lp.sum() + ent.sum()).backward()
    assert fl.CALLS["exact_top"] > exact


@CUDA
def test_ce_gradient_scale_is_not_rounded_to_half():
    # p(target) ~ 0 (an untrained head): each row's largest gradient entry is ~ -1 / n_items. Written into the half
    # buffer as such it rounds to half(1 / n) on every row; n = 255 puts that 0.39% off in bf16.
    if not _half_ok("cuda"):
        pytest.skip("needs bf16")
    g = torch.Generator(device = "cpu").manual_seed(0)
    N, H, V, n = 255, 64, 4096, 255
    h = (torch.randn(N, H, generator = g) * 4).cuda().bfloat16().requires_grad_(True)
    w = (torch.randn(V, H, generator = g) * 4).cuda().bfloat16()
    t = torch.randint(0, V, (N,), generator = g).cuda()
    loss = fused_linear_cross_entropy(h, w, None, t, divisor = n)
    (got,) = torch.autograd.grad(loss, (h,))
    h64 = h.detach().double().requires_grad_(True)
    ref = torch.nn.functional.cross_entropy(h64 @ w.double().t(), t, reduction = "sum") / n
    (want,) = torch.autograd.grad(ref, (h64,))
    ratio = float(got.double().norm() / want.norm())
    # Without an fp32-output GEMM (ROCm, torch < 2.8) dz @ W is already rounded to half, so the leftover factor
    # rounds a second time (+0.2% here); still under the 0.39% that writing half(1 / 255) itself costs.
    half_gemm = fl._IS_HIP or not fl._fp32_out_ok(h.device, h.dtype)
    assert abs(ratio - 1.0) < (3e-3 if half_gemm else 1e-3), ratio


@CUDA
@pytest.mark.parametrize("ent_grad", [False, True])
def test_exact_top_backward_with_tied_maxima(monkeypatch, ent_grad):
    # Repeated (or zero-initialised) head rows tie a column group's top logit; its second top is the tied value, not
    # the next distinct one (that gave ~0.5 hidden gradients and NaN entropy gradients).
    if not fl._HAS_TRITON or fl._IS_HIP:
        pytest.skip("needs Triton on CUDA (ROCm has no exact-top backward)")
    N, H, V = 64, 64, 40000
    gen = torch.Generator().manual_seed(0)
    w0 = torch.randn(V, H, generator = gen) / H**0.5
    w0[:4096] = w0[0]
    h0 = torch.randn(N, H, generator = gen)
    t = torch.randint(0, V, (N,), generator = gen).cuda()
    g = torch.randn(N, generator = gen).cuda().double()
    for dtype in _dtypes("cuda")[1:]:  # half only: fp32 heads have no exact-top backward
        if not fl._logits_mode(torch.device("cuda"), dtype)[0]:
            continue

        def grads(exact):
            monkeypatch.setattr(fl, "_BWD_EXACT_TOP", exact)
            h = h0.cuda().to(dtype).requires_grad_(True)
            w = w0.cuda().to(dtype).requires_grad_(True)
            lp, ent = fused_linear_logprobs(h, w, t, return_entropy = True, entropy_requires_grad = ent_grad)
            obj = (lp.double() * g).sum() + (0.1 * ent.double().sum() if ent_grad else 0)
            return torch.autograd.grad(obj, (h, w))

        before = fl.CALLS["exact_top"]
        eh, ew = grads(True)
        assert fl.CALLS["exact_top"] > before
        fh, fw = grads(False)
        hd = h0.cuda().to(dtype).double().requires_grad_(True)
        wd = w0.cuda().to(dtype).double().requires_grad_(True)
        lp, ent = _lp_oracle(hd, wd, None, t, torch.ones(N, dtype = torch.bool, device = "cuda"), 1.0, 0.0, 0.0, 0.0)[:2]
        rh, rw = torch.autograd.grad((lp * g).sum() + (0.1 * ent.sum() if ent_grad else 0), (hd, wd))
        assert torch.isfinite(eh).all() and torch.isfinite(ew).all()
        assert _rel(eh, rh) <= 1.5 * _rel(fh, rh) + 1e-6 and _rel(ew, rw) <= 1.5 * _rel(fw, rw) + 1e-6, \
            (dtype, _rel(eh, rh), _rel(fh, rh), _rel(ew, rw), _rel(fw, rw))
    monkeypatch.setattr(fl, "_BWD_EXACT_TOP", True)


@CUDA
@pytest.mark.parametrize("V", [20000, 20003])
def test_fp16_without_fp32_output_gemms_does_not_overflow(monkeypatch, V):
    # torch < 2.8 / ROCm: no fp32-output GEMM. The frozen head's forward gradient is scaled up in fp16, so a half GEMM
    # result can pass 65504 even when the true gradient is representable.
    monkeypatch.setattr(fl, "_fp32_out_ok", lambda *a, **k: False)
    N, H = 64, 64
    gen = torch.Generator().manual_seed(1)
    w = (16 * torch.randn(V, H, generator = gen) / H**0.5).cuda().half()
    h0 = (0.1 * torch.randn(N, H, generator = gen)).cuda().half()
    t = torch.randint(0, V, (N,), generator = gen).cuda()
    h = h0.clone().requires_grad_(True)
    (gh,) = torch.autograd.grad(fused_linear_logprobs(h, w, t).double().sum(), (h,))
    hd = h0.double().requires_grad_(True)
    lp = _lp_oracle(hd, w.double(), None, t, torch.ones(N, dtype = torch.bool, device = "cuda"), 1.0, 0.0, 0.0, 0.0)[0]
    (rh,) = torch.autograd.grad(lp.sum(), (hd,))
    assert torch.isfinite(gh).all() and _rel(gh, rh) < 2e-2, _rel(gh, rh)
    h = h0.clone().requires_grad_(True)
    (gc,) = torch.autograd.grad(fused_linear_cross_entropy(h, w, None, t, divisor = N), (h,))
    assert torch.isfinite(gc).all()


@CUDA
def test_softcap_keeps_the_fp32_recompute():
    # A hard softcap (5) made the half recompute's head / hidden gradient ~3.4x the fp32 recompute's error.
    if not fl._HAS_TRITON or fl._IS_HIP:
        pytest.skip("needs Triton on CUDA")
    for dtype in _dtypes("cuda")[1:]:
        if not fl._logits_mode(torch.device("cuda"), dtype)[0]:
            continue
        h, w, t = _peaked("cuda", dtype, 128, 64, 40000)
        h.requires_grad_(True)
        w.requires_grad_(True)
        for cap, engaged in ((5.0, False), (0.0, True)):
            before = fl.CALLS["exact_top"]
            lp = fused_linear_logprobs(h, w, t, logit_softcapping = cap)
            torch.autograd.grad(lp.sum(), (h, w))
            assert (fl.CALLS["exact_top"] > before) == engaged, (dtype, cap)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("V", [3001, 40001])
def test_lazy_entropy_only_logged_keeps_the_forward_gradient(device, V):
    # entropy_requires_grad=None and the entropy only logged (TRL 1.15 DPO): a frozen head takes the forward gradient,
    # bit for bit the entropy_requires_grad=False path, and no recompute backward runs.
    for dtype in _dtypes(device):
        h, w, b, t = _make(device, dtype, 37, 24, V, with_bias = True)
        g = torch.randn(37, device = device, dtype = torch.float64)

        def run(erg):
            hh = h.detach().clone().requires_grad_(True)
            lp, ent = fused_linear_logprobs(hh, w, t, bias = b, return_entropy = True, entropy_requires_grad = erg,
                                            temperature = 0.7)
            (gh,) = torch.autograd.grad((lp.double() * g).sum(), (hh,))
            return lp.detach(), ent.detach(), gh

        bwd = fl.CALLS["logprobs_bwd"]
        lazy = run(None)
        assert fl.CALLS["logprobs_bwd"] == bwd, "a logged entropy must not run the recompute backward"
        off = run(False)
        for x, y in zip(lazy, off):
            assert torch.equal(x, y), (device, dtype)
        on = run(True)  # the eager entropy gradient: the recompute route
        assert fl.CALLS["logprobs_bwd"] > bwd
        # Same log-probs; the recompute route's entropy agrees bit for bit in half precision and to fp32 rounding
        # order (an ulp or two, equally accurate against fp64) with fp32 inputs.
        assert torch.equal(lazy[0], on[0])
        if dtype == torch.float32:
            torch.testing.assert_close(lazy[1], on[1], rtol = 0, atol = 4e-6)
        else:
            assert torch.equal(lazy[1], on[1])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("V", [3001, 40001])
def test_lazy_entropy_gradient_when_used(device, V):
    # A lazy entropy that does reach the loss (GRPO entropy bonus) gets its gradient: forward gradient for the
    # log-probs plus a recomputed entropy term, as accurate as the eager entropy gradient against fp64.
    for dtype in _dtypes(device):
        h, w, b, t = _make(device, dtype, 37, 24, V, with_bias = True)
        g = torch.randn(37, device = device, dtype = torch.float64)
        ge = 0.3 * torch.randn(37, device = device, dtype = torch.float64)
        mask = torch.ones(37, dtype = torch.bool, device = device)
        mask[::5] = False

        def run(erg):
            hh = h.detach().clone().requires_grad_(True)
            lp, ent = fused_linear_logprobs(hh, w, t, bias = b, mask = mask, return_entropy = True,
                                            entropy_requires_grad = erg, temperature = 0.7)
            (gh,) = torch.autograd.grad((lp.double() * g).sum() + (ent.double() * ge).sum(), (hh,))
            return gh

        bwd = fl.CALLS["logprobs_bwd"]
        lazy = run(None)
        assert fl.CALLS["logprobs_bwd"] == bwd + 1
        eager = run(True)
        hd = h.detach().double().requires_grad_(True)
        lp, ent = _lp_oracle(hd, w.double(), b.double(), t, mask, 0.7, 0.0, 0.0, 0.0)[:2]
        (ref,) = torch.autograd.grad((lp * g).sum() + (ent * ge).sum(), (hd,))
        _close(lazy, ref, _tol(dtype), f"lazy {device} {dtype}")
        # Same backward as the eager entropy gradient (one recompute, rounded once).
        assert _rel(lazy, ref) <= 1.05 * _rel(eager, ref) + 1e-9, (device, dtype, _rel(lazy, ref), _rel(eager, ref))
        assert (lazy[~mask] == 0).all()


@pytest.mark.parametrize("device", DEVICES)
def test_lazy_entropy_with_a_trainable_head_recomputes(device):
    # A trainable head has no forward gradient: a lazy entropy is then simply differentiable (recompute route).
    h, w, b, t = _make(device, torch.float32, 20, 16, 501, train_head = True, with_bias = True)
    lp, ent = fused_linear_logprobs(h, w, t, bias = b, return_entropy = True, entropy_requires_grad = None)
    got = torch.autograd.grad(lp.sum() + ent.sum(), (h, w, b))
    lp2, ent2 = fused_linear_logprobs(h, w, t, bias = b, return_entropy = True, entropy_requires_grad = True)
    want = torch.autograd.grad(lp2.sum() + ent2.sum(), (h, w, b))
    for x, y in zip(got, want):
        assert torch.equal(x, y)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
@pytest.mark.parametrize("use_ent", [False, True])
def test_lazy_entropy_compiles_without_graph_breaks(use_ent):
    if not fl._HAS_TRITON:
        pytest.skip("Inductor compiles GPU code through Triton")
    import torch._dynamo
    torch._dynamo.reset()
    h, w, _, t = _make("cuda", _dtypes("cuda")[-1], 64, 32, 40001)

    def f(hh):
        lp, ent = fused_linear_logprobs(hh, w, t, return_entropy = True, entropy_requires_grad = None)
        return lp.float().sum() + (0.1 * ent.float().sum() if use_ent else 0.0 * ent.detach().float().sum())

    hh = h.detach().clone().requires_grad_(True)
    compiled = torch.compile(f, fullgraph = True)
    (gc,) = torch.autograd.grad(compiled(hh), (hh,))
    he = h.detach().clone().requires_grad_(True)
    (ge,) = torch.autograd.grad(f(he), (he,))
    torch.testing.assert_close(gc.float(), ge.float(), rtol = 0, atol = 0)


@CUDA
@pytest.mark.parametrize("n_keep", [1, 777, 4095, 4096, 4097])
def test_small_batches_take_the_tile_and_stay_bit_identical(monkeypatch, n_keep):
    # The small-card rule (threshold 0): below 4096 kept rows too, the frozen pass and its no-grad reference take
    # [rows, <= 16384] fp32 tiles (TRL's [rows, 32768] bf16 transient), and no-grad, frozen and trainable passes are
    # bit-identical.
    if not fl._HAS_TRITON or fl._IS_HIP:
        pytest.skip("needs Triton on CUDA")
    for name in ("UNSLOTH_FUSED_LINEAR_FLASH", "UNSLOTH_FUSED_LINEAR_FWD_TILE_COLS"):
        if os.environ.get(name) is not None:
            pytest.skip(f"{name} is set")
    monkeypatch.setattr(fl, "_FLASH_MIN_ROWS", 0)
    shapes = []
    real = fl._Buffers.__init__

    def record(self, rows, n_cols, *a, **k):
        shapes.append(n_cols)
        real(self, rows, n_cols, *a, **k)

    monkeypatch.setattr(fl._Buffers, "__init__", record)
    V, H, N = 70001, 64, n_keep + 3
    for dtype in _dtypes("cuda")[1:]:
        if not fl._logits_mode(torch.device("cuda"), dtype)[0]:
            continue
        gen = torch.Generator().manual_seed(n_keep)
        h = torch.randn(N, H, generator = gen).cuda().to(dtype)
        w = (torch.randn(V, H, generator = gen) / 8).cuda().to(dtype)
        t = torch.randint(0, V, (N,), generator = gen).cuda()
        mask = torch.ones(N, dtype = torch.bool, device = "cuda")
        mask[:3] = False
        with torch.no_grad():
            ref = fused_linear_logprobs(h, w, t, mask = mask)
        shapes.clear()
        flash = fl.CALLS["flash"]
        frozen = fused_linear_logprobs(h.clone().requires_grad_(True), w, t, mask = mask)
        assert fl.CALLS["flash"] == flash + 1, (dtype, n_keep)
        assert shapes and max(shapes) <= 16384, shapes
        trainable = fused_linear_logprobs(h.clone().requires_grad_(True), w.clone().requires_grad_(True), t, mask = mask)
        assert torch.equal(frozen.detach(), ref) and torch.equal(trainable.detach(), ref), (dtype, n_keep)


def test_tile_cap_leaves_large_and_half_layouts_alone(monkeypatch):
    # From 4096 rows the fp32 budget tile is already 16384 columns; half-logit layouts keep the budget tile.
    dev = torch.device("cpu")
    monkeypatch.setattr(fl, "_budget_bytes", lambda device = None: 2 * 1024**3)
    # The CUDA tile budget (ROCm's 8 MB one already keeps half tiles under 16384 columns) and an explicit cap.
    monkeypatch.setattr(fl, "_BWD_TILE_BYTES", 256 * 1024**2)
    monkeypatch.setattr(fl, "_BWD_ROWS", 4096)
    monkeypatch.setattr(fl, "_FWD_TILE_COLS", 16384)
    for n in (4096, 8192, 65536):
        for bpl in (2, 4):
            capped = fl._tiled_layout(n, 151936, 0, bpl, dev)
            monkeypatch.setattr(fl, "_FWD_TILE_COLS", 0)
            budget = fl._tiled_layout(n, 151936, 0, bpl, dev)
            monkeypatch.setattr(fl, "_FWD_TILE_COLS", 16384)
            assert capped == budget, (n, bpl, capped, budget)
    assert fl._tiled_layout(1536, 151936, 0, 4, dev)[1] == 16384
    assert fl._tiled_layout(1536, 151936, 0, 2, dev)[1] > 16384


def test_flash_min_rows_follows_the_card_size(monkeypatch):
    # Up to 24 GiB (T4 / L4 / consumer cards) the tiles are faster and TRL-sized at every row count; larger cards keep
    # [rows, V] chunks below 4096 rows, where they are faster (A100 2-17%). The env setting wins everywhere.
    class Props:
        def __init__(self, gib):
            self.total_memory = int(gib * 1024**3)

    monkeypatch.setattr(fl, "_FLASH_MIN_ROWS", None)
    for gib, want in ((15, 0), (22.5, 0), (24, 0), (40, 4096), (80, 4096), (96, 4096), (180, 4096)):
        fl._flash_min_rows_auto.cache_clear()
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device, gib = gib: Props(gib))
        assert fl._flash_min_rows(torch.device("cuda", 0)) == want, gib
    fl._flash_min_rows_auto.cache_clear()
    assert fl._flash_min_rows(torch.device("cpu")) == 4096
    monkeypatch.setattr(fl, "_IS_HIP", True)  # ROCm keeps the previous threshold on any card size
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: Props(16))
    fl._flash_min_rows_auto.cache_clear()
    assert fl._flash_min_rows(torch.device("cuda", 0)) == 4096
    monkeypatch.setattr(fl, "_IS_HIP", False)
    monkeypatch.setattr(fl, "_FLASH_MIN_ROWS", 123)
    assert fl._flash_min_rows(torch.device("cuda", 0)) == 123
    fl._flash_min_rows_auto.cache_clear()
