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

import numpy
import pytest
import torch

import unsloth_zoo.fused_losses.fused_linear as fl
from unsloth_zoo.fused_losses.fused_linear import fused_linear_cross_entropy, fused_linear_logprobs

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")


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


@CUDA
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


@CUDA
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
        torch.ops.unsloth_zoo.fused_linear_logprobs.default,
        (h, w, b, t.clamp(min = 0), mask, 0.8, 0.0, 0.0, 0.0, torch.float32, True, True, True, True, False,
         True, True, True, 4),
    )


@CUDA
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


@CUDA
@CUDA
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
        torch.ops.unsloth_zoo.fused_linear_logprobs.default,
        (h, w, b, t, None, 1.0, 0.0, 0.0, 0.0, torch.float32, False, False, False, False, True,
         True, False, False, 4),
        test_utils = ("test_faketensor",),
    )
    n = h.shape[0]
    torch.library.opcheck(
        torch.ops.unsloth_zoo.fused_linear_logprobs_bwd.default,
        (h, w, b, t, None, torch.zeros(n), torch.empty(0), torch.ones(n), torch.empty(0), 1.0, 0.0, 0.0, 0.0,
         torch.float32, True, False, False, 4),
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
    h, w, _, t = _make("cuda", torch.float32, 300, 32, 20000, train_head = True)
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


@CUDA
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
