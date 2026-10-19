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

"""Flash-style forward gradient (frozen head, vocab tiles): fp64 oracle, never less accurate than the [rows, V]
chunks it replaces, bit-identical no-grad / frozen / trainable passes, the kill switch, and cross entropy."""

import pytest
import torch

import unsloth_zoo.fused_losses.fused_linear as fl
from unsloth_zoo.fused_losses.fused_linear import fused_linear_cross_entropy, fused_linear_logprobs

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and fl._HAS_TRITON and not fl._IS_HIP),
    reason = "needs CUDA and Triton (ROCm keeps the chunked path)",
)


@pytest.fixture(autouse = True)
def _flash_on(monkeypatch):
    if not fl._logits_mode(torch.device("cuda"), torch.float16)[0]:
        pytest.skip("needs fp32-output GEMMs (torch >= 2.8)")
    monkeypatch.setattr(fl, "_FLASH", True)
    yield
    broken = set(fl._TRITON_BROKEN)
    fl._TRITON_BROKEN.clear()
    assert not broken, f"Triton fell back to torch on {broken}"


def _dtypes():
    return [torch.float16] + ([torch.bfloat16] if torch.cuda.get_device_capability()[0] >= 8 else [])


def _transform(z, mul, cap):
    if mul:
        z = z * mul
    if cap:
        z = cap * torch.tanh(z / cap)
    return z


def _inputs(dtype, N, H, V, seed = 0, late_max = False, with_bias = False):
    # Peaked rows (p ~ 0.6 / 0.3). late_max: the likely tokens sit in the last vocab tile, so every row's running
    # max moves after the first tiles and the accumulator is rescaled.
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(V, H, generator = g)
    w = w / w.norm(dim = 1, keepdim = True)
    lo = V - 4000 if late_max else 0
    a = torch.randint(lo, V, (N,), generator = g)
    b = torch.randint(lo, V, (N,), generator = g)
    h = 13.7 * (w[a] + 0.95 * w[b]) + 0.5 * torch.randn(N, H, generator = g) / H**0.5
    t = torch.where(torch.rand(N, generator = g) < 0.7, a, b)
    bias = (0.1 * torch.randn(V, generator = g)).cuda().to(dtype) if with_bias else None
    mask = torch.rand(N, generator = g) >= 0.2
    return h.cuda().to(dtype), w.cuda().to(dtype), bias, t.cuda(), mask.cuda()


def _oracle(h, w, bias, t, mask, T, mul, cap, g):
    hd = h.detach().double().requires_grad_(True)
    z = hd @ w.double().t() + (bias.double() if bias is not None else 0)
    lp = (_transform(z, mul, cap) / T).log_softmax(-1).gather(-1, t[:, None])[:, 0] * mask.double()
    (grad,) = torch.autograd.grad((lp * g).sum(), (hd,))
    return lp.detach(), grad


def _rel(x, y):
    return float((x.double() - y).norm() / y.norm())


def _frozen(h, w, t, g, **kw):
    hh = h.detach().clone().requires_grad_(True)
    lp = fused_linear_logprobs(hh, w, t, **kw)
    (grad,) = torch.autograd.grad((lp.double() * g).sum(), (hh,))
    return lp.detach(), grad


@pytest.mark.parametrize("T,mul,cap,with_bias", [(1.0, 0.0, 0.0, False), (0.7, 1.5, 30.0, True)])
def test_flash_matches_oracle_and_is_never_less_accurate_than_chunks(monkeypatch, T, mul, cap, with_bias):
    for dtype in _dtypes():
        h, w, bias, t, mask = _inputs(dtype, 160, 64, 40001, with_bias = with_bias)
        g = torch.randn(160, device = "cuda", dtype = torch.float64)
        kw = dict(bias = bias, mask = mask, temperature = T, logit_scale_multiply = mul, logit_softcapping = cap)
        n = fl.CALLS["flash"]
        lp, grad = _frozen(h, w, t, g, **kw)
        assert fl.CALLS["flash"] == n + 1
        monkeypatch.setattr(fl, "_FLASH", False)
        lp_c, grad_c = _frozen(h, w, t, g, **kw)
        assert fl.CALLS["flash"] == n + 1
        monkeypatch.setattr(fl, "_FLASH", True)
        lp64, grad64 = _oracle(h, w, bias, t, mask, T, mul, cap, g)
        assert float((lp.double() - lp64).abs().max()) < 2e-4, dtype
        assert _rel(grad, grad64) < 1e-2, (dtype, _rel(grad, grad64))
        assert _rel(grad, grad64) <= 1.05 * _rel(grad_c, grad64) + 1e-6, (dtype, _rel(grad, grad64), _rel(grad_c, grad64))
        # Dropped rows get no gradient.
        assert not grad[~mask].any()


def test_flash_rescales_when_the_row_max_moves_late(monkeypatch):
    # Every row's max is in the last tile; more rows than one wave of programs (a cross-warp race on the running max
    # only showed past the first wave), and several row chunks.
    monkeypatch.setattr(fl, "_BWD_ROWS", 1536)
    dtype = _dtypes()[-1]
    h, w, _, t, mask = _inputs(dtype, 4096, 64, 40001, seed = 1, late_max = True)
    g = torch.randn(4096, device = "cuda", dtype = torch.float64)
    lp, grad = _frozen(h, w, t, g, mask = mask)
    monkeypatch.setattr(fl, "_FLASH", False)
    _, grad_c = _frozen(h, w, t, g, mask = mask)
    lp64, grad64 = _oracle(h, w, None, t, mask, 1.0, 0.0, 0.0, g)

    def rows(x):
        # Rows with p(target) ~ 1 have a ~1e-6 gradient: the relative error there is the same on both paths.
        return ((x.double() - grad64).norm(dim = 1) / grad64.norm(dim = 1).clamp_min(1e-30))[mask]

    assert _rel(grad, grad64) <= 1.05 * _rel(grad_c, grad64) + 1e-6, (_rel(grad, grad64), _rel(grad_c, grad64))
    assert float(rows(grad).max()) <= 1.05 * float(rows(grad_c).max()) + 1e-6
    assert float(rows(grad).median()) < 5e-3
    assert float((lp.double() - lp64).abs().max()) < 2e-4


@pytest.mark.parametrize("flash", [True, False])
def test_no_grad_frozen_and_trainable_passes_are_bit_identical(monkeypatch, flash):
    # Several row chunks and vocab tiles, compacted rows. Under the kill switch the frozen and no-grad passes stay
    # chunked (still identical to each other) and the flash path never runs.
    monkeypatch.setattr(fl, "_FLASH", flash)
    monkeypatch.setattr(fl, "_BWD_ROWS", 48)
    for dtype in _dtypes():
        h, w, bias, t, mask = _inputs(dtype, 150, 64, 40001, seed = 2, with_bias = True)
        kw = dict(bias = bias, mask = mask, temperature = 0.9, return_entropy = True, entropy_requires_grad = False)
        n = fl.CALLS["flash"]
        with torch.no_grad():
            ref = fused_linear_logprobs(h, w, t, **kw)
        hh = h.clone().requires_grad_(True)
        frozen = fused_linear_logprobs(hh, w, t, **kw)
        frozen[0].sum().backward()
        assert fl.CALLS["flash"] == n + flash
        for x, y in zip(frozen, ref):
            assert torch.equal(x.detach(), y), (dtype, flash)
        if flash:
            ww = w.clone().requires_grad_(True)
            trainable = fused_linear_logprobs(h, ww, t, **kw)
            ent = fused_linear_logprobs(hh, w, t, **{**kw, "entropy_requires_grad": True})
            for x, y, z in zip(trainable, ent, ref):
                assert torch.equal(x.detach(), z) and torch.equal(y.detach(), z), dtype


@pytest.mark.parametrize("cap,mul,scaling", [(0.0, 0.0, None), (30.0, 0.5, 1024.0)])
def test_flash_cross_entropy(monkeypatch, cap, mul, scaling):
    for dtype in _dtypes():
        h, w, _, t, _ = _inputs(dtype, 200, 64, 40001, seed = 3)
        t[::4] = -100
        n_items = int((t != -100).sum())

        def run(**kw):
            hh = h.detach().clone().requires_grad_(True)
            loss = fused_linear_cross_entropy(hh, w, None, t, divisor = n_items, scaling = scaling,
                                              logit_scale_multiply = mul, logit_softcapping = cap, **kw)
            (grad,) = torch.autograd.grad(loss, (hh,))
            return loss.detach().double(), grad

        n = fl.CALLS["flash"]
        loss, grad = run()
        assert fl.CALLS["flash"] == n + 1
        run(label_smoothing = 0.1)  # label smoothing keeps the chunked path
        assert fl.CALLS["flash"] == n + 1
        monkeypatch.setattr(fl, "_FLASH", False)
        loss_c, grad_c = run()
        monkeypatch.setattr(fl, "_FLASH", True)
        hd = h.detach().double().requires_grad_(True)
        keep = t != -100
        z = _transform(hd @ w.double().t(), mul, cap)
        ref = torch.nn.functional.cross_entropy(z[keep], t[keep], reduction = "sum") / n_items
        (grad64,) = torch.autograd.grad(ref, (hd,))
        assert abs(float(loss) - float(ref)) < 1e-4 * max(abs(float(ref)), 1.0)
        assert _rel(grad, grad64) < 1e-2 and _rel(grad, grad64) <= 1.05 * _rel(grad_c, grad64) + 1e-6, \
            (dtype, _rel(grad, grad64), _rel(grad_c, grad64))
        assert not grad[~keep].any()


def test_trainable_head_cross_entropy_keeps_the_chunked_path():
    h, w, _, t, _ = _inputs(_dtypes()[-1], 64, 64, 40001, seed = 4)
    w.requires_grad_(True)
    n = fl.CALLS["flash"]
    fused_linear_cross_entropy(h.requires_grad_(True), w, None, t, divisor = 64).backward()
    assert fl.CALLS["flash"] == n and w.grad is not None


def test_flash_transient_is_one_vocab_tile(monkeypatch):
    # The forward gradient's logits buffer is [rows, tile], not [rows, V].
    shapes = []
    real = fl._Buffers.__init__

    def record(self, rows, n_cols, *a, **k):
        shapes.append((rows, n_cols))
        real(self, rows, n_cols, *a, **k)

    monkeypatch.setattr(fl._Buffers, "__init__", record)
    h, w, _, t, _ = _inputs(_dtypes()[-1], 64, 64, 70001, seed = 5)
    _frozen(h, w, t, torch.ones(64, device = "cuda", dtype = torch.float64))
    assert shapes and all(cols <= max(fl._FWD_TILE_COLS, fl._SPLIT_COLS) for _, cols in shapes), shapes


@pytest.mark.parametrize("kind", ["ce", "lp"])
def test_flash_fullgraph_compile_no_breaks(kind):
    import torch._dynamo
    from torch._dynamo.testing import CompileCounterWithBackend
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()
    cnt = CompileCounterWithBackend("inductor")
    h, w, _, t, mask = _inputs(_dtypes()[-1], 96, 64, 40001, seed = 6)

    def step(h, t, mask):
        if kind == "ce":
            return fused_linear_cross_entropy(h, w, None, torch.where(mask, t, -100), divisor = mask.sum())
        return -(fused_linear_logprobs(h, w, t, mask = mask)).sum() / mask.sum()

    compiled = torch.compile(step, backend = cnt, fullgraph = True)
    n = fl.CALLS["flash"]
    hc = h.clone().requires_grad_(True)
    compiled(hc, t, mask).backward()
    he = h.clone().requires_grad_(True)
    step(he, t, mask).backward()
    assert fl.CALLS["flash"] == n + 2
    assert torch.equal(hc.grad, he.grad)
    assert sum(counters["graph_break"].values()) == 0, dict(counters["graph_break"])
    assert cnt.frame_count == 1
