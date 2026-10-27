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

"""Fixed rows (no compaction): the log-prob op on every row with dropped rows zeroed, taken while a CUDA graph is
captured (nonzero and the kept-row count are a host sync, not allowed there) or with
UNSLOTH_FUSED_LINEAR_FIXED_ROWS=1. It must give what compaction gives, and capture with its backward."""

import pytest
import torch

import unsloth_zoo.fused_losses.fused_linear as fl

N, H, V = 37, 24, 101


def _case(seed, dtype, device = "cpu"):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(N, H, generator = g)
    w = torch.randn(V, H, generator = g) / 4
    b = torch.randn(V, generator = g) / 4
    t = torch.randint(0, V, (N,), generator = g)
    mask = torch.rand(N, generator = g) > 0.3
    t[3], t[8], t[9] = -100, V + 3, -1  # out of range: dropped whatever the mask says
    h[~mask] = 3.0e4 if dtype != torch.float16 else 6.0e4  # a dropped row that overflows the logits if computed
    to = lambda x: x.to(device)  # noqa: E731
    return to(h).to(dtype), to(w).to(dtype), to(b).to(dtype), to(t), to(mask)


def _run(fixed, monkeypatch, h, w, b, t, mask, head, ent, lazy):
    monkeypatch.setattr(fl, "_FIXED_ROWS", fixed)
    h = h.clone().requires_grad_()
    w = w.clone().requires_grad_(head)
    b = b.clone().requires_grad_(head)
    kw = dict(bias = b, mask = mask, return_entropy = ent, return_mean_logits = True, return_is_top1 = True,
              entropy_requires_grad = None if lazy else True)
    out = fl.fused_linear_logprobs(h, w, t, **kw)
    lp, e, mean, top1 = out if ent else (out[0], None, out[1], out[2])
    g = torch.linspace(-1, 1, N, device = h.device)  # nonzero on dropped rows too: they must still get no gradient
    loss = (lp * g).sum() + ((e * g.flip(0)).sum() if ent else 0)
    loss.backward()
    grads = (h.grad, w.grad, b.grad) if head else (h.grad,)
    return (lp, mean, top1) + ((e,) if ent else ()) + grads


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("head,ent,lazy", [(False, False, False), (True, False, False), (False, True, True),
                                           (True, True, False)])
def test_fixed_rows_equal_compaction(dtype, head, ent, lazy, monkeypatch):
    case = _case(0, dtype)
    ref = _run(False, monkeypatch, *case, head, ent, lazy)
    got = _run(True, monkeypatch, *case, head, ent, lazy)
    # bf16 bit for bit; fp32 to summation order (GEMMs / head-gradient sums over all rows instead of the kept ones
    # block, so round, differently).
    tol = dict(rtol = 0, atol = 0) if dtype != torch.float32 else dict(rtol = 1e-5, atol = 1e-7)
    for r, x in zip(ref, got):
        assert torch.isfinite(x.float()).all()
        if x.dtype == torch.bool:
            assert torch.equal(x, r)
        else:
            torch.testing.assert_close(x, r, **tol)
    dropped = ~case[4] | (case[3] < 0) | (case[3] >= V)
    assert (got[0][dropped] == 0).all() and (got[len(got) - (3 if head else 1)][dropped] == 0).all()


@pytest.mark.parametrize("head", [False, True])
def test_fixed_rows_dropped_row_keeps_its_normalizer(head, monkeypatch):
    # A dropped row still computes exp(z - lse) in the backward (times a zero gradient): its saved lse must stay its
    # real one, or a large bias / low temperature turns that into 0 * inf.
    h, w = torch.randn(2, 4), torch.randn(4, 4)
    b, t, mask = torch.tensor([0.0, 10.0, 0.0, 0.0]), torch.tensor([1, 2]), torch.tensor([True, False])
    def run(fixed):
        monkeypatch.setattr(fl, "_FIXED_ROWS", fixed)
        hh, ww, bb = h.clone().requires_grad_(), w.clone().requires_grad_(head), b.clone().requires_grad_(head)
        lp = fl.fused_linear_logprobs(hh, ww, t, bias = bb, mask = mask, temperature = 0.1)
        lp.sum().backward()
        return (lp, hh.grad) + ((ww.grad, bb.grad) if head else ())
    for r, x in zip(run(False), run(True)):
        assert torch.isfinite(x).all()
        torch.testing.assert_close(x, r, rtol = 1e-5, atol = 1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA")
@pytest.mark.parametrize("head", [False, True])
def test_logprobs_capture_in_a_cuda_graph(head):
    """Forward + backward of the public log-prob API captured in a CUDA graph, replayed on new data, equal to eager."""
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    h0, w0, b0, t0, m0 = _case(1, dtype, "cuda")
    h = h0.clone().requires_grad_()
    w = w0.clone().requires_grad_(head)
    t, m = t0.clone(), m0.clone()

    def step():
        lp, e = fl.fused_linear_logprobs(h, w, t, mask = m, return_entropy = True)
        (lp.sum() - 0.1 * e.sum()).backward()
        return lp, e

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            h.grad = w.grad = None
            step()
    torch.cuda.current_stream().wait_stream(s)
    h.grad = w.grad = None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        lp, e = step()
    for seed in (2, 3):
        h1, w1, b1, t1, m1 = _case(seed, dtype, "cuda")
        with torch.no_grad():
            h.copy_(h1)
        t.copy_(t1)
        m.copy_(m1)
        graph.replay()
        torch.cuda.synchronize()
        hr = h1.clone().requires_grad_()
        wr = w0.clone().requires_grad_(head)
        lr, er = fl.fused_linear_logprobs(hr, wr, t1, mask = m1, return_entropy = True)
        (lr.sum() - 0.1 * er.sum()).backward()
        torch.testing.assert_close(lp, lr, rtol = 1e-4, atol = 1e-4)
        torch.testing.assert_close(e, er, rtol = 1e-4, atol = 1e-4)
        torch.testing.assert_close(h.grad.float(), hr.grad.float(), rtol = 2e-2, atol = 2e-3)
        if head:
            torch.testing.assert_close(w.grad.float(), wr.grad.float(), rtol = 2e-2, atol = 2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA")
def test_fixed_rows_never_sync_the_host(monkeypatch):
    """UNSLOTH_FUSED_LINEAR_FIXED_ROWS=1: forward + backward with CUDA sync debugging set to raise."""
    monkeypatch.setattr(fl, "_FIXED_ROWS", True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    h0, w0, _, t, m = _case(4, dtype, "cuda")
    h = h0.clone().requires_grad_()
    w = w0.clone().requires_grad_()
    for _ in range(2):  # warm (Triton JIT, cached device queries) before checking
        lp, e = fl.fused_linear_logprobs(h, w, t, mask = m, return_entropy = True)
        torch.autograd.grad(lp.sum() + e.sum(), [h, w])
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        lp, e = fl.fused_linear_logprobs(h, w, t, mask = m, return_entropy = True)
        torch.autograd.grad(lp.sum() + e.sum(), [h, w])
    finally:
        torch.cuda.set_sync_debug_mode(0)
