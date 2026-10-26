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

"""Margin sweep (arXiv 2610.31642 Section 2) through the public entry points, CPU torch bodies and CUDA Triton.

One row per call, so the head gradient's target row is exactly (1 - p_t) h = -expm1(log p_t) h: the target
gradient, which collapses to 0 with the loss once the margin passes ~16.6 nats in fp32."""

import pytest
import torch

import unsloth_zoo.fused_losses.fused_linear as fl
from unsloth_zoo.fused_losses.fused_linear import fused_linear_cross_entropy, fused_linear_logprobs

H = 64
MARGINS = (5.0, 10.0, 17.0, 20.0, 30.0, 50.0, 80.0)


def _devices():
    out = [("cpu", torch.float32)]
    if torch.cuda.is_available():
        out += [("cuda", torch.float32), ("cuda", torch.bfloat16)]
    return out


def _peaked(V, margin, device, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(V, H, generator = g, dtype = torch.float64) / H**0.5
    h = torch.randn(1, H, generator = g, dtype = torch.float64)
    h /= h.norm()
    t = torch.randint(0, V, (1,), generator = g)
    zo = (h @ w.t())[0]
    zo[t] = -float("inf")
    w[t] = (margin + float(torch.logsumexp(zo, 0))) * h[0]
    return h.to(device, dtype), w.to(device, dtype), t.to(device)


def _oracle(h, w, t):
    z = (h.double() @ w.double().t())
    zo = z.scatter(1, t[:, None], -float("inf"))
    lp = -torch.nn.functional.softplus(torch.logsumexp(zo, 1) - z.gather(1, t[:, None])[:, 0])
    return lp, -torch.expm1(lp)[:, None] * h.double()


def _rel(x, y):
    return float((x.double() - y).abs().max() / y.abs().max().clamp_min(1e-300))


def _sweep(V, device, dtype, entry):
    rows = []
    for i, margin in enumerate(MARGINS):
        h, w, t = _peaked(V, margin, device, dtype, seed = i)
        ww = w.clone().requires_grad_(True)
        if entry == "ce":
            val = -fused_linear_cross_entropy(h, ww, None, t, divisor = 1.0)
        else:
            val = fused_linear_logprobs(h, ww, t).sum()
        (gw,) = torch.autograd.grad(val, (ww,))
        lp, gt = _oracle(h, w, t)
        rows.append((margin, float(lp), _rel(val.detach().reshape(1), lp), _rel(gw[t], gt)))
    return rows


@pytest.mark.parametrize("entry", ["ce", "logprobs"])
@pytest.mark.parametrize("V", [32000, 152064])
@pytest.mark.parametrize("device,dtype", _devices())
def test_margin_sweep_matches_fp64_oracle(V, device, dtype, entry):
    tol = 1e-3 if dtype == torch.float32 else 2e-2
    rows = _sweep(V, device, dtype, entry)
    for margin, lp, lp_rel, g_rel in rows:
        print(f"{device} {dtype} V={V} {entry} margin={margin:4.0f} logp={lp:.3e} lp_rel={lp_rel:.2e} g_rel={g_rel:.2e}")
    for margin, lp, lp_rel, g_rel in rows:
        assert lp_rel < tol and g_rel < tol, (margin, lp, lp_rel, g_rel)


@pytest.mark.skipif(not torch.cuda.is_available() or not fl._HAS_TRITON, reason = "needs CUDA + Triton")
def test_expm1_fallback_is_subnormal_safe():
    import triton
    import triton.language as tl

    @triton.jit
    def k(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        o = tl.arange(0, BLOCK)
        ok = o < n
        tl.store(y_ptr + o, fl._fl_expm1_kahan(tl.load(x_ptr + o, mask = ok, other = 0.0)), mask = ok)

    x = torch.tensor([0.0, -1e-37, -1e-30, -1e-12, -1e-7, -1e-3, -0.5, -0.999, -1.0, -3.0, -20.0, -88.0, -200.0,
                      -float("inf"), 1e-30, 1e-6, 0.5, 2.0, 50.0, 100.0], device = "cuda")
    y = torch.empty_like(x)
    k[(1,)](x, y, x.numel(), BLOCK = 32)
    ref = torch.expm1(x.double()).float().double()
    finite = torch.isfinite(ref)
    rel = (y.double() - ref).abs() / ref.abs().clamp_min(1e-300)
    small = finite & (ref != 0) & (x.abs() < 1)
    assert torch.equal(y[x == 0], x[x == 0])
    # |x| < 1 is the cancelling range (Kahan); past it u - 1 carries tl.exp's own few-ulp error.
    assert float(rel[small].max()) < 1e-6, rel
    assert float(rel[finite & ~small & (ref != 0)].max()) < 4e-6, rel
    assert torch.isinf(y[~finite]).all()
