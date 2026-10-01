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

"""Compilable Sinkhorn-Knopp for the DeepSeek-V4 / V4.1 manifold hyper-connection (mHC) mixers.

The mixers project a [..., hc, hc] matrix onto the doubly-stochastic manifold with a chain of
`x / (x.sum(dim) + eps)` normalisations (one column step, then `iters - 1` row + column pairs).
They used to run eager (`DISABLE_COMPILE_MODULES`), which costs ~230 tiny fp32 launches per call.

Compiled as written, Inductor emits one reduction kernel per normalisation (~40 forward, ~40
backward per call). `unsloth_sinkhorn_knopp` instead treats each token's n x n matrix as n * n
pointwise tensors, so the chain is pointwise and fuses into a few pointwise kernels (about 30 per
mixer call forward + backward on M07, against about 90 compiled as written).
Its backward is explicit: the iterates are recomputed from the saved input and each normalisation
`y = x / d`, `d = sum(x) + eps` is differentiated as `gx = (gy - sum(gy * y)) / d`, the exact
derivative with no `x / d / d` term. Outside a compiled region it runs the transformers loop, so
eager stays bitwise identical to the stock module.

The compiler swaps the transformers loop for this call through `MODULE_FORWARD_SOURCE_REWRITES`
(`unsloth_zoo/compiler.py`) when `UNSLOTH_DSV4_MHC_FAST=unrolled`; the default compiles the stock
loop (torch >= 2.13) and `UNSLOTH_DSV4_MHC_FAST=0` keeps the eager mixer. See `mhc_fast_mode`.
"""

import os

import torch

__all__ = [
    "unsloth_sinkhorn_knopp",
    "mhc_fast_mode",
    "MHC_SINKHORN_SOURCE",
    "MHC_SINKHORN_REPLACEMENT",
]


def _torch_at_least(major, minor):
    try:
        version = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    except Exception:
        return False
    return version >= (major, minor)


def mhc_fast_mode():
    """`UNSLOTH_DSV4_MHC_FAST` for the DeepSeek-V4 / V4.1 mixer:
    `0` keeps it eager; `stock` compiles the transformers loop as is; `unrolled` compiles it through
    `unsloth_sinkhorn_knopp` (about 3x fewer mixer kernels than `stock`, about 3x slower to compile,
    and not bitwise reproducible run to run unless TORCHINDUCTOR_DETERMINISTIC=1). Unset: `stock` on
    torch >= 2.13, where the compiled stock mixer was checked finite and close to eager, else eager."""
    value = os.environ.get("UNSLOTH_DSV4_MHC_FAST", "").strip().lower()
    if value in ("0", "false", "off", "no", "eager"):
        return None
    if value in ("unrolled", "stock"):
        return value
    return "stock" if _torch_at_least(2, 13) else None


# Exact transformers text (deepseek_v4 and the deepseek_v41 port share it); a source rewrite only
# applies when this occurs exactly once in the mixer's forward.
MHC_SINKHORN_SOURCE = (
    "        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)\n"
    "        for _ in range(self.hc_sinkhorn_iters - 1):\n"
    "            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)\n"
    "            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)\n"
)
MHC_SINKHORN_REPLACEMENT = (
    "        comb = unsloth_sinkhorn_knopp(comb, self.hc_sinkhorn_iters, self.hc_eps)\n"
)


# Above this the unrolled form (n * n scalars per token) stops paying for its graph size.
_MAX_UNROLLED_N = 8


def _schedule(iters):
    # One column step, then `iters - 1` row + column pairs (dim -1 sums a row, dim -2 a column).
    return [-2] + [-1, -2] * (int(iters) - 1)


def _stock_chain(comb, iters, eps):
    # transformers' loop, op for op
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(int(iters) - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return comb


def _entries(t):
    # [..., n, n] -> n x n list of [...] views: every step becomes pointwise, so Inductor fuses the
    # whole chain into one kernel instead of one reduction kernel per normalisation.
    return [list(row.unbind(-1)) for row in t.unbind(-2)]


def _join(c):
    return torch.stack([torch.stack(row, dim=-1) for row in c], dim=-2)


def _normalise(c, dim, eps):
    """One `x / (x.sum(dim) + eps)` step; returns the new entries and the n denominators."""
    n = len(c)
    if dim == -1:
        dens = []
        for i in range(n):
            s = c[i][0]
            for j in range(1, n):
                s = s + c[i][j]
            dens.append(s + eps)
        return [[c[i][j] / dens[i] for j in range(n)] for i in range(n)], dens
    dens = []
    for j in range(n):
        s = c[0][j]
        for i in range(1, n):
            s = s + c[i][j]
        dens.append(s + eps)
    return [[c[i][j] / dens[j] for j in range(n)] for i in range(n)], dens


class _SinkhornKnopp(torch.autograd.Function):
    """Unrolled chain with an explicit backward. Saves only the input; the backward recomputes the
    iterates and differentiates each `y = x / d`, `d = sum(x) + eps`, as `gx = (gy - sum(gy * y)) / d`."""

    @staticmethod
    def forward(ctx, comb, iters, eps):
        ctx.save_for_backward(comb)
        ctx.iters = iters
        ctx.eps = eps
        c = _entries(comb)
        for dim in _schedule(iters):
            c, _ = _normalise(c, dim, eps)
        return _join(c)

    @staticmethod
    def backward(ctx, grad):
        (comb,) = ctx.saved_tensors
        dims = _schedule(ctx.iters)
        c = _entries(comb)
        outs, dens = [], []
        for dim in dims:
            c, den = _normalise(c, dim, ctx.eps)
            outs.append(c)
            dens.append(den)
        g = _entries(grad.to(comb.dtype))
        n = len(g)
        for dim, y, den in zip(reversed(dims), reversed(outs), reversed(dens)):
            if dim == -1:
                new = []
                for i in range(n):
                    t = g[i][0] * y[i][0]
                    for j in range(1, n):
                        t = t + g[i][j] * y[i][j]
                    new.append([(g[i][j] - t) / den[i] for j in range(n)])
                g = new
            else:
                new = [[None] * n for _ in range(n)]
                for j in range(n):
                    t = g[0][j] * y[0][j]
                    for i in range(1, n):
                        t = t + g[i][j] * y[i][j]
                    for i in range(n):
                        new[i][j] = (g[i][j] - t) / den[j]
                g = new
        return _join(g), None, None


def unsloth_sinkhorn_knopp(comb, iters, eps):
    """Sinkhorn-Knopp on the last two dims exactly as transformers runs it: one column step, then
    `iters - 1` row + column steps (so `iters <= 1` is a single column step there too).

    Eager (and kill switch / compile off) runs transformers' own loop, so it is bitwise the stock
    module. Under torch.compile a square [..., n, n] input with n <= 8 takes the unrolled chain."""
    n = comb.shape[-1]
    if (
        torch.compiler.is_compiling()
        and comb.dim() >= 2
        and comb.shape[-2] == n
        and n <= _MAX_UNROLLED_N
    ):
        return _SinkhornKnopp.apply(comb, int(iters), float(eps))
    return _stock_chain(comb, iters, eps)
