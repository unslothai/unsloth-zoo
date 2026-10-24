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

"""Compilable Sinkhorn-Knopp for the DeepSeek-V4 / V4.1 and GLM-5.3-Flash mHC mixers (see `mhc_fast_mode`).

`unsloth_sinkhorn_knopp` unrolls each token's n x n matrix into n * n pointwise tensors so Inductor
fuses the chain instead of emitting one reduction kernel per normalisation. Its backward
recomputes the iterates and uses `gx = (gy - sum(gy * y)) / d` per step (no `x / d / d` term).
Eager runs transformers' own loop, so it stays bitwise identical to the stock module.

`unsloth_mhc_projection` replaces the mixer's FP32 norm + projection outside torch.compile: plain
autograd keeps the FP32 upcast of the `[tokens, hc_mult * hidden]` input for backward (twice the
bytes of the bf16 streams, per mixer site); this saves only the bf16 input and reruns the cast,
norm and projection in backward, bitwise identical. Compiled blocks keep the stock math.
"""

import os

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

__all__ = [
    "unsloth_sinkhorn_knopp",
    "unsloth_mhc_projection",
    "mhc_fast_mode",
    "MHC_SINKHORN_SOURCE",
    "MHC_SINKHORN_REPLACEMENT",
    "MHC_PROJECTION_SOURCE",
    "MHC_PROJECTION_REPLACEMENT",
]


def _torch_at_least(major, minor):
    try:
        version = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    except Exception:
        return False
    return version >= (major, minor)


def mhc_fast_mode():
    """`UNSLOTH_DSV4_MHC_FAST`: `0` eager, `stock` compile as is, `unrolled` via `unsloth_sinkhorn_knopp`
    (slower to compile, not run-to-run reproducible without TORCHINDUCTOR_DETERMINISTIC=1).
    Unset: `stock` on torch >= 2.13 (checked finite there), else eager."""
    value = os.environ.get("UNSLOTH_DSV4_MHC_FAST", "").strip().lower()
    if value in ("0", "false", "off", "no", "eager"):
        return None
    if value in ("unrolled", "stock"):
        return value
    return "stock" if _torch_at_least(2, 13) else None


# Must occur exactly once in the mixer forward; transformers 5.8.0's slice-based mixer does not match.
MHC_SINKHORN_SOURCE = (
    "        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)\n"
    "        for _ in range(self.hc_sinkhorn_iters - 1):\n"
    "            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)\n"
    "            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)\n"
)
MHC_SINKHORN_REPLACEMENT = (
    "        comb = unsloth_sinkhorn_knopp(comb, self.hc_sinkhorn_iters, self.hc_eps)\n"
)

# Must occur exactly once in the mixer forward (DeepSeek-V4 / V4.1, GLM-5.3-Flash; transformers 5.17).
MHC_PROJECTION_SOURCE = (
    "        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())\n"
    "        pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split([hc, hc, hc * hc], dim=-1)\n"
)
MHC_PROJECTION_REPLACEMENT = (
    "        pre_w, post_w, comb_w = unsloth_mhc_projection(hidden_streams, self.input_norm, self.fn).split(\n"
    "            [hc, hc, hc * hc], dim=-1\n"
    "        )\n"
)


def _mhc_projection(hidden_streams, input_norm, fn):
    flat = input_norm(hidden_streams.flatten(start_dim=2).float())
    return F.linear(flat, fn.float())


def unsloth_mhc_projection(hidden_streams, input_norm, fn):
    """`F.linear(input_norm(hidden_streams.flatten(2).float()), fn.float())`, saving `hidden_streams`
    (alive anyway) for backward instead of its FP32 upcast; backward reruns the same math."""
    if (
        torch.compiler.is_compiling()
        or not torch.is_grad_enabled()
        or hidden_streams.dtype == torch.float32
        or not (hidden_streams.requires_grad or fn.requires_grad)
    ):
        return _mhc_projection(hidden_streams, input_norm, fn)
    return checkpoint(
        _mhc_projection, hidden_streams, input_norm, fn,
        use_reentrant = False, preserve_rng_state = False,
    )


_MAX_UNROLLED_N = 8


def _schedule(iters):
    return [-2] + [-1, -2] * (int(iters) - 1)


def _stock_chain(comb, iters, eps):
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(int(iters) - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return comb


def _entries(t):
    return [list(row.unbind(-1)) for row in t.unbind(-2)]


def _join(c):
    return torch.stack([torch.stack(row, dim=-1) for row in c], dim=-2)


def _normalise(c, dim, eps):
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
    """transformers' Sinkhorn-Knopp schedule; the unrolled chain only under torch.compile for square n <= 8."""
    n = comb.shape[-1]
    if (
        torch.compiler.is_compiling()
        and comb.dim() >= 2
        and comb.shape[-2] == n
        and n <= _MAX_UNROLLED_N
    ):
        return _SinkhornKnopp.apply(comb, int(iters), float(eps))
    return _stock_chain(comb, iters, eps)
