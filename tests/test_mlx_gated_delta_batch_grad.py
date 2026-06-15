# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Batched gradient-parity regression test for the gated-delta custom VJP.

Guards the fix from #776: the `_chunked_vjp` backward used to accumulate
per-step gradients with `d_x = d_x.at[:, t].add(...)`. On the MLX 0.31.2 Metal
backend that scatter-add reads the update with a wrong batch stride, silently
corrupting `d_q/d_k/d_v/d_g/d_beta` for every batch row past index 0 (only with
batch size > 1; row 0 stays correct). The fix collects per-step grads in lists,
reverses them after the reverse-time loop, and `mx.stack`s them.

The committed `test_mlx_gated_delta.py` only checks finite gradients at B = 1, so
it passes on both the buggy and fixed code. These tests instead pin the custom
VJP against a plain autodiff reference (`mx.value_and_grad` of the unrolled
recurrence) at B > 1, which is exactly the failure surface:

  * on Metal, the pre-fix code fails the parity assertion (corrupt rows > 0);
  * on any backend, a misordered stack (e.g. a dropped `reverse()`) or a
    reintroduced scatter-add also fails it.

These run against a REAL `mlx` install (Apple Silicon Metal, or Linux CPU/CUDA
if `mlx` is present); they are skipped where `mlx` cannot be imported, e.g. the
torch-shim CI used by `test_mlx_gated_delta.py`. Target runtime is a few seconds.
"""

from __future__ import annotations

import itertools

import pytest

try:
    import mlx.core as mx
    import mlx.nn as nn  # noqa: F401  (gated_delta_vjp imports it at module load)
    from unsloth_zoo.gated_delta_vjp import gated_delta_ops_efficient, _gated_delta_step
    _HAS_MLX = True
except Exception:
    _HAS_MLX = False

_SKIP_REASON = (
    "Requires a real `mlx` install (Apple Silicon Metal, or Linux mlx[cpu]/"
    "mlx[cuda]); the batched gated-delta scatter regression cannot be checked "
    "under the torch-shim CI."
)

if not _HAS_MLX:
    print(
        "\n[test_mlx_gated_delta_batch_grad] SKIPPING: no real `mlx` import. "
        "These batched gradient-parity tests need real MLX (run on Apple "
        "Silicon, or Linux with mlx[cpu]/mlx[cuda]).\n"
    )

mlx_only = pytest.mark.skipif(not _HAS_MLX, reason=_SKIP_REASON)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_inputs(B, T, Hk, Dk, Hv, Dv, vectorized_gating, with_mask, identical_rows=False, seed=0):
    mx.random.seed(seed)

    def rnd(*shape):
        x = mx.random.normal(shape).astype(mx.float32)
        if identical_rows and B > 1 and shape[0] == B:
            x = mx.broadcast_to(x[:1], x.shape)
        return x

    q = rnd(B, T, Hk, Dk)
    k = rnd(B, T, Hk, Dk)
    v = rnd(B, T, Hv, Dv)
    # scalar gating -> g[:, t] is (B, Hv); vectorized -> (B, Hv, Dk)
    g = mx.sigmoid(rnd(B, T, Hv, Dk) if vectorized_gating else rnd(B, T, Hv))
    beta = mx.sigmoid(rnd(B, T, Hv))

    mask = None
    if with_mask:
        row = [True] * T
        if T >= 4:
            row[2] = False
            row[T - 1] = False
        mask = mx.broadcast_to(mx.array(row, dtype=mx.bool_)[None, :], (B, T))

    mx.eval(q, k, v, g, beta)
    if mask is not None:
        mx.eval(mask)
    return q, k, v, g, beta, mask


def _plain_reference(q, k, v, g, beta, mask):
    """Plain unrolled recurrence (autodiff ground truth).

    Mirrors gated_delta_ops_efficient's preprocessing (GQA head repeat + zero
    initial state) but with no custom VJP and no chunking, so MLX autodiff
    differentiates it directly.
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    qr, kr = q, k
    repeat_factor = Hv // Hk
    if repeat_factor > 1:
        qr = mx.repeat(q, repeat_factor, -2)
        kr = mx.repeat(k, repeat_factor, -2)
    state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    ys = []
    for t in range(T):
        m = mask[:, t] if mask is not None else None
        y, state = _gated_delta_step(qr[:, t], kr[:, t], v[:, t], g[:, t], beta[:, t], state, m)
        ys.append(y)
    return mx.stack(ys, axis=1)


def _grads(forward, q, k, v, g, beta, mask):
    def loss(q_, k_, v_, g_, beta_):
        out = forward(q_, k_, v_, g_, beta_, mask)
        y = out[0] if isinstance(out, tuple) else out
        return y.astype(mx.float32).sum()

    _, grads = mx.value_and_grad(loss, argnums=(0, 1, 2, 3, 4))(q, k, v, g, beta)
    mx.eval(grads)
    return grads


def _efficient(q, k, v, g, beta, mask):
    return gated_delta_ops_efficient(q, k, v, g, beta, mask=mask)


def _rel_l2(a, b):
    num = sum(mx.sum((x.astype(mx.float32) - y.astype(mx.float32)) ** 2).item() for x, y in zip(a, b)) ** 0.5
    den = sum(mx.sum(y.astype(mx.float32) ** 2).item() for y in b) ** 0.5 + 1e-12
    return num / den


_NAMES = ("q", "k", "v", "g", "beta")

# B>1, both gating forms, GQA on/off, single- and multi-chunk (CHUNK_SIZE=min(64,T)),
# and masked/unmasked. Small dims keep this well under a second per case.
_PARITY_CASES = [
    (B, T, gqa, mask_on, vec)
    for B in (2, 3, 4)
    for T, gqa, mask_on, vec in [
        (8, False, False, False),    # base, single chunk
        (8, True, True, False),      # GQA + mask
        (8, False, False, True),     # vectorized gating
        (96, True, False, False),    # multi-chunk + GQA
        (96, False, True, True),     # multi-chunk + mask + vectorized
    ]
]


@mlx_only
@pytest.mark.parametrize("B,T,gqa,mask_on,vec", _PARITY_CASES)
def test_batched_grad_matches_plain_recurrence(B, T, gqa, mask_on, vec):
    """Custom-VJP gradients must match plain-autodiff gradients for B > 1.

    Pre-#776 on Metal this fails: rows > 0 carry corrupt gradients from the
    wrong-batch-stride scatter-add.
    """
    Hk = 2
    Hv = Hk * (2 if gqa else 1)
    Dk = Dv = 16
    inputs = _make_inputs(B, T, Hk, Dk, Hv, Dv, vec, mask_on)

    g_eff = _grads(_efficient, *inputs)
    g_ref = _grads(_plain_reference, *inputs)

    rel = _rel_l2(g_eff, g_ref)
    assert rel < 2e-4, f"batched VJP grads diverge from plain recurrence (rel L2 {rel:.2e})"
    for name, a, b in zip(_NAMES, g_eff, g_ref):
        assert mx.allclose(a, b, atol=1e-4, rtol=1e-3).item(), f"grad mismatch on d_{name}"


@mlx_only
@pytest.mark.parametrize("gqa,vec", list(itertools.product([False, True], [False, True])))
def test_identical_rows_give_identical_grads(gqa, vec):
    """With identical batch rows, every row's gradient must be identical.

    The pre-#776 bug corrupted only rows > 0 (row 0 stayed correct), so identical
    inputs produced row-dependent gradients; here the per-row spread must be ~0.
    """
    B, T = 4, 24
    Hk = 2
    Hv = Hk * (2 if gqa else 1)
    Dk = Dv = 16
    inputs = _make_inputs(B, T, Hk, Dk, Hv, Dv, vec, with_mask=False, identical_rows=True)

    g_eff = _grads(_efficient, *inputs)
    for name, grad in zip(_NAMES, g_eff):
        norms = [mx.sqrt(mx.sum(grad[b].astype(mx.float32) ** 2)).item() for b in range(B)]
        lo, hi = min(norms), max(norms)
        spread = (hi - lo) / (sum(norms) / len(norms) + 1e-12)
        assert spread < 1e-4, f"d_{name} grad norms differ across identical rows (spread {spread:.2e})"
