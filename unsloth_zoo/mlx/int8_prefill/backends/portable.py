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
"""The W8A8 algorithm in plain MLX ops, runnable on any backend.

This is what makes the project testable. The Metal kernels need an M5 and there is none
in CI, but the *arithmetic* is defined by the algorithm rather than by Metal, so
everything that decides output quality -- the block Walsh-Hadamard rotation, per-row
activation quantization, the per-channel weight requant, the scale application -- can be exercised on Linux, and on
an M1 runner, against the same reference the Metal path is checked against.

Numerical caveat, stated because it bounds what a test using this can claim: the
accumulation here is `mx.matmul` in float32, not int32. Products of int8 values are
exact, but a sum of K of them can reach `127*127*K`, which exceeds float32's exactly
representable integer range (2**24) once K is past about 1000. On real weight and
activation distributions the sums are far smaller than worst case and the difference is
immaterial, but a test wanting *bit-exact* agreement with an int32 GEMM must either keep
K small or compare against numpy int32 -- see
tests/test_mlx_int8_prefill_numerics.py::TestPortableBackend::test_gemm_matches_exact_int32,
which does the latter.
"""

import mlx.core as mx

from ..scales import dequantize_f32, rotate

INT8_MAX = 127.0


def quantize_rows(x, rot=0):
    """Per-row (per-token) symmetric int8 activation quantization, after the block
    Walsh-Hadamard rotation when `rot` is nonzero.

    Returns (xq int8 [M, K], xs float32 [M]). Rows that are entirely zero get a floor
    scale rather than a division by zero.
    """
    xf = x.astype(mx.float32)
    if rot:
        xf = rotate(xf, rot)
    xs = mx.maximum(mx.abs(xf).max(axis=-1), 1e-8) / INT8_MAX
    xq = mx.clip(mx.round(xf / xs[:, None]), -INT8_MAX, INT8_MAX)
    return xq.astype(mx.int8), xs


def requantize_weight(weight, scales, biases, ws, bits, group_size, rot=0):
    """Packed affine weights -> per-output-channel symmetric int8 [N, K], rotated like the
    activations when `rot` is nonzero.

    The Metal backend fuses this into one kernel that reads the packed words directly.
    Here we go through `mx.dequantize`, which is the same arithmetic with an
    intermediate.
    """
    dq = dequantize_f32(weight, scales, biases, bits, group_size)
    if rot:
        dq = rotate(dq, rot)
    wq = mx.clip(mx.round(dq / ws[:, None]), -INT8_MAX, INT8_MAX)
    return wq.astype(mx.int8)


def int8_gemm(xq, xs, wq, ws, rot=0):
    """(xq @ wq.T) with the per-row and per-channel scales folded back in, and the
    rotation's factor of `rot` (H @ H.T = rot * I) divided out."""
    acc = mx.matmul(xq.astype(mx.float32), wq.astype(mx.float32).T)
    out = acc * xs[:, None] * ws[None, :]
    return out * (1.0 / rot) if rot else out


def matmul(x, entry, out_dtype=mx.bfloat16):
    """Full W8A8 path for a registered weight. `x` may carry leading batch dims."""
    k = x.shape[-1]
    flat = x.reshape(-1, k)
    xq, xs = quantize_rows(flat, entry.rot)
    wq = requantize_weight(
        entry.w, entry.scales, entry.biases, entry.ws, entry.bits, entry.group_size,
        entry.rot,
    )
    out = int8_gemm(xq, xs, wq, entry.ws, entry.rot)
    return out.astype(out_dtype).reshape(*x.shape[:-1], entry.n)
