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
"""The two MetalPerformancePrimitives-free kernels against the portable reference.

Needs a real Metal device but not an M5: these kernels carry no MPP include, so any Apple
Silicon runner executes them. They hold the packed-nibble unpacking and the Walsh-Hadamard
butterflies, the likeliest places for a silent wrong-answer bug and the parts Linux cannot
check at all. Run unrotated (rot 0) and rotated (rot 32).

A tolerance of one int8 LSB: Metal's rint rounds halves to even while mx.round does not,
and the butterfly sums in a different order than mx.hadamard_transform, so ties can
differ by one and nothing else may. The share of entries that differ at all is bounded
too, so a systematic off-by-one cannot hide under the LSB bar.
"""

import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip(
        reason="needs a Metal device; the macOS leg of mlx-ci.yml runs this file",
        allow_module_level=True,
    )

import mlx.nn as nn

from unsloth_zoo.mlx.int8_prefill import scales
from unsloth_zoo.mlx.int8_prefill.backends import metal_mpp, portable

ROTS = [0, scales.ROTATE_BLOCK]
MAX_DIFFERING = 1e-3


def _assert_close_codes(got, want):
    d = mx.abs(got.astype(mx.int32) - want.astype(mx.int32))
    diff = int(d.max().item())
    frac = float((d > 0).astype(mx.float32).mean().item())
    assert diff <= 1, f"max abs code diff {diff}"
    assert frac < MAX_DIFFERING, f"{frac:.2e} of codes differ"


@pytest.mark.parametrize("rot", ROTS)
@pytest.mark.parametrize("bits,group_size", [(4, 32), (4, 64), (4, 128), (8, 32), (8, 64), (8, 128)])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_requant_matches_portable(rot, bits, group_size, dtype):
    lin = nn.Linear(1024, 2048, bias=False)
    lin.weight = lin.weight.astype(dtype)
    ql = nn.QuantizedLinear.from_linear(lin, group_size=group_size, bits=bits)
    ws = scales.channel_scale(
        ql["weight"], ql["scales"], ql["biases"], bits, group_size,
        exact=False, rotate=bool(rot),
    )
    got = metal_mpp.requantize_weight(
        ql["weight"], ql["scales"], ql["biases"], ws, bits, group_size, rot
    )
    want = portable.requantize_weight(
        ql["weight"], ql["scales"], ql["biases"], ws, bits, group_size, rot
    )
    mx.eval(got, want)
    _assert_close_codes(got, want)


@pytest.mark.parametrize("rot", ROTS)
@pytest.mark.parametrize("k", [1024, 1056])  # 1056 = 33 blocks: a partial final pass
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_row_quant_matches_portable(rot, k, dtype):
    x = mx.random.normal((640, k))
    x = x.at[:, 7].multiply(50.0).astype(dtype)
    gq, gs = metal_mpp.quantize_rows(x, rot)
    wq, ws = portable.quantize_rows(x, rot)
    mx.eval(gq, gs, wq, ws)
    _assert_close_codes(gq, wq)
    srel = float((mx.abs(gs - ws) / ws).max().item())
    assert srel < 1e-5, f"max relative scale diff {srel:.3e}"
