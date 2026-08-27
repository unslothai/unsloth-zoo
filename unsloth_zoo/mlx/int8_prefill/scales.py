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
"""Per-output-channel int8 scales for an affine-quantized weight.

An affine weight stores `w = scales * q + biases` with `q` an integer in
`[0, 2**bits - 1]`. To requantize into symmetric int8 we need `max|w|` along each output
channel. Two ways to get it:

**bound** (default, free). Within a group the extremes of `w` can only occur at `q = 0`
or `q = 2**bits - 1`, so `max|w| <= max(|b|, |lvl*s + b|)`, computed from the scale and
bias tensors alone -- no dequantization, ~10 MB for a 27B model, computed once.

**exact** (opt-in). Dequantize a layer at a time and take the true absmax.

Measured against stock `mx.quantize` output the two agree almost everywhere, because
MLX's affine quantizer sets `bias = group min`, so code 0 is attained in every group and
the bound is tight. For float32 and float16 storage they are identical. For bfloat16 a
thin tail -- 0.1% of channels, worst case 8/7 -- comes out loose, because bf16 rounding
of the group scale can leave the top code unused, making the group's true maximum
`13*s + b` rather than `15*s + b`. Measured end to end, correcting that tail moves the
matmul's mean and max relative error by nothing (0.00875 either way), so the bound stays
the default. `tests/mlx_int8/test_numerics.py` asserts the distribution, so if MLX ever
changes its quantizer the tests fail rather than quality degrading silently on hardware
we cannot test on.

Where they differ is learned or clipped quantizers -- DWQ (`mlx_lm/quant/dwq.py:97`
learns scales by gradient descent), AWQ, GPTQ -- which do not attain the extremes by
construction. There the bound over-estimates, which costs int8 range. Hence the flag.

A note on what this does *not* fix: the real precision cost of this technique is
per-channel versus per-group granularity. A channel whose groups differ in range by a
factor R gets an int8 step of `max_g range_g / 254` against a native `range_g / lvl`.
At 4 bits that tolerates R up to ~17; at 8 bits it tolerates almost nothing. Exact
absmax does not change that -- only finer int8 granularity would.
"""

import logging
import os

import mlx.core as mx

logger = logging.getLogger(__name__)

INT8_MAX = 127.0
_EPS = 1e-8


def use_exact_scales() -> bool:
    return os.environ.get("UNSLOTH_MLX_INT8_EXACT_SCALES", "0").lower() in (
        "1", "true", "yes", "on",
    )


def channel_scale_bound(scales, biases, bits):
    """Per-channel int8 scale from the quant metadata alone. No dequantization."""
    s = scales.astype(mx.float32)
    b = biases.astype(mx.float32)
    lvl = float((1 << bits) - 1)
    per_group = mx.maximum(mx.abs(b), mx.abs(lvl * s + b))
    return mx.maximum(per_group.max(axis=-1), _EPS) / INT8_MAX


def channel_scale_exact(weight, scales, biases, bits, group_size):
    """Per-channel int8 scale from the true dequantized absmax.

    Materializes one layer in high precision, then drops it. Caller should `mx.eval` and
    move on so peak memory stays at one layer.
    """
    dq = mx.dequantize(
        weight, scales, biases, group_size=group_size, bits=bits, mode="affine"
    )
    return mx.maximum(mx.abs(dq).max(axis=-1).astype(mx.float32), _EPS) / INT8_MAX


def channel_scale(weight, scales, biases, bits, group_size, exact=None):
    """Per-channel int8 scale, `exact` defaulting to the environment."""
    if exact is None:
        exact = use_exact_scales()
    if exact:
        return channel_scale_exact(weight, scales, biases, bits, group_size)
    return channel_scale_bound(scales, biases, bits)


def group_range_ratio(scales, bits):
    """Per-channel R = max_g range_g / min_g range_g, the quantity that decides whether
    requantizing to per-channel int8 is safe.

    Diagnostic only. R is what the headroom argument in the module docstring is about,
    so this is what to report when a model's quality regresses under the int8 path.
    """
    s = mx.abs(scales.astype(mx.float32))
    lvl = float((1 << bits) - 1)
    rng = s * lvl
    return rng.max(axis=-1) / mx.maximum(rng.min(axis=-1), _EPS)
