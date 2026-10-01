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
the default. `tests/test_mlx_int8_prefill_numerics.py` asserts the distribution, so if MLX ever
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

**Rotation** (default on). Measured on real 4-bit checkpoints, the weight requant above is
the small half of the error: almost all of it is the per-row activation quantization,
because one outlier feature sets the row's scale and flattens everything else onto a few
int8 levels. Both operands are therefore multiplied by a block-diagonal Walsh-Hadamard
matrix H (blocks of `ROTATE_BLOCK` = 32, the SIMD width) before they are quantized. H is
orthogonal up to a factor of 32 (`H @ H.T = 32 * I`), so `(x H)(w H)^T = 32 * x w^T`
exactly and the GEMM is unchanged; the rotation spreads an outlier across its block,
which shrinks the row's absmax relative to its typical entry. End to end against stock
`mx.quantized_matmul`, this cut logits KLD about 5x (Qwen3-1.7B-4bit 3.6e-2 -> 6.7e-3,
Llama-3.2-1B-4bit 1.3e-2 -> 2.8e-3). The rotated weight has no analytic bound, so its
scale is always the exact absmax, computed once here. `UNSLOTH_MLX_INT8_ROTATE=0`
restores the unrotated arithmetic.
"""

import logging
import os

import mlx.core as mx

logger = logging.getLogger(__name__)

INT8_MAX = 127.0
_EPS = 1e-8

# Walsh-Hadamard block size. 32 is the Metal SIMD width, so the activation kernel rotates a
# block with five simd_shuffle_xor stages and no threadgroup memory. 64 and 128 measured
# marginally better on one model and no better on another, which does not pay for
# cross-simdgroup shuffles. Eligibility already requires K % 32 == 0.
ROTATE_BLOCK = 32

# Rows of a weight rotated at once while computing its scale, to cap warmup memory at
# a slice of one layer in float32.
_ROTATE_CHUNK_ROWS = 4096


def use_exact_scales() -> bool:
    return os.environ.get("UNSLOTH_MLX_INT8_EXACT_SCALES", "0").lower() in (
        "1", "true", "yes", "on",
    )


def use_rotation() -> bool:
    return os.environ.get("UNSLOTH_MLX_INT8_ROTATE", "1").lower() not in (
        "0", "false", "no", "off",
    )


def dequantize_f32(weight, scales, biases, bits, group_size):
    """Dequantize with the metadata widened to float32 first, which is what the Metal
    requant kernel does. Dequantizing in the metadata dtype and widening afterwards would
    round every bf16 weight before the int8 step ever sees it."""
    return mx.dequantize(
        weight, scales.astype(mx.float32), biases.astype(mx.float32),
        group_size=group_size, bits=bits, mode="affine",
    )


def rotate(a, block=ROTATE_BLOCK):
    """Unnormalized block-diagonal Walsh-Hadamard along the last axis (Sylvester order,
    the same butterfly the Metal kernels run). The 1/block it leaves behind is folded into
    the GEMM epilogue."""
    shape = a.shape
    a = a.reshape(*shape[:-1], shape[-1] // block, block)
    return mx.hadamard_transform(a, scale=1.0).reshape(shape)


def channel_scale_rotated(weight, scales, biases, bits, group_size):
    """Per-channel int8 scale of the rotated weight: its exact absmax, a row chunk at a
    time so warmup never holds more than a slice of one layer in float32."""
    n = weight.shape[0]
    parts = []
    for start in range(0, n, _ROTATE_CHUNK_ROWS):
        stop = min(start + _ROTATE_CHUNK_ROWS, n)
        dq = dequantize_f32(
            weight[start:stop], scales[start:stop], biases[start:stop], bits, group_size
        )
        part = mx.abs(rotate(dq)).max(axis=-1)
        mx.eval(part)
        parts.append(part)
    absmax = parts[0] if len(parts) == 1 else mx.concatenate(parts)
    return mx.maximum(absmax, _EPS) / INT8_MAX


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


def channel_scale(weight, scales, biases, bits, group_size, exact=None, rotate=None):
    """Per-channel int8 scale for the arithmetic the entry will run: the rotated weight's
    exact absmax when rotating (the default), else the bound or exact absmax of the plain
    dequantized weight, `exact` defaulting to the environment."""
    if rotate is None:
        rotate = use_rotation()
    if rotate:
        return channel_scale_rotated(weight, scales, biases, bits, group_size)
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
