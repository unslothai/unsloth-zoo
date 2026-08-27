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
"""Which quantized weights may take the W8A8 int8 path.

Table-driven so widening support is a data change. The rules are deliberately
conservative: a weight that fails any of them keeps MLX's stock 4-bit kernels, which
is always correct, merely slower.
"""

import os

# Only calls with at least this many rows (tokens) take the int8 path. Below it MLX's
# quantized kernels win -- they are roughly 2x faster than a 16-bit GEMM at decode
# sizes. This is also what keeps generation on the stock path: decode presents 1 row,
# or a draft block's worth under speculative decoding, far below the threshold.
ROW_THRESHOLD = int(os.environ.get("UNSLOTH_MLX_INT8_ROW_THRESHOLD", "512"))

# GEMM tile shape. N is NOT ceil-divided when building the launch grid, so an N that is
# not a multiple of TN would silently drop output columns. Enforce it here rather than
# trusting the kernel.
TILE_N = 128
TILE_M = 128

# K must be a multiple of this for the tiled loads to stay in bounds.
K_MULTIPLE = 32

# Below this, the fixed cost of quantizing activations is not repaid.
MIN_DIM = 1024

# Above this an output projection is almost certainly an lm_head, where prefill needs
# logits for one position only and the requant would be pure overhead.
MAX_OUT = 32768

# (bits, group_size) combinations the requant kernel can unpack.
#
# The kernel walks packed uint32 words, so it needs 32 % bits == 0 (whole values per
# word) and (group_size * bits) % 32 == 0 (a group never straddles a word). Affine bits
# 3/5/6 are a dense bitstream where values cross word boundaries; they are permanently
# excluded rather than merely unimplemented.
_SUPPORTED_4BIT = {(4, 32), (4, 64), (4, 128)}

# Affine 8-bit is off by default and this is not conservatism, it is arithmetic. See
# plans/sparkling-popping-kay.md: requantizing affine-8 to symmetric int8 has zero bits
# of headroom, so any channel whose inter-group range ratio exceeds ~1.2 comes out
# strictly worse than the 8-bit weights it started from. At 4 bits the same requant has
# roughly 16x of cushion, which is the only reason this technique works at all.
_SUPPORTED_8BIT = {(8, 32), (8, 64), (8, 128)}


def allow_8bit() -> bool:
    return os.environ.get("UNSLOTH_MLX_INT8_ALLOW_8BIT", "0").lower() in (
        "1", "true", "yes", "on",
    )


def supported_formats() -> frozenset:
    fmts = set(_SUPPORTED_4BIT)
    if allow_8bit():
        fmts |= _SUPPORTED_8BIT
    return frozenset(fmts)


def is_eligible(n, k, bits, group_size, mode="affine", has_biases=True):
    """Return (ok, reason). `reason` is None when eligible, else a short diagnostic.

    `n` is the output dim, `k` the input dim, both in elements (not packed words).
    """
    if mode != "affine":
        # mxfp4 / mxfp8 / nvfp4 carry no biases and use e8m0/e4m3 scales; the requant
        # kernel's affine arithmetic does not apply.
        return False, f"mode={mode!r} is not affine"
    if not has_biases:
        return False, "affine weight without biases"
    if (bits, group_size) not in supported_formats():
        if (bits, group_size) in _SUPPORTED_8BIT:
            return False, "affine 8-bit requires UNSLOTH_MLX_INT8_ALLOW_8BIT=1"
        return False, f"unsupported (bits={bits}, group_size={group_size})"
    if n % TILE_N:
        return False, f"N={n} is not a multiple of {TILE_N}"
    if k % K_MULTIPLE:
        return False, f"K={k} is not a multiple of {K_MULTIPLE}"
    if min(n, k) < MIN_DIM:
        return False, f"min(N,K)={min(n, k)} below {MIN_DIM}"
    if n > MAX_OUT:
        return False, f"N={n} above {MAX_OUT} (probably an lm_head)"
    return True, None


def rows_of(x, k_dim):
    """Row count a matmul will see, flattening every leading batch dim."""
    return x.size // k_dim
