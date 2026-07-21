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

"""Regression test for ORPO odds-ratio numerical stability in float16.

The odds-ratio term uses log(1 - p) = log(-expm1(logp)) with a 1e-12 floor.
In float16 that floor underflows to 0.0, so a perfectly-predicted response row
(logp -> 0, e.g. an empty response span) makes mx.log(0.0) = -inf and yields
NaN gradients. The fix computes the term in float32 (mirroring TRL's
ORPOTrainer.odds_ratio_loss upcast). This test pins that a float16 logp row
with an exact 0.0 stays finite.
"""

from __future__ import annotations

import math
import sys

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    shim_prefixes = ("mlx", "mlx_lm", "mlx_vlm")
    real_mlx_modules = {
        name: module
        for name, module in sys.modules.items()
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in shim_prefixes)
    }
    from mlx_simulation import simulate_mlx_on_torch
    from mlx_simulation.mlx_stub import _MLXFinder
    simulate_mlx_on_torch()
    for name in list(sys.modules):
        if name == "unsloth_zoo.mlx" or name.startswith("unsloth_zoo.mlx."):
            sys.modules.pop(name, None)
    yield
    for name in list(sys.modules):
        if (
            name == "unsloth_zoo.mlx" or name.startswith("unsloth_zoo.mlx.")
            or any(name == prefix or name.startswith(f"{prefix}.") for prefix in shim_prefixes)
        ):
            sys.modules.pop(name, None)
    sys.meta_path[:] = [
        finder for finder in sys.meta_path
        if not isinstance(finder, _MLXFinder)
    ]
    sys.modules.update(real_mlx_modules)


def test_naive_float16_odds_ratio_underflows_to_inf():
    """Document the bug: the pre-fix float16 expression is non-finite on logp=0."""
    import mlx.core as mx

    logp_c = mx.array([0.0], dtype=mx.float16)  # perfectly-predicted chosen row
    # Pre-fix formulation: floor built in the input (float16) dtype.
    val_c = mx.maximum(-mx.expm1(logp_c), mx.array(1e-12, logp_c.dtype))
    assert float(val_c.astype(mx.float32)[0]) == 0.0  # 1e-12 underflowed to 0.0
    assert not math.isfinite(float(mx.log(val_c).astype(mx.float32)[0]))


def test_orpo_odds_ratio_loss_finite_on_perfect_float16_row():
    """The fixed helper stays finite on a float16 logp row containing 0.0."""
    import mlx.core as mx
    from unsloth_zoo.mlx.utils import _orpo_odds_ratio_loss

    # Row 0: perfectly-predicted chosen response (logp -> 0). Row 1: ordinary.
    logp_c = mx.array([0.0, -0.5], dtype=mx.float16)
    logp_r = mx.array([-1.0, -2.0], dtype=mx.float16)
    or_loss = _orpo_odds_ratio_loss(logp_c, logp_r)
    val = float(or_loss.astype(mx.float32))
    assert math.isfinite(val), f"ORPO odds-ratio loss not finite: {val}"
    assert or_loss.dtype == mx.float32


def test_orpo_odds_ratio_loss_matches_float32_reference():
    """float16 inputs give the same result as float32 inputs (upcast parity)."""
    import mlx.core as mx
    from unsloth_zoo.mlx.utils import _orpo_odds_ratio_loss

    logp_c16 = mx.array([-0.2, -0.5, -1.5], dtype=mx.float16)
    logp_r16 = mx.array([-0.9, -2.0, -0.3], dtype=mx.float16)
    logp_c32 = logp_c16.astype(mx.float32)
    logp_r32 = logp_r16.astype(mx.float32)
    a = float(_orpo_odds_ratio_loss(logp_c16, logp_r16).astype(mx.float32))
    b = float(_orpo_odds_ratio_loss(logp_c32, logp_r32).astype(mx.float32))
    assert math.isfinite(a) and math.isfinite(b)
    assert abs(a - b) < 1e-2
