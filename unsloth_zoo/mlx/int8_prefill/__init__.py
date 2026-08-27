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
"""int8 W8A8 prefill acceleration for MLX quantized models.

MLX computes quantized matmuls in 16-bit regardless of how the weights are stored, so a
4-bit model prefills no faster than an 8-bit one. On Apple M5, whose neural accelerators
run int8 at roughly twice the fp16 rate, routing prefill-sized matmuls through an
int8 x int8 -> int32 GEMM recovers that. Decode is memory-bound and stays on MLX's
kernels untouched.

Usage::

    from unsloth_zoo.mlx import int8_prefill

    if int8_prefill.enable():          # no-op and False off M5
        int8_prefill.warmup(model)     # required: builds the allow-list

Everything is opt-in and fails closed. On any other hardware, any other MLX version, or
any unexpected error, `enable()` returns False and MLX is left exactly as it was.

Environment:
    UNSLOTH_MLX_INT8_PREFILL=0|1|force   kill switch / skip the capability probe
    UNSLOTH_MLX_INT8_BACKEND=metal_mpp|portable
    UNSLOTH_MLX_INT8_ROW_THRESHOLD=512   below this, calls keep MLX's kernels
    UNSLOTH_MLX_INT8_ALLOW_8BIT=1        affine 8-bit (off by default, see eligibility)
    UNSLOTH_MLX_INT8_EXACT_SCALES=1      exact absmax instead of the analytic bound
    UNSLOTH_MLX_INT8_VERIFY=1            shadow mode: log max relative error per call
"""

import logging

from . import backends, capability, eligibility, patch, registry, scales

__all__ = [
    "enable", "disable", "warmup", "is_supported", "is_enabled", "self_test",
    "reason", "registered",
]

logger = logging.getLogger(__name__)

__version__ = "0.1.0"


def is_supported():
    """Whether this machine can run the int8 path. Cached, never raises."""
    return capability.is_supported()


def reason():
    """The capability verdict's explanation, for logs and bug reports."""
    return capability.reason()


def enable(force=False):
    """Install the patch if this machine supports it.

    Returns True if the patch is now active. `force=True` skips the capability probe and
    is for testing on hardware that cannot run the real kernels -- pair it with
    UNSLOTH_MLX_INT8_BACKEND=portable.
    """
    if not force and not is_supported():
        logger.info("Unsloth: MLX int8 prefill not enabled (%s)", capability.reason())
        return False
    patch.apply()
    return True


def disable():
    """Restore MLX's original op and drop the allow-list."""
    patch.revert()
    registry.clear()
    patch.reset_selftest()


def is_enabled():
    return patch.is_patched()


def warmup(model, scope="all", exact_scales=None):
    """Register a model's eligible projections. Required -- nothing is accelerated
    until a weight is registered.

    This is also where every `mx.eval` in the module happens, which is what leaves the
    hot path safe to run inside `mx.compile`.
    """
    return registry.warmup(model, scope=scope, exact_scales=exact_scales)


def self_test(**kwargs):
    """Verify registered shapes against MLX's own op on this device. See patch.self_test."""
    return patch.self_test(**kwargs)


def registered():
    """Names of the currently registered modules."""
    return registry.names()
