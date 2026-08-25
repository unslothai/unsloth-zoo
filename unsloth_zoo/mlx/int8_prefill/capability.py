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
"""Can this machine run the int8 W8A8 prefill path?

There is no MLX API for Apple GPU generation, Metal version, or Metal Performance
Primitives availability, so the only sound test is to compile an MPP int8 kernel, run
it, and check the numbers. Two traps this has to route around:

  * `mx.fast.metal_kernel(...)` *constructs* successfully with no Metal backend at all
    and only raises "[metal_kernel] No Metal back-end" at call time, so construction
    proves nothing.
  * `mx.metal.device_info()` is deprecated and on a CUDA build cheerfully returns the
    CUDA device dict, so it is not a Metal probe either.

Everything is layered cheapest-first, cached, lazy (never at import), and fails closed:
any unexpected exception anywhere means "unsupported", never a raised error into a
user's generate call.
"""

import logging
import os
import platform
import sys

logger = logging.getLogger(__name__)

_MLX_MIN = (0, 29)
_MACOS_MIN = 26  # Metal 4 / MetalPerformancePrimitives

_verdict = None
_reason = None


def _env_override():
    return os.environ.get("UNSLOTH_MLX_INT8_PREFILL", "").strip().lower()


def _macos_major():
    try:
        return int(platform.mac_ver()[0].split(".")[0])
    except (ValueError, IndexError):
        return 0


def _probe_mpp_int8_gemm():
    """Compile and run a small int8 MPP GEMM and compare it against an exact reference.

    A machine that lacks the int8 tensor-op path fails at kernel compilation; a machine
    that has it but computes wrongly fails the comparison. Both mean unsupported.

    The reference is a float32 matmul rather than an integer one, because `mx.matmul`
    accepts only inexact types. That costs nothing here: with K=256 the largest possible
    partial sum is 127*127*256 = 4.13e6, well inside float32's exactly representable
    integer range of 2**24, so the comparison stays exact and needs no tolerance. K must
    not be raised past ~1000 without revisiting that.
    """
    import mlx.core as mx

    from .backends.metal_mpp import build_probe_kernel

    M = N = 128
    K = 256
    kernel = build_probe_kernel(N=N, K=K)

    # Deterministic, spans the signed int8 range, and is not symmetric about zero so a
    # sign or transpose error cannot cancel out.
    xq = ((mx.arange(M * K) % 251) - 125).reshape(M, K).astype(mx.int8)
    wq = ((mx.arange(N * K) % 241) - 120).reshape(N, K).astype(mx.int8)

    got = kernel(
        inputs=[xq, wq, mx.array([M], dtype=mx.int32)],
        grid=(N // 128 * 32 * 8, (M + 127) // 128, 1),
        threadgroup=(32 * 8, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[mx.int32],
    )[0]
    want = mx.matmul(xq.astype(mx.float32), wq.astype(mx.float32).T).astype(mx.int32)
    mx.eval(got, want)
    return bool(mx.array_equal(got, want).item())


def _decide():
    env = _env_override()
    if env in ("0", "off", "false", "no"):
        return False, "disabled by UNSLOTH_MLX_INT8_PREFILL"

    if sys.platform != "darwin":
        return False, f"not macOS (sys.platform={sys.platform!r})"

    try:
        import mlx.core as mx
    except ImportError:
        return False, "mlx is not installed"

    if not mx.metal.is_available():
        return False, "no Metal backend (mx.metal.is_available() is False)"

    try:
        version = tuple(int(p) for p in mx.__version__.split(".")[:2])
    except (ValueError, AttributeError):
        version = (0, 0)
    if version < _MLX_MIN:
        return False, f"mlx {mx.__version__} below minimum {'.'.join(map(str, _MLX_MIN))}"

    major = _macos_major()
    if major < _MACOS_MIN:
        return False, f"macOS {major} below {_MACOS_MIN} (no Metal Performance Primitives)"

    # Logged, never branched on: we do not know Apple's M5 architecture string and there
    # will be an M6. The probe below is the actual discriminator.
    try:
        logger.debug("Unsloth: MLX device_info=%r", mx.device_info())
    except Exception:
        pass

    if env in ("1", "force", "true", "yes", "on"):
        return True, "forced by UNSLOTH_MLX_INT8_PREFILL (probe skipped)"

    try:
        if _probe_mpp_int8_gemm():
            return True, "int8 MPP GEMM probe passed"
        return False, "int8 MPP GEMM probe produced wrong results"
    except BaseException as exc:  # a probe must never take down the caller
        return False, f"int8 MPP GEMM probe failed: {type(exc).__name__}: {exc}"


def is_supported():
    """True if the int8 path may be used here. Cached; safe to call in a hot loop."""
    global _verdict, _reason
    if _verdict is None:
        try:
            _verdict, _reason = _decide()
        except BaseException as exc:
            _verdict, _reason = False, f"capability probe raised {type(exc).__name__}: {exc}"
        logger.info(
            "Unsloth: MLX int8 prefill %s (%s)",
            "available" if _verdict else "unavailable",
            _reason,
        )
    return _verdict


def reason():
    """Why `is_supported()` decided what it did. Runs the probe if it has not yet."""
    is_supported()
    return _reason


def reset():
    """Forget the cached verdict. For tests that manipulate the environment."""
    global _verdict, _reason
    _verdict = None
    _reason = None
