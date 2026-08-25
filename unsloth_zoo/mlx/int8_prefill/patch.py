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
"""The monkey patch itself.

`mx.quantized_matmul` is replaced rather than `nn.QuantizedLinear.__call__`, because the
op is where every quantized projection converges: tied lm_heads reach it through
`QuantizedEmbedding.as_linear`, DeepSeek-family MLA through `QuantizedMultiLinear`, and
sharded models through the distributed linears. Patching the layer would need five
separate class patches and would still miss anything calling the op directly.

Probed on MLX 0.32.1: assignment on the nanobind module works, `functools.wraps`
round-trips so `disable()` has a handle, and nothing in mlx, mlx-lm or unsloth_zoo
imports `quantized_matmul` by value -- the `sys.modules` sweep below is insurance
against a future by-value import, not a present requirement.
"""

import functools
import logging
import os
import sys

import mlx.core as mx

from . import backends, capability, registry
from .eligibility import ROW_THRESHOLD

logger = logging.getLogger(__name__)

_ORIG_QMM = mx.quantized_matmul
_ACT_DTYPES = (mx.bfloat16, mx.float16, mx.float32)

_disabled_by_selftest = False


def _verify_mode():
    return os.environ.get("UNSLOTH_MLX_INT8_VERIFY", "0").lower() in (
        "1", "true", "yes", "on",
    )


def _make_fn(entry, backend):
    """A differentiable W8A8 callable for one registered weight.

    `mx.fast.metal_kernel` outputs carry no vjp, so without this a LoRA backward through
    an intercepted prefill would fail deep inside the trainer -- Unsloth's MLX path
    wraps `nn.QuantizedLinear` in `mlx_lm.tuner.lora.LoRALinear`, whose `__call__` needs
    `dL/dx`. The gradient delegates to the stock 4-bit op, so it is exact rather than a
    straight-through approximation: only the forward is quantized further.
    """
    bits, gs = entry.bits, entry.group_size
    w, scales, biases = entry.w, entry.scales, entry.biases

    @mx.custom_function
    def fn(x):
        return backend.matmul(x, entry, out_dtype=x.dtype)

    @fn.vjp
    def _(primals, cotangent, output):
        # MLX hands a single-argument custom_function its primal directly rather than as
        # a 1-tuple. We do not need it either way: the gradient w.r.t. x is cotangent @ W,
        # and W is captured above.
        return _ORIG_QMM(cotangent, w, scales, biases, False, gs, bits, "affine")

    return fn


def _dispatch(x, entry):
    backend = backends.select()
    if entry.fn is None:
        entry.fn = _make_fn(entry, backend)
    out = entry.fn(x)

    if _verify_mode():
        from .backends import portable

        ref = _ORIG_QMM(
            x, entry.w, entry.scales, entry.biases, True, entry.group_size,
            entry.bits, "affine",
        )
        err = mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max()
        denom = mx.maximum(mx.abs(ref.astype(mx.float32)).max(), 1e-8)
        mx.eval(err, denom)
        logger.info(
            "Unsloth: MLX int8 verify %s rows=%d max_rel_err=%.3e",
            entry.name, x.size // x.shape[-1], (err / denom).item(),
        )
    return out


@functools.wraps(_ORIG_QMM)
def _patched_qmm(x, w, /, *args, **kwargs):
    # Callers pass these positionally as often as not -- mlx-lm/mlx_lm/models/base.py:84
    # does `mx.quantized_matmul(queries, *q_keys, transpose=..., ...)` -- so bind by hand
    # rather than assuming keywords.
    n = len(args)
    biases     = args[1] if n > 1 else kwargs.get("biases")
    transpose  = args[2] if n > 2 else kwargs.get("transpose", True)
    group_size = args[3] if n > 3 else kwargs.get("group_size")
    bits       = args[4] if n > 4 else kwargs.get("bits")
    mode       = args[5] if n > 5 else kwargs.get("mode", "affine")

    if (
        not _disabled_by_selftest
        and kwargs.get("stream") is None  # a metal_kernel launch cannot honour a stream
        and transpose is True
        and mode == "affine"
        and biases is not None
        and getattr(w, "ndim", 0) == 2
        and x.dtype in _ACT_DTYPES
    ):
        entry = registry.get(w)
        if (
            entry is not None
            and (group_size is None or group_size == entry.group_size)
            and (bits is None or bits == entry.bits)
            and x.size // x.shape[-1] >= ROW_THRESHOLD
        ):
            return _dispatch(x, entry)

    return _ORIG_QMM(x, w, *args, **kwargs)


_patched_qmm.__unsloth_int8_prefill__ = True


def is_patched():
    return getattr(mx.quantized_matmul, "__unsloth_int8_prefill__", False)


def apply():
    """Install the patch. Returns True if it was installed by this call."""
    if is_patched():
        return False
    mx.quantized_matmul = _patched_qmm
    for mod in tuple(sys.modules.values()):
        if mod is not None and getattr(mod, "quantized_matmul", None) is _ORIG_QMM:
            try:
                setattr(mod, "quantized_matmul", _patched_qmm)
            except Exception:
                pass
    logger.info(
        "Unsloth: MLX int8 prefill patch applied (row threshold %d, backend %s)",
        ROW_THRESHOLD, backends.select().__name__.rsplit(".", 1)[-1],
    )
    return True


def revert():
    """Restore MLX's original op."""
    if not is_patched():
        return False
    mx.quantized_matmul = _ORIG_QMM
    for mod in tuple(sys.modules.values()):
        if mod is not None and getattr(mod, "quantized_matmul", None) is _patched_qmm:
            try:
                setattr(mod, "quantized_matmul", _ORIG_QMM)
            except Exception:
                pass
    return True


def self_test(max_entries=4):
    """Run every registered shape through the real kernels and compare exactly.

    This is the safety net for hardware we have never run on. The Metal GEMM's tile
    conventions -- the transposed-right extents, the cooperative-tensor index ordering,
    the ragged-M epilogue guard -- are asserted in comments and checked nowhere, so if
    any of them is wrong the failure mode is wrong tokens, not an exception. Comparing
    against the stock op over the shapes actually registered turns that into a clean
    disable.

    Returns (ok, detail). On failure the patch stops dispatching permanently.
    """
    global _disabled_by_selftest

    entries = list(registry._entries.values())[:max_entries]
    if not entries:
        return True, "no registered weights to test"

    backend = backends.select()
    for entry in entries:
        # A ragged row count on purpose: M % 128 != 0 exercises the epilogue guard, and
        # a full-tile-only test would miss a broken one.
        for rows in (ROW_THRESHOLD, ROW_THRESHOLD + 37):
            x = mx.random.normal((rows, entry.k)).astype(mx.bfloat16)
            try:
                got = backend.matmul(x, entry, out_dtype=mx.float32)
                want = _ORIG_QMM(
                    x, entry.w, entry.scales, entry.biases, True,
                    entry.group_size, entry.bits, "affine",
                ).astype(mx.float32)
                err = (mx.abs(got - want).max()
                       / mx.maximum(mx.abs(want).max(), 1e-8))
                mx.eval(err)
                rel = err.item()
            except BaseException as exc:
                _disabled_by_selftest = True
                return False, f"{entry.name} rows={rows}: {type(exc).__name__}: {exc}"
            # W8A8 is lossy by design, so this is a sanity bar, not an equality check:
            # a correct kernel lands well inside it and a transposed or mis-indexed one
            # is nowhere near.
            if not (rel < 0.15):
                _disabled_by_selftest = True
                return False, f"{entry.name} rows={rows}: max_rel_err={rel:.3e}"

    return True, f"{len(entries)} weight(s) verified"


def reset_selftest():
    global _disabled_by_selftest
    _disabled_by_selftest = False
