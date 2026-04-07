from __future__ import annotations

import os
from typing import Any

try:
    import mlx.core as mx
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False

if _MLX_AVAILABLE:
    from .runtime_cce import (
        LEGACY_RUNTIME_VARIANT_ALIASES,
        RUNTIME_VARIANT_INFO,
        SUPPORTED_RUNTIME_VARIANTS,
        make_chunked_cross_entropy_loss,
        make_runtime_cce_loss_fused_finalize,
    )
else:
    LEGACY_RUNTIME_VARIANT_ALIASES = {}
    RUNTIME_VARIANT_INFO = {}
    SUPPORTED_RUNTIME_VARIANTS = ()

    def make_chunked_cross_entropy_loss(**kwargs):
        raise RuntimeError("mlx is required for CCE loss but is not installed")

    def make_runtime_cce_loss_fused_finalize(**kwargs):
        raise RuntimeError("mlx is required for CCE loss but is not installed")

__all__ = [
    "install_mlx_fast_cce_loss",
    "make_chunked_cross_entropy_loss",
    "make_runtime_cce_loss_fused_finalize",
    "SUPPORTED_RUNTIME_VARIANTS",
    "RUNTIME_VARIANT_INFO",
    "LEGACY_RUNTIME_VARIANT_ALIASES",
]


_RUNTIME_CCE_CACHE: dict[tuple[int, float, int, str, bool, int | None, int | None, str], Any] = {}


def _get_runtime_cce(
    *,
    ignore_index: int,
    logit_softcap: float,
    chunk_size: int = 0,
    runtime_variant: str = "balanced",
    quantized: bool = False,
    group_size: int | None = None,
    bits: int | None = None,
    mode: str = "affine",
):
    key = (ignore_index, logit_softcap, chunk_size, runtime_variant, quantized, group_size, bits, mode)
    runtime_cce = _RUNTIME_CCE_CACHE.get(key)
    if runtime_cce is None:
        runtime_cce, _ = make_chunked_cross_entropy_loss(
            ignore_index=ignore_index,
            logit_softcap=logit_softcap,
            chunk_size=chunk_size,
            runtime_variant=runtime_variant,
            quantized=quantized,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        _RUNTIME_CCE_CACHE[key] = runtime_cce
    return runtime_cce


def install_mlx_fast_cce_loss(*, override: bool = False):
    if not _MLX_AVAILABLE:
        raise RuntimeError("mlx is required for CCE loss but is not installed")

    if hasattr(mx.fast, "cce_loss") and not override:
        return mx.fast.cce_loss

    def cce_loss(
        hidden,
        weight,
        targets,
        *,
        scales=None,
        biases=None,
        group_size=None,
        bits=None,
        ignore_index: int = -100,
        logit_softcap: float = 0.0,
        chunk_size: int = 0,
        runtime_variant: str | None = None,
        mode: str = "affine",
        **_kwargs,
    ):
        if hidden.ndim < 2:
            raise ValueError("hidden must have at least 2 dimensions")
        if weight.ndim != 2:
            raise ValueError("weight must have exactly 2 dimensions")

        runtime_variant = runtime_variant or os.environ.get("MLX_CCE_RUNTIME_VARIANT", "balanced")
        hidden_flat = hidden.reshape((-1, hidden.shape[-1]))
        targets_flat = targets.reshape((-1,)).astype(mx.int32)

        if scales is not None:
            if group_size is None or bits is None:
                raise ValueError("group_size and bits are required for quantized CCE")
            if biases is None:
                biases = mx.zeros_like(scales)
            runtime_cce = _get_runtime_cce(
                ignore_index=ignore_index,
                logit_softcap=logit_softcap,
                chunk_size=chunk_size,
                runtime_variant=runtime_variant,
                quantized=True,
                group_size=group_size,
                bits=bits,
                mode=mode,
            )
            losses = runtime_cce(hidden_flat, weight, scales, biases, targets_flat)
            return losses.reshape(targets.shape)

        if weight.dtype != hidden.dtype:
            weight = weight.astype(hidden.dtype)

        runtime_cce = _get_runtime_cce(
            ignore_index=ignore_index,
            logit_softcap=logit_softcap,
            chunk_size=chunk_size,
            runtime_variant=runtime_variant,
        )
        losses = runtime_cce(hidden_flat, weight, targets_flat)
        return losses.reshape(targets.shape)

    cce_loss._unsloth_runtime_cce = True
    cce_loss._unsloth_supports_quantized = True
    mx.fast.cce_loss = cce_loss
    return cce_loss
