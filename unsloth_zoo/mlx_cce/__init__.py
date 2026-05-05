from __future__ import annotations

from typing import Any

try:
    import mlx.core as mx
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False

if _MLX_AVAILABLE:
    from .runtime_cce import (
        make_chunked_cross_entropy_loss,
        make_runtime_cce_loss_fused_finalize,
    )
else:
    def make_chunked_cross_entropy_loss(**kwargs):
        raise RuntimeError("mlx is required for CCE loss but is not installed")

    def make_runtime_cce_loss_fused_finalize(**kwargs):
        raise RuntimeError("mlx is required for CCE loss but is not installed")

__all__ = [
    "make_chunked_cross_entropy_loss",
    "make_runtime_cce_loss_fused_finalize",
    "clear_cce_cache",
]


_RUNTIME_CCE_CACHE: dict[tuple[int, float, int, bool, int | None, int | None, str], Any] = {}


def _get_runtime_cce(
    *,
    ignore_index: int,
    logit_softcap: float,
    chunk_size: int = 0,
    quantized: bool = False,
    group_size: int | None = None,
    bits: int | None = None,
    mode: str = "affine",
):
    key = (ignore_index, logit_softcap, chunk_size, quantized, group_size, bits, mode)
    runtime_cce = _RUNTIME_CCE_CACHE.get(key)
    if runtime_cce is None:
        runtime_cce, _ = make_chunked_cross_entropy_loss(
            ignore_index=ignore_index,
            logit_softcap=logit_softcap,
            chunk_size=chunk_size,
            quantized=quantized,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        _RUNTIME_CCE_CACHE[key] = runtime_cce
    return runtime_cce


def clear_cce_cache():
    """Clear the CCE kernel cache.

    Call this when switching models or devices to free cached Metal
    kernels and compiled functions that may hold stale references.
    """
    _RUNTIME_CCE_CACHE.clear()
