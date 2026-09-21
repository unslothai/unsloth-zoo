# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Opt-in memory-efficient Gemma-4 31B D=512 global attention on AMD gfx906.

PyTorch SDPA on legacy gfx906 can select a math path for head_dim=512, making
attention memory scale quadratically with sequence length. This router keeps all
existing SDPA semantics by default and only substitutes a tiled Triton kernel
when the call is provably the Gemma-4 31B D=512 dense causal training shape.

Enable with::

    UNSLOTH_GEMMA4_GFX906_GLOBAL=1

The patch is deliberately opt-in because SDPA is substantially faster. It is
training-mode only (including reentrant checkpoint pack forwards under
``torch.no_grad()``), requires Triton 3.8+, and accepts only mask-free causal
attention or an exact boolean / 0-and-negative-infinity causal mask.
``UNSLOTH_GEMMA4_GFX906_GLOBAL_MIN_SEQ`` defaults to 1024 and is a memory-policy
threshold, not a performance crossover point.
"""

import functools
import os
import torch
from packaging.version import Version

from .common import TEMPORARY_PATCHES, logger
from .utils import raise_error

__all__ = [
    "patch_gemma4_gfx906_global_attention",
    "gemma4_gfx906_global_stats",
    "maybe_gemma4_gfx906_global_attention",
]

_ORIG_SDPA = [None]
_ENGAGED = [0]
_MIN_TRITON_VERSION = Version("3.8.0")


def _enabled():
    return os.environ.get("UNSLOTH_GEMMA4_GFX906_GLOBAL", "0") == "1"


def _min_seq_len():
    try:
        return max(1, int(os.environ.get("UNSLOTH_GEMMA4_GFX906_GLOBAL_MIN_SEQ", "1024")))
    except Exception:
        return 1024


def _is_gfx906_tensor(x):
    if getattr(torch.version, "hip", None) is None or not torch.is_tensor(x) or x.device.type != "cuda":
        return False
    try:
        props = torch.cuda.get_device_properties(x.device)
        return str(getattr(props, "gcnArchName", "")).split(":", 1)[0] == "gfx906"
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def _triton_supported():
    """True only for the Triton range actually validated by this kernel.

    The repo permits Triton 3.0+, but D=512 dKV uses dot shapes that older
    Triton releases reject. Keep the feature off rather than failing in backward
    after a successful forward on an unvalidated compiler.
    """
    try:
        import triton
        return Version(str(triton.__version__)) >= _MIN_TRITON_VERSION
    except Exception:
        return False


def _mask_is_exact_causal(mask, S, batch_size, device, _block=1024):
    """Verify an explicit mask is exactly dense causal attention.

    This intentionally does not reuse the sliding-window mask cache: a cached
    verdict can be stale after an in-place mutation or can have been computed
    for a different window. Float masks are accepted only when allowed entries
    are exactly zero and blocked entries are negative infinity; finite additive
    biases are real semantics and must stay on the original backend.
    """
    if mask is None:
        return True
    if not torch.is_tensor(mask) or mask.dim() != 4:
        return False
    if mask.requires_grad or mask.device != device:
        return False
    if mask.dtype != torch.bool and not torch.is_floating_point(mask):
        return False
    if mask.shape[0] not in (1, batch_size) or mask.shape[1] != 1:
        return False
    if mask.shape[-2] != S or mask.shape[-1] != S:
        return False
    if torch.compiler.is_compiling():
        return False

    m = mask[:, 0]
    idx = torch.arange(S, device=mask.device)
    is_bool = m.dtype == torch.bool
    for start in range(0, S, _block):
        rows = idx[start : start + _block]
        causal = idx[None, :] <= rows[:, None]
        block = m[:, start : start + _block, :]
        if is_bool:
            match = block == causal
        else:
            match = torch.where(causal, block == 0, torch.isneginf(block) & (block < 0))
        if not bool(match.all().item()):
            return False
    return True


def _eligible(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout,
    is_causal,
    has_cache=False,
):
    if not _enabled() or torch.compiler.is_compiling():
        return False
    # Reentrant gradient checkpointing runs its first training forward under
    # torch.no_grad(). Gate on module.training, not grad mode, so checkpoint pack
    # and backward recompute use the same backend. Eval/inference remains on the
    # existing backend.
    if not getattr(module, "training", False):
        return False
    if has_cache:
        return False
    if not type(module).__name__.startswith("Gemma4") or getattr(module, "is_sliding", False):
        return False
    if not (_is_gfx906_tensor(query) and query.dim() == key.dim() == value.dim() == 4):
        return False
    if key.device != query.device or value.device != query.device:
        return False
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32) or key.dtype != query.dtype or value.dtype != query.dtype:
        return False
    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        return False
    if (
        query.shape[1] != 32
        or key.shape[1] != 4
        or value.shape[1] != 4
        or query.shape[-1] != 512
        or key.shape[-1] != 512
        or value.shape[-1] != 512
    ):
        return False
    Sq, Sk, Sv = query.shape[2], key.shape[2], value.shape[2]
    if Sq != Sk or Sq != Sv or Sq < _min_seq_len():
        return False
    if float(dropout or 0.0) != 0.0:
        return False
    if not _triton_supported():
        return False

    causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    # With an explicit exact causal mask, SDPA may report is_causal=False; the
    # verified mask is sufficient. Multimodal, packed and padded masks all fail
    # this exact probe and defer to the original backend.
    if attention_mask is None and not causal:
        return False
    return _mask_is_exact_causal(attention_mask, Sq, query.shape[0], query.device)


def maybe_gemma4_gfx906_global_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout=0.0,
    scaling=None,
    is_causal=None,
    has_cache=False,
):
    """Return raw (B, H, S, D) attention output when the gfx906 path applies.

    ``None`` means the caller must keep its existing attention backend.  This
    shape-preserving helper is shared by the normal attention registry wrapper
    and Gemma-4's FORCE_FLOAT32 forward, which intentionally calls raw SDPA
    directly instead of the registry.
    """
    if not _eligible(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout,
        is_causal,
        has_cache=has_cache,
    ):
        return None
    from ._gemma4_gfx906_global_kernels import gemma4_gfx906_global_attention

    # Preserve SDPA's public contract: scale=None means 1/sqrt(head_dim), not
    # module.scaling. Gemma4's normal caller passes self.scaling explicitly.
    scale = query.shape[-1] ** -0.5 if scaling is None else scaling
    out = gemma4_gfx906_global_attention(query, key, value, float(scale))
    _ENGAGED[0] += 1
    if _ENGAGED[0] == 1:
        logger.info_once(
            "Unsloth: gfx906 memory-efficient Gemma-4 31B D=512 global attention engaged."
        )
    return out


def _sdpa_maybe_gfx906_global(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout=0.0,
    scaling=None,
    is_causal=None,
    _fallback=None,
    **kwargs,
):
    out = maybe_gemma4_gfx906_global_attention(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=dropout,
        scaling=scaling,
        is_causal=is_causal,
    )
    if out is not None:
        return out.transpose(1, 2).contiguous(), None

    fallback = _fallback if _fallback is not None else _ORIG_SDPA[0]
    if fallback is None:
        raise RuntimeError("gfx906 Gemma-4 SDPA wrapper has no fallback backend")
    return fallback(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=dropout,
        scaling=scaling,
        is_causal=is_causal,
        **kwargs,
    )


def gemma4_gfx906_global_stats():
    return {
        "engaged": _ENGAGED[0],
        "enabled": _enabled(),
        "min_seq_len": _min_seq_len(),
        "triton_supported": _triton_supported(),
    }


def _make_gfx906_global_wrapper(original):
    """Capture the wrapped backend per installation; never via mutable global state."""
    router = _sdpa_maybe_gfx906_global

    @functools.wraps(original)
    def wrapped(module, query, key, value, attention_mask,
                dropout=0.0, scaling=None, is_causal=None, **kwargs):
        return router(
            module, query, key, value, attention_mask,
            dropout=dropout, scaling=scaling, is_causal=is_causal,
            _fallback=original, **kwargs,
        )

    try:
        for name, value in vars(original).items():
            if name.startswith("_unsloth_"):
                setattr(wrapped, name, value)
    except Exception:
        pass
    wrapped._unsloth_gemma4_gfx906_global = True
    wrapped._unsloth_gemma4_gfx906_original = original
    return wrapped


def patch_gemma4_gfx906_global_attention():
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception as e:
        return raise_error("transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS", e)

    try:
        current = ALL_ATTENTION_FUNCTIONS["sdpa"]
    except Exception as e:
        return raise_error("ALL_ATTENTION_FUNCTIONS['sdpa']", e)

    if getattr(current, "_unsloth_gemma4_gfx906_global", False):
        return
    # Keep this only for direct unit calls of _sdpa_maybe_gfx906_global. The
    # installed wrapper captures its own fallback, so later patch passes/module
    # reloads cannot rewrite a shared pointer into a recursive wrapper chain.
    _ORIG_SDPA[0] = current
    ALL_ATTENTION_FUNCTIONS["sdpa"] = _make_gfx906_global_wrapper(current)


TEMPORARY_PATCHES.append(patch_gemma4_gfx906_global_attention)
