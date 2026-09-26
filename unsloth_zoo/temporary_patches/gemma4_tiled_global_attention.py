# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# ============================================================================
# Memory-bounded router for Gemma-4 global (head_dim 512) attention.
#
# Gemma-4 global layers use head_dim 512, which FlashAttention-2 and the fused
# SDPA backends do not take on every GPU; SDPA then falls back to its math path
# and attention memory grows as O(S^2). This routes eligible global layers
# through the tiled Triton kernel in _triton_causal_attention_d512 (O(S) memory)
# and defers everything else to the backend it wrapped.
#
# It trades speed for memory (the tiled kernel is much slower than SDPA), so it
# is opt-in: UNSLOTH_GEMMA4_TILED_GLOBAL=1. UNSLOTH_GEMMA4_TILED_GLOBAL_MIN_SEQ
# (default 1024) keeps short sequences on SDPA. Both are read at call time.
#
# Engages only for training-mode, cache-free, dropout-free, dense causal
# attention (no mask, or a mask verified to be exactly causal) with equal Q/K/V
# lengths, Hq % Hkv == 0 and D=512, on a device in _VALIDATED_ARCHS with Triton
# >= 3.8. Training mode rather than grad mode is the gate so that the no_grad
# pack forward of reentrant checkpointing and its recompute pick one backend.
# Once selected, kernel errors propagate; they are not retried through SDPA.
# ============================================================================

import contextvars
import functools
import inspect
import os
import torch
from packaging.version import Version

from .common import TEMPORARY_PATCHES, logger
from .utils import raise_error

__all__ = [
    "patch_gemma4_tiled_global_attention",
    "gemma4_tiled_global_stats",
    "maybe_gemma4_tiled_global_attention",
]

# Architectures this path has been validated on. Extend only with numerical
# and memory results on that hardware; the kernel itself is not arch-specific.
_VALIDATED_ARCHS = frozenset({"gfx906"})
_HEAD_DIM = 512
_MIN_TRITON_VERSION = Version("3.8.0")

_ORIG_SDPA = [None]      # only for direct calls; installed wrappers capture their own
_ENGAGED = [0]
_CACHE_PRESENT = contextvars.ContextVar("unsloth_gemma4_tiled_global_cache", default = False)


def _enabled():
    return os.environ.get("UNSLOTH_GEMMA4_TILED_GLOBAL", "0") == "1"


def _min_seq_len():
    try:
        return max(1, int(os.environ.get("UNSLOTH_GEMMA4_TILED_GLOBAL_MIN_SEQ", "1024")))
    except Exception:
        return 1024


def _device_arch(x):
    """gfx arch of a HIP tensor's device, e.g. 'gfx906'; None for anything else."""
    if getattr(torch.version, "hip", None) is None or not torch.is_tensor(x) or x.device.type != "cuda":
        return None
    try:
        arch = getattr(torch.cuda.get_device_properties(x.device), "gcnArchName", "")
        return str(arch).split(":", 1)[0] or None
    except Exception:
        return None


def _on_validated_device(x):
    return _device_arch(x) in _VALIDATED_ARCHS


@functools.lru_cache(maxsize = 1)
def _triton_supported():
    # Older Triton rejects the D=512 dK/dV dot shapes at backward compile time,
    # after a forward already succeeded, so gate before the forward instead.
    try:
        import triton
        return Version(str(triton.__version__)) >= _MIN_TRITON_VERSION
    except Exception:
        return False


def _mask_is_exact_causal(mask, S, batch_size, device, query_dtype, _block = 1024):
    """True for None or a mask that is exactly dense causal, element by element.

    Accepts bool, or float (fp32 or the query dtype, as SDPA does) with exactly 0
    on allowed and -inf on blocked positions. A finite bias is real semantics and
    is rejected. The verdict is never cached: masks can be mutated in place.
    """
    if mask is None:
        return True
    if not torch.is_tensor(mask) or mask.dim() != 4:
        return False
    if mask.requires_grad or mask.device != device:
        return False
    if mask.dtype not in (torch.bool, torch.float32, query_dtype):
        return False
    if mask.shape[0] not in (1, batch_size) or mask.shape[1] != 1:
        return False
    if mask.shape[-2] != S or mask.shape[-1] != S:
        return False
    if torch.compiler.is_compiling():
        return False

    m = mask[:, 0]
    idx = torch.arange(S, device = mask.device)
    is_bool = m.dtype == torch.bool
    # Row blocks keep the transient comparison at (block, S) instead of (S, S).
    for start in range(0, S, _block):
        rows = idx[start : start + _block]
        causal = idx[None, :] <= rows[:, None]
        block = m[:, start : start + _block, :]
        match = (block == causal) if is_bool else torch.where(causal, block == 0, torch.isneginf(block))
        if not bool(match.all().item()):
            return False
    return True


def _eligible(module, query, key, value, attention_mask, dropout, is_causal, has_cache = False):
    if not _enabled() or torch.compiler.is_compiling():
        return False
    if not getattr(module, "training", False) or has_cache:
        return False
    if not type(module).__name__.startswith("Gemma4") or getattr(module, "is_sliding", False):
        return False
    if not (query.dim() == key.dim() == value.dim() == 4):
        return False
    if not _on_validated_device(query):
        return False
    if key.device != query.device or value.device != query.device:
        return False
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return False
    B, Hq, Sq, D = query.shape
    if key.shape[0] != B or value.shape[0] != B or key.shape[1] != value.shape[1]:
        return False
    Hkv = key.shape[1]
    if Hkv == 0 or Hq % Hkv != 0:
        return False
    if D != _HEAD_DIM or key.shape[-1] != _HEAD_DIM or value.shape[-1] != _HEAD_DIM:
        return False
    if key.shape[2] != Sq or value.shape[2] != Sq or Sq < _min_seq_len():
        return False
    if float(dropout or 0.0) != 0.0:
        return False
    if not _triton_supported():
        return False

    causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    # An explicit exact-causal mask suffices even when the caller passes
    # is_causal=False; padded, packed and multimodal masks fail the probe.
    if attention_mask is None and not causal:
        return False
    return _mask_is_exact_causal(attention_mask, Sq, B, query.device, query.dtype)


def maybe_gemma4_tiled_global_attention(
    module, query, key, value, attention_mask,
    dropout = 0.0, scaling = None, is_causal = None, has_cache = False,
):
    """(B, H, S, D) attention output if the tiled path applies, else None.

    Shared by the registry wrapper below and gemma4_float32, whose FORCE_FLOAT32
    forward calls raw SDPA instead of going through the registry.
    """
    if not _eligible(module, query, key, value, attention_mask, dropout, is_causal, has_cache = has_cache):
        return None
    from ._triton_causal_attention_d512 import causal_attention_d512

    # SDPA semantics: scale=None means 1/sqrt(D). Gemma-4 passes self.scaling.
    scale = query.shape[-1] ** -0.5 if scaling is None else scaling
    out = causal_attention_d512(query, key, value, float(scale))
    _ENGAGED[0] += 1
    if _ENGAGED[0] == 1:
        logger.info_once("Unsloth: Gemma-4 global attention is using the tiled D=512 Triton kernel.")
    return out


def _sdpa_maybe_tiled_global(
    module, query, key, value, attention_mask,
    dropout = 0.0, scaling = None, is_causal = None, _fallback = None, **kwargs,
):
    has_cache = (
        bool(_CACHE_PRESENT.get())
        or kwargs.get("past_key_values", None) is not None
        or kwargs.get("past_key_value", None) is not None
        or getattr(module, "_unsloth_shared_kv_carrier", None) is not None
    )
    out = maybe_gemma4_tiled_global_attention(
        module, query, key, value, attention_mask,
        dropout = dropout, scaling = scaling, is_causal = is_causal, has_cache = has_cache,
    )
    if out is not None:
        return out.transpose(1, 2).contiguous(), None

    fallback = _fallback if _fallback is not None else _ORIG_SDPA[0]
    if fallback is None:
        raise RuntimeError("Unsloth: Gemma-4 tiled global SDPA wrapper has no fallback backend")
    return fallback(
        module, query, key, value, attention_mask,
        dropout = dropout, scaling = scaling, is_causal = is_causal, **kwargs,
    )


def gemma4_tiled_global_stats():
    return {
        "engaged": _ENGAGED[0],
        "enabled": _enabled(),
        "min_seq_len": _min_seq_len(),
        "triton_supported": _triton_supported(),
        "validated_archs": sorted(_VALIDATED_ARCHS),
    }


def _make_tiled_global_wrapper(original):
    """Wrap one registry entry, capturing its fallback in the closure."""
    router = _sdpa_maybe_tiled_global

    @functools.wraps(original)
    def wrapped(module, query, key, value, attention_mask,
                dropout = 0.0, scaling = None, is_causal = None, **kwargs):
        return router(
            module, query, key, value, attention_mask,
            dropout = dropout, scaling = scaling, is_causal = is_causal,
            _fallback = original, **kwargs,
        )

    # Carry sibling wrappers' sentinels so each patch stays idempotent whichever
    # order the Gemma-4 SDPA wrappers were installed in.
    try:
        for name, attr in vars(original).items():
            if name.startswith("_unsloth_"):
                setattr(wrapped, name, attr)
    except Exception:
        pass
    wrapped._unsloth_gemma4_tiled_global = True
    wrapped._unsloth_gemma4_tiled_global_original = original
    return wrapped


def _make_gemma4_cache_scope_wrapper(original):
    """Record whether this Gemma4TextAttention.forward call carries a cache.

    The attention forward consumes past_key_values before calling the registry,
    and after an empty-cache prefill Q/K/V lengths are equal, so the registry
    cannot tell that call from a cache-free one. A ContextVar scoped to the
    forward carries that bit. Set on every call, whether or not the feature is
    enabled at entry, so toggling the env var mid-forward cannot open a gap.
    """
    try:
        signature = inspect.signature(original)
    except (TypeError, ValueError):
        signature = None

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        cache = kwargs.get("past_key_values", None)
        if "past_key_values" not in kwargs and signature is not None:
            try:
                cache = signature.bind_partial(*args, **kwargs).arguments.get("past_key_values", None)
            except TypeError:
                cache = None
        token = _CACHE_PRESENT.set(cache is not None)
        try:
            return original(*args, **kwargs)
        finally:
            _CACHE_PRESENT.reset(token)

    wrapped._unsloth_gemma4_tiled_global_cache_scope = True
    return wrapped


def _patch_gemma4_cache_scope():
    try:
        from transformers.models.gemma4 import modeling_gemma4
        current = modeling_gemma4.Gemma4TextAttention.forward
    except Exception:
        return False
    if not getattr(current, "_unsloth_gemma4_tiled_global_cache_scope", False):
        modeling_gemma4.Gemma4TextAttention.forward = _make_gemma4_cache_scope_wrapper(current)
    return True


def patch_gemma4_tiled_global_attention():
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception as e:
        return raise_error("transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS", e)

    # Fail closed: without the cache scope an empty-cache prefill looks
    # cache-free to the registry, so do not install the router at all.
    # Re-run every pass in case another Gemma-4 patch replaced forward.
    if not _patch_gemma4_cache_scope():
        return

    try:
        current = ALL_ATTENTION_FUNCTIONS["sdpa"]
    except Exception as e:
        return raise_error("ALL_ATTENTION_FUNCTIONS['sdpa']", e)
    if getattr(current, "_unsloth_gemma4_tiled_global", False):
        return
    _ORIG_SDPA[0] = current
    # Direct assignment, as in gemma4_flash_sliding: register() does not update
    # the mapping layers read.
    ALL_ATTENTION_FUNCTIONS["sdpa"] = _make_tiled_global_wrapper(current)


TEMPORARY_PATCHES.append(patch_gemma4_tiled_global_attention)
