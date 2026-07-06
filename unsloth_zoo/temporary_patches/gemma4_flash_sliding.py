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
# Fast sliding-window attention router for Gemma-4.
#
# Gemma-4's head_dim 512 global layers force FlashAttention-2 off model-wide, so
# every layer falls back to O(S^2) SDPA. A causal sliding window only needs keys
# in (i-w, i] for query i, so route just the sliding layers (head_dim <= 256,
# plain causal band mask, no padding) through a fast O(S*w) kernel; global and
# padded / non-band masks defer to the original SDPA. One default-on wrapper
# picks the kernel automatically:
#   * FlashAttention-2 window_size=(w-1, 0) when flash-attn is importable and the
#     dtype is fp16/bf16, or
#   * a pure-SDPA banded (block-local) kernel otherwise,
# so the sliding-window speedup is automatic with or without flash-attn. Both
# use a byte-identical band probe, so they engage / defer identically; backward
# is exact and numerics match full SDPA + band mask. On by default;
# UNSLOTH_GEMMA4_FLASH_SLIDING=0 reverts to plain SDPA. UNSLOTH_BANDED_SDPA=1
# forces the pure-SDPA banded kernel even when flash-attn is present.
# ============================================================================

import os
import functools
import torch
from .common import TEMPORARY_PATCHES, logger
from .utils import raise_error
from .gemma4_banded_attention import _banded_sdpa_core, _mask_is_plain_band

__all__ = ["patch_gemma4_flash_sliding"]

try:
    from flash_attn import flash_attn_func as _flash_attn_func
    _HAS_FA2 = True
except Exception:
    _flash_attn_func = None
    _HAS_FA2 = False

_ORIG_SDPA = [None]      # boxed reference to the wrapped sdpa function
_ENGAGED = [0]           # count of FA2-path invocations (debug)
_BANDED_ENGAGED = [0]    # count of banded-path invocations (debug)


@functools.lru_cache(maxsize=1)
def _enabled():
    """Whether the gemma-4 sliding-window fast router is active.

    On by default and effectively always on: the router auto-prefers
    FlashAttention-2's window kernel when flash-attn is importable, and the
    pure-SDPA banded fallback works on every setup and dtype. So "always on if
    Flash Attention exists and/or if it works" holds without gating on _HAS_FA2:
    the "or if it works" case is covered by the universal banded path, which is
    why this must NOT be gated on _HAS_FA2 (that would wrongly disable the
    banded fallback). Set UNSLOTH_GEMMA4_FLASH_SLIDING=0 to revert to plain SDPA.

    Cached with maxsize=1 since the env var is read once per process. Any code or
    test that toggles UNSLOTH_GEMMA4_FLASH_SLIDING at runtime must call
    _enabled.cache_clear() afterwards for the change to take effect.
    """
    return os.environ.get("UNSLOTH_GEMMA4_FLASH_SLIDING", "1") != "0"


@functools.lru_cache(maxsize=1)
def _force_banded():
    """Optional override: force the pure-SDPA banded kernel even when flash-attn is
    importable. The router already prefers FA2 automatically, so this is only for
    benchmarking or working around a flash-attn issue; off by default.

    Cached with maxsize=1 since the env var is read once per process. Any code or
    test that toggles UNSLOTH_BANDED_SDPA at runtime must call
    _force_banded.cache_clear() afterwards for the change to take effect.
    """
    return os.environ.get("UNSLOTH_BANDED_SDPA", "0") == "1"


def _sdpa_maybe_flash_sliding(module, query, key, value, attention_mask,
                              dropout=0.0, scaling=None, is_causal=None, **kwargs):
    # Shared layer + band gate for both fast kernels (no _HAS_FA2 / dtype clause
    # here, so the banded fallback is reachable without flash-attn).
    if (_enabled()
            and getattr(module, "is_sliding", False)
            and type(module).__name__.startswith("Gemma4")
            and query.dim() == 4 and query.shape[-1] <= 256):
        w = kwargs.get("sliding_window", None) or getattr(module, "sliding_window", None)
        Sq = query.shape[2]
        Sk = key.shape[2]
        # Mirror SDPA's derivation. With no explicit mask, only a causal module
        # may take the causal window; a bidirectional call (is_causal False) must
        # stay bidirectional instead of being forced causal.
        causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
        if (w and Sq == Sk and Sq > w
                and (attention_mask is not None or causal)
                and _mask_is_plain_band(attention_mask, Sq, w)):
            # Prefer FlashAttention-2's window kernel when importable and the dtype
            # is supported; otherwise fall to the pure-SDPA banded kernel so the
            # O(S*w) speedup is automatic with or without flash-attn.
            if _HAS_FA2 and not _force_banded() and query.dtype in (torch.float16, torch.bfloat16):
                try:
                    # flash_attn_func wants (B, S, H, D); a causal window of w keys
                    # is window_size=(w-1, 0). FA2 handles GQA (H q, Hkv kv heads).
                    out = _flash_attn_func(
                        query.transpose(1, 2),
                        key.transpose(1, 2),
                        value.transpose(1, 2),
                        dropout_p=dropout if module.training else 0.0,
                        softmax_scale=scaling,
                        causal=True,
                        window_size=(w - 1, 0),
                    )
                    _ENGAGED[0] += 1
                    if _ENGAGED[0] == 1:
                        logger.info_once(
                            f"Unsloth: FlashAttention-2 sliding window engaged for gemma-4 "
                            f"(window={w})."
                        )
                    return out, None                      # (B, S, H, D), no weights
                except Exception as e:
                    logger.warning_once(f"Unsloth: gemma-4 FA2 sliding fell back to SDPA ({e})")
            else:
                try:
                    # Pure-SDPA block-local kernel; batch-general. ng folds the kv
                    # heads up to the q heads exactly as SDPA's GQA expansion does.
                    ng = getattr(module, "num_key_value_groups", query.shape[1] // key.shape[1])
                    out = _banded_sdpa_core(
                        query, key, value, w, scaling,
                        dropout if module.training else 0.0, ng,
                    )
                    _BANDED_ENGAGED[0] += 1
                    if _BANDED_ENGAGED[0] == 1:
                        logger.info_once(
                            f"Unsloth: banded sliding SDPA engaged for gemma-4 "
                            f"(window={w})."
                        )
                    return out, None                      # (B, S, H, D), no weights
                except Exception as e:
                    logger.warning_once(f"Unsloth: gemma-4 banded sliding fell back to SDPA ({e})")
    return _ORIG_SDPA[0](module, query, key, value, attention_mask,
                         dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)


def patch_gemma4_flash_sliding():
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception as e:
        raise_error("transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS", e)
        return
    try:
        current = ALL_ATTENTION_FUNCTIONS["sdpa"]
    except Exception as e:
        raise_error("ALL_ATTENTION_FUNCTIONS['sdpa']", e)
        return
    if getattr(current, "_unsloth_gemma4_flash", False):
        return
    _ORIG_SDPA[0] = current
    _sdpa_maybe_flash_sliding._unsloth_gemma4_flash = True
    # Direct assignment: AttentionInterface.register() does not update the
    # global mapping that layers read via ALL_ATTENTION_FUNCTIONS["sdpa"].
    ALL_ATTENTION_FUNCTIONS["sdpa"] = _sdpa_maybe_flash_sliding


TEMPORARY_PATCHES.append(patch_gemma4_flash_sliding)
