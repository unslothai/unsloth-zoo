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
# FlashAttention-2 for Gemma-4 sliding-window layers.
#
# Gemma-4's head_dim 512 global layers force FA2 off model-wide, so all 30
# layers fall back to O(S^2) SDPA. But a causal sliding window is exactly
# FA2's window_size=(w-1, 0), so route just the sliding layers (head_dim <=
# 256, plain causal band mask, no padding) through FA2; global layers and
# padded / non-band masks defer unchanged to the original SDPA. Backward is
# exact through FA2; numerics match SDPA to bf16 rounding.
#
# On by default when flash-attn is importable; UNSLOTH_GEMMA4_FLASH_SLIDING=0
# reverts to SDPA.
# ============================================================================

import os
import torch
from .common import TEMPORARY_PATCHES, logger
from .utils import raise_error

__all__ = ["patch_gemma4_flash_sliding"]

try:
    from flash_attn import flash_attn_func as _flash_attn_func
    _HAS_FA2 = True
except Exception:
    _flash_attn_func = None
    _HAS_FA2 = False

_BAND_OK = {}        # id(attention_mask) -> bool (plain causal band, no padding)
_ORIG_SDPA = [None]  # boxed reference to the wrapped sdpa function
_ENGAGED = [0]       # count of FA2-path invocations (debug)


def _enabled():
    return os.environ.get("UNSLOTH_GEMMA4_FLASH_SLIDING", "1") != "0"


def _mask_is_plain_band(mask, S, w):
    """True if `mask` is exactly the causal + sliding band with no token padding."""
    if mask is None:
        return True
    if not torch.is_tensor(mask) or mask.dim() != 4:
        return False
    key = id(mask)
    cached = _BAND_OK.get(key)
    if cached is not None:
        return cached
    if len(_BAND_OK) > 64:
        _BAND_OK.clear()
    m = mask[:, 0]
    if m.shape[-2] != S or m.shape[-1] != S:
        _BAND_OK[key] = False
        return False
    allowed = m if m.dtype == torch.bool else (m > -1e4)
    dev = m.device
    cand = sorted({x for x in (0, 1, w - 1, w, w + 1, 2 * w, S // 2, S - 1, S - w, S - w - 1) if 0 <= x < S})
    probes = torch.tensor(cand, device=dev)
    # Probe EVERY batch element: with right padding only the longest sample
    # keeps a clean band, and FA2's window_size cannot express per-sample
    # padding, so any deviating element must reject the whole batch.
    rows = allowed[:, probes]                             # (B, P, S)
    idx = torch.arange(S, device=dev)[None, None, :]
    cnt = rows.sum(-1)
    minidx = torch.where(rows, idx, torch.full_like(idx, S)).amin(-1)
    maxidx = torch.where(rows, idx, torch.full_like(idx, -1)).amax(-1)
    lo = torch.clamp(probes - w + 1, min=0)[None, :]
    hi = probes[None, :]
    ok = bool(((cnt == (hi - lo + 1)) & (minidx == lo) & (maxidx == hi)).all().item())
    _BAND_OK[key] = ok
    return ok


def _sdpa_maybe_flash_sliding(module, query, key, value, attention_mask,
                              dropout=0.0, scaling=None, is_causal=None, **kwargs):
    if (_enabled() and _HAS_FA2
            and getattr(module, "is_sliding", False)
            and type(module).__name__.startswith("Gemma4")
            and query.dim() == 4 and query.shape[-1] <= 256
            and query.dtype in (torch.float16, torch.bfloat16)):
        w = kwargs.get("sliding_window", None) or getattr(module, "sliding_window", None)
        Sq = query.shape[2]
        Sk = key.shape[2]
        if (w and Sq == Sk and Sq > w and _mask_is_plain_band(attention_mask, Sq, w)):
            try:
                # flash_attn_func wants (B, S, H, D); a causal window of w keys is
                # window_size=(w-1, 0). FA2 handles GQA (H q heads, Hkv kv heads).
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
                return out, None                          # (B, S, H, D), no weights
            except Exception as e:
                logger.warning_once(f"Unsloth: gemma-4 FA2 sliding fell back to SDPA ({e})")
    return _ORIG_SDPA[0](module, query, key, value, attention_mask,
                         dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)


def patch_gemma4_flash_sliding():
    if not _HAS_FA2:
        return
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception as e:
        return raise_error("transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS", e)
    try:
        current = ALL_ATTENTION_FUNCTIONS["sdpa"]
    except Exception as e:
        return raise_error("ALL_ATTENTION_FUNCTIONS['sdpa']", e)
    if getattr(current, "_unsloth_gemma4_flash", False):
        return  # already wrapped
    _ORIG_SDPA[0] = current
    _sdpa_maybe_flash_sliding._unsloth_gemma4_flash = True
    # Direct assignment: AttentionInterface.register() does not update the
    # global mapping that layers read via ALL_ATTENTION_FUNCTIONS["sdpa"].
    ALL_ATTENTION_FUNCTIONS["sdpa"] = _sdpa_maybe_flash_sliding


TEMPORARY_PATCHES.append(patch_gemma4_flash_sliding)
