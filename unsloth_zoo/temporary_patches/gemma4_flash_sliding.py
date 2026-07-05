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
# Gemma-4's head_dim 512 global layers force FA2 off model-wide, so every layer
# falls back to O(S^2) SDPA. A causal sliding window is exactly FA2's
# window_size=(w-1, 0), so route only the sliding layers (head_dim <= 256, plain
# causal band mask, no padding) through FA2; global and padded / non-band masks
# defer to the original SDPA. Backward is exact; numerics match SDPA to bf16.
# On by default with flash-attn; UNSLOTH_GEMMA4_FLASH_SLIDING=0 reverts to SDPA.
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

_ORIG_SDPA = [None]  # boxed reference to the wrapped sdpa function
_ENGAGED = [0]       # count of FA2-path invocations (debug)


def _enabled():
    return os.environ.get("UNSLOTH_GEMMA4_FLASH_SLIDING", "1") != "0"


def _mask_is_plain_band(mask, S, w, _block=1024):
    """True if `mask` is exactly the causal + sliding band with no token padding."""
    if mask is None:
        return True
    if not torch.is_tensor(mask) or mask.dim() != 4:
        return False
    # Cache the verdict on the tensor itself. An id()-keyed dict can alias a
    # freed mask to a newly allocated one and return a stale result; binding to
    # the tensor lifetime avoids that. `is not None` preserves a cached False.
    cached = getattr(mask, "_unsloth_plain_band", None)
    if cached is not None:
        return cached
    if mask.shape[-2] != S or mask.shape[-1] != S:
        try: mask._unsloth_plain_band = False
        except (AttributeError, RuntimeError): pass
        return False
    # Verify the WHOLE mask over every batch element and head. A sampled-row
    # probe would miss a packed-sequence boundary that falls between the sampled
    # rows and let FA2 attend across samples. A plain band allows key j for
    # query i iff i - w < j <= i; anything else (padding, packing) must reject.
    # Scan in row blocks so we never materialise a second S x S tensor: at
    # 16k/32k contexts a full S x S band plus its comparison would add multiple
    # GB of transient memory and can OOM before the FA2 path even runs. Each
    # block only builds a (block, S) band. A float mask must additionally carry
    # no in-band bias (exactly 0) so routing to FA2 cannot silently drop it.
    is_bool = mask.dtype == torch.bool
    idx = torch.arange(S, device=mask.device)
    ok = True
    for start in range(0, S, _block):
        rows = idx[start : start + _block]                                       # (br,)
        band = (idx[None, :] <= rows[:, None]) & (idx[None, :] > rows[:, None] - w)  # (br, S)
        msl = mask[..., start : start + _block, :]                               # (..., br, S)
        match = (msl == band) if is_bool else torch.where(band, msl == 0, msl <= -1e4)
        if not bool(match.all().item()):
            ok = False
            break
    try: mask._unsloth_plain_band = ok
    except (AttributeError, RuntimeError): pass
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
        # Mirror SDPA's derivation. With no explicit mask, only a causal module
        # may take the causal FA2 window; a bidirectional call (is_causal False)
        # must stay bidirectional instead of being forced causal.
        causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
        if (w and Sq == Sk and Sq > w
                and (attention_mask is not None or causal)
                and _mask_is_plain_band(attention_mask, Sq, w)):
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
