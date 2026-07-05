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
# along with this program.  If not, see https://www.gnu.org/licenses/
#
# ============================================================================
# Banded (block-local) SDPA for sliding-window attention layers.
#
# A sliding window only needs keys in (i-w, i] for query i, but SDPA with a band
# mask still materialises the full O(S^2) score matrix. Compute it in O(S*w):
# cut the sequence into w-sized blocks, let query block b attend key blocks b-1
# and b (length 2w) under one static (w, 2w) mask (block 0 masks its phantom
# previous block), and fold blocks into the batch dim for a single SDPA call.
# Backward is exact through SDPA; fp32 rel ~1e-8 vs full SDPA + band mask.
# Engaged only when the mask is exactly the causal+sliding band with no token
# padding (fully verified, so packed masks defer), else defers to SDPA.
# Opt-in UNSLOTH_BANDED_SDPA=1.
# ============================================================================

import os
import torch
import torch.nn.functional as F
from .common import logger

__all__ = ["maybe_banded_sliding"]

_MASK_CACHE = {}    # (w, nb, device) -> (nb,1,w,2w) bool block mask
_ENGAGED = [0]      # count of banded-path invocations (debug)


def _enabled():
    return os.environ.get("UNSLOTH_BANDED_SDPA", "0") == "1"


def _block_mask(w, nb, device):
    key = (w, nb, str(device))
    m = _MASK_CACHE.get(key)
    if m is not None:
        return m
    qi = torch.arange(w, device=device)[:, None]
    ki = torch.arange(2 * w, device=device)[None, :]
    base = (ki > qi) & (ki <= qi + w)                     # (w, 2w) local band
    m = base[None].expand(nb, w, 2 * w).clone()           # (nb, w, 2w)
    m[0] &= (torch.arange(2 * w, device=device)[None, :] >= w)  # block 0: drop phantom prev-block
    m = m[:, None]                                         # (nb, 1, w, 2w)
    _MASK_CACHE[key] = m
    return m


def _mask_is_plain_band(mask, S, w, _block=1024):
    """True only if `mask` is exactly the causal+sliding band with no token padding.

    Every row is verified (not a sampled subset), so packed or padded masks, whose
    extra segment boundaries the banded path cannot honour, are always rejected. The
    verdict is stashed on the tensor itself, so it can never collide with a recycled
    object id.
    """
    if mask is None:
        return True
    if not torch.is_tensor(mask) or mask.dim() != 4:
        return False
    cached = getattr(mask, "_unsloth_plain_band", None)
    if cached is not None:
        return cached
    # The banded path applies a single block mask to every head, so a per-head
    # (shape[1] > 1) mask cannot be honoured; reject it instead of validating
    # only head 0 and silently dropping the rest.
    if mask.shape[1] != 1 or mask.shape[-2] != S or mask.shape[-1] != S:
        ok = False
    else:
        m = mask[0, 0]
        # Scan in row blocks so we never materialise a second dense S x S band
        # plus its comparison: at 16k/32k that is gigabytes of transient GPU
        # memory. Each block only builds a (block, S) band. A float mask must
        # additionally carry no in-band bias (exactly 0), else the banded path
        # would replace it with a boolean mask and silently drop the bias.
        is_bool = m.dtype == torch.bool
        idx = torch.arange(S, device=m.device)
        ok = True
        for start in range(0, S, _block):
            rows = idx[start : start + _block]                                       # (br,)
            band = (idx[None, :] <= rows[:, None]) & (idx[None, :] > rows[:, None] - w)  # (br, S)
            msl = m[start : start + _block, :]                                       # (br, S)
            match = (msl == band) if is_bool else torch.where(band, msl == 0, msl <= -1e4)
            if not bool(match.all().item()):
                ok = False
                break
    try:
        mask._unsloth_plain_band = ok
    except Exception:
        pass
    return ok


def _banded_sdpa_core(query, key, value, w, scaling, dropout, num_key_value_groups):
    # query: (B,H,S,d)   key/value: (B,Hkv,S,d)
    B, H, S, d = query.shape
    Hkv = key.shape[1]
    if num_key_value_groups > 1:
        key = key[:, :, None].expand(B, Hkv, num_key_value_groups, S, d).reshape(B, H, S, d)
        value = value[:, :, None].expand(B, Hkv, num_key_value_groups, S, d).reshape(B, H, S, d)
    Sp = ((S + w - 1) // w) * w
    if Sp != S:
        query = F.pad(query, (0, 0, 0, Sp - S))
        key = F.pad(key, (0, 0, 0, Sp - S))
        value = F.pad(value, (0, 0, 0, Sp - S))
    nb = Sp // w
    qb = query.reshape(B, H, nb, w, d).permute(0, 2, 1, 3, 4).reshape(B * nb, H, w, d)
    kpad = F.pad(key, (0, 0, w, 0))                       # (B,H, w+Sp, d)
    vpad = F.pad(value, (0, 0, w, 0))
    kw = kpad.unfold(2, 2 * w, w).permute(0, 1, 2, 4, 3)  # (B,H,nb,2w,d)
    vw = vpad.unfold(2, 2 * w, w).permute(0, 1, 2, 4, 3)
    kw = kw.permute(0, 2, 1, 3, 4).reshape(B * nb, H, 2 * w, d)
    vw = vw.permute(0, 2, 1, 3, 4).reshape(B * nb, H, 2 * w, d)
    mask = _block_mask(w, nb, query.device)
    mask = mask.expand(B, nb, 1, w, 2 * w).reshape(B * nb, 1, w, 2 * w)
    out = F.scaled_dot_product_attention(qb, kw, vw, attn_mask=mask, dropout_p=dropout, scale=scaling)
    out = out.reshape(B, nb, H, w, d).permute(0, 2, 1, 3, 4).reshape(B, H, Sp, d)[:, :, :S]
    return out.transpose(1, 2).contiguous()               # (B,S,H,d) to match sdpa_attention_forward


def maybe_banded_sliding(module, query, key, value, attention_mask,
                         dropout=0.0, scaling=None, sliding_window=None):
    """Return (attn_output, None) via the banded fast path, or None to defer.

    Called from unsloth's sdpa wrapper for sliding-window layers. Engages only
    for Gemma-4-style sliding attention (head_dim <= 256) whose mask is exactly
    the causal+sliding band with no token padding; otherwise returns None so the
    caller keeps its normal SDPA path. Opt-in via UNSLOTH_BANDED_SDPA=1.
    """
    if not _enabled():
        return None
    if not (getattr(module, "is_sliding", False) and type(module).__name__.startswith("Gemma4")):
        return None
    if query.dim() != 4 or query.shape[-1] > 256:
        return None
    w = sliding_window or getattr(module, "sliding_window", None)
    if not w:
        return None
    Sq, Sk = query.shape[2], key.shape[2]
    if not (Sq == Sk and Sq > w and query.shape[0] == 1):
        return None
    if not _mask_is_plain_band(attention_mask, Sq, w):
        return None
    try:
        ng = getattr(module, "num_key_value_groups", query.shape[1] // key.shape[1])
        out = _banded_sdpa_core(query, key, value, w, scaling,
                                dropout if module.training else 0.0, ng)
    except Exception as e:
        logger.warning_once(f"Unsloth: banded sdpa fell back to full sdpa ({e})")
        return None
    _ENGAGED[0] += 1
    if _ENGAGED[0] == 1:
        logger.info_once(f"Unsloth: banded sliding SDPA engaged (window={w}, seq={Sq}).")
    return out, None
