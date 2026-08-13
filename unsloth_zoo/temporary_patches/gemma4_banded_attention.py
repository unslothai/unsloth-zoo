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
#
# This is the pure-SDPA fallback kernel for the gemma-4 sliding-window router
# in gemma4_flash_sliding.py: it delivers the same O(S*w) speedup when
# FlashAttention-2 is not importable. The router owns all routing decisions;
# this module only exports the block-local kernel and the shared band probe.
# ============================================================================

import torch
import torch.nn.functional as F

__all__ = []

_MASK_CACHE = {}    # (w, nb, device) -> (nb,1,w,2w) bool block mask


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
    # FIFO bound: varying seq lengths yield many (w, nb) keys, so evict the oldest
    # entry once the cache hits the cap (dict is insertion-ordered) to keep this a
    # pure recompute-on-miss optimization rather than an unbounded leak.
    if len(_MASK_CACHE) >= 8:
        _MASK_CACHE.pop(next(iter(_MASK_CACHE)))
    _MASK_CACHE[key] = m
    return m


def _mask_is_plain_band(mask, S, w, _block=1024):
    """True only if `mask` is exactly the causal+sliding band with no token padding.

    Shared by both the FlashAttention-2 window path and the banded SDPA fallback,
    so both engage/defer on provably identical decisions. Every row of every batch
    element is verified (not a sampled subset), so packed or padded masks, whose
    extra segment boundaries neither path can honour, are always rejected. Per-head
    (shape[1] > 1) masks are rejected as well: the banded kernel applies one block
    mask to every head, so it cannot honour them. The verdict is stashed on the
    tensor itself, so it can never collide with a recycled object id.
    """
    if mask is None:
        return True
    if not torch.is_tensor(mask) or mask.dim() != 4:
        return False
    # Under torch.compile the band verdict drives Python control flow via .item()
    # (a data-dependent graph break -> hard error under fullgraph) and mutates a
    # tensor attribute, neither of which is traceable. Defer to the original SDPA
    # while compiling so the graph stays intact; correctness is unchanged.
    if torch.compiler.is_compiling():
        return False
    cached = getattr(mask, "_unsloth_plain_band", None)
    if cached is not None:
        return cached
    # Reject per-head masks (shape[1] > 1): a single block mask / window is
    # applied to every head, so a distinct per-head mask cannot be honoured.
    if mask.shape[1] != 1 or mask.shape[-2] != S or mask.shape[-1] != S:
        ok = False
    else:
        # Validate EVERY batch element, not just batch 0: a padded batch can carry
        # per-sample padding (for example left padding during batched generation)
        # that neither the window nor the block kernel honours, and checking only
        # batch 0 would route a padded batch to the fast path and drop the padding.
        m = mask[:, 0]                                    # (B, S, S)
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
            msl = m[:, start : start + _block, :]                                     # (B, br, S)
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
