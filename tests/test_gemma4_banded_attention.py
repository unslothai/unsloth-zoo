"""Banded (block-local) sliding-window SDPA must match full SDPA + band mask.

`_banded_sdpa_core` computes sliding-window attention in O(S*w) by folding
w-sized query blocks into the batch dimension, each attending its own and the
previous key block under a static (w, 2w) mask. These tests compare it against
full SDPA with the explicit band mask (query i attends keys (i-w, i]) on CPU,
including GQA and sequence lengths that do not divide the window.
"""

import pytest
import torch
import torch.nn.functional as F

from unsloth_zoo.temporary_patches.gemma4_banded_attention import (
    _banded_sdpa_core,
    _block_mask,
    _mask_is_plain_band,
)


def _band_mask_full(S, w, device="cpu"):
    qi = torch.arange(S, device=device)[:, None]
    kj = torch.arange(S, device=device)[None, :]
    return (kj <= qi) & (kj > qi - w)


def _reference(q, k, v, w, scale):
    B, H, S, d = q.shape
    Hkv = k.shape[1]
    ng = H // Hkv
    if ng > 1:
        k = k[:, :, None].expand(B, Hkv, ng, S, d).reshape(B, H, S, d)
        v = v[:, :, None].expand(B, Hkv, ng, S, d).reshape(B, H, S, d)
    m = _band_mask_full(S, w, q.device)[None, None]
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=m, scale=scale)
    return out.transpose(1, 2).contiguous()  # (B,S,H,d), matching the core


@pytest.mark.parametrize("S,w,H,Hkv,d", [
    (256, 64, 4, 4, 32),     # no GQA, S divisible by w
    (300, 64, 8, 2, 32),     # GQA 4, ragged tail block
    (513, 128, 4, 1, 16),    # MQA, S = 4w + 1
])
def test_matches_full_sdpa_fp32(S, w, H, Hkv, d):
    torch.manual_seed(0)
    q = torch.randn(1, H, S, d)
    k = torch.randn(1, Hkv, S, d)
    v = torch.randn(1, Hkv, S, d)
    scale = d ** -0.5
    ref = _reference(q, k, v, w, scale)
    out = _banded_sdpa_core(q, k, v, w, scale, 0.0, H // Hkv)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_gradients_match_fp32():
    torch.manual_seed(1)
    S, w, H, d = 256, 64, 2, 16
    q1 = torch.randn(1, H, S, d, requires_grad=True)
    k1 = torch.randn(1, H, S, d, requires_grad=True)
    v1 = torch.randn(1, H, S, d, requires_grad=True)
    q2 = q1.detach().clone().requires_grad_(True)
    k2 = k1.detach().clone().requires_grad_(True)
    v2 = v1.detach().clone().requires_grad_(True)
    scale = d ** -0.5

    _reference(q1, k1, v1, w, scale).square().sum().backward()
    _banded_sdpa_core(q2, k2, v2, w, scale, 0.0, 1).square().sum().backward()

    torch.testing.assert_close(q2.grad, q1.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(k2.grad, k1.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(v2.grad, v1.grad, rtol=1e-4, atol=1e-5)


def test_block_mask_block0_masks_phantom_prev():
    w, nb = 8, 3
    m = _block_mask(w, nb, torch.device("cpu"))
    assert m.shape == (nb, 1, w, 2 * w)
    # Block 0 must not attend the (padded) previous block.
    assert not m[0, 0, :, :w].any()
    # Later blocks follow the same local band for every block index.
    assert torch.equal(m[1], m[2])


def test_mask_is_plain_band_detection():
    S, w = 128, 32
    band = _band_mask_full(S, w)[None, None]
    assert _mask_is_plain_band(band, S, w)
    # Padding one token out of the band pattern must be rejected.
    padded = band.clone()
    padded[..., S - 1, 0:1] = True
    assert not _mask_is_plain_band(padded, S, w)
    assert _mask_is_plain_band(None, S, w)
