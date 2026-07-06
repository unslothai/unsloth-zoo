"""Banded (block-local) sliding-window SDPA must match full SDPA + band mask.

`_banded_sdpa_core` computes sliding-window attention in O(S*w) by folding
w-sized query blocks into the batch dimension, each attending its own and the
previous key block under a static (w, 2w) mask. These tests compare it against
full SDPA with the explicit band mask (query i attends keys (i-w, i]) on CPU,
including GQA and sequence lengths that do not divide the window.

They also cover the unified sliding-window router in gemma4_flash_sliding.py:
with flash-attn forced unavailable the router must fall back to this banded
kernel and still match full SDPA + band mask, and UNSLOTH_BANDED_SDPA=1 must
force the banded kernel even when flash-attn looks present.
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


def test_batch_general_matches_full_sdpa_fp32():
    # The banded core is batch-general (the router drops the batch == 1 gate), so
    # a B > 1 unpadded band must match full SDPA + band mask on every element.
    torch.manual_seed(2)
    B, S, w, H, Hkv, d = 3, 300, 64, 8, 2, 32
    q = torch.randn(B, H, S, d)
    k = torch.randn(B, Hkv, S, d)
    v = torch.randn(B, Hkv, S, d)
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


def test_mask_is_plain_band_rejects_non_probe_violation():
    # A band broken only at an interior row (a packed-sequence boundary) must be
    # rejected: the verifier checks every row, not a sampled subset.
    S, w = 128, 32
    packed = _band_mask_full(S, w)[None, None].clone()
    packed[..., 50, 40] = False    # drop an in-band key at a non-probe row
    assert not _mask_is_plain_band(packed, S, w)


def test_mask_is_plain_band_rejects_per_head_mask():
    # A 4D mask with a head dim > 1 cannot be honoured by the single block mask
    # applied to every head, so it must be rejected even if head 0 is a band.
    S, w = 128, 32
    band = _band_mask_full(S, w)
    per_head = band[None, None].expand(1, 4, S, S).clone()   # (1, 4, S, S)
    assert not _mask_is_plain_band(per_head, S, w)
    assert _mask_is_plain_band(band[None, None], S, w)       # (1, 1, S, S) still ok


def test_mask_is_plain_band_rejects_per_batch_padding():
    # The probe validates every batch element, not just batch 0: a batch whose
    # element 0 is a full band but a later element carries padding must be
    # rejected so the fast path never runs on an unhonoured padded sample.
    S, w = 128, 32
    band = _band_mask_full(S, w)                              # (S, S)
    batched = band[None, None].expand(1, 1, S, S).clone()
    batched = batched.expand(3, 1, S, S).clone()             # (3, 1, S, S), all bands
    assert _mask_is_plain_band(batched.clone(), S, w)        # every element a band -> ok
    padded = batched.clone()
    padded[2, 0, :, 0] = False                               # break the band only in element 2
    assert not _mask_is_plain_band(padded, S, w)


def test_mask_is_plain_band_rejects_inband_bias():
    # A finite in-band bias (soft penalty) must be rejected: the banded path
    # swaps in a boolean block mask and would silently drop the bias.
    S, w = 128, 32
    band = _band_mask_full(S, w)
    biased = torch.where(band, -5.0, float("-inf")).to(torch.float32)[None, None]
    assert not _mask_is_plain_band(biased, S, w)
    clean = torch.where(band, 0.0, float("-inf")).to(torch.float32)[None, None]
    assert _mask_is_plain_band(clean, S, w)


def test_mask_is_plain_band_chunked_equals_single_block():
    # Row-chunked verification must agree with a single-block scan on every mask.
    S, w = 96, 16
    band = _band_mask_full(S, w)
    packed = band.clone(); packed[50, 40] = False
    cases = {
        "band_bool": band[None, None],
        "band_float": torch.where(band, 0.0, float("-inf")).to(torch.float32)[None, None],
        "packed_bool": packed[None, None],
    }
    for name, m in cases.items():
        full = _mask_is_plain_band(m.clone(), S, w, _block=S)
        chunked = _mask_is_plain_band(m.clone(), S, w, _block=7)  # crosses block edges
        assert full == chunked, name


def test_mask_is_plain_band_survives_id_reuse():
    # A recycled object id must not produce a stale cached verdict for packed masks.
    S, w = 128, 32
    template = _band_mask_full(S, w)[None, None].clone()
    template[..., 50, 40] = False    # packed-style violation on a non-probe row
    valid = _band_mask_full(S, w)[None, None]
    assert _mask_is_plain_band(valid, S, w)    # caches a True verdict on this object
    del valid                                  # frees its id for reuse
    for _ in range(5000):
        packed = template.clone()              # a distinct object, may land on the freed id
        assert not _mask_is_plain_band(packed, S, w)
        del packed


class Gemma4SlidingFake:
    """Minimal stand-in for a Gemma-4 sliding-window attention module.

    The router gates on ``is_sliding``, a class name starting with ``Gemma4`` and
    ``head_dim <= 256``; this reproduces exactly those attributes so the routing
    branch can be exercised without loading a real model.
    """
    is_sliding = True
    is_causal = True
    training = False

    def __init__(self, w, ng):
        self.sliding_window = w
        self.num_key_value_groups = ng


@pytest.mark.parametrize("with_mask", [False, True])
def test_router_falls_back_to_banded_without_flash(monkeypatch, with_mask):
    # With flash-attn forced unavailable the unified wrapper must route a sliding
    # Gemma-4 plain-band case through _banded_sdpa_core (never the wrapped SDPA)
    # and match full SDPA + band mask. Runs on CPU regardless of flash-attn.
    from unsloth_zoo.temporary_patches import gemma4_flash_sliding as gf

    monkeypatch.setattr(gf, "_HAS_FA2", False)
    monkeypatch.setattr(gf, "_flash_attn_func", None)
    monkeypatch.setenv("UNSLOTH_GEMMA4_FLASH_SLIDING", "1")
    gf._enabled.cache_clear()  # _enabled is lru_cache(1); re-read the toggled env
    monkeypatch.delenv("UNSLOTH_BANDED_SDPA", raising=False)

    S, w, H, Hkv, d = 300, 64, 8, 2, 32
    torch.manual_seed(0)
    q = torch.randn(1, H, S, d)
    k = torch.randn(1, Hkv, S, d)
    v = torch.randn(1, Hkv, S, d)
    scale = d ** -0.5
    ng = H // Hkv
    module = Gemma4SlidingFake(w, ng)

    if with_mask:
        attn_mask = _band_mask_full(S, w)[None, None].clone()
    else:
        attn_mask = None

    def _boom(*a, **kw):
        raise AssertionError("router fell through to SDPA instead of the banded kernel")

    orig = gf._ORIG_SDPA[0]
    gf._ORIG_SDPA[0] = _boom
    before = gf._BANDED_ENGAGED[0]
    try:
        out, weights = gf._sdpa_maybe_flash_sliding(
            module, q, k, v, attn_mask, dropout=0.0, scaling=scale, is_causal=None,
        )
    finally:
        gf._ORIG_SDPA[0] = orig

    assert weights is None
    assert gf._BANDED_ENGAGED[0] == before + 1, "banded path was not engaged"
    ref = _reference(q, k, v, w, scale)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_router_force_banded_env_skips_flash(monkeypatch):
    # UNSLOTH_BANDED_SDPA=1 must force the banded kernel even when flash-attn is
    # importable: the flash function must never be called.
    from unsloth_zoo.temporary_patches import gemma4_flash_sliding as gf

    def _flash_should_not_run(*a, **kw):
        raise AssertionError("flash_attn_func called despite UNSLOTH_BANDED_SDPA=1")

    monkeypatch.setattr(gf, "_HAS_FA2", True)
    monkeypatch.setattr(gf, "_flash_attn_func", _flash_should_not_run)
    monkeypatch.setenv("UNSLOTH_GEMMA4_FLASH_SLIDING", "1")
    gf._enabled.cache_clear()  # _enabled is lru_cache(1); re-read the toggled env
    monkeypatch.setenv("UNSLOTH_BANDED_SDPA", "1")

    S, w, H, Hkv, d = 256, 64, 4, 4, 32
    torch.manual_seed(3)
    q = torch.randn(1, H, S, d)
    k = torch.randn(1, Hkv, S, d)
    v = torch.randn(1, Hkv, S, d)
    scale = d ** -0.5
    module = Gemma4SlidingFake(w, H // Hkv)

    orig = gf._ORIG_SDPA[0]
    gf._ORIG_SDPA[0] = lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("router fell through to SDPA instead of the banded kernel"))
    before = gf._BANDED_ENGAGED[0]
    try:
        out, weights = gf._sdpa_maybe_flash_sliding(
            module, q, k, v, None, dropout=0.0, scaling=scale, is_causal=None,
        )
    finally:
        gf._ORIG_SDPA[0] = orig

    assert weights is None
    assert gf._BANDED_ENGAGED[0] == before + 1
    ref = _reference(q, k, v, w, scale)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_router_kill_switch_defers_to_sdpa(monkeypatch):
    # UNSLOTH_GEMMA4_FLASH_SLIDING=0 must disable both fast kernels and defer to
    # the wrapped SDPA, neither FA2 nor banded may engage.
    from unsloth_zoo.temporary_patches import gemma4_flash_sliding as gf

    monkeypatch.setattr(gf, "_HAS_FA2", False)
    monkeypatch.setattr(gf, "_flash_attn_func", None)
    monkeypatch.setenv("UNSLOTH_GEMMA4_FLASH_SLIDING", "0")
    gf._enabled.cache_clear()  # _enabled is lru_cache(1); re-read the toggled env

    S, w, H, d = 256, 64, 4, 32
    torch.manual_seed(4)
    q = torch.randn(1, H, S, d)
    k = torch.randn(1, H, S, d)
    v = torch.randn(1, H, S, d)
    scale = d ** -0.5
    module = Gemma4SlidingFake(w, 1)

    sentinel = object()
    called = {}

    def _fake_sdpa(mod, qq, kk, vv, am, dropout=0.0, scaling=None, is_causal=None, **kw):
        called["hit"] = True
        return sentinel, None

    banded_before = gf._BANDED_ENGAGED[0]
    orig = gf._ORIG_SDPA[0]
    gf._ORIG_SDPA[0] = _fake_sdpa
    try:
        out, _ = gf._sdpa_maybe_flash_sliding(
            module, q, k, v, None, dropout=0.0, scaling=scale, is_causal=None,
        )
    finally:
        gf._ORIG_SDPA[0] = orig

    assert called.get("hit") is True
    assert out is sentinel
    assert gf._BANDED_ENGAGED[0] == banded_before, "banded engaged despite kill switch"
