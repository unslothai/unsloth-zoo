"""FA2 sliding-window path for gemma-4 must match full SDPA with the band mask.

Gemma-4 mixes head_dim=256 sliding-window layers with head_dim=512 global
layers, so FlashAttention-2 (head_dim <= 256) is disabled model-wide and the
sliding layers fall back to full O(S^2) SDPA. Routing the sliding layers to
FA2 with window_size=(w-1, 0) reproduces the causal sliding band exactly. This
checks forward + backward parity against SDPA with the explicit band mask, and
that the mask probe accepts a genuine band / rejects a padded one.
"""
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

flash_attn_func = pytest.importorskip("flash_attn").flash_attn_func


def _full_sdpa_band(q, k, v, w, scaling, ng):
    B, H, S, d = q.shape
    Hkv = k.shape[1]
    if ng > 1:
        k = k[:, :, None].expand(B, Hkv, ng, S, d).reshape(B, H, S, d)
        v = v[:, :, None].expand(B, Hkv, ng, S, d).reshape(B, H, S, d)
    qi = torch.arange(S, device=q.device)[:, None]
    ki = torch.arange(S, device=q.device)[None, :]
    allowed = (ki <= qi) & (ki > qi - w)
    mask = torch.zeros(S, S, device=q.device, dtype=q.dtype).masked_fill(~allowed, float("-inf"))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[None, None], scale=scaling)
    return out.transpose(1, 2).contiguous()


@pytest.mark.parametrize("S,w", [(4096, 1024), (8192, 1024)])
@pytest.mark.parametrize("H,Hkv", [(16, 8), (8, 8)])
def test_fa2_sliding_matches_sdpa_band(S, w, H, Hkv):
    torch.manual_seed(0)
    d = 256
    q = torch.randn(1, H, S, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, Hkv, S, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, Hkv, S, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    scaling = d ** -0.5
    ng = H // Hkv

    ref = _full_sdpa_band(q, k, v, w, scaling, ng)
    fa = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                         softmax_scale=scaling, causal=True, window_size=(w - 1, 0))
    fwd_rel = (fa.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-9)
    assert fwd_rel < 3e-2, f"forward rel {fwd_rel:.2e}"

    g = torch.randn_like(ref)
    (ref * g).sum().backward()
    dq_ref = q.grad.clone(); q.grad = None
    (fa * g).sum().backward()
    dq_rel = (q.grad.float() - dq_ref.float()).norm() / dq_ref.float().norm().clamp_min(1e-9)
    assert dq_rel < 3e-2, f"dq rel {dq_rel:.2e}"


def test_mask_probe_accepts_band_rejects_padding():
    from unsloth_zoo.temporary_patches import gemma4_flash_sliding as gf
    S, w = 512, 128
    qi = torch.arange(S, device="cuda")[:, None]
    ki = torch.arange(S, device="cuda")[None, :]
    band = (ki <= qi) & (ki > qi - w)
    mask = torch.where(band, 0.0, float("-inf")).to(torch.float32)[None, None]
    assert gf._mask_is_plain_band(mask, S, w) is True
    padded = mask.clone()
    padded[..., :, S // 2] = float("-inf")
    assert gf._mask_is_plain_band(padded, S, w) is False


def _band(S, w, device="cuda"):
    idx = torch.arange(S, device=device)
    return (idx[None, :] <= idx[:, None]) & (idx[None, :] > idx[:, None] - w)


def test_mask_probe_rejects_packed_boundary():
    # An interior band break (a packed-sequence boundary) must be rejected; the
    # whole-mask verification does not sample rows.
    from unsloth_zoo.temporary_patches import gemma4_flash_sliding as gf
    S, w = 512, 128
    band = _band(S, w)
    packed = torch.where(band, 0.0, float("-inf")).to(torch.float32)[None, None]
    packed[..., 300, 200] = float("-inf")     # drop an in-band key at an interior row
    assert gf._mask_is_plain_band(packed, S, w) is False


def test_float_mask_inband_bias_rejected():
    # A finite in-band bias would be silently dropped when routing to FA2, so a
    # float mask is only accepted when in-band entries are exactly 0.
    from unsloth_zoo.temporary_patches import gemma4_flash_sliding as gf
    S, w = 256, 64
    band = _band(S, w)
    biased = torch.where(band, -5.0, float("-inf")).to(torch.float32)[None, None]
    assert gf._mask_is_plain_band(biased, S, w) is False
    clean = torch.where(band, 0.0, float("-inf")).to(torch.float32)[None, None]
    assert gf._mask_is_plain_band(clean, S, w) is True


def test_chunked_verification_matches_dense():
    # Row-chunked verification must agree with a single-block scan on band,
    # padded, and packed masks (the memory win needs no test).
    from unsloth_zoo.temporary_patches import gemma4_flash_sliding as gf
    S, w = 96, 16
    band = _band(S, w)
    band_f = torch.where(band, 0.0, float("-inf")).to(torch.float32)[None, None]
    padded = band_f.clone(); padded[..., :, S // 2] = float("-inf")
    packed = band_f.clone(); packed[..., 50, 40] = float("-inf")
    for name, m in {"band_bool": band[None, None], "band_float": band_f,
                    "padded": padded, "packed": packed}.items():
        full = gf._mask_is_plain_band(m.clone(), S, w, _block=S)
        chunked = gf._mask_is_plain_band(m.clone(), S, w, _block=7)  # crosses block edges
        assert full == chunked, name


class Gemma4SlidingFake:
    """Minimal stand-in for a Gemma-4 sliding-window attention module: just the
    attributes the router gates on (is_sliding, class name, head_dim <= 256)."""
    is_sliding = True
    is_causal = True
    training = False

    def __init__(self, w, ng):
        self.sliding_window = w
        self.num_key_value_groups = ng


def test_router_routes_to_fa2_and_matches_band(monkeypatch):
    # With flash-attn present the unified wrapper must take the FA2 window branch
    # (never the wrapped SDPA) and match full SDPA + band mask.
    from unsloth_zoo.temporary_patches import gemma4_flash_sliding as gf

    monkeypatch.setenv("UNSLOTH_GEMMA4_FLASH_SLIDING", "1")
    gf._enabled.cache_clear()  # _enabled is lru_cache(1); re-read the toggled env
    monkeypatch.delenv("UNSLOTH_BANDED_SDPA", raising=False)
    gf._force_banded.cache_clear()  # _force_banded is lru_cache(1); re-read the toggled env
    assert gf._HAS_FA2, "flash_attn imported by the test but not by the router"

    S, w, H, Hkv, d = 4096, 1024, 16, 8, 256
    torch.manual_seed(0)
    q = torch.randn(1, H, S, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, Hkv, S, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, Hkv, S, d, device="cuda", dtype=torch.bfloat16)
    scaling = d ** -0.5
    ng = H // Hkv
    module = Gemma4SlidingFake(w, ng)

    def _boom(*a, **kw):
        raise AssertionError("router did not take the FA2 path")

    orig = gf._ORIG_SDPA[0]
    gf._ORIG_SDPA[0] = _boom
    before = gf._ENGAGED[0]
    try:
        out, weights = gf._sdpa_maybe_flash_sliding(
            module, q, k, v, None, dropout=0.0, scaling=scaling, is_causal=None,
        )
    finally:
        gf._ORIG_SDPA[0] = orig

    assert weights is None
    assert gf._ENGAGED[0] == before + 1, "FA2 path was not engaged"
    ref = _full_sdpa_band(q, k, v, w, scaling, ng)
    rel = (out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-9)
    assert rel < 3e-2, f"router FA2 rel {rel:.2e}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
