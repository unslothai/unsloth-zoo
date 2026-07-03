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


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
