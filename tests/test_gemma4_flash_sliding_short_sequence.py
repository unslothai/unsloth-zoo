# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""A sequence no longer than the sliding window must still leave the masked SDPA path."""
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class Gemma4SlidingFake(torch.nn.Module):
    def __init__(self, w, ng):
        super().__init__()
        self.is_sliding = True
        self.is_causal = True
        self.sliding_window = w
        self.num_key_value_groups = ng
        self.training = True


def _causal_ref(q, k, v, scaling, ng):
    B, H, S, d = q.shape
    Hkv = k.shape[1]
    if ng > 1:
        k = k[:, :, None].expand(B, Hkv, ng, S, d).reshape(B, H, S, d)
        v = v[:, :, None].expand(B, Hkv, ng, S, d).reshape(B, H, S, d)
    out = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True, scale=scaling)
    return out.transpose(1, 2).contiguous()


def _route(gf, module, q, k, v, scaling, mask=None):
    def _boom(*a, **kw):
        raise AssertionError("router fell through to the wrapped (masked) SDPA")
    orig = gf._ORIG_SDPA[0]
    gf._ORIG_SDPA[0] = _boom
    try:
        return gf._sdpa_maybe_flash_sliding(module, q, k, v, mask, dropout=0.0, scaling=scaling, is_causal=None)
    finally:
        gf._ORIG_SDPA[0] = orig


@pytest.mark.parametrize("S,w", [(1024, 1024), (512, 1024)])
@pytest.mark.parametrize("banded", [False, True])
def test_short_sequence_leaves_masked_sdpa(monkeypatch, S, w, banded):
    from unsloth_zoo.temporary_patches import gemma4_flash_sliding as gf
    monkeypatch.setenv("UNSLOTH_GEMMA4_FLASH_SLIDING", "1"); gf._enabled.cache_clear()
    if banded:
        monkeypatch.setenv("UNSLOTH_BANDED_SDPA", "1")     # no flash-attn: the causal SDPA branch
    else:
        monkeypatch.delenv("UNSLOTH_BANDED_SDPA", raising=False)
    gf._force_banded.cache_clear()
    if not banded and not gf._HAS_FA2:
        pytest.skip("flash_attn not importable; the FA2 arm cannot be exercised")
    H, Hkv, d = 16, 8, 256
    torch.manual_seed(0)
    q = torch.randn(2, H, S, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(2, Hkv, S, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(2, Hkv, S, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    scaling = d ** -0.5
    module = Gemma4SlidingFake(w, H // Hkv)
    fa2_before, causal_before = gf._ENGAGED[0], gf._CAUSAL_ENGAGED[0]
    out, weights = _route(gf, module, q, k, v, scaling)
    assert weights is None
    if banded:
        assert gf._CAUSAL_ENGAGED[0] == causal_before + 1, "mask-free causal SDPA was not engaged"
    else:
        assert gf._ENGAGED[0] == fa2_before + 1, "FA2 path was not engaged"
    ref = _causal_ref(q, k, v, scaling, H // Hkv)
    rel = (out.float() - ref).norm() / ref.norm().clamp_min(1e-9)
    assert rel < 3e-2, f"forward rel {rel:.2e}"
    g = torch.randn_like(ref)
    (ref * g).sum().backward(); dq_ref = q.grad.float().clone(); q.grad = None
    (out.float() * g).sum().backward()
    dq_rel = (q.grad.float() - dq_ref).norm() / dq_ref.norm().clamp_min(1e-9)
    assert dq_rel < 3e-2, f"dq rel {dq_rel:.2e}"


def test_short_sequence_with_explicit_causal_mask_engages(monkeypatch):
    from unsloth_zoo.temporary_patches import gemma4_flash_sliding as gf
    monkeypatch.setenv("UNSLOTH_GEMMA4_FLASH_SLIDING", "1"); gf._enabled.cache_clear()
    monkeypatch.setenv("UNSLOTH_BANDED_SDPA", "1"); gf._force_banded.cache_clear()
    S, w, H, Hkv, d = 512, 1024, 16, 8, 256
    q = torch.randn(1, H, S, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, Hkv, S, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, Hkv, S, d, device="cuda", dtype=torch.bfloat16)
    idx = torch.arange(S, device="cuda")
    mask = (idx[None, :] <= idx[:, None])[None, None]
    out, _ = _route(gf, Gemma4SlidingFake(w, H // Hkv), q, k, v, d ** -0.5, mask=mask)
    ref = _causal_ref(q, k, v, d ** -0.5, H // Hkv)
    assert ((out.float() - ref).norm() / ref.norm()) < 3e-2


def test_padded_short_sequence_still_defers(monkeypatch):
    """Padding is not a band: the wrapped SDPA must still get it."""
    from unsloth_zoo.temporary_patches import gemma4_flash_sliding as gf
    monkeypatch.setenv("UNSLOTH_GEMMA4_FLASH_SLIDING", "1"); gf._enabled.cache_clear()
    S, w, H, Hkv, d = 512, 1024, 16, 8, 256
    q = torch.randn(1, H, S, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, Hkv, S, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, Hkv, S, d, device="cuda", dtype=torch.bfloat16)
    idx = torch.arange(S, device="cuda")
    mask = (idx[None, :] <= idx[:, None]).clone()
    mask[:, :8] = False                                     # left padding
    mask = mask[None, None]
    with pytest.raises(AssertionError, match="fell through"):
        _route(gf, Gemma4SlidingFake(w, H // Hkv), q, k, v, d ** -0.5, mask=mask)
