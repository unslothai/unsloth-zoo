"""Muse Glimmer sliding layers must route to the banded O(S*w) kernel, and only there.

Muse Glimmer interleaves 39 sliding_attention layers (window 2048) with 13
full_attention layers, and hands every sliding layer a dense [B, 1, S, S] band
mask, so each one pays O(S^2) for a 2048 key window. The router in
``muse_glimmer_banded_attention`` sends exactly those layers to the block-local
kernel shared with gemma-4 and defers everything else.

These run on CPU (float32, small shapes), so they are real CI coverage rather
than GPU-only. They pin three things: the gate defers in every case it must,
the engaged path matches a reference causal+sliding SDPA, and installing the
router is idempotent across the three passes unsloth makes over
TEMPORARY_PATCHES without breaking gemma-4's own re-entry guard.
"""
import pytest
import torch
import torch.nn.functional as F

from unsloth_zoo.temporary_patches import muse_glimmer_banded_attention as mg

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
W = 16
S_ON = mg._MIN_SEQ_MULTIPLE_OF_WINDOW * W          # first length that engages
H, HKV, D = 4, 2, 8


def _reference_band(q, k, v, w, scaling, ng):
    """Full SDPA with the explicit causal+sliding band: what the model does today."""
    B, Hq, S, d = q.shape
    Hkv = k.shape[1]
    if ng > 1:
        k = k[:, :, None].expand(B, Hkv, ng, S, d).reshape(B, Hq, S, d)
        v = v[:, :, None].expand(B, Hkv, ng, S, d).reshape(B, Hq, S, d)
    idx = torch.arange(S, device=q.device)
    allowed = (idx[None, :] <= idx[:, None]) & (idx[None, :] > idx[:, None] - w)
    mask = torch.zeros(S, S, device=q.device, dtype=q.dtype).masked_fill(
        ~allowed, float("-inf"))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[None, None], scale=scaling)
    return out.transpose(1, 2).contiguous()


def _band_mask(S, w, device=DEVICE):
    idx = torch.arange(S, device=device)
    return ((idx[None, :] <= idx[:, None]) & (idx[None, :] > idx[:, None] - w))[None, None]


class MuseGlimmerTextAttention:
    """Stand-in carrying only the attributes the router gates on."""
    is_causal = True
    training = False

    def __init__(self, w=W, ng=H // HKV, is_local_attention=True):
        self.sliding_window = w
        self.num_key_value_groups = ng
        self.is_local_attention = is_local_attention


class MuseGlimmerVisionAttention(MuseGlimmerTextAttention):
    """Vision tower class name: must never be routed, whatever its flags say."""


@pytest.fixture(autouse=True)
def _clean_router(monkeypatch):
    """Every test drives the router directly, so give it a known wrapped sdpa and
    a clean env. The env readers are lru_cache(1), so they need cache_clear()."""
    monkeypatch.delenv("UNSLOTH_MUSE_GLIMMER_BANDED_SDPA", raising=False)
    monkeypatch.delenv("UNSLOTH_MUSE_GLIMMER_BANDED_FORCE", raising=False)
    mg._enabled.cache_clear()
    mg._force_banded.cache_clear()
    calls = []

    def fake_sdpa(module, q, k, v, mask, dropout=0.0, scaling=None, is_causal=None, **kw):
        calls.append({"module": module, "mask": mask, "is_causal": is_causal})
        return "DEFERRED", None

    prev = mg._ORIG_SDPA[0]
    mg._ORIG_SDPA[0] = fake_sdpa
    counters = (mg._ENGAGED[0], mg._BANDED_ENGAGED[0], mg._DEFERRED[0])
    yield calls
    mg._ORIG_SDPA[0] = prev
    mg._ENGAGED[0], mg._BANDED_ENGAGED[0], mg._DEFERRED[0] = counters
    mg._enabled.cache_clear()
    mg._force_banded.cache_clear()


def _qkv(S, dtype=torch.float32, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(1, H, S, D, generator=g).to(DEVICE, dtype)
    k = torch.randn(1, HKV, S, D, generator=g).to(DEVICE, dtype)
    v = torch.randn(1, HKV, S, D, generator=g).to(DEVICE, dtype)
    return q, k, v


def _route(module, S, mask=None, is_causal=None, q=None, k=None, v=None, **kw):
    if q is None:
        q, k, v = _qkv(S)
    return mg._sdpa_maybe_muse_glimmer_banded(
        module, q, k, v, mask, dropout=0.0, scaling=D ** -0.5, is_causal=is_causal, **kw)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_registered_in_temporary_patches():
    from unsloth_zoo.temporary_patches.common import TEMPORARY_PATCHES
    assert mg.patch_muse_glimmer_banded_sliding_attention in TEMPORARY_PATCHES


# ---------------------------------------------------------------------------
# The gate must defer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("S", [W, 2 * W, 3 * W - 1])
def test_defers_below_three_windows(_clean_router, S):
    """Blocking into (w, 2w) tiles is a measured loss below 2w and break-even at
    2w, so short-context finetunes must be left completely alone."""
    out, weights = _route(MuseGlimmerTextAttention(), S)
    assert out == "DEFERRED" and weights is None
    assert len(_clean_router) == 1


def test_defers_on_full_attention_layer(_clean_router):
    out, _ = _route(MuseGlimmerTextAttention(is_local_attention=False), S_ON)
    assert out == "DEFERRED"


def test_defers_on_vision_attention(_clean_router):
    out, _ = _route(MuseGlimmerVisionAttention(), S_ON)
    assert out == "DEFERRED"


def test_defers_on_foreign_module(_clean_router):
    class LlamaAttention:
        is_local_attention = True
        is_causal = True
        training = False
        sliding_window = W
        num_key_value_groups = H // HKV

    out, _ = _route(LlamaAttention(), S_ON)
    assert out == "DEFERRED"


def test_defers_on_decode_step(_clean_router):
    """Sq=1 against a long cache: the block kernel has no meaning there."""
    g = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(1, H, 1, D, generator=g).to(DEVICE)
    k = torch.randn(1, HKV, S_ON, D, generator=g).to(DEVICE)
    v = torch.randn(1, HKV, S_ON, D, generator=g).to(DEVICE)
    out, _ = _route(MuseGlimmerTextAttention(), S_ON, q=q, k=k, v=v)
    assert out == "DEFERRED"


def test_defers_on_padded_mask(_clean_router):
    mask = _band_mask(S_ON, W).clone()
    mask[..., :, S_ON // 2] = False                 # a padded key column
    out, _ = _route(MuseGlimmerTextAttention(), S_ON, mask=mask)
    assert out == "DEFERRED"


def test_defers_on_per_head_mask(_clean_router):
    mask = _band_mask(S_ON, W).expand(1, H, S_ON, S_ON).clone()
    out, _ = _route(MuseGlimmerTextAttention(), S_ON, mask=mask)
    assert out == "DEFERRED"


def test_defers_on_bidirectional_call_without_mask(_clean_router):
    out, _ = _route(MuseGlimmerTextAttention(), S_ON, mask=None, is_causal=False)
    assert out == "DEFERRED"


def test_defers_when_training_dropout_is_active(_clean_router):
    module = MuseGlimmerTextAttention()
    module.training = True
    q, k, v = _qkv(S_ON)
    out, _ = mg._sdpa_maybe_muse_glimmer_banded(
        module, q, k, v, None, dropout=0.1, scaling=D ** -0.5, is_causal=None)
    assert out == "DEFERRED"


def test_env_off_defers_everywhere(_clean_router, monkeypatch):
    monkeypatch.setenv("UNSLOTH_MUSE_GLIMMER_BANDED_SDPA", "0")
    mg._enabled.cache_clear()
    out, _ = _route(MuseGlimmerTextAttention(), S_ON)
    assert out == "DEFERRED"


# ---------------------------------------------------------------------------
# The gate must engage, and match the reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("with_mask", [False, True])
def test_engages_and_matches_reference(_clean_router, with_mask):
    S = S_ON
    q, k, v = _qkv(S)
    mask = _band_mask(S, W) if with_mask else None
    before = mg._BANDED_ENGAGED[0]
    out, weights = _route(MuseGlimmerTextAttention(), S, mask=mask, q=q, k=k, v=v)
    assert not _clean_router, "router deferred instead of engaging"
    assert weights is None
    assert mg._BANDED_ENGAGED[0] == before + 1
    ref = _reference_band(q, k, v, W, D ** -0.5, H // HKV)
    assert out.shape == ref.shape
    rel = (out - ref).norm() / ref.norm().clamp_min(1e-9)
    assert rel < 1e-5, f"forward rel {rel:.2e}"


def test_backward_matches_reference(_clean_router):
    S = S_ON
    q, k, v = _qkv(S)
    q_r, k_r, v_r = (t.clone().requires_grad_(True) for t in (q, k, v))
    q_b, k_b, v_b = (t.clone().requires_grad_(True) for t in (q, k, v))
    ref = _reference_band(q_r, k_r, v_r, W, D ** -0.5, H // HKV)
    out, _ = _route(MuseGlimmerTextAttention(), S, q=q_b, k=k_b, v=v_b)
    g = torch.randn_like(ref)
    (ref * g).sum().backward()
    (out * g).sum().backward()
    for name, a, b in (("dq", q_r.grad, q_b.grad),
                       ("dk", k_r.grad, k_b.grad),
                       ("dv", v_r.grad, v_b.grad)):
        rel = (a - b).norm() / a.norm().clamp_min(1e-9)
        assert rel < 1e-5, f"{name} rel {rel:.2e}"


def test_banded_engages_when_sliding_window_comes_from_kwargs(_clean_router):
    module = MuseGlimmerTextAttention()
    module.sliding_window = None
    before = mg._BANDED_ENGAGED[0]
    _route(module, S_ON, sliding_window=W)
    assert mg._BANDED_ENGAGED[0] == before + 1


# ---------------------------------------------------------------------------
# Installation is idempotent, and does not break gemma-4's re-entry guard
# ---------------------------------------------------------------------------

def _sdpa_registry():
    modeling_utils = pytest.importorskip("transformers.modeling_utils")
    return modeling_utils.ALL_ATTENTION_FUNCTIONS


def test_patch_is_idempotent():
    """unsloth applies TEMPORARY_PATCHES three times per process (init,
    pre_compile, post_compile), so a second install must not stack a wrapper."""
    registry = _sdpa_registry()
    original = registry["sdpa"]
    prev_box = mg._ORIG_SDPA[0]

    # Importing unsloth already ran TEMPORARY_PATCHES, so start from a known
    # plain entry rather than from whatever is installed in this process.
    def plain_sdpa(module, q, k, v, mask, **kw):
        return "PLAIN", None

    try:
        registry["sdpa"] = plain_sdpa
        mg.patch_muse_glimmer_banded_sliding_attention()
        installed = registry["sdpa"]
        assert installed is not plain_sdpa
        mg.patch_muse_glimmer_banded_sliding_attention()
        assert registry["sdpa"] is installed
        assert mg._ORIG_SDPA[0] is plain_sdpa
    finally:
        registry["sdpa"] = original
        mg._ORIG_SDPA[0] = prev_box


def test_patch_preserves_wrapped_router_sentinels():
    """gemma-4's router re-entry guard reads its sentinel off the installed sdpa
    entry only. Once this router wraps it, that sentinel must still be visible,
    otherwise gemma-4 would wrap itself again on the next pass."""
    registry = _sdpa_registry()
    original = registry["sdpa"]
    prev_box = mg._ORIG_SDPA[0]

    def gemma4_like(module, q, k, v, mask, **kw):
        return "GEMMA4", None

    gemma4_like._unsloth_gemma4_flash = True
    try:
        registry["sdpa"] = gemma4_like
        mg.patch_muse_glimmer_banded_sliding_attention()
        installed = registry["sdpa"]
        assert installed is not gemma4_like
        assert getattr(installed, "_unsloth_gemma4_flash", False) is True
        assert getattr(installed, "_unsloth_muse_glimmer_banded", False) is True
        mg.unpatch_muse_glimmer_banded_sliding_attention()
        assert registry["sdpa"] is gemma4_like
    finally:
        registry["sdpa"] = original
        mg._ORIG_SDPA[0] = prev_box
        # The sentinel was forwarded onto the module-level wrapper; drop it again
        # so it cannot leak into any later test.
        vars(mg._sdpa_maybe_muse_glimmer_banded).pop("_unsloth_gemma4_flash", None)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
