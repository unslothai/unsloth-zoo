"""Routing and numerical tests for the opt-in Gemma-4 tiled D=512 global attention."""

from __future__ import annotations

import inspect
import math
import sys
import types

import pytest
import torch
import torch.nn.functional as F

from unsloth_zoo.temporary_patches import gemma4_tiled_global_attention as gg

KERNEL_MODULE = "unsloth_zoo.temporary_patches._triton_causal_attention_d512"


class Gemma4GlobalFake:
    is_sliding = False
    is_causal = True
    scaling = 1.0
    training = True


class Gemma4SlidingFake(Gemma4GlobalFake):
    is_sliding = True


class OtherAttentionFake:
    training = True


@pytest.fixture
def routed(monkeypatch):
    """Enable the router and pretend every tensor lives on a validated device."""
    monkeypatch.setenv("UNSLOTH_GEMMA4_TILED_GLOBAL", "1")
    monkeypatch.setenv("UNSLOTH_GEMMA4_TILED_GLOBAL_MIN_SEQ", "4")
    monkeypatch.setattr(gg, "_triton_supported", lambda: True)
    monkeypatch.setattr(gg, "_device_arch", lambda x: "gfx906")
    return monkeypatch


def _fake_kernel(monkeypatch, fn):
    monkeypatch.setitem(sys.modules, KERNEL_MODULE, types.SimpleNamespace(causal_attention_d512 = fn))


def _forbid_kernel(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("cache-bearing call reached the tiled kernel")
    _fake_kernel(monkeypatch, forbidden)


def _recording_fallback():
    calls = []

    def fallback(*args, **kwargs):
        calls.append(True)
        return "fallback", None
    return fallback, calls


def _qkv(S = 4, dtype = torch.float16, Hq = 32, Hkv = 4, D = 512):
    q = torch.zeros(1, Hq, S, D, dtype = dtype, requires_grad = True)
    k = torch.zeros(1, Hkv, S, D, dtype = dtype, requires_grad = True)
    v = torch.zeros_like(k, requires_grad = True)
    return q, k, v


def _causal(S):
    idx = torch.arange(S)
    return idx[None, :] <= idx[:, None]


def _float_causal(S, dtype = torch.float32):
    return torch.where(_causal(S), 0.0, float("-inf")).to(dtype)[None, None]


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def test_env_switches_are_read_at_call_time(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GEMMA4_TILED_GLOBAL", "0")
    monkeypatch.setenv("UNSLOTH_GEMMA4_TILED_GLOBAL_MIN_SEQ", "1024")
    assert gg._enabled() is False and gg._min_seq_len() == 1024
    monkeypatch.setenv("UNSLOTH_GEMMA4_TILED_GLOBAL", "1")
    monkeypatch.setenv("UNSLOTH_GEMMA4_TILED_GLOBAL_MIN_SEQ", "4096")
    assert gg._enabled() is True and gg._min_seq_len() == 4096


def test_router_is_opt_in(routed):
    routed.setenv("UNSLOTH_GEMMA4_TILED_GLOBAL", "0")
    assert gg._eligible(Gemma4GlobalFake(), *_qkv(), None, 0.0, True) is False


def test_router_accepts_only_the_narrow_training_shape(routed):
    q, k, v = _qkv()
    module = Gemma4GlobalFake()
    assert gg._eligible(module, q, k, v, None, 0.0, True) is True
    assert gg._eligible(module, q.float(), k.float(), v.float(), None, 0.0, True) is True
    assert gg._eligible(Gemma4SlidingFake(), q, k, v, None, 0.0, True) is False
    assert gg._eligible(OtherAttentionFake(), q, k, v, None, 0.0, True) is False
    assert gg._eligible(module, q, k[:, :, :-1], v[:, :, :-1], None, 0.0, True) is False
    assert gg._eligible(module, q, k, v, None, 0.1, True) is False
    assert gg._eligible(module, q, k, v, None, 0.0, False) is False
    assert gg._eligible(module, q, k, v, None, 0.0, True, has_cache = True) is False
    assert gg._eligible(module, *_qkv(D = 256), None, 0.0, True) is False
    assert gg._eligible(module, *_qkv(Hq = 32, Hkv = 3), None, 0.0, True) is False
    assert gg._eligible(module, *_qkv(S = 3), None, 0.0, True) is False

    module.training = False
    assert gg._eligible(module, q, k, v, None, 0.0, True) is False
    module.training = True
    with torch.no_grad():
        # Reentrant checkpoint packs run under no_grad in training mode and
        # must pick the same backend as the recompute.
        assert gg._eligible(module, q, k, v, None, 0.0, True) is True


@pytest.mark.parametrize("Hq,Hkv", [(32, 4), (16, 2), (8, 1), (8, 8)])
def test_router_accepts_any_gqa_ratio(routed, Hq, Hkv):
    assert gg._eligible(Gemma4GlobalFake(), *_qkv(Hq = Hq, Hkv = Hkv), None, 0.0, True) is True


@pytest.mark.parametrize("arch", [None, "gfx90a", "gfx942", "gfx1100"])
def test_router_declines_devices_outside_the_validated_set(routed, arch):
    routed.setattr(gg, "_device_arch", lambda x: arch)
    assert gg._eligible(Gemma4GlobalFake(), *_qkv(), None, 0.0, True) is False


def test_router_declines_cpu_tensors_without_arch_override(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GEMMA4_TILED_GLOBAL", "1")
    monkeypatch.setenv("UNSLOTH_GEMMA4_TILED_GLOBAL_MIN_SEQ", "4")
    monkeypatch.setattr(gg, "_triton_supported", lambda: True)
    assert gg._device_arch(torch.zeros(1)) is None
    assert gg._eligible(Gemma4GlobalFake(), *_qkv(), None, 0.0, True) is False


def test_triton_version_gate_is_part_of_eligibility(routed):
    routed.setattr(gg, "_triton_supported", lambda: False)
    assert gg._eligible(Gemma4GlobalFake(), *_qkv(), None, 0.0, True) is False


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------

def test_exact_causal_mask_is_accepted_and_padding_is_not(routed):
    q, k, v = _qkv(S = 8)
    module = Gemma4GlobalFake()
    mask = _float_causal(8)
    # An exact causal mask suffices even when the caller says is_causal=False.
    assert gg._eligible(module, q, k, v, mask, 0.0, False) is True
    padded = mask.clone()
    padded[..., :, 3] = float("-inf")
    assert gg._eligible(module, q, k, v, padded, 0.0, False) is False


def test_mask_probe_ignores_sliding_verdict_and_sees_mutation(routed):
    from unsloth_zoo.temporary_patches.gemma4_banded_attention import _mask_is_plain_band

    q, k, v = _qkv(S = 4)
    module = Gemma4GlobalFake()
    idx = torch.arange(4)
    sliding = ((idx[None, :] <= idx[:, None]) & (idx[None, :] > idx[:, None] - 2))[None, None]
    # The sliding helper stashes a True verdict for w=2 on the tensor; the
    # global probe must not read it as proof of full causal.
    assert _mask_is_plain_band(sliding, 4, 2) is True
    assert gg._eligible(module, q, k, v, sliding, 0.0, False) is False

    causal = _causal(4)[None, None].clone()
    assert gg._eligible(module, q, k, v, causal, 0.0, False) is True
    causal[..., 3, 0] = False
    assert gg._eligible(module, q, k, v, causal, 0.0, False) is False


def test_mask_probe_rejects_bias_grad_and_bad_metadata(routed):
    q, k, v = _qkv(S = 4)
    module = Gemma4GlobalFake()
    exact = _float_causal(4)
    assert gg._eligible(module, q, k, v, exact, 0.0, False) is True

    rejected = [
        torch.where(_causal(4), 0.0, -10000.0)[None, None],           # finite bias
        exact.clone().requires_grad_(True),                            # mask grad
        _causal(4).to(torch.int32)[None, None],                        # integer
        exact.expand(2, -1, -1, -1).clone(),                           # wrong batch
        exact.expand(1, 2, -1, -1).clone(),                            # per head
        torch.empty((1, 1, 4, 4), dtype = torch.bool, device = "meta"),  # device
    ]
    for mask in rejected:
        assert gg._eligible(module, q, k, v, mask, 0.0, False) is False


def test_mask_probe_follows_sdpa_dtype_contract(routed):
    module = Gemma4GlobalFake()
    q16, k16, v16 = _qkv(dtype = torch.float16)
    for mask in (_causal(4)[None, None], _float_causal(4, torch.float16), _float_causal(4, torch.float32)):
        assert gg._eligible(module, q16, k16, v16, mask, 0.0, False) is True

    q32, k32, v32 = _qkv(dtype = torch.float32)
    for bad_dtype in (torch.float16, torch.bfloat16, torch.float64):
        assert gg._eligible(module, q32, k32, v32, _float_causal(4, bad_dtype), 0.0, False) is False

    # An SDPA-invalid mask goes to the original backend unchanged.
    fallback, calls = _recording_fallback()
    out = gg._sdpa_maybe_tiled_global(
        module, q32, k32, v32, _float_causal(4, torch.float64),
        dropout = 0.0, scaling = 1.0, is_causal = False, _fallback = fallback,
    )
    assert out == ("fallback", None) and calls == [True]


# ---------------------------------------------------------------------------
# Cache scoping
# ---------------------------------------------------------------------------

def _registry_forward(q, k, v, fallback):
    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values = None):
        return gg._sdpa_maybe_tiled_global(
            self, q, k, v, None,
            dropout = 0.0, scaling = 1.0, is_causal = True, _fallback = fallback,
        )
    return forward


def test_cache_scope_keeps_equal_length_prefill_on_fallback(routed):
    q, k, v = _qkv()
    module = Gemma4GlobalFake()
    _fake_kernel(routed, lambda q, k, v, scale: q)
    fallback, calls = _recording_fallback()
    wrapped = gg._make_gemma4_cache_scope_wrapper(_registry_forward(q, k, v, fallback))

    # After an empty-cache update Q/K/V lengths are equal; only the scope can
    # tell this call apart from a cache-free one.
    assert wrapped(module, None, None, None, object()) == ("fallback", None)
    assert wrapped(module, None, None, None, past_key_values = object()) == ("fallback", None)
    assert calls == [True, True]
    assert gg._CACHE_PRESENT.get() is False

    out, weights = wrapped(module, None, None, None, None)
    assert out.shape == (1, 4, 32, 512) and weights is None
    assert calls == [True, True]

    # Registry callers that pass the cache directly are rejected too.
    assert gg._sdpa_maybe_tiled_global(
        module, q, k, v, None,
        dropout = 0.0, scaling = 1.0, is_causal = True,
        past_key_values = object(), _fallback = fallback,
    ) == ("fallback", None)
    assert calls == [True, True, True]


def test_cache_scope_sees_positional_cache_through_carrier_wrapper(routed):
    from unsloth_zoo.temporary_patches.gemma4 import _make_gemma4_attention_carrier_forward

    q, k, v = _qkv()
    _forbid_kernel(routed)
    fallback, calls = _recording_fallback()
    carrier = _make_gemma4_attention_carrier_forward(_registry_forward(q, k, v, fallback))
    # The carrier wrapper must keep the HF signature for bind_partial to work.
    assert "past_key_values" in inspect.signature(carrier).parameters
    wrapped = gg._make_gemma4_cache_scope_wrapper(carrier)

    assert wrapped(Gemma4GlobalFake(), None, None, None, object()) == ("fallback", None)
    assert calls == [True]
    assert gg._CACHE_PRESENT.get() is False


def test_cache_scope_is_set_even_when_disabled_at_entry(routed):
    q, k, v = _qkv()
    routed.setenv("UNSLOTH_GEMMA4_TILED_GLOBAL", "0")
    _forbid_kernel(routed)
    fallback, calls = _recording_fallback()
    inner = _registry_forward(q, k, v, fallback)

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values = None):
        routed.setenv("UNSLOTH_GEMMA4_TILED_GLOBAL", "1")
        return inner(self, hidden_states, position_embeddings, attention_mask, past_key_values)

    wrapped = gg._make_gemma4_cache_scope_wrapper(forward)
    assert wrapped(Gemma4GlobalFake(), None, None, None, object()) == ("fallback", None)
    assert calls == [True]


def test_real_gemma4_attention_empty_dynamic_cache_prefill_uses_fallback(routed):
    pytest.importorskip("transformers.models.gemma4")
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.gemma4 import modeling_gemma4 as modeling
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from unsloth_zoo.temporary_patches.gemma4 import _make_gemma4_attention_carrier_forward

    routed.setenv("UNSLOTH_GEMMA4_TILED_GLOBAL_MIN_SEQ", "1")
    _forbid_kernel(routed)
    seen = []

    def fallback(module, query, key, value, attention_mask, **kwargs):
        seen.append((query.shape[2], key.shape[2], value.shape[2]))
        return query.transpose(1, 2).contiguous(), None

    routed.setitem(ALL_ATTENTION_FUNCTIONS, "sdpa", gg._make_tiled_global_wrapper(fallback))

    # Expose the older (hidden, pos, mask, past_key_values) positional API over
    # whatever the installed Gemma4TextAttention takes, then stack the carrier
    # and cache-scope wrappers the way the patches do.
    real_forward = modeling.Gemma4TextAttention.forward
    needs_shared = "shared_kv_states" in inspect.signature(real_forward).parameters

    def positional_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values = None, **kwargs):
        if needs_shared:
            kwargs.setdefault("shared_kv_states", {})
        return real_forward(
            self, hidden_states, position_embeddings, attention_mask,
            past_key_values = past_key_values, **kwargs,
        )

    carrier = _make_gemma4_attention_carrier_forward(positional_forward)
    routed.setattr(modeling.Gemma4TextAttention, "forward", gg._make_gemma4_cache_scope_wrapper(carrier))

    config = Gemma4TextConfig(
        hidden_size = 64, intermediate_size = 128, num_hidden_layers = 1,
        num_attention_heads = 32, num_key_value_heads = 4, head_dim = 256,
        num_global_key_value_heads = 4, global_head_dim = 512,
        layer_types = ["full_attention"], attention_dropout = 0.0, use_cache = True,
        rope_parameters = {"rope_type": "default", "rope_theta": 10000.0},
    )
    config.rope_parameters = {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
        "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1_000_000.0},
    }
    config._attn_implementation = "sdpa"
    layer = modeling.Gemma4TextAttention(config, 0).train()
    rotary = modeling.Gemma4TextRotaryEmbedding(config, device = "cpu", layer_type = "full_attention")
    S = 2
    hidden = torch.randn(1, S, config.hidden_size)
    with torch.no_grad():
        position_embeddings = rotary(hidden, torch.arange(S)[None], "full_attention")

    cache = DynamicCache(config = config)
    output, weights = layer(hidden, position_embeddings, None, cache)   # cache positional

    assert output.shape == hidden.shape and weights is None
    assert cache.get_seq_length(0) == S
    assert seen == [(S, S, S)]
    assert gg._CACHE_PRESENT.get() is False


# ---------------------------------------------------------------------------
# Wrapper lifecycle and call contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order", [("sliding", "global"), ("global", "sliding")])
def test_sdpa_wrappers_are_order_independent_and_reload_safe(order):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from unsloth_zoo.temporary_patches import gemma4_flash_sliding as gf

    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    saved = (gg._ORIG_SDPA[0], gf._ORIG_SDPA[0])
    try:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = lambda *args, **kwargs: ("base", "weights")
        patches = {"sliding": gf.patch_gemma4_flash_sliding, "global": gg.patch_gemma4_tiled_global_attention}
        for name in order:
            patches[name]()
        # TEMPORARY_PATCHES runs every patch again in later phases.
        gf.patch_gemma4_flash_sliding()
        gg.patch_gemma4_tiled_global_attention()
        stacked = ALL_ATTENTION_FUNCTIONS["sdpa"]
        assert getattr(stacked, "_unsloth_gemma4_flash", False)
        assert getattr(stacked, "_unsloth_gemma4_tiled_global", False)

        q, k, v = _qkv()
        assert stacked(OtherAttentionFake(), q, k, v, None) == ("base", "weights")
        # A module reload resets the boxes; installed wrappers must not use them.
        gg._ORIG_SDPA[0] = None
        gf._ORIG_SDPA[0] = None
        assert stacked(OtherAttentionFake(), q, k, v, None) == ("base", "weights")
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = original
        gg._ORIG_SDPA[0], gf._ORIG_SDPA[0] = saved


def test_eval_mode_delegates_to_wrapped_sdpa(routed):
    module = Gemma4GlobalFake()
    module.training = False
    fallback, calls = _recording_fallback()
    out = gg._sdpa_maybe_tiled_global(
        module, *_qkv(), None, dropout = 0.0, scaling = 1.0, is_causal = True, _fallback = fallback,
    )
    assert out == ("fallback", None) and calls == [True]


def test_reentrant_checkpoint_pack_and_recompute_choose_same_backend(routed):
    from unsloth_zoo.gradient_checkpointing import Unsloth_Gradient_Checkpointer

    module = Gemma4GlobalFake()
    seen = []
    x = torch.randn(1, 32, 4, 512, requires_grad = True)

    def fn(hidden):
        seen.append((torch.is_grad_enabled(), gg._eligible(module, hidden, hidden[:, :4], hidden[:, :4], None, 0.0, True)))
        return hidden.square().sum()

    Unsloth_Gradient_Checkpointer.apply(fn, x).backward()
    assert seen == [(False, True), (True, True)]


def test_scale_none_means_sdpa_default(routed):
    scales = []
    _fake_kernel(routed, lambda q, k, v, scale: scales.append(scale) or q)
    q, k, v = _qkv()
    out = gg.maybe_gemma4_tiled_global_attention(
        Gemma4GlobalFake(), q, k, v, None, dropout = 0.0, scaling = None, is_causal = True,
    )
    assert out is q
    assert scales == [pytest.approx(1.0 / math.sqrt(512.0))]


def test_kernel_error_is_not_retried_through_sdpa(routed):
    def boom(*args, **kwargs):
        raise RuntimeError("kernel launch failed")
    _fake_kernel(routed, boom)
    fallback, calls = _recording_fallback()
    with pytest.raises(RuntimeError, match = "kernel launch failed"):
        gg._sdpa_maybe_tiled_global(
            Gemma4GlobalFake(), *_qkv(), None,
            dropout = 0.0, scaling = 1.0, is_causal = True, _fallback = fallback,
        )
    assert calls == []


def test_kernel_rejects_malformed_inputs_before_launch():
    pytest.importorskip("triton")
    from unsloth_zoo.temporary_patches._triton_causal_attention_d512 import causal_attention_d512

    def t(*shape, dtype = torch.float16):
        return torch.zeros(*shape, dtype = dtype)

    q, k, v = t(1, 32, 4, 512), t(1, 4, 4, 512), t(1, 4, 4, 512)
    malformed = [
        ((q, k, t(1, 3, 4, 512)), "same \\(B, Hkv, S\\)"),
        ((q, t(1, 4, 4, 256), v), "head_dim 512"),
        ((q, k, t(1, 4, 4, 256)), "head_dim 512"),
        ((q, t(2, 4, 4, 512), t(2, 4, 4, 512)), "matching batch"),
        ((q, t(1, 4, 5, 512), t(1, 4, 5, 512)), "matching batch and sequence"),
        ((q, t(1, 3, 4, 512), t(1, 3, 4, 512)), "Hq % Hkv"),
        ((q, k.float(), v), "one dtype"),
    ]
    for args, message in malformed:
        with pytest.raises(ValueError, match = message):
            causal_attention_d512(*args, 1.0)


def test_force_float32_disabled_leaves_attention_forward_unchanged(monkeypatch):
    pytest.importorskip("transformers.models.gemma4")
    from transformers.models.gemma4 import modeling_gemma4 as modeling
    from unsloth_zoo.temporary_patches.gemma4_float32 import patch_Gemma4TextAttention

    original_forward = modeling.Gemma4TextAttention.forward
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "0")
    patch_Gemma4TextAttention()
    assert modeling.Gemma4TextAttention.forward is original_forward


# ---------------------------------------------------------------------------
# Numerics on a validated device
# ---------------------------------------------------------------------------

def _on_validated_gpu():
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        return False
    try:
        arch = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")).split(":", 1)[0]
    except Exception:
        return False
    return arch in gg._VALIDATED_ARCHS


needs_validated_gpu = pytest.mark.skipif(not _on_validated_gpu(), reason = "needs a GPU in _VALIDATED_ARCHS")

# Per dtype: whole-tensor relative tolerance, then per-row mixed L2 (rtol, atol).
TOL = {
    torch.float16: (3e-3, 6e-3, 1.0),
    torch.bfloat16: (3e-2, 6e-2, 6.0),
    torch.float32: (5e-4, 1e-3, 2e-2),
}


def _assert_rows_close(actual, expected, *, rtol, atol):
    """Every row within atol + rtol * |row|; whole-tensor norms hide a bad row."""
    row_diff = torch.linalg.vector_norm(actual.float() - expected.float(), dim = -1)
    limit = atol + rtol * torch.linalg.vector_norm(expected.float(), dim = -1)
    if bool((row_diff > limit).any().item()):
        worst = (row_diff / limit.clamp_min(1e-12)).amax().item()
        raise AssertionError(f"localized row error exceeds tolerance (worst normalized excess={worst:.4g})")


@pytest.mark.parametrize("dtype", list(TOL))
def test_row_guard_catches_one_bad_row_that_head_norm_misses(dtype):
    tol, rtol, atol = TOL[dtype]
    corruption = {torch.float16: 0.10, torch.bfloat16: 1.00, torch.float32: 0.01}[dtype]
    reference = torch.full((1, 1, 127, 512), 10.0)
    actual = reference.clone()
    actual[..., 17, :] *= 1.0 + corruption
    head_rel = (torch.linalg.vector_norm(actual - reference, dim = (-2, -1))
                / torch.linalg.vector_norm(reference, dim = (-2, -1)))
    assert head_rel.amax().item() < tol * 4
    with pytest.raises(AssertionError, match = "localized row error"):
        _assert_rows_close(actual, reference, rtol = rtol, atol = atol)


def _compare_with_sdpa(*, dtype, B = 1, Hq = 32, Hkv = 4, S = 127, scale = 1.0, seed = 3407,
                       noncontiguous = False, rows = True):
    from unsloth_zoo.temporary_patches._triton_causal_attention_d512 import causal_attention_d512

    torch.manual_seed(seed)

    def make(H):
        if noncontiguous:
            return torch.randn(B, H, S, 1024, device = "cuda", dtype = dtype)[..., ::2].requires_grad_()
        return torch.randn(B, H, S, 512, device = "cuda", dtype = dtype, requires_grad = True)

    q, k, v = make(Hq), make(Hkv), make(Hkv)
    qr, kr, vr = (x.detach().clone().requires_grad_(True) for x in (q, k, v))
    out = causal_attention_d512(q, k, v, scale)
    ref = F.scaled_dot_product_attention(
        qr, kr.repeat_interleave(Hq // Hkv, dim = 1), vr.repeat_interleave(Hq // Hkv, dim = 1),
        is_causal = True, scale = scale,
    )
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad)
    torch.cuda.synchronize()

    tol, rtol, atol = TOL[dtype]
    for actual, expected in ((out, ref), (q.grad, qr.grad), (k.grad, kr.grad), (v.grad, vr.grad)):
        assert torch.isfinite(actual).all()
        rel = ((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-9)).item()
        assert rel < tol
        if rows:
            _assert_rows_close(actual, expected, rtol = rtol, atol = atol)


@needs_validated_gpu
@pytest.mark.parametrize("dtype", list(TOL))
@pytest.mark.parametrize("S", [127, 1025])
def test_kernel_matches_sdpa(dtype, S):
    _compare_with_sdpa(dtype = dtype, S = S, seed = 3407 + S)


@needs_validated_gpu
@pytest.mark.parametrize("S", [15, 16, 17, 31, 32, 33])
def test_kernel_matches_sdpa_at_tile_boundaries(S):
    _compare_with_sdpa(dtype = torch.float16, S = S, scale = 1.0 / math.sqrt(512.0), seed = 3407 + S, rows = False)


@needs_validated_gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_kernel_matches_sdpa_batch_two(dtype):
    _compare_with_sdpa(dtype = dtype, B = 2, S = 64, seed = 777)


@needs_validated_gpu
def test_kernel_matches_sdpa_noncontiguous():
    _compare_with_sdpa(dtype = torch.float16, S = 33, seed = 8128, noncontiguous = True, rows = False)


@needs_validated_gpu
@pytest.mark.parametrize("dtype", list(TOL))
@pytest.mark.parametrize("Hq,Hkv", [(16, 2), (8, 1), (8, 8)])
def test_kernel_matches_sdpa_other_gqa_ratios(dtype, Hq, Hkv):
    _compare_with_sdpa(dtype = dtype, Hq = Hq, Hkv = Hkv, S = 97, seed = 42 + Hq + Hkv)


@needs_validated_gpu
def test_kernel_rejects_higher_order_graph_construction():
    from unsloth_zoo.temporary_patches._triton_causal_attention_d512 import causal_attention_d512

    torch.manual_seed(3407)
    q = torch.randn(1, 32, 8, 512, device = "cuda", requires_grad = True)
    k = torch.randn(1, 4, 8, 512, device = "cuda", requires_grad = True)
    v = torch.randn_like(k, requires_grad = True)
    loss = causal_attention_d512(q, k, v, 1.0 / math.sqrt(512.0)).sum() + q.square().sum()
    with pytest.raises(RuntimeError, match = "first-order gradients only"):
        torch.autograd.grad(loss, (q, k, v), create_graph = True)


@needs_validated_gpu
def test_force_float32_real_gemma4_attention_checkpoint_matches_sdpa(monkeypatch):
    """Real 31B global-layer shape through FORCE_FLOAT32 and reentrant checkpointing."""
    from transformers.models.gemma4 import modeling_gemma4 as modeling
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from unsloth_zoo.gradient_checkpointing import Unsloth_Gradient_Checkpointer
    from unsloth_zoo.temporary_patches.gemma4_float32 import patch_Gemma4TextAttention

    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    monkeypatch.setenv("UNSLOTH_GEMMA4_TILED_GLOBAL_MIN_SEQ", "1")
    original_forward = modeling.Gemma4TextAttention.forward
    try:
        patch_Gemma4TextAttention()
        config = Gemma4TextConfig(
            hidden_size = 5376, intermediate_size = 21504, num_hidden_layers = 1,
            num_attention_heads = 32, num_key_value_heads = 4, head_dim = 256,
            num_global_key_value_heads = 4, global_head_dim = 512, attention_k_eq_v = True,
            layer_types = ["full_attention"], attention_dropout = 0.0, use_cache = False,
            rope_parameters = {"rope_type": "default", "rope_theta": 10000.0},
        )
        config.rope_parameters = {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
            "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1_000_000.0},
        }
        config._attn_implementation = "sdpa"
        layer = modeling.Gemma4TextAttention(config, 0).to(device = "cuda", dtype = torch.float16).train()
        layer.requires_grad_(False)
        rotary = modeling.Gemma4TextRotaryEmbedding(config, device = "cuda", layer_type = "full_attention")

        S = 32
        torch.manual_seed(3407)
        base = torch.randn(1, S, 5376, device = "cuda", dtype = torch.float16)
        with torch.no_grad():
            position_embeddings = rotary(base, torch.arange(S, device = "cuda")[None], "full_attention")
        upstream_grad = torch.randn_like(base)

        def run(enabled):
            monkeypatch.setenv("UNSLOTH_GEMMA4_TILED_GLOBAL", enabled)
            hidden = base.detach().clone().requires_grad_(True)
            before = gg.gemma4_tiled_global_stats()["engaged"]
            output = Unsloth_Gradient_Checkpointer.apply(
                lambda h: layer(h, position_embeddings = position_embeddings, attention_mask = None)[0],
                hidden,
            )
            output.backward(upstream_grad)
            torch.cuda.synchronize()
            return output.detach(), hidden.grad.detach(), gg.gemma4_tiled_global_stats()["engaged"] - before

        ref_out, ref_grad, ref_engaged = run("0")
        out, grad, engaged = run("1")

        def rel(a, b):
            return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-9)).item()

        assert ref_engaged == 0
        # Once in the no_grad pack forward, once in the backward recompute.
        assert engaged == 2
        assert rel(out, ref_out) < 5e-4 and rel(grad, ref_grad) < 5e-4
        assert torch.isfinite(out).all() and torch.isfinite(grad).all()
    finally:
        modeling.Gemma4TextAttention.forward = original_forward
