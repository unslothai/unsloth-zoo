"""Routing and numerical tests for the opt-in gfx906 Gemma-4 31B global kernel."""

from __future__ import annotations

import math
import sys
import types

import pytest
import torch
import torch.nn.functional as F

from unsloth_zoo.temporary_patches import gemma4_gfx906_global_attention as gg


class Gemma4GlobalFake:
    is_sliding = False
    is_causal = True
    scaling = 1.0
    training = True


class Gemma4SlidingFake(Gemma4GlobalFake):
    is_sliding = True


class OtherAttentionFake:
    training = True


def _reset_env_cache(monkeypatch, *, enabled="1", min_seq="4"):
    monkeypatch.setenv("UNSLOTH_GEMMA4_GFX906_GLOBAL", enabled)
    monkeypatch.setenv("UNSLOTH_GEMMA4_GFX906_GLOBAL_MIN_SEQ", min_seq)
    monkeypatch.setattr(gg, "_triton_supported", lambda: True)


def test_env_switches_are_not_stale(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GEMMA4_GFX906_GLOBAL", "0")
    monkeypatch.setenv("UNSLOTH_GEMMA4_GFX906_GLOBAL_MIN_SEQ", "1024")
    assert gg._enabled() is False
    assert gg._min_seq_len() == 1024
    monkeypatch.setenv("UNSLOTH_GEMMA4_GFX906_GLOBAL", "1")
    monkeypatch.setenv("UNSLOTH_GEMMA4_GFX906_GLOBAL_MIN_SEQ", "4096")
    assert gg._enabled() is True
    assert gg._min_seq_len() == 4096


def _cpu_qkv(S=4, dtype=torch.float16):
    q = torch.zeros(1, 32, S, 512, dtype=dtype, requires_grad=True)
    k = torch.zeros(1, 4, S, 512, dtype=dtype, requires_grad=True)
    v = torch.zeros_like(k, requires_grad=True)
    return q, k, v


def _assert_rowwise_mixed_close(actual, expected, *, rtol, atol):
    """Require every row to satisfy a mixed absolute/relative L2 tolerance."""
    row_diff = torch.linalg.vector_norm(actual.float() - expected.float(), dim=-1)
    row_ref = torch.linalg.vector_norm(expected.float(), dim=-1)
    limit = atol + rtol * row_ref
    violations = row_diff > limit
    if bool(violations.any().item()):
        worst = (row_diff / limit.clamp_min(1e-12)).amax().item()
        raise AssertionError(
            f"localized row error exceeds tolerance (worst normalized excess={worst:.4g})"
        )


def test_router_is_opt_in(monkeypatch):
    _reset_env_cache(monkeypatch, enabled="0")
    monkeypatch.setattr(gg, "_is_gfx906_tensor", lambda x: True)
    q, k, v = _cpu_qkv()
    assert gg._eligible(Gemma4GlobalFake(), q, k, v, None, 0.0, True) is False


def test_router_accepts_only_narrow_training_shape(monkeypatch):
    _reset_env_cache(monkeypatch)
    monkeypatch.setattr(gg, "_is_gfx906_tensor", lambda x: True)
    q, k, v = _cpu_qkv()
    module = Gemma4GlobalFake()

    assert gg._eligible(module, q, k, v, None, 0.0, True) is True
    assert gg._eligible(Gemma4SlidingFake(), q, k, v, None, 0.0, True) is False
    assert gg._eligible(module, q.float(), k.float(), v.float(), None, 0.0, True) is True
    assert gg._eligible(module, q, k[:, :, :-1], v[:, :, :-1], None, 0.0, True) is False
    assert gg._eligible(module, q, k, v, None, 0.1, True) is False
    assert gg._eligible(module, q, k, v, None, 0.0, False) is False
    assert gg._eligible(module, q, k, v, None, 0.0, True, has_cache=True) is False
    module.training = False
    assert gg._eligible(module, q, k, v, None, 0.0, True) is False
    module.training = True
    with torch.no_grad():
        # Reentrant checkpoint packs execute under no_grad; training mode must
        # still select the same backend that recompute will use.
        assert gg._eligible(module, q, k, v, None, 0.0, True) is True


def test_router_accepts_exact_causal_mask_rejects_padding(monkeypatch):
    _reset_env_cache(monkeypatch)
    monkeypatch.setattr(gg, "_is_gfx906_tensor", lambda x: True)
    q, k, v = _cpu_qkv(S=8)
    module = Gemma4GlobalFake()
    idx = torch.arange(8)
    causal = idx[None, :] <= idx[:, None]
    mask = torch.where(causal, 0.0, float("-inf"))[None, None]

    # Explicit exact causal mask is sufficient even when is_causal=False.
    assert gg._eligible(module, q, k, v, mask, 0.0, False) is True
    padded = mask.clone()
    padded[..., :, 3] = float("-inf")
    assert gg._eligible(module, q, k, v, padded, 0.0, False) is False


def test_global_mask_verifier_ignores_sliding_cache_and_mutation(monkeypatch):
    from unsloth_zoo.temporary_patches.gemma4_banded_attention import _mask_is_plain_band

    _reset_env_cache(monkeypatch)
    monkeypatch.setattr(gg, "_is_gfx906_tensor", lambda x: True)
    q, k, v = _cpu_qkv(S=4)
    module = Gemma4GlobalFake()
    idx = torch.arange(4)
    sliding = ((idx[None, :] <= idx[:, None]) & (idx[None, :] > idx[:, None] - 2))[None, None]

    # Seed the shared sliding helper's tensor attribute with a True verdict for
    # w=2. The global verifier must not reuse it as proof of full causal w=S.
    assert _mask_is_plain_band(sliding, 4, 2) is True
    assert gg._eligible(module, q, k, v, sliding, 0.0, False) is False

    causal = (idx[None, :] <= idx[:, None])[None, None].clone()
    assert gg._eligible(module, q, k, v, causal, 0.0, False) is True
    causal[..., 3, 0] = False
    assert gg._eligible(module, q, k, v, causal, 0.0, False) is False


def test_global_mask_verifier_rejects_finite_bias_grad_and_bad_metadata(monkeypatch):
    _reset_env_cache(monkeypatch)
    monkeypatch.setattr(gg, "_is_gfx906_tensor", lambda x: True)
    q, k, v = _cpu_qkv(S=4)
    module = Gemma4GlobalFake()
    idx = torch.arange(4)
    causal = idx[None, :] <= idx[:, None]

    exact = torch.where(causal, 0.0, float("-inf"))[None, None]
    assert gg._eligible(module, q, k, v, exact, 0.0, False) is True

    finite = torch.where(causal, 0.0, -10000.0)[None, None]
    assert gg._eligible(module, q, k, v, finite, 0.0, False) is False

    grad_mask = exact.clone().requires_grad_(True)
    assert gg._eligible(module, q, k, v, grad_mask, 0.0, False) is False

    integer_mask = causal.to(torch.int32)[None, None]
    assert gg._eligible(module, q, k, v, integer_mask, 0.0, False) is False

    wrong_batch = exact.expand(2, -1, -1, -1).clone()
    assert gg._eligible(module, q, k, v, wrong_batch, 0.0, False) is False

    per_head = exact.expand(1, 2, -1, -1).clone()
    assert gg._eligible(module, q, k, v, per_head, 0.0, False) is False

    other_device = torch.empty((1, 1, 4, 4), dtype=torch.bool, device="meta")
    assert gg._eligible(module, q, k, v, other_device, 0.0, False) is False


def test_global_mask_verifier_matches_sdpa_dtype_contract(monkeypatch):
    _reset_env_cache(monkeypatch)
    monkeypatch.setattr(gg, "_is_gfx906_tensor", lambda x: True)
    module = Gemma4GlobalFake()
    idx = torch.arange(4)
    causal = idx[None, :] <= idx[:, None]

    q16, k16, v16 = _cpu_qkv(dtype=torch.float16)
    bool_mask = causal[None, None]
    fp16_mask = torch.where(causal, 0.0, float("-inf")).to(torch.float16)[None, None]
    fp32_mask = fp16_mask.to(torch.float32)
    assert gg._eligible(module, q16, k16, v16, bool_mask, 0.0, False) is True
    assert gg._eligible(module, q16, k16, v16, fp16_mask, 0.0, False) is True
    assert gg._eligible(module, q16, k16, v16, fp32_mask, 0.0, False) is True

    q32, k32, v32 = _cpu_qkv(dtype=torch.float32)
    for bad_dtype in (torch.float16, torch.bfloat16, torch.float64):
        bad = torch.where(causal, 0.0, float("-inf")).to(bad_dtype)[None, None]
        assert gg._eligible(module, q32, k32, v32, bad, 0.0, False) is False

    fallback_calls = []

    def fallback(*args, **kwargs):
        fallback_calls.append(True)
        return "fallback", None

    bad = torch.where(causal, 0.0, float("-inf")).to(torch.float64)[None, None]
    assert gg._sdpa_maybe_gfx906_global(
        module,
        q32,
        k32,
        v32,
        bad,
        dropout=0.0,
        scaling=1.0,
        is_causal=False,
        _fallback=fallback,
    ) == ("fallback", None)
    assert fallback_calls == [True]


def test_cache_scope_blocks_equal_length_registry_prefill(monkeypatch):
    _reset_env_cache(monkeypatch)
    monkeypatch.setattr(gg, "_is_gfx906_tensor", lambda x: True)
    q, k, v = _cpu_qkv()
    module = Gemma4GlobalFake()
    fallback_calls = []

    fake_kernel_module = types.SimpleNamespace(
        gemma4_gfx906_global_attention=lambda q, k, v, scale: q
    )
    monkeypatch.setitem(
        sys.modules,
        "unsloth_zoo.temporary_patches._gemma4_gfx906_global_kernels",
        fake_kernel_module,
    )

    def fallback(*args, **kwargs):
        fallback_calls.append(True)
        return "fallback", None

    def attention_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
    ):
        return gg._sdpa_maybe_gfx906_global(
            self,
            q,
            k,
            v,
            None,
            dropout=0.0,
            scaling=1.0,
            is_causal=True,
            _fallback=fallback,
        )

    wrapped = gg._make_gemma4_cache_scope_wrapper(attention_forward)

    # Empty-cache prefill still has equal Q/K/V lengths after cache.update().
    # The forward scope, not tensor lengths, must keep it on the original backend.
    assert wrapped(module, None, None, None, object()) == ("fallback", None)
    assert wrapped(
        module, None, None, None, past_key_values=object()
    ) == ("fallback", None)
    assert fallback_calls == [True, True]
    assert gg._CACHE_PRESENT.get() is False

    # No-cache training remains eligible and therefore does not call fallback.
    out, weights = wrapped(module, None, None, None, None)
    assert out.shape == (1, 4, 32, 512)
    assert weights is None
    assert fallback_calls == [True, True]

    # A custom registry caller that supplies cache state directly is also
    # rejected even when it bypasses Gemma4TextAttention.forward.
    assert gg._sdpa_maybe_gfx906_global(
        module,
        q,
        k,
        v,
        None,
        dropout=0.0,
        scaling=1.0,
        is_causal=True,
        past_key_values=object(),
        _fallback=fallback,
    ) == ("fallback", None)
    assert fallback_calls == [True, True, True]


def test_real_gemma4_empty_dynamic_cache_prefill_uses_registry_fallback(monkeypatch):
    """Exercise the Transformers call graph that originally exposed M2."""
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4 import modeling_gemma4 as modeling

    _reset_env_cache(monkeypatch, min_seq="1")
    monkeypatch.setattr(gg, "_is_gfx906_tensor", lambda x: True)

    fallback_calls = []

    def fallback(module, query, key, value, attention_mask, **kwargs):
        fallback_calls.append((query.shape[2], key.shape[2], value.shape[2]))
        return query.transpose(1, 2).contiguous(), None

    def forbidden_kernel(*args, **kwargs):
        raise AssertionError("cache-bearing prefill must not enter custom kernel")

    monkeypatch.setitem(
        sys.modules,
        "unsloth_zoo.temporary_patches._gemma4_gfx906_global_kernels",
        types.SimpleNamespace(gemma4_gfx906_global_attention=forbidden_kernel),
    )
    monkeypatch.setitem(
        ALL_ATTENTION_FUNCTIONS,
        "sdpa",
        gg._make_gfx906_global_wrapper(fallback),
    )
    monkeypatch.setattr(
        modeling.Gemma4TextAttention,
        "forward",
        gg._make_gemma4_cache_scope_wrapper(modeling.Gemma4TextAttention.forward),
    )

    config = Gemma4TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=256,
        num_global_key_value_heads=4,
        global_head_dim=512,
        layer_types=["full_attention"],
        attention_dropout=0.0,
        use_cache=True,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    config.rope_parameters = {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
        "full_attention": {
            "rope_type": "proportional",
            "partial_rotary_factor": 0.25,
            "rope_theta": 1_000_000.0,
        },
    }
    config._attn_implementation = "sdpa"
    layer = modeling.Gemma4TextAttention(config, 0).train()
    rotary = modeling.Gemma4TextRotaryEmbedding(
        config, device="cpu", layer_type="full_attention"
    )
    S = 2
    hidden = torch.randn(1, S, config.hidden_size)
    position_ids = torch.arange(S).unsqueeze(0)
    with torch.no_grad():
        position_embeddings = rotary(hidden, position_ids, "full_attention")

    cache = DynamicCache(config=config)
    output, weights = layer(
        hidden,
        position_embeddings=position_embeddings,
        attention_mask=None,
        shared_kv_states={},
        past_key_values=cache,
    )

    assert output.shape == hidden.shape
    assert weights is None
    assert cache.get_seq_length(0) == S
    # The empty cache has already been updated, so Q/K/V lengths are equal;
    # cache scope is the only reason this routes to fallback.
    assert fallback_calls == [(S, S, S)]
    assert gg._CACHE_PRESENT.get() is False


def test_triton_version_gate_is_part_of_eligibility(monkeypatch):
    _reset_env_cache(monkeypatch)
    monkeypatch.setattr(gg, "_is_gfx906_tensor", lambda x: True)
    monkeypatch.setattr(gg, "_triton_supported", lambda: False)
    q, k, v = _cpu_qkv()
    assert gg._eligible(Gemma4GlobalFake(), q, k, v, None, 0.0, True) is False


@pytest.mark.parametrize("order", [("sliding", "global"), ("global", "sliding")])
def test_global_and_sliding_wrappers_are_order_independent_and_reload_safe(monkeypatch, order):
    """Both install orders remain finite and captured fallbacks survive reset/reload state."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from unsloth_zoo.temporary_patches import gemma4_flash_sliding as gf

    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    old_global_orig = gg._ORIG_SDPA[0]
    old_sliding_orig = gf._ORIG_SDPA[0]
    try:
        def base(*args, **kwargs):
            return "base", "weights"

        ALL_ATTENTION_FUNCTIONS["sdpa"] = base
        for which in order:
            if which == "sliding":
                gf.patch_gemma4_flash_sliding()
            else:
                gg.patch_gemma4_gfx906_global_attention()
        # Run both patches again, as TEMPORARY_PATCHES does across phases.
        gf.patch_gemma4_flash_sliding()
        gg.patch_gemma4_gfx906_global_attention()
        stacked = ALL_ATTENTION_FUNCTIONS["sdpa"]
        assert getattr(stacked, "_unsloth_gemma4_flash", False)
        assert getattr(stacked, "_unsloth_gemma4_gfx906_global", False)

        q, k, v = _cpu_qkv()
        assert stacked(OtherAttentionFake(), q, k, v, None) == ("base", "weights")

        # Simulate module state being reset by reload. Production wrappers must
        # use their captured fallback cells, not these mutable compatibility boxes.
        gg._ORIG_SDPA[0] = None
        gf._ORIG_SDPA[0] = None
        assert stacked(OtherAttentionFake(), q, k, v, None) == ("base", "weights")
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = original
        gg._ORIG_SDPA[0] = old_global_orig
        gf._ORIG_SDPA[0] = old_sliding_orig


def test_eval_path_delegates_to_wrapped_sdpa(monkeypatch):
    _reset_env_cache(monkeypatch)
    monkeypatch.setattr(gg, "_is_gfx906_tensor", lambda x: True)
    q, k, v = _cpu_qkv()
    seen = []

    def fallback(*args, **kwargs):
        seen.append(True)
        return "fallback", "weights"

    module = Gemma4GlobalFake()
    module.training = False
    out = gg._sdpa_maybe_gfx906_global(
        module, q, k, v, None,
        dropout=0.0, scaling=1.0, is_causal=True, _fallback=fallback,
    )
    assert out == ("fallback", "weights")
    assert seen == [True]


def test_reentrant_checkpoint_pack_and_recompute_choose_same_backend(monkeypatch):
    from unsloth_zoo.gradient_checkpointing import Unsloth_Gradient_Checkpointer

    _reset_env_cache(monkeypatch)
    monkeypatch.setattr(gg, "_is_gfx906_tensor", lambda x: True)
    module = Gemma4GlobalFake()
    seen = []
    x = torch.randn(1, 32, 4, 512, requires_grad=True)

    def fn(hidden):
        q = hidden
        k = hidden[:, :4]
        v = hidden[:, :4]
        seen.append((torch.is_grad_enabled(), gg._eligible(module, q, k, v, None, 0.0, True)))
        return hidden.square().sum()

    Unsloth_Gradient_Checkpointer.apply(fn, x).backward()
    assert seen == [(False, True), (True, True)]


def test_scale_none_preserves_sdpa_default(monkeypatch):
    _reset_env_cache(monkeypatch)
    monkeypatch.setattr(gg, "_is_gfx906_tensor", lambda x: True)
    q, k, v = _cpu_qkv()
    seen = []
    fake_kernel_module = types.SimpleNamespace(
        gemma4_gfx906_global_attention=lambda q, k, v, scale: seen.append(scale) or q
    )
    monkeypatch.setitem(
        sys.modules,
        "unsloth_zoo.temporary_patches._gemma4_gfx906_global_kernels",
        fake_kernel_module,
    )
    out = gg.maybe_gemma4_gfx906_global_attention(
        Gemma4GlobalFake(), q, k, v, None, dropout=0.0, scaling=None, is_causal=True
    )
    assert out is q
    assert seen == [pytest.approx(1.0 / math.sqrt(512.0))]


def test_kernel_runtime_error_is_not_swallowed_into_sdpa_fallback(monkeypatch):
    _reset_env_cache(monkeypatch)
    monkeypatch.setattr(gg, "_is_gfx906_tensor", lambda x: True)
    q, k, v = _cpu_qkv()
    fallback_calls = []

    def boom(*args, **kwargs):
        raise RuntimeError("kernel launch failed")

    fake_kernel_module = types.SimpleNamespace(gemma4_gfx906_global_attention=boom)
    monkeypatch.setitem(
        sys.modules,
        "unsloth_zoo.temporary_patches._gemma4_gfx906_global_kernels",
        fake_kernel_module,
    )

    def fallback(*args, **kwargs):
        fallback_calls.append(True)
        return "fallback", None

    with pytest.raises(RuntimeError, match="kernel launch failed"):
        gg._sdpa_maybe_gfx906_global(
            Gemma4GlobalFake(), q, k, v, None,
            dropout=0.0, scaling=1.0, is_causal=True, _fallback=fallback,
        )
    assert fallback_calls == []


def _runtime_is_gfx906():
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        return False
    try:
        arch = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""))
        return arch.split(":", 1)[0] == "gfx906"
    except Exception:
        return False


@pytest.mark.skipif(not _runtime_is_gfx906(), reason="needs AMD gfx906")
@pytest.mark.parametrize(
    "dtype,tol,row_rtol,row_atol",
    [
        (torch.float16, 3e-3, 6e-3, 1.0),
        (torch.bfloat16, 3e-2, 6e-2, 6.0),
        (torch.float32, 5e-4, 1e-3, 2e-2),
    ],
)
def test_kernel_forward_backward_matches_sdpa(dtype, tol, row_rtol, row_atol):
    from unsloth_zoo.temporary_patches._gemma4_gfx906_global_kernels import (
        gemma4_gfx906_global_attention,
    )

    torch.manual_seed(3407)
    S = 127  # odd length exercises tail masks
    q = torch.randn(1, 32, S, 512, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(1, 4, S, 512, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    qr = q.detach().clone().requires_grad_(True)
    kr = k.detach().clone().requires_grad_(True)
    vr = v.detach().clone().requires_grad_(True)

    out = gemma4_gfx906_global_attention(q, k, v, 1.0)
    ref = F.scaled_dot_product_attention(
        qr,
        kr.repeat_interleave(8, dim=1),
        vr.repeat_interleave(8, dim=1),
        is_causal=True,
        scale=1.0,
    )
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad)
    torch.cuda.synchronize()

    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-9)).item()

    def max_group_rel(a, b, dims):
        diff = torch.linalg.vector_norm(a.float() - b.float(), dim=dims)
        denom = torch.linalg.vector_norm(b.float(), dim=dims).clamp_min(1e-6)
        return (diff / denom).amax().item()

    assert rel(out, ref) < tol
    assert rel(q.grad, qr.grad) < tol
    assert rel(k.grad, kr.grad) < tol
    assert rel(v.grad, vr.grad) < tol
    # Whole-tensor norms can hide a localized head/row failure. Keep separate
    # worst-group checks with head/row tolerances derived from the same dtype
    # budget rather than only one aggregate scalar.
    for actual, expected in (
        (out, ref),
        (q.grad, qr.grad),
        (k.grad, kr.grad),
        (v.grad, vr.grad),
    ):
        assert max_group_rel(actual, expected, (-2, -1)) < tol * 4
        _assert_rowwise_mixed_close(
            actual, expected, rtol=row_rtol, atol=row_atol
        )
    assert all(torch.isfinite(x).all() for x in (out, q.grad, k.grad, v.grad))


@pytest.mark.parametrize(
    "tol,row_rtol,row_atol,corruption",
    [
        (3e-3, 6e-3, 1.0, 0.10),
        (3e-2, 6e-2, 6.0, 1.00),
        (5e-4, 1e-3, 2e-2, 0.01),
    ],
)
def test_rowwise_guard_detects_corruption_hidden_by_head_norm(
    tol, row_rtol, row_atol, corruption
):
    reference = torch.full((1, 1, 127, 512), 10.0)
    actual = reference.clone()
    actual[..., 17, :] *= 1.0 + corruption

    head_diff = torch.linalg.vector_norm(actual - reference, dim=(-2, -1))
    head_ref = torch.linalg.vector_norm(reference, dim=(-2, -1))
    assert (head_diff / head_ref).amax().item() < tol * 4
    with pytest.raises(AssertionError, match="localized row error"):
        _assert_rowwise_mixed_close(
            actual, reference, rtol=row_rtol, atol=row_atol
        )


@pytest.mark.skipif(not _runtime_is_gfx906(), reason="needs AMD gfx906")
def test_kernel_rejects_higher_order_graph_construction():
    from unsloth_zoo.temporary_patches._gemma4_gfx906_global_kernels import (
        gemma4_gfx906_global_attention,
    )

    torch.manual_seed(3407)
    S = 8
    q = torch.randn(1, 32, S, 512, device="cuda", dtype=torch.float32, requires_grad=True)
    k = torch.randn(1, 4, S, 512, device="cuda", dtype=torch.float32, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    out = gemma4_gfx906_global_attention(q, k, v, 1.0 / math.sqrt(512.0))
    loss = out.sum() + q.square().sum()

    with pytest.raises(RuntimeError, match="first-order gradients only"):
        torch.autograd.grad(loss, (q, k, v), create_graph=True)


@pytest.mark.skipif(not _runtime_is_gfx906(), reason="needs AMD gfx906")
@pytest.mark.parametrize(
    "dtype,tol",
    [(torch.float16, 3e-3), (torch.float32, 5e-4)],
)
def test_kernel_batch_two_matches_sdpa(dtype, tol):
    from unsloth_zoo.temporary_patches._gemma4_gfx906_global_kernels import (
        gemma4_gfx906_global_attention,
    )

    torch.manual_seed(777)
    B, S = 2, 64
    q = torch.randn(B, 32, S, 512, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(B, 4, S, 512, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    qr = q.detach().clone().requires_grad_(True)
    kr = k.detach().clone().requires_grad_(True)
    vr = v.detach().clone().requires_grad_(True)

    out = gemma4_gfx906_global_attention(q, k, v, 1.0)
    ref = F.scaled_dot_product_attention(
        qr,
        kr.repeat_interleave(8, dim=1),
        vr.repeat_interleave(8, dim=1),
        is_causal=True,
        scale=1.0,
    )
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad)
    torch.cuda.synchronize()

    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-9)).item()

    assert rel(out, ref) < tol
    assert rel(q.grad, qr.grad) < tol
    assert rel(k.grad, kr.grad) < tol
    assert rel(v.grad, vr.grad) < tol


@pytest.mark.skipif(not _runtime_is_gfx906(), reason="needs AMD gfx906")
@pytest.mark.parametrize("S", [15, 16, 17, 31, 32, 33])
def test_kernel_tile_boundaries_match_sdpa(S):
    from unsloth_zoo.temporary_patches._gemma4_gfx906_global_kernels import (
        gemma4_gfx906_global_attention,
    )

    torch.manual_seed(3407 + S)
    scale = 1.0 / math.sqrt(512.0)
    q = torch.randn(1, 32, S, 512, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(1, 4, S, 512, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    qr = q.detach().clone().requires_grad_(True)
    kr = k.detach().clone().requires_grad_(True)
    vr = v.detach().clone().requires_grad_(True)

    out = gemma4_gfx906_global_attention(q, k, v, scale)
    ref = F.scaled_dot_product_attention(
        qr,
        kr.repeat_interleave(8, dim=1),
        vr.repeat_interleave(8, dim=1),
        is_causal=True,
        scale=scale,
    )
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad)
    torch.cuda.synchronize()

    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-9)).item()

    assert rel(out, ref) < 3e-3
    assert rel(q.grad, qr.grad) < 3e-3
    assert rel(k.grad, kr.grad) < 3e-3
    assert rel(v.grad, vr.grad) < 3e-3


@pytest.mark.skipif(not _runtime_is_gfx906(), reason="needs AMD gfx906")
def test_kernel_noncontiguous_qkv_matches_sdpa():
    from unsloth_zoo.temporary_patches._gemma4_gfx906_global_kernels import (
        gemma4_gfx906_global_attention,
    )

    torch.manual_seed(8128)
    S = 33
    q = torch.randn(1, 32, S, 1024, device="cuda", dtype=torch.float16)[..., ::2].requires_grad_()
    k = torch.randn(1, 4, S, 1024, device="cuda", dtype=torch.float16)[..., ::2].requires_grad_()
    v = torch.randn(1, 4, S, 1024, device="cuda", dtype=torch.float16)[..., ::2].requires_grad_()
    assert not q.is_contiguous() and not k.is_contiguous() and not v.is_contiguous()
    qr = q.detach().clone().requires_grad_(True)
    kr = k.detach().clone().requires_grad_(True)
    vr = v.detach().clone().requires_grad_(True)

    out = gemma4_gfx906_global_attention(q, k, v, 1.0)
    ref = F.scaled_dot_product_attention(
        qr,
        kr.repeat_interleave(8, dim=1),
        vr.repeat_interleave(8, dim=1),
        is_causal=True,
        scale=1.0,
    )
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad)
    torch.cuda.synchronize()

    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-9)).item()

    assert rel(out, ref) < 3e-3
    assert rel(q.grad, qr.grad) < 3e-3
    assert rel(k.grad, kr.grad) < 3e-3
    assert rel(v.grad, vr.grad) < 3e-3


@pytest.mark.skipif(not _runtime_is_gfx906(), reason="needs AMD gfx906")
@pytest.mark.parametrize(
    "dtype,tol",
    [(torch.float16, 3e-3), (torch.float32, 5e-4)],
)
def test_kernel_long_1025_matches_sdpa(dtype, tol):
    """Cross many Q/KV tiles at a length just above the default 1024 threshold."""
    from unsloth_zoo.temporary_patches._gemma4_gfx906_global_kernels import (
        gemma4_gfx906_global_attention,
    )

    torch.manual_seed(911)
    S = 1025
    q = torch.randn(1, 32, S, 512, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(1, 4, S, 512, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    qr = q.detach().clone().requires_grad_(True)
    kr = k.detach().clone().requires_grad_(True)
    vr = v.detach().clone().requires_grad_(True)

    out = gemma4_gfx906_global_attention(q, k, v, 1.0)
    ref = F.scaled_dot_product_attention(
        qr,
        kr.repeat_interleave(8, dim=1),
        vr.repeat_interleave(8, dim=1),
        is_causal=True,
        scale=1.0,
    )
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad)
    torch.cuda.synchronize()

    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-9)).item()

    assert rel(out, ref) < tol
    assert rel(q.grad, qr.grad) < tol
    assert rel(k.grad, kr.grad) < tol
    assert rel(v.grad, vr.grad) < tol
    assert all(torch.isfinite(x).all() for x in (out, q.grad, k.grad, v.grad))


def test_force_float32_disabled_leaves_attention_forward_unchanged(monkeypatch):
    from transformers.models.gemma4 import modeling_gemma4 as modeling
    from unsloth_zoo.temporary_patches.gemma4_float32 import patch_Gemma4TextAttention

    original_forward = modeling.Gemma4TextAttention.forward
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "0")
    patch_Gemma4TextAttention()
    assert modeling.Gemma4TextAttention.forward is original_forward


@pytest.mark.skipif(not _runtime_is_gfx906(), reason="needs AMD gfx906")
def test_force_float32_real_gemma4_attention_checkpoint_matches_sdpa(monkeypatch):
    """Real 31B attention shape: checkpoint pack + recompute use the Triton path."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4 import modeling_gemma4 as modeling
    from unsloth_zoo.gradient_checkpointing import Unsloth_Gradient_Checkpointer
    from unsloth_zoo.temporary_patches.gemma4_float32 import patch_Gemma4TextAttention

    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    monkeypatch.setenv("UNSLOTH_GEMMA4_GFX906_GLOBAL_MIN_SEQ", "1")
    original_forward = modeling.Gemma4TextAttention.forward
    try:
        patch_Gemma4TextAttention()
        config = Gemma4TextConfig(
            hidden_size=5376,
            intermediate_size=21504,
            num_hidden_layers=1,
            num_attention_heads=32,
            num_key_value_heads=4,
            head_dim=256,
            num_global_key_value_heads=4,
            global_head_dim=512,
            attention_k_eq_v=True,
            layer_types=["full_attention"],
            attention_dropout=0.0,
            use_cache=False,
            # transformers 5.5 validates rope_parameters before Gemma4's
            # per-layer nested representation is consumed by the rotary module.
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        )
        config.rope_parameters = {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
            "full_attention": {
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
                "rope_theta": 1_000_000.0,
            },
        }
        config._attn_implementation = "sdpa"
        layer = modeling.Gemma4TextAttention(config, 0).to(
            device="cuda", dtype=torch.float16
        ).train()
        for parameter in layer.parameters():
            parameter.requires_grad_(False)

        rotary = modeling.Gemma4TextRotaryEmbedding(
            config, device="cuda", layer_type="full_attention"
        )
        S = 32
        torch.manual_seed(3407)
        base = torch.randn(1, S, 5376, device="cuda", dtype=torch.float16)
        position_ids = torch.arange(S, device="cuda").unsqueeze(0)
        with torch.no_grad():
            position_embeddings = rotary(base, position_ids, "full_attention")
        upstream_grad = torch.randn_like(base)

        def run(enabled):
            monkeypatch.setenv("UNSLOTH_GEMMA4_GFX906_GLOBAL", enabled)
            hidden = base.detach().clone().requires_grad_(True)
            before = gg.gemma4_gfx906_global_stats()["engaged"]

            def forward_only(h):
                return layer(
                    h,
                    position_embeddings=position_embeddings,
                    attention_mask=None,
                )[0]

            output = Unsloth_Gradient_Checkpointer.apply(forward_only, hidden)
            output.backward(upstream_grad)
            torch.cuda.synchronize()
            after = gg.gemma4_gfx906_global_stats()["engaged"]
            return output.detach(), hidden.grad.detach(), after - before

        reference_output, reference_grad, reference_engaged = run("0")
        triton_output, triton_grad, triton_engaged = run("1")

        def rel(a, b):
            return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-9)).item()

        assert reference_engaged == 0
        # Reentrant checkpointing executes the layer once during the no-grad pack
        # and once during backward recompute. Both must use the memory fallback.
        assert triton_engaged == 2
        assert rel(triton_output, reference_output) < 5e-4
        assert rel(triton_grad, reference_grad) < 5e-4
        assert torch.isfinite(triton_output).all()
        assert torch.isfinite(triton_grad).all()
    finally:
        modeling.Gemma4TextAttention.forward = original_forward
