"""CPU tests for the Gemma-4 vision pooler float16 overflow fix.

The buggy transformers (<= 5.9.x) Gemma4VisionPooler scales pooled activations
by sqrt(hidden_size) in the input dtype, so fp16 activations ~2300 overflow to
inf (2300 * 33.94 > 65504) and NaN the loss. The zoo patch upcasts the pooler
input to fp32 (reproducing the upstream >= 5.10.1 fixed pooler's fp32 output),
lets standardization run in fp32, and casts last_hidden_state back to the
pixel_values dtype in Gemma4VisionModel.forward.

Synthetic classes below mirror the exact 5.5.0 source shapes so the tests run
on any installed transformers (including 4.57.6, which has no gemma4 at all).
"""

import math
import types

import pytest
import torch

from unsloth_zoo.temporary_patches.gemma4 import (
    _gemma4_vision_pooler_status,
    _patch_gemma4_vision_pooler_fp16,
    patch_Gemma4VisionPoolerFP16,
)
from unsloth_zoo.temporary_patches.common import TEMPORARY_PATCHES

HIDDEN = 1152  # 26B-A4B / 31B vision tower width; sqrt = 33.94


def make_buggy_classes(standardize=True):
    """Fresh classes per test (the patch mutates class attributes)."""

    class TinyPooler(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.root_hidden_size = HIDDEN**0.5

        def forward(self, hidden_states, pixel_position_ids, padding_positions, output_length=None):
            hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)
            hidden_states *= self.root_hidden_size
            return hidden_states, padding_positions

    class TinyVisionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pooler = TinyPooler()
            self.standardize = standardize
            self.register_buffer("std_bias", torch.full((HIDDEN,), 50000.0, dtype=torch.float16))
            self.register_buffer("std_scale", torch.full((HIDDEN,), 0.02, dtype=torch.float16))

        def forward(self, pixel_values, padding_positions=None):
            if padding_positions is None:
                padding_positions = torch.zeros(pixel_values.shape[:2], dtype=torch.bool)
            hidden_states, pooler_mask = self.pooler(pixel_values, None, padding_positions)
            if self.standardize:
                hidden_states = (hidden_states - self.std_bias) * self.std_scale
            return types.SimpleNamespace(last_hidden_state=hidden_states)

    return TinyPooler, TinyVisionModel


def make_fixed_pooler():
    class FixedPooler(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.root_hidden_size = HIDDEN**0.5

        def forward(self, hidden_states, pixel_position_ids, padding_positions, output_length=None):
            hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)
            hidden_states = hidden_states.float() * self.root_hidden_size
            return hidden_states, padding_positions

    return FixedPooler


def fp16_overflow_input():
    # amax ~2300 in fp16: finite before the pooler, 2300 * 33.94 = 78k > 65504 after.
    x = torch.linspace(-2300.0, 2300.0, 2 * HIDDEN, dtype=torch.float32)
    return x.reshape(1, 2, HIDDEN).to(torch.float16)


def test_status_classification():
    TinyPooler, _ = make_buggy_classes()
    assert _gemma4_vision_pooler_status(TinyPooler) == "buggy"
    assert _gemma4_vision_pooler_status(make_fixed_pooler()) == "fixed"

    class Drifted(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states * self.some_other_scale

    assert _gemma4_vision_pooler_status(Drifted) == "unknown"


def test_unpatched_fp16_overflows():
    _, TinyVisionModel = make_buggy_classes()
    out = TinyVisionModel()(fp16_overflow_input()).last_hidden_state
    assert not torch.isfinite(out).all(), "expected the buggy pooler to overflow fp16"


def test_patched_fp16_finite_and_matches_fp32_reference():
    TinyPooler, TinyVisionModel = make_buggy_classes()
    x = fp16_overflow_input()

    # fp32 reference on the UNPATCHED classes.
    ref = TinyVisionModel()(x.float()).last_hidden_state

    assert _patch_gemma4_vision_pooler_fp16(TinyPooler, TinyVisionModel) == "patched"
    out = TinyVisionModel()(x).last_hidden_state
    assert out.dtype == torch.float16, "caller must cast back to the pixel_values dtype"
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), ref, rtol=2e-3, atol=2.0)


def test_patched_standardize_false_casts_back():
    TinyPooler, TinyVisionModel = make_buggy_classes(standardize=False)
    assert _patch_gemma4_vision_pooler_fp16(TinyPooler, TinyVisionModel) == "patched"
    out = TinyVisionModel()(fp16_overflow_input()).last_hidden_state
    assert out.dtype == torch.float16
    # No standardization to cancel the overflow: fp32 78k saturates the fp16
    # cast to inf, matching what upstream >= 5.10.1 does for standardize=False.
    assert torch.isfinite(out).any()


def test_bf16_and_fp32_bit_identical():
    for dtype in (torch.bfloat16, torch.float32):
        TinyPooler, TinyVisionModel = make_buggy_classes()
        x = fp16_overflow_input().to(dtype)
        before_model = TinyVisionModel().to(dtype)  # real bf16/fp32 models have uniform buffers
        before = before_model(x).last_hidden_state
        assert _patch_gemma4_vision_pooler_fp16(TinyPooler, TinyVisionModel) == "patched"
        after = TinyVisionModel().to(dtype)(x).last_hidden_state
        assert after.dtype == before.dtype
        assert torch.equal(after, before), f"{dtype} must be untouched"


def test_fixed_source_is_skipped():
    FixedPooler = make_fixed_pooler()
    _, TinyVisionModel = make_buggy_classes()
    original_pooler = FixedPooler.forward
    original_vision = TinyVisionModel.forward
    assert _patch_gemma4_vision_pooler_fp16(FixedPooler, TinyVisionModel) == "fixed"
    assert FixedPooler.forward is original_pooler
    assert TinyVisionModel.forward is original_vision


def test_drift_source_is_skipped_atomically():
    class Drifted(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states * self.some_other_scale

    _, TinyVisionModel = make_buggy_classes()
    original_pooler = Drifted.forward
    original_vision = TinyVisionModel.forward
    assert _patch_gemma4_vision_pooler_fp16(Drifted, TinyVisionModel) == "unknown"
    assert Drifted.forward is original_pooler
    assert TinyVisionModel.forward is original_vision, "drift must skip BOTH classes"


def test_idempotent():
    TinyPooler, TinyVisionModel = make_buggy_classes()
    assert _patch_gemma4_vision_pooler_fp16(TinyPooler, TinyVisionModel) == "patched"
    once_pooler = TinyPooler.forward
    once_vision = TinyVisionModel.forward
    assert _patch_gemma4_vision_pooler_fp16(TinyPooler, TinyVisionModel) == "already"
    assert TinyPooler.forward is once_pooler
    assert TinyVisionModel.forward is once_vision


def test_kwarg_calls_handled():
    TinyPooler, TinyVisionModel = make_buggy_classes()
    assert _patch_gemma4_vision_pooler_fp16(TinyPooler, TinyVisionModel) == "patched"
    x = fp16_overflow_input()
    pooler = TinyPooler()
    padding = torch.zeros(x.shape[:2], dtype=torch.bool)
    hs, _ = pooler(hidden_states=x, pixel_position_ids=None, padding_positions=padding)
    assert hs.dtype == torch.float32 and torch.isfinite(hs).all()
    out = TinyVisionModel()(pixel_values=x).last_hidden_state
    assert out.dtype == torch.float16 and torch.isfinite(out).all()


def test_gradients_flow_through_patch():
    TinyPooler, TinyVisionModel = make_buggy_classes()
    assert _patch_gemma4_vision_pooler_fp16(TinyPooler, TinyVisionModel) == "patched"
    x = fp16_overflow_input().requires_grad_(True)
    out = TinyVisionModel()(x).last_hidden_state
    out.float().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_wrapper_registered():
    assert patch_Gemma4VisionPoolerFP16 in TEMPORARY_PATCHES


def test_real_transformers_source_canary():
    """Drift guard: the installed pooler must be a known-buggy or known-fixed
    shape. Fails loudly when upstream rewrites the pooler so the pattern list
    gets refreshed."""
    modeling = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
    pooler_cls = getattr(modeling, "Gemma4VisionPooler", None)
    if pooler_cls is None:
        pytest.skip("transformers has no Gemma4VisionPooler")
    status = _gemma4_vision_pooler_status(pooler_cls)
    assert status in ("buggy", "fixed"), (
        f"Gemma4VisionPooler.forward drifted (status={status}) - update "
        "_gemma4_vision_pooler_status patterns and re-verify the fp16 fix"
    )


def test_real_transformers_patch_applies_or_skips():
    """End-to-end on the installed transformers: the public wrapper either
    patches (buggy source), skips (fixed source), or no-ops (no gemma4)."""
    try:
        import transformers.models.gemma4.modeling_gemma4 as modeling
    except ImportError:
        patch_Gemma4VisionPoolerFP16()  # must no-op without raising
        return
    pooler_cls = getattr(modeling, "Gemma4VisionPooler", None)
    vision_cls = getattr(modeling, "Gemma4VisionModel", None)
    if pooler_cls is None or vision_cls is None:
        patch_Gemma4VisionPoolerFP16()
        return
    status_before = _gemma4_vision_pooler_status(pooler_cls)
    patch_Gemma4VisionPoolerFP16()
    if status_before == "buggy":
        assert getattr(pooler_cls.forward, "_unsloth_vision_pooler_fp16", False)
        assert getattr(vision_cls.forward, "_unsloth_vision_pooler_fp16", False)
    elif status_before == "fixed":
        assert not getattr(pooler_cls.forward, "_unsloth_vision_pooler_fp16", False)
