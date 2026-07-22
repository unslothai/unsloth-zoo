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

"""CPU tests for the Gemma-4 vision pooler float16 overflow fix.

The buggy transformers (<= 5.9.x) Gemma4VisionPooler scales pooled activations
by sqrt(hidden_size) in the input dtype, so fp16 activations ~2300 overflow to
inf (2300 * 33.94 > 65504) and NaN the loss. The zoo patch upcasts the pooler
input to fp32 (matching the upstream >= 5.10.1 fixed pooler's fp32 output up
to fp16 rounding), lets standardization run in fp32, and casts
last_hidden_state back to the encoder's working dtype (the patch embedder's
input_proj weight dtype, i.e. upstream's inputs_embeds.dtype) in
Gemma4VisionModel.forward.

Synthetic classes below mirror the exact 5.5.0 source shapes so the tests run
on any installed transformers (including 4.57.6, which has no gemma4 at all).
"""

import types

import pytest
import torch

from unsloth_zoo.temporary_patches.gemma4 import (
    _gemma4_vision_cast_dtype,
    _gemma4_vision_pooler_status,
    _patch_gemma4_vision_pooler_fp16,
    patch_Gemma4VisionPoolerFP16,
)
from unsloth_zoo.temporary_patches.common import TEMPORARY_PATCHES

HIDDEN = 1152  # 26B-A4B / 31B vision tower width; sqrt = 33.94


def make_buggy_classes(standardize=True, model_dtype=torch.float16, return_tuple=False):
    """Fresh classes per test (the patch mutates class attributes)."""

    class TinyPooler(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.root_hidden_size = HIDDEN**0.5

        def forward(self, hidden_states, pixel_position_ids, padding_positions, output_length=None):
            hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)
            hidden_states *= self.root_hidden_size
            return hidden_states, padding_positions

    class TinyPatchEmbedder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)

        def forward(self, pixel_values):
            # Like upstream: the embedder casts pixels to its weight dtype.
            # Identity pass instead of the matmul to keep amax controlled.
            return pixel_values.to(self.input_proj.weight.dtype)

    class TinyVisionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embedder = TinyPatchEmbedder()
            self.pooler = TinyPooler()
            self.standardize = standardize
            self.register_buffer("std_bias", torch.full((HIDDEN,), 50000.0, dtype=model_dtype))
            self.register_buffer("std_scale", torch.full((HIDDEN,), 0.02, dtype=model_dtype))

        def forward(self, pixel_values, padding_positions=None):
            hidden_states = self.patch_embedder(pixel_values)
            if padding_positions is None:
                padding_positions = torch.zeros(hidden_states.shape[:2], dtype=torch.bool)
            hidden_states, pooler_mask = self.pooler(hidden_states, None, padding_positions)
            if self.standardize:
                hidden_states = (hidden_states - self.std_bias) * self.std_scale
            if return_tuple:
                return (hidden_states,)
            return types.SimpleNamespace(last_hidden_state=hidden_states)

    return TinyPooler, TinyVisionModel


def make_model(vision_cls, model_dtype=torch.float16):
    model = vision_cls()
    return model.to(model_dtype)


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


def fp16_overflow_input(dtype=torch.float16):
    # amax ~2300 in fp16: finite before the pooler, 2300 * 33.94 = 78k > 65504 after.
    x = torch.linspace(-2300.0, 2300.0, 2 * HIDDEN, dtype=torch.float32)
    return x.reshape(1, 2, HIDDEN).to(dtype)


def test_status_classification():
    TinyPooler, _ = make_buggy_classes()
    assert _gemma4_vision_pooler_status(TinyPooler) == "buggy"
    assert _gemma4_vision_pooler_status(make_fixed_pooler()) == "fixed"

    class Drifted(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states * self.some_other_scale

    assert _gemma4_vision_pooler_status(Drifted) == "unknown"


def test_status_unknown_when_source_unavailable():
    # exec'd class: inspect.getsource fails -> fail open to "unknown"/skip.
    namespace = {"torch": torch}
    exec(
        "class NoSourcePooler(torch.nn.Module):\n"
        "    def forward(self, hidden_states):\n"
        "        return hidden_states * self.root_hidden_size\n",
        namespace,
    )
    assert _gemma4_vision_pooler_status(namespace["NoSourcePooler"]) == "unknown"


def test_unpatched_fp16_overflows():
    _, TinyVisionModel = make_buggy_classes()
    out = make_model(TinyVisionModel)(fp16_overflow_input()).last_hidden_state
    assert not torch.isfinite(out).all(), "expected the buggy pooler to overflow fp16"


def test_patched_fp16_finite_and_matches_fp32_reference():
    TinyPooler, TinyVisionModel = make_buggy_classes()
    x = fp16_overflow_input()

    # fp32 reference on the UNPATCHED classes.
    _, RefVisionModel = make_buggy_classes()
    ref = make_model(RefVisionModel, torch.float32)(x.float()).last_hidden_state

    assert _patch_gemma4_vision_pooler_fp16(TinyPooler, TinyVisionModel) == "patched"
    out = make_model(TinyVisionModel)(x).last_hidden_state
    assert out.dtype == torch.float16, "caller must cast back to the encoder dtype"
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), ref, rtol=2e-3, atol=2.0)


def test_patched_standardize_false_casts_back():
    TinyPooler, TinyVisionModel = make_buggy_classes(standardize=False)
    assert _patch_gemma4_vision_pooler_fp16(TinyPooler, TinyVisionModel) == "patched"
    x = fp16_overflow_input()
    out = make_model(TinyVisionModel)(x).last_hidden_state
    assert out.dtype == torch.float16
    # Exact upstream >= 5.10.1 semantics for standardize=False: fp32 scale
    # then a saturating fp16 cast (values past 65504 become inf).
    expected = (x.float() * HIDDEN**0.5).to(torch.float16)
    assert torch.equal(out, expected)


def test_mixed_fp32_pixels_fp16_tower_casts_to_tower_dtype():
    # Standard HF processor output is fp32; the embedder casts it to fp16.
    # The cast-back target must be the tower dtype, NOT the pixel dtype -
    # otherwise fp32 features hit the fp16 embed_vision projection downstream.
    TinyPooler, TinyVisionModel = make_buggy_classes()
    assert _patch_gemma4_vision_pooler_fp16(TinyPooler, TinyVisionModel) == "patched"
    out = make_model(TinyVisionModel)(fp16_overflow_input(torch.float32)).last_hidden_state
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()


def test_mixed_fp32_pixels_bf16_tower_untouched():
    # bf16 tower: the pooler gate and the fp16-only cast must both stay off.
    TinyPooler, TinyVisionModel = make_buggy_classes(model_dtype=torch.bfloat16)
    x = fp16_overflow_input(torch.float32)
    before = make_model(TinyVisionModel, torch.bfloat16)(x).last_hidden_state
    assert _patch_gemma4_vision_pooler_fp16(TinyPooler, TinyVisionModel) == "patched"
    after = make_model(TinyVisionModel, torch.bfloat16)(x).last_hidden_state
    assert after.dtype == before.dtype
    assert torch.equal(after, before)


def test_bf16_and_fp32_bit_identical():
    for dtype in (torch.bfloat16, torch.float32):
        TinyPooler, TinyVisionModel = make_buggy_classes(model_dtype=dtype)
        x = fp16_overflow_input(dtype)
        before = make_model(TinyVisionModel, dtype)(x).last_hidden_state
        assert _patch_gemma4_vision_pooler_fp16(TinyPooler, TinyVisionModel) == "patched"
        after = make_model(TinyVisionModel, dtype)(x).last_hidden_state
        assert after.dtype == before.dtype
        assert torch.equal(after, before), f"{dtype} must be untouched"


def test_tuple_output_cast():
    # transformers decorators return a plain tuple under return_dict=False;
    # the cast-back must not silently skip that path.
    TinyPooler, TinyVisionModel = make_buggy_classes(return_tuple=True)
    assert _patch_gemma4_vision_pooler_fp16(TinyPooler, TinyVisionModel) == "patched"
    out = make_model(TinyVisionModel)(fp16_overflow_input())
    assert isinstance(out, tuple)
    assert out[0].dtype == torch.float16
    assert torch.isfinite(out[0]).all()


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


def test_repair_reinstalls_missing_vision_wrapper():
    # A marked pooler without the paired vision wrapper (external rebind or
    # an interrupt between installs cannot produce this - the pooler marker
    # is the commit point - but external code can) must be repaired, never
    # reported "already".
    TinyPooler, TinyVisionModel = make_buggy_classes()
    original_vision = TinyVisionModel.forward
    assert _patch_gemma4_vision_pooler_fp16(TinyPooler, TinyVisionModel) == "patched"
    TinyVisionModel.forward = original_vision  # simulate external rebind
    assert _patch_gemma4_vision_pooler_fp16(TinyPooler, TinyVisionModel) == "repaired"
    assert getattr(TinyVisionModel.forward, "_unsloth_vision_pooler_fp16", False)
    out = make_model(TinyVisionModel)(fp16_overflow_input()).last_hidden_state
    assert out.dtype == torch.float16 and torch.isfinite(out).all()


def test_kwarg_calls_handled():
    TinyPooler, TinyVisionModel = make_buggy_classes()
    assert _patch_gemma4_vision_pooler_fp16(TinyPooler, TinyVisionModel) == "patched"
    x = fp16_overflow_input()
    pooler = TinyPooler()
    padding = torch.zeros(x.shape[:2], dtype=torch.bool)
    hs, _ = pooler(hidden_states=x, pixel_position_ids=None, padding_positions=padding)
    assert hs.dtype == torch.float32 and torch.isfinite(hs).all()
    out = make_model(TinyVisionModel)(pixel_values=x).last_hidden_state
    assert out.dtype == torch.float16 and torch.isfinite(out).all()


def test_cast_dtype_helper_fallbacks():
    TinyPooler, TinyVisionModel = make_buggy_classes()
    model = make_model(TinyVisionModel)
    assert _gemma4_vision_cast_dtype(model, None) == torch.float16
    bare = types.SimpleNamespace()  # no patch_embedder: fall back to pixels
    assert _gemma4_vision_cast_dtype(bare, torch.zeros(1, dtype=torch.bfloat16)) == torch.bfloat16
    assert _gemma4_vision_cast_dtype(bare, [torch.zeros(1, dtype=torch.float16)]) == torch.float16
    assert _gemma4_vision_cast_dtype(bare, None) is None


def test_gradients_flow_through_patch():
    TinyPooler, TinyVisionModel = make_buggy_classes()
    assert _patch_gemma4_vision_pooler_fp16(TinyPooler, TinyVisionModel) == "patched"
    x = fp16_overflow_input().requires_grad_(True)
    out = make_model(TinyVisionModel)(x).last_hidden_state
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
    patches (buggy source), skips (fixed source), or no-ops (no gemma4).
    NOTE: on buggy installs this mutates the process-global transformers
    classes (that is the shipping behavior); other tests here only use
    per-test synthetic classes, so ordering does not matter."""
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
