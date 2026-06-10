# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Structural pins for the PR #659 cleanup (not the FP8/bnb4bit kernels):

1. The unused post-load FP8 scale reattach machinery is gone.
2. _call_with_temporary_moe_weights is unified in moe_utils (removed from
   moe_utils_fp8 and moe_utils_bnb4bit).
3. The PEFT bnb4bit MoE merge/unmerge patches live in moe_utils_bnb4bit,
   not misc.py.
"""

from __future__ import annotations

import pytest


_DEAD_FP8_REATTACH_SYMBOLS = (
    "_maybe_patch_glm4_stacked_moe_fp8_scales",
    "_do_glm4_scale_patching",
    "maybe_patch_stacked_moe_expert_fp8_scales",
    "_maybe_attach_dropped_moe_fp8_scales",
    "_attach_stacked_scales",
    "_attach_per_expert_scales",
    "_annotate_block_size",
    "_resolve_safetensors_shards",
)


def test_dead_fp8_reattach_machinery_is_gone():
    from unsloth_zoo.temporary_patches import moe_utils_fp8
    leftover = [s for s in _DEAD_FP8_REATTACH_SYMBOLS if hasattr(moe_utils_fp8, s)]
    assert not leftover, f"dead FP8 reattach symbols still present: {leftover}"


def test_moe_utils_fp8_does_not_open_safetensors_at_load_time():
    """The deleted reattach path was the only consumer of safetensors /
    hf_hub_download in moe_utils_fp8; nothing should reopen checkpoint shards."""
    from pathlib import Path
    import unsloth_zoo.temporary_patches.moe_utils_fp8 as m
    src = Path(m.__file__).read_text()
    forbidden = ("safetensors.safe_open", "hf_hub_download", "safetensors.torch")
    for needle in forbidden:
        assert needle not in src, (
            f"moe_utils_fp8.py still mentions {needle!r}; "
            "the post-load reattach machinery should have been removed."
        )


def test_call_with_temporary_moe_weights_unified():
    """One canonical helper in moe_utils; FP8 and bnb4bit import it."""
    from unsloth_zoo.temporary_patches import moe_utils, moe_utils_fp8, moe_utils_bnb4bit

    assert callable(getattr(moe_utils, "swap_moe_weights_for_call", None))
    assert not hasattr(moe_utils_fp8, "_call_with_temporary_moe_weights")
    assert not hasattr(moe_utils_bnb4bit, "_call_with_temporary_moe_weights")


def test_peft_bnb4bit_moe_patches_live_in_moe_utils_bnb4bit():
    """The three PEFT-bnb4bit helpers moved out of misc.py."""
    from unsloth_zoo.temporary_patches import misc, moe_utils_bnb4bit

    moved = (
        "_ParamShapeProxy",
        "patch_peft_param_wrapper_4bit_expert_shape",
        "patch_peft_param_wrapper_merge_4bit",
    )
    for sym in moved:
        assert hasattr(moe_utils_bnb4bit, sym), f"missing in moe_utils_bnb4bit: {sym}"
        assert not hasattr(misc, sym), f"still in misc.py: {sym}"


def test_temporary_patches_register_expected_count():
    """Phase 3 must NOT silently drop a TEMPORARY_PATCHES.append entry.
    moe_utils_bnb4bit now registers 3 more patches than before."""
    from unsloth_zoo.temporary_patches import moe_utils_bnb4bit
    import inspect

    src = inspect.getsource(moe_utils_bnb4bit)
    # Count is sticky to the file; if Phase 3 forgets the .append, we drop one.
    count = src.count("TEMPORARY_PATCHES.append(")
    assert count == 6, (
        f"moe_utils_bnb4bit registers {count} patches; expected 6 "
        "(4 original bnb4bit patches + the 2 PEFT MoE patches relocated from misc.py)."
    )


def test_transformers_v5_moe_quant_gate_false_without_param_needs_quantization(monkeypatch):
    """Transformers v4-style Bnb4BitHfQuantizer has no v5 quantization gate.
    PR #659's v5-only patch set must not register for that API shape."""
    from transformers.quantizers import quantizer_bnb_4bit
    from unsloth_zoo.temporary_patches import common

    class V4StyleBnb4BitHfQuantizer:
        pass

    monkeypatch.setattr(
        quantizer_bnb_4bit,
        "Bnb4BitHfQuantizer",
        V4StyleBnb4BitHfQuantizer,
    )
    common.is_transformers_v5_moe_quantization_available.cache_clear()
    try:
        assert common.is_transformers_v5_moe_quantization_available() is False
    finally:
        common.is_transformers_v5_moe_quantization_available.cache_clear()


def test_bnb4bit_param_needs_quantization_patch_noops_when_hook_absent(monkeypatch):
    """The direct traceback was an AttributeError on this missing hook."""
    from transformers.quantizers import quantizer_bnb_4bit
    from unsloth_zoo.temporary_patches import moe_utils_bnb4bit

    class V4StyleBnb4BitHfQuantizer:
        pass

    monkeypatch.setattr(
        quantizer_bnb_4bit,
        "Bnb4BitHfQuantizer",
        V4StyleBnb4BitHfQuantizer,
    )

    moe_utils_bnb4bit.patch_bnb4bit_quantizer_param_needs_quantization()
    assert not hasattr(V4StyleBnb4BitHfQuantizer, "param_needs_quantization")


def test_v5_moe_quant_registration_helpers_respect_gate(monkeypatch):
    """If capability detection says v4/non-v5, registration must be a no-op."""
    from unsloth_zoo.temporary_patches.common import TEMPORARY_PATCHES
    from unsloth_zoo.temporary_patches import moe_utils_bnb4bit, moe_utils_fp8

    before = list(TEMPORARY_PATCHES)
    monkeypatch.setattr(
        moe_utils_bnb4bit,
        "is_transformers_v5_moe_quantization_available",
        lambda: False,
    )
    monkeypatch.setattr(
        moe_utils_fp8,
        "is_transformers_v5_moe_quantization_available",
        lambda: False,
    )

    moe_utils_bnb4bit._register_transformers_v5_moe_bnb4bit_patches()
    moe_utils_fp8._register_transformers_v5_moe_fp8_patches()
    assert TEMPORARY_PATCHES == before


def test_v5_moe_quant_patches_registered_when_capability_exists():
    """On the v5 API shape, PR #659's v5-only patches remain active."""
    from unsloth_zoo.temporary_patches import common, moe_utils_bnb4bit, moe_utils_fp8

    common.is_transformers_v5_moe_quantization_available.cache_clear()
    if not common.is_transformers_v5_moe_quantization_available():
        pytest.skip("Transformers v5 MoE quantization APIs are not available")

    expected = {
        moe_utils_bnb4bit.patch_bnb4bit_quantize_convert,
        moe_utils_bnb4bit.patch_bnb4bit_quantizer_param_needs_quantization,
        moe_utils_bnb4bit.patch_bnb4bit_quantizer_process_model,
        moe_utils_bnb4bit.patch_transformers_weight_converter_kwargs,
        moe_utils_bnb4bit.patch_peft_param_wrapper_4bit_expert_shape,
        moe_utils_bnb4bit.patch_peft_param_wrapper_merge_4bit,
        moe_utils_fp8.patch_fp8_experts_interface,
        moe_utils_fp8.patch_fp8_validate_quantization_for_training,
    }
    registered = set(common.TEMPORARY_PATCHES)
    missing = expected - registered
    assert not missing, f"v5 MoE quant patch(es) not registered: {missing}"
