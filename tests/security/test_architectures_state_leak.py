# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Rejection must not leave SiglipVisionModel._init_weights patched; arch check runs before patching."""

from __future__ import annotations

import pytest


def _siglip():
    from transformers.models.siglip.modeling_siglip import SiglipVisionModel
    return SiglipVisionModel


def test_a_rejected_architecture_leaves_init_weights_untouched():
    import transformers
    from unsloth_zoo.empty_model import create_empty_vision_model

    siglip = _siglip()
    before = siglip._init_weights
    assert not hasattr(siglip, "_original_initialize_weights")

    config = transformers.SiglipVisionConfig()
    config.architectures = ["AutoTokenizer"]

    with pytest.raises(ValueError, match = "not a model architecture"):
        create_empty_vision_model(config)

    assert siglip._init_weights is before, (
        "_init_weights is still the no-op, so every later SigLIP model skips init"
    )
    assert not hasattr(siglip, "_original_initialize_weights"), (
        "the restore sentinel was left behind"
    )


def test_the_check_precedes_the_patch_in_the_source():
    """Pins the check-before-patch ordering; moving the guard below the patch reintroduces the leak."""
    import inspect
    from unsloth_zoo import empty_model

    source = inspect.getsource(empty_model.create_empty_vision_model)
    guard = source.find("_is_known_architecture")
    patch = source.find("_original_initialize_weights")
    assert guard != -1 and patch != -1
    assert guard < patch, "the architecture guard must run before the SigLIP patch"


def test_cancelling_mid_load_still_restores_init_weights(monkeypatch):
    """KeyboardInterrupt mid-load must not leave the patch installed; needs finally, not except Exception."""
    import transformers
    from unsloth_zoo import empty_model

    siglip = _siglip()
    before = siglip._init_weights

    def cancel(*args, **kwargs):
        raise KeyboardInterrupt("simulated cell cancellation")

    monkeypatch.setattr(empty_model, "get_model_type", lambda config: "siglip_vision_model")
    monkeypatch.setattr(transformers, "SiglipVisionModel", type(siglip), raising = False)

    config = transformers.SiglipVisionConfig()
    config.architectures = ["SiglipVisionModel"]
    monkeypatch.setattr("accelerate.init_empty_weights", cancel, raising = False)

    with pytest.raises(KeyboardInterrupt):
        empty_model.create_empty_vision_model(config)

    assert siglip._init_weights is before, (
        "a cancelled load left _init_weights as the no-op for the whole process"
    )
    assert not hasattr(siglip, "_original_initialize_weights")


def test_a_nested_call_does_not_restore_the_outer_patch():
    """Only the call that installed the patch may remove it."""
    import inspect
    from unsloth_zoo import empty_model

    source = inspect.getsource(empty_model.create_empty_vision_model)
    assert "patched_here" in source, "the ownership flag is gone"
    assert "finally:" in source, "the restore is not in a finally"
    finally_at = source.index("finally:")
    assert "patched_here and hasattr" in source[finally_at:], (
        "the finally must restore only when this call installed the patch"
    )
