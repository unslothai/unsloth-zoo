# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Rejecting an architecture must not leave process-wide state changed.

`create_empty_vision_model` replaces `SiglipVisionModel._init_weights` with a no-op so
meta-device construction skips initialisation, and restores it at the end. A guard that
raises between those two points would leave every later SigLIP model in the process
skipping weight init, which is silent and would not show up as an error anywhere near
the cause. So the architecture check runs before the patch, not after.

CPU-only and network-free: the config is built in memory and rejected before anything
is constructed.
"""

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
    """Pins the ordering the test above relies on, so a later edit cannot silently
    reintroduce the leak by moving the guard back down."""
    import inspect
    from unsloth_zoo import empty_model

    source = inspect.getsource(empty_model.create_empty_vision_model)
    guard = source.find("_is_known_architecture")
    patch = source.find("_original_initialize_weights")
    assert guard != -1 and patch != -1
    assert guard < patch, "the architecture guard must run before the SigLIP patch"
