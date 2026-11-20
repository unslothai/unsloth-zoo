# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A repo's image_processor_type may only select a real processor class; transformers.pipeline would import its auto_map code despite trust_remote_code=False."""

import pytest


@pytest.fixture(autouse = True, scope = "module")
def _install_shim():
    # The shim is torch-backed and `[core]` has no torch on darwin/arm64; unguarded, like every tests/test_mlx_*.py, this gives 7 collection ERRORs, not skips.
    pytest.importorskip("torch", reason = "the MLX simulation shim is torch-backed")
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


def _build(preprocessor_config, processor_config = None, model_type = "llava"):
    from unsloth_zoo.mlx.loader import _build_vlm_image_processor_from_config
    return _build_vlm_image_processor_from_config(
        # Nonexistent, so the AutoImageProcessor fallback cannot mask the result.
        "/nonexistent/unsloth-zoo-test-model",
        processor_config if processor_config is not None else {},
        preprocessor_config,
        model_type,
        trust_remote_code = False,
    )


def test_pipeline_is_not_selectable_as_an_image_processor():
    """The regression: `pipeline` is a function, not a processing class."""
    import transformers

    calls = []
    original = transformers.pipeline

    def recording_pipeline(*args, **kwargs):
        calls.append(kwargs)
        raise AssertionError("transformers.pipeline must never be called from here")

    transformers.pipeline = recording_pipeline
    try:
        result = _build({
            "image_processor_type": "pipeline",
            "task": "image-classification",
            "model": "attacker/repo",
            "trust_remote_code": True,
        })
    finally:
        transformers.pipeline = original

    assert not calls, f"a repo-supplied name reached transformers.pipeline with {calls}"
    assert result is None


@pytest.mark.parametrize("name", [
    "pipeline",             # the documented case above
    "logging",              # a submodule, not a class
    "AutoImageProcessor",   # a factory, and not a processing base subclass
])
def test_non_processing_names_are_refused(name):
    from unsloth_zoo.mlx.loader import _is_processor_like_class
    import transformers

    assert not _is_processor_like_class(
        getattr(transformers, name, None)
    ), f"{name} must not be selectable by a repository-supplied name"


def test_a_real_image_processor_still_builds():
    """A real image processor must still build; in Transformers 5 CLIPImageProcessor resolves to CLIPImageProcessorPil without torchvision, as in tests-security."""
    from transformers.image_processing_base import ImageProcessingMixin

    built = _build({
        "image_processor_type": "CLIPImageProcessor",
        "size": {"shortest_edge": 224},
    })
    assert isinstance(built, ImageProcessingMixin)
    assert type(built).__name__ in ("CLIPImageProcessor", "CLIPImageProcessorPil")
    # The sidecar keys still reach the constructor; the gate only filters the callee.
    assert built.size == {"shortest_edge": 224}


def test_repo_supplied_trust_remote_code_is_dropped():
    """Remote-code consent belongs to the caller, never to a downloaded file."""
    from transformers.image_processing_base import ImageProcessingMixin

    built = _build({
        "image_processor_type": "CLIPImageProcessor",
        "size": {"shortest_edge": 224},
        "trust_remote_code": True,
    })
    assert isinstance(built, ImageProcessingMixin)
    assert getattr(built, "trust_remote_code", False) is not True


def test_transformers_fallback_resolver_requires_a_processing_class():
    """The sibling resolver reached the same namespace with only an isinstance check."""
    from unsloth_zoo.mlx.loader import _resolve_mlx_vlm_processor_class

    # `TrainingArguments` is a class, so an `isinstance(x, type)` gate passes it.
    assert _resolve_mlx_vlm_processor_class("llava", "TrainingArguments") is None
    assert _resolve_mlx_vlm_processor_class("llava", "LlavaProcessor") is not None
