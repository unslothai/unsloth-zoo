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

"""A name out of a model repo may only select a real processing class.

`_build_vlm_image_processor_from_config` takes `image_processor_type` out of the
downloaded repo's `processor_config.json` / `preprocessor_config.json`, resolves it
against the `transformers` top-level namespace and calls the result with the rest of
that same JSON object as keyword arguments. `transformers` exports module-level
functions there as well as classes, and `pipeline` is one of them: it accepts
`trust_remote_code`, and honouring it imports the repository's own `auto_map` Python.
So the resolved object has to be checked before it is called, on a path the user
reached with `trust_remote_code=False`.
"""

import pytest


@pytest.fixture(autouse = True, scope = "module")
def _install_shim():
    # The MLX simulation shim is torch-backed, and `[core]` does not install torch on
    # darwin/arm64. Every `tests/test_mlx_*.py` calls this unguarded, which is fine for
    # a Linux-only file; this one sits in the security suite, so an absent torch has to
    # come out as a skip and not as a collection ERROR that reds the gate wholesale.
    pytest.importorskip("torch", reason = "the MLX simulation shim is torch-backed")
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


def _build(preprocessor_config, processor_config = None, model_type = "llava"):
    from unsloth_zoo.mlx.loader import _build_vlm_image_processor_from_config
    return _build_vlm_image_processor_from_config(
        # A path that does not exist, so the AutoImageProcessor fallback below the
        # resolver cannot succeed and mask the result.
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
    """The gate must not break the case this code exists for.

    Not an exact-name assertion. `CLIPImageProcessor` is a `TorchvisionBackend` in
    Transformers 5, and with torchvision absent the lazy module resolves that name to
    `CLIPImageProcessorPil` instead. torchvision is not a dependency of this project,
    and the `tests-security` job installs `.[core]` only, so the PIL class is what the
    hard gate actually builds. Both are CLIP image processors deriving from
    `ImageProcessingMixin`, which is what the gate under test has to keep admitting.
    """
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
