"""The #6028 backport must fire only for installs whose unsloth lacks the guard.

unsloth's generate forces a static cache, which drops Gemma's bidirectional
image block overlay at prefill. Current unsloth gates that itself; this shim
covers older ones by clearing `_supports_static_cache`, the single flag
unsloth/models/vision.py consults, and must stay out of the way otherwise.
"""
import sys
import types

import pytest

import unsloth_zoo.temporary_patches  # noqa: F401  (registers the patches)
from unsloth_zoo.temporary_patches.gemma4 import patch_Gemma4_static_cache_backport


@pytest.fixture
def fake_gemma(request):
    """A gemma4_unified-shaped module: one class with the overlay, one without."""
    mod_name = "transformers.models.gemma4_unified.modeling_gemma4_unified"
    saved = sys.modules.get(mod_name)
    module = types.ModuleType(mod_name)

    class WithOverlay:
        @staticmethod
        def create_masks_for_generate(*args, **kwargs):
            return None

    class WithoutOverlay:
        pass

    module.WithOverlay = WithOverlay
    module.WithoutOverlay = WithoutOverlay
    module.NotAClass = 42
    sys.modules[mod_name] = module
    yield module
    if saved is None:
        sys.modules.pop(mod_name, None)
    else:
        sys.modules[mod_name] = saved


@pytest.fixture
def fake_vision(request):
    """Stand in for unsloth.models.vision without importing unsloth."""
    def _install(has_guard):
        module = types.ModuleType("unsloth.models.vision")
        if has_guard:
            module._needs_bidirectional_multimodal_mask = lambda *a, **k: False
        sys.modules["unsloth.models.vision"] = module
        return module

    saved = sys.modules.get("unsloth.models.vision")
    yield _install
    if saved is None:
        sys.modules.pop("unsloth.models.vision", None)
    else:
        sys.modules["unsloth.models.vision"] = saved


def test_old_unsloth_gets_the_backport(fake_gemma, fake_vision):
    fake_vision(has_guard=False)
    patch_Gemma4_static_cache_backport(phase="post_compile")
    assert fake_gemma.WithOverlay._supports_static_cache is False


def test_current_unsloth_is_left_alone(fake_gemma, fake_vision):
    fake_vision(has_guard=True)
    patch_Gemma4_static_cache_backport(phase="post_compile")
    assert not hasattr(fake_gemma.WithOverlay, "_supports_static_cache")


def test_classes_without_the_overlay_are_untouched(fake_gemma, fake_vision):
    """Causal VLMs must keep the static cache path."""
    fake_vision(has_guard=False)
    patch_Gemma4_static_cache_backport(phase="post_compile")
    assert not hasattr(fake_gemma.WithoutOverlay, "_supports_static_cache")


def test_unsloth_not_imported_yet_is_a_no_op(fake_gemma, fake_vision):
    """At the init phase vision is absent, so the version check cannot be made
    and the safe answer is to do nothing."""
    sys.modules.pop("unsloth.models.vision", None)
    patch_Gemma4_static_cache_backport(phase="post_compile")
    assert not hasattr(fake_gemma.WithOverlay, "_supports_static_cache")


@pytest.mark.parametrize("phase", ["init", "pre_compile", "anything_else"])
def test_only_runs_at_post_compile(fake_gemma, fake_vision, phase):
    fake_vision(has_guard=False)
    patch_Gemma4_static_cache_backport(phase=phase)
    assert not hasattr(fake_gemma.WithOverlay, "_supports_static_cache")


def test_registered_with_a_phase_parameter():
    """unsloth calls the patch with `phase` only if it declares the parameter,
    so a rename there would silently make this run at init."""
    import inspect
    assert "phase" in inspect.signature(patch_Gemma4_static_cache_backport).parameters


def test_unsloth_vision_reads_the_flag_we_clear():
    """Pin the unsloth-side contract: the shim is pointless if that attribute
    stops being what selects the cache."""
    vision = pytest.importorskip("unsloth.models.vision")
    src = inspect_source(vision)
    assert "_supports_static_cache" in src


def inspect_source(module):
    import inspect
    return inspect.getsource(module)
