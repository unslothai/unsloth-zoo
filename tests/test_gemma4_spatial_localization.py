"""Tests for the Gemma4 multimodal projection patch (issue #6028 follow-up).

These tests are written to FAIL if the patch regresses, which the previous
shape/dtype-only versions did not:

  - `test_projection_preserves_lora_delta` fails if the patch reads
    `embedding_projection.weight` instead of calling the module, because a PEFT
    LoRA adapter on that projector would then contribute nothing and receive no
    gradient.
  - `test_projection_matches_fp64_reference` pins the numerics against an fp64
    reference rather than only checking the output dtype.
  - `test_upstream_signatures` pins the upstream signature the patch relies on,
    matching the convention in tests/test_temporary_patches_exhaustive.py.
"""
import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def _apply_patches():
    import unsloth_zoo.temporary_patches  # noqa: F401  (applies TEMPORARY_PATCHES)


def _configs(mm_dim=64, text_dim=32):
    from transformers.models.gemma4.configuration_gemma4 import (
        Gemma4TextConfig,
        Gemma4VisionConfig,
    )
    vision_config = Gemma4VisionConfig(
        output_proj_dims=mm_dim, hidden_size=mm_dim, rms_norm_eps=1e-6,
    )
    text_config = Gemma4TextConfig(hidden_size=text_dim)
    return vision_config, text_config


def _embedder(mm_dim=64, text_dim=32):
    from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder
    vision_config, text_config = _configs(mm_dim, text_dim)
    return Gemma4MultimodalEmbedder(vision_config, text_config).float()


def test_config_construction_does_not_raise():
    """The KV-shared proxy hides `num_kv_shared_layers`; it must hide it from
    iteration too, or upstream's validate_token_ids raises AttributeError."""
    _configs()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_projection_shape_and_dtype(dtype):
    embedder = _embedder().to(dtype)
    out = embedder(torch.randn(2, 16, 64, dtype=dtype))
    assert out.shape == (2, 16, 32)
    assert out.dtype == dtype


def test_projection_matches_fp64_reference():
    """Patched forward must stay close to an fp64 reference of the same math."""
    embedder = _embedder()
    x = torch.randn(2, 16, 64)

    norm = embedder.embedding_pre_projection_norm
    ref = (norm._norm(x.double())) @ embedder.embedding_projection.weight.double().T

    out = embedder(x)
    assert torch.allclose(out.double(), ref, atol=1e-4), (
        f"max|out-ref| = {float((out.double() - ref).abs().max())}"
    )


def test_projection_preserves_lora_delta():
    """A LoRA adapter on `embedding_projection` must affect the output AND get
    a gradient. Reading `.weight` instead of calling the module silently drops
    both, which is what `finetune_vision_layers` / `finetune_audio_layers`
    attach here."""
    peft = pytest.importorskip("peft")

    embedder = _embedder()
    x = torch.randn(2, 16, 64)
    with torch.no_grad():
        baseline = embedder(x).clone()

    wrapped = peft.get_peft_model(
        _embedder(),
        peft.LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.0,
            target_modules=["embedding_projection"], bias="none",
        ),
    )
    inner = wrapped.base_model.model
    proj = inner.embedding_projection
    with torch.no_grad():
        proj.base_layer.weight.copy_(embedder.embedding_projection.weight)
        # PEFT zero-inits lora_B, so without this the delta is zero and this
        # test would pass even with the projection bypassed.
        torch.nn.init.normal_(proj.lora_A["default"].weight, std=0.5)
        torch.nn.init.normal_(proj.lora_B["default"].weight, std=0.5)

    with torch.no_grad():
        adapted = inner(x)
    assert not torch.allclose(adapted, baseline), (
        "LoRA adapter on embedding_projection had no effect on the output: "
        "the projection path is bypassing the PEFT wrapper"
    )

    inner.zero_grad(set_to_none=True)
    out = inner(x)
    assert out.requires_grad, (
        "patched forward produced an output with no grad_fn: nothing in the "
        "projection path is trainable"
    )
    out.square().sum().backward()
    grad = proj.lora_B["default"].weight.grad
    assert grad is not None and float(grad.norm()) > 0.0, (
        "lora_B received no gradient: the adapter cannot train"
    )


def test_upstream_signatures():
    """Pin the upstream signature the patch replaces (see
    tests/test_temporary_patches_exhaustive.py for this convention)."""
    import inspect

    from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder

    unpatched = getattr(
        Gemma4MultimodalEmbedder,
        "_unsloth_original_forward",
        Gemma4MultimodalEmbedder.forward,
    )
    params = list(inspect.signature(unpatched).parameters)
    assert params[:2] == ["self", "inputs_embeds"], params


def test_temporary_patch_registered():
    from unsloth_zoo.temporary_patches import TEMPORARY_PATCHES

    names = [p.__name__ for p in TEMPORARY_PATCHES]
    assert "patch_Gemma4MultimodalEmbedder_forward" in names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
