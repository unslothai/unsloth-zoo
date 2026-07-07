"""Regression: NEFTune's runtime ``__class__`` swap must not drop the quantized
``embed_tokens`` from the resolved quantization map.

MLXTrainer._install_neftune reassigns ``embed_tokens.__class__`` to a subclass
(``_NEFTuneEmbed``) for the duration of training, which includes the adapter
save. ``_effective_mlx_quantization_map`` used to detect quantized layers by an
exact class-name match (``type(module).__name__ in {...}``), so the subclassed
embedding was silently dropped from the saved ``base_resolved_quantization_map``.
On reload the saved map was then validated against the unmodified base (a real
``QuantizedEmbedding``), producing "unexpected quantized modules:
['model.embed_tokens']" and a hard failure. The fix switches both copies of the
scan to ``isinstance``.

Apple-Silicon/Metal only (needs a real quantized MLX model).
"""

import pytest

try:
    import mlx.core as mx
    _METAL = mx.metal.is_available()
except Exception:
    _METAL = False

metal_only = pytest.mark.skipif(not _METAL, reason="requires Apple Silicon Metal")

MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"


def _neftune_swap(emb):
    """Mimic MLXTrainer._install_neftune's ``__class__`` reassignment."""
    base = type(emb)
    emb.__class__ = type("_NEFTuneEmbed", (base,), {})
    return base


@metal_only
def test_neftune_subclass_kept_in_quantization_map():
    """Both scans must still recognise a subclassed quantized embedding."""
    from unsloth_zoo.mlx import loader as mlx_loader
    from unsloth_zoo.mlx import utils as mlx_utils
    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.utils import _get_text_model

    model, _ = FastMLXModel.from_pretrained(MODEL, max_seq_length=128)
    emb = _get_text_model(model).model.embed_tokens

    for _effective in (
        mlx_utils._effective_mlx_quantization_map,
        mlx_loader._effective_mlx_quantization_map,
    ):
        base_map = _effective(model)
        assert any("embed_tokens" in k for k in base_map), \
            f"embed_tokens missing from base map: {sorted(base_map)}"

        original = _neftune_swap(emb)
        try:
            swapped_map = _effective(model)
        finally:
            emb.__class__ = original

        assert swapped_map == base_map, (
            "NEFTune subclass dropped a quantized module from the map: "
            f"missing {sorted(set(base_map) - set(swapped_map))}"
        )


@metal_only
def test_neftune_saved_map_reloads_against_base():
    """A map written while NEFTune is active must validate against the
    unmodified base (reproduces the reload ValueError before the fix)."""
    from unsloth_zoo.mlx.loader import (
        FastMLXModel,
        _effective_mlx_quantization_map,
        _validate_mlx_adapter_base,
    )
    from unsloth_zoo.mlx.utils import _get_text_model

    model, _ = FastMLXModel.from_pretrained(MODEL, max_seq_length=128)
    emb = _get_text_model(model).model.embed_tokens

    original = _neftune_swap(emb)
    try:
        saved_map = _effective_mlx_quantization_map(model)
    finally:
        emb.__class__ = original

    # Reload validates the saved map against the now-unmodified base model.
    adapter_cfg = {"base_resolved_quantization_map": saved_map}
    _validate_mlx_adapter_base(model, adapter_cfg)  # must not raise


@metal_only
def test_neftune_subclass_is_name_transparent():
    """The real _install_neftune subclass must report the base class name, so
    every save-time name-based check sees through the transparent stand-in."""
    import mlx.nn as nn
    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig
    from unsloth_zoo.mlx.utils import _get_text_model

    model, tok = FastMLXModel.from_pretrained(MODEL, max_seq_length=128)
    emb = _get_text_model(model).model.embed_tokens
    base_name = type(emb).__name__  # QuantizedEmbedding

    tr = MLXTrainer(
        model=model, tokenizer=tok, train_dataset=[{"text": "x"}],
        args=MLXTrainingConfig(neftune_noise_alpha=5.0,
                               output_dir="/tmp/neftune_name", report_to="none"),
    )
    tr._install_neftune()
    try:
        assert type(emb).__name__ == base_name, \
            f"subclass leaked its name into introspection: {type(emb).__name__}"
        assert isinstance(emb, (nn.QuantizedLinear, nn.QuantizedEmbedding))
    finally:
        tr._remove_neftune()
    assert type(emb).__name__ == base_name


@metal_only
def test_neftune_preserves_dora_name_detection():
    """A DoRA-adapted embed_tokens must still satisfy the save-time
    `type(module).__name__.startswith("DoRA")` check while NEFTune is active."""
    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig
    from unsloth_zoo.mlx.utils import _get_text_model

    model, tok = FastMLXModel.from_pretrained(MODEL, max_seq_length=128)
    emb = _get_text_model(model).model.embed_tokens
    real_base = type(emb)
    # A DoRA-named class exercises the name check. (Real DoRA wraps via
    # composition instead; that topology is covered by the wrapped-embedding
    # test below.)
    emb.__class__ = type("DoRAEmbedding", (real_base,), {})
    try:
        tr = MLXTrainer(
            model=model, tokenizer=tok, train_dataset=[{"text": "x"}],
            args=MLXTrainingConfig(neftune_noise_alpha=5.0,
                                   output_dir="/tmp/neftune_dora", report_to="none"),
        )
        tr._install_neftune()
        try:
            assert type(emb).__name__.startswith("DoRA"), \
                f"NEFTune broke DoRA save-time detection: {type(emb).__name__}"
        finally:
            tr._remove_neftune()
        assert type(emb).__name__ == "DoRAEmbedding"
    finally:
        emb.__class__ = real_base


@metal_only
def test_lora_wrapped_embedding_map_key_and_reload():
    """A real LoRA/DoRA-wrapped quantized embedding (what target_modules
    including embed_tokens produces) must keep its canonical ``embed_tokens``
    key in both map scans, and a map saved from the wrapped model must
    validate against the unwrapped base. Before the fix the wrapper leaked
    the key as ``embed_tokens.embedding`` and reload failed with
    missing/unexpected quantized modules."""
    from mlx_lm.tuner.dora import DoRAEmbedding
    from mlx_lm.tuner.lora import LoRAEmbedding
    from unsloth_zoo.mlx import loader as mlx_loader
    from unsloth_zoo.mlx import utils as mlx_utils
    from unsloth_zoo.mlx.loader import FastMLXModel, _validate_mlx_adapter_base
    from unsloth_zoo.mlx.utils import _get_text_model

    for wrapper_cls in (LoRAEmbedding, DoRAEmbedding):
        model, _ = FastMLXModel.from_pretrained(MODEL, max_seq_length=128)
        inner = _get_text_model(model).model
        base_map = mlx_loader._effective_mlx_quantization_map(model)
        assert "model.embed_tokens" in base_map, sorted(base_map)

        inner.embed_tokens = wrapper_cls.from_base(inner.embed_tokens, r=8)
        for _effective in (
            mlx_utils._effective_mlx_quantization_map,
            mlx_loader._effective_mlx_quantization_map,
        ):
            wrapped_map = _effective(model)
            assert wrapped_map == base_map, (
                f"{wrapper_cls.__name__} changed the map: "
                f"{sorted(set(wrapped_map) ^ set(base_map))}"
            )

        # Reload validates the saved-while-wrapped map against a fresh base.
        fresh, _ = FastMLXModel.from_pretrained(MODEL, max_seq_length=128)
        _validate_mlx_adapter_base(
            fresh, {"base_resolved_quantization_map": wrapped_map}
        )  # must not raise
