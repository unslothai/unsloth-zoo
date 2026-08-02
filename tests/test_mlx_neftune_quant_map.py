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


# --- VLM NEFTune: which module the noise lands on -------------------------

VLM_MODEL = "mlx-community/SmolVLM-256M-Instruct-4bit"
H, V = 4, 32


@pytest.fixture(scope="module")
def vlm():
    from unsloth_zoo.mlx.loader import FastMLXModel
    return FastMLXModel.from_pretrained(VLM_MODEL, max_seq_length=128)[0]


def _install(model, alpha=5.0):
    """Drive the real installer without building a whole trainer."""
    from types import SimpleNamespace
    from unsloth_zoo.mlx.trainer import MLXTrainer
    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.model, trainer._is_vlm = model, True
    trainer.args = SimpleNamespace(neftune_noise_alpha=alpha)
    trainer._install_neftune()
    return trainer


def _tree(**o):
    """A wrapper whose embed path can be given each awkward topology."""
    import mlx.core as mx, mlx.nn as nn

    class Adapter(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.embedding = inner
        def __call__(self, x):                  # zero delta, as at LoRA init
            return self.embedding(x) + mx.zeros((1, 3, H))

    class Wrapper(nn.Module):
        def get_input_embeddings(self, input_ids, pixel_values=None):
            if o.get("raises"):
                raise RuntimeError("this path needs pixel values")
            bb = self.language_model.model
            if o.get("renamed"):
                return bb.shared(input_ids)
            if o.get("decoy"):
                bb.decoy(input_ids)
            h = bb.embed_tokens(input_ids)
            if o.get("per_layer"):
                bb.embed_tokens_per_layer(input_ids)
            if o.get("sibling"):
                h = bb.embed_norm(h)            # same shape, encloses nothing
            return h

    model = Wrapper()
    model.language_model = nn.Module()
    bb = model.language_model.model = nn.Module()
    bb.embed_tokens, bb.lm_head = nn.Embedding(V, H), nn.Linear(H, V)
    if o.get("per_layer"):
        bb.embed_tokens_per_layer = nn.Embedding(V, H * 3)
    if o.get("sibling"):
        bb.embed_norm = nn.RMSNorm(H)
    if o.get("decoy"):
        bb.decoy = nn.Embedding(V, H * 2)
    if o.get("adapter"):
        bb.embed_tokens = Adapter(bb.embed_tokens)
    if o.get("renamed"):
        # florence2 calls its embedding `shared`; nothing may key on the name
        bb.shared = bb.embed_tokens
        del bb.embed_tokens
    if o.get("alias"):
        bb.second_view = bb.embed_tokens
    model._adapter_cls = Adapter
    return model


@metal_only
def test_vlm_neftune_lands_on_the_module_the_embed_path_uses(vlm):
    """A quantized embedding behind an untied head of identical weight shape."""
    import mlx.core as mx
    from unsloth_zoo.mlx.utils import _probe_vlm_embedding_module

    selected = _probe_vlm_embedding_module(vlm)
    named = dict(vlm.named_modules())
    assert named.get("language_model.embed_tokens") is selected
    assert "Quantized" in type(selected).__name__, type(selected).__name__
    head = named["language_model.lm_head"]
    base_cls, alpha = type(selected), 5.0
    ids = mx.array([list(range(8))], dtype=mx.int32)
    selected._set_training_mode(False)
    before = selected(ids).astype(mx.float32)
    head_before = head(before)
    trainer = _install(vlm, alpha=alpha)
    try:
        assert trainer._neftune_emb is selected
        assert type(selected) is not base_cls
        clean = selected(ids).astype(mx.float32)
        # Repeats prove determinism, not cleanliness; match pre-install.
        assert mx.array_equal(clean, before), "eval output must be untouched"
        assert mx.array_equal(head(before), head_before), "the head was perturbed"
        selected._set_training_mode(True)
        noise = selected(ids).astype(mx.float32) - clean
        again = selected(ids).astype(mx.float32) - clean
    finally:
        trainer._remove_neftune()
    assert type(selected) is base_cls

    # Eval gave the clean reference, so a training draw is the noise itself.
    scale = alpha / ((clean.shape[-1] * clean.shape[-2]) ** 0.5)
    assert mx.max(mx.abs(noise)).item() <= scale * 1.02
    rms = mx.sqrt(mx.mean(noise * noise)).item()
    assert rms == pytest.approx(scale / (3 ** 0.5), rel=0.15), rms
    # A clean row survives the aggregate, and a constant offset survives all of
    # it; require every row, redrawn per call.
    assert mx.min(mx.max(mx.abs(noise), axis=-1)).item() > 0
    assert not mx.array_equal(noise, again), "noise is redrawn per call"
    distinct = len({round(v, 6) for v in noise.flatten().tolist()})
    assert distinct > 64, f"a real draw is not {distinct} repeated values"


@metal_only
@pytest.mark.parametrize("opts,expected", [
    ({"per_layer": True}, "embed_tokens"),  # gemma4: on the path, other width
    ({"decoy": True}, "embed_tokens"),      # completes first: order cannot decide
    ({"adapter": True}, "adapter"),         # zero delta: values cannot decide
    ({"sibling": True}, None),              # inkling: neither encloses
    ({"renamed": True}, "shared"),          # florence2: not called embed_tokens
])
def test_probe_picks_the_embedding_or_declines(opts, expected):
    from unsloth_zoo.mlx.utils import _identify_vlm_embedding_module
    model = _tree(**opts)
    found = _identify_vlm_embedding_module(model)
    if expected is None:
        assert found is None
    elif expected == "adapter":
        assert isinstance(found, model._adapter_cls)
        assert found is not found.embedding
    else:
        assert found is getattr(model.language_model.model, expected)


@metal_only
@pytest.mark.parametrize("opts,resolves", [
    ({}, True), ({"sibling": True}, False), ({"raises": True}, False),
    # florence2 hands one embedding to encoder and decoder alike
    ({"alias": True}, True),
])
def test_probe_leaves_no_trace(opts, resolves):
    """Cleanup has to hold on the declining and raising paths too."""
    import mlx.core as mx, mlx.nn as nn
    from unsloth_zoo.mlx.utils import _probe_vlm_embedding_module
    class Noisy(nn.Module):
        """Draws in any mode, and counts mode flips, so neither check is
        vacuous."""
        def __init__(self, inner):
            super().__init__()
            self.inner, self.flips = inner, 0
        def _set_training_mode(self, mode):
            super()._set_training_mode(mode)
            self.flips += 1
        def __call__(self, x):
            return self.inner(x) + mx.random.uniform(-1e-6, 1e-6, (1, 3, H))

    model = _tree(**opts)
    bb = model.language_model.model
    aliased = bb.embed_tokens
    bb.embed_tokens = Noisy(aliased)
    assert not opts.get("alias") or sum(
        1 for _, m in model.named_modules() if m is aliased) > 1
    bb.lm_head._set_training_mode(False)
    flags = {p: m.training for p, m in model.named_modules()}
    classes = {p: type(m) for p, m in model.named_modules()}
    assert len(set(flags.values())) > 1, "a uniform tree hides a root-only restore"
    keys = [mx.array(k) for k in mx.random.state]
    want = mx.random.uniform(shape=(3,))
    mx.eval(want)
    mx.random.state[:] = keys
    assert (_probe_vlm_embedding_module(model) is not None) is resolves
    assert {p: m.training for p, m in model.named_modules()} == flags
    assert {p: type(m) for p, m in model.named_modules()} == classes
    got = mx.random.uniform(shape=(3,))
    mx.eval(got)
    assert mx.array_equal(want, got), "probe consumed the caller's stream"
    assert bb.embed_tokens.flips == 0, "a mode flip requantizes real layers"


@metal_only
def test_neftune_declines_for_value_comparing_families():
    """gemma3n finds merged positions by comparing embeddings; noise breaks it."""
    from types import SimpleNamespace
    from unsloth_zoo.mlx import utils as mlx_utils
    model = _tree()
    model.language_model.model.config = SimpleNamespace(
        model_type=sorted(mlx_utils._VLM_EMBED_SCALE_FAMILIES)[0], hidden_size=H)
    assert mlx_utils._vlm_compares_embedding_values(model)
    assert getattr(_install(model), "_neftune_emb", None) is None
