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
    _MLX = True
    _METAL = mx.metal.is_available()
except Exception:
    _MLX = _METAL = False

metal_only = pytest.mark.skipif(not _METAL, reason="requires Apple Silicon Metal")
# Identification, scale resolution and cleanup are pure logic over a synthetic
# tree: real mlx, no Metal, no downloads, so they gate the Linux job too.
mlx_only = pytest.mark.skipif(not _MLX, reason="requires the mlx runtime")

# mlx 0.32 made mx.random.state a sentinel refusing item assignment, so tests
# rewind through the same reseed pair the production code uses.
_RNG_STATE_WRITABLE = _MLX and isinstance(mx.random.state, list)

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
    """The one test here that needs a real 4-bit VLM, or a skip saying why not.

    Building it is a third-party download plus a transformers processor lookup,
    both outside anything NEFTune touches, and both able to fail on a runner for
    reasons this branch cannot fix: conftest redirects HF_HOME to a fresh tmp dir
    per session, and the mlx-community repo's processor has to be resolvable by
    the installed transformers. A skip names that; an error would report it as a
    NEFTune failure. Scoped to construction only, so a real regression in the
    probe still fails loudly below.
    """
    from unsloth_zoo.mlx.loader import FastMLXModel
    try:
        return FastMLXModel.from_pretrained(VLM_MODEL, max_seq_length=128)[0]
    except Exception as e:
        pytest.skip(f"could not build {VLM_MODEL} on this runner: {type(e).__name__}: {e}")


def _install(model, alpha=5.0, is_vlm=None):
    """Drive the real installer without building a whole trainer."""
    from types import SimpleNamespace
    from unsloth_zoo.mlx.trainer import MLXTrainer
    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.model = model
    trainer._is_vlm = hasattr(model, "language_model") if is_vlm is None else is_vlm
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


@mlx_only
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


@mlx_only
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
    from unsloth_zoo.mlx.utils import _mlx_rng_key, _restore_mlx_rng_key
    key = _mlx_rng_key()
    assert key is not None, "this mlx exposes no rewindable key; the probe cannot be checked"
    want = mx.random.uniform(shape=(3,))
    mx.eval(want)
    _restore_mlx_rng_key(key)
    assert (_probe_vlm_embedding_module(model) is not None) is resolves
    assert {p: m.training for p, m in model.named_modules()} == flags
    assert {p: type(m) for p, m in model.named_modules()} == classes
    got = mx.random.uniform(shape=(3,))
    mx.eval(got)
    assert mx.array_equal(want, got), "probe consumed the caller's stream"
    assert bb.embed_tokens.flips == 0, "a mode flip requantizes real layers"


@mlx_only
def test_neftune_declines_for_value_comparing_families():
    """gemma3n finds merged positions by comparing embeddings; noise breaks it."""
    from types import SimpleNamespace
    from unsloth_zoo.mlx import utils as mlx_utils
    model = _tree()
    model.language_model.model.config = SimpleNamespace(
        model_type=sorted(mlx_utils._VLM_EMBED_SCALE_FAMILIES)[0], hidden_size=H)
    assert mlx_utils._vlm_compares_embedding_values(model)
    assert getattr(_install(model), "_neftune_emb", None) is None


# --- NEFTune noise magnitude on families that scale after the embedding ----

GEMMA3_TEXT = "mlx-community/gemma-3-270m-it-4bit"


@mlx_only
@pytest.mark.parametrize("model_type,inside,scaled", [
    # transformers moved gemma and gemma2's multiply inside the embedding
    # partway through the supported version range, so the answer follows the
    # installed transformers rather than a fixed list.
    ("gemma", True, True),
    ("gemma", False, False),
    ("gemma2", True, True),
    ("gemma2", False, False),
    ("gemma3_text", True, True),
    ("gemma3n_text", True, True),
    # nothing multiplies after the embedding here, so there is nothing to cancel
    ("qwen3", True, False),
])
def test_embed_scale_follows_the_installed_transformers(monkeypatch, model_type, inside, scaled):
    from types import SimpleNamespace
    from unsloth_zoo.mlx import utils as mlx_utils

    monkeypatch.setattr(mlx_utils, "_transformers_scales_inside_embedding",
                        lambda _mt: inside)
    model = _tree()
    model.language_model.model.config = SimpleNamespace(
        model_type=model_type, hidden_size=H)
    found = mlx_utils._neftune_embed_scale(model)
    assert found == (pytest.approx(H ** 0.5) if scaled else None)


@mlx_only
def test_embed_scale_probe_reads_the_installed_transformers():
    """The probe must answer from the installed package, not a table."""
    import importlib.util
    import pathlib
    import sys
    from unittest import mock
    from unsloth_zoo.mlx.utils import _transformers_scales_inside_embedding as probe

    # gemma3 has scaled inside the embedding in every transformers that has it.
    assert probe("gemma3_text") is True
    # A plain-embedding architecture, and one transformers does not know at all.
    assert probe("qwen3") is False
    assert probe("not_a_real_architecture") is None
    # It must answer without importing: a modeling module needs torch, which is
    # not installed alongside MLX on macOS arm64.
    sys.modules.pop("transformers.models.gemma3.modeling_gemma3", None)
    probe.cache_clear()
    with mock.patch.dict(sys.modules, {"torch": None}):
        assert probe("gemma3_text") is True, "the probe needs torch importable"
    probe.cache_clear()

    # And it must go through the loader, so a package that is not loose files
    # on disk still answers rather than silently reporting "no such thing".
    spec = importlib.util.find_spec("transformers.models.gemma3.modeling_gemma3")
    assert spec.loader.get_source(spec.name).count("ScaledWordEmbedding") > 0

    # A package shipped without source still answers, from the compiled code.
    loader = type(spec.loader)
    with mock.patch.object(loader, "get_source", lambda self, name: None):
        assert probe("gemma3_text") is True, "source-less package must still answer"
    probe.cache_clear()
    # Only when neither is available is the question genuinely unanswerable.
    def _raise(self, name):
        raise OSError("no code")
    with mock.patch.object(loader, "get_source", lambda self, name: None), \
         mock.patch.object(loader, "get_code", _raise):
        assert probe("gemma3_text") is None
    probe.cache_clear()


@mlx_only
@pytest.mark.parametrize("model_type,scaled", [
    # transformers 4.x predates these, so the probe cannot answer; they have
    # scaled inside the embedding in every transformers that has them.
    ("gemma4_text", True),
    ("gemma4_unified_text", True),
    ("gemma", False),      # genuinely version-dependent, so not assumed
    # minicpm3 is unaskable for a different reason: no built-in arch in the
    # supported range, so the reference is trust_remote_code, which scales
    # OUTSIDE the embedding as mlx-lm does. See the dedicated test below.
    ("minicpm3", False),
])
def test_embed_scale_falls_back_when_transformers_cannot_be_asked(
        monkeypatch, model_type, scaled):
    from types import SimpleNamespace
    from unsloth_zoo.mlx import utils as mlx_utils

    monkeypatch.setattr(mlx_utils, "_transformers_scales_inside_embedding",
                        lambda _mt: None)
    model = _tree()
    backbone = model.language_model.model
    backbone.config = SimpleNamespace(model_type=model_type, hidden_size=H)
    backbone.embed_scale = H ** 0.5
    found = mlx_utils._neftune_embed_scale(model)
    assert found == (pytest.approx(H ** 0.5) if scaled else None)


@mlx_only
@pytest.mark.parametrize("key,value,expected", [
    ("scale_emb", 12.0, 12.0),      # minicpm3 scales by a config value, not a sqrt
    (None, None, H ** 0.5),
])
def test_embed_scale_reads_a_non_sqrt_factor(monkeypatch, key, value, expected):
    from types import SimpleNamespace
    from unsloth_zoo.mlx import utils as mlx_utils

    monkeypatch.setattr(mlx_utils, "_transformers_scales_inside_embedding",
                        lambda _mt: True)
    model = _tree()
    fields = {"model_type": "gemma3_text", "hidden_size": H}
    if key:
        fields[key] = value
    model.language_model.model.config = SimpleNamespace(**fields)
    assert mlx_utils._neftune_embed_scale(model) == pytest.approx(expected)


@mlx_only
def test_embed_scale_prefers_the_backbone_attribute():
    """gemma4 exposes the multiply it uses; a derived value could disagree."""
    from types import SimpleNamespace
    from unsloth_zoo.mlx.utils import _neftune_embed_scale

    model = _tree()
    backbone = model.language_model.model
    backbone.config = SimpleNamespace(model_type="gemma4_text", hidden_size=H)
    backbone.embed_scale = 7.5
    assert _neftune_embed_scale(model) == pytest.approx(7.5)


@metal_only
def test_neftune_noise_is_divided_by_a_post_embedding_scale():
    """On these families the model multiplies after the embedding returns, so
    the injected noise has to be smaller by that factor to land where
    transformers puts it."""
    import mlx.core as mx
    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.utils import _neftune_embed_scale

    model, _ = FastMLXModel.from_pretrained(GEMMA3_TEXT, max_seq_length=128)
    embed_scale = _neftune_embed_scale(model)
    assert embed_scale == pytest.approx(640 ** 0.5), embed_scale

    alpha, ids = 15.0, mx.array([list(range(8))], dtype=mx.int32)
    trainer = _install(model, alpha=alpha)
    emb = trainer._neftune_emb
    try:
        emb._set_training_mode(False)
        clean = emb(ids).astype(mx.float32)
        emb._set_training_mode(True)
        noise = emb(ids).astype(mx.float32) - clean
    finally:
        trainer._remove_neftune()
    uncorrected = alpha / ((clean.shape[-1] * clean.shape[-2]) ** 0.5) / (3 ** 0.5)
    rms = mx.sqrt(mx.mean(noise * noise)).item()
    assert rms == pytest.approx(uncorrected / embed_scale, rel=0.15), rms


@metal_only
def test_compiled_step_carries_the_rng_so_noise_is_redrawn():
    """A compiled step traces its body once. Unless the random state is one of
    the compiled inputs and outputs, every step replays a single draw and the
    noise becomes a fixed offset -- which is not NEFTune at all, and is
    invisible from the loss."""
    import mlx.core as mx
    from unittest import mock
    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    model, tok = FastMLXModel.from_pretrained(MODEL, max_seq_length=64)
    model = FastMLXModel.get_peft_model(model, r=4)
    captured, real_compile = [], mx.compile

    def _recording_compile(fn, *args, **kwargs):
        captured.append(kwargs.get("inputs"))
        return real_compile(fn, *args, **kwargs)

    trainer = MLXTrainer(
        model=model, tokenizer=tok,
        train_dataset=[{"text": f"row {i}"} for i in range(4)],
        args=MLXTrainingConfig(
            neftune_noise_alpha=5.0, max_steps=2, compile=True,
            per_device_train_batch_size=1, gradient_accumulation_steps=1,
            warmup_steps=0, output_dir="/tmp/neftune_compile", report_to="none",
        ),
    )
    with mock.patch.object(mx, "compile", _recording_compile):
        trainer.train()

    states = [s for s in captured if s is not None]
    assert states, "compile never ran, so this proves nothing"
    assert any(any(part is mx.random.state for part in state) for state in states), \
        "the random state is not compiled in; every step would replay one draw"


@mlx_only
@pytest.mark.parametrize("state", [(), None])
def test_probe_survives_a_backend_without_a_list_random_state(monkeypatch, state):
    """A stub backend need not present the random state as a mutable list.
    Identification still has to run: the snapshot is a courtesy, not a
    precondition."""
    import mlx.core as mx
    from unsloth_zoo.mlx.utils import _probe_vlm_embedding_module

    model = _tree()
    monkeypatch.setattr(mx.random, "state", state, raising=False)
    found = _probe_vlm_embedding_module(model)
    assert found is model.language_model.model.embed_tokens


@mlx_only
@pytest.mark.parametrize("family,embed_scale,expected", [
    ("gemma3", None, H ** 0.5),           # sqrt(hidden_size), computed inline
    ("gemma3n", None, H ** 0.5),
    ("gemma4_unified", 7.0, 7.0),         # scale carried by the backbone
])
def test_embed_scale_reads_both_spellings_of_a_family(
        monkeypatch, family, embed_scale, expected):
    """A tower arrives as either `<family>` or `<family>_text` depending on the
    wrapper, and the transformers lookup normalises before asking, so the family
    sets it falls back to have to answer the same for both."""
    from types import SimpleNamespace
    from unsloth_zoo.mlx import utils as mlx_utils

    # Force the fallback: this is the path that reads the family sets.
    monkeypatch.setattr(mlx_utils, "_transformers_scales_inside_embedding",
                        lambda _mt: None)
    found = []
    for model_type in (family, f"{family}_text"):
        model = _tree()
        backbone = model.language_model.model
        backbone.config = SimpleNamespace(model_type=model_type, hidden_size=H)
        if embed_scale is not None:
            backbone.embed_scale = embed_scale
        found.append(mlx_utils._neftune_embed_scale(model))
    assert found[0] == found[1] == pytest.approx(expected), found


@mlx_only
def test_probe_survives_the_backends_own_random_state():
    """The shapes above are hypothetical; this is the one that actually ships.

    Nothing is monkeypatched: identification and the rewind have to work against
    whatever random state this mlx really exposes.
    """
    import mlx.core as mx
    from unsloth_zoo.mlx.utils import _probe_vlm_embedding_module

    model = _tree()
    found = _probe_vlm_embedding_module(model)
    assert found is model.language_model.model.embed_tokens
    # And the sentinel must survive: rebinding it to a list shadows the object
    # mx.compile captured, which stops a compiled step redrawing.
    assert isinstance(mx.random.state, list) is _RNG_STATE_WRITABLE

    # The rewind must work on whatever this mlx exposes, not just the pre-0.32
    # list form.
    import mlx.nn as nn
    from unsloth_zoo.mlx.utils import _mlx_rng_key, _restore_mlx_rng_key

    class Noisy(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def __call__(self, x):
            return self.inner(x) + mx.random.uniform(-1e-6, 1e-6, (1, 3, H))

    model = _tree()
    backbone = model.language_model.model
    backbone.embed_tokens = Noisy(backbone.embed_tokens)
    key = _mlx_rng_key()
    assert key is not None
    want = mx.random.uniform(shape=(3,))
    mx.eval(want)
    _restore_mlx_rng_key(key)
    assert _probe_vlm_embedding_module(model) is backbone.embed_tokens
    got = mx.random.uniform(shape=(3,))
    mx.eval(got)
    assert mx.array_equal(want, got), \
        "the probe advanced the caller's stream on this mlx"


@mlx_only
def test_minicpm3_keeps_the_scale_transformers_actually_applies():
    """minicpm3 must not be corrected while transformers ships no built-in arch.

    Then the only reference is trust_remote_code, which multiplies by scale_emb
    outside the embedding just as mlx-lm does, so both sides already agree.
    Derived rather than hardcoded: a transformers carrying the scaled class
    flips this on its own.
    """
    import importlib.util
    from types import SimpleNamespace
    from unsloth_zoo.mlx import utils as mlx_utils

    builtin = None
    try:
        builtin = importlib.util.find_spec(
            "transformers.models.minicpm3.modeling_minicpm3")
    except Exception:
        builtin = None

    model = _tree()
    model.language_model.model.config = SimpleNamespace(
        model_type="minicpm3", hidden_size=H, scale_emb=12.0)
    got = mlx_utils._neftune_embed_scale(model)

    if builtin is None:
        assert got is None, (
            f"minicpm3 corrected by {got} with no built-in transformers arch; "
            f"that makes MLX {got}x weaker than the trust_remote_code reference")
    else:
        assert got == pytest.approx(12.0)


@mlx_only
def test_neftune_forwards_extra_call_arguments():
    """The probe can return a wrapper taking more than one argument; the swapped
    __call__ must forward them."""
    import mlx.core as mx
    import mlx.nn as nn
    from types import SimpleNamespace

    class MultiArg(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.embedding = inner
        def __call__(self, x, pixel_values=None, mask=None):
            return self.embedding(x) + mx.zeros((1, 3, H))

    model = _tree()
    backbone = model.language_model.model
    backbone.config = SimpleNamespace(model_type="llama", hidden_size=H)
    backbone.embed_tokens = MultiArg(backbone.embed_tokens)

    _install(model, alpha=5.0, is_vlm=True)
    emb = backbone.embed_tokens
    assert getattr(emb, "_unsloth_neftune_active", False), "probe declined"
    emb.train()
    out = emb(mx.array([[0, 1, 2]]), pixel_values=None, mask=None)
    mx.eval(out)
    assert out.shape == (1, 3, H)


@mlx_only
@pytest.mark.parametrize("model_type", ["gemma3n_text", "gemma3n"])
def test_refusal_covers_both_spellings_of_a_value_comparing_family(model_type):
    """The refusal must not depend on which spelling the wrapper exposes.

    `_vlm_embed_scale` compares the raw model type, so the bare name slips past
    it. A missed refusal injects noise into that comparison; a spurious one only
    trains un-noised, so the gate is conservative across both.
    """
    from types import SimpleNamespace
    from unsloth_zoo.mlx.utils import _vlm_compares_embedding_values

    model = _tree()
    model.language_model.model.config = SimpleNamespace(
        model_type=model_type, hidden_size=H)
    assert _vlm_compares_embedding_values(model) is True


@mlx_only
def test_neftune_noise_is_zero_mean():
    """RMS and max both survive a sign change, so neither pins the distribution.

    transformers draws U(-m, m); U(0, m) matches on both while shifting every
    dimension positively. Only the mean separates them.
    """
    import mlx.core as mx
    from types import SimpleNamespace

    model = _tree()
    backbone = model.language_model.model
    backbone.config = SimpleNamespace(model_type="llama", hidden_size=H)
    _install(model, alpha=15.0, is_vlm=False)
    emb = backbone.embed_tokens
    emb.train()

    ids = mx.array([[0, 1, 2]])
    base_call = type(emb).__mro__[1].__call__
    total = 0.0
    draws = 400
    for _ in range(draws):
        noise = emb(ids) - base_call(emb, ids)
        mx.eval(noise)
        total += float(mx.mean(noise).item())
    mean = total / draws
    # sd of the mean over draws*3*H samples of U(-m, m); 6 sigma is still tiny
    # next to the m/2 offset a one-sided draw would produce.
    m = 15.0 / ((3 * H) ** 0.5)
    tolerance = 6.0 * m / (3.0 ** 0.5) / ((draws * 3 * H) ** 0.5)
    assert abs(mean) < tolerance, (
        f"noise mean {mean} exceeds {tolerance}; a one-sided draw would sit "
        f"near {m / 2}")


@mlx_only
def test_embed_scale_failure_does_not_abort_training():
    """A config that cannot be reduced to a number degrades to uncorrected noise,
    like every other lookup in _install_neftune, rather than taking train() down."""
    import mlx.core as mx
    from types import SimpleNamespace

    class Hostile:
        @property
        def model_type(self):
            raise RuntimeError("this config does not want to be read")

    model = _tree()
    backbone = model.language_model.model
    backbone.config = Hostile()
    _install(model, alpha=5.0, is_vlm=False)
    emb = backbone.embed_tokens
    assert getattr(emb, "_unsloth_neftune_active", False), \
        "a hostile config must not prevent NEFTune from installing un-corrected"
    emb.train()
    mx.eval(emb(mx.array([[0, 1, 2]])))
