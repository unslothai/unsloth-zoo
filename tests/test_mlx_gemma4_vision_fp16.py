# SPDX-License-Identifier: AGPL-3.0-only
"""Gemma 4 vision pooling in fp16 (the default on Apple M1/M2): the sqrt(hidden_size) scale passes
65,504 on towers without clipped linears, and the inf becomes a NaN soft token, so the reply is the
last vocabulary id over and over (unsloth#15473). HF computes that step in fp32; so must we."""

import pytest

mx = pytest.importorskip("mlx.core")
from mlx_simulation import mlx_is_simulated

if mlx_is_simulated():
    pytest.skip("Requires native MLX", allow_module_level = True)
gemma4 = pytest.importorskip("mlx_vlm.models.gemma4.gemma4")

from mlx.utils import tree_map
from mlx_vlm.models.gemma4 import vision
from mlx_vlm.models.gemma4.config import ModelConfig, TextConfig, VisionConfig
from unsloth_zoo.mlx.loader import _convert_mlx_dtype, _fix_gemma4_vision_pooler_fp32

# trl-internal-testing/tiny-Gemma4ForConditionalGeneration, vocabulary trimmed, standardize on.
_VISION = {
    "model_type": "gemma4_vision", "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2,
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64, "global_head_dim": 64,
    "patch_size": 16, "pooling_kernel_size": 3, "position_embedding_size": 630, "default_output_length": 70,
    "rms_norm_eps": 1e-6, "rope_parameters": {"rope_theta": 100.0, "rope_type": "default"},
    "standardize": True, "use_clipped_linears": False,
}
_TEXT = {
    "model_type": "gemma4_text", "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2,
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64, "global_head_dim": 64,
    "hidden_size_per_layer_input": 16, "vocab_size": 1024, "vocab_size_per_layer_input": 1024,
    "layer_types": ["sliding_attention", "full_attention"], "num_kv_shared_layers": 0,
}
_IMAGE_TOKEN = 1000
_PIXELS = mx.ones((1, 3, 48, 48))
_IDS = mx.array([[2, _IMAGE_TOKEN, 10]])


def _model(dtype):
    config = ModelConfig.from_dict({
        "model_type": "gemma4", "image_token_id": _IMAGE_TOKEN, "audio_token_id": 1001,
        "video_token_id": 1002, "boi_token_id": 1003, "eoi_token_id": 1004,
        "text_config": _TEXT, "vision_config": _VISION, "audio_config": None,
    })
    config.text_config = TextConfig.from_dict(_TEXT)
    config.vision_config = VisionConfig.from_dict(_VISION)
    mx.random.seed(3407)
    model = gemma4.Model(config)
    tower = model.vision_tower
    # 768 patch values x 26 embed near 2e4: pooled x sqrt(16) reaches 8e4, as the 26B tower's
    # ~2.5e3 x sqrt(1152) does. std_scale brings it back to the real tower's single digits.
    embedder = tower.patch_embedder
    embedder.input_proj.weight = mx.full(embedder.input_proj.weight.shape, 26.0)
    tower.std_scale = mx.full(tower.std_scale.shape, 1e-4)
    model.update(tree_map(lambda v: v.astype(mx.bfloat16) if mx.issubdtype(v.dtype, mx.floating) else v,
                          model.parameters()))
    _convert_mlx_dtype(model, dtype, model_type = "gemma4")
    return model


def _embeds(model):
    out = model.get_input_embeddings(input_ids = _IDS, pixel_values = _PIXELS)
    return out.inputs_embeds if hasattr(out, "inputs_embeds") else out[0]


@pytest.fixture
def unpatched(monkeypatch):
    for cls in (vision.VisionPooler, vision.VisionModel):
        monkeypatch.setattr(cls, "__call__", getattr(cls.__call__, "__wrapped__", cls.__call__))


def test_setup_overflows_fp16_without_the_fix(unpatched):
    model = _model(mx.float16)
    assert not bool(mx.isfinite(_embeds(model)).all())


def test_fp16_soft_tokens_finite():
    assert _fix_gemma4_vision_pooler_fp32()
    model = _model(mx.float16)
    feats = model.vision_tower(_PIXELS.astype(mx.float16))
    embeds = _embeds(model)
    assert feats.dtype == mx.float16
    assert embeds.dtype == mx.float16
    assert bool(mx.isfinite(embeds).all())
    want = _embeds(_model(mx.float32))
    assert float(mx.abs(embeds.astype(mx.float32) - want).max()) < 5e-2 * (1 + float(mx.abs(want).max()))


def test_bf16_unchanged(monkeypatch):
    assert _fix_gemma4_vision_pooler_fp32()
    patched = _embeds(_model(mx.bfloat16))
    for cls in (vision.VisionPooler, vision.VisionModel):
        monkeypatch.setattr(cls, "__call__", cls.__call__.__wrapped__)
    stock = _embeds(_model(mx.bfloat16))
    assert patched.dtype == stock.dtype == mx.bfloat16
    assert bool(mx.array_equal(patched, stock))


def test_idempotent():
    assert _fix_gemma4_vision_pooler_fp32()
    pool = vision.VisionPooler.__call__
    assert _fix_gemma4_vision_pooler_fp32()
    assert vision.VisionPooler.__call__ is pool
    assert not hasattr(pool.__wrapped__, "__wrapped__")
