# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Exercise real Gemma4 and Unified text modules on CPU, without checkpoints."""

import inspect

import pytest
import torch

from unsloth_zoo.temporary_patches import gemma4_float32 as patches
from unsloth_zoo.temporary_patches import utils as patch_utils


def test_older_transformers_without_unified_keeps_gemma4(monkeypatch):
    pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
    original_import = patches.importlib.import_module

    def import_without_unified(name, package=None):
        if name == "transformers.models.gemma4_unified.modeling_gemma4_unified":
            raise ModuleNotFoundError(name)
        return original_import(name, package)

    monkeypatch.setattr(patches.importlib, "import_module", import_without_unified)
    assert [prefix for _, prefix in patches._gemma4_text_variants()] == ["Gemma4"]


@pytest.fixture(params=[("gemma4", "Gemma4"), ("gemma4_unified", "Gemma4Unified")])
def variant(request, monkeypatch):
    name, prefix = request.param
    module = pytest.importorskip(f"transformers.models.{name}.modeling_{name}")
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    monkeypatch.setattr(patch_utils, "UNSLOTH_COMPILE_DISABLE", True)
    for patched_module, patched_prefix in patches._gemma4_text_variants():
        for suffix in ("TextScaledWordEmbedding", "RMSNorm", "TextAttention"):
            cls = getattr(patched_module, patched_prefix + suffix)
            monkeypatch.setattr(cls, "forward", cls.forward)
    for name in ("_gemma4_rms_norm_scaled", "_gemma4_rms_norm_unscaled"):
        fn = getattr(patches, name)
        while hasattr(fn, "__wrapped__"):
            fn = fn.__wrapped__
        monkeypatch.setattr(patches, name, fn)
    return module, prefix


def _install(module, prefix):
    for suffix, patcher in (
        ("TextScaledWordEmbedding", patches.patch_Gemma4TextScaledWordEmbedding),
        ("RMSNorm", patches.patch_Gemma4RMSNorm),
        ("TextAttention", patches.patch_Gemma4TextAttention),
    ):
        cls = getattr(module, prefix + suffix)
        before = cls.forward
        patcher()
        assert cls.forward is not before, f"{cls.__name__} was not patched"


def _config(module, prefix, shared=False):
    config = getattr(module, prefix + "TextConfig")(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, global_head_dim=8, num_global_key_value_heads=1,
        layer_types=["sliding_attention", "full_attention"] * 2,
        sliding_window=2, attention_k_eq_v=True,
        num_kv_shared_layers=2 if shared else 0,
    )
    config._attn_implementation = "sdpa"
    return config


def _positions():
    angles = torch.randn(1, 3, 8, dtype=torch.float32)
    return angles.cos(), angles.sin()


def _mask(sliding):
    positions = torch.arange(3)
    allowed = positions[:, None] >= positions[None, :]
    if sliding:
        allowed &= positions[:, None] - positions[None, :] < 2
    return allowed[None, None]


def test_float32_rope_reproduces_unpatched_dtype_failure(variant):
    module, prefix = variant
    attention = getattr(module, prefix + "TextAttention")(_config(module, prefix), 0).to(torch.bfloat16)
    inputs = torch.randn(1, 3, 16, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="same dtype"):
        attention(inputs, _positions(), _mask(True), shared_kv_states={})


@pytest.mark.parametrize("layer", [0, 1])
def test_patched_attention_matches_fp32_reference_and_backward(variant, monkeypatch, layer):
    module, prefix = variant
    seen = []
    sdpa = torch.nn.functional.scaled_dot_product_attention

    def capture(q, k, v, **kwargs):
        seen.append((q.dtype, k.dtype, v.dtype))
        return sdpa(q, k, v, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", capture)
    _install(module, prefix)
    attention = getattr(module, prefix + "TextAttention")(_config(module, prefix), layer).half()
    inputs = torch.randn(1, 3, 16, dtype=torch.float16, requires_grad=True)
    cos, sin = _positions()
    mask = _mask(layer == 0)
    out, _ = attention(inputs, (cos, sin), mask, shared_kv_states={})
    assert seen == [(torch.float32,) * 3]
    assert (attention.v_proj is None) == (layer == 1)

    shape = (1, 3, -1, 8)
    q = attention.q_norm(attention.q_proj(inputs).view(shape)).float()
    raw_k = attention.k_proj(inputs).view(shape)
    raw_v = attention.v_proj(inputs).view(shape) if attention.v_proj is not None else raw_k
    k = attention.k_norm(raw_k).float()
    v = attention.v_norm(raw_v).float().transpose(1, 2)
    q = module.apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2).transpose(1, 2)
    k = module.apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=2).transpose(1, 2)
    ref = sdpa(q, k, v, attn_mask=mask, scale=1.0, enable_gqa=True)
    ref = attention.o_proj(ref.transpose(1, 2).reshape(1, 3, 16).half())
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    out.float().square().mean().backward()
    assert torch.isfinite(out).all() and torch.isfinite(inputs.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in attention.parameters())


def test_embedding_and_norm_keep_residual_finite(variant):
    module, prefix = variant
    _install(module, prefix)
    embedding = getattr(module, prefix + "TextScaledWordEmbedding")(32, 16, 0, embed_scale=4).half()
    with torch.no_grad():
        embedding.weight.fill_(20000)
    hidden = embedding(torch.tensor([[1, 2, 3]]))
    assert hidden.dtype == torch.float32
    assert torch.isfinite(hidden).all() and hidden.max() > torch.finfo(torch.float16).max
    for scaled in (True, False):
        norm = getattr(module, prefix + "RMSNorm")(16, with_scale=scaled).half()
        out = norm(hidden)
        assert out.dtype == torch.float16 and torch.isfinite(out).all()
        out.float().square().mean().backward(retain_graph=True)
    assert torch.isfinite(embedding.weight.grad).all()


def test_shared_kv_reuses_finite_producer_states(variant):
    module, prefix = variant
    _install(module, prefix)
    config = _config(module, prefix, shared=True)
    cls = getattr(module, prefix + "TextAttention")
    producer, consumer = cls(config, 0).half(), cls(config, 2).half()
    assert producer.store_full_length_kv and consumer.is_kv_shared_layer
    inputs = torch.randn(1, 3, 16, dtype=torch.float16, requires_grad=True)
    shared = {}
    positions = _positions()
    first, _ = producer(inputs, positions, _mask(True), shared_kv_states=shared)
    out, _ = consumer(inputs, positions, _mask(True), shared_kv_states=shared)
    assert all(t.dtype == torch.float32 for t in shared["sliding_attention"])
    (first.float().square().mean() + out.float().square().mean()).backward()
    assert torch.isfinite(inputs.grad).all()
    assert producer.k_proj.weight.grad is not None


def test_bfloat16_mode_leaves_original_forwards_unchanged(variant, monkeypatch):
    module, prefix = variant
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "0")
    classes = [getattr(module, prefix + suffix) for suffix in ("TextScaledWordEmbedding", "RMSNorm", "TextAttention")]
    before = [cls.forward for cls in classes]
    patches.patch_Gemma4TextScaledWordEmbedding()
    patches.patch_Gemma4RMSNorm()
    patches.patch_Gemma4TextAttention()
    assert [cls.forward for cls in classes] == before


def test_unified_image_batch_backward_with_checkpointing(variant):
    module, prefix = variant
    if prefix != "Gemma4Unified":
        pytest.skip("Unified uses an encoder-free vision embedder")
    from transformers.models.gemma4_unified.configuration_gemma4_unified import (
        Gemma4UnifiedConfig,
        Gemma4UnifiedVisionConfig,
    )
    _install(module, prefix)
    config = Gemma4UnifiedConfig(
        text_config=_config(module, prefix),
        vision_config=Gemma4UnifiedVisionConfig(
            patch_size=2, pooling_kernel_size=1, mm_embed_dim=16,
            mm_posemb_size=8, output_proj_dims=16,
        ),
        image_token_id=28, video_token_id=29, audio_token_id=30,
        boi_token_id=26, eoi_token_id=27,
    )
    config._attn_implementation = "sdpa"
    model = module.Gemma4UnifiedForConditionalGeneration(config).half().train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    tokens = torch.tensor([[3, 28, 28, 4, 5, 6]])
    loss = model(
        input_ids=tokens, labels=tokens, use_cache=False,
        pixel_values=torch.randn(1, 2, 12),
        image_position_ids=torch.tensor([[[0, 0], [1, 0]]]),
        mm_token_type_ids=(tokens == 28).long(),
    ).loss
    assert torch.isfinite(loss)
    loss.backward()
    grads = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
    assert any("patch_dense" in name for name in grads)
    assert any("q_proj" in name for name in grads)
    assert all(torch.isfinite(grad).all() for grad in grads.values())


def test_unified_vision_projection_preserves_values_above_fp16_range(monkeypatch):
    module = pytest.importorskip("transformers.models.gemma4_unified.modeling_gemma4_unified")
    from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedVisionConfig
    cls = module.Gemma4UnifiedVisionEmbedder
    original_forward = cls.forward
    monkeypatch.setattr(cls, "__init__", cls.__init__)
    monkeypatch.setattr(cls, "forward", cls.forward)
    monkeypatch.setattr(cls, "_unsloth_vision_fp32_patched", False, raising=False)
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    patches.patch_Gemma4UnifiedVisionEmbedder()
    # Unsloth's compiler picks what to compile from this source; it must stay upstream's.
    assert inspect.getsource(cls.forward) == inspect.getsource(original_forward)
    vision_config = Gemma4UnifiedVisionConfig(patch_size=2, pooling_kernel_size=1,
        mm_embed_dim=16, mm_posemb_size=8, output_proj_dims=16)
    model = cls(vision_config, _config(module, "Gemma4Unified")).half()
    for child in model.modules():
        if hasattr(child, "_pre_set_compute_dtype"):
            child.to(child._pre_set_compute_dtype)
    with torch.no_grad():
        model.patch_ln1.weight.fill_(500)
        model.patch_dense.weight.zero_()
        model.patch_dense.weight[:, 0] = torch.linspace(1000, 2000, 16)
    pixels = torch.arange(24).reshape(1, 2, 12).float()
    positions = torch.tensor([[[0, 0], [1, 0]]])
    projected = []
    handle = model.patch_dense.register_forward_hook(lambda m, a, out: projected.append(out))
    # Base norm keeps input dtype unless forced-fp32 patches are installed.
    model.multimodal_embedder.float()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(pixels, positions)
    handle.remove()
    # Newer transformers return BaseModelOutputWithPooling (and accept return_dict), older a tensor.
    out = getattr(out, "pooler_output", out)
    if "return_dict" in inspect.signature(original_forward).parameters:
        assert isinstance(model(pixels, positions, return_dict=False), tuple)
    assert projected[0].dtype == torch.float32
    assert projected[0].abs().max() > torch.finfo(torch.float16).max
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    assert torch.isfinite(model.patch_dense.weight.grad).all()
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "0")
    ordinary = cls(vision_config, _config(module, "Gemma4Unified"))
    assert not ordinary._unsloth_vision_fp32
    assert not hasattr(ordinary.patch_dense, "_pre_set_compute_dtype")
