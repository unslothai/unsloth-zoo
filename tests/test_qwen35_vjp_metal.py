"""Metal-only regression tests for unsloth-zoo PR 738.

PR 738 fixes the second crash in unslothai/unsloth#6002:
``ValueError: [Primitive::vjp] Not implemented for CustomKernel`` raised during
MLX LoRA training of qwen3_5-family VLMs on Apple Silicon. The non-differentiable
paths are the GatedDeltaNet custom Metal kernel (``gated_delta_kernel``) and the
fused MRoPE Metal kernel (``MRoPERotaryEmbedding.apply_rotary`` fused path).

These bugs ONLY manifest on Metal (``mx.metal.is_available()`` and the default
device is the GPU). On CPU, mlx-vlm already falls back to differentiable ops, so
every test here is skipped on non-Metal machines with a loud notice.

Run on an Apple Silicon machine (CI installs unsloth-zoo with ``pip install -e .``):

    pytest tests/test_pr738_qwen35_vjp_metal.py -v

Target runtime: well under ~2 minutes on an M1.
"""

import pytest

try:
    import mlx.core as mx
    import mlx.nn as nn
    _HAS_METAL = mx.metal.is_available() and mx.default_device() == mx.gpu
except Exception:
    _HAS_METAL = False
_SKIP_REASON = (
    "Requires Apple Silicon Metal GPU (mx.metal.is_available() and default "
    "device == gpu); the qwen3_5 VJP crash is Metal-kernel-specific and cannot "
    "be reproduced on the CPU backend."
)

if not _HAS_METAL:
    print(
        "\n[test_pr738_qwen35_vjp_metal] SKIPPING ALL TESTS: no Metal GPU "
        "detected. These tests reproduce a Metal-only [Primitive::vjp] "
        "CustomKernel crash and only run on Apple Silicon.\n"
    )

metal_only = pytest.mark.skipif(not _HAS_METAL, reason=_SKIP_REASON)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _tiny_text_config(full_attention_interval=2):
    """Tiny qwen3_5 TextConfig.

    full_attention_interval=2 -> layer 0 is linear (GatedDeltaNet), layer 1 is
    full attention; exercises both the GDN VJP fix and the MRoPE fix in one
    model. head_dim 128 * partial_rotary_factor 0.25 = 32 = sum([11, 11, 10]).
    """
    from mlx_vlm.models.qwen3_5.config import TextConfig

    return TextConfig(
        model_type="qwen3_5",
        hidden_size=64,
        intermediate_size=128,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=1,
        max_position_embeddings=512,
        head_dim=128,
        full_attention_interval=full_attention_interval,
    )


def _flatten_grads(grads):
    flat = []

    def collect(node):
        if isinstance(node, mx.array):
            flat.append(node)
        elif isinstance(node, dict):
            for v in node.values():
                collect(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                collect(v)

    collect(grads)
    return flat


def _gdn_inputs(B=2, T=6, Hk=1, Hv=2, Dk=128, Dv=128, seed=0):
    """Inputs matching mlx-vlm gated_delta_update; Dk=128 so the Metal kernel
    (which requires Dk a multiple of 32) is actually exercised."""
    mx.random.seed(seed)
    dt = mx.bfloat16
    q = mx.random.normal((B, T, Hk, Dk)).astype(dt)
    k = mx.random.normal((B, T, Hk, Dk)).astype(dt)
    v = mx.random.normal((B, T, Hv, Dv)).astype(dt)
    a = mx.random.normal((B, T, Hv))
    b = mx.random.normal((B, T, Hv))
    A_log = mx.random.normal((Hv,))
    dt_bias = mx.random.normal((Hv,))
    mx.eval(q, k, v, a, b, A_log, dt_bias)
    return q, k, v, a, b, A_log, dt_bias


# --------------------------------------------------------------------------- #
# (a) GDN: unpatched grad through the Metal kernel raises the VJP ValueError
# --------------------------------------------------------------------------- #
@metal_only
def test_gated_delta_kernel_grad_raises_without_patch():
    """Proves the bug exists: differentiating the Metal gated_delta_kernel
    (use_kernel=True, state=None) raises [Primitive::vjp] CustomKernel."""
    import importlib

    import mlx_vlm.models.qwen3_5.gated_delta as vlm_gd

    importlib.reload(vlm_gd)  # ensure pristine (unpatched) module
    assert not getattr(vlm_gd, "_unsloth_gated_delta_patched", False)

    q, k, v, a, b, A_log, dt_bias = _gdn_inputs()

    def loss(q_, k_, v_):
        out = vlm_gd.gated_delta_update(
            q_, k_, v_, a, b, A_log, dt_bias,
            state=None, mask=None, use_kernel=True,
        )
        # The raw Metal kernel path returns a list of outputs; ops paths
        # return a tuple or a bare array.
        y = out[0] if isinstance(out, (tuple, list)) else out
        return y.astype(mx.float32).sum()

    with pytest.raises(ValueError) as exc:
        val, _ = mx.value_and_grad(loss, argnums=(0, 1, 2))(q, k, v)
        mx.eval(val)
    assert "vjp" in str(exc.value).lower() or "CustomKernel" in str(exc.value), exc.value


# --------------------------------------------------------------------------- #
# (b) GDN: patch fixes grad; output matches the use_kernel=False reference
# --------------------------------------------------------------------------- #
@metal_only
def test_patch_gated_delta_vlm_fixes_grad_and_matches_reference():
    import importlib

    import mlx_vlm.models.qwen3_5.gated_delta as vlm_gd
    from unsloth_zoo.gated_delta_vjp import patch_gated_delta_vlm

    importlib.reload(vlm_gd)
    # Reference forward (differentiable ops path) BEFORE patching.
    q, k, v, a, b, A_log, dt_bias = _gdn_inputs()
    ref_out, _ = vlm_gd.gated_delta_update(
        q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=False
    )
    mx.eval(ref_out)

    patch_gated_delta_vlm()
    assert getattr(vlm_gd, "_unsloth_gated_delta_patched", False)

    def loss(q_, k_, v_):
        out = vlm_gd.gated_delta_update(
            q_, k_, v_, a, b, A_log, dt_bias,
            state=None, mask=None, use_kernel=True,
        )
        y = out[0] if isinstance(out, tuple) else out
        return y.astype(mx.float32).sum()

    val, (dq, dk, dv) = mx.value_and_grad(loss, argnums=(0, 1, 2))(q, k, v)
    mx.eval(val, dq, dk, dv)

    for name, g in (("dq", dq), ("dk", dk), ("dv", dv)):
        assert bool(mx.all(mx.isfinite(g))), f"{name} non-finite after patch"
    assert any(float(mx.abs(g).max()) > 0 for g in (dq, dk, dv)), "all grads zero"

    pat_out, _ = vlm_gd.gated_delta_update(
        q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True
    )
    mx.eval(pat_out)
    assert mx.allclose(
        ref_out.astype(mx.float32), pat_out.astype(mx.float32), rtol=2e-2, atol=2e-2
    ), float(mx.abs(ref_out.astype(mx.float32) - pat_out.astype(mx.float32)).max())


# --------------------------------------------------------------------------- #
# (c) MRoPE: fused apply is non-differentiable; flip makes grad work; fused vs
#     fallback forward match
# --------------------------------------------------------------------------- #
@metal_only
def test_disable_fused_mrope_fixes_rotary_grad():
    import mlx_vlm.models.qwen3_5.language as qlang
    from unsloth_zoo.mlx.loader import _disable_fused_mrope

    cfg = _tiny_text_config(full_attention_interval=1)  # all attention layers
    model = qlang.Qwen3_5Model(cfg)
    mx.eval(model.parameters())

    rotaries = [
        layer.self_attn.rotary_emb for layer in model.layers if not layer.is_linear
    ]
    assert rotaries, "no rotary modules built"
    # On Metal the fused kernel path is active.
    assert all(r.fused_apply for r in rotaries), "expected fused_apply True on Metal"

    head_dim = cfg.head_dim
    B, H, L = 1, cfg.num_attention_heads, 4
    q = mx.random.normal((B, H, L, head_dim)).astype(mx.bfloat16)
    k = mx.random.normal((B, 1, L, head_dim)).astype(mx.bfloat16)
    pos = mx.tile(mx.expand_dims(mx.arange(L), 0)[None], (3, 1, 1))
    rot = rotaries[0]

    # Fused forward (reference output) before flipping.
    fused_q, fused_k = rot.apply_rotary(q, k, pos, unsqueeze_dim=1)
    mx.eval(fused_q, fused_k)

    # Pre-flip: grad through fused apply raises the VJP error.
    def loss(q_, k_):
        oq, ok = rot.apply_rotary(q_, k_, pos, unsqueeze_dim=1)
        return (oq.astype(mx.float32).sum() + ok.astype(mx.float32).sum())

    with pytest.raises(ValueError) as exc:
        val, _ = mx.value_and_grad(loss, argnums=(0, 1))(q, k)
        mx.eval(val)
    assert "vjp" in str(exc.value).lower() or "CustomKernel" in str(exc.value), exc.value

    # Apply the fix.
    _disable_fused_mrope(model)
    assert not any(r.fused_apply for r in rotaries), "fused_apply still True after fix"

    val, (dq, dk) = mx.value_and_grad(loss, argnums=(0, 1))(q, k)
    mx.eval(val, dq, dk)
    assert bool(mx.all(mx.isfinite(dq))) and bool(mx.all(mx.isfinite(dk)))
    assert float(mx.abs(dq).max()) > 0

    # Fallback forward should match the fused forward.
    fb_q, fb_k = rot.apply_rotary(q, k, pos, unsqueeze_dim=1)
    mx.eval(fb_q, fb_k)
    assert mx.allclose(
        fused_q.astype(mx.float32), fb_q.astype(mx.float32), rtol=2e-2, atol=2e-2
    ), float(mx.abs(fused_q.astype(mx.float32) - fb_q.astype(mx.float32)).max())
    assert mx.allclose(
        fused_k.astype(mx.float32), fb_k.astype(mx.float32), rtol=2e-2, atol=2e-2
    )


# --------------------------------------------------------------------------- #
# (d) End-to-end: one value_and_grad step on a GDN + attention model
# --------------------------------------------------------------------------- #
@metal_only
def test_end_to_end_training_step_all_patches():
    import mlx_vlm.models.qwen3_5.language as qlang
    from unsloth_zoo.gated_delta_vjp import patch_gated_delta, patch_gated_delta_vlm
    from unsloth_zoo.mlx.loader import _disable_fused_mrope, _fix_qwen35_attention_cache

    cfg = _tiny_text_config(full_attention_interval=2)  # 1 GDN + 1 attention layer
    model = qlang.Qwen3_5Model(cfg)
    mx.eval(model.parameters())
    # The model must report training so use_kernel=not self.training picks the
    # ops path on the inference branch; the patch handles state=None regardless.
    model.train()

    # Apply the full PR 738 patch set, exactly as trainer.py does.
    _fix_qwen35_attention_cache(model)
    _disable_fused_mrope(model)
    patch_gated_delta()
    patch_gated_delta_vlm()

    inputs = mx.array([[1, 2, 3, 4, 5, 6]])

    def loss_fn(m):
        out = m(inputs)
        return out.astype(mx.float32).sum()

    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    mx.eval(loss, grads)

    assert bool(mx.isfinite(loss)), "non-finite loss"
    flat = _flatten_grads(grads)
    assert flat, "no gradients produced"
    assert all(bool(mx.all(mx.isfinite(g))) for g in flat), "non-finite grads"
    assert any(float(mx.abs(g).max()) > 0 for g in flat), "all grads zero"
