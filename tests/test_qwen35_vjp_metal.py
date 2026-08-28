"""Metal-only regression tests for unsloth-zoo PR 738.

PR 738 fixes the second crash in unslothai/unsloth#6002:
``ValueError: [Primitive::vjp] Not implemented for CustomKernel`` raised during
MLX LoRA training of qwen3_5-family VLMs on Apple Silicon. The non-differentiable
paths are the GatedDeltaNet custom Metal kernel (``gated_delta_kernel``) and the
fused MRoPE Metal kernel (``MRoPERotaryEmbedding.apply_rotary`` fused path).

These bugs ONLY manifest on Metal (``mx.metal.is_available()`` and the default
device is the GPU). On CPU, mlx-vlm already falls back to differentiable ops, so
every test here is skipped on non-Metal machines with a loud notice.

Target runtime: well under ~2 minutes on an M1.
"""

import pytest

mx = pytest.importorskip("mlx.core", reason="MLX is not installed.")
import mlx.nn as nn

try:
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
def test_patch_gated_delta_vlm_fixes_grad_and_matches_reference(monkeypatch):
    import importlib

    import mlx_vlm.models.qwen3_5.gated_delta as vlm_gd
    from unsloth_zoo.gated_delta_vjp import patch_gated_delta_vlm

    importlib.reload(vlm_gd)
    vlm_gd._unsloth_gated_delta_patched = False
    # Reference forward (differentiable ops path) BEFORE patching.
    q, k, v, a, b, A_log, dt_bias = _gdn_inputs()
    ref_out, _ = vlm_gd.gated_delta_update(
        q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=False
    )
    mx.eval(ref_out)

    patch_gated_delta_vlm()
    assert vlm_gd.gated_delta_update.__name__ == "patched_vlm_gated_delta_update"

    def loss(q_, k_, v_):
        out = vlm_gd.gated_delta_update(
            q_, k_, v_, a, b, A_log, dt_bias,
            state=None, mask=None, use_kernel=False,
        )
        y = out[0] if isinstance(out, tuple) else out
        return y.astype(mx.float32).sum()

    # An empty cache alone is prefill: the fused kernel returns a raw list.
    assert isinstance(vlm_gd.gated_delta_update(
        q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True
    ), list)

    # Upstream's fallbacks differentiate too, and which exist varies by version.
    for _n in ("gated_delta_ops", "gated_delta_chunked"):
        monkeypatch.setattr(vlm_gd, _n, lambda *a: pytest.fail("upstream"), raising=False)

    val, (dq, dk, dv) = mx.value_and_grad(loss, argnums=(0, 1, 2))(q, k, v)
    pat_out, _ = vlm_gd.gated_delta_update(
        q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=False
    )
    mx.eval(val, dq, dk, dv, pat_out)

    assert all(bool(mx.all(mx.isfinite(g))) for g in (dq, dk, dv)), "non-finite grads"
    assert any(float(mx.abs(g).max()) > 0 for g in (dq, dk, dv)), "all grads zero"
    assert mx.allclose(ref_out.astype(mx.float32), pat_out.astype(mx.float32),
                       rtol=2e-2, atol=2e-2)


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

    # Pre-flip: grad through fused apply raises the VJP error on mlx releases
    # whose fused rotary kernel has no VJP. mlx 0.32.0 added one, so only the
    # message is asserted, not that it raises at all.
    def loss(q_, k_):
        oq, ok = rot.apply_rotary(q_, k_, pos, unsqueeze_dim=1)
        return (oq.astype(mx.float32).sum() + ok.astype(mx.float32).sum())

    try:
        val, _ = mx.value_and_grad(loss, argnums=(0, 1))(q, k)
        mx.eval(val)
    except ValueError as exc:
        assert "vjp" in str(exc).lower() or "CustomKernel" in str(exc), exc

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
    # `use_kernel=not self.training` asks for the differentiable path.
    model.train()

    # Trainer order: patch_gated_delta's sweep warns about an unpatched copy.
    _fix_qwen35_attention_cache(model)
    _disable_fused_mrope(model)
    patch_gated_delta_vlm()
    patch_gated_delta()

    # Grads alone prove nothing: upstream's ops path differentiates too.
    assert qlang.gated_delta_update.__name__ == "patched_vlm_gated_delta_update"

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


# Two tolerances per output. The global one bounds error against the tensor's
# largest magnitude; the elementwise one bounds each output against its own
# magnitude, so a single badly-computed small entry cannot hide behind a large
# one. bf16 inputs quantize y to ~4e-3 either way; the fp32 state carries no
# such rounding and is held to fp32 accumulation error.
_Y_TOL = {mx.bfloat16: 5e-3, mx.float32: 5e-6}
_Y_ELEM_TOL = {mx.bfloat16: 8e-3, mx.float32: 2e-5}
_STATE_TOL = 1e-6
_STATE_ELEM_TOL = 2e-5


def _inputs(B, T, Hk, Hv, Dk, Dv, dtype, vectorized, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((B, T, Hk, Dk)).astype(dtype)
    k = mx.random.normal((B, T, Hk, Dk)).astype(dtype)
    k = k / mx.linalg.norm(k.astype(mx.float32), axis=-1, keepdims=True).astype(dtype)
    v = mx.random.normal((B, T, Hv, Dv)).astype(dtype)
    g_shape = (B, T, Hv, Dk) if vectorized else (B, T, Hv)
    g = mx.exp(-mx.random.uniform(shape=g_shape).astype(mx.float32) * 0.5)
    beta = mx.random.uniform(shape=(B, T, Hv)).astype(dtype)
    state = mx.random.normal((B, Hv, Dv, Dk)).astype(mx.float32) * 0.1
    if Hv > Hk:
        q, k = mx.repeat(q, Hv // Hk, -2), mx.repeat(k, Hv // Hk, -2)
    return q, k, v, g, beta, state


def _reference(q, k, v, g, beta, state):
    q, k, v = (x.astype(mx.float32) for x in (q, k, v))
    beta = beta.astype(mx.float32)
    s, ys = state, []
    for t in range(q.shape[1]):
        s = s * (g[:, t][..., None, None] if g.ndim == 3 else g[:, t][..., None, :])
        delta = (v[:, t] - (s * k[:, t][..., None, :]).sum(-1)) * beta[:, t][..., None]
        s = s + k[:, t][..., None, :] * delta[..., None]
        ys.append((s * q[:, t][..., None, :]).sum(-1))
    return mx.stack(ys, axis=1), s


def _rel(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b)) / mx.max(mx.abs(b)))


def _elem_rel(a, b, floor=0.01):
    """Worst deviation relative to each element's own magnitude.

    The denominator carries a floor of ``floor`` x the tensor's scale so that
    outputs that are near zero by construction do not divide by ~0.
    """
    d = mx.abs(a.astype(mx.float32) - b)
    return float(mx.max(d / (mx.abs(b) + floor * mx.max(mx.abs(b)))))


# (B, T, Hk, Hv, Dk, Dv, dtype, vectorized) -- one case per P the layout can
# derive (2, 4, 8, 16, 32, so every step of the shuffle reduction runs), the
# segment widths that are not 16 (W=20, 24, 28), both gating shapes, both input
# dtypes, threadgroups that do and do not fill a simdgroup, and a T that is not
# a multiple of the time block.
_CASES = [
    (1, 64, 16, 32, 128, 128, mx.bfloat16, False),
    (2, 77, 16, 32, 128, 128, mx.bfloat16, False),
    (1, 64, 32, 32, 128, 128, mx.float32, False),
    (1, 33, 4, 8, 64, 64, mx.bfloat16, False),
    (1, 40, 2, 4, 96, 128, mx.bfloat16, False),
    (1, 64, 8, 16, 128, 128, mx.bfloat16, True),
    (1, 20, 2, 4, 32, 128, mx.float32, False),   # P=2, DB=128
    (1, 20, 2, 4, 160, 128, mx.float32, False),  # W=20, Dk not a power of two
    (1, 20, 2, 4, 256, 128, mx.float32, False),  # P=16, TB=8
    (1, 20, 2, 4, 128, 3, mx.float32, False),    # 24 threads: partial simdgroup
    (1, 20, 2, 4, 32, 6, mx.float32, True),      # 12 threads, per-column gating
    (1, 20, 2, 4, 224, 128, mx.float32, False),  # W=28
    (1, 20, 2, 4, 512, 128, mx.bfloat16, False), # P=32: the off=16 shuffle step
]


def _case_id(c):
    return f"B{c[0]}T{c[1]}_Dk{c[4]}Dv{c[5]}_{'vec' if c[7] else 'scalar'}_{str(c[6])[10:]}"


@metal_only
@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_blocked_forward_matches_fp32_reference(case):
    from unsloth_zoo.gated_delta_vjp import gated_delta_blocked_forward

    q, k, v, g, beta, state = _inputs(*case)
    out = gated_delta_blocked_forward(q, k, v, g, beta, state)
    assert out is not None, "layout should be supported for this shape"
    y, s_out = out
    assert (y.dtype, s_out.dtype) == (q.dtype, state.dtype)
    y_ref, s_ref = _reference(q, k, v, g, beta, state)
    assert _rel(y, y_ref) < _Y_TOL[case[6]]
    assert _elem_rel(y, y_ref) < _Y_ELEM_TOL[case[6]]
    assert _rel(s_out, s_ref) < _STATE_TOL
    assert _elem_rel(s_out, s_ref) < _STATE_ELEM_TOL


@metal_only
def test_blocked_forward_is_causal_per_step_and_independent_per_v_row():
    """Perturb a late timestep and a non-zero v row; check both directions."""
    from unsloth_zoo.gated_delta_vjp import gated_delta_blocked_forward

    q, k, v, g, beta, state = _inputs(1, 40, 4, 8, 128, 128, mx.float32, False)
    y0, _ = gated_delta_blocked_forward(q, k, v, g, beta, state)

    t_p, dv_p = 37, 71  # deliberately not step 0 and not row 0
    v2 = mx.array(v)
    v2[0, t_p, 5, dv_p] = v2[0, t_p, 5, dv_p] + 1.0
    y1, _ = gated_delta_blocked_forward(q, k, v2, g, beta, state)
    d = mx.abs(y1 - y0)
    assert float(mx.max(d[:, :t_p])) == 0.0, "a step must not affect earlier steps"
    assert float(mx.max(d[:, t_p:, 5, dv_p])) > 0.0, "later steps of the row must move"
    assert float(mx.max(d[:, :, 4])) == 0.0, "other v heads must not move"
    other = mx.concatenate([d[:, :, 5, :dv_p], d[:, :, 5, dv_p + 1:]], axis=-1)
    assert float(mx.max(other)) == 0.0, "other rows of the same head must not move"


@metal_only
def test_unsupported_layout_falls_back_to_the_stock_kernel():
    from unsloth_zoo.gated_delta_vjp import (
        _blocked_layout, gated_delta_blocked_forward, gated_delta_kernel_efficient,
    )

    assert _blocked_layout(40, 128, mx.bfloat16, False) is None  # Dk % 16 != 0
    assert _blocked_layout(128, 40, mx.bfloat16, False) is None  # Dv % DB != 0
    assert _blocked_layout(128, 128, mx.bfloat16, False)[:2] == (8, 16)
    # Dv=40 satisfies gated_delta_kernel_supported but derives no blocked
    # layout, so the whole path has to run on the stock forward.
    q, k, v, g, beta, state = _inputs(1, 40, 2, 4, 128, 40, mx.float32, False)
    assert gated_delta_blocked_forward(q, k, v, g, beta, state) is None
    y, s = gated_delta_kernel_efficient(q, k, v, g, beta, state)
    y_ref, s_ref = _reference(q, k, v, g, beta, state)
    assert _rel(y, y_ref) < _Y_TOL[mx.float32] and _rel(s, s_ref) < _STATE_TOL


@metal_only
def test_kernel_efficient_forward_uses_the_blocked_kernel():
    import unsloth_zoo.gated_delta_vjp as gdv

    q, k, v, g, beta, state = _inputs(1, 96, 32, 32, 128, 128, mx.float32, False)
    calls, real = [], gdv.gated_delta_blocked_forward
    gdv.gated_delta_blocked_forward = lambda *a: (calls.append(1), real(*a))[1]
    try:
        y, s = gdv.gated_delta_kernel_efficient(q, k, v, g, beta, state)
    finally:
        gdv.gated_delta_blocked_forward = real
    assert calls, "the chunked forward must dispatch the blocked kernel"
    y_ref, s_ref = _reference(q, k, v, g, beta, state)
    assert _rel(y, y_ref) < _Y_TOL[mx.float32]
    assert _elem_rel(y, y_ref) < _Y_ELEM_TOL[mx.float32]
    assert _rel(s, s_ref) < _STATE_TOL
