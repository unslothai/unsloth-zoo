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


# (b) GDN: patch fixes grad; output matches the use_kernel=False reference
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


# (c) MRoPE: fused apply is non-differentiable; flip makes grad work; fused vs
#     fallback forward match
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


# (d) End-to-end: one value_and_grad step on a GDN + attention model
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


# The elementwise bound divides by each entry's own magnitude plus 1% of the
# tensor's largest, so a small bad entry cannot hide behind a large one.
_Y_TOL = {mx.bfloat16: 5e-3, mx.float16: 5e-4, mx.float32: 5e-6}
_Y_ELEM_TOL = {mx.bfloat16: 8e-3, mx.float16: 1e-3, mx.float32: 2e-5}
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
    return q, k, v, g, beta, state


def _gqa_repeated(args):
    q, k, v, g, beta, state = args
    r = v.shape[-2] // q.shape[-2]
    return (mx.repeat(q, r, -2), mx.repeat(k, r, -2), v, g, beta, state) if r > 1 else args


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


def _grads_of(fn, args, w, ws):
    def inner(q, k, v, g, beta, state):
        y, s = fn(q, k, v, g, beta, state)
        return (y.astype(mx.float32) * w).sum() + (s.astype(mx.float32) * ws).sum()
    return mx.value_and_grad(inner, argnums=(0, 1, 2, 3, 4, 5))(*args)


def _rel(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b)) / mx.max(mx.abs(b)))


def _elem_rel(a, b, floor=0.01):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b)
                        / (mx.abs(b) + floor * mx.max(mx.abs(b)))))


# (B, T, Hk, Hv, Dk, Dv, dtype, vectorized). Together they exercise every thread
# prefix and segment width the layout derives, both gatings, all dtypes, a ragged T.
_CASES = [
    (1, 64, 16, 32, 128, 128, mx.bfloat16, False),
    (2, 77, 16, 32, 128, 128, mx.bfloat16, False),
    (1, 64, 32, 32, 128, 128, mx.float32, False),
    (1, 33, 4, 8, 64, 64, mx.bfloat16, False),
    (1, 40, 2, 4, 96, 128, mx.bfloat16, False),   # W=24 forward, W=12 backward
    (1, 64, 8, 16, 128, 128, mx.bfloat16, True),
    (1, 20, 2, 4, 32, 128, mx.float32, False),    # P=1 forward, P=2 backward
    (1, 20, 2, 4, 160, 128, mx.float32, False),   # W=20, Dk not a power of two
    (1, 20, 2, 4, 256, 128, mx.float32, False),   # P=16 backward
    (1, 20, 2, 4, 128, 4, mx.float32, False),     # 32 threads, one simdgroup
    (1, 20, 2, 4, 128, 40, mx.float32, False),    # Dv not a multiple of the cap
    (1, 20, 2, 4, 224, 128, mx.float32, False),
    (1, 20, 2, 4, 256, 128, mx.float32, True),    # backward needs a narrowed threadgroup
    (1, 20, 2, 4, 64, 128, mx.float32, True),     # per-column gating, P=2/P=4
    (1, 20, 2, 4, 128, 2, mx.float32, False),
    (1, 20, 2, 4, 128, 1, mx.float32, True),      # P=32, W=4: the off=16 shuffle
    (1, 20, 2, 8, 128, 128, mx.bfloat16, False),
    (1, 20, 2, 4, 128, 128, mx.float16, False),
]


def _case_id(c):
    return f"B{c[0]}T{c[1]}_Dk{c[4]}Dv{c[5]}_{'vec' if c[7] else 'scalar'}_{str(c[6])[9:]}"


@metal_only
@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_blocked_forward_matches_fp32_reference(case):
    from unsloth_zoo.gated_delta_vjp import gated_delta_blocked_forward

    q, k, v, g, beta, state = _gqa_repeated(_inputs(*case))
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
def test_only_calls_the_kernels_can_answer_for_are_admitted():
    from unsloth_zoo.gated_delta_vjp import (
        _blocked_layouts, gated_delta_blocked_backward, gated_delta_blocked_forward,
        gated_delta_kernel_supported,
    )

    q, k, v, g, beta, state = _inputs(1, 8, 1, 1, 288, 128, mx.float32, False)
    assert _blocked_layouts(q, g, v) is None   # Dk=288 passes every other rule
    assert not gated_delta_kernel_supported(q, g, None, v, k)
    assert gated_delta_blocked_forward(q, k, v, g, beta, state) is None

    ok_q, ok_k, ok_v, ok_g, _, _ = _inputs(1, 8, 1, 1, 128, 128, mx.float32, False)
    bf_q, bf_k, bf_v, bf_g = (x.astype(mx.bfloat16) for x in (ok_q, ok_k, ok_v, ok_g))
    assert gated_delta_kernel_supported(ok_q, ok_g, None, ok_v, ok_k)
    assert gated_delta_kernel_supported(bf_q, bf_g, None, bf_v, bf_k)
    for partial in ((ok_q, ok_g, None), (ok_q, ok_g, None, ok_v), (q, g, None)):
        assert not gated_delta_kernel_supported(*partial)
    for mix in ((ok_q, ok_g, ok_v, bf_k), (ok_q, ok_g, bf_v, ok_k),
                (bf_q, bf_g, bf_v, bf_k.astype(mx.float16))):
        assert not gated_delta_kernel_supported(mix[0], mix[1], None, *mix[2:])
    fwd, bwd = gated_delta_blocked_forward, gated_delta_blocked_backward
    args = list(_inputs(1, 16, 4, 4, 128, 128, mx.float32, False))
    dy, d_state = mx.random.normal((1, 16, 4, 128)), mx.zeros_like(args[5])
    assert fwd(*args) is not None and bwd(*args, dy, d_state) is not None
    for i in (1, 2):
        mixed = list(args)
        mixed[i] = mixed[i].astype(mx.bfloat16)
        assert fwd(*mixed) is None and bwd(*mixed, dy, d_state) is None
    assert bwd(*args, dy.astype(mx.bfloat16), d_state) is None


@metal_only
@pytest.mark.parametrize("rest_dtype", [mx.float32, mx.bfloat16])
def test_every_output_keeps_the_dtype_its_primal_came_in(rest_dtype):
    import unsloth_zoo.gated_delta_vjp as gdv

    q, k, v, *rest = _inputs(1, 100, 4, 4, 128, 128, mx.bfloat16, False)
    g, beta, state = (x.astype(rest_dtype) for x in rest)
    primals = (q, k, v, g, beta, state)
    mx.random.seed(13)
    w, ws = mx.random.normal((1, 100, 4, 128)), mx.random.normal((1, 4, 128, 128))
    ref = _reference(q, k, v, g, beta, state.astype(mx.float32))[1]
    for out in (gdv.gated_delta_kernel_efficient(q, k, v, g, beta, state)[1],
                gdv.gated_delta_blocked_forward(q, k, v, g, beta, state)[1]):
        assert out.dtype == rest_dtype and _rel(out, ref) < 5e-3
    # Not elementwise: bf16 leaves three digits, so cancellation dominates.
    for name, a, b, primal in zip(
            "q k v g beta state".split(),
            _grads_of(gdv.gated_delta_kernel_efficient, primals, w, ws)[1],
            _grads_of(gdv.gated_delta_ops_efficient, primals, w, ws)[1], primals):
        assert a.dtype == primal.dtype and a.shape == primal.shape, name
        assert _rel(a, b) < 1e-2, name


@metal_only
@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_gradients_match_autodiff_over_the_same_recurrence(case):
    """Different accumulation orders, so the elementwise bound exposes a misroute."""
    import unsloth_zoo.gated_delta_vjp as gdv

    args = _inputs(*case)
    B, T, Hk, Hv, Dk, Dv, dtype, _ = case
    mx.random.seed(11)
    w = mx.random.normal((B, T, Hv, Dv))
    ws = mx.random.normal((B, Hv, Dv, Dk))
    tol, elem_tol = (2e-6, 2e-5) if dtype == mx.float32 else (5e-3, 2e-2)
    l_k, g_k = _grads_of(gdv.gated_delta_kernel_efficient, args, w, ws)
    l_o, g_o = _grads_of(gdv.gated_delta_ops_efficient, args, w, ws)
    assert abs(float(l_k) - float(l_o)) <= tol * max(abs(float(l_o)), 1.0)
    for name, a, b, primal in zip("q k v g beta state".split(), g_k, g_o, args):
        assert a.shape == b.shape and a.dtype == primal.dtype, name
        assert _rel(a, b) < tol, name
        assert _elem_rel(a, b) < elem_tol, name


@metal_only
def test_backward_is_causal_and_reaches_non_first_positions():
    """Perturbing dy at one late step moves d_q only at that step and head, and d_v
    all the way back from it; the rest sits at the atomic reductions' noise floor."""
    import unsloth_zoo.gated_delta_vjp as gdv

    q, k, v, g, beta, state = _inputs(1, 40, 4, 8, 128, 128, mx.float32, False)
    t_p, hv_p, dv_p = 31, 5, 97   # deliberately none of them index 0
    hq_p = hv_p // (8 // 4)       # d_q is folded back to the 4 key heads

    def grads(dy):
        def inner(q, k, v, g, beta, state):
            y, _ = gdv.gated_delta_kernel_efficient(q, k, v, g, beta, state)
            return (y.astype(mx.float32) * dy).sum()
        return mx.grad(inner, argnums=(0, 2))(q, k, v, g, beta, state)

    mx.random.seed(7)
    dy0 = mx.random.normal((1, 40, 8, 128))
    dy1 = mx.array(dy0)
    dy1[0, t_p, hv_p, dv_p] += 1.0
    (dq0, dv0), (dq1, dv1) = grads(dy0), grads(dy1)
    repeats = [grads(dy0) for _ in range(4)]
    floor = 8 * max([1e-12] + [float(mx.max(mx.abs(a - b)))
                               for r in repeats for a, b in zip(r, (dq0, dv0))])
    d_q, d_v = mx.abs(dq1 - dq0), mx.abs(dv1 - dv0)

    assert max(float(mx.max(d_q[:, :t_p])), float(mx.max(d_q[:, t_p + 1:]))) <= floor
    assert max(float(mx.max(d_q[:, t_p, :hq_p])), float(mx.max(d_q[:, t_p, hq_p+1:]))) <= floor
    assert float(mx.max(d_q[:, t_p, hq_p])) > 1e3 * floor
    # d_v's trail runs the whole way back, not just to the step the scan entered on.
    assert float(mx.max(d_v[:, t_p + 1:])) <= floor, "d_v cannot look ahead"
    assert float(d_v[0, t_p, hv_p, dv_p]) > 1e3 * floor
    assert float(mx.min(d_v[0, :t_p + 1, hv_p, dv_p])) > 10 * floor
    other = mx.concatenate([d_v[:, :, hv_p, :dv_p], d_v[:, :, hv_p, dv_p + 1:]], axis=-1)
    assert float(mx.max(other)) == 0.0, "d_v must stay in its own state row"
    assert float(mx.max(d_v[:, :, :hv_p]) + mx.max(d_v[:, :, hv_p + 1:])) == 0.0


@metal_only
def test_the_chunk_length_is_the_longest_affordable_and_never_visible():
    """A memory decision, so 64 and 512 agree bit for bit; the only seam coverage."""
    import unsloth_zoo.gated_delta_vjp as gdv

    bill = lambda B, Hv, Dk, Dv, n: B * Hv * Dv * Dk * 4 * max(0, n - 1)
    for B, Hv, Dk, Dv, TB in ((1, 32, 128, 128, 8), (8, 32, 128, 64, 8),
                              (64, 32, 128, 128, 4), (1, 32, 256, 256, 8)):
        chunk = gdv._kernel_chunk_size(B, Hv, Dk, Dv, TB)
        assert chunk % TB == 0 and gdv._KERNEL_CHUNK_MIN <= chunk <= gdv._KERNEL_CHUNK_MAX
        if chunk > gdv._KERNEL_CHUNK_MIN:
            assert bill(B, Hv, Dk, Dv, chunk // TB) <= gdv._KERNEL_CHECKPOINT_BUDGET
        if chunk < gdv._KERNEL_CHUNK_MAX:
            assert bill(B, Hv, Dk, Dv, chunk // TB + 1) > gdv._KERNEL_CHECKPOINT_BUDGET
    assert gdv._kernel_chunk_size(1, 0, 128, 128, 8) == gdv._KERNEL_CHUNK_MAX

    q, k, v, g, beta, state = _inputs(1, 300, 4, 4, 128, 128, mx.float32, False)
    state, out, chunks = state.astype(mx.bfloat16), [], []
    real, fwd = gdv._kernel_chunk_size, gdv.gated_delta_blocked_forward
    gdv.gated_delta_blocked_forward = lambda *a: (chunks.append(a[0].shape[1]), fwd(*a))[1]
    try:
        for length in (64, 512):
            gdv._kernel_chunk_size = lambda *a, n=length: n
            out.append(gdv.gated_delta_kernel_efficient(q, k, v, g, beta, state))
    finally:
        gdv._kernel_chunk_size, gdv.gated_delta_blocked_forward = real, fwd
    assert chunks == [64, 64, 64, 64, 44, 300], "the loop has to use the length"

    assert (out[0][0] == out[1][0]).all() and (out[0][1] == out[1][1]).all()
    assert out[0][1].dtype == mx.bfloat16 and out[0][1].shape == state.shape


@metal_only
@pytest.mark.parametrize("T", [8, 64, 100, 128])
def test_backward_checkpoints_block_boundaries_not_steps(T):
    import unsloth_zoo.gated_delta_vjp as gdv

    B, Hv, Dk, Dv = 1, 32, 128, 128
    q, k, v, g, beta, state = _inputs(B, T, Hv, Hv, Dk, Dv, mx.bfloat16, False)
    dy = mx.random.normal((B, T, Hv, Dv)).astype(mx.bfloat16)
    TB = gdv._blocked_layouts(q, g, v)[1][3]
    shapes, real = [], gdv._get_blocked_kernels(False)

    def spy(**kwargs):
        shapes.append(kwargs["output_shapes"])
        return real[1](**kwargs)

    gdv._GD_BLOCKED[False] = (real[0], spy, real[2])
    try:
        mx.eval(gdv.gated_delta_blocked_backward(
            q, k, v, g, beta, state, dy, mx.zeros_like(state)))
    finally:
        gdv._GD_BLOCKED[False] = real

    (ckpt_shape, kv_shape), = shapes
    assert ckpt_shape == (B, Hv, max(1, -(-T // TB) - 1), Dv, Dk)
    assert kv_shape == (B, Hv, T, Dv)
