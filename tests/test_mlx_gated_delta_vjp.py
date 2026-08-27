# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# Gated-delta VJP tests:
#   * consumer-binding sweep: stale `from .gated_delta import ...` bindings
#     must be rebound by identity, foreign impls left alone (torch shim on CI).
#   * structural gated-delta detection for the patch trigger.
#   * gradient parity vs PLAIN AUTODIFF for the ops and fused-kernel VJP,
#     B >= 2 (mx `.at[:, t].add` corrupted rows past the first on mlx 0.31,
#     fixed by ml-explore/mlx#3483). Metal-only.
#   * kernel routing: training calls must reach the fused-kernel VJP.
#   * the training window that turns those patches on, the index detachment it
#     installs, and the fusions it must disable.

from __future__ import annotations

import importlib
import importlib.util
import sys
import types

import pytest

_HAS_REAL_MLX = importlib.util.find_spec("mlx") is not None
if not _HAS_REAL_MLX:
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()

import mlx.core as mx  # noqa: E402  (real, or the torch shim on CI)
import mlx.nn as nn  # noqa: E402

from unsloth_zoo.mlx.loader import (  # noqa: E402
    _disable_fused_input_projections, _disable_fused_mrope)
from unsloth_zoo.mlx.utils import (  # noqa: E402
    _MLX_INDEX_OP_NAMES, acquire_mlx_training_patches, mlx_training_patches_active,
    pause_mlx_training_patches, release_mlx_training_patches,
    resume_mlx_training_patches)

_HAS_METAL = _HAS_REAL_MLX and mx.metal.is_available()
requires_metal = pytest.mark.skipif(
    not _HAS_METAL, reason="needs Apple Silicon Metal GPU"
)
requires_real_mlx = pytest.mark.skipif(not _HAS_REAL_MLX, reason="needs real MLX")

# Snapshot the REAL mlx/mlx_lm modules now, before sibling test files install
# the mlx_simulation torch-stub into sys.modules, so the code under test
# resolves the real stack regardless of order. The explicit import pulls in
# mlx_lm.models.gated_delta (the kernel path from-imports it at call time).
if _HAS_REAL_MLX:
    import mlx_lm.models.gated_delta  # noqa: F401

_REAL_MODULES = (
    {
        name: module
        for name, module in sys.modules.items()
        if name == "mlx" or name.startswith(("mlx.", "mlx_lm", "mlx_vlm"))
    }
    if _HAS_REAL_MLX
    else {}
)
_ZOO_MODULES = ("unsloth_zoo.gated_delta_vjp",)


@pytest.fixture(autouse=True, scope="module")
def _restore_real_mlx_modules():
    if not _HAS_REAL_MLX:
        yield
        return
    displaced = {name: sys.modules.get(name) for name in _REAL_MODULES}
    displaced_zoo = {name: sys.modules.pop(name, None) for name in _ZOO_MODULES}
    sys.modules.update(_REAL_MODULES)
    for name in _ZOO_MODULES:
        importlib.import_module(name)
    yield
    for name in _ZOO_MODULES:
        sys.modules.pop(name, None)
    for swapped in (displaced_zoo, displaced):
        for name, module in swapped.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


# -- consumer-binding sweep ---------------------------------------------------


@pytest.fixture()
def fake_mlx_lm(monkeypatch):
    """Install a minimal fake mlx_lm.models.gated_delta + consumer modules."""

    def original_gated_delta_update(*args, **kwargs):
        raise AssertionError("unpatched gated_delta_update must not run")

    gated_delta = types.ModuleType("mlx_lm.models.gated_delta")
    gated_delta.gated_delta_update = original_gated_delta_update
    gated_delta.compute_g = lambda *a, **k: None
    gated_delta.gated_delta_kernel = lambda *a, **k: None

    models = types.ModuleType("mlx_lm.models")
    models.gated_delta = gated_delta
    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.models = models

    consumer_names = (
        "mlx_lm.models.qwen3_5",
        "mlx_lm.models.qwen3_next",
        "mlx_vlm.models.qwen3_5.language",
    )
    consumers = {}
    for name in consumer_names:
        module = types.ModuleType(name)
        module.gated_delta_update = original_gated_delta_update
        consumers[name] = module

    def foreign_gated_delta_update(*args, **kwargs):
        return "foreign"

    foreign = types.ModuleType("mlx_vlm.models.qwen3_5.gated_delta")
    foreign.gated_delta_update = foreign_gated_delta_update

    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.gated_delta", gated_delta)
    for name, module in consumers.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(
        sys.modules, "mlx_vlm.models.qwen3_5.gated_delta", foreign,
    )

    return types.SimpleNamespace(
        gated_delta=gated_delta,
        original=original_gated_delta_update,
        consumers=consumers,
        foreign=foreign,
        foreign_fn=foreign_gated_delta_update,
    )


def _patch():
    from unsloth_zoo.gated_delta_vjp import patch_gated_delta
    patch_gated_delta()


def test_sweep_rebinds_stale_consumers_only(fake_mlx_lm):
    _patch()
    patched = fake_mlx_lm.gated_delta.gated_delta_update
    assert patched is not fake_mlx_lm.original
    assert fake_mlx_lm.gated_delta._unsloth_gated_delta_patched
    for name, module in fake_mlx_lm.consumers.items():
        assert module.gated_delta_update is patched, f"{name} still stale"
    # Foreign impls (a different function, e.g. mlx-vlm >= 0.6's own module) stay.
    assert fake_mlx_lm.foreign.gated_delta_update is fake_mlx_lm.foreign_fn


def test_second_call_sweeps_consumers_imported_after_first_patch(
    fake_mlx_lm, monkeypatch,
):
    _patch()
    patched = fake_mlx_lm.gated_delta.gated_delta_update

    late = types.ModuleType("mlx_lm.models.kimi_linear")
    late.gated_delta_update = fake_mlx_lm.original
    monkeypatch.setitem(sys.modules, "mlx_lm.models.kimi_linear", late)

    _patch()
    assert late.gated_delta_update is patched
    # No double-wrap: the patched function is stable across calls.
    assert fake_mlx_lm.gated_delta.gated_delta_update is patched


# -- structural gated-delta detection -----------------------------------------


def test_structural_detection():
    from unsloth_zoo.mlx.compile import model_has_gated_delta_layers

    class _GatedDeltaNet:
        def __init__(self):
            self.A_log = object()
            self.dt_bias = object()

    class _Mamba2Mixer:
        """Same parameters, non-delta class name: must NOT match."""

        def __init__(self):
            self.A_log = object()
            self.dt_bias = object()

    class _FakeModel:
        def __init__(self, layer):
            self._layers = [("layers.0", layer)] if layer is not None else []

        def named_modules(self):
            return list(self._layers)

    class _Broken:
        def named_modules(self):
            raise RuntimeError("no module tree")

    assert model_has_gated_delta_layers(_FakeModel(_GatedDeltaNet()))
    assert not model_has_gated_delta_layers(_FakeModel(_Mamba2Mixer()))
    assert not model_has_gated_delta_layers(_FakeModel(None))
    assert not model_has_gated_delta_layers(_Broken())


def test_structural_detection_matches_unnamed_linear_attention(monkeypatch):
    """GLM-5.x holds the gate-decay pair on a `Glm5NextForgetGate` under a mixer
    named `Glm5NextLinearAttention`; what separates it from an SSM gate carrying the
    same parameters is that its defining module binds `gated_delta_update`."""
    from unsloth_zoo.mlx.compile import model_has_gated_delta_layers

    class _ForgetGate:
        A_log = dt_bias = object()

    class _SelfAttn: pass       # same module, no gate decay: must NOT match

    _model = lambda *m: types.SimpleNamespace(named_modules=lambda: list(enumerate(m)))

    for suffix, binds_update in (("linear_attention", True), ("ssm", False)):
        module = types.ModuleType(f"fake_pkg.{suffix}")
        if binds_update:
            module.gated_delta_update = lambda *a, **k: None
        monkeypatch.setitem(sys.modules, module.__name__, module)
        _ForgetGate.__module__ = _SelfAttn.__module__ = module.__name__
        mixer, gate, attn = types.SimpleNamespace(), _ForgetGate(), _SelfAttn()
        assert bool(model_has_gated_delta_layers(_model(mixer, gate))) is binds_update
        assert not model_has_gated_delta_layers(_model(mixer, attn))


# -- gradient parity vs plain autodiff (Metal only) ---------------------------


def _plain_reference(q, k, v, g, beta, state):
    from unsloth_zoo.gated_delta_vjp import _gated_delta_step
    T = q.shape[1]
    Hv, Hk = v.shape[-2], q.shape[-2]
    if (r := Hv // Hk) > 1:
        q = mx.repeat(q, r, -2)
        k = mx.repeat(k, r, -2)
    ys, s = [], state
    for t in range(T):
        y, s = _gated_delta_step(q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t], s)
        ys.append(y)
    return mx.stack(ys, axis=1), s


def _make_case(B, T, Hk, Hv, Dk, Dv, dtype, vectorized=False):
    mx.random.seed(0)
    q = (mx.random.normal((B, T, Hk, Dk)) * 0.1).astype(dtype)
    k = (mx.random.normal((B, T, Hk, Dk)) * 0.1).astype(dtype)
    v = (mx.random.normal((B, T, Hv, Dv)) * 0.1).astype(dtype)
    g_shape = (B, T, Hv, Dk) if vectorized else (B, T, Hv)
    g = mx.sigmoid(mx.random.normal(g_shape)) * 0.9
    beta = mx.sigmoid(mx.random.normal((B, T, Hv)))
    state = mx.random.normal((B, Hv, Dv, Dk)) * 0.1
    dy = (mx.random.normal((B, T, Hv, Dv)) * 0.5).astype(dtype)
    dso = mx.random.normal((B, Hv, Dv, Dk)) * 0.5
    return [q, k, v, g, beta, state], [dy, dso]


CASES = [
    # (B, T, Hk, Hv, Dk, Dv, dtype, tol, vectorized) — B >= 2 everywhere.
    # vectorized=True exercises kimi_linear-style per-column gating.
    (2, 96, 2, 4, 64, 32, mx.float32, 5e-4, False),
    (3, 70, 2, 4, 32, 16, mx.float32, 5e-4, False),
    (2, 130, 4, 4, 96, 64, mx.bfloat16, 2e-2, False),
    (2, 96, 4, 4, 64, 64, mx.float32, 5e-4, True),
    (2, 130, 4, 4, 128, 128, mx.bfloat16, 2e-2, True),
]
CASE_IDS = ["b2-gqa", "b3-gqa", "b2-bf16", "b2-vec-kimi", "b2-vec-bf16"]
IMPLEMENTATIONS = ["gated_delta_ops_efficient", "gated_delta_kernel_efficient"]


@requires_metal
@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_vjp_matches_plain_autodiff(impl, case):
    import unsloth_zoo.gated_delta_vjp as gv
    fn = getattr(gv, impl)
    *cfg, dtype, tol, vectorized = case
    args, cots = _make_case(*cfg, dtype, vectorized=vectorized)
    _, truth = mx.vjp(lambda *a: _plain_reference(*a), args, cots)
    _, got = mx.vjp(lambda *a: fn(*a), args, cots)
    mx.eval(truth, got)
    names = ["d_q", "d_k", "d_v", "d_g", "d_beta", "d_state"]
    for name, t_, g_ in zip(names, truth, got):
        diff = float(mx.abs(t_.astype(mx.float32) - g_.astype(mx.float32)).max())
        assert diff < tol, f"{impl}: {name} diff {diff} exceeds {tol}"


@requires_metal
def test_patched_update_routes_training_to_kernel_path(monkeypatch):
    """A call site asking for the differentiable path, or an open window, takes the
    kernel VJP; an empty cache alone must not. Only the spy proves which ran."""
    import unsloth_zoo.gated_delta_vjp as gv

    called = {}
    real = gv.gated_delta_kernel_efficient

    def spy(*args, **kwargs):
        called["kernel"] = True
        return real(*args, **kwargs)

    monkeypatch.setattr(gv, "gated_delta_kernel_efficient", spy)
    gv.patch_gated_delta()
    from mlx_lm.models import gated_delta as gd

    B, T, Hk, Hv, Dk, Dv = 1, 8, 2, 2, 32, 16
    q = mx.random.normal((B, T, Hk, Dk)) * 0.1
    k = mx.random.normal((B, T, Hk, Dk)) * 0.1
    v = mx.random.normal((B, T, Hv, Dv)) * 0.1
    a = mx.random.normal((B, T, Hv))
    b = mx.random.normal((B, T, Hv))
    A_log = mx.random.normal((Hv,))
    dt_bias = mx.random.normal((Hv,))
    mx.eval(gd.gated_delta_update(q, k, v, a, b, A_log, dt_bias, state=None))
    assert not called, "uncached inference was routed to the training VJP"

    acquire_mlx_training_patches()
    try:
        mx.eval(gd.gated_delta_update(q, k, v, a, b, A_log, dt_bias, state=None))
    finally:
        release_mlx_training_patches()
    assert called.get("kernel"), "training call did not route to kernel VJP"

    called.clear()
    mx.eval(gd.gated_delta_update(q, k, v, a, b, A_log, dt_bias, state=None,
                                  use_kernel=False))
    assert called.get("kernel"), "use_kernel=False did not get the efficient VJP"


def test_vlm_patch_rebinds_both_namespaces_and_sweep_skips_it(
    fake_mlx_lm, monkeypatch,
):
    """patch_gated_delta_vlm covers mlx_vlm >= 0.6's own module (a distinct
    function the identity sweep leaves alone), and the sweep must treat the
    sibling patch as owned, not foreign."""
    calls = {}

    def vlm_original(q, k, v, a, b, A_log, dt_bias,
                     state=None, mask=None, use_kernel=True):
        calls["inference"] = state
        return "y", state

    vlm_gd = types.ModuleType("mlx_vlm.models.qwen3_5.gated_delta")
    vlm_gd.gated_delta_update = vlm_original
    vlm_pkg = types.ModuleType("mlx_vlm.models.qwen3_5")
    vlm_pkg.gated_delta = vlm_gd
    vlm_pkg.language = fake_mlx_lm.consumers["mlx_vlm.models.qwen3_5.language"]
    vlm_pkg.language.gated_delta_update = vlm_original
    monkeypatch.setitem(sys.modules, "mlx_vlm.models.qwen3_5", vlm_pkg)
    monkeypatch.setitem(
        sys.modules, "mlx_vlm.models.qwen3_5.gated_delta", vlm_gd,
    )

    from unsloth_zoo.gated_delta_vjp import patch_gated_delta_vlm
    patch_gated_delta_vlm()

    patched = vlm_gd.gated_delta_update
    assert patched is not vlm_original
    assert vlm_pkg.language.gated_delta_update is patched
    assert vlm_gd._unsloth_gated_delta_patched

    # Inference (state provided) delegates to the original implementation.
    y, state = patched(*[object()] * 7, state="kv-cache")
    assert (y, state) == ("y", "kv-cache") and calls["inference"] == "kv-cache"

    # The sweep recognizes the sibling patch instead of warning "foreign".
    _patch()
    assert vlm_pkg.language.gated_delta_update is patched


@requires_metal
def test_kernel_dispatch_guards_partial_threadgroup_rows():
    """Dv not divisible by the threadgroup row count must fall back to the
    ops VJP: the backward kernel's shared-memory pre-reduction would read
    uninitialized slots in a partial trailing threadgroup."""
    import unsloth_zoo.gated_delta_vjp as gv

    q = mx.zeros((1, 8, 2, 32))
    g = mx.zeros((1, 8, 2))
    ok_v = mx.zeros((1, 8, 2, 16))
    bad_v = mx.zeros((1, 8, 2, 30))
    assert gv.gated_delta_kernel_supported(q, g, None, ok_v)
    assert not gv.gated_delta_kernel_supported(q, g, None, bad_v)


# -- training window and index detachment -------------------------------------

@pytest.fixture
def index_stop():
    acquire_mlx_training_patches()
    try:
        yield
    finally:
        release_mlx_training_patches()


def test_window_depth_accounting():
    """Trainer runs overlap and the patches are global: an inner window must not
    unpatch an outer one, and a pause must refuse while anyone else holds it."""
    originals = {name: getattr(mx, name) for name in _MLX_INDEX_OP_NAMES}
    assert not mlx_training_patches_active()
    acquire_mlx_training_patches()
    acquire_mlx_training_patches()
    # The outer run still needs the window, so this pause must be refused.
    assert pause_mlx_training_patches() is False
    assert mlx_training_patches_active() and mx.take_along_axis._unsloth_index_stop_gradient
    release_mlx_training_patches()
    assert pause_mlx_training_patches() is True
    assert not mlx_training_patches_active()
    assert not hasattr(mx.take_along_axis, "_unsloth_index_stop_gradient")
    resume_mlx_training_patches(True)
    assert all(getattr(mx, n)._unsloth_index_stop_gradient for n in _MLX_INDEX_OP_NAMES)
    release_mlx_training_patches()
    assert not mlx_training_patches_active()
    assert all(getattr(mx, n) is originals[n] for n in _MLX_INDEX_OP_NAMES)
    # Outside a run there is nothing to close, and nothing to reopen.
    resume_mlx_training_patches(pause_mlx_training_patches())
    assert not mlx_training_patches_active()


@requires_real_mlx
def test_checkpointed_layer_detaches_integer_arguments():
    """A layer that embeds the token ids it was handed derives a gather index,
    and `mx.checkpoint` makes every argument a primal MLX wants a gradient for."""
    from unsloth_zoo.mlx.utils import _patch_layer_class_for_gc, _unpatch_layer_class_gc

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed, self.proj = nn.Embedding(8, 4), nn.Linear(4, 4)

        def __call__(self, h, ids):
            return self.proj(h) + self.embed(ids)

    layer, ids, h = _Layer(), mx.array([0, 1, 2]), mx.zeros((3, 4))
    plain = nn.value_and_grad(layer, lambda m: m(h, ids).sum())(layer)
    _patch_layer_class_for_gc(_Layer)
    try:
        checkpointed = nn.value_and_grad(layer, lambda m: m(h, ids).sum())(layer)
        mx.eval(plain, checkpointed)
    finally:
        _unpatch_layer_class_gc(_Layer)
    assert checkpointed[0].item() == plain[0].item()
    assert mx.allclose(checkpointed[1]["proj"]["weight"], plain[1]["proj"]["weight"]).item()


@requires_real_mlx
def test_router_gradient_matches_detached_reference(index_stop):
    """Top-k routing in the shape every MoE block uses. An all-zero grad would
    mean the score path was severed with the index path, leaving it untrained."""
    x, w = mx.random.normal((4, 8)), mx.random.normal((8, 6))
    raw = mx.argpartition._unsloth_index_original

    def router(w, argpartition=mx.argpartition, detach=lambda i: i):
        gates = mx.softmax(x @ w, axis=-1)
        inds = detach(argpartition(gates, kth=-2, axis=-1)[..., -2:])
        return mx.take_along_axis(gates, inds, axis=-1).sum()

    grad = mx.grad(router)(w)
    expected = mx.grad(lambda w: router(w, raw, mx.stop_gradient))(w)
    mx.eval(grad, expected)
    assert mx.allclose(grad, expected, atol=1e-6).item()
    assert mx.abs(grad).sum().item() > 0


def _gather_sort_loss(w):
    """SwitchGLU's `x[argsort(indices)]` produces the index inside __getitem__."""
    h = mx.random.normal((8, 4)) @ w
    return h[mx.argsort(h[:, 0])].sum()


def _sparse_mask_loss(w):
    """GLM-5.x's mask index never passes through an arg* op at all."""
    h = mx.random.normal((2, 6)) @ w
    safe = mx.where(h[:, :2] > 0, mx.array([[0, 2], [1, 3]]), 3)
    scattered = mx.put_along_axis(mx.zeros_like(h), safe, mx.array(1.0), axis=-1)
    return (h * scattered).sum()


@requires_real_mlx
@pytest.mark.parametrize("loss, shape",
                         [(_gather_sort_loss, (4, 4)), (_sparse_mask_loss, (6, 4))])
def test_index_derived_graphs_stay_differentiable(loss, shape, index_stop):
    grad = mx.grad(loss)(mx.random.normal(shape))
    mx.eval(grad)
    assert mx.abs(grad).sum().item() > 0


@requires_real_mlx
def test_only_the_index_argument_is_detached(index_stop):
    """MLX differentiates integer arrays; detaching every one would drop a real gradient."""
    data = mx.array([[10, 20], [30, 40]])
    grad = mx.grad(
        lambda d: mx.take_along_axis(d * 2, mx.array([[0], [1]]), axis=-1).sum()
    )(data)
    mx.eval(grad)
    assert grad.tolist() == [[2, 0], [0, 2]]

    # SwitchGLU's quantized path passes lhs_indices=None, which must not detach.
    a, b = mx.random.normal((4, 3, 5)), mx.random.normal((4, 5, 2))
    rhs = mx.array([0, 2, 1, 3], dtype=mx.uint32)
    assert mx.gather_mm(a, b, None, rhs).shape == (4, 3, 2)
    assert mx.gather_mm(a, b, lhs_indices=None, rhs_indices=rhs).shape == (4, 3, 2)


# -- the shared mlx-vlm gated-delta module ------------------------------------

# mlx-vlm 0.6.5 keeps the shared module under `text_models`; 0.6.6 moved it up.
_SHARED_GATED_DELTA = ("mlx_vlm.models.gated_delta",
                       "mlx_vlm.models.text_models.gated_delta")


@pytest.mark.parametrize("shared_name", _SHARED_GATED_DELTA)
def test_shared_patch_rebinds_consumers_and_forwards_lower_bound(shared_name, monkeypatch):
    seen = {}

    def original(q, k, v, a, b, A_log, dt_bias,
                 state=None, mask=None, use_kernel=True, **kw):
        seen["kw"] = kw
        return "cached", state

    shared = types.ModuleType(shared_name)
    consumer = types.ModuleType("mlx_vlm.models.glm5_next.language")
    models_pkg = types.ModuleType("mlx_vlm.models")
    shared.gated_delta_update = consumer.gated_delta_update = original
    models_pkg.gated_delta = shared
    monkeypatch.setitem(sys.modules, "mlx_vlm.models", models_pkg)
    # Mutually exclusive layouts: hide whichever one is not under test.
    for _name in _SHARED_GATED_DELTA:
        monkeypatch.delitem(sys.modules, _name, raising=False)
    monkeypatch.setitem(sys.modules, shared_name, shared)
    monkeypatch.setitem(sys.modules, "mlx_vlm.models.glm5_next.language", consumer)

    from unsloth_zoo.gated_delta_vjp import patch_gated_delta_vlm_shared
    patch_gated_delta_vlm_shared()

    patched = shared.gated_delta_update
    assert patched is not original
    assert consumer.gated_delta_update is patched
    assert shared._unsloth_gated_delta_patched

    # A cached call keeps the fused kernel and must carry the gate lower bound.
    assert patched(*[object()] * 7, state="kv", lower_bound=-5.0) == ("cached", "kv")
    assert seen["kw"] == {"lower_bound": -5.0}

    # Prefill is uncached too, so an empty state is not a training signal.
    assert patched(*[object()] * 7) == ("cached", None)
    # mlx-vlm had no `lower_bound` here before 0.6.9; do not invent one for it.
    assert seen["kw"] == {}

    # Inside a window it takes the training branch, which must gate through
    # `compute_g_safe`: GLM-5.x clamps the decay and qwen3_5 does not.
    import unsloth_zoo.gated_delta_vjp as gv
    shared.compute_g = lambda *a: pytest.fail("bounded gate took the plain path")
    shared.compute_g_safe = lambda A_log, a, dt, lb: seen.setdefault("bound", lb)
    monkeypatch.setattr(gv, "gated_delta_kernel_supported", lambda *a: False)
    monkeypatch.setattr(gv, "gated_delta_ops_efficient", lambda *a: "vjp")
    z = mx.zeros((1, 4, 2, 8))
    acquire_mlx_training_patches()
    try:
        assert patched(*[z] * 7, lower_bound=-5.0) == "vjp"
    finally:
        release_mlx_training_patches()
    assert seen["bound"] == -5.0


# -- fusions and caches the trainer must turn off -----------------------------

class _FakeModel:
    def __init__(self, *modules):
        self._modules = modules

    def modules(self):
        return list(self._modules)

    def named_modules(self):
        return [(f"layers.{i}", m) for i, m in enumerate(self._modules)]


def test_qwen35_attention_detection_follows_the_mro():
    from unsloth_zoo.mlx.compile import model_has_qwen35_attention_layers

    base = type("Qwen3_5Attention", (), {})
    subclass = type("Qwen4ExpAttention", (base,), {})
    assert model_has_qwen35_attention_layers(_FakeModel(subclass()))
    assert not model_has_qwen35_attention_layers(_FakeModel(object()))


@pytest.mark.parametrize("disable, flag, extra", [
    (_disable_fused_input_projections, "fuse_in", {"_fused_ready": True}),
    (_disable_fused_mrope, "fused_apply", {}),
])
def test_disabling_a_fusion_targets_only_fused_modules(disable, flag, extra):
    """The returned modules are what the trainer re-fuses once training is over."""
    fused = types.SimpleNamespace(**{flag: True}, **extra)
    plain = types.SimpleNamespace()
    assert disable(_FakeModel(fused, plain)) == [fused]
    assert getattr(fused, flag) is False
    assert not hasattr(plain, flag)
    # The projection fusion also drops the concatenation it cached.
    assert not extra or fused._fused_ready is False
    assert disable(_FakeModel(fused, plain)) == []


@pytest.mark.skipif(
    importlib.util.find_spec("mlx_vlm.models.glm5_next") is None
    or not _HAS_REAL_MLX,
    reason="needs mlx-vlm with glm5_next on real MLX",
)
def test_unfused_projection_matches_the_fused_one():
    if sys.modules.get("mlx.core") is not mx:
        pytest.skip("another suite installed the MLX shim")
    from mlx_vlm.models.glm5_next.config import TextConfig
    from mlx_vlm.models.glm5_next.language import Glm5NextLinearAttention

    config = TextConfig(
        model_type="glm5_next_text", vocab_size=64, hidden_size=64, intermediate_size=128,
        moe_intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=4, n_shared_experts=1, n_routed_experts=4, index_topk=8,
        routed_scaling_factor=1.0, kv_lora_rank=16, q_lora_rank=32, qk_rope_head_dim=0,
        v_head_dim=32, qk_nope_head_dim=32, num_experts_per_tok=2, index_n_heads=2,
        first_k_dense_replace=0, max_position_embeddings=256, rms_norm_eps=1e-5,
        index_head_dim=32, layer_types=["linear_attention"], mlp_layer_types=["dense"],
        linear_attn_config={"num_heads": 2, "head_dim": 32,
                            "short_conv_kernel_size": 4, "gate_lower_bound": -5.0},
    )
    module = Glm5NextLinearAttention(config)
    mx.eval(module.parameters())
    x = mx.random.normal((1, 16, config.hidden_size))

    fused = module(x)
    mx.eval(fused)

    # A zero-initialized adapter leaves the output unchanged, but the fusion
    # reads `.weight` off the projection and LoRALinear has none.
    from mlx_lm.tuner.lora import LoRALinear
    module.update_modules({"q_proj": LoRALinear.from_base(module.q_proj, r=4)})
    module._fused_ready = False
    with pytest.raises(AttributeError):
        module(x)

    assert _disable_fused_input_projections(_FakeModel(module)) == [module]
    unfused = module(x)
    mx.eval(unfused)
    assert mx.allclose(fused, unfused, atol=1e-5, rtol=1e-5).item()
