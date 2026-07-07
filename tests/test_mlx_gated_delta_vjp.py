# Unsloth Zoo - Utilities for Unsloth
# Gated-delta VJP tests:
#   * consumer-binding sweep: stale `from .gated_delta import ...` bindings
#     must be rebound by identity, foreign impls left alone (torch shim on CI).
#   * structural gated-delta detection for the patch trigger.
#   * gradient parity vs PLAIN AUTODIFF for the ops and fused-kernel VJP,
#     B >= 2 (mx `.at[:, t].add` corrupted rows past the first on mlx 0.31,
#     fixed by ml-explore/mlx#3483). Metal-only.
#   * kernel routing: training calls must reach the fused-kernel VJP.

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

_HAS_METAL = _HAS_REAL_MLX and mx.metal.is_available()
requires_metal = pytest.mark.skipif(
    not _HAS_METAL, reason="needs Apple Silicon Metal GPU"
)

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
    """state=None + no mask must take the kernel VJP."""
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
    y, s = gd.gated_delta_update(q, k, v, a, b, A_log, dt_bias, state=None)
    mx.eval(y, s)
    assert called.get("kernel"), "training call did not route to kernel VJP"


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
