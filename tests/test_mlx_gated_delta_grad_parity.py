# Unsloth Zoo - Utilities for Unsloth
# Pin GatedDeltaNet VJP gradient correctness against plain autodiff:
#   * The ops VJP must match plain autodiff for B >= 2 (mx `.at[:, t].add`
#     scatters read updates with a wrong batch stride on mlx 0.31, which
#     silently corrupted every batch row past the first).
#   * The fused-kernel VJP must match the same ground truth.
# Requires real MLX + Metal; skipped on CI runners without it.

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("requires Apple Silicon Metal GPU", allow_module_level=True)

# Snapshot the REAL mlx/mlx_lm modules at collection time. Sibling test
# files install the mlx_simulation torch-stub into sys.modules at fixture
# time; this module needs the real stack regardless of execution order,
# and must hand the stubs back afterwards. The unsloth_zoo gated-delta
# modules are re-imported fresh inside the fixture so their module-level
# `mx` binds whichever side is active — never imported at collection,
# which would leak real bindings into the stub-based sibling tests.
import importlib  # noqa: E402
import sys  # noqa: E402

import mlx_lm.models.gated_delta  # noqa: F401, E402

_REAL_MODULES = {
    name: module
    for name, module in sys.modules.items()
    if name == "mlx" or name.startswith(("mlx.", "mlx_lm", "mlx_vlm"))
}
_ZOO_MODULES = ("unsloth_zoo.gated_delta_vjp",)


@pytest.fixture(autouse=True, scope="module")
def _restore_real_mlx_modules():
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


def _assert_grads_match(fn, args, cots, tol):
    _, truth = mx.vjp(lambda *a: _plain_reference(*a), args, cots)
    _, got = mx.vjp(lambda *a: fn(*a), args, cots)
    mx.eval(truth, got)
    names = ["d_q", "d_k", "d_v", "d_g", "d_beta", "d_state"]
    for name, t_, g_ in zip(names, truth, got):
        diff = float(mx.abs(t_.astype(mx.float32) - g_.astype(mx.float32)).max())
        assert diff < tol, f"{name} diff {diff} exceeds {tol}"


CASES = [
    # (B, T, Hk, Hv, Dk, Dv, dtype, tol, vectorized) — B >= 2 everywhere:
    # batch rows past the first are exactly what the .at[] scatter bug
    # corrupted. vectorized=True exercises kimi_linear-style per-column
    # gating (g: [B, T, Hv, Dk]).
    (2, 96, 2, 4, 64, 32, mx.float32, 5e-4, False),
    (3, 70, 2, 4, 32, 16, mx.float32, 5e-4, False),
    (2, 130, 4, 4, 96, 64, mx.bfloat16, 2e-2, False),
    (2, 96, 4, 4, 64, 64, mx.float32, 5e-4, True),
    (2, 130, 4, 4, 128, 128, mx.bfloat16, 2e-2, True),
]
CASE_IDS = ["b2-gqa", "b3-gqa", "b2-bf16", "b2-vec-kimi", "b2-vec-bf16"]


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_ops_vjp_matches_plain_autodiff(case):
    from unsloth_zoo.gated_delta_vjp import gated_delta_ops_efficient
    *cfg, dtype, tol, vectorized = case
    args, cots = _make_case(*cfg, dtype, vectorized=vectorized)
    _assert_grads_match(gated_delta_ops_efficient, args, cots, tol)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_kernel_vjp_matches_plain_autodiff(case):
    from unsloth_zoo.gated_delta_vjp import gated_delta_kernel_efficient
    *cfg, dtype, tol, vectorized = case
    args, cots = _make_case(*cfg, dtype, vectorized=vectorized)
    _assert_grads_match(gated_delta_kernel_efficient, args, cots, tol)


def test_patched_update_routes_training_to_kernel_path(monkeypatch):
    """state=None + no mask + scalar gating must take the kernel VJP."""
    import unsloth_zoo.gated_delta_vjp as gk

    called = {}
    real = gk.gated_delta_kernel_efficient

    def spy(*args, **kwargs):
        called["kernel"] = True
        return real(*args, **kwargs)

    monkeypatch.setattr(gk, "gated_delta_kernel_efficient", spy)
    from unsloth_zoo.gated_delta_vjp import patch_gated_delta
    patch_gated_delta()
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
