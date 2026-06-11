# Unsloth Zoo - Utilities for Unsloth
# Pin patch_gated_delta consumer-binding sweep:
#   * Consumers that did `from .gated_delta import gated_delta_update` hold a
#     stale module-level binding; patching the source module alone never
#     reaches their call sites. The sweep must rebind them by identity.
#   * Foreign implementations (same attribute name, different function — e.g.
#     mlx-vlm >= 0.6 ships its own gated_delta module) must NOT be touched.
#   * A second patch_gated_delta() call must re-sweep without double-wrapping.

from __future__ import annotations

import sys
import types

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_mlx_shim():
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


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

    # Stale from-import consumers (mirrors qwen3_5 / qwen3_next / mlx-vlm).
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

    # Foreign implementation with the same attribute name (mlx-vlm >= 0.6).
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


def test_source_module_is_patched(fake_mlx_lm):
    _patch()
    assert fake_mlx_lm.gated_delta.gated_delta_update is not fake_mlx_lm.original
    assert fake_mlx_lm.gated_delta._unsloth_gated_delta_patched


def test_stale_consumer_bindings_are_rebound(fake_mlx_lm):
    _patch()
    patched = fake_mlx_lm.gated_delta.gated_delta_update
    for name, module in fake_mlx_lm.consumers.items():
        assert module.gated_delta_update is patched, f"{name} still stale"


def test_foreign_implementations_are_left_alone(fake_mlx_lm):
    _patch()
    assert (
        fake_mlx_lm.foreign.gated_delta_update is fake_mlx_lm.foreign_fn
    ), "foreign gated_delta_update must not be replaced"


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
