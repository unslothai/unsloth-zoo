# Unsloth Zoo - Utilities for Unsloth
# Pin MLXTrainingConfig grad-clip resolution across all three knobs:
#   max_grad_leaf_norm  proportional per-leaf L2 cap (cheap, direction-preserving)
#   max_grad_value      elementwise clamp (historical contract; explicit positives win)
#   max_grad_norm       global L2 (HF parity; cross-tree reduction, pays memory)
# Default (all None) -> ("leaf_norm", 1.0); explicit 0.0 disables that knob.

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_mlx_shim():
    # Evict unsloth_zoo.mlx.* so a module bound to real mlx is rebuilt against the shim (as _install_shim does).
    import sys

    from mlx_simulation import (
        restore_modules,
        simulate_mlx_on_torch,
        snapshot_modules,
    )
    from mlx_simulation.mlx_stub import _MLXFinder

    shim_prefixes = ("mlx", "mlx_lm", "mlx_vlm")

    def _owned(name):
        return (
            name == "unsloth_zoo.mlx" or name.startswith("unsloth_zoo.mlx.")
            or any(name == prefix or name.startswith(f"{prefix}.") for prefix in shim_prefixes)
        )

    real_mlx_modules = snapshot_modules(_owned)
    simulate_mlx_on_torch()
    for name in list(sys.modules):
        if name == "unsloth_zoo.mlx" or name.startswith("unsloth_zoo.mlx."):
            sys.modules.pop(name, None)
    yield
    sys.meta_path[:] = [
        finder for finder in sys.meta_path
        if not isinstance(finder, _MLXFinder)
    ]
    restore_modules(real_mlx_modules, _owned)


def _resolve(raw_mgv=None, raw_mgln=None, max_grad_norm=0.0):
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig, _resolve_mlx_grad_clipping

    cfg = MLXTrainingConfig(
        max_grad_norm=max_grad_norm,
        max_grad_value=raw_mgv,
        max_grad_leaf_norm=raw_mgln,
        output_dir="/tmp/x",
    )
    return _resolve_mlx_grad_clipping(cfg)


def test_field_defaults_are_none_sentinels():
    """Defaults are sentinels meaning 'use MLX cheap default'."""
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    cfg = MLXTrainingConfig(output_dir="/tmp/x")
    assert cfg.max_grad_value is None
    assert cfg.max_grad_leaf_norm is None


def test_fields_accept_explicit_positive():
    """Fields accept positive floats for power users opting in."""
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    cfg = MLXTrainingConfig(
        max_grad_value=2.5,
        max_grad_leaf_norm=1.5,
        output_dir="/tmp/x",
    )
    assert cfg.max_grad_value == 2.5
    assert cfg.max_grad_leaf_norm == 1.5


# Each row: knobs passed in -> (max_grad_norm, max_grad_value, max_grad_leaf_norm, mode).
@pytest.mark.parametrize("kwargs, expected", [
    # Default (all None) is the cheap per-leaf clip at 1.0.
    pytest.param(dict(max_grad_norm=0.0), (0.0, 0.0, 1.0, "leaf_norm"), id="default_uses_cheap_leaf_norm"),
    pytest.param(dict(max_grad_norm=1.0), (1.0, 0.0, 0.0, "global_norm"), id="user_max_grad_norm_wins_over_default"),
    # Explicit 0.0 disables the cheap default.
    pytest.param(dict(raw_mgv=0.0, max_grad_norm=0.0), (0.0, 0.0, 0.0, "none"), id="explicit_zero_disables_cheap_default"),
    pytest.param(dict(raw_mgv=0.0, max_grad_norm=1.0), (1.0, 0.0, 0.0, "global_norm"), id="explicit_zero_lets_max_grad_norm_through"),
    # An explicit positive max_grad_value wins over everything (historical contract).
    pytest.param(dict(raw_mgv=2.0, max_grad_norm=1.0), (0.0, 2.0, 0.0, "value"), id="explicit_positive_overrides_max_grad_norm"),
    pytest.param(dict(raw_mgv=5.0, max_grad_norm=0.0), (0.0, 5.0, 0.0, "value"), id="explicit_positive_alone"),
    pytest.param(dict(raw_mgln=1.3, max_grad_norm=1.0), (0.0, 0.0, 1.3, "leaf_norm"), id="explicit_leaf_norm_overrides_max_grad_norm"),
    pytest.param(dict(raw_mgv=2.0, raw_mgln=1.3), (0.0, 2.0, 0.0, "value"), id="max_grad_value_wins_over_leaf_norm_when_both_positive"),
])
def test_resolution(kwargs, expected):
    assert _resolve(**kwargs) == expected


def test_leaf_norm_and_value_clipping_have_distinct_results():
    import mlx.core as mx
    import numpy as np
    from unsloth_zoo.mlx.trainer import _clip_grad_by_leaf_norm, _clip_grad_by_value

    grad = {"weight": mx.array([3.0, 4.0])}
    value_clipped = _clip_grad_by_value(grad, 2.0)
    leaf_clipped = _clip_grad_by_leaf_norm(grad, 2.5)
    np.testing.assert_allclose(np.array(value_clipped["weight"]), [2.0, 2.0])
    np.testing.assert_allclose(np.array(leaf_clipped["weight"]), [1.5, 2.0], rtol=1e-6)
