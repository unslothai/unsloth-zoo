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
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


def _resolve(raw_mgv=None, raw_mgln=None, max_grad_norm=0.0):
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig, _resolve_mlx_grad_clipping

    cfg = MLXTrainingConfig(
        max_grad_norm=max_grad_norm,
        max_grad_value=raw_mgv,
        max_grad_leaf_norm=raw_mgln,
        output_dir="/tmp/x",
    )
    return _resolve_mlx_grad_clipping(cfg)


# -- field defaults ---------------------------------------------------------


def test_field_defaults_are_none_sentinels():
    """Defaults are sentinels meaning 'use MLX cheap default'."""
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    cfg = MLXTrainingConfig(output_dir="/tmp/x")
    assert cfg.max_grad_value is None
    assert cfg.max_grad_leaf_norm is None


def test_fields_accept_none():
    """Fields accept None and round-trip through the dataclass."""
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    cfg = MLXTrainingConfig(
        max_grad_value=None,
        max_grad_leaf_norm=None,
        output_dir="/tmp/x",
    )
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


# -- resolution semantics ---------------------------------------------------


def test_default_uses_cheap_leaf_norm():
    """Default (all None, max_grad_norm=0.0) -> leaf norm clip at 1.0."""
    mgn, mgv, mgln, mode = _resolve(max_grad_norm=0.0)
    assert mgn == 0.0
    assert mgv == 0.0
    assert mgln == 1.0
    assert mode == "leaf_norm"


def test_user_max_grad_norm_wins_over_default():
    """User passes max_grad_norm=1.0 with defaults -> global norm only."""
    mgn, mgv, mgln, mode = _resolve(max_grad_norm=1.0)
    assert mgn == 1.0
    assert mgv == 0.0
    assert mgln == 0.0
    assert mode == "global_norm"


def test_explicit_zero_disables_cheap_default():
    """Explicit 0.0 disables cheap clipping. With no max_grad_norm, no clip."""
    mgn, mgv, mgln, mode = _resolve(raw_mgv=0.0, max_grad_norm=0.0)
    assert mgn == 0.0
    assert mgv == 0.0
    assert mgln == 0.0
    assert mode == "none"


def test_explicit_zero_lets_max_grad_norm_through():
    """Explicit cheap 0.0 + max_grad_norm=1.0 -> only norm clipping."""
    mgn, mgv, mgln, mode = _resolve(raw_mgv=0.0, max_grad_norm=1.0)
    assert mgn == 1.0
    assert mgv == 0.0
    assert mgln == 0.0
    assert mode == "global_norm"


def test_explicit_positive_overrides_max_grad_norm():
    """Explicit max_grad_value=2.0 with max_grad_norm=1.0 -> elementwise
    wins (existing rule), max_grad_norm zeroed."""
    mgn, mgv, mgln, mode = _resolve(raw_mgv=2.0, max_grad_norm=1.0)
    assert mgn == 0.0
    assert mgv == 2.0
    assert mgln == 0.0
    assert mode == "value"


def test_explicit_positive_alone():
    """User passes max_grad_value=5.0 with no max_grad_norm -> elementwise at 5."""
    mgn, mgv, mgln, mode = _resolve(raw_mgv=5.0, max_grad_norm=0.0)
    assert mgn == 0.0
    assert mgv == 5.0
    assert mgln == 0.0
    assert mode == "value"


def test_explicit_leaf_norm_overrides_max_grad_norm():
    """Explicit max_grad_leaf_norm uses proportional clipping and avoids global norm."""
    mgn, mgv, mgln, mode = _resolve(raw_mgln=1.3, max_grad_norm=1.0)
    assert mgn == 0.0
    assert mgv == 0.0
    assert mgln == 1.3
    assert mode == "leaf_norm"


def test_max_grad_value_wins_over_leaf_norm_when_both_positive():
    """Keep max_grad_value's public elementwise meaning if both knobs are set."""
    mgn, mgv, mgln, mode = _resolve(raw_mgv=2.0, raw_mgln=1.3)
    assert mgn == 0.0
    assert mgv == 2.0
    assert mgln == 0.0
    assert mode == "value"


# -- trainer source assertions (defense-in-depth) ---------------------------


def test_trainer_source_pins_resolution_rule():
    """Source-level pin: trainer.py contains the four-branch resolution.
    Cheap defense against a future refactor silently regressing the rule."""
    import inspect
    from unsloth_zoo.mlx import trainer as T

    src = (
        inspect.getsource(T._resolve_mlx_grad_clipping)
        + inspect.getsource(T.MLXTrainer._train_inner)
    )
    assert "max_grad_value" in src
    assert "max_grad_leaf_norm" in src
    assert 'return 0.0, 0.0, 1.0, "leaf_norm"' in src


def test_source_distinguishes_leaf_norm_from_elementwise_value_clip():
    """Pin the API split: value is elementwise, leaf_norm is proportional."""
    import inspect
    from unsloth_zoo.mlx import trainer as T

    value_src = inspect.getsource(T._clip_grad_by_value)
    leaf_src = inspect.getsource(T._clip_grad_by_leaf_norm)

    assert "mx.clip" in value_src
    assert "mx.sqrt(mx.sum" in leaf_src
    assert "return g * scale.astype(g.dtype)" in leaf_src
    assert "mx.clip" not in leaf_src
