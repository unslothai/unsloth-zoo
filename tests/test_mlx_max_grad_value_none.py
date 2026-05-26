# Unsloth Zoo - Utilities for Unsloth
# Pin MLXTrainingConfig.max_grad_value resolution:
#   * None (default) -> cheap MLX elementwise clip at 1.0, unless
#     max_grad_norm > 0 is also passed (then global-norm wins).
#   * 0.0 -> explicitly disabled.
#   * positive -> explicit elementwise opt-in; overrides max_grad_norm.

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_mlx_shim():
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


def _resolve(raw_mgv, max_grad_norm):
    """Mirror trainer.py's internal resolution. Returns the (max_grad_value,
    max_grad_norm) pair the step function will actually use."""
    user_set = raw_mgv is not None
    if user_set:
        mgv = float(raw_mgv or 0.0)
        if max_grad_norm > 0 and mgv > 0:
            max_grad_norm = 0.0
    elif max_grad_norm > 0:
        mgv = 0.0
    else:
        mgv = 1.0
    return mgv, max_grad_norm


# -- field defaults ---------------------------------------------------------


def test_field_default_is_none_sentinel():
    """Default is None (a sentinel meaning 'use MLX cheap default')."""
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    cfg = MLXTrainingConfig(output_dir="/tmp/x")
    assert cfg.max_grad_value is None


def test_field_accepts_none():
    """Field accepts None and round-trips through the dataclass."""
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    cfg = MLXTrainingConfig(max_grad_value=None, output_dir="/tmp/x")
    assert cfg.max_grad_value is None


def test_field_accepts_explicit_positive():
    """Field accepts positive floats for power users opting in."""
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    cfg = MLXTrainingConfig(max_grad_value=2.5, output_dir="/tmp/x")
    assert cfg.max_grad_value == 2.5


# -- resolution semantics ---------------------------------------------------


def test_default_uses_cheap_elementwise():
    """Default (None, max_grad_norm=0.0) -> elementwise clip at 1.0."""
    mgv, mgn = _resolve(raw_mgv=None, max_grad_norm=0.0)
    assert mgv == 1.0
    assert mgn == 0.0


def test_user_max_grad_norm_wins_over_default():
    """User passes max_grad_norm=1.0 with default max_grad_value=None ->
    global-norm clipping wins, elementwise disabled. HF/TRL parity."""
    mgv, mgn = _resolve(raw_mgv=None, max_grad_norm=1.0)
    assert mgv == 0.0
    assert mgn == 1.0


def test_explicit_zero_disables_elementwise():
    """Explicit 0.0 disables elementwise. With no max_grad_norm,
    nothing clips."""
    mgv, mgn = _resolve(raw_mgv=0.0, max_grad_norm=0.0)
    assert mgv == 0.0
    assert mgn == 0.0


def test_explicit_zero_lets_max_grad_norm_through():
    """Explicit max_grad_value=0.0 + max_grad_norm=1.0 -> only norm clipping."""
    mgv, mgn = _resolve(raw_mgv=0.0, max_grad_norm=1.0)
    assert mgv == 0.0
    assert mgn == 1.0


def test_explicit_positive_overrides_max_grad_norm():
    """Explicit max_grad_value=2.0 with max_grad_norm=1.0 -> elementwise
    wins (existing rule), max_grad_norm zeroed."""
    mgv, mgn = _resolve(raw_mgv=2.0, max_grad_norm=1.0)
    assert mgv == 2.0
    assert mgn == 0.0


def test_explicit_positive_alone():
    """User passes max_grad_value=5.0 with no max_grad_norm -> elementwise at 5."""
    mgv, mgn = _resolve(raw_mgv=5.0, max_grad_norm=0.0)
    assert mgv == 5.0
    assert mgn == 0.0


# -- trainer source assertions (defense-in-depth) ---------------------------


def test_trainer_source_pins_resolution_rule():
    """Source-level pin: trainer.py contains the four-branch resolution.
    Cheap defense against a future refactor silently regressing the rule."""
    import inspect
    from unsloth_zoo.mlx import trainer as T

    src = inspect.getsource(T.MLXTrainer.train) + inspect.getsource(T.MLXTrainer._train_inner)
    assert "_user_set_mgv = _raw_mgv is not None" in src
    assert "elif max_grad_norm > 0:" in src
    assert "max_grad_value = 1.0" in src
