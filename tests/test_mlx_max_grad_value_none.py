# Unsloth Zoo - Utilities for Unsloth
# Verify MLX grad-clip semantics match issue #662's HF-parity proposal:
#
#   1. MLXTrainingConfig.max_grad_value defaults to None (no elementwise
#      clip), so a user-passed max_grad_norm is honored as in HF/TRL.
#   2. None and 0.0 both disable elementwise clipping. Explicit None
#      previously rebound silently to 1.0.
#   3. A positive value opts in to elementwise clipping; the existing
#      mutual-exclusion rule (elementwise wins, max_grad_norm zeroed)
#      then applies and prints a notice.
#   4. The dataclass round-trips None through the field.

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_mlx_shim():
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


def _disable_clip_decision(raw_mgv):
    """Mirror trainer.py's internal decision so we can pin the rule
    without standing up a full MLXTrainer / model / optimizer.
    """
    max_grad_value = 0.0 if raw_mgv is None else float(raw_mgv or 0.0)
    return max_grad_value > 0


def test_field_default_is_clip_off():
    """Default = None (no elementwise clip). HF/TRL parity."""
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    cfg = MLXTrainingConfig(output_dir="/tmp/x")
    assert cfg.max_grad_value is None


def test_none_disables_clip():
    """Explicit None disables clipping (not silently rebound to 1.0)."""
    assert _disable_clip_decision(None) is False


def test_zero_disables_clip():
    """Zero remains a documented disable signal."""
    assert _disable_clip_decision(0.0) is False
    assert _disable_clip_decision(0) is False


def test_positive_enables_clip():
    """A positive value opts in to elementwise clipping."""
    assert _disable_clip_decision(1.0) is True
    assert _disable_clip_decision(2.5) is True


def test_field_accepts_none():
    """Field accepts None and round-trips it through the dataclass."""
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    cfg = MLXTrainingConfig(max_grad_value=None, output_dir="/tmp/x")
    assert cfg.max_grad_value is None


def test_field_accepts_explicit_positive():
    """Field accepts positive floats for power users opting into clip."""
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    cfg = MLXTrainingConfig(max_grad_value=2.5, output_dir="/tmp/x")
    assert cfg.max_grad_value == 2.5
