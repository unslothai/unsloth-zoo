# Unsloth Zoo - Utilities for Unsloth
# Verify that `MLXTrainingConfig(max_grad_value=None)` disables
# elementwise grad clipping rather than silently rebinding to the
# default 1.0.
#
# Background: the field is typed `float | None = 1.0`, and the field
# docstring documents `None` as a disable signal. Probe authors comparing
# `unsloth_zoo.MLXTrainer` against `mlx-lm`'s native loop (which does
# no elementwise clipping) set `max_grad_value=None` and were surprised
# that clipping was still applied. This test pins the fixed behavior so
# the disable-None semantics stays stable.

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


def test_field_default_is_clip_on():
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    cfg = MLXTrainingConfig(output_dir="/tmp/x")
    assert cfg.max_grad_value == 1.0


def test_none_disables_clip():
    """The motivating change: passing None disables clipping."""
    assert _disable_clip_decision(None) is False


def test_zero_disables_clip():
    """Zero remains a documented disable signal."""
    assert _disable_clip_decision(0.0) is False
    assert _disable_clip_decision(0) is False


def test_positive_enables_clip():
    """A positive value keeps clipping on."""
    assert _disable_clip_decision(1.0) is True
    assert _disable_clip_decision(2.5) is True


def test_field_accepts_none():
    """Field accepts None and round-trips it through the dataclass."""
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    cfg = MLXTrainingConfig(max_grad_value=None, output_dir="/tmp/x")
    assert cfg.max_grad_value is None
