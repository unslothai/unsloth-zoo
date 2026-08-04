# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""adam_epsilon on the MLX trainer (HF TrainingArguments parity).

HF exposes ``adam_epsilon`` and the CUDA path forwards it to the optimizer
(``unsloth/trainer.py:350``, alongside the betas). MLX carried ``adam_beta1`` /
``adam_beta2`` but not the epsilon, and because ``MLXTrainingConfig`` rejects
unknown kwargs, passing it raised ``TypeError`` before training began rather
than being quietly ignored.

CPU-pure: builds optimizers through the MLX simulation shim, no weights and no
Metal. ``optimizer._kw`` is the shim's record of the kwargs MLX was called with.
"""

from __future__ import annotations

import pytest
import torch  # noqa: F401  (the simulation shim runs MLX ops on torch)


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    """Install-only, matching tests/test_mlx_double_quant_reject.py. Deliberately
    no teardown: popping mlx / unsloth_zoo.mlx modules on the way out breaks
    later files that hold references to them (this file sorts before
    test_mlx_generate.py, which does)."""
    from mlx_simulation import simulate_mlx_on_torch

    simulate_mlx_on_torch()


def _trainer(**config_kwargs):
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    class DummyModel:
        def trainable_parameters(self):
            return {}

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.model = DummyModel()
    trainer.args = MLXTrainingConfig(**config_kwargs)
    return trainer


def test_config_accepts_adam_epsilon():
    """The regression itself: MLXTrainingConfig validates its kwargs, so
    adam_epsilon=... used to raise TypeError before a step ever ran."""
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    assert MLXTrainingConfig(adam_epsilon=1e-6).adam_epsilon == pytest.approx(1e-6)
    assert MLXTrainingConfig().adam_epsilon is None


@pytest.mark.parametrize("optim_name", ["adamw", "adam"])
def test_adam_epsilon_reaches_the_optimizer(optim_name):
    optimizer = _trainer(optim=optim_name, adam_epsilon=1e-6)._build_optimizer(
        total_steps=4
    )
    assert optimizer._kw["eps"] == pytest.approx(1e-6)


@pytest.mark.parametrize("optim_name", ["adamw", "adam"])
def test_unset_epsilon_leaves_the_mlx_default_untouched(optim_name):
    """None means "not requested": nothing is forwarded, so MLX's own default
    (1e-8, matching HF) stands and the unset path is bitwise unchanged."""
    optimizer = _trainer(optim=optim_name)._build_optimizer(total_steps=4)
    assert "eps" not in optimizer._kw


def test_epsilon_composes_with_betas():
    optimizer = _trainer(
        optim="adamw", adam_beta1=0.85, adam_beta2=0.95, adam_epsilon=1e-7,
    )._build_optimizer(total_steps=4)
    assert optimizer._kw["eps"] == pytest.approx(1e-7)
    assert optimizer._kw["betas"] == (pytest.approx(0.85), pytest.approx(0.95))


@pytest.mark.parametrize("optim_name", ["sgd", "muon", "lion"])
def test_epsilon_is_not_forwarded_to_epsilon_free_optimizers(optim_name):
    """SGD/Muon/Lion accept no epsilon, so forwarding one would be a TypeError.
    HF ignores adam_epsilon for these too, so setting it stays harmless."""
    optimizer = _trainer(
        optim=optim_name, adam_epsilon=1e-6,
    )._build_optimizer(total_steps=4)
    assert "eps" not in getattr(optimizer, "_kw", {})


def test_adafactor_does_not_receive_the_scalar_epsilon():
    """MLX's Adafactor eps is a 2-tuple with different meaning from HF's scalar,
    so adam_epsilon must not leak into it. The CUDA path scopes it the same way."""
    optimizer = _trainer(
        optim="adafactor", adam_epsilon=1e-6,
    )._build_optimizer(total_steps=4)
    assert "eps" not in getattr(optimizer, "_kw", {})


def test_appended_field_stays_an_exact_suffix():
    """MLXTrainingConfig binds positional args by field order, so a new field
    must be appended last and keep _MLX_CONFIG_OPTIONAL_COPY_FIELDS a suffix of
    the field list; inserting mid-list would shift every later field's slot."""
    from dataclasses import fields as dataclass_fields
    from unsloth_zoo.mlx.trainer import (
        MLXTrainingConfig,
        _MLX_CONFIG_OPTIONAL_COPY_FIELDS,
    )

    names = [f.name for f in dataclass_fields(MLXTrainingConfig) if f.init]
    assert "adam_epsilon" in _MLX_CONFIG_OPTIONAL_COPY_FIELDS
    # Deliberately not `names[-1] == "adam_epsilon"`: the convention is that
    # each new field is appended and registered here, so pinning this field as
    # the permanent last one would fire on the next unrelated addition. The
    # suffix property below is the invariant that actually has to hold.
    tail = tuple(names[-len(_MLX_CONFIG_OPTIONAL_COPY_FIELDS):])
    assert tail == _MLX_CONFIG_OPTIONAL_COPY_FIELDS
