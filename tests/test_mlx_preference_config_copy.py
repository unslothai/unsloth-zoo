# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Regression test: a legacy config dump must keep its warmup_ratio schedule.

MLXTrainingConfig.__init__ treats a wholesale copy carrying a DEFAULT
warmup_steps beside a non-default warmup_ratio as implicit, so the ratio wins.
That check (``copied_all_fields``) tolerates fields appended after the copy was
written. The ORPO/DPO fields were not in that tolerated set, so a dump from
before they existed failed the check, warmup_steps was read as explicit, and a
defaulted 5 silently overrode the ratio schedule.

This is the keyword path -- ``MLXTrainingConfig(**legacy_dict)`` -- not the
positional one; the copy tolerance exists precisely to serve round-tripped
configs.
"""

from __future__ import annotations

import sys

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    shim_prefixes = ("mlx", "mlx_lm", "mlx_vlm")
    real_mlx_modules = {
        name: module
        for name, module in sys.modules.items()
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in shim_prefixes)
    }
    from mlx_simulation import simulate_mlx_on_torch
    from mlx_simulation.mlx_stub import _MLXFinder
    simulate_mlx_on_torch()
    for name in list(sys.modules):
        if name == "unsloth_zoo.mlx" or name.startswith("unsloth_zoo.mlx."):
            sys.modules.pop(name, None)
    yield
    for name in list(sys.modules):
        if (
            name == "unsloth_zoo.mlx" or name.startswith("unsloth_zoo.mlx.")
            or any(name == prefix or name.startswith(f"{prefix}.") for prefix in shim_prefixes)
        ):
            sys.modules.pop(name, None)
    sys.meta_path[:] = [
        finder for finder in sys.meta_path
        if not isinstance(finder, _MLXFinder)
    ]
    sys.modules.update(real_mlx_modules)


PREFERENCE_FIELDS = ("orpo_beta", "dpo_beta", "reference_free", "append_eos")


def _legacy_dump(omit):
    """Every init field except ``omit``, at its default, as an older dump would be."""
    from dataclasses import fields
    from unsloth_zoo.mlx.trainer import (
        MLXTrainingConfig, _MLX_CONFIG_OPTIONAL_COPY_FIELDS,
    )

    tolerated = set(_MLX_CONFIG_OPTIONAL_COPY_FIELDS) | {
        "compile_max_variants", "label_smoothing_factor", "report_grad_norm",
    }
    return {
        field.name: getattr(MLXTrainingConfig, field.name)
        for field in fields(MLXTrainingConfig)
        if field.init and field.name not in omit and field.name not in tolerated
    }


def _resolved_warmup(config, total_steps=100):
    from unsloth_zoo.mlx.trainer import MLXTrainer

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.args = config
    return trainer._resolve_warmup_steps(total_steps)


def test_legacy_dump_without_preference_fields_keeps_ratio_schedule():
    """warmup_ratio must win: the omitted fields are appended, not user intent."""
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    dump = _legacy_dump(omit=PREFERENCE_FIELDS)
    dump["warmup_steps"] = MLXTrainingConfig.warmup_steps   # the default, 5
    dump["warmup_ratio"] = 0.1                              # non-default

    config = MLXTrainingConfig(**dump)
    assert config._unsloth_mlx_warmup_steps_explicit is False
    assert _resolved_warmup(config, 100) == 10              # ceil(0.1 * 100)


def test_dump_including_preference_fields_is_unchanged():
    """The same dump WITH the fields already behaved correctly; pin that."""
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    dump = _legacy_dump(omit=())
    dump["warmup_steps"] = MLXTrainingConfig.warmup_steps
    dump["warmup_ratio"] = 0.1

    config = MLXTrainingConfig(**dump)
    assert config._unsloth_mlx_warmup_steps_explicit is False
    assert _resolved_warmup(config, 100) == 10


def test_explicit_non_default_warmup_steps_still_wins():
    """A genuinely explicit warmup_steps must still override the ratio."""
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    dump = _legacy_dump(omit=PREFERENCE_FIELDS)
    dump["warmup_steps"] = 7                                # NOT the default
    dump["warmup_ratio"] = 0.1

    config = MLXTrainingConfig(**dump)
    assert config._unsloth_mlx_warmup_steps_explicit is True
    assert _resolved_warmup(config, 100) == 7


def test_preference_fields_are_tolerated_by_the_copy_check():
    """Pin the tolerance directly so a field rename cannot silently drop it."""
    from dataclasses import fields
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    declared = {f.name for f in fields(MLXTrainingConfig) if f.init}
    for name in PREFERENCE_FIELDS:
        assert name in declared, f"{name} is no longer a config field"

    dump = _legacy_dump(omit=PREFERENCE_FIELDS)
    config = MLXTrainingConfig(**dump)
    for name in PREFERENCE_FIELDS:
        assert getattr(config, name) == getattr(MLXTrainingConfig, name)
