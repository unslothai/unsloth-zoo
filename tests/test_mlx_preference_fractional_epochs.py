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

"""Regression test: fractional num_train_epochs on the preference path.

num_train_epochs is a float. The epoch-based preference builder truncated it
with int(), so 1.5 and 0.5 both collapsed onto a single pass -- 1.5 trained
short and 0.5 trained a whole extra pass. The SFT ordered path
(_create_ordered_text_plan) already builds to a whole-accumulation-window
budget; the preference builder now uses the same formula.

Reference values were measured from create_ordered_batches at the same shape
(6 rows, batch 2, accum 2 -> 3 micro-batches and 2 steps per pass):

    epochs   batches   steps
      0.5          2       1
      1.0          3       2
      1.5          5       3
      2.0          6       4

The whole-number rows are pinned alongside the fractional ones so a budget
regression fails loudly rather than only shifting the fractional cases.
"""

from __future__ import annotations

import sys
import types

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


N_ROWS, BATCH_SIZE, GRAD_ACCUM = 6, 2, 2

# (num_train_epochs, batches, steps) measured from the SFT ordered path.
EXPECTED = [
    (0.5, 2, 1),
    (1.0, 3, 2),
    (1.5, 5, 3),
    (2.0, 6, 4),
]


class _Tok:
    eos_token_id = 7
    bos_token = None

    def encode(self, text, add_special_tokens=True):
        return [(ord(ch) % 90) + 8 for ch in text]


def _rows(n=N_ROWS):
    return [{"prompt": f"p{i} ", "chosen": "good", "rejected": "bad"}
            for i in range(n)]


def _args(epochs):
    return types.SimpleNamespace(
        max_steps=0, num_train_epochs=epochs,
        gradient_accumulation_steps=GRAD_ACCUM,
    )


def _build(epochs):
    from unsloth_zoo.mlx.utils import create_preference_batches

    return create_preference_batches(
        _rows(), _Tok(), BATCH_SIZE, 64, dataset_order="torch_randperm",
        num_epochs=epochs, grad_accum=GRAD_ACCUM, seed=0,
    )


@pytest.mark.parametrize("epochs,batches,steps", EXPECTED)
def test_epoch_budget_matches_the_sft_ordered_path(epochs, batches, steps):
    from unsloth_zoo.mlx.trainer import _resolve_training_steps

    plan = _build(epochs)
    assert len(plan) == batches, f"{epochs} epochs: wrong batch count"
    resolved = _resolve_training_steps(
        _args(epochs), plan, None, includes_epochs=True,
    )
    assert resolved == steps, f"{epochs} epochs: wrong step budget"


def test_fractional_epochs_are_not_collapsed_onto_one_pass():
    """The reported defect: 0.5, 1.0 and 1.5 all produced one pass."""
    counts = {epochs: len(_build(epochs)) for epochs in (0.5, 1.0, 1.5)}
    assert counts[0.5] < counts[1.0] < counts[1.5], counts


def test_one_pass_is_still_the_cycle_length():
    """The pass length itself is unchanged by the epoch budget."""
    for epochs, _batches, _steps in EXPECTED:
        assert _build(epochs).cycle_length == 3


def test_whole_epochs_are_byte_identical_to_the_integer_build():
    """A whole count must produce exactly the passes it always did."""
    import math

    for epochs in (1.0, 2.0, 3.0):
        plan = _build(epochs)
        assert len(plan) == int(epochs) * 3, f"{epochs} epochs changed"
        assert not math.isclose(len(plan), 0)


def test_default_order_reshuffles_batch_visits_each_epoch():
    from unsloth_zoo.mlx.utils import create_preference_batches

    plan = create_preference_batches(
        _rows(), _Tok(), BATCH_SIZE, 64, dataset_order="default",
        num_epochs=3, grad_accum=GRAD_ACCUM, seed=11,
    )
    assert len(plan) == 9

    def signature(batch_data):
        batch, _lengths, _labels = batch_data
        return tuple(int(row[1]) for row in batch.tolist()[:BATCH_SIZE])

    passes = [
        [signature(item) for item in plan[start:start + 3]]
        for start in range(0, len(plan), 3)
    ]
    first_pass = create_preference_batches(
        _rows(), _Tok(), BATCH_SIZE, 64, dataset_order="default", seed=11,
    )
    assert passes[0] == [signature(item) for item in first_pass]
    assert all(sorted(block) == sorted(passes[0]) for block in passes[1:])
    assert any(block != passes[0] for block in passes[1:])
