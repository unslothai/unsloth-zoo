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

"""Regression tests: preference batches must expose their one-pass length.

``create_preference_batches`` is eager, so it returns materialized batches
rather than a lazy ``FiniteTextBatchPlan``. The trainer still needs the
micro-batch count of ONE dataset pass to force HF's epoch-final optimizer step:
``_mlx_epoch_microbatches`` and ``_callback_batches_per_epoch`` both read it as
``getattr(batches, "cycle_length", None)``.

Returning a bare ``list`` left that None under ``max_steps > 0``, so a ragged
pass never force-updated on its tail -- 3 preference batches with
``gradient_accumulation_steps=2`` should update on batch 2 and then apply the
epoch tail (batch 3) by itself, but instead accumulated batch 3 together with
the next pass's batch 1. That changes the effective gradient and moves the
callback/save/eval epoch boundaries off the SFT finite-plan path.

``PreferenceBatchList`` carries the count. The pass length is deliberately NOT
clamped to ``len(batches)``: ``num_epochs`` expansion makes the returned list
several passes long, and a ``num_batches`` horizon can truncate it below one
pass -- the same convention the SFT/VLM plans use.
"""

from __future__ import annotations

import math
import pickle
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


class _StubTokenizer:
    """Minimal encoder: char codes, so prompt ids prefix prompt+completion ids."""

    eos_token_id = 7
    bos_token = None

    def encode(self, text, add_special_tokens=True):
        return [(ord(ch) % 90) + 8 for ch in text]


def _dataset(n_pairs):
    return [
        {"prompt": f"prompt number {i:03d} ", "chosen": "good answer",
         "rejected": "bad answer"}
        for i in range(n_pairs)
    ]


def _build(n_pairs, batch_size, **kwargs):
    from unsloth_zoo.mlx.utils import create_preference_batches

    return create_preference_batches(
        _dataset(n_pairs), _StubTokenizer(), batch_size,
        max_seq_length=64, **kwargs,
    )


def _args(max_steps=0, num_train_epochs=1.0):
    return types.SimpleNamespace(
        max_steps=max_steps, num_train_epochs=num_train_epochs,
    )


# --------------------------------------------------------------------------
# 1. The count itself.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_pairs,batch_size", [(6, 2), (5, 2), (3, 2), (1, 4), (7, 3)])
def test_cycle_length_is_one_pass(n_pairs, batch_size):
    batches = _build(n_pairs, batch_size)
    expected = math.ceil(n_pairs / batch_size)
    assert batches.cycle_length == expected
    assert len(batches) == expected  # single pass, no horizon: they agree


def test_cycle_length_survives_num_batches_truncation():
    """A sub-one-pass horizon must not shrink the reported pass length."""
    batches = _build(10, 2, num_batches=2)  # one pass would be 5 batches
    assert len(batches) == 2
    assert batches.cycle_length == 5


def test_cycle_length_reports_one_pass_not_all_epochs():
    """num_epochs materializes several passes; the count stays per-pass."""
    batches = _build(6, 2, dataset_order="torch_randperm", num_epochs=3, seed=0)
    assert len(batches) == 9        # 3 passes x 3 batches
    assert batches.cycle_length == 3


def test_empty_dataset_reports_no_cycle():
    batches = _build(0, 2)
    assert len(batches) == 0
    assert batches.cycle_length is None


# --------------------------------------------------------------------------
# 2. What the trainer does with it -- codex's ragged-tail scenario.
# --------------------------------------------------------------------------

def test_bare_list_reports_no_epoch_boundary():
    """Document the bug: a plain list gives the helper nothing to work with."""
    from unsloth_zoo.mlx.trainer import _mlx_epoch_microbatches

    batches = list(_build(3, 1))  # strip the subclass -> pre-fix behavior
    assert _mlx_epoch_microbatches(_args(max_steps=5), batches) is None


def test_max_steps_run_exposes_ragged_epoch_boundary():
    """3 batches, grad_accum=2, max_steps>0: the tail boundary is visible."""
    from unsloth_zoo.mlx.trainer import _mlx_epoch_microbatches

    batches = _build(3, 1)
    assert len(batches) == 3
    assert _mlx_epoch_microbatches(_args(max_steps=5), batches) == 3


def test_epoch_based_run_still_reports_a_boundary():
    from unsloth_zoo.mlx.trainer import _mlx_epoch_microbatches

    batches = _build(3, 1)
    assert _mlx_epoch_microbatches(_args(num_train_epochs=1.0), batches) == 3


# --------------------------------------------------------------------------
# 3. The container must stay a drop-in list.
# --------------------------------------------------------------------------

def test_behaves_like_the_list_it_replaces():
    batches = _build(6, 2)
    assert isinstance(batches, list)
    plain = list(batches)
    assert len(batches) == len(plain)
    # The trainer modulo-cycles this; indexing must be unchanged.
    assert all(batches[i % len(batches)] is plain[i % len(plain)] for i in range(13))


def test_not_mistaken_for_a_lazy_plan():
    """Eager batches must not enter the refetch / lazy-plan branches."""
    from unsloth_zoo.mlx.trainer import (
        _EAGER_REFETCHABLE_PLAN_TYPES, _FINITE_BATCH_PLAN_TYPES,
    )

    batches = _build(4, 2)
    assert not isinstance(batches, _FINITE_BATCH_PLAN_TYPES)
    assert not isinstance(batches, _EAGER_REFETCHABLE_PLAN_TYPES)


def test_pickle_round_trip_keeps_cycle_length():
    from unsloth_zoo.mlx.utils import PreferenceBatchList

    original = PreferenceBatchList([1, 2, 3], cycle_length=2)
    restored = pickle.loads(pickle.dumps(original))
    assert list(restored) == [1, 2, 3]
    assert restored.cycle_length == 2
    assert isinstance(restored, PreferenceBatchList)
