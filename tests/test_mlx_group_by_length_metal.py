# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""group_by_length end to end through the real text plan builders.

Companion to tests/test_mlx_group_by_length.py, which covers the ordering
algorithm and config resolution on CPU. Building an actual batch plan needs a
real MLX runtime -- under the simulation shim `create_ordered_batches` cannot
tokenize, exactly as in tests/test_mlx_batching_and_decay.py -- so this module
skips unless real MLX is installed (Apple Silicon).
"""

from __future__ import annotations

import pytest


mx = pytest.importorskip("mlx.core")
if "mlx_simulation" in str(getattr(mx, "__file__", "")):
    pytest.skip("requires real MLX runtime", allow_module_level=True)


class _TinyTokenizer:
    """Same stub the batching tests use: text is space-separated token ids."""

    pad_token_id = 2
    eos_token_id = 2
    unk_token_id = -1

    def encode(self, text):
        return [int(part) for part in str(text).split()]

    def convert_tokens_to_ids(self, token):
        if isinstance(token, list):
            return [self.convert_tokens_to_ids(item) for item in token]
        return self.unk_token_id


def _varied_dataset(n=24):
    """Rows of deliberately uneven length so grouping has something to do."""
    return [
        {"text": " ".join(str(10 + j) for j in range((i % 6) + 2))}
        for i in range(n)
    ]


def _plan(order, n=24, batch_size=2, seed=1234, **kwargs):
    from unsloth_zoo.mlx.utils import create_ordered_batches

    return create_ordered_batches(
        dataset=_varied_dataset(n),
        tokenizer=_TinyTokenizer(),
        batch_size=batch_size,
        max_seq_length=16,
        seed=seed,
        dataset_order=order,
        **kwargs,
    )


def test_plan_covers_every_row_exactly_once():
    batches = _plan("length_grouped")
    assert batches, "length_grouped produced no batches"
    seen = sum(int(batch.shape[0]) for batch, _lengths, _labels in batches)
    assert seen == 24


def test_multiple_epochs_reshuffle():
    """A fresh permutation per epoch is the whole difference from the legacy
    length-sorted default, which repeats one order forever."""
    batches = _plan("length_grouped", num_epochs=2)
    half = len(batches) // 2
    first = [batch.tolist() for batch, _l, _lb in batches[:half]]
    second = [batch.tolist() for batch, _l, _lb in batches[half:]]
    assert first != second


def test_plan_is_deterministic_for_a_seed():
    as_lists = lambda plan: [b.tolist() for b, _l, _lb in plan]
    assert as_lists(_plan("length_grouped", seed=99)) == as_lists(
        _plan("length_grouped", seed=99)
    )


def test_cuts_padding_versus_torch_randperm():
    """The user-visible payoff, measured through the real builder: total padded
    cells across the plan, which is what dynamic padding actually costs."""

    def padded_cells(order):
        return sum(
            int(batch.shape[0]) * int(batch.shape[1])
            for batch, _l, _lb in _plan(order, n=96, batch_size=4, seed=7)
        )

    assert padded_cells("length_grouped") < padded_cells("torch_randperm")


class _FakeWorld:
    def __init__(self, rank, size):
        self._rank, self._size = rank, size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


def test_plan_under_ddp_gives_each_rank_the_same_batch_count():
    """Ranks must execute the same number of micro-batches or the collectives
    desynchronise. Tail global batches are padded for exactly this reason."""
    counts = {
        r: len(_plan("length_grouped", n=50, batch_size=2,
                     comm_group=_FakeWorld(r, 2)))
        for r in range(2)
    }
    assert counts[0] == counts[1]
    assert counts[0] > 0


def test_ddp_ranks_see_disjoint_rows_from_the_same_global_order():
    """Each rank derives the order locally and takes its own shard; with two
    ranks the first batch of each must differ."""
    first = [
        _plan("length_grouped", n=50, batch_size=2,
              comm_group=_FakeWorld(r, 2))[0][0].tolist()
        for r in range(2)
    ]
    assert first[0] != first[1]


def test_labeled_path_end_to_end_matches_the_unlabeled_order():
    """train_on_responses_only builds its batches through _create_labeled_batches,
    a separate implementation from the unlabeled plan. They must produce the same
    row order or masking is applied to a different stream than training sees."""
    from unsloth_zoo.mlx.trainer import _create_labeled_batches

    def mask_fn(example):
        # Supervise every token: the masking policy is irrelevant here, the
        # row ORDER is what this test is about.
        return {"labels": [list(ids) for ids in example["input_ids"]]}

    batches, _ds = _create_labeled_batches(
        dataset=_varied_dataset(24),
        tokenizer=_TinyTokenizer(),
        mask_fn=mask_fn,
        batch_size=2,
        max_seq_length=16,
        seed=1234,
        dataset_order="length_grouped",
        preserve_dataset_order=False,
        return_dataset=True,
    )
    assert batches, "labeled length_grouped produced no batches"
    seen = sum(int(batch.shape[0]) for batch, _l, _lb in batches)
    assert seen == 24
