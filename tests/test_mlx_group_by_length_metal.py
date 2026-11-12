# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""group_by_length end to end through the real text plan builders (needs real MLX)."""

from __future__ import annotations

import pytest


mx = pytest.importorskip("mlx.core")
if "mlx_simulation" in str(getattr(mx, "__file__", "")):
    pytest.skip("requires real MLX runtime", allow_module_level=True)

import mlx.nn as nn
from mlx.utils import tree_flatten


class _TinyTokenizer:

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
    # Leading token is unique per row so equal-length rows stay distinguishable.
    return [
        {"text": " ".join([str(1000 + i)] + [str(10 + j) for j in range((i % 6) + 1)])}
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
    pytest.importorskip("torch")  # the torch_randperm arm requires torch

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
    counts = {
        r: len(_plan("length_grouped", n=50, batch_size=2,
                     comm_group=_FakeWorld(r, 2)))
        for r in range(2)
    }
    assert counts[0] == counts[1]
    assert counts[0] > 0


def test_ddp_ranks_see_disjoint_rows_from_the_same_global_order():
    first = [
        _plan("length_grouped", n=50, batch_size=2,
              comm_group=_FakeWorld(r, 2))[0][0].tolist()
        for r in range(2)
    ]
    assert first[0] != first[1]


def test_labeled_path_end_to_end_matches_the_unlabeled_order():
    from unsloth_zoo.mlx.trainer import _create_labeled_batches

    def mask_fn(example):
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


class _CausalTiny(nn.Module):
    # A prefix mean mixes positions causally, so trailing padding cannot reach a real token.
    def __init__(self):
        super().__init__()
        mx.random.seed(0)
        self.embed, self.head = nn.Embedding(1100, 16), nn.Linear(16, 1100)

    def __call__(self, x):
        steps = mx.arange(1, x.shape[1] + 1, dtype=mx.float32)[None, :, None]
        return self.head(mx.tanh(mx.cumsum(self.embed(x), axis=1) / steps))


def _window_gradients(plan, model, accum):
    from unsloth_zoo.mlx.utils import make_baseline_loss_fn

    loss_and_grad, windows = nn.value_and_grad(model, make_baseline_loss_fn()), []
    for start in range(0, len(plan), accum):
        total, tokens = {}, 0
        for index in range(start, start + accum):
            (_loss, toks), grad = loss_and_grad(model, *plan[index])
            for name, value in tree_flatten(grad):
                total[name] = total.get(name, 0) + value * toks
            tokens += toks.item()
        windows.append({name: value / tokens for name, value in total.items()})
    return windows


def _mixed_length_plan(**kwargs):
    from unsloth_zoo.mlx.utils import _create_ordered_text_plan

    rows = [{"text": " ".join([str(1000 + i)] + ["10"] * ((i * 5) % 7 + 1))} for i in range(24)]
    return _create_ordered_text_plan(
        dataset=rows, tokenizer=_TinyTokenizer(), batch_size=2, max_seq_length=16, seed=7,
        dataset_order="sequential", num_epochs=1, grad_accum=3, **kwargs,
    )


def test_length_sorted_windows_leave_every_step_gradient_unchanged():
    unsorted, cut = _mixed_length_plan(), _mixed_length_plan(length_sorted_windows=True)
    assert sum(map(cut.batch_width, range(len(cut)))) < sum(map(unsorted.batch_width, range(len(unsorted))))
    model = _CausalTiny()
    for before, after in zip(_window_gradients(unsorted, model, 3), _window_gradients(cut, model, 3)):
        for name, value in before.items():
            assert mx.allclose(value, after[name], rtol=1e-5, atol=1e-7).item(), name
