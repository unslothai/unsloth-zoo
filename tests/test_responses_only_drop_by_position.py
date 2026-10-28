# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""``train_on_responses_only`` drops fully masked rows by the positions its label scan found.

The scan over the ``labels`` column already lists every fully masked row. The filter that
removes them used to decode every column of every row again to re-derive the same answer;
it now keeps a row exactly when its position is not on that list. Survivors must be the
same rows in the same order, including on shuffled / selected views, formatted datasets and
under ``num_proc``; a custom transform still re-reads the labels.

CPU-pure and offline: the tokenizer is a local stub.
"""

import random
import types

import pytest
from datasets import Dataset

from unsloth_zoo.dataset_utils import train_on_responses_only

USER_ID, ASSISTANT_ID = 1, 2


class _StubTokenizer:
    def __call__(self, text, add_special_tokens = False):
        ids = {"<|user|>": [USER_ID], "<|assistant|>": [ASSISTANT_ID]}.get(text)
        return types.SimpleNamespace(input_ids = ids if ids is not None else [ord(c) for c in text])

    def convert_ids_to_tokens(self, ids):
        return [str(i) for i in ids]


def _rows(n, frac_masked, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        body = [rng.randrange(10, 1000) for _ in range(8)]
        if rng.random() < frac_masked:
            out.append([USER_ID] + body + [rng.randrange(10, 1000)])  # no response marker: fully masked
        else:
            out.append([USER_ID] + body[:3] + [ASSISTANT_ID] + body[3:])
    return out


def _run(dataset, capsys = None):
    trainer = types.SimpleNamespace(
        train_dataset = dataset, eval_dataset = None,
        args = types.SimpleNamespace(packing = False),
    )
    train_on_responses_only(trainer, "<|user|>", "<|assistant|>", tokenizer = _StubTokenizer())
    return trainer.train_dataset


def _expected(dataset):
    return [row for row in dataset.with_format(None)["input_ids"] if ASSISTANT_ID in row]


class _Unreadable(dict):
    def __getitem__(self, key):
        raise AssertionError(f"filter predicate read column {key!r}")

    def get(self, key, default = None):
        raise AssertionError(f"filter predicate read column {key!r}")


def test_filter_decides_by_position_without_reading_rows(monkeypatch):
    seen = []
    real_filter = Dataset.filter

    def spy(self, function, *args, **kwargs):
        seen.append((function, kwargs))
        return real_filter(self, function, *args, **kwargs)

    monkeypatch.setattr(Dataset, "filter", spy)
    dataset = Dataset.from_dict({"input_ids": _rows(50, 0.3, 0)})
    out = _run(dataset)
    assert out["input_ids"] == _expected(dataset)
    assert len(seen) == 1
    function, kwargs = seen[0]
    # The predicate answers from row positions alone: handing it rows that cannot be read
    # must still give the same keep mask the run used.
    masked = {i for i, row in enumerate(dataset["input_ids"]) if ASSISTANT_ID not in row}
    if kwargs.get("batched") and kwargs.get("with_indices"):
        mask = function(_Unreadable(), list(range(len(dataset))))
    else:
        mask = [function(_Unreadable()) for _ in range(len(dataset))]
    assert list(mask) == [i not in masked for i in range(len(dataset))]


@pytest.mark.parametrize(
    "make",
    [
        lambda: Dataset.from_dict({"input_ids": _rows(300, 0.25, 1)}),
        lambda: Dataset.from_dict({"input_ids": _rows(2500, 0.4, 2)}).shuffle(seed = 3).select(range(7, 2400, 3)),
        lambda: Dataset.from_dict({"input_ids": _rows(300, 0.25, 4)}).with_format("torch"),
        lambda: Dataset.from_dict({"input_ids": _rows(1001, 0.5, 5)}),
        lambda: Dataset.from_dict({"input_ids": _rows(1, 0.0, 6) + _rows(40, 1.0, 7)}),
    ],
    ids = ["plain", "shuffled_selected", "torch_format", "batch_edge", "one_survivor"],
)
def test_survivors_are_the_rows_with_a_response(make):
    dataset = make()
    out = _run(dataset)
    got = [list(map(int, row)) for row in out.with_format(None)["input_ids"]]
    assert got == _expected(dataset)


def test_custom_transform_still_rereads_labels(monkeypatch):
    seen = []
    real_filter = Dataset.filter

    def spy(self, function, *args, **kwargs):
        seen.append(kwargs)
        return real_filter(self, function, *args, **kwargs)

    monkeypatch.setattr(Dataset, "filter", spy)
    dataset = Dataset.from_dict({"input_ids": _rows(60, 0.3, 8)}).with_transform(lambda batch: batch)
    out = _run(dataset)
    assert len(out) == len(_expected(dataset.with_format(None)))
    assert seen and not seen[0].get("with_indices")
