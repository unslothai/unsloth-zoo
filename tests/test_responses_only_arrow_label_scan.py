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

"""The fully masked label scan reads Arrow offsets and values and finds the same rows as the per-row scan."""

import random
import types

import pytest
from datasets import Dataset, Features, Sequence, Value, concatenate_datasets

import unsloth_zoo.dataset_utils as dataset_utils
from unsloth_zoo.dataset_utils import _fully_masked_rows_from_arrow, train_on_responses_only


def _row_by_row(dataset):
    # The scan this replaces, verbatim.
    dropped, idx = [], 0
    for batch in dataset.select_columns(["labels"]).iter(batch_size = 1000):
        for labels in batch["labels"]:
            if labels is not None:
                if getattr(labels, "tolist", None): labels = labels.tolist()
                if not any(l != -100 for l in labels): dropped.append(idx)
            idx += 1
    return dropped


def _labels(n, seed, nulls = False, empties = False):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        width = rng.randrange(1, 40)
        row = [-100] * width
        kind = rng.random()
        if kind < 0.6:
            start = rng.randrange(width)
            row[start:] = [rng.randrange(0, 120) for _ in range(width - start)]
        elif kind < 0.7:
            row[rng.randrange(width)] = rng.choice([0, -1, -99, -101, 7])
        if nulls and rng.random() < 0.05: row = None
        if empties and rng.random() < 0.05: row = []
        out.append(row)
    if nulls: out[-3:] = [None, None, [-100, -100]]
    return out


def _dataset(labels, dtype = "int64"):
    features = Features({"input_ids": Sequence(Value("int64")), "labels": Sequence(Value(dtype))})
    ids = [[1] * (len(x) if x is not None else 1) for x in labels]
    return Dataset.from_dict({"input_ids": ids, "labels": labels}, features = features)


def _shapes():
    base = _dataset(_labels(3000, 0, nulls = True, empties = True))
    yield "plain", base
    yield "int32", _dataset(_labels(2500, 1, nulls = True), "int32")
    yield "int8", _dataset(_labels(2500, 2), "int8")
    yield "shuffled", base.shuffle(seed = 3)
    yield "selected", base.select(range(1500, 2999, 3))
    yield "filtered", base.filter(lambda e: len(e["input_ids"]) % 3 != 0)
    yield "sharded", base.shard(4, 1)
    yield "concatenated", concatenate_datasets([base.select(range(700)), _dataset(_labels(900, 4, empties = True))])
    yield "all_masked", _dataset([[-100] * 5] * 1200 + [[]] * 10)
    yield "none_masked", _dataset([[1, -100]] * 1200)
    yield "numpy", base.with_format("numpy")
    yield "torch", base.with_format("torch")
    yield "python", base.with_format("python")
    yield "labels_only_format", base.with_format("numpy", columns = ["labels"])
    yield "empty", base.select([])


@pytest.mark.parametrize("name,dataset", list(_shapes()), ids = [n for n, _ in _shapes()])
def test_same_rows_as_the_row_by_row_scan(name, dataset):
    try:
        expected = _row_by_row(dataset)
    except ImportError as error:
        # datasets 3.x numpy/torch formatters import torchvision.io.VideoReader, gone from new torchvision.
        pytest.skip(f"this datasets build cannot read the {name} format here: {error}")
    for batch_size in (7, 1000, 10_000):
        got = _fully_masked_rows_from_arrow(dataset, batch_size = batch_size)
        assert got == expected, name
        assert all(type(i) is int for i in got)


def test_unsupported_columns_keep_the_row_by_row_scan():
    nested = Dataset.from_dict({"labels": [[[-100, 1]], [[-100]]]})
    floats = Dataset.from_dict({"labels": [[-100.0, 1.0], [-100.0]]})
    inner_null = Dataset.from_dict({"labels": [[-100, None], [-100]]})
    unsigned = _dataset([[0, 1], [2]], "uint8")
    custom = _dataset([[-100], [1]]).with_transform(lambda b: b)
    hidden = _dataset([[-100], [1]]).with_format("numpy", columns = ["input_ids"])
    pandas = _dataset([[-100], [1]]).with_format("pandas")
    for dataset in (nested, floats, inner_null, unsigned, custom, hidden, pandas):
        assert _fully_masked_rows_from_arrow(dataset) is None


USER_ID, ASSISTANT_ID = 1, 2


class _StubTokenizer:
    def __call__(self, text, add_special_tokens = False):
        ids = {"<|user|>": [USER_ID], "<|assistant|>": [ASSISTANT_ID]}.get(text)
        return types.SimpleNamespace(input_ids = ids if ids is not None else [ord(c) for c in text])

    def convert_ids_to_tokens(self, ids):
        return [str(i) for i in ids]


def _token_rows(n, seed):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        body = [rng.randrange(10, 1000) for _ in range(rng.randrange(3, 12))]
        if rng.random() < 0.3:
            rows.append([USER_ID] + body)
        else:
            rows.append([USER_ID] + body[:2] + [ASSISTANT_ID] + body[2:])
    return rows


def _mask(dataset):
    trainer = types.SimpleNamespace(train_dataset = dataset, eval_dataset = None, args = types.SimpleNamespace(packing = False))
    try:
        train_on_responses_only(trainer, "<|user|>", "<|assistant|>", tokenizer = _StubTokenizer())
    except ValueError as error:
        # Torch-formatted variable length rows are fully masked before the scan on main
        # too; the refusal has to stay the same.
        return str(error)
    out = trainer.train_dataset.with_format(None)
    return out["input_ids"], out["labels"]


@pytest.mark.parametrize("fmt", [None, "numpy", "torch"])
def test_masked_split_matches_the_row_by_row_scan(monkeypatch, fmt, capsys):
    dataset = Dataset.from_dict({"input_ids": _token_rows(4000, 5)})
    if fmt is not None: dataset = dataset.with_format(fmt)
    try:
        dataset[:1]
    except ImportError as error:
        pytest.skip(f"this datasets build cannot read the {fmt} format here: {error}")
    calls = []
    real = dataset_utils._fully_masked_rows_from_arrow
    monkeypatch.setattr(dataset_utils, "_fully_masked_rows_from_arrow", lambda d, **k: calls.append(1) or real(d, **k))
    fast = _mask(dataset)
    fast_log = capsys.readouterr().out
    assert calls == [1]
    monkeypatch.setattr(dataset_utils, "_fully_masked_rows_from_arrow", lambda d, **k: None)
    slow = _mask(dataset)
    assert fast == slow
    assert fast_log == capsys.readouterr().out
    if fmt == "torch" and isinstance(fast, str): return
    assert "Removed" in fast_log
    assert all(ASSISTANT_ID in row for row in fast[0])
