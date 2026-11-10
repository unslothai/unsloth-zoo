# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""fix_untrained_tokens counts input_ids of a plain datasets.Dataset from Arrow instead of a batched .map, which
builds Python lists of every column; counts and the rewritten rows must match the .map path exactly."""

import itertools

import numpy as np
import pytest
import torch

datasets = pytest.importorskip("datasets")

from unsloth_zoo import tokenizer_utils

V = 16


def _reference_counts(train_dataset):
    final_counts = np.zeros(V, dtype = np.int64)

    def mapping(examples):
        counter = np.fromiter(itertools.chain.from_iterable(examples["input_ids"]), dtype = np.int32)
        np.add.at(final_counts, counter, 1)

    tokenizer_utils._count_input_ids(train_dataset, mapping)
    return final_counts


def _base():
    rng = np.random.default_rng(0)
    rows = [
        {"input_ids" : rng.integers(0, V, rng.integers(1, 12)).tolist(), "labels" : [-100], "text" : "x"}
        for _ in range(200)
    ]
    return datasets.Dataset.from_list(rows)


_VARIANTS = {
    "plain"          : lambda d: d,
    "select"         : lambda d: d.select(range(5, 150, 3)),
    "shuffle_select" : lambda d: d.shuffle(seed = 1).select(range(40)),
    "filter"         : lambda d: d.filter(lambda r: len(r["input_ids"]) % 2 == 0),
    "torch"          : lambda d: d.with_format("torch"),
    "empty"          : lambda d: d.select([]),
}


@pytest.mark.parametrize("flush_ids", [1 << 20, 5])
@pytest.mark.parametrize("batch_rows", [10_000, 7])
@pytest.mark.parametrize("name", list(_VARIANTS))
def test_arrow_counts_match_the_map_path(name, batch_rows, flush_ids, monkeypatch):
    monkeypatch.setattr(tokenizer_utils, "_COUNT_INPUT_IDS_ARROW_BATCH_ROWS", batch_rows)
    monkeypatch.setattr(tokenizer_utils, "_COUNT_INPUT_IDS_ARROW_FLUSH_IDS", flush_ids)
    train_dataset = _VARIANTS[name](_base())
    final_counts = np.zeros(V, dtype = np.int64)

    assert tokenizer_utils._count_input_ids_arrow(train_dataset, final_counts) is True
    assert final_counts.tolist() == _reference_counts(train_dataset).tolist()


@pytest.mark.parametrize("rows", [[{"input_ids" : [1, V + 3]}], [{"input_ids" : [1, -1]}]])
def test_out_of_range_ids_fall_back_untouched(rows):
    final_counts = np.zeros(V, dtype = np.int64)
    assert tokenizer_utils._count_input_ids_arrow(datasets.Dataset.from_list(rows), final_counts) is False
    assert final_counts.sum() == 0


def test_custom_transform_and_plain_lists_fall_back():
    final_counts = np.zeros(V, dtype = np.int64)
    transformed = _base().with_transform(lambda b: {"input_ids" : [[1] for _ in b["input_ids"]]})
    assert tokenizer_utils._count_input_ids_arrow(transformed, final_counts) is False
    assert tokenizer_utils._count_input_ids_arrow([{"input_ids" : [1]}], final_counts) is False
    assert final_counts.sum() == 0


class _Tokenizer:
    chat_template = None

    def __len__(self):
        return V

    def convert_ids_to_tokens(self, ids):
        return [f"<t{i}>" for i in ids]


class _Model:
    def __init__(self):
        generator = torch.Generator().manual_seed(0)
        self.embedding = torch.nn.Embedding(V, 4)
        self.lm_head = torch.nn.Linear(4, V, bias = False)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.randn(V, 4, generator = generator))
            self.lm_head.weight.copy_(torch.randn(V, 4, generator = generator))
            self.embedding.weight[V - 3:] = 0
            self.lm_head.weight[V - 3:] = 0

        class _Config:
            _name_or_path = "unsloth/tiny-test-model"

        self.config = _Config()

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.lm_head


def test_fix_untrained_tokens_skips_the_batched_map_and_matches_it(monkeypatch):
    train_dataset = _base().shuffle(seed = 2).select(range(120))
    fast = _Model()
    tokenizer_utils.fix_untrained_tokens(fast, _Tokenizer(), train_dataset)

    monkeypatch.setattr(tokenizer_utils, "_count_input_ids_arrow", lambda *args: False)
    slow = _Model()
    tokenizer_utils.fix_untrained_tokens(slow, _Tokenizer(), train_dataset)
    assert torch.equal(fast.embedding.weight, slow.embedding.weight)
    assert torch.equal(fast.lm_head.weight, slow.lm_head.weight)
    assert not torch.equal(fast.embedding.weight[V - 3:], torch.zeros(3, 4))

    monkeypatch.undo()
    calls = []
    monkeypatch.setattr(datasets.Dataset, "map", lambda self, *args, **kwargs: calls.append(kwargs.get("desc")))
    tokenizer_utils.fix_untrained_tokens(_Model(), _Tokenizer(), train_dataset)
    assert "Counting untrained tokens" not in calls


def test_a_late_out_of_range_batch_leaves_the_counts_untouched(monkeypatch):
    monkeypatch.setattr(tokenizer_utils, "_COUNT_INPUT_IDS_ARROW_BATCH_ROWS", 2)
    train_dataset = datasets.Dataset.from_list([{"input_ids" : [1, 2]}] * 5 + [{"input_ids" : [V + 1]}])
    final_counts = np.zeros(V, dtype = np.int64)
    assert tokenizer_utils._count_input_ids_arrow(train_dataset, final_counts) is False
    assert final_counts.tolist() == [0] * V
