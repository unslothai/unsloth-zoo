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

"""The fully masked row filter and the label builder read mask columns a batch at a time on a map-style
Dataset, with the same rows and labels as row by row; an IterableDataset stays row by row. CPU-only."""

from types import SimpleNamespace

import pytest
from datasets import Dataset

import unsloth_zoo.dataset_utils as dataset_utils
from unsloth_zoo.dataset_utils import sft_prepare_dataset

ROWS = [
    {"input_ids": [10, 11, 12, 13, 14], "assistant_masks": [0, 0, 0, 1, 1], "completion_mask": [0, 1, 1, 1, 1]},
    {"input_ids": [20, 21, 22], "assistant_masks": [0, 1, 0], "completion_mask": [0, 0, 1]},
    {"input_ids": [30, 31], "assistant_masks": [0, 0], "completion_mask": [1, 1]},
]


def _prepare(dataset, completion_only_loss):
    trainer = SimpleNamespace(model = None, completion_only_loss = completion_only_loss, data_collator = None)
    tokenizer = SimpleNamespace(pad_token_id = 0, bos_token = None, chat_template = None)
    args = SimpleNamespace(max_length = 16, dataset_text_field = "text", remove_unused_columns = True, packing_strategy = "bfd")
    return sft_prepare_dataset(trainer, dataset, tokenizer, args, False, None, "train")


@pytest.fixture
def spy(monkeypatch):
    monkeypatch.setattr(dataset_utils, "trl_collator_applies_masks", lambda: False)
    monkeypatch.setattr(dataset_utils, "trl_collator_ignores_labels", lambda collator = None: False)
    calls = []
    original_map = Dataset.map
    def map(self, function = None, *args, **kwargs):
        calls.append(kwargs.get("batched", False))
        return original_map(self, function, *args, **kwargs)
    monkeypatch.setattr(Dataset, "map", map)
    return calls


@pytest.mark.parametrize("completion_only_loss", [False, True])
def test_labels_are_built_a_batch_at_a_time(spy, completion_only_loss):
    out = _prepare(Dataset.from_list(ROWS), completion_only_loss)
    assert spy == [True], spy
    masks = ["assistant_masks", "completion_mask"] if completion_only_loss else ["assistant_masks"]
    expected = [[t if all(r[m][i] for m in masks) else -100 for i, t in enumerate(r["input_ids"])] for r in ROWS]
    assert out["labels"] == expected
    # completion_mask is no label input without completion_only_loss, so it stays a column.
    assert out.column_names == (["input_ids", "labels"] if completion_only_loss else ["input_ids", "completion_mask", "labels"])


def test_formatted_dataset_keeps_row_by_row_labels(spy):
    out = _prepare(Dataset.from_list(ROWS).with_format("python"), False)
    assert out["labels"][0] == [-100, -100, -100, 13, 14]
    plain = _prepare(Dataset.from_list(ROWS).with_format("numpy"), False)
    assert [list(x) for x in plain["labels"]] == out["labels"]


def test_iterable_dataset_keeps_row_by_row_labels(monkeypatch):
    monkeypatch.setattr(dataset_utils, "trl_collator_applies_masks", lambda: False)
    monkeypatch.setattr(dataset_utils, "trl_collator_ignores_labels", lambda collator = None: False)
    out = _prepare(Dataset.from_list(ROWS).to_iterable_dataset(), False)
    assert [row["labels"] for row in out] == [[-100, -100, -100, 13, 14], [-100, 21, -100], [-100, -100]]


def test_fully_masked_rows_are_dropped_by_a_batched_filter(monkeypatch):
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
    except Exception as e:  # offline runner
        pytest.skip(reason = f"offline runner, tokenizer download failed: {e}")
    # Keep the mask columns (TRL >= 1.7 would turn them into labels): this test reads assistant_masks.
    monkeypatch.setattr(dataset_utils, "trl_collator_applies_masks", lambda: True)
    calls = []
    original_filter = Dataset.filter
    def filter(self, function = None, *args, **kwargs):
        calls.append((kwargs.get("batched", False), kwargs.get("input_columns")))
        return original_filter(self, function, *args, **kwargs)
    monkeypatch.setattr(Dataset, "filter", filter)
    rows = [
        {"messages": [{"role": "user", "content": "hi " * (40 if i % 3 == 0 else 1)}, {"role": "assistant", "content": f"ok {i}"}]}
        for i in range(12)
    ]
    trainer = SimpleNamespace(chat_template = None, _unsloth_assistant_mask_fallback = True, data_collator = None,
                              model = None, completion_only_loss = None)
    args = SimpleNamespace(max_length = 32, dataset_text_field = "text", remove_unused_columns = True,
                           assistant_only_loss = True, dataset_num_proc = 1, packing_strategy = "bfd")
    out = sft_prepare_dataset(trainer, Dataset.from_list(rows), tokenizer, args, False, None, "train")
    assert calls == [(True, ["assistant_masks"])], calls
    # Rows whose 40-word prompt fills the 32 tokens lose every assistant token and are dropped, in order.
    assert len(out) == 8
    assert all(any(m) for m in out["assistant_masks"])
