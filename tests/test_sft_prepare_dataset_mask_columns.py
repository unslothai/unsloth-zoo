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

"""Pretokenized completion_mask / assistant_masks train only masked tokens on TRL < 1.7 (collator applies masks) and >=
1.7 (_prepare_dataset builds labels); CPU-only, offline."""

import pytest
import torch
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling

import unsloth_zoo.dataset_utils as dataset_utils
from unsloth_zoo.dataset_utils import sft_prepare_dataset

trl_sft = pytest.importorskip("trl.trainer.sft_trainer")
trl_data_utils = pytest.importorskip("trl.data_utils")


class Tokenizer:
    pad_token_id = 0
    bos_token = None
    chat_template = None


class Args:
    max_length = 16
    dataset_text_field = "text"
    remove_unused_columns = True
    packing_strategy = "bfd"


class Trainer:
    def __init__(self, completion_only_loss = True):
        self.model = None
        self.completion_only_loss = completion_only_loss
        self.data_collator = trl_sft.DataCollatorForLanguageModeling(pad_token_id = 0)


ROWS = [
    {"input_ids": [10, 11, 12, 13, 14], "completion_mask": [0, 0, 0, 1, 1], "assistant_masks": [0, 0, 0, 1, 1]},
    {"input_ids": [20, 21, 22],         "completion_mask": [0, 1, 1],       "assistant_masks": [0, 1, 1]},
]
ALL_TOKENS = [11, 12, 13, 14, 21, 22]


def trained_tokens(dataset, packing, completion_only_loss = True):
    kwargs = {"pad_token_id": 0, "padding_free": packing}
    if "completion_only_loss" in trl_sft.DataCollatorForLanguageModeling.__dataclass_fields__:
        kwargs["completion_only_loss"] = completion_only_loss
    collator = trl_sft.DataCollatorForLanguageModeling(**kwargs)
    batch = collator([dataset[i] for i in range(len(dataset))])
    ids, labels = batch["input_ids"].flatten().tolist(), batch["labels"].flatten().tolist()
    # Position 0 of a row is never a target (no previous token), whatever the masks say.
    return sorted(t for t, y in zip(ids, labels) if y != -100 and t not in (10, 20))


@pytest.fixture
def real_pack_dataset(monkeypatch):
    monkeypatch.setattr(dataset_utils, "pack_dataset", trl_data_utils.pack_dataset, raising = False)


@pytest.mark.parametrize("packing", [False, True])
@pytest.mark.parametrize("mask", ["completion_mask", "assistant_masks"])
def test_mask_column_trains_only_masked_tokens(mask, packing, real_pack_dataset):
    dataset = Dataset.from_list([{"input_ids": r["input_ids"], mask: r[mask]} for r in ROWS])
    out = sft_prepare_dataset(Trainer(), dataset, Tokenizer(), Args(), packing, None, "train")
    assert trained_tokens(out, packing) == [13, 14, 21, 22]


def test_completion_mask_ignored_without_completion_only_loss(real_pack_dataset):
    dataset = Dataset.from_list([{"input_ids": r["input_ids"], "completion_mask": r["completion_mask"]} for r in ROWS])
    trainer = Trainer(completion_only_loss = False)
    out = sft_prepare_dataset(trainer, dataset, Tokenizer(), Args(), False, None, "train")
    assert trained_tokens(out, False, completion_only_loss = False) == ALL_TOKENS


@pytest.mark.parametrize("mask", ["completion_mask", "assistant_masks"])
def test_mask_column_keeps_trl_collator(mask):
    trainer = Trainer()
    collator = trainer.data_collator
    dataset = Dataset.from_list([{"input_ids": r["input_ids"], mask: r[mask]} for r in ROWS])
    sft_prepare_dataset(trainer, dataset, Tokenizer(), Args(), False, None, "train")
    assert trainer.data_collator is collator


def test_plain_input_ids_still_get_the_lm_collator():
    trainer = Trainer()
    dataset = Dataset.from_list([{"input_ids": r["input_ids"]} for r in ROWS])
    out = sft_prepare_dataset(trainer, dataset, Tokenizer(), Args(), False, None, "train")
    assert type(trainer.data_collator) is DataCollatorForLanguageModeling
    assert "labels" not in out.column_names


class Trl018Collator:
    """TRL 0.18's collator: labels from input_ids, completion_mask applied, a "labels" column never read."""

    def __init__(self, pad_token_id, completion_only_loss = True, pad_to_multiple_of = None):
        self.pad_token_id = pad_token_id
        self.completion_only_loss = completion_only_loss

    def __call__(self, examples):
        def pad(rows, value):
            return torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(r) for r in rows], batch_first = True, padding_value = value
            )

        labels = pad([e["input_ids"] for e in examples], -100)
        if self.completion_only_loss and "completion_mask" in examples[0]:
            labels[pad([e["completion_mask"] for e in examples], 0) == 0] = -100
        return {"input_ids": pad([e["input_ids"] for e in examples], self.pad_token_id), "labels": labels}


@pytest.fixture
def trl_018_collator(monkeypatch):
    monkeypatch.setattr(trl_sft, "DataCollatorForLanguageModeling", Trl018Collator)
    monkeypatch.setattr(dataset_utils, "_TRL_COLLATOR_APPLIES_MASKS", None)
    yield
    dataset_utils._TRL_COLLATOR_APPLIES_MASKS = None


def test_trl_018_collator_is_detected(trl_018_collator):
    assert dataset_utils.trl_collator_applies_masks() is False
    assert dataset_utils.trl_collator_ignores_labels() is True


def test_trl_018_completion_mask_reaches_its_collator(trl_018_collator):
    dataset = Dataset.from_list([{"input_ids": r["input_ids"], "completion_mask": r["completion_mask"]} for r in ROWS])
    out = sft_prepare_dataset(Trainer(), dataset, Tokenizer(), Args(), False, None, "train")
    labels = Trl018Collator(pad_token_id = 0)([out[i] for i in range(len(out))])["labels"]
    assert sorted(x for x in labels.flatten().tolist() if x != -100) == [13, 14, 21, 22]


@pytest.mark.parametrize("collator", ["flattening", "seq2seq"])
def test_trl_018_label_reading_collator_gets_labels_only(trl_018_collator, collator):
    from transformers import DataCollatorForSeq2Seq, DataCollatorWithFlattening

    trainer = Trainer()
    trainer.data_collator = DataCollatorWithFlattening() if collator == "flattening" else DataCollatorForSeq2Seq(None)
    dataset = Dataset.from_list([{"input_ids": r["input_ids"], "completion_mask": r["completion_mask"]} for r in ROWS])
    out = sft_prepare_dataset(trainer, dataset, Tokenizer(), Args(), False, None, "train")
    assert set(out.column_names) == {"input_ids", "labels"}
    batch = DataCollatorWithFlattening(return_tensors = "pt")([out[i] for i in range(len(out))])
    ids, labels = batch["input_ids"].flatten().tolist(), batch["labels"].flatten().tolist()
    assert sorted(t for t, y in zip(ids, labels) if y != -100) == [13, 14, 21, 22]


def test_trl_018_collator_found_in_init_before_it_is_stored(trl_018_collator):
    from transformers import DataCollatorWithFlattening

    class InitTrainer(Trainer):
        def __init__(self):
            self.model = None
            self.completion_only_loss = True
            data_collator = DataCollatorWithFlattening()  # noqa: F841 -- read from this frame, as TRL 0.18's __init__
            dataset = Dataset.from_list([{"input_ids": r["input_ids"], "completion_mask": r["completion_mask"]} for r in ROWS])
            def prepare(split):  # a frame between, like TRL's dict eval comprehension before Python 3.12
                return sft_prepare_dataset(self, dataset, Tokenizer(), Args(), False, None, split)
            self.out = sft_prepare_dataset(self, dataset, Tokenizer(), Args(), False, None, "train")
            self.eval_out = prepare("eval")

    trainer = InitTrainer()
    assert set(trainer.out.column_names) == {"input_ids", "labels"}
    assert set(trainer.eval_out.column_names) == {"input_ids", "labels"}
