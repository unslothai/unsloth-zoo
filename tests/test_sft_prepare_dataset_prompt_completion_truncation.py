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

"""Prompt/completion rows in sft_prepare_dataset cut to max_length like TRL: truncation_mode picks the side and
max_length=None is no limit (it used to raise TypeError); CPU-only, offline."""

import pytest
from datasets import Dataset

from unsloth_zoo.dataset_utils import sft_prepare_dataset

PROMPT = "p1 p2 p3 p4 p5 p6"
COMPLETION = "c1 c2 c3 c4"
IDS = {word: i + 1 for i, word in enumerate((PROMPT + " " + COMPLETION).split())}
FULL = [IDS[word] for word in (PROMPT + " " + COMPLETION).split()]
COMPLETION_IDS = FULL[-4:]


class Tokenizer:
    pad_token_id = 0
    bos_token = None
    eos_token = None
    chat_template = None

    def __call__(self, texts, add_special_tokens = True, **kwargs):
        if isinstance(texts, str):
            return {"input_ids": [IDS[word] for word in texts.split()]}
        return {"input_ids": [[IDS[word] for word in text.split()] for text in texts]}


class Args:
    dataset_text_field = "text"
    dataset_num_proc = None
    completion_only_loss = True
    packing = False

    def __init__(self, max_length, **overrides):
        self.max_length = max_length
        self.__dict__.update(overrides)


class Trainer:
    model = None
    data_collator = None
    completion_only_loss = True


def prepare(max_length, **overrides):
    rows = [{"prompt": PROMPT + " ", "completion": COMPLETION}] * 2
    dataset = sft_prepare_dataset(
        Trainer(), Dataset.from_list(rows), Tokenizer(), Args(max_length, **overrides), False, None, "train",
    )
    row = dataset[0]
    # TRL >= 1.7 folds completion_mask into labels; older TRL keeps the column for its collator.
    keep = row["labels"] if "labels" in row else row["completion_mask"]
    trained = [t for t, k in zip(row["input_ids"], keep) if k not in (0, -100)]
    return list(row["input_ids"]), trained


def test_keep_end_keeps_the_tail():
    ids, trained = prepare(7, truncation_mode = "keep_end")
    assert ids == FULL[-7:]
    assert trained == COMPLETION_IDS


def test_keep_start_is_the_default():
    ids, trained = prepare(7)
    assert ids == FULL[:7]
    assert trained == FULL[6:7]
    assert prepare(7, truncation_mode = "keep_start")[0] == FULL[:7]


def test_no_max_length_keeps_the_whole_row():
    ids, trained = prepare(None)
    assert ids == FULL
    assert trained == COMPLETION_IDS
