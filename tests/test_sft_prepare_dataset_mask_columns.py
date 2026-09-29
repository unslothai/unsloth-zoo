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

"""Pretokenized rows carrying completion_mask / assistant_masks keep their masks.

TRL's SFT collator reads those columns and sets the masked tokens to -100.
sft_prepare_dataset used to swap it for transformers' collator, which ignores
them, and packing selected them away. CPU-only and offline.
"""

import pytest
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
    def __init__(self):
        self.model = None
        self.data_collator = trl_sft.DataCollatorForLanguageModeling(pad_token_id = 0)


ROWS = [
    {"input_ids": [10, 11, 12, 13, 14], "completion_mask": [0, 0, 0, 1, 1], "assistant_masks": [0, 0, 0, 1, 1]},
    {"input_ids": [20, 21, 22],         "completion_mask": [0, 1, 1],       "assistant_masks": [0, 1, 1]},
]


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
    sft_prepare_dataset(trainer, dataset, Tokenizer(), Args(), False, None, "train")
    assert type(trainer.data_collator) is DataCollatorForLanguageModeling


def test_packing_keeps_mask_columns(monkeypatch):
    monkeypatch.setattr(dataset_utils, "pack_dataset", trl_data_utils.pack_dataset, raising = False)
    dataset = Dataset.from_list(ROWS)
    packed = sft_prepare_dataset(Trainer(), dataset, Tokenizer(), Args(), True, None, "train")
    assert {"completion_mask", "assistant_masks"} <= set(packed.column_names)
    # bfd packs both rows into one 8-token sequence; the masks travel with their tokens.
    row = packed[0]
    ids, mask = row["input_ids"], row["assistant_masks"]
    assert sorted(t for t, m in zip(ids, mask) if m) == [13, 14, 21, 22]
