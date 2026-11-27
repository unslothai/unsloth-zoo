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

"""train_on_responses_only keeps the batch-width rounding of the text collator it replaces (unslothai/unsloth#19567).

Padded SFT batches run through TRL's DataCollatorForLanguageModeling, which applies SFTConfig.pad_to_multiple_of.
The swap to DataCollatorForSeq2Seq used to start from defaults, so batches were padded to their longest row only.
CPU-pure, offline: a real fast tokenizer built from a word-level vocabulary.
"""

import pytest
from datasets import Dataset
from transformers import (
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    PreTrainedTokenizerFast,
)

from unsloth_zoo.dataset_utils import train_on_responses_only


INSTRUCTION_PART = "<|user|>"
RESPONSE_PART = "<|assistant|>"


def _tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit

    words = ["<pad>", "<unk>", INSTRUCTION_PART, RESPONSE_PART] + [f"w{i}" for i in range(32)]
    backend = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token = "<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = backend, pad_token = "<pad>", unk_token = "<unk>",
        additional_special_tokens = [INSTRUCTION_PART, RESPONSE_PART],
    )
    tokenizer.padding_side = "right"
    return tokenizer


def _rows(tokenizer):
    texts = [
        f"{INSTRUCTION_PART} w1 w2 {RESPONSE_PART} w3",
        f"{INSTRUCTION_PART} w4 w5 w6 w7 {RESPONSE_PART} w8 w9 w10 w11 w12",
    ]
    return Dataset.from_dict({"input_ids": [tokenizer(t)["input_ids"] for t in texts]})


class _Trainer:
    def __init__(self, collator, tokenizer):
        self.data_collator = collator
        self.processing_class = tokenizer
        self.train_dataset = _rows(tokenizer)
        self.eval_dataset = None
        self.args = type("Args", (), {
            "packing": False, "max_length": 64, "dataset_text_field": "text",
        })()


def _width(trainer):
    batch = trainer.data_collator([trainer.train_dataset[i] for i in range(len(trainer.train_dataset))])
    assert batch["input_ids"].shape == batch["labels"].shape
    return batch["input_ids"].shape[1]


def _trl_text_collator(**kwargs):
    sft = pytest.importorskip("trl.trainer.sft_trainer")
    return sft.DataCollatorForLanguageModeling(**kwargs)


def test_trl_text_collator_pad_to_multiple_of_survives_the_swap():
    """The reported case: SFTConfig(pad_to_multiple_of = 64) with padded (not padding-free) batches."""
    tokenizer = _tokenizer()
    trainer = _Trainer(_trl_text_collator(pad_token_id = tokenizer.pad_token_id, pad_to_multiple_of = 64), tokenizer)
    assert _width(trainer) == 64

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert isinstance(out.data_collator, DataCollatorForSeq2Seq)
    assert out.data_collator.pad_to_multiple_of == 64
    assert _width(out) == 64


def test_trl_text_collator_without_rounding_still_pads_to_the_longest_row():
    tokenizer = _tokenizer()
    trainer = _Trainer(_trl_text_collator(pad_token_id = tokenizer.pad_token_id), tokenizer)

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert out.data_collator.pad_to_multiple_of is None
    assert _width(out) == len(out.train_dataset[1]["input_ids"])


def test_only_the_rounding_is_taken_from_a_trl_collator():
    """TRL's fields other than pad_to_multiple_of are not DataCollatorForSeq2Seq settings."""
    tokenizer = _tokenizer()
    collator = _trl_text_collator(pad_token_id = tokenizer.pad_token_id, pad_to_multiple_of = 8)

    out = train_on_responses_only(_Trainer(collator, tokenizer), INSTRUCTION_PART, RESPONSE_PART)

    assert out.data_collator.pad_to_multiple_of == 8
    assert out.data_collator.padding is True
    assert out.data_collator.max_length is None
    assert out.data_collator.label_pad_token_id == -100
    assert _width(out) == 16


def test_padding_collator_on_a_real_tokenizer_keeps_its_settings():
    """A pad-delegating collator built on a tokenizer that can pad is replaced too, not only one holding a processor."""
    tokenizer = _tokenizer()
    collator = DataCollatorWithPadding(tokenizer, padding = "max_length", max_length = 24, pad_to_multiple_of = 8)

    out = train_on_responses_only(_Trainer(collator, tokenizer), INSTRUCTION_PART, RESPONSE_PART)

    assert out.data_collator.padding == "max_length"
    assert out.data_collator.max_length == 24
    assert out.data_collator.pad_to_multiple_of == 8
    assert _width(out) == 24


def test_a_custom_collator_attribute_is_not_read_as_a_setting():
    """Same-named fields on a user's own collator need not mean the same thing, so they are still left behind."""
    class MyCollator:
        pad_to_multiple_of = 64
        def __call__(self, features):
            return {"mine": True}

    tokenizer = _tokenizer()
    out = train_on_responses_only(_Trainer(MyCollator(), tokenizer), INSTRUCTION_PART, RESPONSE_PART)

    assert isinstance(out.data_collator, DataCollatorForSeq2Seq)
    assert out.data_collator.pad_to_multiple_of is None


@pytest.mark.parametrize("model_max_length, expected", [(32, 32), (int(1e30), None)])
def test_max_length_padding_without_a_max_length(model_max_length, expected):
    """padding = "max_length" with no max_length: inputs and labels end at the same width.

    A real model_max_length is the target. The no-limit sentinel pads to the longest row instead.
    """
    tokenizer = _tokenizer()
    tokenizer.model_max_length = model_max_length
    collator = DataCollatorWithPadding(tokenizer, padding = "max_length")

    out = train_on_responses_only(_Trainer(collator, tokenizer), INSTRUCTION_PART, RESPONSE_PART)

    width = _width(out)
    assert width == (expected or len(out.train_dataset[1]["input_ids"]))
