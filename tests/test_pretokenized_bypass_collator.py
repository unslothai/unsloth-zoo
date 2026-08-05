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

"""What the pretokenized bypass is allowed to hand to the text path.

The bypass lets a text-only fine-tune of a multimodal model reach dataset-level
masking instead of being refused. Two things it must not do on the way:

* fire for rows that still carry images, since the text path replaces the
  collator and the user's image handling would go with it; and
* leave behind a `DataCollatorForSeq2Seq` that pads through a processor, which
  has no `.pad`, so the run dies on the first batch.

CPU-pure and offline: everything here is a local stub, no weights are loaded.
"""

import re

import pytest
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, DataCollatorWithPadding

from unsloth_zoo.dataset_utils import train_on_responses_only


INSTRUCTION_PART = "<|user|>"
RESPONSE_PART = "<|assistant|>"
USER_ID, ASSISTANT_ID, PAD_ID = 1, 2, 0
ROW = [USER_ID, 10, 11, ASSISTANT_ID, 20, 21]


class _Encoding(dict):
    """Mapping (what `datasets.map` wants) that also answers `.input_ids`."""
    @property
    def input_ids(self):
        return self["input_ids"]


class StubTokenizer:
    """A text tokenizer: it can pad, which is the whole point below."""
    padding_side = "right"
    pad_token_id = PAD_ID
    model_input_names = ["input_ids", "attention_mask"]

    @staticmethod
    def _ids(text):
        # Markers are single tokens; everything else is one token per character.
        ids = []
        for piece in re.split(f"({re.escape(INSTRUCTION_PART)}|{re.escape(RESPONSE_PART)})", text):
            if piece == INSTRUCTION_PART:   ids.append(USER_ID)
            elif piece == RESPONSE_PART:    ids.append(ASSISTANT_ID)
            else:                           ids.extend(ord(c) for c in piece)
        return ids

    def __call__(self, text, add_special_tokens = False, **kwargs):
        if isinstance(text, (list, tuple)):
            batch = [self._ids(t) for t in text]
            return _Encoding(input_ids = batch,
                             attention_mask = [[1] * len(ids) for ids in batch])
        return _Encoding(input_ids = self._ids(text))

    def pad(self, features, padding = True, max_length = None,
            pad_to_multiple_of = None, return_tensors = None, **kwargs):
        width = max(len(f["input_ids"]) for f in features)
        return {
            "input_ids": [f["input_ids"] + [PAD_ID] * (width - len(f["input_ids"]))
                          for f in features],
        }


class StubProcessor:
    """What a multimodal model hands the trainer as its tokenizer. No `.pad`."""
    def __init__(self):
        self.image_processor = object()
        self.tokenizer = StubTokenizer()


class MyVisionCollator:
    """A user's own collator, holding the processor so it can batch images."""
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        return {"mine": True}


class StubTrainer:
    def __init__(self, collator, train_dataset):
        self.data_collator = collator
        self.train_dataset = train_dataset
        self.eval_dataset = None
        self.processing_class = StubProcessor()
        self.args = type("Args", (), {
            "packing": False, "max_length": 64, "dataset_text_field": "text",
        })()


def _text_rows():
    return Dataset.from_dict({"input_ids": [list(ROW), list(ROW)]})


def _multimodal_rows():
    return Dataset.from_dict({
        "input_ids": [list(ROW), list(ROW)],
        "pixel_values": [[[0.0] * 4], [[0.0] * 4]],
        "image_grid_thw": [[1, 2, 2], [1, 2, 2]],
    })


# ---- images must not be dropped -------------------------------------------

@pytest.mark.parametrize("extra_column", [
    "pixel_values", "pixel_values_videos", "image_grid_thw", "input_features",
])
def test_multimodal_rows_are_still_refused(extra_column):
    """`input_ids` alone does not make a row text-only. The text path swaps the
    collator for a text one, so firing here would silently drop the user's
    image handling and mis-shape (or fail to batch) the image columns."""
    collator = MyVisionCollator(StubProcessor())
    dataset = Dataset.from_dict({
        "input_ids": [list(ROW), list(ROW)],
        extra_column: [[0.0], [0.0]],
    })
    trainer = StubTrainer(collator, dataset)

    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert trainer.data_collator is collator, "the user's collator was replaced"


def test_multimodal_iterable_rows_are_still_refused():
    """No column_names, so the row peek has to make the same call."""
    trainer = StubTrainer(MyVisionCollator(StubProcessor()),
                          _multimodal_rows().to_iterable_dataset())
    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)


# ---- the run has to survive the first batch --------------------------------

def test_a_processor_backed_seq2seq_collator_is_rebuilt():
    """`DataCollatorForSeq2Seq(tokenizer = processor)` is the collator the
    bypass exists for. Left alone it pads via `self.tokenizer.pad`, which a
    processor does not have, so the first batch dies with
    `AttributeError: ... object has no attribute 'pad'`."""
    processor = StubProcessor()
    trainer = StubTrainer(DataCollatorForSeq2Seq(tokenizer = processor),
                          _text_rows())

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    collator = out.data_collator
    assert isinstance(collator, DataCollatorForSeq2Seq)
    assert hasattr(collator.tokenizer, "pad"), \
        f"still padding through {type(collator.tokenizer).__name__}"

    rows = [out.train_dataset[i] for i in range(len(out.train_dataset))]
    batch = collator(rows)     # used to raise AttributeError
    assert "input_ids" in batch and "labels" in batch


def test_a_tokenizer_backed_seq2seq_collator_is_left_alone():
    """The pre-existing path: a seq2seq collator that already pads through a
    real tokenizer keeps its own settings."""
    collator = DataCollatorForSeq2Seq(tokenizer = StubTokenizer(),
                                      pad_to_multiple_of = 8)
    trainer = StubTrainer(collator, _text_rows())
    trainer.data_collator = collator
    # Not a vision collator, so this exercises the plain text path.
    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)
    assert out.data_collator is collator
    assert out.data_collator.pad_to_multiple_of == 8


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# ---- the follow-on cases the first round of this fix missed ---------------

def _text_only_trainer():
    return StubTrainer(MyVisionCollator(StubProcessor()), _text_rows())


def test_a_multimodal_eval_split_still_refuses():
    """The collator is swapped for the whole trainer, so a text-only train set
    beside a multimodal eval set must not clear the guard."""
    trainer = _text_only_trainer()
    trainer.eval_dataset = _multimodal_rows()
    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)


def test_a_multimodal_split_in_an_eval_dict_still_refuses():
    trainer = _text_only_trainer()
    trainer.eval_dataset = {"a": _text_rows(), "b": _multimodal_rows()}
    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)


def test_a_text_only_eval_split_is_fine():
    trainer = _text_only_trainer()
    trainer.eval_dataset = _text_rows()
    train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)


@pytest.mark.parametrize("column", [
    "image_pixel_values", "audio_input_features", "audio_embed_sizes",
    "flattened_patches", "high_res_pixel_values",
])
def test_processor_specific_column_names_are_refused_too(column):
    """phi4_multimodal, pix2struct and kosmos-2.5 do not spell these the way
    the common families do."""
    dataset = Dataset.from_dict({
        "input_ids": [list(ROW), list(ROW)],
        column: [[0.0], [0.0]],
    })
    trainer = StubTrainer(MyVisionCollator(StubProcessor()), dataset)
    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)


def test_a_processor_backed_collator_is_rebuilt_even_with_packing():
    """It raises on its first batch whether or not packing is on, so the
    tokenizer repair cannot be gated on packing the way the swap is."""
    trainer = StubTrainer(DataCollatorForSeq2Seq(tokenizer = StubProcessor()),
                          _text_rows())
    trainer.args.packing = True
    train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)
    assert hasattr(trainer.data_collator.tokenizer, "pad")


def test_the_rebuild_keeps_the_settings_the_caller_chose():
    collator = DataCollatorForSeq2Seq(tokenizer = StubProcessor(),
                                      pad_to_multiple_of = 8,
                                      label_pad_token_id = -123)
    trainer = StubTrainer(collator, _text_rows())
    train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)
    assert trainer.data_collator.pad_to_multiple_of == 8
    assert trainer.data_collator.label_pad_token_id == -123
    assert hasattr(trainer.data_collator.tokenizer, "pad")


# ---- raw text-only splits reach the text path too --------------------------

def _raw_text_rows(n = 2):
    return Dataset.from_dict({
        "text": [f"{INSTRUCTION_PART}q{i}{RESPONSE_PART}a{i}" for i in range(n)],
    })


def test_a_raw_text_only_eval_split_is_allowed_through():
    """A text-only eval split that is not pretokenized is tokenized by the text
    path with the inner text tokenizer, and the same dataset-level masking then
    applies to it. Refusing it would make adding evaluation to a supported
    text-only VLM run fail unless the user pretokenizes eval by hand."""
    trainer = _text_only_trainer()
    trainer.eval_dataset = _raw_text_rows()

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    eval_split = out.eval_dataset
    assert "labels" in eval_split.column_names, "eval was never tokenized/masked"
    row = eval_split[0]
    unmasked = [i for i, l in enumerate(row["labels"]) if l != -100]
    assert unmasked, "eval labels are all -100"
    # Only the answer is trained on; the question and both markers stay masked.
    assert [row["input_ids"][i] for i in unmasked] == [ord("a"), ord("0")]
    assert row["labels"][row["input_ids"].index(ASSISTANT_ID)] == -100


def test_a_raw_text_only_split_in_an_eval_dict_is_allowed_through():
    trainer = _text_only_trainer()
    trainer.eval_dataset = {"pretok": _text_rows(), "raw": _raw_text_rows()}
    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)
    assert "labels" in out.eval_dataset["raw"].column_names


def test_a_raw_split_carrying_images_is_still_refused():
    """Raw does not mean text-only: an image column means the text path would
    throw the user's image handling away with the collator."""
    trainer = _text_only_trainer()
    trainer.eval_dataset = Dataset.from_dict({
        "text": [f"{INSTRUCTION_PART}q{RESPONSE_PART}a"],
        "images": [[0.0]],
    })
    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)


def test_a_raw_split_with_no_text_column_is_still_refused():
    """Conversational multimodal data carries its images inside the turns, so
    the columns alone look text-only. There is nothing for the text path to
    tokenize, so this must keep the old refusal rather than empty the split."""
    trainer = _text_only_trainer()
    trainer.eval_dataset = Dataset.from_dict({"messages": [[{"role": "user"}]]})
    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)


# ---- non-seq2seq collators that pad through a processor --------------------

def test_a_processor_backed_padding_collator_is_repaired_under_packing():
    """`DataCollatorWithPadding(tokenizer = processor)` pads through
    `self.tokenizer.pad` exactly like the seq2seq one does, so packing being on
    must not leave it attached: the first batch dies with
    `AttributeError: ... object has no attribute 'pad'` either way."""
    collator = DataCollatorWithPadding(tokenizer = StubProcessor())
    trainer = StubTrainer(collator, _text_rows())
    trainer.args.packing = True

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert out.data_collator is not collator, "still padding through the processor"
    assert hasattr(out.data_collator.tokenizer, "pad")
    rows = [out.train_dataset[i] for i in range(len(out.train_dataset))]
    batch = out.data_collator(rows)   # used to raise AttributeError
    assert "input_ids" in batch and "labels" in batch


def test_a_collator_holding_only_a_processor_is_repaired_under_packing():
    """TRL's `DataCollatorForVisionLanguageModeling` keeps the processor under
    `.processor`, not `.tokenizer`, and has the same problem."""
    trainer = _text_only_trainer()          # MyVisionCollator holds `.processor`
    trainer.args.packing = True
    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)
    assert isinstance(out.data_collator, DataCollatorForSeq2Seq)
    assert hasattr(out.data_collator.tokenizer, "pad")


class PackingCollator:
    """Stand-in for TRL's packing `DataCollatorForLanguageModeling`: it takes a
    bare `pad_token_id` and holds no tokenizer or processor at all."""
    def __init__(self):
        self.pad_token_id = PAD_ID

    def __call__(self, features):
        return {"packed": True}


def test_a_packing_collator_that_holds_no_tokenizer_is_left_alone():
    """The reason the swap is gated on packing: a packing collator builds the
    packed `position_ids` itself, and DataCollatorForSeq2Seq would not. Nothing
    above may widen far enough to catch one."""
    collator = PackingCollator()
    trainer = StubTrainer(collator, _text_rows())
    trainer.args.packing = True
    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)
    assert out.data_collator is collator


def test_a_real_tokenizer_padding_collator_is_untouched_under_packing():
    """It can pad on its own, so packing keeps it as before."""
    collator = DataCollatorWithPadding(tokenizer = StubTokenizer())
    trainer = StubTrainer(collator, _text_rows())
    trainer.args.packing = True
    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)
    assert out.data_collator is collator
