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
masking instead of being refused. Two things it must not do on the way: fire for
rows that still carry images (the text path replaces the collator, and the user's
image handling with it), or leave behind a collator padding through a processor,
which has no `.pad`, so the run dies on the first batch. CPU-pure and offline.
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
    """`input_ids` alone does not make a row text-only: the text path swaps in a
    text collator, silently dropping the user's image handling."""
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
    """`DataCollatorForSeq2Seq(tokenizer = processor)` is what the bypass exists
    for: left alone it pads via a `self.tokenizer.pad` a processor does not have."""
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
    """The pre-existing path: one padding through a real tokenizer keeps its settings."""
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
    """The swap is trainer-wide, so a text-only train set beside a multimodal
    eval set must not clear the guard."""
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
    """phi4_multimodal, pix2struct and kosmos-2.5 spell these their own way."""
    dataset = Dataset.from_dict({
        "input_ids": [list(ROW), list(ROW)],
        column: [[0.0], [0.0]],
    })
    trainer = StubTrainer(MyVisionCollator(StubProcessor()), dataset)
    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)


def test_a_processor_backed_collator_is_rebuilt_even_with_packing():
    """It raises on the first batch either way, so the repair is not packing-gated."""
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
    """A raw text-only eval split is tokenized by the text path with the inner
    tokenizer and masked the same way, so users need not pretokenize it by hand."""
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
    """Raw does not mean text-only: an image column still needs its collator."""
    trainer = _text_only_trainer()
    trainer.eval_dataset = Dataset.from_dict({
        "text": [f"{INSTRUCTION_PART}q{RESPONSE_PART}a"],
        "images": [[0.0]],
    })
    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)


def test_a_raw_split_with_no_text_column_is_still_refused():
    """Conversational multimodal data hides its images inside the turns, so the
    columns look text-only. Nothing to tokenize, so refuse rather than empty it."""
    trainer = _text_only_trainer()
    trainer.eval_dataset = Dataset.from_dict({"messages": [[{"role": "user"}]]})
    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)


# ---- non-seq2seq collators that pad through a processor --------------------

def test_a_processor_backed_padding_collator_is_repaired_under_packing():
    """`DataCollatorWithPadding(tokenizer = processor)` pads through the same
    missing `.pad`, so packing being on must not leave it attached."""
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
    `.processor`, not `.tokenizer`, with the same problem."""
    trainer = _text_only_trainer()          # MyVisionCollator holds `.processor`
    trainer.args.packing = True
    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)
    assert isinstance(out.data_collator, DataCollatorForSeq2Seq)
    assert hasattr(out.data_collator.tokenizer, "pad")


class PackingCollator:
    """Stand-in for TRL's packing `DataCollatorForLanguageModeling`: a bare
    `pad_token_id`, no tokenizer or processor at all."""
    def __init__(self):
        self.pad_token_id = PAD_ID

    def __call__(self, features):
        return {"packed": True}


def test_a_packing_collator_that_holds_no_tokenizer_is_left_alone():
    """Why the swap is packing-gated: a packing collator builds the packed
    `position_ids` itself and DataCollatorForSeq2Seq would not."""
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


# ---- raw columns must not reach the collator -------------------------------

class StrictPadTokenizer(StubTokenizer):
    """Pads like the real thing: it tensorizes every key it is handed, so a raw
    string column raises exactly as `tokenizer.pad(return_tensors = "pt")` does."""
    def pad(self, features, **kwargs):
        for feature in features:
            for key, value in feature.items():
                probe = value
                while isinstance(probe, (list, tuple)) and probe:
                    probe = probe[0]
                if not isinstance(probe, (int, float, bool)):
                    raise ValueError(
                        f"Unable to create tensor ... (`{key}` in this case)"
                    )
        return StubTokenizer.pad(self, features, **kwargs)


class StrictProcessor(StubProcessor):
    def __init__(self):
        super().__init__()
        self.tokenizer = StrictPadTokenizer()


def _collate_every_row(trainer):
    dataset = trainer.train_dataset
    return trainer.data_collator([dataset[i] for i in range(len(dataset))])


def test_a_leftover_text_column_never_reaches_the_collator():
    """Pretokenizing with `dataset.map` and no `remove_columns` leaves `text`
    behind. Unused-column removal is off for token-type-id models, so the raw
    string reaches the swapped-in collator and dies while tensorizing."""
    dataset = Dataset.from_dict({
        "input_ids": [list(ROW), list(ROW)],
        "text": ["q0", "q1"],
    })
    trainer = StubTrainer(MyVisionCollator(StrictProcessor()), dataset)
    trainer.processing_class = StrictProcessor()

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert "text" not in out.train_dataset.column_names
    _collate_every_row(out)     # used to raise ValueError while tensorizing


def test_a_leftover_conversational_column_never_reaches_the_collator():
    dataset = Dataset.from_dict({
        "input_ids": [list(ROW), list(ROW)],
        "messages": [[{"role": "user", "content": "q"}]] * 2,
    })
    trainer = StubTrainer(MyVisionCollator(StrictProcessor()), dataset)
    trainer.processing_class = StrictProcessor()

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert "messages" not in out.train_dataset.column_names
    _collate_every_row(out)


def test_numeric_model_columns_survive_the_strip():
    """token_type_ids is exactly why unused-column removal was turned off, so
    dropping raw columns must not take it (or any other numeric column) along."""
    dataset = Dataset.from_dict({
        "input_ids": [list(ROW), list(ROW)],
        "attention_mask": [[1] * len(ROW)] * 2,
        "token_type_ids": [[0] * len(ROW)] * 2,
        "text": ["q0", "q1"],
    })
    trainer = StubTrainer(MyVisionCollator(StrictProcessor()), dataset)
    trainer.processing_class = StrictProcessor()

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    for column in ("input_ids", "attention_mask", "token_type_ids", "labels"):
        assert column in out.train_dataset.column_names, column
    assert "text" not in out.train_dataset.column_names


def test_a_tokenized_eval_split_keeps_no_raw_text():
    """The raw eval split the bypass tokenizes for the user must come back with
    the text column replaced, not beside it."""
    trainer = StubTrainer(MyVisionCollator(StrictProcessor()), _text_rows())
    trainer.processing_class = StrictProcessor()
    trainer.eval_dataset = {"raw": _raw_text_rows(), "pretok": _text_rows()}

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert "text" not in out.eval_dataset["raw"].column_names
    assert "labels" in out.eval_dataset["raw"].column_names


# ---- only model inputs may reach the collator ------------------------------

class TensorizingPadTokenizer(StrictPadTokenizer):
    """Pads the keys a real tokenizer knows about and tensorizes the rest, so a
    ragged leftover raises exactly as `tokenizer.pad(return_tensors = "pt")` does."""
    _PADDED = ("input_ids", "attention_mask", "token_type_ids", "special_tokens_mask")

    def pad(self, features, **kwargs):
        for key in features[0]:
            if key in self._PADDED: continue
            widths = {len(f[key]) if isinstance(f[key], (list, tuple)) else None
                      for f in features}
            if len(widths) > 1:
                raise ValueError(
                    f"Unable to create tensor ... (`{key}` in this case)"
                )
        return StrictPadTokenizer.pad(self, features, **kwargs)


class TensorizingProcessor(StubProcessor):
    def __init__(self):
        super().__init__()
        self.tokenizer = TensorizingPadTokenizer()


def _bypass(dataset, model = None):
    trainer = StubTrainer(MyVisionCollator(TensorizingProcessor()), dataset)
    trainer.processing_class = TensorizingProcessor()
    if model is not None: trainer.model = model
    return train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)


def test_a_scalar_label_column_never_reaches_the_collator():
    """`DataCollatorForSeq2Seq` reads `label` in preference to `labels`, so a
    kept scalar `label` both loses the masked labels and dies on `len(int)`."""
    dataset = Dataset.from_dict({
        "input_ids": [list(ROW), list(ROW)],
        "label": [0, 1],
    })
    out = _bypass(dataset)

    batch = _collate_every_row(out)     # used to raise TypeError: len() of int
    assert "label" not in out.train_dataset.column_names
    assert len(batch["labels"][0]) == len(ROW)


def test_a_ragged_numeric_column_never_reaches_the_collator():
    """Pretokenizing a prompt/completion split leaves `prompt_ids` behind. It is
    numeric, but `tokenizer.pad` pads only its own keys, so stacking it fails."""
    dataset = Dataset.from_dict({
        "input_ids": [list(ROW), list(ROW)],
        "prompt_ids": [[1, 2], [1, 2, 3]],
    })
    out = _bypass(dataset)

    _collate_every_row(out)             # used to raise ValueError while tensorizing
    assert "prompt_ids" not in out.train_dataset.column_names


def test_a_numeric_metadata_column_is_dropped():
    """`id`/`sample_idx` tensorize cleanly but are not model inputs."""
    dataset = Dataset.from_dict({
        "input_ids": [list(ROW), list(ROW)],
        "id": [7, 8],
        "sample_idx": [0, 1],
    })
    out = _bypass(dataset)

    assert "id" not in out.train_dataset.column_names
    assert "sample_idx" not in out.train_dataset.column_names
    assert "input_ids" in out.train_dataset.column_names


def test_model_and_processor_declared_inputs_survive():
    """The kept names come from the processor and the model's own forward, so a
    multimodal input this file has never heard of still gets through."""
    class FakeModel:
        def forward(self, input_ids = None, attention_mask = None, labels = None,
                    pixel_values = None, deepstack_visual_embeds = None, **kwargs):
            pass

    processor_names = ["input_ids", "attention_mask", "mm_token_type_ids"]
    dataset = Dataset.from_dict({
        "input_ids": [list(ROW), list(ROW)],
        "token_type_ids": [[0] * len(ROW)] * 2,
        "mm_token_type_ids": [[0] * len(ROW)] * 2,
        "deepstack_visual_embeds": [[0] * len(ROW)] * 2,
        "id": [1, 2],
    })
    trainer = StubTrainer(MyVisionCollator(TensorizingProcessor()), dataset)
    trainer.processing_class = TensorizingProcessor()
    trainer.processing_class.tokenizer.model_input_names = processor_names
    trainer.model = FakeModel()

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    kept = out.train_dataset.column_names
    for column in ("input_ids", "labels", "token_type_ids", "mm_token_type_ids",
                   "deepstack_visual_embeds"):
        assert column in kept, column
    assert "id" not in kept


# ---- the multimodal set follows the installed processor --------------------

class DerivedProcessor(StubProcessor):
    """A processor whose image half declares its own output names, the way every
    transformers image processor / feature extractor does."""
    def __init__(self, image_names = (), own_names = ()):
        super().__init__()
        self.image_processor = type("ImageProcessor", (), {
            "model_input_names": list(image_names),
        })()
        if own_names:
            self.model_input_names = list(self.tokenizer.model_input_names) + list(own_names)


def _refuses(dataset, processor):
    trainer = StubTrainer(MyVisionCollator(processor), dataset)
    trainer.processing_class = processor
    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)
    return trainer


@pytest.mark.parametrize("column", ["image_patches", "image_patches_indices"])
def test_fuyu_style_image_columns_are_refused(column):
    """Fuyu spells its preprocessed images `image_patches`/`image_patches_indices`
    beside `input_ids`, so the columns alone look text-only."""
    from transformers.models.fuyu.image_processing_fuyu import FuyuImageProcessor

    processor = DerivedProcessor(
        image_names = FuyuImageProcessor.model_input_names,
        # FuyuProcessor.model_input_names adds this one on top of the image half.
        own_names = ["image_patches", "image_patches_indices"],
    )
    dataset = Dataset.from_dict({
        "input_ids": [list(ROW), list(ROW)],
        column: [[0.0], [0.0]],
    })
    trainer = _refuses(dataset, processor)
    assert isinstance(trainer.data_collator, MyVisionCollator)


def test_a_column_name_this_file_has_never_heard_of_is_refused():
    """The point of deriving: a processor output no static list mentions still
    keeps the user's collator."""
    processor = DerivedProcessor(image_names = ["widget_patches", "widget_offsets"])
    dataset = Dataset.from_dict({
        "input_ids": [list(ROW), list(ROW)],
        "widget_patches": [[0.0], [0.0]],
    })
    _refuses(dataset, processor)


def test_a_raw_eval_split_with_a_derived_column_is_refused():
    """`_eval_split_is_raw_text_only` reads the same set."""
    processor = DerivedProcessor(image_names = ["widget_patches"])
    trainer = StubTrainer(MyVisionCollator(processor), _text_rows())
    trainer.processing_class = processor
    trainer.eval_dataset = Dataset.from_dict({
        "text": [f"{INSTRUCTION_PART}q{RESPONSE_PART}a"],
        "widget_patches": [[0.0]],
    })
    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)


def test_deriving_never_denylists_the_text_columns():
    """An image processor that repeats `input_ids`/`attention_mask` must not turn
    every text-only row into a refusal."""
    processor = DerivedProcessor(
        image_names = ["input_ids", "attention_mask", "token_type_ids", "labels"],
    )
    trainer = StubTrainer(MyVisionCollator(processor), _text_rows())
    trainer.processing_class = processor
    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)
    assert "labels" in out.train_dataset.column_names


# ---- a DatasetDict eval is a dict of splits, not one dataset ---------------

def test_a_datasetdict_eval_is_normalized_per_split():
    """`type(x) is dict` is False for a DatasetDict, so the eval branches used to
    treat the whole mapping as one dataset: `column_names` came back a dict, the
    raw `text` column was never dropped, and the collator died tensorizing it."""
    from datasets import DatasetDict

    trainer = StubTrainer(MyVisionCollator(StrictProcessor()), _text_rows())
    trainer.processing_class = StrictProcessor()
    trainer.eval_dataset = DatasetDict({"raw": _raw_text_rows(),
                                        "pretok": _text_rows()})

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    for key in ("raw", "pretok"):
        columns = out.eval_dataset[key].column_names
        assert "labels" in columns, f"{key} was never masked"
        assert "text" not in columns, f"{key} kept its raw text column"
        split = out.eval_dataset[key]
        out.data_collator([split[i] for i in range(len(split))])


def test_a_multimodal_split_in_a_datasetdict_eval_still_refuses():
    from datasets import DatasetDict

    trainer = _text_only_trainer()
    trainer.eval_dataset = DatasetDict({"a": _text_rows(), "b": _multimodal_rows()})
    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)


def test_an_iterable_datasetdict_eval_is_normalized_per_split():
    from datasets import IterableDatasetDict

    trainer = StubTrainer(MyVisionCollator(StubProcessor()), _text_rows())
    trainer.eval_dataset = IterableDatasetDict(
        {"a": _text_rows().to_iterable_dataset()},
    )
    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)
    assert "labels" in next(iter(out.eval_dataset["a"]))


# ---- a fresh streaming dataset carries no batch_size -----------------------

def test_a_pretokenized_streaming_dataset_reaches_masking():
    """The bypass accepts an IterableDataset, so the map below must not read a
    `_ex_iterable.batch_size` a fresh ArrowExamplesIterable does not have."""
    dataset = _text_rows().to_iterable_dataset()
    assert not hasattr(dataset._ex_iterable, "batch_size")

    trainer = StubTrainer(MyVisionCollator(StubProcessor()), dataset)
    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    row = next(iter(out.train_dataset))     # used to raise AttributeError
    assert [l for l in row["labels"] if l != -100] == [20, 21]


def test_a_columnless_streaming_dataset_reaches_masking():
    """`load_dataset(streaming = True)` can resolve no features, so the bypass
    peeks a row instead - and the same map still has to run."""
    dataset = _text_rows().to_iterable_dataset()
    dataset._info.features = None       # what an unresolved stream looks like
    assert dataset.column_names is None
    assert not hasattr(dataset._ex_iterable, "batch_size")

    trainer = StubTrainer(MyVisionCollator(StubProcessor()), dataset)
    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)
    row = next(iter(out.train_dataset))     # used to raise AttributeError
    assert [l for l in row["labels"] if l != -100] == [20, 21]
