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
    `.processor`, not `.tokenizer`, with the same problem. It rebuilds labels
    through that processor, so it is a class we know how to answer for and the
    repair still runs with packing on."""
    from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

    trainer = _text_only_trainer()
    trainer.data_collator = DataCollatorForVisionLanguageModeling(processor = StubProcessor())
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


class LabelRebuildingVisionCollator:
    """A user's vision collator holding the processor for images and the text
    tokenizer for padding, and rebuilding `labels` from `input_ids` on collate."""
    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer

    def __call__(self, features):
        batch = self.tokenizer.pad(features)
        batch["labels"] = [list(ids) for ids in batch["input_ids"]]
        return batch


def test_a_label_rebuilding_vision_collator_is_replaced():
    """It pads through a real tokenizer, so the `.pad` repair does not fire. Left
    attached it rebuilds `labels` at collate time, throwing away the dataset-level
    mask this function just wrote: training on prompts."""
    collator = LabelRebuildingVisionCollator(StubProcessor())
    trainer = StubTrainer(collator, _text_rows())

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert out.data_collator is not collator, "the masked labels get overwritten"
    rows = [out.train_dataset[i] for i in range(len(out.train_dataset))]
    labels = out.data_collator(rows)["labels"]
    assert -100 in list(labels[0]), "the response mask did not survive collation"


def test_a_label_rebuilding_vision_collator_is_refused_under_packing():
    """With packing on the same collator has no right answer: this file cannot
    tell a custom label rebuilder from a custom packer, and both readings lose
    something silently. Refuse instead of picking one."""
    collator = LabelRebuildingVisionCollator(StubProcessor())
    trainer = StubTrainer(collator, _text_rows())
    trainer.args.packing = True

    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)


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
    # Literal copy of FuyuImageProcessor.model_input_names: importing it pulls in
    # torchvision (transformers 5.x), which is not a dependency of this package.
    fuyu_image_names = ["images", "image_input_ids", "image_patches",
                        "image_patch_indices_per_batch",
                        "image_patch_indices_per_subsequence"]
    processor = DerivedProcessor(
        image_names = fuyu_image_names,
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


@pytest.mark.parametrize("column", ["num_crops", "num_tiles"])
def test_an_integer_side_car_is_refused_without_the_processor(column):
    """Gemma 3 declares `num_crops` and Llama 3.2 Vision `num_tiles` beside
    `pixel_values`. Their dtype is an ordinary int, so the schema calls them
    plain and only the name can refuse - and a processor that will not name its
    own outputs leaves the static set as the only thing that knows."""
    processor = DerivedProcessor()          # declares nothing to derive from
    dataset = Dataset.from_dict({
        "input_ids": [list(ROW), list(ROW)],
        column: [4, 4],
    })
    _refuses(dataset, processor)


def test_a_raw_eval_split_with_a_derived_column_is_refused():
    """`_split_is_raw_text_only` reads the same set."""
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


# ---- a text column is a column that actually holds text --------------------

def _messages_with_an_image():
    return [[
        {"role": "user", "content": [{"type": "image", "image": "cat.png"},
                                     {"type": "text",  "text": "q"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
    ]]


def _messages_of_plain_text():
    return [[
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]]


def test_pretokenized_rows_hiding_images_in_messages_are_refused():
    """The column names look text-only, but `messages` carries inline image parts
    the user's collator is what turns into pixels. The strip drops the column, so
    the images would be lost silently."""
    dataset = Dataset.from_dict({
        "input_ids": [list(ROW)],
        "messages": _messages_with_an_image(),
    })
    collator = MyVisionCollator(StubProcessor())
    trainer = StubTrainer(collator, dataset)

    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert trainer.data_collator is collator, "the user's collator was replaced"
    assert "messages" in trainer.train_dataset.column_names


def test_a_pretokenized_iterable_hiding_images_in_messages_is_refused():
    """No column_names to read, so the row peek has to make the same call - and
    it must not eat the row it peeked."""
    dataset = Dataset.from_dict({
        "input_ids": [list(ROW)],
        "messages": _messages_with_an_image(),
    }).to_iterable_dataset()
    trainer = StubTrainer(MyVisionCollator(StubProcessor()), dataset)

    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert len(list(trainer.train_dataset)) == 1, "the peek consumed the stream"


def test_an_eval_split_hiding_images_in_messages_is_refused():
    dataset = Dataset.from_dict({
        "input_ids": [list(ROW)],
        "messages": _messages_with_an_image(),
    })
    trainer = _text_only_trainer()
    trainer.eval_dataset = dataset
    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)


@pytest.mark.parametrize("kind", ["input_image", "input_video"])
def test_pretokenized_rows_hiding_an_input_image_part_are_refused(kind):
    """`input_image`/`input_video` are the other spelling of an inline media part
    (the one `mlx/loader.py` already recognises), and name media just like
    `image` does."""
    dataset = Dataset.from_dict({
        "input_ids": [list(ROW)],
        "messages": [[
            {"role": "user", "content": [{"type": kind, kind: "cat.png"},
                                         {"type": "text", "text": "q"}]},
        ]],
    })
    collator = MyVisionCollator(StubProcessor())
    trainer = StubTrainer(collator, dataset)

    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert trainer.data_collator is collator, "the user's collator was replaced"
    assert "messages" in trainer.train_dataset.column_names


def test_pretokenized_rows_with_plain_text_messages_still_pass():
    """The other half of the same check: a conversation of plain strings holds no
    image handling to lose, so `dataset.map` without remove_columns keeps working."""
    dataset = Dataset.from_dict({
        "input_ids": [list(ROW)],
        "messages": _messages_of_plain_text(),
    })
    trainer = StubTrainer(MyVisionCollator(StrictProcessor()), dataset)
    trainer.processing_class = StrictProcessor()

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert "messages" not in out.train_dataset.column_names
    _collate_every_row(out)


def test_dataset_text_field_pointing_at_conversations_is_refused():
    """`dataset_text_field = "messages"` used to clear the guard by name alone,
    and the raw conversations then reached the plain tokenizer with no chat
    template applied, failing after the collator had already been swapped."""
    trainer = _text_only_trainer()
    trainer.args.dataset_text_field = "messages"
    trainer.eval_dataset = Dataset.from_dict({"messages": _messages_of_plain_text()})

    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)


def test_dataset_text_field_pointing_at_a_real_string_column_is_allowed():
    """The bypass is about strings, not about the name `text`."""
    trainer = _text_only_trainer()
    trainer.args.dataset_text_field = "prompt"
    trainer.eval_dataset = Dataset.from_dict({
        "prompt": [f"{INSTRUCTION_PART}q{RESPONSE_PART}a"],
    })

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert "labels" in out.eval_dataset.column_names
    assert "prompt" not in out.eval_dataset.column_names


def test_a_raw_eval_split_whose_text_column_is_structured_is_refused():
    """Same column name, non-string rows: `text` holding message dicts is not
    something the plain tokenizer can encode."""
    trainer = _text_only_trainer()
    trainer.eval_dataset = Dataset.from_dict({"text": _messages_of_plain_text()})
    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)


def test_a_raw_iterable_eval_split_of_real_text_is_still_allowed():
    trainer = _text_only_trainer()
    trainer.eval_dataset = _raw_text_rows().to_iterable_dataset()
    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)
    assert "labels" in next(iter(out.eval_dataset))


# ---- one row is not the whole split ----------------------------------------

def _rows_with_an_image_at(n, image_at):
    """`n` pretokenized rows; only row `image_at` carries an image, under a column
    name no static list mentions. Exactly what a partly-illustrated set looks like."""
    import io
    from PIL import Image as PILImage
    from datasets import Image as ImageFeature

    buffer = io.BytesIO()
    PILImage.new("RGB", (2, 2)).save(buffer, format = "PNG")
    media = [{"bytes": buffer.getvalue(), "path": None} if i == image_at else None
             for i in range(n)]
    return Dataset.from_dict({
        "input_ids": [list(ROW)] * n, "media": media,
    }).cast_column("media", ImageFeature())


def test_a_later_row_hiding_an_image_is_refused():
    """Row 0 has no image, so the old one-row peek cleared the guard and the strip
    below then dropped the column, images and all."""
    n = 5
    collator = MyVisionCollator(StubProcessor())
    trainer = StubTrainer(collator, _rows_with_an_image_at(n, image_at = n - 1))

    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert trainer.data_collator is collator
    assert "media" in trainer.train_dataset.column_names


def test_the_bounded_scan_reaches_the_end_of_a_long_split():
    """The scan is a small constant, so it samples across the split (first and
    last row included) rather than the first N rows."""
    n = 200
    trainer = StubTrainer(MyVisionCollator(StubProcessor()),
                          _rows_with_an_image_at(n, image_at = n - 1))
    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)


def test_a_long_plain_text_split_is_still_allowed():
    """The other half: scanning more rows must not refuse an honest text run."""
    n = 200
    dataset = Dataset.from_dict({
        "input_ids": [list(ROW)] * n,
        "messages": _messages_of_plain_text() * n,
    })
    trainer = StubTrainer(MyVisionCollator(StrictProcessor()), dataset)
    trainer.processing_class = StrictProcessor()

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert "messages" not in out.train_dataset.column_names


def test_a_later_streaming_row_hiding_an_image_is_refused():
    """Streaming rows arrive as they were written, so row 0 really can be plain
    while a later one carries inline image parts."""
    from datasets import IterableDataset

    rows = [{"input_ids": list(ROW), "messages": _messages_of_plain_text()[0]},
            {"input_ids": list(ROW), "messages": _messages_of_plain_text()[0]},
            {"input_ids": list(ROW), "messages": _messages_with_an_image()[0]}]
    dataset = IterableDataset.from_generator(lambda: iter(rows))
    trainer = StubTrainer(MyVisionCollator(StubProcessor()), dataset)

    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert len(list(trainer.train_dataset)) == len(rows), "the peek consumed the stream"


def test_a_raw_eval_split_whose_later_row_is_not_text_is_refused():
    """The same one-row peek also admitted a mixed raw eval split to the plain
    tokenizer, which cannot encode a null."""
    trainer = _text_only_trainer()
    trainer.eval_dataset = Dataset.from_dict({
        "text": [f"{INSTRUCTION_PART}q{RESPONSE_PART}a", None],
    })
    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)


def test_a_raw_eval_split_whose_null_is_past_the_sample_is_refused():
    """`Value('string')` is nullable, so the dtype still reads as text and the
    bounded sample skips row 20. Left through, the plain tokenizer meets the null
    halfway through `map` and dies there instead of refusing here."""
    n, null_at = 200, 20
    texts = [f"{INSTRUCTION_PART}q{RESPONSE_PART}a"] * n
    texts[null_at] = None
    trainer = _text_only_trainer()
    trainer.eval_dataset = Dataset.from_dict({"text": texts})
    assert null_at not in {i * (n - 1) // 15 for i in range(16)}, "row is sampled"

    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)


# ---- what the whole-column scan is allowed to cost -------------------------

@pytest.mark.parametrize("column", ["urls", "paths", "files"])
def test_an_ambiguous_column_of_media_lists_is_refused(column):
    """A plural media column holds a list of URLs per row, and `List(string)` is
    provably text, so the bypass used to strip the column and train on the text
    alone."""
    n = 4
    trainer = StubTrainer(MyVisionCollator(StrictProcessor()), Dataset.from_dict({
        "input_ids": [list(ROW)] * n,
        column: [["https://example.com/cat.jpg", "https://example.com/dog.png"]] * n,
    }))
    trainer.processing_class = StrictProcessor()

    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)


def test_an_ambiguous_image_column_is_never_decoded_by_the_scan(monkeypatch):
    """`media` is an ambiguous name, but its dtype is `PIL.Image.Image`, so no
    string in it can name a file. Scanning it anyway decoded every image (~9ms
    and ~3MB of pixels a row) only to reach the refusal the schema already gave."""
    import io
    import datasets.features.image as image_feature
    from PIL import Image as PILImage
    from datasets import Image as ImageFeature

    n = 300
    buffer = io.BytesIO()
    PILImage.new("RGB", (8, 8)).save(buffer, format = "PNG")
    blob = {"bytes": buffer.getvalue(), "path": None}
    dataset = Dataset.from_dict({
        "input_ids": [list(ROW)] * n, "media": [blob] * n,
    }).cast_column("media", ImageFeature())
    decoded = []
    original = image_feature.Image.decode_example
    monkeypatch.setattr(image_feature.Image, "decode_example",
                        lambda self, value, **kw: (decoded.append(1),
                                                   original(self, value, **kw))[1])
    trainer = StubTrainer(MyVisionCollator(StubProcessor()), dataset)

    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    # Only the bounded row sample may decode; the whole-column scan may not.
    # 16 per split, and the train split is now scanned as well as the eval one
    # (train no longer has to be pretokenized), so the bound is per split. What
    # matters is that it stays a constant and does not grow with `n`.
    assert len(decoded) <= 16 * 2, f"{len(decoded)} images decoded for {n} rows"
    assert len(decoded) < n / 4, "the sample is scaling with the dataset"


# ---- the processor may live only on the collator ---------------------------

def test_the_collators_processor_derives_the_multimodal_columns():
    """With the public `tokenizer =` override the multimodal processor is no
    longer `processing_class`, but the collator still holds it, so its derived
    output names must still block the bypass."""
    processor = DerivedProcessor(image_names = ["widget_patches"])
    dataset = Dataset.from_dict({
        "input_ids": [list(ROW)],
        "widget_patches": [[0.0]],
    })
    collator = MyVisionCollator(processor)
    trainer = StubTrainer(collator, dataset)

    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART,
                                tokenizer = StubTokenizer())

    assert trainer.data_collator is collator
    assert "widget_patches" in trainer.train_dataset.column_names


def test_the_tokenizer_override_still_allows_a_text_only_run():
    """Deriving from the collator only adds names, so a text-only split passes."""
    trainer = StubTrainer(MyVisionCollator(StrictProcessor()), _text_rows())
    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART,
                                  tokenizer = StrictPadTokenizer())
    assert "labels" in out.train_dataset.column_names


# ---- a streaming split cannot be filtered, so sample its labels -------------

def _unmatched_rows(n = 3):
    # No response marker anywhere: every label ends up -100.
    return Dataset.from_dict({"input_ids": [[10, 11, 12]] * n})


def test_a_fully_masked_streaming_train_split_is_reported():
    """`_filter_fully_masked` and `fix_zero_training_loss` both skip streaming, so
    the bypass would otherwise start a run with no training signal at all."""
    dataset = _unmatched_rows().to_iterable_dataset()
    trainer = StubTrainer(MyVisionCollator(StubProcessor()), dataset)

    with pytest.raises(ValueError, match = "nothing to train on"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)


def test_the_streaming_label_check_does_not_consume_the_stream():
    dataset = _unmatched_rows(n = 5).to_iterable_dataset()
    trainer = StubTrainer(MyVisionCollator(StubProcessor()), dataset)
    with pytest.raises(ValueError, match = "nothing to train on"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)
    assert len(list(dataset)) == 5


def test_a_fully_masked_streaming_eval_split_names_the_split():
    trainer = StubTrainer(MyVisionCollator(StubProcessor()), _text_rows())
    trainer.eval_dataset = {"bad": _unmatched_rows().to_iterable_dataset()}
    with pytest.raises(ValueError, match = r"eval_dataset\[bad\]"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)


def test_a_partly_masked_streaming_split_still_trains():
    """Only an all -100 sample is an error; rows without a response are normal."""
    from datasets import concatenate_datasets

    dataset = concatenate_datasets([_unmatched_rows(n = 2), _text_rows()])
    trainer = StubTrainer(MyVisionCollator(StubProcessor()), dataset.to_iterable_dataset())

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert len(list(out.train_dataset)) == 4


# ---- a 16-row sample is not the whole split either --------------------------

def test_an_image_on_an_unsampled_row_is_refused():
    """The bounded scan reads 16 fixed positions, so a 200-row split checks rows
    0, 13, 26, ... and an image on row 5 was never looked at. The column type is
    uniform down the whole split, so the schema settles it for every row."""
    n = 200
    collator = MyVisionCollator(StubProcessor())
    trainer = StubTrainer(collator, _rows_with_an_image_at(n, image_at = 5))

    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert trainer.data_collator is collator
    assert "media" in trainer.train_dataset.column_names, "the images were dropped"


def test_a_raw_eval_split_with_an_image_on_an_unsampled_row_is_refused():
    """`_split_is_raw_text_only` samples the same 16 positions, so the eval
    half needs the schema just as much as the train half does."""
    import io
    from PIL import Image as PILImage
    from datasets import Image as ImageFeature

    buffer = io.BytesIO()
    PILImage.new("RGB", (2, 2)).save(buffer, format = "PNG")
    n = 200
    media = [{"bytes": buffer.getvalue(), "path": None} if i == 5 else None
             for i in range(n)]
    eval_split = Dataset.from_dict({
        "text": [f"{INSTRUCTION_PART}q{RESPONSE_PART}a"] * n, "media": media,
    }).cast_column("media", ImageFeature())

    trainer = _text_only_trainer()
    trainer.eval_dataset = eval_split
    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert "media" in trainer.eval_dataset.column_names


# ---- a masked prefix is not a masked stream --------------------------------

def test_a_stream_whose_responses_start_later_still_trains():
    """A sorted or filtered stream can put every prompt-only row first. The
    bounded prefix cannot see past row 16, so it must not refuse the run."""
    from datasets import IterableDataset

    rows = [{"input_ids": [10, 11, 12]} for _ in range(16)]
    rows += [{"input_ids": list(ROW)} for _ in range(4)]
    dataset = IterableDataset.from_generator(lambda: iter(rows))
    trainer = StubTrainer(MyVisionCollator(StubProcessor()), dataset)

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    got = list(out.train_dataset)
    assert len(got) == len(rows)
    assert [l for l in got[-1]["labels"] if l != -100] == [20, 21]


def test_a_stream_masked_past_the_prefix_warns_instead(capsys):
    trainer = StubTrainer(MyVisionCollator(StubProcessor()),
                          _unmatched_rows(n = 40).to_iterable_dataset())
    train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)
    assert "nothing to train on" in capsys.readouterr().out


# ---- empty splits ----------------------------------------------------------

def test_an_empty_raw_eval_split_is_handled():
    """`_maybe_tokenize_dataset` peeks one row, and an empty split has none."""
    trainer = StubTrainer(MyVisionCollator(StubProcessor()), _text_rows())
    trainer.eval_dataset = Dataset.from_dict({"text": []})

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert len(out.eval_dataset) == 0
    assert "labels" in out.train_dataset.column_names


def test_an_empty_pretokenized_eval_split_is_handled():
    trainer = StubTrainer(MyVisionCollator(StubProcessor()), _text_rows())
    trainer.eval_dataset = Dataset.from_dict({"input_ids": []})
    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)
    assert len(out.eval_dataset) == 0


# ---- a trainer need not have a train split ---------------------------------

def test_an_eval_only_trainer_is_not_a_crash():
    """`Trainer(eval_dataset = ...)` with no train split: the final all-masked
    check calls len() on it, so it must not run with nothing there."""
    trainer = StubTrainer(MyVisionCollator(StubProcessor()), None)
    trainer.eval_dataset = _text_rows()

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert out.train_dataset is None
    assert "labels" in out.eval_dataset.column_names


def test_an_eval_only_trainer_on_the_plain_text_path_is_not_a_crash():
    """Same call, reached without the bypass at all."""
    trainer = StubTrainer(DataCollatorForSeq2Seq(tokenizer = StubTokenizer()), None)
    trainer.eval_dataset = _text_rows()
    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)
    assert "labels" in out.eval_dataset.column_names


# ---- a media URL/path column is media, even as a plain string ---------------

@pytest.mark.parametrize("column", [
    "image_url", "video_url", "audio_url", "input_image", "input_video",
    "image_path", "img_url", "image_file",
])
def test_a_top_level_media_url_column_is_refused(column):
    """A pretokenized VLM set often stores its media as a URL string beside
    `input_ids`. The value is a string, so only the column name says it is media,
    and dropping it would leave a text-only run over a vision dataset."""
    collator = MyVisionCollator(StubProcessor())
    dataset = Dataset.from_dict({
        "input_ids": [list(ROW), list(ROW)],
        column: ["https://example.com/cat.jpg"] * 2,
    })
    trainer = StubTrainer(collator, dataset)

    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert trainer.data_collator is collator, "the user's collator was replaced"
    assert column in trainer.train_dataset.column_names, "the media column was dropped"


@pytest.mark.parametrize("column, value", [
    ("path",      "/data/images/0001.png"),
    ("file_name", "clips/intro.mp4"),
    ("url",       "https://example.com/a.jpeg?width=64"),
    ("uri",       "data:image/png;base64,AAAA"),
])
def test_an_ambiguous_column_pointing_at_media_is_refused(column, value):
    """`path`/`url` are media only sometimes, so the value decides."""
    collator = MyVisionCollator(StubProcessor())
    dataset = Dataset.from_dict({
        "input_ids": [list(ROW), list(ROW)],
        column: [value] * 2,
    })
    trainer = StubTrainer(collator, dataset)
    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)
    assert trainer.data_collator is collator


@pytest.mark.parametrize("column, value", [
    ("path",      "corpus/shard_0007.jsonl"),
    ("path",      "wiki/en/Anarchism"),
    ("url",       "https://en.wikipedia.org/wiki/Anarchism"),
    ("file_name", "notes.txt"),
    ("source",    "wikipedia"),
    ("doc_id",    "id-42"),
])
def test_a_benign_metadata_column_is_still_allowed(column, value):
    """The other direction: `path` is a source file in plenty of text corpora, so
    a name alone must not refuse an ordinary text-only run."""
    trainer = StubTrainer(MyVisionCollator(StubProcessor()), Dataset.from_dict({
        "input_ids": [list(ROW), list(ROW)],
        column: [value] * 2,
    }))

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert "labels" in out.train_dataset.column_names, "the text path never ran"
    assert column not in out.train_dataset.column_names


def test_a_raw_eval_split_with_a_media_url_column_is_refused():
    """The same column on a raw eval split, which `_maybe_tokenize_dataset` would
    tokenize from `text` and strip the media from."""
    trainer = _text_only_trainer()
    trainer.eval_dataset = Dataset.from_dict({
        "text": [f"{INSTRUCTION_PART}q{RESPONSE_PART}a"],
        "image_url": ["https://example.com/cat.png"],
    })
    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)


# ---- an ambiguous media column is read to the end ---------------------------

def test_an_ambiguous_media_column_is_scanned_past_the_sample():
    """`url`/`path` are `string` either way, so the schema cannot rule them out
    and the 16-row sample skips row 5. Read the column instead: it decodes no
    images, so a full scan is cheap."""
    n = 200
    urls = ["https://en.wikipedia.org/wiki/Anarchism"] * n
    urls[5] = "https://example.com/cat.jpg"
    collator = MyVisionCollator(StubProcessor())
    trainer = StubTrainer(collator, Dataset.from_dict({
        "input_ids": [list(ROW)] * n, "url": urls,
    }))

    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert trainer.data_collator is collator
    assert "url" in trainer.train_dataset.column_names, "the media column was dropped"


def test_a_raw_eval_split_with_an_unsampled_media_url_is_refused():
    """The eval half samples the same 16 positions, so it needs the scan too."""
    n = 200
    paths = ["corpus/shard_0007.jsonl"] * n
    paths[5] = "/data/images/0001.png"
    trainer = _text_only_trainer()
    trainer.eval_dataset = Dataset.from_dict({
        "text": [f"{INSTRUCTION_PART}q{RESPONSE_PART}a"] * n, "path": paths,
    })
    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)


def test_a_wholly_benign_ambiguous_column_still_passes_the_scan():
    """The scan must not turn every text corpus with a `path` column into a refusal."""
    n = 200
    trainer = StubTrainer(MyVisionCollator(StubProcessor()), Dataset.from_dict({
        "input_ids": [list(ROW)] * n, "path": [f"corpus/shard_{i}.jsonl" for i in range(n)],
    }))
    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)
    assert "labels" in out.train_dataset.column_names, "the text path never ran"


# ---- a collator holding an image processor directly -------------------------

class DirectImageProcessorCollator:
    """`_is_vision_collator` accepts a bare `collator.image_processor`, so the
    column names that half declares have to be derived from it as well."""
    def __init__(self, image_names):
        self.image_processor = type("ImageProcessor", (), {
            "model_input_names": list(image_names),
        })()

    def __call__(self, features):
        return {"mine": True}


def test_columns_of_a_directly_held_image_processor_are_multimodal():
    collator = DirectImageProcessorCollator(["my_pixels"])
    trainer = StubTrainer(collator, Dataset.from_dict({
        "input_ids": [list(ROW), list(ROW)], "my_pixels": [[0.0, 1.0], [0.0, 1.0]],
    }))

    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert trainer.data_collator is collator
    assert "my_pixels" in trainer.train_dataset.column_names, "the media column was dropped"


def test_a_directly_held_image_processor_still_lets_text_only_rows_through():
    """The derivation must widen the multimodal set, not refuse every run."""
    trainer = StubTrainer(DirectImageProcessorCollator(["my_pixels"]), _text_rows())
    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)
    assert "labels" in out.train_dataset.column_names, "the text path never ran"


# ---- what the second review round found ------------------------------------

@pytest.mark.parametrize("suffix", [".avif", ".jfif", ".jpe", ".apng", ".ogv", ".wma"])
def test_less_common_media_suffixes_are_recognized(suffix):
    """A suffix missing from the list makes the column look like prose, so the
    bypass fires and the media column is dropped: VLM rows trained as text."""
    trainer = StubTrainer(MyVisionCollator(StubProcessor()), Dataset.from_dict({
        "input_ids": [list(ROW), list(ROW)],
        "url": [f"https://example.com/cat{suffix}", f"https://example.com/dog{suffix}"],
    }))
    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)
    assert "url" in trainer.train_dataset.column_names, "the media column was dropped"


def test_an_unscannable_ambiguous_column_is_refused():
    """A stream cannot be read past its prefix, so a `cat.jpg` further down goes
    unseen. Assuming text there drops the column silently, so refuse instead."""
    n = 40
    urls = ["some prose, not a file"] * n
    urls[30] = "https://example.com/cat.jpg"        # past the 16-row prefix
    dataset = Dataset.from_dict({"input_ids": [list(ROW)] * n, "url": urls})
    trainer = StubTrainer(MyVisionCollator(StubProcessor()),
                          dataset.to_iterable_dataset())

    with pytest.raises(ValueError, match = "cannot be read past its first rows") as excinfo:
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)
    assert "['url']" in str(excinfo.value), "the refusal does not name the column"


def test_an_unscannable_non_string_column_is_not_called_media():
    """Only a string can name a file, so an ambiguous name over ints stays text."""
    n = 40
    trainer = StubTrainer(MyVisionCollator(StubProcessor()), Dataset.from_dict({
        "input_ids": [list(ROW)] * n, "file": list(range(n)),
    }).to_iterable_dataset())
    train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)


def test_a_padding_collator_rebuild_keeps_the_padding_the_caller_chose():
    """`DataCollatorWithPadding` forwards these to `tokenizer.pad` exactly as the
    replacement does, so dropping them turns max_length padding into dynamic."""
    collator = DataCollatorWithPadding(tokenizer = StubProcessor(),
                                       padding = "max_length", max_length = 32,
                                       pad_to_multiple_of = 8)
    trainer = StubTrainer(collator, _text_rows())

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert out.data_collator is not collator
    assert out.data_collator.padding == "max_length"
    assert out.data_collator.max_length == 32
    assert out.data_collator.pad_to_multiple_of == 8
    assert hasattr(out.data_collator.tokenizer, "pad")


class OffsetTokenizer(StubTokenizer):
    """The caller's own text tokenizer: same text, a different vocabulary."""
    OFF = 1000

    @classmethod
    def _ids(cls, text):
        return [i + cls.OFF for i in StubTokenizer._ids(text)]


def test_a_raw_eval_split_is_tokenized_with_the_override_tokenizer():
    """The response markers were tokenized with the `tokenizer =` override, so
    encoding the eval split with the trainer's processor gives IDs that can
    never match and the whole split comes back masked."""
    trainer = StubTrainer(MyVisionCollator(StubProcessor()),
                          Dataset.from_dict({"input_ids": [[i + OffsetTokenizer.OFF
                                                            for i in ROW]] * 2}))
    trainer.eval_dataset = _raw_text_rows()

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART,
                                  tokenizer = OffsetTokenizer())

    row = out.eval_dataset[0]
    assert all(i >= OffsetTokenizer.OFF for i in row["input_ids"]), \
        "eval was encoded with the trainer's processor, not the override"
    assert any(l != -100 for l in row["labels"]), "eval labels are all -100"


def test_precomputed_labels_survive_the_raw_column_drop():
    """A raw split can carry token-level `labels` the caller already masked;
    the masking pass intersects with them, so they must not be removed with the
    raw text or every response token gets un-masked."""
    text = f"{INSTRUCTION_PART}ab{RESPONSE_PART}cd"
    ids = StubTokenizer._ids(text)
    old = [-100] * len(ids)
    old[-2] = ids[-2]                    # the caller masked the final token

    trainer = StubTrainer(DataCollatorForSeq2Seq(tokenizer = StubTokenizer()),
                          Dataset.from_dict({"text": [text] * 2,
                                             "labels": [list(old)] * 2}))
    trainer.processing_class = StubTokenizer()

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert out.train_dataset[0]["labels"] == old, "the caller's mask was lost"


# ---- provenance structs are not media --------------------------------------

def _meta_trainer(meta, n = 2, iterable = False):
    dataset = Dataset.from_dict({"input_ids": [list(ROW)] * n, "meta": meta})
    if iterable: dataset = dataset.to_iterable_dataset()
    return StubTrainer(MyVisionCollator(StubProcessor()), dataset)


def test_a_nested_provenance_struct_is_not_refused_as_media():
    """`meta = {"url": ..., "path": ...}` is what every web-scraped text corpus
    carries. Judged on the key name alone it refused a text-only run, while the
    same strings in top-level `path`/`url` columns are value-scanned and pass."""
    trainer = _meta_trainer([{"url": "https://en.wikipedia.org/wiki/Cat",
                              "path": "corpus/shard.jsonl",
                              "source": "wiki"}] * 2)

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert any(l != -100 for l in out.train_dataset[0]["labels"])


def test_a_nested_media_reference_is_still_refused():
    """The value, not the key, is what decides: a `cat.jpg` under the same key
    still points at an image the text path would drop."""
    trainer = _meta_trainer([{"path": "images/cat.jpg"}] * 2)
    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)


def test_a_nested_media_reference_past_the_sample_is_still_refused():
    """The 16-row sample never reads row 137, so the whole column is scanned."""
    n = 300
    meta = [{"path": "corpus/shard.jsonl"} for _ in range(n)]
    meta[137] = {"path": "images/cat.jpg"}
    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(_meta_trainer(meta, n), INSTRUCTION_PART, RESPONSE_PART)


def test_an_unscannable_nested_provenance_struct_is_refused_by_name():
    """A stream hands over no more than a prefix, so the rows it never reads
    cannot be called text; the refusal names the column to drop."""
    trainer = _meta_trainer([{"path": "corpus/shard.jsonl"}] * 40, n = 40,
                            iterable = True)
    with pytest.raises(ValueError, match = "cannot be read past its first rows") as excinfo:
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)
    assert "['meta']" in str(excinfo.value), "the refusal does not name the column"


def test_an_inline_image_url_part_is_still_refused():
    """`image_url` is unambiguous, so a chat turn holding one never reaches the
    value scan - the struct is refused on the key."""
    trainer = StubTrainer(MyVisionCollator(StubProcessor()), Dataset.from_dict({
        "input_ids": [list(ROW)] * 2,
        "messages": [[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://x/cat.jpg"}}]}]] * 2,
    }))
    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)


def test_an_undecoded_image_struct_is_still_refused():
    """`datasets.Image(decode = False)` is `{"bytes": ..., "path": ...}`; `bytes`
    stays a hard reject so relaxing `path` cannot let an image through."""
    trainer = _meta_trainer([{"bytes": b"\x89PNG", "path": "a"}] * 2)
    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)


# ---- what the third review round found --------------------------------------

@pytest.mark.parametrize("key, value", [
    ("image_path",     "images/cat.jpg"),
    ("video_path",     "clips/intro.mp4"),
    ("audio_path",     "clips/take1.wav"),
    ("image_file",     "images/cat.png"),
    ("image_filename", "cat.jpg"),
    ("image_url",      "https://example.com/cat.jpg"),
])
def test_a_nested_path_style_media_key_is_refused(key, value):
    """The same name is media at the top level, so it is media one level down
    too: a turn carrying `{"image_path": ...}` holds a plain string, which is
    what the schema and the sampled values both see, so only the key can say it
    is an image the text path would drop."""
    trainer = StubTrainer(MyVisionCollator(StubProcessor()), Dataset.from_dict({
        "input_ids": [list(ROW)] * 2,
        "messages": [[{"role": "user", "content": "describe", key: value}]] * 2,
    }))

    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert "messages" in trainer.train_dataset.column_names, "the media was dropped"


def test_a_nested_media_path_in_a_plain_struct_is_refused():
    """Not only conversations: a `meta` struct pointing at the media file is the
    same silent text-only run."""
    trainer = _meta_trainer([{"video_path": "clips/intro.mp4",
                              "source": "youtube"}] * 2)
    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)


@pytest.mark.parametrize("column, value", [
    ("video_filename",  "intro.mp4"),
    ("audio_filename",  "take1.wav"),
    ("image_filename",  "cat.jpg"),
    ("video_filenames", ["intro.mp4"]),
    ("audio_filenames", ["take1.wav"]),
])
def test_a_media_filename_column_is_refused(column, value):
    """`*_filename` is as ordinary a spelling as `*_path`, and only the image one
    was listed: the column is `string`, so nothing else can catch it."""
    collator = MyVisionCollator(StubProcessor())
    trainer = StubTrainer(collator, Dataset.from_dict({
        "input_ids": [list(ROW)] * 2,
        column: [value] * 2,
    }))

    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert trainer.data_collator is collator, "the user's collator was replaced"
    assert column in trainer.train_dataset.column_names, "the media column was dropped"


@pytest.mark.parametrize("key, value", [
    ("file_path",  "images/cat.jpg"),
    ("filepath",   "images/cat.jpg"),
    ("file_name",  "cat.jpg"),
    ("filename",   "cat.jpg"),
    ("file",       "clips/intro.mp4"),
    ("uri",        "https://example.com/cat.jpg"),
    ("media",      "clips/take1.wav"),
    ("source_url", "https://example.com/cat.png"),
    ("paths",      ["images/cat.jpg"]),
    ("urls",       ["https://example.com/cat.jpg"]),
])
def test_a_nested_generic_media_alias_is_scanned_by_value(key, value):
    """Only `path`/`url` were value-scanned one level down, so every other
    generic spelling the top level treats as ambiguous was called text and the
    media it pointed at was dropped with the column."""
    trainer = _meta_trainer([{key: value}] * 2)

    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert "meta" in trainer.train_dataset.column_names, "the media was dropped"


def test_a_nested_generic_alias_holding_provenance_still_passes():
    """The value still decides: a shard path under the same keys is what a text
    corpus carries, so widening the nested set must not refuse it."""
    trainer = _meta_trainer([{"file_path": "corpus/shard.jsonl",
                              "uri": "s3://bucket/key.txt"}] * 2)

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert any(l != -100 for l in out.train_dataset[0]["labels"])


class ProcessorWithoutImages:
    """A processor half with no `.pad` and no `image_processor`, so the vision
    bypass never sees it and only the `.pad` repair can fire."""
    def __init__(self):
        self.feature_extractor = object()
        self.tokenizer = StubTokenizer()


class SelfPackingCollator:
    """A user's own packing collator: it holds a processor for its own use and
    concatenates the batch itself, so it never calls `.pad` on anything."""
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        ids, position_ids = [], []
        for feature in features:
            ids += list(feature["input_ids"])
            position_ids += list(range(len(feature["input_ids"])))
        return {"input_ids": [ids], "position_ids": [position_ids]}


def test_a_self_packing_collator_holding_a_processor_is_left_alone():
    """Holding a processor is not proof of padding through it. This one packs and
    batches itself, so replacing it with DataCollatorForSeq2Seq silently drops the
    packed `position_ids` - exactly what the packing gate exists to prevent."""
    collator = SelfPackingCollator(ProcessorWithoutImages())
    trainer = StubTrainer(collator, _text_rows())
    trainer.processing_class = StubTokenizer()      # a plain text run
    trainer.args.packing = True

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert out.data_collator is collator, "the custom packing collator was replaced"
    rows = [out.train_dataset[i] for i in range(len(out.train_dataset))]
    assert "position_ids" in out.data_collator(rows), "its packing did not survive"


def test_a_seq2seq_collator_holding_a_processor_is_still_repaired_under_packing():
    """The other direction: a class that really does pad through `.pad` is still
    rebuilt even with packing on, since it dies on the first batch either way."""
    collator = DataCollatorForSeq2Seq(tokenizer = ProcessorWithoutImages())
    trainer = StubTrainer(collator, _text_rows())
    trainer.processing_class = StubTokenizer()
    trainer.args.packing = True

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert out.data_collator is not collator, "still padding through the processor"
    assert hasattr(out.data_collator.tokenizer, "pad")


# --- what the fourth review round found -----------------------------------

def test_a_model_input_column_survives_raw_text_tokenization():
    """A field the model is fed but the tokenizer does not recreate must survive.

    The raw-text tokenize path removed every original column except `labels`, so
    a per-row `sample_weight` (or any custom auxiliary target declared by
    `model.forward`) vanished before the later model-input keep-list could save
    it, leaving a missing required forward argument at train time.
    """
    class _ModelWithExtraInput:
        def forward(self, input_ids = None, attention_mask = None, labels = None,
                    sample_weight = None):
            raise AssertionError("not called")

    trainer = _text_only_trainer()
    rows = _raw_text_rows(2).add_column("sample_weight", [0.25, 0.75])
    trainer.eval_dataset = rows
    trainer.model = _ModelWithExtraInput()

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    columns = set(out.eval_dataset.column_names)
    assert "sample_weight" in columns, sorted(columns)
    assert out.eval_dataset[0]["sample_weight"] == 0.25
    # The raw text it replaced is still gone: the collator cannot stack strings.
    assert "text" not in columns, sorted(columns)


def test_a_bare_image_column_of_paths_is_refused():
    """`img` holding "cat.jpg" is media, and looked like text on schema alone.

    `_MEDIA_KEYS` already treats the bare names as unambiguous media one level
    down; the top-level list only carried the compound spellings, so a
    pretokenized VLM split keeping its images in a plain `img` column passed the
    bypass and had the column dropped before training.
    """
    trainer = _text_only_trainer()
    trainer.train_dataset = Dataset.from_dict({
        "input_ids": [list(ROW), list(ROW)],
        "img": ["images/cat.jpg", "images/dog.png"],
    })

    with pytest.raises(Exception):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)


def test_a_token_classification_collator_keeps_its_padding_settings():
    """Every pad-delegating collator's padding fields must survive the repair.

    `DataCollatorForTokenClassification` is a separate class, not a subclass of
    `DataCollatorWithPadding`, so an isinstance check on that one class dropped
    `padding = "max_length"` and silently reshaped every batch to dynamic.
    """
    from transformers import DataCollatorForTokenClassification

    collator = DataCollatorForTokenClassification(
        tokenizer = StubProcessor(),
        padding = "max_length",
        max_length = 128,
        pad_to_multiple_of = 8,
    )
    trainer = _text_only_trainer()
    trainer.data_collator = collator

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    repaired = out.data_collator
    assert repaired.padding == "max_length", repaired.padding
    assert repaired.max_length == 128, repaired.max_length
    assert repaired.pad_to_multiple_of == 8, repaired.pad_to_multiple_of


# --- what the fifth review round found ---------------------------------------

def test_a_raw_eval_split_whose_full_scan_fails_is_refused():
    """A failed exhaustive scan proves nothing, so it must not read as proof.

    A custom transform that needs a column `select_columns([name])` removed makes
    the whole-column scan raise, and only the 16 sampled rows were ever checked.
    Calling the column all-strings there lets the bypass through, and a `None`
    further down then crashes `_maybe_tokenize_dataset` partway.
    """
    def _needs_both(batch):
        return {"text": [t + str(w) for t, w in zip(batch["text"], batch["weight"])]}

    trainer = _text_only_trainer()
    trainer.eval_dataset = Dataset.from_dict({
        "text": [f"{INSTRUCTION_PART}q{i}{RESPONSE_PART}a{i}" for i in range(4)],
        "weight": [1, 2, 3, 4],
    }).with_transform(_needs_both)

    with pytest.raises(ValueError, match = "does not support response-only") as excinfo:
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)
    assert "['text']" in str(excinfo.value), "the refusal does not name the column"


@pytest.mark.parametrize("fmt", ["torch", "numpy"])
def test_a_formatted_numeric_column_still_reaches_masking(fmt):
    """A schema-proven column must not be re-judged through its row value.

    Under `with_format("torch")`/`with_format("numpy")` an auxiliary numeric
    column such as `sample_weight` or `source_id` comes back as a tensor/array,
    which `_is_plain_text` does not recognise, so a perfectly good text-only run
    was refused with the vision-collator error. The schema already judged every
    row of that column, so the row check has nothing left to add.
    """
    trainer = _text_only_trainer()
    trainer.train_dataset = Dataset.from_dict({
        "input_ids": [list(ROW)] * 4,
        "sample_weight": [0.25, 0.75, 0.5, 1.0],
        "source_id": [1, 2, 3, 4],
    }).with_format(fmt)

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert "labels" in out.train_dataset.column_names, "the split was never masked"


@pytest.mark.parametrize("fmt", ["torch", "numpy"])
def test_a_formatted_image_column_is_still_refused(fmt):
    """The guard that matters: an image is a numeric tensor under these formats
    too, so skipping the row check must only ever cover what the schema proved."""
    from datasets import Features, Image, Sequence, Value
    from PIL import Image as PILImage

    picture = PILImage.new("RGB", (2, 2))
    dataset = Dataset.from_dict(
        {"input_ids": [list(ROW)] * 2, "picture": [picture, picture]},
        features = Features({
            "input_ids": Sequence(Value("int32")),
            "picture": Image(),
        }),
    ).with_format(fmt)
    trainer = _text_only_trainer()
    trainer.train_dataset = dataset

    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)


def test_remove_unused_columns_false_keeps_the_users_columns():
    """`remove_unused_columns = False` is an explicit instruction, not a hint.

    A custom `Trainer.compute_loss` that pops `sample_weight` before calling the
    model leaves that field out of `model.forward`, so the model-input keep-list
    deleted it and the weighting was silently lost. HF's Trainer honours the flag;
    so must the strip.
    """
    trainer = _text_only_trainer()
    trainer.train_dataset = Dataset.from_dict({
        "input_ids": [list(ROW), list(ROW)],
        "sample_weight": [0.25, 0.75],
    })
    trainer.args.remove_unused_columns = False

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    columns = set(out.train_dataset.column_names)
    assert "sample_weight" in columns, sorted(columns)
    assert out.train_dataset[0]["sample_weight"] == 0.25


def test_a_bypassed_self_packing_collator_is_refused_under_packing():
    """Neither answer is right here, so neither may be taken silently.

    `_is_vision_collator` matches any collator merely holding a processor, and
    the bypass disjunct is not packing-gated, so a custom self-packing collator
    was replaced with `DataCollatorForSeq2Seq` and its packing, its
    `position_ids` and any block-attention inputs went with it. Keeping it is no
    safer: its `__call__` may rebuild `labels` over the mask just written.
    """
    collator = SelfPackingCollator(StubProcessor())   # a processor, so bypassed
    trainer = StubTrainer(collator, _text_rows())
    trainer.args.packing = True

    with pytest.raises(ValueError, match = "does not support response-only") as excinfo:
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)
    assert "SelfPackingCollator" in str(excinfo.value), "the refusal does not name the collator"
    assert trainer.data_collator is collator, "the user's collator was replaced anyway"


def test_a_bypassed_self_packing_collator_is_untouched_without_packing():
    """The blast radius: with packing off there is nothing to lose, so the
    replacement runs exactly as before."""
    collator = SelfPackingCollator(StubProcessor())
    trainer = StubTrainer(collator, _text_rows())

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert isinstance(out.data_collator, DataCollatorForSeq2Seq)



def test_remove_unused_columns_false_survives_raw_text_tokenization():
    """The opt-out has to hold on the raw path too, not just the pretokenized one.

    Tokenization runs first, and its own strip kept only `labels` and the declared
    forward parameters, so a `sample_weight` that a custom `compute_loss` pops was
    already deleted by the time the model-input keep-list consulted the flag.
    """
    trainer = _text_only_trainer()
    trainer.eval_dataset = _raw_text_rows(2).add_column("sample_weight", [0.25, 0.75])
    trainer.args.remove_unused_columns = False

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    columns = set(out.eval_dataset.column_names)
    assert "sample_weight" in columns, sorted(columns)
    assert out.eval_dataset[0]["sample_weight"] == 0.25
    # The text just consumed still goes: no collator can stack a string.
    assert "text" not in columns, sorted(columns)


def test_remove_unused_columns_false_survives_a_raw_train_split():
    """Same strip, the train side of it, reached with a plain text collator."""
    rows = _raw_text_rows(2).add_column("sample_weight", [0.25, 0.75])
    trainer = StubTrainer(DataCollatorForSeq2Seq(tokenizer = StubTokenizer()), rows)
    trainer.args.remove_unused_columns = False

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    columns = set(out.train_dataset.column_names)
    assert "sample_weight" in columns, sorted(columns)
    assert "text" not in columns, sorted(columns)


# ---- round N: three more Codex items ---------------------------------------

def test_the_packing_refusal_lands_before_the_dataset_is_touched():
    """It is a deterministic configuration error, so it must not cost a full
    tokenize/map/mask/filter pass over a large corpus first -- and the failed
    call used to leave the trainer's datasets rewritten behind it."""
    collator = LabelRebuildingVisionCollator(StubProcessor())
    rows = Dataset.from_dict({"text": ["a" * 4, "b" * 4]})
    trainer = StubTrainer(collator, rows)
    trainer.args.packing = True
    before = trainer.train_dataset

    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert trainer.train_dataset is before, "the split was mutated before refusing"
    assert trainer.train_dataset.column_names == ["text"]


def test_the_packing_refusal_does_not_map_the_eval_split_either():
    collator = LabelRebuildingVisionCollator(StubProcessor())
    trainer = StubTrainer(collator, _text_rows())
    trainer.eval_dataset = Dataset.from_dict({"input_ids": [list(ROW)]})
    trainer.args.packing = True
    before = trainer.eval_dataset

    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)
    assert trainer.eval_dataset is before


def test_a_type_tagged_struct_column_is_still_scanned():
    """`{"type": "image", "content": "cat.jpg"}` is all `string`, so the schema
    called the column plain and marked it proven -- which is exactly what makes
    `_row_is_plain_text` skip it, so the `type` tag was never read and the image
    column was dropped, training the row as text."""
    rows = Dataset.from_dict({
        "input_ids": [list(ROW), list(ROW)],
        "part": [{"type": "text", "content": "hello"},
                 {"type": "image", "content": "cat.jpg"}],
    })
    trainer = StubTrainer(MyVisionCollator(StubProcessor()), rows)
    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)


def test_a_type_tagged_struct_of_real_text_still_passes():
    """The scan answers per value, so an ordinary tagged text part is fine and
    the run is not refused for carrying a `type` field."""
    rows = Dataset.from_dict({
        "input_ids": [list(ROW), list(ROW)],
        "part": [{"type": "text", "content": "hello"},
                 {"type": "text", "content": "there"}],
    })
    trainer = StubTrainer(MyVisionCollator(StubProcessor()), rows)
    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)
    assert out is trainer


def test_a_nested_type_tag_is_scanned_too():
    """A tag one level down inside a list of turns is the same shape."""
    rows = Dataset.from_dict({
        "input_ids": [list(ROW), list(ROW)],
        "turns": [[{"type": "text", "content": "hi"}],
                  [{"type": "image", "content": "dog.png"}]],
    })
    trainer = StubTrainer(MyVisionCollator(StubProcessor()), rows)
    with pytest.raises(ValueError, match = "does not support response-only"):
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)


def test_a_custom_label_pad_value_survives_the_repair():
    """DataCollatorForSeq2Seq takes `label_pad_token_id` too, so a caller who
    chose one keeps it; dropping it padded with -100 instead."""
    from transformers import DataCollatorForTokenClassification
    collator = DataCollatorForTokenClassification(
        tokenizer = StubProcessor(), padding = "max_length", max_length = 32,
        label_pad_token_id = -1,
    )
    trainer = StubTrainer(collator, _text_rows())
    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)
    assert isinstance(out.data_collator, DataCollatorForSeq2Seq)
    assert out.data_collator.label_pad_token_id == -1
    assert out.data_collator.padding == "max_length"
    assert out.data_collator.max_length == 32


def test_the_default_label_pad_value_is_unchanged():
    """The copy must not move the default off -100 for everyone else."""
    from transformers import DataCollatorForTokenClassification
    collator = DataCollatorForTokenClassification(tokenizer = StubProcessor())
    trainer = StubTrainer(collator, _text_rows())
    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)
    assert out.data_collator.label_pad_token_id == -100
