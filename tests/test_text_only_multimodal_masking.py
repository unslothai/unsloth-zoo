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

"""train_on_responses_only on a text-only fine-tune of a multimodal model must not hit the vision-collator check;
already-tokenized rows keep dataset-level masking; CPU-only with a stub tokenizer."""

import pytest
from datasets import Dataset

from unsloth_zoo.dataset_utils import train_on_responses_only


INSTRUCTION_PART = "<|user|>"
RESPONSE_PART = "<|assistant|>"
USER_ID, ASSISTANT_ID = 1, 2
ROW = [USER_ID, 10, 11, ASSISTANT_ID, 20, 21]


class StubEncoding(dict):
    """Mimics real tokenizer output, a dict that also allows attribute access; datasets.map rejects non-dict output and the raw-text path passes it straight in."""
    __getattr__ = dict.__getitem__


class StubTokenizer:
    # Accepts **kwargs because the raw-text path calls with truncation/max_length, which nothing reached while that path still refused.
    def __call__(self, text, add_special_tokens = False, **kwargs):
        if isinstance(text, (list, tuple)):
            ids = [self(t).input_ids for t in text]
            return StubEncoding(input_ids = ids,
                                attention_mask = [[1] * len(r) for r in ids])
        ids = []
        rest = text
        while rest:
            for marker, tid in ((INSTRUCTION_PART, USER_ID), (RESPONSE_PART, ASSISTANT_ID)):
                if rest.startswith(marker):
                    ids.append(tid)
                    rest = rest[len(marker):]
                    break
            else:
                ids.append(ord(rest[0]))
                rest = rest[1:]
        return StubEncoding(input_ids = ids, attention_mask = [1] * len(ids))


class StubProcessor:
    """A processor: what a multimodal model hands the trainer as its tokenizer."""
    def __init__(self):
        self.image_processor = object()
        self.tokenizer = StubTokenizer()


class TextCollator:
    """A plain text collator that happens to hold the processor."""
    def __init__(self, processor):
        self.tokenizer = processor


class UnslothVisionDataCollator:
    """Name-matched by the real check; rebuilds labels at collate time."""
    def __init__(self, processor):
        self.processor = processor
        self.image_processor = processor.image_processor


class StubTrainer:
    def __init__(self, collator, train_dataset):
        self.data_collator = collator
        self.train_dataset = train_dataset
        self.eval_dataset = None
        self.processing_class = StubProcessor()
        self.args = type("Args", (), {"packing": False})()


def _pretokenized():
    return Dataset.from_dict({"input_ids": [list(ROW), list(ROW)]})


def _raw_text():
    return Dataset.from_dict({"text": ["<|user|>hi<|assistant|>yes"] * 2})


def test_pretokenized_rows_get_dataset_level_masking():
    """The regression: this used to raise ValueError."""
    processor = StubProcessor()
    trainer = StubTrainer(TextCollator(processor), _pretokenized())

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    labels = out.train_dataset["labels"]
    assert len(labels) == 2
    for row in labels:
        assert row[:3] == [-100, -100, -100], row
        assert any(label != -100 for label in row), "everything was masked"


def test_raw_text_rows_are_tokenized_and_masked():
    """Raw text must be tokenized and masked, not refused: with TRL 0.22.2 the text column is gone after
    _maybe_tokenize_dataset, leaving nothing for a collator to rebuild labels from; refusing would have broken four
    shipped notebooks."""
    processor = StubProcessor()
    trainer = StubTrainer(TextCollator(processor), _raw_text())

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert "text" not in out.train_dataset.column_names, \
        "the consumed text column must not survive for a collator to re-read"
    labels = out.train_dataset["labels"]
    for row in labels:
        assert any(l == -100 for l in row), "nothing was masked"
        assert any(l != -100 for l in row), "everything was masked"


def test_unsloth_vision_collator_is_configured_not_bypassed():
    """The real vision path must be untouched even with pretokenized rows."""
    processor = StubProcessor()
    collator = UnslothVisionDataCollator(processor)
    trainer = StubTrainer(collator, _pretokenized())

    out = train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)

    assert out is trainer
    assert callable(collator.train_on_responses_only)
    assert "labels" not in out.train_dataset.column_names


def test_iterable_dataset_clears_the_guard():
    """IterableDataset has no column_names, so detection peeks a row instead; only the guard is asserted, not the text path behind it."""
    trainer = StubTrainer(TextCollator(StubProcessor()),
                          _pretokenized().to_iterable_dataset())

    try:
        train_on_responses_only(trainer, INSTRUCTION_PART, RESPONSE_PART)
    except ValueError as exception:
        pytest.fail(f"pretokenized iterable dataset was refused: {exception}")
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
