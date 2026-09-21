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

"""The collator swap must not throw away a padding-free batch.

Unsloth auto-enables `padding_free` and wraps `torch_call` to add
`packed_seq_lengths`; replacing the instance dropped both while
`args.padding_free` stayed True. Real TRL collator, not a stand-in: a mock
would assert nothing about its flattening or its use of `labels`.
"""

import inspect
import re

import pytest

torch = pytest.importorskip("torch")
sft_trainer = pytest.importorskip("trl.trainer.sft_trainer")
from transformers import DataCollatorForSeq2Seq  # noqa: E402
from transformers import DataCollatorForLanguageModeling as HFLanguageModeling  # noqa: E402
from datasets import Dataset  # noqa: E402

from unsloth_zoo.dataset_utils import (  # noqa: E402
    is_trl_padding_free_collator,
    train_on_responses_only,
)

TRLLanguageModeling = sft_trainer.DataCollatorForLanguageModeling

INSTRUCTION_PART = "<|user|>"
RESPONSE_PART = "<|assistant|>"
USER_ID, ASSISTANT_ID, PAD_ID = 1, 2, 0

# Unequal, so padded [2, 7] and flattened [1, 11] cannot be confused.
LONG_ROW = [USER_ID, 10, 11, 12, ASSISTANT_ID, 20, 21]
SHORT_ROW = [USER_ID, 13, ASSISTANT_ID, 22]


class _Encoding(dict):
    """Mapping for `datasets.map` that also answers `.input_ids`."""
    @property
    def input_ids(self):
        return self["input_ids"]


class StubTokenizer:
    padding_side = "right"
    pad_token_id = PAD_ID
    model_input_names = ["input_ids", "attention_mask"]

    @staticmethod
    def _ids(text):
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
        return {"input_ids": [f["input_ids"] + [PAD_ID] * (width - len(f["input_ids"]))
                              for f in features]}


class StubTrainer:
    def __init__(self, collator, train_dataset, packing = False):
        self.data_collator = collator
        self.train_dataset = train_dataset
        self.eval_dataset = None
        self.processing_class = StubTokenizer()
        self.args = type("Args", (), {
            "packing": packing, "max_length": 64, "dataset_text_field": "text",
        })()


def _rows():
    return Dataset.from_dict({"input_ids": [list(LONG_ROW), list(SHORT_ROW)]})


# Across `trl>=0.18.2,!=0.19.0,<=1.13.0` the signature moves: `padding_free`
# arrives in 0.19.1, `completion_only_loss` is gone by 1.13.0. Build from it.
_TRL_FIELDS = set(inspect.signature(TRLLanguageModeling.__init__).parameters)
_PADDING_FREE_SUPPORTED = "padding_free" in _TRL_FIELDS

needs_padding_free = pytest.mark.skipif(
    not _PADDING_FREE_SUPPORTED,
    reason = "this TRL has no padding-free collator, so there is nothing to preserve",
)


def _trl_collator(padding_free = True, cls = None):
    cls = cls or TRLLanguageModeling
    fields = set(inspect.signature(cls.__init__).parameters)
    kwargs = {"pad_token_id": PAD_ID}
    if "padding_free" in fields:        kwargs["padding_free"] = padding_free
    if "completion_only_loss" in fields: kwargs["completion_only_loss"] = False
    return cls(**kwargs)


def _wrap_like_unsloth(collator):
    """What `enable_padding_free_metadata` installs, lost with the instance."""
    original = collator.torch_call

    def torch_call_with_lengths(examples):
        lengths = [len(e["input_ids"]) for e in examples]
        batch = original(examples)
        batch["packed_seq_lengths"] = torch.tensor(lengths, dtype = torch.int32)
        return batch

    collator.torch_call = torch_call_with_lengths
    collator._unsloth_padding_free_lengths_wrapped = True
    return collator


def _collate(trainer):
    rows = [trainer.train_dataset[i] for i in range(len(trainer.train_dataset))]
    return trainer.data_collator(rows)


@needs_padding_free
def test_a_padding_free_trl_collator_is_not_swapped_for_a_padding_one():
    collator = _trl_collator()
    out = train_on_responses_only(StubTrainer(collator, _rows()),
                                  INSTRUCTION_PART, RESPONSE_PART)
    assert out.data_collator is collator
    batch = _collate(out)
    assert list(batch["input_ids"].shape) == [1, len(LONG_ROW) + len(SHORT_ROW)]
    # TRL carried position ids under `attention_mask` before 0.20, `position_ids` since.
    positions = batch.get("position_ids", batch.get("attention_mask"))
    assert positions is not None
    assert positions.flatten().tolist() == \
        list(range(len(LONG_ROW))) + list(range(len(SHORT_ROW)))


@needs_padding_free
def test_unsloths_sequence_length_wrapper_survives_the_masking_pass():
    collator = _wrap_like_unsloth(_trl_collator())
    out = train_on_responses_only(StubTrainer(collator, _rows()),
                                  INSTRUCTION_PART, RESPONSE_PART)
    assert getattr(out.data_collator, "_unsloth_padding_free_lengths_wrapped", False)
    batch = _collate(out)
    assert batch["packed_seq_lengths"].tolist() == [len(LONG_ROW), len(SHORT_ROW)]


@needs_padding_free
def test_the_preserved_collator_keeps_the_response_only_labels():
    """TRL reads the masking pass's `labels` instead of rebuilding from `input_ids`."""
    out = train_on_responses_only(StubTrainer(_trl_collator(), _rows()),
                                  INSTRUCTION_PART, RESPONSE_PART)
    batch = _collate(out)
    ids = batch["input_ids"].flatten().tolist()
    labels = batch["labels"].flatten().tolist()
    supervised = [i for i, l in zip(ids, labels) if l != -100]
    # After each ASSISTANT marker only; TRL also masks index 0 of every sequence.
    assert supervised == [20, 21, 22]
    assert all(i == l for i, l in zip(ids, labels) if l != -100)


# Negative controls: each fails under a specific widening of the guard.
def test_a_collator_that_merely_carries_padding_free_is_still_replaced():
    class NotATrlCollator:
        padding_free = True
        def __call__(self, features): return {"mine": True}

    out = train_on_responses_only(StubTrainer(NotATrlCollator(), _rows()),
                                  INSTRUCTION_PART, RESPONSE_PART)
    assert isinstance(out.data_collator, DataCollatorForSeq2Seq)


def test_the_transformers_language_modeling_collator_is_not_exempted():
    """Same class name, different project: the module prefix separates them."""
    collator = HFLanguageModeling(tokenizer = StubTokenizer(), mlm = False)
    collator.padding_free = True
    out = train_on_responses_only(StubTrainer(collator, _rows()),
                                  INSTRUCTION_PART, RESPONSE_PART)
    assert isinstance(out.data_collator, DataCollatorForSeq2Seq)


@needs_padding_free
def test_a_trl_collator_with_padding_free_off_is_still_replaced():
    """The exemption is about the mode, not the class."""
    out = train_on_responses_only(StubTrainer(_trl_collator(padding_free = False), _rows()),
                                  INSTRUCTION_PART, RESPONSE_PART)
    assert isinstance(out.data_collator, DataCollatorForSeq2Seq)


@needs_padding_free
def test_an_ordinary_subclass_is_exempted_through_its_ancestor():
    class UsersOwnCollator(TRLLanguageModeling):
        pass

    collator = _trl_collator(cls = UsersOwnCollator)
    out = train_on_responses_only(StubTrainer(collator, _rows()),
                                  INSTRUCTION_PART, RESPONSE_PART)
    assert out.data_collator is collator


# The predicate on its own, so a shim can import it instead of re-deriving the
# MRO walk, and so these cases do not need a trainer to reach it.
@needs_padding_free
def test_the_predicate_accepts_only_trls_padding_free_collator():
    class NotATrlCollator:
        padding_free = True

    class UsersOwnCollator(TRLLanguageModeling):
        pass

    hf = HFLanguageModeling(tokenizer = StubTokenizer(), mlm = False)
    hf.padding_free = True

    assert is_trl_padding_free_collator(_trl_collator()) is True
    assert is_trl_padding_free_collator(_trl_collator(cls = UsersOwnCollator)) is True
    assert is_trl_padding_free_collator(_trl_collator(padding_free = False)) is False
    assert is_trl_padding_free_collator(NotATrlCollator()) is False
    assert is_trl_padding_free_collator(hf) is False
    assert is_trl_padding_free_collator(None) is False
    assert is_trl_padding_free_collator(DataCollatorForSeq2Seq(StubTokenizer())) is False
