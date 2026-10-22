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

"""Double-BOS detection in ``sft_prepare_dataset``.

Detection is ``test_text.startswith(bos_token) or bos_token in chat_template``. Arm B
needs a literal BOS, which Llama-3 templates never have (they emit ``{{- bos_token }}``),
so arm A is the only live detector - and it read ``row[field][0]``, the first CHARACTER
of a string, which no multi-character BOS can match.

Assertions read tokenized output, not stub state: under ``fork`` the map body runs in
child processes whose recorded state is invisible to the parent.
"""

import pytest
from datasets import Dataset

from unsloth_zoo.dataset_utils import sft_prepare_dataset


BOS = "<|begin_of_text|>"
BOS_ID = 128000
CONTENT_IDS = [10, 11, 12]

# BOS via a Jinja variable, so the literal is absent and arm B cannot fire.
JINJA_BOS_TEMPLATE = "{{- bos_token }}\n{%- for m in messages %}{{ m['content'] }}{%- endfor %}"
# Carries the literal, which arm B is supposed to catch.
LITERAL_BOS_TEMPLATE = "{{ '" + BOS + "' }}{{ messages }}"


class BosStubTokenizer:
    """Encodes the add_special_tokens decision into its output: one leading BOS id if the
    text starts with BOS, one more if the caller asks for specials. Correct -> one, bug -> two."""

    def __init__(self, bos_token=BOS, chat_template=JINJA_BOS_TEMPLATE):
        self.bos_token = bos_token
        self.chat_template = chat_template

    def __call__(
        self,
        texts,
        truncation=True,
        max_length=None,
        return_token_type_ids=False,
        add_special_tokens=True,
    ):
        if isinstance(texts, str):
            texts = [texts]
        rows = []
        for text in texts:
            # list-valued column arrives as a list of lists under batched=True
            if isinstance(text, (list, tuple)):
                text = text[0] if len(text) != 0 else ""
            ids = list(CONTENT_IDS)
            if self.bos_token is not None and text.startswith(self.bos_token):
                ids = [BOS_ID] + ids
            if self.bos_token is not None and add_special_tokens:
                ids = [BOS_ID] + ids
            rows.append(ids)
        return {"input_ids": rows}


class Args:
    def __init__(self, dataset_text_field="text"):
        self.max_length = 64
        self.dataset_text_field = dataset_text_field
        self.remove_unused_columns = True


class DummyTrainer:
    """Stands in for SFTTrainer: only .model is read and .data_collator is set."""

    def __init__(self):
        self.model = None
        self.data_collator = None


def _prepare(dataset, tokenizer, dataset_text_field="text"):
    prepared = sft_prepare_dataset(
        DummyTrainer(),
        dataset,
        tokenizer,
        Args(dataset_text_field),
        packing=False,
        formatting_func=None,
        dataset_name="train",
    )
    rows = [list(row["input_ids"]) for row in prepared]
    assert rows, "sft_prepare_dataset returned no rows"
    return rows


def _leading_bos_count(ids):
    count = 0
    for token in ids:
        if token != BOS_ID:
            break
        count += 1
    return count


def test_text_already_starting_with_bos_is_not_double_bos():
    """Unpatched, arm A never fires and every row gets two BOS ids."""
    dataset = Dataset.from_dict({"text": [BOS + "hello world", BOS + "second row"]})

    rows = _prepare(dataset, BosStubTokenizer())

    for ids in rows:
        assert _leading_bos_count(ids) == 1, (
            f"expected exactly one leading BOS id, got {_leading_bos_count(ids)} "
            f"in {ids} - a second {BOS!r} was prepended to text that already had one"
        )


def test_text_without_bos_still_gets_exactly_one():
    """The other direction: text with no BOS must still get one."""
    dataset = Dataset.from_dict({"text": ["hello world", "second row"]})

    rows = _prepare(dataset, BosStubTokenizer())

    for ids in rows:
        assert _leading_bos_count(ids) == 1, (
            f"expected exactly one leading BOS id, got {_leading_bos_count(ids)} in {ids}"
        )


def test_literal_bos_in_chat_template_still_detected():
    """Arm B alone. The text carries NO BOS so arm A cannot fire; with BOS-carrying
    text here the test would pass even with arm B deleted."""
    dataset = Dataset.from_dict({"text": ["hello world", "second row"]})

    rows = _prepare(dataset, BosStubTokenizer(chat_template=LITERAL_BOS_TEMPLATE))

    for ids in rows:
        assert _leading_bos_count(ids) == 0, (
            f"literal BOS in the chat template no longer suppresses the extra BOS: {ids}"
        )


def test_list_valued_text_field_is_unwrapped():
    """A list-valued field: element 0 really is the text, so unwrapping is correct."""
    dataset = Dataset.from_dict({"text": [[BOS + "hello world"], [BOS + "second row"]]})

    rows = _prepare(dataset, BosStubTokenizer())

    for ids in rows:
        assert _leading_bos_count(ids) == 1, (
            f"double BOS not detected for a list-valued text field: {ids}"
        )


def test_no_bos_token_on_tokenizer_is_a_noop():
    """No bos_token (Qwen2.5): must not crash, no BOS id may appear."""
    dataset = Dataset.from_dict({"text": ["hello world", "second row"]})

    rows = _prepare(dataset, BosStubTokenizer(bos_token=None, chat_template="{{ messages }}"))

    for ids in rows:
        assert _leading_bos_count(ids) == 0, ids
        assert ids == CONTENT_IDS, ids
