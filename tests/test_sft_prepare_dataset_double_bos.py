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

``sft_prepare_dataset`` decides whether to pass ``add_special_tokens=False`` to the
tokenizer so a dataset whose text ALREADY carries the BOS token does not get a
second one prepended. Detection has two arms:

    (test_text is not None and test_text.startswith(bos_token)) or bos_token in chat_template

Arm B only fires when the chat template contains the BOS token *as a literal*. The
Llama-3 family emits it via the Jinja variable ``{{- bos_token }}``, so the literal
is absent and arm A is the only detector. Arm A read ``row[field][0]``, which indexes
a plain string and yields its first CHARACTER, so ``startswith(bos_token)`` could
never be true for any multi-character BOS.

These tests assert on the TOKENIZED OUTPUT rather than on a flag recorded by the
stub. ``sft_prepare_dataset`` hands ``num_proc`` to ``Dataset.map`` whenever the
multiprocessing start method is ``fork`` (Linux), so the map body runs in child
processes and any state a stub records there is invisible to the parent. Reading
the returned dataset is what crosses that boundary - and doubled BOS ids in the
tokenized rows are the user-visible harm anyway.

CPU-pure and offline: the tokenizer/processor is a local stub, no weights are loaded.
"""

import pytest
from datasets import Dataset

from unsloth_zoo.dataset_utils import sft_prepare_dataset


BOS = "<|begin_of_text|>"
BOS_ID = 128000
CONTENT_IDS = [10, 11, 12]

# Mirrors the Llama-3 chat template: BOS is emitted through a Jinja variable, so the
# literal token string never appears in the template source and arm B cannot fire.
JINJA_BOS_TEMPLATE = "{{- bos_token }}\n{%- for m in messages %}{{ m['content'] }}{%- endfor %}"
# A template that does carry the literal, which arm B is supposed to catch.
LITERAL_BOS_TEMPLATE = "{{ '" + BOS + "' }}{{ messages }}"


class BosStubTokenizer:
    """Encodes the add_special_tokens decision into its OUTPUT.

    A leading BOS id appears once for a text that already starts with the BOS
    string, and once more when the caller asks for special tokens. A correct
    caller therefore yields exactly one leading BOS id in every case; the bug
    yields two.
    """

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
            # A list-valued text column arrives here as a list of lists under
            # batched=True. Mirror the unwrap the caller does for its BOS probe so
            # this stub can still tokenize that shape.
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
    """Stands in for SFTTrainer: sft_prepare_dataset only reads .model and sets
    .data_collator."""

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
    """The regression: text that already carries BOS must not be given a second
    one. Unpatched, arm A never fires and every row starts with two BOS ids."""
    dataset = Dataset.from_dict({"text": [BOS + "hello world", BOS + "second row"]})

    rows = _prepare(dataset, BosStubTokenizer())

    for ids in rows:
        assert _leading_bos_count(ids) == 1, (
            f"expected exactly one leading BOS id, got {_leading_bos_count(ids)} "
            f"in {ids} - a second {BOS!r} was prepended to text that already had one"
        )


def test_text_without_bos_still_gets_exactly_one():
    """Guard the other direction: a dataset with no leading BOS must still have one
    added by the tokenizer."""
    dataset = Dataset.from_dict({"text": ["hello world", "second row"]})

    rows = _prepare(dataset, BosStubTokenizer())

    for ids in rows:
        assert _leading_bos_count(ids) == 1, (
            f"expected exactly one leading BOS id, got {_leading_bos_count(ids)} in {ids}"
        )


def test_literal_bos_in_chat_template_still_detected():
    """Arm B must keep working: a template holding the literal BOS disables
    add_special_tokens on its own.

    The text deliberately carries NO BOS, so arm A cannot fire and the template literal is
    the only thing that can suppress the extra token. With BOS-carrying text here instead,
    arm A alone satisfies the assertion and the test passes even with arm B deleted."""
    dataset = Dataset.from_dict({"text": ["hello world", "second row"]})

    rows = _prepare(dataset, BosStubTokenizer(chat_template=LITERAL_BOS_TEMPLATE))

    for ids in rows:
        assert _leading_bos_count(ids) == 0, (
            f"literal BOS in the chat template no longer suppresses the extra BOS: {ids}"
        )


def test_list_valued_text_field_is_unwrapped():
    """Some datasets store the text field as a list of strings. Indexing element 0
    is correct there, and detection must still fire."""
    dataset = Dataset.from_dict({"text": [[BOS + "hello world"], [BOS + "second row"]]})

    rows = _prepare(dataset, BosStubTokenizer())

    for ids in rows:
        assert _leading_bos_count(ids) == 1, (
            f"double BOS not detected for a list-valued text field: {ids}"
        )


def test_no_bos_token_on_tokenizer_is_a_noop():
    """A tokenizer without a BOS token (e.g. Qwen2.5) must not crash, and no BOS id
    may appear."""
    dataset = Dataset.from_dict({"text": ["hello world", "second row"]})

    rows = _prepare(dataset, BosStubTokenizer(bos_token=None, chat_template="{{ messages }}"))

    for ids in rows:
        assert _leading_bos_count(ids) == 0, ids
        assert ids == CONTENT_IDS, ids
