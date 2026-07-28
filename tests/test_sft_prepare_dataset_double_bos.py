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
never be true for any multi-character BOS. These tests pin arm A.

CPU-pure and offline: the tokenizer/processor are local stubs, no weights are loaded.
"""

import pytest
from datasets import Dataset

from unsloth_zoo.dataset_utils import sft_prepare_dataset


BOS = "<|begin_of_text|>"

# Mirrors the Llama-3 chat template: BOS is emitted through a Jinja variable, so the
# literal token string never appears in the template source and arm B cannot fire.
JINJA_BOS_TEMPLATE = "{{- bos_token }}\n{%- for m in messages %}{{ m['content'] }}{%- endfor %}"


class RecordingTokenizer:
    """Minimal tokenizer stub that records the add_special_tokens it was called with."""

    def __init__(self, bos_token=BOS, chat_template=JINJA_BOS_TEMPLATE):
        self.bos_token = bos_token
        self.chat_template = chat_template
        self.add_special_tokens_seen = []

    def __call__(
        self,
        texts,
        truncation=True,
        max_length=None,
        return_token_type_ids=False,
        add_special_tokens=True,
    ):
        self.add_special_tokens_seen.append(add_special_tokens)
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": [[1, 2, 3] for _ in texts]}


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


def _run(dataset, tokenizer, dataset_text_field="text"):
    trainer = DummyTrainer()
    sft_prepare_dataset(
        trainer,
        dataset,
        tokenizer,
        Args(dataset_text_field),
        packing=False,
        formatting_func=None,
        dataset_name="train",
    )
    assert tokenizer.add_special_tokens_seen, "tokenizer was never invoked"
    return tokenizer.add_special_tokens_seen


def test_text_already_starting_with_bos_disables_add_special_tokens():
    """The regression: text that already carries BOS must tokenize with
    add_special_tokens=False, otherwise every sequence gets a doubled BOS."""
    dataset = Dataset.from_dict({"text": [BOS + "hello world", BOS + "second row"]})
    tokenizer = RecordingTokenizer()

    seen = _run(dataset, tokenizer)

    assert all(flag is False for flag in seen), (
        "double BOS not detected: sft_prepare_dataset tokenized with "
        f"add_special_tokens={seen}, so a second {BOS!r} is prepended to every row"
    )


def test_text_without_bos_keeps_add_special_tokens():
    """Guard the other direction: a dataset with no leading BOS must keep
    add_special_tokens=True so the tokenizer still adds one."""
    dataset = Dataset.from_dict({"text": ["hello world", "second row"]})
    tokenizer = RecordingTokenizer()

    seen = _run(dataset, tokenizer)

    assert all(flag is True for flag in seen), (
        f"add_special_tokens was disabled for text that has no leading BOS: {seen}"
    )


def test_literal_bos_in_chat_template_still_detected():
    """Arm B must keep working: a template holding the literal BOS disables
    add_special_tokens even when the text itself does not start with BOS."""
    dataset = Dataset.from_dict({"text": ["hello world", "second row"]})
    tokenizer = RecordingTokenizer(chat_template="{{ '" + BOS + "' }}{{ messages }}")

    seen = _run(dataset, tokenizer)

    assert all(flag is False for flag in seen), (
        f"literal BOS in the chat template no longer disables add_special_tokens: {seen}"
    )


def test_list_valued_text_field_is_unwrapped():
    """Some datasets store the text field as a list of strings. Indexing element 0
    is correct there, and detection must still fire."""
    dataset = Dataset.from_dict({"text": [[BOS + "hello world"], [BOS + "second row"]]})
    tokenizer = RecordingTokenizer()

    seen = _run(dataset, tokenizer)

    assert all(flag is False for flag in seen), (
        f"double BOS not detected for a list-valued text field: {seen}"
    )


def test_no_bos_token_on_tokenizer_is_a_noop():
    """A tokenizer without a BOS token (e.g. Qwen2.5) must not crash and must keep
    add_special_tokens=True."""
    dataset = Dataset.from_dict({"text": ["hello world", "second row"]})
    tokenizer = RecordingTokenizer(bos_token=None, chat_template="{{ messages }}")

    seen = _run(dataset, tokenizer)

    assert all(flag is True for flag in seen), seen
