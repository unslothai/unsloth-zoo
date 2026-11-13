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

"""Conversational prompt/completion rows get None-filled keys from Arrow's heterogeneous struct
unification; sft_prepare_dataset must strip them like the messages path does, so 'is defined'
template branches never render '(None)' into the text or trained labels. CPU-only, offline tokenizer."""

import pytest

pytest.importorskip("datasets")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from unsloth_zoo.dataset_utils import sft_prepare_dataset


def build_tokenizer():
    # Whitespace tokenizer whose template emits a name tag behind `is defined`.
    words = "<pad> <eos> <unk> user: assistant: hi hey hello hi there (None) (bob) (ann)".split()
    vocab = {w: i for i, w in enumerate(words)}
    tok = Tokenizer(models.WordLevel(vocab = vocab, unk_token = "<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    t = PreTrainedTokenizerFast(
        tokenizer_object = tok, eos_token = "<eos>", pad_token = "<pad>", unk_token = "<unk>",
    )
    t.chat_template = (
        "{%- for m in messages %}"
        "{{ m['role'] }}: "
        "{% if m['name'] is defined %}({{ m['name'] }}) {% endif %}"
        "{{ m['content'] }} \n"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}assistant: {%- endif %}"
    )
    return t


class Args:
    max_length = 512
    max_seq_length = 512
    dataset_text_field = "text"
    remove_unused_columns = True
    dataset_num_proc = None
    assistant_only_loss = False
    completion_only_loss = True


class Trainer:
    def __init__(self):
        self.model = None
        self.data_collator = None
        self.completion_only_loss = True


def rows():
    # Row 1 carries no 'name' on any message; Arrow unifies the struct with name=None.
    return Dataset.from_list([
        {
            "prompt": [{"role": "user", "content": "hi", "name": "bob"}],
            "completion": [{"role": "assistant", "content": "hello", "name": "ann"}],
        },
        {
            "prompt": [{"role": "user", "content": "hey"}],
            "completion": [{"role": "assistant", "content": "hi there"}],
        },
    ])


def _clean(messages):
    return [
        {k: v for k, v in m.items() if v is not None} if isinstance(m, dict) else m
        for m in messages
    ]


def test_pc_rows_strip_none_keys_like_the_messages_path():
    tokenizer = build_tokenizer()
    dataset = rows()
    # Arrow unifies heterogeneous message structs, filling 'name' with None on row 1.
    assert dataset[1]["prompt"][0].get("name", "missing") is None

    prepared = sft_prepare_dataset(
        Trainer(), dataset, tokenizer, Args(),
        packing = False, formatting_func = None, dataset_name = "train",
    )
    expected = tokenizer.apply_chat_template(
        _clean(dataset[1]["prompt"]) + _clean(dataset[1]["completion"]),
        tokenize = True, return_dict = False,
    )
    produced = list(prepared[1]["input_ids"])
    assert produced == expected, produced
