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

"""sft_prepare_dataset probes formatting_func on a single example (the TRL per-example contract), so the
batched tokenize map must call it once per row and flatten, not once per column-dict batch. CPU-only."""

from datasets import Dataset

from unsloth_zoo.dataset_utils import sft_prepare_dataset

ROWS = {
    "question": ["what is 1+1", "what is 2+2", "what is 3+3", "what is 4+4"],
    "answer": ["2", "4", "6", "8"],
}


class TinyTokenizer:
    """Char-level tokenizer, enough for sft_prepare_dataset's text path."""

    bos_token = "<bos>"
    eos_token = "<eos>"
    chat_template = ""

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for text in texts:
            if not isinstance(text, str):
                raise TypeError(f"tokenizer got non-str input: {type(text)}")
            out.append([ord(c) for c in text])
        return {"input_ids": out}


class Args:
    max_length = 512
    max_seq_length = 512
    dataset_text_field = "text"
    remove_unused_columns = True
    dataset_num_proc = None
    assistant_only_loss = False


class Trainer:
    def __init__(self):
        self.model = None
        self.data_collator = None


def formatting_func(example):
    # TRL's per-example contract: one example in, list[str] out.
    return [f"Q: {example['question']}\nA: {example['answer']}"]


def test_formatting_func_is_applied_per_example_not_per_batch():
    dataset = Dataset.from_dict(ROWS)
    prepared = sft_prepare_dataset(
        Trainer(), dataset, TinyTokenizer(), Args(),
        packing = False, formatting_func = formatting_func, dataset_name = "train",
    )
    expected = [f"Q: {q}\nA: {a}" for q, a in zip(ROWS["question"], ROWS["answer"])]
    actual = ["".join(chr(t) for t in row["input_ids"]) for row in prepared]
    assert len(actual) == len(ROWS["question"])
    assert actual == expected
