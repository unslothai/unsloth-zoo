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

"""Prompt/completion rows whose prompt ends in whitespace: byte-level BPE merges that space into the completion's
first token, so masking `len(tokenize(prompt))` tokens hid the start of the answer (a one-word answer was not
trained at all). The completion now starts where the two tokenizations diverge, as in TRL's SFTTrainer."""

import pytest
from datasets import Dataset, IterableDataset

tokenizers = pytest.importorskip("tokenizers")
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

import unsloth_zoo.dataset_utils as dataset_utils  # noqa: E402
from unsloth_zoo.dataset_utils import _prompt_completion_boundary, sft_prepare_dataset  # noqa: E402


@pytest.fixture(autouse = True)
def _keep_completion_mask(monkeypatch):
    # Keep completion_mask rather than folding it into labels, which depends on the installed TRL.
    monkeypatch.setattr(dataset_utils, "trl_collator_applies_masks", lambda: True)


# Qwen2 / Qwen3 pre-tokenizer: a leading space stays attached to the following letters.
QWEN2_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+"
)
EOS = "<|im_end|>"
# ChatML without trimming, like Qwen2.5 / Typhoon 2.5: the generation prompt ends in "\n".
CHATML = (
    "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>\\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)

ROWS = [
    # The prompt's trailing space merges with the answer: " two", " สอง".
    {"prompt": "Question: one plus one? Answer: ", "completion": "two"},
    {"prompt": "คำถาม: หนึ่งบวกหนึ่ง คำตอบ: ", "completion": "สอง"},
    # Controls: the space already sits on the completion side, so nothing merges.
    {"prompt": "Question: one plus one? Answer:", "completion": " two"},
    {"prompt": "คำถาม: หนึ่งบวกหนึ่ง คำตอบ:", "completion": " สอง"},
]


@pytest.fixture(scope = "module")
def tokenizer():
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN2_SPLIT), behavior = "isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space = False, use_regex = False),
    ])
    tok.decoder = decoders.ByteLevel()
    corpus = [row["prompt"].rstrip() + " " + row["completion"].strip() for row in ROWS] * 50
    trainer = trainers.BpeTrainer(
        vocab_size = 600, special_tokens = [EOS, "<|im_start|>"],
        initial_alphabet = pre_tokenizers.ByteLevel.alphabet(),
    )
    corpus += ["<|im_start|>user\nQuestion<|im_end|>\n<|im_start|>assistant\n\nAnswer.<|im_end|>\n"] * 50
    tok.train_from_iterator(corpus, trainer = trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object = tok, eos_token = EOS, pad_token = EOS)
    fast.chat_template = CHATML
    return fast


class Args:
    max_length = 128
    dataset_text_field = "text"
    remove_unused_columns = True
    dataset_num_proc = None


class DummyTrainer:
    def __init__(self):
        self.model = None
        self.data_collator = None


def _trained_texts(dataset, tok):
    prepared = sft_prepare_dataset(
        DummyTrainer(), dataset, tok, Args(), packing = False, formatting_func = None, dataset_name = "train",
    )
    out = []
    for row in prepared:
        ids, mask = list(row["input_ids"]), list(row["completion_mask"])
        assert len(ids) == len(mask)
        out.append(tok.decode([t for t, m in zip(ids, mask) if m]))
    return out


def test_premise_trailing_space_merges_into_the_answer(tokenizer):
    for row in ROWS[:2]:
        prompt_ids = tokenizer(row["prompt"]).input_ids
        joined_ids = tokenizer(row["prompt"] + row["completion"]).input_ids
        assert joined_ids[:len(prompt_ids)] != prompt_ids, row


@pytest.mark.parametrize("streaming", [False, True], ids = ["batched", "row_by_row"])
def test_answer_is_trained_when_prompt_ends_in_space(tokenizer, streaming):
    dataset = Dataset.from_list(ROWS)
    if streaming:
        dataset = IterableDataset.from_generator(lambda: iter(ROWS))
    texts = _trained_texts(dataset, tokenizer)
    for row, text in zip(ROWS, texts):
        assert text == " " + row["completion"].strip() + EOS, (row, text)
        # No prompt text leaks into the trained span.
        assert row["prompt"].strip().split()[-1] not in text


def test_boundary_unchanged_when_prompt_is_a_prefix():
    assert _prompt_completion_boundary([1, 2, 3], [1, 2, 3, 4, 5]) == 3
    assert _prompt_completion_boundary([1, 2, 3], [1, 2]) == 2


def test_boundary_is_where_tokenizations_diverge():
    # [.., 7] is the prompt's lone trailing space; [.., 9] is " two" in the joined ids.
    assert _prompt_completion_boundary([1, 2, 7], [1, 2, 9, 5]) == 2
    assert _prompt_completion_boundary([1, 2, 7], [3, 9]) == 0


def test_conversational_rows_keep_the_answer_and_mask_the_prompt(tokenizer):
    rows = [
        {"prompt": [{"role": "user", "content": "Question"}], "completion": [{"role": "assistant", "content": c}]}
        for c in ("Answer.", "\nAnswer.")
    ]
    texts = _trained_texts(Dataset.from_list(rows), tokenizer)
    for text in texts:
        assert "Answer.<|im_end|>" in text, text
        assert "Question" not in text and "user" not in text and "<|im_start|>" not in text, text
