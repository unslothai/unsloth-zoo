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


# Qwen2-style pre-tokenizer groups runs of newlines, so the marker's trailing "\n" merges with an answer
# that starts with "\n" ("\n\n" is one token) and the exact marker never appears (unsloth-zoo#2363).

import pytest

tokenizers = pytest.importorskip("tokenizers")
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from unsloth_zoo.dataset_utils import train_on_responses_only  # noqa: E402

QWEN2_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+"
)
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"


def _render(question, answer):
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>\n"


def _tokenizer():
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN2_SPLIT), behavior = "isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space = False, use_regex = False),
    ])
    tok.decoder = decoders.ByteLevel()
    corpus = [_render("Question", a) for a in ("Answer.", "\nAnswer.", "One\n\nTwo")] * 50
    trainer = trainers.BpeTrainer(
        vocab_size = 400, special_tokens = ["<|im_start|>", "<|im_end|>"],
        initial_alphabet = pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(corpus, trainer = trainer)
    return PreTrainedTokenizerFast(tokenizer_object = tok, eos_token = "<|im_end|>", pad_token = "<|im_end|>")


def _trained(tok, question, answer):
    masker = train_on_responses_only(
        None, INSTRUCTION_PART, RESPONSE_PART, tokenizer = tok, return_function = True,
    )
    ids = tok(_render(question, answer), add_special_tokens = False).input_ids
    labels = masker({"input_ids": [ids]})["labels"][0]
    return tok.decode([t for t, lab in zip(ids, labels) if lab != -100])


def test_premise_marker_newline_merges_with_answer():
    tok = _tokenizer()
    marker = tok(RESPONSE_PART, add_special_tokens = False).input_ids
    row = tok(_render("Question", "\nAnswer."), add_special_tokens = False).input_ids
    assert not any(row[i : i + len(marker)] == marker for i in range(len(row)))


@pytest.mark.parametrize("answer", ["Answer.", "\nAnswer.", "\n\nAnswer."])
def test_answer_trained_with_or_without_leading_newline(answer):
    text = _trained(_tokenizer(), "Question", answer)
    assert answer.lstrip("\n") + "<|im_end|>" in text
    assert "Question" not in text and "user" not in text
