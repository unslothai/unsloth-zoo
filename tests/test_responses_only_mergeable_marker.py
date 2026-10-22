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

# aya-vision renders <|START_RESPONSE|> as plain text; byte-level BPE glues its "|>" onto the answer
# ("|>\\sigma" -> "|>\\"), so the full marker never appears and the row trained on nothing.

import pytest

tokenizers = pytest.importorskip("tokenizers")
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from unsloth_zoo.dataset_utils import train_on_responses_only  # noqa: E402

SPECIALS = ["<PAD>", "<BOS>", "<EOT>", "<USER>", "<CHAT>"]
INSTRUCTION_PART = "<|END_RESPONSE|><EOT><USER>"
RESPONSE_PART = "<CHAT><|START_RESPONSE|>"
ANSWERS = ["\\sigma ^ { a }", "{ \\frac { N } { M } }", "D _ { \\mu }", "x + y = 1"]


def _render(question, answer):
    return f"<BOS><USER>{question}<EOT>{RESPONSE_PART}{answer}<|END_RESPONSE|><EOT>"


def _tokenizer():
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space = False)
    tok.decoder = decoders.ByteLevel()
    corpus = [_render("Write the LaTeX.", a) for a in ANSWERS] * 50
    trainer = trainers.BpeTrainer(
        vocab_size = 600, special_tokens = SPECIALS,
        initial_alphabet = pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(corpus, trainer = trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object = tok, bos_token = "<BOS>", eos_token = "<EOT>", pad_token = "<PAD>",
        additional_special_tokens = ["<USER>", "<CHAT>"],
    )


def test_premise_marker_edge_merges_with_answer():
    tok = _tokenizer()
    marker = tok(RESPONSE_PART, add_special_tokens = False).input_ids
    row = tok(_render("Write the LaTeX.", ANSWERS[0]), add_special_tokens = False).input_ids
    assert not any(row[i : i + len(marker)] == marker for i in range(len(row)))


@pytest.mark.parametrize("answer", ANSWERS)
def test_every_row_supervises_its_answer(answer):
    tok = _tokenizer()
    masker = train_on_responses_only(
        None, INSTRUCTION_PART, RESPONSE_PART, tokenizer = tok, return_function = True,
    )
    ids = tok(_render("Write the LaTeX.", answer), add_special_tokens = False).input_ids
    labels = masker({"input_ids": [ids]})["labels"][0]
    kept = [t for t, l in zip(ids, labels) if l != -100]
    text = tok.decode(kept)
    assert answer in text
    assert "Write the LaTeX" not in text and "<USER>" not in text


def test_special_token_markers_keep_their_core():
    from unsloth_zoo.dataset_utils import _find_common_token_ids
    tok = _tokenizer()
    ids = tok("<EOT><USER>", add_special_tokens = False).input_ids
    core, left, right = _find_common_token_ids("<EOT><USER>", tok, force_match = True)
    assert (core, left, right) == (ids, [], [])


def test_plain_text_only_marker_keeps_its_core():
    # No added token to anchor a shrunk core, so the unstable edge stays required.
    from unsloth_zoo.dataset_utils import _find_common_token_ids
    tok = _tokenizer()
    ids = tok("<|START_RESPONSE|>", add_special_tokens = False).input_ids
    assert _find_common_token_ids("<|START_RESPONSE|>", tok, force_match = True) == (ids, [], [])
