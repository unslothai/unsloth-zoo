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


"""A placeholder token held only on the processor (Inkling 2) must still be masked from the loss."""

import pytest

pytest.importorskip("torch")


class _Tokenizer:
    unk_token_id = 0
    pad_token_id = 1

    def __init__(self, vocab):
        self.vocab = vocab

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(t, self.unk_token_id) for t in tokens]


class _Processor:
    def __init__(self, tokenizer, image_token):
        self.tokenizer = tokenizer
        self.image_token = image_token


def _ids(obj):
    from unsloth_zoo.vision_utils import get_padding_tokens_ids

    return set(get_padding_tokens_ids(obj).tolist())


def test_processor_only_image_token_is_a_padding_id():
    tok = _Tokenizer({"<|unused_200054|>": 200054, "<|end_message|>": 200004})
    ids = _ids(_Processor(tok, "<|unused_200054|>"))
    assert 200054 in ids
    assert 1 in ids  # pad
    assert 200004 not in ids and 0 not in ids


def test_tokenizer_image_token_still_counts_and_none_is_ignored():
    tok = _Tokenizer({"<image_soft_token>": 7})
    tok.image_token = "<image_soft_token>"
    assert 7 in _ids(_Processor(tok, None))
    assert 7 in _ids(tok)
