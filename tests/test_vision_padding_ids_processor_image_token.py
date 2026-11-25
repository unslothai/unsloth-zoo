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
        # Fast tokenizers raise on None / AddedToken, so the helper must only forward non-empty strings.
        assert all(isinstance(t, str) and t for t in tokens), tokens
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
    # Outside IMAGE_TOKENS, so only the tokenizer.image_token lookup can find it.
    tok = _Tokenizer({"<|test_tokenizer_only_image|>": 7})
    tok.image_token = "<|test_tokenizer_only_image|>"
    assert _ids(_Processor(tok, None)) == {1, 7}
    assert _ids(tok) == {1, 7}


@pytest.mark.parametrize("invalid", [None, "", object()])
def test_invalid_processor_token_is_ignored(invalid):
    tok = _Tokenizer({"": 9})
    proc = _Processor(tok, invalid)
    proc.audio_token = invalid
    proc.video_token = invalid
    assert _ids(proc) == {1}


def test_processor_only_audio_and_video_tokens_are_padding_ids():
    tok = _Tokenizer({"<sound>": 500, "<|test_video_slot|>": 501, "<|end_message|>": 200004})
    proc = _Processor(tok, None)
    proc.audio_token = "<sound>"
    proc.video_token = "<|test_video_slot|>"
    assert _ids(proc) == {1, 500, 501}


class _AudioTokenizer(_Tokenizer):
    padding_side = "right"


def _truncating_collator(max_seq_length):
    from unsloth_zoo.vision_utils import UnslothVisionDataCollator

    collator = UnslothVisionDataCollator.__new__(UnslothVisionDataCollator)
    # Audio Flamingo 3 shape: <sound> lives on the processor, not the tokenizer.
    proc = _Processor(_AudioTokenizer({"<sound>": 500}), None)
    proc.audio_token = "<sound>"
    collator.processor = proc
    collator.max_seq_length = max_seq_length
    collator.truncation = True
    return collator


def test_truncation_into_processor_only_audio_span_raises():
    torch = pytest.importorskip("torch")
    batch = {
        "input_ids": torch.tensor([[2, 3, 500, 500, 500, 6]]),
        "attention_mask": torch.ones(1, 6, dtype=torch.long),
    }
    with pytest.raises(ValueError, match="audio tokens"):
        _truncating_collator(3)._truncate_sequence_tensors(batch, seq_len=6)


def test_truncation_of_text_only_keeps_processor_only_audio_span():
    torch = pytest.importorskip("torch")
    batch = {
        "input_ids": torch.tensor([[2, 500, 500, 500, 5, 6]]),
        "attention_mask": torch.ones(1, 6, dtype=torch.long),
    }
    out = _truncating_collator(4)._truncate_sequence_tensors(batch, seq_len=6)
    assert out["input_ids"].tolist() == [[2, 500, 500, 500]]


def test_truncation_ignores_non_string_tokenizer_audio_token():
    torch = pytest.importorskip("torch")
    collator = _truncating_collator(4)
    collator.processor.tokenizer.audio_token = object()  # AddedToken-like: a fast tokenizer raises TypeError on it
    batch = {
        "input_ids": torch.tensor([[2, 500, 500, 500, 5, 6]]),
        "attention_mask": torch.ones(1, 6, dtype=torch.long),
    }
    assert collator._truncate_sequence_tensors(batch, seq_len=6)["input_ids"].tolist() == [[2, 500, 500, 500]]
