# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Prompt-completion collation: token type id routing.

Gemma 3n emits ``token_type_ids`` and Gemma 4 emits ``mm_token_type_ids``.
Both must be concatenated, flushed, and truncated in lock-step with
``input_ids``; before this fix the mm variant was left in the output as a
stale prompt-width copy whose vision block positions no longer lined up
with the flushed sequence (silent attention mis-masking on Gemma 4 12B+).

Hermetic CPU tests with a stub whitespace processor, no model or network.
"""

from __future__ import annotations

import torch

from unsloth_zoo.vision_utils import UnslothVisionDataCollator

PAD_ID = 0
IMG_ID = 7
VOCAB = {"<img>": IMG_ID, "a": 1, "b": 2, "x": 3, "y": 4, "z": 5, "w": 6}


class _FakeTokenizer:
    pad_token_id = PAD_ID
    padding_side = "right"

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return VOCAB.get(tokens, -1)
        return [VOCAB.get(t, -1) for t in tokens]


class _FakeFeatureExtractor:
    sampling_rate = 16000


class _FakeProcessor:
    """Whitespace tokenizer emitting a configurable token-type-ids key."""

    def __init__(self, tt_key):
        self.tokenizer = _FakeTokenizer()
        self.feature_extractor = _FakeFeatureExtractor()
        self.tt_key = tt_key

    def __call__(self, text, padding=True, padding_side="right", return_tensors="pt",
                 add_special_tokens=False, **kwargs):
        rows = [[VOCAB[t] for t in s.split()] for s in text]
        width = max(len(r) for r in rows)
        ids, mask = [], []
        for r in rows:
            pad = [PAD_ID] * (width - len(r))
            if padding_side == "left":
                ids.append(pad + r)
                mask.append([0] * len(pad) + [1] * len(r))
            else:
                ids.append(r + pad)
                mask.append([1] * len(r) + [0] * len(pad))
        out = {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(mask),
        }
        if self.tt_key is not None:
            # Mirrors ProcessorMixin.create_mm_token_type_ids: 0 text, 1 image
            out[self.tt_key] = (out["input_ids"] == IMG_ID).long()
        return out


def make_collator(tt_key, max_seq_length=None):
    collator = UnslothVisionDataCollator.__new__(UnslothVisionDataCollator)
    collator.processor = _FakeProcessor(tt_key)
    collator.formatting_func = None
    collator.max_seq_length = max_seq_length
    collator.truncation = max_seq_length is not None
    collator.ignore_index = -100
    collator.completion_only_loss = True
    collator.pad_to_multiple_of = None
    collator.image_size = None
    collator.patch_size = 14
    collator.padding_token_ids = torch.tensor([PAD_ID, IMG_ID])
    return collator


# Skewed lengths: row 0 has the long prompt (with an image token), row 1 has
# the long completion, so prompt left pads and completion right pads both
# appear and the flush genuinely moves tokens.
EXAMPLES = [
    {"prompt": "<img> a b", "completion": "x"},
    {"prompt": "a", "completion": "x y z w"},
]


def test_mm_token_type_ids_routed_through_pc_path():
    out = make_collator("mm_token_type_ids")(EXAMPLES)
    assert "token_type_ids" not in out
    mm = out["mm_token_type_ids"]
    # Stale pre-fix copy kept the prompt-only width (2, 3)
    assert mm.shape == out["input_ids"].shape
    # Type ids must move in lock-step with input_ids through flush/truncate
    assert torch.equal(mm, (out["input_ids"] == IMG_ID).long())


def test_token_type_ids_still_routed():
    out = make_collator("token_type_ids")(EXAMPLES)
    assert "mm_token_type_ids" not in out
    tt = out["token_type_ids"]
    assert tt.shape == out["input_ids"].shape
    assert torch.equal(tt, (out["input_ids"] == IMG_ID).long())


def test_mm_token_type_ids_truncation_stays_aligned():
    out = make_collator("mm_token_type_ids", max_seq_length=4)(EXAMPLES)
    mm = out["mm_token_type_ids"]
    assert mm.shape == out["input_ids"].shape == (2, 4)
    assert torch.equal(mm, (out["input_ids"] == IMG_ID).long())


def test_no_type_ids_emitted_is_a_noop():
    out = make_collator(None)(EXAMPLES)
    assert "token_type_ids" not in out and "mm_token_type_ids" not in out
    assert out["input_ids"].shape == out["attention_mask"].shape
