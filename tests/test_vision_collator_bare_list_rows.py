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

"""_select_messages_or_raw lets a row BE the messages list, so media extraction must not call .get on it; a bare
list-of-turns example collates like the same turns under a 'messages' key. CPU-only stub processor."""

import pytest
import torch

from unsloth_zoo.vision_utils import UnslothVisionDataCollator

PAD_ID = 0
IMG_ID = 7
VOCAB = {"<img>": IMG_ID, "<end>": 9, "a": 1, "b": 2, "x": 3, "y": 4, "z": 5, "w": 6}


class _FakeTokenizer:
    pad_token_id = PAD_ID
    padding_side = "right"

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return VOCAB.get(tokens, -1)
        return [VOCAB.get(t, -1) for t in tokens]


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.seen_images = []

    def apply_chat_template(self, messages, **kwargs):
        words = []
        for m in messages:
            content = m["content"]
            if isinstance(content, str):
                words.append(content)
                continue
            for part in content:
                words.append("<img>" if part.get("type") == "image" else part["text"])
        return " ".join(words)

    def __call__(self, text, padding=True, padding_side="right", return_tensors="pt",
                 add_special_tokens=False, **kwargs):
        self.seen_images.append(kwargs.get("images"))
        rows = [[VOCAB[t] for t in s.split()] for s in text]
        width = max(len(r) for r in rows)
        ids, mask = [], []
        for r in rows:
            pad = [PAD_ID] * (width - len(r))
            ids.append(r + pad)
            mask.append([1] * len(r) + [0] * len(pad))
        out = {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}
        if kwargs.get("images") is not None:
            out["pixel_values"] = torch.zeros(len(ids), 1, 2, 2)
        return out


def make_collator():
    collator = UnslothVisionDataCollator.__new__(UnslothVisionDataCollator)
    collator.processor = _FakeProcessor()
    collator.formatting_func = None
    collator.max_seq_length = None
    collator.truncation = False
    collator.ignore_index = -100
    collator.completion_only_loss = False
    collator.pad_to_multiple_of = None
    collator.image_size = None
    collator.patch_size = 14
    collator.padding_token_ids = torch.tensor([PAD_ID, IMG_ID])
    collator.assistant_single_content = False
    collator.dtype = torch.float32
    return collator


MESSAGES = [
    {"role": "user", "content": [{"type": "text", "text": "a"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "x"}]},
]


def test_a_bare_list_of_turns_collates_like_a_messages_key():
    """The row itself is the messages list; the extractors used to call example.get on it -> AttributeError."""
    collator = make_collator()
    out = collator([MESSAGES])
    assert out["input_ids"].tolist() == [[1, 3]]
    assert out["labels"].tolist() == [[1, 3]]


def test_a_bare_list_row_still_reads_images_embedded_in_its_turns():
    """Media can only live inside the message parts on a bare row; it must reach the processor."""
    Image = pytest.importorskip("PIL.Image")
    collator = make_collator()
    image = Image.new("RGB", (64, 64))
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "a"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "x"}]},
    ]
    out = collator([messages])
    assert len(collator.processor.seen_images[0][0]) == 1
    assert "pixel_values" in out
