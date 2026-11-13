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

"""pad_token_id == eos_token_id (the standard recipe when a tokenizer ships no pad token) must not hide real eos
targets: pads mask by POSITION (attention_mask == 0), media placeholders by id. CPU-only stub processor."""

import torch

from unsloth_zoo.vision_utils import UnslothVisionDataCollator

EOS_ID = 9
IMG_ID = 7
VOCAB = {"<end>": EOS_ID, "<img>": IMG_ID, "a": 1, "b": 2, "x": 3, "y": 4, "z": 5, "w": 6}


class _FakeTokenizer:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
        self.padding_side = "right"

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return VOCAB.get(tokens, -1)
        return [VOCAB.get(t, -1) for t in tokens]


class _FakeProcessor:
    """Every rendered turn ends with <end>: the attended eos token is a real target, not padding."""

    def __init__(self, pad_token_id):
        self.tokenizer = _FakeTokenizer(pad_token_id)

    def apply_chat_template(self, messages, **kwargs):
        out = []
        for m in messages:
            content = m["content"]
            if isinstance(content, str):
                out.append(content)
            else:
                out += [part["text"] for part in content if part.get("type") == "text"]
            out.append("<end>")
        return " ".join(out)

    def __call__(self, text, padding=True, padding_side="right", return_tensors="pt",
                 add_special_tokens=False, **kwargs):
        pad_id = self.tokenizer.pad_token_id
        rows = [[VOCAB[t] for t in s.split()] for s in text]
        width = max(len(r) for r in rows)
        ids, mask = [], []
        for r in rows:
            pad = [pad_id] * (width - len(r))
            if padding_side == "left":
                ids.append(pad + r)
                mask.append([0] * len(pad) + [1] * len(r))
            else:
                ids.append(r + pad)
                mask.append([1] * len(r) + [0] * len(pad))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


def make_collator(pad_token_id, padding_ids):
    collator = UnslothVisionDataCollator.__new__(UnslothVisionDataCollator)
    collator.processor = _FakeProcessor(pad_token_id)
    collator.formatting_func = None
    collator.max_seq_length = None
    collator.truncation = False
    collator.ignore_index = -100
    collator.completion_only_loss = False
    collator.pad_to_multiple_of = None
    collator.image_size = None
    collator.patch_size = 14
    collator.padding_token_ids = torch.tensor(padding_ids)
    collator.assistant_single_content = False
    return collator


def _messages(text):
    return [{"role": "user", "content": [{"type": "text", "text": t}]} for t in text.split()]


def test_pad_equal_eos_masks_only_pad_positions_on_the_messages_path():
    """pad == eos: the attended <end> after each turn stays a target; only attention_mask == 0 pads mask."""
    collator = make_collator(EOS_ID, [EOS_ID, IMG_ID])
    out = collator([
        {"messages": _messages("a") + _messages("x")},
        {"messages": _messages("a")},
    ])
    ids, labels, attn = out["input_ids"], out["labels"], out["attention_mask"]
    # Row 0 renders "a <end> x <end>", row 1 "a <end>" padded on the right.
    assert ids.tolist() == [[1, EOS_ID, 3, EOS_ID], [1, EOS_ID, EOS_ID, EOS_ID]]
    assert attn.tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
    for row in range(2):
        for pos in range(ids.shape[1]):
            if attn[row, pos] == 0:
                assert labels[row, pos] == -100, "pad positions must mask"
            else:
                assert labels[row, pos] == ids[row, pos], (
                    f"attended token {ids[row, pos].item()} at ({row}, {pos}) was masked"
                )


def test_pad_equal_eos_keeps_the_completion_eos_on_the_pc_path():
    """Same aliasing through _collate_prompt_completion: an attended eos inside completion survives."""
    collator = make_collator(EOS_ID, [EOS_ID, IMG_ID])
    collator.completion_only_loss = True
    out = collator([
        {"prompt": "a <end>", "completion": "x <end>"},
        {"prompt": "a b", "completion": "x"},
    ])
    ids, labels, attn = out["input_ids"], out["labels"], out["attention_mask"]
    assert ids.tolist() == [[1, EOS_ID, 3, EOS_ID], [1, 2, 3, EOS_ID]]
    assert attn.tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]
    # completion_only_loss leaves only the completion span; inside it every attended token trains.
    assert labels[0].tolist() == [-100, -100, 3, EOS_ID]
    assert labels[1].tolist() == [-100, -100, 3, -100]


def test_a_distinct_pad_token_still_masks_exactly_the_pad_positions():
    """A real pad token: pads mask at pad positions, everything attended keeps its label."""
    collator = make_collator(0, [0, IMG_ID])
    out = collator([
        {"messages": _messages("a") + _messages("x")},
        {"messages": _messages("a")},
    ])
    ids, labels, attn = out["input_ids"], out["labels"], out["attention_mask"]
    assert ids.tolist() == [[1, EOS_ID, 3, EOS_ID], [1, EOS_ID, 0, 0]]
    for row in range(2):
        for pos in range(ids.shape[1]):
            if attn[row, pos] == 0:
                assert labels[row, pos] == -100
            else:
                assert labels[row, pos] == ids[row, pos]
