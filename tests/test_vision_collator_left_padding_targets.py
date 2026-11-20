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

"""A left-padded batch must train on the same targets as its rows collated alone (batch size 2 equals 2 accumulation
steps of batch size 1); CPU only, fake processor."""

import pytest
import torch

from unsloth_zoo.vision_utils import UnslothVisionDataCollator

PAD_ID = 0
BOS_ID = 2
VOCAB = {"<bos>": BOS_ID, "a": 3, "b": 4, "x": 5, "y": 6, "z": 7, "w": 8}


class _FakeTokenizer:
    pad_token_id = PAD_ID

    def __init__(self, padding_side):
        self.padding_side = padding_side

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return VOCAB.get(tokens, -1)
        return [VOCAB.get(t, -1) for t in tokens]


class _FakeProcessor:
    """Pads on its tokenizer's side unless told otherwise, like a transformers processor."""

    def __init__(self, padding_side):
        self.tokenizer = _FakeTokenizer(padding_side)

    def __call__(self, text, padding=True, padding_side=None, return_tensors="pt",
                 add_special_tokens=False, **kwargs):
        side = padding_side or self.tokenizer.padding_side
        rows = [[VOCAB[t] for t in s.split()] for s in text]
        width = max(len(r) for r in rows)
        ids, mask = [], []
        for r in rows:
            pad = [PAD_ID] * (width - len(r))
            if side == "left":
                ids.append(pad + r)
                mask.append([0] * len(pad) + [1] * len(r))
            else:
                ids.append(r + pad)
                mask.append([1] * len(r) + [0] * len(pad))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


def make_collator(padding_side):
    collator = UnslothVisionDataCollator.__new__(UnslothVisionDataCollator)
    collator.processor = _FakeProcessor(padding_side)
    collator.formatting_func = None
    collator.max_seq_length = None
    collator.truncation = False
    collator.ignore_index = -100
    collator.completion_only_loss = False
    collator.pad_to_multiple_of = None
    collator.image_size = None
    collator.patch_size = 14
    collator.padding_token_ids = torch.tensor([PAD_ID])
    return collator


# Unequal lengths, so the batch pads the second row.
EXAMPLES = [
    {"prompt": "<bos> a b", "completion": "x y z w"},
    {"prompt": "<bos> a", "completion": "x"},
]


def _shifted_targets(input_ids, labels):
    """(previous input, target) for every position the causal LM loss trains on."""
    return [
        (int(input_ids[t - 1]), int(labels[t]))
        for t in range(1, labels.shape[0])
        if int(labels[t]) != -100
    ]


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_batched_rows_train_on_the_same_targets_as_single_rows(padding_side):
    collator = make_collator(padding_side)
    batched = collator(EXAMPLES)
    for i, example in enumerate(EXAMPLES):
        single = collator([example])
        assert _shifted_targets(batched["input_ids"][i], batched["labels"][i]) == \
            _shifted_targets(single["input_ids"][0], single["labels"][0]), (padding_side, i)


def test_no_target_is_predicted_from_padding():
    out = make_collator("left")(EXAMPLES)
    previous_is_pad = out["attention_mask"][:, :-1] == 0
    assert not (out["labels"][:, 1:] != -100)[previous_is_pad].any()


def test_total_target_count_matches_gradient_accumulation():
    """num_items_in_batch for batch size 2 equals the sum over 2 batch-size-1 micro-batches."""
    collator = make_collator("left")
    batched = collator(EXAMPLES)
    singles = [collator([e]) for e in EXAMPLES]
    n_batched = int((batched["labels"][:, 1:] != -100).sum())
    n_singles = sum(int((s["labels"][:, 1:] != -100).sum()) for s in singles)
    assert n_batched == n_singles


def test_mask_helper_ignores_missing_or_mismatched_attention_mask():
    collator = make_collator("left")
    labels = torch.tensor([[5, 6, 7]])
    collator._mask_targets_after_padding(labels, None)
    collator._mask_targets_after_padding(labels, torch.ones(1, 2, dtype = torch.long))
    assert labels.tolist() == [[5, 6, 7]]
