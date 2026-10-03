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

"""Mllama collation: batches mixing image and text-only rows, and prompt-completion cross_attention_mask.

Hermetic CPU tests with a stub processor that follows MllamaProcessor: it rejects a batch where only
some rows have images, and emits a per-token cross_attention_mask that is 1 from the image token on.
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from unsloth_zoo.vision_utils import UnslothVisionDataCollator

PAD_ID = 0
IMG_ID = 7
VOCAB = {"<img>": IMG_ID, "a": 1, "b": 2, "x": 3, "y": 4, "z": 5, "w": 6}
TILES = 2
MIXED_ERROR = "If a batch of text is provided, there should be either no images or at least one image per sample"


class _FakeTokenizer:
    pad_token_id = PAD_ID
    padding_side = "right"

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return VOCAB.get(tokens, -1)
        return [VOCAB.get(t, -1) for t in tokens]


class _FakeMllamaProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.calls = 0

    def __call__(self, text, images=None, padding=True, padding_side=None, return_tensors="pt",
                 add_special_tokens=False, **kwargs):
        self.calls += 1
        padding_side = padding_side or self.tokenizer.padding_side
        if images is not None and any(len(row) == 0 for row in images):
            raise ValueError(MIXED_ERROR)
        rows = [[VOCAB[t] for t in s.split()] for s in text]
        width = max(len(r) for r in rows)
        ids, mask, cross = [], [], []
        n_img = max(len(row) for row in images) if images is not None else 0
        for b, r in enumerate(rows):
            pad = [PAD_ID] * (width - len(r))
            row_ids = pad + r if padding_side == "left" else r + pad
            ids.append(row_ids)
            mask.append([0] * len(pad) + [1] * len(r) if padding_side == "left" else [1] * len(r) + [0] * len(pad))
            seen = torch.zeros(width, max(n_img, 1), TILES, dtype = torch.long)
            k = 0
            for t, tok in enumerate(row_ids):
                if tok == IMG_ID:
                    k += 1
                seen[t, :k] = 1
            cross.append(seen)
        out = {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}
        if images is not None:
            out["cross_attention_mask"] = torch.stack(cross)
            B = len(rows)
            out["pixel_values"] = torch.full((B, n_img, TILES, 3, 2, 2), 5.0)
            out["aspect_ratio_ids"] = torch.full((B, n_img), 3)
            out["aspect_ratio_mask"] = torch.ones((B, n_img, TILES), dtype = torch.long)
            for b, row in enumerate(images):
                out["pixel_values"][b, len(row):] = 0
                out["aspect_ratio_ids"][b, len(row):] = 0
                out["aspect_ratio_mask"][b, len(row):] = 0
                out["aspect_ratio_mask"][b, len(row):, 0] = 1
        return out


def make_collator():
    collator = UnslothVisionDataCollator.__new__(UnslothVisionDataCollator)
    collator.processor = _FakeMllamaProcessor()
    collator.formatting_func = None
    collator.max_seq_length = None
    collator.truncation = False
    collator.ignore_index = -100
    collator.completion_only_loss = True
    collator.pad_to_multiple_of = None
    collator.image_size = None
    collator.patch_size = 14
    collator.resize_dimension = 0
    collator.dtype = torch.float32
    collator.padding_token_ids = torch.tensor([PAD_ID, IMG_ID])
    return collator


IMG = Image.new("RGB", (4, 4))


@pytest.mark.parametrize("padding_side", ["right", "left"])
@pytest.mark.parametrize("layout", [(1, 0), (0, 1), (0, 1, 0), (2, 0, 1)])
def test_mixed_image_and_text_rows(layout, padding_side):
    collator = make_collator()
    texts = ["<img> " * n + ("a b" if n else "x y z w a b") for n in layout]
    images = [[IMG] * n for n in layout]
    kwargs = dict(text = texts, images = images, padding = True, return_tensors = "pt", padding_side = padding_side)
    out = collator._call_processor(kwargs, True)

    for j, n in enumerate(layout):
        alone = collator.processor(text = [texts[j]], images = [images[j]] if n else None, padding_side = padding_side)
        m = out["attention_mask"][j].bool()
        assert torch.equal(out["input_ids"][j][m], alone["input_ids"][0])
        pad_pos = (~m).nonzero().flatten()
        if len(pad_pos):
            assert (pad_pos[0] == 0) == (padding_side == "left")
        assert (out["input_ids"][j][~m] == PAD_ID).all()
        if n:
            assert torch.equal(out["cross_attention_mask"][j][m][:, :n], alone["cross_attention_mask"][0][:, :n])
            assert (out["pixel_values"][j, :n] == 5).all()
        else:
            # Text-only rows never attend to an image and carry Mllama's empty-slot values
            assert (out["cross_attention_mask"][j] == 0).all()
            assert (out["pixel_values"][j] == 0).all()
            assert (out["aspect_ratio_ids"][j] == 0).all()
            assert (out["aspect_ratio_mask"][j][:, 0] == 1).all() and (out["aspect_ratio_mask"][j][:, 1:] == 0).all()
    assert out["cross_attention_mask"].shape[:2] == out["input_ids"].shape


def test_other_value_errors_still_raise():
    collator = make_collator()
    with pytest.raises(ValueError, match = "at least one image"):
        collator._call_processor(dict(text = ["x", "<img> a"], images = [[], [IMG]]), False)


def test_all_image_rows_take_one_processor_call():
    collator = make_collator()
    out = collator._call_processor(dict(text = ["<img> a", "<img> x y"], images = [[IMG], [IMG]], padding_side = "right"), True)
    assert collator.processor.calls == 1
    assert out["input_ids"].shape == (2, 3)


def test_prompt_completion_cross_attention_mask_covers_completion():
    collator = make_collator()
    examples = [
        {"images": [IMG], "prompt": "<img> a b", "completion": "x"},
        {"images": [], "prompt": "a", "completion": "x y z w"},
    ]
    out = collator(examples)
    cross = out["cross_attention_mask"]
    assert cross.shape[:2] == out["input_ids"].shape
    m0 = out["attention_mask"][0].bool()
    ids0 = out["input_ids"][0][m0]
    # Every token from the image token on, completion included, attends to the image
    assert ids0.tolist() == [IMG_ID, 1, 2, 3]
    assert (cross[0][m0][:, 0] == 1).all()
    assert (cross[1] == 0).all()
    assert (cross[0][~m0] == 0).all()


def test_prompt_completion_cross_attention_mask_all_image_rows():
    collator = make_collator()
    examples = [
        {"images": [IMG], "prompt": "<img> a", "completion": "x y z"},
        {"images": [IMG], "prompt": "a <img> b", "completion": "w"},
    ]
    out = collator(examples)
    cross, attn = out["cross_attention_mask"], out["attention_mask"].bool()
    assert cross.shape[:2] == out["input_ids"].shape
    for j in range(2):
        ids = out["input_ids"][j][attn[j]]
        after_image = torch.cumsum(ids == IMG_ID, 0) > 0
        assert torch.equal(cross[j][attn[j]][:, 0, 0].bool(), after_image)


def test_image_axis_equal_to_token_width_is_not_padded():
    # An image-only prompt is one token wide, the same size as its image axis
    collator = make_collator()
    texts = ["<img>", "x y z w a b"]
    out = collator._call_processor(dict(text = texts, images = [[IMG], []], padding_side = "left"), True)
    assert out["input_ids"].shape == (2, 6)
    assert out["cross_attention_mask"].shape == (2, 6, 1, TILES)
    assert out["pixel_values"].shape[1] == 1
    assert out["aspect_ratio_ids"].shape == (2, 1)
    assert out["aspect_ratio_mask"].shape == (2, 1, TILES)
    assert (out["pixel_values"][0] == 5).all() and (out["pixel_values"][1] == 0).all()


@pytest.mark.parametrize("padding_side", ["right", "left"])
@pytest.mark.parametrize("pad_to_multiple_of", [None, 3])
def test_prompt_completion_cross_attention_mask_follows_truncation(padding_side, pad_to_multiple_of):
    collator = make_collator()
    collator.processor.tokenizer.padding_side = padding_side
    collator.max_seq_length = 4
    collator.pad_to_multiple_of = pad_to_multiple_of
    examples = [
        {"images": [IMG], "prompt": "a <img> b", "completion": "x y z w"},
        {"images": [IMG], "prompt": "<img> a", "completion": "x"},
    ]
    out = collator(examples)
    cross, attn = out["cross_attention_mask"], out["attention_mask"].bool()
    assert cross.shape[:2] == out["input_ids"].shape
    assert out["input_ids"][0][attn[0]].tolist() == [1, IMG_ID, 2, 3]
    for j in range(2):
        ids = out["input_ids"][j][attn[j]]
        after_image = torch.cumsum(ids == IMG_ID, 0) > 0
        assert torch.equal(cross[j][attn[j]][:, 0, 0].bool(), after_image)
        assert (cross[j][~attn[j]] == 0).all()
