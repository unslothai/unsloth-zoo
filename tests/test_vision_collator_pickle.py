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

"""UnslothVisionDataCollator pickles for spawn DataLoader workers (Windows, macOS, or an explicit
spawn start method) with every resize_dimension and train_on_responses_only."""

import pickle
from types import SimpleNamespace

import pytest
import torch

from unsloth_zoo.dataset_utils import train_on_responses_only
from unsloth_zoo.vision_utils import UnslothVisionDataCollator

USER, ASSISTANT = "<|user|>", "<|assistant|>"
MARKERS = {USER: 1, ASSISTANT: 2}


class _Embedding:
    weight = torch.zeros(1, dtype = torch.float32)


class _Model:
    config = SimpleNamespace(torch_dtype = "float32", model_type = "qwen2_5_vl")

    def get_input_embeddings(self):
        return _Embedding()


class _Encoding(dict):
    __getattr__ = dict.__getitem__


class _Tokenizer:
    """Markers are single ids; other text is per-character ordinals."""

    def __call__(self, text, add_special_tokens = False, **kwargs):
        if isinstance(text, (list, tuple)):
            ids = [self(t).input_ids for t in text]
            return _Encoding(input_ids = ids, attention_mask = [[1] * len(r) for r in ids])
        ids, rest = [], text
        while rest:
            for marker, tid in MARKERS.items():
                if rest.startswith(marker):
                    ids.append(tid)
                    rest = rest[len(marker):]
                    break
            else:
                ids.append(ord(rest[0]))
                rest = rest[1:]
        return _Encoding(input_ids = ids, attention_mask = [1] * len(ids))

    def convert_tokens_to_ids(self, tokens):
        return [0] * len(tokens)


class _Processor:
    image_processor = SimpleNamespace()
    tokenizer = _Tokenizer()

    def convert_tokens_to_ids(self, tokens):
        return [0] * len(tokens)

    def apply_chat_template(self, *args, **kwargs):
        return "hello"


@pytest.mark.parametrize("dimension, side", [(0, 3), (1, 7), ("max", 7), ("min", 3)])
def test_collator_pickles_for_spawn_workers(dimension, side):
    collator = UnslothVisionDataCollator(_Model(), _Processor(), resize_dimension = dimension)
    restored = pickle.loads(pickle.dumps(collator))
    image = SimpleNamespace(size = (3, 7))
    assert collator.size_func(image) == side
    assert restored.size_func(image) == side
    assert restored.resize_dimension == dimension


ROW = [1, 10, 11, 2, 20, 21]


def test_response_masker_pickles_and_masks_the_same():
    masker = train_on_responses_only(
        None, USER, ASSISTANT, tokenizer = _Tokenizer(), return_function = True
    )
    restored = pickle.loads(pickle.dumps(masker))
    want = masker({"input_ids": [list(ROW)]})["labels"]
    assert want[0] == [-100, -100, -100, -100, 20, 21]
    assert restored({"input_ids": [list(ROW)]})["labels"] == want


def test_collator_with_response_masking_pickles():
    collator = UnslothVisionDataCollator(
        _Model(), _Processor(), train_on_responses_only = True,
        instruction_part = USER, response_part = ASSISTANT,
    )
    restored = pickle.loads(pickle.dumps(collator))
    batch = {"input_ids": [list(ROW)]}
    assert restored.train_on_responses_only(batch)["labels"] == \
        collator.train_on_responses_only(batch)["labels"]
