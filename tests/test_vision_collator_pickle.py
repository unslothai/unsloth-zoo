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

"""UnslothVisionDataCollator pickles for spawn DataLoader workers (Windows, macOS, or an explicit spawn start
method), for every resize_dimension; its image-side function used to be a constructor-local lambda."""

import pickle
from types import SimpleNamespace

import pytest
import torch

from unsloth_zoo.vision_utils import UnslothVisionDataCollator


class _Embedding:
    weight = torch.zeros(1, dtype = torch.float32)


class _Model:
    config = SimpleNamespace(torch_dtype = "float32", model_type = "qwen2_5_vl")

    def get_input_embeddings(self):
        return _Embedding()


class _Processor:
    image_processor = SimpleNamespace()

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
