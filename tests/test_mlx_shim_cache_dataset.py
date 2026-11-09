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

"""The torch shim's mlx_lm CacheDataset keeps mlx-lm's contract: an item is ``data.process(data[idx])``, computed once.
Returning the raw row instead made every text plan built through the shim empty."""

from __future__ import annotations

import types

import pytest
import torch  # noqa: F401  (the simulation shim runs MLX ops on torch)


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    """No teardown: later files hold references to the mlx modules."""
    from mlx_simulation import simulate_mlx_on_torch

    simulate_mlx_on_torch()


class _Rows:
    def __init__(self, rows):
        self.rows = rows
        self.processed = []

    def __getitem__(self, idx):
        return self.rows[idx]

    def __len__(self):
        return len(self.rows)

    def process(self, row):
        self.processed.append(row["text"])
        return ([int(part) for part in row["text"].split()], 0)


def test_items_are_processed_once():
    from mlx_simulation.mlx_lm_stub import CacheDataset

    rows = _Rows([{"text": "1 2 3"}, {"text": "4 5"}])
    cache = CacheDataset(rows)
    assert len(cache) == 2
    assert cache[0] == ([1, 2, 3], 0)
    assert cache[0] == ([1, 2, 3], 0)
    assert cache[1] == ([4, 5], 0)
    assert rows.processed == ["1 2 3", "4 5"]
    assert cache.itemlen(0) == len(rows[0])


def test_a_dataset_without_process_passes_through():
    from mlx_simulation.mlx_lm_stub import CacheDataset

    assert CacheDataset([([1, 2], [1, 2])])[0] == ([1, 2], [1, 2])


def test_an_ordered_text_plan_built_through_the_shim_has_rows():
    from unsloth_zoo.mlx.utils import _create_ordered_text_plan

    tokenizer = types.SimpleNamespace(
        chat_template=None, eos_token_id=None, pad_token_id=0,
        encode=lambda text, add_special_tokens=True: [int(part) for part in str(text).split()],
    )
    texts = [{"text": " ".join(str(500 + i + j) for j in range(3 + i))} for i in range(4)]
    plan = _create_ordered_text_plan(texts, tokenizer, 2, 64, seed=3407, dataset_order="sequential")
    assert len(plan) > 0
