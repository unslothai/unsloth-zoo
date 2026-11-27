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

"""grpo_vision_chunks over a dedup video bank gives each chunk the flat pixel_values_videos."""

import pytest
import torch

from unsloth_zoo.rl_replacements import (
    GRPO_VISION_INDEX_KEYS,
    GRPO_VISION_KEYS,
    grpo_get_vision_inputs,
    grpo_video_bank_rows,
    grpo_vision_chunks,
)


def _layouts(order, videos_per_prompt):
    """order: prompt id per sample; prompt p has videos_per_prompt[p] videos of 2 * (v + 1) rows."""
    bank, grids, starts = [], {}, {}
    row = 0
    for p, n in enumerate(videos_per_prompt):
        for v in range(n):
            k = 2 * (v + 1)
            bank.append(p * 100 + v * 10 + torch.arange(k * 3, dtype = torch.float32).reshape(k, 3))
            grids[(p, v)] = torch.tensor([1, 1, k])
            starts[(p, v)] = (row, row + k)
            row += k
    bank = torch.cat(bank)
    flat_rows, grid, rows = [], [], []
    for p in order:
        for v in range(videos_per_prompt[p]):
            a, b = starts[(p, v)]
            flat_rows.append(bank[a:b])
            grid.append(grids[(p, v)])
            rows.append([a, b])
    num_videos = [videos_per_prompt[p] for p in order]
    grid = torch.stack(grid)
    flat = {"pixel_values_videos": torch.cat(flat_rows), "video_grid_thw": grid, "num_videos": num_videos}
    banked = {"pixel_values_videos": bank, "video_grid_thw": grid, "num_videos": num_videos,
              "video_bank_rows": torch.tensor(rows, dtype = torch.long)}
    return flat, banked


@pytest.mark.parametrize("order", [[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 0, 1, 2], [1, 1]])
@pytest.mark.parametrize("batch_size", [1, 2, 3, 8])
def test_bank_chunks_match_flat(order, batch_size):
    flat, banked = _layouts(order, [1, 2, 0])
    n = len(order)
    a = grpo_vision_chunks(grpo_get_vision_inputs(flat), n, batch_size)
    b = grpo_vision_chunks(grpo_get_vision_inputs(banked), n, batch_size)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.keys() == y.keys()
        for key in x:
            assert torch.equal(x[key], y[key]) if torch.is_tensor(x[key]) else x[key] == y[key]


def test_contiguous_range_is_a_view():
    bank = torch.arange(30.0).reshape(10, 3)
    rows = torch.tensor([[0, 2], [2, 6], [0, 2]])
    view = grpo_video_bank_rows(bank, rows, 0, 2)
    assert view.untyped_storage().data_ptr() == bank.untyped_storage().data_ptr()
    assert torch.equal(grpo_video_bank_rows(bank, rows, 1, 3), torch.cat([bank[2:6], bank[0:2]]))


def test_index_is_not_a_model_key():
    assert not set(GRPO_VISION_INDEX_KEYS) & set(GRPO_VISION_KEYS)
    assert "video_bank_rows" not in grpo_get_vision_inputs({"pixel_values": None})


def test_mismatched_index_raises():
    _, banked = _layouts([0, 0], [1])
    banked["video_bank_rows"] = torch.tensor([[0, 2], [0, 3]])
    with pytest.raises(ValueError, match = "video_bank_rows"):
        grpo_vision_chunks(banked, 2, 1)


def test_noncontiguous_chunks_are_built_only_when_read():
    # Noncontiguous bank rows are copies: build chunks per read, else all copies live at once.
    import unsloth_zoo.rl_replacements as rr

    flat, banked = _layouts([0, 1, 0, 1, 0, 1, 0, 1], [1, 2])
    chunks = grpo_vision_chunks(grpo_get_vision_inputs(banked), 8, 2)
    stored = [list.__getitem__(chunks, i)["pixel_values_videos"] for i in range(len(chunks))]
    assert all(isinstance(v, rr._GrpoBankRows) for v in stored)
    ref = grpo_vision_chunks(grpo_get_vision_inputs(flat), 8, 2)
    for x, y in zip(ref, chunks):
        assert torch.is_tensor(y["pixel_values_videos"])
        assert torch.equal(x["pixel_values_videos"], y["pixel_values_videos"])
    assert torch.equal(chunks[1]["pixel_values_videos"], ref[1]["pixel_values_videos"])
    assert isinstance(chunks[1:], rr._GrpoVisionChunks)
