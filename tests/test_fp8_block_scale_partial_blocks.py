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
"""Rebuilding a block FP8 projection from vLLM must accept dims smaller than or not multiples of the block; the scale
grid has ceil(dim / block) entries, not dim // block."""

import pytest

from unsloth_zoo.vllm_utils import _block_scale_grid_matches, _block_scale_rows


@pytest.mark.parametrize("weight, scale", [
    ((6144, 4096), (48, 32)),   # aligned (qwen3 qkv)
    ((64, 128), (1, 1)),        # out_features below the block
    ((128, 64), (1, 1)),        # in_features below the block
    ((200, 300), (2, 3)),       # partial last block on both sides
])
def test_valid_grids_match(weight, scale):
    assert _block_scale_grid_matches(weight, scale, (128, 128))


@pytest.mark.parametrize("weight, scale", [
    ((0, 4096), (48, 32)),      # wrongly sliced [0, x] weight
    ((6144, 4096), (47, 32)),   # one block row short
    ((64, 128), (1, 2)),        # grid wider than the weight
])
def test_wrong_grids_are_refused(weight, scale):
    assert not _block_scale_grid_matches(weight, scale, (128, 128))


def _slices(widths, block):
    offsets = [0]
    for w in widths:
        offsets.append(offsets[-1] + w)
    return [_block_scale_rows(offsets[k], offsets[k + 1], block) for k in range(len(widths))]


def test_aligned_fused_slices_are_disjoint():
    assert _slices([4096, 1024, 1024], 128) == [(0, 32), (32, 40), (40, 48)]


def test_small_projections_inside_one_block_share_it():
    assert _slices([16, 16, 16], 128) == [(0, 1), (0, 1), (0, 1)]
    assert _slices([200], 128) == [(0, 2)]


def test_a_slice_starting_mid_block_and_crossing_a_boundary_is_refused():
    with pytest.raises(ValueError, match = "crosses into the next"):
        _slices([200, 100], 128)


def test_per_row_scales_slice_exactly():
    assert _slices([3, 5], 1) == [(0, 3), (3, 8)]
