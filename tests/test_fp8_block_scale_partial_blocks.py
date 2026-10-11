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
"""Rebuilding a block FP8 projection from vLLM must accept a dimension smaller than (or not a
multiple of) its block: the scale grid has ceil(dim / block) entries, not dim // block."""

import pytest

from unsloth_zoo.vllm_utils import _block_scale_grid_matches


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
