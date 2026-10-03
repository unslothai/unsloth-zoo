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
"""TiledMLP must hold its tile size at every batch size.

`TiledMLP.forward` flattens to (bsz * qlen, hd) and splits that into `n_shards` tiles of
`chunk_size`, appending whatever is left over as one final tile. The arctic strategy counted the
shards over `qlen` alone, so with bsz > 1 the final tile absorbed the rest of the batch: the
tiling still produced the right numbers, but the peak activation it exists to bound grew with
the batch instead of staying at one tile.

CPU only: a tiny Linear stands in for the MLP and records the rows it is handed.
"""

import pytest
import torch

from unsloth_zoo.tiled_mlp import patch_mlp


class _RecordingMLP(torch.nn.Module):

    def __init__(self, hidden):
        super().__init__()
        self.up = torch.nn.Linear(hidden, hidden, bias = False)
        self.rows_seen = []

    def forward(self, x):
        self.rows_seen.append(x.shape[-2])
        return self.up(x)
pass


@pytest.mark.parametrize(("bsz", "qlen"), [(1, 32), (2, 32), (4, 32), (8, 64)])
def test_arctic_tiles_stay_within_one_chunk(bsz, qlen):
    hidden = 8  # the arctic strategy uses hd as the chunk size
    mlp = _RecordingMLP(hidden)
    reference = _RecordingMLP.forward
    patch_mlp(mlp, target_arctic = True)

    x = torch.randn(bsz, qlen, hidden)
    out = mlp(x)

    # No tile may exceed the chunk size, whatever the batch size. Before, the trailing tile held
    # (bsz - 1) * qlen extra rows: 448 of them for a 8 x 64 batch against a chunk size of 8.
    assert mlp.rows_seen, "the tiled forward never called the wrapped MLP"
    assert max(mlp.rows_seen) <= hidden, f"tile of {max(mlp.rows_seen)} rows exceeds chunk size {hidden}"
    assert sum(mlp.rows_seen) == bsz * qlen, "the tiles must cover the flattened batch exactly once"

    # The tiling is a memory optimisation, so the output has to be unchanged.
    expected = reference(mlp, x.reshape(1, -1, hidden)).reshape(bsz, qlen, hidden)
    torch.testing.assert_close(out, expected)
pass
