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

import torch

from unsloth_zoo.saving_utils import _save_size_in_bytes, get_torch_storage_size_new


def test_a_tied_embedding_counts_once():
    w = torch.zeros(4096, 1024, dtype = torch.bfloat16)
    sd = {"model.embed_tokens.weight": w, "lm_head.weight": w, "norm.weight": torch.zeros(1024, dtype = torch.bfloat16)}
    assert _save_size_in_bytes(sd, 2) == w.nbytes + 1024 * 2


def test_views_of_one_buffer_each_count():
    arena = torch.zeros(3 * 1024, 1024, dtype = torch.bfloat16)
    q, k, v = arena.chunk(3, 0)
    sd = {"q": q, "k": k, "v": v}
    assert _save_size_in_bytes(sd, 2) == sum(get_torch_storage_size_new(x, 2) for x in sd.values())


def test_meta_tensors_are_never_merged():
    a, b = torch.empty(64, 64, device = "meta"), torch.empty(64, 64, device = "meta")
    assert _save_size_in_bytes({"a": a, "b": b}, 2) >= 2 * 64 * 64
