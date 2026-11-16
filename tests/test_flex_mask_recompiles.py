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

"""Training flex masks keep one mask_mod structure for padded and unpadded rows of any length:
tiny Gemma-3 vision recompiled every block at the second length and at the first unpadded batch."""

import pytest
import torch
from torch._dynamo.testing import CompileCounter

masking_utils = pytest.importorskip("transformers.masking_utils")


@pytest.fixture
def built(monkeypatch):
    from unsloth_zoo.temporary_patches import misc

    getattr(misc, "patch_flex_mask_keeps_padding_mask_mod", lambda: None)()
    mods = []
    monkeypatch.setattr(masking_utils, "create_block_mask", lambda mask_mod, **kw: mods.append(mask_mod))
    return mods


def _mask_mod(built, attention_mask, fn = None):
    L = attention_mask.shape[1]
    (fn or masking_utils.flex_attention_mask)(batch_size = 2, q_length = L, kv_length = L, attention_mask = attention_mask)
    return built.pop()


def _rows(lengths, L):
    return torch.stack([torch.arange(L) < n for n in lengths])


def _dense(mask_mod, L):
    q, kv = torch.arange(L)[:, None], torch.arange(L)[None, :]
    return torch.stack([mask_mod(torch.tensor(b), torch.tensor(0), q, kv) for b in range(2)])


def test_padded_and_unpadded_batches_share_one_graph(built):
    cc = CompileCounter()
    compiled = torch.compile(_dense, backend = cc, dynamic = None)
    torch._dynamo.reset()
    for lengths, L in (((6, 4), 6), ((9, 9), 9), ((7, 5), 7), ((11, 11), 11), ((8, 3), 8)):
        compiled(_mask_mod(built, _rows(lengths, L)), L)
    assert cc.frame_count <= 2, f"{cc.frame_count} graphs for 5 micro batches"


@pytest.mark.parametrize("lengths", [(6, 6), (6, 3)])
def test_mask_values_unchanged(built, lengths):
    rows = _rows(lengths, 6)
    ours = _dense(_mask_mod(built, rows), 6)
    stock = _dense(_mask_mod(built, rows, getattr(masking_utils.flex_attention_mask, "__wrapped__", None)), 6)
    assert torch.equal(ours, stock)
    with torch.no_grad():  # inference keeps transformers' own mask_mod
        _mask_mod(built, rows)
