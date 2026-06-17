# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""The resized-shard rewrite has two branches (streaming temp file vs low-disk
in-place save_file). They must produce identical tensors, and both must preserve
unchanged tensors byte-for-byte while swapping in the resized ones. CPU-only.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from unsloth_zoo.saving_utils import (
    _estimate_resized_shard_bytes,
    _inplace_rewrite_resized_shard,
    _stream_rewrite_resized_shard_and_replace,
)


def _read_header(path):
    with open(path, "rb") as f:
        length_of_header = int.from_bytes(f.read(8), "little")
        header_metadata = json.loads(f.read(length_of_header))
    return length_of_header, header_metadata


def _make_shard(path, tensors):
    save_file(tensors, str(path), metadata = {"format": "pt"})


@pytest.fixture
def base_shard(tmp_path):
    torch.manual_seed(0)
    tensors = {
        "model.embed_tokens.weight": torch.randn(10, 8),   # grows below
        "model.norm.weight":         torch.randn(8),       # pass-through
        "lm_head.weight":            torch.randn(10, 8),    # pass-through
    }
    path = tmp_path / "model.safetensors"
    _make_shard(path, tensors)
    return path, tensors


def test_streaming_and_inplace_resized_rewrites_match(base_shard, tmp_path):
    src, original = base_shard
    length_of_header, header_metadata = _read_header(src)

    # Vocab grew 10 -> 13: the merged embedding no longer fits its byte slot.
    grown = torch.randn(13, 8)
    resized = {"model.embed_tokens.weight": grown}

    stream_path = tmp_path / "stream.safetensors"
    inplace_path = tmp_path / "inplace.safetensors"
    shutil.copy2(src, stream_path)
    shutil.copy2(src, inplace_path)

    _stream_rewrite_resized_shard_and_replace(
        str(stream_path), str(tmp_path), header_metadata, length_of_header, dict(resized)
    )
    _inplace_rewrite_resized_shard(str(inplace_path), header_metadata, dict(resized))

    with safe_open(str(stream_path), framework = "pt") as a, \
         safe_open(str(inplace_path), framework = "pt") as b:
        assert sorted(a.keys()) == sorted(b.keys()) == sorted(original.keys())
        for key in original:
            ta, tb = a.get_tensor(key), b.get_tensor(key)
            # Both branches agree with each other...
            assert torch.equal(ta, tb), f"branch mismatch for {key}"
            # ...the resized tensor is the grown one, the rest are untouched.
            expected = grown if key == "model.embed_tokens.weight" else original[key]
            assert torch.equal(ta, expected), f"wrong content for {key}"


def test_estimate_tracks_resized_growth(base_shard):
    src, _ = base_shard
    length_of_header, header_metadata = _read_header(src)
    base_size = src.stat().st_size

    grown = torch.randn(13, 8)  # +3 rows * 8 cols * 4 bytes = +96 bytes of data
    est = _estimate_resized_shard_bytes(
        header_metadata, {"model.embed_tokens.weight": grown}, length_of_header
    )
    assert est > base_size
    assert est - base_size >= (13 - 10) * 8 * 4
