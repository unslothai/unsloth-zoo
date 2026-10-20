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

"""The persistent Triton scratch buffer must never be handed to a kernel launched on another card
(device_map with experts on several GPUs)."""
import sys
import types

import pytest
import torch

from unsloth_zoo.temporary_patches import moe_utils


@pytest.fixture
def allocator(monkeypatch):
    fake_triton = types.SimpleNamespace(set_allocator = lambda fn: captured.setdefault("fn", fn))
    captured, current, allocs = {}, {"d": 0}, []
    real_empty = torch.empty

    def empty(size, device = None, dtype = None):
        t = real_empty(size, dtype = dtype)
        t.fake_device = torch.device(device)
        allocs.append((t.fake_device, size))
        return t

    monkeypatch.setitem(sys.modules, "triton", fake_triton)
    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current["d"])
    monkeypatch.setattr(moe_utils, "_TRITON_ALLOCATOR_INITIALIZED", False)
    fresh = {} if isinstance(moe_utils._PERSISTENT_BUFFER, dict) else None   # head: per-device dict
    monkeypatch.setattr(moe_utils, "_PERSISTENT_BUFFER", fresh)
    moe_utils._init_triton_allocator()
    assert moe_utils._TRITON_ALLOCATOR_INITIALIZED
    return captured["fn"], current, allocs


def test_each_card_gets_its_own_scratch(allocator):
    alloc, current, allocs = allocator
    first = alloc(1000, 128, None)
    current["d"] = 1
    other = alloc(100, 128, None)
    assert other is not first
    assert other.fake_device == torch.device("cuda", 1)
    current["d"] = 0
    assert alloc(100, 128, None) is first   # still reused on its own card
    current["d"] = 1
    assert alloc(100, 128, None) is other


def test_single_card_reuse_and_growth_unchanged(allocator):
    alloc, current, allocs = allocator
    a = alloc(1000, 128, None)
    assert alloc(500, 128, None) is a and alloc(1024, 128, None) is a
    b = alloc(5000, 128, None)
    assert b is not a and alloc(10, 128, None) is b
    assert [size for _, size in allocs] == [int(1024 * 1.1), int(5120 * 1.1)]
    assert all(d.type == "cuda" for d, _ in allocs)
