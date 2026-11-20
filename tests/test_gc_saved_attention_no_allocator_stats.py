# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""Automatic saved attention with no allocator statistics: a global CUDAPluggableAllocator raises from
getDeviceStats, so the budget's peak / capacity reads raised out of the attention call and aborted the
training step. Without readings the call must recompute as with saved attention off. CPU, with the
CUDA allocator queries replaced by ones that raise."""
import pathlib

import pytest
import torch

SA = pytest.importorskip("unsloth_zoo.saved_attention")

CUDA = torch.device("cuda", 0)


def _raise(*args, **kwargs):
    raise RuntimeError("CUDAPluggableAllocator does not yet support getDeviceStats")


def test_saved_attention_module_is_this_tree():
    root = pathlib.Path(__file__).resolve().parents[1]
    assert pathlib.Path(SA.__file__).resolve() == root / "unsloth_zoo" / "saved_attention.py"


def test_device_readings_without_allocator_stats(monkeypatch):
    for name in ("memory_stats_as_nested_dict", "mem_get_info", "memory_reserved", "memory_allocated"):
        monkeypatch.setattr(torch.cuda, name, _raise)
    assert SA._device_peak(CUDA) is None
    assert SA._device_capacity(CUDA) is None
    if hasattr(SA, "_device_allocated"):
        assert SA._device_allocated(CUDA) is None


def test_admission_declines_without_a_capacity(monkeypatch):
    monkeypatch.setattr(SA, "_device_peak", lambda device: 1 << 30)
    monkeypatch.setattr(SA, "_device_capacity", lambda device: None)
    monkeypatch.setattr(SA, "_AUTO_ACTIVE", [True])
    monkeypatch.setattr(SA, "_BUDGETS", {})
    query = torch.empty((1, 4, 64, 16), dtype = torch.bfloat16, device = "meta")
    budget = SA._BUDGETS[query.device] = SA._Budget()
    budget.measured = 1 << 40  # sizes already measured: only the capacity can refuse
    frame = SA.SavedAttentionFrame(auto = True)
    assert SA._auto_admit(frame, [query]) is False
