# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""The gradient checkpointing pinned host pool is capped by free host RAM per local rank.

Past the cap new and regrown offload buffers are pageable instead of pinned, and nothing
raises. CPU only: torch.empty is spied so a pinned request is recorded, not performed.
"""
import types

import pytest
import torch


@pytest.fixture
def gc_module(monkeypatch):
    monkeypatch.setenv("UNSLOTH_ALLOW_CPU", "1")
    try:
        from unsloth_zoo import gradient_checkpointing as gc_module
    except ImportError as e:
        pytest.skip(f"unsloth_zoo import unavailable: {e}")
    monkeypatch.setattr(gc_module, "PINNED_MEMORY_AVAILABLE", True, raising = False)
    monkeypatch.setattr(gc_module, "_WARNED_ABOUT_PINNED_MEMORY", False, raising = False)
    monkeypatch.setattr(gc_module, "CPU_BUFFERS", [], raising = False)
    monkeypatch.setattr(gc_module, "GC_PINNED_BYTES", 0, raising = False)
    monkeypatch.setattr(gc_module, "GC_PINNED_CAP", None, raising = False)
    monkeypatch.delenv("UNSLOTH_DISABLE_PINNED_MEMORY", raising = False)
    monkeypatch.delenv("UNSLOTH_GC_PINNED_MAX_GB", raising = False)
    return gc_module


@pytest.fixture
def pin_requests(monkeypatch, gc_module):
    """Pretend pinning works: record the request, hand back a buffer marked pinned."""
    requests = []
    real_empty = torch.empty

    def empty(*args, pin_memory = False, **kwargs):
        requests.append(pin_memory)
        buffer = real_empty(*args, **kwargs)
        buffer._fake_pinned = pin_memory
        return buffer

    def mark(buffer):
        setattr(buffer, gc_module.HOST_PINNED_ATTR, getattr(buffer, "_fake_pinned", False))
        return buffer

    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(gc_module, "_mark_host_buffer", mark)
    return requests


def test_pins_until_the_cap_then_goes_pageable(monkeypatch, gc_module, pin_requests):
    monkeypatch.setenv("UNSLOTH_GC_PINNED_MAX_GB", str(3000 / (1 << 30)))
    a = gc_module._new_host_buffer(1000, torch.float16)   # 2000 bytes, fits
    b = gc_module._new_host_buffer(1000, torch.float16)   # 4000 total, over the cap
    assert pin_requests == [True, False]
    assert getattr(a, gc_module.HOST_PINNED_ATTR) and not getattr(b, gc_module.HOST_PINNED_ATTR)
    assert gc_module.GC_PINNED_BYTES == 2000


def test_growth_past_the_cap_swaps_to_pageable(monkeypatch, gc_module, pin_requests):
    monkeypatch.setenv("UNSLOTH_GC_PINNED_MAX_GB", str(3000 / (1 << 30)))
    a = gc_module._new_host_buffer(1000, torch.float16)
    assert gc_module.GC_PINNED_BYTES == 2000
    grown = gc_module._grow_host_buffer(a, 4000)          # 8000 bytes would exceed 3000
    assert grown.numel() == 4000
    assert not getattr(grown, gc_module.HOST_PINNED_ATTR)
    assert gc_module.GC_PINNED_BYTES == 0                  # the pinned slot was given up


def test_growth_within_the_cap_stays_pinned(monkeypatch, gc_module, pin_requests):
    monkeypatch.setenv("UNSLOTH_GC_PINNED_MAX_GB", "1")
    a = gc_module._new_host_buffer(1000, torch.float16)
    grown = gc_module._grow_host_buffer(a, 4000)
    assert grown is a and getattr(grown, gc_module.HOST_PINNED_ATTR)
    assert gc_module.GC_PINNED_BYTES == 8000


def test_cap_is_free_ram_minus_reserve_split_by_local_ranks(monkeypatch, gc_module):
    gib = 1 << 30
    fake = types.SimpleNamespace(
        virtual_memory = lambda: types.SimpleNamespace(total = 100 * gib, available = 60 * gib),
    )
    monkeypatch.setitem(__import__("sys").modules, "psutil", fake)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    # Reserve is max(4 GiB, 15% of 100 GiB) = 15 GiB, so (60 - 15) / 4 ranks.
    assert gc_module._compute_pinned_cap() == (45 * gib) // 4


def test_unmeasurable_host_ram_keeps_the_pool_uncapped(monkeypatch, gc_module, pin_requests):
    monkeypatch.setitem(__import__("sys").modules, "psutil", None)  # import raises
    assert gc_module._compute_pinned_cap() == -1
    for _ in range(3):
        gc_module._new_host_buffer(1 << 20, torch.float16)
    assert pin_requests == [True, True, True]


def test_failed_pin_releases_its_reservation(monkeypatch, gc_module):
    monkeypatch.setenv("UNSLOTH_GC_PINNED_MAX_GB", "1")
    real_empty = torch.empty

    def refuse_pinned(*args, pin_memory = False, **kwargs):
        if pin_memory: raise RuntimeError("CUDA error: out of memory")
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", refuse_pinned)
    buffer = gc_module._new_host_buffer(1000, torch.float16)
    assert buffer.numel() == 1000
    assert gc_module.GC_PINNED_BYTES == 0
