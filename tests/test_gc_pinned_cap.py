# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""Past the pinned cap (free host RAM per local rank) offload buffers go pageable, never raise.
CPU only: torch.empty is spied so a pinned request is recorded, not performed."""
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
    monkeypatch.setattr(gc_module, "_PINNED_CACHED", {}, raising = False)
    monkeypatch.setattr(gc_module, "_HOST_HELD_BASELINE", None, raising = False)
    monkeypatch.setattr(gc_module, "_host_allocator_held_bytes", lambda: None, raising = False)
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
    a = gc_module._new_host_buffer(1000, torch.float16)   # 2000 bytes, a 2048 block, fits
    b = gc_module._new_host_buffer(1000, torch.float16)   # 4096 total, over the cap
    assert pin_requests == [True, False]
    assert getattr(a, gc_module.HOST_PINNED_ATTR) and not getattr(b, gc_module.HOST_PINNED_ATTR)
    assert gc_module.GC_PINNED_BYTES == 2048


def test_growth_past_the_cap_swaps_to_pageable(monkeypatch, gc_module, pin_requests):
    monkeypatch.setenv("UNSLOTH_GC_PINNED_MAX_GB", str(3000 / (1 << 30)))
    a = gc_module._new_host_buffer(1000, torch.float16)
    assert gc_module.GC_PINNED_BYTES == 2048
    grown = gc_module._grow_host_buffer(a, 4000)          # 8000 bytes would exceed 3000
    assert grown.numel() == 4000
    assert not getattr(grown, gc_module.HOST_PINNED_ATTR)
    # The given up block stays in torch's host cache, so it stays charged.
    assert gc_module.GC_PINNED_BYTES == 2048 and gc_module._PINNED_CACHED == {2048: 1}


def test_growth_within_the_cap_stays_pinned(monkeypatch, gc_module, pin_requests):
    monkeypatch.setenv("UNSLOTH_GC_PINNED_MAX_GB", "1")
    a = gc_module._new_host_buffer(1000, torch.float16)
    grown = gc_module._grow_host_buffer(a, 4000)
    assert grown is a and getattr(grown, gc_module.HOST_PINNED_ATTR)
    # New 8192 block plus the old 2048 block resize_ returned to the host cache.
    assert gc_module.GC_PINNED_BYTES == 8192 + 2048


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
    assert not gc_module.PINNED_MEMORY_AVAILABLE


def test_rounded_block_counts_against_the_cap(monkeypatch, gc_module):
    monkeypatch.setenv("UNSLOTH_GC_PINNED_MAX_GB", str(6000 / (1 << 30)))
    assert gc_module._reserve_pinned(2049)          # 4096 block
    assert not gc_module._reserve_pinned(2049)      # 8192 > 6000, though 4098 requested bytes fit
    assert gc_module.GC_PINNED_BYTES == 4096


def test_released_blocks_stay_charged_and_are_reused(monkeypatch, gc_module):
    monkeypatch.setenv("UNSLOTH_GC_PINNED_MAX_GB", str(10000 / (1 << 30)))
    assert gc_module._reserve_pinned(3000)          # 4096 block
    gc_module._release_pinned(3000)
    assert gc_module.GC_PINNED_BYTES == 4096
    # Another size class cannot pile on top of the cached block past the cap.
    assert not gc_module._reserve_pinned(7000)      # 4096 cached + 8192 > 10000
    # The same size class takes the cached block back without a new charge.
    assert gc_module._reserve_pinned(4000)
    assert gc_module.GC_PINNED_BYTES == 4096 and gc_module._PINNED_CACHED == {4096: 0}


def test_emptied_host_cache_drops_the_cached_charge(monkeypatch, gc_module):
    monkeypatch.setenv("UNSLOTH_GC_PINNED_MAX_GB", str(10000 / (1 << 30)))
    held = {"bytes": 0}
    monkeypatch.setattr(gc_module, "_host_allocator_held_bytes", lambda: held["bytes"])
    assert gc_module._reserve_pinned(3000)
    held["bytes"] = 4096
    gc_module._release_pinned(3000)
    held["bytes"] = 0                               # torch.cuda.empty_host_cache()
    assert gc_module._reserve_pinned(7000)          # 8192 fits once the 4096 is gone
    assert gc_module.GC_PINNED_BYTES == 8192 and gc_module._PINNED_CACHED == {}


def test_allocator_growth_beyond_our_charge_counts(monkeypatch, gc_module):
    monkeypatch.setenv("UNSLOTH_GC_PINNED_MAX_GB", str(10000 / (1 << 30)))
    held = {"bytes": 1000}                          # someone else's, before the cap was computed
    monkeypatch.setattr(gc_module, "_host_allocator_held_bytes", lambda: held["bytes"])
    assert gc_module._reserve_pinned(4096)
    held["bytes"] = 1000 + 4096 + 4096              # a cached block of ours was taken by another user
    assert not gc_module._reserve_pinned(4096)      # 8192 already held past the baseline
