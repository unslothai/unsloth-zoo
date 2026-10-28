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
    monkeypatch.setattr(gc_module, "_host_allocator_free_bytes", lambda: None, raising = False)
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


def test_pageable_slot_is_pinned_again_when_it_grows_after_ram_frees(monkeypatch, gc_module, pin_requests):
    # A low-RAM moment sends one growth pageable; the next growth must not stay pageable for good.
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising = False)
    gib = 1 << 30
    host = _fake_host(monkeypatch, total_gib = 12, available_gib = 8)
    a = gc_module._new_host_buffer(1000, torch.float16)
    host.available = 0
    b = gc_module._grow_host_buffer(a, 4000)
    assert not getattr(b, gc_module.HOST_PINNED_ATTR)
    host.available = 8 * gib
    c = gc_module._grow_host_buffer(b, 8000)
    assert c.numel() == 8000 and getattr(c, gc_module.HOST_PINNED_ATTR)
    assert pin_requests == [True, False, True]


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
    monkeypatch.setattr(gc_module, "_empty_host_cache", lambda: False)  # no flush available
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


def test_cached_block_taken_by_another_user_is_not_reused_past_the_cap(monkeypatch, gc_module):
    monkeypatch.setenv("UNSLOTH_GC_PINNED_MAX_GB", str(6000 / (1 << 30)))
    held = {"bytes": 0}
    free = {"bytes": 0}
    monkeypatch.setattr(gc_module, "_host_allocator_held_bytes", lambda: held["bytes"])
    monkeypatch.setattr(gc_module, "_host_allocator_free_bytes", lambda: free["bytes"])
    # Draining torch's event queue finds no free block of this size: the other user has it.
    monkeypatch.setattr(gc_module, "_host_cache_reuses_block", lambda block: False, raising = False)
    assert gc_module._reserve_pinned(4096)
    held["bytes"] = 4096
    gc_module._release_pinned(4096)                 # cached, but a pinned DataLoader takes it
    free["bytes"] = 0
    # A new 4096 block would make 8192 pinned, past the 6000 cap.
    assert not gc_module._reserve_pinned(4096)
    # Still reusable when the allocator really holds it free.
    gc_module._release_pinned(4096)
    free["bytes"] = 4096
    assert gc_module._reserve_pinned(4096)


def test_regrowth_within_retained_storage_reserves_nothing(monkeypatch, gc_module, pin_requests):
    monkeypatch.setenv("UNSLOTH_GC_PINNED_MAX_GB", "1")
    a = gc_module._new_host_buffer(4000, torch.float16)   # 8000 bytes, 8192 block
    a.resize_(1000)                                         # reset shrinks numel, keeps storage
    grown = gc_module._grow_host_buffer(a, 3000)            # 6000 bytes fit the 8000 storage
    assert grown is a and getattr(grown, gc_module.HOST_PINNED_ATTR)
    assert gc_module.GC_PINNED_BYTES == 8192 and gc_module._PINNED_CACHED == {}


def _fake_host(monkeypatch, total_gib, available_gib):
    gib = 1 << 30
    host = types.SimpleNamespace(total = int(total_gib * gib), available = int(available_gib * gib))
    fake = types.SimpleNamespace(virtual_memory = lambda: host)
    monkeypatch.setitem(__import__("sys").modules, "psutil", fake)
    return host


def test_single_rank_low_ram_host_still_pins_what_fits(monkeypatch, gc_module, pin_requests):
    # 8 GiB host with 3 GiB available: the 4 GiB reserve meant for other ranks must not
    # force a single process to pageable copies that pinned fine before the cap existed.
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising = False)
    _fake_host(monkeypatch, total_gib = 8, available_gib = 3)
    for _ in range(8):
        gc_module._new_host_buffer(16 << 20, torch.float16)   # 32 MiB each
    assert pin_requests == [True] * 8


def test_single_rank_refuses_a_block_that_does_not_fit_now(monkeypatch, gc_module, pin_requests):
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising = False)
    gib = 1 << 30
    host = _fake_host(monkeypatch, total_gib = 12, available_gib = 3)
    assert gc_module._reserve_pinned(1 * gib)
    # Available RAM is re-read per new block: the host filled up since the first one.
    host.available = int(1.5 * gib)
    assert not gc_module._reserve_pinned(1 * gib)
    host.available = 4 * gib
    assert gc_module._reserve_pinned(1 * gib)


def test_several_local_ranks_keep_the_split_cap(monkeypatch, gc_module):
    _fake_host(monkeypatch, total_gib = 8, available_gib = 3)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    assert gc_module._compute_pinned_cap() == 0
    assert not gc_module._reserve_pinned(32 << 20)


def test_single_rank_frees_dead_cached_blocks_before_going_pageable(monkeypatch, gc_module):
    # A grown pinned buffer leaves its old block in torch's host cache. When RAM runs short
    # those blocks are returned to the OS first, so they do not sit beside the pageable copies.
    gib = 1 << 30
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising = False)
    monkeypatch.setattr(gc_module, "DEVICE_TYPE", "cuda", raising = False)
    host = _fake_host(monkeypatch, total_gib = 8, available_gib = 3)
    half = gib // 2
    assert gc_module._reserve_pinned(half)
    gc_module._release_pinned(half)                  # 512 MiB block now dead in the host cache
    emptied = []

    def empty_host_cache():
        emptied.append(True)
        host.available += half

    monkeypatch.setattr(torch._C, "_host_emptyCache", empty_host_cache, raising = False)
    # 1 GiB + 0.8 GiB headroom does not fit in 1.5 GiB, but does once the dead block is freed.
    host.available = int(1.5 * gib)
    assert gc_module._reserve_pinned(1 * gib)
    assert emptied == [True]
    assert gc_module.GC_PINNED_BYTES == 1 * gib and gc_module._PINNED_CACHED == {}
    # Nothing cached any more: a block that still does not fit goes pageable without a flush.
    host.available = int(1.5 * gib)
    assert not gc_module._reserve_pinned(1 * gib)
    assert emptied == [True]


def test_baseline_is_taken_once_host_stats_appear(monkeypatch, gc_module):
    # The cap can be computed before CUDA initialises (host_memory_stats() is empty then): the
    # foreign-bytes baseline must be latched at the first reading, not stay unset for good.
    monkeypatch.setenv("UNSLOTH_GC_PINNED_MAX_GB", str(10000 / (1 << 30)))
    held = {"bytes": None}
    monkeypatch.setattr(gc_module, "_host_allocator_held_bytes", lambda: held["bytes"])
    assert gc_module._reserve_pinned(4096)
    held["bytes"] = 1000 + 4096                     # 1000 bytes pinned by someone else
    assert gc_module._reserve_pinned(1024)
    held["bytes"] = 1000 + 5120 + 4096              # a cached block of ours was taken by another user
    assert not gc_module._reserve_pinned(2048)      # 9216 past the baseline + 2048 > 10000


def test_cap_flushes_our_released_blocks_before_going_pageable(monkeypatch, gc_module):
    # A grown slot's old block counts against the cap until torch frees it. At the cap those dead
    # blocks are returned to the OS once, so a bigger block can still be pinned.
    monkeypatch.setenv("UNSLOTH_GC_PINNED_MAX_GB", str(10000 / (1 << 30)))
    monkeypatch.setattr(gc_module, "DEVICE_TYPE", "cuda", raising = False)
    monkeypatch.setattr(gc_module, "_synchronize_used_devices", lambda: None)
    emptied = []
    monkeypatch.setattr(torch._C, "_host_emptyCache", lambda: emptied.append(True), raising = False)
    assert gc_module._reserve_pinned(3000)          # 4096 block
    gc_module._release_pinned(3000)                 # the slot grew: 4096 is dead in the host cache
    assert gc_module._reserve_pinned(7000)          # 8192 fits once the 4096 is gone
    assert emptied == [True]
    assert gc_module.GC_PINNED_BYTES == 8192 and gc_module._PINNED_CACHED == {}
    # Nothing of ours cached: at the cap it goes pageable without a flush.
    assert not gc_module._reserve_pinned(7000)
    assert emptied == [True]


def test_cap_flush_counts_only_blocks_the_allocator_released(monkeypatch, gc_module):
    # Our released 4096 block was taken by another pinned user (a DataLoader): the flush cannot
    # free it, so an 8192 request still does not fit under a 10000 cap.
    monkeypatch.setenv("UNSLOTH_GC_PINNED_MAX_GB", str(10000 / (1 << 30)))
    monkeypatch.setattr(gc_module, "DEVICE_TYPE", "cuda", raising = False)
    monkeypatch.setattr(gc_module, "_synchronize_used_devices", lambda: None)
    monkeypatch.setattr(gc_module, "_host_allocator_free_bytes", lambda: 0)
    held = {"bytes": 0}
    monkeypatch.setattr(gc_module, "_host_allocator_held_bytes", lambda: held["bytes"])
    emptied = []
    monkeypatch.setattr(torch._C, "_host_emptyCache", lambda: emptied.append(True), raising = False)
    assert gc_module._reserve_pinned(3000)          # 4096 block
    held["bytes"] = 4096
    gc_module._release_pinned(3000)                 # cached, then reused by another pinned user
    assert not gc_module._reserve_pinned(7000)
    assert emptied == [True]
    # Once the allocator really frees it, the same request fits.
    held["bytes"] = 0
    assert gc_module._reserve_pinned(7000)


# torch 2.13+ host allocator settings: above pinned_max_round_threshold_mb (or
# pinned_max_cached_size_mb) a pinned block is the exact request, and above
# pinned_max_cached_size_mb a freed block goes back to the OS instead of the cache.
MIB = 1 << 20


@pytest.fixture
def host_alloc_conf(monkeypatch, gc_module):
    def set_conf(settings, knobs = True):
        monkeypatch.setattr(torch._C, "_accelerator_getAllocatorSettings", lambda: settings, raising = False)
        monkeypatch.setattr(gc_module, "_HOST_ALLOC_KNOBS", knobs, raising = False)
        monkeypatch.setattr(gc_module, "_HOST_ALLOC_POLICY", (None, None, None), raising = False)
        monkeypatch.setattr(gc_module, "_WARNED_SHORT_BLOCK", True, raising = False)
    return set_conf


def test_round_threshold_charges_the_exact_request(host_alloc_conf, gc_module):
    host_alloc_conf("expandable_segments:True,pinned_max_round_threshold_mb:1")
    assert gc_module._pinned_block_bytes(3 * MIB + 5) == 3 * MIB + 5
    assert gc_module._pinned_block_bytes(MIB - 7) == MIB      # at or below: still a power of two
    assert gc_module._pinned_block_bytes(3000) == 4096


def test_round_threshold_lets_more_buffers_pin_under_the_cap(monkeypatch, host_alloc_conf, gc_module, pin_requests):
    host_alloc_conf("pinned_max_round_threshold_mb:1")
    monkeypatch.setenv("UNSLOTH_GC_PINNED_MAX_GB", str(7 * MIB / (1 << 30)))
    # Two 3 MiB slots are 6 MiB really pinned; charged as 4 + 4 MiB the second went pageable.
    gc_module._new_host_buffer(3 * MIB // 2, torch.float16)
    gc_module._new_host_buffer(3 * MIB // 2, torch.float16)
    assert pin_requests == [True, True]
    assert gc_module.GC_PINNED_BYTES == 6 * MIB


def test_block_above_max_cached_size_is_not_kept_charged(host_alloc_conf, gc_module):
    host_alloc_conf("pinned_max_cached_size_mb:2")
    assert gc_module._reserve_pinned(3 * MIB)
    assert gc_module.GC_PINNED_BYTES == 3 * MIB
    gc_module._release_pinned(3 * MIB)
    # torch freed it, so nothing of ours is cached and the bytes are no longer held.
    assert gc_module.GC_PINNED_BYTES == 0 and not any(gc_module._PINNED_CACHED.values())
    # Small blocks are still rounded and cached.
    assert gc_module._reserve_pinned(3000)
    gc_module._release_pinned(3000)
    assert gc_module.GC_PINNED_BYTES == 4096 and gc_module._PINNED_CACHED == {4096: 1}


def test_unset_or_unsupported_settings_keep_power_of_two(host_alloc_conf, gc_module):
    host_alloc_conf("")
    assert gc_module._pinned_block_bytes(3 * MIB + 5) == 4 * MIB
    # torch 2.12 accepts the key but ignores it.
    host_alloc_conf("pinned_max_round_threshold_mb:1", knobs = False)
    assert gc_module._pinned_block_bytes(3 * MIB + 5) == 4 * MIB


def test_settings_change_is_picked_up(host_alloc_conf, gc_module):
    host_alloc_conf("pinned_max_round_threshold_mb:1")
    assert gc_module._pinned_block_bytes(3 * MIB) == 3 * MIB
    host_alloc_conf("pinned_max_round_threshold_mb:8")
    assert gc_module._pinned_block_bytes(3 * MIB) == 4 * MIB
