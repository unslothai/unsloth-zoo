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
    monkeypatch.setattr(gc_module, "_PINNED_LIVE", {}, raising = False)
    monkeypatch.setattr(gc_module, "_HOST_HELD_BASELINE", None, raising = False)
    monkeypatch.setattr(gc_module, "_PINNED_LIVE_PEAK", 0, raising = False)
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
    gc_module._new_host_buffer(3 * MIB // 2, torch.float16)
    gc_module._new_host_buffer(3 * MIB // 2, torch.float16)
    assert pin_requests == [True, True]
    assert gc_module.GC_PINNED_BYTES == 6 * MIB


def test_block_above_max_cached_size_is_not_kept_charged(host_alloc_conf, gc_module):
    host_alloc_conf("pinned_max_cached_size_mb:2")
    assert gc_module._reserve_pinned(3 * MIB)
    assert gc_module.GC_PINNED_BYTES == 3 * MIB
    gc_module._release_pinned(3 * MIB)
    assert gc_module.GC_PINNED_BYTES == 0 and not any(gc_module._PINNED_CACHED.values())
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


def test_release_uses_the_block_charged_at_allocation(monkeypatch, host_alloc_conf, gc_module):
    host_alloc_conf("pinned_max_round_threshold_mb:1")
    monkeypatch.setenv("UNSLOTH_GC_PINNED_MAX_GB", str(4 * MIB / (1 << 30)))
    flushes = []
    monkeypatch.setattr(gc_module, "_empty_host_cache", lambda: flushes.append(1) or False)
    assert gc_module._reserve_pinned(3 * MIB)       # exact 3 MiB block above the threshold
    assert gc_module._reserve_pinned(MIB)
    assert gc_module.GC_PINNED_BYTES == 4 * MIB
    host_alloc_conf("pinned_max_round_threshold_mb:8")
    gc_module._release_pinned(3 * MIB)
    gc_module._release_pinned(MIB)
    assert gc_module._PINNED_CACHED == {3 * MIB: 1, MIB: 1}
    assert gc_module.GC_PINNED_BYTES == 4 * MIB and not gc_module._PINNED_LIVE
    # No cached 4 MiB block: the request must not take the cached-block shortcut past the cap.
    assert not gc_module._reserve_pinned(4 * MIB)
    assert flushes and gc_module.GC_PINNED_BYTES == 4 * MIB


def test_concurrent_releases_keep_the_cap_bookkeeping_consistent(gc_module, monkeypatch):
    # Split-model backward releases on autograd worker threads; the read-modify-write over the cap
    # globals must not lose updates (a lost update leaks charged bytes and offload goes pageable early).
    import sys, threading
    MIB = 1 << 20
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        leaked = errors = 0
        for _ in range(20):
            gc_module.GC_PINNED_BYTES, gc_module._PINNED_CACHED, gc_module._PINNED_LIVE = 0, {}, {}
            for _ in range(400): assert gc_module._reserve_pinned(MIB)
            bad = []
            def release():
                for _ in range(200):
                    try: gc_module._release_pinned(MIB)
                    except Exception as e: bad.append(e)
            threads = [threading.Thread(target = release) for _ in range(2)]
            for t in threads: t.start()
            for t in threads: t.join()
            cached = sum(k * v for k, v in gc_module._PINNED_CACHED.items())
            leaked += (gc_module.GC_PINNED_BYTES - cached) // MIB
            errors += len(bad)
    finally:
        sys.setswitchinterval(old)
    assert errors == 0
    assert leaked == 0


# torch 2.13+ counted path, against test_gc_pinned_counted's model of torch's CachingHostAllocator.
from test_gc_pinned_counted import allocator, pin, unpin  # noqa: E402,F401


@pytest.fixture
def counted(monkeypatch, gc_module, allocator):
    monkeypatch.setattr(gc_module, "DEVICE_TYPE", "cuda", raising = False)
    monkeypatch.setattr(gc_module, "_PINNED_RELEASED", 0, raising = False)
    monkeypatch.setattr(gc_module, "_WARNED_ABOUT_PINNED_CAP", True, raising = False)
    return allocator


def _auto_cap(monkeypatch, cap_bytes):
    """Free host RAM worth `cap_bytes` per rank over two local ranks (reserve: 15% of 100 GiB)."""
    gib = 1 << 30
    _fake_host(monkeypatch, total_gib = 100, available_gib = 15 + 2 * cap_bytes / gib)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")


def test_freed_baseline_pin_cannot_hide_our_cached_blocks(gc_module, counted):
    # A user-set cap bounds every pinned byte the allocator holds: with a 6 MiB foreign pin, one of our
    # 4 MiB blocks released (cached) and the foreign block back to the OS, nothing pins past 10 MiB.
    fake = counted("pinned_max_cached_size_mb:4", cap_bytes = 10 * MIB)
    foreign = fake.allocate(6 * MIB)
    live = [(n, pin(gc_module, fake, n)) for n in (3 * MIB, 3 * MIB)]
    live = [(n, block) for n, block in live if block]
    assert live
    unpin(gc_module, fake, *live.pop())
    fake.release(foreign)                        # above the cached max: returned to the OS
    pin(gc_module, fake, 5 * MIB)
    assert fake.peak <= 10 * MIB


@pytest.mark.parametrize("user_cap", [True, False])
def test_reused_larger_cached_block_cannot_lift_the_cap(monkeypatch, gc_module, counted, user_cap):
    # Codex 4406863867, torch 2.14: a 3 MiB request reuses a cached 3.5 MiB block, so our own count runs
    # 0.5 MiB short. Once the 6 MiB foreign pin is gone, 3 + 0.5 MiB more must still respect the cap.
    if not user_cap: _auto_cap(monkeypatch, 10 * MIB)
    fake = counted("pinned_max_round_threshold_mb:1", cap_bytes = 10 * MIB if user_cap else None)
    foreign = fake.allocate(6 * MIB)
    live = [(n, pin(gc_module, fake, n)) for n in (int(3.5 * MIB), int(3.5 * MIB))]
    live = [(n, block) for n, block in live if block]
    unpin(gc_module, fake, *live.pop())
    live.append((3 * MIB, pin(gc_module, fake, 3 * MIB)))
    fake.release(foreign)
    fake.empty_cache()
    live += [(n, pin(gc_module, fake, n)) for n in (3 * MIB, MIB // 2)]
    assert fake.held() <= gc_module.GC_PINNED_CAP   # all ours now: 10.5 MiB before the fix
    assert fake.peak <= gc_module.GC_PINNED_CAP
    if user_cap: assert gc_module.GC_PINNED_CAP == 10 * MIB


def test_auto_cap_is_not_reduced_by_foreign_pins_present_when_computed(monkeypatch, gc_module, counted):
    # Free RAM already excludes the 6 MiB pinned, so the auto cap adds it back.
    _auto_cap(monkeypatch, 10 * MIB)
    fake = counted("pinned_max_round_threshold_mb:1")
    fake.allocate(6 * MIB)
    ours = [pin(gc_module, fake, 3 * MIB) for _ in range(4)]
    assert all(ours[:3]) and ours[3] is None
    assert gc_module.GC_PINNED_CAP == 16 * MIB and fake.peak == 15 * MIB


def test_concurrent_reservations_cannot_spend_the_same_headroom(monkeypatch, gc_module, counted):
    # Cap 11 MiB, 3 MiB of ours live, 5 MiB pinned by another user since the baseline. A reserves 2 MiB and
    # stalls before torch pins it; B's 3 MiB must not pass the cap check on the stale count (13 MiB).
    import threading
    fake = counted("pinned_max_round_threshold_mb:1", cap_bytes = 11 * MIB)
    live = gc_module._new_host_buffer(3 * MIB, torch.uint8)
    fake.allocate(5 * MIB)
    assert fake.held() == 8 * MIB
    real_empty = torch.empty                    # the model's pinned allocation
    inside, go = threading.Event(), threading.Event()

    def stalled_empty(*args, pin_memory = False, **kwargs):
        if pin_memory and threading.current_thread().name == "A":
            inside.set()
            go.wait(10)
        return real_empty(*args, pin_memory = pin_memory, **kwargs)

    monkeypatch.setattr(torch, "empty", stalled_empty)
    out = {}
    a = threading.Thread(target = lambda: out.update(a = gc_module._new_host_buffer(2 * MIB, torch.uint8)), name = "A")
    b = threading.Thread(target = lambda: out.update(b = gc_module._new_host_buffer(3 * MIB, torch.uint8)), name = "B")
    a.start()
    assert inside.wait(10)
    b.start()
    b.join(1)                                    # unfixed: B passes the check and pins while A stalls
    go.set()
    a.join(10)
    b.join(10)
    assert "a" in out and "b" in out and live is not None
    assert fake.peak <= 11 * MIB


def _pending_block_setup(monkeypatch, gc_module, cap_bytes):
    """One 4096 block reserved, released while its copy event is still pending (reads as active)."""
    monkeypatch.setenv("UNSLOTH_GC_PINNED_MAX_GB", str(cap_bytes / (1 << 30)))
    held = {"bytes": 0}
    flushes = []
    real_empty = torch.empty

    def empty(*args, pin_memory = False, **kwargs):
        if pin_memory: held["bytes"] += 4096    # the allocator cannot reuse the pending block yet
        return real_empty(*args, **kwargs)

    def flush():
        flushes.append(True)
        return False

    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(gc_module, "_host_allocator_held_bytes", lambda: held["bytes"])
    monkeypatch.setattr(gc_module, "_host_allocator_free_bytes", lambda: 0)
    monkeypatch.setattr(gc_module, "_empty_host_cache", flush)
    assert gc_module._reserve_pinned(4096)
    held["bytes"] = 4096
    gc_module._release_pinned(4096)
    return held, flushes


def test_pending_cached_block_under_the_cap_pins_one_more_without_a_flush(monkeypatch, gc_module):
    # A flush syncs every device and unpins every cached block; under the cap one more block is enough.
    held, flushes = _pending_block_setup(monkeypatch, gc_module, 1 << 20)
    assert gc_module._reserve_pinned(4096)
    assert flushes == []
    assert gc_module.GC_PINNED_BYTES == 8192 == held["bytes"]
    assert gc_module._PINNED_CACHED == {4096: 1}      # the pending block is still ours, reused later
    monkeypatch.setattr(gc_module, "_host_allocator_free_bytes", lambda: 4096)  # its copy finished
    assert gc_module._reserve_pinned(4096)              # the next request takes it, no new pin
    assert held["bytes"] == 8192 and gc_module.GC_PINNED_BYTES == 8192


def test_pending_cached_block_at_the_cap_still_flushes(monkeypatch, gc_module):
    held, flushes = _pending_block_setup(monkeypatch, gc_module, 6000)
    gc_module._reserve_pinned(4096)
    assert flushes                                       # one more block would pass the cap
    assert gc_module.GC_PINNED_BYTES <= 6000


def test_pending_blocks_are_kept_for_one_extra_generation_only(monkeypatch, gc_module):
    # Far under the cap, but a CPU running several steps ahead must not pin one more block per
    # step in flight: past two live sets (one live, one queued) the old flush runs again.
    held, flushes = _pending_block_setup(monkeypatch, gc_module, 1 << 20)
    assert gc_module._reserve_pinned(4096)              # second generation: kept, no flush
    assert flushes == [] and held["bytes"] == 8192
    assert gc_module._PINNED_CACHED == {4096: 1} and gc_module.GC_PINNED_BYTES == 8192
    gc_module._release_pinned(4096)                     # its copy is queued too
    gc_module._reserve_pinned(4096)                     # a third generation would pass 2 x 4096
    assert flushes
    assert gc_module.GC_PINNED_BYTES <= 8192


def test_pending_bound_follows_the_whole_live_set(monkeypatch, gc_module):
    # The bound is two whole live sets, not two blocks: a backward releases every block of the
    # step, and the next forward re-reserves them while their copies are still queued.
    held, flushes = _pending_block_setup(monkeypatch, gc_module, 1 << 20)
    monkeypatch.setattr(gc_module, "_host_allocator_free_bytes", lambda: 4096)  # first step done
    for _ in range(4): assert gc_module._reserve_pinned(4096)  # cached block reused, 3 new ones
    held["bytes"] = 4 * 4096
    monkeypatch.setattr(gc_module, "_host_allocator_free_bytes", lambda: 0)
    for _ in range(4): gc_module._release_pinned(4096)        # all four copies still queued
    assert gc_module._PINNED_CACHED == {4096: 4} and gc_module._PINNED_LIVE_PEAK == 4 * 4096
    for _ in range(4): assert gc_module._reserve_pinned(4096)  # one extra generation: all kept
    assert flushes == [] and held["bytes"] == gc_module.GC_PINNED_BYTES == 8 * 4096
    gc_module._release_pinned(4096)
    gc_module._reserve_pinned(4096)                            # past 2 x the live set: flush
    assert flushes
