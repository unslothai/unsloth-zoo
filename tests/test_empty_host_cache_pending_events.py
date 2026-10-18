# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""A pinned offload slot that goes pageable on growth leaves its old block in torch's host cache.
Its last use was a non_blocking copy, so torch.cuda.host_memory_stats() still counts it active
until the allocator processes its event. _empty_host_cache must still flush it."""
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
    monkeypatch.setattr(gc_module, "GC_PINNED_BYTES", 0, raising = False)
    monkeypatch.setattr(gc_module, "GC_PINNED_CAP", gc_module._PIN_PER_BLOCK, raising = False)
    monkeypatch.setattr(gc_module, "_PINNED_CACHED", {}, raising = False)
    monkeypatch.setattr(gc_module, "_HOST_HELD_BASELINE", None, raising = False)
    monkeypatch.delenv("UNSLOTH_DISABLE_PINNED_MEMORY", raising = False)
    monkeypatch.delenv("UNSLOTH_GC_PINNED_MAX_GB", raising = False)
    return gc_module


def test_flushes_our_released_blocks_when_stats_report_none_free(gc_module, monkeypatch):
    calls = []
    monkeypatch.setattr(gc_module, "DEVICE_TYPE", "cuda", raising = False)
    monkeypatch.setattr(gc_module, "_host_allocator_free_bytes", lambda: 0, raising = False)
    monkeypatch.setattr(torch._C, "_host_emptyCache", lambda: calls.append(1), raising = False)
    gc_module._PINNED_CACHED[1 << 24] = 1
    gc_module.GC_PINNED_BYTES = 1 << 24
    assert gc_module._empty_host_cache() is True
    assert calls == [1]
    assert gc_module.GC_PINNED_BYTES == 0 and not gc_module._PINNED_CACHED


def test_nothing_released_and_nothing_free_skips_the_flush(gc_module, monkeypatch):
    calls = []
    monkeypatch.setattr(gc_module, "DEVICE_TYPE", "cuda", raising = False)
    monkeypatch.setattr(gc_module, "_host_allocator_free_bytes", lambda: 0, raising = False)
    monkeypatch.setattr(torch._C, "_host_emptyCache", lambda: calls.append(1), raising = False)
    assert gc_module._empty_host_cache() is False
    assert calls == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA pinned memory")
def test_pageable_fallback_frees_block_last_used_by_async_copy(gc_module, monkeypatch):
    torch.zeros(1, device = "cuda")
    torch._C._host_emptyCache()
    base = torch.cuda.host_memory_stats()["allocated_bytes.current"]
    monkeypatch.setattr(gc_module, "_block_fits_host_ram", lambda block: False)
    n = 4 << 20
    x = gc_module._mark_host_buffer(torch.empty(n, dtype = torch.float16, pin_memory = True))
    gc_module.GC_PINNED_BYTES = gc_module._pinned_block_bytes(gc_module._storage_bytes(x))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x.copy_(torch.ones(n, device = "cuda", dtype = torch.float16), non_blocking = True)
    torch.cuda.synchronize()
    grown = gc_module._grow_host_buffer(x, 2 * n)  # host RAM "full": pageable
    assert not grown.is_pinned()
    del x
    assert torch.cuda.host_memory_stats()["allocated_bytes.current"] > base
    assert gc_module._empty_host_cache() is True
    assert torch.cuda.host_memory_stats()["allocated_bytes.current"] == base


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA pinned memory")
def test_flush_waits_for_a_copy_still_in_flight(gc_module):
    # A released block whose non_blocking copy has not finished stays queued in torch's host
    # cache; the flush must not drop its bookkeeping while the allocator still holds it.
    torch.cuda.synchronize()
    torch._C._host_emptyCache()
    base = torch.cuda.host_memory_stats()["allocated_bytes.current"]
    n = 4 << 20
    x = gc_module._mark_host_buffer(torch.empty(n, dtype = torch.float16, pin_memory = True))
    block = gc_module._pinned_block_bytes(gc_module._storage_bytes(x))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(200_000_000)  # keep the copy in flight while we flush
        torch.ones(n, device = "cuda", dtype = torch.float16).copy_(x, non_blocking = True)
    del x
    gc_module._PINNED_CACHED[block] = 1
    gc_module.GC_PINNED_BYTES = block
    assert gc_module._empty_host_cache() is True
    assert not gc_module._PINNED_CACHED and gc_module.GC_PINNED_BYTES == 0
    assert torch.cuda.host_memory_stats()["allocated_bytes.current"] == base
