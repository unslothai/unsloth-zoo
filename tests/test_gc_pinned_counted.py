# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""With torch 2.13+ host allocator settings (pinned_max_round_threshold_mb / pinned_max_cached_size_mb)
the pinned cap reads the allocator's own byte count. CPU only: a small model of torch 2.14's
CachingHostAllocator (aten/src/ATen/core/CachingHostAllocator.h: Log2 buckets, a request takes any
cached block in its bucket at least as large, blocks above pinned_max_cached_size_mb are destroyed
once their copy events are processed) stands in for host_memory_stats / pinned allocations."""
import weakref

import pytest
import torch

MIB = 1 << 20


class FakeHostAllocator:
    def __init__(self, round_max = float("inf"), cached_max = float("inf")):
        self.round_max, self.cached_max = round_max, cached_max
        self.blocks = set()      # every block the allocator owns (allocated_bytes)
        self.free = {}           # bucket -> list of cached free block sizes, as ids
        self.pending = []        # (block, event done) released with a copy event outstanding
        self.events_done = False
        self.peak_after_alloc = 0
        self.peak = 0            # over every allocation, trial ones included
        self._next = 0

    def _new(self, size):
        self._next += 1
        block = (self._next, size)
        self.blocks.add(block)
        self.peak = max(self.peak, self.held())
        return block

    @staticmethod
    def _bucket(size):
        return (int(size) - 1).bit_length()

    def held(self):
        return sum(size for _, size in self.blocks)

    def active(self):
        cached = sum(size for blocks in self.free.values() for _, size in blocks)
        return self.held() - cached

    def _cache(self, block):
        if block[1] > self.cached_max: self.blocks.discard(block)
        else: self.free.setdefault(self._bucket(block[1]), []).append(block)

    def process_events(self):
        if not self.events_done: return
        pending, self.pending = self.pending, []
        for block in pending: self._cache(block)

    def allocate(self, size):
        self.process_events()
        if size <= self.round_max and size <= self.cached_max: size = 1 << (int(size) - 1).bit_length()
        bucket = self.free.get(self._bucket(size), [])
        for i in range(len(bucket) - 1, -1, -1):
            if bucket[i][1] >= size: return bucket.pop(i)
        return self._new(size)

    def release(self, block, copy_event = False):
        if copy_event: self.pending.append(block)
        else: self._cache(block)

    def empty_cache(self):
        self.process_events()
        for blocks in self.free.values():
            for block in blocks: self.blocks.discard(block)
        self.free.clear()


@pytest.fixture
def gc_module(monkeypatch):
    monkeypatch.setenv("UNSLOTH_ALLOW_CPU", "1")
    try:
        from unsloth_zoo import gradient_checkpointing as gc_module
    except ImportError as e:
        pytest.skip(f"unsloth_zoo import unavailable: {e}")
    monkeypatch.setattr(gc_module, "PINNED_MEMORY_AVAILABLE", True, raising = False)
    monkeypatch.setattr(gc_module, "GC_PINNED_BYTES", 0, raising = False)
    monkeypatch.setattr(gc_module, "GC_PINNED_CAP", None, raising = False)
    monkeypatch.setattr(gc_module, "_WARNED_ABOUT_PINNED_CAP", True, raising = False)
    monkeypatch.setattr(gc_module, "_PINNED_CACHED", {}, raising = False)
    monkeypatch.setattr(gc_module, "_PINNED_LIVE", {}, raising = False)
    monkeypatch.setattr(gc_module, "_PINNED_RELEASED", 0, raising = False)
    monkeypatch.setattr(gc_module, "_HOST_HELD_BASELINE", None, raising = False)
    monkeypatch.setattr(gc_module, "DEVICE_TYPE", "cuda", raising = False)
    monkeypatch.delenv("UNSLOTH_GC_PINNED_MAX_GB", raising = False)
    return gc_module


@pytest.fixture
def allocator(monkeypatch, gc_module):
    """Install the fake allocator for `settings`; returns a factory."""
    def install(settings, cap_bytes = None):
        round_max, cached_max = gc_module._parse_host_alloc_policy(settings)
        fake = FakeHostAllocator(round_max, cached_max)
        monkeypatch.setattr(torch._C, "_accelerator_getAllocatorSettings", lambda: settings, raising = False)
        monkeypatch.setattr(gc_module, "_HOST_ALLOC_KNOBS", True, raising = False)
        monkeypatch.setattr(gc_module, "_HOST_ALLOC_POLICY", (None, None, None), raising = False)
        monkeypatch.setattr(gc_module, "_WARNED_SHORT_BLOCK", True, raising = False)
        monkeypatch.setattr(gc_module, "_host_allocator_held_bytes", fake.held, raising = False)
        monkeypatch.setattr(gc_module, "_host_allocator_free_bytes", lambda: fake.held() - fake.active(), raising = False)
        monkeypatch.setattr(torch._C, "_host_emptyCache", fake.empty_cache, raising = False)

        def synchronize():
            fake.events_done = True
        monkeypatch.setattr(gc_module, "_synchronize_used_devices", synchronize, raising = False)
        real_empty = torch.empty

        def empty(*args, pin_memory = False, **kwargs):
            out = real_empty(*args, **kwargs)
            if pin_memory:  # the reuse probe inside _reserve_pinned
                block = fake.allocate(out.numel() * out.element_size())
                weakref.finalize(out, fake.release, block)
            return out
        monkeypatch.setattr(torch, "empty", empty)
        if cap_bytes is not None:
            monkeypatch.setenv("UNSLOTH_GC_PINNED_MAX_GB", str(cap_bytes / (1 << 30)))
        return fake
    return install


def pin(gc_module, fake, nbytes):
    """What an offload buffer does: reserve, then torch really allocates. None = went pageable."""
    if not gc_module._reserve_pinned(nbytes): return None
    block = fake.allocate(nbytes)
    fake.peak_after_alloc = max(fake.peak_after_alloc, fake.held())
    return block


def unpin(gc_module, fake, nbytes, block, copy_event = False):
    gc_module._release_pinned(nbytes)
    fake.release(block, copy_event)


def test_same_bucket_reuse_of_a_larger_block_cannot_bypass_the_cap(gc_module, allocator):
    # Codex 4405695356: torch 2.14, threshold 1, cap 10.5 MiB.
    fake = allocator("pinned_max_round_threshold_mb:1", cap_bytes = 10.5 * MIB)
    a = pin(gc_module, fake, int(3.5 * MIB))
    b = pin(gc_module, fake, 2 * MIB)
    c = pin(gc_module, fake, 2 * MIB)
    assert a and b and c and fake.held() == pytest.approx(7.5 * MIB, abs = 64)
    unpin(gc_module, fake, int(3.5 * MIB), a)
    d = pin(gc_module, fake, 3 * MIB)            # torch hands back the cached 3.5 MiB block
    assert d == a and fake.held() == pytest.approx(7.5 * MIB, abs = 64)
    unpin(gc_module, fake, 2 * MIB, b)
    unpin(gc_module, fake, 2 * MIB, c)
    e = pin(gc_module, fake, int(3.5 * MIB))      # no 3.5 MiB block is cached any more
    # Real pinned bytes stay under the cap at every allocation (the simulation reached 11 MiB).
    assert fake.peak <= 10.5 * MIB + 64
    assert e is not None and fake.held() == pytest.approx(7 * MIB, abs = 64)  # the flush returned the two 2 MiB blocks


def test_released_block_waiting_on_its_copy_event_is_flushed(gc_module, allocator):
    # Codex 4405695372: cached max 2 MiB, a 3 MiB buffer last used by a non_blocking copy.
    fake = allocator("pinned_max_cached_size_mb:2", cap_bytes = 4 * MIB)
    a = pin(gc_module, fake, 3 * MIB)
    assert a and fake.held() == pytest.approx(3 * MIB, abs = 64)
    unpin(gc_module, fake, 3 * MIB, a, copy_event = True)
    assert fake.held() == pytest.approx(3 * MIB, abs = 64)                # still pinned in torch's event queue
    # The next offload must not stay pageable until an unrelated allocation drains the queue.
    b = pin(gc_module, fake, 3 * MIB)
    assert b is not None and fake.held() == pytest.approx(3 * MIB, abs = 64)
    assert fake.peak <= 4 * MIB


def test_completed_copy_event_is_drained_without_a_device_sync(monkeypatch, gc_module, allocator):
    fake = allocator("pinned_max_cached_size_mb:2", cap_bytes = 4 * MIB)
    a = pin(gc_module, fake, 3 * MIB)
    unpin(gc_module, fake, 3 * MIB, a, copy_event = True)
    fake.events_done = True                      # the copy finished; torch has not looked yet
    syncs = []
    monkeypatch.setattr(gc_module, "_synchronize_used_devices", lambda: syncs.append(1))
    assert pin(gc_module, fake, 3 * MIB) is not None
    assert syncs == [] and fake.peak <= 4 * MIB + 64


def test_released_pending_block_is_flushed_in_per_block_mode(monkeypatch, gc_module, allocator):
    fake = allocator("pinned_max_cached_size_mb:2")
    gc_module.GC_PINNED_CAP = gc_module._PIN_PER_BLOCK
    # Host RAM fits the block only once the pending 3 MiB block is returned to the OS.
    monkeypatch.setattr(gc_module, "_block_fits_host_ram", lambda block: fake.held() + block <= 4 * MIB)
    a = pin(gc_module, fake, 3 * MIB)
    unpin(gc_module, fake, 3 * MIB, a, copy_event = True)
    assert pin(gc_module, fake, 3 * MIB) is not None and fake.held() == pytest.approx(3 * MIB, abs = 64)


def test_settings_change_while_live_cannot_take_a_phantom_cached_block(gc_module, allocator):
    # The earlier P2 (release at the allocation-time size), against the allocator's own count.
    fake = allocator("pinned_max_round_threshold_mb:1", cap_bytes = 4 * MIB)
    a = pin(gc_module, fake, 3 * MIB)
    b = pin(gc_module, fake, MIB)
    assert a and b and fake.held() == pytest.approx(4 * MIB, abs = 64)
    unpin(gc_module, fake, 3 * MIB, a)
    unpin(gc_module, fake, MIB, b)
    # A 4 MiB request finds no 4 MiB block cached; the flush returns the 3 and 1 MiB blocks first.
    c = pin(gc_module, fake, 4 * MIB)
    assert fake.peak <= 4 * MIB + 64 and c is not None and fake.held() == pytest.approx(4 * MIB, abs = 64)


def test_cached_blocks_at_the_cap_are_flushed_not_overshot(gc_module, allocator):
    # At the cap with only our cached blocks: return them and pin anew, never pin a trial block first.
    fake = allocator("pinned_max_round_threshold_mb:1", cap_bytes = 6 * MIB)
    a = pin(gc_module, fake, 3 * MIB)
    b = pin(gc_module, fake, 3 * MIB)
    unpin(gc_module, fake, 3 * MIB, a)
    unpin(gc_module, fake, 3 * MIB, b)
    assert pin(gc_module, fake, 4 * MIB) is not None
    assert pin(gc_module, fake, 2 * MIB) is not None
    assert fake.peak <= 6 * MIB + 64 and fake.held() == pytest.approx(6 * MIB, abs = 64)


def test_full_cap_with_nothing_cached_refuses_without_probing(monkeypatch, gc_module, allocator):
    fake = allocator("pinned_max_round_threshold_mb:1", cap_bytes = 4 * MIB)
    assert pin(gc_module, fake, 3 * MIB)
    flushes = []
    monkeypatch.setattr(gc_module, "_empty_host_cache", lambda: flushes.append(1) or False)
    assert pin(gc_module, fake, 3 * MIB) is None
    assert flushes == [] and fake.held() == pytest.approx(3 * MIB, abs = 64)


def test_unset_settings_keep_the_simulated_path(monkeypatch, gc_module, allocator):
    fake = allocator("", cap_bytes = 8 * MIB)
    counted = []
    monkeypatch.setattr(gc_module, "_reserve_pinned_counted", lambda *a: counted.append(a) or True, raising = False)
    a = pin(gc_module, fake, 3 * MIB)
    assert a and counted == [] and gc_module.GC_PINNED_BYTES == 4 * MIB
    unpin(gc_module, fake, 3 * MIB, a)
    assert gc_module._PINNED_CACHED == {4 * MIB: 1} and gc_module._PINNED_RELEASED == 0


@pytest.mark.parametrize("settings", ["pinned_max_round_threshold_mb:1", "pinned_max_cached_size_mb:1"])
def test_foreign_pin_between_reservations_counts_against_the_cap(gc_module, allocator, settings):
    # Codex 4405981969: cap 10 MiB, reserve 3, another user pins 6 (training_utils.py's input_ids /
    # labels, a pin_memory DataLoader), reserve 3. 12 MiB would really be held.
    fake = allocator(settings, cap_bytes = 10 * MIB)
    a = pin(gc_module, fake, 3 * MIB)
    assert a and fake.held() == pytest.approx(3 * MIB, abs = 64)
    foreign = fake.allocate(6 * MIB)
    assert pin(gc_module, fake, 3 * MIB) is None
    assert fake.held() == pytest.approx(9 * MIB, abs = 64)
    # Once the foreign block is gone the next offload pins again.
    fake.release(foreign)
    fake.empty_cache()
    assert pin(gc_module, fake, 3 * MIB) is not None
    assert fake.peak <= 10 * MIB + 64


def test_counted_path_reads_the_allocator_on_every_reservation(monkeypatch, gc_module, allocator):
    fake = allocator("pinned_max_round_threshold_mb:1", cap_bytes = 10 * MIB)
    reads = []
    held = fake.held
    monkeypatch.setattr(gc_module, "_host_allocator_held_bytes", lambda: reads.append(1) or held())
    blocks = [pin(gc_module, fake, 3 * MIB) for _ in range(3)]
    assert all(blocks) and len(reads) == 3
    assert pin(gc_module, fake, 3 * MIB) is None and len(reads) >= 4
