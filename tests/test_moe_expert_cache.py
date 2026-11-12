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
"""UNSLOTH_MOE_EXPERT_CACHE budget logic with a simulated device memory (CPU tensors, mocked mem_get_info)."""
import pytest
import torch

from unsloth_zoo.temporary_patches import moe_expert_cache as ec

GiB = 2**30


class FakeParam:
    requires_grad = False

    def __init__(self):
        self.t = torch.empty(4, dtype = torch.uint8)

    def data_ptr(self):
        return self.t.data_ptr()


@pytest.fixture
def mem(monkeypatch):
    """total 40 GiB, the step peak and free memory set by the test; cached bytes are taken from free."""
    state = {"total": 40 * GiB, "peak": 20 * GiB, "free_base": 20 * GiB}

    def fake_mem(device):
        return state["free_base"] - ec._cached_bytes(), state["total"], state["peak"]

    monkeypatch.setattr(ec, "_mem", fake_mem)
    monkeypatch.setattr(ec, "_CHECK_EVERY_S", 0.0)
    monkeypatch.setenv("UNSLOTH_MOE_EXPERT_CACHE", "1")
    ec.clear_expert_cache()
    for k in ec.EXPERT_CACHE_STATS:
        ec.EXPERT_CACHE_STATS[k] = 0
    yield state
    ec.clear_expert_cache()


def stack(gib):
    # A CPU stand-in whose element count encodes the size; storage is tiny via expand.
    t = torch.zeros(1, dtype = torch.bfloat16).expand(int(gib * GiB // 2))
    return t


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("UNSLOTH_MOE_EXPERT_CACHE", raising = False)
    assert not ec.expert_cache_enabled()
    p = FakeParam()
    for _ in range(10):
        ec.expert_cache_offer(p, stack(1))
    assert ec.expert_cache_lookup(p) is None


def test_admits_only_after_a_full_step(mem):
    p = FakeParam()
    w = stack(1)
    for i in range(ec.ADMIT_AFTER - 1):
        ec.expert_cache_offer(p, w)
        assert ec.expert_cache_lookup(p) is None
    ec.expert_cache_offer(p, w)
    assert ec.expert_cache_lookup(p) is w
    assert ec.EXPERT_CACHE_STATS["admits"] == 1 and ec.EXPERT_CACHE_STATS["hits"] == 1


def test_budget_is_total_minus_uncached_peak_minus_margin(mem):
    # budget = 40 - 20 - max(1, 2) = 18 GiB: six 3 GiB stacks fit, the seventh is declined.
    params = [FakeParam() for _ in range(7)]
    stacks = [stack(3) for _ in params]
    mem["free_base"] = 40 * GiB    # plenty free right now: only the peak-based budget limits
    for _ in range(ec.ADMIT_AFTER):
        for p, w in zip(params, stacks):
            ec.expert_cache_offer(p, w)
        mem["peak"] = 20 * GiB + ec._cached_bytes()   # later peaks include the cache; the budget must not shrink
    hits = [ec.expert_cache_lookup(p) is not None for p in params]
    assert hits == [True] * 6 + [False]
    assert ec.EXPERT_CACHE_STATS["declined"] >= 1


def test_gib_cap(mem, monkeypatch):
    monkeypatch.setenv("UNSLOTH_MOE_EXPERT_CACHE", "5")
    mem["free_base"] = 40 * GiB
    params = [FakeParam() for _ in range(3)]
    for _ in range(ec.ADMIT_AFTER):
        for p in params:
            ec.expert_cache_offer(p, stack(2))
    assert sum(ec.expert_cache_lookup(p) is not None for p in params) == 2


def test_eviction_when_free_memory_drops(mem):
    mem["free_base"] = 30 * GiB
    params = [FakeParam() for _ in range(4)]
    for _ in range(ec.ADMIT_AFTER):
        for p in params:
            ec.expert_cache_offer(p, stack(2))
    assert ec._cached_bytes() == 8 * GiB
    # A ballast takes memory: free + cached-unused falls to 4.5 GiB minus the cache; margin is 2 GiB (5% of 40).
    mem["free_base"] = int(4.5 * GiB)
    assert ec.expert_cache_lookup(params[3]) is not None     # most recent one survives
    assert ec._cached_bytes() == 2 * GiB                     # 3 LRU stacks dropped: free -3.5 -> 2.5 GiB >= 2
    assert ec.EXPERT_CACHE_STATS["evictions"] == 3
    assert ec.expert_cache_lookup(params[0]) is None


def test_stale_param_is_dropped(mem):
    mem["free_base"] = 40 * GiB
    p = FakeParam()
    for _ in range(ec.ADMIT_AFTER):
        ec.expert_cache_offer(p, stack(1))
    p.t = torch.empty(8, dtype = torch.uint8)   # storage swapped (reload / requantize)
    assert ec.expert_cache_lookup(p) is None


def test_trainable_param_never_cached(mem):
    mem["free_base"] = 40 * GiB
    p = FakeParam()
    p.requires_grad = True
    for _ in range(ec.ADMIT_AFTER + 2):
        ec.expert_cache_offer(p, stack(1))
    assert ec.expert_cache_lookup(p) is None


def test_clear_on_oom(mem):
    mem["free_base"] = 40 * GiB
    p = FakeParam()
    for _ in range(ec.ADMIT_AFTER):
        ec.expert_cache_offer(p, stack(1))
    ec.clear_expert_cache(oom = True)
    assert ec.expert_cache_lookup(p) is None and ec.EXPERT_CACHE_STATS["oom_clears"] == 1
