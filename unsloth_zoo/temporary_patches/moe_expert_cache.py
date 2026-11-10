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
"""VRAM-budgeted cache of dequantized (16-bit) frozen 4-bit expert stacks.

A frozen NF4 expert stack is dequantized up to three times per layer per step (no-grad forward, checkpoint replay,
backward). The decode is deterministic, so keeping the 16-bit copy of as many layers as fit is exact and removes those
decodes for the cached layers.

UNSLOTH_MOE_EXPERT_CACHE:
  0 (default)  off.
  1 / auto     admit a stack only after it was decoded ADMIT_AFTER times (a full training step has run, so the
               allocator's peak holds that step's activations), while
               cached + stack <= total - peak_reserved_without_cache - margin.
  <float>      the same, capped at that many GiB.
Safety: before every lookup the device's free + cached-unused memory is checked (rate limited); under the margin the
least recently used stacks are dropped until it is met again. clear_expert_cache() drops everything; a dequant that hits
OOM drops the cache and retries once (moe_utils). The cache is host state: traced callers decode as before (no graph
break) unless UNSLOTH_MOE_EXPERT_CACHE_TRACED=1, which reaches the cache through a compiler-disabled call (one graph
break per decode).
"""

__all__ = [
    "EXPERT_CACHE_STATS",
    "expert_cache_enabled",
    "expert_cache_lookup",
    "expert_cache_offer",
    "clear_expert_cache",
]

import os
import threading
import time
import weakref
from collections import OrderedDict

import torch

EXPERT_CACHE_STATS = {"hits": 0, "misses": 0, "admits": 0, "evictions": 0, "declined": 0, "oom_clears": 0, "bytes": 0}

ADMIT_AFTER = 4            # decodes of one stack before it may be cached
MARGIN_MIN_BYTES = 1 << 30
MARGIN_FRAC = 0.05
_CHECK_EVERY_S = 0.25      # free-memory check rate per device

_lock = threading.Lock()
_CACHE = OrderedDict()     # key -> (weakref to param, data_ptr, tensor)
_SEEN = {}                 # key -> decode count
_LAST_CHECK = {}           # device index -> monotonic time
_PEAK0 = {}                # device index -> allocator peak before anything was cached


def _mode():
    raw = os.environ.get("UNSLOTH_MOE_EXPERT_CACHE", "0").strip().lower()
    if raw in ("", "0", "off", "false", "no"):
        return None
    if raw in ("1", "auto", "on", "true", "yes"):
        return float("inf")
    try:
        gib = float(raw)
    except ValueError:
        return None
    return gib * 2**30 if gib > 0 else None


def expert_cache_enabled() -> bool:
    return _mode() is not None


def _key(param):
    return id(param)


def _margin(total):
    return max(MARGIN_MIN_BYTES, int(MARGIN_FRAC * total))


def _mem(device):
    """(free incl. cached-unused, total, peak reserved) or None when unmeasurable."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        free, total = torch.cuda.mem_get_info(device)
        free += torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        return free, total, torch.cuda.max_memory_reserved(device)
    except Exception:
        return None


def _cached_bytes(device = None):
    n = 0
    for ref, ptr, t in _CACHE.values():
        if device is None or t.device == device:
            n += t.numel() * t.element_size()
    return n


def _enforce_margin(device, force = False):
    """Drop least recently used stacks of `device` while free + cached-unused memory is under the margin."""
    now = time.monotonic()
    idx = device.index if device.index is not None else -1
    if not force and now - _LAST_CHECK.get(idx, 0.0) < _CHECK_EVERY_S:
        return
    _LAST_CHECK[idx] = now
    while True:
        m = _mem(device)
        if m is None or m[0] >= _margin(m[1]):
            break
        k = next((k for k, v in _CACHE.items() if v[2].device == device), None)
        if k is None:
            break
        del _CACHE[k]   # its memory returns to the allocator once the caller drops its own reference
        EXPERT_CACHE_STATS["evictions"] += 1
    EXPERT_CACHE_STATS["bytes"] = _cached_bytes()


def expert_cache_lookup(param):
    """The cached 16-bit stack for a frozen 4-bit param, or None."""
    if torch.compiler.is_compiling() or _mode() is None:
        return None
    with _lock:
        k = _key(param)
        hit = _CACHE.get(k)
        if hit is not None:
            ref, ptr, t = hit
            if ref() is not param or param.data_ptr() != ptr:
                del _CACHE[k]
                hit = None
        if hit is None:
            EXPERT_CACHE_STATS["misses"] += 1
            return None
        _CACHE.move_to_end(k)
        _enforce_margin(t.device)
        if k not in _CACHE:
            EXPERT_CACHE_STATS["misses"] += 1
            return None
        EXPERT_CACHE_STATS["hits"] += 1
        return t


def expert_cache_offer(param, weight):
    """Called with a freshly decoded full stack; keeps a reference when the budget allows (no copy)."""
    cap = _mode()
    if cap is None or torch.compiler.is_compiling() or param.requires_grad:
        return
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return
    with _lock:
        k = _key(param)
        n = _SEEN.get(k, 0) + 1
        _SEEN[k] = n
        if n < ADMIT_AFTER or k in _CACHE:
            return
        device = weight.device
        m = _mem(device)
        if m is None:
            EXPERT_CACHE_STATS["declined"] += 1
            return
        free, total, peak = m
        need = weight.numel() * weight.element_size()
        cached = _cached_bytes(device)
        # Budget against the peak seen before the first admission (one full uncached step), so every cached byte
        # adds to that peak and the margin stays free at it.
        idx = device.index if device.index is not None else -1
        if not cached:
            _PEAK0[idx] = peak
        budget = total - _PEAK0.get(idx, peak) - _margin(total)
        if cached + need > min(budget, cap) or free - need < _margin(total):
            EXPERT_CACHE_STATS["declined"] += 1
            return
        _CACHE[k] = (weakref.ref(param), param.data_ptr(), weight)
        EXPERT_CACHE_STATS["admits"] += 1
        EXPERT_CACHE_STATS["bytes"] = _cached_bytes()


def clear_expert_cache(oom = False):
    with _lock:
        _CACHE.clear()
        _SEEN.clear()
        _PEAK0.clear()
        if oom:
            EXPERT_CACHE_STATS["oom_clears"] += 1
        EXPERT_CACHE_STATS["bytes"] = 0
