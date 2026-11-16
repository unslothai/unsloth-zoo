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
  0 (default)  off (also "", off, false, no, any zero / negative / unparsable value).
  1 / auto     admit a stack only after it was decoded ADMIT_AFTER times (a full training step has run, so the
               allocator's peak holds that step's activations), while
               cached + stack <= total - peak_reserved_without_cache - margin.
  <float>      the same, capped at that many GiB.
Safety: before every lookup the device's free + cached-unused memory is checked (rate limited); under the margin the
least recently used stacks are dropped until it is met again. clear_expert_cache() drops everything; a dequant that hits
OOM drops the cache and retries once (with_expert_cache_oom_retry). The cache is host state: traced callers decode as
before (no graph break) unless UNSLOTH_MOE_EXPERT_CACHE_TRACED=1, which reaches the cache through a compiler-disabled
call (one graph break per decode).
An entry is dropped when its param dies (weakref callback) and on any change to the packed bytes or quant state
(storage pointer, in-place version counters, QuantState identity / metadata). It is only served on the CUDA stream that
produced it and never while a CUDA graph is being captured.
"""

__all__ = [
    "EXPERT_CACHE_STATS",
    "expert_cache_enabled",
    "expert_cache_lookup",
    "expert_cache_offer",
    "clear_expert_cache",
    "parse_expert_cache_mode",
    "with_expert_cache_oom_retry",
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

# Reentrant: a weakref callback (a param dying, maybe in a gc pass triggered under the lock) takes it too.
_lock = threading.RLock()
_CACHE = OrderedDict()     # key -> (source token, producing stream id, tensor)
_SEEN = {}                 # key -> decode count
_REFS = {}                 # key -> weakref to the param; its callback drops the key's entries
_LAST_CHECK = {}           # device index -> monotonic time
_PEAK0 = {}                # device index -> allocator peak before anything was cached

_OFF = ("", "0", "off", "false", "no")
_ON = ("1", "auto", "on", "true", "yes")


def parse_expert_cache_mode(raw):
    """UNSLOTH_MOE_EXPERT_CACHE -> None (off), inf (auto) or a byte cap. The one parser every gate uses. Raises
    nothing (traced callers evaluate it), so anything unrecognised, zero or negative is off."""
    raw = (raw or "0").strip().lower()
    if raw in _OFF:
        return None
    if raw in _ON:
        return float("inf")
    if not raw.replace(".", "", 1).isdigit():
        return None
    gib = float(raw)
    return gib * 2**30 if gib > 0 else None


def _mode():
    return parse_expert_cache_mode(os.environ.get("UNSLOTH_MOE_EXPERT_CACHE", "0"))


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
    for _, _, t in list(_CACHE.values()):
        if device is None or t.device == device:
            n += t.numel() * t.element_size()
    return n


def _tensor_token(t):
    if isinstance(t, torch.Tensor):
        # _version counts in-place writes (copy_, load_state_dict); it raises on inference tensors.
        return (id(t), t.data_ptr(), t._version, t.dtype, tuple(t.shape))
    if t is None or isinstance(t, (bool, int, float, str)):
        return t
    return id(t)


def _source_token(param):
    """What the decoded stack depends on: the packed storage and its in-place version, the QuantState objects and their
    tensors / metadata. None when that cannot be tracked (inference tensors carry no version counter)."""
    try:
        qs = getattr(param, "quant_state", None)
        s2 = getattr(qs, "state2", None)
        parts = [param.data_ptr(), getattr(param, "_version", None), id(qs), id(s2),
                 tuple(getattr(param, "_original_shape", None) or ())]
        for st in (qs, s2):
            if st is None:
                continue
            shape = getattr(st, "shape", None)
            parts.append((getattr(st, "dtype", None), getattr(st, "blocksize", None),
                          getattr(st, "quant_type", None), tuple(shape) if shape is not None else None))
            for name in ("absmax", "code", "offset"):
                parts.append(_tensor_token(getattr(st, name, None)))
        return tuple(parts)
    except RuntimeError:
        return None


def _stream_id(t):
    if not t.is_cuda:
        return None
    return torch.cuda.current_stream(t.device).cuda_stream


def _capturing(device = None):
    """Whether a CUDA graph capture is running on the current device or on `device` (the operand's GPU):
    `is_current_stream_capturing` only asks the current device, and an operand on another GPU runs on that
    GPU's current stream."""
    if not torch.cuda.is_available():
        return False
    if torch.cuda.is_current_stream_capturing():
        return True
    if isinstance(device, torch.device) and device.type == "cuda" and device.index is not None \
            and device.index != torch.cuda.current_device():
        with torch.cuda.device(device):
            return torch.cuda.is_current_stream_capturing()
    return False


def _forget(key, ref):
    """Weakref callback: the param died, so its decoded stack must not outlive it."""
    with _lock:
        if _REFS.get(key) is not ref:
            return
        _drop(key)
        EXPERT_CACHE_STATS["bytes"] = _cached_bytes()


def _drop(key):
    _CACHE.pop(key, None)
    _SEEN.pop(key, None)
    _REFS.pop(key, None)


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
        k = next((k for k, v in list(_CACHE.items()) if v[2].device == device), None)
        if k is None:
            break
        _CACHE.pop(k, None)   # its memory returns to the allocator once the caller drops its own reference
        EXPERT_CACHE_STATS["evictions"] += 1
    EXPERT_CACHE_STATS["bytes"] = _cached_bytes()


def expert_cache_lookup(param, count = True):
    """The cached 16-bit stack for a frozen 4-bit param, or None. count = False probes without touching the stats."""
    if torch.compiler.is_compiling() or _mode() is None or _capturing(getattr(param, "device", None)):
        return None
    with _lock:
        k = _key(param)
        hit = _CACHE.get(k)
        if hit is not None:
            ref = _REFS.get(k)
            token = _source_token(param)
            if ref is None or ref() is not param or token is None or token != hit[0]:
                _drop(k)   # reloaded / requantized / edited in place: decode again
                EXPERT_CACHE_STATS["bytes"] = _cached_bytes()
                hit = None
            elif hit[1] != _stream_id(hit[2]):
                hit = None   # another stream: nothing orders it after the producer, so decode there instead
        if hit is None:
            if count:
                EXPERT_CACHE_STATS["misses"] += 1
            return None
        t = hit[2]
        _CACHE.move_to_end(k)
        _enforce_margin(t.device)
        if k not in _CACHE:
            if count:
                EXPERT_CACHE_STATS["misses"] += 1
            return None
        if count:
            EXPERT_CACHE_STATS["hits"] += 1
        return t


def expert_cache_offer(param, weight):
    """Called with a freshly decoded full stack; keeps a reference when the budget allows (no copy)."""
    cap = _mode()
    if cap is None or torch.compiler.is_compiling() or param.requires_grad:
        return
    # An inference tensor cannot be saved for backward by a later training step.
    if weight.is_inference() or _capturing(weight.device):
        return
    with _lock:
        k = _key(param)
        ref = _REFS.get(k)
        if ref is None or ref() is not param:
            _drop(k)
            try:
                ref = weakref.ref(param, lambda r, key = k: _forget(key, r))
            except TypeError:
                return
            _REFS[k] = ref
        n = _SEEN.get(k, 0) + 1
        _SEEN[k] = n
        if n < ADMIT_AFTER:
            return
        old = _CACHE.get(k)
        if old is not None:
            if old[2].dtype == weight.dtype:
                return
            _CACHE.pop(k, None)   # the compute dtype changed: the old entry never hits again
        token = _source_token(param)
        if token is None:
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
        # adds to that peak and the margin stays free at it. Kept through evictions and OOM clears: the allocator peak
        # by then includes the cache and would shrink the budget for good.
        idx = device.index if device.index is not None else -1
        if idx not in _PEAK0:
            _PEAK0[idx] = peak
        budget = total - _PEAK0[idx] - _margin(total)
        # `free` was measured with `weight` already allocated: keeping it costs nothing more right now.
        if cached + need > min(budget, cap) or free < _margin(total):
            EXPERT_CACHE_STATS["declined"] += 1
            return
        _CACHE[k] = (token, _stream_id(weight), weight)
        EXPERT_CACHE_STATS["admits"] += 1
        EXPERT_CACHE_STATS["bytes"] = _cached_bytes()


def clear_expert_cache(oom = False):
    """Drop every cached stack. oom = True (a decode ran out of memory) keeps the uncached peak baseline."""
    with _lock:
        _CACHE.clear()
        _SEEN.clear()
        _REFS.clear()
        if oom:
            EXPERT_CACHE_STATS["oom_clears"] += 1
        else:
            _PEAK0.clear()
        EXPERT_CACHE_STATS["bytes"] = 0


def with_expert_cache_oom_retry(fn):
    """fn(); on CUDA OOM while stacks are cached, drop the cache and run fn once more (after the except block, so the
    failed call's frames and tensors are released first)."""
    try:
        return fn()
    except torch.cuda.OutOfMemoryError:
        if not EXPERT_CACHE_STATS.get("bytes"):
            raise
        clear_expert_cache(oom = True)
    torch.cuda.empty_cache()
    return fn()
