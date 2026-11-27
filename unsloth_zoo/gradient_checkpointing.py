# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
import numpy as np
from typing import Union, Optional, List, Any, Callable, Tuple
from contextlib import contextmanager, nullcontext
import os
import functools
import inspect
import sys
import warnings
import gc
import collections
import heapq
import threading
import weakref
from .utils import _get_dtype, Version, is_distributed
from .saved_attention import (
    new_saved_attention_frame,
    saved_attention_scope,
    save_attention_requested,
    patch_transformers_saved_attention,
)
# Re-exported; lives in integrated_device.py so the import-time allocator block avoids torch.
from .integrated_device import (
    KEEP_ON_DEVICE_ENV,
    _any_device_integrated,
    _env_tristate,
    unified_memory_device_indices,
)
from .device_type import (
    is_hip,
    get_device_type,
    DEVICE_TYPE,
    DEVICE_TYPE_TORCH,
    DEVICE_COUNT,
    ALLOW_PREQUANTIZED_MODELS,
)

__all__ = [
    "calculate_n_gradient_checkpoints",
    "prepare_n_gradient_checkpoints",
    "Unsloth_Offloaded_Gradient_Checkpointer",
    "unsloth_offloaded_gradient_checkpoint",
    "patch_unsloth_gradient_checkpointing",
    "unpatch_unsloth_gradient_checkpointing",

    "Unsloth_Gradient_Checkpointer",
    "unsloth_gradient_checkpoint",
    "patch_gradient_checkpointing",
    "unpatch_gradient_checkpointing",

    "patch_unsloth_smart_gradient_checkpointing",
    "unpatch_unsloth_smart_gradient_checkpointing",
    "reset_unsloth_gradient_checkpointing_buffers",
    "NonReentrantActivationOffloader",
    "non_reentrant_offload_hooks",
    "activation_offloader_for",
    "get_activation_offload_stats",
]

INITIAL_CPU_BUFFER_SIZE = 128 * 1024       # per CPU buffer
INITIAL_GPU_BUFFER_SIZE = 2 * 256 * 2048   # per GPU buffer
INITIAL_CPU_BUFFER_COUNT = 200             # number of CPU buffers
DOUBLE_BUFFER_HEADROOM = 512 * 1024 * 1024 # min free CUDA memory to enable double buffering


@functools.cache
def _double_buffer_disabled():
    # Cached at first GC init, never at import (no early GPU probe); off on integrated devices (no
    # H2D copy to hide, ~2x slower, no memory saved); UNSLOTH_DISABLE_DOUBLE_BUFFER=0/1 forces it.
    env = os.environ.get("UNSLOTH_DISABLE_DOUBLE_BUFFER")
    if env is not None: return env == "1"
    if DEVICE_TYPE not in ("cuda", "hip"): return False
    return _any_device_integrated() or bool(_keep_on_device_indices())


@functools.cache
def _keep_on_device_indices():
    # Unified-memory devices (AMD gfx1150/1151, NVIDIA GB10) keep inputs on device since offloading
    # saves nothing; cached at first GC init after device selection, never at import;
    # UNSLOTH_GC_KEEP_ON_DEVICE=1 keeps all, 0 offloads all.
    if DEVICE_TYPE not in ("cuda", "hip"): return frozenset()
    return unified_memory_device_indices()


_WARNED_KEEP_ON_DEVICE = False
_WARNED_KEEP_LOW_MEMORY = False
# Free memory below which unified devices offload again; their budget may be under host RAM.
KEEP_ON_DEVICE_HEADROOM_MIN  = 2 * 1024 * 1024 * 1024
KEEP_ON_DEVICE_HEADROOM_FRAC = 0.05
# _KEEP_BUDGET tracks free as last mem_get_info minus later allocs; re-reads see outside use.
_KEEP_BUDGET = {}
_KEEP_BUDGET_REFRESH = 256


def _device_free_bytes(device_index):
    free, total = torch.cuda.mem_get_info(device_index)
    # Blocks torch has cached but not handed out are free for the next activation too.
    cached = torch.cuda.memory_reserved(device_index) - torch.cuda.memory_allocated(device_index)
    return int(free) + max(int(cached), 0), int(total)


def _keeps_on_device(device_index, nbytes = 0):
    global _WARNED_KEEP_ON_DEVICE, _WARNED_KEEP_LOW_MEMORY
    keep = _keep_on_device_indices()
    if not keep or device_index not in keep: return False
    try:
        allocated = int(torch.cuda.memory_allocated(device_index))
    except Exception:
        return False # cannot tell: offload, as before
    budget = _KEEP_BUDGET.get(device_index)
    projected = None if budget is None else budget[0] - max(allocated - budget[1], 0)
    if projected is None or projected - nbytes < 2 * budget[2] or budget[3] >= _KEEP_BUDGET_REFRESH:
        try:
            free, total = _device_free_bytes(device_index)
        except Exception:
            return False # cannot tell: offload, as before
        headroom = max(KEEP_ON_DEVICE_HEADROOM_MIN, int(total * KEEP_ON_DEVICE_HEADROOM_FRAC))
        budget = _KEEP_BUDGET[device_index] = [free, allocated, headroom, 0]
        projected = free
    budget[3] += 1
    if projected - nbytes < budget[2]:
        if not _WARNED_KEEP_LOW_MEMORY:
            _WARNED_KEEP_LOW_MEMORY = True
            print("Unsloth: GPU memory is nearly full, so gradient checkpointing offloads activations to RAM again.")
        return False
    if not _WARNED_KEEP_ON_DEVICE:
        _WARNED_KEEP_ON_DEVICE = True
        if _keep_on_device_forced():
            # A discrete GPU can be selected too: do not claim it shares memory with the CPU.
            print(f"Unsloth: {KEEP_ON_DEVICE_ENV}=1, so gradient checkpointing keeps activations on the device.")
        else:
            print("Unsloth: GPU shares memory with the CPU, so gradient checkpointing keeps activations on the device.")
    return True


def _keep_on_device_forced():
    return _env_tristate(KEEP_ON_DEVICE_ENV) is True


# WSL2 caps pinned memory near 1-2GB, else pageable; unslothai/unsloth 338, 1552, 1744, 1797.
PINNED_MEMORY_AVAILABLE = True
_WARNED_ABOUT_PINNED_MEMORY = False


def _pinned_memory_disabled():
    return os.environ.get("UNSLOTH_DISABLE_PINNED_MEMORY", "0") == "1"


def _is_host_alloc_oom(exception):
    # torch >= 2.9 raises AcceleratorError, older torch RuntimeError; both subclass RuntimeError.
    text = str(exception).lower()
    return ("out of memory" in text) or ("cudaerrormemoryallocation" in text)


def _warn_pinned_unavailable(detail):
    global PINNED_MEMORY_AVAILABLE, _WARNED_ABOUT_PINNED_MEMORY
    PINNED_MEMORY_AVAILABLE = False
    if _WARNED_ABOUT_PINNED_MEMORY: return
    _WARNED_ABOUT_PINNED_MEMORY = True
    print(
        "Unsloth: Could not allocate pinned (page-locked) host memory for gradient "
        "checkpointing offload, so pageable host memory will be used instead. "
        "Training continues and results are unchanged, but the offload copies no "
        "longer overlap with compute, so each step is somewhat slower.\n"
        f"Unsloth: The allocator reported: {detail}\n"
        "Unsloth: This is normal under WSL2, which caps pinned memory per process. "
        "Set UNSLOTH_DISABLE_PINNED_MEMORY=1 to skip pinning from the start."
    )


# Pinned flag lives on the buffer (is_pinned() is a driver query): a drifted parallel list would
# mark pageable buffers pinned, so async copies read pageable memory; unmarked costs a sync copy.
HOST_PINNED_ATTR = "_unsloth_is_pinned"


def _mark_host_buffer(buffer):
    """Record whether `buffer` is page-locked. Call wherever a host buffer is made or regrown."""
    try:
        pinned = buffer.is_pinned()
    except Exception:
        pinned = False
    try:
        setattr(buffer, HOST_PINNED_ATTR, pinned)
    except AttributeError:
        pass  # exotic tensor subclass; readers fall back to False
    return buffer


def _new_host_buffer(numel, dtype):
    nbytes = numel * dtype.itemsize
    if PINNED_MEMORY_AVAILABLE and not _pinned_memory_disabled():
        with _PINNED_LOCK:  # held through allocation: no double-spent headroom
            if _reserve_pinned(nbytes):
                try:
                    return _mark_host_buffer(torch.empty(numel, dtype = dtype, device = "cpu", pin_memory = True))
                except RuntimeError as e:
                    _release_pinned(nbytes)
                    if not _is_host_alloc_oom(e): raise
                    _warn_pinned_unavailable(e)
    return _mark_host_buffer(torch.empty(numel, dtype = dtype, device = "cpu"))


def _grow_host_buffer(buffer, new_size):
    """resize_ on a pinned buffer issues a fresh, larger cudaHostAlloc, which can fail."""
    if new_size <= buffer.numel(): return _mark_host_buffer(buffer)
    new_size = _exact_growth_numel(new_size, buffer.element_size())
    old_bytes = _storage_bytes(buffer)
    if new_size * buffer.element_size() <= old_bytes:
        # A shrunk buffer keeps its storage, so this resize_ allocates nothing.
        buffer.resize_(new_size)
        return _mark_host_buffer(buffer)
    new_bytes = max(old_bytes, new_size * buffer.element_size())
    pinned = getattr(buffer, HOST_PINNED_ATTR, False)
    if not pinned:
        # Pageable slots take a fresh block when rewritten, so they re-pin once RAM allows.
        return _new_host_buffer(new_size, buffer.dtype)
    # resize_ takes a new pinned block and hands the old one back to the host cache.
    with _PINNED_LOCK:  # held through allocation: no double-spent headroom
        if pinned and not _reserve_pinned(new_bytes):
            _release_pinned(old_bytes)
            replacement = _mark_host_buffer(torch.empty(new_size, dtype = buffer.dtype, device = "cpu"))
            if GC_PINNED_CAP == _PIN_PER_BLOCK:
                # Free the old block rather than leave it cached beside the pageable copy.
                try:
                    buffer.resize_(0)
                    buffer.untyped_storage().resize_(0)  # Tensor.resize_ alone never shrinks a storage
                except Exception: pass
                else: _empty_host_cache()
            return replacement
        try:
            buffer.resize_(new_size)
            if pinned: _release_pinned(old_bytes)
            _note_outgrown_block(old_bytes)
            return _mark_host_buffer(buffer)
        except RuntimeError as e:
            if pinned: _release_pinned(new_bytes)
            if not _is_host_alloc_oom(e) or not buffer.is_pinned(): raise
            _release_pinned(old_bytes)
            _warn_pinned_unavailable(e)
            return _mark_host_buffer(torch.empty(new_size, dtype = buffer.dtype, device = "cpu"))


# Host allocator rounds pinned blocks to powers of two and keeps freed ones pinned until
# empty_host_cache (torch 2.6-2.15; 2.13+ settings turn both off above a size), so GC_PINNED_BYTES
# charges rounded blocks; a release moves one to _PINNED_CACHED until host_memory_stats shows less
# held.
GC_PINNED_BYTES = 0
GC_PINNED_CAP = None        # None = not computed yet, -1 = uncapped, -2 = per block (one rank)
_PIN_PER_BLOCK = -2
_PIN_HEADROOM_BYTES = 1 << 30
_WARNED_ABOUT_PINNED_CAP = False
_PINNED_CACHED = {}         # rounded block bytes -> count released but still cached
_PINNED_LIVE = {}           # requested bytes -> blocks charged at allocation, still live
_HOST_HELD_BASELINE = None  # allocator bytes not ours when the cap was computed
_PINNED_RELEASED = 0        # settings mode: blocks released since the last flush (may wait on events)


# torch 2.13+: blocks above the round threshold are exact; torch 2.12 ignores these keys.
_HOST_ALLOC_POLICY = (None, None, None)
_TORCH_MINOR = Version(torch.__version__).release[:2]
_HOST_ALLOC_KNOBS = _TORCH_MINOR >= (2, 13)
_INF = float("inf")


def _env_allocator_settings():
    # The variable torch parsed at startup: the first one set, in c10/core/AllocatorConfig.cpp's order.
    for var in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        value = os.environ.get(var)
        if value is not None: return value
    return ""


def _host_allocator_settings():
    getter = getattr(torch._C, "_accelerator_getAllocatorSettings", None)
    if getter is not None:
        try: return getter() or ""
        except Exception: pass
    return _env_allocator_settings()


def _parse_host_alloc_policy(settings, prior = (_INF, _INF)):
    round_max, cached_max = prior
    for item in settings.split(","):
        key, _, value = item.partition(":")
        key = key.strip()
        if key not in ("pinned_max_round_threshold_mb", "pinned_max_cached_size_mb"): continue
        try: limit = int(float(value.strip()) * (1 << 20))
        except ValueError: continue
        if key == "pinned_max_round_threshold_mb": round_max = limit
        else: cached_max = limit
    return round_max, cached_max


def _host_alloc_policy():
    """(round threshold, max cached size) in bytes; inf when unset or unsupported by this torch."""
    global _HOST_ALLOC_POLICY
    if not _HOST_ALLOC_KNOBS: return _INF, _INF
    settings = _host_allocator_settings()
    policy = _HOST_ALLOC_POLICY
    if settings != policy[0]:
        # Fold into the policy in effect: torch reports only the last, so omitted keys persist.
        prior = policy[1:] if policy[0] is not None else _parse_host_alloc_policy(_env_allocator_settings())
        policy = _HOST_ALLOC_POLICY = (settings, *_parse_host_alloc_policy(settings, prior))
        if policy[1] != _INF: _warn_short_block_reuse()
    return policy[1], policy[2]


_WARNED_SHORT_BLOCK = False
def _warn_short_block_reuse():
    # torch 2.13 get_free_block can hand out a shorter unrounded block (fixed in 2.14).
    global _WARNED_SHORT_BLOCK
    if _WARNED_SHORT_BLOCK or _TORCH_MINOR >= (2, 14): return
    _WARNED_SHORT_BLOCK = True
    warnings.warn(
        "Unsloth: pinned_max_round_threshold_mb is set on torch 2.13, whose pinned host cache can "
        "reuse a smaller block for a larger request above that threshold (fixed in torch 2.14). "
        "Gradient checkpointing offload grows pinned buffers, so upgrade torch or unset it.",
        stacklevel = 2,
    )


# Slot = hidden*tokens*2 bytes, often not a power of two (Qwen3-4B at 32k: 160MB pins as 256MB);
# torch 2.14+ CUDA sets pinned_max_round_threshold_mb so blocks above _GC_PINNED_EXACT_MB pin exact
# (2.13 can hand out short blocks), unless user set a pinned knob; UNSLOTH_GC_PINNED_EXACT=0 off.
_GC_PINNED_EXACT_MB = 4
_GC_PINNED_EXACT = False
_PINNED_KNOBS = ("pinned_max_round_threshold_mb", "pinned_max_cached_size_mb")


def _maybe_enable_exact_pinned():
    global _GC_PINNED_EXACT
    if _GC_PINNED_EXACT: return True
    if os.environ.get("UNSLOTH_GC_PINNED_EXACT", "1") == "0": return False
    if DEVICE_TYPE != "cuda" or getattr(torch.version, "hip", None) or _TORCH_MINOR < (2, 14): return False
    setter = getattr(torch._C, "_accelerator_setAllocatorSettings", None)
    if setter is None: return False
    current = _host_allocator_settings()
    user = [current] + [os.environ.get(var) or "" for var in
        ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF")]
    if any(knob in text for text in user for knob in _PINNED_KNOBS): return False
    # torch reports only the latest settings string: a pinned knob set by an earlier update stays in effect.
    if any(v != _INF for v in _host_alloc_policy()): return False
    # Each call resets max_split_size_mb, roundup_power2_divisions and garbage_collection_threshold
    # to defaults, so pass the settings in effect back in with ours appended.
    try: setter(",".join(filter(None, (current, f"pinned_max_round_threshold_mb:{_GC_PINNED_EXACT_MB}"))))
    except Exception: return False
    _GC_PINNED_EXACT = True
    return True


def _exact_growth_numel(numel, itemsize):
    """Exact blocks are only reused by requests no larger, so a slot that grows with the sequence length
    would pin a fresh block each time. Grow in 1/8 power-of-two steps: few growths, under 12.5% slack."""
    nbytes = numel * itemsize
    if not _GC_PINNED_EXACT or nbytes <= (_GC_PINNED_EXACT_MB << 20): return numel
    step = 1 << (nbytes.bit_length() - 4)
    return -(-(-(-nbytes // step) * step) // itemsize)


_EXACT_OUTGROWN = False


def _note_outgrown_block(old_bytes):
    """An outgrown exact block fits none of the larger requests that follow, so it would stay pinned in
    torch's cache. Free it now if its copies are done (no sync: last used in an earlier step), else at the
    next step's first offload, after a sync."""
    global _EXACT_OUTGROWN
    if not _GC_PINNED_EXACT or old_bytes <= (_GC_PINNED_EXACT_MB << 20): return
    _EXACT_OUTGROWN = True
    try: torch._C._host_emptyCache()
    except Exception: pass


def _flush_outgrown_blocks():
    global _EXACT_OUTGROWN
    if not _EXACT_OUTGROWN: return
    _EXACT_OUTGROWN = False
    with _PINNED_LOCK: _empty_host_cache()


def _pinned_block_bytes(nbytes):
    """Bytes torch's host allocator really pins for an `nbytes` request."""
    if nbytes <= 0: return 0
    round_max, cached_max = _host_alloc_policy()
    if nbytes > round_max or nbytes > cached_max: return int(nbytes)
    return 1 << (int(nbytes) - 1).bit_length()


def _pinned_block_cached(block):
    """False when torch frees a released block of this size instead of caching it."""
    return block <= _host_alloc_policy()[1]


def _host_allocator_held_bytes():
    """Pinned bytes the caching host allocator holds (live + cached), None if unknown."""
    if DEVICE_TYPE not in ("cuda", "hip"): return None
    try: return int(torch.cuda.host_memory_stats()["allocated_bytes.current"])
    except Exception: return None  # torch < 2.7 has no host_memory_stats


def _host_allocator_free_bytes():
    """Pinned bytes cached free in the host allocator (any user's), None if unknown."""
    if DEVICE_TYPE not in ("cuda", "hip"): return None
    try:
        stats = torch.cuda.host_memory_stats()
        return int(stats["allocated_bytes.current"]) - int(stats["active_bytes.current"])
    except Exception: return None


def _host_cache_reuses_block(block):
    """A pinned block freed after a non_blocking copy waits in torch's event queue, which only an
    allocation drains (host_memory_stats does not), so it reads as not free. Allocate and drop a
    block of that size to drain it: True if the allocator reused a cached block, False if it had
    to pin a new one, which is released again so the probe never holds pinned memory past the cap."""
    held = _host_allocator_held_bytes()
    if held is None: return False
    try:
        probe = torch.empty(block, dtype = torch.uint8, device = "cpu", pin_memory = True)
    except Exception as exception:
        if not _is_host_alloc_oom(exception): raise
        return False
    after = _host_allocator_held_bytes()
    del probe
    if after is not None and after <= held: return True
    _empty_host_cache()
    return False


def _compute_pinned_cap(held = None):
    """`held`: counted mode, the allocator's pinned bytes now (everyone's), which the cap then bounds."""
    env = os.environ.get("UNSLOTH_GC_PINNED_MAX_GB")
    if env:
        try: return max(0, int(float(env) * (1 << 30)))
        except ValueError: pass
    try:
        from .block_swap import _PIN_RESERVE_MIN_BYTES, _PIN_RESERVE_FRACTION, _local_world_size
        from .memory_budget import host_memory_bytes
        total, available = host_memory_bytes()
    except Exception:
        return -1  # unmeasurable: keep the old uncapped behaviour
    if _local_world_size() == 1: return _PIN_PER_BLOCK
    reserve = max(_PIN_RESERVE_MIN_BYTES, int(total * _PIN_RESERVE_FRACTION))
    # Free RAM already excludes pinned bytes, so add back what is held (ours, or all in counted mode).
    return max(0, available - reserve) // _local_world_size() + (GC_PINNED_BYTES if held is None else held)


def _block_fits_host_ram(block):
    """One local rank: pin unless the block would leave under min(1 GiB, 10% of RAM) available."""
    try:
        from .memory_budget import host_memory_bytes
        total, available = host_memory_bytes()
    except Exception:
        return True
    return block + min(_PIN_HEADROOM_BYTES, total // 10) <= available


def _synchronize_used_devices():
    """Wait for this process's work on every device it already uses (a split model offloads on
    devices other than the current one). Devices it never touched are skipped, so no context is
    created on them. A failure here is a real device error and is raised."""
    if not torch.cuda.is_available(): return
    for index in range(torch.cuda.device_count()):
        if torch.cuda.memory_reserved(index) > 0:
            torch.cuda.synchronize(index)
pass


def _empty_host_cache():
    """Return cached free pinned blocks to the OS. A pinned buffer that grows leaves its old block
    in torch's host cache, and once growth goes pageable nothing reuses it. True if any were freed."""
    global GC_PINNED_BYTES, _PINNED_RELEASED
    if DEVICE_TYPE not in ("cuda", "hip"): return False
    ours = sum(k * v for k, v in _PINNED_CACHED.items())
    free = _host_allocator_free_bytes()
    # host_memory_stats() misses pending non_blocking copies; released blocks justify a flush.
    released = _PINNED_RELEASED
    if not (ours or released or free): return False
    if ours or released:
        # Sync released copies first: _host_emptyCache frees only blocks with completed copies.
        _synchronize_used_devices()
    try: torch._C._host_emptyCache()
    except Exception: return False
    GC_PINNED_BYTES = max(0, GC_PINNED_BYTES - ours)
    _PINNED_CACHED.clear()
    _PINNED_RELEASED = 0
    return True


def _storage_bytes(buffer):
    # resize_ never shrinks a storage, so numel can understate what stays pinned.
    try: return buffer.untyped_storage().nbytes()
    except Exception: return buffer.numel() * buffer.element_size()


def _release_pool_pinned(start = 0):
    """Release every pinned CPU_BUFFERS slot from `start` on; call before freeing them."""
    if not CPU_BUFFERS: return
    for x in CPU_BUFFERS[start:]:
        if x is not None and getattr(x, HOST_PINNED_ATTR, False):
            _release_pinned(_storage_bytes(x))


def _reset_pinned_cap():
    global GC_PINNED_CAP, _WARNED_ABOUT_PINNED_CAP
    GC_PINNED_CAP = None
    _WARNED_ABOUT_PINNED_CAP = False


def _reserve_pinned_unlocked(nbytes):
    """Count `nbytes` against the pinned cap; False means allocate pageable instead."""
    global GC_PINNED_BYTES, GC_PINNED_CAP, _WARNED_ABOUT_PINNED_CAP, _HOST_HELD_BASELINE
    block = _pinned_block_bytes(nbytes)
    if _host_alloc_settings_active():
        counted = _reserve_pinned_counted(nbytes, block)
        if counted is not None: return counted
    if _PINNED_CACHED.get(block, 0) > 0:
        _PINNED_CACHED[block] -= 1
        free = _host_allocator_free_bytes()
        # Fixed cap: refusal is sticky, so check the event queue; per-block re-checks host RAM.
        if free is None or free >= block or (
            GC_PINNED_CAP is not None and GC_PINNED_CAP >= 0 and _host_cache_reuses_block(block)
        ):
            _charge_live(nbytes, block)
            return True  # the allocator hands back the cached block, already charged
        # Another pinned user took the block: it is no longer ours, and this request is new.
        GC_PINNED_BYTES = max(0, GC_PINNED_BYTES - block)
    held = _host_allocator_held_bytes()
    if GC_PINNED_CAP is None:
        GC_PINNED_CAP = _compute_pinned_cap()
        _HOST_HELD_BASELINE = None if held is None else max(0, held - GC_PINNED_BYTES)
    elif _HOST_HELD_BASELINE is None and held is not None:
        # host_memory_stats() is empty before CUDA init: take the baseline at the first reading.
        _HOST_HELD_BASELINE = max(0, held - GC_PINNED_BYTES)
    used = GC_PINNED_BYTES
    if held is not None:
        if held < GC_PINNED_BYTES:
            # The host cache was emptied: whatever we released is gone.
            live = GC_PINNED_BYTES - sum(k * v for k, v in _PINNED_CACHED.items())
            _PINNED_CACHED.clear()
            GC_PINNED_BYTES = used = max(live, held)
        if _HOST_HELD_BASELINE is not None:
            # A cached block taken by another pinned user makes our next request a new one.
            used = max(used, held - _HOST_HELD_BASELINE)
    if GC_PINNED_CAP == _PIN_PER_BLOCK:
        if not _block_fits_host_ram(block) and not (_empty_host_cache() and _block_fits_host_ram(block)):
            _warn_host_ram_full()
            return False
    elif GC_PINNED_CAP >= 0 and used + block > GC_PINNED_CAP and not (
        _PINNED_CACHED and any(_PINNED_CACHED.values()) and _flushed_cached_fits(block, used)
    ):
        _warn_pinned_cap()
        return False
    GC_PINNED_BYTES += block
    _charge_live(nbytes, block)
    return True


def _warn_host_ram_full():
    global _WARNED_ABOUT_PINNED_CAP
    if _WARNED_ABOUT_PINNED_CAP: return
    _WARNED_ABOUT_PINNED_CAP = True
    print(
        "Unsloth: Host RAM is nearly full, so gradient checkpointing offload buffers that "
        "do not fit use pageable memory. Training continues; "
        "set UNSLOTH_GC_PINNED_MAX_GB to set a fixed pinned cap instead."
    )


def _warn_pinned_cap():
    global _WARNED_ABOUT_PINNED_CAP
    if _WARNED_ABOUT_PINNED_CAP: return
    _WARNED_ABOUT_PINNED_CAP = True
    print(
        f"Unsloth: Gradient checkpointing pinned host memory reached its cap of "
        f"{GC_PINNED_CAP / (1 << 30):.1f}GB (free host RAM split across local ranks), "
        "so further offload buffers use pageable memory. Training continues; "
        "set UNSLOTH_GC_PINNED_MAX_GB to change the cap."
    )


# With torch 2.13+ settings the allocator count (live + cached + event-pending) is the ground truth.
def _host_alloc_settings_active():
    if not _HOST_ALLOC_KNOBS: return False
    round_max, cached_max = _host_alloc_policy()
    return round_max != _INF or cached_max != _INF


def _reserve_pinned_counted(nbytes, block):
    """None if the allocator cannot be read (the simulated path then decides)."""
    global GC_PINNED_BYTES, GC_PINNED_CAP
    # Re-read every time: other users pin between ours; the cap bounds all held bytes.
    held = _host_allocator_held_bytes()
    if held is None: return None
    if GC_PINNED_CAP is None:
        GC_PINNED_CAP = _compute_pinned_cap(held)
    if GC_PINNED_CAP == _PIN_PER_BLOCK:
        if not _block_fits_host_ram(block) and not (_empty_host_cache() and _block_fits_host_ram(block)):
            _warn_host_ram_full()
            return False
    elif GC_PINNED_CAP >= 0 and held + block > GC_PINNED_CAP:
        if not _counted_slow_path_fits(block):
            _warn_pinned_cap()
            return False
    GC_PINNED_BYTES += block
    _charge_live(nbytes, -block)
    return True


def _counted_slow_path_fits(block):
    """Over the cap: process pending copy events, then flush the cache and re-read. Never trial-allocate
    `block`: an uncached one would itself pin past the cap."""
    free = _host_allocator_free_bytes() or 0
    if not (_PINNED_RELEASED or free > 0):
        return False
    # A tiny pinned allocation processes completed copy events (host_memory_stats does not).
    try: drain = torch.empty(1, dtype = torch.uint8, device = "cpu", pin_memory = True)
    except Exception as exception:
        if not _is_host_alloc_oom(exception): raise
    else: del drain
    held = _host_allocator_held_bytes()
    if held is not None and held + block <= GC_PINNED_CAP: return True
    if not _empty_host_cache(): return False
    held = _host_allocator_held_bytes()
    return held is not None and held + block <= GC_PINNED_CAP


def _charge_live(nbytes, block):
    blocks = _PINNED_LIVE.get(nbytes)
    if blocks is None: _PINNED_LIVE[nbytes] = [block]
    else: blocks.append(block)


def _flushed_cached_fits(block, used):
    """Cap mode: flush our cached released blocks; True if `block` then fits under the cap."""
    ours = sum(k * v for k, v in _PINNED_CACHED.items())
    if not _empty_host_cache(): return False
    held = _host_allocator_held_bytes()
    if held is None or _HOST_HELD_BASELINE is None:
        return max(0, used - ours) + block <= GC_PINNED_CAP
    return max(GC_PINNED_BYTES, held - _HOST_HELD_BASELINE) + block <= GC_PINNED_CAP


def _release_pinned_unlocked(nbytes):
    """The block stays pinned in torch's host cache, so it stays charged until reused; one above
    pinned_max_cached_size_mb (torch 2.13+) goes back to the OS, so it stops being charged."""
    global GC_PINNED_BYTES, _PINNED_RELEASED
    blocks = _PINNED_LIVE.get(nbytes)
    if blocks:
        block = blocks.pop()
        if not blocks: del _PINNED_LIVE[nbytes]
    else:
        block = _pinned_block_bytes(nbytes)
    if not block: return
    if block < 0:
        GC_PINNED_BYTES = max(0, GC_PINNED_BYTES + block)
        _PINNED_RELEASED += 1
        return
    if _pinned_block_cached(block):
        _PINNED_CACHED[block] = _PINNED_CACHED.get(block, 0) + 1
    else:
        GC_PINNED_BYTES = max(0, GC_PINNED_BYTES - block)



# Autograd threads also release; spanning cap check and allocation stops double-spent headroom.
_PINNED_LOCK = threading.RLock()


def _reserve_pinned(nbytes):
    """Count `nbytes` against the pinned cap; False means allocate pageable instead."""
    with _PINNED_LOCK:
        return _reserve_pinned_unlocked(nbytes)


def _release_pinned(nbytes):
    with _PINNED_LOCK:
        return _release_pinned_unlocked(nbytes)

def _view_bytes_as(buffer, nbytes, dtype, shape):
    """Reinterpret raw buffer bytes, never cast: the init dtype can differ from the activation's."""
    if buffer.dtype == dtype: return buffer[:nbytes // dtype.itemsize].view(shape)
    return buffer.view(torch.uint8)[:nbytes].view(dtype).view(shape)


# UNSLOTH_OFFLOAD_FP32=exact|bf16 forces; auto stores autocast fp32 offloads as bf16 (fp16 overflows
# gemma3 residuals, spiked grad norms 20x on T4; bf16 drifts LoRA grads ~0.1%), else exact.
def _fp32_offload_mode():
    mode = os.environ.get("UNSLOTH_OFFLOAD_FP32", "auto").strip().lower()
    if mode in ("exact", "fp32", "float32"): return "exact"
    if mode in ("bf16", "bfloat16"): return "bf16"
    return "auto"
FP32_OFFLOAD_MODE = _fp32_offload_mode()
FP32_OFFLOAD_EXACT = FP32_OFFLOAD_MODE == "exact"


def _autocast_enabled(device_type):
    try:
        return torch.is_autocast_enabled(device_type)
    except TypeError:  # torch < 2.4 has no device argument
        return torch.is_autocast_cpu_enabled() if device_type == "cpu" else torch.is_autocast_enabled()


def _offload_store_dtype(tensor):
    """Dtype an offloaded activation is stored in; the recompute always gets tensor.dtype back."""
    dtype = tensor.dtype
    if dtype != torch.float32 or FP32_OFFLOAD_EXACT: return dtype
    if FP32_OFFLOAD_MODE == "bf16" or _autocast_enabled(tensor.device.type): return torch.bfloat16
    return dtype


@contextmanager
def _no_inference_mode():
    # Create GC buffers outside inference_mode, or a later offload copy_ fails (unsloth#3828).
    try:
        leave_inference = torch.inference_mode(False)
    except (TypeError, AttributeError):
        leave_inference = nullcontext()  # older torch lacks inference_mode(bool)
    with leave_inference, torch.no_grad():
        yield

torch_version = torch.__version__
if Version(torch_version) < Version("2.4.0"):
    torch_amp_custom_fwd = torch.cuda.amp.custom_fwd
    torch_amp_custom_bwd = torch.cuda.amp.custom_bwd
else:
    _amp_device_type = DEVICE_TYPE_TORCH if DEVICE_TYPE in ("xpu", "npu") else "cuda"
    torch_amp_custom_fwd = torch.amp.custom_fwd(device_type = _amp_device_type)
    torch_amp_custom_bwd = torch.amp.custom_bwd(device_type = _amp_device_type)
pass


def _calculate_n_gradient_checkpoints(
    n_layers : int,
    method   : Optional[Union[str, int]] = "sqrt",
) -> List[int]:
    assert(type(n_layers) is int and n_layers > 0)

    if method is None: method = "sqrt"

    if method == "sqrt":
        n_checkpoints = int(n_layers**0.5)
    elif type(method) is int and method > 0:
        n_checkpoints = int(np.ceil(n_layers / method))
    else:
        raise ValueError("method must be 'sqrt' or an int >0 and <= n_layers.")

    size = n_layers // n_checkpoints
    sizes = np.full(n_checkpoints, size, dtype = int)
    leftovers = n_layers % n_checkpoints
    for k in range(leftovers):
        sizes[n_checkpoints-1-k] += 1
    boundaries = np.hstack((0, np.cumsum(sizes)))
    boundaries = boundaries.tolist()
    return boundaries
pass


def calculate_n_gradient_checkpoints(
    n_layers              : int,
    layers_per_checkpoint : Optional[Union[str, int]] = "sqrt",
) -> List[int]:
    assert(type(n_layers) is int and n_layers > 0)

    if layers_per_checkpoint is None or layers_per_checkpoint == 1:
        return None

    boundaries = _calculate_n_gradient_checkpoints(n_layers, layers_per_checkpoint)

    assert(boundaries[0] == 0 and boundaries[-1] == n_layers)
    assert(min(boundaries) == 0 and max(boundaries) == n_layers)
    assert(np.diff(boundaries).min() >= 0)
    return boundaries
pass


def prepare_n_gradient_checkpoints(
    model                 : Any,
    layers_per_checkpoint : Optional[Union[str, int]] = "sqrt",
    use_reentrant         : Optional[bool] = True,
) -> None:
    """
    Calculates where to place the gradient checkpoints given n_layers.

    Args:
        model: Any LlamaModel with layers.
        layers_per_checkpoint (`Union[str, int]`, *optional*):
            Can either be `sqrt` or an integer for how many layers per checkpoint you want.
            The more, the less memory usage, but can be slower. Default is `sqrt`.
            Choose 1 for Pytorch gradient checkpointing. 2 to wrap 2 layers in 1 module etc.
        use_reentrant (`bool`, *optional*):
            https://github.com/pytorch/pytorch/blob/main/torch/utils/checkpoint.py#L354
            Optimal gradient checkpointing algorithm `use_reentrant=False` which will
            be the default in future Pytorch versions doesn't seem to work??
    """
    _model = None
    if hasattr(model, "layers"):
        _model = model
    elif hasattr(model, "model"):
        if hasattr(model.model, "layers"):
            _model = model.model
    if _model is None:
        raise TypeError("`model` or `model.model` does not have attribute `layers`. Are you sure this is a model?")
    pass

    if use_reentrant is False:
        use_reentrant = True
    pass

    n_layers = len(_model.layers)
    boundaries = calculate_n_gradient_checkpoints(n_layers, layers_per_checkpoint)
    _model._gradient_checkpointing_boundaries    = boundaries
    _model._gradient_checkpointing_use_reentrant = use_reentrant
pass


# ---------------------------------------------------------------------------
# Gradient-checkpoint recompute marker
# ---------------------------------------------------------------------------
# Thread-local, True only while checkpoint backward re-runs forward (MoE recompute policy reads it);
# context manager always restores the prior value, is idempotent, safe under nesting and exceptions,
# and never alters checkpointing.
_GC_RECOMPUTE_TLS = threading.local()


def in_gradient_checkpoint_recompute() -> bool:
    """True while a gradient-checkpoint backward is recomputing its forward."""
    return getattr(_GC_RECOMPUTE_TLS, "active", False)


@contextmanager
def _gradient_checkpoint_recompute_marker(function = None):
    previous = getattr(_GC_RECOMPUTE_TLS, "active", False)
    previous_function = getattr(_GC_RECOMPUTE_TLS, "function", None)
    _GC_RECOMPUTE_TLS.active = True
    _GC_RECOMPUTE_TLS.function = function
    try:
        yield
    finally:
        _GC_RECOMPUTE_TLS.active = previous
        _GC_RECOMPUTE_TLS.function = previous_function


def _module_forward_chain(module):
    function = getattr(type(module), "forward", None)
    for _ in range(8):
        if function is None:
            return
        yield function
        function = getattr(function, "__wrapped__", None)
pass


def _calls_whole_module(function, module):
    if not isinstance(module, torch.nn.Module):
        return False
    if function in (torch.nn.Module.__call__, torch.nn.Module._call_impl):
        return True
    return any(function is forward for forward in _module_forward_chain(module))
pass


def _replayed_module(function):
    """Module whose call is the whole checkpointed function, else None. Any shape that could add work
    after the module returns (lambdas, other methods) must resolve to None."""
    # All Unsloth Zoo code licensed under LGPLv3
    for _ in range(8):
        if isinstance(function, _KeywordArgumentCall):
            function = function.function
            continue
        if isinstance(function, torch.nn.Module):
            # An overridden __call__ may transform forward's output.
            return function if type(function).__call__ in (torch.nn.Module.__call__, torch.nn.Module._call_impl) else None
        if isinstance(function, functools.partial):
            inner = function.func
            if inspect.ismethod(inner):
                owner = inner.__self__
                return owner if _calls_whole_module(inner.__func__, owner) else None
            owner = function.args[0] if function.args else None
            return owner if _calls_whole_module(inner, owner) else None
        if inspect.ismethod(function):
            owner = function.__self__
            return owner if _calls_whole_module(function.__func__, owner) else None
        module = getattr(function, "_unsloth_checkpoint_module", None)
        return module if isinstance(module, torch.nn.Module) else None
    return None
pass


def gradient_checkpoint_replay_module():
    """During a reentrant checkpoint replay, the module whose output nothing else reads; else None."""
    if not getattr(_GC_RECOMPUTE_TLS, "active", False):
        return None
    function = getattr(_GC_RECOMPUTE_TLS, "function", None)
    if function is None:
        return None
    module = _replayed_module(function)
    # Forward hooks may read the output.
    if module is None or module._forward_hooks or torch.nn.modules.module._global_forward_hooks:
        return None
    return module


# ---------------------------------------------------------------------------
# Same MoE routing in the checkpoint replay as in the forward that made the loss
# ---------------------------------------------------------------------------
# Replay reuses first-pass MoE routing (compile rounds differ by a bf16 ulp); read by eager or
# opaque ops, never traced code, so no graph guards; UNSLOTH_MOE_GC_SAME_ROUTING=0 disables.
_GC_ROUTING_TLS = threading.local()
MOE_GC_ROUTING_STATS = collections.Counter()   # stored / reused / mismatch / unmatched
GC_SELECTION_STATS = collections.Counter()     # stored / reused / mismatch / unmatched


class CheckpointRoutingFrame:
    """First-pass routing in call order. Kept for the checkpoint context's lifetime and read through a cursor reset
    at each replay, so a second backward (retain_graph) replays the same routing."""
    __slots__ = (
        "entries", "cursor", "replaying", "current", "kept", "dequant_chunks", "selections", "selection_cursor",
        "__weakref__",
    )

    def __init__(self):
        self.entries = []
        self.cursor = 0
        self.replaying = False
        self.current = None   # entry index of the MoE call being routed now (moe_checkpoint_call)
        self.kept = {}        # first-pass frozen base expert outputs the replay reuses (moe_gc_keep_base)
        self.dequant_chunks = {}   # first-pass expert dequant chunk decisions the replay reuses (moe_utils)
        self.selections = []       # first-pass (key, value) of no-grad selection ops (checkpoint_replayed_selection)
        self.selection_cursor = 0


def _new_checkpoint_routing_frame(tensors):
    """A frame only when the checkpoint will run backward (some input requires grad)."""
    if os.environ.get("UNSLOTH_MOE_GC_SAME_ROUTING", "1") == "0":
        return None
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.requires_grad:
            return CheckpointRoutingFrame()
    return None


@contextmanager
def _checkpoint_routing_scope(frame, replay):
    previous = getattr(_GC_ROUTING_TLS, "frame", None)
    if frame is not None:
        frame.replaying = replay
        frame.cursor = 0
        frame.current = None
        frame.selection_cursor = 0
    _GC_ROUTING_TLS.frame = frame
    try:
        yield
    finally:
        _GC_ROUTING_TLS.frame = previous
        if frame is not None and replay and frame.cursor < len(frame.entries):
            MOE_GC_ROUTING_STATS["unmatched"] += len(frame.entries) - frame.cursor
        if frame is not None and replay and frame.selection_cursor < len(frame.selections):
            GC_SELECTION_STATS["unmatched"] += len(frame.selections) - frame.selection_cursor


def checkpoint_replayed_selection(key, compute):
    """`compute()` for a no-grad selection op inside a checkpointed call (a sparse-attention indexer's top-k): the
    first pass stores its result in the routing frame, the replay returns the stored one instead of running it again.

    The selection gets no gradient and depends only on the call's inputs, so the replay skips its cost and keeps the
    forward's picks (a recompute can flip ties). Entries match by call order and `key`; a mismatch runs `compute()`
    from there on. Eager only: traced code and calls outside a frame just run `compute()`.
    UNSLOTH_GC_REPLAY_SELECTION=0 disables."""
    frame = getattr(_GC_ROUTING_TLS, "frame", None)
    if frame is None or torch.compiler.is_compiling() or os.environ.get("UNSLOTH_GC_REPLAY_SELECTION", "1") == "0":
        return compute()
    if not frame.replaying:
        value = compute()
        frame.selections.append((key, value))
        GC_SELECTION_STATS["stored"] += 1
        return value
    cursor = frame.selection_cursor
    if cursor < len(frame.selections) and frame.selections[cursor][0] == key:
        frame.selection_cursor = cursor + 1
        GC_SELECTION_STATS["reused"] += 1
        return frame.selections[cursor][1]
    GC_SELECTION_STATS["mismatch"] += 1
    frame.selection_cursor = len(frame.selections)
    return compute()


@contextmanager
def suspended_checkpoint_routing():
    """No routing frame inside: for throwaway probe forwards that are not the call making the loss."""
    previous = getattr(_GC_ROUTING_TLS, "frame", None)
    _GC_ROUTING_TLS.frame = None
    try:
        yield
    finally:
        _GC_ROUTING_TLS.frame = previous


def _checkpoint_route_eager(top_k_index, top_k_weights, grad_mode):
    """(ids, weights) the experts must use: the first pass's in a replay, else the given ones."""
    frame = getattr(_GC_ROUTING_TLS, "frame", None)
    if frame is None:
        return None
    frame.current = None
    if not frame.replaying:
        # Only no-grad calls make the loss's forward; a grad-mode trial inside it does not.
        if not grad_mode:
            frame.entries.append((top_k_index.detach().clone(), top_k_weights.detach().clone()))
            frame.current = len(frame.entries) - 1
            MOE_GC_ROUTING_STATS["stored"] += 1
        return None
    if not grad_mode or frame.cursor >= len(frame.entries):
        return None
    index, weights = frame.entries[frame.cursor]
    frame.cursor += 1
    if (
        index.shape != top_k_index.shape or index.dtype != top_k_index.dtype or index.device != top_k_index.device
        or weights.shape != top_k_weights.shape or weights.dtype != top_k_weights.dtype
        or weights.device != top_k_weights.device
    ):
        # Not the same call (a layer that routes differently in its replay): keep the replay's routing.
        MOE_GC_ROUTING_STATS["mismatch"] += 1
        frame.cursor = len(frame.entries)
        return None
    frame.current = frame.cursor - 1
    MOE_GC_ROUTING_STATS["reused"] += 1
    return index, weights


def moe_checkpoint_call():
    """(frame, entry index, replaying) for the MoE experts call whose routing was just stored (first pass) or
    reused (replay), else None; consumed by the read, so a later call without routing gets None. Eager only."""
    frame = getattr(_GC_ROUTING_TLS, "frame", None)
    if frame is None:
        return None
    current = frame.current
    frame.current = None
    if current is None:
        return None
    return frame, current, frame.replaying


def _moe_checkpoint_route_impl(
    top_k_index: torch.Tensor, top_k_weights: torch.Tensor, grad_mode: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Resolved at call time: a reloaded copy of this module keeps the op registered by the first.
    found = sys.modules[__name__]._checkpoint_route_eager(top_k_index, top_k_weights, grad_mode)
    if found is None:
        return top_k_index.clone(), top_k_weights.clone()
    # Clone: Inductor may write later ops in place into this output, corrupting the stored routing.
    return found[0].clone(), found[1].clone()


def _moe_checkpoint_route_fake(top_k_index, top_k_weights, grad_mode):
    return torch.empty_like(top_k_index), torch.empty_like(top_k_weights)


def _register_moe_checkpoint_route_op():
    name = "unsloth_zoo::moe_checkpoint_route"
    existing = getattr(getattr(torch.ops, "unsloth_zoo", None), "moe_checkpoint_route", None)
    if existing is not None:   # a reloaded copy of this module
        return existing.default
    # Host-side state: a CUDA graph replay would skip the body, so keep the op out of CUDA graphs.
    kwargs = {}
    tag = getattr(getattr(torch, "Tag", None), "cudagraph_unsafe", None)
    if tag is not None and "tags" in inspect.signature(torch.library.custom_op).parameters:
        kwargs["tags"] = (tag,)
    op = torch.library.custom_op(name, mutates_args = (), **kwargs)(_moe_checkpoint_route_impl)
    op.register_fake(_moe_checkpoint_route_fake)
    return op


_moe_checkpoint_route_op = _register_moe_checkpoint_route_op()


def _blend_checkpoint_routing(top_k_index, top_k_weights, index, weights):
    """Routing weights with the first pass's values and, where it is exact, the replay's gradient.

    ``weights`` (first pass, no grad) give the values. A token whose replay picked the same expert set
    (in any slot order) takes the replay's gradient, matched slot by slot: the same set under the same
    normalization is the same differentiable function, so the gradient is exact. A token whose replay
    picked a different set takes no router gradient: the replay's weights are normalized over experts
    the loss never used, so their gradient would reach an expert the first pass did not select, and the
    first pass's set has no differentiable weights here. Aligned [T, k] weights are matched per slot;
    dense [T, E] weights line up already. When nothing changed (``index`` equals ``top_k_index``) the
    values are bitwise ``top_k_weights`` and so is the gradient."""
    if top_k_weights.shape == top_k_index.shape:
        match = index.unsqueeze(-1) == top_k_index.unsqueeze(-2)          # [T, k, k]
        same_set = match.any(-1).all(-1, keepdim = True)                   # [T, 1]
        aligned = (match.to(top_k_weights.dtype) * top_k_weights.unsqueeze(-2)).sum(-1)
        delta = aligned - aligned.detach()
    else:
        same_set = (index.sort(-1).values == top_k_index.sort(-1).values).all(-1, keepdim = True)
        delta = top_k_weights - top_k_weights.detach()
    return weights + torch.where(same_set, delta, torch.zeros_like(delta))


def moe_checkpoint_routing(module, top_k_index, top_k_weights):
    """The (top-k ids, routing weights) an MoE experts call must use.

    Inside an Unsloth gradient-checkpoint replay these are the ids the first (no-grad) pass used,
    so backward differentiates the routing that produced the loss. Everywhere else they are the
    given tensors (eager: the same objects). Training mode only."""
    if not isinstance(top_k_index, torch.Tensor) or not isinstance(top_k_weights, torch.Tensor) \
            or not getattr(module, "training", False):
        return top_k_index, top_k_weights
    grad_mode = torch.is_grad_enabled()
    if not torch.compiler.is_compiling():
        found = _checkpoint_route_eager(top_k_index, top_k_weights, grad_mode)
        if found is None:
            return top_k_index, top_k_weights
        index, weights = found
    else:
        index, weights = _moe_checkpoint_route_op(top_k_index, top_k_weights.detach(), grad_mode)
    if not grad_mode:
        return index, weights
    return index, _blend_checkpoint_routing(top_k_index, top_k_weights, index, weights)


def _detach_checkpoint_args(args):
    """Detach the differentiable tensors in ``args`` for a reentrant recompute.

    ``ctx.args`` are the tensors as they were handed to ``Function.apply``; a
    Python ``autograd.Function`` does not strip their ``grad_fn``. Re-running the
    block on them under ``torch.enable_grad()`` and calling
    ``torch.autograd.backward`` on the result therefore walks straight into the
    history *behind* those tensors from inside this backward - a nested backward
    over a graph the outer engine still intends to visit, which frees it early
    ("Trying to backward through the graph a second time") and delivers the
    gradient as a side effect rather than as this Function's output.

    Detaching stops the traversal at the recompute boundary, exactly as
    ``UnslothCheckpointFunction`` and torch's own reentrant checkpoint do, and
    the caller returns the collected ``.grad`` so the engine keeps propagating.

    Returns ``(recompute_args, grad_holders)`` where ``grad_holders[i]`` is the
    detached tensor to read a gradient off, or ``None`` for inputs that take no
    gradient.
    """
    # All Unsloth Zoo code licensed under LGPLv3
    recompute_args = []
    grad_holders   = []
    for arg in args:
        if torch.is_tensor(arg) and arg.requires_grad:
            detached = arg.detach()
            detached.requires_grad_(True)
            recompute_args.append(detached)
            grad_holders  .append(detached)
        else:
            recompute_args.append(arg)
            grad_holders  .append(None)
    return recompute_args, grad_holders
pass


class Unsloth_Offloaded_Gradient_Checkpointer(torch.autograd.Function):
    """
    All Unsloth Zoo code licensed under LGPLv3
    Saves VRAM by smartly offloading to RAM.
    Tiny hit to performance, since we mask the movement via non blocking calls.
    """
    @staticmethod
    @torch_amp_custom_fwd
    def forward(ctx, forward_function, hidden_states, *args):
        ctx.device = hidden_states.device
        saved_hidden_states = hidden_states.to("cpu", non_blocking = True)
        ctx._routing_frame = _new_checkpoint_routing_frame((hidden_states, *args))
        with torch.no_grad(), _checkpoint_routing_scope(ctx._routing_frame, False):
            output = forward_function(hidden_states, *args)
        ctx.save_for_backward(saved_hidden_states)
        ctx.forward_function = forward_function
        ctx.args = args
        return output
    pass

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, dY):
        (hidden_states,) = ctx.saved_tensors
        hidden_states = hidden_states.to(ctx.device, non_blocking = True).detach()
        hidden_states.requires_grad_(True)
        recompute_args, grad_holders = _detach_checkpoint_args(ctx.args)
        with torch.enable_grad(), _gradient_checkpoint_recompute_marker(ctx.forward_function), \
                _checkpoint_routing_scope(getattr(ctx, "_routing_frame", None), True):
            output = ctx.forward_function(hidden_states, *recompute_args)
        # forward() stores whatever a layer returns, but backward() must unpack a 1-tuple;
        # bare-tensor blocks (Qwen2-VL vision, later transformers) used to raise ValueError too many
        # values to unpack.
        if isinstance(output, tuple):
            (output,) = output
        torch.autograd.backward(output, dY)
        # Extra-input grads go back to the engine, not accumulate in the nested backward.
        return (None, hidden_states.grad,) + tuple(
            None if holder is None else holder.grad for holder in grad_holders
        )
pass


class Unsloth_Gradient_Checkpointer(torch.autograd.Function):
    """
    All Unsloth Zoo code licensed under LGPLv3
    Same as normal gradient checkpointing but cleaner
    """
    @staticmethod
    @torch_amp_custom_fwd
    def forward(ctx, forward_function, hidden_states, *args):
        ctx._routing_frame = _new_checkpoint_routing_frame((hidden_states, *args))
        with torch.no_grad(), _checkpoint_routing_scope(ctx._routing_frame, False):
            output = forward_function(hidden_states, *args)
        ctx.save_for_backward(hidden_states)
        ctx.forward_function = forward_function
        ctx.args = args
        return output
    pass

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, *dY):
        (hidden_states,) = ctx.saved_tensors
        hidden_states = hidden_states.detach()
        hidden_states.requires_grad_(True)
        recompute_args, grad_holders = _detach_checkpoint_args(ctx.args)
        with torch.enable_grad(), _gradient_checkpoint_recompute_marker(ctx.forward_function), \
                _checkpoint_routing_scope(getattr(ctx, "_routing_frame", None), True):
            output = ctx.forward_function(hidden_states, *recompute_args)
        outputs = output if isinstance(output, tuple) else (output,)
        pairs = [(o, g) for o, g in zip(outputs, dY)
                 if torch.is_tensor(o) and o.requires_grad and g is not None]
        if not pairs:
            # As torch's reentrant checkpoint: nothing to backprop is an error, not a step with no gradients.
            raise RuntimeError("none of output has requires_grad=True, this checkpoint() is not necessary")
        torch.autograd.backward([o for o, _ in pairs], [g for _, g in pairs])
        # Extra-input grads go back to the engine, not accumulate in the nested backward.
        return (None, hidden_states.grad,) + tuple(
            None if holder is None else holder.grad for holder in grad_holders
        )
pass




# Checkpoint-only keywords the reentrant shims ignore to keep behaviour; binding one to the wrapped
# block raises TypeError; early_stop (torch 2.10, 2.13) is ignored when use_reentrant=True.
_TORCH_CHECKPOINT_KEYWORDS_LITERAL = frozenset({
    "preserve_rng_state",
    "context_fn",
    "determinism_check",
    "debug",
    "early_stop",
})


def _torch_checkpoint_keywords():
    """The literal set above, widened by whatever torch's own ``checkpoint``
    actually declares.

    Read off the live signature so a keyword a later torch adds (the same way it
    added ``early_stop``) is dropped rather than bound onto the block. Falls back
    to the literal set if the signature cannot be read, and unions rather than
    replaces so a signature already replaced by one of the shims below can only
    ever widen the set. ``use_reentrant`` is excluded because every shim names it
    explicitly, so it never reaches ``**kwargs``."""
    # All Unsloth Zoo code licensed under LGPLv3
    names = set(_TORCH_CHECKPOINT_KEYWORDS_LITERAL)
    try:
        checkpoint = getattr(
            torch.utils.checkpoint, "_unsloth_pristine_checkpoint", None,
        ) or torch.utils.checkpoint.checkpoint
        for name, parameter in inspect.signature(checkpoint).parameters.items():
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
                names.add(name)
    except Exception:
        pass
    names.discard("use_reentrant")
    return frozenset(names)


_TORCH_CHECKPOINT_KEYWORDS = _torch_checkpoint_keywords()


class _KeywordArgumentCall:
    """Re-form keyword arguments from trailing positional arguments.

    ``functools.partial(function, **extra)`` is the obvious way to turn a
    keyword call into the positional-only call an ``autograd.Function`` can
    make, but a tensor closed over that way is invisible to autograd: it never
    appears in the Function's input list, so the Function cannot return a
    gradient for it. The reentrant checkpointers re-run the wrapped block under
    ``torch.enable_grad()`` and then call ``torch.autograd.backward`` on the
    recomputed output, so a captured non-leaf tensor is walked *through* - its
    own history is traversed inside the nested backward. When that history is
    shared (the same tensor handed to several checkpointed layers, or also
    feeding the loss) the first nested backward frees it and the next traversal
    raises ``Trying to backward through the graph a second time``; when it does
    not raise, the gradient arrives as a side effect of the nested call instead
    of as a Function output.

    So tensors are passed as ordinary positional inputs to the Function - which
    is what makes them visible to autograd - and this callable puts them back
    into keyword position, taking them off the tail of the positional list.
    Non-tensor keywords have no gradient and stay bound directly.
    """
    # All Unsloth Zoo code licensed under LGPLv3
    __slots__ = ("function", "keys", "constants",)

    def __init__(self, function, keys, constants):
        self.function  = function
        self.keys      = keys
        self.constants = constants
    pass

    def __call__(self, *args):
        split = len(args) - len(self.keys)
        kwargs = dict(self.constants)
        kwargs.update(zip(self.keys, args[split:]))
        return self.function(*args[:split], **kwargs)
pass


def _bind_checkpoint_kwargs(function, kwargs):
    """Bind leftover keyword arguments into ``function``.

    ``checkpoint(fn, *args, **kwargs)`` means "call ``fn(*args, **kwargs)``";
    that is what torch's non-reentrant path does. The Unsloth checkpointers are
    ``torch.autograd.Function`` subclasses and can only forward positionals, so
    the keywords are bound into the callable instead - the same trick
    ``transformers.modeling_layers.GradientCheckpointingLayer.__call__`` uses.

    Before this, ``unsloth_gradient_checkpoint`` /
    ``unsloth_offloaded_gradient_checkpoint`` dropped them silently (the wrapped
    block then ran with different arguments and no diagnostic) while
    ``unsloth_checkpoint`` raised ``Unexpected keyword arguments``. Binding makes
    all three agree and never turns a working call into an error.

    Returns ``(callable, extra_positional_args)``. The extra arguments are the
    keyword values that require grad; the caller must append them to the
    Function's positional inputs so autograd can see them, and
    ``_KeywordArgumentCall`` moves them back into keyword position before the
    block is called. Returns ``function`` unchanged with an empty tuple when
    there is nothing to bind, so the ordinary empty-kwargs path allocates
    nothing and the no-tensor path keeps the cheap ``functools.partial``.
    """
    # All Unsloth Zoo code licensed under LGPLv3
    if not kwargs:
        return function, ()
    extra = {k : v for k, v in kwargs.items() if k not in _TORCH_CHECKPOINT_KEYWORDS}
    if not extra:
        return function, ()
    keys = tuple(
        k for k, v in extra.items() if torch.is_tensor(v) and v.requires_grad
    )
    if not keys:
        return functools.partial(function, **extra), ()
    constants = {k : v for k, v in extra.items() if k not in keys}
    return _KeywordArgumentCall(function, keys, constants), \
        tuple(extra[k] for k in keys)
pass


@torch._disable_dynamo
def _pristine_for_compiled_checkpoint(use_reentrant, kwargs):
    # torch.compile's checkpoint HOP (tag_activation_checkpoint_impl) re-enters torch.utils.checkpoint.checkpoint while
    # AOT autograd traces it, with use_reentrant = False and a context_fn; the shims must hand that to torch's own.
    context_fn = kwargs.get("context_fn")
    if use_reentrant is False and context_fn is not None and context_fn is not torch.utils.checkpoint.noop_context_fn:
        return getattr(torch.utils.checkpoint, "_unsloth_pristine_checkpoint", None)
    return None
pass


NO_GRAD_CHECKPOINT_STATS = collections.Counter()   # checkpoint calls run directly under grad off


def _call_without_checkpoint(function, *args):
    """Grad off records no graph, so run the block directly: skips the offload copy, routing /
    attention frames and layer counters (GRPO / DPO ref logps, eval under no_grad)."""
    # All Unsloth Zoo code licensed under LGPLv3
    NO_GRAD_CHECKPOINT_STATS["calls"] += 1
    with _checkpoint_routing_scope(None, False), saved_attention_scope(None, False):
        return function(*args)
pass


@torch._disable_dynamo
def unsloth_gradient_checkpoint(function, *args, use_reentrant = None, **kwargs):
    pristine = _pristine_for_compiled_checkpoint(use_reentrant, kwargs)
    if pristine is not None:
        return pristine(function, *args, use_reentrant = False, **kwargs)
    function, tensor_args = _bind_checkpoint_kwargs(function, kwargs)
    if not torch.is_grad_enabled():
        return _call_without_checkpoint(function, *args, *tensor_args)
    return Unsloth_Gradient_Checkpointer.apply(function, *args, *tensor_args)
pass


# Names of the checkpoint shims below; none is the pristine torch fn.
_UNSLOTH_CKPT_SHIM_NAMES = frozenset({
    "unsloth_checkpoint",
    "unsloth_gradient_checkpoint",
    "unsloth_offloaded_gradient_checkpoint",
})


def _capture_pristine_checkpoint_once():
    # Stash torch's checkpoint on first patch only: a second patch overwrites _old_checkpoint with
    # the shim, losing it; the Gemma-4 KV-sharing fix reads it to force use_reentrant=False.
    ck = torch.utils.checkpoint
    if getattr(ck, "_unsloth_pristine_checkpoint", None) is not None:
        return
    current = getattr(ck, "checkpoint", None)
    if current is not None and getattr(current, "__name__", "") not in _UNSLOTH_CKPT_SHIM_NAMES:
        ck._unsloth_pristine_checkpoint = current


def _torch_checkpoint():
    """torch's own checkpoint, looked up per call. Smart gradient checkpointing replaces
    torch.utils.checkpoint.checkpoint with unsloth_checkpoint, which forces the reentrant, CPU-offloading
    UnslothCheckpointFunction: a chunk loop inside a decoder layer would offload every chunk, advance the
    per-layer offload counters, and drop the grads of parameters the chunk only reaches through a closure."""
    ck = torch.utils.checkpoint
    for func in (
        getattr(ck, "_unsloth_pristine_checkpoint", None),
        ck.checkpoint,
        getattr(ck, "_old_checkpoint", None),
    ):
        if getattr(func, "__module__", None) == ck.__name__:
            return func
    return ck.checkpoint


def patch_unsloth_gradient_checkpointing():
    print("Unsloth: Patched gradient checkpointing for long context finetuning.")
    import torch.utils
    if torch.utils.checkpoint.checkpoint.__name__ == "unsloth_offloaded_gradient_checkpoint": return
    _capture_pristine_checkpoint_once()
    torch.utils.checkpoint._old_checkpoint = torch.utils.checkpoint.checkpoint
    torch.utils.checkpoint.checkpoint = unsloth_offloaded_gradient_checkpoint
    _bind_transformers_checkpoint(unsloth_offloaded_gradient_checkpoint)
    os.environ["UNSLOTH_PATCHED"] = "1"
pass


def patch_gradient_checkpointing():
    print("Unsloth: Patched gradient checkpointing.")
    import torch.utils
    if torch.utils.checkpoint.checkpoint.__name__ == "unsloth_gradient_checkpoint": return
    _capture_pristine_checkpoint_once()
    torch.utils.checkpoint._old_checkpoint = torch.utils.checkpoint.checkpoint
    torch.utils.checkpoint.checkpoint = unsloth_gradient_checkpoint
    _bind_transformers_checkpoint(unsloth_gradient_checkpoint)
    os.environ["UNSLOTH_PATCHED"] = "1"
pass


def _bind_transformers_checkpoint(shim):
    # Keep transformers' prior checkpoint (maybe a wrapper) like _old_checkpoint for unpatch.
    import transformers.modeling_utils
    current = getattr(transformers.modeling_utils, "checkpoint", None)
    if current is not shim:
        transformers.modeling_utils._unsloth_old_checkpoint = current
    transformers.modeling_utils.checkpoint = shim
pass


def _restore_transformers_checkpoint(shim):
    # The patch points transformers.modeling_utils.checkpoint at the shim as well, and that is
    # the one gradient_checkpointing_enable() wraps. Left behind, a model set up for vanilla
    # gradient checkpointing later in the process still runs the shim, including the offloaded
    # one that allocates pinned host buffers.
    import transformers.modeling_utils
    if getattr(transformers.modeling_utils, "checkpoint", None) is not shim: return
    previous = transformers.modeling_utils.__dict__.pop("_unsloth_old_checkpoint", None)
    transformers.modeling_utils.checkpoint = (
        previous if previous is not None else torch.utils.checkpoint.checkpoint
    )
pass


def unpatch_unsloth_gradient_checkpointing():
    import torch.utils
    if hasattr(torch.utils.checkpoint, "_old_checkpoint"):
        torch.utils.checkpoint.checkpoint = torch.utils.checkpoint._old_checkpoint
        del torch.utils.checkpoint._old_checkpoint
    pass
    _restore_transformers_checkpoint(unsloth_offloaded_gradient_checkpoint)
pass


def unpatch_gradient_checkpointing():
    import torch.utils
    if hasattr(torch.utils.checkpoint, "_old_checkpoint"):
        torch.utils.checkpoint.checkpoint = torch.utils.checkpoint._old_checkpoint
        del torch.utils.checkpoint._old_checkpoint
    pass
    _restore_transformers_checkpoint(unsloth_gradient_checkpoint)
pass


from torch.utils.checkpoint import (
    check_backward_validity,
    _infer_device_type,
    _get_autocast_kwargs,
    _get_device_module,
    get_device_states,
    detach_variable,
    contextlib,
    DefaultDeviceType,
)
# Added [device_type] in Torch 2.5!
def set_device_states(devices, states, *, device_type=None) -> None:
    """Set RNG states for the given devices.

    device_type defaults to ``DefaultDeviceType.get_device_type()`` (cuda).
    """
    if device_type is None:
        device_type = DefaultDeviceType.get_device_type()
    if device_type == "meta":
        return
    device_module = _get_device_module(device_type)
    for device, state in zip(devices, states):
        device_module.set_rng_state(state, device)
pass


def _cuda_tensor_arg_devices(args):
    """Device ids of args that are all plain CUDA tensors (transformers passes just hidden_states), else None.
    Matches _infer_device_type + get_device_states for that case without their per-layer pytree walks."""
    device_ids = []
    for arg in args:
        if (type(arg) is not torch.Tensor and type(arg) is not torch.nn.Parameter) or arg.device.type != "cuda":
            return None
        device_ids.append(arg.get_device())
    return device_ids or None
pass

global CPU_BUFFERS
global CPU_INDEX
global GPU_BUFFERS
global GPU_BUFFERS_B
global USE_DOUBLE_BUFFER
global BACKWARD_PASS
global EXTRA_STREAMS
global MAIN_STREAMS
global MINIMUM_SIZE
global USE_UNSLOTH_GC
global LAST_GC_INDEX
global FIRST_PASS
global CURRENT_GC_INDEX
global BUFFER_EVENTS_A
global BUFFER_EVENTS_B
global NEXT_BUFFER_SLOT

if DEVICE_TYPE in ("cuda", "hip"):
    torch_gpu_stream = torch.cuda.stream
elif DEVICE_TYPE == "xpu":
    torch_gpu_stream = torch.xpu.stream
elif DEVICE_TYPE == "npu":
    torch_gpu_stream = torch.npu.stream


def _compute_stream(device_index):
    """Current compute stream; never cache it at init (callers may train under `torch.cuda.stream(s)`)."""
    if DEVICE_TYPE in ("cuda", "hip"):
        return torch.cuda.current_stream(device_index)
    elif DEVICE_TYPE == "xpu":
        return torch.xpu.current_stream(device_index)
    elif DEVICE_TYPE == "npu":
        return torch.npu.current_stream(device_index)
    raise RuntimeError(f"Unsloth: no compute stream for device type {DEVICE_TYPE}")
pass


def _new_device_event():
    if DEVICE_TYPE in ("cuda", "hip"):
        return torch.cuda.Event()
    elif DEVICE_TYPE == "xpu":
        return torch.xpu.Event()
    elif DEVICE_TYPE == "npu":
        return torch.npu.Event()
    raise RuntimeError(f"Unsloth: no device events for device type {DEVICE_TYPE}")
pass


# UNSLOTH_GC_PREFETCH=1: during checkpoint k's backward, reload k-1 into the other slot.
# Only along a verified sequential chain; otherwise falls back to on-demand reload.
GC_PREFETCH = os.environ.get("UNSLOTH_GC_PREFETCH", "0") == "1"
GC_PREFETCH_STATS = {"issued": 0, "consumed": 0, "dropped": 0}
OFFLOAD_GENERATION = 0      # bumped at the first offload of every forward
OFFLOAD_SLOT_META = []      # per CPU_INDEX: (generation, device_index, buffer_slot, new_size, shape, store_dtype)
PREFETCHED = {}             # device_index -> (generation, cpu_index, buffer_slot)
LAST_BACKWARD_INDEX = {}    # "last" -> (generation, cpu_index) of the last offloaded backward, any device
COPY_EVENTS = {}            # (device_index, buffer_slot) -> event recorded after that slot's reload


def _reset_gc_prefetch_state():
    PREFETCHED.clear()
    LAST_BACKWARD_INDEX.clear()
    del OFFLOAD_SLOT_META[:]
pass


def _copy_event(device_index, buffer_slot):
    key = (device_index, buffer_slot)
    event = COPY_EVENTS.get(key)
    if event is None:
        event = COPY_EVENTS[key] = _new_device_event()
    return event
pass


def _prefetch_offloaded_input(cpu_index, device_index, generation, current_slot):
    """Queue reload of `cpu_index` into its slot; any doubt returns False (on-demand reload)."""
    if not GC_PREFETCH or not USE_DOUBLE_BUFFER or GPU_BUFFERS_B is None: return False
    if cpu_index < 0 or cpu_index >= len(OFFLOAD_SLOT_META): return False
    meta = OFFLOAD_SLOT_META[cpu_index]
    if meta is None: return False
    meta_generation, meta_device, buffer_slot, new_size, shape, store_dtype = meta
    if meta_generation != generation or meta_device != device_index: return False
    # The other slot only: the current one is about to be read by this recompute.
    if buffer_slot == current_slot: return False
    if PREFETCHED.get(device_index) == (generation, cpu_index, buffer_slot): return False
    host_buffer = CPU_BUFFERS[cpu_index] if cpu_index < len(CPU_BUFFERS) else None
    if host_buffer is None or not getattr(host_buffer, HOST_PINNED_ATTR, False): return False
    gpu_staging = GPU_BUFFERS_B[device_index] if buffer_slot == 1 else GPU_BUFFERS[device_index]
    if gpu_staging.numel() * gpu_staging.element_size() < new_size: return False
    if host_buffer.numel() * host_buffer.element_size() < new_size: return False
    PREFETCHED.pop(device_index, None)
    try:
        buffer = _view_bytes_as(gpu_staging, new_size, store_dtype, shape)
        x = _view_bytes_as(host_buffer, new_size, store_dtype, shape)
        EXTRA_STREAM = EXTRA_STREAMS[device_index]
        event_buffer = BUFFER_EVENTS_B if buffer_slot == 1 else BUFFER_EVENTS_A
        EXTRA_STREAM.wait_event(event_buffer[device_index])
        with torch_gpu_stream(EXTRA_STREAM):
            buffer.copy_(x, non_blocking = True)
            _copy_event(device_index, buffer_slot).record(EXTRA_STREAM)
    except RuntimeError:
        GC_PREFETCH_STATS["dropped"] += 1
        return False
    PREFETCHED[device_index] = (generation, cpu_index, buffer_slot)
    GC_PREFETCH_STATS["issued"] += 1
    return True
pass


def _take_prefetched(device_index, generation, cpu_index, buffer_slot):
    """Consume a matching prefetch once; a stale or mismatched one is dropped."""
    entry = PREFETCHED.pop(device_index, None)
    if entry is None: return False
    if entry == (generation, cpu_index, buffer_slot):
        GC_PREFETCH_STATS["consumed"] += 1
        return True
    GC_PREFETCH_STATS["dropped"] += 1
    return False
pass

class _PerDeviceSlots(list):
    """One slot per visible device, created on first access to that device.

    Eager allocation on every GPU made each torchrun rank open a context on every card.
    Iteration yields None for untouched devices and never allocates.
    """
    def __init__(self, n, factory):
        super().__init__([None] * n)
        self._factory = factory

    def __getitem__(self, index):
        value = list.__getitem__(self, index)
        if value is None and isinstance(index, int):
            value = self._factory(index)
            list.__setitem__(self, index, value)
        return value
pass


def _allocated_slots(slots):
    """Slots already created, without creating the rest."""
    if slots is None: return []
    return [x for x in slots if x is not None]
pass


def _free_gpu_buffers_b():
    """Release buffer B on every device that has one and turn double buffering off."""
    global GPU_BUFFERS_B
    global USE_DOUBLE_BUFFER
    for buffer in _allocated_slots(GPU_BUFFERS_B):
        if hasattr(buffer, "resize_"): buffer.resize_(0)
    GPU_BUFFERS_B = None
    USE_DOUBLE_BUFFER = False
pass


def _new_gpu_buffer(device_index):
    with _no_inference_mode():
        return torch.empty(INITIAL_GPU_BUFFER_SIZE, dtype = GPU_BUFFER_DTYPE, device = f"{DEVICE_TYPE_TORCH}:{device_index}")
pass


def _new_extra_stream(device_index):
    # A bare Stream() lands on the current device, so create each card's side stream on that card
    if DEVICE_TYPE == "npu":
        return torch.npu.Stream(device = device_index)
    elif DEVICE_TYPE_TORCH == "cuda":
        return torch.cuda.Stream(device = torch.device(f"cuda:{device_index}"))
    return torch.xpu.Stream(device = torch.device(f"xpu:{device_index}"))
pass


def _main_stream(device_index):
    if DEVICE_TYPE in ("cuda", "hip"):
        return torch.cuda.default_stream(torch.device(f"cuda:{device_index}"))
    elif DEVICE_TYPE == "xpu":
        return torch.xpu.current_stream(torch.device(f"xpu:{device_index}"))
    return torch.npu.default_stream(device_index)
pass


def _ensure_device_buffers(device_index):
    """Create this device's staging buffer(s) and side stream before the offload uses them."""
    try:
        GPU_BUFFERS[device_index]
    except Exception as e:
        if "out of memory" not in str(e).lower():
            print("="*10 + "\n")
            print("Unsloth: Your setup does not support `PYTORCH_CUDA_ALLOC_CONF`\n")
            print("Please set `import os; os.environ['PYTORCH_CUDA_ALLOC_CONF'] = '';`\n")
            print("Then re-run Unsloth from the start.")
            print("="*10 + "\n")
        raise
    EXTRA_STREAMS[device_index]
    if GPU_BUFFERS_B is not None:
        try:
            GPU_BUFFERS_B[device_index]
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            _free_gpu_buffers_b()
pass


def _visible_device_count():
    if DEVICE_TYPE in ("cuda", "hip"):
        return torch.cuda.device_count()
    elif DEVICE_TYPE == "npu":
        return torch.npu.device_count()
    return torch.xpu.device_count()
pass


def _double_buffer_event_ctor():
    if DEVICE_TYPE in ("cuda", "hip"):
        return torch.cuda.Event
    elif DEVICE_TYPE == "xpu":
        return torch.xpu.Event
    elif DEVICE_TYPE == "npu":
        return torch.npu.Event
    raise RuntimeError(f"Double buffering unsupported on {DEVICE_TYPE}")
pass


CPU_BUFFERS = []
CPU_INDEX = None
# None until initialize_unsloth_gradient_checkpointing runs (and again after unpatching): the "initialized"
# sentinel. CPU_BUFFERS cannot be one, it stays [] on purpose when every device keeps its inputs.
GPU_BUFFERS = None
GPU_BUFFER_DTYPE = None

# Slots are leased until their backward so 1F1B never reuses a pending one; lowest free first.
_FREE_CPU_SLOTS = []    # min-heap of returned slot indices
_CPU_SLOT_TOP = 0       # slots [0, _CPU_SLOT_TOP) have been handed out at least once
_CPU_SLOT_EPOCH = 0     # bumped by every pool reset; a lease from an older epoch is ignored
# A slot released on one device's side stream and leased on another waits on the release event.
_SLOT_FENCES = {}       # slot index -> (device index, event recorded at release)
# A retain_graph=True graph re-reads its slot on the next backward: hold it until the graph is freed.
_SLOT_HOLDERS = {}      # slot index -> weakref to the autograd ctx whose backward returned it
_RETAINED_SLOTS = {}    # slot index -> weakref to a ctx whose graph is still retained


def _reset_cpu_slots():
    global _CPU_SLOT_TOP, _CPU_SLOT_EPOCH
    del _FREE_CPU_SLOTS[:]
    _CPU_SLOT_TOP = 0
    _CPU_SLOT_EPOCH += 1
    _SLOT_FENCES.clear()
    _SLOT_HOLDERS.clear()
    _RETAINED_SLOTS.clear()
pass


def _graph_retained(ctx_ref):
    ctx = ctx_ref()
    if ctx is None: return False
    try:
        ctx._raw_saved_tensors
    except (RuntimeError, AttributeError):
        return False
    return True
pass


def _reclaim_retained_slots():
    for index, ctx_ref in list(_RETAINED_SLOTS.items()):
        if not _graph_retained(ctx_ref):
            del _RETAINED_SLOTS[index]
            heapq.heappush(_FREE_CPU_SLOTS, index)
pass


class _CpuSlotLease:
    """One CPU_BUFFERS slot held by an offloaded checkpoint until its backward has read it."""
    __slots__ = ("index", "epoch", "device", "stream", "slot")

    def __init__(self, device = None, stream = None):
        global _CPU_SLOT_TOP
        if _RETAINED_SLOTS: _reclaim_retained_slots()
        self.index = None
        while _FREE_CPU_SLOTS:
            index = heapq.heappop(_FREE_CPU_SLOTS)
            holder = _SLOT_HOLDERS.pop(index, None)
            if holder is not None and _graph_retained(holder):
                _RETAINED_SLOTS[index] = holder
                continue
            self.index = index
            break
        if self.index is None:
            self.index = _CPU_SLOT_TOP
            _CPU_SLOT_TOP += 1
        self.slot = self.index
        self.epoch = _CPU_SLOT_EPOCH
        self.device, self.stream = device, stream
        fence = _SLOT_FENCES.pop(self.index, None)
        if fence is not None and fence[0] != device and stream is not None:
            stream.wait_event(fence[1])

    def release(self, ctx = None):
        index, self.index = self.index, None
        if index is None and ctx is not None and self.stream is not None and self.epoch == _CPU_SLOT_EPOCH:
            # A retained graph's next backward reloads the slot again: fence that reload, not the first.
            holder = _SLOT_HOLDERS.get(self.slot) or _RETAINED_SLOTS.get(self.slot)
            if holder is not None and holder() is ctx:
                _SLOT_FENCES[self.slot] = (self.device, self.stream.record_event())
        if index is not None and self.epoch == _CPU_SLOT_EPOCH:
            if self.stream is not None:
                _SLOT_FENCES[index] = (self.device, self.stream.record_event())
            if ctx is not None:
                try:
                    _SLOT_HOLDERS[index] = weakref.ref(ctx)
                except TypeError:
                    pass
            heapq.heappush(_FREE_CPU_SLOTS, index)

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass  # interpreter shutdown
pass

def initialize_unsloth_gradient_checkpointing(dtype = None):
    # All Unsloth Zoo code licensed under LGPLv3
    global FP32_OFFLOAD_EXACT, FP32_OFFLOAD_MODE
    FP32_OFFLOAD_MODE = _fp32_offload_mode()
    FP32_OFFLOAD_EXACT = FP32_OFFLOAD_MODE == "exact"
    _maybe_enable_exact_pinned()
    global GC_PREFETCH
    GC_PREFETCH = os.environ.get("UNSLOTH_GC_PREFETCH", "0") == "1"
    _reset_gc_prefetch_state()
    global CPU_BUFFERS
    global CPU_INDEX
    global GPU_BUFFERS
    global GPU_BUFFERS_B
    global USE_DOUBLE_BUFFER
    global BACKWARD_PASS
    global EXTRA_STREAMS
    global MAIN_STREAMS
    global MINIMUM_SIZE
    global USE_UNSLOTH_GC
    global LAST_GC_INDEX
    global FIRST_PASS
    global CURRENT_GC_INDEX
    global BUFFER_EVENTS_A
    global BUFFER_EVENTS_B
    global NEXT_BUFFER_SLOT
    _release_pool_pinned()
    _reset_pinned_cap()
    CPU_BUFFERS = []
    CPU_INDEX = 0
    _reset_cpu_slots()
    _KEEP_BUDGET.clear()

    if dtype is None:
        if DEVICE_TYPE == "cuda":
            # Check visible GPUs: device_map can put pre-Ampere layers beside Ampere device 0.
            from .device_type import _visible_cuda_capabilities
            SUPPORTS_BFLOAT16 = all(major >= 8 for major, _ in _visible_cuda_capabilities())
        elif DEVICE_TYPE == "hip":
            # Ask rather than assume: RDNA 1/2 (gfx101x, gfx103x) have no native bf16,
            # so Triton picks a dot intrinsic LLVM cannot lower and the process dies
            # with no Python exception (unslothai/unsloth issue 7922). Unpatched ROCm
            # still answers True here, so this is inert until unsloth patches the probe.
            SUPPORTS_BFLOAT16 = torch.cuda.is_bf16_supported()
        elif DEVICE_TYPE == "xpu":
            # One buffer dtype for the pool below, which covers every visible XPU.
            from .device_type import _xpu_bf16_supported
            try:
                _xpu_indices = list(range(torch.xpu.device_count()))
            except Exception:
                _xpu_indices = None
            SUPPORTS_BFLOAT16 = _xpu_bf16_supported(_xpu_indices or None)
        elif DEVICE_TYPE == "npu":
            SUPPORTS_BFLOAT16 = torch.npu.is_bf16_supported()
        dtype = torch.bfloat16 if SUPPORTS_BFLOAT16 else torch.float16
    pass

    # Re-probe and clear the warning latch, or a second model falls back silently.
    global PINNED_MEMORY_AVAILABLE
    global _WARNED_ABOUT_PINNED_MEMORY
    PINNED_MEMORY_AVAILABLE = True
    _WARNED_ABOUT_PINNED_MEMORY = False
    # Every visible device keeps its inputs: no slot is ever leased, so skip the pinned pool (it grows on demand).
    keep = _keep_on_device_indices()
    if not (keep and len(keep) >= _visible_device_count()):
        with _no_inference_mode():
            for i in range(INITIAL_CPU_BUFFER_COUNT):
                x = _new_host_buffer(INITIAL_CPU_BUFFER_SIZE, dtype)
                CPU_BUFFERS.append(x)
    pass

    global GPU_BUFFER_DTYPE
    GPU_BUFFER_DTYPE = dtype
    n_gpus = _visible_device_count()
    NEXT_BUFFER_SLOT = [0] * n_gpus
    GPU_BUFFERS = _PerDeviceSlots(n_gpus, _new_gpu_buffer)
    # Double buffering: auto-off on unified memory, or via env var
    if _double_buffer_disabled():
        GPU_BUFFERS_B = None
        USE_DOUBLE_BUFFER = False
        BUFFER_EVENTS_A = None
        BUFFER_EVENTS_B = None
    else:
        GPU_BUFFERS_B = _PerDeviceSlots(n_gpus, _new_gpu_buffer)
        USE_DOUBLE_BUFFER = False # enabled after first pass if CUDA free memory > DOUBLE_BUFFER_HEADROOM
        # Per-buffer events prevent double-buffer races; bound to a device on first record.
        event_ctor = _double_buffer_event_ctor()
        BUFFER_EVENTS_A = tuple([event_ctor() for _ in range(n_gpus)])
        BUFFER_EVENTS_B = tuple([event_ctor() for _ in range(n_gpus)])

    BACKWARD_PASS = True
    EXTRA_STREAMS = _PerDeviceSlots(n_gpus, _new_extra_stream)
    MAIN_STREAMS  = _PerDeviceSlots(n_gpus, _main_stream)

    # Minimum size to enable Unsloth GC is 2MB (in bytes) -> 32 layers = 64MB
    MINIMUM_SIZE = 2 * 1024 * 1024
    USE_UNSLOTH_GC = True

    # Don't offload the last layer - uses more VRAM and is slower
    # See https://github.com/pytorch/torchtune/pull/1443
    LAST_GC_INDEX = 0
    FIRST_PASS = True
    CURRENT_GC_INDEX = 0
pass


class UnslothCheckpointFunction(torch.autograd.Function):

    @classmethod
    def apply(cls, run_function, preserve_rng_state, *args):
        # Also installed as torch.utils.checkpoint.CheckpointFunction, so pre-bound torch checkpoints land here.
        if not torch.is_grad_enabled() and not torch.compiler.is_compiling():
            return _call_without_checkpoint(run_function, *args)
        return super().apply(run_function, preserve_rng_state, *args)
    pass

    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        # All Unsloth Zoo code licensed under LGPLv3
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        # Handle autocast enabled for cpu AND gpu.
        cuda_devices = _cuda_tensor_arg_devices(args)
        ctx.device_type = "cuda" if cuda_devices is not None else _infer_device_type(*args)
        ctx.device_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs(
            ctx.device_type
        )
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident: we can't
            # anticipate run_function initializing it later, so don't stash here.
            ctx.had_device_in_fwd = False
            device_module = _get_device_module(ctx.device_type)
            if getattr(device_module, "_initialized", False):
                ctx.had_device_in_fwd = True
                if cuda_devices is not None:
                    ctx.fwd_devices = cuda_devices
                    ctx.fwd_device_states = [torch.cuda.get_rng_state(device) for device in cuda_devices]
                else:
                    ctx.fwd_devices, ctx.fwd_device_states = get_device_states(*args)

        # Save non-tensor inputs in ctx, keep a placeholder None for tensors
        # to be filled out during the backward.
        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        ctx._requires_gradient = False
        # Read unconditionally in backward, so it needs a value on every path
        # that never takes the offload branch below.
        ctx._saved_metadata = (None, None, None, None, None, None, None,)
        use_gpu_buffer = False

        for i, arg in enumerate(args):
            if torch.is_tensor(arg):
                # ANY differentiable input means this Function has a gradient to
                # produce, exactly as torch's own reentrant CheckpointFunction
                # (which saves unconditionally) does. Keying this off input zero
                # alone made backward bail out early - and it is reached, since
                # autograd only calls backward when some input requires grad -
                # so the engine got a single None instead of one gradient per
                # input and raised `returned an incorrect number of gradients`.
                # A frozen first activation next to a differentiable later one
                # is ordinary: an untrained embedding feeding layer 0 alongside
                # a learned positional/query tensor, or a tensor keyword routed
                # positionally by _bind_checkpoint_kwargs.
                if arg.requires_grad: ctx._requires_gradient = True

                if i == 0 and arg.requires_grad:
                    global FIRST_PASS
                    global LAST_GC_INDEX
                    if FIRST_PASS:
                        # Save last layer index so next run skips offloading it
                        # (saves VRAM and time). See
                        # https://github.com/pytorch/torchtune/pull/1443
                        LAST_GC_INDEX += 1
                    pass
                    global CURRENT_GC_INDEX
                    CURRENT_GC_INDEX += 1

                    # The cutoff counts stored bytes: bf16-stored float32 (autocast) offloads exactly where it always
                    # did; exactly stored float32 (no autocast) counts 4 bytes, so 2-4MB activations now offload too.
                    store_dtype = _offload_store_dtype(arg)
                    new_size = arg.numel() * store_dtype.itemsize

                    global MINIMUM_SIZE
                    global CPU_INDEX
                    if new_size > MINIMUM_SIZE and ((CURRENT_GC_INDEX != LAST_GC_INDEX) or FIRST_PASS) and \
                            not _keeps_on_device(arg.device.index, new_size):
                        use_gpu_buffer = True
                        global CPU_BUFFERS
                        global GPU_BUFFERS
                        global GPU_BUFFERS_B
                        global USE_DOUBLE_BUFFER
                        global BACKWARD_PASS
                        global EXTRA_STREAMS
                        global MAIN_STREAMS
                        device = arg.device
                        device_index = device.index
                        _ensure_device_buffers(device_index)
                        GPU_BUFFER   = GPU_BUFFERS  [device_index]
                        MAIN_STREAM  = _compute_stream(device_index)
                        EXTRA_STREAM = EXTRA_STREAMS[device_index]

                        # New graph. Host slots are not rewound: pending graphs keep theirs.
                        if BACKWARD_PASS:
                            BACKWARD_PASS = False
                            _flush_outgrown_blocks()
                            # A new graph: earlier prefetch descriptors and slot records are stale.
                            global OFFLOAD_GENERATION
                            OFFLOAD_GENERATION += 1
                            _reset_gc_prefetch_state()
                            if not FIRST_PASS and not USE_DOUBLE_BUFFER and GPU_BUFFERS_B is not None:
                                try:
                                    if DEVICE_TYPE in ("cuda", "hip"):
                                        free_mem, _ = torch.cuda.mem_get_info(device_index)
                                    elif DEVICE_TYPE == "xpu":
                                        free_mem, _ = torch.xpu.mem_get_info(device_index)
                                    elif DEVICE_TYPE == "npu":
                                        free_mem, _ = torch.npu.mem_get_info(device_index)
                                    else:
                                        free_mem = 0
                                except Exception as e:
                                    free_mem = 0
                                if free_mem > DOUBLE_BUFFER_HEADROOM:
                                    USE_DOUBLE_BUFFER = True
                                    print(f"Unsloth: Double buffering enabled (parallel H2D + compute) for backward pass.")
                                else:
                                    _free_gpu_buffers_b()
                        pass

                        lease = _CpuSlotLease(device_index, EXTRA_STREAM)
                        CPU_INDEX = lease.index
                        # Extend buffer size
                        while CPU_INDEX >= len(CPU_BUFFERS):
                            with _no_inference_mode():
                                x = _new_host_buffer(-(-new_size // GPU_BUFFER.element_size()), GPU_BUFFER.dtype)
                            CPU_BUFFERS.append(x)
                        pass

                        x = CPU_BUFFERS[CPU_INDEX]
                        shape = arg.shape
                        # Elements of each buffer's own dtype needed for new_size bytes, rounded up.
                        host_numel = -(-new_size // x.element_size())
                        if host_numel > x.numel():
                            with _no_inference_mode():
                                x = _grow_host_buffer(x, host_numel)
                            # Backward reads this slot by index, so store the replacement back.
                            CPU_BUFFERS[CPU_INDEX] = x
                        gpu_numel = -(-new_size // GPU_BUFFER.element_size())
                        if gpu_numel > GPU_BUFFER.numel():
                            try:
                                GPU_BUFFER.resize_(gpu_numel)
                            except RuntimeError as e:
                                if "out of memory" not in str(e).lower():
                                    raise
                                # Clear buffer B and resize the single buffer
                                if GPU_BUFFERS_B is not None:
                                    _free_gpu_buffers_b()
                                    print("Unsloth: Disabled double buffering due to insufficient VRAM.")
                                    GPU_BUFFER.resize_(gpu_numel)
                                else:
                                    raise
                        # Resize buffer B when double buffering; disable + free B on OOM
                        if USE_DOUBLE_BUFFER:
                            GPU_BUFFER_B = GPU_BUFFERS_B[device_index]
                            gpu_b_numel = -(-new_size // GPU_BUFFER_B.element_size())
                            if gpu_b_numel > GPU_BUFFER_B.numel():
                                try:
                                    GPU_BUFFER_B.resize_(gpu_b_numel)
                                except RuntimeError as e:
                                    if "out of memory" not in str(e).lower():
                                        raise
                                    _free_gpu_buffers_b()
                                    print("Unsloth: Disabled double buffering due to insufficient VRAM.")

                        # Read the cached flag off the slot itself, before the view is
                        # taken: a view is a fresh object and does not carry the attribute.
                        host_is_pinned = getattr(x, HOST_PINNED_ATTR, False)
                        # See https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams
                        EXTRA_STREAM.wait_stream(MAIN_STREAM)
                        # Casting to the buffer dtype on reload made the recompute see it (LLVM abort on ROCm gfx10),
                        # so the bytes are viewed as store_dtype here and cast back to arg.dtype in backward.
                        x = _view_bytes_as(x, new_size, store_dtype, shape)
                        # x is a normal (non-inference) buffer, so copy_ is safe (unsloth#3828).
                        with torch_gpu_stream(EXTRA_STREAM):
                            # Only a pinned destination gives a genuinely async copy.
                            x.copy_(arg, non_blocking = host_is_pinned)

                        global NEXT_BUFFER_SLOT
                        buffer_slot = NEXT_BUFFER_SLOT[device_index]
                        NEXT_BUFFER_SLOT[device_index] ^= 1
                        ctx._saved_metadata = (new_size, shape, CPU_INDEX, device_index, MAIN_STREAM, EXTRA_STREAM, buffer_slot,)
                        ctx._saved_dtype = arg.dtype
                        ctx._store_dtype = store_dtype
                        # Prefetch only follows records made by a top level forward of this graph.
                        ctx._offload_generation = None if in_gradient_checkpoint_recompute() else OFFLOAD_GENERATION
                        while len(OFFLOAD_SLOT_META) <= CPU_INDEX: OFFLOAD_SLOT_META.append(None)
                        OFFLOAD_SLOT_META[CPU_INDEX] = None if ctx._offload_generation is None else \
                            (OFFLOAD_GENERATION, device_index, buffer_slot, new_size, shape, store_dtype)
                        ctx._cpu_slot = lease
                        CPU_INDEX += 1
                        tensor_inputs.append(None)

                        global USE_UNSLOTH_GC
                        if USE_UNSLOTH_GC:
                            print("Unsloth: Will smartly offload gradients to save VRAM!")
                            USE_UNSLOTH_GC = False
                    else:
                        ctx._saved_metadata = (None, None, None, None, None, None, None,)
                        tensor_inputs.append(arg)
                    pass
                else:
                    tensor_inputs.append(arg)
                pass
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)
            pass
        pass
        if ctx._requires_gradient: ctx.save_for_backward(*tensor_inputs)

        ctx._routing_frame = _new_checkpoint_routing_frame(args) if ctx._requires_gradient else None
        ctx._attention_frame = new_saved_attention_frame(args) if ctx._requires_gradient else None
        with torch.no_grad(), _checkpoint_routing_scope(ctx._routing_frame, False), \
                saved_attention_scope(ctx._attention_frame, False):
            outputs = run_function(*args)

        if use_gpu_buffer: MAIN_STREAM.wait_stream(EXTRA_STREAM)
        return outputs
    pass


    @staticmethod
    def backward(ctx, *args):
        # All Unsloth Zoo code licensed under LGPLv3
        if not ctx._requires_gradient: return None

        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "When use_reentrant=True, torch.utils.checkpoint is incompatible"
                " with .grad() or passing an `inputs` parameter to .backward()."
                " To resolve this error, you can either set use_reentrant=False,"
                " or call .backward() without passing the `inputs` argument."
            )

        # Copy the list to avoid modifying original list.
        inputs = list(ctx.inputs)
        tensor_indices = ctx.tensor_indices
        tensors = ctx.saved_tensors

        new_size, shape, CPU_INDEX, device_index, MAIN_STREAM, EXTRA_STREAM, buffer_slot = ctx._saved_metadata
        if CPU_INDEX is not None:
            global GPU_BUFFER
            global USE_DOUBLE_BUFFER
            global GPU_BUFFERS_B
            global BUFFER_EVENTS_A
            global BUFFER_EVENTS_B
            # Select buffer from per-device buffer_slot
            saved_dtype = ctx._saved_dtype
            store_dtype = getattr(ctx, "_store_dtype", saved_dtype)
            gpu_staging = GPU_BUFFERS_B[device_index] if (USE_DOUBLE_BUFFER and buffer_slot == 1) else GPU_BUFFERS[device_index]
            buffer = _view_bytes_as(gpu_staging, new_size, store_dtype, shape)
            host_buffer = CPU_BUFFERS[CPU_INDEX]
            host_is_pinned = getattr(host_buffer, HOST_PINNED_ATTR, False)
            x = _view_bytes_as(host_buffer, new_size, store_dtype, shape)

            MAIN_STREAM = _compute_stream(device_index)
            generation = getattr(ctx, "_offload_generation", None)
            use_prefetch = GC_PREFETCH and USE_DOUBLE_BUFFER and generation is not None \
                and not in_gradient_checkpoint_recompute()
            prefetched = use_prefetch and _take_prefetched(device_index, generation, CPU_INDEX, buffer_slot)
            if not prefetched:
                # See https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams
                if USE_DOUBLE_BUFFER:
                    # Wait for the last compute on THIS buffer to finish
                    event_buffer = BUFFER_EVENTS_B if buffer_slot == 1 else BUFFER_EVENTS_A
                    EXTRA_STREAM.wait_event(event_buffer[device_index])
                else:
                    # Single buffer mode: wait for the compute stream
                    EXTRA_STREAM.wait_stream(MAIN_STREAM)

                # buffer is a normal (non-inference) buffer, so this reload copy_ is safe (unsloth#3828).
                with torch_gpu_stream(EXTRA_STREAM):
                    buffer.copy_(x, non_blocking = host_is_pinned)
                    if use_prefetch: _copy_event(device_index, buffer_slot).record(EXTRA_STREAM)
            pass
            if use_prefetch:
                # Sequential chain only: this backward follows CPU_INDEX + 1, or is the top of the graph.
                last = LAST_BACKWARD_INDEX.get("last")
                in_chain = (last == (generation, CPU_INDEX + 1)) or \
                    ((last is None or last[0] != generation) and CPU_INDEX == len(OFFLOAD_SLOT_META) - 1)
                LAST_BACKWARD_INDEX["last"] = (generation, CPU_INDEX)
                if in_chain:
                    _prefetch_offloaded_input(CPU_INDEX - 1, device_index, generation, buffer_slot)
            pass
            # Reload is queued on EXTRA_STREAM, as is the next offload into it: safe to reuse now
            # unless the graph is retained (checked at lease time).
            lease = getattr(ctx, "_cpu_slot", None)
            if lease is not None: lease.release(ctx)
        else:
            # No GPU buffer seen
            if len(tensor_indices) != 0:
                inputs[tensor_indices[0]] = tensors[0]
        pass

        # Fill in inputs with appropriate saved tensors.
        for i, idx in enumerate(tensor_indices[1:], start = 1):
            inputs[idx] = tensors[i]
        pass

        global BACKWARD_PASS
        BACKWARD_PASS = True
        global FIRST_PASS
        FIRST_PASS = False
        global CURRENT_GC_INDEX
        CURRENT_GC_INDEX = 0

        # Stash the surrounding rng state, mimic the forward state, then
        # restore the surrounding state when done.
        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_device_in_fwd:
            rng_devices = ctx.fwd_devices
        with torch.random.fork_rng(
            devices=rng_devices, enabled=ctx.preserve_rng_state, device_type=ctx.device_type
        ):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_device_in_fwd:
                    set_device_states(ctx.fwd_devices, ctx.fwd_device_states, device_type=ctx.device_type)

            device_autocast_ctx = torch.amp.autocast(
                device_type=ctx.device_type, **ctx.device_autocast_kwargs
            ) if torch.amp.is_autocast_available(ctx.device_type) else nullcontext()

            # detached_inputs = detach_variable(tuple(inputs))
            detached_inputs = []
            for inp in inputs:
                if not isinstance(inp, torch.Tensor):
                    detached_inputs.append(inp)
                    continue
                x = inp.detach()
                x.requires_grad = inp.requires_grad
                detached_inputs.append(x)
            pass

            # Wait for GPU buffer to finish
            if CPU_INDEX is not None:
                if use_prefetch:
                    # Only this slot's copy: a prefetch queued behind it must not stall the recompute.
                    MAIN_STREAM.wait_event(_copy_event(device_index, buffer_slot))
                else:
                    MAIN_STREAM.wait_stream(EXTRA_STREAM)
                x = buffer.detach() if store_dtype == saved_dtype else buffer.to(saved_dtype)
                x.requires_grad_(True)
                detached_inputs[0] = x
            pass

            with torch.enable_grad(), device_autocast_ctx, torch.amp.autocast("cpu", **ctx.cpu_autocast_kwargs), _gradient_checkpoint_recompute_marker(ctx.run_function), \
                    _checkpoint_routing_scope(getattr(ctx, "_routing_frame", None), True), \
                    saved_attention_scope(getattr(ctx, "_attention_frame", None), True):  # type: ignore[attr-defined]
                outputs = ctx.run_function(*detached_inputs)
            pass
        pass

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # run backward() with only tensor that requires grad
        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(outputs)):
            if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
                outputs_with_grad.append(outputs[i])
                args_with_grad.append(args[i])
        pass

        if len(outputs_with_grad) == 0:
            pass
            # raise RuntimeError(
            #     "none of output has requires_grad=True,"
            #     " this checkpoint() is not necessary"
            # )
        else:
            torch.autograd.backward(outputs_with_grad, args_with_grad)
        pass

        # Record event after compute so the copy stream can wait on it
        if CPU_INDEX is not None and USE_DOUBLE_BUFFER:
            event_buffer = BUFFER_EVENTS_B if buffer_slot == 1 else BUFFER_EVENTS_A
            event_buffer[device_index].record(MAIN_STREAM)

        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else None
            for inp in detached_inputs
        )
        # Clear all memory
        for i in range(len(detached_inputs)):
            detached_inputs[i] = None
            inputs[i] = None
        pass

        return (None, None) + grads
pass


def _is_compiling():
    try: return torch.compiler.is_compiling()
    except AttributeError: return False
pass


GC_OFFLOAD_MIN_BYTES = 0
_ALL_OFFLOADERS = weakref.WeakSet()


def _env_nonnegative_int(name, default):
    try:
        return max(0, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default
pass


def _new_offload_stats():
    stats = {"pinned": 0, "pageable": 0, "kept": 0, "prefetched": 0, "prefetch_hits": 0, "forwards": 0,
             "offloaded": 0, "offloaded_bytes": 0, "restored": 0, "restored_bytes": 0, "prefetched_bytes": 0}
    for reason in ("small", "expanded", "unified"):
        stats["kept_" + reason] = 0
        stats["kept_" + reason + "_bytes"] = 0
    return stats
pass


def _release_keepalive(entry, stream = None):
    """Drop an inflight D2H source once its packing stream is ordered after the copy."""
    copied, owner = entry[1], entry[2]
    if copied is not None: (owner or stream).wait_event(copied)
    entry[0] = entry[1] = entry[2] = None
pass


class _ForwardOrder(list):
    """Weak refs to one forward's records in pack order. Records hold their own order, so the offloader keeps only
    the newest and an abandoned forward's metadata dies with its records."""
    __slots__ = ("forward_id", "backward_started", "counted", "__weakref__")

    def __init__(self, forward_id = 0):
        super().__init__()
        self.forward_id = forward_id
        self.backward_started = False
        self.counted = False
pass


class _OffloadedActivation:
    """One saved activation: a host copy (`copied` marks the D2H), or still a device tensor."""
    __slots__ = ("device", "host", "copied", "device_tensor", "ready", "ready_stream", "index", "pinned_bytes",
                 "unpacked", "group", "keepalive", "order", "nbytes", "__weakref__")

    def __init__(self, device, index, order = None):
        self.order         = order
        self.nbytes        = 0
        self.device        = device
        self.host          = None
        self.copied        = None
        self.device_tensor = None
        self.ready         = None  # set only on a prefetched device_tensor
        self.ready_stream  = None  # the stream that allocated it
        self.index         = index
        self.pinned_bytes  = 0
        self.unpacked      = False
        self.group         = 0     # which hooks() entry (checkpoint call) packed it
        self.keepalive     = None  # its inflight entry: [source, copied, packing stream]

    def __del__(self):
        try:
            # Graph dropped before the keepalive was retired: release the source now, after its copy.
            if self.keepalive is not None: _release_keepalive(self.keepalive)
            if self.pinned_bytes: _release_pinned(self.pinned_bytes)
            # Prefetch dropped unread: order the allocating stream after the reload still writing it.
            if self.ready is not None: self.ready_stream.wait_event(self.ready)
        except Exception:
            pass  # interpreter shutdown
pass


class NonReentrantActivationOffloader:
    """
    All Unsloth Zoo code licensed under LGPLv3
    saved_tensors_hooks for torch's non-reentrant checkpoint: the checkpoint inputs that
    need grad go to pinned host memory on a side stream, so the D2H overlaps the next
    layers instead of queueing on the compute stream.

    UNSLOTH_GC_OFFLOAD_BACKEND=unsloth_stream (from the datta0/unsloth-zoo FSDP2 prototype)
    also keeps the last UNSLOTH_GC_STREAM_KEEP_LAST (default 1) activations on the device,
    whose backward comes first, and reloads layer i-1 on a second side stream while layer i
    recomputes. Costs up to one more layer's checkpoint inputs of VRAM each way.

    Saved tensors under UNSLOTH_GC_OFFLOAD_MIN_BYTES (default 0, off) and expanded views stay on
    the device. Prefetch and keep-last stay inside the forward a record belongs to.
    """
    def __init__(self, stream_mode = None):
        if stream_mode is None:
            stream_mode = os.environ.get("UNSLOTH_GC_OFFLOAD_BACKEND", "").strip().lower() == "unsloth_stream"
        self.stream_mode = bool(stream_mode)
        try:
            self.keep_last = max(0, int(os.environ.get("UNSLOTH_GC_STREAM_KEEP_LAST", "1")))
        except ValueError:
            self.keep_last = 1
        # 0: copy on the compute stream (single GPU default). N > 0: side stream, at most N sources
        # kept alive per device, released after the compute stream waits on the copy. No
        # record_stream on the side stream: its block reuse depends on how far the CPU runs ahead
        # (distributed default 1).
        default_inflight = 1 if is_distributed() else 0
        try:
            self.max_inflight = max(0, int(os.environ.get("UNSLOTH_GC_MAX_INFLIGHT_COPIES", default_inflight)))
        except ValueError:
            self.max_inflight = default_inflight
        self.min_bytes = _env_nonnegative_int("UNSLOTH_GC_OFFLOAD_MIN_BYTES", GC_OFFLOAD_MIN_BYTES)
        # Device indices on unified memory: their saved inputs stay put (see _keep_on_device_indices).
        self.keep_on_device = _keep_on_device_indices()
        self.inflight = {}
        self.d2h_streams = {}
        self.h2d_streams = {}
        self.owned = False
        self.forward_id = 0
        # Weak, so a graph dropped without backward frees its activations. Newest forward only.
        self.order = _ForwardOrder(0)
        self.group = 0
        self.stats = _new_offload_stats()
        _ALL_OFFLOADERS.add(self)
    pass

    @property
    def backward_started(self):
        return self.order.backward_started

    @backward_started.setter
    def backward_started(self, value):
        self.order.backward_started = bool(value)

    def begin_forward(self):
        self.forward_id += 1
        # Counted here: an empty order is reused, not replaced.
        self.stats["forwards"] += 1
        if len(self.order) or self.order.backward_started:
            self.order = _ForwardOrder(self.forward_id)
        self.order.counted = True
    pass

    def _current_order(self):
        order = self.order
        # No owner, no pre-hook: a new forward starts after backward began or the newest record died.
        if order.backward_started or (not self.owned and len(order) and order[-1]() is None):
            self.forward_id += 1
            order = self.order = _ForwardOrder(self.forward_id)
        if not order.counted:
            order.counted = True
            self.stats["forwards"] += 1
        return order
    pass

    def _keep_reason(self, x, nbytes):
        if nbytes < self.min_bytes:
            return "small"
        # Expanded views only: a host copy would materialise every repeat. Other views are offloaded, as
        # they may be the last owner of a larger allocation.
        try:
            storage_bytes = x.untyped_storage().nbytes()
        except Exception:
            return None
        if storage_bytes < nbytes:
            return "expanded"
        return None
    pass

    def _keep(self, reason, nbytes):
        stats = self.stats
        stats["kept_" + reason] += 1
        stats["kept_" + reason + "_bytes"] += nbytes
    pass

    def _stream(self, cache, device_index):
        stream = cache.get(device_index)
        if stream is None:
            stream = cache[device_index] = self._new_stream(device_index)
        return stream
    pass

    # Device seams, overridden by the CPU lifecycle tests.
    def _on_host(self, x):
        return x.device.type == "cpu"

    def _streamable(self, x):
        return DEVICE_TYPE in ("cuda", "hip") and x.device.type == "cuda" and not _is_compiling()

    def _new_stream(self, device_index):
        return _new_extra_stream(device_index)

    def _current_stream(self, device_index):
        return torch.cuda.current_stream(device_index)

    def _stream_ctx(self, stream):
        return torch.cuda.stream(stream)

    def _new_event(self):
        return torch.cuda.Event()

    def _record_stream(self, x, stream):
        x.record_stream(stream)

    def _host_copy(self, x, non_blocking = True):
        return x.to("cpu", non_blocking = non_blocking)

    def _device_copy(self, host, device):
        return host.to(device, non_blocking = True)

    def _empty_on_device(self, host, device):
        return torch.empty_like(host, device = device)

    def _to_host(self, record, x):
        nbytes = x.numel() * x.element_size()
        if PINNED_MEMORY_AVAILABLE and not _pinned_memory_disabled():
            with _PINNED_LOCK:  # held through allocation: no double-spent headroom
                if _reserve_pinned(nbytes) and self._to_pinned(record, x, nbytes): return
        record.host = self._host_copy(x, non_blocking = False)
        self.stats["pageable"] += 1
        self._count_offload(nbytes)
    pass

    def _count_offload(self, nbytes):
        stats = self.stats
        stats["offloaded"] += 1
        stats["offloaded_bytes"] += nbytes
    pass

    def _to_pinned(self, record, x, nbytes):
        """Non_blocking D2H into a pinned block already reserved; False (released) if pinning failed."""
        device_index = x.device.index
        if self.max_inflight == 0:
            try:
                host = self._host_copy(x)
            except RuntimeError as e:
                _release_pinned(nbytes)
                if not _is_host_alloc_oom(e): raise
                _warn_pinned_unavailable(e)
                return False
            record.host, record.pinned_bytes = host, nbytes
            self.stats["pinned"] += 1
            self._count_offload(nbytes)
            return True
        inflight = self.inflight.get(device_index)
        if inflight is None:
            inflight = self.inflight[device_index] = collections.deque()
        compute = self._current_stream(device_index)
        self._retire(inflight, compute, self.max_inflight - 1)
        side = self._stream(self.d2h_streams, device_index)
        side.wait_stream(compute)
        try:
            with self._stream_ctx(side):
                host = self._host_copy(x)
                copied = self._new_event()
                copied.record(side)
        except RuntimeError as e:
            _release_pinned(nbytes)
            if not _is_host_alloc_oom(e): raise
            _warn_pinned_unavailable(e)
            return False
        else:
            # Keepalive: x's block must outlive the copy; record_stream is a no-op on its own stream.
            self._record_stream(x, compute)
            entry = [x, copied, compute]
            inflight.append(entry)
            record.host, record.copied, record.pinned_bytes, record.keepalive = host, copied, nbytes, entry
            self.stats["pinned"] += 1
            self._count_offload(nbytes)
            return True
    pass

    @staticmethod
    def _retire(inflight, compute, keep):
        while inflight and (len(inflight) > keep or inflight[0][1] is None or inflight[0][1].query()):
            _release_keepalive(inflight.popleft(), compute)
    pass

    def _drain(self):
        """Release every keepalive (backward reads the host copies after waiting on them)."""
        for device_index, inflight in self.inflight.items():
            if inflight: self._retire(inflight, self._current_stream(device_index), 0)
    pass

    def pack(self, x):
        if not x.requires_grad:
            return None, x
        if self._on_host(x):
            # Detached like the keep branches below: a CPU op saving its own output would hold itself.
            return None, x.detach()
        # Traced: offload as before (the budget check reads allocator state).
        if self.keep_on_device and x.device.index in self.keep_on_device and not _is_compiling():
            nbytes = x.numel() * x.element_size()
            if _keeps_on_device(x.device.index, nbytes):
                self._keep("unified", nbytes)
                return None, x.detach()
        if _is_compiling():
            # Traced: no size or storage checks, which would add guards on symbolic shapes.
            return x.device, x.to("cpu", non_blocking = True)
        nbytes = x.numel() * x.element_size()
        reason = self._keep_reason(x, nbytes)
        if reason is not None:
            self._keep(reason, nbytes)
            # Detached: an op that saves its own output would otherwise hold itself through its grad_fn.
            return None, x.detach()
        if not self._streamable(x):
            self._count_offload(nbytes)
            # The third item marks an eager copy, so unpack counts its restore.
            return x.device, x.to("cpu", non_blocking = True), nbytes
        order = self._current_order()
        record = _OffloadedActivation(x.device, len(order), order)
        record.group = self.group
        record.nbytes = nbytes
        order.append(weakref.ref(record))
        x = x.detach()
        if self.stream_mode and self.keep_last > 0:
            record.device_tensor = x
            self.stats["kept"] += 1
            if record.index >= self.keep_last:
                older = order[record.index - self.keep_last]()
                if older is not None and older.host is None and older.device_tensor is not None:
                    self._to_host(older, older.device_tensor)
                    older.device_tensor = None
                    self.stats["kept"] -= 1
        else:
            self._to_host(record, x)
        return record
    pass

    def _prefetch(self, index, order = None):
        if order is None: order = self.order
        if index < 0 or index >= len(order): return
        record = order[index]()
        # A checkpoint with several saved inputs unpacks them in order, so index - 1 can be an
        # input of the same checkpoint that was already handed out: reloading it duplicates it.
        if record is None or record.unpacked or record.host is None or record.device_tensor is not None: return
        device_index = record.device.index
        compute = self._current_stream(device_index)
        side = self._stream(self.h2d_streams, device_index)
        # Allocated on the compute stream, which consumes it; the side stream waits for
        # that stream's earlier users of the block and for this record's D2H.
        try:
            x = self._empty_on_device(record.host, record.device)
        except torch.cuda.OutOfMemoryError:
            return  # unpack reloads it in place instead
        side.wait_stream(compute)
        if record.copied is not None: side.wait_event(record.copied)
        with self._stream_ctx(side):
            x.copy_(record.host, non_blocking = True)
            ready = self._new_event()
            ready.record(side)
        # No record_stream: compute waits on `ready` in unpack, or in __del__ if never read.
        record.device_tensor, record.ready, record.ready_stream = x, ready, compute
        self.stats["prefetched"] += 1
        self.stats["prefetched_bytes"] += record.nbytes
    pass

    def _prefetch_target(self, index, group, order = None):
        """The newest record before `index` still to be reloaded. A checkpoint with several saved
        inputs unpacks them back to back: skipping its own handed out inputs and the ones already
        prefetched makes each unpack queue one input of the previous checkpoint, so all of them
        are reloaded early, not only the one at index - 1. Single input checkpoints get index - 1.
        Never past the previous checkpoint, so at most its inputs wait on the device, and only
        index - 1 when the hooks were not entered per checkpoint (one group for every record)."""
        if order is None: order = self.order
        start = index - 1
        index = start
        while index >= 0:
            record = order[index]() if index < len(order) else None
            if record is not None and (record.group < group - 1 or (
                record.group == group and not record.unpacked and index != start
            )):
                return -1
            if record is None or not (record.unpacked or record.device_tensor is not None): break
            index -= 1
        return index
    pass

    def unpack(self, packed):
        if type(packed) is tuple:
            device, x = packed[0], packed[1]
            if device is None: return x
            if len(packed) == 3: self._count_restore(packed[2])
            return x.to(device, non_blocking = True)
        record = packed
        order = record.order
        if order is None: order = self.order
        if not order.backward_started: self._drain()
        order.backward_started = True
        record.unpacked = True
        device_index = record.device.index
        compute = self._current_stream(device_index)
        if record.device_tensor is not None:
            x = record.device_tensor
            if record.ready is not None:
                compute.wait_event(record.ready)
                # Consumed on another stream than it was allocated on: its free must wait for this one.
                if compute != record.ready_stream: self._record_stream(x, compute)
                # One shot: a retain_graph rerun reloads from the host copy.
                record.device_tensor, record.ready, record.ready_stream = None, None, None
                self.stats["prefetch_hits"] += 1
                self._count_restore(record.nbytes)
        else:
            if record.copied is not None: compute.wait_event(record.copied)
            x = self._device_copy(record.host, record.device)
            self._count_restore(record.nbytes)
        if self.stream_mode: self._prefetch(self._prefetch_target(record.index, record.group, order), order)
        return x
    pass

    def _count_restore(self, nbytes):
        stats = self.stats
        stats["restored"] += 1
        stats["restored_bytes"] += nbytes
    pass

    def hooks(self):
        # One call per checkpoint (vision.py wraps each checkpoint call): groups its saved inputs.
        self.group += 1
        return torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)
pass


NONREENTRANT_OFFLOADER = None
# Weak keys: the offloader never refers back to its owner.
_OWNED_OFFLOADERS = weakref.WeakKeyDictionary()
_OWNED_OFFLOADERS_LOCK = threading.Lock()


def _new_activation_offloader():
    return NonReentrantActivationOffloader()
pass


def _begin_forward_hook(module, args):
    # Module level and looked up per call: the model stays picklable and a copy never drives the original's.
    if _is_compiling(): return
    offloader = _OWNED_OFFLOADERS.get(module)
    if offloader is not None: offloader.begin_forward()
pass


def activation_offloader_for(owner):
    """
    All Unsloth Zoo code licensed under LGPLv3
    Per-owner offloader, created on first use, so two models never touch each other's saved inputs.
    """
    with _OWNED_OFFLOADERS_LOCK:
        offloader = _OWNED_OFFLOADERS.get(owner)
        if offloader is None:
            offloader = _new_activation_offloader()
            offloader.owned = hasattr(owner, "register_forward_pre_hook")
            # A copied or unpickled model already carries the hook.
            if offloader.owned and _begin_forward_hook not in getattr(owner, "_forward_pre_hooks", {}).values():
                owner.register_forward_pre_hook(_begin_forward_hook)
            _OWNED_OFFLOADERS[owner] = offloader
    return offloader
pass


def non_reentrant_offload_hooks(owner = None):
    """saved_tensors_hooks context that offloads non-reentrant checkpoint inputs to the host."""
    if owner is not None:
        return activation_offloader_for(owner).hooks()
    global NONREENTRANT_OFFLOADER
    if NONREENTRANT_OFFLOADER is None:
        NONREENTRANT_OFFLOADER = _new_activation_offloader()
    return NONREENTRANT_OFFLOADER.hooks()
pass


def get_activation_offload_stats(reset = False):
    """
    All Unsloth Zoo code licensed under LGPLv3
    Counters summed over every live non-reentrant offloader: tensors and bytes offloaded / restored, prefetches
    and their hits, and what stayed on the device and why (kept_small, kept_expanded).
    "forwards": owned = every model forward, unowned = once per inferred forward.
    `reset = True` zeroes them after reading.
    """
    total = _new_offload_stats()
    for offloader in list(_ALL_OFFLOADERS):
        for key, value in offloader.stats.items():
            total[key] = total.get(key, 0) + value
        if reset:
            offloader.stats = _new_offload_stats()
    return total
pass


from torch.utils.checkpoint import (
    ContextManager,
    _DEFAULT_DETERMINISM_MODE,
    _checkpoint_without_reentrant_generator,
    noop_context_fn,
)
@torch._disable_dynamo
def unsloth_checkpoint(
    function,
    *args,
    use_reentrant: Optional[bool] = None,
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    determinism_check: str = _DEFAULT_DETERMINISM_MODE,
    debug: bool = False,
    **kwargs
):
    r"""Activation checkpoint a model or part of the model.

    Trades compute for memory: forward in checkpointed regions omits saving
    tensors for backward and recomputes them during the backward pass.

    This is Unsloth's drop-in for ``torch.utils.checkpoint.checkpoint``; it
    forces ``use_reentrant=True`` to route through UnslothCheckpointFunction
    (smart CPU offloading). The ``context_fn`` / ``determinism_check`` /
    ``debug`` args are only valid with ``use_reentrant=False`` and raise here.

    .. warning::

        If :attr:`function` behaves differently in backward vs forward (e.g.
        via a global), recomputation may error or yield wrong gradients.

    Args:
        function: forward pass to run; must handle the passed input tuple.
        args: inputs to :attr:`function`.

    Returns:
        Output of running :attr:`function` on :attr:`*args`.
    """
    pristine = _pristine_for_compiled_checkpoint(
        use_reentrant, dict(kwargs, context_fn = context_fn),
    )
    if pristine is not None:
        return pristine(
            function, *args, use_reentrant = False, context_fn = context_fn,
            determinism_check = determinism_check, debug = debug, **kwargs
        )
    # Force use_reentrant=True so UnslothCheckpointFunction (smart CPU offloading)
    # is always used. This is safe because unsloth_checkpoint is only active when
    # smart GC is patched; when unpatched, the original torch checkpoint is restored.
    # Fixes transformers 5.2 which defaults use_reentrant=False, bypassing Unsloth.
    use_reentrant = True

    # Hack to mix *args with **kwargs in a python 2.7-compliant way
    preserve = kwargs.pop("preserve_rng_state", True)
    if kwargs and use_reentrant:
        # torch raises here, but only because *the caller* asked for the
        # reentrant path. We force use_reentrant = True above, so raising would
        # turn a call that works on unpatched torch (non-reentrant checkpoint
        # forwards **kwargs straight to `function`) into a crash the moment
        # Unsloth's smart gradient checkpointing is installed. Bind them into
        # the callable instead - same observable behaviour, no keyword reaches
        # UnslothCheckpointFunction. See _bind_checkpoint_kwargs.
        function, extra_args = _bind_checkpoint_kwargs(function, kwargs)
        args = args + extra_args
        kwargs = {}

    if use_reentrant:
        if context_fn is not noop_context_fn or debug is not False:
            raise ValueError(
                "Passing `context_fn` or `debug` is only supported when "
                "use_reentrant=False."
            )
        if not torch.is_grad_enabled():
            return _call_without_checkpoint(function, *args)
        return UnslothCheckpointFunction.apply(function, preserve, *args)
    else:
        gen = _checkpoint_without_reentrant_generator(
            function, preserve, context_fn, determinism_check, debug, *args, **kwargs
        )
        # Runs pre-forward logic
        next(gen)
        ret = function(*args, **kwargs)
        # Runs post-forward logic
        try:
            next(gen)
        except StopIteration:
            return ret
pass


def patch_unsloth_smart_gradient_checkpointing(dtype = None):
    # All Unsloth Zoo code licensed under LGPLv3
    if torch.utils.checkpoint.CheckpointFunction.__name__ != "UnslothCheckpointFunction":
        initialize_unsloth_gradient_checkpointing(dtype)
        torch.utils.checkpoint._old_CheckpointFunction = torch.utils.checkpoint.CheckpointFunction
        torch.utils.checkpoint.CheckpointFunction = UnslothCheckpointFunction

    if torch.utils.checkpoint.checkpoint.__name__ != "unsloth_checkpoint":
        _capture_pristine_checkpoint_once()
        torch.utils.checkpoint._old_checkpoint = torch.utils.checkpoint.checkpoint
        torch.utils.checkpoint.checkpoint = unsloth_checkpoint

    # Always patch transformers.modeling_utils.checkpoint so
    # gradient_checkpointing_enable() wraps unsloth_checkpoint; otherwise
    # transformers 5.2's use_reentrant=False default bypasses
    # UnslothCheckpointFunction. Outside the conditional above since
    # torch.utils.checkpoint may already be patched while this one is not.
    import transformers.modeling_utils
    transformers.modeling_utils.checkpoint = unsloth_checkpoint
    # Configuration only: which devices keep anything is decided per checkpoint frame.
    if save_attention_requested():
        patch_transformers_saved_attention()
pass


def unpatch_unsloth_smart_gradient_checkpointing():
    # All Unsloth Zoo code licensed under LGPLv3
    if (torch.utils.checkpoint.CheckpointFunction.__name__ == "UnslothCheckpointFunction") and \
        hasattr(torch.utils.checkpoint, "_old_CheckpointFunction"):

        torch.utils.checkpoint.CheckpointFunction = torch.utils.checkpoint._old_CheckpointFunction
        global CPU_BUFFERS
        global GPU_BUFFERS
        global GPU_BUFFERS_B
        global USE_DOUBLE_BUFFER
        global BUFFER_EVENTS_A
        global BUFFER_EVENTS_B
        global NEXT_BUFFER_SLOT
        _release_pool_pinned()
        for i in range(len(CPU_BUFFERS)):
            if hasattr(CPU_BUFFERS[i], "resize_"): CPU_BUFFERS[i].resize_(0)
            if type(CPU_BUFFERS) is list: CPU_BUFFERS[i] = None
        # Only devices that were used: indexing an untouched slot would allocate it.
        for buffer in _allocated_slots(GPU_BUFFERS):
            if hasattr(buffer, "resize_"): buffer.resize_(0)
        # resize_(0) keeps the storage: drop the loop reference so empty_cache() below can release it.
        buffer = None
        if GPU_BUFFERS_B is not None:
            _free_gpu_buffers_b()
        CPU_BUFFERS = None
        _reset_cpu_slots()
        GPU_BUFFERS = None
        BUFFER_EVENTS_A = None
        BUFFER_EVENTS_B = None
        NEXT_BUFFER_SLOT = None
        # Collect first so tensors only reachable through cycles are freed before the cache is emptied.
        gc.collect()
        if DEVICE_TYPE == "npu": torch.npu.empty_cache()
        elif DEVICE_TYPE == "xpu": torch.xpu.empty_cache()
        else: torch.cuda.empty_cache()

    if (torch.utils.checkpoint.checkpoint.__name__ == "unsloth_checkpoint") and \
        hasattr(torch.utils.checkpoint, "_old_checkpoint"):

        torch.utils.checkpoint.checkpoint = torch.utils.checkpoint._old_checkpoint

    # Restore transformers.modeling_utils.checkpoint independently: an earlier
    # unpatch_unsloth_gradient_checkpointing() (e.g. training_utils.py:201) may
    # have deleted _old_checkpoint and restored torch.utils.checkpoint,
    # making the condition above False. Prefer _old_checkpoint, else the
    # already-restored torch.utils.checkpoint.checkpoint.
    import transformers.modeling_utils
    if getattr(transformers.modeling_utils, "checkpoint", None) is unsloth_checkpoint:
        transformers.modeling_utils.checkpoint = getattr(
            torch.utils.checkpoint, "_old_checkpoint",
            torch.utils.checkpoint.checkpoint
        )
pass


def reset_unsloth_gradient_checkpointing_buffers():
    """
    All Unsloth Zoo code licensed under LGPLv3

    Reset CPU_BUFFERS and GPU_BUFFERS to their initial sizes after training.

    Call after trainer.train() to free training-allocated memory while keeping
    buffers ready for another run. Unlike unpatch_unsloth_smart_gradient_checkpointing,
    this neither destroys the buffers nor unpatches checkpointing.
    """
    global CPU_BUFFERS
    global GPU_BUFFERS
    global CPU_INDEX
    global BACKWARD_PASS
    global LAST_GC_INDEX
    global FIRST_PASS
    global CURRENT_GC_INDEX
    global USE_UNSLOTH_GC
    global NEXT_BUFFER_SLOT
    global GPU_BUFFERS_B
    global USE_DOUBLE_BUFFER
    global BUFFER_EVENTS_A
    global BUFFER_EVENTS_B

    # GPU_BUFFERS is the "initialized" sentinel. An empty CPU_BUFFERS is a valid initialized pool (every
    # device keeps its inputs), and its run state (first pass, last checkpoint index) still has to be reset.
    if CPU_BUFFERS is None or GPU_BUFFERS is None:
        return

    _release_pool_pinned(INITIAL_CPU_BUFFER_COUNT)
    # Reset CPU buffers to initial size; free any added during training
    for i in range(len(CPU_BUFFERS)):
        if i < INITIAL_CPU_BUFFER_COUNT:
            if CPU_BUFFERS[i] is not None and hasattr(CPU_BUFFERS[i], "resize_"):
                CPU_BUFFERS[i].resize_(INITIAL_CPU_BUFFER_SIZE)
                # resize_ keeps the object and its pinning, but re-read rather than assume.
                _mark_host_buffer(CPU_BUFFERS[i])
        else:
            if CPU_BUFFERS[i] is not None and hasattr(CPU_BUFFERS[i], "resize_"):
                CPU_BUFFERS[i].resize_(0)
            CPU_BUFFERS[i] = None
    pass

    # Trim the list back to initial count if it grew
    if len(CPU_BUFFERS) > INITIAL_CPU_BUFFER_COUNT:
        del CPU_BUFFERS[INITIAL_CPU_BUFFER_COUNT:]
    pass

    # Reset GPU buffers to initial size
    for buffer in _allocated_slots(GPU_BUFFERS):
        if hasattr(buffer, "resize_"):
            buffer.resize_(INITIAL_GPU_BUFFER_SIZE)
    pass

    # Reset state for a fresh training run
    CPU_INDEX = 0
    _reset_cpu_slots()
    BACKWARD_PASS = True
    _reset_gc_prefetch_state()
    LAST_GC_INDEX = 0
    FIRST_PASS = True
    CURRENT_GC_INDEX = 0
    USE_UNSLOTH_GC = True  # re-enable the "Will smartly offload" message
    if NEXT_BUFFER_SLOT is not None:
        for i in range(len(NEXT_BUFFER_SLOT)):
            NEXT_BUFFER_SLOT[i] = 0

    # Reset double buffering if buffer B still exists, or try to re-allocate
    if _double_buffer_disabled():
        if GPU_BUFFERS_B is not None:
            _free_gpu_buffers_b()
        USE_DOUBLE_BUFFER = False
    elif GPU_BUFFERS_B is not None:
        for buffer in _allocated_slots(GPU_BUFFERS_B):
            if hasattr(buffer, "resize_"):
                buffer.resize_(INITIAL_GPU_BUFFER_SIZE)
        USE_DOUBLE_BUFFER = False
    else:
        try:
            global GPU_BUFFER_DTYPE
            if GPU_BUFFER_DTYPE is None and _allocated_slots(GPU_BUFFERS):
                GPU_BUFFER_DTYPE = _allocated_slots(GPU_BUFFERS)[0].dtype
            n_gpus = len(GPU_BUFFERS)
            GPU_BUFFERS_B = _PerDeviceSlots(n_gpus, _new_gpu_buffer)
            event_ctor = _double_buffer_event_ctor()
            BUFFER_EVENTS_A = tuple([event_ctor() for _ in range(n_gpus)])
            BUFFER_EVENTS_B = tuple([event_ctor() for _ in range(n_gpus)])
            USE_DOUBLE_BUFFER = False
        except RuntimeError:
            pass

    # Collect first so tensors only reachable through cycles are freed before the cache is emptied.
    gc.collect()
    if DEVICE_TYPE == "npu": torch.npu.empty_cache()
    elif DEVICE_TYPE == "xpu": torch.xpu.empty_cache()
    else: torch.cuda.empty_cache()
pass


@torch._disable_dynamo
def unsloth_offloaded_gradient_checkpoint(function, *args, use_reentrant = None, **kwargs):
    global CPU_BUFFERS
    pristine = _pristine_for_compiled_checkpoint(use_reentrant, kwargs)
    if pristine is not None:
        return pristine(function, *args, use_reentrant = False, **kwargs)
    preserve = kwargs.pop("preserve_rng_state", True)
    function, tensor_args = _bind_checkpoint_kwargs(function, kwargs)
    if not torch.is_grad_enabled():
        return _call_without_checkpoint(function, *args, *tensor_args)
    # Test None, not emptiness: CPU_BUFFERS stays [] when all devices keep inputs (by design).
    if CPU_BUFFERS is None or GPU_BUFFERS is None:
        initialize_unsloth_gradient_checkpointing(args[0].dtype)
    return UnslothCheckpointFunction.apply(function, preserve, *args, *tensor_args)

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
