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
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Selective gradient checkpointing: with UNSLOTH_GC_SAVE_ATTENTION=1 (GPU) or =offload (pinned
host), the first pass keeps each attention kernel's outputs (output, logsumexp, rng state) and the
backward replay of the same checkpointed call reuses them instead of recomputing attention. Kernels
the dispatcher cannot see, mismatched replays and random (dropout) calls are recomputed as before.

Unset (or =auto): Unsloth's attention dispatch routes long training calls here on measured GPUs only
(save_attention_auto_min_seq), and each output is kept only while the device memory budget allows
it. =0 turns everything off."""

import collections
import os
import threading
import weakref
from contextlib import contextmanager

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map_only, tree_unflatten

__all__ = [
    "SAVED_ATTENTION_STATS",
    "SavedAttentionFrame",
    "save_attention_enabled",
    "save_attention_offload",
    "save_attention_auto_min_seq",
    "new_saved_attention_frame",
    "saved_attention_scope",
    "saved_attention_active",
    "saved_attention_call",
    "saved_function_call",
    "patch_transformers_saved_attention",
]

SAVED_ATTENTION_STATS = collections.Counter()

_TLS = threading.local()


def _save_attention_setting():
    return os.environ.get("UNSLOTH_GC_SAVE_ATTENTION", "auto").strip().lower() or "auto"


_WARNED_HIP = [False]


def save_attention_enabled():
    if _save_attention_setting() not in ("1", "true", "yes", "on", "offload"):
        return False
    # ROCm SDPA kernels are not intercepted: nothing is kept, only graph breaks are added.
    if getattr(torch.version, "hip", None):
        if not _WARNED_HIP[0]:
            _WARNED_HIP[0] = True
            print("Unsloth: UNSLOTH_GC_SAVE_ATTENTION is not supported on ROCm yet and is ignored.")
        return False
    return True


def save_attention_offload():
    return _save_attention_setting() == "offload"


# Automatic mode: compute capability -> the shortest query length where Colab measured it faster
# than the A/A noise (Qwen3-4B LoRA): T4 and A100 from 8K, L4 from 16K (noisy and +1.6% at 8K).
_AUTO_MIN_SEQ = {(7, 5): 8192, (8, 0): 8192, (8, 9): 16384}
_AUTO_ACTIVE = [False]


def save_attention_auto_min_seq():
    """Automatic mode: the query length from which training attention calls should go through
    saved_attention_call, or None when the mode is off (=0, an explicit setting, ROCm, CPU, an
    unmeasured GPU). A caller that routes calls marks the mode active, which turns on the frames."""
    if _save_attention_setting() != "auto":
        return None
    if getattr(torch.version, "hip", None) or not torch.cuda.is_available():
        return None
    try:
        capability = tuple(torch.cuda.get_device_capability())
    except Exception:
        return None
    min_seq = _AUTO_MIN_SEQ.get(capability)
    if min_seq is None:
        return None
    try:
        min_seq = int(os.environ.get("UNSLOTH_GC_SAVE_ATTENTION_MIN_SEQ", min_seq))
    except ValueError:
        pass
    _AUTO_ACTIVE[0] = True
    return max(min_seq, 1)


class _Budget:
    """Kept bytes on one device against the headroom left by the run's own peak.

    The peak without kept outputs is learnt from the allocator's running peak (never reset): every
    change of the kept bytes is preceded by a reading, so a new peak is charged with exactly the
    bytes kept at that time. Nothing is kept for a call larger than any call already seen through a
    whole step (its peak is not known yet)."""
    __slots__ = ("live", "seen_peak", "base_peak", "measured", "capacity")

    def __init__(self):
        self.live = 0
        self.seen_peak = None
        self.base_peak = None
        self.measured = 0
        self.capacity = None


_BUDGETS = {}
_AUTO_MARGIN_BYTES = 1 << 30
_AUTO_MARGIN_FRACTION = 0.06


def _device_peak(device):
    if device.type != "cuda":
        return None
    return torch.cuda.memory_stats_as_nested_dict(device)["allocated_bytes"]["all"]["peak"]


def _device_capacity(device):
    if device.type != "cuda":
        return None
    free, total = torch.cuda.mem_get_info(device)
    capacity = free + torch.cuda.memory_reserved(device)
    get_fraction = getattr(torch.cuda, "get_per_process_memory_fraction", None)
    if get_fraction is not None:
        try:
            capacity = min(capacity, int(total * get_fraction(device)))
        except Exception:
            pass
    return capacity


def _observe(budget, device):
    peak = _device_peak(device)
    if peak is None:
        return False
    if peak != budget.seen_peak:
        # Also after a reset of the peak stats: the new peak is still charged with today's kept bytes.
        without = peak - budget.live
        budget.base_peak = without if budget.base_peak is None else max(budget.base_peak, without)
        budget.seen_peak = peak
    return True


def _auto_admit(frame, tensors):
    query = tensors[0] if tensors else None
    if not isinstance(query, torch.Tensor) or query.dim() < 2:
        return False
    # Output like Q plus a float32 logsumexp per query row.
    estimate = _bytes(query) + (query.numel() // max(query.shape[-1], 1)) * 4
    frame.estimate[query.device] = max(frame.estimate.get(query.device, 0), estimate)
    budget = _BUDGETS.get(query.device)
    if budget is None:
        budget = _BUDGETS[query.device] = _Budget()
    if not _observe(budget, query.device):
        return False
    if budget.live == 0 or budget.capacity is None:
        budget.capacity = _device_capacity(query.device)
    if estimate > budget.measured:
        SAVED_ATTENTION_STATS["declined_unmeasured"] += 1
        return False
    capacity = budget.capacity
    margin = max(_AUTO_MARGIN_BYTES, int(capacity * _AUTO_MARGIN_FRACTION))
    if budget.live + estimate > capacity - margin - budget.base_peak:
        SAVED_ATTENTION_STATS["declined_budget"] += 1
        return False
    return True


def _auto_keep(frame, entry):
    device = None
    kept = 0
    for _, _, out in entry:
        for t in tree_flatten(out)[0]:
            if isinstance(t, torch.Tensor):
                device = t.device if device is None else device
                kept += _bytes(t)
    budget = _BUDGETS.get(device) if device is not None else None
    if budget is None:
        return
    # The kernel may have raised the peak: charge it before the kept bytes grow.
    _observe(budget, device)
    budget.live += kept
    frame.kept.append((device, kept))
    SAVED_ATTENTION_STATS["kept_bytes"] += kept


def _auto_release(frame, measured):
    # Without a reading first, a later peak reading is charged with fewer kept bytes than it held,
    # which only overstates the run's own peak: safe, but the replay reads first to stay exact.
    for device, kept in frame.kept:
        budget = _BUDGETS[device]
        if measured:
            _observe(budget, device)
        budget.live = max(budget.live - kept, 0)
    frame.kept.clear()
    # Per device: a replay only measures the peaks of the devices its calls ran on.
    if measured:
        for device, estimate in frame.estimate.items():
            budget = _BUDGETS.get(device)
            if budget is not None:
                budget.measured = max(budget.measured, estimate)


class SavedAttentionFrame:
    __slots__ = ("entries", "cursor", "replaying", "offload", "loaded", "auto", "kept", "estimate", "__weakref__")

    def __init__(self, auto = False):
        self.entries = []
        self.cursor = 0
        self.replaying = False
        self.offload = save_attention_offload()
        self.loaded = False
        self.auto = auto
        self.kept = []
        self.estimate = {}

    def __del__(self):
        try:
            if self.kept:
                _auto_release(self, False)
        except Exception:
            pass


def new_saved_attention_frame(tensors):
    if save_attention_enabled():
        auto = False
    elif _AUTO_ACTIVE[0] and _save_attention_setting() == "auto":
        auto = True
    else:
        return None
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.requires_grad:
            return SavedAttentionFrame(auto)
    return None


@contextmanager
def saved_attention_scope(frame, replay):
    previous = getattr(_TLS, "frame", None)
    if frame is not None:
        frame.replaying = replay
        frame.cursor = 0
        if replay and frame.auto:
            # From here the kept outputs stand in for the ones the replay would allocate anyway.
            _auto_release(frame, True)
        if frame.offload:
            if replay:
                _reload_frame(frame)
                _prefetch_before(frame)
            else:
                _OFFLOADED_FRAMES.append(weakref.ref(frame))
    _TLS.frame = frame
    try:
        yield
    finally:
        _TLS.frame = previous
        if frame is not None and replay:
            left = sum(1 for entry in frame.entries if entry)
            if left:
                SAVED_ATTENTION_STATS["unmatched"] += left
            # A second backward over a retained graph finds no entries and recomputes.
            frame.entries.clear()
            if frame.offload:
                _forget_frame(frame)


def saved_attention_active():
    """Runtime state: never read it in traced code."""
    return getattr(_TLS, "frame", None) is not None


_ATEN_ATTENTION_NAMES = (
    "_scaled_dot_product_flash_attention",
    "_scaled_dot_product_efficient_attention",
    "_scaled_dot_product_cudnn_attention",
    "_scaled_dot_product_flash_attention_for_cpu",
    "_scaled_dot_product_fused_attention_overrideable",
    "_flash_attention_forward",
    "_efficient_attention_forward",
)
_OTHER_NAMESPACES = ("flash_attn", "torch_attn")
_IS_ATTENTION = {}


def _aten_attention_ops():
    ops = set()
    for name in _ATEN_ATTENTION_NAMES:
        packet = getattr(torch.ops.aten, name, None)
        if packet is not None and hasattr(packet, "default"):
            ops.add(packet.default)
    return ops


_ATEN_ATTENTION_OPS = _aten_attention_ops()


def _is_attention_forward(func):
    found = _IS_ATTENTION.get(func)
    if found is not None:
        return found
    found = func in _ATEN_ATTENTION_OPS
    if not found:
        namespace = str(getattr(func, "namespace", ""))
        name = str(getattr(func, "__name__", func)).lower()
        found = any(ns in namespace for ns in _OTHER_NAMESPACES) and (
            "fwd" in name or "forward" in name or "varlen_attn" in name
        ) and not ("bwd" in name or "backward" in name)
    if found:
        schema = getattr(func, "_schema", None)
        # Out= / in-place variants write into a caller's buffer.
        if schema is not None and getattr(schema, "is_mutable", False):
            found = False
    _IS_ATTENTION[func] = found
    return found


def _signature(args, kwargs):
    leaves, _ = tree_flatten((args, kwargs))
    key = []
    for leaf in leaves:
        if isinstance(leaf, torch.Tensor):
            key.append((tuple(leaf.shape), leaf.dtype, leaf.device))
        elif leaf is None or isinstance(leaf, (bool, int, float, str)):
            key.append(leaf)
        else:
            key.append(type(leaf).__name__)
    return tuple(key)


def _detach(t):
    return t.detach()


def _rng_state(tensors):
    devices = sorted({t.device.index for t in tensors if isinstance(t, torch.Tensor) and t.is_cuda})
    return [torch.get_rng_state()] + [torch.cuda.get_rng_state(index) for index in devices]


def _rng_advanced(before, tensors):
    # A kept output skips this call's draws in the replay, shifting every later random op of the layer.
    return any(not torch.equal(a, b) for a, b in zip(before, _rng_state(tensors)))


_OFFLOAD_MIN_BYTES = 1 << 20
_OFFLOADED_FRAMES = []  # weakrefs, first-pass order
_SIDE_STREAMS = {}


class _OnHost:
    __slots__ = ("host", "device")

    def __init__(self, host, device):
        self.host = host
        self.device = device


class _Reloading:
    __slots__ = ("tensor", "event")

    def __init__(self, tensor, event):
        self.tensor = tensor
        self.event = event


def _side_stream(device):
    stream = _SIDE_STREAMS.get(device.index)
    if stream is None:
        stream = torch.cuda.Stream(device = device)
        _SIDE_STREAMS[device.index] = stream
    return stream


def _to_host(t):
    if not isinstance(t, torch.Tensor) or not t.is_cuda or t.numel() * t.element_size() < _OFFLOAD_MIN_BYTES:
        return t
    stream = _side_stream(t.device)
    stream.wait_stream(torch.cuda.current_stream(t.device))
    host = torch.empty_like(t, device = "cpu", pin_memory = True)
    with torch.cuda.stream(stream):
        host.copy_(t, non_blocking = True)
    t.record_stream(stream)
    SAVED_ATTENTION_STATS["offloaded_bytes"] += t.numel() * t.element_size()
    return _OnHost(host, t.device)


def _to_device(item):
    if not isinstance(item, _OnHost):
        return item
    stream = _side_stream(item.device)
    with torch.cuda.stream(stream):
        gpu = torch.empty_like(item.host, device = item.device)
        gpu.copy_(item.host, non_blocking = True)
        event = torch.cuda.Event()
        event.record(stream)
    return _Reloading(gpu, event)


def _ready(item):
    if isinstance(item, _OnHost):
        item = _to_device(item)
    if isinstance(item, _Reloading):
        current = torch.cuda.current_stream(item.tensor.device)
        current.wait_event(item.event)
        item.tensor.record_stream(current)
        return item.tensor
    return item


def _map_outputs(fn, out):
    if isinstance(out, (tuple, list)):
        return type(out)(_map_outputs(fn, o) for o in out)
    return fn(out)


def _offload_entry(entry):
    for i, (func, sig, out) in enumerate(entry):
        entry[i] = (func, sig, _map_outputs(_to_host, out))


def _reload_frame(frame):
    if frame.loaded:
        return
    frame.loaded = True
    for found in frame.entries:
        if found and found[0] == "kernel":
            entry = found[1]
            for i, item in enumerate(entry):
                if item is not None:
                    func, sig, out = item
                    entry[i] = (func, sig, _map_outputs(_to_device, out))


def _prefetch_before(frame):
    previous = None
    for ref in _OFFLOADED_FRAMES:
        other = ref()
        if other is frame:
            break
        if other is not None and other.entries:
            previous = other
    if previous is not None and not previous.replaying:
        _reload_frame(previous)
        SAVED_ATTENTION_STATS["prefetched"] += 1


def _forget_frame(frame):
    _OFFLOADED_FRAMES[:] = [ref for ref in _OFFLOADED_FRAMES if ref() is not None and ref() is not frame]


class _StoreMode(TorchDispatchMode):
    def __init__(self, entry):
        super().__init__()
        self.entry = entry
        self.random = False

    def __torch_dispatch__(self, func, types, args = (), kwargs = None):
        kwargs = kwargs or {}
        if not _is_attention_forward(func):
            return func(*args, **kwargs)
        flat = tree_flatten((args, kwargs))[0]
        before = _rng_state(flat)
        out = func(*args, **kwargs)
        if _rng_advanced(before, flat):
            self.random = True
        else:
            self.entry.append((func, _signature(args, kwargs), tree_map_only(torch.Tensor, _detach, out)))
        return out


class _ReplayMode(TorchDispatchMode):
    def __init__(self, entry):
        super().__init__()
        self.entry = entry
        self.cursor = 0
        self.failed = False

    def __torch_dispatch__(self, func, types, args = (), kwargs = None):
        kwargs = kwargs or {}
        if not self.failed and _is_attention_forward(func):
            entry = self.entry
            if self.cursor < len(entry) and entry[self.cursor][0] is func \
                    and entry[self.cursor][1] == _signature(args, kwargs):
                cached = _map_outputs(_ready, entry[self.cursor][2])
                entry[self.cursor] = None
                self.cursor += 1
                SAVED_ATTENTION_STATS["reused"] += 1
                return tree_map_only(torch.Tensor, _detach, cached)
            self.failed = True
            SAVED_ATTENTION_STATS["mismatch"] += 1
        return func(*args, **kwargs)


def _first_pass_leaves(tensors):
    # Match the replay's grad branches (memory-efficient attention only returns lse when grad is needed).
    leaves = []
    for t in tensors:
        if isinstance(t, torch.Tensor) and (t.is_floating_point() or t.is_complex()):
            t = t.detach().requires_grad_(True)
        leaves.append(t)
    return leaves


def _take_entry(frame):
    if frame.cursor >= len(frame.entries):
        return None
    entry = frame.entries[frame.cursor]
    frame.entries[frame.cursor] = None
    frame.cursor += 1
    return entry


@torch.compiler.disable
def saved_attention_call(function, *tensors):
    """``function(*tensors)``, keeping its attention kernel outputs for the checkpoint replay."""
    frame = getattr(_TLS, "frame", None)
    if frame is None:
        return function(*tensors)
    if not frame.replaying:
        # Grad mode here is not the loss forward (block compile's trial run, nested regions).
        if torch.is_grad_enabled():
            return function(*tensors)
        if frame.auto and not _auto_admit(frame, tensors):
            frame.entries.append(None)
            return function(*tensors)
        entry = []
        mode = _StoreMode(entry)
        with torch.enable_grad(), mode:
            out = function(*_first_pass_leaves(tensors))
        out = tree_map_only(torch.Tensor, _detach, out)
        if mode.random:
            # Dropout: recompute so the replay draws the same.
            frame.entries.append(None)
            SAVED_ATTENTION_STATS["random"] += 1
        elif entry:
            if frame.offload:
                _offload_entry(entry)
            frame.entries.append(("kernel", entry))
            if frame.auto:
                _auto_keep(frame, entry)
            SAVED_ATTENTION_STATS["stored"] += len(entry)
        else:
            frame.entries.append(None)
            SAVED_ATTENTION_STATS["no_kernel"] += 1
        return out
    if not torch.is_grad_enabled():
        return function(*tensors)
    entry = _take_entry(frame)
    if not entry or entry[0] != "kernel":
        return function(*tensors)
    mode = _ReplayMode(entry[1])
    with mode:
        out = function(*tensors)
    left = sum(1 for e in entry[1] if e is not None)
    if left:
        SAVED_ATTENTION_STATS["unmatched"] += left
    return out


def _extent(t):
    lo = t.storage_offset()
    return lo, lo + 1 + sum((n - 1) * st for n, st in zip(t.shape, t.stride()) if n > 0)


def _same_view_of(t, leaf):
    if t.device != leaf.device or t.dtype != leaf.dtype or t.numel() == 0 or leaf.numel() == 0:
        return False
    if t.untyped_storage().data_ptr() != leaf.untyped_storage().data_ptr():
        return False
    lo, hi = _extent(t)
    leaf_lo, leaf_hi = _extent(leaf)
    return leaf_lo <= lo and hi <= leaf_hi


class _FunctionEntry:
    __slots__ = ("outputs", "leaves", "layouts", "replay", "spec", "kept_bytes")

    def __init__(self):
        self.outputs = None
        self.leaves = None
        self.layouts = None
        self.replay = None
        self.spec = None
        self.kept_bytes = 0


_TENSOR_SLOT = object()
_FUNCTION_SLACK_BYTES = 4 << 20


def _bytes(t):
    return t.numel() * t.element_size()


class _ReplayFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, entry, *inputs):
        ctx.set_materialize_grads(False)
        ctx.entry = entry
        ctx.save_for_backward(*inputs)
        # Detached: the kept graph must not join the outer graph.
        return tuple(o.detach() for o in entry.outputs)

    @staticmethod
    def backward(ctx, *grads):
        entry = ctx.entry
        ctx.entry = None
        replay_inputs = ctx.saved_tensors
        replay = []
        for (shape, stride, _, _), x in zip(entry.layouts, replay_inputs):
            if x.stride() != stride:
                # The kept graph addresses Q / K / V through the first pass's strides.
                y = torch.empty_strided(shape, stride, dtype = x.dtype, device = x.device)
                y.copy_(x)
                x = y
            replay.append(x.detach())
        entry.replay = replay
        differentiable = [leaf is not None and x.requires_grad for leaf, x in zip(entry.leaves, replay_inputs)]
        pairs = [(o, g) for o, g in zip(entry.outputs, grads) if g is not None and o.requires_grad]
        wanted = [leaf for leaf, d in zip(entry.leaves, differentiable) if d]
        if not pairs or not wanted:
            entry.replay = None
            entry.outputs = None
            return (None,) + tuple(None for _ in replay_inputs)
        found = torch.autograd.grad(
            [o for o, _ in pairs], wanted, [g for _, g in pairs], allow_unused = True,
        )
        entry.replay = None
        entry.outputs = None
        entry.leaves = None
        found = iter(found)
        input_grads = tuple(next(found) if d else None for d in differentiable)
        return (None,) + input_grads


@torch.compiler.disable
def saved_function_call(function, n_inputs, *args, **kwargs):
    """Keeps the first pass's autograd graph with Q / K / V (``args[:n_inputs]``) as placeholders;
    backward runs it on the replay's Q / K / V. For kernels the dispatcher cannot see (FlexAttention)."""
    frame = getattr(_TLS, "frame", None)
    if frame is None:
        return function(*args, **kwargs)
    inputs = args[:n_inputs]
    if not frame.replaying:
        if torch.is_grad_enabled() or not all(isinstance(x, torch.Tensor) for x in inputs):
            return function(*args, **kwargs)
        if frame.offload:
            # A kept graph cannot move to the host; keeping it would pin every layer's output on the device.
            frame.entries.append(None)
            SAVED_ATTENTION_STATS["not_offloaded"] += 1
            return function(*args, **kwargs)
        entry = _FunctionEntry()
        leaves = _first_pass_leaves(inputs)
        kept = []

        def pack(t):
            for i, leaf in enumerate(leaves):
                if _same_view_of(t, leaf):
                    return ("leaf", i, t.shape, t.stride(), t.storage_offset() - leaf.storage_offset())
            kept.append(t)
            return t

        def unpack(p):
            if isinstance(p, tuple) and len(p) == 5 and p[0] == "leaf":
                _, i, shape, stride, offset = p
                base = entry.replay[i]
                return base.as_strided(shape, stride, base.storage_offset() + offset)
            return p

        flat_args = tree_flatten((args, kwargs))[0]
        before = _rng_state(flat_args)
        with torch.enable_grad(), torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            out = function(*leaves, *args[n_inputs:], **kwargs)
        random = _rng_advanced(before, flat_args)
        flat, spec = tree_flatten(out)
        outputs = [o for o in flat if isinstance(o, torch.Tensor)]
        template = [_TENSOR_SLOT if isinstance(o, torch.Tensor) else o for o in flat]
        out_bytes = sum(_bytes(o) for o in outputs)
        seen = set()
        kept_bytes = 0
        for t in kept:
            key = (t.untyped_storage().data_ptr(), t.dtype)
            if key not in seen:
                seen.add(key)
                kept_bytes += t.untyped_storage().nbytes()
        detached = tree_map_only(torch.Tensor, _detach, out)
        if random:
            frame.entries.append(None)
            SAVED_ATTENTION_STATS["random"] += 1
            return detached
        # A graph holding expanded K / V or score matrices is not worth keeping.
        if not outputs or not any(o.requires_grad for o in outputs) or \
                kept_bytes > out_bytes + out_bytes // 4 + _FUNCTION_SLACK_BYTES:
            frame.entries.append(None)
            SAVED_ATTENTION_STATS["too_big" if outputs else "no_kernel"] += 1
            return detached
        entry.outputs = outputs
        entry.layouts = [(x.shape, x.stride(), x.dtype, x.device) for x in inputs]
        for i, (leaf, x) in enumerate(zip(leaves, inputs)):
            if leaf is x:
                leaves[i] = None
            else:
                # The kept graph holds the leaf (AccumulateGrad), which shares Q / K / V storage: swap in
                # a stand-in so it keeps its autograd identity but not the storage.
                leaf.data = torch.empty((), dtype = leaf.dtype, device = leaf.device).expand(leaf.shape)
        kept.clear()
        entry.leaves = leaves
        entry.spec = (template, spec)
        entry.kept_bytes = kept_bytes
        frame.entries.append(("function", entry))
        SAVED_ATTENTION_STATS["stored"] += 1
        return detached
    if not torch.is_grad_enabled():
        return function(*args, **kwargs)
    found = _take_entry(frame)
    if not found or found[0] != "function":
        return function(*args, **kwargs)
    entry = found[1]
    if len(inputs) != len(entry.layouts) or any(
        not isinstance(x, torch.Tensor) or x.shape != shape or x.dtype != dtype or x.device != device
        for x, (shape, _, dtype, device) in zip(inputs, entry.layouts)
    ):
        SAVED_ATTENTION_STATS["mismatch"] += 1
        return function(*args, **kwargs)
    SAVED_ATTENTION_STATS["reused"] += 1
    outputs = iter(_ReplayFunction.apply(entry, *inputs))
    flat = [next(outputs) if leaf is _TENSOR_SLOT else leaf for leaf in entry.spec[0]]
    return tree_unflatten(flat, entry.spec[1])


_PATCHED_MARK = "_unsloth_saved_attention"


def _wrap_attention_interface(fn):
    if getattr(fn, _PATCHED_MARK, False):
        return fn
    import functools

    @functools.wraps(fn)
    def attention_forward(module, query, key, value, *args, **kwargs):
        if not getattr(module, "training", False):
            return fn(module, query, key, value, *args, **kwargs)
        return saved_attention_call(
            lambda q, k, v: fn(module, q, k, v, *args, **kwargs), query, key, value,
        )

    setattr(attention_forward, _PATCHED_MARK, True)
    return attention_forward


def _wrap_flex_kernel(fn):
    if getattr(fn, _PATCHED_MARK, False):
        return fn
    import functools

    @functools.wraps(fn)
    def flex_kernel(query, key, value, *args, **kwargs):
        if not kwargs.get("training", True):
            return fn(query, key, value, *args, **kwargs)
        return saved_function_call(fn, 3, query, key, value, *args, **kwargs)

    setattr(flex_kernel, _PATCHED_MARK, True)
    return flex_kernel


def patch_transformers_saved_attention():
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception:
        return False
    for name in ("sdpa", "flash_attention_2", "flash_attention_3"):
        fn = ALL_ATTENTION_FUNCTIONS.get(name)
        if fn is not None:
            ALL_ATTENTION_FUNCTIONS[name] = _wrap_attention_interface(fn)
    try:
        import transformers.integrations.flex_attention as hf_flex
    except Exception:
        return True
    fn = getattr(hf_flex, "compile_friendly_flex_attention", None)
    if fn is not None:
        hf_flex.compile_friendly_flex_attention = _wrap_flex_kernel(fn)
    return True
