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

"""Layer-major gradient accumulation for block swap (offload_layers).

With gradient accumulation every micro-batch streams every swapped layer in again, once for the forward and once for
the backward recompute, so a GA = 4 step copies (or reads from disk) every layer 8 times. A layer-major window runs
all micro-batches of one optimizer step through a layer while it is on the card instead:

* forward: each micro-batch's own training step runs in a worker thread; the threads take turns (one runs at a time,
  in a fixed order), so every micro-batch passes decoder layer l before any passes layer l + 1. A layer runs without
  gradients there, and its input is kept, as gradient checkpointing keeps it.
* every layer output the model sees is a leaf; the loss part (final norm, head, loss, the trainer's own scaling and
  backward) runs per micro-batch, in order, and leaves its gradient on the last one (a loss on an intermediate
  hidden state on that one).
* backward: the layers are recomputed and backpropagated last to first, every micro-batch per layer, from the kept
  inputs and the gradients the loss part left on the leaves.

Each micro-batch keeps its own shapes, kernels and loss normalisation, and every weight gradient accumulates
micro-batch 0 first, as plain accumulation does, so losses and gradients are bitwise those of standard gradient
accumulation with per-layer checkpointing. Dropout masks are replayed in the recompute (drawn in layer-major order,
so with dropout they differ from standard accumulation's). The price is memory: every micro-batch's layer inputs are
held at once (gradient checkpointing holds one micro-batch's), see `boundary_bytes`.

Anything outside the protocol (a layer called out of order or twice, hidden states changed or written in place between
layers, a layer input other than the hidden states that needs gradients, a mapping passed between layers (shared KV
states), a checkpoint around the layer call, forward hooks or wrappers added to a layer during the window, saved-tensor
hooks, a loss part writing a decoder weight's gradient) aborts the window before any decoder-layer gradient exists;
layers that share key / value states are refused up front. Where standard training runs a layer's forward under grad
mode (no reentrant checkpoint), a recompute whose hidden states differ in any bit from the window's forward rolls the
window back too. Every abort, and every error a window raises, puts back the gradients and RNG state it found and
releases the block swap's held blocks; after an abort the caller runs the step the standard way.
UNSLOTH_OFFLOAD_LAYER_MAJOR = 0 turns it off.

Data parallel: the caller passes `agree` (a MIN all-reduce over the ranks), runs the micro-batches on the module
inside DDP (no reduction in their backwards) and reduces the gradients once after a window. The ranks vote before the
first loss part and after the decoder backward, so they all keep their windows or all roll back.
"""

import functools
import os
from collections.abc import Mapping
import sys
import threading
import traceback
from contextlib import ExitStack, contextmanager, nullcontext

import torch
import torch.utils.checkpoint

__all__ = [
    "LayerMajorAbort",
    "layer_major_mode",
    "layer_major_stats",
    "boundary_bytes",
    "run_layer_major",
    "stall_fraction",
    "swap_totals",
]

_STATS = {"windows": 0, "micro_batches": 0, "layer_calls": 0, "recomputes": 0, "aborts": 0, "last_abort": None}
_TLS = threading.local()
_CACHE_KWARGS = ("past_key_value", "past_key_values", "cache", "layer_past")


class LayerMajorAbort(BaseException):
    """Unwinds a micro-batch's worker (BaseException: trainer code catching Exception must not swallow it)."""


def layer_major_mode():
    """'off', 'on' (whenever eligible) or 'auto' (only where block swap measured copy stalls) from
    UNSLOTH_OFFLOAD_LAYER_MAJOR."""
    v = os.environ.get("UNSLOTH_OFFLOAD_LAYER_MAJOR", "auto").strip().lower()
    if v in ("0", "false", "off", "no"):
        return "off"
    if v in ("1", "true", "on", "yes", "force"):
        return "on"
    return "auto"


def layer_major_stats(reset = False):
    out = dict(_STATS)
    if reset:
        for k in _STATS:
            _STATS[k] = None if k == "last_abort" else 0
    return out


def swap_totals(swapper):
    """A BlockSwap's cumulative copy / stall / compute (and disk read wait) totals, without resetting its stats()
    window; None when it keeps none."""
    try:
        swapper._harvest()
        out = dict(swapper._tot)
    except Exception:
        return None
    tier = getattr(swapper, "_disk_tier", None)
    if tier is not None:
        try:
            out.update({k: v for k, v in tier.stats().items() if k.startswith("disk_")})
        except Exception:
            pass
    return out


def stall_fraction(before, after):
    """Share of the swapped layers' GPU time spent waiting for copies (or disk reads) between two swap_totals()."""
    if before is None or after is None:
        return None
    d = {k: after.get(k, 0) - before.get(k, 0) for k in after}
    stall = max(d.get("stall_ms", 0), d.get("disk_wait_ms", 0))
    busy = d.get("compute_ms", 0) + stall
    return stall / busy if busy > 0 else None


def boundary_bytes(num_tokens, hidden_size, num_layers, element_size = 2):
    """Device bytes a window holds for the layer inputs of micro-batches with `num_tokens` (rows x columns of
    each) tokens: every layer's input of every micro-batch, plus the last layer's output (the loss part's leaf)."""
    return sum(int(t) for t in num_tokens) * int(hidden_size) * (int(num_layers) + 1) * int(element_size)


class _ReplayAbort(Exception):
    """A decoder recompute that does not match its forward: the window rolls back."""


_DISAGREE = "another rank stopped its layer-major window"


class _Agreement:
    """Data-parallel votes on one window; `agree(ok, stage)` is a MIN all-reduce. Every rank votes "forward" (before
    any loss part's collectives) then "commit" (after the decoder backward); a rank stopping early votes False."""
    def __init__(self, agree):
        self.agree = agree
        self.stage = 0 if agree is not None else 2   # 0: forward vote due, 1: commit vote due, 2: done

    def vote(self, ok):
        if self.stage == 2:
            return ok and self.agree is None
        stage = "forward" if self.stage == 0 else "commit"
        result = bool(self.agree(bool(ok), stage))
        self.stage = 2 if (stage == "commit" or not result) else 1
        return result

    def close(self):
        """A window that stops early: False for every vote still due."""
        while self.stage != 2:
            self.vote(False)


def _saved_tensor_hooks_active():
    """Whether saved-tensor hooks (activation offloading, save_on_cpu) are active on this thread. Torch without the
    query: entering disable_saved_tensors_hooks raises while hooks are registered."""
    try:
        return torch._C._autograd._top_saved_tensors_default_hooks(False) is not None
    except (AttributeError, TypeError):
        pass
    try:
        with torch.autograd.graph.disable_saved_tensors_hooks("unsloth layer-major probe"):
            return False
    except RuntimeError:
        return True


def _is_cache(obj):
    """A mutable KV / state cache object (transformers Cache, Mamba cache_params and the like)."""
    if obj is None or isinstance(obj, (torch.Tensor, bool, int, float, str, tuple, list, dict)):
        return False
    cache_utils = sys.modules.get("transformers.cache_utils")
    base = getattr(cache_utils, "Cache", None)
    if isinstance(base, type) and isinstance(obj, base):
        return True
    return "Cache" in type(obj).__name__ or "cache" in type(obj).__name__


def _recorders_beyond_hidden_states():
    """transformers' active output collector (5.x) when it records more than hidden states (attentions, router
    logits): those are captured inside the decoder, under the window's no-grad forward."""
    mod = sys.modules.get("transformers.utils.output_capturing")
    collector = getattr(mod, "_active_collector", None)
    try:
        active = collector.get() if collector is not None else None
    except Exception:
        return False
    return bool(active) and bool(set(active.keys()) - {"hidden_states", "_hidden_states_layers"})


def _recompile_limit(error):
    exc = sys.modules.get("torch._dynamo.exc")
    kinds = tuple(cls for name in ("FailOnRecompileLimitHit", "RecompileLimitExceeded", "CacheLimitExceeded")
                  if isinstance(cls := getattr(exc, name, None), type))
    return bool(kinds) and isinstance(error, kinds)


def _all_rng_state():
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() and torch.cuda.is_initialized() else None
    return torch.get_rng_state(), cuda


def _set_all_rng_state(state):
    cpu, cuda = state
    torch.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)


def _autocast_state():
    state = []
    devices = ("cuda", "cpu") if torch.cuda.is_available() else ("cpu",)
    for dev in devices:
        try:
            state.append((dev, torch.is_autocast_enabled(dev), torch.get_autocast_dtype(dev)))
        except (TypeError, RuntimeError, AttributeError):
            pass
    return tuple(state), torch.is_autocast_cache_enabled()


def _autocast(state):
    devices, cache = state
    stack = ExitStack()
    for dev, enabled, dtype in devices:
        stack.enter_context(torch.autocast(device_type = dev, enabled = enabled, dtype = dtype,
                                           cache_enabled = cache))
    return stack


def _rng_state(devices):
    """CPU generator state and those of the CUDA devices a layer call runs on (its parameters' and its input's):
    the recompute must draw the same dropout masks."""
    return torch.get_rng_state(), [(d, torch.cuda.get_rng_state(d)) for d in devices]


@contextmanager
def _replay_rng(state):
    cpu, devices = state
    with torch.random.fork_rng(devices = [d for d, _ in devices], enabled = True):
        torch.set_rng_state(cpu)
        for d, dev_state in devices:
            torch.cuda.set_rng_state(dev_state, d)
        yield


@contextmanager
def _checkpointing_off(layer):
    """The window is the checkpoint: the layer's own checkpoint wrapper must pass the call straight through."""
    value = getattr(layer, "gradient_checkpointing", False)
    if not value:
        yield
        return
    # The flag can be a class property (Unsloth's skip-aware flag keeps its value under another instance key and has
    # no deleter), so restore whatever instance keys the assignment wrote instead of deleting the attribute.
    state = layer.__dict__
    before = dict(state)
    layer.gradient_checkpointing = False
    written = {k: (k in before, before.get(k)) for k, v in state.items() if k not in before or before[k] is not v}
    try:
        yield
    finally:
        if written:
            for k, (had, old) in written.items():
                if had:
                    state[k] = old
                else:
                    state.pop(k, None)
        else:
            # A setter that keeps the value outside the instance dict: assign the old value back through it.
            layer.gradient_checkpointing = value


def _uncached(layer, kwargs):
    """What a checkpointing layer wrapper does to the call's kwargs before it checkpoints (transformers clears the
    cache: the backward replay would update it twice)."""
    if not (getattr(layer, "gradient_checkpointing", False) and layer.training):
        return kwargs
    if kwargs.get("use_cache"):
        kwargs["use_cache"] = False
    if not getattr(layer, "_can_checkpoint_with_cache", False):
        for k in _CACHE_KWARGS:
            if kwargs.get(k) is not None:
                kwargs[k] = None
    return kwargs


def _hidden(args, kwargs):
    if args and isinstance(args[0], torch.Tensor):
        return args[0], 0
    h = kwargs.get("hidden_states")
    if isinstance(h, torch.Tensor):
        return h, "hidden_states"
    return None, None


def _with_hidden(args, kwargs, where, x):
    if where == 0:
        return (x,) + tuple(args[1:]), kwargs
    kwargs = dict(kwargs)
    kwargs[where] = x
    return tuple(args), kwargs


def _needs_grad(obj, depth = 0):
    if isinstance(obj, torch.Tensor):
        return obj.requires_grad
    if depth < 3 and isinstance(obj, (tuple, list)):
        return any(_needs_grad(o, depth + 1) for o in obj)
    if depth < 3 and isinstance(obj, Mapping):
        return any(_needs_grad(o, depth + 1) for o in obj.values())
    return False


def _shared_state(obj, depth = 0, empties = None):
    """A non-empty mapping or list among a decoder layer's inputs: mutable per-call state one layer can write and a
    later one read. The window's no-grad forward would hand the later layer detached tensors. Tuples (position
    embeddings) are looked into. Empty ones are collected into `empties`: the call must leave them empty. Plain dicts
    and UserDicts passed directly (Gemma 4's KV-sharing mapping) are handled by _SharedMappings instead."""
    if isinstance(obj, (Mapping, list)):
        if len(obj) or empties is None:
            return True
        empties.append(obj)
        return False
    if depth < 3 and isinstance(obj, tuple):
        return any(_shared_state(o, depth + 1, empties) for o in obj)
    return False


class _RecordingDict(dict):
    """A copy of a layer's mapping argument that records reads of keys an earlier decoder layer wrote."""
    def __init__(self, items, written, bad):
        super().__init__(items)
        self._written, self._bad = written, bad

    def _read(self, key):
        if key in self._written:
            self._bad.append(key)

    def __getitem__(self, key):
        self._read(key)
        return super().__getitem__(key)

    def get(self, key, default = None):
        self._read(key)
        return super().get(key, default)

    def __contains__(self, key):
        self._read(key)
        return super().__contains__(key)

    def _read_all(self):
        if self._written:
            self._bad.extend(self._written)

    def __iter__(self):
        self._read_all()
        return super().__iter__()

    def keys(self):
        self._read_all()
        return super().keys()

    def values(self):
        self._read_all()
        return super().values()

    def items(self):
        self._read_all()
        return super().items()


class _SharedMappings:
    """Mappings passed directly to every decoder layer (Gemma 4's KV-sharing mapping, which a layer writes even when
    no layer reads it back). Each call gets a recording copy; what the call writes is copied into the real mapping
    after it. A layer reading a key another layer of the window wrote would get the no-grad forward's tensors where
    standard training passes ones with gradients: that aborts (KV consumers are also refused up front)."""
    def __init__(self):
        self.written = {}   # id(real mapping) -> keys written during the window

    @staticmethod
    def accepted(obj):
        from collections import UserDict
        return type(obj) is dict or isinstance(obj, UserDict)

    def wrap(self, obj, bad):
        """(proxy to pass, real store, the proxy's store, keys written so far)."""
        from collections import UserDict
        written = self.written.setdefault(id(obj), set())
        if isinstance(obj, UserDict):
            proxy = type(obj).__new__(type(obj))
            proxy.__dict__.update(obj.__dict__)
            proxy.data = _RecordingDict(obj.data, written, bad)
            return proxy, obj.data, proxy.data, written
        proxy = _RecordingDict(obj, written, bad)
        return proxy, obj, proxy, written

    @staticmethod
    def commit(real, copy, written):
        """Copy the call's writes into the real store; False when the call deleted a key."""
        keys = set(dict.keys(copy))
        if any(k not in keys for k in list(real.keys())):
            return False
        for k in keys:
            v = dict.__getitem__(copy, k)
            if k not in real or real[k] is not v:
                real[k] = v
                written.add(k)
        return True


def _grad_observable(obj, allowed, limit = 4096):
    """Whether a gradient flowing into `obj` (tensors, also inside tuples / lists / mappings) reaches any leaf other
    than those in `allowed` (ids): a trainable parameter, or any other tensor whose .grad standard training fills.
    Gemma 4's per-layer inputs under LoRA are computed from frozen weights and the input embeddings, so they reach
    only the micro-batch's first decoder input (the leaf enable_input_require_grads marks), which the window
    backpropagates itself. Too large a graph to walk counts as observable."""
    pending, nodes, seen = [obj], [], set()
    while pending:
        o = pending.pop()
        if isinstance(o, torch.Tensor):
            if o.requires_grad:
                if o.grad_fn is None:
                    if id(o) not in allowed:
                        return True
                else:
                    nodes.append(o.grad_fn)
        elif isinstance(o, (tuple, list)):
            pending.extend(o)
        elif isinstance(o, Mapping):
            pending.extend(o.values())
    while nodes:
        node = nodes.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        if len(seen) > limit:
            return True
        variable = getattr(node, "variable", None)
        if isinstance(variable, torch.Tensor) and id(variable) not in allowed:
            return True
        nodes.extend(fn for fn, _ in getattr(node, "next_functions", ()))
    return False


def _detach_tensors(obj, depth = 0):
    if isinstance(obj, torch.Tensor):
        return obj.detach() if obj.requires_grad else obj
    if depth < 3 and type(obj) in (tuple, list):
        return type(obj)(_detach_tensors(o, depth + 1) for o in obj)
    return obj


def _contains_cache(values):
    """A live cache anywhere in the call's arguments, nested in tuples / lists / mappings too."""
    pending, seen = list(values), set()
    while pending:
        value = pending.pop()
        if _is_cache(value):
            return True
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, Mapping):
            pending.extend(value.values())
        elif isinstance(value, (tuple, list)):
            pending.extend(value)
    return False


def _output_metadata(out):
    """The container type and the plain values after the hidden states (floats by their bytes: -0.0 and NaN)."""
    import struct

    values = tuple(out[1:]) if type(out) in (tuple, list) else ()
    return type(out), tuple((type(v), struct.pack("!d", v) if type(v) is float else v) for v in values)


def _stock_checkpoint(fn):
    """torch's own checkpoint function, by provenance (the module it was defined in), not by the current value of
    torch.utils.checkpoint.checkpoint, which an integration can rebind to a wrapper: the function body must come
    from torch.utils.checkpoint, and an outer wrapper may only be torch's own dynamo-disable decorator."""
    ck = torch.utils.checkpoint
    body = getattr(fn, "__wrapped__", fn)
    code = getattr(body, "__code__", None)
    if code is None or getattr(body, "__globals__", None) is not vars(ck) or code.co_name != "checkpoint":
        return False
    if fn is body:
        return True
    outer = getattr(fn, "__globals__", {}) or {}
    return outer.get("__name__") == "torch._compile" and getattr(fn, "__wrapped__", None) is body


def _kv_shared_layers(layers):
    """Decoder layers that reuse another layer's key / value states (Gemma 4 is_kv_shared_layer)."""
    for layer in layers:
        for module in layer.modules():
            if getattr(module, "is_kv_shared_layer", False) is True:
                return True
    return False


_CKPT_SHIMS = ("unsloth_checkpoint", "unsloth_gradient_checkpoint", "unsloth_offloaded_gradient_checkpoint")


def _unsloth_checkpoint_function():
    """Unsloth's smart gradient checkpointing is patched in: torch's reentrant checkpoint and unsloth_checkpoint
    both run UnslothCheckpointFunction (offloaded inputs, MoE routing frames)."""
    return getattr(torch.utils.checkpoint.CheckpointFunction, "__name__", "") == "UnslothCheckpointFunction"


@contextmanager
def _unsloth_frames(layer, x, swapper = None):
    """Around the window's no-grad forward of a layer whose checkpoint is Unsloth's (a shim, or torch's reentrant
    checkpoint once Unsloth's CheckpointFunction is patched in). Yields [refusal reason or None, routing frame,
    (scope, saved-attention frame) or None]:
    standard training gives that checkpoint a MoE routing frame its recompute replays, and UnslothCheckpointFunction
    may store a float32 input in bfloat16 for its recompute. The window does neither, so an input stored that way
    is refused up front and a layer that records routing in the frame is refused after its forward."""
    state = [None, None, None]
    gc = sys.modules.get("unsloth_zoo.gradient_checkpointing")
    fn = getattr(layer, "_gradient_checkpointing_func", None)
    for _ in range(4):
        attrs = getattr(fn, "__dict__", None) or {}
        if isinstance(fn, functools.partial):
            fn = fn.func
        elif swapper is not None and attrs.get("_unsloth_block_swap") is swapper and \
                attrs.get("_unsloth_block_swap_wrapper") is fn:
            fn = fn.__wrapped__   # block swap's own residency wrapper, as _checkpoint_mode resolves it
        else:
            break
    shim = gc is not None and any(fn is getattr(gc, name, None) for name in _CKPT_SHIMS)
    if gc is None or not (shim or _unsloth_checkpoint_function()):
        yield state
        return
    store = getattr(gc, "_offload_store_dtype", None)
    # Every path but unsloth_gradient_checkpoint (which keeps the input as is) runs UnslothCheckpointFunction.
    exact = shim and fn is getattr(gc, "unsloth_gradient_checkpoint", None)
    if not exact and store is not None and store(x) != x.dtype:
        state[0] = "Unsloth's checkpoint stores float32 layer inputs in bfloat16 for the recompute"
        yield state
        return
    with ExitStack() as stack:
        scope, new_frame = getattr(gc, "_checkpoint_routing_scope", None), getattr(gc, "_new_checkpoint_routing_frame", None)
        frame = new_frame((x,)) if scope is not None and new_frame is not None else None
        if frame is not None:
            state[1] = frame
            stack.enter_context(scope(frame, False))
        saved = sys.modules.get("unsloth_zoo.saved_attention")
        new_attention = getattr(saved, "new_saved_attention_frame", None)
        attention_scope = getattr(saved, "saved_attention_scope", None)
        if not exact and new_attention is not None and attention_scope is not None:
            # UnslothCheckpointFunction keeps the forward's attention outputs (UNSLOTH_GC_SAVE_ATTENTION, or its
            # automatic mode at long sequences) for its recompute: the window keeps them the same way, one frame
            # per layer and micro-batch, replayed by that recompute.
            attention = new_attention((x,))
            if attention is not None:
                state[2] = (attention_scope, attention)
                stack.enter_context(attention_scope(attention, False))
        yield state


def _checkpoint_mode(layer, swapper = None):
    """(no_grad_forward, problem) for the layer's own checkpoint wrapper, which the window bypasses. Only these
    function objects (identity, not names: functools.wraps copies those) are known: torch's checkpoint (current or
    the pristine one Unsloth keeps; what Unsloth's own layers carry), optionally a functools.partial binding nothing
    but use_reentrant, the checkpoint wrapper this block swap installed around it, and Unsloth's checkpoint shims
    (reentrant; their MoE routing and saved-attention frames are handled per call). Anything else is refused. no_grad_forward: standard training runs the
    layer's forward without gradients (reentrant: use_reentrant True or unset, which torch resolves to True), so the
    window's no-grad forward and grad-mode recompute are exactly standard accumulation's, also where the two differ
    in the last bit (a compiled block's inference and training graphs)."""
    if not (getattr(layer, "gradient_checkpointing", False) and getattr(layer, "training", False)):
        return False, None
    fn, reentrant = getattr(layer, "_gradient_checkpointing_func", None), None
    shims = sys.modules.get("unsloth_zoo.gradient_checkpointing")
    shim_fns = {id(f) for f in (getattr(shims, name, None) for name in _CKPT_SHIMS) if f is not None}
    for _ in range(4):
        if fn is None:
            break
        if isinstance(fn, functools.partial):
            if reentrant is not None or fn.args or set(fn.keywords) - {"use_reentrant"}:
                return False, "a decoder layer's checkpoint has custom options (context_fn and the like)"
            reentrant = fn.keywords.get("use_reentrant")
            fn = fn.func
            continue
        if id(fn) in shim_fns:
            # Reentrant whatever use_reentrant says: a no-grad forward, then a grad-mode recompute from the kept
            # (or offloaded and reloaded) input. Their routing / saved-attention frames: _unsloth_frames.
            return True, None
        if _stock_checkpoint(fn):
            return reentrant is None or reentrant is True, None
        attrs = getattr(fn, "__dict__", None) or {}
        if swapper is not None and attrs.get("_unsloth_block_swap") is swapper and \
                attrs.get("_unsloth_block_swap_wrapper") is fn and reentrant is None:
            fn = fn.__wrapped__   # block swap's own residency wrapper around the layer's checkpoint
            continue
        break
    return False, "a decoder layer's checkpoint wrapper is not one the window can reproduce"


def _operator_modes_active():
    """TorchFunctionMode / TorchDispatchMode on this thread: thread-local, so workers would run without them."""
    for probe in (getattr(torch._C, "_len_torch_function_stack", None),
                  getattr(torch._C, "_len_torch_dispatch_stack", None),
                  getattr(getattr(torch, "_ops", None), "_len_torch_dispatch_stack_pre_dispatch", None)):
        try:
            if probe is not None and probe():
                return True
        except Exception:
            return True
    return False


_INT_VIEW = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _differs(a, b):
    """Device-side flag: whether two tensors of the same shape and dtype differ in any bit (NaN-safe)."""
    view = _INT_VIEW.get(a.element_size())
    a, b = a.detach().contiguous(), b.detach().contiguous()
    if view is None or a.dtype.is_complex:
        return a.ne(b).any()
    return a.view(view).ne(b.view(view)).any()


def _split_output(out):
    """(hidden, rebuild) for a layer output that is the hidden states or a plain tuple / list led by them."""
    if isinstance(out, torch.Tensor):
        return out, lambda h: h
    if type(out) in (tuple, list) and out and isinstance(out[0], torch.Tensor):
        # Attention weights / router logits (also nested in containers) computed without gradients would silently
        # lose their gradient: only plain metadata may follow the hidden states.
        if any(type(o) not in (type(None), bool, int, float, str) for o in out[1:]):
            return None, None
        rest = tuple(out[1:])
        kind = type(out)
        return out[0], lambda h: kind((h,) + rest)
    return None, None


class _Window:
    def __init__(self, layers, n, swapper, device, streams):
        self.layers = list(layers)
        self.L = len(self.layers)
        self.n = n
        self.index = {id(m): i for i, m in enumerate(self.layers)}
        self.swapper = swapper
        self.device = device
        self.streams = streams
        self.input_hooks = []
        self.cv = threading.Condition()
        self.turn = 0
        self.alive = [True] * n
        self.next_layer = [0] * n
        self.last_out = [None] * n
        self.last_version = [None] * n
        self.h0 = [None] * n
        self.saved = [[None] * n for _ in range(self.L)]
        self.attention = [[None] * n for _ in range(self.L)]   # (scope, saved-attention frame) per call
        # Every layer's output as the model sees it: a leaf, so a loss on intermediate hidden states (and the
        # loss part on the last one) leaves its gradient there for the backward.
        self.outs = [[None] * n for _ in range(self.L)]
        self.results = [None] * n
        self.error = None
        self.abort_reason = None
        self.held = set()
        self.signatures = [self._call_signature(l) for l in range(self.L)]
        self.layer_devices = [{p.device for p in layer.parameters() if p.is_cuda} for layer in self.layers]
        # Layers whose standard forward runs under grad mode: the window's no-grad forward must match its recompute.
        self.check_values = [False] * self.L
        self.in_backward = False
        self.input_used = [False] * n
        self.mappings = _SharedMappings()

    def _call_signature(self, l):
        """Forward hooks and instance-level forwards in layer l (and the global hooks): what a per-call output
        recorder changes."""
        from torch.nn.modules import module as nn_module
        sig = [len(getattr(nn_module, "_global_forward_hooks", ())), len(getattr(nn_module, "_global_forward_pre_hooks", ()))]
        for mod in self.layers[l].modules():
            sig += (len(mod._forward_hooks), len(mod._forward_pre_hooks), id(mod.__dict__.get("forward")))
        return tuple(sig)

    # ---- turn taking: exactly one worker runs at a time, in micro-batch order ----
    def _wait_turn(self, m):
        with self.cv:
            while self.turn != m and self.abort_reason is None:
                self.cv.wait()
            if self.abort_reason is not None:
                raise LayerMajorAbort(self.abort_reason)

    def _next_alive(self, m):
        for k in range(1, self.n + 1):
            j = (m + k) % self.n
            if self.alive[j]:
                return j
        return None

    def _pass_turn(self, m):
        with self.cv:
            self.turn = self._next_alive(m)
            self.cv.notify_all()

    def _abort(self, reason):
        with self.cv:
            if self.abort_reason is None:
                self.abort_reason = reason
            self.cv.notify_all()
        raise LayerMajorAbort(reason)

    # ---- the decoder layer call inside a worker ----
    def layer_call(self, layer, orig_call, args, kwargs):
        m = _TLS.micro
        l = self.index.get(id(layer))
        if l is None:
            # Another model's layer of the same class (a reference model): not part of the window.
            return orig_call(layer, *args, **kwargs)
        if l != self.next_layer[m]:
            self._abort(f"micro-batch {m} called decoder layer {l} where layer {self.next_layer[m]} was due")
        if not torch.is_grad_enabled():
            self._abort("a decoder layer ran with gradients off (a checkpoint around the layer call)")
        h, where = _hidden(args, kwargs)
        if h is None:
            self._abort("a decoder layer call without a hidden_states tensor")
        if l > 0 and (h is not self.last_out[m] or h._version != self.last_version[m]):
            self._abort("the hidden states changed between two decoder layers")
        if where == 0:
            others = (args[1:], kwargs)
        else:
            others = (args, {k: v for k, v in kwargs.items() if k != where})
        if _needs_grad(others):
            first = h if l == 0 else self.h0[m]
            if first is None or not first.is_leaf or _grad_observable(others, {id(first)}):
                self._abort("a decoder layer input other than the hidden states needs gradients that reach a "
                            "trained weight (such as Gemma 4's per-layer inputs with a trainable projection)")
            # Their gradient reaches nothing but this micro-batch's first decoder input, a leaf (frozen per-layer
            # embeddings under LoRA): passed without one, so the recompute does not backpropagate into the input
            # side once per layer; that leaf's own .grad is the only value that differs, and nothing trains it.
            if where == 0:
                args = (args[0],) + _detach_tensors(tuple(args[1:]))
            else:
                args = _detach_tensors(tuple(args))
            kwargs = {k: (v if k == where else _detach_tensors(v)) for k, v in kwargs.items()}
            others = (args[1:], {k: v for k, v in kwargs.items() if k != where}) if where == 0 else \
                (args, {k: v for k, v in kwargs.items() if k != where})
        # Plain dicts / UserDicts passed directly: recording copies (_SharedMappings); anything else that is a
        # non-empty mapping or list (or one inside a tuple) aborts.
        shared, bad = [], []
        if where == 0:
            args = (args[0],) + tuple(self._share(a, shared, bad) for a in args[1:])
        else:
            args = tuple(self._share(a, shared, bad) for a in args)
        kwargs = {k: (v if k == where else self._share(v, shared, bad)) for k, v in kwargs.items()}
        empties = []
        plain = [v for v in (*(args[1:] if where == 0 else args), *(v for k, v in kwargs.items() if k != where))
                 if not any(v is entry[0] for entry in shared)]
        if _shared_state(tuple(plain), empties = empties):
            self._abort("a decoder layer input is a mapping or list (state shared between layers, such as shared KV)")
        if _operator_modes_active():
            self._abort("a TorchFunctionMode / TorchDispatchMode is active inside the micro-batch")
        if _saved_tensor_hooks_active():
            # Entered inside the micro-batch (a callback): the window's decoder replay would bypass them.
            self._abort("saved-tensor hooks are active inside the micro-batch")
        if _recorders_beyond_hidden_states():
            self._abort("transformers is recording decoder outputs other than hidden states (output_attentions)")
        if self._call_signature(l) != self.signatures[l]:
            # Per-call output recorders (transformers' output_hidden_states / output_attentions capture wrappers,
            # hooks) would collect every interleaved micro-batch's outputs.
            self._abort("a decoder layer's forward or hooks changed during the window (output recorders)")
        if l == 0 and h.requires_grad:
            self.h0[m] = h
            # The input-side graph is traversed once, after the layers: a loss part that backpropagates into it
            # directly (a loss on the embedding output) would consume it first.
            self.input_hooks.append(h.register_hook(functools.partial(self._input_grad, m)))
        # Same requires_grad as the input a reentrant checkpoint passes its no-grad forward (compiled guards).
        x = h.detach().requires_grad_(h.requires_grad)
        if _contains_cache((*others[0], *others[1].values())):
            # A live KV / state cache, positional or under any keyword (deprecated aliases included), checked on
            # the call as made: the recompute would update it a second time.
            self._abort("a decoder layer call with a live KV cache")
        no_grad_forward, problem = _checkpoint_mode(layer, self.swapper)
        if problem is not None:
            self._abort(problem)
        if no_grad_forward and (where != 0 or not h.requires_grad):
            # A reentrant checkpoint only produces gradients from a positional input that needs them: without
            # one, standard training gets no decoder gradient at all here.
            self._abort("a reentrant checkpoint without a positional hidden state that needs gradients")
        self.check_values[l] = self.check_values[l] or not no_grad_forward
        kwargs = _uncached(layer, dict(kwargs))
        args2, kwargs2 = _with_hidden(args, kwargs, where, x)
        devices = self.layer_devices[l] | ({x.device} if x.is_cuda else set())
        self.saved[l][m] = (args2, kwargs2, where, _autocast_state(), _rng_state(sorted(devices, key = str)))
        if m == 0 and self.swapper is not None:
            self.swapper.hold(l)
            self.held.add(l)
        with torch.no_grad(), _checkpointing_off(layer), _unsloth_frames(layer, x, self.swapper) as frames:
            if frames[0] is not None:
                self._abort(frames[0])
            out = orig_call(layer, *args2, **kwargs2)
        if any(len(c) for c in empties):
            self._abort("a decoder layer wrote into a mapping or list passed to every layer (shared KV states)")
        if bad:
            self._abort("a decoder layer read a mapping entry another decoder layer wrote (shared KV states)")
        for _, real, copy, written in shared:
            if not _SharedMappings.commit(real, copy, written):
                self._abort("a decoder layer removed entries from a mapping passed to every layer")
        if frames[1] is not None and frames[1].entries:
            self._abort("a decoder layer routes MoE experts under Unsloth's checkpoint, whose recompute replays it")
        self.attention[l][m] = frames[2]
        out_h, rebuild = _split_output(out)
        if out_h is None:
            self._abort("a decoder layer output that is not the hidden states (or a tuple led by them)")
        self.saved[l][m] += (_output_metadata(out),)
        if l in self.held and m == self.n - 1:
            self.held.discard(l)
            self.swapper.unhold(l)
        if l == self.L - 1 and m == self.n - 1 and self.agreement.stage == 0:
            # Vote before any loss part issues its own collectives.
            if not self.agreement.vote(self.abort_reason is None):
                self._abort(_DISAGREE)
        _STATS["layer_calls"] += 1
        self.next_layer[m] = l + 1
        # The output tensor itself becomes the leaf (it has no graph): anything that kept a reference to it inside
        # the call (transformers' hidden-state capture) then holds the leaf, and a loss on it reaches the layers.
        leaf = out_h.requires_grad_(True) if out_h is not x and out_h.is_leaf else out_h.detach().requires_grad_(True)
        self.outs[l][m] = leaf
        out = rebuild(leaf)
        self.last_out[m], self.last_version[m] = leaf, leaf._version
        # Let the other micro-batches through this layer; after the last layer the loss parts run in order.
        self._pass_turn(m)
        self._wait_turn(m)
        return out

    def _share(self, value, shared, bad):
        if not _SharedMappings.accepted(value):
            return value
        entry = self.mappings.wrap(value, bad)
        shared.append(entry)
        return entry[0]

    def _input_grad(self, m, grad):
        if not self.in_backward:
            self.input_used[m] = True

    def _worker(self, m, fn):
        _TLS.window, _TLS.micro = self, m
        try:
            self._wait_turn(m)
            with ExitStack() as stack:
                if self.device is not None and self.device.type == "cuda":
                    # The caller's current stream on every device the model runs on (thread-local, like the
                    # current device): inputs produced there are ordered before the micro-batch's kernels.
                    for stream in self.streams:
                        stack.enter_context(torch.cuda.stream(stream))
                    stack.enter_context(torch.cuda.device(self.device))
                # Grad mode and autocast are thread-local: the caller's apply to its micro-batches.
                stack.enter_context(torch.set_grad_enabled(self.caller_grad))
                stack.enter_context(_autocast(self.caller_autocast))
                self.results[m] = fn()
            if self.next_layer[m] != self.L and self.abort_reason is None:
                with self.cv:
                    self.abort_reason = (f"micro-batch {m} ran {self.next_layer[m]} of {self.L} decoder layers")
        except LayerMajorAbort as e:
            # Also one a micro-batch raised itself (not through _abort): the window must not count as complete.
            with self.cv:
                if self.abort_reason is None:
                    self.abort_reason = str(e) or f"micro-batch {m} aborted the window"
        except RuntimeError as e:
            # The model writes in place into a layer output between layers (a leaf here): no window.
            inplace = "leaf Variable that requires grad" in str(e)
            with self.cv:
                if self.error is None and not inplace:
                    self.error = e
                if self.abort_reason is None:
                    self.abort_reason = ("the model writes in place into the hidden states between layers" if inplace
                                         else f"micro-batch {m} raised {type(e).__name__}: {e}")
        except BaseException as e:   # surfaced by the caller once every worker is done
            with self.cv:
                if self.error is None:
                    self.error = e
                if self.abort_reason is None:
                    self.abort_reason = f"micro-batch {m} raised {type(e).__name__}: {e}"
        finally:
            _TLS.window = None
            with self.cv:
                self.alive[m] = False
                self.turn = self._next_alive(m)
                self.cv.notify_all()

    # ---- backward, layer by layer, on the calling thread ----
    def backward(self, orig_calls):
        self.in_backward = True
        g = [None] * self.n
        # One flag per device, read once after the decoder backward: where standard training's forward runs under
        # grad mode, a recompute whose hidden states differ in any bit from the window's no-grad forward (a layer
        # that computes something else under grad mode) would train a different function than standard.
        differs = {}
        for l in range(self.L - 1, -1, -1):
            layer = self.layers[l]
            orig_call = orig_calls[type(layer)]
            if self.swapper is not None:
                self.swapper.hold(l)
            try:
                for m in range(self.n):
                    saved, self.saved[l][m] = self.saved[l][m], None
                    attention, self.attention[l][m] = self.attention[l][m], None
                    leaf, self.outs[l][m] = self.outs[l][m], None
                    # The loss part's gradient on the last output, or a loss on an intermediate one.
                    extra = leaf.grad if leaf is not None else None
                    if extra is not None:
                        g[m] = extra if g[m] is None else g[m] + extra
                    del extra
                    if saved is None or g[m] is None:
                        del leaf
                        continue
                    args2, kwargs2, where, ac, rng, metadata = saved
                    x = args2[0] if where == 0 else kwargs2[where]
                    x.requires_grad_(True)
                    with torch.enable_grad(), _autocast(ac), _checkpointing_off(layer), _replay_rng(rng), \
                            (attention[0](attention[1], True) if attention is not None else nullcontext()):
                        out = orig_call(layer, *args2, **kwargs2)
                    out_h, _ = _split_output(out)
                    if out_h is None or not out_h.requires_grad or leaf is None or \
                            (out_h.shape, out_h.dtype, out_h.device) != (leaf.shape, leaf.dtype, leaf.device):
                        raise _ReplayAbort("a decoder recompute returned an output its forward did not")
                    if self.check_values[l] and _output_metadata(out) != metadata:
                        raise _ReplayAbort("a decoder recompute changed the layer's output metadata")
                    if self.check_values[l]:
                        with torch.no_grad():
                            flag = _differs(out_h, leaf)
                            prev = differs.get(flag.device)
                            differs[flag.device] = flag if prev is None else prev.logical_or_(flag)
                        del flag, prev
                    del leaf
                    torch.autograd.backward(out_h, g[m])
                    _STATS["recomputes"] += 1
                    g[m] = x.grad
                    del out, out_h, x, args2, kwargs2, saved
            finally:
                if self.swapper is not None:
                    self.swapper.unhold(l)
        if any(bool(flag) for flag in differs.values()):
            raise _ReplayAbort("a decoder recompute changed the hidden states its forward produced")
        differs.clear()
        # Input-side modules (a trainable embedding): their graph ends at layer 0's input, in micro-batch order.
        for m in range(self.n):
            if self.h0[m] is not None and g[m] is not None:
                torch.autograd.backward(self.h0[m], g[m])
        self.h0 = [None] * self.n


class _Workers:
    """Long-lived worker threads: a fresh thread pays CUDA library set-up (handles, workspaces) on its first
    launches, tens of ms per thread per step on a small model."""
    def __init__(self):
        self.lock = threading.Lock()
        self.queues = []

    def _loop(self, q):
        while True:
            job = q.get()
            if job is None:
                return
            window, m, fn, done = job
            try:
                window._worker(m, fn)
            finally:
                # An idle worker must not keep the last step's model, trainer and batch alive.
                job = window = fn = None
                done.set()

    def run(self, window, fns, events):
        """Submit one job per micro-batch, appending each one's done event to `events`. The window's lock is held
        while submitting: no worker enters the model until every job is in, and an interrupted submission aborts
        the window, so the workers already given a job leave instead of waiting for the others."""
        import queue
        with self.lock:
            while len(self.queues) < len(fns):
                q = queue.SimpleQueue()
                threading.Thread(target = self._loop, args = (q,), daemon = True,
                                 name = f"unsloth-layer-major-{len(self.queues)}").start()
                self.queues.append(q)
            queues = self.queues[:len(fns)]
        with window.cv:
            try:
                for m, (q, fn) in enumerate(zip(queues, fns)):
                    done = threading.Event()
                    q.put((window, m, fn, done))
                    events.append(done)
            except BaseException:
                if window.abort_reason is None:
                    window.abort_reason = "interrupted while submitting micro-batches"
                window.cv.notify_all()
                raise


_WORKERS = _Workers()


@contextmanager
def _intercept(window, layers):
    """Route the decoder layers' calls through the window, for its worker threads only (class-level, undone after,
    also when setting up fails halfway). Each interceptor handles instances of exactly its class, so a derived class
    calling its parent's __call__ is intercepted once."""
    orig_calls = {cls: cls.__call__ for cls in {type(layer) for layer in layers}}
    patched = []
    try:
        for cls, orig in orig_calls.items():
            patched.append((cls, "__call__" in cls.__dict__, orig))

            def call(self, *args, _cls = cls, _orig = orig, **kwargs):
                w = getattr(_TLS, "window", None)
                if w is None or type(self) is not _cls:
                    return _orig(self, *args, **kwargs)
                return w.layer_call(self, _orig, args, kwargs)

            cls.__call__ = call
        yield orig_calls
    finally:
        for cls, had, orig in reversed(patched):
            if had:
                cls.__call__ = orig
            elif "__call__" in cls.__dict__:
                del cls.__call__


_WINDOW_LOCK = threading.Lock()
# The worker threads exist only in the process that started them: a fork child inherits their queues, not them.
_WINDOW_PID = os.getpid()


def run_layer_major(layers, micro_fns, swapper = None, model = None, agree = None):
    """Run `micro_fns` (one full training step per micro-batch: forward, loss, its scaling and backward) layer-major
    over `layers`. Returns (results, None) with every gradient accumulated, or (None, reason) after a safe abort or
    refusal: every trainable gradient of `model` (and of `layers`) and the RNG state are as they were, so the caller
    can run the same micro-batches the standard way. A real error in a micro-batch is raised. One window per process
    at a time: a concurrent caller is refused (the decoder classes and the worker pool are shared).

    Data parallel needs `agree` (see _Agreement): all ranks keep or all roll back. Micro-batches must not reduce
    gradients (run them on the module inside DDP); the caller reduces after the window."""
    agreement = _Agreement(agree)
    if os.getpid() != _WINDOW_PID:
        agreement.close()
        return None, "layer-major workers cannot be reused after fork"
    if model is None:
        # Every trainable gradient the micro-batches can write must be snapshotted for a rollback.
        agreement.close()
        return None, "layer-major accumulation needs the full model (gradient rollback)"
    if not _WINDOW_LOCK.acquire(blocking = False):
        agreement.close()
        return None, "another layer-major window is running in this process"
    try:
        reset = getattr(swapper, "reset", None)
        try:
            results, why = _run_layer_major(layers, micro_fns, swapper, model, agreement)
        except BaseException:
            # The gradients and RNG are already put back; blocks a raising window left pending are released.
            if reset is not None:
                reset()
            raise
        if results is None:
            if reset is not None:
                reset()       # an interrupted window leaves blocks pending: release them all
            agreement.close()
        return results, why
    finally:
        _WINDOW_LOCK.release()


def _decoder_forward_hooks(layers, swapper):
    """Forward (pre-)hooks in the decoder layers other than the block swap's own, or global module forward hooks."""
    from torch.nn.modules import module as nn_module

    if getattr(nn_module, "_global_forward_pre_hooks", None) or getattr(nn_module, "_global_forward_hooks", None):
        return True
    own = {handle.id for handle in getattr(swapper, "handles", ()) if hasattr(handle, "id")}
    for layer in layers:
        for module in layer.modules():
            if (set(module._forward_pre_hooks) | set(module._forward_hooks)) - own:
                return True
    return False


def _tied_trainable_io(model):
    """A trainable parameter shared by the input embedding and the output head: the window adds the head's
    contributions of every micro-batch before any embedding contribution (standard accumulation interleaves)."""
    get_in, get_out = getattr(model, "get_input_embeddings", None), getattr(model, "get_output_embeddings", None)
    if not (callable(get_in) and callable(get_out)):
        return False
    try:
        emb, head = get_in(), get_out()
    except (NotImplementedError, AttributeError):
        return False
    if emb is None or head is None:
        return False
    trainable = {id(p) for p in emb.parameters() if p.requires_grad}
    return any(id(p) in trainable for p in head.parameters())


def _run_layer_major(layers, micro_fns, swapper, model, agreement):
    n = len(micro_fns)
    if n < 1 or not len(layers):
        return None, "nothing to run"
    if agreement.agree is None and torch.distributed.is_available() and torch.distributed.is_initialized() and \
            torch.distributed.get_world_size() > 1:
        # DDP would reduce before the decoder backward exists, and ranks could disagree on rollback.
        return None, "distributed training (gradient reduction would miss the decoder backward)"
    if model is not None and _tied_trainable_io(model):
        return None, "the input embedding and the output head share a trainable weight"
    if _kv_shared_layers(layers):
        # A consumer layer replays before its producer and would read the producer's no-grad key / value states.
        return None, "decoder layers share key / value states (KV sharing)"
    if _operator_modes_active():
        return None, "a TorchFunctionMode / TorchDispatchMode is active (the workers would run without it)"
    if _decoder_forward_hooks(layers, swapper):
        # Standard training runs them once around a checkpointed forward; the window calls the layer twice (no-grad
        # forward and grad-mode recompute).
        return None, "a decoder layer has forward hooks the window cannot reproduce"
    if _saved_tensor_hooks_active():
        # Saved-tensor hooks (activation offloading) are thread-local: the workers' loss parts would run without.
        return None, "saved-tensor hooks are active (activation offloading)"
    # Decoder weights get gradients only from the window's backward; one the loss parts write (a regulariser on a
    # decoder weight) would be added before every recompute contribution instead of interleaved with them.
    decoder, seen_dec = [], set()
    for layer in layers:
        for p in layer.parameters():
            if not p.requires_grad:
                continue
            if id(p) in seen_dec:
                # Standard accumulation adds its layers' contributions micro-batch by micro-batch, the window
                # layer by layer.
                return None, "decoder layers share a trainable weight"
            seen_dec.add(id(p))
            decoder.append(p)
    # A loss part can also write gradients directly (a regulariser on a decoder weight): an abort puts back all.
    params, seen = [], set()
    for owner in ([model] if model is not None else []) + list(layers):
        for p in owner.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                params.append(p)
    try:
        outside = [(p, None if p.grad is None else p.grad.clone()) for p in params]
    except torch.OutOfMemoryError:
        return None, "out of memory while saving the gradients a window would have to restore"
    rng = _all_rng_state()
    watched = [(p, p.grad, None if p.grad is None else p.grad._version) for p in decoder]

    def restore():
        for p, g in outside:
            p.grad = g
        _set_all_rng_state(rng)   # draws made by the abandoned window (NEFTune noise, dropout) are not consumed

    def roll_back(reason):
        restore()
        _STATS["aborts"] += 1
        _STATS["last_abort"] = reason
        return None, reason

    scope = getattr(swapper, "layer_major_window", None)
    try:
        with scope() if scope is not None else nullcontext():
            results, why = _execute_window(layers, micro_fns, swapper, model, n, watched, roll_back, agreement)
        if results is not None:
            if scope is not None:
                try:
                    swapper._end_step()   # the step's single end, once its whole backward is done
                except torch.OutOfMemoryError:
                    # Only the optional prefetch for the next step ran out: the step's gradients are complete.
                    reset = getattr(swapper, "reset", None)
                    if reset is not None:
                        reset()
            _STATS["windows"] += 1
            _STATS["micro_batches"] += n
        return results, why
    except BaseException:
        # A real error (or an interrupt) propagates, but a caller that catches it must not inherit a partial
        # accumulation or consumed RNG draws.
        restore()
        raise


def _execute_window(layers, micro_fns, swapper, model, n, watched, roll_back, agreement):
    device, streams = None, ()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        device = torch.device("cuda", torch.cuda.current_device())
        devices = {device} | {torch.device(d) for d in getattr(swapper, "_peak_devs", ())}
        for owner in ([model] if model is not None else []) + list(layers):
            devices.update(t.device for t in (*owner.parameters(), *owner.buffers()) if t.is_cuda)
        streams = tuple(torch.cuda.current_stream(d) for d in sorted(devices, key = lambda d: d.index or 0))
    w = _Window(layers, n, swapper, device, streams)
    w.agreement = agreement
    w.caller_grad, w.caller_autocast = torch.is_grad_enabled(), _autocast_state()
    try:
        with _intercept(w, layers) as orig_calls:
            events = []
            try:
                _WORKERS.run(w, micro_fns, events)
                for done in events:
                    while not done.wait(0.5):
                        pass
            except BaseException:
                with w.cv:
                    if w.abort_reason is None:
                        w.abort_reason = "interrupted"
                    w.cv.notify_all()
                for done in events:
                    done.wait()
                raise
            if w.abort_reason is None and any(out is not None and out.grad is not None
                                              for row in w.outs[:-1] for out in row):
                # Autograd would interleave those contributions with the decoder's; the window adds them first.
                w.abort_reason = "a loss directly uses an intermediate decoder output"
            if w.abort_reason is None and any(out is None or out.grad is None for out in w.outs[-1]):
                # A targeted backward (inputs = ...) or a loss that never reaches the last decoder output.
                w.abort_reason = "a micro-batch did not backpropagate into the last decoder output"
            if w.abort_reason is None and any(w.input_used):
                w.abort_reason = ("a loss part backpropagated into the first decoder layer's input directly "
                                  "(a loss on the embedding output)")
            if w.abort_reason is None and any(p.grad is not g or (g is not None and g._version != v)
                                              for p, g, v in watched):
                w.abort_reason = "a loss part wrote a decoder layer weight's gradient (a regulariser on it)"
    finally:
        # The first-input hooks only watch the loss parts; a reused input tensor must not keep the window alive.
        for handle in w.input_hooks:
            handle.remove()
        w.input_hooks.clear()
        if w.swapper is not None:
            for l in sorted(w.held):
                w.swapper.unhold(l)
            w.held.clear()
    if w.abort_reason is not None:
        w.saved = w.outs = w.h0 = None
        error, w.error = w.error, None
        if error is not None and (agreement.stage == 1 or
                                  (not isinstance(error, torch.OutOfMemoryError) and not _recompile_limit(error))):
            # After the forward vote, loss parts may have run collectives another rank is still in: a rollback
            # vote could pair with them, so raise as a standard step would.
            raise error
        if error is not None:
            # Its traceback's frames hold the failed forward's tensors (and the window): free them now, before
            # the caller's standard step needs the memory.
            traceback.clear_frames(error.__traceback__)
            error.__traceback__ = None
        del error
        return roll_back(w.abort_reason)
    try:
        w.backward(orig_calls)
    except (torch.OutOfMemoryError, _ReplayAbort, Exception) as error:
        # The recompute builds a graph the no-grad forward did not: it can run out where the forward fit, or hit
        # a compiled block's recompile limit. Anything else is a real error.
        if not isinstance(error, (torch.OutOfMemoryError, _ReplayAbort)) and not _recompile_limit(error):
            raise
        reason = ("out of memory in the decoder backward" if isinstance(error, torch.OutOfMemoryError)
                  else str(error) if isinstance(error, _ReplayAbort)
                  else "torch.compile recompile limit in the decoder backward")
        traceback.clear_frames(error.__traceback__)
        del error
        w.saved = w.outs = w.h0 = None
        return roll_back(reason)
    if not agreement.vote(True):
        w.saved = w.outs = w.h0 = None
        return roll_back(_DISAGREE)
    return w.results, None
