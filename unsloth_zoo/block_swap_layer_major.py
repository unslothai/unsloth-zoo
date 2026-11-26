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
layers, a layer input other than the hidden states that needs gradients, a checkpoint around the layer call, forward
hooks or wrappers added to a layer during the window, saved-tensor hooks) aborts the window before any decoder-layer
gradient exists; gradients the loss part already wrote are put back, and the caller runs the step
the standard way. UNSLOTH_OFFLOAD_LAYER_MAJOR = 0 turns it off.
"""

import functools
import os
import threading
from contextlib import ExitStack, contextmanager

import torch

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


def _rng_state(device):
    """CPU and device generator states before a layer call: its recompute must draw the same dropout masks."""
    cuda = device is not None and device.type == "cuda"
    return torch.get_rng_state(), (torch.cuda.get_rng_state(device) if cuda else None), device


@contextmanager
def _replay_rng(state):
    cpu, dev_state, device = state
    devices = [device] if dev_state is not None else []
    with torch.random.fork_rng(devices = devices, enabled = True):
        torch.set_rng_state(cpu)
        if dev_state is not None:
            torch.cuda.set_rng_state(dev_state, device)
        yield


@contextmanager
def _checkpointing_off(layer):
    """The window is the checkpoint: the layer's own checkpoint wrapper must pass the call straight through."""
    if not getattr(layer, "gradient_checkpointing", False):
        yield
        return
    had = "gradient_checkpointing" in layer.__dict__
    old = layer.__dict__.get("gradient_checkpointing")
    layer.gradient_checkpointing = False
    try:
        yield
    finally:
        if had:
            layer.gradient_checkpointing = old
        else:
            del layer.gradient_checkpointing


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
    if depth < 3 and isinstance(obj, dict):
        return any(_needs_grad(o, depth + 1) for o in obj.values())
    return False


def _split_output(out):
    """(hidden, rebuild) for a layer output that is the hidden states or a plain tuple / list led by them."""
    if isinstance(out, torch.Tensor):
        return out, lambda h: h
    if type(out) in (tuple, list) and out and isinstance(out[0], torch.Tensor):
        # Attention weights / router logits computed without gradients would silently lose their gradient.
        if any(isinstance(o, torch.Tensor) for o in out[1:]):
            return None, None
        rest = tuple(out[1:])
        kind = type(out)
        return out[0], lambda h: kind((h,) + rest)
    return None, None


class _Window:
    def __init__(self, layers, n, swapper, device, stream):
        self.layers = list(layers)
        self.L = len(self.layers)
        self.n = n
        self.index = {id(m): i for i, m in enumerate(self.layers)}
        self.swapper = swapper
        self.device = device
        self.stream = stream
        self.cv = threading.Condition()
        self.turn = 0
        self.alive = [True] * n
        self.next_layer = [0] * n
        self.last_out = [None] * n
        self.last_version = [None] * n
        self.h0 = [None] * n
        self.saved = [[None] * n for _ in range(self.L)]
        # Every layer's output as the model sees it: a leaf, so a loss on intermediate hidden states (and the
        # loss part on the last one) leaves its gradient there for the backward.
        self.outs = [[None] * n for _ in range(self.L)]
        self.results = [None] * n
        self.error = None
        self.abort_reason = None
        self.held = set()
        self.signatures = [self._call_signature(l) for l in range(self.L)]
        self.in_backward = False
        self.input_used = [False] * n

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
            self._abort("a decoder layer input other than the hidden states needs gradients")
        if self._call_signature(l) != self.signatures[l]:
            # Per-call output recorders (transformers' output_hidden_states / output_attentions capture wrappers,
            # hooks) would collect every interleaved micro-batch's outputs.
            self._abort("a decoder layer's forward or hooks changed during the window (output recorders)")
        if l == 0 and h.requires_grad:
            self.h0[m] = h
            # The input-side graph is traversed once, after the layers: a loss part that backpropagates into it
            # directly (a loss on the embedding output) would consume it first.
            h.register_hook(functools.partial(self._input_grad, m))
        x = h.detach()
        kwargs = _uncached(layer, dict(kwargs))
        if not getattr(layer, "_can_checkpoint_with_cache", False) and any(
                kwargs.get(k) is not None for k in _CACHE_KWARGS):
            # A live KV cache (use_cache without checkpointing): the recompute would update it a second time.
            self._abort("a decoder layer call with a live KV cache")
        args2, kwargs2 = _with_hidden(args, kwargs, where, x)
        self.saved[l][m] = (args2, kwargs2, where, _autocast_state(), _rng_state(self.device))
        if m == 0 and self.swapper is not None:
            self.swapper.hold(l)
            self.held.add(l)
        with torch.no_grad(), _checkpointing_off(layer):
            out = orig_call(layer, *args2, **kwargs2)
        out_h, rebuild = _split_output(out)
        if out_h is None:
            self._abort("a decoder layer output that is not the hidden states (or a tuple led by them)")
        if l in self.held and m == self.n - 1:
            self.held.discard(l)
            self.swapper.unhold(l)
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

    def _input_grad(self, m, grad):
        if not self.in_backward:
            self.input_used[m] = True

    def _worker(self, m, fn):
        _TLS.window, _TLS.micro = self, m
        try:
            self._wait_turn(m)
            with ExitStack() as stack:
                if self.device is not None and self.device.type == "cuda":
                    stack.enter_context(torch.cuda.device(self.device))
                    if self.stream is not None:
                        stack.enter_context(torch.cuda.stream(self.stream))
                # Grad mode and autocast are thread-local: the caller's apply to its micro-batches.
                stack.enter_context(torch.set_grad_enabled(self.caller_grad))
                stack.enter_context(_autocast(self.caller_autocast))
                self.results[m] = fn()
            if self.next_layer[m] != self.L and self.abort_reason is None:
                with self.cv:
                    self.abort_reason = (f"micro-batch {m} ran {self.next_layer[m]} of {self.L} decoder layers")
        except LayerMajorAbort:
            pass
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
        for l in range(self.L - 1, -1, -1):
            layer = self.layers[l]
            orig_call = orig_calls[type(layer)]
            if self.swapper is not None:
                self.swapper.hold(l)
            try:
                for m in range(self.n):
                    saved, self.saved[l][m] = self.saved[l][m], None
                    leaf, self.outs[l][m] = self.outs[l][m], None
                    # The loss part's gradient on the last output, or a loss on an intermediate one.
                    extra = leaf.grad if leaf is not None else None
                    if extra is not None:
                        g[m] = extra if g[m] is None else g[m] + extra
                    del leaf, extra
                    if saved is None or g[m] is None:
                        continue
                    args2, kwargs2, where, ac, rng = saved
                    x = args2[0] if where == 0 else kwargs2[where]
                    x.requires_grad_(True)
                    with torch.enable_grad(), _autocast(ac), _checkpointing_off(layer), _replay_rng(rng):
                        out = orig_call(layer, *args2, **kwargs2)
                    out_h, _ = _split_output(out)
                    torch.autograd.backward(out_h, g[m])
                    _STATS["recomputes"] += 1
                    g[m] = x.grad
                    del out, out_h, x, args2, kwargs2, saved
            finally:
                if self.swapper is not None:
                    self.swapper.unhold(l)
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

    def run(self, window, fns):
        import queue
        with self.lock:
            while len(self.queues) < len(fns):
                q = queue.SimpleQueue()
                threading.Thread(target = self._loop, args = (q,), daemon = True,
                                 name = f"unsloth-layer-major-{len(self.queues)}").start()
                self.queues.append(q)
            queues = self.queues[:len(fns)]
        events = []
        for m, (q, fn) in enumerate(zip(queues, fns)):
            done = threading.Event()
            events.append(done)
            q.put((window, m, fn, done))
        return events


_WORKERS = _Workers()


@contextmanager
def _intercept(window, layers):
    """Route the decoder layers' calls through the window, for its worker threads only (class-level, undone after)."""
    patched, orig_calls = [], {}
    for cls in {type(layer) for layer in layers}:
        orig = cls.__call__
        orig_calls[cls] = orig
        had = "__call__" in cls.__dict__

        def call(self, *args, _orig = orig, **kwargs):
            w = getattr(_TLS, "window", None)
            if w is None:
                return _orig(self, *args, **kwargs)
            return w.layer_call(self, _orig, args, kwargs)

        cls.__call__ = call
        patched.append((cls, had, orig))
    try:
        yield orig_calls
    finally:
        for cls, had, orig in patched:
            if had:
                cls.__call__ = orig
            else:
                del cls.__call__


def run_layer_major(layers, micro_fns, swapper = None, model = None):
    """Run `micro_fns` (one full training step per micro-batch: forward, loss, its scaling and backward) layer-major
    over `layers`. Returns (results, None) with every gradient accumulated, or (None, reason) after a safe abort:
    every trainable gradient of `model` (and of `layers`) is as it was, so the caller can run the same micro-batches
    the standard way. A real error in a micro-batch is raised."""
    n = len(micro_fns)
    if n < 1 or not len(layers):
        return None, "nothing to run"
    if torch.distributed.is_available() and torch.distributed.is_initialized() and \
            torch.distributed.get_world_size() > 1:
        # DDP reduces in the loss parts' backwards, before the window's decoder backward exists.
        return None, "distributed training (gradient reduction would miss the decoder backward)"
    # A loss part can also write gradients directly (a regulariser on a decoder weight): an abort puts back all.
    params, seen = [], set()
    for owner in ([model] if model is not None else []) + list(layers):
        for p in owner.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                params.append(p)
    outside = [(p, None if p.grad is None else p.grad.clone()) for p in params]
    try:
        hooks = torch._C._autograd._top_saved_tensors_default_hooks(False)
    except (AttributeError, TypeError, RuntimeError):   # torch < 2.7 has no query
        hooks = None
    if hooks is not None:
        # Saved-tensor hooks (activation offloading) are thread-local: the workers' loss parts would run without.
        return None, "saved-tensor hooks are active (activation offloading)"
    device = stream = None
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        device = torch.device("cuda", torch.cuda.current_device())
        stream = torch.cuda.current_stream(device)
    w = _Window(layers, n, swapper, device, stream)
    w.caller_grad, w.caller_autocast = torch.is_grad_enabled(), _autocast_state()
    try:
        with _intercept(w, layers) as orig_calls:
            events = _WORKERS.run(w, micro_fns)
            try:
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
            if w.abort_reason is None and any(w.input_used):
                w.abort_reason = ("a loss part backpropagated into the first decoder layer's input directly "
                                  "(a loss on the embedding output)")
    finally:
        if w.swapper is not None:
            for l in sorted(w.held):
                w.swapper.unhold(l)
            w.held.clear()
    if w.abort_reason is not None:
        for p, g in outside:
            p.grad = g
        w.saved = w.outs = w.h0 = None
        _STATS["aborts"] += 1
        _STATS["last_abort"] = w.abort_reason
        if w.error is not None and not isinstance(w.error, torch.OutOfMemoryError):
            raise w.error
        return None, w.abort_reason
    try:
        w.backward(orig_calls)
    except torch.OutOfMemoryError:
        # The recompute builds a graph the no-grad forward did not: it can run out where the forward fit.
        w.saved = w.outs = w.h0 = None
        for p, g in outside:
            p.grad = g
        _STATS["aborts"] += 1
        _STATS["last_abort"] = "out of memory in the decoder backward"
        return None, _STATS["last_abort"]
    _STATS["windows"] += 1
    _STATS["micro_batches"] += n
    return w.results, None
