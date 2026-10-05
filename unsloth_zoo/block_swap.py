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

"""Block swap: keep frozen decoder layers in pinned host RAM during training.

Host copy is authoritative (frozen under LoRA), so eviction never copies back.
The recompute sweep runs in reverse, so prefetch direction follows grad mode.
Slots come from a fixed depth + 1 pool: per-fetch allocation leaves the peak unchanged.
"""
import os
import sys
import torch
from contextlib import contextmanager, nullcontext
import functools
import gc
import importlib

__all__ = [
    "BlockSwap",
    "find_decoder_layers",
    "swap_indices",
    "estimate_training_reserve_bytes",
    "lora_param_count",
    "auto_swap_indices",
    "build_host_layers",
    "load_layers_to_host",
    "extra_input_embeddings",
    "usable_device_bytes",
]


@contextmanager
def _no_inference_mode():
    # Inference tensors would make later copy_ and autograd saves raise.
    try:
        leave_inference = torch.inference_mode(False)
    except (TypeError, AttributeError):
        leave_inference = nullcontext()
    with leave_inference, torch.no_grad():
        yield

# WSL2 caps pinned memory: fall back to pageable.
_PINNED_MEMORY_AVAILABLE = True
# torch's pinned allocator rounds each allocation up to a power of two, so blocks share packed chunks.
_ALIGN = 512
_CHUNK_BYTES = 256 << 20
# Pin only while this much host RAM stays free: max(4 GiB, 15%); the rest stays pageable.
_PIN_RESERVE_MIN_BYTES = 4 << 30
_PIN_RESERVE_FRACTION = 0.15


def _host_copy(t):
    return t.detach().to("cpu", copy = True).contiguous()


def _pow2_ceil(n):
    return 1 << max(0, (int(n) - 1).bit_length())


def _pin_budget():
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.available - max(_PIN_RESERVE_MIN_BYTES, int(vm.total * _PIN_RESERVE_FRACTION))
    except Exception:
        return 0


class _Registered:
    """Page-locks the first `nbytes` of pageable `buf` in place (exact size, no allocator rounding)."""

    def __init__(self, buf, nbytes):
        self.buf, self.ptr = buf, None
        # cudaHostRegisterPortable: pinned for every device's context, since blocks may fetch onto several cards.
        rc = torch.cuda.cudart().cudaHostRegister(buf.data_ptr(), nbytes, 1)
        if int(rc) != 0:
            raise RuntimeError(f"cudaHostRegister failed: {rc}")
        self.ptr = buf.data_ptr()

    def __del__(self):
        # Before the buffer can be freed: a freed range left registered would pin whatever lands there next.
        if self.ptr is not None:
            try:
                torch.cuda.cudart().cudaHostUnregister(self.ptr)
            except Exception:
                pass
            self.ptr = None


def _host_chunk(nbytes, budget):
    """A uint8 host buffer, pinned while `budget` allows; returns (buffer, bytes pinned)."""
    global _PINNED_MEMORY_AVAILABLE
    if (_PINNED_MEMORY_AVAILABLE and nbytes <= budget
            and os.environ.get("UNSLOTH_DISABLE_PINNED_MEMORY", "0") != "1"):
        try:
            return torch.empty(nbytes, dtype = torch.uint8, pin_memory = True), nbytes
        except RuntimeError as e:
            # Some builds report only "cudaErrorMemoryAllocation".
            text = str(e).lower()
            if "out of memory" not in text and "cudaerrormemoryallocation" not in text: raise
            _PINNED_MEMORY_AVAILABLE = False
    return torch.empty(nbytes, dtype = torch.uint8), 0


_CHECKPOINT_FILE = os.path.join("torch", "utils", "checkpoint.py")


def _autograd_keeps_weights():
    if not torch.is_grad_enabled():
        return False
    # Non-reentrant checkpoint forward saves nothing; its recompute refetches through the hooks.
    try:
        hooks = torch._C._autograd._top_saved_tensors_default_hooks(False)
        return hooks is None or not getattr(hooks[0], "__qualname__", "").startswith("_checkpoint_hook.")
    except (AttributeError, TypeError):
        pass
    # Older torch (2.6) has no hook getter: a grad-on forward inside checkpoint() is the non-reentrant one.
    f = sys._getframe(1)
    while f is not None:
        code = f.f_code
        if code.co_name == "checkpoint" and code.co_filename.endswith(_CHECKPOINT_FILE):
            return False
        f = f.f_back
    return True


def _swappable(module):
    out = []
    for name, p in module.named_parameters(recurse = True):
        if p.requires_grad or "lora_" in name:
            continue
        out.append((name, p))
    return out


class _Block:
    __slots__ = ("params", "host", "devices", "index", "streams", "events", "resident", "slot", "sig",
                 "pending", "layout", "sizes", "src", "empties", "host_buf", "home")

    def __init__(self, layer, streams, device, shared = ()):
        self.params, self.host, self.devices = [], [], []
        index = {}
        for name, p in _swappable(layer):
            if id(p) in shared:
                continue
            index[id(p)] = len(self.params)
            self.params.append(p)
            self.devices.append(device if p.data.device.type == "cpu" else p.data.device)
        self.index = index
        # Each device's weights sit back to back, so a fetch is one copy per device, not one per tensor.
        self.layout, self.sizes = [], {}
        for p, d in zip(self.params, self.devices):
            off = self.sizes.get(d, 0)
            self.layout.append(off)
            self.sizes[d] = off + -(-p.data.nbytes // _ALIGN) * _ALIGN
        for d in self.devices:
            if d not in streams:
                streams[d] = torch.cuda.Stream(device = d)
        self.streams = streams
        self.events = {d: torch.cuda.Event() for d in self.sizes}
        self.empties = [torch.empty(0, device = d, dtype = p.dtype) for p, d in zip(self.params, self.devices)]
        self.resident = True
        self.slot = None
        # Ran with grad on and autograd holds its weights until the backward hook fires.
        self.pending = False
        self.src, self.host_buf = {}, {}
        self.home = self.devices[0] if self.devices else None
        self.sig = tuple((tuple(p.shape), p.dtype, d, off) for p, d, off in zip(self.params, self.devices, self.layout))

    def nbytes(self):
        return sum(self.sizes.values())

    def pack(self, regions):
        """Copy the weights into host `regions` {device: uint8 view}; host[i] become views into them."""
        self.host_buf = regions
        with torch.no_grad():
            for p, d, off in zip(self.params, self.devices, self.layout):
                view = regions[d][off:off + p.data.nbytes].view(p.dtype).view(p.shape)
                view.copy_(p.data)
                self.host.append(view)

    def evict(self):
        if not self.resident:
            return
        for p, e in zip(self.params, self.empties):
            p.data = e
        self.resident = False
        self.pending = False
        freed, self.slot = self.slot, None
        return freed

    def prefetch(self, slot):
        if self.resident:
            return
        bufs, views = slot
        for d, event in self.events.items():
            stream = self.streams[d]
            stream.wait_stream(torch.cuda.current_stream(d))
            with torch.cuda.stream(stream):
                bufs[d].copy_(self.host_buf[d], non_blocking = True)
                event.record(stream)
        for p, v in zip(self.params, views):
            p.data = v
        self.slot = slot
        self.resident = True

    def wait(self):
        for d, event in self.events.items():
            torch.cuda.current_stream(d).wait_event(event)


def _to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj if obj.device == device else obj.to(device, non_blocking = True)
    if isinstance(obj, (tuple, list)):
        moved = [_to_device(o, device) for o in obj]
        return type(obj)(*moved) if hasattr(obj, "_fields") else type(obj)(moved)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


def _opaque(hook):
    # Stream copies, event waits and `.data` swaps cannot be traced; compiled callers break around hooks.
    disable = getattr(getattr(torch, "compiler", None), "disable", None)
    return disable(hook) if disable is not None else hook


def swap_indices(total, n, placement = "spread"):
    """"spread": evenly spaced, ending at the last layer (each copy hides behind total / n layers); "tail": last n."""
    n = max(0, min(int(n), total))
    if placement == "tail":
        return list(range(total - n, total))
    return [(k + 1) * total // n - 1 for k in range(n)]


class BlockSwap:
    """Install on a layer list; layers at `n` (a count or explicit indices) live on the host."""

    def __init__(self, layers, n, prefetch_depth = 2, device = None, placement = "tail"):
        self.blocks, self.handles = [], []
        self.depth = max(1, prefetch_depth)
        self.streams = {}
        self.free = {}
        self._grew = False
        self._chunks = []
        self.pinned_bytes = 0
        if isinstance(n, int):
            self.indices = swap_indices(len(layers), n, placement)
        else:
            self.indices = sorted({int(i) % len(layers) for i in n})
        self.pos = {li: k for k, li in enumerate(self.indices)}
        self.start = self.indices[0] if self.indices else len(layers)
        self._input_hooked = [False] * len(self.indices)
        self.spans_devices, self.layer_device = False, None
        if not self.indices:
            return
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = torch.device(device)
        swapped = [layers[i] for i in self.indices]
        # A param shared by two layers stays on the card: evicting it under one empties it for the other.
        owners = {}
        for li, layer in enumerate(layers):
            for _, p in _swappable(layer):
                owners.setdefault(id(p), set()).add(li)
        shared = {pid for pid, o in owners.items() if len(o) > 1}
        self.blocks = [_Block(layer, self.streams, self.device, shared) for layer in swapped]
        homes = {b.home for b in self.blocks if b.home is not None}
        for li, layer in enumerate(layers):
            if li not in self.pos:
                homes.update(p.device for _, p in _swappable(layer) if p.device.type == "cuda")
        # Code that bypasses the pre-hook (fast decode loops) must check this: only the hook moves inputs across cards.
        self.spans_devices = len(homes) > 1
        self.layer_device = next(iter(homes)) if len(homes) == 1 else None
        for layer in swapped:
            for name, p in layer.named_parameters():
                if (p.device.type == "cpu") and (p.requires_grad or "lora_" in name or id(p) in shared):
                    p.data = p.data.to(self.device)

        # Evict before building the pool (originals + pool would OOM); roll back on any failure.
        try:
            for i, layer in enumerate(swapped):
                self.handles.append(layer.register_forward_pre_hook(self._pre(i), with_kwargs = True))
                self.handles.append(layer.register_forward_hook(self._post(i)))
                # Evicted weights are empty: state_dict must read the host copy.
                self.handles.append(layer._register_state_dict_hook(self._state_dict(i)))

            self._pack()
            total = self.host_bytes()
            if self.pinned_bytes < total:
                # Pageable copies are several times slower than pinned ones and cannot hide behind compute.
                print(f"Unsloth: offload_layers pinned {self.pinned_bytes / 2**30:.1f} of {total / 2**30:.1f} GiB of host "
                      "memory; the rest is copied from pageable memory, several times slower. Free host RAM "
                      "or lower offload_layers for full speed.")

            sigs = [b.sig for b in self.blocks]
            for sig in set(sigs):
                self.free[sig] = [self._new_slot(sig) for _ in range(min(self.depth + 1, sigs.count(sig)))]

            self._arm(forward = True)
        except Exception:
            self.remove()
            raise

    def _pack(self):
        # Block by block, releasing each original as soon as it is packed: never two host copies of the tail.
        if self._pack_registered():
            return
        left = sum(b.nbytes() for b in self.blocks)
        chunk = max(_CHUNK_BYTES, _pow2_ceil(max(max(b.sizes.values(), default = 0) for b in self.blocks)))
        budget = _pin_budget()
        buf, used = None, 0
        for b in self.blocks:
            regions = {}
            for d, need in b.sizes.items():
                if buf is None or used + need > buf.numel():
                    buf, pinned = _host_chunk(chunk if left >= chunk else _pow2_ceil(left), budget - self.pinned_bytes)
                    self.pinned_bytes += pinned
                    self._chunks.append(buf)
                    used = 0
                regions[d] = buf[used:used + need]
                used += need
                left -= need
            b.pack(regions)
            self._release(b)

    def _pack_registered(self):
        # One exact-size pageable arena, page-locked in place up to the budget (cudaHostRegister).
        global _PINNED_MEMORY_AVAILABLE
        if (not _PINNED_MEMORY_AVAILABLE or os.environ.get("UNSLOTH_DISABLE_PINNED_MEMORY", "0") == "1"
                or not hasattr(torch.cuda.cudart(), "cudaHostRegister")):
            return False
        total = sum(b.nbytes() for b in self.blocks)
        arena = torch.empty(total, dtype = torch.uint8)
        used, ends = 0, []
        for b in self.blocks:
            regions = {}
            for d, need in b.sizes.items():
                regions[d] = arena[used:used + need]
                used += need
            b.pack(regions)
            self._release(b)
            ends.append(used)
        # Pin whole blocks only: a copy from half-pinned memory is staged synchronously.
        budget = _pin_budget()
        pin = max([e for e in ends if e <= budget], default = 0)
        self._chunks.append(arena)
        if pin > 0:
            try:
                self._chunks.append(_Registered(arena, pin))
                self.pinned_bytes = pin
            except Exception:
                _PINNED_MEMORY_AVAILABLE = False
        return True

    def _release(self, block):
        freed = block.evict()
        if freed is not None:
            self.free[block.sig].append(freed)

    def _acquire(self, block, steal):
        free = self.free[block.sig]
        if not free and steal:
            # Only after an interrupted step; prefetches never steal (could evict the running block).
            # A pending block's slot is still read by backward: reusing it corrupts gradients silently.
            for b in self.blocks:
                if b is not block and b.sig == block.sig and b.resident and b.slot is not None and not b.pending:
                    self._release(b)
                    break
            else:
                # Grad-on forward order (no per-layer checkpoint, or several layers in one region).
                if not self._grew:
                    print("Unsloth: block_swap needs per-layer gradient checkpointing to save memory; "
                          "growing the slot pool instead so gradients stay correct.")
                    self._grew = True
                return self._new_slot(block.sig)
        return free.pop() if free else None

    def _new_slot(self, sig):
        sizes = {}
        for shape, dt, dv, off in sig:
            n = torch.Size(shape).numel() * torch.empty(0, dtype = dt).element_size()
            sizes[dv] = max(sizes.get(dv, 0), off + -(-n // _ALIGN) * _ALIGN)
        with _no_inference_mode():
            bufs = {dv: torch.empty(n, dtype = torch.uint8, device = dv) for dv, n in sizes.items()}
            views = [bufs[dv][off:off + torch.Size(shape).numel() * torch.empty(0, dtype = dt).element_size()]
                     .view(dt).view(shape) for shape, dt, dv, off in sig]
        return bufs, views

    def _fetch(self, block, steal = False):
        if block.resident:
            return
        slot = self._acquire(block, steal)
        if slot is not None:
            block.prefetch(slot)

    def _arm(self, forward):
        rng = range(self.depth) if forward else range(len(self.blocks) - 1,
                                                      len(self.blocks) - 1 - self.depth, -1)
        for i in rng:
            if 0 <= i < len(self.blocks):
                self._fetch(self.blocks[i])

    def _pre(self, i):
        def hook(module, args, kwargs):
            b = self.blocks[i]
            self._fetch(b, steal = True)
            b.wait()
            # Weights kept for backward = recompute sweep, which walks layers in reverse.
            nxt = i - self.depth if _autograd_keeps_weights() else i + self.depth
            if 0 <= nxt < len(self.blocks):
                self._fetch(self.blocks[nxt])
            # Release on the input-grad hook: a full backward hook adds a node that reorders grad sums of shared inputs.
            x = next((a for a in args if isinstance(a, torch.Tensor)), None)
            if x is None:
                x = kwargs.get("hidden_states")
            moved = None
            if isinstance(x, torch.Tensor) and b.home is not None and x.device != b.home:
                # A host-loaded tail fetches onto the head's card, not necessarily where the hidden states are.
                moved = (_to_device(args, b.home), _to_device(kwargs, b.home))
                x = next((a for a in moved[0] if isinstance(a, torch.Tensor)), None)
                if x is None:
                    x = moved[1].get("hidden_states")
            hooked = False
            if torch.is_grad_enabled() and isinstance(x, torch.Tensor) and x.requires_grad:
                x.register_hook(self._bwd(i))
                hooked = True
            self._input_hooked[i] = hooked
            return moved
        return _opaque(hook)

    def _post(self, i):
        def hook(module, args, output):
            training = getattr(module, "training", False)
            if not _autograd_keeps_weights():
                # A training forward's last blocks are the first ones recompute needs: keep them.
                if not (training and i >= len(self.blocks) - self.depth):
                    self._release(self.blocks[i])
                # Inference: the next forward starts at the first block, so start fetching it now.
                if not training and i == len(self.blocks) - 1:
                    self._arm(forward = True)
                return output
            out = output[0] if isinstance(output, (tuple, list)) else output
            if not (isinstance(out, torch.Tensor) and out.requires_grad):
                self._release(self.blocks[i])
                return output
            self.blocks[i].pending = True
            if not self._input_hooked[i]:
                # No grad-requiring input to hook: release once the whole backward finishes.
                release = self._bwd(i)
                out.register_hook(
                    lambda g: torch.autograd.Variable._execution_engine.queue_callback(lambda: release(None)))
            return output
        return _opaque(hook)

    def _bwd(self, i):
        def hook(grad):
            self._release(self.blocks[i])
            if i == 0:
                self._arm(forward = True)
        return _opaque(hook)

    def _state_dict(self, i):
        def hook(module, state_dict, prefix, local_metadata):
            # Per call: PEFT renames weights after install, and every alias must get the host copy.
            b = self.blocks[i]
            for name, p in module.named_parameters(remove_duplicate = False):
                idx = b.index.get(id(p))
                if idx is not None and prefix + name in state_dict:
                    state_dict[prefix + name] = b.host[idx]
        return hook

    @_opaque
    def enter(self, idx):
        """Make layer `idx` resident for code that bypasses the forward hooks (fast decode)."""
        i = self.pos.get(idx, -1)
        if 0 <= i < len(self.blocks):
            b = self.blocks[i]
            self._fetch(b, steal = True)
            b.wait()
            if i + self.depth < len(self.blocks):
                self._fetch(self.blocks[i + self.depth])

    @_opaque
    def leave(self, idx):
        i = self.pos.get(idx, -1)
        if 0 <= i < len(self.blocks):
            self._release(self.blocks[i])
            if i == len(self.blocks) - 1:
                self._arm(forward = True)

    def reset(self):
        for b in self.blocks:
            self._release(b)
        self._arm(forward = True)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        # Free the pool first, else the restore can OOM.
        self.free = {}
        with _no_inference_mode():
            for b in self.blocks:
                if not b.resident:
                    for p, h, d in zip(b.params, b.host, b.devices):
                        p.data = h.to(d, copy = True)
                    b.resident = True
        # Callers holding this object (the fast decode loop) must see a no-op, not an empty pool.
        self.blocks = []
        self._chunks = []
        for dev in self.streams:
            torch.cuda.synchronize(dev)

    def host_bytes(self):
        return sum(b.nbytes() for b in self.blocks)

    def pool_bytes(self):
        held = [b.slot for b in self.blocks if b.slot is not None]
        free = [slot for slots in self.free.values() for slot in slots]
        return sum(t.numel() for bufs, _ in held + free for t in bufs.values())

    def signatures(self):
        return len(self.free)

    def resident_count(self):
        return sum(b.resident for b in self.blocks)


def _text_config(config):
    candidates = []
    get = getattr(config, "get_text_config", None)
    if callable(get):
        try:
            candidates.append(get())
        except Exception:
            pass
    # T5Gemma on transformers 4.56 returns the outer config, which has no width: try the sub-configs.
    candidates += [getattr(config, "text_config", None), getattr(config, "decoder", None), config]
    for c in candidates:
        if c is not None and any(getattr(c, k, None) for k in ("hidden_size", "n_embd", "d_model")):
            return c
    return next(c for c in candidates if c is not None)


# Measured 42 on Llama-3.1-8B 4-bit with Unsloth checkpointing, rounded up.
_ACTIVATION_BYTES_PER_TOKEN_HIDDEN = 48


def estimate_training_reserve_bytes(config, seq_len, batch_size = 1, extra_bytes = 0,
                                    logit_rows = 2048, safety_bytes = 256 << 20, fragmentation = 1 / 16):
    """Activations + fp32 logits (`logit_rows` rows) + `extra_bytes`, plus `fragmentation` for split allocator blocks."""
    text = _text_config(config)
    hidden = (getattr(text, "hidden_size", None) or getattr(text, "n_embd", None)
              or getattr(text, "d_model", None) or 0)
    vocab = getattr(text, "vocab_size", None) or 0
    tokens = max(1, int(seq_len)) * max(1, int(batch_size))
    activations = tokens * int(hidden) * _ACTIVATION_BYTES_PER_TOKEN_HIDDEN
    logits = min(tokens, int(logit_rows)) * int(vocab) * 4
    base = activations + logits + int(extra_bytes)
    return int(base + base * fragmentation + int(safety_bytes))


def lora_param_count(layers, r = 16):
    """r * (in + out) per linear in `layers`, per expert on fused [experts, in, out] weights."""
    total = 0
    for layer in layers:
        for module in layer.modules():
            i, o = getattr(module, "in_features", None), getattr(module, "out_features", None)
            if isinstance(i, int) and isinstance(o, int) and not list(module.children()):
                total += r * (i + o)
                continue
            weight = getattr(module, "weight", None)
            # transformers Conv1D (GPT-2 family): `nf` outputs, weight stored [in, out].
            if isinstance(getattr(module, "nf", None), int) and weight is not None and weight.dim() == 2:
                total += r * (weight.shape[0] + weight.shape[1])
                continue
            for name, p in module.named_parameters(recurse = False):
                if p.dim() == 3 and "lora_" not in name:
                    total += r * p.shape[0] * (p.shape[1] + p.shape[2])
    return total


def usable_device_bytes(device):
    """Bytes this process can still allocate on `device`."""
    return _free_device_bytes(device)


def _free_device_bytes(device):
    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    # Blocks torch's caching allocator holds but does not use are free to this process.
    usable = free + torch.cuda.memory_reserved(device) - allocated
    # torch.cuda.set_per_process_memory_fraction caps this process below what the card has free.
    get_fraction = getattr(torch.cuda, "get_per_process_memory_fraction", None)
    try:
        fraction = get_fraction(device) if get_fraction is not None else 1.0
    except Exception:
        fraction = 1.0
    if fraction < 1.0:
        usable = min(usable, int(total * fraction) - allocated)
    return max(0, int(usable))


def _layer_signature(params):
    return tuple((tuple(p.shape), p.dtype) for _, p in params)


def _pool_bytes(sizes, depth, sigs = None):
    # One pool per shape signature (as BlockSwap keys them), depth + 1 slots each, fewer if fewer blocks share it.
    depth = max(1, depth)  # as BlockSwap
    counts = {}
    for i, b in enumerate(sizes):
        key = sigs[i] if sigs is not None else b
        size, count = counts.get(key, (b, 0))
        counts[key] = (max(size, b), count + 1)
    return sum(b * min(depth + 1, c) for b, c in counts.values())


def auto_swap_indices(layers, reserve_bytes, prefetch_depth = 2, free_bytes = None):
    """Fewest layers to move to host RAM so each GPU keeps `reserve_bytes` free, net of the slot pool.

    Returns (indices, shortfall_left); shortfall_left > 0 when one layer per device must stay and it is not enough."""
    # Params shared across layers stay on the card under BlockSwap, so they free nothing.
    owners = {}
    for i, layer in enumerate(layers):
        for _, p in _swappable(layer):
            owners.setdefault(id(p), set()).add(i)
    by_device = {}
    for i, layer in enumerate(layers):
        params = [(k, p) for k, p in _swappable(layer) if len(owners[id(p)]) == 1]
        if not params or params[0][1].device.type != "cuda":
            continue
        by_device.setdefault(params[0][1].device, []).append(
            (i, sum(p.nbytes for _, p in params), _layer_signature(params))
        )
    chosen, left = [], 0
    for device, items in by_device.items():
        free = (free_bytes or {}).get(device)
        if free is None:
            free = _free_device_bytes(device)
        need = int(reserve_bytes) - free
        if need <= 0:
            continue
        pick = []
        for n in range(1, len(items)):
            pick = [items[k] for k in swap_indices(len(items), n)]
            sizes, sigs = [b for _, b, _ in pick], [g for _, _, g in pick]
            if sum(sizes) - _pool_bytes(sizes, prefetch_depth, sigs) >= need:
                break
        else:
            sizes, sigs = [b for _, b, _ in pick], [g for _, _, g in pick]
            left = max(left, need - (sum(sizes) - _pool_bytes(sizes, prefetch_depth, sigs)))
        chosen += [i for i, _, _ in pick]
    return sorted(chosen), left


def find_decoder_layers(model):
    m = model
    for _ in range(6):
        for attr in ("model", "base_model", "transformer", "language_model"):
            inner = getattr(m, attr, None)
            if inner is not None and hasattr(inner, "layers"):
                return inner.layers
            if inner is not None and inner is not m:
                m = inner
                break
        else:
            break
    if hasattr(m, "layers"):
        return m.layers
    # Other layouts (`transformer.h`, `decoder.block`): the heaviest block list; classes may differ (Mllama).
    best, best_bytes = None, 0
    for mod in model.modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 2:
            sizes = [sum(p.nbytes for n, p in c.named_parameters() if "lora_" not in n) for c in mod]
            if min(sizes) > 0 and sum(sizes) > best_bytes:
                best, best_bytes = mod, sum(sizes)
    if best is not None:
        return best
    raise RuntimeError("could not locate decoder layers")


def build_host_layers(make_layer, first_idx, count, tensors, device, compute_dtype,
                      quantize_4bit = False, skip_modules = (), prefix = "model.layers."):
    """Build layers [first_idx, first_idx + count) with frozen weights in pinned host RAM; quant_state stays on `device`."""
    import bitsandbytes as bnb
    device = torch.device(device)
    if any(".quant_state." in k for k in tensors):
        quantize_4bit = False
    out = []
    for idx in range(first_idx, first_idx + count):
        with torch.device("meta"):
            layer = make_layer(idx)
        base = f"{prefix}{idx}."
        for mname, mod in list(layer.named_modules()):
            if not isinstance(mod, torch.nn.Linear):
                continue
            key = base + mname + ".weight"
            stats = {k[len(key) + 1:]: tensors[k] for k in tensors if k.startswith(key + ".")}
            bias = None
            if mod.bias is not None:
                bias = torch.nn.Parameter(_host_copy(tensors[base + mname + ".bias"]().to(compute_dtype)),
                                          requires_grad = False)
            skipped = any(s in mname.split(".") for s in skip_modules)
            if stats or (quantize_4bit and not skipped):
                if stats:
                    w = bnb.nn.Params4bit.from_prequantized(
                        tensors[key](), {k: v() for k, v in stats.items()}, requires_grad = False, device = device)
                else:
                    w = bnb.nn.Params4bit(tensors[key]().to(compute_dtype), requires_grad = False,
                                          compress_statistics = True, quant_type = "nf4",
                                          quant_storage = torch.uint8).to(device)
                new = bnb.nn.Linear4bit(mod.in_features, mod.out_features, bias = bias is not None,
                                        compute_dtype = compute_dtype, quant_type = w.quant_type,
                                        quant_storage = w.quant_storage, device = "meta")
                w.module = new
                # quant_state keeps the checkpoint dtype; match the compute dtype the resident layers got.
                w.quant_state.dtype = compute_dtype
                new.weight = w
                new.quant_state = w.quant_state
                w.data = _host_copy(w.data)
                torch.cuda.empty_cache()
            else:
                new = torch.nn.Linear(mod.in_features, mod.out_features, bias = bias is not None, device = "meta")
                new.weight = torch.nn.Parameter(_host_copy(tensors[key]().to(compute_dtype)),
                                                requires_grad = False)
            if bias is not None:
                new.bias = bias
            parent_name, _, child = mname.rpartition(".")
            setattr(layer.get_submodule(parent_name) if parent_name else layer, child, new)
        for pname, p in list(layer.named_parameters()):
            if p.device.type != "meta":
                continue
            t = tensors[base + pname]()
            if t.is_floating_point():
                t = t.to(compute_dtype)
            parent_name, _, child = pname.rpartition(".")
            parent = layer.get_submodule(parent_name) if parent_name else layer
            setattr(parent, child, torch.nn.Parameter(_host_copy(t), requires_grad = False))
        for bname, b in layer.named_buffers():
            if b.device.type == "meta":
                raise RuntimeError(f"block_swap: {base}{bname} is a buffer with no checkpoint value")
        out.append(layer)
    return out


# Extra token tables (Gemma 3n / 4 per-layer embeddings) worth host RAM; small position tables stay.
EXTRA_EMBEDDING_MIN_BYTES = 256 << 20


def extra_input_embeddings(model):
    """Large nn.Embeddings besides the input embedding; lm_head only ties to the input one, so never shared."""
    try:
        main = model.get_input_embeddings()
    except Exception:
        main = None
    out = []
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if module is main or not isinstance(module, torch.nn.Embedding) or weight is None:
            continue
        if weight.numel() * weight.element_size() >= EXTRA_EMBEDDING_MIN_BYTES:
            out.append((name, module))
    return out


class _HostLoad:
    """What `load_layers_to_host` moved: the layer list and the indices living in host RAM."""

    def __init__(self, n, placement, embeddings = False):
        self.n, self.placement, self.want_embeddings = n, placement, embeddings
        self.layers, self.indices, self.prefixes = None, [], {}
        self.done = set()
        self.embedding_prefixes, self.embeddings = {}, []

    def bind(self, model):
        if self.layers is not None:
            return
        layers = find_decoder_layers(model)
        name = next((k for k, m in model.named_modules() if m is layers), None)
        self.layers = layers
        self.indices = swap_indices(len(layers), self.n, self.placement) if isinstance(self.n, int) \
            else sorted({int(i) % len(layers) for i in self.n})
        if name is not None:
            self.prefixes = {f"{name}.{i}.": i for i in self.indices}
        if self.want_embeddings:
            self.embedding_prefixes = {f"{n}.": m for n, m in extra_input_embeddings(model)}

    def evict_embedding(self, module):
        weight = getattr(module, "weight", None)
        if module in self.embeddings or weight is None or weight.device.type == "meta":
            return
        with torch.no_grad():
            weight.requires_grad_(False)
            if weight.device.type != "cpu":
                weight.data = weight.data.to("cpu")
        self.embeddings.append(module)

    def evict(self, i, force = False):
        if i in self.done:
            return
        layer = self.layers[i]
        params = list(layer.parameters())
        if not force and any(p.device.type == "meta" for p in params):
            return
        with torch.no_grad():
            for p in params:
                if p.is_floating_point():
                    p.requires_grad_(False)
                if p.device.type not in ("cpu", "meta"):
                    p.data = p.data.to("cpu")
        self.done.add(i)

    def on_param(self, model, target_name):
        self.bind(model)
        for prefix, i in self.prefixes.items():
            if target_name.startswith(prefix):
                self.evict(i)
                return
        for prefix, module in self.embedding_prefixes.items():
            if target_name.startswith(prefix):
                self.evict_embedding(module)
                return


@contextmanager
def load_layers_to_host(n, placement = "spread", embeddings = False):
    """Wrap `from_pretrained`: each chosen layer moves to host RAM once its last (quantized) weight lands.

    Install `BlockSwap` on the yielded `.layers` / `.indices` afterwards; `embeddings` also streams
    `extra_input_embeddings` (`.embeddings`, the caller hooks their lookups)."""
    state = _HostLoad(n, placement, embeddings)
    patched = []
    try:
        core = importlib.import_module("transformers.core_model_loading")
        original = getattr(core, "set_param_for_module", None)
    except ImportError:
        core, original = None, None
    if original is not None:
        # transformers 5: the one sink every loaded parameter goes through.
        @functools.wraps(original)
        def set_param_for_module(*args, **kwargs):
            out = original(*args, **kwargs)
            model = kwargs.get("model", args[0] if args else None)
            name = kwargs.get("target_name", args[1] if len(args) > 1 else None)
            if model is not None and isinstance(name, str):
                state.on_param(model, name)
            return out
        core.set_param_for_module = set_param_for_module
        patched.append((core, "set_param_for_module", original))
    else:
        # transformers 4: weights load shard by shard; sweep after each shard.
        mu = importlib.import_module("transformers.modeling_utils")
        original = mu._load_state_dict_into_meta_model

        @functools.wraps(original)
        def _load_state_dict_into_meta_model(model, *args, **kwargs):
            out = original(model, *args, **kwargs)
            state.bind(model)
            for i in state.indices:
                state.evict(i)
            # The plan counted streamed tables off the card during the load, so they leave per shard too.
            for module in state.embedding_prefixes.values():
                state.evict_embedding(module)
            return out
        mu._load_state_dict_into_meta_model = _load_state_dict_into_meta_model
        patched.append((mu, "_load_state_dict_into_meta_model", original))
    # The allocator warmup reserves the whole model's bytes on the card before loading: skip it.
    mu = importlib.import_module("transformers.modeling_utils")
    warmup = getattr(mu, "caching_allocator_warmup", None)
    if warmup is not None:
        mu.caching_allocator_warmup = lambda *args, **kwargs: None
        patched.append((mu, "caching_allocator_warmup", warmup))
    # The async loader materializes every tensor on the card up front; load one at a time.
    old_env = os.environ.get("HF_DEACTIVATE_ASYNC_LOAD")
    os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
    try:
        yield state
    finally:
        for module, attr, fn in patched:
            setattr(module, attr, fn)
        if old_env is None:
            os.environ.pop("HF_DEACTIVATE_ASYNC_LOAD", None)
        else:
            os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = old_env
    # Layers with weights the checkpoint lacked finished only once missing keys were initialized.
    if state.layers is not None:
        for i in state.indices:
            state.evict(i, force = True)
        for module in state.embedding_prefixes.values():
            state.evict_embedding(module)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
