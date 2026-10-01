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

__all__ = [
    "BlockSwap",
    "find_decoder_layers",
    "swap_indices",
    "build_host_layers",
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
# torch's pinned allocator rounds every allocation up to a power of two (a 4.5 MiB blob holds 8 MiB),
# so blocks are packed into shared power-of-two chunks, as diffusion group offload does.
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
        rc = torch.cuda.cudart().cudaHostRegister(buf.data_ptr(), nbytes, 0)
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
                 "pending", "layout", "sizes", "src", "empties", "host_buf")

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


def swap_indices(total, n, placement = "spread"):
    """Which of `total` layers to swap. "spread" spaces them evenly (ending at the last layer), so each
    copy hides behind total / n layers of compute instead of one; "tail" takes the last n."""
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
        if not self.indices:
            return
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = torch.device(device)
        swapped = [layers[i] for i in self.indices]
        # A param reachable from two layers (tied / shared blocks) stays on the card: evicting it
        # under one layer would empty it for the other.
        owners = {}
        for li, layer in enumerate(layers):
            for _, p in _swappable(layer):
                owners.setdefault(id(p), set()).add(li)
        shared = {pid for pid, o in owners.items() if len(o) > 1}
        self.blocks = [_Block(layer, self.streams, self.device, shared) for layer in swapped]
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
                # Pageable copies run at a fraction of pinned bandwidth (5-7 vs 55 GB/s measured) and cannot hide.
                print(f"Unsloth: block_swap pinned {self.pinned_bytes / 2**30:.1f} of {total / 2**30:.1f} GiB of host "
                      "memory; the rest is copied from pageable memory, several times slower. Free host RAM "
                      "or lower block_swap_layers for full speed.")

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
            # The grad of the block's input is complete only once the block's backward has run, so its
            # hook is the release point. A tensor hook adds no autograd node, unlike a full backward hook,
            # which reorders gradient sums of tensors fed to several blocks (cross-attention states).
            hooked = False
            if torch.is_grad_enabled():
                x = next((a for a in args if isinstance(a, torch.Tensor)), None)
                if x is None:
                    x = kwargs.get("hidden_states")
                if isinstance(x, torch.Tensor) and x.requires_grad:
                    x.register_hook(self._bwd(i))
                    hooked = True
            self._input_hooked[i] = hooked
            return None
        return hook

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
                # Nothing will backpropagate through this call.
                self._release(self.blocks[i])
                return output
            self.blocks[i].pending = True
            if not self._input_hooked[i]:
                # No grad-requiring input to hook: release once the whole backward finishes.
                release = self._bwd(i)
                out.register_hook(
                    lambda g: torch.autograd.Variable._execution_engine.queue_callback(lambda: release(None)))
            return output
        return hook

    def _bwd(self, i):
        def hook(grad):
            self._release(self.blocks[i])
            if i == 0:
                self._arm(forward = True)
        return hook

    def _state_dict(self, i):
        def hook(module, state_dict, prefix, local_metadata):
            # Per call: PEFT renames weights after install, and every alias must get the host copy.
            b = self.blocks[i]
            for name, p in module.named_parameters(remove_duplicate = False):
                idx = b.index.get(id(p))
                if idx is not None and prefix + name in state_dict:
                    state_dict[prefix + name] = b.host[idx]
        return hook

    def enter(self, idx):
        """Make layer `idx` resident for code that bypasses the forward hooks (fast decode)."""
        i = self.pos.get(idx, -1)
        if 0 <= i < len(self.blocks):
            b = self.blocks[i]
            self._fetch(b, steal = True)
            b.wait()
            if i + self.depth < len(self.blocks):
                self._fetch(self.blocks[i + self.depth])

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
    # Other layouts (`transformer.h`, `decoder.block`, `decoder.layers`): the list of blocks holding the
    # most weight. Classes may differ (Mllama interleaves cross-attention layers).
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
    """Build layers [first_idx, first_idx + count) with frozen weights in pinned host RAM.
    `tensors` maps checkpoint keys to zero-arg loaders; quant_state stays on `device`."""
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
