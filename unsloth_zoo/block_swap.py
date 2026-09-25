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
"""Block swap: keep frozen decoder layers in pinned host RAM during training.

Base weights are frozen under LoRA, so eviction never copies back: the host
tensor stays authoritative and the device copy is dropped. For bnb Params4bit
only the packed uint8 blob travels; quant_state is small and stays resident.

Three things this has to get right.

Direction. Unsloth's checkpointer runs the layer twice per step -- once in
forward under no_grad, once during backward under enable_grad -- and the second
sweep walks the layers in reverse. A prefetcher that only looks forward fetches
layers that already finished, which then sit resident and save nothing.

Allocation. Fetching into a fresh tensor each time leaves the allocator's
high-water mark near the no-swap baseline: average residency drops but the peak
does not, so the card you need is barely smaller. Slots come from a fixed pool
instead, sized depth + 1, which is what makes the peak fall.

Streams. The copy runs on a side stream while the layer consumes the weight on
the default one. Pool slots are never returned to the allocator, so nothing can
recycle them underneath a copy; the hazard is instead reusing a slot while its
last reader is still running, which is why the side stream waits on the consumer
before writing and the consumer waits on the event before reading.
"""
import os
import torch

__all__ = [
    "BlockSwap",
    "find_decoder_layers",
]

# Pinning lets the H2D prefetch overlap with compute, but WSL2 caps the pinned
# budget and a single .pin_memory() can OOM there while ordinary RAM is free.
# Fall back to pageable host memory (slower, no overlap) instead of failing, and
# honor UNSLOTH_DISABLE_PINNED_MEMORY, mirroring gradient_checkpointing. Kept
# local because this module is loaded standalone (no package-relative imports).
_PINNED_MEMORY_AVAILABLE = True


def _to_pinned_host(t):
    global _PINNED_MEMORY_AVAILABLE
    host = t.to("cpu", copy = True)
    if not _PINNED_MEMORY_AVAILABLE or os.environ.get("UNSLOTH_DISABLE_PINNED_MEMORY", "0") == "1":
        return host
    try:
        return host.pin_memory()
    except RuntimeError as e:
        # Some torch/CUDA builds report pinned exhaustion as the compact
        # "cudaErrorMemoryAllocation" with no spaced "out of memory" phrase;
        # treat both as host-alloc OOM, mirroring gradient_checkpointing.
        text = str(e).lower()
        if "out of memory" not in text and "cudaerrormemoryallocation" not in text: raise
        _PINNED_MEMORY_AVAILABLE = False
        return host


def _swappable(module):
    out = []
    for name, p in module.named_parameters(recurse = True):
        if p.requires_grad or "lora_" in name:
            continue
        out.append((name, p))
    return out


class _Block:
    __slots__ = ("params", "host", "devices", "names", "stream", "event", "resident", "slot", "sig")

    def __init__(self, layer, streams):
        self.params, self.host, self.devices, self.names = [], [], [], []
        for name, p in _swappable(layer):
            self.params.append(p)
            self.host.append(_to_pinned_host(p.data))
            self.devices.append(p.data.device)
            self.names.append(name)
        # One side stream per device, so a sharded model keeps each card's
        # copies on that card's stream.
        dev = self.devices[0] if self.devices else torch.device("cuda", 0)
        if dev not in streams:
            streams[dev] = torch.cuda.Stream(device = dev)
        self.stream = streams[dev]
        self.event = torch.cuda.Event()
        self.resident = True
        self.slot = None
        # Hashable shape signature; blocks with the same one share a pool.
        self.sig = tuple((tuple(h.shape), h.dtype, d) for h, d in zip(self.host, self.devices))

    def evict(self):
        if not self.resident:
            return
        for p, d in zip(self.params, self.devices):
            p.data = torch.empty(0, device = d, dtype = p.data.dtype)
        self.resident = False
        freed, self.slot = self.slot, None
        return freed

    def prefetch(self, slot):
        """Copy into `slot`, a pre-allocated set of device tensors from the pool.

        Allocating per fetch instead leaves the allocator's high-water mark near
        the no-swap baseline: average residency falls but the peak does not, so
        the card you need is barely smaller. A fixed pool keeps device use flat.
        """
        if self.resident:
            return
        self.stream.wait_stream(torch.cuda.current_stream(self.stream.device))
        with torch.cuda.stream(self.stream):
            for dst, h in zip(slot, self.host):
                dst.copy_(h, non_blocking = True)
            self.event.record(self.stream)
        for p, dst in zip(self.params, slot):
            p.data = dst
        self.slot = slot
        self.resident = True

    def wait(self):
        torch.cuda.current_stream(self.stream.device).wait_event(self.event)

    def nbytes(self):
        return sum(h.numel() * h.element_size() for h in self.host)


class BlockSwap:
    """Install on a decoder-layer list; the tail `n` of them live on the host."""

    def __init__(self, layers, n, prefetch_depth = 2):
        # Fully initialize state up front so a disabled swap (n <= 0) can still
        # be managed through reset()/pool_bytes()/signatures() without blowing up.
        self.blocks, self.handles = [], []
        self.depth = max(1, prefetch_depth)
        self.streams = {}
        self.free = {}
        self.start = len(layers)
        if n <= 0:
            return
        n = min(n, len(layers))
        self.start = len(layers) - n
        self.blocks = [_Block(l, self.streams) for l in layers[self.start:]]

        for i, layer in enumerate(layers[self.start:]):
            self.handles.append(layer.register_forward_pre_hook(self._pre(i)))
            self.handles.append(layer.register_forward_hook(self._post(i)))
            self.handles.append(layer.register_full_backward_hook(self._bwd(i)))
            # Evicted blocks hold empty weight tensors, so a state_dict() taken
            # mid-training (periodic full-model checkpoints, save_pretrained on a
            # merged model) would serialize zero-length base weights. The host
            # copy is authoritative for these frozen params, so substitute it for
            # every swapped weight regardless of residency.
            self.handles.append(layer._register_state_dict_hook(self._state_dict(i)))

        # depth + 1 slots per shape signature is the most that can be live at
        # once: the block being consumed plus the ones in flight. Layers are not
        # all alike -- Unsloth's dynamic 4-bit quants leave some blocks
        # unquantized -- so each distinct signature gets its own pool. Cap by the
        # number of blocks with that signature: a signature with fewer than
        # depth + 1 blocks can never have more than that many slots live, and
        # over-allocating just holds extra device copies for nothing.
        # Drop the original device weights before allocating the pool. While
        # they are all still resident, originals + pool is a higher peak than
        # the steady state (unswapped weights + depth + 1 slots) and can OOM in
        # exactly the memory-constrained setups this is meant to fit. Eviction
        # is safe here: every block's slot is still None, so _release frees
        # device storage without touching the not-yet-built pool.
        #
        # Between eviction and the pool being built the layers hold empty weight
        # tensors, so a pool allocation OOM here would leave the model broken
        # with no object for the caller to recover through. Restore weights and
        # pull the hooks on any failure so the caller can fall back to the
        # untouched model, then re-raise.
        try:
            for b in self.blocks:
                self._release(b)

            sigs = [b.sig for b in self.blocks]
            for sig in set(sigs):
                self.free[sig] = [
                    [torch.empty(shape, dtype = dt, device = dv) for shape, dt, dv in sig]
                    for _ in range(min(self.depth + 1, sigs.count(sig)))
                ]

            self._arm(forward = True)
        except Exception:
            self.remove()
            raise

    def _release(self, block):
        freed = block.evict()
        if freed is not None:
            self.free[block.sig].append(freed)

    def _acquire(self, sig):
        free = self.free[sig]
        if not free:
            # Every slot of this shape is live. With depth + 1 per signature
            # this should not happen, so it is a guard, not a hot path.
            for b in self.blocks:
                if b.sig == sig and b.resident and b.slot is not None:
                    self._release(b)
                    break
        return free.pop() if free else None

    def _fetch(self, block):
        if block.resident:
            return
        slot = self._acquire(block.sig)
        if slot is not None:
            block.prefetch(slot)

    def _arm(self, forward):
        rng = range(self.depth) if forward else range(len(self.blocks) - 1,
                                                      len(self.blocks) - 1 - self.depth, -1)
        for i in rng:
            if 0 <= i < len(self.blocks):
                self._fetch(self.blocks[i])

    def _pre(self, i):
        def hook(module, args):
            b = self.blocks[i]
            self._fetch(b)
            b.wait()
            # grad off means the checkpointer's first sweep, which runs forward.
            # grad on means the recompute sweep, which the engine walks in
            # reverse, so the next layer wanted is the one below this.
            nxt = i + self.depth if not torch.is_grad_enabled() else i - self.depth
            if 0 <= nxt < len(self.blocks):
                self._fetch(self.blocks[nxt])
            return None
        return hook

    def _post(self, i):
        def hook(module, args, output):
            # Under no_grad nothing reads this weight again until recompute, so
            # drop it now. With grad on, the graph this pass just built still
            # has to be walked, so eviction waits for the backward hook.
            if not torch.is_grad_enabled():
                self._release(self.blocks[i])
            return output
        return hook

    def _bwd(self, i):
        def hook(module, grad_input, grad_output):
            self._release(self.blocks[i])
            # Block 0 is the last swapped block the backward reaches, so its
            # hook marks the end of this step's use of the pool. Arming the next
            # forward's leading fetches here hides them behind the remaining
            # backward and the optimizer step, and it means no caller has to
            # know a step boundary exists. Every micro-batch is a full
            # forward + backward, so gradient accumulation needs nothing extra.
            if i == 0:
                self._arm(forward = True)
        return hook

    def _state_dict(self, i):
        def hook(module, state_dict, prefix, local_metadata):
            b = self.blocks[i]
            for name, host in zip(b.names, b.host):
                key = prefix + name
                if key in state_dict:
                    state_dict[key] = host
        return hook

    def reset(self):
        """Manual step-boundary reset. Not needed in training; the backward
        hook arms the next step itself. Kept for callers that run forward-only
        loops and want the leading prefetches back in flight."""
        for b in self.blocks:
            self._release(b)
        self._arm(forward = True)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        # Free the pool before restoring: the full restore needs real
        # allocations, not pool slots, and holding the unused free slots while
        # allocating them can OOM on a card sized for the swapped model (and
        # would fire again inside the constructor's rollback, which calls here).
        self.free = {}
        for b in self.blocks:
            if not b.resident:
                for p, h, d in zip(b.params, b.host, b.devices):
                    p.data = h.to(d, copy = True)
                b.resident = True
        # Sync every device we prefetched on, not just the current one: a sharded
        # model can have an in-flight copy on another card, and the pre-hook that
        # would have waited on its event is already removed above.
        for dev in self.streams:
            torch.cuda.synchronize(dev)

    def host_bytes(self):
        return sum(b.nbytes() for b in self.blocks)

    def pool_bytes(self):
        """Device bytes the pool owns, free slots and held ones alike."""
        held = [b.slot for b in self.blocks if b.slot is not None]
        free = [slot for slots in self.free.values() for slot in slots]
        return sum(t.numel() * t.element_size() for slot in held + free for t in slot)

    def signatures(self):
        return len(self.free)

    def resident_count(self):
        return sum(b.resident for b in self.blocks)


def find_decoder_layers(model):
    """The decoder layer ModuleList, however the wrapper buries it."""
    m = model
    for _ in range(6):
        for attr in ("model", "base_model", "transformer", "language_model"):
            inner = getattr(m, attr, None)
            if inner is not None and hasattr(inner, "layers"):
                return inner.layers
            if inner is not None:
                m = inner
                break
        else:
            break
    if hasattr(m, "layers"):
        return m.layers
    raise RuntimeError("could not locate decoder layers")
