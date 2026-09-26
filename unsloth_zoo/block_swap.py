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
import torch
from contextlib import contextmanager, nullcontext

__all__ = [
    "BlockSwap",
    "find_decoder_layers",
    "build_host_layers",
]


@contextmanager
def _no_inference_mode():
    # Inference tensors would make later copy_ and autograd saves raise.
    try:
        leave_inference = torch.inference_mode(False)
    except (TypeError, AttributeError):
        leave_inference = nullcontext()  # older torch lacks inference_mode(bool)
    with leave_inference, torch.no_grad():
        yield

# WSL2 caps pinned memory: fall back to pageable. Local since this module loads standalone.
_PINNED_MEMORY_AVAILABLE = True


def _to_pinned_host(t):
    global _PINNED_MEMORY_AVAILABLE
    host = t.to("cpu", copy = True)
    if not _PINNED_MEMORY_AVAILABLE or os.environ.get("UNSLOTH_DISABLE_PINNED_MEMORY", "0") == "1":
        return host
    try:
        return host.pin_memory()
    except RuntimeError as e:
        # Some builds report only "cudaErrorMemoryAllocation".
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
    __slots__ = ("params", "host", "devices", "index", "streams", "events", "resident", "slot", "sig")

    def __init__(self, layer, streams, device):
        self.params, self.host, self.devices = [], [], []
        index = {}
        for name, p in _swappable(layer):
            index[id(p)] = len(self.params)
            self.params.append(p)
            if p.data.device.type == "cpu":
                self.host.append(p.data if p.data.is_pinned() else _to_pinned_host(p.data))
                self.devices.append(device)
            else:
                self.host.append(_to_pinned_host(p.data))
                self.devices.append(p.data.device)
        self.index = index
        # One stream + event per device: sharded blocks must sync every card.
        for d in self.devices:
            if d not in streams:
                streams[d] = torch.cuda.Stream(device = d)
        self.streams = streams
        self.events = {d: torch.cuda.Event() for d in self.devices}
        self.resident = True
        self.slot = None
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
        if self.resident:
            return
        for d, event in self.events.items():
            stream = self.streams[d]
            stream.wait_stream(torch.cuda.current_stream(d))
            with torch.cuda.stream(stream):
                for dst, h, pd in zip(slot, self.host, self.devices):
                    if pd == d:
                        dst.copy_(h, non_blocking = True)
                event.record(stream)
        for p, dst in zip(self.params, slot):
            p.data = dst
        self.slot = slot
        self.resident = True

    def wait(self):
        for d, event in self.events.items():
            torch.cuda.current_stream(d).wait_event(event)

    def nbytes(self):
        return sum(h.numel() * h.element_size() for h in self.host)


class BlockSwap:
    """Install on a decoder-layer list; the tail `n` of them live on the host."""

    def __init__(self, layers, n, prefetch_depth = 2, device = None):
        self.blocks, self.handles = [], []
        self.depth = max(1, prefetch_depth)
        self.streams = {}
        self.free = {}
        self.start = len(layers)
        if n <= 0:
            return
        n = min(n, len(layers))
        self.start = len(layers) - n
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = torch.device(device)
        self.blocks = [_Block(l, self.streams, self.device) for l in layers[self.start:]]
        for layer in layers[self.start:]:
            for name, p in layer.named_parameters():
                if (p.requires_grad or "lora_" in name) and p.device.type == "cpu":
                    p.data = p.data.to(self.device)

        # A param shared with any other layer (swapped or not) gets evicted under it.
        unswapped = set()
        for layer in layers[:self.start]:
            for _, p in _swappable(layer):
                unswapped.add(id(p))
        seen = set()
        for b in self.blocks:
            for p in b.params:
                if id(p) in seen or id(p) in unswapped:
                    raise ValueError(
                        "block_swap: a Parameter is shared with another decoder "
                        "layer; exclude the shared layer or reduce the swap depth.")
                seen.add(id(p))

        # Evict before building the pool (originals + pool would OOM); roll back on any failure.
        try:
            # register_full_backward_hook can raise on legacy hooks: keep inside rollback.
            for i, layer in enumerate(layers[self.start:]):
                self.handles.append(layer.register_forward_pre_hook(self._pre(i)))
                self.handles.append(layer.register_forward_hook(self._post(i)))
                self.handles.append(layer.register_full_backward_hook(self._bwd(i)))
                # Evicted weights are empty: state_dict must read the host copy.
                self.handles.append(layer._register_state_dict_hook(self._state_dict(i)))

            for b in self.blocks:
                self._release(b)

            sigs = [b.sig for b in self.blocks]
            with _no_inference_mode():
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

    def _acquire(self, block, steal):
        free = self.free[block.sig]
        if not free and steal:
            # Only after an interrupted step (backward died, blocks left resident).
            # Prefetches never steal: they could evict the block about to run.
            for b in self.blocks:
                if b is not block and b.sig == block.sig and b.resident and b.slot is not None:
                    self._release(b)
                    break
        return free.pop() if free else None

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
        def hook(module, args):
            b = self.blocks[i]
            self._fetch(b, steal = True)
            b.wait()
            # grad on = recompute sweep, which walks layers in reverse.
            nxt = i + self.depth if not torch.is_grad_enabled() else i - self.depth
            if 0 <= nxt < len(self.blocks):
                self._fetch(self.blocks[nxt])
            return None
        return hook

    def _post(self, i):
        def hook(module, args, output):
            # With grad on, backward still needs the weight; the backward hook evicts.
            if not torch.is_grad_enabled():
                self._release(self.blocks[i])
            return output
        return hook

    def _bwd(self, i):
        def hook(module, grad_input, grad_output):
            self._release(self.blocks[i])
            # Block 0 ends the step: arm next forward's fetches behind the optimizer.
            if i == 0:
                self._arm(forward = True)
        return hook

    def _state_dict(self, i):
        def hook(module, state_dict, prefix, local_metadata):
            # Resolved per call: PEFT wrapping after install renames weights (q_proj.base_layer.weight),
            # and state_dict() emits every alias; a missed one serializes an empty tensor.
            b = self.blocks[i]
            for name, p in module.named_parameters(remove_duplicate = False):
                idx = b.index.get(id(p))
                if idx is not None and prefix + name in state_dict:
                    state_dict[prefix + name] = b.host[idx]
        return hook

    def enter(self, idx):
        """Make layer `idx` resident for code that bypasses the forward hooks (fast decode)."""
        i = idx - self.start
        if 0 <= i < len(self.blocks):
            self._pre(i)(None, None)

    def leave(self, idx):
        i = idx - self.start
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
        for dev in self.streams:
            torch.cuda.synchronize(dev)

    def host_bytes(self):
        return sum(b.nbytes() for b in self.blocks)

    def pool_bytes(self):
        held = [b.slot for b in self.blocks if b.slot is not None]
        free = [slot for slots in self.free.values() for slot in slots]
        return sum(t.numel() * t.element_size() for slot in held + free for t in slot)

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
            # base_model may resolve to self.
            if inner is not None and inner is not m:
                m = inner
                break
        else:
            break
    if hasattr(m, "layers"):
        return m.layers
    raise RuntimeError("could not locate decoder layers")


def build_host_layers(make_layer, first_idx, count, tensors, device, compute_dtype,
                      quantize_4bit = False, skip_modules = (), prefix = "model.layers."):
    """Build layers [first_idx, first_idx + count) with frozen weights in pinned host RAM.
    `tensors` maps checkpoint keys to zero-arg loaders; quant_state stays on `device`."""
    import bitsandbytes as bnb
    device = torch.device(device)
    # Prequantized checkpoints keep unquantized (dynamic) weights plain.
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
                bias = torch.nn.Parameter(_to_pinned_host(tensors[base + mname + ".bias"]().to(compute_dtype)),
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
                # Prequantized quant_state carries the checkpoint dtype (bf16); fp16 GPUs need the compute dtype,
                # which patch_model_and_tokenizer already applied to the layers loaded before these.
                w.quant_state.dtype = compute_dtype
                new.weight = w
                new.quant_state = w.quant_state
                w.data = _to_pinned_host(w.data)
                torch.cuda.empty_cache()
            else:
                new = torch.nn.Linear(mod.in_features, mod.out_features, bias = bias is not None, device = "meta")
                new.weight = torch.nn.Parameter(_to_pinned_host(tensors[key]().to(compute_dtype)),
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
            setattr(parent, child, torch.nn.Parameter(_to_pinned_host(t), requires_grad = False))
        for bname, b in layer.named_buffers():
            if b.device.type == "meta":
                raise RuntimeError(f"block_swap: {base}{bname} is a buffer with no checkpoint value")
        out.append(layer)
    return out
