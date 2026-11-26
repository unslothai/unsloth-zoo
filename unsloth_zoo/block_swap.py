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
Slots come from a depth + 1 pool, not per-fetch allocation; prefetch_depth = "auto" grows or shrinks it.
"""
import os
import sys
import torch
from contextlib import contextmanager, nullcontext
import functools
import gc
import importlib
import importlib.util
import itertools
import tempfile
import stat
import time
import weakref

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
# Copy, stall and compute timings for stats(); UNSLOTH_OFFLOAD_STATS=0 skips the timing events.
_TIMING = os.environ.get("UNSLOTH_OFFLOAD_STATS", "1") != "0"
# Backward leaves the first prefetch_depth swapped blocks on the card for the next forward (0 = release them).
_KEEP_BOTTOM = os.environ.get("UNSLOTH_OFFLOAD_KEEP_FIRST", "1") != "0"
# Lossless bf16 compression of the host copies (fewer PCIe bytes per fetch, decoded on the card); opt in with 1.
_COMPRESS_ENV = "UNSLOTH_BLOCK_SWAP_COMPRESS"
# prefetch_depth = "auto": deepest pool it grows to, and the steps it watches before settling.
_AUTO_MAX_DEPTH = 4
_AUTO_STEPS = 12
# Steps judged per depth: a single step's copy timings are too noisy to decide on.
_AUTO_SAMPLE = 2
_AUTO_STALL = 0.02
# Auto pool growth keeps this much of each card free for temporaries made inside a layer or the loss.
_WORKSPACE_MIN_BYTES = 512 << 20
_WORKSPACE_FRACTION = 0.05


def _compress_requested():
    return os.environ.get(_COMPRESS_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


# Disk-tier blocks are stored with the lossless bf16 codec unless this is 0: the disk is ~17x slower than the bus,
# so a quarter fewer bytes read is a quarter off a disk-bound step, and the card decodes far faster than either.
_DISK_COMPRESS_ENV = "UNSLOTH_OFFLOAD_DISK_COMPRESS"


def _disk_compress_requested():
    return os.environ.get(_DISK_COMPRESS_ENV, "1").strip().lower() not in ("0", "false", "no", "off")


_CODEC = []


def _codec():
    """unsloth_zoo.block_swap_codec (a copy already in sys.modules first: tests load both files standalone)."""
    if not _CODEC:
        mod = sys.modules.get("unsloth_zoo.block_swap_codec")
        if mod is None and __package__ == "unsloth_zoo":
            try:
                from unsloth_zoo import block_swap_codec as mod
            except Exception:
                mod = None
        if mod is None:
            return None
        _CODEC.append(mod)
    return _CODEC[0]


_DISK = []


class _NoDiskTier:
    """Stands in when block_swap_disk cannot be imported: no disk tier, the host RAM refusal stays."""

    @staticmethod
    def disk_tier_reason():
        return "unsloth_zoo.block_swap_disk unavailable"

    @staticmethod
    def host_limit_bytes():
        return None

    @staticmethod
    def disk_ref(p):
        return None


def _disk():
    """unsloth_zoo.block_swap_disk (a copy already in sys.modules first: tests load both files standalone)."""
    if not _DISK:
        mod = sys.modules.get("unsloth_zoo.block_swap_disk")
        if mod is None and __package__ == "unsloth_zoo":
            try:
                from unsloth_zoo import block_swap_disk as mod
            except Exception:
                mod = None
        _DISK.append(mod if mod is not None else _NoDiskTier)
    return _DISK[0]


def _has_disk_ref(p):
    """True for a weight spilled to disk at load whose placeholder BlockSwap has not taken over yet."""
    return getattr(p, "_unsloth_disk", None) is not None and _disk().disk_ref(p) is not None


def _host_copy(t):
    return t.detach().to("cpu", copy = True).contiguous()


def _pow2_ceil(n):
    return 1 << max(0, (int(n) - 1).bit_length())


def _host_reserve(total):
    # max(4 GiB, 15%), but never more than a quarter of a small host or container (a 2 GiB one keeps 0.5 GiB).
    return max(min(_PIN_RESERVE_MIN_BYTES, total // 4), int(total * _PIN_RESERVE_FRACTION))


def _host_free():
    """(total, free after the reserve) host RAM. `_host_memory` already caps psutil by an enforcing cgroup limit
    (psutil reports the whole host inside a container, while the cgroup is what gets the job OOM-killed)."""
    total, available = _host_memory()
    return total, available - _host_reserve(total)


def _local_world_size():
    try:
        return max(1, int(os.environ.get("LOCAL_WORLD_SIZE", "1")))
    except ValueError:
        return 1


def _host_memory():
    """``memory_budget.host_memory_bytes``; plain psutil when this file is loaded on its own."""
    try:
        from unsloth_zoo.memory_budget import host_memory_bytes
    except Exception:
        import psutil
        vm = psutil.virtual_memory()
        return vm.total, vm.available
    return host_memory_bytes()


def _pin_budget():
    # Each local rank sees the same free RAM, so it may only claim its share.
    try:
        return _host_free()[1] // _local_world_size()
    except Exception:
        return 0


# Re-reads of free RAM and the peers' reservations before refusing, while a peer may be mid-copy.
_PEER_RETRIES = 8
_PEER_RETRY_S = 0.25
# Host RAM sessions (a load into host RAM, a BlockSwap install): local ranks run the same sequence of them.
_HOST_SESSION = 0


def _rank_ledger():
    """(directory shared by this launch's local ranks, local rank, local world size), or None outside a multi-rank
    launch. Each rank publishes there the host bytes it still has to allocate in its current session."""
    world = _local_world_size()
    rank = os.environ.get("LOCAL_RANK")
    job = os.environ.get("TORCHELASTIC_RUN_ID") or os.environ.get("MASTER_PORT")
    if world <= 1 or rank is None or not job:
        return None
    try:
        uid = os.getuid() if hasattr(os, "getuid") else 0
        # The launcher is every local rank's parent: a later launch on the same port gets a fresh directory.
        path = os.path.join(tempfile.gettempdir(), f"unsloth_block_swap_{uid}_{job}_{os.getppid()}")
        os.makedirs(path, mode = 0o700, exist_ok = True)
        if not _private_dir(path):
            # Predictable name in a shared temp dir: another OS user may have made it.
            return None
        return path, int(rank), world
    except (OSError, ValueError):
        return None


def _private_dir(path):
    """True if `path` is a real directory owned by this user that no one else can write to."""
    if os.name == "nt" or not hasattr(os, "getuid"):
        return True
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISDIR(st.st_mode) and st.st_uid == os.getuid() and not (st.st_mode & 0o022)


def _publish_host_pending(nbytes):
    ledger = _rank_ledger()
    if ledger is None:
        return
    path, rank, _ = ledger
    try:
        tmp = os.path.join(path, f"{rank}.{os.getpid()}.tmp")
        with open(tmp, "w") as f:
            f.write(f"{_HOST_SESSION} {max(0, int(nbytes))}")
        os.replace(tmp, os.path.join(path, str(rank)))
    except OSError:
        pass


def _others_host_pending(own):
    """Host bytes the other local ranks still have to allocate, or None outside a multi-rank launch. A rank that
    has not reached this session yet is assumed to need `own` (every rank loads the same model); one that is done
    with it published 0, since what it allocated is already gone from available RAM."""
    ledger = _rank_ledger()
    if ledger is None:
        return None
    path, rank, world = ledger
    total = 0
    for r in range(world):
        if r == rank:
            continue
        try:
            with open(os.path.join(path, str(r))) as f:
                session, nbytes = (int(x) for x in f.read().split())
            nbytes = max(0, nbytes)
        except (OSError, ValueError):
            session, nbytes = -1, 0
        # A peer still in an earlier session (loading while this rank already installs) keeps what it published:
        # its remaining load is still to come, and it will need about as much as this rank for this session too.
        total += max(own, nbytes) if session < _HOST_SESSION else nbytes
    return total


def _begin_host_session():
    global _HOST_SESSION
    _HOST_SESSION += 1


def _end_host_session():
    # Everything this session allocated is in host RAM now: available RAM already shows it.
    _publish_host_pending(0)


def _check_host_fit(sizes, on_host = 0, slack = 0):
    """Refuse an offload whose layers (`sizes`, bytes each) would not fit in free host RAM: running it out
    freezes the machine. `on_host` bytes of them already sit in host RAM and are counted as available;
    `slack` is allocator rounding on top of the packed bytes."""
    try:
        total, free = _host_free()
    except Exception:
        return
    need = sum(sizes) + slack
    own = max(0, need - on_host)
    others = _others_host_pending(own)
    if others is None:
        # No shared ledger: every local rank reads the same free RAM, so each may only plan for its share.
        avail = free // _local_world_size() + on_host
    else:
        # Ranks that finished already took their bytes out of `free`; only the others' remaining bytes are held back.
        avail = free - others + on_host
    limit = _disk().host_limit_bytes()
    if limit is not None:
        avail = min(avail, limit)
    # A peer that is copying a layer has already taken those bytes out of `free` but lowers its reservation
    # only once the layer is in: give in-flight peers a moment before refusing.
    retries = _PEER_RETRIES if others else 0
    while need > avail and retries > 0:
        retries -= 1
        time.sleep(_PEER_RETRY_S)
        try:
            total, free = _host_free()
        except Exception:
            return None
        others = _others_host_pending(own)
        avail = free - (others or 0) + on_host
        if limit is not None:
            avail = min(avail, limit)
    if need <= avail:
        _publish_host_pending(own)
        return own
    fit, used = 0, slack
    # Smallest first: the most layers that could fit.
    for size in sorted(sizes):
        if used + size > avail:
            break
        used, fit = used + size, fit + 1
    raise ValueError(
        f"Unsloth: offload_layers asked for {len(sizes)} layers in host RAM, which needs "
        f"{need / 2**30:.1f} GiB, but only {max(avail, 0) / 2**30:.1f} GiB is available after keeping "
        f"{_host_reserve(total) / 2**30:.1f} GiB free for the system. At most {fit} layers would fit; "
        "lower offload_layers or free host RAM."
    )


def _lazy_host_empty(nbytes):
    """Pageable host bytes whose pages are committed only as packing writes them, which the host guard credits.
    torch.use_deterministic_algorithms(True) would otherwise fill (commit) the whole arena up front; every byte is
    overwritten by packing, so skipping that fill never exposes uninitialized memory."""
    det = getattr(getattr(torch.utils, "deterministic", None), "fill_uninitialized_memory", None)
    if det is None or not det:
        return torch.empty(nbytes, dtype = torch.uint8)
    torch.utils.deterministic.fill_uninitialized_memory = False
    try:
        return torch.empty(nbytes, dtype = torch.uint8)
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = det


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


def _can_register():
    return (_PINNED_MEMORY_AVAILABLE and os.environ.get("UNSLOTH_DISABLE_PINNED_MEMORY", "0") != "1"
            and torch.cuda.is_available() and hasattr(torch.cuda.cudart(), "cudaHostRegister"))


def _chunked_commits(block_sizes):
    """Host bytes _pack's chunked fallback has allocated once each block (of {device: bytes} sizes) is packed,
    padding included (same walk)."""
    left = sum(sum(s.values()) for s in block_sizes)
    chunk = max(_CHUNK_BYTES, _pow2_ceil(max((max(s.values(), default = 0) for s in block_sizes), default = 0)))
    total, cap, used, out = 0, None, 0, []
    for sizes in block_sizes:
        for need in sizes.values():
            if cap is None or used + need > cap:
                cap = chunk if left >= chunk else _pow2_ceil(left)
                total += cap
                used = 0
            used += need
            left -= need
        out.append(total)
    return out


# glibc serves allocations below its mmap threshold (dynamic, up to 32 MiB on 64-bit) from the heap, and free()
# keeps those pages resident until malloc_trim. Larger ones are mmapped and go back to the OS on free.
_HEAP_MAX_BYTES = 32 * 2**20
_MALLOC_TRIM = []


def _malloc_trim_fn():
    if not _MALLOC_TRIM:
        fn = None
        if sys.platform.startswith("linux"):
            try:
                import ctypes
                fn = ctypes.CDLL(None).malloc_trim
                fn.argtypes, fn.restype = [ctypes.c_size_t], ctypes.c_int
            except (OSError, AttributeError):
                fn = None
        _MALLOC_TRIM.append(fn)
    return _MALLOC_TRIM[0]


def _heap_trimmable():
    return _malloc_trim_fn() is not None


def _trim_host_heap():
    """Return free heap pages to the OS (glibc malloc_trim, which also madvises free pages inside the heap).
    True when it ran; False where there is no malloc_trim (not glibc)."""
    fn = _malloc_trim_fn()
    if fn is None:
        return False
    fn(0)
    return True


def _creditable_host_storages(swapped, layers, owner):
    """{storage ptr: bytes} of the CPU storages packing `swapped` frees, and {ptr: user set}. Pointers and sizes
    only: a storage reference held past this call would keep the storage alive while packing assumes it is gone."""
    host_storages, users = {}, {}
    # Heap-sized storages give their pages back only through malloc_trim, which packing runs after each credited
    # block; without it (not glibc) only storages large enough to be mmapped are credited.
    trimmable = _heap_trimmable()
    for li, layer in enumerate(swapped):
        for _, p in _swappable(layer):
            if p.device.type == "cpu":
                # Pinned by torch's caching host allocator: freeing it parks the block in that cache, still pinned,
                # and nothing returns it to the OS while packing. Never credited.
                try:
                    if p.data.is_pinned():
                        continue
                except RuntimeError:
                    pass
                st = p.data.untyped_storage()
                users.setdefault(st.data_ptr(), set()).add(li)
                if trimmable or st.nbytes() >= _HEAP_MAX_BYTES:
                    host_storages[st.data_ptr()] = st.nbytes()
    swapped_ids = {id(p) for layer in swapped for _, p in _swappable(layer)}
    # Buffers too: a registered buffer viewing a weight's storage keeps it alive after the weight is rebound.
    if owner is not None:
        everything = itertools.chain(owner.parameters(), owner.buffers())
    else:
        everything = (q for layer in layers for q in itertools.chain(layer.parameters(), layer.buffers()))
    for q in everything:
        if q.device.type == "cpu" and id(q) not in swapped_ids:
            ptr = q.data.untyped_storage().data_ptr()
            if ptr in users:
                users[ptr].add(-1)
    return {ptr: n for ptr, n in host_storages.items() if len(users[ptr]) == 1}, users


class _HostWalk:
    """Host RAM over install's packing walk. Block k's bytes are committed (`commits`, cumulative) before its
    original's storage (`credits[k]`) is released, so the peak is the largest committed-through-k less
    released-before-k, plus `margin` for release lag, and never above committing everything."""

    def __init__(self, commits, credits, margin):
        self.commits = list(commits)
        self.released = [0] + list(itertools.accumulate(credits))[:len(self.commits)]
        self.steps = [c - self.released[k] for k, c in enumerate(self.commits)]
        self.margin = margin

    def pending(self, k):
        """Bytes still to come once blocks 0..k are packed and released (k = -1: before the first)."""
        rest = self.steps[k + 1:]
        if not rest:
            return 0
        now = self.commits[k] - self.released[k + 1] if k >= 0 else 0
        return max(0, min(self.commits[-1], max(rest) + self.margin) - now)


def _chunked_bytes(block_sizes):
    """Host bytes _pack's chunked fallback allocates for blocks with these {device: bytes} sizes."""
    return (_chunked_commits(block_sizes) or [0])[-1]


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


def _checkpoints_in_forward(module):
    """True when this call checkpoints inside `forward` (transformers >= 5.23 or Unsloth's block compile wrapper)."""
    if not (getattr(module, "gradient_checkpointing", False) and module.training):
        return False
    fwd = module.__dict__.get("forward")
    fwd = getattr(fwd, "__func__", fwd) if fwd is not None else type(module).forward
    return getattr(getattr(fwd, "__code__", None), "co_name", None) == "forward_with_checkpointing"


def _graph_task_id():
    """Id of the backward running on this thread, -1 outside one (or on a torch without the query)."""
    get = getattr(torch._C, "_current_graph_task_id", None)
    return get() if get is not None else -1


def _swappable(module):
    kept = {id(p) for m in module.modules() if getattr(m, "_unsloth_host_kept", False) for p in m.parameters()}
    out = []
    for name, p in module.named_parameters(recurse = True):
        if p.requires_grad or "lora_" in name or id(p) in kept:
            continue
        out.append((name, p))
    return out


class _Block:
    __slots__ = ("params", "host", "devices", "index", "streams", "events", "resident", "slot", "sig",
                 "pending", "layout", "sizes", "src", "empties", "host_buf", "home",
                 "t_copy", "copy_open", "copy_pending", "marks", "mark_open", "codec", "decoders",
                 "disk", "disk_store", "disk_runs", "disk_state", "disk_ready", "disk_futs", "stage", "tier")

    def __init__(self, layer, streams, device, shared = ()):
        self.params, self.host, self.devices = [], [], []
        index = {}
        for name, p in _swappable(layer):
            if id(p) in shared:
                continue
            index[id(p)] = len(self.params)
            self.params.append(p)
            self.devices.append(device if p.data.device.type in ("cpu", "meta") else p.data.device)
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
        self.events = {d: torch.cuda.Event(enable_timing = _TIMING) for d in self.sizes}
        self.t_copy = {d: torch.cuda.Event(enable_timing = True) for d in self.sizes} if _TIMING else None
        self.copy_open = False
        self.copy_pending = []
        # Compute-stream marks: before the wait, after it, after the layer ran.
        self.marks = tuple(torch.cuda.Event(enable_timing = True) for _ in range(3)) if _TIMING else None
        self.mark_open = 0
        self.empties = [torch.empty(0, device = d, dtype = p.dtype) for p, d in zip(self.params, self.devices)]
        self.resident = True
        self.slot = None
        # Ran with grad on and autograd holds its weights until the backward hook fires.
        self.pending = False
        self.src, self.host_buf = {}, {}
        # {device: RegionCodec} for regions stored compressed; their host copies decode on demand (host_views).
        self.codec, self.decoders = {}, None
        self.home = self.devices[0] if self.devices else None
        self.sig = tuple((tuple(p.shape), p.dtype, d, off) for p, d, off in zip(self.params, self.devices, self.layout))
        self.disk, self.disk_store, self.disk_runs, self.disk_state = None, None, (), 0
        self.disk_ready, self.disk_futs, self.stage, self.tier = {}, None, None, None
        refs = [_disk().disk_ref(p) if getattr(p, "_unsloth_disk", None) is not None else None for p in self.params]
        if any(r is not None for r in refs):
            if not all(r is not None for r in refs):
                raise RuntimeError("Unsloth: offload_layers found a layer only partly spilled to disk.")
            self.set_disk(refs)

    def set_disk(self, refs):
        """Serve this block from the spill file (rewritten once if its layout differs from the slot's)."""
        mod = _disk()
        store = refs[0].store
        if len(self.sizes) != 1:
            raise RuntimeError("Unsloth: offload_layers serves a layer from disk only when it fetches onto one card.")
        bases = {r.offset - off for r, off in zip(refs, self.layout)}
        if any(r.store is not store for r in refs) or len(bases) != 1 or next(iter(bases)) % mod.ALIGN:
            hold = []
            pieces = []
            for p, r, off in zip(self.params, refs, self.layout):
                buf, owner = mod.read_bytes(r.store, r.offset, r.nbytes)
                hold.append(owner)
                pieces.append((buf.view(r.dtype).view(r.shape) if r.nbytes else buf, off))
            base = store.write_region(pieces, self.nbytes())
            refs = [mod.DiskRef(store, base + off, r.nbytes, r.shape, r.dtype) for r, off in zip(refs, self.layout)]
            del hold, pieces
        else:
            base = next(iter(bases))
        self.disk = {"refs": refs, "base": base}
        self.disk_store = store
        self.disk_runs = ((base, mod._round_up(self.nbytes()), 0),)

    def nbytes(self):
        return sum(self.sizes.values())

    def encode_disk(self, encode_device):
        """Rewrite a raw spilled region (spilled at load) encoded, in place: the encoded region is never longer."""
        if not self.disk_compressible() or self.disk is None or self.compressed():
            return False
        raw, owner = _disk().read_bytes(self.disk_store, self.disk["base"], self.nbytes())
        try:
            return self._store_encoded(raw, encode_device, self.disk_store, self.disk["base"])
        finally:
            del raw, owner

    def _store_encoded(self, raw, encode_device, store, at):
        """Encode the raw region and write it (appended, or over the raw region at `at`). False = keep raw."""
        codec = _codec()
        d = next(iter(self.sizes))
        # A fresh spill encodes over `raw` itself (each chunk is read before its shorter encoding is written), so
        # host RAM holds one region; a rewrite keeps `raw` intact to put back if the write fails.
        out = raw if at is None else None
        try:
            with _no_inference_mode():
                enc = codec.encode_region(raw, device = encode_device, out = out)
        except torch.cuda.OutOfMemoryError:
            enc = None
        if enc is None:
            return False
        parts, c = enc
        pieces = [(t, off) for off, t in parts]
        if out is not None:
            pieces.insert(0, (out[:c.words * 3 // 2], 0))
        try:
            base = store.write_region(pieces, c.nbytes, at = at)
        except BaseException:
            if at is not None:
                # The raw bytes may be half overwritten: put them back before giving up.
                store.write_region([(raw, 0)], raw.numel(), at = at)
            raise
        mod = _disk()
        if at is None:
            refs = [mod.DiskRef(store, base + off, p.data.nbytes, p.data.shape, p.data.dtype)
                    for p, off in zip(self.params, self.layout)]
            self.disk = {"refs": refs, "base": base}
            self.disk_store = store
        self.codec[d] = c
        self.disk_runs = ((base, mod._round_up(c.nbytes), 0),)
        return True

    def pack(self, regions):
        """Copy the weights into host `regions` {device: uint8 view}; host[i] become views into them."""
        self.host_buf = regions
        with torch.no_grad():
            for p, d, off in zip(self.params, self.devices, self.layout):
                view = regions[d][off:off + p.data.nbytes].view(p.dtype).view(p.shape)
                view.copy_(p.data)
                self.host.append(view)

    def compressible(self):
        return self.disk is None and bool(self.params) and all(p.dtype == torch.bfloat16 for p in self.params)

    def spill(self, encode_device = None):
        """Write the weights to the spill file and serve the block from disk; encoded when `encode_device` is
        given and the codec takes the region."""
        mod = _disk()
        store = mod._shared_spill(self.nbytes())
        if encode_device is not None and self.disk_compressible():
            raw = torch.zeros(self.nbytes(), dtype = torch.uint8)
            with torch.no_grad():
                for p, off in zip(self.params, self.layout):
                    raw[off:off + p.data.nbytes].view(p.dtype).view(p.shape).copy_(p.data)
            if self._store_encoded(raw, encode_device, store, None):
                return
            del raw
        base = store.write_region([(p.data, off) for p, off in zip(self.params, self.layout)], self.nbytes())
        self.set_disk([mod.DiskRef(store, base + off, p.data.nbytes, p.data.shape, p.data.dtype)
                       for p, off in zip(self.params, self.layout)])

    def disk_compressible(self):
        return len(self.sizes) == 1 and bool(self.params) and all(p.dtype == torch.bfloat16 for p in self.params)

    def disk_views(self, target = "cpu"):
        """The block's weights read back from disk, on `target`."""
        mod = _disk()
        c = next((c for c in self.codec.values() if c is not None), None) if self.codec else None
        buf, owner = mod.read_bytes(self.disk_store, self.disk["base"], c.nbytes if c is not None else self.nbytes())
        if c is not None:
            buf = _codec().decode_region_cpu(c, buf)
        out = []
        for shape, dt, _, off in self.sig:
            n = torch.Size(shape).numel() * torch.empty(0, dtype = dt).element_size()
            v = buf[off:off + n].view(dt).view(shape)
            out.append(v.clone() if torch.device(target).type == "cpu" else v.to(target, copy = True))
        del owner
        return out

    def pack_compressed(self, arena, used, encode_device):
        """Like pack, but each device's region is stored encoded (when that saves bytes) at arena[used:].
        Returns the next free arena offset."""
        codec = _codec()
        self.host_buf = {}
        with torch.no_grad():
            for d, need in self.sizes.items():
                # Zeroed padding: alignment gaps then cost no escapes.
                raw = torch.zeros(need, dtype = torch.uint8)
                for p, pd, off in zip(self.params, self.devices, self.layout):
                    if pd == d:
                        raw[off:off + p.data.nbytes].view(p.dtype).view(p.shape).copy_(p.data)
                # Encoded chunks go straight into the arena (never larger than the raw region they replace).
                dst = arena[used:used + need]
                enc = None
                dev = torch.device(d) if torch.device(d).type == "cuda" else encode_device
                try:
                    enc = codec.encode_region(raw, device = dev, out = dst)
                except torch.cuda.OutOfMemoryError:
                    # No room on the card to encode: store this region raw (the bytes are already assembled).
                    enc = None
                if enc is None:
                    region = dst
                    region.copy_(raw)
                    self.codec[d] = None
                else:
                    parts, c = enc
                    region = arena[used:used + c.nbytes]
                    codec.write_parts(region, parts)
                    self.codec[d] = c
                del raw
                self.host_buf[d] = region
                used += -(-region.numel() // _ALIGN) * _ALIGN
            for p, d, off in zip(self.params, self.devices, self.layout):
                self.host.append(None if self.codec.get(d) is not None else
                                 self.host_buf[d][off:off + p.data.nbytes].view(p.dtype).view(p.shape))
        return used

    def compressed(self):
        return any(c is not None for c in self.codec.values())

    def wire_bytes(self):
        return sum(t.numel() for t in self.host_buf.values())

    def _region_on(self, d, target):
        """Region d's raw bytes on `target` (decoded on card d when it is compressed)."""
        with _no_inference_mode():
            return self._region_on_impl(d, target)

    def _region_on_impl(self, d, target):
        c = self.codec.get(d)
        if c is None:
            return self.host_buf[d].to(target, copy = True) if torch.device(target).type != "cpu" else self.host_buf[d]
        dec = (self.decoders or {}).get(d)
        to_cpu = torch.device(target).type == "cpu"
        if dec is not None:
            try:
                with _no_inference_mode():
                    out = torch.empty(c.raw_nbytes, dtype = torch.uint8, device = d)
                    stream = self.streams[d]
                    stream.wait_stream(torch.cuda.current_stream(d))
                    done = torch.cuda.Event()
                    dec.fetch(c, self.host_buf[d], out, stream, done)
                    torch.cuda.current_stream(d).wait_event(done)
                    return out if torch.device(target) == torch.device(d) else out.to(target)
            except torch.cuda.OutOfMemoryError:
                if not to_cpu:
                    raise
        out = _codec().decode_region_cpu(c, self.host_buf[d])
        return out if to_cpu else out.to(target)

    def host_views(self, target = "cpu"):
        """The block's weights from the host copy (decoded when compressed). Views of one buffer per device."""
        if self.disk is not None:
            return self.disk_views(target)
        if not self.compressed() and torch.device(target).type == "cpu":
            return self.host
        regions = {d: self._region_on(d, target) for d in self.host_buf}
        with _no_inference_mode():
            return [regions[d][off:off + torch.Size(shape).numel() * torch.empty(0, dtype = dt).element_size()]
                    .view(dt).view(shape) for shape, dt, d, off in self.sig]

    def evict(self):
        if not self.resident:
            return
        if self.disk_state:
            self.tier.cancel(self)
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
        if self.disk is not None:
            self._prefetch_disk(slot)
            return
        if self.copy_open:
            # The last copy is still in flight: keep its events for _harvest_copy, record this one on new ones.
            self.copy_pending.append((self.t_copy, self.events))
            self.t_copy = {d: torch.cuda.Event(enable_timing = True) for d in self.sizes}
            self.events = {d: torch.cuda.Event(enable_timing = True) for d in self.sizes}
        for d, event in self.events.items():
            stream = self.streams[d]
            stream.wait_stream(torch.cuda.current_stream(d))
            with torch.cuda.stream(stream):
                if self.t_copy is not None:
                    self.t_copy[d].record(stream)
                c = self.codec.get(d)
                if c is None:
                    dec = (self.decoders or {}).get(d)
                    if dec is not None:
                        # The slot may still be written by a compressed block's decode that was stolen before
                        # compute waited for it (an interrupted step): order this copy after it.
                        stream.wait_stream(dec.stream)
                    bufs[d].copy_(self.host_buf[d], non_blocking = True)
                    event.record(stream)
            if c is not None:
                # Compressed bytes over the bus, decoded into the slot on the card's decode stream.
                self.decoders[d].fetch(c, self.host_buf[d], bufs[d], stream, event)
        self.copy_open = self.t_copy is not None
        for p, v in zip(self.params, views):
            p.data = v
        self.slot = slot
        self.resident = True

    def _prefetch_disk(self, slot):
        # The card copy (enqueued once the read lands) waits on this event: the slot's previous user.
        if self.copy_open or getattr(self, "copy_pending", None):
            self.copy_pending.append((self.t_copy, self.events))
            self.t_copy = {d: torch.cuda.Event(enable_timing = True) for d in self.sizes} if _TIMING else None
            self.events = {d: torch.cuda.Event(enable_timing = _TIMING) for d in self.sizes}
            self.copy_open = False
        else:
            # Fresh events: the compute stream must never wait on one recorded for an earlier fetch.
            self.events = {d: torch.cuda.Event(enable_timing = _TIMING) for d in self.sizes}
        self.disk_ready = {}
        for d in self.sizes:
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream(d))
            self.disk_ready[d] = ev
        for p, v in zip(self.params, slot[1]):
            p.data = v
        self.slot = slot
        self.resident = True
        self.disk_state = 1
        self.tier.submit(self)

    def wait(self):
        if self.disk_state:
            # Waiting on a not yet recorded event is a no-op: enqueue the copy first.
            self.tier.complete(self)
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


def _output_logits(output):
    out = getattr(output, "logits", None)
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(output, (tuple, list)):
        # return_dict = False with labels puts the scalar loss first and the logits second.
        tensors = [t for t in output[:2] if isinstance(t, torch.Tensor)]
        return max(tensors, key = lambda t: t.numel(), default = None)
    return output


def swap_indices(total, n, placement = "spread"):
    """"spread": evenly spaced, ending at the last layer (each copy hides behind total / n layers); "tail": last n."""
    n = max(0, min(int(n), total))
    if placement == "tail":
        return list(range(total - n, total))
    return [(k + 1) * total // n - 1 for k in range(n)]


class BlockSwap:
    """Install on a layer list; layers at `n` (a count or explicit indices) live on the host.
    `prefetch_depth = "auto"` starts at 1 and deepens while copies stall a step that could hide them; `model`
    (found automatically for lists from `find_decoder_layers`) lets it see the step peak after the last layer."""

    # Defaults for instances built without __init__ (the scheduler tests).
    total = 0
    current = None
    _auto = False
    _settled = True
    _step_queued = False
    _steps = 0
    _last_stall = None
    _sampled = 0
    _tot = None
    _seen = None
    _adapt_seen = None
    _step_peak = None
    _last_peak = None
    _optimizer_reserve = {}
    _reserve_owner = None
    _reserve_layers = ()
    _hwm_fresh = False
    _hwm_base = {}
    _peak_devs = ()
    _to_spill = frozenset()
    _disk_tier = None
    _disk_seen = {}
    _disk_decoders = {}

    def __init__(self, layers, n, prefetch_depth = 2, device = None, placement = "tail", model = None):
        self.blocks, self.handles = [], []
        auto = prefetch_depth == "auto"
        # Auto tunes from the timing events; with UNSLOTH_OFFLOAD_STATS=0 it is the fixed default.
        self._auto = auto and _TIMING
        self._settled = not self._auto
        self._step_queued = False
        self.depth = _resolved_depth(prefetch_depth)
        self.total = len(layers)
        self._tot = dict.fromkeys(_STAT_KEYS, 0.0)
        self._seen = dict(self._tot)
        self._adapt_seen = dict(self._tot)
        self.streams = {}
        self.free = {}
        self._grew = False
        self._chunks = []
        self.pinned_bytes = 0
        self._decoders = {}
        self._compress = False
        self._compress_note = None
        if isinstance(n, int):
            self.indices = swap_indices(len(layers), n, placement)
        else:
            self.indices = sorted({int(i) % len(layers) for i in n})
        self.pos = {li: k for k, li in enumerate(self.indices)}
        self.start = self.indices[0] if self.indices else len(layers)
        self._input_hooked = [False] * len(self.indices)
        # Graph task a grad-on forward of each block ran in (-1: not inside a backward).
        self._forward_task = [-1] * len(self.indices)
        self._ckpt_wrapped = weakref.WeakSet()
        self.spans_devices, self.layer_device = False, None
        if not self.indices:
            # Still a host session (with nothing pending): other local ranks then read this rank as done with it,
            # and every rank's session count stays in step.
            _begin_host_session()
            _end_host_session()
            return
        swapped = [layers[i] for i in self.indices]
        # A param shared by two layers stays on the card: evicting it under one empties it for the other.
        owners = {}
        for li, layer in enumerate(layers):
            for _, p in _swappable(layer):
                owners.setdefault(id(p), set()).add(li)
        shared = {pid for pid, o in owners.items() if len(o) > 1}
        for li, layer in enumerate(layers):
            name = next((n for n, p in layer.named_parameters() if p.requires_grad and _has_disk_ref(p)), None)
            if name is not None:
                raise ValueError(
                    f"Unsloth: layer {li}'s weight {name} lives in offload_layers' disk tier, which holds frozen "
                    "weights only (LoRA / QLoRA). Full finetuning needs every trained weight in GPU or host RAM."
                )
        # Accelerate leaves layers that did not fit at load on `meta`, with the real weights behind its hooks.
        for li in self.indices:
            name = next((n for n, p in _swappable(layers[li])
                         if p.device.type == "meta" and id(p) not in shared and not _has_disk_ref(p)), None)
            if name is not None:
                raise ValueError(
                    f"Unsloth: offload_layers cannot swap layer {li}: its weight {name} is on the meta device, "
                    "so the model did not fit on the GPU at load and accelerate offloaded it to CPU or disk. "
                    "Pass offload_layers to from_pretrained instead, so those layers load straight to host memory."
                )
        # After the meta check, which must refuse before any CUDA call.
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            # One canonical key: params report cuda:N, so a bare "cuda" would split peaks and homes in two.
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.blocks = [_Block(layer, self.streams, self.device, shared) for layer in swapped]
        self._disk_tier, self._to_spill, self._disk_seen, self._disk_decoders = None, set(), {}, {}
        # Every card a block's frozen weights land on gets slots, not only its home card.
        self._peak_devs = sorted({torch.device(dv) for b in self.blocks for dv in (*b.sizes, b.home) if dv is not None
                                  and torch.device(dv).type == "cuda"}, key = lambda d: d.index or 0)
        _begin_host_session()
        # Weights already in host RAM are released as their block is packed. On Linux the registered arena is
        # pageable and fills block by block, so only a couple of blocks are ever held twice (Flash-Next: 79.5 GiB
        # of host layers packed with 34.5 GiB still free); Windows commits that arena up front, so no credit there.
        # The chunked fallback allocates one chunk at a time and releases each block's original right after
        # packing it, so on top of the originals it holds at most one chunk and the block being packed.
        # Per storage: a CPU tensor shared by several selected layers is released (and credited) once.
        # A storage is freed only once every tensor on it is gone: credit only one that lives in a single swapped
        # layer's swappable weights (a 10 GiB storage viewed by ten layers stays resident until the last is packed,
        # and one also viewed from outside the swapped weights is never freed by packing).
        # The whole model when it is known, passed or inferred from find_decoder_layers: an alias outside the
        # decoder list (a tied head, a model-level buffer) keeps the storage alive just the same.
        owner = model
        if owner is None:
            try:
                owner = _LAYER_OWNERS.get(layers)
            except TypeError:  # a plain list of layers
                owner = None
            owner = owner() if owner is not None else None
        host_storages, users = _creditable_host_storages(swapped, layers, owner)
        # A file-backed (mmap) storage is page cache that available RAM already counts: no extra headroom.
        file_backed = _file_backed_ranges()
        # Per block, the credited bytes its packing gives back: the ledger tracks what is still to come from that.
        self._host_credit = [0] * len(swapped)
        for ptr, n in host_storages.items():
            if not _in_ranges(ptr, file_backed):
                self._host_credit[next(iter(users[ptr]))] += n
        # Peak is the largest committed through block k minus released before k, not total less
        # credits.
        def host_walk():
            on_disk = [b.disk is not None or k in self._to_spill for k, b in enumerate(self.blocks)]
            sizes = [0 if d else b.nbytes() for b, d in zip(self.blocks, on_disk)]
            if not _can_register():
                # The chunked fallback commits whole chunks, rounded up to powers of two (padding included).
                commits = _chunked_commits([{} if d else b.sizes for b, d in zip(self.blocks, on_disk)])
            elif os.name == "nt":
                # Windows commits the whole registered arena up front.
                commits = [sum(sizes)] * len(sizes)
            else:
                # The Linux registered arena is pageable and fills block by block as it is written.
                commits = list(itertools.accumulate(sizes))
            # A block's worth of margin over the walk, for release lag; never above committing everything.
            walk = _HostWalk(commits, self._host_credit, max(sizes, default = 0))
            need = commits[-1] if commits else 0
            return sizes, walk, need, walk.pending(-1), need - sum(sizes)

        sizes, self._host_walk, need, peak, slack = host_walk()
        try:
            # Before the host check: compressing packs through one raw region of scratch at a time.
            self._compress = self._compress_check()
        except Exception:
            self._compress = False
        if not self._compress:
            self._decoders = {}
        self._encode_scratch = 0
        if self._compress:
            self._encode_scratch = max((b.nbytes() for b in self.blocks if b.compressible()), default = 0)
        staging = self._staging_bytes()
        try:
            try:
                try:
                    self._host_own = _check_host_fit(sizes + ([staging] if staging else []), on_host = need - peak,
                                                     slack = slack + self._encode_scratch)
                except ValueError:
                    if not self._compress:
                        raise
                    # Fits raw but not with the encoder's scratch: swap raw rather than refuse.
                    self._compress, self._decoders, self._encode_scratch = False, {}, 0
                    self._compress_note = "no host room for the encoder's scratch"
                    self._host_own = _check_host_fit(sizes + ([staging] if staging else []), on_host = need - peak,
                                                     slack = slack)
            except ValueError as e:
                if not self._plan_disk(e):
                    raise
                self._compress, self._decoders, self._encode_scratch = False, {}, 0
                sizes, self._host_walk, need, peak, slack = host_walk()
                self._host_own = _check_host_fit(sizes + [self._staging_bytes()], on_host = need - peak, slack = slack)
        except Exception:
            self._decoders = {}
            _end_host_session()
            raise
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
            if self._auto:
                self._reserve_owner = weakref.ref(owner) if owner is not None else None
                self._reserve_layers = layers
                self._optimizer_reserve = self._optimizer_bytes()
                # Allocator high-water mark at install (load and the pre-swap model): only a rise past it is training.
                self._hwm_base = {_cuda_index(dv): torch.cuda.max_memory_allocated(dv) for dv in self._peak_devs}
            if self._auto and owner is not None:
                # Logits and loss run after the last layer, so only the model's output sees the step's peak.
                self.handles.append(owner.register_forward_hook(_opaque(self._output_peak)))
            for i, layer in enumerate(swapped):
                self.handles.append(layer.register_forward_pre_hook(self._pre(i), with_kwargs = True))
                self.handles.append(layer.register_forward_hook(self._post(i)))
                # Evicted weights are empty: state_dict must read the host copy.
                self.handles.append(layer._register_state_dict_hook(self._state_dict(i)))
            if self._auto:
                # Unswapped layers below the first swapped one run their backward after its hook: sample them too.
                first = min(self.pos, default = 0)
                for li, layer in enumerate(layers[:first]):
                    if li not in self.pos:
                        self.handles.append(
                            layer.register_forward_pre_hook(_opaque(self._plain_peak), with_kwargs = True))

            if self._staging_bytes():
                self._disk_tier = _disk().DiskTier(self._staging_bytes() // 2, slots = 2)
                self._disk_decoders = self._disk_compress_check()
            self._pack()
            if self._disk_decoders and not any(b.compressed() and b.disk is not None for b in self.blocks):
                self._disk_decoders = {}
            if self._compress:
                self._build_decoders()
            # What sits in host RAM: compressed regions are smaller than host_bytes() (the raw size).
            total = sum(b.wire_bytes() for b in self.blocks) if self._compress else self.ram_bytes()
            if self._disk_tier is not None:
                self._report_disk()
            if self.pinned_bytes < total:
                # Pageable copies are several times slower than pinned ones and cannot hide behind compute.
                print(f"Unsloth: offload_layers pinned {self.pinned_bytes / 2**30:.1f} of {total / 2**30:.1f} GiB of host "
                      "memory; the rest is copied from pageable memory, several times slower. Free host RAM "
                      "or lower offload_layers for full speed.")

            sigs = [b.sig for b in self.blocks]
            for sig in set(sigs):
                self.free[sig] = [self._new_slot(sig) for _ in range(min(self.depth + 1, sigs.count(sig)))]

            self._arm(forward = True)
            if self._auto:
                # A fresh allocator peak window from here on: the load's own peak would hide every training peak
                # below it, and temporaries inside a layer (dequantized expert stacks) never reach a hook sample.
                # Before training starts, so a caller measuring the training peak from here reads the same number.
                try:
                    for dv in self._peak_devs:
                        torch.cuda.reset_peak_memory_stats(dv)
                    self._hwm_fresh = True
                except Exception:
                    self._hwm_fresh = False
            # Only once installed: a failed install must leave load-time spills tagged for a retry.
            for b in self.blocks:
                if b.disk is not None:
                    for p in b.params:
                        p.__dict__.pop("_unsloth_disk", None)
        except Exception:
            self.remove()
            raise
        finally:
            _end_host_session()

    def _on_disk(self, k):
        return getattr(self.blocks[k], "disk", None) is not None or k in self._to_spill

    def _staging_bytes(self):
        """Pinned staging the disk tier needs: two slots of its largest block (a read lands while one copies out)."""
        largest = max((b.nbytes() for k, b in enumerate(self.blocks) if self._on_disk(k)), default = 0)
        return 2 * _disk()._round_up(largest) if largest else 0

    def _plan_disk(self, error):
        """Host RAM refused: keep the blocks that fit (sweep order), mark the rest for disk. False = re-raise."""
        mod = _disk()
        reason = mod.disk_tier_reason()
        if reason is not None:
            return False
        movable = [k for k, b in enumerate(self.blocks) if b.disk is None and len(b.sizes) == 1 and b.params]
        if not movable:
            return False
        try:
            _, free = _host_free()
        except Exception:
            return False
        budget = free + sum(self._host_credit)
        limit = mod.host_limit_bytes()
        if limit is not None:
            budget = min(budget, limit)
        largest = max(b.nbytes() for k, b in enumerate(self.blocks) if k in movable or b.disk is not None)
        # Staging ring, plus a block of release lag.
        ring = 2 * mod._round_up(largest)
        if budget < ring:
            raise ValueError(
                f"Unsloth: offload_layers cannot fit even the disk tier's staging ring ({ring / 2**30:.2f} GiB, two "
                f"of the largest swapped layer) in host RAM ({max(budget, 0) / 2**30:.2f} GiB usable). {error}")
        budget -= ring + largest
        used, spill = 0, set()
        for k, b in enumerate(self.blocks):
            if b.disk is not None:
                continue
            if k in movable and used + b.nbytes() > budget:
                spill.add(k)
            else:
                used += b.nbytes()
        if not spill:
            return False
        spread = _disk_placement(movable, len(spill), self.depth)
        if spread is not None and sum(self.blocks[k].nbytes() for k in movable if k not in spread) \
                + sum(b.nbytes() for k, b in enumerate(self.blocks) if b.disk is None and k not in movable) <= budget:
            spill = spread
        mod._shared_spill(sum(self.blocks[k].nbytes() for k in spill))
        self._to_spill = spill
        return True

    def _report_disk(self):
        disk = [b for b in self.blocks if b.disk is not None]
        if not disk:
            return
        nbytes = sum(b.nbytes() for b in disk)
        stored = sum(n for b in disk for _, n, _ in b.disk_runs)
        store = disk[0].disk_store
        packed = (f", stored losslessly compressed as {stored / 2**30:.2f} GiB" if any(b.compressed() for b in disk)
                  else "")
        print(f"Unsloth: offload_layers serves {len(disk)} of {len(self.blocks)} swapped layers "
              f"({nbytes / 2**30:.2f} GiB{packed}) from disk at {store.directory}, since GPU and host RAM cannot hold "
              f"them ({self.ram_bytes() / 2**30:.2f} GiB stay in host RAM). Each forward and each backward reads them "
              f"again, about {2 * stored / 2**30:.2f} GiB from disk per micro-batch, so steps are disk bound: use a "
              f"larger batch with fewer gradient accumulation steps. {_disk()._ENV}=0 refuses instead.")

    def _compress_check(self):
        """Compression is opt in, needs the registered arena (pageable chunks keep raw copies) and a device
        decode that round trips exactly on every card a compressible block fetches onto."""
        if not _compress_requested() or not _can_register():
            return False
        codec = _codec()
        if codec is None:
            self._compress_note = "codec unavailable"
            return False
        devs = sorted({torch.device(d) for b in self.blocks if b.compressible() for d in b.sizes
                       if torch.device(d).type == "cuda"}, key = lambda d: d.index or 0)
        if not devs:
            return False
        largest = {}
        for b in self.blocks:
            if b.compressible():
                for d, n in b.sizes.items():
                    largest[torch.device(d)] = max(largest.get(torch.device(d), 0), n)
        with _no_inference_mode():
            for d in devs:
                ok, triton_on, why = codec.self_check(d)
                if ok and not triton_on:
                    # The torch decode is exact but slower than the bus (about 4 GB/s on an L4): no gain without Triton.
                    ok, why = False, "no Triton decode"
                if not ok:
                    self._compress_note = f"self-check failed on {d}: {why}"
                    print(f"Unsloth: offload_layers compression is off ({self._compress_note}); copying raw weights.")
                    return False
                self._compress_note = "triton"
            # Staging ring and aux buffers sized for the largest region, before any slot exists: if they and the
            # mandatory slot pool would not fit together (counting what packing frees on that card), stay raw.
            per_sig = {}
            for b in self.blocks:
                per_sig.setdefault(b.sig, [0, b.sizes])[0] += 1
            for d in devs:
                words = largest[d] // 2
                dec_bytes = (codec._RING * (min(codec.CHUNK, words) * 3 // 2)
                             + 2 * codec.max_aux_nbytes(largest[d]))
                slots = sum(min(self.depth + 1, n) * sizes.get(d, 0) for n, sizes in per_sig.values())
                reclaim = sum(p.data.nbytes for b in self.blocks for p in b.params
                              if p.data.device.type == "cuda" and torch.device(p.data.device) == d)
                try:
                    free = torch.cuda.mem_get_info(d)[0] + torch.cuda.memory_reserved(d) - torch.cuda.memory_allocated(d)
                except Exception:
                    free = None
                if free is not None and free + reclaim < slots + dec_bytes:
                    self._compress_note = "no room for the decode buffers next to the slot pool"
                    print(f"Unsloth: offload_layers compression is off ({self._compress_note}); copying raw weights.")
                    return False
            try:
                for d in devs:
                    words = largest[d] // 2
                    self._decoders[d] = codec.DeviceDecoder(d, min(codec.CHUNK, words) * 3 // 2,
                                                            codec.max_aux_nbytes(largest[d]), use_triton = True)
            except torch.cuda.OutOfMemoryError:
                self._decoders = {}
                self._compress_note = "no room for the decode buffers"
                print(f"Unsloth: offload_layers compression is off ({self._compress_note}); copying raw weights.")
                return False
        return True

    def _disk_compress_check(self):
        """{card: DeviceDecoder} for the disk tier's bf16 blocks, or {} (codec off, no Triton decode, no room)."""
        if not _disk_compress_requested():
            return {}
        codec = _codec()
        if codec is None:
            return {}
        largest = {}
        for k, b in enumerate(self.blocks):
            if self._on_disk(k) and b.disk_compressible():
                d = torch.device(next(iter(b.sizes)))
                if d.type == "cuda":
                    largest[d] = max(largest.get(d, 0), b.nbytes())
        out = {}
        try:
            with _no_inference_mode():
                for d, n in largest.items():
                    ok, triton_on, why = codec.self_check(d)
                    if not (ok and triton_on):
                        # The torch decode is exact but slower than a fast disk; keep those blocks raw.
                        print(f"Unsloth: offload_layers stores disk layers raw ({why or 'no Triton decode'}).")
                        return {}
                    words = n // 2
                    out[d] = codec.DeviceDecoder(d, min(codec.CHUNK, words) * 3 // 2, codec.max_aux_nbytes(n),
                                                 use_triton = True)
        except torch.cuda.OutOfMemoryError:
            return {}
        return out

    def _release_decoders(self):
        """Drop every decoder's GPU buffers (callers drained the decode streams first)."""
        self._decoders = {}
        for b in self.blocks:
            if getattr(b, "decoders", None) is not None:
                b.decoders = None

    def _build_decoders(self):
        used = {d for b in self.blocks for d, c in b.codec.items() if c is not None}
        self._decoders = {d: dec for d, dec in self._decoders.items() if d in used}
        if not self._decoders:
            self._compress = False
        for b in self.blocks:
            b.decoders = self._decoders
        raw = sum(b.nbytes() for b in self.blocks if b.compressed())
        if raw:
            stored = sum(b.wire_bytes() for b in self.blocks if b.compressed())
            n = sum(b.compressed() for b in self.blocks)
            print(f"Unsloth: offload_layers stores {n} of {len(self.blocks)} swapped layers compressed (lossless), "
                  f"{stored / 2**30:.2f} of {raw / 2**30:.2f} GiB ({100 * stored / raw:.0f}%), so fetches copy fewer bytes.")

    def _pack(self):
        # Block by block, releasing each original as soon as it is packed: never two host copies of the tail.
        if self._pack_registered():
            return
        left = sum(b.nbytes() for k, b in enumerate(self.blocks) if not self._on_disk(k))
        chunk = max(_CHUNK_BYTES, _pow2_ceil(max((max(b.sizes.values(), default = 0)
                                                  for k, b in enumerate(self.blocks) if not self._on_disk(k)),
                                                 default = 0)))
        budget = _pin_budget()
        buf, used = None, 0
        for k, b in enumerate(self.blocks):
            if self._to_disk(k, b):
                continue
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
            self._publish_packed(k)

    def _publish_packed(self, k):
        # Other local ranks hold back only the rise still to come, chunk padding already committed
        # included; not capped at the first estimate, since freeing a storage larger than its block
        # raises free RAM the next blocks need.
        credit = getattr(self, "_host_credit", None)
        if credit is not None and k < len(credit) and credit[k] > 0:
            # The credit assumes the originals' pages are gone now: heap-served ones need an explicit trim.
            _trim_host_heap()
        own = getattr(self, "_host_own", None)
        walk = getattr(self, "_host_walk", None)
        if own is not None and walk is not None:
            scratch = getattr(self, "_encode_scratch", 0)
            if scratch and not any(b.compressible() for b in self.blocks[k + 1:]):
                scratch = 0
            _publish_host_pending(walk.pending(k) + scratch)

    def _pack_registered(self):
        # One exact-size pageable arena, page-locked in place up to the budget (cudaHostRegister).
        global _PINNED_MEMORY_AVAILABLE
        if not _can_register():
            return False
        total = sum(b.nbytes() for k, b in enumerate(self.blocks) if not self._on_disk(k))
        arena = _lazy_host_empty(total)
        used, ends = 0, []
        for k, b in enumerate(self.blocks):
            if self._to_disk(k, b):
                ends.append(used)
                continue
            if self._compress and b.compressible():
                # Encoded regions are never larger than raw ones, so they fit the arena; on Linux its unused tail
                # is never written and so never committed (and never pinned).
                used = b.pack_compressed(arena, used, self.device)
            else:
                regions = {}
                for d, need in b.sizes.items():
                    regions[d] = arena[used:used + need]
                    used += need
                b.pack(regions)
            self._release(b)
            ends.append(used)
            self._publish_packed(k)
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

    def _to_disk(self, k, b):
        """Packing step of a disk-tier block: spill it unless spilled at load, then release it."""
        if not self._on_disk(k):
            return False
        dec = self._disk_decoders.get(torch.device(next(iter(b.sizes)))) if self._disk_decoders and b.sizes else None
        if b.disk is None:
            b.spill(encode_device = dec.device if dec is not None else None)
        elif dec is not None:
            b.encode_disk(dec.device)
        if b.compressed():
            b.decoders = self._disk_decoders
        b.tier = self._disk_tier
        self._release(b)
        self._publish_packed(k)
        return True

    def _release(self, block):
        freed = block.evict()
        if freed is not None:
            self.free[block.sig].append(freed)

    def _keep(self, block):
        """Backward is done with the block, but it stays on the card for the next forward."""
        block.pending = False

    def _acquire(self, block, steal):
        free = self.free[block.sig]
        if not free and steal:
            # Only after an interrupted step, since prefetches never steal (could evict the running block); a pending
            # block's slot is still read by backward, so reuse corrupts gradients silently.
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
        if getattr(block, "copy_open", False) or getattr(block, "copy_pending", None):
            self._harvest_copy(block)
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
            if _checkpoints_in_forward(module) and self._wrap_checkpoint(module, i):
                # The swap runs inside the checkpointed function (_wrap_checkpoint); here only the device move.
                return self._move(i, args, kwargs)
            return self._enter(i, args, kwargs)
        return _opaque(hook)

    def _move(self, i, args, kwargs):
        home = self.blocks[i].home
        x = next((a for a in args if isinstance(a, torch.Tensor)), None)
        if x is None:
            x = kwargs.get("hidden_states")
        if isinstance(x, torch.Tensor) and home is not None and x.device != home:
            return _to_device(args, home), _to_device(kwargs, home)
        return None

    def _wrap_checkpoint(self, module, i):
        # Checkpoint inside `forward`: module hooks never run in the recompute, so wrap the checkpointed function.
        fn = getattr(module, "_gradient_checkpointing_func", None)
        if not callable(fn):
            return False
        seen = fn
        while seen is not None:   # ours may sit under routing replay's _RecordingCheckpoint
            if getattr(seen, "_unsloth_block_swap", None) is self:
                return True
            seen = getattr(seen, "_unsloth_replay_checkpoint", None) or getattr(seen, "__wrapped__", None)
        inner = fn

        def run_in_checkpoint(function, *args, **kw):
            def run(*a, **k):
                self._enter(i, a, {**(getattr(function, "keywords", None) or {}), **k})
                return self._leave(i, module, function(*a, **k))
            return inner(run, *args, **kw)

        run_in_checkpoint._unsloth_block_swap = self
        run_in_checkpoint.__wrapped__ = inner
        if hasattr(inner, "_unsloth_replay_checkpoint"):   # routing replay then sees it as wrapped, not rewrapping
            run_in_checkpoint._unsloth_replay_checkpoint = inner._unsloth_replay_checkpoint
        module._gradient_checkpointing_func = run_in_checkpoint
        self.__dict__.setdefault("_ckpt_wrapped", weakref.WeakSet()).add(module)
        return True

    @_opaque
    def _enter(self, i, args, kwargs):
        b = self.blocks[i]
        self.current = self.indices[i]
        tier = getattr(self, "_disk_tier", None)
        if tier is not None:
            tier.pump()
            if b.disk is not None and any(p.requires_grad for p in b.params):
                raise ValueError("Unsloth: offload_layers' disk tier holds frozen weights only; a weight of layer "
                                 f"{self.current} was unfrozen (full finetuning needs it in GPU or host RAM).")
        if self._auto and not self._settled:
            self._note_block_peak(b)
        self._fetch(b, steal = True)
        marks = getattr(b, "marks", None)
        # A sample still in flight is kept for _harvest: re-recording its events would lose it.
        # mark_open: 1 = waiting sample open, 2 = full sample recorded, 3 = wait-only sample closed (the forward
        # hook never ran: a recompute that stopped early) but its events may still be in flight.
        in_flight = marks is not None and (
            (b.mark_open == 2 and not marks[2].query()) or (b.mark_open in (1, 3) and not marks[1].query()))
        if marks is not None and not in_flight:
            self._harvest_marks(b)
            stream = torch.cuda.current_stream(b.home)
            marks[0].record(stream)
            b.wait()
            marks[1].record(stream)
            b.mark_open = 1
        else:
            if marks is not None and b.mark_open == 1:
                b.mark_open = 3   # keep it for _harvest; this layer call records no sample of its own
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
            self._forward_task[i] = _graph_task_id()
        self._input_hooked[i] = hooked
        return moved

    def _post(self, i):
        def hook(module, args, output):
            if _checkpoints_in_forward(module) and getattr(
                    getattr(module, "_gradient_checkpointing_func", None), "_unsloth_block_swap", None) is self:
                return output
            return self._leave(i, module, output)
        return _opaque(hook)

    @_opaque
    def _leave(self, i, module, output):
        b = self.blocks[i]
        if getattr(b, "mark_open", 0) == 1:
            b.marks[2].record(torch.cuda.current_stream(b.home))
            b.mark_open = 2
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

    def _bwd(self, i):
        def hook(grad):
            if self._auto and not self._settled:
                self._note_block_peak(self.blocks[i])
            # The first `depth` blocks are the next forward's first fetches (_end_step arms them into the same
            # pool): keep them instead of copying the same bytes in again.
            release = self._keep if _KEEP_BOTTOM and i < getattr(self, "depth", 0) else self._release
            # Reentrant checkpoint: the input grad can land before a token-id-only branch (Qwen4Exp n-gram keys)
            # reads its frozen weights, so release only once the inner backward is done.
            task = _graph_task_id()
            if grad is not None and task != -1 and self._forward_task[i] not in (-1, task):
                try:
                    torch.autograd.Variable._execution_engine.queue_callback(lambda: release(self.blocks[i]))
                except RuntimeError:
                    release(self.blocks[i])
            else:
                release(self.blocks[i])
            # Queue here only when the output's grad hook did not; else a checkpoint replay ends the
            # step twice.
            if i == 0 and not self._step_queued:
                self._step_queued = True
                try:
                    torch.autograd.Variable._execution_engine.queue_callback(self._end_step)
                except RuntimeError:  # not inside a backward
                    self._end_step()
            elif i == 0:
                # A deferred release (a first swapped layer whose input needs no grad) can run after the queued
                # _end_step armed the next forward while this block was still resident: arm again, no new step.
                self._arm(forward = True)
        return _opaque(hook)

    def _clear_step_queued(self):
        self._step_queued = False

    def _end_step(self):
        # Backward is done: a wait sample whose forward hook never ran (a recompute that stopped early) is complete.
        for b in self.blocks:
            if getattr(b, "marks", None) is not None and getattr(b, "mark_open", 0) == 1:
                b.mark_open = 3
        if self._step_queued:
            # Clear the mark queued behind deferred releases, so a late _bwd(0) cannot end the step
            # again.
            try:
                torch.autograd.Variable._execution_engine.queue_callback(self._clear_step_queued)
            except RuntimeError:
                self._step_queued = False
        if self._auto and not self._settled:
            # Grads accumulated in the layers below the first swapped one are still allocated here.
            for dv in self._peak_devs:
                self._note_peak(dv)
        self._last_peak, self._step_peak = self._step_peak or {}, {}
        self._steps += 1
        if self._steps == 1 and self._tot is not None:
            # Step 1 includes copies armed at install and warmup: windows start after it. Wait for its events once,
            # so none of its samples is still pending (and later counted in the first window) when the baseline is set.
            for dv in {d for d in self.streams if isinstance(d, torch.device) and d.type == "cuda"} | set(self._peak_devs):
                torch.cuda.synchronize(dv)
            self._harvest()
            self._seen, self._adapt_seen = dict(self._tot), dict(self._tot)
        elif not self._settled:
            self._adapt_depth()
        self._arm(forward = True)

    def _plain_peak(self, module, args, kwargs):
        if not self._auto or self._settled or not torch.is_grad_enabled():
            return
        homes = self._peak_devs
        for dv in homes:
            self._note_peak(dv)
        x = next((a for a in args if isinstance(a, torch.Tensor)), None)
        if x is None:
            x = kwargs.get("hidden_states")
        if isinstance(x, torch.Tensor) and x.requires_grad:
            def peak(grad):
                if self._auto and not self._settled:
                    for dv in homes:
                        self._note_peak(dv)
            x.register_hook(_opaque(peak))

    def _state_dict(self, i):
        def hook(module, state_dict, prefix, local_metadata):
            # Per call: PEFT renames weights after install, and every alias must get the host copy.
            b = self.blocks[i]
            # Every caller gets real values: peft's adapter save selects keys after this (biases, modules_to_save,
            # embeddings), so nothing may be left as an evicted placeholder. A compressed block decodes once per
            # call, which costs its raw size in host RAM for as long as the caller keeps the dict.
            host = None if (b.compressed() or b.disk is not None) else b.host
            for name, p in module.named_parameters(remove_duplicate = False):
                idx = b.index.get(id(p))
                if idx is not None and prefix + name in state_dict:
                    if host is None:
                        host = b.host_views()
                    state_dict[prefix + name] = host[idx]
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
        for m in list(getattr(self, "_ckpt_wrapped", ())):
            parent, fn = None, m.__dict__.get("_gradient_checkpointing_func")
            while fn is not None and getattr(fn, "_unsloth_block_swap", None) is not self:
                parent, fn = fn, getattr(fn, "_unsloth_replay_checkpoint", None)   # under routing replay
            if fn is None:
                continue
            if parent is None:
                m._gradient_checkpointing_func = fn.__wrapped__
            else:
                parent._unsloth_replay_checkpoint = fn.__wrapped__
        self._ckpt_wrapped = weakref.WeakSet()
        tier = getattr(self, "_disk_tier", None)
        if tier is not None:
            tier.drain()
        # Drain every copy / decode stream before any slot, staging buffer or host arena can be freed.
        decs = list((getattr(self, "_decoders", {}) or {}).values()) + list((getattr(self, "_disk_decoders", {}) or {}).values())
        for dev in list(self.streams) + [dec.stream.device for dec in decs]:
            try:
                torch.cuda.synchronize(dev)
            except Exception:
                pass
        # Free the pool first, else the restore can OOM.
        self.free = {}
        with _no_inference_mode():
            for b in self.blocks:
                if not b.resident:
                    if getattr(b, "disk", None) is not None:
                        for p, t in zip(b.params, b.disk_views(b.devices[0])):
                            p.data = t
                            p.__dict__.pop("_unsloth_disk", None)
                    elif getattr(b, "codec", None) and b.compressed():
                        # Decoded on the card one region at a time, then each weight gets its own storage; with
                        # no room for a decoded region next to the weights, decoded on the host instead.
                        for d in list(b.host_buf):
                            mine = [(p, sig) for p, pd, sig in zip(b.params, b.devices, b.sig) if pd == d]

                            def restore(region):
                                # All copies first, then rebind: an OOM midway leaves every weight as it was.
                                out = []
                                for p, (shape, dt, _, off) in mine:
                                    n = torch.Size(shape).numel() * torch.empty(0, dtype = dt).element_size()
                                    out.append(region[off:off + n].view(dt).view(shape).to(d, copy = True))
                                for (p, _), t in zip(mine, out):
                                    p.data = t

                            oom = False
                            try:
                                restore(b._region_on(d, d))
                            except torch.cuda.OutOfMemoryError:
                                oom = True
                            if oom:
                                # Outside the except block, so its traceback no longer holds the decoded region.
                                # No room for a decoded region beside the weights: decode on the host, upload each.
                                # The decode streams were drained above, so their staging ring and aux buffers can
                                # go now (the rest of the restore decodes on the host); empty_cache can then hand
                                # their memory to the uploads.
                                self._release_decoders()
                                torch.cuda.empty_cache()
                                c = b.codec.get(d)
                                restore(_codec().decode_region_cpu(c, b.host_buf[d]) if c is not None else b.host_buf[d])
                    else:
                        for p, h, d in zip(b.params, b.host, b.devices):
                            p.data = h.to(d, copy = True)
                    b.resident = True
        # Callers holding this object (the fast decode loop) must see a no-op, not an empty pool.
        self.blocks = []
        self._decoders = {}
        self._disk_decoders = {}
        self._chunks = []
        if tier is not None:
            tier.close()
            self._disk_tier = None
        # A removed controller must not keep the model's layers (and their restored GPU weights) alive.
        self._reserve_layers, self._reserve_owner = (), None
        for dev in self.streams:
            torch.cuda.synchronize(dev)

    @contextmanager
    def shaped(self):
        """Expose evicted weight shapes to PEFT using zero-stride placeholders."""
        held = []
        try:
            for b in self.blocks:
                if not b.resident:
                    for p, e, sig in zip(b.params, b.empties, b.sig):
                        held.append((b, p, e))
                        p.data = e.new_zeros(()).expand(sig[0])
            yield
        finally:
            # A block fetched inside the context holds real weights now: leave it.
            for b, p, e in held:
                if not b.resident:
                    p.data = e

    def host_bytes(self):
        """Bytes of the swapped layers, wherever they live off the card (host RAM or the disk tier)."""
        return sum(b.nbytes() for b in self.blocks)

    def ram_bytes(self):
        return sum(b.nbytes() for b in self.blocks if getattr(b, "disk", None) is None)

    def disk_bytes(self):
        return sum(b.nbytes() for b in self.blocks if getattr(b, "disk", None) is not None)

    def compression_stats(self):
        """Engagement of the lossless fetch codec: compressed layers, stored vs raw bytes, decoded fetches."""
        blocks = [b for b in self.blocks if getattr(b, "codec", None) and b.compressed()]
        decs = getattr(self, "_decoders", {}) or {}
        return {
            "enabled": bool(getattr(self, "_compress", False)),
            "decode": getattr(self, "_compress_note", None),
            "layers": len(blocks),
            "raw_bytes": sum(b.nbytes() for b in blocks),
            "stored_bytes": sum(b.wire_bytes() for b in blocks),
            "fetches": sum(d.fetches for d in decs.values()),
            "triton": all(d.use_triton for d in decs.values()) if decs else None,
            "fallback": next((d.failed for d in decs.values() if d.failed), None),
            "decoder_bytes": sum(d.nbytes() for d in decs.values()),
        }

    def pool_bytes(self):
        held = [b.slot for b in self.blocks if b.slot is not None]
        free = [slot for slots in self.free.values() for slot in slots]
        return sum(t.numel() for bufs, _ in held + free for t in bufs.values())

    def signatures(self):
        return len(self.free)

    def resident_count(self):
        return sum(b.resident for b in self.blocks)

    def _harvest_copy(self, b):
        # Only finished copies are read: query() never blocks the host.
        def done(events):
            return all(e.query() for e in events.values())

        def add(starts, ends):
            # max(0, ...): HIP can order timestamps of events recorded while a stream sat idle backwards.
            self._tot["copy_ms"] += sum(max(0.0, starts[d].elapsed_time(e)) for d, e in ends.items())
            self._tot["copies"] += 1
            self._tot["copy_bytes"] += b.nbytes()
            if "wire_bytes" in self._tot:
                wire = getattr(b, "wire_bytes", None)
                self._tot["wire_bytes"] += wire() if wire is not None else b.nbytes()

        keep = []
        for starts, ends in getattr(b, "copy_pending", []):
            if done(ends):
                add(starts, ends)
            else:
                keep.append((starts, ends))
        b.copy_pending = keep
        if b.copy_open and done(b.events):
            add(b.t_copy, b.events)
            b.copy_open = False

    def _harvest_marks(self, b):
        if b.mark_open == 2 and b.marks[2].query():
            self._tot["stall_ms"] += max(0.0, b.marks[0].elapsed_time(b.marks[1]))
            self._tot["compute_ms"] += max(0.0, b.marks[1].elapsed_time(b.marks[2]))
            self._tot["layers"] += 1
        elif b.mark_open in (1, 3) and b.marks[1].query():
            # The forward hook never ran (a checkpoint recompute that stopped early): the wait still happened.
            self._tot["stall_ms"] += max(0.0, b.marks[0].elapsed_time(b.marks[1]))
        b.mark_open = 0

    def _harvest(self):
        for b in self.blocks:
            if getattr(b, "copy_open", False) or getattr(b, "copy_pending", None):
                self._harvest_copy(b)
            if (getattr(b, "mark_open", 0) == 2 and b.marks[2].query()) or (
                    getattr(b, "mark_open", 0) == 3 and b.marks[1].query()):
                self._harvest_marks(b)

    def _since(self, seen):
        return {k: self._tot[k] - seen[k] for k in _STAT_KEYS}

    def _adapt_depth(self):
        # Between steps only: the "keep the last depth blocks" rule in _post must not change mid-sweep.
        self._harvest()
        d = self._since(self._adapt_seen)
        if d["layers"] == 0 or d["copies"] == 0:
            return
        self._sampled += 1
        if self._sampled < _AUTO_SAMPLE and self._steps < _AUTO_STEPS:
            return
        self._sampled = 0
        self._adapt_seen = dict(self._tot)
        busy = d["compute_ms"] + d["stall_ms"]
        stall = d["stall_ms"] / busy if busy > 0 else 0.0
        # Measured, not predicted: a slot that does not cut the stall (a bus slower than compute, or noise)
        # is given back, so a slow bus costs one slot for a couple of steps.
        if self._last_stall is not None and stall > 0.75 * self._last_stall:
            self._shrink_pool()
            self.depth -= 1
            self._settled = True
        elif stall > _AUTO_STALL and self.depth < _AUTO_MAX_DEPTH and self._grow_pool():
            self.depth += 1
            self._last_stall = stall
        else:
            self._settled = True
        if self._steps >= _AUTO_STEPS:
            self._settled = True
        if self._settled:
            print(f"Unsloth: offload_layers prefetch_depth = 'auto' settled on {self.depth} "
                  f"(copies stalled {100 * stall:.1f}% of the swapped layers' time).")

    def _output_peak(self, module, args, output):
        if not self._auto or self._settled:
            return
        self._step_queued = False
        homes = self._peak_devs
        for dv in homes:
            self._note_peak(dv)
        out = _output_logits(output)
        node = getattr(out, "grad_fn", None) if isinstance(out, torch.Tensor) else None
        # When the output's grad lands, the op that consumed the output (the loss) has just held the output,
        # its grad and a temporary of the same size; only the grad is still allocated. A node hook, not a
        # tensor hook: callers keep only the loss, and a tensor hook dies with the dropped output.
        hooks = [(node, out.device, out.nbytes)] if node is not None else []
        # The loss too: fused cross entropy (Unsloth's) computes it from the hidden states, so returned logits
        # (UNSLOTH_RETURN_LOGITS=1) need not be its ancestors and backward from the loss never reaches them.
        loss = getattr(output, "loss", None)
        if loss is None and isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
            loss = output[0] if output[0].numel() == 1 else None
        loss_node = getattr(loss, "grad_fn", None) if isinstance(loss, torch.Tensor) else None
        if loss_node is not None and loss_node is not node:
            hooks.append((loss_node, None, 0))
        for node, dev, size in hooks:

            def grad_peak(*_, dev = dev, size = size):
                if self._auto and not self._settled:
                    for dv in homes:
                        self._note_peak(dv, extra = 2 * size if dev is not None and _cuda_index(dv) == _cuda_index(dev)
                                        else 0)
                    # Queued from the outer graph, so it runs after the whole backward: a reentrant checkpoint's
                    # inner backward per layer, layers below the first swapped one and a frozen first block included.
                    if not self._step_queued:
                        self._step_queued = True
                        torch.autograd.Variable._execution_engine.queue_callback(self._end_step)
            register = getattr(node, "register_prehook", None) or node.register_hook
            register(_opaque(grad_peak))

    def _note_block_peak(self, b):
        for dv in b.sizes or (b.home,):
            self._note_peak(dv)

    def _note_peak(self, dv, extra = 0):
        # Our own per-step high-water mark: torch's peak counter still holds the load and the pre-swap model.
        if dv is None:   # a block with nothing swappable (all trainable, or only shared weights) has no home card
            return
        if self._step_peak is None:
            self._step_peak = {}
        k = torch.device(dv).index
        self._step_peak[k] = max(self._step_peak.get(k, 0), torch.cuda.memory_allocated(dv) + extra)

    def _optimizer_bytes(self):
        """AdamW's two fp32 moments + the foreach temporary per trainable param (12 bytes), per card. Read from the
        current params: adapters attached after install (PEFT on a swapped base) count too."""
        owner = self._reserve_owner() if self._reserve_owner is not None else None
        params = owner.parameters() if owner is not None else \
            (q for layer in self._reserve_layers for q in layer.parameters())
        reserve, seen = {}, set()
        for q in params:
            if q.requires_grad and q.is_cuda and id(q) not in seen:
                seen.add(id(q))
                k = _cuda_index(q.device)
                reserve[k] = reserve.get(k, 0) + q.numel() * 12
        return reserve

    def _grow_pool(self):
        """One more slot for every pool that can use one, if each card still has room for it."""
        sigs = [b.sig for b in self.blocks]
        want = [s for s in self.free if sigs.count(s) > self.depth + 1]
        if not want:
            # Every block already has a slot of its own (signatures alternating A/B/A/B): a deeper lookahead
            # allocates nothing and only starts each copy earlier, up to one block short of the whole sweep.
            return self.depth + 1 < len(self.blocks)
        need = {}
        for s in want:
            size = {}
            for shape, dt, dv, off in s:
                n = torch.Size(shape).numel() * torch.empty(0, dtype = dt).element_size()
                size[dv] = max(size.get(dv, 0), off + -(-n // _ALIGN) * _ALIGN)
            for dv, n in size.items():
                need[dv] = need.get(dv, 0) + n
        fresh = self._optimizer_bytes() if (self._reserve_owner is not None or self._reserve_layers) else {}
        reserve = {k: max(fresh.get(k, 0), self._optimizer_reserve.get(k, 0)) for k in {*fresh, *self._optimizer_reserve}}
        for dv, n in need.items():
            # Between steps the activations are gone; keep room for the next step's peak.
            k = _cuda_index(dv)
            peak = (self._last_peak or {}).get(torch.device(dv).index, 0)
            # Hook samples miss temporaries made and freed inside a layer or the loss: once the allocator's
            # high-water mark has risen past where it stood at install, that mark is a real training peak.
            hwm = torch.cuda.max_memory_allocated(dv)
            if self._hwm_fresh or hwm > self._hwm_base.get(k, hwm):
                peak = max(peak, hwm)
            transient = max(0, peak - torch.cuda.memory_allocated(dv))
            # Still keep a workspace margin for intra-op temporaries the samples cannot see.
            transient += max(_WORKSPACE_MIN_BYTES, int(torch.cuda.get_device_properties(dv).total_memory * _WORKSPACE_FRACTION))
            # optimizer.step() runs after every peak sample: keep room for AdamW's two moments (lazily allocated on
            # the first step) and the foreach temporary, as plan_block_swap budgets them.
            transient += reserve.get(_cuda_index(dv), 0)
            if _free_device_bytes(dv) - transient < 2 * n:
                return False
        for s in want:
            self.free[s].append(self._new_slot(s))
        return True

    def _shrink_pool(self):
        sigs = [b.sig for b in self.blocks]
        for s, free in self.free.items():
            held = sum(b.sig == s and b.slot is not None for b in self.blocks)
            if free and held + len(free) > min(self.depth, sigs.count(s)):
                free.pop()

    def stats(self, reset = True):
        """Where each swapped layer is, plus copy / stall / compute totals since the last call."""
        self._harvest()
        window = self._since(self._seen)
        if reset:
            self._seen = dict(self._tot)
        state = {}
        for li, b in zip(self.indices, self.blocks):
            on_disk = getattr(b, "disk", None) is not None
            if not b.resident:
                state[li] = "disk" if on_disk else "host"
            elif on_disk and b.disk_state:
                state[li] = "reading"
            elif b.pending:
                state[li] = "held"
            elif not all(e.query() for e in b.events.values()):
                state[li] = "copying"
            else:
                state[li] = "gpu"
        return {
            "total_layers": self.total,
            "swapped": list(self.indices),
            "state": state,
            "current": self.current,
            "prefetch_depth": self.depth,
            "auto_depth": self._auto,
            "depth_settled": self._settled,
            "host_bytes": self.host_bytes(),
            "ram_bytes": self.ram_bytes(),
            "pinned_bytes": self.pinned_bytes,
            "pool_bytes": self.pool_bytes(),
            "timing": _TIMING,
            "compression": self.compression_stats(),
            **window,
            **self._disk_stats(reset),
        }

    def _disk_stats(self, reset):
        """Disk layers, plus reads / bytes / time since the last stats()."""
        tier = getattr(self, "_disk_tier", None)
        disk = [b for b in self.blocks if getattr(b, "disk", None) is not None]
        out = {"disk_layers": len(disk), "disk_bytes": sum(b.nbytes() for b in disk),
               "disk_dir": disk[0].disk_store.directory if disk else None}
        now = tier.stats() if tier is not None else {}
        seen = getattr(self, "_disk_seen", None) or {}
        for k in ("disk_reads", "disk_read_bytes", "disk_read_ms", "disk_wait_ms"):
            out[k] = now.get(k, 0) - seen.get(k, 0)
        if reset and tier is not None:
            self._disk_seen = dict(now)
        return out


_STAT_KEYS = ("copies", "copy_ms", "copy_bytes", "wire_bytes", "stall_ms", "compute_ms", "layers")


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


def _file_backed_ranges():
    """[(start, end)] of this process's file-backed mappings (Linux /proc/self/maps); [] elsewhere."""
    out = []
    try:
        with open("/proc/self/maps", "r", encoding = "utf-8", errors = "replace") as f:
            for line in f:
                fields = line.split(maxsplit = 5)
                if len(fields) < 6 or not fields[5].strip().startswith("/") or fields[4] == "0":
                    continue   # anonymous, [heap], [stack] or a pseudo file (inode 0)
                lo, hi = (int(x, 16) for x in fields[0].split("-"))
                out.append((lo, hi))
    except OSError:
        return []
    return out


def _in_ranges(ptr, ranges):
    return any(lo <= ptr < hi for lo, hi in ranges)


def _cuda_index(dv):
    """The card index of a CUDA device (a bare "cuda" is the current card); None for anything else."""
    d = torch.device(dv)
    if d.type != "cuda":
        return None
    return d.index if d.index is not None else torch.cuda.current_device()


def _resolved_depth(prefetch_depth):
    """The depth BlockSwap starts at: "auto" starts at 1 (2 when timing is off, the fixed default)."""
    if prefetch_depth == "auto":
        return 1 if _TIMING else 2
    return max(1, int(prefetch_depth))


def _pool_bytes(sizes, depth, sigs = None):
    # One pool per shape signature (as BlockSwap keys them), depth + 1 slots each, fewer if fewer blocks share it.
    depth = _resolved_depth(depth)  # as BlockSwap
    counts = {}
    for i, b in enumerate(sizes):
        key = sigs[i] if sigs is not None else b
        size, count = counts.get(key, (b, 0))
        counts[key] = (max(size, b), count + 1)
    return sum(b * min(depth + 1, c) for b, c in counts.values())


def auto_swap_indices(layers, reserve_bytes, prefetch_depth = 2, free_bytes = None, tiers = False,
                      host_free_bytes = None):
    """Fewest layers to move to host RAM so each GPU keeps `reserve_bytes` free, net of the slot pool.

    Returns (indices, shortfall_left); shortfall_left > 0 when one layer per device must stay and it is not enough.
    `tiers = True` also returns the indices host RAM cannot hold (`host_free_bytes`, default the measured free RAM
    less the system reserve and UNSLOTH_OFFLOAD_HOST_LIMIT), which BlockSwap serves from the disk tier."""
    indices, left = _auto_swap_indices(layers, reserve_bytes, prefetch_depth, free_bytes)
    if not tiers:
        return indices, left
    return indices, left, _disk_tier_indices(layers, indices, host_free_bytes, _resolved_depth(prefetch_depth))


_DISK_PLACEMENT_ENV = "UNSLOTH_OFFLOAD_DISK_PLACEMENT"


def _disk_placement(movable, n_disk, depth):
    """Which of `movable` (swapped block positions, sweep order) to serve from disk, `n_disk` of them; None keeps
    the tail (UNSLOTH_OFFLOAD_DISK_PLACEMENT=tail).

    Disk blocks go first where a fetch is skipped: the last `depth` blocks stay on the card from forward to
    backward, and the first `depth` from backward to the next forward. The rest are spread evenly between the
    host RAM blocks, so a read is always in flight while RAM blocks compute, instead of the disk idling through a
    run of RAM blocks and then stalling through a run of disk blocks."""
    if os.environ.get(_DISK_PLACEMENT_ENV, "spread").strip().lower() == "tail":
        return None
    m = list(movable)
    n_disk = min(n_disk, len(m))
    if n_disk <= 0:
        return set()
    ends = []
    depth = max(0, int(depth))
    for j in range(depth):
        ends.append(len(m) - 1 - j)
        if _KEEP_BOTTOM:
            ends.append(j)
    pick = []
    for j in ends:
        if len(pick) < n_disk and 0 <= j < len(m) and j not in pick:
            pick.append(j)
    rest = [j for j in range(len(m)) if j not in pick]
    left = n_disk - len(pick)
    if left > 0:
        pick += [rest[(2 * t + 1) * len(rest) // (2 * left)] for t in range(left)]
    return {m[j] for j in pick}


def _disk_tier_indices(layers, indices, host_free_bytes = None, prefetch_depth = 2):
    """The swapped `indices` host RAM cannot hold, as BlockSwap's install picks them."""
    if not indices:
        return []
    sizes = [sum(p.nbytes for _, p in _swappable(layers[i])) for i in indices]
    if host_free_bytes is None:
        try:
            host_free_bytes = _host_free()[1] + sum(
                p.nbytes for i in indices for _, p in _swappable(layers[i]) if p.device.type == "cpu")
        except Exception:
            return []
        limit = _disk().host_limit_bytes()
        if limit is not None:
            host_free_bytes = min(host_free_bytes, limit)
    if sum(sizes) <= host_free_bytes:
        return []
    budget = host_free_bytes - 2 * _disk()._round_up(max(sizes)) - max(sizes)
    used, out = 0, []
    for i, n in zip(indices, sizes):
        if used + n > budget:
            out.append(i)
        else:
            used += n
    spread = _disk_placement(range(len(indices)), len(out), prefetch_depth) if out else None
    if spread is not None and sum(n for j, n in enumerate(sizes) if j not in spread) <= budget:
        return [indices[j] for j in sorted(spread)]
    return out


def _auto_swap_indices(layers, reserve_bytes, prefetch_depth = 2, free_bytes = None):
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


def _mark_host_kept(model):
    names = getattr(model, "_no_placement_params", None)
    if not names or os.environ.get("UNSLOTH_PLACE_NO_PLACEMENT_PARAMS", "0") == "1":
        return
    params = list(model.named_parameters(remove_duplicate = False))
    # Tag the whole owner module, not the parameter: a host load replaces the Parameter.
    try:
        from .device_map_planner import no_placement_matcher
    except ImportError:  # loaded as a standalone file (tests): same rule, inline
        def no_placement_matcher(model, names):
            dotted = [n for n in names if "." in n]
            bare = {n for n in names if "." not in n}
            no_split = set(getattr(model, "_no_split_modules", None) or [])
            roots = {p for p, m in model.named_modules() if type(m).__name__ in no_split} if bare else set()
            return lambda name: any(name == n or name.endswith("." + n) for n in dotted) or (
                name.rpartition(".")[2] in bare and name.rpartition(".")[0] in roots
            )
    match = no_placement_matcher(model, names)
    for owner in {n.rsplit(".", 1)[0] for n, _ in params if match(n)}:
        model.get_submodule(owner)._unsloth_host_kept = True


# Layer list -> the model find_decoder_layers found it in: BlockSwap watches that model's output for the step peak.
_LAYER_OWNERS = weakref.WeakKeyDictionary()


def find_decoder_layers(model):
    _mark_host_kept(model)
    layers = _find_decoder_layers(model)
    try:
        _LAYER_OWNERS[layers] = weakref.ref(model)
    except TypeError:
        pass
    return layers


def _find_decoder_layers(model):
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


def _host_layer_bytes(make_layer, idx, tensors, compute_dtype, quantize_4bit, skip_modules, prefix):
    """Host bytes build_host_layers will copy for layer `idx`: packed 4-bit Linear weights (prequantized or
    quantized here) take half a byte per element, everything else the compute dtype."""
    with torch.device("meta"):
        layer = make_layer(idx)
    width = torch.empty(0, dtype = compute_dtype).element_size()
    base, total, done = f"{prefix}{idx}.", 0, set()
    for mname, mod in layer.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        key = base + mname + ".weight"
        packed = any(k.startswith(key + ".") for k in tensors) or (
            quantize_4bit and not any(s in mname.split(".") for s in skip_modules))
        n = mod.weight.numel()
        total += (n + 1) // 2 if packed else n * width
        done.add(id(mod.weight))
        if mod.bias is not None:
            total += mod.bias.numel() * width
            done.add(id(mod.bias))
    for p in layer.parameters():
        if id(p) not in done:
            total += p.numel() * (width if p.is_floating_point() else p.element_size())
    return total


def _spill_layer_weights(layer, shared_ids, need_bytes):
    """Spill a frozen layer's swappable weights in _Block's slot layout (a fetch is one read); the rest goes to
    host RAM as usual. Returns the region's bytes."""
    params = [p for _, p in _swappable(layer) if id(p) not in shared_ids]
    layout, off = [], 0
    for p in params:
        layout.append(off)
        off += -(-p.data.nbytes // _ALIGN) * _ALIGN
    if params:
        _disk().spill_params(params, layout, off, need_bytes = need_bytes(off))
    spilled = {id(p) for p in params}
    with torch.no_grad():
        for p in layer.parameters():
            if id(p) not in spilled and p.device.type not in ("cpu", "meta"):
                p.data = p.data.to("cpu")
    return off


def _ram_layers_beside_disk(sizes):
    """How many of `sizes` fit host RAM beside the staging ring; None when the disk tier is off."""
    mod = _disk()
    if mod.disk_tier_reason() is not None or not sizes:
        return None
    largest = max(sizes)
    extra = [2 * mod._round_up(largest), largest]
    for m in range(len(sizes), -1, -1):
        try:
            _check_host_fit(list(sizes[:m]) + extra)
            return m
        except ValueError:
            continue
    return None


def build_host_layers(make_layer, first_idx, count, tensors, device, compute_dtype,
                      quantize_4bit = False, skip_modules = (), prefix = "model.layers."):
    """Build layers [first_idx, first_idx + count) with frozen weights in pinned host RAM; quant_state stays on `device`."""
    import bitsandbytes as bnb
    device = torch.device(device)
    if any(".quant_state." in k for k in tensors):
        quantize_4bit = False
    # Refuse before copying anything when the layers cannot fit in host RAM (sized on meta, no tensor read).
    _begin_host_session()
    try:
        return _build_host_layers(make_layer, first_idx, count, tensors, device, compute_dtype, quantize_4bit,
                                  skip_modules, prefix, bnb)
    finally:
        _end_host_session()


def _build_host_layers(make_layer, first_idx, count, tensors, device, compute_dtype, quantize_4bit, skip_modules,
                       prefix, bnb):
    sizes = [_host_layer_bytes(make_layer, idx, tensors, compute_dtype, quantize_4bit, skip_modules, prefix)
             for idx in range(first_idx, first_idx + count)]
    ram = len(sizes)
    try:
        _check_host_fit(sizes)
    except ValueError:
        ram = _ram_layers_beside_disk(sizes)
        if ram is None:
            raise
    out = []
    pending = sum(sizes)
    for idx in range(first_idx, first_idx + count):
        if out:
            # The layers built so far are in available RAM already: other local ranks hold back only the rest.
            pending -= sizes[len(out) - 1]
            _publish_host_pending(pending)
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
        if len(out) >= ram:
            left = len(sizes) - len(out)
            _spill_layer_weights(layer, (), lambda first, left = left: first * left)
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


def _counted_params(module):
    """Params a host move costs: a host-kept table (`_no_placement_params`, tagged by _mark_host_kept) is already
    in host RAM before any layer moves, so it is neither charged again nor credited."""
    kept = set()
    for m in module.modules():
        if getattr(m, "_unsloth_host_kept", False):
            kept.update(id(p) for p in m.parameters())
    return [p for p in module.parameters() if id(p) not in kept]


def _counted_named(module):
    ids = {id(p) for p in _counted_params(module)}
    return [(n, p) for n, p in module.named_parameters() if id(p) in ids]


# Loaded bytes per declared (meta) byte of a quantized weight before any layer of that kind has loaded:
# bitsandbytes declares the unquantized shape in the compute dtype (16-bit) and stores 4 or 8 bits per value.
_QUANT_PRIOR = {"4bit": 0.25, "8bit": 0.5}


def _param_kind(layer, name, p):
    cls = type(p).__name__
    if cls == "Params4bit":
        return "4bit"
    if cls == "Int8Params":
        return "8bit"
    if name.endswith("weight"):
        owner = layer.get_submodule(name.rpartition(".")[0]) if "." in name else layer
        mcls = type(owner).__name__
        if "4bit" in mcls:
            return "4bit"
        if "8bit" in mcls:
            return "8bit"
    return "dense"


class _HostLoad:
    """What `load_layers_to_host` moved: the layer list and the indices living in host RAM."""

    def __init__(self, n, placement, embeddings = False):
        self.n, self.placement, self.want_embeddings = n, placement, embeddings
        self.layers, self.indices, self.prefixes = None, [], {}
        self.done, self.done_bytes = set(), []
        self.moved_storages = set()
        self.ratios = {}   # loaded bytes / declared meta bytes, per moved layer
        self.embedding_prefixes, self.embeddings, self.embedding_bytes = {}, [], []
        # Each selected layer's bytes as declared on meta, to scale a loaded layer's size to the others'.
        self.shape_bytes, self.last_done = {}, None
        self.owned_names = {}
        # Ratios are per kind and layer: an unquantized skip layer loads full size, the rest at 4
        # bits.
        self.decl, self.kind_ratios = {}, {}
        self.cpu_to_come = 0
        self.disk, self.disk_sizes = set(), []

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
        # A param shared by several selected layers moves once: charge it to the first.
        seen = set()
        self.shape_bytes, self.owned_names = {}, {}
        for i in self.indices:
            own = [(n, p) for n, p in _counted_named(layers[i]) if id(p) not in seen]
            seen.update(id(p) for _, p in own)
            self.shape_bytes[i] = sum(p.nbytes for _, p in own)
            # By name: loading replaces the meta Parameter objects, the names stay.
            self.owned_names[i] = {n for n, _ in own}
            self.decl[i] = {n: (_param_kind(layers[i], n, p), p.nbytes) for n, p in own}
        if self.want_embeddings:
            # A table inside a selected layer moves (and is charged) with that layer, not a second time here.
            inside = tuple(self.prefixes)
            self.embedding_prefixes = {f"{n}.": m for n, m in extra_input_embeddings(model)
                                       if not (inside and f"{n}.".startswith(inside))}

    def _layers_to_come(self, size, ref):
        """Bytes of the selected layers not moved yet. A layer already materialized counts its own bytes; one still
        on meta is scaled from its declared size by the smallest storage ratio seen so far (layer `ref` takes
        `size` bytes loaded), since layers can be quantized differently (skip modules) and each later eviction
        re-checks with real sizes."""
        left = [j for j in self.indices if j not in self.done]
        # The layer being moved first: a storage it shares with a later-loaded selected layer is its own (credited
        # by the caller), never also counted in cpu_to_come for the other layer.
        if ref in left:
            left.remove(ref)
            left.insert(0, ref)
        base = self.shape_bytes.get(ref, 0)
        ratios = list(self.ratios.values()) + ([size / base] if base else [])
        ratio = min(ratios) if ratios else None
        ref_kinds = self._kind_sample(ref)

        def projected(j, names = None):
            # Declared bytes scaled per kind: by the smallest ratio seen for that kind, else the kind's prior.
            decl = self.decl.get(j)
            if not decl:
                return None
            total = 0
            for n, (kind, nbytes) in decl.items():
                if names is not None and n not in names:
                    continue
                seen = list(self.kind_ratios.get(kind, {}).values())
                if kind in ref_kinds:
                    seen.append(ref_kinds[kind])
                total += nbytes * (min(seen) if seen else _QUANT_PRIOR.get(kind, 1.0))
            return int(total)
        out = []
        # Per storage, as evict() moves them: a weight shared by several selected layers (or already moved) once.
        seen = set(self.moved_storages)
        # Bytes of those layers (other than `ref`, which the caller credits itself) already in host RAM: a
        # Transformers 4 shard the device map put on CPU. Evicting them copies nothing.
        self.cpu_to_come = 0
        for j in left:
            named = _counted_named(self.layers[j])
            loaded = [p for _, p in named if p.device.type != "meta"]
            owned = self.owned_names.get(j)
            still_meta = [p for n, p in named if p.device.type == "meta" and (owned is None or n in owned)]
            ptrs = [p.untyped_storage().data_ptr() for p in loaded]
            fresh = [p for p, ptr in zip(loaded, ptrs) if ptr not in seen]
            seen.update(ptrs)
            if j != ref:
                self.cpu_to_come += sum(p.nbytes for p in fresh if p.device.type == "cpu")
            if loaded and not still_meta:
                out.append(sum(p.nbytes for p in fresh))
            elif loaded:
                # Partly loaded (a shard boundary inside the layer): what landed, plus the rest scaled from meta.
                rest = projected(j, {n for n, p in named if p.device.type == "meta" and (owned is None or n in owned)})
                if rest is None:
                    rest = int(sum(p.nbytes for p in still_meta) * (ratio or 1.0))
                out.append(sum(p.nbytes for p in fresh) + rest)
            elif (est := projected(j)) is not None:
                out.append(est)
            elif ratio is None:
                out.append(size)
            else:
                out.append(int(self.shape_bytes.get(j, base) * ratio))
        return out

    def _kind_sample(self, j):
        """{kind: loaded / declared bytes} over layer j's params of that kind that have landed."""
        decl = self.decl.get(j)
        if not decl or self.layers is None:
            return {}
        loaded, declared = {}, {}
        for n, p in _counted_named(self.layers[j]):
            if n not in decl or p.device.type == "meta":
                continue
            kind, nbytes = decl[n]
            loaded[kind] = loaded.get(kind, 0) + p.nbytes
            declared[kind] = declared.get(kind, 0) + nbytes
        return {k: loaded[k] / declared[k] for k in loaded if declared[k]}

    def _pending_embedding_bytes(self, skip = None):
        # Streamed tables still to come; a meta weight already knows its size.
        return [m.weight.nbytes for m in self.embedding_prefixes.values()
                if m not in self.embeddings and m is not skip and getattr(m, "weight", None) is not None]

    def _pending_embedding_cpu_bytes(self, skip = None):
        # Streamed tables still to come that already sit in host RAM (host-kept, or a CPU shard): moving them copies nothing.
        return sum(m.weight.nbytes for m in self.embedding_prefixes.values()
                   if m not in self.embeddings and m is not skip and getattr(m, "weight", None) is not None
                   and m.weight.device.type == "cpu")

    def evict_embedding(self, module):
        weight = getattr(module, "weight", None)
        if module in self.embeddings or weight is None or weight.device.type == "meta":
            return
        size = self.done_bytes[-1] if self.done_bytes else 0
        to_come = self._layers_to_come(size, self.last_done)
        copied = 0 if weight.device.type == "cpu" else weight.nbytes
        own = _check_host_fit(self.done_bytes + self.embedding_bytes + [weight.nbytes]
                              + self._pending_embedding_bytes(skip = module) + to_come,
                              sum(self.done_bytes) + sum(self.embedding_bytes) + (weight.nbytes - copied)
                              + self.cpu_to_come + self._pending_embedding_cpu_bytes(skip = module))
        with torch.no_grad():
            weight.requires_grad_(False)
            if weight.device.type != "cpu":
                weight.data = weight.data.to("cpu")
        self.embeddings.append(module)
        self.embedding_bytes.append(weight.nbytes)
        if own is not None:
            # Now in host RAM, so out of what other local ranks must hold back for this one.
            _publish_host_pending(max(0, own - copied))

    def evict(self, i, force = False):
        if i in self.done:
            return
        layer = self.layers[i]
        params = list(layer.parameters())
        if not force and any(p.device.type == "meta" for p in params):
            return
        # Counted once per storage across layers: a param another selected layer already moved costs nothing more.
        counted = {id(p) for p in _counted_params(layer)
                   if p.device.type == "meta" or p.untyped_storage().data_ptr() not in self.moved_storages}
        # Layers still to come are scaled from this one by their meta sizes; the first check runs before any moved.
        size = sum(p.nbytes for p in params if p.device.type != "meta" and id(p) in counted)
        # The storage ratio compares like with like: the params bind() charged to this layer, loaded vs declared.
        # A shared weight this layer happens to load first is not in its declared bytes.
        owned = self.owned_names.get(i)
        owned_size = size if owned is None else sum(
            p.nbytes for n, p in _counted_named(layer) if n in owned and p.device.type != "meta")
        to_come = self._layers_to_come(owned_size, i)
        copied = sum(p.nbytes for p in params if p.device.type not in ("cpu", "meta") and id(p) in counted)
        on_host = sum(self.done_bytes) + sum(self.embedding_bytes) + \
            sum(p.nbytes for p in params if p.device.type == "cpu" and id(p) in counted)
        try:
            if self.disk:
                raise ValueError("spilling")
            own = _check_host_fit(self.done_bytes + self.embedding_bytes + self._pending_embedding_bytes() + to_come,
                                  on_host + self.cpu_to_come + self._pending_embedding_cpu_bytes())
        except ValueError:
            own = self._ram_or_disk(i, layer, params, size, on_host)
            if own is False:
                raise
            if own == "disk":
                return
        with torch.no_grad():
            for p in params:
                if p.is_floating_point():
                    p.requires_grad_(False)
                if p.device.type not in ("cpu", "meta"):
                    p.data = p.data.to("cpu")
        self.moved_storages.update(p.untyped_storage().data_ptr() for p in params
                                   if p.device.type != "meta" and id(p) in counted)
        self.done.add(i)
        self.done_bytes.append(size)
        if self.shape_bytes.get(i):
            self.ratios[i] = owned_size / self.shape_bytes[i]
        for kind, r in self._kind_sample(i).items():
            self.kind_ratios.setdefault(kind, {})[i] = r
        self.last_done = i
        if own is not None:
            # Now in host RAM, so out of what other local ranks must hold back for this one.
            _publish_host_pending(max(0, own - copied))

    def _ram_or_disk(self, i, layer, params, size, on_host):
        """Host RAM if layer i fits beside those there and the staging ring, else disk ("disk"); False = tier off."""
        mod = _disk()
        if mod.disk_tier_reason() is not None or any(p.device.type == "meta" and not _has_disk_ref(p) for p in params):
            return False
        largest = max([size] + self.disk_sizes)
        staging = 2 * mod._round_up(largest)
        try:
            # Streamed tables cannot go to disk: they keep their room.
            return _check_host_fit(self.done_bytes + self.embedding_bytes + self._pending_embedding_bytes()
                                   + [size, staging, largest], on_host + self._pending_embedding_cpu_bytes())
        except ValueError:
            pass
        self._spill_layer(i, layer)
        return "disk"

    def _spill_layer(self, i, layer):
        """Spill layer i's frozen weights; they become placeholders."""
        mod = _disk()
        with torch.no_grad():
            for p in layer.parameters():
                if p.is_floating_point():
                    p.requires_grad_(False)
        others = {id(p) for j, other in enumerate(self.layers) if j != i for p in other.parameters()}
        left = len([j for j in self.indices if j not in self.done])
        off = _spill_layer_weights(layer, others, lambda first: first * max(1, left))
        self.done.add(i)
        self.done_bytes.append(0)
        self.disk.add(i)
        self.disk_sizes.append(off)
        self.last_done = i
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    _begin_host_session()
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
            model = kwargs.get("model", args[0] if args else None)
            name = kwargs.get("target_name", args[1] if len(args) > 1 else None)
            if model is not None:
                # Before the first weight lands, so bind() reads every layer's size on meta.
                state.bind(model)
            out = original(*args, **kwargs)
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
            state.bind(model)
            out = original(model, *args, **kwargs)
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
        _end_host_session()
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
