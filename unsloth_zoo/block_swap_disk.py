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

"""Read-only disk tier for block swap: frozen layers that fit neither the GPU nor host RAM.

Each such layer is written once (at load, or when the swap is installed) to an unlinked spill file, laid out
exactly like its slot on the card, in the weights' own storage format (4-bit, packed fp4, fp8 stay as they are).
A fetch reads the layer with O_DIRECT, so no page cache competes with the pinned host tier, into a small pinned
staging ring on worker threads, then copies it to the card's slot on the swap's copy stream.
Only used where host RAM would otherwise refuse the offload; UNSLOTH_OFFLOAD_DISK=0 restores that refusal.
"""

import os
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import torch

__all__ = [
    "disk_tier_enabled",
    "disk_tier_reason",
    "host_limit_bytes",
    "SpillFile",
    "DiskTier",
    "DiskRef",
    "spill_params",
    "placeholder_like",
    "disk_ref",
    "read_bytes",
]

ALIGN = 4096
_ENV = "UNSLOTH_OFFLOAD_DISK"
_DIR_ENV = "UNSLOTH_OFFLOAD_DISK_DIR"
_LIMIT_ENV = "UNSLOTH_OFFLOAD_HOST_LIMIT"
_QD_ENV = "UNSLOTH_OFFLOAD_DISK_QD"
_CHUNK_ENV = "UNSLOTH_OFFLOAD_DISK_CHUNK_MB"
_WRITE_CHUNK = 64 << 20
_MARGIN_BYTES = 2 << 30
# Filesystems that live in RAM: spilling there costs the RAM it is meant to save.
_RAM_FS = {"tmpfs", "ramfs", "devtmpfs", "hugetlbfs", "squashfs", "proc", "sysfs"}


def _round_up(n, a = ALIGN):
    return -(-int(n) // a) * a


def disk_tier_reason():
    """Why the disk tier cannot be used here, or None."""
    value = os.environ.get(_ENV, "auto").strip().lower()
    if value in ("0", "false", "no", "off", "never"):
        return f"{_ENV}=0"
    if not sys.platform.startswith("linux") or not hasattr(os, "O_DIRECT") or not hasattr(os, "preadv"):
        return "needs Linux (O_DIRECT reads)"
    try:
        world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    except ValueError:
        world = 1
    if world > 1:
        return "one process per host only (local ranks would share one disk)"
    return None


def disk_tier_enabled():
    return disk_tier_reason() is None


def _parse_bytes(text):
    text = str(text).strip().upper().replace(" ", "")
    for suffix, mult in (("TIB", 2**40), ("GIB", 2**30), ("MIB", 2**20), ("KIB", 2**10),
                         ("TB", 10**12), ("GB", 10**9), ("MB", 10**6), ("KB", 10**3), ("B", 1)):
        if text.endswith(suffix):
            return int(float(text[:-len(suffix)]) * mult)
    return int(float(text))


def host_limit_bytes():
    """UNSLOTH_OFFLOAD_HOST_LIMIT: most host RAM the offloaded layers may take (e.g. "40GB"); None = no cap."""
    value = os.environ.get(_LIMIT_ENV, "").strip()
    if not value:
        return None
    try:
        return max(0, _parse_bytes(value))
    except ValueError:
        return None


def _fs_source(path):
    """(filesystem type, source device) of the mount holding `path`, or (None, None)."""
    best, kind, source = "", None, None
    try:
        real = os.path.realpath(path)
        with open("/proc/mounts", "r", encoding = "utf-8", errors = "replace") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mnt = parts[1].replace("\\040", " ")
                if (real == mnt or real.startswith(mnt.rstrip("/") + "/")) and len(mnt) >= len(best):
                    best, kind, source = mnt, parts[2], parts[0]
    except OSError:
        return None, None
    return kind, source


def _fs_type(path):
    return _fs_source(path)[0]


# Measured on Colab and Kaggle: reads with O_DIRECT through overlay and loop mounts were served from page cache
# (3.5-5 GB/s, buff/cache growing by the file size), so they would take the RAM a spill is meant to save.
_LAYERED_FS = {"overlay", "fuse", "fuse.gcsfuse", "aufs", "9p", "virtiofs", "nfs", "nfs4", "cifs", "smb3"}


def _layered(path):
    kind, source = _fs_source(path)
    return kind in _LAYERED_FS or (source or "").startswith("/dev/loop")


def _candidate_dirs():
    out = []
    env = os.environ.get(_DIR_ENV, "").strip()
    if env:
        return [env]
    for d in ("/mnt/local-scratch", os.environ.get("HF_HUB_CACHE"), os.environ.get("HF_HOME"),
              os.path.expanduser("~/.cache"), os.getcwd()):
        if d and d not in out:
            out.append(d)
    import tempfile
    t = tempfile.gettempdir()
    if t not in out:
        out.append(t)
    return out


def pick_spill_dir(need_bytes):
    """The writable, disk-backed directory with the most free space that holds `need_bytes` plus a margin."""
    best, best_key, best_free, seen = None, None, -1, []
    for d in _candidate_dirs():
        try:
            os.makedirs(d, exist_ok = True)
            if not os.access(d, os.W_OK):
                continue
            fs = _fs_type(d)
            if fs in _RAM_FS:
                seen.append(f"{d} ({fs}, in RAM)")
                continue
            st = os.statvfs(d)
            free = st.f_bavail * st.f_frsize
        except (OSError, AttributeError):  # no statvfs off POSIX
            continue
        seen.append(f"{d} ({free / 2**30:.0f} GiB free)")
        key = (free >= need_bytes + _MARGIN_BYTES, not _layered(d), free)
        if best_key is None or key > best_key:
            best, best_key, best_free = d, key, free
    if best is None or best_free < need_bytes + _MARGIN_BYTES:
        raise ValueError(
            f"Unsloth: offload_layers needs {need_bytes / 2**30:.1f} GiB of disk to spill the layers host RAM "
            f"cannot hold, but no directory has that much free: {', '.join(seen) or 'none writable'}. "
            f"Set {_DIR_ENV} to a local SSD with room, or lower max_seq_length / free host RAM."
        )
    return best


class _HostBuf:
    """Host bytes aligned for O_DIRECT, page-locked in place (exact size, unlike torch's power-of-two pinned
    allocator) when asked and possible. Unregistered before the memory goes."""

    def __init__(self, nbytes, pin):
        self.raw = torch.empty(nbytes + ALIGN, dtype = torch.uint8)
        off = (-self.raw.data_ptr()) % ALIGN
        self.view = self.raw[off:off + nbytes]
        self.ptr = None
        if pin and nbytes and torch.cuda.is_available() and os.environ.get("UNSLOTH_DISABLE_PINNED_MEMORY", "0") != "1":
            try:
                cudart = torch.cuda.cudart()
                if hasattr(cudart, "cudaHostRegister") and int(cudart.cudaHostRegister(self.view.data_ptr(), nbytes, 1)) == 0:
                    self.ptr = self.view.data_ptr()
            except Exception:
                self.ptr = None

    @property
    def pinned(self):
        return self.ptr is not None

    def __del__(self):
        if self.ptr is not None:
            try:
                torch.cuda.cudart().cudaHostUnregister(self.ptr)
            except Exception:
                pass
            self.ptr = None


def _aligned_empty(nbytes, pin):
    """(owner, aligned uint8 view of nbytes) in host memory; pinned when asked and possible."""
    owner = _HostBuf(nbytes, pin)
    return owner, owner.view


class DiskRef:
    """Where one spilled weight lives: (file, byte offset, bytes) plus the shape and dtype it had, and the
    placeholder its parameter holds until BlockSwap takes it over (one element, expanded to the shape)."""
    __slots__ = ("store", "offset", "nbytes", "shape", "dtype", "placeholder")

    def __init__(self, store, offset, nbytes, shape, dtype, placeholder = None):
        self.store, self.offset, self.nbytes = store, int(offset), int(nbytes)
        self.shape, self.dtype, self.placeholder = tuple(shape), dtype, placeholder


def placeholder_like(t):
    """Shape and dtype of `t` on its device for one element of memory: shape checks pass, nothing is resident.
    (A meta tensor cannot replace a CPU or CUDA parameter's data in place.)"""
    return torch.zeros((), dtype = t.dtype, device = t.device).expand(t.shape)


def disk_ref(p):
    """The DiskRef of a parameter spilled at load whose data is still the placeholder, else None. Raises when the
    data was rewritten in between (a dtype cast, a copy): the spill would no longer describe the weight."""
    ref = getattr(p, "_unsloth_disk", None)
    if ref is None or ref.placeholder is None:
        return None
    data = p.data
    if data.data_ptr() == ref.placeholder.data_ptr() and tuple(data.shape) == ref.shape and data.dtype == ref.dtype \
            and all(st == 0 for st in data.stride()):
        return ref
    raise RuntimeError(
        f"Unsloth: a weight offload_layers spilled to disk at load ({ref.dtype} {list(ref.shape)}) was rewritten "
        f"before training started (now {data.dtype} {list(data.shape)}); load without offload_layers, or with "
        "enough host RAM, to change it."
    )


class SpillFile:
    """An unlinked file on local disk, appended region by region with O_DIRECT, read back with O_DIRECT."""

    def __init__(self, need_bytes, directory = None):
        directory = directory or pick_spill_dir(need_bytes)
        path = os.path.join(directory, f".unsloth_offload_{os.getpid()}_{id(self):x}.bin")
        self.directory = directory
        self.path = path
        self.direct = True
        try:
            self.fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_DIRECT, 0o600)
        except OSError:
            # A filesystem without O_DIRECT: buffered, with the page cache dropped behind every read.
            self.fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            self.direct = False
        # Unlinked at once: the space comes back when the process exits, crash included.
        try:
            os.unlink(path)
        except OSError:
            pass
        self.size = 0
        self.written = 0
        self.lock = threading.Lock()
        self._bounce = None

    def _bounce_buf(self):
        if self._bounce is None:
            self._bounce = _aligned_empty(_WRITE_CHUNK, pin = torch.cuda.is_available())
        return self._bounce[1]

    def write_region(self, pieces, region_bytes, at = None):
        """Append one region of `region_bytes`: `pieces` [(tensor, offset in region)] laid back to back, the gaps
        zeroed. Returns the region's file offset (ALIGN aligned). `at` rewrites a region already in the file
        there instead (the caller keeps it no longer than the region it replaces)."""
        total = _round_up(region_bytes)
        with self.lock:
            if at is None:
                base = self.size
                self.size = base + total
            else:
                base = int(at)
                if base % ALIGN or base + total > self.size:
                    raise ValueError("Unsloth: a spill file rewrite must stay inside an aligned region already written.")
        flat = []
        for t, off in pieces:
            src = t.detach()
            if not src.is_contiguous():
                src = src.contiguous()
            flat.append((src.reshape(-1).view(torch.uint8) if src.numel() else src.reshape(-1).view(torch.uint8),
                         int(off)))
        flat.sort(key = lambda x: x[1])
        buf = self._bounce_buf()
        pos = 0
        while pos < total:
            n = min(_WRITE_CHUNK, total - pos)
            window = buf[:n]
            window.zero_()
            for src, off in flat:
                lo, hi = max(off, pos), min(off + src.numel(), pos + n)
                if lo < hi:
                    window[lo - pos:hi - pos].copy_(src[lo - off:hi - off])
            mv = memoryview(window.numpy())
            done = 0
            while done < n:
                done += os.pwrite(self.fd, mv[done:], base + pos + done)
            pos += n
        if at is None:
            self.written += total
        return base

    def read(self, offset, dst):
        """Fill uint8 `dst` (ALIGN aligned, length a multiple of ALIGN) from file `offset`."""
        mv = memoryview(dst.numpy())
        n, done = dst.numel(), 0
        while done < n:
            got = os.preadv(self.fd, [mv[done:]], offset + done)
            if got <= 0:
                raise OSError(f"Unsloth: short read from the offload spill file at {offset + done}")
            done += got
        if not self.direct:
            try:
                os.posix_fadvise(self.fd, offset, n, os.POSIX_FADV_DONTNEED)
            except (AttributeError, OSError):
                pass

    def close(self):
        fd, self.fd = getattr(self, "fd", None), None
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass

    def __del__(self):
        self.close()


_SPILL = []


def _shared_spill(need_bytes):
    """One spill file per process: layers spilled at load and at install share it."""
    if not _SPILL or _SPILL[0].fd is None:
        _SPILL[:] = [SpillFile(need_bytes)]
    return _SPILL[0]


def spill_params(params, layout, region_bytes, need_bytes = None, store = None):
    """Write `params` (at their `layout` offsets in one region) to the spill file and turn each into a one-element
    placeholder tagged with its DiskRef. The originals' storages go once nothing else holds them."""
    store = store or _shared_spill(need_bytes if need_bytes is not None else region_bytes)
    base = store.write_region([(p.data, off) for p, off in zip(params, layout)], region_bytes)
    with torch.no_grad():
        for p, off in zip(params, layout):
            hold = placeholder_like(p.data)
            ref = DiskRef(store, base + off, p.data.nbytes, p.data.shape, p.data.dtype, hold)
            p.data = hold
            p._unsloth_disk = ref
    return store, base


def _qd():
    try:
        return max(1, int(os.environ.get(_QD_ENV, "4")))
    except ValueError:
        return 4


def _chunk():
    try:
        return max(1, int(os.environ.get(_CHUNK_ENV, "64"))) << 20
    except ValueError:
        return 64 << 20


class DiskTier:
    """Pinned staging ring + reader threads for the blocks BlockSwap serves from disk.

    submit(block) starts its read once a staging slot is free; pump() enqueues the card copy of every finished
    read; complete(block) blocks the host until that block's copy is enqueued (the compute stream then waits on
    the block's events as for a host-RAM block)."""

    def __init__(self, slot_bytes, slots = 2):
        self.slot_bytes = _round_up(slot_bytes)
        self.ring = [_aligned_empty(self.slot_bytes, pin = True) for _ in range(max(1, slots))]
        self.pinned = all(r[0].pinned for r in self.ring)
        self.free = deque(range(len(self.ring)))
        self.slot_done = [None] * len(self.ring)   # last card copy out of each staging slot
        self.waiting = deque()
        self.inflight = deque()                    # blocks being read (or read, card copy not enqueued yet)
        self.orphans = []                          # (staging slot, futures) of reads whose block was evicted
        self.pool = ThreadPoolExecutor(max_workers = _qd(), thread_name_prefix = "unsloth_disk")
        self.chunk = _chunk()
        self.reads = 0
        self.read_bytes = 0
        self.wait_ms = 0.0
        self.read_ms = 0.0
        self._lock = threading.Lock()

    def host_bytes(self):
        return sum(r[1].numel() for r in self.ring)

    def _read_run(self, store, offset, dst):
        t0 = time.perf_counter()
        store.read(offset, dst)
        dt = (time.perf_counter() - t0) * 1000.0
        with self._lock:
            self.read_ms += dt
            self.read_bytes += dst.numel()

    def _start(self, block, sid):
        block.stage = sid
        stage = self.ring[sid][1]
        futs = []
        for file_off, length, stage_off in block.disk_runs:
            pos = 0
            while pos < length:
                n = min(self.chunk, length - pos)
                futs.append(self.pool.submit(self._read_run, block.disk_store, file_off + pos,
                                             stage[stage_off + pos:stage_off + pos + n]))
                pos += n
        block.disk_futs = futs
        self.reads += 1
        self.inflight.append(block)

    def _take_slot(self, block_until = False):
        for sid in list(self.free):
            ev = self.slot_done[sid]
            if ev is None or ev.query():
                self.free.remove(sid)
                return sid
        if block_until and self.free:
            sid = self.free.popleft()
            self.slot_done[sid].synchronize()
            return sid
        return None

    def _reap_orphans(self, block_until = False):
        keep = []
        for sid, futs in self.orphans:
            if block_until or all(f.done() for f in futs):
                for f in futs:
                    f.result()
                self.free.append(sid)
            else:
                keep.append((sid, futs))
        self.orphans = keep

    def submit(self, block):
        sid = self._take_slot()
        if sid is None:
            self.waiting.append(block)
        else:
            self._start(block, sid)

    def _copy_to_card(self, block):
        """Enqueue the staging -> slot copy of a block whose read finished; frees its staging slot after it."""
        stage = self.ring[block.stage][1]
        bufs, _ = block.slot
        codecs = getattr(block, "codec", None) or {}
        for d, event in block.events.items():
            stream = block.streams[d]
            stream.wait_event(block.disk_ready[d])
            # The slot may still be written by a decode into it from a block stolen or reset mid-step.
            fences = getattr(block, "fences", None)
            ev = fences.pop(bufs[d].data_ptr(), None) if fences else None
            if ev is not None:
                stream.wait_event(ev)
            c = codecs.get(d)
            with torch.cuda.stream(stream):
                if block.t_copy is not None:
                    block.t_copy[d].record(stream)
                if c is None:
                    n = block.sizes[d]
                    bufs[d][:n].copy_(stage[:n], non_blocking = True)
                    event.record(stream)
            if c is not None:
                # Stored encoded (lossless): the encoded bytes go over the bus and decode into the slot on the card.
                block.decoders[d].fetch(c, stage[:c.nbytes], bufs[d], stream, event)
        done = torch.cuda.Event()
        done.record(block.streams[next(iter(block.events))])
        self.slot_done[block.stage] = done
        self.free.append(block.stage)
        block.stage = None
        block.disk_futs = None
        block.copy_open = block.t_copy is not None
        block.disk_state = 0

    def pump(self):
        if self.orphans:
            self._reap_orphans()
        while self.inflight and all(f.done() for f in self.inflight[0].disk_futs):
            b = self.inflight.popleft()
            for f in b.disk_futs:
                f.result()
            self._copy_to_card(b)
        while self.waiting:
            sid = self._take_slot()
            if sid is None:
                break
            self._start(self.waiting.popleft(), sid)

    def complete(self, block):
        if not block.disk_state:
            return
        t0 = time.perf_counter()
        while block in self.waiting:
            # No staging slot: finish the reads ahead of it, needed first anyway.
            if self.inflight:
                self._finish(self.inflight[0])
            elif self.orphans:
                self._reap_orphans(block_until = True)
            sid = self._take_slot(block_until = True)
            if sid is not None:
                if self.waiting[0] is block:
                    self.waiting.popleft()
                    self._start(block, sid)
                else:
                    self._start(self.waiting.popleft(), sid)
        while block.disk_state:
            self._finish(self.inflight[0])
        self.wait_ms += (time.perf_counter() - t0) * 1000.0

    def _finish(self, b):
        for f in b.disk_futs:
            f.result()
        self.inflight.remove(b)
        self._copy_to_card(b)

    def cancel(self, block):
        """The block is evicted before its read reached the card: drop it, keep the slot until the read lands."""
        if not block.disk_state:
            return
        if block in self.waiting:
            self.waiting.remove(block)
        elif block in self.inflight:
            self.inflight.remove(block)
            self.orphans.append((block.stage, block.disk_futs))
        block.stage = None
        block.disk_futs = None
        block.disk_state = 0

    def drain(self):
        """Every read done and every card copy enqueued (callers synchronize the copy streams after)."""
        while self.waiting or self.inflight:
            if self.inflight:
                self._finish(self.inflight[0])
            else:
                self.complete(self.waiting[0])
        self._reap_orphans(block_until = True)

    def stats(self):
        return {"disk_reads": self.reads, "disk_read_bytes": self.read_bytes, "disk_read_ms": self.read_ms,
                "disk_wait_ms": self.wait_ms}

    def close(self):
        try:
            self.drain()
        except Exception:
            pass
        self.pool.shutdown(wait = True)
        self.ring = []


def read_bytes(store, offset, nbytes):
    """`nbytes` at any file `offset`, read synchronously into fresh host memory (state_dict, restore, relayout):
    (uint8 view, owner to keep alive while the view is used)."""
    lo = offset - offset % ALIGN
    hi = _round_up(offset + nbytes)
    owner, buf = _aligned_empty(max(hi - lo, ALIGN), pin = False)
    store.read(lo, buf[:hi - lo] if hi > lo else buf)
    return buf[offset - lo:offset - lo + nbytes], owner
