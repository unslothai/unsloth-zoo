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

"""Disk-backed frozen embedding tables (Qwen4Exp n-gram PLE table, DeepSeek-V4.1 engram).

The table is never concatenated or copied into anonymous memory. Each checkpoint tensor that
holds a row range of the table is memory mapped read-only straight from its safetensors file,
advised ``MADV_RANDOM`` (a 320 B row must not fault a 128 KiB readahead window), and gathered
per step with ``torch.index_select`` on the mapped view (ATen faults pages from many threads,
which gives NVMe queue depth without per-row Python). Only the unique rows of a batch are read,
moved to the ids device (pinned + ``non_blocking`` when CUDA), decoded there, then expanded with
the inverse index.

Modes (``UNSLOTH_FROZEN_TABLE``):
    auto  gpu if the table is under 5% of free VRAM, ram if under 50% of available host RAM, else disk
    disk  the mapping above; page cache is reclaimable so there is no anonymous RSS
    ram   the same mapping plus a sequential ``MADV_WILLNEED`` warm-up
    gpu   the planes are materialized on the device and gathered there
Reader (``UNSLOTH_FROZEN_TABLE_READER``): ``mmap`` (default) or ``pread`` (thread pool of
``os.preadv`` over coalesced row runs, for filesystems where mmap faults are slow).
"""

from __future__ import annotations

__all__ = [
    "FrozenTableStore",
    "UnslothFrozenTableEmbedding",
    "FROZEN_TABLE_STATS",
    "read_safetensors_header",
    "resolve_tensor_locations",
    "resolve_frozen_table_mode",
    "frozen_table_next_step",
    "decode_bf16_rows",
    "make_fp8_block_decode",
    "reset_frozen_table_stats",
    "bind_frozen_tables",
    "install_frozen_table_load_hook",
    "iter_frozen_table_modules",
]

import json
import mmap
import os
import struct
import threading
import warnings
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

FROZEN_TABLE_STATS = {
    "gathers": 0,
    "rows": 0,
    "unique_rows": 0,
    "memo_hits": 0,
    "bytes_read_est": 0,
}
_STATS_LOCK = threading.Lock()
_STEP = [0]


def reset_frozen_table_stats():
    with _STATS_LOCK:
        for k in FROZEN_TABLE_STATS:
            FROZEN_TABLE_STATS[k] = 0


def frozen_table_next_step() -> int:
    """Invalidate every module memo (call once per optimizer step if ids tensors are reused in place)."""
    _STEP[0] += 1
    return _STEP[0]


def _bump(**kw):
    with _STATS_LOCK:
        for k, v in kw.items():
            FROZEN_TABLE_STATS[k] += int(v)


# ----------------------------------------------------------------------------------------------
# safetensors headers
# ----------------------------------------------------------------------------------------------
_ST_DTYPES = {
    "BOOL": (torch.bool, 1),
    "U8": (torch.uint8, 1),
    "I8": (torch.int8, 1),
    "I16": (torch.int16, 2),
    "I32": (torch.int32, 4),
    "I64": (torch.int64, 8),
    "F16": (torch.float16, 2),
    "BF16": (torch.bfloat16, 2),
    "F32": (torch.float32, 4),
    "F64": (torch.float64, 8),
}
if hasattr(torch, "float8_e4m3fn"):
    _ST_DTYPES["F8_E4M3"] = (torch.float8_e4m3fn, 1)
if hasattr(torch, "float8_e5m2"):
    _ST_DTYPES["F8_E5M2"] = (torch.float8_e5m2, 1)
if hasattr(torch, "float8_e8m0fnu"):
    _ST_DTYPES["F8_E8M0"] = (torch.float8_e8m0fnu, 1)


def read_safetensors_header(path: str) -> Tuple[dict, int]:
    """Return (header dict, absolute byte offset of the data section)."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"Unsloth: {path} is not a safetensors file (short header).")
        (n,) = struct.unpack("<Q", raw)
        if n > 100 * 1024 * 1024:
            raise ValueError(f"Unsloth: {path} has an implausible safetensors header length {n}.")
        header = json.loads(f.read(n))
    return header, 8 + n


@dataclass(frozen=True)
class TensorLocation:
    path: str
    name: str
    dtype: str  # safetensors dtype string
    shape: Tuple[int, ...]
    offset: int  # absolute byte offset in the file
    nbytes: int


def _checkpoint_files(source: Union[str, Sequence[str]]) -> Tuple[Optional[dict], List[str]]:
    """``source`` = an index json, a directory, a single .safetensors file, or a list of files.
    Returns (weight_map or None, files)."""
    if isinstance(source, (list, tuple)):
        files = [str(x) for x in source]
        return None, [f for f in files if f.endswith(".safetensors")]
    source = str(source)
    if os.path.isdir(source):
        index = os.path.join(source, "model.safetensors.index.json")
        if os.path.exists(index):
            source = index
        elif os.path.exists(os.path.join(source, "model.safetensors")):
            return None, [os.path.join(source, "model.safetensors")]
        else:
            files = sorted(
                os.path.join(source, f) for f in os.listdir(source) if f.endswith(".safetensors")
            )
            if not files:
                raise FileNotFoundError(f"Unsloth: no safetensors files in {source}.")
            return None, files
    if source.endswith(".json"):
        with open(source, "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        base = os.path.dirname(source)
        weight_map = {k: os.path.join(base, v) for k, v in weight_map.items()}
        return weight_map, sorted(set(weight_map.values()))
    return None, [source]


def resolve_tensor_locations(
    source: Union[str, Sequence[str]],
    names: Optional[Iterable[str]] = None,
    predicate: Optional[Callable[[str], bool]] = None,
) -> Dict[str, TensorLocation]:
    """Find tensors by exact ``names`` or ``predicate(name)`` across a checkpoint, reading headers only."""
    weight_map, files = _checkpoint_files(source)
    wanted = set(names) if names is not None else None
    if weight_map is not None:
        files = sorted({
            weight_map[k] for k in weight_map
            if (wanted is not None and k in wanted) or (predicate is not None and predicate(k))
        })
    out = {}
    for path in files:
        header, data_start = read_safetensors_header(path)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if not ((wanted is not None and name in wanted) or (predicate is not None and predicate(name))):
                continue
            begin, end = meta["data_offsets"]
            out[name] = TensorLocation(
                path=os.path.realpath(path),
                name=name,
                dtype=meta["dtype"],
                shape=tuple(int(x) for x in meta["shape"]),
                offset=data_start + int(begin),
                nbytes=int(end) - int(begin),
            )
    if wanted is not None:
        missing = wanted - set(out)
        if missing:
            raise KeyError(f"Unsloth: frozen table tensors not found in checkpoint: {sorted(missing)[:4]}")
    return out


# ----------------------------------------------------------------------------------------------
# Mapped files (shared across tables, closed when the last table using them dies)
# ----------------------------------------------------------------------------------------------
_PAGE = mmap.PAGESIZE
_FILES_LOCK = threading.Lock()
_FILES: Dict[str, "_MappedFile"] = {}


def _madvise(mm: mmap.mmap, start: int, length: int, advice: int) -> bool:
    """madvise(2) on a page-aligned range; no-op where unsupported. Test hook."""
    fn = getattr(mm, "madvise", None)
    if fn is None or advice is None:
        return False
    try:
        fn(advice, start, length)
        return True
    except (OSError, ValueError):
        return False


def _aligned_range(offset: int, nbytes: int, file_size: int) -> Tuple[int, int]:
    start = (offset // _PAGE) * _PAGE
    end = min(((offset + nbytes + _PAGE - 1) // _PAGE) * _PAGE, file_size)
    return start, end - start


class _MappedFile:
    def __init__(self, path: str):
        self.path = path
        self.fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        self.size = os.fstat(self.fd).st_size
        self.mm = mmap.mmap(self.fd, self.size, access=mmap.ACCESS_READ) if self.size else None
        self.refs = 0
        self._whole = None

    def whole(self) -> torch.Tensor:
        if self._whole is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)  # read-only buffer: we never write to it
                self._whole = torch.frombuffer(self.mm, dtype=torch.uint8)
        return self._whole

    def close(self):
        self._whole = None
        if self.mm is not None:
            try:
                self.mm.close()
            except BufferError:
                pass  # a view is still alive somewhere; the GC unmaps it
            self.mm = None
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None


def _acquire_file(path: str) -> _MappedFile:
    with _FILES_LOCK:
        f = _FILES.get(path)
        if f is None or f.mm is None:
            f = _FILES[path] = _MappedFile(path)
        f.refs += 1
        return f


def _release_files(paths: List[str]):
    with _FILES_LOCK:
        for p in paths:
            f = _FILES.get(p)
            if f is None:
                continue
            f.refs -= 1
            if f.refs <= 0:
                _FILES.pop(p, None)
                f.close()


# ----------------------------------------------------------------------------------------------
# Decoders
# ----------------------------------------------------------------------------------------------
def decode_bf16_rows(planes: Dict[str, torch.Tensor], dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Single plane of bf16 rows (Qwen4Exp n-gram table)."""
    (raw,) = planes.values()
    rows = raw.view(torch.bfloat16)
    return rows if dtype is None or dtype == rows.dtype else rows.to(dtype)


def make_fp8_block_decode(block_size: int = 32, weight_plane: str = "weight", scale_plane: str = "scale"):
    """fp8 e4m3 rows times per-block e8m0 scales, exactly DeepseekV41EngramEmbedding.forward:
    ``values.float().unflatten(-1, (-1, block)) * scales.float().unsqueeze(-1)`` (float32 result)."""

    def decode(planes: Dict[str, torch.Tensor], dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        values = planes[weight_plane].view(torch.float8_e4m3fn)
        scales = planes[scale_plane].view(torch.float8_e8m0fnu).float()
        out = values.float().unflatten(-1, (-1, block_size)) * scales.unsqueeze(-1)
        out = out.flatten(-2)
        return out if dtype is None or dtype == out.dtype else out.to(dtype)

    decode.block_size = block_size
    return decode


# ----------------------------------------------------------------------------------------------
# Mode
# ----------------------------------------------------------------------------------------------
_MODES = ("auto", "disk", "ram", "gpu")


def _available_host_bytes() -> int:
    # psutil capped by an enforcing cgroup limit (a container's RAM, not the host's); works on Windows and macOS.
    try:
        from unsloth_zoo.memory_budget import host_memory_bytes
        return int(host_memory_bytes()[1])
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return 0


def resolve_frozen_table_mode(table_bytes: int, device: Optional[torch.device] = None, mode: Optional[str] = None) -> str:
    mode = (mode or os.environ.get("UNSLOTH_FROZEN_TABLE", "auto")).strip().lower()
    if mode not in _MODES:
        raise ValueError(f"Unsloth: UNSLOTH_FROZEN_TABLE must be one of {_MODES}, got {mode!r}.")
    if mode != "auto":
        return mode
    try:
        if torch.cuda.is_available():
            dev = device if device is not None and getattr(device, "type", None) == "cuda" else None
            free, _ = torch.cuda.mem_get_info(dev)
            if table_bytes < 0.05 * free:
                return "gpu"
    except Exception:
        pass
    if table_bytes < 0.5 * _available_host_bytes():
        return "ram"
    return "disk"


# ----------------------------------------------------------------------------------------------
# Store
# ----------------------------------------------------------------------------------------------
@dataclass
class _Shard:
    file: _MappedFile
    offset: int
    rows: int
    row_start: int
    view: Optional[torch.Tensor]  # uint8 [rows, row_bytes] zero-copy view of the mapping


class _Plane:
    def __init__(self, name: str, locations: Sequence[TensorLocation]):
        if not locations:
            raise ValueError(f"Unsloth: frozen table plane {name!r} has no tensors.")
        self.name = name
        self.st_dtype = locations[0].dtype
        cols = locations[0].shape[1:]
        for loc in locations:
            if loc.dtype != self.st_dtype or loc.shape[1:] != cols:
                raise ValueError(
                    f"Unsloth: frozen table plane {name!r} mixes dtypes/shapes "
                    f"({loc.name}: {loc.dtype} {loc.shape} vs {self.st_dtype} {cols})."
                )
            if loc.shape[0] and loc.nbytes % loc.shape[0]:
                raise ValueError(f"Unsloth: {loc.name} size is not a whole number of rows.")
        self.cols = cols
        self.row_bytes = locations[0].nbytes // max(locations[0].shape[0], 1)
        self.locations = list(locations)
        self.shards: List[_Shard] = []
        self.num_rows = sum(loc.shape[0] for loc in locations)
        self.starts = None


class FrozenTableStore:
    """Read-only row table made of one or more planes (same row count), each a list of row shards
    in safetensors files. ``gather(ids)`` returns decoded rows on ``ids.device``."""

    def __init__(
        self,
        planes: Dict[str, Sequence[TensorLocation]],
        decode: Optional[Callable] = None,
        mode: Optional[str] = None,
        reader: Optional[str] = None,
        device: Optional[torch.device] = None,
        pread_threads: Optional[int] = None,
        name: str = "table",
    ):
        self.name = name
        self.planes = {k: _Plane(k, v) for k, v in planes.items()}
        rows = {p.num_rows for p in self.planes.values()}
        if len(rows) != 1:
            raise ValueError(f"Unsloth: frozen table planes disagree on row count: { {k: p.num_rows for k, p in self.planes.items()} }")
        self.num_rows = rows.pop()
        self.row_bytes = sum(p.row_bytes for p in self.planes.values())
        self.nbytes = self.num_rows * self.row_bytes
        self.decode = decode or decode_bf16_rows
        self.mode = resolve_frozen_table_mode(self.nbytes, device, mode)
        self.reader = (reader or os.environ.get("UNSLOTH_FROZEN_TABLE_READER", "mmap")).strip().lower()
        if self.reader not in ("mmap", "pread"):
            raise ValueError(f"Unsloth: UNSLOTH_FROZEN_TABLE_READER must be mmap or pread, got {self.reader!r}.")
        if self.reader == "pread" and not hasattr(os, "pread"):
            self.reader = "mmap"
        self.pread_threads = pread_threads or int(os.environ.get("UNSLOTH_FROZEN_TABLE_THREADS", "0")) or min(32, (os.cpu_count() or 4) * 2)
        self._pool = None
        self._staging: Dict[Tuple[str, torch.device], torch.Tensor] = {}
        self._staging_event = None
        self._device_planes: Dict[str, torch.Tensor] = {}
        self.madvise_calls: List[Tuple[str, int, int, int]] = []

        paths = []
        try:
            for plane in self.planes.values():
                row = 0
                starts = []
                for loc in plane.locations:
                    f = _acquire_file(loc.path)
                    paths.append(loc.path)
                    if loc.offset + loc.nbytes > f.size:
                        raise ValueError(f"Unsloth: {loc.name} extends past the end of {loc.path}.")
                    view = f.whole()[loc.offset: loc.offset + loc.nbytes].view(loc.shape[0], plane.row_bytes) if loc.shape[0] else None
                    plane.shards.append(_Shard(f, loc.offset, loc.shape[0], row, view))
                    starts.append(row)
                    row += loc.shape[0]
                plane.starts = torch.tensor(starts, dtype=torch.int64)
        except Exception:
            _release_files(paths)
            raise
        self._paths = paths
        self._finalizer = weakref.finalize(self, _release_files, list(paths))
        self._advise()
        if self.mode == "gpu":
            self._materialize(device)

    # -- construction helpers ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        source: Union[str, Sequence[str]],
        plane_tensor_names: Dict[str, Sequence[str]],
        **kwargs,
    ) -> "FrozenTableStore":
        """``plane_tensor_names`` = {plane: [tensor names in row order]}."""
        names = [n for v in plane_tensor_names.values() for n in v]
        locs = resolve_tensor_locations(source, names=names)
        return cls({p: [locs[n] for n in v] for p, v in plane_tensor_names.items()}, **kwargs)

    def close(self):
        for p in self.planes.values():
            for s in p.shards:
                s.view = None
        self._device_planes.clear()
        self._staging.clear()
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
        self._finalizer()

    def __deepcopy__(self, memo):
        return self  # read-only, shared

    def __reduce__(self):
        raise TypeError("Unsloth: FrozenTableStore maps checkpoint files and cannot be pickled; rebind after load.")

    def __repr__(self):
        planes = ", ".join(f"{k}:{p.st_dtype}x{p.row_bytes}B/{len(p.shards)} shards" for k, p in self.planes.items())
        return f"FrozenTableStore({self.name}, rows={self.num_rows}, {planes}, mode={self.mode}, reader={self.reader})"

    # -- paging advice -------------------------------------------------------------------------
    def _advise(self):
        rand = getattr(mmap, "MADV_RANDOM", None)
        willneed = getattr(mmap, "MADV_WILLNEED", None)
        for plane in self.planes.values():
            for s in plane.shards:
                if s.rows == 0 or s.file.mm is None:
                    continue
                start, length = _aligned_range(s.offset, s.rows * plane.row_bytes, s.file.size)
                if self.mode in ("disk", "ram") and _madvise(s.file.mm, start, length, rand):
                    self.madvise_calls.append((s.file.path, start, length, rand))
                if self.mode == "ram":
                    chunk = 256 * 1024 * 1024
                    for off in range(start, start + length, chunk):
                        n = min(chunk, start + length - off)
                        if _madvise(s.file.mm, off, n, willneed):
                            self.madvise_calls.append((s.file.path, off, n, willneed))

    def _materialize(self, device):
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        for name, plane in self.planes.items():
            out = torch.empty((self.num_rows, plane.row_bytes), dtype=torch.uint8, device=device)
            for s in plane.shards:
                if s.rows:
                    out[s.row_start: s.row_start + s.rows].copy_(s.view)
            self._device_planes[name] = out

    # -- gather --------------------------------------------------------------------------------
    def _staging_buffer(self, plane: _Plane, rows: int, pin: bool) -> torch.Tensor:
        if not pin:
            return torch.empty((rows, plane.row_bytes), dtype=torch.uint8)
        if self._staging_event is not None:
            self._staging_event.synchronize()  # previous H2D out of this buffer has finished
        key = (plane.name, pin)
        buf = self._staging.get(key)
        if buf is None or buf.shape[0] < rows:
            buf = torch.empty((max(rows, 1) * 5 // 4 + 1, plane.row_bytes), dtype=torch.uint8, pin_memory=True)
            self._staging[key] = buf
        return buf[:rows]

    def _segments(self, plane: _Plane, uniq: torch.Tensor):
        """Split sorted unique ids into (shard, local ids, out slice) without Python per-row work."""
        cuts = torch.searchsorted(uniq, plane.starts).tolist() + [uniq.numel()]
        for i, s in enumerate(plane.shards):
            a, b = cuts[i], cuts[i + 1]
            if b > a:
                yield s, uniq[a:b] - s.row_start, a, b

    def _read_mmap(self, plane: _Plane, uniq: torch.Tensor, out: torch.Tensor):
        for s, local, a, b in self._segments(plane, uniq):
            torch.index_select(s.view, 0, local, out=out[a:b])

    def _read_pread(self, plane: _Plane, uniq: torch.Tensor, out: torch.Tensor):
        rb = plane.row_bytes
        out_np = out.numpy()
        jobs = []
        for s, local, a, b in self._segments(plane, uniq):
            ids = local.tolist()
            # coalesce consecutive rows into one read
            run_start = 0
            for j in range(1, len(ids) + 1):
                if j == len(ids) or ids[j] != ids[j - 1] + 1:
                    jobs.append((s.file.fd, s.offset + ids[run_start] * rb, a + run_start, a + j))
                    run_start = j
        if not jobs:
            return

        def work(chunk):
            preadv = getattr(os, "preadv", None)
            for fd, off, lo, hi in chunk:
                view = memoryview(out_np[lo:hi]).cast("B")
                want = len(view)
                done = 0
                while done < want:
                    if preadv is not None:
                        n = preadv(fd, [view[done:]], off + done)
                    else:
                        data = os.pread(fd, want - done, off + done)
                        n = len(data)
                        view[done: done + n] = data
                    if n <= 0:
                        raise OSError(f"Unsloth: short read from frozen table file at offset {off + done}.")
                    done += n

        n = min(self.pread_threads, len(jobs))
        if n <= 1:
            work(jobs)
            return
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.pread_threads, thread_name_prefix="unsloth_frozen_table")
        step = (len(jobs) + n - 1) // n
        list(self._pool.map(work, [jobs[i: i + step] for i in range(0, len(jobs), step)]))

    def gather_raw_unique(self, uniq_cpu: torch.Tensor, device: torch.device) -> Dict[str, torch.Tensor]:
        """Raw uint8 rows for sorted unique int64 CPU ids, on ``device``."""
        pin = device.type == "cuda" and torch.cuda.is_available()
        out = {}
        for name, plane in self.planes.items():
            if name in self._device_planes:
                dev_plane = self._device_planes[name]
                out[name] = dev_plane.index_select(0, uniq_cpu.to(dev_plane.device, non_blocking=True)).to(device)
                continue
            buf = self._staging_buffer(plane, uniq_cpu.numel(), pin)
            if self.reader == "pread":
                self._read_pread(plane, uniq_cpu, buf)
            else:
                self._read_mmap(plane, uniq_cpu, buf)
            out[name] = buf.to(device, non_blocking=True) if pin else (buf if device.type == "cpu" else buf.to(device))
        if pin:
            ev = torch.cuda.Event()
            ev.record()
            self._staging_event = ev
        return out

    def gather(self, ids: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Rows for ``ids`` (any shape, integer) on ``ids.device``: shape ``ids.shape + row shape``."""
        if ids.is_floating_point() or ids.is_complex():
            raise TypeError("Unsloth: frozen table ids must be integers.")
        device = ids.device
        flat = ids.reshape(-1).long()
        if flat.numel() == 0:
            dec = self.decode({k: torch.empty((0, p.row_bytes), dtype=torch.uint8, device=device) for k, p in self.planes.items()}, dtype)
            return dec.reshape(*ids.shape, *dec.shape[1:])
        uniq, inv = torch.unique(flat, sorted=True, return_inverse=True)
        uniq_cpu = uniq.cpu() if uniq.device.type != "cpu" else uniq
        lo, hi = int(uniq_cpu[0]), int(uniq_cpu[-1])
        if lo < 0 or hi >= self.num_rows:
            raise IndexError(f"Unsloth: frozen table id out of range [{lo}, {hi}] for {self.num_rows} rows.")
        raw = self.gather_raw_unique(uniq_cpu, device)
        rows = self.decode(raw, dtype)
        out = rows.index_select(0, inv.to(rows.device))
        _bump(gathers=1, rows=flat.numel(), unique_rows=uniq_cpu.numel(),
              bytes_read_est=0 if self._device_planes else uniq_cpu.numel() * self.row_bytes)
        return out.reshape(*ids.shape, *rows.shape[1:])


# ----------------------------------------------------------------------------------------------
# Module
# ----------------------------------------------------------------------------------------------
class UnslothFrozenTableEmbedding(nn.Module):
    """Parameter-free stand-in for a frozen ``nn.Embedding``: the table lives in a
    :class:`FrozenTableStore` kept as a plain attribute, so ``state_dict``, ``.to()``,
    ``named_parameters``, accelerate hooks, FSDP/DDP and optimizers never see it."""

    def __init__(self, num_embeddings: int, embedding_dim: int, dtype: Optional[torch.dtype] = None,
                 table_name: str = "", row_shape: Optional[Tuple[int, ...]] = None):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.row_shape = tuple(row_shape) if row_shape is not None else (self.embedding_dim,)
        self.table_dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.table_name = table_name
        self.frozen_table_store: Optional[FrozenTableStore] = None
        self._memo = None

    # Upstream forwards read ``self.<table>.weight.device`` to pick where to run the lookup; a
    # meta tensor tells them "not placed", so they leave the ids where they are.
    @property
    def weight(self) -> torch.Tensor:
        return torch.empty((self.num_embeddings, *self.row_shape), dtype=self.table_dtype, device="meta")

    @property
    def is_bound(self) -> bool:
        return self.frozen_table_store is not None

    def bind(self, store: FrozenTableStore):
        if store.num_rows != self.num_embeddings:
            raise ValueError(
                f"Unsloth: frozen table {self.table_name or ''} expects {self.num_embeddings} rows, "
                f"the checkpoint has {store.num_rows}."
            )
        self.frozen_table_store = store
        self._memo = None
        return self

    def _apply(self, fn, recurse=True):
        # Track dtype casts (model.to(torch.float32), .half()) without owning any tensor.
        try:
            probe = fn(torch.empty(0, dtype=self.table_dtype))
            if isinstance(probe, torch.Tensor) and probe.is_floating_point():
                if probe.dtype != self.table_dtype:
                    self._memo = None
                self.table_dtype = probe.dtype
        except Exception:
            pass
        return super()._apply(fn, recurse)

    def extra_repr(self):
        return f"{self.num_embeddings}, {self.embedding_dim}, dtype={self.table_dtype}, store={self.frozen_table_store!r}"

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        store = self.frozen_table_store
        if store is None:
            raise RuntimeError(
                f"Unsloth: frozen table {self.table_name or ''} is not bound to its checkpoint. Call "
                "unsloth_zoo.frozen_table_store.bind_frozen_tables(model, checkpoint) after loading, "
                "or set UNSLOTH_FROZEN_TABLE=gpu to load it as a normal embedding."
            )
        memo = self._memo
        step = _STEP[0]
        if memo is not None:
            m_ids, m_key, m_rows = memo
            key = (ids.data_ptr(), ids._version, tuple(ids.shape), tuple(ids.stride()), ids.device, step, self.table_dtype)
            if m_key == key or (
                m_key[2] == key[2] and m_key[4] == key[4] and m_key[5] == step and m_key[6] == key[6]
                and m_ids.dtype == ids.dtype and torch.equal(m_ids, ids)
            ):
                _bump(memo_hits=1)
                return m_rows
        with torch.no_grad():
            rows = store.gather(ids, dtype=self.table_dtype)
        key = (ids.data_ptr(), ids._version, tuple(ids.shape), tuple(ids.stride()), ids.device, step, self.table_dtype)
        # Keep a reference to ids: while held, no other tensor can take its address.
        self._memo = (ids, key, rows)
        return rows


def iter_frozen_table_modules(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, UnslothFrozenTableEmbedding):
            yield name, module


# ----------------------------------------------------------------------------------------------
# Binding to a checkpoint
# ----------------------------------------------------------------------------------------------
# Plane name -> checkpoint key suffix regexes after the table's prefix, tried in order. A regex
# group 1 (if any) is the shard index used for row order.
DEFAULT_CHECKPOINT_PLANES = {"weight": (r"shard_(\d+)\.weight", r"weight")}


def _checkpoint_keys(source) -> Tuple[List[str], Union[str, List[str]]]:
    weight_map, files = _checkpoint_files(source)
    if weight_map is not None:
        return list(weight_map), source
    keys = []
    for path in files:
        header, _ = read_safetensors_header(path)
        keys.extend(k for k in header if k != "__metadata__")
    return keys, files


def _common_suffix(a: List[str], b: List[str]) -> int:
    n = 0
    while n < len(a) and n < len(b) and a[-1 - n] == b[-1 - n]:
        n += 1
    return n


def _match_plane(module_name: str, keys: List[str], alternatives: Sequence[str]) -> List[str]:
    import re
    mod_parts = module_name.split(".")
    # The part from the last "layers.<i>" on must match exactly (never bind layer 1's table to layer 14's).
    need = len(mod_parts)
    for i in range(len(mod_parts) - 1, -1, -1):
        if mod_parts[i] == "layers":
            need = len(mod_parts) - i
            break
    else:
        need = min(len(mod_parts), 2)
    for alt in alternatives:
        rx = re.compile(r"^(?P<prefix>.+?)\.(?:" + alt + r")$")
        groups: Dict[str, List[Tuple[int, str]]] = {}
        for k in keys:
            m = rx.match(k)
            if m is None:
                continue
            prefix = m.group("prefix")
            idx = int(m.group(2)) if m.re.groups >= 2 and m.group(2) is not None else 0
            groups.setdefault(prefix, []).append((idx, k))
        scored = []
        for prefix, items in groups.items():
            score = _common_suffix(prefix.split("."), mod_parts)
            if score >= need:
                scored.append((score, prefix, items))
        if not scored:
            continue
        scored.sort(key=lambda x: -x[0])
        if len(scored) > 1 and scored[0][0] == scored[1][0]:
            raise RuntimeError(
                f"Unsloth: frozen table {module_name} matches several checkpoint prefixes: "
                f"{[s[1] for s in scored[:3]]}."
            )
        items = sorted(scored[0][2])
        idx = [i for i, _ in items]
        if len(items) > 1 and idx != list(range(len(items))):
            raise RuntimeError(f"Unsloth: frozen table {module_name} shards are not contiguous: {idx[:8]}...")
        return [k for _, k in items]
    return []


def bind_frozen_tables(
    model: nn.Module,
    checkpoint: Union[str, Sequence[str]],
    mode: Optional[str] = None,
    reader: Optional[str] = None,
    strict: bool = True,
) -> Dict[str, FrozenTableStore]:
    """Bind every :class:`UnslothFrozenTableEmbedding` in ``model`` to its rows in ``checkpoint``
    (index json, directory, safetensors file or list of files). Reads safetensors headers only.
    Raises if a table cannot be found or does not match (never falls back to loading it into RAM)."""
    modules = list(iter_frozen_table_modules(model))
    if not modules:
        return {}
    keys, source = _checkpoint_keys(checkpoint)
    device = None
    for p in model.parameters():
        if p.device.type not in ("meta", "cpu"):
            device = p.device
            break
    bound = {}
    for name, module in modules:
        spec = getattr(module, "checkpoint_planes", None) or DEFAULT_CHECKPOINT_PLANES
        planes = {}
        for plane, alternatives in spec.items():
            found = _match_plane(name, keys, alternatives)
            if not found:
                if not strict:
                    break
                raise RuntimeError(
                    f"Unsloth: could not find frozen table {name} (plane {plane!r}) in the checkpoint. "
                    "Refusing to load it into memory; set UNSLOTH_FROZEN_TABLE=gpu for a normal load."
                )
            planes[plane] = found
        else:
            store = FrozenTableStore.from_checkpoint(
                source, planes,
                decode=getattr(module, "decode_fn", None),
                mode=mode, reader=reader, device=device, name=name,
            )
            first = next(iter(store.planes.values()))
            row_shape = getattr(module, "row_shape", None)
            dec = store.decode(
                {k: torch.zeros((1, p.row_bytes), dtype=torch.uint8) for k, p in store.planes.items()}, None
            )
            if row_shape is not None and tuple(dec.shape[1:]) != tuple(row_shape):
                store.close()
                raise RuntimeError(
                    f"Unsloth: frozen table {name} rows decode to {tuple(dec.shape[1:])}, expected {tuple(row_shape)} "
                    f"(checkpoint {first.st_dtype} {first.cols})."
                )
            module.bind(store)
            bound[name] = store
    return bound


_LOAD_HOOK_INSTALLED = [False]


def install_frozen_table_load_hook() -> bool:
    """Bind frozen tables right after ``PreTrainedModel._load_pretrained_model`` (checkpoint files are known
    there). Idempotent. A model without frozen tables is untouched."""
    if _LOAD_HOOK_INSTALLED[0]:
        return True
    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception:
        return False
    original = PreTrainedModel.__dict__.get("_load_pretrained_model")
    if original is None:
        return False
    original_fn = original.__func__ if isinstance(original, staticmethod) else original
    if getattr(original_fn, "_unsloth_frozen_table_hook", False):
        _LOAD_HOOK_INSTALLED[0] = True
        return True

    def _load_pretrained_model(model, state_dict, checkpoint_files, load_config, *args, **kwargs):
        result = original_fn(model, state_dict, checkpoint_files, load_config, *args, **kwargs)
        unbound = [n for n, m in iter_frozen_table_modules(model) if not m.is_bound]
        if unbound:
            if not checkpoint_files or not str(checkpoint_files[0]).endswith(".safetensors"):
                raise RuntimeError(
                    f"Unsloth: frozen tables {unbound[:2]} need a safetensors checkpoint to stream from. "
                    "Set UNSLOTH_FROZEN_TABLE=gpu for a normal load."
                )
            bind_frozen_tables(model, list(checkpoint_files))
        return result

    _load_pretrained_model._unsloth_frozen_table_hook = True
    _load_pretrained_model.__wrapped__ = original_fn
    PreTrainedModel._load_pretrained_model = staticmethod(_load_pretrained_model)
    _LOAD_HOOK_INSTALLED[0] = True
    return True
