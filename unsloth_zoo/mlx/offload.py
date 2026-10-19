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

"""Weights served from the checkpoint's safetensors on disk instead of from memory.

Per-layer embedding (PLE) tables -- Qwen4-Exp's 128-shard n-gram table, Gemma's per-layer token
embeddings -- are large, and a call reads a few of their rows. ``offload_ple`` replaces them,
before the load evaluates anything, with tables that read the rows a call asks for straight from
the shards, so the offloaded bytes are never loaded. Nothing is converted or written to disk.
"""

import glob
import json
import os
import struct
import threading
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import mlx.nn as nn
import numpy as np

__all__ = ["CheckpointTensors", "DiskEmbedding", "offload_fraction", "offload_ple", "read_ranges"]

# safetensors dtype -> (numpy transport dtype, mlx dtype)
_DTYPES = {
    "BF16": (np.uint16, mx.bfloat16),
    "F16": (np.float16, mx.float16),
    "F32": (np.float32, mx.float32),
    "U32": (np.uint32, mx.uint32),
    "U8": (np.uint8, mx.uint8),
    "I32": (np.int32, mx.int32),
}

_READ_THREADS = 16
# Below this many reads a pool round trip costs more than it overlaps.
_SERIAL_READS = 8
_pool = None
_pool_lock = threading.Lock()


def _reader_pool():
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(_READ_THREADS, thread_name_prefix = "unsloth-offload-read")
        return _pool


def _read_all(fd, view, offset):
    done = 0
    while done < len(view):
        got = os.preadv(fd, [view[done:]], offset + done)
        if got <= 0:
            raise OSError(f"short read at byte {offset + done}")
        done += got


def read_ranges(jobs):
    """Fill each ``(fd, offset, writable memoryview)``. Positional reads release the GIL, so a batch
    spread over the pool runs at the storage's queue depth instead of one read at a time."""

    def run(part):
        for fd, offset, view in part:
            _read_all(fd, view, offset)

    if len(jobs) <= _SERIAL_READS:
        run(jobs)
        return
    parts = min(_READ_THREADS, len(jobs))
    list(_reader_pool().map(run, [jobs[i::parts] for i in range(parts)]))


class CheckpointTensors:
    """Where each tensor of a checkpoint directory lies, from the safetensors headers alone."""

    def __init__(self, model_dir):
        self.specs = {}
        for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
            path = os.path.realpath(path)
            with open(path, "rb") as stream:
                size = struct.unpack("<Q", stream.read(8))[0]
                header = json.loads(stream.read(size))
            for name, spec in header.items():
                if name != "__metadata__":
                    self.specs[name] = (path, spec["dtype"], tuple(spec["shape"]), 8 + size + spec["data_offsets"][0])
        self._fds = {}
        self._lock = threading.Lock()

    def fd(self, path):
        with self._lock:
            if path not in self._fds:
                self._fds[path] = os.open(path, os.O_RDONLY)
            return self._fds[path]

    def close(self):
        with self._lock:
            fds, self._fds = list(self._fds.values()), {}
        for fd in fds:
            os.close(fd)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def offload_fraction(name, value):
    """``None`` (off) or the fraction to offload in (0, 1]."""
    if value is None or value is False or value == 0 or value == "off":
        return None
    fraction = None
    if not isinstance(value, (bool, str)):
        try:
            fraction = float(value)
        except (TypeError, ValueError):
            pass
    if fraction is None or not 0 < fraction <= 1:
        raise ValueError(
            f"Unsloth: {name} must be 'off' or the fraction to offload in (0, 1], got {value!r}."
        )
    return fraction


class _Field:
    __slots__ = ("name", "fd", "offset", "row_bytes", "transport", "dtype", "width", "cast", "map")

    def __init__(self, name, fd, offset, row_bytes, transport, dtype, width, cast, map):
        self.name, self.fd, self.offset, self.row_bytes = name, fd, offset, row_bytes
        self.transport, self.dtype, self.width, self.cast, self.map = transport, dtype, width, cast, map

    def to_mx(self, buffer):
        array = mx.array(buffer.view(self.transport).reshape(-1, self.width))
        if array.dtype != self.dtype:
            array = array.view(self.dtype)  # bf16 travels as uint16
        return array if self.cast is None else array.astype(self.cast)


class _RowTable:
    """A ``[rows, ...]`` tensor and its quantization siblings, read by row. The first ``resident``
    rows are read once at load and held in memory. A row's first lookup reads it with ``pread``,
    concurrently with the call's other new rows; later lookups gather it from the page cache
    through a read-only map, which costs no system call per row."""

    def __init__(self, store, names, resident, casts):
        self.store = store  # owns the file descriptors
        self.fields = []
        rows = None
        for field, name in names.items():
            path, dtype, shape, offset = store.specs[name]
            if rows is not None and shape[0] != rows:
                raise ValueError(f"{name} has {shape[0]} rows, its table {rows}")
            rows = shape[0]
            transport, mx_dtype = _DTYPES[dtype]
            width = int(np.prod(shape[1:]))
            row_bytes = width * np.dtype(transport).itemsize
            self.fields.append(_Field(
                field, store.fd(path), offset, row_bytes, transport, mx_dtype, width, casts.get(field),
                np.memmap(path, dtype = np.uint8, mode = "r", offset = offset, shape = (rows, row_bytes)),
            ))
        self.rows = rows
        self.seen = np.zeros((rows + 7) // 8, dtype = np.uint8)
        self.resident = min(int(resident), rows)
        self.held = []
        if self.resident:
            buffers = self.buffers(self.resident)
            piece, jobs = 64 << 20, []
            for field, buffer in zip(self.fields, buffers):
                flat = memoryview(buffer.reshape(-1))
                jobs.extend(
                    (field.fd, field.offset + start, flat[start:start + piece])
                    for start in range(0, self.resident * field.row_bytes, piece)
                )
            read_ranges(jobs)
            self.held = [field.to_mx(buffer) for field, buffer in zip(self.fields, buffers)]
            mx.eval(self.held)

    def buffers(self, count):
        return [np.empty((count, field.row_bytes), dtype = np.uint8) for field in self.fields]

    def bytes(self, rows):
        return rows * sum(field.row_bytes for field in self.fields)

    def fill(self, local, buffers, at, jobs):
        """Put rows ``local`` (unique, all past the held prefix) into ``buffers`` at ``at``: rows
        read before are copied now, new ones queued on ``jobs``. Returns the new rows."""
        seen = (self.seen[local >> 3] >> (local & 7).astype(np.uint8)) & 1 == 1
        new, new_at = local[~seen], at[~seen].tolist()
        for field, buffer in zip(self.fields, buffers):
            if seen.any():
                buffer[at[seen]] = field.map[local[seen]]
            for row, position in zip(new.tolist(), new_at):
                jobs.append((field.fd, field.offset + row * field.row_bytes, memoryview(buffer[position])))
        return new

    def mark_seen(self, rows):
        np.bitwise_or.at(self.seen, rows >> 3, np.left_shift(1, rows & 7).astype(np.uint8))


def _by_shard(shard):
    """``(shard, positions)`` per run of equal values in the non-decreasing ``shard``."""
    starts = np.concatenate([[0], np.flatnonzero(np.diff(shard)) + 1, [shard.size]])
    for start, end in zip(starts[:-1].tolist(), starts[1:].tolist()):
        if start < end:
            yield int(shard[start]), np.arange(start, end)


class DiskEmbedding(nn.Module):
    """Stands in for an ``nn.Embedding`` / ``nn.QuantizedEmbedding`` (one table) or a row-sharded
    embedding (its shards in row order) whose rows stay on disk. Each looked-up row goes through the
    same ``dequantize`` as the module it replaces, so the output is that module's, bit for bit."""

    def __init__(self, tables, quantization, dims, normalize_zero):
        super().__init__()
        offsets = np.cumsum([0] + [table.rows for table in tables]).astype(np.int64)
        # Plain attributes: containers or arrays here would register as children or parameters.
        object.__setattr__(self, "_tables", tuple(tables))
        object.__setattr__(self, "_offsets", offsets)
        object.__setattr__(self, "quantization", quantization)
        self.dims = dims
        self.normalize_zero = normalize_zero

    def __call__(self, x):
        ids = np.asarray(x.astype(mx.int64)).reshape(-1)
        if ids.size and (ids.min() < 0 or ids.max() >= self._offsets[-1]):
            raise IndexError("embedding index is outside the table")
        unique, inverse = np.unique(ids, return_inverse = True)
        values = self._lookup(unique)
        if unique.size != ids.size or not np.array_equal(unique, ids):
            values = values[mx.array(inverse.reshape(-1).astype(np.int32))]
        if self.normalize_zero:
            # The stock sharded embedding scatter-adds each shard's rows into zeros, which turns
            # -0.0 into +0.0; adding zeros reproduces that bit for bit.
            values = mx.zeros(values.shape, dtype = values.dtype) + values
        return values.reshape(*x.shape, self.dims)

    def _lookup(self, unique):
        tables, offsets = self._tables, self._offsets
        shard = np.searchsorted(offsets, unique, side = "right") - 1
        local = unique - offsets[shard]
        resident = np.asarray([table.resident for table in tables], dtype = np.int64)[shard]
        on_disk = np.flatnonzero(local >= resident)
        pieces = []  # (positions in ``unique``, one array per field)
        if on_disk.size or not unique.size:  # an empty lookup still yields the right dtype
            buffers = tables[0].buffers(on_disk.size)
            jobs, new = [], []
            for index, chosen in _by_shard(shard[on_disk]):
                new.append((index, tables[index].fill(local[on_disk[chosen]], buffers, chosen, jobs)))
            read_ranges(jobs)
            for index, rows in new:
                tables[index].mark_seen(rows)
            pieces.append((on_disk, [field.to_mx(buffer) for field, buffer in zip(tables[0].fields, buffers)]))
        held = np.flatnonzero(local < resident)
        for index, chosen in _by_shard(shard[held]):
            rows = mx.array(local[held[chosen]].astype(np.int32))
            pieces.append((held[chosen], [column[rows] for column in tables[index].held]))
        positions = np.concatenate([at for at, _ in pieces])
        columns = [
            arrays[0] if len(arrays) == 1 else mx.concatenate(arrays)
            for arrays in zip(*(columns for _, columns in pieces))
        ]
        if np.any(positions[1:] < positions[:-1]):
            order = mx.array(np.argsort(positions, kind = "stable").astype(np.int32))
            columns = [column[order] for column in columns]
        if self.quantization is None:
            return columns[0]
        group_size, bits, mode = self.quantization
        return mx.dequantize(
            columns[0],
            scales = columns[1],
            biases = columns[2] if len(columns) > 2 else None,
            group_size = group_size,
            bits = bits,
            mode = mode,
        )

    def offload_bytes(self):
        """``(on disk, resident)`` bytes of the tables this stands in for."""
        return (
            sum(table.bytes(table.rows - table.resident) for table in self._tables),
            sum(column.nbytes for table in self._tables for column in table.held),
        )


# (attribute name, module types): Gemma 3n / Gemma 4 per-layer token embeddings, and
# Qwen4-Exp's row-sharded n-gram table.
_PLE_SITES = (
    ("embed_tokens_per_layer", ("Embedding", "QuantizedEmbedding")),
    ("ngram_embedding", ("ShardedEmbedding",)),
)


def ple_sites(model):
    """``(path, module)`` of every PLE table in ``model``."""
    for path, module in model.named_modules():
        key = path.rsplit(".", 1)[-1]
        if any(key == name and type(module).__name__ in kinds for name, kinds in _PLE_SITES):
            yield path, module


def _checkpoint_name(store, name):
    """The checkpoint's name for a parameter: its MLX name, or the Hugging Face layout of it."""
    if name in store.specs:
        return name
    prefix = "language_model.model."
    if name.startswith(prefix):
        alternate = "model.language_model." + name[len(prefix):]
        if alternate in store.specs:
            return alternate
    return None


def _quantization_of(module):
    if type(module).__name__ != "QuantizedEmbedding":
        return None
    return (module.group_size, module.bits, module.mode)


def _table_for(store, module, path, fraction, cast_for):
    """A ``_RowTable`` over ``module``'s tensors, or the reason they cannot be read from the checkpoint."""
    fields = ["weight"]
    if _quantization_of(module) is not None:
        fields += ["scales"] + (["biases"] if module.get("biases") is not None else [])
    names, casts = {}, {}
    for field in fields:
        parameter = f"{path}.{field}"
        name = _checkpoint_name(store, parameter)
        if name is None:
            return None, f"the checkpoint has no tensor for {parameter}"
        _, dtype, shape, _ = store.specs[name]
        expected = module[field]
        if dtype not in _DTYPES or shape != tuple(expected.shape) or _DTYPES[dtype][1] != expected.dtype:
            return None, (
                f"{name} is {dtype} {list(shape)} in the checkpoint, "
                f"{expected.dtype} {list(expected.shape)} in the model"
            )
        names[field] = name
        casts[field] = cast_for(parameter, expected.dtype)
    resident = int(module["weight"].shape[0] * (1 - fraction))
    return _RowTable(store, names, resident, casts), None


def offload_ple(model, model_dir, fraction, *, cast_dtype = None):
    """Replace every PLE table of ``model`` with one read from ``model_dir``'s shards, keeping the
    first ``1 - fraction`` of each table's rows in memory. Must run before anything evaluates the
    tables. ``cast_dtype`` is the float dtype the load casts parameters to; offloaded float tensors
    get the cast they would have had as parameters.
    Returns ``{"tables", "on_disk_bytes", "resident_bytes", "skipped": [(path, reason)]}``."""
    cast_predicate = getattr(model, "cast_predicate", lambda _: True)

    def cast_for(parameter, dtype):
        if cast_dtype is None or not mx.issubdtype(dtype, mx.floating) or dtype == cast_dtype:
            return None
        return cast_dtype if cast_predicate(parameter) else None

    store = CheckpointTensors(model_dir)
    report = {"tables": 0, "on_disk_bytes": 0, "resident_bytes": 0, "skipped": []}
    for path, module in list(ple_sites(model)):
        sharded = type(module).__name__ == "ShardedEmbedding"
        if sharded:
            parts = [(f"{path}.shards.{i}", shard) for i, shard in enumerate(module.shards)]
        else:
            parts = [(path, module)]
        quantizations = {_quantization_of(part) for _, part in parts}
        if len(quantizations) != 1:
            report["skipped"].append((path, "its shards are quantized differently"))
            continue
        tables, reason = [], None
        for part_path, part in parts:
            table, reason = _table_for(store, part, part_path, fraction, cast_for)
            if table is None:
                break
            tables.append(table)
        if reason is not None:
            report["skipped"].append((path, reason))
            continue
        quantization = quantizations.pop()
        dims = module.dims if sharded or quantization is not None else module["weight"].shape[-1]
        replacement = DiskEmbedding(tables, quantization, dims, normalize_zero = sharded)
        _replace_module(model, path, replacement)
        on_disk, resident = replacement.offload_bytes()
        report["tables"] += 1
        report["on_disk_bytes"] += on_disk
        report["resident_bytes"] += resident
    return report


def _replace_module(model, path, replacement):
    parent = model
    keys = path.split(".")
    for key in keys[:-1]:
        parent = parent[int(key)] if isinstance(parent, list) else parent[key]
    setattr(parent, keys[-1], replacement)
