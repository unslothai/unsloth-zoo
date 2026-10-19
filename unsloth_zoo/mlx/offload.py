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
the shards. ``offload_experts`` does the same for the routed experts of quantized MoE layers, read
into a fixed number of slots as calls route to them. The offloaded bytes are never loaded, and
nothing is converted or written to disk.
"""

import copy
import glob
import json
import math
import os
import struct
import threading
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import mlx.nn as nn
import numpy as np

__all__ = [
    "CheckpointTensors",
    "DiskEmbedding",
    "OffloadSwitchGLU",
    "offload_experts",
    "offload_fraction",
    "offload_ple",
    "offloadable",
    "read_ranges",
]

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
    """``None`` (off), ``"auto"``, or the fraction to offload in (0, 1]."""
    if value is None or value is False or value == 0 or value == "off":
        return None
    if value is True or value == "auto":
        return "auto"
    fraction = None
    if not isinstance(value, (bool, str)):
        try:
            fraction = float(value)
        except (TypeError, ValueError):
            pass
    if fraction is None or not 0 < fraction <= 1:
        raise ValueError(
            f"Unsloth: {name} must be 'off', 'auto' or the fraction to offload in (0, 1], got {value!r}."
        )
    return fraction


class _Field:
    """One checkpoint tensor, read by its leading index (a table row or an expert)."""

    __slots__ = ("name", "fd", "offset", "row_bytes", "transport", "dtype", "shape", "cast", "map")

    def __init__(self, store, name, tensor, cast, mapped = False):
        path, dtype, shape, offset = store.specs[tensor]
        self.transport, self.dtype = _DTYPES[dtype]
        self.name, self.fd, self.offset, self.cast = name, store.fd(path), offset, cast
        self.shape = tuple(shape[1:])
        self.row_bytes = int(np.prod(self.shape)) * np.dtype(self.transport).itemsize
        self.map = None
        if mapped:
            self.map = np.memmap(path, dtype = np.uint8, mode = "r", offset = offset, shape = (shape[0], self.row_bytes))

    def to_mx(self, buffer):
        array = mx.array(buffer.view(self.transport).reshape(-1, *self.shape))
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
            shape = store.specs[name][2]
            if rows is not None and shape[0] != rows:
                raise ValueError(f"{name} has {shape[0]} rows, its table {rows}")
            rows = shape[0]
            self.fields.append(_Field(store, field, name, casts.get(field), mapped = True))
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


def _checkpoint_tensor(store, parameter, expected):
    """``(name, None)`` when the checkpoint stores ``parameter`` as the model holds it, else ``(None, reason)``."""
    name = _checkpoint_name(store, parameter)
    if name is None:
        return None, f"the checkpoint has no tensor for {parameter}"
    _, dtype, shape, _ = store.specs[name]
    if dtype not in _DTYPES or shape != tuple(expected.shape) or _DTYPES[dtype][1] != expected.dtype:
        return None, (
            f"{name} is {dtype} {list(shape)} in the checkpoint, "
            f"{expected.dtype} {list(expected.shape)} in the model"
        )
    return name, None


def _cast_rule(model, cast_dtype):
    """The cast the load gives a float parameter: ``cast_dtype`` under the model's ``cast_predicate``."""
    cast_predicate = getattr(model, "cast_predicate", lambda _: True)

    def cast_for(parameter, dtype):
        if cast_dtype is None or not mx.issubdtype(dtype, mx.floating) or dtype == cast_dtype:
            return None
        return cast_dtype if cast_predicate(parameter) else None

    return cast_for


def _table_fields(store, module, path, cast_for):
    """The checkpoint names and casts of ``module``'s tensors, or the reason they cannot be read."""
    fields = ["weight"]
    if _quantization_of(module) is not None:
        fields += ["scales"] + (["biases"] if module.get("biases") is not None else [])
    names, casts = {}, {}
    for field in fields:
        parameter = f"{path}.{field}"
        name, reason = _checkpoint_tensor(store, parameter, module[field])
        if reason is not None:
            return None, reason
        names[field] = name
        casts[field] = cast_for(parameter, module[field].dtype)
    return (names, casts), None


def _ple_parts(store, path, module, cast_for):
    """``[(part, (names, casts))]`` and the quantization of a PLE table, or why it stays in memory."""
    if type(module).__name__ == "ShardedEmbedding":
        parts = [(f"{path}.shards.{i}", shard) for i, shard in enumerate(module.shards)]
    else:
        parts = [(path, module)]
    quantizations = {_quantization_of(part) for _, part in parts}
    if len(quantizations) != 1:
        return None, None, "its shards are quantized differently"
    found = []
    for part_path, part in parts:
        fields, reason = _table_fields(store, part, part_path, cast_for)
        if reason is not None:
            return None, None, reason
        found.append((part, fields))
    return found, quantizations.pop(), None


def offload_ple(model, model_dir, fraction, *, cast_dtype = None):
    """Replace every PLE table of ``model`` with one read from ``model_dir``'s shards, keeping the
    first ``1 - fraction`` of each table's rows in memory. Must run before anything evaluates the
    tables. ``cast_dtype`` is the float dtype the load casts parameters to; offloaded float tensors
    get the cast they would have had as parameters.
    Returns ``{"tables", "on_disk_bytes", "resident_bytes", "skipped": [(path, reason)]}``."""
    cast_for = _cast_rule(model, cast_dtype)
    store = CheckpointTensors(model_dir)
    report = {"tables": 0, "on_disk_bytes": 0, "resident_bytes": 0, "skipped": []}
    for path, module in list(ple_sites(model)):
        parts, quantization, reason = _ple_parts(store, path, module, cast_for)
        if reason is not None:
            report["skipped"].append((path, reason))
            continue
        tables = [
            _RowTable(store, names, int(part["weight"].shape[0] * (1 - fraction)), casts)
            for part, (names, casts) in parts
        ]
        sharded = type(module).__name__ == "ShardedEmbedding"
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


# Routed experts. The slot cache follows oMLX's MoE expert offload (Apache-2.0,
# omlx/patches/moe_expert_offload.py): contiguous per-layer slot arrays, decayed-frequency
# eviction, the call's own experts never evicted, and over-capacity calls run expert by expert.

_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_EXPERT_FIELDS = ("weight", "scales", "biases", "bias")
# The stock SwitchGLU sorts a call's routes from this many on.
_SORT_ROUTES = 64
_SCORE_DECAY, _SCORE_DECAY_EVERY = 0.7, 4


def _picks_rhs(rows, routes, experts):
    """Whether MLX runs a sorted ``gather_qmm`` as ``gather_qmm_rhs`` (``GatherQMM::eval_gpu``)."""
    return rows == 1 and routes >= 16 and routes // experts >= 4


class _FullTableKernels:
    """Mixed into a slot table's ``QuantizedSwitchLinear``. MLX chooses a sorted call's kernel from
    the table size, and the slot table is smaller than the expert table: where the full table gets
    the per-route kernel, the call runs unsorted, which computes each route alone just as the
    sorted per-route kernel does."""

    def __call__(self, x, indices, sorted_indices = False):
        if sorted_indices and not _picks_rhs(x.shape[-2], indices.size, self.full_experts):
            sorted_indices = False
        return super().__call__(x, indices, sorted_indices = sorted_indices)

    def gather(self, x, indices, sorted_indices):
        return super().__call__(x, indices, sorted_indices = sorted_indices)


_SLOT_CLASSES = {}


def _slot_projection(projection, arrays, full_experts):
    base = type(projection)
    if base not in _SLOT_CLASSES:
        _SLOT_CLASSES[base] = type(f"Slot{base.__name__}", (_FullTableKernels, base), {})
    slot = copy.copy(projection)
    slot.__class__ = _SLOT_CLASSES[base]
    for name, array in arrays.items():
        slot[name] = array
    slot.full_experts = full_experts
    return slot


class _ExpertSlots:
    """One layer's experts in ``capacity`` slots. A miss evicts the expert with the lowest decayed
    routing count (+1 per route, x0.7 every 4 calls) among those the call does not use."""

    def __init__(self, fields, experts, capacity):
        self.fields = fields  # [(projection, name, _Field)]
        self.capacity = capacity
        self.arrays = {
            (projection, name): mx.zeros((capacity, *field.shape), dtype = field.cast or field.dtype)
            for projection, name, field in fields
        }
        mx.eval(list(self.arrays.values()))
        self.slot_of = np.full(experts, -1, dtype = np.int64)
        self.expert_of = np.full(capacity, -1, dtype = np.int64)
        self.map = mx.zeros((experts,), dtype = mx.int32)  # expert -> slot, valid while resident
        self.score = np.zeros(experts, dtype = np.float32)
        self.expert_bytes = sum(field.row_bytes for _, _, field in fields)
        self.calls = self.hits = self.misses = 0

    def ensure(self, routes, experts):
        """Make ``experts`` (unique, at most ``capacity``) resident; ``routes`` are the call's routes."""
        np.add.at(self.score, routes, 1.0)
        self.calls += 1
        if self.calls % _SCORE_DECAY_EVERY == 0:
            self.score *= _SCORE_DECAY
        missing = experts[self.slot_of[experts] < 0]
        self.hits += experts.size - missing.size
        if not missing.size:
            return
        self.misses += missing.size
        slots = self._reserve(missing.size, experts)
        buffers = [np.empty((missing.size, field.row_bytes), dtype = np.uint8) for _, _, field in self.fields]
        jobs = [
            (field.fd, field.offset + expert * field.row_bytes, memoryview(buffer[i]))
            for (_, _, field), buffer in zip(self.fields, buffers)
            for i, expert in enumerate(missing.tolist())
        ]
        read_ranges(jobs)
        at = mx.array(slots.astype(np.int32))
        for (projection, name, field), buffer in zip(self.fields, buffers):
            self.arrays[projection, name][at] = field.to_mx(buffer)
        self.slot_of[missing] = slots
        self.expert_of[slots] = missing
        self.map[mx.array(missing.astype(np.int32))] = at

    def slot_indices(self, indices):
        return self.map[indices].astype(indices.dtype)

    def _reserve(self, count, protected):
        free = np.flatnonzero(self.expert_of < 0)
        if free.size >= count:
            return free[:count]
        held = np.flatnonzero((self.expert_of >= 0) & ~np.isin(self.expert_of, protected))
        victims = held[np.argsort(self.score[self.expert_of[held]], kind = "stable")[:count - free.size]]
        self.slot_of[self.expert_of[victims]] = -1
        return np.concatenate([free, victims])


class _GLUPart:
    """``up_proj`` / ``gate_proj`` / ``activation`` / ``down_proj`` of an ``OffloadSwitchGLU``, for
    callers that run the GLU's steps themselves (mlx-vlm's batch-invariant verifier)."""

    def __init__(self, owner, name):
        self._owner, self._name = owner, name

    def __call__(self, *args, **kwargs):
        return self._owner._part(self._name, *args, **kwargs)


class OffloadSwitchGLU(nn.Module):
    """Stands in for a ``SwitchGLU`` of quantized experts that stay in the checkpoint. A call reads
    the experts it routes to into the slots, then runs a copy of the original module (kept out of
    the module tree) on the slots, so the stock code and kernels compute it. A call routing to more
    experts than the slots hold runs expert by expert, each expert read once (once per position
    where the stock call runs a decode block position by position)."""

    def __init__(self, glu, slots, experts):
        super().__init__()
        object.__setattr__(self, "_glu", glu)
        object.__setattr__(self, "_slots", slots)
        object.__setattr__(self, "_parts", None)
        self.experts = experts
        for name in ("up_proj", "gate_proj", "activation", "down_proj"):
            object.__setattr__(self, name, _GLUPart(self, name))

    def _stock(self):
        glu = self._glu
        if glu.training != self.training:
            glu.train(self.training)
        return glu

    def _routes(self, indices):
        routes = np.asarray(indices).reshape(-1).astype(np.int64)
        return routes, np.unique(routes)

    def __call__(self, x, indices):
        glu = self._stock()
        routes, experts = self._routes(indices)
        slots = self._slots
        if experts.size <= slots.capacity:
            slots.ensure(routes, experts)
            return glu(x, slots.slot_indices(indices))
        # Over the slots: expert by expert, each route computed as the stock call computes it.
        rows = x.reshape(-1, x.shape[-1])
        token = np.arange(routes.size) // indices.shape[-1]
        block = _decode_block(glu, x)
        if block and x.shape[0] * indices.shape[-1] >= _SORT_ROUTES:  # stock runs the block one position at a time
            return mx.concatenate([self(x[:, i:i + 1], indices[:, i:i + 1]) for i in range(x.shape[1])], axis = 1)
        if block and x.shape[0] == 1:
            run = self._block_run(glu, rows, indices.dtype)
        else:  # mlx-vlm computes a batch's decode block one route at a time, as an unsorted gather does
            sort = not block and indices.size >= _SORT_ROUTES and _picks_rhs(1, indices.size, self.experts)
            run = self._gather_run(rows, lambda r, s, srt: (self._glu_rows(glu, r, s, srt)[3],), sort)
        (routed,) = self._by_expert(token, routes, run)
        return routed.reshape(*indices.shape, -1)

    @staticmethod
    def _glu_rows(glu, rows, slots, sort):
        """``(up, gate, activated, down)`` of one route per row, by the stock projections' kernels."""
        up = glu.up_proj.gather(rows, slots, sort)
        gate = glu.gate_proj.gather(rows, slots, sort)
        activated = glu.activation(up, gate)
        return up, gate, activated, glu.down_proj.gather(activated, slots, sort)

    def _by_expert(self, row_of_route, routes, run):
        """``run(experts, rows)`` over every route, expert by expert in chunks the slots can hold, each
        chunk evaluated before the next is read. ``run`` gets each route's expert and source row and
        returns per-route outputs; returns each output as ``[routes, dims]`` in route order."""
        slots = self._slots
        resident = slots.slot_of[routes] >= 0
        # Resident experts first, so a chunk's misses evict only experts already used.
        order = np.lexsort((routes, ~resident))
        ordered = routes[order]
        starts = np.flatnonzero(np.concatenate([[True], ordered[1:] != ordered[:-1]]))
        bounds = list(starts[::slots.capacity]) + [routes.size]
        outputs = []
        for start, end in zip(bounds[:-1], bounds[1:]):
            chunk = ordered[start:end]
            slots.ensure(chunk, np.unique(chunk))
            parts = run(chunk, row_of_route[order[start:end]])
            mx.eval(parts)
            outputs.append(parts)
        inverse = mx.array(np.argsort(order, kind = "stable").astype(np.int32))
        return tuple(mx.concatenate(column)[inverse] for column in zip(*outputs))

    def _gather_run(self, rows, compute, sort):
        """Chunks as stock gathers, one route per row. ``sort`` keeps a chunk on
        ``gather_qmm_rhs`` by sorting it and padding it to the slot table's minimum for that kernel."""
        slots = self._slots
        floor = max(16, 4 * slots.capacity)

        def run(chunk, source):
            count = chunk.size
            if sort and count < floor:
                chunk, source = (np.pad(a, (0, floor - count), mode = "edge") for a in (chunk, source))
            index = slots.map[mx.array(chunk.astype(np.int32))]
            taken = rows[mx.array(source.astype(np.int32))]
            if sort:  # the stock sorted layout: one [1, dims] row per route, flat indices
                index_order = mx.argsort(index)
                parts = compute(mx.expand_dims(taken[index_order], -2), index[index_order], True)
                inverse = mx.argsort(index_order)
                parts = tuple(part[inverse] for part in parts)
            else:
                parts = compute(mx.expand_dims(taken, (-2, -3)), index.reshape(-1, 1), False)
            return tuple(part.reshape(part.shape[0], -1)[:count] for part in parts)

        return run

    def _block_run(self, glu, rows, dtype):
        """Chunks as mlx-vlm decode blocks of the chunk's tokens. Those kernels give a route the same
        result whatever the block's other positions and routes are, so a token's routes in the chunk
        make up its position, padded by repeating one of them, at least two positions per block."""
        slots = self._slots
        from mlx_vlm.models.linear import DECODE_BLOCK_SIZE

        def run(chunk, source):
            tokens, position = np.unique(source, return_inverse = True)
            by_position = np.argsort(position, kind = "stable")
            first = np.searchsorted(position[by_position], np.arange(tokens.size))
            column = np.empty(chunk.size, dtype = np.int64)
            column[by_position] = np.arange(chunk.size) - first[position[by_position]]
            grid = np.repeat(chunk[by_position[first]][:, None], column.max() + 1, axis = 1)
            grid[position, column] = chunk
            outputs = []
            for at in range(0, tokens.size, DECODE_BLOCK_SIZE):
                block_tokens, block_grid = tokens[at:at + DECODE_BLOCK_SIZE], grid[at:at + DECODE_BLOCK_SIZE]
                size = block_tokens.size
                if size == 1:  # the decode-block path needs two positions
                    block_tokens, block_grid = np.repeat(block_tokens, 2), np.repeat(block_grid, 2, axis = 0)
                index = slots.map[mx.array(block_grid.astype(np.int32))].astype(dtype)
                block = rows[mx.array(block_tokens.astype(np.int32))]
                outputs.append(glu(block[None], index[None])[0][:size])
            routed = mx.concatenate(outputs)
            return (routed[mx.array(position.astype(np.int32)), mx.array(column.astype(np.int32))],)

        return run

    def _part(self, name, *args, **kwargs):
        glu = self._stock()
        if name == "activation":
            up, gate = args
            parts = self._parts
            if parts is not None and parts[1] is up and parts[2] is gate:
                return parts[3]
            return glu.activation(up, gate)
        x, indices = args[:2]
        sorted_indices = kwargs.get("sorted_indices", args[2] if len(args) > 2 else False)
        parts = self._parts
        if name != "up_proj" and parts is not None:
            if name == "gate_proj" and parts[0][0] is x and parts[0][1] is indices:
                return parts[2]
            if name == "down_proj" and parts[3] is x and parts[0][1] is indices:
                return parts[4]
        routes, experts = self._routes(indices)
        slots = self._slots
        if experts.size <= slots.capacity:
            slots.ensure(routes, experts)
            index = slots.slot_indices(indices)
            if name != "up_proj":
                return getattr(glu, name)(x, index, sorted_indices = sorted_indices)
            up = glu.up_proj(x, index, sorted_indices = sorted_indices)
            gate = glu.gate_proj(x, index, sorted_indices = sorted_indices)
            activated = glu.activation(up, gate)
            down = glu.down_proj(activated, index, sorted_indices = sorted_indices)
        else:
            # Over the slots: one row per route, unsorted (each route alone), as the caller's
            # per-route kernels compute it.
            if sorted_indices:
                raise ValueError("Unsloth: a sorted call over more experts than the offload slots hold")
            per_route = x.shape[-3] == indices.shape[-1]
            rows = x.reshape(-1, x.shape[-1])
            row_of_route = np.arange(routes.size) // (1 if per_route else indices.shape[-1])
            shape = (*indices.shape, 1, -1)
            if name != "up_proj":
                (out,) = self._by_expert(row_of_route, routes, self._gather_run(
                    rows, lambda r, s, srt: (getattr(glu, name).gather(r, s, srt),), False))
                return out.reshape(shape)
            up, gate, activated, down = (part.reshape(shape) for part in self._by_expert(
                row_of_route, routes, self._gather_run(rows, lambda r, s, srt: self._glu_rows(glu, r, s, srt), False)))
        object.__setattr__(self, "_parts", ((x, indices), up, gate, activated, down))
        return up

    def offload_bytes(self):
        """``(on disk, resident)`` bytes of the experts this stands in for."""
        slots = self._slots
        return self.experts * slots.expert_bytes, sum(array.nbytes for array in slots.arrays.values())


def _decode_block(glu, x):
    """Whether the stock call takes mlx-vlm's decode-block path."""
    if type(glu).__module__ != "mlx_vlm.models.switch_layers":
        return False
    from mlx_vlm.models.linear import DECODE_BLOCK_SIZE
    return not glu.training and x.ndim == 3 and 1 < x.shape[1] <= DECODE_BLOCK_SIZE


def expert_sites(model):
    """``(path, module)`` of every ``SwitchGLU`` of quantized experts in ``model``."""
    for path, module in model.named_modules():
        if type(module).__name__ == "SwitchGLU" and all(
            type(getattr(module, name, None)).__name__ == "QuantizedSwitchLinear" for name in _PROJECTIONS
        ):
            yield path, module


def _top_k(model, path):
    parent = model
    for key in path.split(".")[:-1]:
        parent = parent[int(key)] if isinstance(parent, list) else parent[key]
    for name in ("top_k", "num_experts_per_tok", "n_activated_experts", "experts_per_token"):
        value = getattr(parent, name, None)
        if isinstance(value, int):
            return value
    return None


def _expert_fields(model, store, path, glu, cast_for):
    """``(fields, top_k)`` of a ``SwitchGLU`` whose experts can be read from disk, or why not."""
    top_k = _top_k(model, path)
    if top_k is None:
        return None, None, "its routing top-k is unknown"
    fields = []
    for projection in _PROJECTIONS:
        linear = getattr(glu, projection)
        if getattr(linear, "global_scale", None) is not None:
            return None, None, "its experts have global scales"
        for name in _EXPERT_FIELDS:
            if linear.get(name) is None:
                continue
            parameter = f"{path}.{projection}.{name}"
            tensor, reason = _checkpoint_tensor(store, parameter, linear[name])
            if reason is not None:
                return None, None, reason
            fields.append((projection, name, _Field(store, name, tensor, cast_for(parameter, linear[name].dtype))))
    return fields, top_k, None


def offloadable(model, model_dir, *, cast_dtype = None):
    """What ``offload_ple`` / ``offload_experts`` could take from the unevaluated ``model``, at the
    load's cast: bytes, the experts kept at full offload, and ``ple`` / ``experts``, ``None`` or why not."""
    cast_for = _cast_rule(model, cast_dtype)
    store = CheckpointTensors(model_dir)
    loaded = lambda array, cast: array.size * (cast or array.dtype).size
    found = {
        "ple_bytes": 0, "expert_bytes": 0, "expert_floor_bytes": 0, "experts_per_layer": 0,
        "expert_layer_bytes": 0, "ple": "the model has no per-layer embedding table",
        "experts": "the model has no stacked quantized routed experts",
    }
    skipped = {"ple": None, "experts": None}
    for path, module in ple_sites(model):
        parts, _, reason = _ple_parts(store, path, module, cast_for)
        if reason is not None:
            skipped["ple"] = skipped["ple"] or reason
            continue
        found["ple_bytes"] += sum(loaded(part[field], casts[field]) for part, (names, casts) in parts for field in names)
    for path, glu in expert_sites(model):
        fields, top_k, reason = _expert_fields(model, store, path, glu, cast_for)
        experts = glu.up_proj["weight"].shape[0]
        if reason is None and max(8, top_k) >= experts:
            reason = f"its {experts} experts per layer all stay resident"
        if reason is not None:
            skipped["experts"] = skipped["experts"] or reason
            continue
        size = sum(loaded(getattr(glu, projection)[name], field.cast) for projection, name, field in fields)
        found["expert_bytes"] += size
        found["expert_floor_bytes"] += size * max(8, top_k) // experts
        found["experts_per_layer"] = max(found["experts_per_layer"], experts)
        found["expert_layer_bytes"] = max(found["expert_layer_bytes"], size)
    for key, size in (("ple", "ple_bytes"), ("experts", "expert_bytes")):
        found[key] = None if found[size] else skipped[key] or found[key]
    return found


def offload_experts(model, model_dir, fraction, *, cast_dtype = None):
    """Replace every ``SwitchGLU`` of quantized experts in ``model`` with an ``OffloadSwitchGLU``
    that reads experts from ``model_dir``'s shards into slots for ``1 - fraction`` of each layer's
    experts, at least ``max(8, top_k)``. Must run before anything evaluates the experts;
    ``cast_dtype`` as for ``offload_ple``.
    Returns ``{"layers", "on_disk_bytes", "resident_bytes", "skipped": [(path, reason)]}``."""
    cast_for = _cast_rule(model, cast_dtype)
    store = CheckpointTensors(model_dir)
    report = {"layers": 0, "on_disk_bytes": 0, "resident_bytes": 0, "skipped": []}
    for path, glu in list(expert_sites(model)):
        experts = glu.up_proj["weight"].shape[0]
        fields, top_k, reason = _expert_fields(model, store, path, glu, cast_for)
        if reason is not None:
            report["skipped"].append((path, reason))
            continue
        capacity = max(8, top_k, experts - math.ceil(experts * fraction - 1e-9))
        if capacity >= experts:
            report["skipped"].append((path, f"{experts} experts fit in {capacity} slots"))
            continue
        slots = _ExpertSlots(fields, experts, capacity)
        stock = copy.copy(glu)
        for projection in _PROJECTIONS:
            arrays = {name: slots.arrays[p, name] for p, name, _ in fields if p == projection}
            setattr(stock, projection, _slot_projection(getattr(glu, projection), arrays, experts))
        slots.store = store
        replacement = OffloadSwitchGLU(stock, slots, experts)
        replacement.train(glu.training)
        _replace_module(model, path, replacement)
        on_disk, resident = replacement.offload_bytes()
        report["layers"] += 1
        report["on_disk_bytes"] += on_disk
        report["resident_bytes"] += resident
    return report
