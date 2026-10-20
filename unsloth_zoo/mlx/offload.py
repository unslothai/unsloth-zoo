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
nothing is converted or written to disk. ``plan_offload`` decides how much of each to offload for a
load, priced from the safetensors headers before anything is built.
"""

import contextlib
import copy
import glob
import inspect
import json
import math
import os
import re
import struct
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import mlx.nn as nn
import numpy as np

__all__ = [
    "CheckpointTensors",
    "DiskEmbedding",
    "OffloadSwitchGLU",
    "SAFETY_MARGIN",
    "context_bytes",
    "offload_experts",
    "offload_footprint",
    "offload_inventory",
    "offload_ple",
    "offload_setting",
    "offloadable",
    "plan_offload",
    "read_ranges",
]

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


def offload_setting(value):
    """``from_pretrained``'s ``offload``: a callable, kept as given, or ``{"ple": knob, "experts": knob}``,
    each ``None`` (off), ``"auto"`` or the fraction to offload in (0, 1]. One knob sets both; a dict
    sets each, a missing part off."""
    if callable(value):
        return value
    if isinstance(value, dict):
        if set(value) - {"ple", "experts"}:
            raise ValueError(f"Unsloth: offload takes the parts 'ple' and 'experts', got {sorted(value)}.")
        return {part: _knob(f"offload[{part!r}]", value.get(part)) for part in ("ple", "experts")}
    knob = _knob("offload", value)
    return {"ple": knob, "experts": knob}


def _knob(name, value):
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
    """One checkpoint tensor, read by its leading index (a table row or an expert); or, for experts a
    checkpoint stores apart, one tensor per expert."""

    __slots__ = ("name", "fd", "offset", "row_bytes", "transport", "dtype", "shape", "cast", "map", "apart")

    def __init__(self, store, name, tensor, cast, mapped = False):
        tensors = tensor if isinstance(tensor, list) else [tensor]
        path, dtype, shape, offset = store.specs[tensors[0]]
        self.transport, self.dtype = _DTYPES[dtype]
        self.name, self.fd, self.offset, self.cast = name, store.fd(path), offset, cast
        self.shape = tuple(shape[1:]) if tensor is tensors[0] else tuple(shape)
        self.row_bytes = int(np.prod(self.shape)) * np.dtype(self.transport).itemsize
        self.map = None
        if mapped:
            self.map = np.memmap(path, dtype = np.uint8, mode = "r", offset = offset, shape = (shape[0], self.row_bytes))
        self.apart = None
        if tensor is not tensors[0]:
            self.apart = [(store.fd(store.specs[t][0]), store.specs[t][3]) for t in tensors]

    def location(self, index):
        """``(fd, offset)`` of row or expert ``index``."""
        if self.apart is not None:
            return self.apart[index]
        return self.fd, self.offset + index * self.row_bytes

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
            # Copied from the map, a field at a time: a staging buffer would hold the prefix twice.
            for field in self.fields:
                self.held.append(field.to_mx(field.map[:self.resident]))
                mx.eval(self.held[-1])

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
    """The checkpoint's name for a parameter: its MLX name, the Hugging Face layout of it, or, for a
    text checkpoint mlx-vlm wraps as its language model, the text layout."""
    if name in store.specs:
        return name
    prefix = "language_model.model."
    if name.startswith(prefix):
        for alternate in ("model.language_model.", "model."):
            if alternate + name[len(prefix):] in store.specs:
                return alternate + name[len(prefix):]
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
# DeepSeek's reference names for them, which DeepSeek V4.1 checkpoints keep.
_REFERENCE_NAMES = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}
# DeepSeek V4.1's also takes each route's weight into its activation.
_SWITCH_GLUS = ("SwitchGLU", "DeepseekV41SwitchGLU")
_EXPERT_FIELDS = ("weight", "scales", "biases", "bias")
# The stock SwitchGLU sorts a call's routes from this many on.
_SORT_ROUTES = 64
# Largest single read when a full table reads runs of adjacent experts.
_RUN_BYTES = 8 << 20
# The prefill step a wide decoder layer call runs in: mlx-vlm's and mlx-lm's default.
_PREFILL_STEP = 2048


def _deepseek_v4_step_mask(layer, arguments):
    """DeepSeek V4 builds one sliding-window mask array for the whole call; each step rebuilds it from
    the block's own rotating cache, as chunked prefill does. Calls with image tokens are not causal."""
    if not arguments.get("causal") or arguments.get("position_offset") is not None:
        return None
    from mlx_vlm.models.base import create_attention_mask
    from mlx_vlm.models.cache import CacheList

    cache = arguments["cache"]
    local = cache[0] if isinstance(cache, CacheList) else cache
    window = layer.attn.config.sliding_window
    return lambda h: create_attention_mask(h[:, :, 0, :], local, window_size = window, return_array = True)


# Decoder layers whose call wider than a prefill step may run as consecutive steps on the layer's
# own cache (no state shared with other layers): each token-indexed argument's token axis, and either
# the masks under which the layer attends causally, or a function of (layer, arguments) giving each
# step's mask from its hidden states (None when the call is not causal).
_WIDE_LAYERS = {
    "mlx_vlm.models.qwen4_exp.language.Qwen4ExpDecoderLayer": (
        {"hidden_states": 1, "input_ids": 1, "position_ids": -1}, (None, "causal")),
    "mlx_lm.models.qwen3_moe.Qwen3MoeDecoderLayer": ({"x": 1}, ("causal",)),
    "mlx_vlm.models.deepseek_v4.language.DeepseekV4Block": ({"h": 1, "input_ids": 1}, _deepseek_v4_step_mask),
}
_table_reader_pool = None


def _runs(field, experts):
    """``[fd, offset, first expert, count]`` covering the sorted ``experts``, consecutive experts
    stored back to back in one file merged into one run."""
    runs = []
    for expert in experts.tolist():
        fd, offset = field.location(expert)
        last = runs[-1] if runs else None
        if last and last[0] == fd and last[2] + last[3] == expert and last[1] + last[3] * field.row_bytes == offset:
            last[3] += 1
        else:
            runs.append([fd, offset, expert, 1])
    return runs


def _table_reader():
    """Fills read-ahead tables one at a time (each spreads its reads over the reader pool)."""
    global _table_reader_pool
    with _pool_lock:
        if _table_reader_pool is None:
            _table_reader_pool = ThreadPoolExecutor(1, thread_name_prefix = "unsloth-offload-table")
        return _table_reader_pool


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
        self.map = mx.zeros((experts,), dtype = mx.int32)  # expert -> slot, valid while resident
        # Built here: MLX streams are thread-bound, so a lazy table would fail on any other thread.
        mx.eval(list(self.arrays.values()), self.map)
        self.slot_of = np.full(experts, -1, dtype = np.int64)
        self.expert_of = np.full(capacity, -1, dtype = np.int64)
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
            (*field.location(expert), memoryview(buffer[i]))
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
        object.__setattr__(self, "_full", None)  # the stock module on a full table, while a wide layer runs
        object.__setattr__(self, "_ahead", None)  # (future, module) of a table being read ahead
        object.__setattr__(self, "_next", None)  # the next offloaded layer a wide prefill reaches
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

    def _table(self):
        """The stock module on a new table of all this layer's experts, and the function filling it:
        resident experts copied from the slots, the others read in runs of adjacent experts. ``None``
        when a field is cast at load, so its bytes on disk are not the table's."""
        slots = self._slots
        if any(field.cast is not None for _, _, field in slots.fields):
            return None
        arrays = {
            (projection, name): mx.zeros((self.experts, *field.shape), dtype = field.dtype)
            for projection, name, field in slots.fields
        }
        mx.eval(list(arrays.values()))
        # Taken here: buffers of evaluated arrays, which the filling thread only copies into.
        views = [
            (field, memoryview(arrays[projection, name]).cast("B"), memoryview(slots.arrays[projection, name]).cast("B"))
            for projection, name, field in slots.fields
        ]
        held = np.flatnonzero(slots.slot_of >= 0)
        where, missing = slots.slot_of[held], np.flatnonzero(slots.slot_of < 0)

        def fill():
            jobs = []
            for field, table, resident in views:
                row = field.row_bytes
                rows = np.frombuffer(table, dtype = np.uint8).reshape(self.experts, row)
                rows[held] = np.frombuffer(resident, dtype = np.uint8).reshape(-1, row)[where]
                for fd, offset, first, count in _runs(field, missing):
                    start, size = first * row, count * row
                    jobs += [(fd, offset + at, table[start + at:start + min(size, at + _RUN_BYTES)])
                             for at in range(0, size, _RUN_BYTES)]
            read_ranges(jobs)

        glu = copy.copy(self._glu)
        for projection in _PROJECTIONS:
            linear = copy.copy(getattr(self._glu, projection))
            for (owner, name), array in arrays.items():
                if owner == projection:
                    linear[name] = array
            setattr(glu, projection, linear)
        return glu, fill

    def _join_ahead(self):
        """The module a read-ahead filled, once it is filled; ``None`` when none was started."""
        ahead = self._ahead
        if ahead is None:
            return None
        object.__setattr__(self, "_ahead", None)
        ahead[0].result()
        return ahead[1]

    def _wide(self, read_next):
        """Run calls on a full table until ``_narrow``, reading the next offloaded layer's ahead."""
        glu = self._join_ahead()
        if glu is None:
            made = self._table()
            if made is None:
                return False
            glu, fill = made
            fill()
        glu.train(self.training)
        object.__setattr__(self, "_full", glu)
        following = self._next
        if read_next and following is not None and following._ahead is None:
            made = following._table()
            if made is not None:
                object.__setattr__(following, "_ahead", (_table_reader().submit(made[1]), made[0]))
        return True

    def _narrow(self):
        object.__setattr__(self, "_full", None)

    def __call__(self, x, indices, *weights):
        if self._full is not None:
            return self._full(x, indices, *weights)
        # A read-ahead nothing used must finish before the slots change under its copy.
        self._join_ahead()
        glu = self._stock()
        routes, experts = self._routes(indices)
        slots = self._slots
        if experts.size <= slots.capacity:
            slots.ensure(routes, experts)
            return glu(x, slots.slot_indices(indices), *weights)
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
            run = self._gather_run(rows, lambda r, s, srt, *w: (self._glu_rows(glu, r, s, srt, *w)[3],), sort, *weights)
        (routed,) = self._by_expert(np.arange(routes.size) if weights else token, routes, run)
        return routed.reshape(*indices.shape, -1)

    @staticmethod
    def _glu_rows(glu, rows, slots, sort, *weights):
        """``(up, gate, activated, down)`` of one route per row, by the stock projections' kernels."""
        up = glu.up_proj.gather(rows, slots, sort)
        gate = glu.gate_proj.gather(rows, slots, sort)
        activated = glu.activation(up, gate, *weights)
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

    def _gather_run(self, rows, compute, sort, weights = None):
        """Chunks as stock gathers, one route per row. ``sort`` keeps a chunk on
        ``gather_qmm_rhs`` by sorting it and padding it to the slot table's minimum for that kernel.
        With ``weights`` (``[..., top_k]``), sources are routes and ``compute`` also gets their weights."""
        slots = self._slots
        floor = max(16, 4 * slots.capacity)

        def run(chunk, source):
            count = chunk.size
            if sort and count < floor:
                chunk, source = (np.pad(a, (0, floor - count), mode = "edge") for a in (chunk, source))
            index = slots.map[mx.array(chunk.astype(np.int32))]
            source = mx.array(source.astype(np.int32))
            if weights is None:
                taken, weight = rows[source], ()
            else:
                taken, weight = rows[source // weights.shape[-1]], (weights.reshape(-1)[source],)
            if sort:  # the stock sorted layout: one [1, dims] row per route, flat indices
                index_order = mx.argsort(index)
                weight = tuple(w[index_order][:, None, None] for w in weight)
                parts = compute(mx.expand_dims(taken[index_order], -2), index[index_order], True, *weight)
                inverse = mx.argsort(index_order)
                parts = tuple(part[inverse] for part in parts)
            else:
                weight = tuple(w[:, None, None, None] for w in weight)
                parts = compute(mx.expand_dims(taken, (-2, -3)), index.reshape(-1, 1), False, *weight)
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
        self._join_ahead()
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
            # Over the slots: one unsorted row per route, as the caller's per-route kernels compute it.
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


def _token_slice(value, axis, start, end):
    index = [slice(None)] * value.ndim
    index[axis] = slice(start, end)
    return value[tuple(index)]


class _WideLayer:
    """Mixed into a decoder layer of ``_WIDE_LAYERS`` holding offloaded experts. A call wider than
    ``_PREFILL_STEP`` tokens runs as consecutive steps of it on the layer's cache, which is what a
    chunked prefill computes for the layer, on a table of all its experts read once for the call while
    the next offloaded layer's table is read ahead. A prefill step of several chunks then reads each
    layer's experts once, and leaves the slots decode uses alone."""

    def __call__(self, *args, **kwargs):
        axes = self._wide_axes
        first = next(iter(axes))
        hidden = args[0] if args else kwargs.get(first)
        if not isinstance(hidden, mx.array) or hidden.ndim < 2 or hidden.shape[0] != 1 or hidden.shape[1] <= _PREFILL_STEP:
            return self._whole(hidden, args, kwargs)
        bound = self._wide_signature.bind(self, *args, **kwargs)
        cache, mask, masks = bound.arguments.get("cache"), bound.arguments.get("mask"), self._wide_masks
        step_mask = masks(self, bound.arguments) if callable(masks) and cache is not None else None
        causal = step_mask is not None if callable(masks) else (mask is None or isinstance(mask, str)) and mask in masks
        switch = self._offload_switch
        if cache is None or not causal or not switch._wide(read_next = True):
            return self._whole(hidden, args, kwargs)
        given = {name: bound.arguments.get(name) for name in axes}
        capture = getattr(_prefill_capture, "request", None)
        outputs = []
        try:
            for start in range(0, hidden.shape[1], _PREFILL_STEP):
                for name, axis in axes.items():
                    if given[name] is not None:
                        bound.arguments[name] = _token_slice(given[name], axis, start, start + _PREFILL_STEP)
                if step_mask is not None:
                    bound.arguments["mask"] = step_mask(bound.arguments[first])
                output = super().__call__(*bound.args[1:], **bound.kwargs)
                mx.eval(output, cache.state)
                outputs.append(output)
                if capture is not None and min(start + _PREFILL_STEP, hidden.shape[1]) == capture[0]:
                    capture[2][id(cache)] = capture[1](cache)
        finally:
            switch._narrow()
        return mx.concatenate(outputs, axis = axes[first])

    def _whole(self, hidden, args, kwargs):
        output = super().__call__(*args, **kwargs)
        capture = getattr(_prefill_capture, "request", None)
        if capture is not None and isinstance(hidden, mx.array) and hidden.ndim >= 2 and hidden.shape[1] == capture[0]:
            cache = self._wide_signature.bind(self, *args, **kwargs).arguments.get("cache")
            if cache is not None:
                mx.eval(output, cache.state)
                capture[2][id(cache)] = capture[1](cache)
        return output


_WIDE_CLASSES = {}
_prefill_capture = threading.local()


def wide_prefill_step(model):
    """The step every wide single-row prefill call of ``model`` runs in, exactly as a prefill chunked at
    it, or ``None`` when some layer may run a call whole (dense layers, DeepSeek V4's image rows, cast fields)."""
    modules = dict(model.named_modules())
    switches = [module for module in modules.values() if isinstance(module, OffloadSwitchGLU)]
    wide = {path for path, layer in modules.items() if isinstance(layer, _WideLayer)}
    stacks = {path.rpartition(".")[0] for path in wide}
    stacked = {path for path in modules if path.rpartition(".")[0] in stacks and path.rpartition(".")[2].isdigit()}
    held = {id(modules[path]._offload_switch) for path in wide if not callable(modules[path]._wide_masks)}
    if not switches or stacked - wide or any(
        id(switch) not in held or any(field.cast is not None for _, _, field in switch._slots.fields)
        for switch in switches
    ):
        return None
    return _PREFILL_STEP


@contextlib.contextmanager
def capture_prefill_rows(rows, copy):
    """Yields ``{id(cache): copy(cache)}``, filled by each wide layer of a prefill call in this thread
    once the call's first ``rows`` rows (a multiple of ``wide_prefill_step``) are in its cache."""
    captured = {}
    _prefill_capture.request = (rows, copy, captured)
    try:
        yield captured
    finally:
        _prefill_capture.request = None


def _widen_layers(model, switches):
    """Give each ``_WIDE_LAYERS`` layer holding one of ``switches`` (path -> ``OffloadSwitchGLU``) its
    wide calls, and chain each offloaded layer to the next one of its stack for read-ahead."""
    stacks = {}
    for path, layer in model.named_modules():
        base = type(layer)
        spec = _WIDE_LAYERS.get(f"{base.__module__}.{base.__qualname__}")
        if spec is None:
            continue
        held = [switch for inner, switch in switches.items() if inner.startswith(path + ".")]
        if len(held) != 1:
            continue
        if base not in _WIDE_CLASSES:
            _WIDE_CLASSES[base] = type(base.__name__, (_WideLayer, base), {
                "_wide_axes": spec[0], "_wide_masks": staticmethod(spec[1]) if callable(spec[1]) else spec[1], "_wide_signature": inspect.signature(base.__call__),
            })
        layer.__class__ = _WIDE_CLASSES[base]
        object.__setattr__(layer, "_offload_switch", held[0])
        parent, _, index = path.rpartition(".")
        if index.isdigit():
            stacks.setdefault(parent, []).append((int(index), held[0]))
    for stack in stacks.values():
        stack.sort(key = lambda item: item[0])
        for (_, switch), (_, following) in zip(stack, stack[1:]):
            object.__setattr__(switch, "_next", following)


def _decode_block(glu, x):
    """Whether the stock call takes mlx-vlm's decode-block path (from 0.7.1; none before)."""
    if type(glu).__module__ != "mlx_vlm.models.switch_layers":
        return False
    size = getattr(sys.modules[type(glu).__module__], "DECODE_BLOCK_SIZE", 0)
    return not glu.training and x.ndim == 3 and 1 < x.shape[1] <= size


def expert_sites(model):
    """``(path, module)`` of every ``SwitchGLU`` of quantized experts in ``model``."""
    for path, module in model.named_modules():
        if type(module).__name__ in _SWITCH_GLUS and all(
            type(getattr(module, name, None)).__name__ == "QuantizedSwitchLinear" for name in _PROJECTIONS
        ):
            yield path, module


def _top_k(model, path):
    parent = model
    for key in path.split(".")[:-1]:
        parent = parent[int(key)] if isinstance(parent, list) else parent[key]
    # On the MoE block, or on its router (GLM-5, DeepSeek V4).
    for holder in (parent, getattr(parent, "gate", None)):
        for name in ("top_k", "num_experts_per_tok", "n_activated_experts", "experts_per_token"):
            value = getattr(holder, name, None)
            if isinstance(value, int):
                return value
    return None


def _expert_tensors(store, path, projection, name, expected):
    """``(names, None)`` of a stacked expert parameter's per-expert tensors, for a checkpoint that
    stores each expert apart (mlx-vlm stacks them at load, as for GLM-5), ``(None, reason)`` when they
    do not match, or ``None`` when the checkpoint has none."""
    if not path.endswith(".switch_mlp"):
        return None
    base = path[:-len("switch_mlp")] + "experts."
    for key in (projection, _REFERENCE_NAMES[projection]):
        names = [_checkpoint_name(store, f"{base}{expert}.{key}.{name}") for expert in range(expected.shape[0])]
        if None not in names:
            break
    else:
        return None
    for tensor in names:
        _, dtype, shape, _ = store.specs[tensor]
        if dtype not in _DTYPES or shape != tuple(expected.shape[1:]) or _DTYPES[dtype][1] != expected.dtype:
            return None, (
                f"{tensor} is {dtype} {list(shape)} in the checkpoint, "
                f"{expected.dtype} {list(expected.shape[1:])} an expert in the model"
            )
    return names, None


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
                tensor, reason = _expert_tensors(store, path, projection, name, linear[name]) or (tensor, reason)
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
    replaced = {}
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
        replaced[path] = replacement
        on_disk, resident = replacement.offload_bytes()
        report["layers"] += 1
        report["on_disk_bytes"] += on_disk
        report["resident_bytes"] += resident
    _widen_layers(model, replaced)
    return report


OFFLOAD_CONTEXT_STEP = 256
_PLE_TENSORS = (".embed_tokens_per_layer.", ".ngram_embedding.shards.")
# Tables mlx-vlm always reads row by row from the checkpoint, never resident: DeepSeek V4.1's Engram.
_MAPPED_TENSORS = (".engram.embed.",)
_EXPERT_TENSOR = re.compile(r"\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases|bias|global_scale)$")
_EXPERT_APART = re.compile(r"\.experts\.(\d+)\.(gate_proj|up_proj|down_proj|w1|w2|w3)\.(weight|scales|biases|bias|global_scale)$")
_FLOAT_BYTES = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8}
_NESTED_CONFIGS = ("text_config", "thinker_config", "language_config", "llm_config")
_TOP_K = ("num_experts_per_tok", "top_k", "n_activated_experts", "experts_per_token")
_CACHE_CAP = 1 << 30
# The seen-row bitmaps and read buffers of offloaded PLE tables, and the capped cache.
_PLE_OVERHEAD = (256 << 20) + _CACHE_CAP
PREFILL_STEPS = (2048, 4096, 8192)
# Kept free on top of the estimate (max of 2 GiB and 3%): it is fitted to measured peaks.
SAFETY_MARGIN = (2 << 30, 0.03)
# Disk experts cost ~4x decode at any fraction, so (as llama.cpp's fitter) an unset context shrinks first.
_OFFLOAD_CONTEXT = 8192
_TRANSIENT_RESERVE = 2 << 30


def _text_config(model_dir):
    """The first config, breadth first through the nested text configs, that sizes the decoder."""
    with open(os.path.join(model_dir, "config.json")) as stream:
        config = json.load(stream)
    queue = [config]
    while queue:
        current = queue.pop(0)
        if "num_hidden_layers" in current:
            return current
        queue += [current[key] for key in _NESTED_CONFIGS if isinstance(current.get(key), dict)]
    return config.get("text_config") or config


def _tensors(model_dir, dtype_bytes):
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with open(path, "rb") as stream:
            (length,) = struct.unpack("<Q", stream.read(8))
            header = json.loads(stream.read(length))
        for name, spec in header.items():
            if name != "__metadata__":
                start, end = spec["data_offsets"]
                size = end - start
                width = _FLOAT_BYTES.get(spec["dtype"])
                if dtype_bytes and width:
                    size = size // width * dtype_bytes
                yield name, spec["shape"], size


def offload_inventory(model_dir, dtype_bytes = None):
    """``offload.offloadable``'s sizes plus ``total_bytes``, from the safetensors headers alone (floats
    at ``dtype_bytes`` an element), to price a load before it is made. It counts every quantized
    ``switch_mlp``, so it can count experts a loaded model keeps; the ``mtp.`` head is never counted."""
    total = ple = 0
    layers = {}  # prefix -> [bytes, experts, quantized projections, global scales]
    for name, shape, size in _tensors(model_dir, dtype_bytes):
        if any(part in name for part in _MAPPED_TENSORS):
            continue
        total += size
        if name.startswith("mtp."):
            continue
        if any(part in name for part in _PLE_TENSORS):
            ple += size
            continue
        match = _EXPERT_TENSOR.search(name)
        apart = None if match is not None else _EXPERT_APART.search(name)
        if match is not None or apart is not None:
            layer = layers.setdefault(name[:(match or apart).start()], [0, 0, set(), False])
            layer[0] += size
            projection, field = match.groups() if match is not None else apart.groups()[1:]
            if match is not None and name.endswith(".up_proj.weight"):
                layer[1] = shape[0]
            if apart is not None:
                layer[1] = max(layer[1], int(apart.group(1)) + 1)
            if field == "scales":
                layer[2].add(projection)
            layer[3] |= field == "global_scale"
    config = _text_config(model_dir)
    top_k = next((config[key] for key in _TOP_K if isinstance(config.get(key), int)), None)
    quantized = [layer[:2] for layer in layers.values() if len(layer[2]) == 3 and layer[1] and not layer[3]]
    experts = sum(size for size, _ in quantized)
    floor = sum(size * min(count, max(8, top_k or 0)) // count for size, count in quantized)
    if not layers:
        reason = "the model has no stacked routed experts"
    elif not quantized:
        reason = (
            "its experts have global scales" if any(layer[3] for layer in layers.values())
            else "its routed experts are not quantized"
        )
    elif top_k is None:
        reason = "its routing top-k is unknown"
    elif floor >= experts:
        reason = f"its {max(count for _, count in quantized)} experts per layer all stay resident"
    else:
        reason = None
    return {
        "total_bytes": total,
        "ple_bytes": ple,
        "expert_bytes": experts if reason is None else 0,
        "expert_floor_bytes": floor if reason is None else 0,
        "experts_per_layer": max((count for _, count in quantized), default = 0),
        "expert_layer_bytes": max((size for size, _ in quantized), default = 0) if reason is None else 0,
        "ple": None if ple else "the model has no per-layer embedding table",
        "experts": reason,
    }


def _expert_overhead(inventory):
    """One layer's reads, a slot array's copy a pending computation holds, and the capped cache."""
    return 2 * inventory["expert_layer_bytes"] + _CACHE_CAP


def offload_footprint(inventory, ple, experts):
    """What the offloadable PLE tables and experts hold in memory with ``ple`` / ``experts`` offloaded
    (``None`` for off), working memory included; the rest of the weights stay as they are."""
    ple_bytes = inventory["ple_bytes"] if inventory["ple"] is None else 0
    held = ple_bytes * (1 - (ple or 0)) + (_PLE_OVERHEAD if ple and ple_bytes else 0)
    if experts and inventory["expert_bytes"]:
        kept = max(inventory["expert_floor_bytes"], inventory["expert_bytes"] * (1 - experts))
        return held + kept + _expert_overhead(inventory)
    return held + inventory["expert_bytes"]


def context_bytes(model_dir, tokens, prefill_step = PREFILL_STEPS[0], dtype_bytes = 2):
    """What ``tokens`` of context cost, from the config: the attention KV cache doubled (the
    transients that grow with it), a prefill step's logits and a reserve. Fitted above
    Qwen3.8-Flash-Next's measured peaks (13.7 GiB over the weights at 262K tokens)."""
    config = _text_config(model_dir)
    layers = config.get("num_hidden_layers", 0)
    types = config.get("layer_types")
    if not types:
        interval = config.get("full_attention_interval")
        types = ["full_attention" if not interval or (i + 1) % interval == 0 else "linear_attention" for i in range(layers)]
    heads = config.get("num_key_value_heads") or config.get("num_attention_heads", 0)
    head_dim = config.get("head_dim") or config.get("hidden_size", 0) // max(1, config.get("num_attention_heads", 1))
    if not (types and heads and head_dim):
        raise ValueError("Unsloth: this model's config does not size its attention, so its context cannot be priced.")
    window = config.get("sliding_window") or tokens
    held = sum(tokens if kind == "full_attention" else min(tokens, window) for kind in types if "attention" in kind and kind != "linear_attention")
    return 2 * 2 * heads * head_dim * dtype_bytes * held + 4 * prefill_step * config.get("vocab_size", 0) + _TRANSIENT_RESERVE


def _largest_context(cost, limit, low, high):
    """The largest context in ``[low, high]`` costing at most ``limit``, or ``None``."""
    if cost(low) > limit:
        return None
    if cost(high) <= limit:
        return high
    lo, hi = 0, (high - low) // OFFLOAD_CONTEXT_STEP
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if cost(low + mid * OFFLOAD_CONTEXT_STEP) <= limit:
            lo = mid
        else:
            hi = mid - 1
    return low + lo * OFFLOAD_CONTEXT_STEP


def plan_offload(
    *,
    budget,
    resident,
    inventory,
    context_cost,
    ceiling,
    minimum = 4096,
    context = None,
    prefill_step = None,
    offload = "auto",
):
    """The 50/50 rule: context takes at most half of what the resident weights leave of ``budget``
    less ``SAFETY_MARGIN`` (more only to reach ``minimum``), and the least offload that fits the rest
    is taken -- none, PLE, then PLE and as many experts as needed. Experts leave memory only when an
    unpinned context of ``_OFFLOAD_CONTEXT`` tokens (within ``minimum`` and ``ceiling``) does not fit
    beside them, and then as many as that context needs. With experts offloaded, each prefill step
    re-reads them, so the step with the fewest offloaded bytes per prompt token is taken.
    ``context_cost(tokens, step)`` prices the context. A ``context``, ``prefill_step`` or ``offload``
    part (as ``from_pretrained`` takes it) not ``"auto"`` is kept as given.
    Returns ``{"ple", "experts", "context", "prefill_step"}``, all ``None`` when nothing fits."""
    offload = offload_setting(offload)
    ple_offload, expert_offload = offload["ple"], offload["experts"]
    ple_bytes = inventory["ple_bytes"] if inventory["ple"] is None else 0
    expert_bytes, floor = inventory["expert_bytes"], inventory["expert_floor_bytes"]
    per_layer = inventory["experts_per_layer"]
    room = min(budget - SAFETY_MARGIN[0], budget / (1 + SAFETY_MARGIN[1])) - resident
    nothing = {"ple": None, "experts": None, "context": None, "prefill_step": None}

    def footprint(ple, experts):
        return offload_footprint(inventory, ple, experts)

    ple_choices = [ple_offload] if ple_offload != "auto" else ([None, 1.0] if ple_bytes else [None])
    expert_auto = expert_offload == "auto" and expert_bytes > 0
    expert_fixed = None if expert_offload == "auto" else expert_offload

    def choose(weights_room):
        for ple in ple_choices:
            if footprint(ple, expert_fixed) <= weights_room:
                return ple, expert_fixed
        if not expert_auto:
            return None
        ple = ple_choices[-1]
        keep = weights_room - footprint(ple, None) + expert_bytes - _expert_overhead(inventory)
        kept = min(per_layer, int(keep * per_layer // expert_bytes)) if keep > 0 else 0
        if kept * expert_bytes // per_layer < floor:
            return None
        return ple, (per_layer - kept) / per_layer

    first = prefill_step or PREFILL_STEPS[0]
    if context is None:
        target = _largest_context(lambda t: context_cost(t, first), room / 2, minimum, ceiling) or minimum
        if choose(room - context_cost(target, first)) is None:
            smallest = footprint(ple_choices[-1], 1.0 if expert_auto else expert_fixed)
            target = _largest_context(lambda t: context_cost(t, first), room - smallest, minimum, ceiling)
            if target is None:
                return nothing
    else:
        target = context
    chosen = choose(room - context_cost(target, first))
    if chosen is None:
        return nothing
    if context is None and expert_auto and chosen[1]:
        useful = min(max(minimum, _OFFLOAD_CONTEXT), ceiling)
        at_useful = choose(room - context_cost(useful, first))
        if at_useful is not None:
            chosen, target = at_useful, useful
    step = first
    if prefill_step is None and chosen[1]:
        # Offloaded expert bytes re-read per prompt token; disk speed scales every step alike.
        def per_token(candidate):
            return expert_bytes * candidate[1][1] / candidate[0] if candidate[1] and candidate[1][1] else 0

        best = (first, chosen)
        for candidate_step in PREFILL_STEPS[1:]:
            if candidate_step > target:
                break
            candidate = choose(room - context_cost(target, candidate_step))
            if candidate is not None and per_token((candidate_step, candidate)) < per_token(best):
                best = (candidate_step, candidate)
        step, chosen = best
    ple, experts = chosen
    if context is not None:
        return {"ple": ple, "experts": experts, "context": context, "prefill_step": step}
    fitted = _largest_context(lambda t: context_cost(t, step), room - footprint(ple, experts), target, ceiling)
    return {"ple": ple, "experts": experts, "context": fitted or target, "prefill_step": step}
