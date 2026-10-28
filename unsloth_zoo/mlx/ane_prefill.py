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

"""Dense SwiGLU MLP prefill split between the Neural Engine and the GPU.

The Neural Engine computes intermediate channels [0, k) from an int8 copy of their weights while the GPU computes
the rest from the checkpoint's own; the partial outputs are summed. Only full tiles of rows take the split. The
Neural Engine half is approximate, so this is opt-in through `UNSLOTH_MLX_ANE_PREFILL=1`.
"""

import collections
import concurrent.futures
import functools
import json
import logging
import os
import platform
import queue
import subprocess
import threading
import time
from contextlib import contextmanager
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import ane

__all__ = ["AnePrefillStatus", "prepare_ane_prefill", "release_ane_prefill", "ane_prefill_mlp"]

logger = logging.getLogger(__name__)

OPT_IN = "UNSLOTH_MLX_ANE_PREFILL"
_TILE = 2048
_ALIGN = 128            # channel granularity of the split
_DOWN_SLAB = 4608       # widest down-projection K the Neural Engine runs at full rate
_ACT_SCALE = 1 / 16     # keeps the fp16 activation and its down projection in range
_MIN_GAIN = 0.05        # below this the split is not worth its approximation
_WATCHDOG = 5.0         # seconds a Neural Engine evaluation may take before the split is abandoned
_STATE = "_unsloth_ane_split"
_CACHE_PATH = os.path.join(os.path.expanduser("~"), ".cache", "unsloth", "mlx_ane_prefill.json")
_LOCK = threading.RLock()
_CLASSES = {}
_ABANDONED = []


class AnePrefillStatus(NamedTuple):
    layers: int
    share: float = 0.0
    gpu_ms: float = 0.0
    split_ms: float = 0.0
    seconds: float = 0.0
    reason: str = ""


class _Worker:
    """The thread Neural Engine evaluations run on; a daemon, so a hung evaluation cannot hold up interpreter exit."""

    def __init__(self):
        self.jobs = queue.SimpleQueue()
        threading.Thread(target = self._serve, name = "unsloth-ane", daemon = True).start()

    def _serve(self):
        while True:
            function, future = self.jobs.get()
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(function())
                except BaseException as error:
                    future.set_exception(error)

    def run(self, function):
        """`function()` on the worker; raises `concurrent.futures.TimeoutError` after the watchdog."""
        future = concurrent.futures.Future()
        self.jobs.put((function, future))
        return future.result(timeout = _WATCHDOG)


@functools.cache
def _worker():
    return _Worker()


def _projections(mlp):
    return tuple(getattr(mlp, name, None) for name in ("gate_proj", "up_proj", "down_proj"))


def _eligible(mlp):
    gate, up, down = _projections(mlp)
    # Subclasses too: the dense-prefill and NAX scopes swap the projections' classes while they are entered.
    if mlp.training or not all(isinstance(p, nn.QuantizedLinear) for p in (gate, up, down)):
        return False
    if any("bias" in p or getattr(p, "mode", "affine") != "affine" for p in (gate, up, down)):
        return False
    if len({(p.group_size, p.bits) for p in (gate, up, down)}) != 1:
        return False
    channels, hidden = gate.weight.shape[0], gate.scales.shape[1] * gate.group_size
    return (up.weight.shape[0] == channels and down.weight.shape[0] == hidden
            and down.scales.shape[1] * down.group_size == channels and channels % _ALIGN == 0)


def _shape(mlp):
    gate = _projections(mlp)[0]
    return gate.scales.shape[1] * gate.group_size, gate.weight.shape[0]


def _computes_swiglu(mlp):
    """The module's own call is exactly down(silu(gate(x)) * up(x)), checked bit for bit on its own projections."""
    gate, up, down = _projections(mlp)
    x = mx.random.normal((1, 8, gate.scales.shape[1] * gate.group_size), key = mx.random.key(0)).astype(gate.scales.dtype)
    return mx.array_equal(mlp(x), down(nn.silu(gate(x)) * up(x))).item()


def _dequantized(p):
    return mx.dequantize(p.weight, p.scales, p.biases, group_size = p.group_size, bits = p.bits).astype(mx.float32)


def _int8_rows(w):
    scales = mx.maximum(mx.abs(w).max(axis = 1), 1e-8) / 127
    codes = mx.clip(mx.round(w / scales[:, None]), -127, 127).astype(mx.int8)
    return np.array(codes), np.array(scales.astype(mx.float16))


def _down_slabs(k):
    return next(n for n in range(1, k // _ALIGN + 1) if k % n == 0 and k // n <= _DOWN_SLAB)


def _program(mlp, k, tile):
    """MIL for channels [0, k): down[:, :k] @ (silu(gate[:k] @ x) * up[:k] @ x), x laid out [1, hidden, 1, tile]."""
    gate, up, down = map(_dequantized, _projections(mlp))
    hidden = down.shape[0]
    blob, fn = ane.Blob(), ane.Function("main", x = f"tensor<fp16, [1, {hidden}, 1, {tile}]>")
    gate_up = fn.conv1x1("x", fn.int8_weight(blob, *_int8_rows(mx.concatenate([gate[:k], up[:k]]))), 2 * k, tile)
    g, u = fn.slice_channels(gate_up, 0, k, tile), fn.slice_channels(gate_up, k, k, tile)
    shape = (1, k, 1, tile)
    # silu(g) = g * (1 + tanh(g / 2)) / 2: the Neural Engine's tanh is more accurate than its sigmoid.
    half = fn.const("fp16", "0.5")
    sig = fn.op("add", shape, x = fn.op("mul", shape, x = fn.op("tanh", shape, x = fn.op("mul", shape, x = g, y = half)),
                                        y = half), y = half)
    scaled_up = fn.op("mul", shape, x = u, y = fn.const("fp16", repr(_ACT_SCALE)))
    act = fn.op("mul", shape, x = fn.op("mul", shape, x = g, y = sig), y = scaled_up)
    slabs = _down_slabs(k)
    width, total = k // slabs, None
    for i in range(slabs):
        weight = fn.int8_weight(blob, *_int8_rows(down[:, i * width:(i + 1) * width]))
        part = fn.conv1x1(fn.slice_channels(act, i * width, width, tile), weight, hidden, tile)
        total = part if total is None else fn.op("add", (1, hidden, 1, tile), x = total, y = part)
    y = fn.op("mul", (1, hidden, 1, tile), x = total, y = fn.const("fp16", repr(1 / _ACT_SCALE)))
    return fn.program([y]), blob.bytes()


def _remainder(p, rows = None, columns = None):
    """A QuantizedLinear over rows [rows:] or input columns [columns:] of `p`'s quantized tensors."""
    weight, scales, biases = p.weight, p.scales, p.biases
    if rows is not None:
        weight, scales, biases = weight[rows:], scales[rows:], biases[rows:]
    if columns is not None:
        weight = mx.contiguous(weight[:, columns * p.bits // 32:])
        scales, biases = mx.contiguous(scales[:, columns // p.group_size:]), mx.contiguous(biases[:, columns // p.group_size:])
    piece = nn.QuantizedLinear(p.group_size, 1, bias = False, group_size = p.group_size, bits = p.bits)
    piece.weight, piece.scales, piece.biases = weight, scales, biases
    piece.freeze()
    return piece


class _Surfaces:
    def __init__(self, hidden, tile):
        self.input, self.output = ane.Surface(hidden * tile * 2), ane.Surface(hidden * tile * 2)


class _Split:
    def __init__(self, mlp, k, tile, surfaces, program = None):
        gate, up, down = _projections(mlp)
        self.k, self.tile, self.hidden, self.surfaces = k, tile, down.weight.shape[0], surfaces
        self.program = program or ane.Program(*_program(mlp, k, tile))
        self.request = self.program.request([surfaces.input], [surfaces.output])
        self.rest = (_remainder(gate, rows = k), _remainder(up, rows = k), _remainder(down, columns = k))
        mx.eval([p.parameters() for p in self.rest])
        self.sources = tuple((p, p.weight, p.scales, p.biases) for p in (gate, up, down))
        self.failed = False

    def serves(self, mlp):
        """Whether `mlp` still has the projections and quantized tensors this split was built from."""
        # The object first: an adapter's replacement may have no quantized tensors at all.
        return all(p is q and p.weight is w and p.scales is s and p.biases is b
                   for p, (q, w, s, b) in zip(_projections(mlp), self.sources))

    def close(self):
        self.program.close()

    def evaluate(self):
        """Run the Neural Engine half on the worker. A timed-out evaluation may still be running, so its native objects
        are never released, not even at exit, and since it holds the worker every split stops."""
        try:
            _worker().run(self.request.run)
        except concurrent.futures.TimeoutError:
            for owner in (self.program, self.request, self.surfaces.input, self.surfaces.output):
                owner._finalizer.detach()
            _ABANDONED.append(self)
            raise

    def __call__(self, x):
        """One tile: x [tile, hidden] -> [tile, hidden]."""
        planes = mx.contiguous(x.T.astype(mx.float16))
        mx.eval(planes)
        self.surfaces.input.write(planes)
        gate, up, down = self.rest
        y = down(nn.silu(gate(x)) * up(x))
        mx.async_eval(y)
        self.evaluate()
        partial = mx.array(self.surfaces.output.array(np.float16, (self.hidden, self.tile)))
        if not mx.all(mx.isfinite(partial)).item():
            raise ane.AneError("the Neural Engine produced a non-finite partial output")
        return (y.astype(mx.float32) + partial.T.astype(mx.float32)).astype(y.dtype)


def _split_call(base):
    def call(self, x):
        with _LOCK:   # held from the lookup on, so a release cannot unload the split under this call
            return routed(self, x)

    def routed(self, x):
        split = self.__dict__.get(_STATE)
        if split is not None and not split.serves(self):   # e.g. an adapter replaced a projection
            self.__dict__.pop(_STATE, None)
            split.close()
            split = None
        rows = x.size // x.shape[-1]
        if split is None or split.failed or _ABANDONED or self.training or rows < split.tile:
            return base.__call__(self, x)
        flat, tiles = x.reshape(rows, x.shape[-1]), rows // split.tile
        try:
            parts = [split(flat[i * split.tile:(i + 1) * split.tile]) for i in range(tiles)]
        except (ane.AneError, concurrent.futures.TimeoutError) as error:
            split.failed = True
            logger.warning("Neural Engine prefill split failed (%s); this layer uses the GPU from now on", error)
            return base.__call__(self, x)
        if tiles * split.tile < rows:
            parts.append(base.__call__(self, flat[tiles * split.tile:][None]).reshape(-1, x.shape[-1]))
        return mx.concatenate(parts).reshape(x.shape)
    return call


def _split_class(base):
    cls = _CLASSES.get(base)
    if cls is None:
        cls = _CLASSES[base] = type(f"_AneSplit{base.__name__}", (base,), {"__call__": _split_call(base)})
        cls._unsloth_ane_native = base
    return cls


def _machine_key(mlp, tile):
    gate, _, down = _projections(mlp)
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output = True, text = True,
                              timeout = 5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        chip = platform.machine()
    build = platform.mac_ver()[0]
    return (f"{chip}|macOS {build}|mlx {mx.__version__}|{down.weight.shape[0]}x{gate.weight.shape[0]}|"
            f"q{gate.bits}g{gate.group_size}|{gate.scales.dtype}|tile {tile}")


def _load_cache():
    try:
        with open(_CACHE_PATH) as file:
            return json.load(file)
    except (OSError, ValueError):
        return {}


def _store_cache(key, entry):
    cache = _load_cache()
    cache[key] = entry
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok = True)
    temporary = f"{_CACHE_PATH}.{os.getpid()}"
    with open(temporary, "w") as file:
        json.dump(cache, file, indent = 1, sort_keys = True)
    os.replace(temporary, _CACHE_PATH)


def _median_ms(function, repeats = 7, warm_seconds = 1.0):
    started = time.perf_counter()
    while time.perf_counter() - started < warm_seconds:   # an idle GPU needs a moment to reach its clocks
        mx.eval(function())
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        mx.eval(function())
        times.append(time.perf_counter() - started)
    return float(np.median(times) * 1e3)


def _aligned(channels, share):
    return int(min(max(round(channels * share / _ALIGN), 1), channels // _ALIGN - 1)) * _ALIGN


def _calibrate(model, mlp, tile, surfaces):
    """(share, GPU ms, split ms, split at that share) for one tile: a tile costs about max(g (1 - s), a s), lowest at
    s = g / (g + a); the chosen share is timed end to end since both units share memory bandwidth."""
    from .inference import dense_prefill_linear   # the GPU route a generation would take

    gate = _projections(mlp)[0]
    channels, hidden = gate.weight.shape[0], gate.scales.shape[1] * gate.group_size
    x = (mx.random.normal((tile, hidden), key = mx.random.key(1)) * 0.5).astype(gate.scales.dtype)
    with dense_prefill_linear(model, _modules = tuple(mlp.named_modules())):
        gpu = _median_ms(lambda: mlp(x[None]))
    probe = _Split(mlp, _aligned(channels, 0.5), tile, surfaces)

    def neural_engine():
        probe.evaluate()
        return mx.array(0)

    per_share = _median_ms(neural_engine) / (probe.k / channels)
    k = _aligned(channels, gpu / (gpu + per_share))
    split = probe if k == probe.k else _Split(mlp, k, tile, surfaces)
    split_ms = _median_ms(lambda: split(x))
    return k / channels, gpu, split_ms, split


def prepare_ane_prefill(model, *, tile = _TILE, share = None):
    """Compile the Neural Engine half of every eligible dense SwiGLU MLP in `model` and attach it.

    `share` fixes the fraction of intermediate channels the Neural Engine takes; by default it is calibrated on the
    first layer and cached per machine, macOS build, MLX version and shape, and a split that is not at least 5% faster
    than the GPU alone is not attached. Returns an `AnePrefillStatus`; `layers == 0` with a `reason` when nothing was
    attached.
    """
    started = time.perf_counter()
    if not ane.available():
        return AnePrefillStatus(0, reason = "no Neural Engine runtime")
    if mx.default_device() != mx.gpu:
        return AnePrefillStatus(0, reason = "the default device is not the GPU")
    if tile % 32:
        raise ValueError("tile must be a multiple of 32 rows")
    with _LOCK:
        mlps = []
        for _, module in model.named_modules():
            if module.__dict__.get(_STATE) is None and _projections(module)[0] is not None and _eligible(module):
                if all(m is not module for m in mlps):
                    mlps.append(module)
        mlps = [m for m in mlps if _computes_swiglu(m)]
        if not mlps:
            return AnePrefillStatus(0, reason = "no dense SwiGLU MLP with affine quantized projections")
        # The language model's shape: the layers share one calibration and one pair of surfaces, so a vision
        # tower's MLPs of another width stay native.
        shape = collections.Counter(map(_shape, mlps)).most_common(1)[0][0]
        mlps = [m for m in mlps if _shape(m) == shape]
        surfaces = _Surfaces(shape[0], tile)
        key, first = _machine_key(mlps[0], tile), None
        cached = _load_cache().get(key) if share is None else None
        if share is not None:
            gpu = split_ms = 0.0
        elif cached is not None:
            share, gpu, split_ms = cached["share"], cached["gpu_ms"], cached["split_ms"]
        else:
            try:
                share, gpu, split_ms, first = _calibrate(model, mlps[0], tile, surfaces)
            except (ane.AneError, concurrent.futures.TimeoutError) as error:
                return AnePrefillStatus(0, seconds = time.perf_counter() - started, reason = f"calibration failed: {error}")
            if split_ms > gpu * (1 - _MIN_GAIN):
                share = 0.0
            _store_cache(key, {"share": share, "gpu_ms": gpu, "split_ms": split_ms})
        if not share:
            return AnePrefillStatus(0, 0.0, gpu, split_ms, time.perf_counter() - started,
                                    f"the split ({split_ms:.1f} ms) is not 5% faster than the GPU ({gpu:.1f} ms)")
        attached = []
        try:
            for mlp in mlps:
                k = _aligned(_projections(mlp)[0].weight.shape[0], share)
                split = first if first is not None and mlp is mlps[0] and first.k == k else None
                mlp.__dict__[_STATE] = split or _Split(mlp, k, tile, surfaces)
                attached.append(mlp)
        except ane.AneError as error:
            logger.warning("Neural Engine prefill stopped after %d of %d layers: %s", len(attached), len(mlps), error)
        status = AnePrefillStatus(len(attached), share, gpu, split_ms, time.perf_counter() - started)
        logger.info("Neural Engine prefill: %d layers at a %.0f%% channel share (one-tile MLP %.1f -> %.1f ms), "
                    "prepared in %.1f s", status.layers, 100 * share, gpu, split_ms, status.seconds)
        return status


def release_ane_prefill(model):
    """Detach and unload every Neural Engine split in `model`."""
    with _LOCK:
        for _, module in model.named_modules():
            split = module.__dict__.pop(_STATE, None)
            if split is not None:
                split.close()


@contextmanager
def ane_prefill_mlp(model, *, _modules = None):
    """Route prefill tiles of the MLPs `prepare_ane_prefill` attached through their Neural Engine split.

    Prepares the model on first entry. Calls of fewer rows than one tile, training and a split that failed keep
    the native module.
    """
    changed = []
    try:
        with _LOCK:
            modules = _modules if _modules is not None else tuple(model.named_modules())
            if not getattr(model, "_unsloth_ane_prepared", False):
                model.__dict__["_unsloth_ane_prepared"] = True
                status = prepare_ane_prefill(model)
                if not status.layers:
                    logger.info("Neural Engine prefill is off: %s", status.reason)
            for _, module in modules:
                if module.__dict__.get(_STATE) is None or any(module is m for m in changed):
                    continue
                if module.__dict__.get("_unsloth_ane_scopes"):
                    module._unsloth_ane_scopes += 1
                else:
                    module.__class__ = _split_class(type(module))
                    module._unsloth_ane_scopes = 1
                changed.append(module)
        yield model
    finally:
        with _LOCK:
            for module in changed:
                module._unsloth_ane_scopes -= 1
                if not module._unsloth_ane_scopes:
                    module.__class__ = module._unsloth_ane_native
                    module.__dict__.pop("_unsloth_ane_scopes")
