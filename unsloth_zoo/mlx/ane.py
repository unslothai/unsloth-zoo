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

"""The Apple Neural Engine through macOS's private AppleNeuralEngine runtime, from ctypes, and the dense SwiGLU
MLP prefill split it serves.

Programs are MIL text compiled in memory, with fp16 tensors exchanged through IOSurfaces. The runtime is
undocumented and can change with any macOS release: every entry point here raises `AneError` rather than
guessing, and `available()` is False wherever the classes this module drives are missing.

The prefill split: the Neural Engine computes intermediate channels [0, k) of each eligible MLP from an int8
copy of their weights while the GPU computes the rest from the checkpoint's own; the partial outputs are
summed. Only full tiles of rows take it. The Neural Engine half is approximate, so the split is opt-in through
`UNSLOTH_MLX_ANE_PREFILL=1`.
"""

import collections
import concurrent.futures
import ctypes
import ctypes.util
import fcntl
import functools
import json
import logging
import os
import platform
import queue
import shutil
import struct
import subprocess
import sys
import threading
import time
import weakref
from contextlib import contextmanager
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

__all__ = ["AneError", "available", "Surface", "Program", "Blob", "Function",
           "AnePrefillStatus", "prepare_ane_prefill", "release_ane_prefill", "ane_prefill_mlp"]

_FRAMEWORK = "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine"
_QOS = 21  # user-interactive
_COMPILE_TRIES = 5
_WEIGHT_FILE = "@model_path/weights/weight.bin"


class AneError(RuntimeError):
    pass


class _Runtime:
    """ctypes bindings to the Objective-C runtime, IOSurface and the private framework."""

    def __init__(self):
        objc = ctypes.CDLL(ctypes.util.find_library("objc"))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.objc_autoreleasePoolPush.restype = ctypes.c_void_p
        objc.objc_autoreleasePoolPop.argtypes = [ctypes.c_void_p]
        self.objc = objc
        self.send = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
        ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
        surface = ctypes.CDLL("/System/Library/Frameworks/IOSurface.framework/IOSurface")
        surface.IOSurfaceCreate.restype = ctypes.c_void_p
        surface.IOSurfaceCreate.argtypes = [ctypes.c_void_p]
        surface.IOSurfaceGetBaseAddress.restype = ctypes.c_void_p
        surface.IOSurfaceGetBaseAddress.argtypes = [ctypes.c_void_p]
        self.surface = surface
        foundation = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        foundation.CFRelease.argtypes = [ctypes.c_void_p]
        self.cf_release = foundation.CFRelease
        ctypes.CDLL(_FRAMEWORK)
        self._prototypes = {}
        self._selectors = {}

    def cls(self, name):
        found = self.objc.objc_getClass(name.encode())
        if not found:
            raise AneError(f"the AppleNeuralEngine runtime has no class {name}")
        return found

    # A call is cast to its exact prototype: arm64 passes variadic arguments differently.
    def call(self, receiver, selector, *args, restype = ctypes.c_void_p, argtypes = None):
        argtypes = tuple(argtypes) if argtypes is not None else (ctypes.c_void_p,) * len(args)
        function = self._prototypes.get((restype, argtypes))
        if function is None:
            function = ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *argtypes)(self.send)
            self._prototypes[(restype, argtypes)] = function
        sel = self._selectors.get(selector)
        if sel is None:
            sel = self._selectors[selector] = self.objc.sel_registerName(selector.encode())
        return function(receiver, sel, *args)

    def pool(self):
        return _Pool(self.objc)

    def string(self, text):
        return self.call(self.cls("NSString"), "stringWithUTF8String:", text.encode(), argtypes = [ctypes.c_char_p])

    def number(self, value):
        return self.call(self.cls("NSNumber"), "numberWithUnsignedLongLong:", value, argtypes = [ctypes.c_ulonglong])

    def data(self, payload):
        return self.call(self.cls("NSData"), "dataWithBytes:length:", payload, len(payload),
                         argtypes = [ctypes.c_char_p, ctypes.c_ulong])

    def dictionary(self, pairs):
        keys = (ctypes.c_void_p * len(pairs))(*(k for k, _ in pairs))
        values = (ctypes.c_void_p * len(pairs))(*(v for _, v in pairs))
        return self.call(self.cls("NSDictionary"), "dictionaryWithObjects:forKeys:count:", values, keys, len(pairs),
                         argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong])

    def array(self, items):
        values = (ctypes.c_void_p * len(items))(*items)
        return self.call(self.cls("NSArray"), "arrayWithObjects:count:", values, len(items),
                         argtypes = [ctypes.c_void_p, ctypes.c_ulong])

    def text(self, obj):
        if not obj:
            return "no error object"
        described = self.call(obj, "description")
        return ctypes.string_at(self.call(described, "UTF8String")).decode(errors = "replace")

    def retain(self, obj):
        return self.call(obj, "retain")

    def release(self, obj):
        self.call(obj, "release", restype = None)


class _Pool:
    def __init__(self, objc):
        self.objc = objc

    def __enter__(self):
        self.token = self.objc.objc_autoreleasePoolPush()

    def __exit__(self, *exc):
        self.objc.objc_autoreleasePoolPop(self.token)


@functools.cache
def _runtime():
    return _Runtime()


@functools.cache
def available():
    """True when the private runtime loads and reports a Neural Engine."""
    if sys.platform != "darwin":   # the framework lives in the dyld shared cache, not at its path on disk
        return False
    try:
        rt = _runtime()
        for name in ("_ANEInMemoryModelDescriptor", "_ANEInMemoryModel", "_ANERequest", "_ANEIOSurfaceObject"):
            rt.cls(name)
        return bool(rt.call(rt.cls("_ANEDeviceInfo"), "hasANE", restype = ctypes.c_bool))
    except (OSError, AneError):
        return False


class Surface:
    """An IOSurface the Neural Engine reads or writes, mapped into this process."""

    def __init__(self, nbytes):
        rt = _runtime()
        self.nbytes = -(-nbytes // 16384) * 16384
        with rt.pool():
            keys = ("kIOSurfaceWidth", "kIOSurfaceHeight", "kIOSurfaceBytesPerElement", "kIOSurfaceBytesPerRow",
                    "kIOSurfaceAllocSize", "kIOSurfacePixelFormat")
            values = (self.nbytes, 1, 1, self.nbytes, self.nbytes, 0)
            properties = rt.dictionary([(ctypes.c_void_p.in_dll(rt.surface, key).value, rt.number(value))
                                        for key, value in zip(keys, values)])
            self.ref = rt.surface.IOSurfaceCreate(properties)
            if not self.ref:
                raise AneError(f"IOSurfaceCreate failed for {self.nbytes} bytes")
            self.base = rt.surface.IOSurfaceGetBaseAddress(self.ref)
            self.handle = rt.retain(rt.call(rt.cls("_ANEIOSurfaceObject"), "objectWithIOSurface:", self.ref))
        self._finalizer = weakref.finalize(self, _free_surface, self.handle, self.ref)

    def array(self, dtype, shape):
        """A numpy view of the surface's memory; valid while this Surface is alive."""
        count = int(np.prod(shape))
        nbytes = np.dtype(dtype).itemsize * count
        if nbytes > self.nbytes:
            raise ValueError(f"{shape} {dtype} needs {nbytes} bytes, the surface holds {self.nbytes}")
        return np.frombuffer((ctypes.c_char * nbytes).from_address(self.base), dtype, count).reshape(shape)

    def write(self, source):
        """Copy a C-contiguous buffer (numpy, or an evaluated `mx.array` through `np.asarray`) to the surface."""
        source = np.asarray(source)
        if not source.flags.c_contiguous or source.nbytes > self.nbytes:
            raise ValueError("the source must be C-contiguous and fit the surface")
        ctypes.memmove(self.base, source.ctypes.data, source.nbytes)


def _free_surface(handle, ref):
    rt = _runtime()
    rt.release(handle)
    rt.cf_release(ref)


class Program:
    """A compiled and loaded MIL program.

    The runtime stages a program under `$TMPDIR/<its identifier>` and reads the compiled form from there for as
    long as it stays loaded, so the directory lives until `close()`; only the MIL and weight sources go after
    loading.
    """

    def __init__(self, mil, weights):
        rt = _runtime()
        if not available():
            raise AneError("no Neural Engine runtime on this machine")
        started = time.perf_counter()
        with rt.pool():
            files = rt.dictionary([(rt.string(_WEIGHT_FILE), rt.dictionary([
                (rt.string("offset"), rt.number(0)), (rt.string("data"), rt.data(weights))]))])
            descriptor = rt.call(rt.cls("_ANEInMemoryModelDescriptor"), "modelWithMILText:weights:optionsPlist:",
                                 rt.data(mil.encode()), files, None)
            model = descriptor and rt.call(rt.cls("_ANEInMemoryModel"), "inMemoryModelWithDescriptor:", descriptor)
            if not model:
                raise AneError("the runtime rejected the program description")
            identifier = ctypes.string_at(rt.call(rt.call(model, "hexStringIdentifier"), "UTF8String")).decode()
            self.directory = os.path.join(os.environ.get("TMPDIR", "/tmp"), identifier)
            self.marker = None
            self.cached = self._compiled(model)
            if not self.cached:
                # A program the runtime compiles here is served from this directory, which identical programs in
                # other processes share: each marks it, and the last one removes it.
                with _directory_lock(self.directory):
                    self.cached = self._compiled(model)
                    if not self.cached:
                        self.marker = self._compile(model, mil, weights)
            try:
                self._invoke(model, "loadWithQoS:options:error:", "load")
            except BaseException:
                if self.marker is not None:
                    with _directory_lock(self.directory):
                        _release_directory(self.directory, self.marker)
                raise
            self.model = rt.retain(model)
        self.seconds = time.perf_counter() - started
        self._finalizer = weakref.finalize(self, _unload, self.model, self.directory, self.marker)

    @staticmethod
    def _compiled(model):
        return bool(_runtime().call(model, "compiledModelExists", restype = ctypes.c_bool))

    def _compile(self, model, mil, weights):
        """Stage the sources, compile and drop them; returns this program's owner marker. Called with the lock held."""
        marker = os.path.join(self.directory, f".unsloth-{os.getpid()}-{id(self)}")
        sources = (os.path.join(self.directory, "model.mil"), os.path.join(self.directory, "weights", "weight.bin"))
        os.makedirs(os.path.dirname(sources[1]), exist_ok = True)
        open(marker, "w").close()
        try:
            with open(sources[0], "w") as file:
                file.write(mil)
            with open(sources[1], "wb") as file:
                file.write(weights)
            self._invoke(model, "compileWithQoS:options:error:", "compile")
        except BaseException:
            _release_directory(self.directory, marker)
            raise
        for source in sources:
            os.remove(source)
        return marker

    @staticmethod
    def _invoke(model, selector, what):
        rt = _runtime()
        for attempt in range(_COMPILE_TRIES):
            error = ctypes.c_void_p()
            if rt.call(model, selector, _QOS, rt.dictionary([]), ctypes.byref(error), restype = ctypes.c_bool,
                       argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]):
                return
            message = rt.text(error.value)
            # The compiler fails some programs nondeterministically; the same program builds on a retry.
            if "Compiler internal error" not in message and attempt:
                break
            time.sleep(0.2)
        raise AneError(f"{what} failed: {message}")

    def request(self, inputs, outputs):
        """A reusable request binding `inputs` and `outputs` (Surfaces) to the program's tensors in order."""
        return _Request(self, inputs, outputs)

    def close(self):
        self._finalizer()


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


@contextmanager
def _directory_lock(directory):
    with open(f"{directory}.lock", "w") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        yield


def _release_directory(directory, marker):
    """Drop `marker`; remove the directory once no live process marks it. Called with its lock held."""
    try:
        os.remove(marker)
        owners = [name for name in os.listdir(directory) if name.startswith(".unsloth-")]
    except OSError:
        return
    if not any(_alive(int(name.split("-")[1])) for name in owners):
        shutil.rmtree(directory, ignore_errors = True)


def _unload(model, directory, marker):
    rt = _runtime()
    error = ctypes.c_void_p()
    with rt.pool():
        rt.call(model, "unloadWithQoS:error:", _QOS, ctypes.byref(error), restype = ctypes.c_bool,
                argtypes = [ctypes.c_uint, ctypes.c_void_p])
    rt.release(model)
    if marker is not None:
        with _directory_lock(directory):
            _release_directory(directory, marker)


class _Request:
    def __init__(self, program, inputs, outputs):
        rt = _runtime()
        self.program, self.surfaces = program, (tuple(inputs), tuple(outputs))  # kept alive with the request
        with rt.pool():
            request = rt.call(
                rt.cls("_ANERequest"),
                "requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:",
                rt.array([s.handle for s in inputs]), rt.array([rt.number(i) for i in range(len(inputs))]),
                rt.array([s.handle for s in outputs]), rt.array([rt.number(i) for i in range(len(outputs))]),
                None, None, rt.number(0))
            if not request:
                raise AneError("the runtime rejected the request")
            self.handle = rt.retain(request)
        self._finalizer = weakref.finalize(self, rt.release, self.handle)

    def run(self):
        """Evaluate once, blocking until the outputs are written. ctypes drops the GIL for the call."""
        rt = _runtime()
        error = ctypes.c_void_p()
        with rt.pool():
            ok = rt.call(self.program.model, "evaluateWithQoS:options:request:error:", _QOS, rt.dictionary([]),
                         self.handle, ctypes.byref(error), restype = ctypes.c_bool,
                         argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p])
            if not ok:
                raise AneError(f"evaluate failed: {rt.text(error.value)}")


class Blob:
    """A program's weight file: a 64-byte header, then each tensor after its own 64-byte header, 64-byte aligned."""

    def __init__(self):
        header = bytearray(64)
        header[0], header[4] = 1, 2
        self.chunks, self.size = [bytes(header)], 64

    def add(self, array):
        """Append a tensor; returns the offset a MIL BLOBFILE reference names."""
        payload = np.ascontiguousarray(array).tobytes()
        offset, pad = self.size, -len(payload) % 64
        self.chunks += [struct.pack("<IIQQ", 0xDEADBEEF, 1, len(payload), offset + 64) + bytes(40), payload, bytes(pad)]
        self.size += 64 + len(payload) + pad
        return offset

    def bytes(self):
        return b"".join(self.chunks)


def _shape(*dims):
    return ", ".join(map(str, dims))


class Function:
    """Builds one MIL function: values are named in order of definition and typed fp16 unless stated."""

    def __init__(self, name, **inputs):
        self.name, self.inputs, self.lines, self.count = name, inputs, [], 0

    def value(self, type_, expression):
        self.count += 1
        name = f"v{self.count}"
        tag = 'name=string("%s")' % name
        self.lines.append(f"    {type_} {name} = {expression.format(name = tag)};")
        return name

    def const(self, type_, literal):
        return self.value(type_, f"const()[{{name}}, val={type_}({literal})]")

    def op(self, op, shape, **operands):
        args = ", ".join(f"{key}={value}" for key, value in operands.items())
        return self.value(f"tensor<fp16, [{_shape(*shape)}]>", f"{op}({args})[{{name}}]")

    def int8_weight(self, blob, codes, scales):
        """A [rows, cols, 1, 1] fp16 weight stored as int8 `codes` with one fp16 scale per row."""
        rows, cols = codes.shape
        codes_at, scales_at = blob.add(codes.astype(np.int8)), blob.add(scales.astype(np.float16))
        return self.value(
            f"tensor<fp16, [{_shape(rows, cols, 1, 1)}]>",
            "constexpr_affine_dequantize()[axis=int32(0), {name}, "
            f"quantized_data=tensor<int8, [{_shape(rows, cols, 1, 1)}]>(BLOBFILE(path=string(\"{_WEIGHT_FILE}\"), "
            f"offset=uint64({codes_at}))), scale=tensor<fp16, [{rows}]>(BLOBFILE(path=string(\"{_WEIGHT_FILE}\"), "
            f"offset=uint64({scales_at}))), zero_point=int8(0)]")

    def conv1x1(self, x, weight, channels, rows):
        """`weight @ x` for x laid out [1, in_channels, 1, rows]."""
        if not hasattr(self, "_conv"):
            self._conv = dict(
                dilations = self.const("tensor<int32, [2]>", "[1, 1]"), groups = self.const("int32", "1"),
                pad = self.const("tensor<int32, [4]>", "[0, 0, 0, 0]"), pad_type = self.const("string", '"valid"'),
                strides = self.const("tensor<int32, [2]>", "[1, 1]"))
        return self.op("conv", (1, channels, 1, rows), **self._conv, weight = weight, x = x)

    def slice_channels(self, x, start, channels, rows):
        begin = self.const("tensor<int32, [4]>", f"[0, {start}, 0, 0]")
        size = self.const("tensor<int32, [4]>", f"[1, {channels}, 1, {rows}]")
        return self.op("slice_by_size", (1, channels, 1, rows), x = x, begin = begin, size = size)

    def program(self, outputs):
        arguments = ", ".join(f"{type_} {name}" for name, type_ in self.inputs.items())
        return ('program(1.3)\n[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, '
                '{"coremlc-version", "3505.4.1"}})]\n{\n'
                f"  func {self.name}<ios18>({arguments}) {{\n" + "\n".join(self.lines) +
                f"\n  }} -> ({', '.join(outputs)});\n}}\n")


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


def _mlp_shape(mlp):
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
    blob, fn = Blob(), Function("main", x = f"tensor<fp16, [1, {hidden}, 1, {tile}]>")
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
        self.input, self.output = Surface(hidden * tile * 2), Surface(hidden * tile * 2)


class _Split:
    def __init__(self, mlp, k, tile, surfaces, program = None):
        gate, up, down = _projections(mlp)
        self.k, self.tile, self.hidden, self.surfaces = k, tile, down.weight.shape[0], surfaces
        self.program = program or Program(*_program(mlp, k, tile))
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
            raise AneError("the Neural Engine produced a non-finite partial output")
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
        except (AneError, concurrent.futures.TimeoutError) as error:
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
    if not available():
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
        shape = collections.Counter(map(_mlp_shape, mlps)).most_common(1)[0][0]
        mlps = [m for m in mlps if _mlp_shape(m) == shape]
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
            except (AneError, concurrent.futures.TimeoutError) as error:
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
        except AneError as error:
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
