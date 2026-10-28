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

"""The Apple Neural Engine through macOS's private AppleNeuralEngine runtime, from ctypes.

Programs are MIL text compiled in memory, with fp16 tensors exchanged through IOSurfaces. The runtime is
undocumented and can change with any macOS release: every entry point here raises `AneError` rather than
guessing, and `available()` is False wherever the classes this module drives are missing.
"""

import ctypes
import ctypes.util
import fcntl
import functools
import os
import shutil
import struct
import sys
import time
import weakref
from contextlib import contextmanager

import numpy as np

__all__ = ["AneError", "available", "Surface", "Program", "Blob", "Function"]

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
