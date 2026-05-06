# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
"""
MLX -> torch simulation stub for Linux/CUDA hosts.

Mirrors the triton_stub.py pattern: install a MetaPathFinder so any
`import mlx.X.Y.Z` resolves to a permissive stub module, with the
explicitly-needed symbols routed to torch.

Activation: call `simulate_mlx_on_torch()` from `mlx_simulation.py`
*before* anything imports mlx (or set UNSLOTH_FORCE_MLX_SIMULATION=1
in the environment so `unsloth_zoo/__init__.py` does it for you).

Covers `mlx.core` (aliased as `mx`) plus its submodules:
  - mlx.core.random, mlx.core.linalg, mlx.core.fft
  - mlx.core.fast (rms_norm/layer_norm/sdpa/rope/metal_kernel/cuda_kernel)
  - mlx.core.metal, mlx.core.cuda, mlx.core.distributed

`mlx.nn`, `mlx.optimizers`, `mlx.utils`, `mlx_lm`, `mlx_vlm` live in
sibling stub files.
"""

from __future__ import annotations

import functools
import math
import os
import sys
import types
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec

import torch


# ---------------------------------------------------------------------------
# Permissive module + Noop sentinel — same shape as triton_stub.
# ---------------------------------------------------------------------------
class _PermissiveModule(types.ModuleType):
    """Module where any missing attribute returns a `_Noop`.

    The Noop raises NotImplementedError on call so missed translations
    fail loudly instead of silently no-opping.
    """
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _Noop(f"{self.__name__}.{name}")


class _Noop:
    """Permissive attribute stub that raises on call.

    Anything mlx actually exercises is implemented explicitly below; if
    a `_Noop` ever lands as a callable the call raises with the symbol
    name so we know what to add.
    """
    __slots__ = ("_name",)

    def __init__(self, name="stub"):
        self._name = name

    def __call__(self, *a, **kw):
        raise NotImplementedError(
            f"mlx-shim: {self._name!r} was called but is not implemented. "
            f"Add a concrete trampoline in unsloth_zoo/stubs/mlx_stub.py "
            f"(or a sibling stub file) and route to torch."
        )

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _Noop(f"{self._name}.{name}")

    def __repr__(self):
        return f"<mlx_stub._Noop: {self._name}>"

    def __bool__(self):
        return False

    def __mro_entries__(self, bases):
        return (object,)


def _make_module(name, attrs=None):
    mod = _PermissiveModule(name)
    mod.__path__ = []
    mod.__package__ = name
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    return mod


# ---------------------------------------------------------------------------
# Meta-path finder — catches any `import mlx.X.Y.Z` not seeded explicitly.
# ---------------------------------------------------------------------------
class _MLXLoader:
    def create_module(self, spec):
        return _make_module(spec.name)

    def exec_module(self, module):
        parts = module.__name__.rsplit(".", 1)
        if len(parts) == 2:
            parent = sys.modules.get(parts[0])
            if parent is not None:
                setattr(parent, parts[1], module)


_MLX_PREFIXES = ("mlx", "mlx_lm", "mlx_vlm")


class _MLXFinder(MetaPathFinder):
    """Intercepts any `import mlx*` or `mlx.X.Y` not already in sys.modules."""
    _loader = _MLXLoader()

    def find_spec(self, fullname, path, target=None):
        # Match top-level 'mlx', 'mlx_lm', 'mlx_vlm' or any submodule.
        for prefix in _MLX_PREFIXES:
            if fullname == prefix or fullname.startswith(prefix + "."):
                if fullname not in sys.modules:
                    return ModuleSpec(fullname, self._loader,
                                      origin="mlx_stub",
                                      is_package=True)
                return None
        return None


# ---------------------------------------------------------------------------
# Module-level metadata. `mx` is `mlx.core` in user code, so this module
# IS mlx.core (we register both `mlx` and `mlx.core` to point here).
# ---------------------------------------------------------------------------
__version__ = "0.31.2"
__path__ = []  # makes this module look like a package


def __getattr__(name):
    """Module-level __getattr__: any missing attr returns _Noop.

    This is what makes `mx.some_op_we_havent_implemented_yet` not crash
    at import time but raise with a clear name when actually called.
    """
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    return _Noop(f"mlx.core.{name}")


# ---------------------------------------------------------------------------
# Dtypes — alias to torch dtypes so .astype(mx.float32) works directly.
# ---------------------------------------------------------------------------
float32 = torch.float32
float16 = torch.float16
bfloat16 = torch.bfloat16
float64 = torch.float64
int8 = torch.int8
int16 = torch.int16
int32 = torch.int32
int64 = torch.int64
uint8 = torch.uint8
uint16 = getattr(torch, "uint16", torch.int32)
uint32 = getattr(torch, "uint32", torch.int64)
uint64 = getattr(torch, "uint64", torch.int64)
bool_ = torch.bool
complex64 = torch.complex64

# Dtype "kinds" used for issubdtype checks
floating = "floating"
integer = "integer"
signedinteger = "signedinteger"
unsignedinteger = "unsignedinteger"
inexact = "inexact"


# Special scalars
inf = math.inf
nan = math.nan


def issubdtype(dtype, kind):
    """Approximate mx.issubdtype(dtype, kind).

    MLX uses string-like dtype kinds; we accept either an mlx-style
    string ('floating', 'integer', etc.) or a torch dtype.
    """
    floating_dtypes = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    integer_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64,
                      torch.uint8)
    if kind == "floating" or kind is floating:
        return dtype in floating_dtypes
    if kind == "integer" or kind is integer:
        return dtype in integer_dtypes
    if kind == "inexact" or kind is inexact:
        return dtype in floating_dtypes
    if isinstance(kind, torch.dtype):
        return dtype == kind
    return False


def finfo(dtype):
    """mx.finfo(dtype) -> torch.finfo(dtype)."""
    return torch.finfo(dtype)


def iinfo(dtype):
    return torch.iinfo(dtype)


# ---------------------------------------------------------------------------
# Devices / streams — opaque handles. mx.gpu / mx.cpu are constants.
# ---------------------------------------------------------------------------
class _Device:
    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return f"Device({self._name})"

    def __eq__(self, other):
        return isinstance(other, _Device) and self._name == other._name

    def __hash__(self):
        return hash(self._name)


gpu = _Device("gpu")
cpu = _Device("cpu")


def default_device():
    if torch.cuda.is_available():
        return gpu
    return cpu


def set_default_device(device):
    pass  # no-op


# ---------------------------------------------------------------------------
# Tier 1: PR-B fresh symbols + memory probes. Most are no-ops on a Linux
# host that's pretending to be Apple Silicon.
# ---------------------------------------------------------------------------
def synchronize(*args, **kw):
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def clear_cache():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_memory_limit(limit, *args, **kw):
    return None


def set_cache_limit(limit, *args, **kw):
    return None


def set_wired_limit(limit, *args, **kw):
    return None


def get_peak_memory():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated()
    return 0


def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def device_info():
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return {
            "device_name": props.name,
            "max_recommended_working_set_size": int(props.total_memory),
            "memory_size": int(props.total_memory),
        }
    return {
        "device_name": "cpu",
        "max_recommended_working_set_size": 0,
        "memory_size": 0,
    }


def eval(*args):
    """mx.eval — force-realize lazy graph. No-op in eager torch."""
    return None


def async_eval(*args):
    return None


def stop_gradient(x):
    return x.detach() if isinstance(x, torch.Tensor) else x


def depends(*args):
    """mx.depends — explicit dependency edge. No-op in eager torch."""
    return None


# ---------------------------------------------------------------------------
# Tier 2: trivial passthroughs to torch.
# ---------------------------------------------------------------------------
# Reductions / elementwise / shape ops mapped 1:1.

class _ArrayMeta(type):
    """Metaclass that makes any torch.Tensor an instance of mx.array.

    Lets PR-A code's `isinstance(x, mx.array)` return True without
    requiring x to be a special subclass, and lets `mx.array | None`
    PEP 604 type unions work.
    """
    def __instancecheck__(cls, instance):
        return isinstance(instance, torch.Tensor)


class array(metaclass=_ArrayMeta):
    """mx.array — class for type/isinstance use; __new__ returns torch.Tensor.

    Calling `array(data, dtype=...)` constructs a torch.Tensor.  Used as
    `mx.array | None` in type annotations (PEP 604).  `isinstance(x,
    mx.array)` returns True for any torch.Tensor.
    """
    def __new__(cls, data=None, dtype=None):
        if data is None:
            return torch.tensor([])
        if isinstance(data, torch.Tensor):
            return data.to(dtype) if dtype is not None else data
        return torch.tensor(data, dtype=dtype)


def asarray(data, dtype=None):
    return array(data, dtype=dtype)


# Direct trampolines — keep the kwarg surface compatible.
def add(a, b, **kw): return torch.add(a, b)
def subtract(a, b, **kw): return torch.subtract(a, b)
def multiply(a, b, **kw): return torch.multiply(a, b)
def divide(a, b, **kw): return torch.divide(a, b)
def floor_divide(a, b, **kw): return torch.floor_divide(a, b)
def remainder(a, b, **kw): return torch.remainder(a, b)
def power(a, b, **kw): return torch.pow(a, b)
def negative(a, **kw): return torch.neg(a)
def abs(a, **kw): return torch.abs(a)


def matmul(a, b, **kw): return torch.matmul(a, b)
def addmm(c, a, b, alpha=1.0, beta=1.0, **kw): return torch.addmm(c, a, b, alpha=alpha, beta=beta)
def inner(a, b, **kw): return torch.inner(a, b)
def outer(a, b, **kw): return torch.outer(a, b)
def tensordot(a, b, dims=2, **kw): return torch.tensordot(a, b, dims=dims)
def einsum(eq, *operands, **kw): return torch.einsum(eq, *operands)


def exp(a, **kw): return torch.exp(a)
def expm1(a, **kw): return torch.expm1(a)
def log(a, **kw): return torch.log(a)
def log1p(a, **kw): return torch.log1p(a)
def log2(a, **kw): return torch.log2(a)
def log10(a, **kw): return torch.log10(a)
def logaddexp(a, b, **kw): return torch.logaddexp(a, b)
def logsumexp(a, axis=None, keepdims=False, **kw):
    if axis is None:
        return torch.logsumexp(a.flatten(), dim=0)
    return torch.logsumexp(a, dim=axis, keepdim=keepdims)
def softmax(a, axis=-1, **kw): return torch.softmax(a, dim=axis)
def sqrt(a, **kw): return torch.sqrt(a)
def rsqrt(a, **kw): return torch.rsqrt(a)
def square(a, **kw): return torch.square(a)
def reciprocal(a, **kw): return torch.reciprocal(a)
def sigmoid(a, **kw): return torch.sigmoid(a)
def sin(a, **kw): return torch.sin(a)
def cos(a, **kw): return torch.cos(a)
def tan(a, **kw): return torch.tan(a)
def sinh(a, **kw): return torch.sinh(a)
def cosh(a, **kw): return torch.cosh(a)
def tanh(a, **kw): return torch.tanh(a)
def erf(a, **kw): return torch.erf(a)
def erfinv(a, **kw): return torch.erfinv(a)
def ceil(a, **kw): return torch.ceil(a)
def floor(a, **kw): return torch.floor(a)
def round(a, **kw): return torch.round(a)
def sign(a, **kw): return torch.sign(a)
def conj(a, **kw): return torch.conj(a)


def maximum(a, b, **kw): return torch.maximum(a, b)
def minimum(a, b, **kw): return torch.minimum(a, b)
def clip(a, lo=None, hi=None, **kw): return torch.clamp(a, min=lo, max=hi)


def equal(a, b, **kw): return torch.eq(a, b)
def not_equal(a, b, **kw): return torch.ne(a, b)
def less(a, b, **kw): return torch.lt(a, b)
def less_equal(a, b, **kw): return torch.le(a, b)
def greater(a, b, **kw): return torch.gt(a, b)
def greater_equal(a, b, **kw): return torch.ge(a, b)
def isfinite(a, **kw): return torch.isfinite(a)
def isinf(a, **kw): return torch.isinf(a)
def isnan(a, **kw): return torch.isnan(a)
def isclose(a, b, rtol=1e-5, atol=1e-8, **kw): return torch.isclose(a, b, rtol=rtol, atol=atol)
def allclose(a, b, rtol=1e-5, atol=1e-8, **kw): return torch.allclose(a, b, rtol=rtol, atol=atol)
def array_equal(a, b, **kw): return torch.equal(a, b)


def logical_and(a, b, **kw): return torch.logical_and(a, b)
def logical_or(a, b, **kw): return torch.logical_or(a, b)
def logical_not(a, **kw): return torch.logical_not(a)


def where(cond, a, b, **kw): return torch.where(cond, a, b)


# Reductions: MLX uses `axis=` and `keepdims=`; torch uses `dim=` and `keepdim=`.
def _reduce_axis(fn, a, axis=None, keepdims=False, **kw):
    if axis is None:
        return fn(a)
    return fn(a, dim=axis, keepdim=keepdims)


def all(a, axis=None, keepdims=False, **kw): return _reduce_axis(torch.all, a, axis, keepdims)
def any(a, axis=None, keepdims=False, **kw): return _reduce_axis(torch.any, a, axis, keepdims)
def sum(a, axis=None, keepdims=False, **kw):
    if axis is None: return torch.sum(a)
    return torch.sum(a, dim=axis, keepdim=keepdims)
def mean(a, axis=None, keepdims=False, **kw):
    if axis is None: return torch.mean(a)
    return torch.mean(a, dim=axis, keepdim=keepdims)
def std(a, axis=None, keepdims=False, **kw):
    if axis is None: return torch.std(a)
    return torch.std(a, dim=axis, keepdim=keepdims)
def var(a, axis=None, keepdims=False, **kw):
    if axis is None: return torch.var(a)
    return torch.var(a, dim=axis, keepdim=keepdims)
def prod(a, axis=None, keepdims=False, **kw):
    if axis is None: return torch.prod(a)
    return torch.prod(a, dim=axis, keepdim=keepdims)


def max(a, axis=None, keepdims=False, **kw):
    if axis is None: return torch.amax(a)
    return torch.amax(a, dim=axis, keepdim=keepdims)


def min(a, axis=None, keepdims=False, **kw):
    if axis is None: return torch.amin(a)
    return torch.amin(a, dim=axis, keepdim=keepdims)


def argmax(a, axis=None, keepdims=False, **kw):
    return torch.argmax(a, dim=axis, keepdim=keepdims) if axis is not None else torch.argmax(a)


def argmin(a, axis=None, keepdims=False, **kw):
    return torch.argmin(a, dim=axis, keepdim=keepdims) if axis is not None else torch.argmin(a)


def argsort(a, axis=-1, **kw): return torch.argsort(a, dim=axis)


def cumsum(a, axis=-1, **kw):
    return torch.cumsum(a, dim=axis)


def cumprod(a, axis=-1, **kw):
    return torch.cumprod(a, dim=axis)


def topk(a, k, axis=-1, **kw):
    return torch.topk(a, k=k, dim=axis)


# Constructors
def zeros(shape, dtype=None, **kw):
    if isinstance(shape, int): shape = (shape,)
    return torch.zeros(*shape, dtype=dtype)
def zeros_like(a, dtype=None, **kw): return torch.zeros_like(a, dtype=dtype)
def ones(shape, dtype=None, **kw):
    if isinstance(shape, int): shape = (shape,)
    return torch.ones(*shape, dtype=dtype)
def ones_like(a, dtype=None, **kw): return torch.ones_like(a, dtype=dtype)
def full(shape, fill_value, dtype=None, **kw):
    if isinstance(shape, int): shape = (shape,)
    return torch.full(shape, fill_value, dtype=dtype)
def empty(shape, dtype=None, **kw):
    if isinstance(shape, int): shape = (shape,)
    return torch.empty(*shape, dtype=dtype)
def arange(*args, dtype=None, **kw):
    return torch.arange(*args, dtype=dtype)
def linspace(start, stop, num=50, dtype=None, **kw):
    return torch.linspace(start, stop, steps=num, dtype=dtype)
def eye(n, m=None, k=0, dtype=None, **kw):
    if m is None: m = n
    return torch.eye(n, m, dtype=dtype)
def identity(n, dtype=None, **kw):
    return torch.eye(n, dtype=dtype)


# Shape ops
def reshape(a, shape, **kw): return a.reshape(*shape) if isinstance(shape, (tuple, list)) else a.reshape(shape)
def flatten(a, start_axis=0, end_axis=-1, **kw): return torch.flatten(a, start_dim=start_axis, end_dim=end_axis)
def squeeze(a, axis=None, **kw):
    if axis is None: return torch.squeeze(a)
    return torch.squeeze(a, dim=axis)
def expand_dims(a, axis, **kw):
    """MLX accepts axis as int OR tuple of ints (insert several singleton dims)."""
    if isinstance(axis, (tuple, list)):
        # MLX inserts dims in the order given, indexing into the *expanded* shape.
        # Process ascending so each insertion's index is correct after prior ones.
        out = a
        for ax in sorted(int(x) for x in axis):
            out = torch.unsqueeze(out, dim=ax)
        return out
    return torch.unsqueeze(a, dim=axis)
def transpose(a, axes=None, **kw):
    if axes is None: return a.T if a.ndim == 2 else a.permute(*reversed(range(a.ndim)))
    return a.permute(*axes)
def permute_dims(a, axes, **kw): return a.permute(*axes)
def swapaxes(a, axis1, axis2, **kw): return torch.swapaxes(a, axis1, axis2)
def moveaxis(a, source, destination, **kw): return torch.movedim(a, source, destination)
def broadcast_to(a, shape, **kw): return torch.broadcast_to(a, shape)
def broadcast_arrays(*arrays, **kw): return torch.broadcast_tensors(*arrays)
def concatenate(arrays, axis=0, **kw): return torch.cat(arrays, dim=axis)
def concat(arrays, axis=0, **kw): return torch.cat(arrays, dim=axis)
def stack(arrays, axis=0, **kw): return torch.stack(arrays, dim=axis)
def split(a, indices_or_sections, axis=0, **kw):
    if isinstance(indices_or_sections, int):
        return torch.chunk(a, indices_or_sections, dim=axis)
    return torch.tensor_split(a, indices_or_sections, dim=axis)
def tile(a, reps, **kw):
    if isinstance(reps, int): reps = (reps,)
    return a.tile(*reps)
def repeat(a, repeats, axis=None, **kw):
    if axis is None:
        return a.flatten().repeat_interleave(repeats)
    return torch.repeat_interleave(a, repeats, dim=axis)
def pad(a, pad_width, mode="constant", constant_values=0, **kw):
    # MLX: pad_width is per-axis ((before, after), ...) or single int.
    # torch.nn.functional.pad expects flat (left, right) pairs in REVERSE axis order.
    import torch.nn.functional as F
    if isinstance(pad_width, int):
        pad_width = ((pad_width, pad_width),) * a.ndim
    elif isinstance(pad_width, (tuple, list)) and len(pad_width) > 0 and isinstance(pad_width[0], int):
        pad_width = (tuple(pad_width),) * a.ndim
    flat = []
    for before, after in reversed(list(pad_width)):
        flat.extend([before, after])
    return F.pad(a, flat, mode=mode, value=constant_values)
def roll(a, shift, axis=None, **kw):
    return torch.roll(a, shifts=shift, dims=axis)
def take(a, indices, axis=None, **kw):
    if axis is None:
        return a.flatten()[indices]
    return torch.index_select(a, dim=axis, index=indices)
def take_along_axis(a, indices, axis=None, **kw):
    # torch.take_along_dim demands int64 indices; MLX accepts int32.
    if indices.dtype != torch.int64:
        indices = indices.to(torch.int64)
    if axis is None:
        return torch.take_along_dim(a, indices, dim=None)
    return torch.take_along_dim(a, indices, dim=axis)


def diag(a, k=0, **kw): return torch.diag(a, diagonal=k)
def diagonal(a, offset=0, axis1=0, axis2=1, **kw): return torch.diagonal(a, offset=offset, dim1=axis1, dim2=axis2)
def trace(a, offset=0, axis1=0, axis2=1, **kw): return torch.diagonal(a, offset=offset, dim1=axis1, dim2=axis2).sum(-1)
def tri(n, m=None, k=0, dtype=None, **kw): return torch.tril(torch.ones(n, m if m is not None else n, dtype=dtype), diagonal=k)
def tril(a, k=0, **kw): return torch.tril(a, diagonal=k)
def triu(a, k=0, **kw): return torch.triu(a, diagonal=k)


def meshgrid(*xs, indexing="xy", **kw):
    return tuple(torch.meshgrid(*xs, indexing=indexing))


# Bitwise
def bitwise_and(a, b, **kw): return torch.bitwise_and(a, b)
def bitwise_or(a, b, **kw): return torch.bitwise_or(a, b)
def bitwise_xor(a, b, **kw): return torch.bitwise_xor(a, b)
def bitwise_invert(a, **kw): return torch.bitwise_not(a)
def left_shift(a, b, **kw): return torch.bitwise_left_shift(a, b)
def right_shift(a, b, **kw): return torch.bitwise_right_shift(a, b)


# Slice update — out-of-place index_put
def slice_update(arr, update, start_indices, axes=None, **kw):
    out = arr.clone()
    indices = start_indices if not isinstance(start_indices, torch.Tensor) else start_indices.tolist()
    if isinstance(indices, int):
        indices = (indices,)
    slices = tuple(slice(int(s), int(s) + dim) for s, dim in zip(indices, update.shape))
    out[slices] = update
    return out


def slice_(arr, start_indices, axes=None, slice_size=None, **kw):
    """mx.slice — out-of-place slice (functional)."""
    if isinstance(start_indices, torch.Tensor):
        start_indices = start_indices.tolist()
    if isinstance(slice_size, torch.Tensor):
        slice_size = slice_size.tolist()
    slices = tuple(slice(int(s), int(s) + int(d)) for s, d in zip(start_indices, slice_size))
    return arr[slices]
# Expose as `slice` on the module (without shadowing the builtin)
globals()["slice"] = slice_


def contiguous(a, **kw):
    return a.contiguous() if hasattr(a, "contiguous") else a


# ---------------------------------------------------------------------------
# Submodules: random, linalg, fft, fast, metal, cuda, distributed
# ---------------------------------------------------------------------------

# --- mx.random — JAX-style key plumbing on top of torch.Generator ----
def _rng_seed(s):
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _rng_key(s):
    return torch.tensor([int(s), 0], dtype=torch.uint32)


def _rng_split(key, num=2):
    base = int(key[0]) if hasattr(key, "__getitem__") else int(key)
    return [torch.tensor([base + i, 0], dtype=torch.uint32) for i in range(num)]


def _rng_normal(shape, dtype=torch.float32, key=None, loc=0.0, scale=1.0):
    if isinstance(shape, int):
        shape = (shape,)
    return torch.randn(*shape, dtype=dtype) * scale + loc


def _rng_uniform(low=0.0, high=1.0, shape=(), dtype=torch.float32, key=None):
    if isinstance(shape, int):
        shape = (shape,)
    return torch.empty(*shape, dtype=dtype).uniform_(low, high)


def _rng_randint(low, high, shape=(), dtype=torch.int32, key=None):
    if isinstance(shape, int):
        shape = (shape,)
    return torch.randint(low, high, shape, dtype=dtype)


def _rng_bernoulli(p=0.5, shape=(), key=None):
    if isinstance(shape, int):
        shape = (shape,)
    if shape:
        return torch.bernoulli(torch.full(shape, p)).bool()
    return torch.bernoulli(torch.tensor(p)).bool()


_rng_state = {"counter": 0}


def _rng_get_state(): return _rng_state.copy()


def _rng_set_state(s): _rng_state.update(s)


random = _make_module("mlx.core.random", {
    "seed": _rng_seed,
    "key": _rng_key,
    "split": _rng_split,
    "normal": _rng_normal,
    "uniform": _rng_uniform,
    "randint": _rng_randint,
    "bernoulli": _rng_bernoulli,
    "state": _rng_get_state,
    "save_state": _rng_get_state,
    "load_state": _rng_set_state,
})


# --- mx.linalg ---
def _la_norm(a, ord=None, axis=None, keepdims=False, **kw):
    return torch.linalg.norm(a, ord=ord, dim=axis, keepdim=keepdims)


linalg = _make_module("mlx.core.linalg", {
    "norm": _la_norm,
    "qr": lambda a, **kw: torch.linalg.qr(a),
    "svd": lambda a, **kw: torch.linalg.svd(a),
    "inv": lambda a, **kw: torch.linalg.inv(a),
    "pinv": lambda a, **kw: torch.linalg.pinv(a),
    "cholesky": lambda a, **kw: torch.linalg.cholesky(a),
    "solve": lambda a, b, **kw: torch.linalg.solve(a, b),
    "eig": lambda a, **kw: torch.linalg.eig(a),
    "eigh": lambda a, **kw: torch.linalg.eigh(a),
    "cross": lambda a, b, **kw: torch.linalg.cross(a, b),
})


# --- mx.fft ---
fft_mod = _make_module("mlx.core.fft", {
    "fft":   lambda a, **kw: torch.fft.fft(a),
    "ifft":  lambda a, **kw: torch.fft.ifft(a),
    "fft2":  lambda a, **kw: torch.fft.fft2(a),
    "ifft2": lambda a, **kw: torch.fft.ifft2(a),
    "fftn":  lambda a, **kw: torch.fft.fftn(a),
    "ifftn": lambda a, **kw: torch.fft.ifftn(a),
    "rfft":  lambda a, **kw: torch.fft.rfft(a),
    "irfft": lambda a, **kw: torch.fft.irfft(a),
    "fftshift":  lambda a, **kw: torch.fft.fftshift(a),
    "ifftshift": lambda a, **kw: torch.fft.ifftshift(a),
})


# --- mx.fast ---
def _fast_sdpa(q, k, v, scale=None, mask=None, **kw):
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, scale=scale, is_causal=False)


def _fast_rms_norm(x, weight, eps=1e-5, **kw):
    if hasattr(torch.nn.functional, "rms_norm"):
        # torch >= 2.4
        return torch.nn.functional.rms_norm(x, weight.shape if weight is not None else x.shape[-1:],
                                            weight=weight, eps=eps)
    rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)
    out = x * rms
    if weight is not None:
        out = out * weight
    return out


def _fast_layer_norm(x, weight, bias, eps=1e-5, **kw):
    return torch.nn.functional.layer_norm(
        x,
        weight.shape if weight is not None else x.shape[-1:],
        weight=weight, bias=bias, eps=eps,
    )


def _fast_metal_kernel(name=None, source=None, input_names=None, output_names=None,
                       header=None, ensure_row_contiguous=True, **kw):
    """Stub for mx.fast.metal_kernel.

    PR-A's CCE module checks `mx.metal.is_available()` first and falls
    back to pure-Python paths when False. We never reach this path in
    the simulation flow. Returning a callable that raises with the
    kernel name catches any unexpected callsite.
    """
    def _kernel(*args, **kwargs):
        raise NotImplementedError(
            f"mlx-shim: mx.fast.metal_kernel(name={name!r}) was invoked but "
            f"the simulation expects mx.metal.is_available()=False so the "
            f"pure-Python fallback fires instead. Check the calling code's "
            f"`if kernel is None` guard."
        )
    return _kernel


fast = _make_module("mlx.core.fast", {
    "scaled_dot_product_attention": _fast_sdpa,
    "rms_norm": _fast_rms_norm,
    "layer_norm": _fast_layer_norm,
    "metal_kernel": _fast_metal_kernel,
    "cuda_kernel": _fast_metal_kernel,
    "precompiled_cuda_kernel": _fast_metal_kernel,
})


# --- mx.metal ---
def _metal_is_available():
    return False  # we're pretending to be Apple but with no actual Metal


def _metal_start_capture(*args, **kw):
    return None


def _metal_stop_capture(*args, **kw):
    return None


def _metal_get_active_memory():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated()
    return 0


def _metal_device_info():
    return device_info()


metal = _make_module("mlx.core.metal", {
    "is_available": _metal_is_available,
    "start_capture": _metal_start_capture,
    "stop_capture": _metal_stop_capture,
    "get_active_memory": _metal_get_active_memory,
    "get_peak_memory": get_peak_memory,
    "reset_peak_memory": reset_peak_memory,
    "set_memory_limit": set_memory_limit,
    "set_cache_limit": set_cache_limit,
    "set_wired_limit": set_wired_limit,
    "device_info": _metal_device_info,
})


# --- mx.cuda ---
cuda = _make_module("mlx.core.cuda", {
    "is_available": lambda: torch.cuda.is_available(),
})


# --- mx.distributed ---
def _dist_is_available(): return False
def _dist_init(*args, **kw): return None
def _dist_pass(x, *args, **kw): return x
def _dist_send(*args, **kw): return None
def _dist_recv(*args, **kw): return None


distributed = _make_module("mlx.core.distributed", {
    "is_available": _dist_is_available,
    "init": _dist_init,
    "all_sum": _dist_pass,
    "all_max": _dist_pass,
    "all_min": _dist_pass,
    "all_gather": _dist_pass,
    "sum_scatter": _dist_pass,
    "send": _dist_send,
    "recv": _dist_recv,
    "recv_like": _dist_recv,
})


# ---------------------------------------------------------------------------
# Saving / loading — safetensors + npz
# ---------------------------------------------------------------------------
def save_safetensors(path, arrays, metadata=None, **kw):
    from safetensors.torch import save_file
    if not isinstance(arrays, dict):
        raise TypeError("save_safetensors expects a dict of name -> array")
    save_file(arrays, path, metadata=metadata)


def load(path, **kw):
    """Generic mx.load — sniff filetype."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    if path.endswith(".npz"):
        import numpy as np
        return dict(np.load(path))
    if path.endswith(".npy"):
        import numpy as np
        return np.load(path)
    raise ValueError(f"mlx-shim: mx.load: unsupported file type {path!r}")


def save(path, array, **kw):
    if path.endswith(".npy"):
        import numpy as np
        np.save(path, array.cpu().numpy() if isinstance(array, torch.Tensor) else array)
    else:
        raise ValueError(f"mlx-shim: mx.save: unsupported file type {path!r}")


def savez(path, **arrays):
    import numpy as np
    np.savez(path, **{k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                      for k, v in arrays.items()})


def savez_compressed(path, **arrays):
    import numpy as np
    np.savez_compressed(path, **{k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                                 for k, v in arrays.items()})


# ---------------------------------------------------------------------------
# Lazy-graph functions: compile, checkpoint, eval, vmap, grad, custom_function
# Bodies live in mlx_helpers.compile_passthrough and mlx_helpers.custom_function.
# ---------------------------------------------------------------------------
def compile(fn=None, inputs=None, outputs=None, shapeless=False, **kw):
    """mx.compile — identity decorator (or torch.compile passthrough via env)."""
    if fn is None:
        # Used as @mx.compile(...) — return a decorator
        def deco(f):
            if os.environ.get("UNSLOTH_MLX_SIM_TORCH_COMPILE", "0") == "1":
                return torch.compile(f, dynamic=shapeless)
            return f
        return deco
    if os.environ.get("UNSLOTH_MLX_SIM_TORCH_COMPILE", "0") == "1":
        return torch.compile(fn, dynamic=shapeless)
    return fn


def checkpoint(fn):
    def wrapped(*args, **kw):
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False, **kw)
    return wrapped


def custom_function(fn):
    """mx.custom_function — wraps fn as a closure-style differentiable op.

    Concrete implementation in mlx_helpers/custom_function.py (delegated
    via a late import to avoid circular imports during `simulate_mlx_on_torch`).
    """
    from .mlx_helpers.custom_function import make_custom_function
    return make_custom_function(fn)


def grad(fn, argnums=0, **kw):
    return torch.func.grad(fn, argnums=argnums)


def value_and_grad(fn, argnums=0, **kw):
    return torch.func.grad_and_value(fn, argnums=argnums)


def vmap(fn, in_axes=0, out_axes=0, **kw):
    return torch.func.vmap(fn, in_dims=in_axes, out_dims=out_axes)


def vjp(fn, primals, cotangents=None, **kw):
    out, vjp_fn = torch.func.vjp(fn, *primals)
    if cotangents is None:
        return out, vjp_fn
    return out, vjp_fn(cotangents)


def jvp(fn, primals, tangents, **kw):
    return torch.func.jvp(fn, primals, tangents)


# Quantization — affine implemented; mxfp4/nvfp4/mxfp8 raise.
def dequantize(w, scales, biases, group_size=None, bits=None, mode="affine",
               global_scale=None, dtype=None, **kw):
    from .mlx_helpers.quant import dequantize_affine
    if mode != "affine":
        raise NotImplementedError(
            f"mlx-shim: mx.dequantize mode={mode!r} not implemented; "
            f"only 'affine' is supported. Quantize on a Mac if you need {mode}."
        )
    return dequantize_affine(w, scales, biases, group_size, bits, dtype=dtype)


def quantized_matmul(x, w, scales, biases, transpose=False, group_size=64, bits=4,
                     mode="affine", **kw):
    from .mlx_helpers.quant import dequantize_affine
    if mode != "affine":
        raise NotImplementedError(
            f"mlx-shim: mx.quantized_matmul mode={mode!r} not implemented."
        )
    w_fp = dequantize_affine(w, scales, biases, group_size, bits, dtype=x.dtype)
    return x @ (w_fp.T if transpose else w_fp)


def quantize(*args, **kw):
    raise NotImplementedError(
        "mlx-shim: mx.quantize is not implemented in the simulation. "
        "Use real MLX on a Mac to produce quantized weights, then dequantize "
        "in the simulation via mx.dequantize."
    )


# ---------------------------------------------------------------------------
# inject_into_sys_modules — registers this module under `mlx`/`mlx.core`
# plus all named submodules.
# ---------------------------------------------------------------------------
def inject_into_sys_modules():
    """Install the meta-path finder and register seeded submodules."""
    import builtins
    if not builtins.any(isinstance(f, _MLXFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _MLXFinder())

    this = sys.modules[__name__]
    sys.modules.update({
        "mlx":                    this,
        "mlx.core":               this,        # `mx` is imported as `mlx.core`
        "mlx.core.random":        random,
        "mlx.core.linalg":        linalg,
        "mlx.core.fft":           fft_mod,
        "mlx.core.fast":          fast,
        "mlx.core.metal":         metal,
        "mlx.core.cuda":          cuda,
        "mlx.core.distributed":   distributed,
    })
    # `import mlx.core as mx` is equivalent to `import mlx.core; mx = mlx.core`
    # — the second step is ATTRIBUTE access on the `mlx` module.  We register
    # `mlx` and `mlx.core` as the same stub module, so `mlx.core` must resolve
    # to `this` via attribute access.  Without this, module-level __getattr__
    # would return a `_Noop("mlx.core.core")` instead.
    this.core = this
    this.random = random
    this.linalg = linalg
    this.fft = fft_mod
    this.fast = fast
    this.metal = metal
    this.cuda = cuda
    this.distributed = distributed
