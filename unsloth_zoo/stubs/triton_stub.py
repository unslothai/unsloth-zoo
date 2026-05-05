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

"""
Triton stub for Apple Silicon / MLX.

Any `import triton.X.Y.Z` auto-resolves to a permissive stub module.
Attribute access on unknown attrs returns a safe no-op.
Only injected on macOS ARM64 with MLX (gated in unsloth_zoo/__init__.py).
"""

import types
import math
import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec


# ---------------------------------------------------------------------------
# Permissive module — auto-returns safe defaults for missing attributes
# ---------------------------------------------------------------------------
class _PermissiveModule(types.ModuleType):
    """Module where any missing attribute returns a safe no-op."""
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _Noop(f"{self.__name__}.{name}")


class _Noop:
    """Permissive attribute-only stub that supports chained access and use
    as a base class. Calling it raises loudly: anything triton actually
    needs at module-load time (``triton.jit``, ``cdiv``, ``autotune``,
    decorators, dtypes) is defined explicitly elsewhere in this file and
    won't hit ``_Noop``. A call landing here means an unexpected CUDA
    code path was reached on Apple Silicon — fail loudly instead of
    returning ``None``.
    """
    def __init__(self, name="stub"):
        self._name = name
    def __call__(self, *a, **kw):
        raise NotImplementedError(
            f"Unsloth: '{self._name}' was called on Apple Silicon / MLX, "
            f"where triton is stubbed out. The caller likely hit a CUDA-only "
            f"kernel path that should be guarded before reaching here."
        )
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _Noop(f"{self._name}.{name}")
    def __repr__(self):
        return f"<triton_stub: {self._name}>"
    def __bool__(self):
        return False
    def __mro_entries__(self, bases):
        # When used as a base class (e.g. `class Foo(KernelInterface)`),
        # resolve to `object` so the class definition succeeds.
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
# Meta-path finder + loader (find_spec protocol — works Python 3.10+)
# ---------------------------------------------------------------------------
class _TritonLoader:
    """Loader that creates permissive stub modules."""
    def create_module(self, spec):
        return _make_module(spec.name)

    def exec_module(self, module):
        # Set as attr on parent so `triton.X.Y` attribute access also works
        parts = module.__name__.rsplit(".", 1)
        if len(parts) == 2:
            parent = sys.modules.get(parts[0])
            if parent is not None:
                setattr(parent, parts[1], module)


class _TritonFinder(MetaPathFinder):
    """Intercepts any `import triton` or `import triton.*`."""
    _loader = _TritonLoader()

    def find_spec(self, fullname, path, target=None):
        if fullname == "triton" or fullname.startswith("triton."):
            if fullname not in sys.modules:
                return ModuleSpec(fullname, self._loader,
                                  origin="triton_stub",
                                  is_package=True)
        return None


# ---------------------------------------------------------------------------
# Version & core API (used at module level in unsloth kernel files)
# ---------------------------------------------------------------------------
__version__ = "3.0.0"
__path__ = []  # Makes this module look like a package for `import triton.X`

def __getattr__(name):
    """Module-level __getattr__ (Python 3.7+): any missing attr returns a _Noop."""
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    return _Noop(f"triton.{name}")

def jit(fn=None, **kwargs):
    if fn is None:
        return lambda f: f
    return fn

def heuristics(values):
    def decorator(fn): return fn
    return decorator

def autotune(configs=None, key=None, **kwargs):
    def decorator(fn): return fn
    return decorator

def cdiv(a, b):
    return math.ceil(a / b)

def next_power_of_2(n):
    if n <= 0: return 1
    n -= 1
    n |= n >> 1; n |= n >> 2; n |= n >> 4
    n |= n >> 8; n |= n >> 16; n |= n >> 32
    return n + 1

class Config:
    def __init__(self, kwargs=None, num_warps=4, num_stages=2, **extra):
        self.kwargs = kwargs or {}
        self.num_warps = num_warps
        self.num_stages = num_stages


# ---------------------------------------------------------------------------
# triton.language — needs constexpr/dtype for kernel annotations
# ---------------------------------------------------------------------------
class _ConstExpr:
    def __init__(self, value=None): self.value = value
    def __class_getitem__(cls, item): return cls

class dtype:
    def __init__(self, name="void"): self.name = name

language = _make_module("triton.language", {
    "constexpr": _ConstExpr,
    "dtype": dtype,
    "float32": "float32", "float16": "float16", "bfloat16": "bfloat16",
    "int32": "int32", "int64": "int64", "int8": "int8",
    "uint8": "uint8", "int1": "int1",
})
language.math = language

# triton.runtime — needs driver.active for target detection
class _CurrentTarget:
    arch = 0; backend = "stub"

class _ActiveDriver:
    @staticmethod
    def get_current_target(): return _CurrentTarget()
    utils = _Noop("driver.utils")

runtime = _make_module("triton.runtime", {
    "driver": _make_module("triton.runtime.driver", {"active": _ActiveDriver()}),
    "errors": _make_module("triton.runtime.errors", {
        "OutOfResources": type("OutOfResources", (Exception,), {}),
    }),
})

# triton.backends — needs empty `backends` dict
backends = _make_module("triton.backends", {"backends": {}})


def inject_into_sys_modules():
    """Install finder and seed core modules."""
    if not any(isinstance(f, _TritonFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _TritonFinder())

    this = sys.modules[__name__]
    sys.modules.update({
        "triton":                this,
        "triton.language":       language,
        "triton.runtime":        runtime,
        "triton.runtime.driver": runtime.driver,
        "triton.runtime.errors": runtime.errors,
        "triton.backends":       backends,
    })
    # Everything else auto-created by _TritonFinder on demand
