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
Bitsandbytes stub for Apple Silicon / MLX.

Any `import bitsandbytes.X.Y` auto-resolves to a permissive stub module.
Only injected on macOS ARM64 with MLX (gated in unsloth_zoo/__init__.py).
"""

import types
import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec


class _PermissiveModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _Noop(f"{self.__name__}.{name}")

class _Noop:
    """Permissive attribute-only stub. Calling it raises loudly so silent
    None-returns can't corrupt downstream tensors (e.g. a previous version
    let ``bnb.functional.quantize_4bit(weight, ...)`` produce ``None``).
    Optional-feature probes that use ``hasattr`` or ``if bnb.foo`` still
    work via ``__getattr__`` and ``__bool__``.
    """
    def __init__(self, name="stub"): self._name = name
    def __call__(self, *a, **kw):
        raise NotImplementedError(
            f"Unsloth: '{self._name}' was called on Apple Silicon / MLX, "
            f"where bitsandbytes is stubbed out. This usually means the "
            f"caller hit a CUDA-only code path that should be guarded by "
            f"a device check before reaching here."
        )
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _Noop(f"{self._name}.{name}")
    def __bool__(self): return False


def _make_module(name, attrs=None):
    mod = _PermissiveModule(name)
    mod.__path__ = []
    mod.__package__ = name
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    return mod


class _BnbLoader:
    def create_module(self, spec): return _make_module(spec.name)
    def exec_module(self, module):
        parts = module.__name__.rsplit(".", 1)
        if len(parts) == 2:
            parent = sys.modules.get(parts[0])
            if parent is not None:
                setattr(parent, parts[1], module)


class _BnbFinder(MetaPathFinder):
    _loader = _BnbLoader()
    def find_spec(self, fullname, path, target=None):
        if fullname == "bitsandbytes" or fullname.startswith("bitsandbytes."):
            if fullname not in sys.modules:
                return ModuleSpec(fullname, self._loader,
                                  origin="bitsandbytes_stub",
                                  is_package=True)
        return None


__version__ = "0.46.0"
__path__ = []

def __getattr__(name):
    """Module-level __getattr__ (Python 3.7+): any missing attr returns a _Noop."""
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    return _Noop(f"bitsandbytes.{name}")


def inject_into_sys_modules():
    if not any(isinstance(f, _BnbFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _BnbFinder())
    sys.modules["bitsandbytes"] = sys.modules[__name__]
