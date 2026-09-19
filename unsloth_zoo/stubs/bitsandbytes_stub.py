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
Bitsandbytes stub for hosts that skip GPU init (gated in unsloth_zoo/__init__.py).

Any `import bitsandbytes.X.Y` auto-resolves to a permissive stub module. Injected only
when no real bitsandbytes is installed: shadowing a working one makes bnb-quantized
checkpoints unloadable.
"""

import importlib.util
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
    def __init__(self, *args, **kwargs):
        # Accept any args, incl. the (name, bases, namespace) triple Python
        # passes when a stubbed class is subclassed at import (e.g. Linear4bit).
        self._name = args[0] if args else kwargs.get("name", "stub")
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
    # Set on every stub module, including finder-minted ones: without a real attribute,
    # the permissive __getattr__ answers the stub check with a falsy _Noop and every
    # caller reads "this is a real wheel".
    mod.IS_UNSLOTH_STUB = True
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
# Lets callers tell this stub from real bitsandbytes (e.g. drift tests skip on it).
IS_UNSLOTH_STUB = True

def __getattr__(name):
    """Module-level __getattr__ (Python 3.7+): any missing attr returns a _Noop."""
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    return _Noop(f"bitsandbytes.{name}")


def real_bitsandbytes_available():
    """Whether a real (non-stub) bitsandbytes is installed.

    Locates it rather than importing it: importing pulls in torch, which costs about a
    second on the hosts that skip GPU init to avoid exactly that.
    """
    existing = sys.modules.get("bitsandbytes")
    if existing is not None:
        return not getattr(existing, "IS_UNSLOTH_STUB", False)
    try:
        spec = importlib.util.find_spec("bitsandbytes")
    except Exception:  # noqa: BLE001 -- an unlocatable install is not usable either
        return False
    # A namespace package (loaderless, e.g. a half-removed install) imports to an empty
    # module; standing aside for it leaves the caller with neither a wheel nor the stub.
    return spec is not None and spec.loader is not None


def inject_into_sys_modules():
    if not any(isinstance(f, _BnbFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _BnbFinder())
    sys.modules["bitsandbytes"] = sys.modules[__name__]
