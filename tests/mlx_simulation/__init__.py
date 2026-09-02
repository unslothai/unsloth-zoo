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
# Single entry point: simulate_mlx_on_torch()
"""
Run this BEFORE any code that imports `mlx`, `mlx_lm`, or `mlx_vlm`.

Usage:

    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()
    import unsloth   # now succeeds on Linux+CUDA, treating itself as Apple Silicon
"""

from __future__ import annotations


def simulate_mlx_on_torch(*, fake_apple_silicon: bool = True):
    """Install MLX-on-torch shims into sys.modules.

    Order matters:
      1. Spoof ``platform.system()`` / ``platform.machine()`` so PR-B's
         ``_IS_MLX`` gate (Darwin+arm64) activates on Linux hosts.
         Pass ``fake_apple_silicon=False`` to skip this if you only want
         MLX symbol routing without flipping the dispatch flag.
      2. Monkey-patch torch.Tensor with MLX-only methods (.astype,
         .expand_dims, .at[]) BEFORE any unsloth_zoo MLX module is
         imported, so module-level `mx.array` annotations and method
         calls resolve.
      3. mlx.core (and the MetaPathFinder) must be in place before any
         submodule import succeeds.
      4. mlx.nn / mlx.optimizers / mlx.utils inject after.
      5. mlx_lm / mlx_vlm inject last because they may transitively
         import mlx.core during their own setup.
    """
    # Force torch to fully load with the REAL platform.system() so its
    # native libs resolve correctly.  Only AFTER that do we spoof platform.
    import torch  # noqa: F401

    if fake_apple_silicon:
        _spoof_apple_silicon_platform()

    from .mlx_helpers.array_proxy import patch_tensor_with_mlx_methods
    patch_tensor_with_mlx_methods()

    from . import mlx_stub
    mlx_stub.inject_into_sys_modules()

    from . import mlx_utils_stub
    mlx_utils_stub.inject_into_sys_modules()

    from . import mlx_nn_stub
    mlx_nn_stub.inject_into_sys_modules()

    from . import mlx_optimizers_stub
    mlx_optimizers_stub.inject_into_sys_modules()

    from . import mlx_lm_stub
    mlx_lm_stub.inject_into_sys_modules()

    from . import mlx_vlm_stub
    mlx_vlm_stub.inject_into_sys_modules()


_PLATFORM_SPOOFED = False

# Only these see the fake host. The spoof is process-wide and permanent, so
# anything collected after an MLX test module inherits it: inductor caches its
# CPU vector ISA from platform.machine(), and "arm64" on x86_64 yields an empty
# list, after which every torch.compile emits uncompilable `at::vec` C++.
#
# Exact module names, not packages. Allow-listing all of `unsloth_zoo` still lied
# to unrelated host-sensitive code in it: unsloth_zoo.llama_cpp reads the host to
# pick a prebuilt archive, so a later llama.cpp call on Linux x86_64 would fetch
# the macOS arm64 build. These two modules are the only places the MLX gate reads
# the host; everything downstream of them consumes the cached boolean.
_SPOOF_CONSUMERS = frozenset({
    "unsloth",                  # unsloth/__init__.py::_is_mlx_available
    "unsloth_zoo.mlx.runtime",  # is_mlx_available
})


def _spoof_apple_silicon_platform():
    """Make platform.system()=='Darwin' and platform.machine()=='arm64'.

    Idempotent.  PR-B's _IS_MLX gate in unsloth/__init__.py uses these
    to decide between MLX and CUDA dispatch.

    Scoped to the immediate caller: only _SPOOF_CONSUMERS see the lie.
    """
    global _PLATFORM_SPOOFED
    if _PLATFORM_SPOOFED:
        return
    _PLATFORM_SPOOFED = True

    import platform
    import sys

    def _scoped(real, fake):
        def spoofed():
            # depth 1 is the real reading module: functools.cache and other
            # C-level wrappers push no Python frame.
            name = sys._getframe(1).f_globals.get("__name__", "")
            return fake if name in _SPOOF_CONSUMERS else real()
        return spoofed

    if not hasattr(platform, "_orig_system_for_mlx_shim"):
        platform._orig_system_for_mlx_shim = platform.system
        platform.system = _scoped(platform._orig_system_for_mlx_shim, "Darwin")
    if not hasattr(platform, "_orig_machine_for_mlx_shim"):
        platform._orig_machine_for_mlx_shim = platform.machine
        platform.machine = _scoped(platform._orig_machine_for_mlx_shim, "arm64")


def mlx_is_simulated() -> bool:
    """True when the ``mlx`` in ``sys.modules`` is this shim rather than a real one.

    ``simulate_mlx_on_torch`` installs process-wide, and one test module calls it
    while being IMPORTED. Collection imports every module in the session, so a
    later module's ``import mlx`` can succeed against the shim without that module
    ever having asked for it -- and whether it does depends on collection order,
    which under ``-n N --dist loadfile`` is not stable. A test that needs real mlx
    semantics must ask this, not just whether the import worked.
    """
    import sys

    module = sys.modules.get("mlx.core") or sys.modules.get("mlx")
    if module is None:
        return False
    origin = f"{getattr(module, '__name__', '')} {getattr(module, '__file__', '') or ''}"
    return "mlx_simulation" in origin


def snapshot_modules(is_owned):
    """Record every currently-imported module the caller is about to disturb."""
    import sys

    return {name: module for name, module in sys.modules.items() if is_owned(name)}


def restore_modules(saved, is_owned):
    """Undo a fixture's sys.modules surgery, PARENT PACKAGE ATTRIBUTES INCLUDED.

    Putting the old entries back in sys.modules is only half of it. Dropping
    `unsloth_zoo.mlx` and letting it be re-imported makes the import machinery
    bind `unsloth_zoo.mlx = <the new module>` as an attribute on the parent
    package, and that binding outlives the sys.modules entry that produced it.

    The two then disagree, because they are read by different things:
    `import unsloth_zoo.mlx.trainer as t` walks parent attributes, while a
    `from unsloth_zoo.mlx.trainer import MLXTrainer` inside the code under test
    resolves through sys.modules. A file that does both is left holding two live
    copies of one module, and `isinstance(trainer, MLXTrainer)` is False for an
    object the other copy built. That is not hypothetical: it is what
    dataset_utils.train_on_responses_only does to decide whether it is looking at
    an MLXTrainer, and it silently took the Hugging Face branch instead.

    So sys.modules and the parent attributes are put back together, or the next
    file inherits a split that only shows up under `-n N --dist loadfile`.
    """
    import sys

    disturbed = sorted(name for name in sys.modules if is_owned(name))
    for name in disturbed:
        sys.modules.pop(name, None)
    sys.modules.update(saved)

    # Shortest names first so a parent is itself restored before its children are
    # re-pointed at it.
    for name in sorted(set(disturbed) | set(saved), key = lambda n: (n.count("."), n)):
        parent_name, _, child = name.rpartition(".")
        if not parent_name:
            continue
        parent = sys.modules.get(parent_name)
        if parent is None:
            continue
        module = sys.modules.get(name)
        if module is not None:
            setattr(parent, child, module)
        elif hasattr(parent, child):
            # Imported only while the shim was up, so the attribute is a dangling
            # reference to a module nothing can reach through sys.modules.
            try:
                delattr(parent, child)
            except AttributeError:
                pass
