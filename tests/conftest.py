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

"""pytest conftest: GPU-free harness + MLX simulation suite path.

Combines two pieces of test setup:

1. GPU-free harness for the CPU-only tests already in this directory
   (LoRA extractor shape parity, registration coverage, dtype helpers).
   ``unsloth_zoo.__init__`` calls ``device_type.get_device_type()`` at
   import time, which raises ``NotImplementedError`` on CI runners
   without CUDA / XPU / HIP visible. We pre-load the real
   ``unsloth_zoo.device_type`` under a temporarily-True
   ``torch.cuda.is_available()`` so the @cache permanently captures
   ``"cuda"`` and the package import chain succeeds. When a real
   accelerator IS available the pre-load is skipped and the real
   detection runs.

2. ``tests/`` is added to ``sys.path`` so the bundled MLX-on-torch
   simulation suite can ``from mlx_simulation import ...``. The shim is
   opt-in test infrastructure: it activates only when a test calls
   ``simulate_mlx_on_torch()`` and never touches production imports of
   ``unsloth_zoo``.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types


# ---------------------------------------------------------------------------
# 1. GPU-free harness: pre-load device_type so importing unsloth_zoo
#    without CUDA / XPU / HIP visible doesn't raise.
# ---------------------------------------------------------------------------

def _has_real_accelerator() -> bool:
    try:
        import torch
    except Exception:
        return False
    try:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return True
    except Exception:
        pass
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return True
    except Exception:
        pass
    try:
        if hasattr(torch, "accelerator") and torch.accelerator.is_available():
            return True
    except Exception:
        pass
    return False


def _preload_real_device_type() -> bool:
    """Pre-load the REAL ``unsloth_zoo.device_type`` module under a
    temporarily-mocked ``torch.cuda.is_available()`` so its
    ``DEVICE_TYPE = get_device_type()`` initialization succeeds without
    a real accelerator. Returns True on success; returns False if
    torch is not importable at all (the security-audit CI job runs
    tests/security/ without installing torch, and those tests don't
    need the preload).
    """
    if "unsloth_zoo.device_type" in sys.modules:
        return True
    pkg_spec = importlib.util.find_spec("unsloth_zoo")
    if pkg_spec is None or not pkg_spec.submodule_search_locations:
        return False
    pkg_path = pkg_spec.submodule_search_locations[0]

    import os

    skeleton_already = "unsloth_zoo" in sys.modules
    if not skeleton_already:
        zoo_pkg = types.ModuleType("unsloth_zoo")
        zoo_pkg.__path__ = [pkg_path]
        zoo_pkg.__spec__ = pkg_spec
        zoo_pkg.__package__ = "unsloth_zoo"
        sys.modules["unsloth_zoo"] = zoo_pkg

    try:
        if "unsloth_zoo.utils" not in sys.modules:
            utils_path = os.path.join(pkg_path, "utils.py")
            utils_spec = importlib.util.spec_from_file_location(
                "unsloth_zoo.utils", utils_path,
            )
            utils_mod = importlib.util.module_from_spec(utils_spec)
            sys.modules["unsloth_zoo.utils"] = utils_mod
            try:
                utils_spec.loader.exec_module(utils_mod)
            except ModuleNotFoundError as exc:
                # Tests that don't need torch (e.g. the tests/security
                # subtree which only exercises scanner regex tables and
                # subprocess invocations) shouldn't be blocked by the
                # device-type preload when torch isn't installed. Pop
                # the half-built modules and bail out gracefully.
                if "torch" in str(exc):
                    sys.modules.pop("unsloth_zoo.utils", None)
                    if not skeleton_already:
                        sys.modules.pop("unsloth_zoo", None)
                    return False
                raise

        device_type_path = os.path.join(pkg_path, "device_type.py")
        dt_spec = importlib.util.spec_from_file_location(
            "unsloth_zoo.device_type", device_type_path,
        )
        dt_mod = importlib.util.module_from_spec(dt_spec)
        sys.modules["unsloth_zoo.device_type"] = dt_mod

        import torch
        _orig_is_avail = torch.cuda.is_available
        torch.cuda.is_available = lambda: True  # type: ignore[assignment]
        try:
            dt_spec.loader.exec_module(dt_mod)
        finally:
            torch.cuda.is_available = _orig_is_avail
    finally:
        if not skeleton_already:
            sys.modules.pop("unsloth_zoo", None)

    return True


def _patch_torch_cuda_for_import() -> None:
    """Guard concrete ``torch.cuda.*`` calls that ``unsloth_zoo.*``
    modules make at IMPORT time on CPU-only CI runners.

    Three crash classes covered:

    1. ``torch.cuda.memory.mem_get_info`` -- some
       ``unsloth_zoo.temporary_patches.*`` modules call this at
       module init. Return a plausible (free, total) pair so the
       memory-availability arithmetic succeeds.

    2. ``torch.cuda.get_device_capability`` -- called at module top
       level in ``unsloth_zoo/compiler.py:87`` and
       ``unsloth_zoo/loss_utils.py:39`` to gate the cut_cross_entropy
       import on Ampere+. CPU-only torch raises ``AssertionError:
       Torch not compiled with CUDA enabled``, blocking every test
       that does ``importlib.import_module("unsloth_zoo.compiler")``
       or ``...loss_utils``. Patch to return ``(8, 0)`` so the
       capability check passes (Ampere-equivalent); the actual
       cut_cross_entropy import is try/except-wrapped anyway.

    3. ``torch.cuda.get_device_properties`` -- similar shape, used
       by other temporary_patches sites. Return a minimal namespace
       with ``major`` / ``minor`` / ``total_memory`` attributes.
    """
    try:
        import torch  # type: ignore
        import torch.cuda.memory as _cuda_memory  # type: ignore
        _cuda_memory.mem_get_info = lambda *a, **k: (0, 80 * 1024 ** 3)
    except Exception:
        return
    try:
        torch.cuda.get_device_capability = lambda *a, **k: (8, 0)  # type: ignore[assignment]
    except Exception:
        pass
    try:
        class _StubDeviceProps:
            major = 8
            minor = 0
            total_memory = 80 * 1024 ** 3
            multi_processor_count = 108
            name = "stub"
        torch.cuda.get_device_properties = lambda *a, **k: _StubDeviceProps()  # type: ignore[assignment]
    except Exception:
        pass


if not _has_real_accelerator():
    if not _preload_real_device_type():
        stub = types.ModuleType("unsloth_zoo.device_type")
        stub.DEVICE_TYPE = "cuda"
        stub.DEVICE_TYPE_TORCH = "cuda"
        stub.DEVICE_COUNT = 1
        stub.ALLOW_PREQUANTIZED_MODELS = False
        stub.is_hip = lambda: False
        stub.get_device_type = lambda: "cuda"
        stub.get_device_count = lambda: 1
        stub.device_synchronize = lambda *a, **k: None
        stub.device_empty_cache = lambda *a, **k: None
        stub.device_is_bf16_supported = lambda *a, **k: False
        sys.modules["unsloth_zoo.device_type"] = stub
    _patch_torch_cuda_for_import()


# ---------------------------------------------------------------------------
# 2. Make ``tests/mlx_simulation`` importable as ``mlx_simulation`` for
#    the MLX-on-torch shim suite.
# ---------------------------------------------------------------------------

_TESTS_DIR = pathlib.Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))


# ---------------------------------------------------------------------------
# 3. Apply zoo-local upstream-drift fixes (triton CompiledKernel attrs,
#    vLLM GuidedDecodingParams rename, peft transformers_weight_conversion
#    shim, etc.). The production import path applies these via
#    ``unsloth_zoo/__init__.py``, but the GPU-free test harness above
#    deliberately avoids importing the full ``unsloth_zoo`` package
#    (which requires CUDA / torch device initialization). Load just
#    the standalone import-fixes module by file path so the drift
#    detectors in ``test_upstream_import_fixes_drift.py`` see the
#    same patched state a real zoo install would.
# ---------------------------------------------------------------------------

def _apply_zoo_import_fixes_for_tests() -> None:
    try:
        pkg_spec = importlib.util.find_spec("unsloth_zoo")
    except Exception:
        return
    if pkg_spec is None or not pkg_spec.submodule_search_locations:
        return
    import os as _os
    fix_path = _os.path.join(
        pkg_spec.submodule_search_locations[0], "import_fixes.py",
    )
    if not _os.path.exists(fix_path):
        return
    mod_name = "unsloth_zoo.import_fixes"
    # Track whether WE installed the parent-package skeleton, so we can
    # pop it after loading import_fixes.py. Leaving a half-initialised
    # ``unsloth_zoo`` in sys.modules confuses other tests (e.g.
    # test_zoo_history_regressions_deep.py imports submodules off the
    # real package and relies on the full __init__.py having run).
    _installed_skeleton = False
    if mod_name in sys.modules:
        mod = sys.modules[mod_name]
    else:
        # Submodule import requires SOME parent ``unsloth_zoo`` entry in
        # sys.modules. Reuse one if a sibling conftest step already
        # installed it (and don't pop in that case); otherwise install a
        # bare skeleton and pop on the way out.
        if "unsloth_zoo" not in sys.modules:
            zoo_pkg = types.ModuleType("unsloth_zoo")
            zoo_pkg.__path__ = list(pkg_spec.submodule_search_locations)
            zoo_pkg.__spec__ = pkg_spec
            zoo_pkg.__package__ = "unsloth_zoo"
            zoo_pkg.__file__ = _os.path.join(
                pkg_spec.submodule_search_locations[0], "__init__.py",
            )
            sys.modules["unsloth_zoo"] = zoo_pkg
            _installed_skeleton = True
        spec = importlib.util.spec_from_file_location(mod_name, fix_path)
        if spec is None or spec.loader is None:
            if _installed_skeleton:
                sys.modules.pop("unsloth_zoo", None)
            return
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:
            sys.modules.pop(mod_name, None)
            if _installed_skeleton:
                sys.modules.pop("unsloth_zoo", None)
            return
    apply = getattr(mod, "apply_import_fixes", None)
    if apply is None:
        if _installed_skeleton:
            sys.modules.pop("unsloth_zoo", None)
        return
    try:
        apply()
    except Exception:
        # Individual fixes are already wrapped; if the entrypoint itself
        # blows up, don't take pytest collection down.
        pass
    finally:
        # Drop our scratch skeleton so subsequent ``import unsloth_zoo``
        # / ``importlib.import_module("unsloth_zoo")`` calls hit the real
        # package init (or whatever skeleton step 1 of this conftest
        # installs lazily on demand) rather than our empty placeholder.
        # The import-fixes module itself stays in sys.modules under
        # ``unsloth_zoo.import_fixes`` -- python's import machinery is
        # happy to find a submodule without an active parent entry.
        if _installed_skeleton:
            sys.modules.pop("unsloth_zoo", None)


_apply_zoo_import_fixes_for_tests()
