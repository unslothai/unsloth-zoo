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
    """Stub torch.cuda.* calls made at IMPORT time on CPU-only CI runners.

    Covers mem_get_info (used by temporary_patches/*), get_device_capability
    (compiler.py:87, loss_utils.py:39 -- gates cut_cross_entropy on Ampere+),
    and get_device_properties. Return (8, 0) so Ampere-gated imports proceed;
    the cut_cross_entropy import itself is try/except wrapped.
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
# 3. Apply upstream-drift fixes (triton CompiledKernel attrs, vLLM rename,
#    peft transformers_weight_conversion shim, etc.) by triggering
#    ``import unsloth``. Fixes live on ``unsloth/import_fixes.py`` and run
#    at unsloth import time; zoo no longer carries a copy. Security-only
#    test suites without unsloth installed keep passing -- ImportError is
#    swallowed below.
# ---------------------------------------------------------------------------

def _apply_upstream_import_fixes_for_tests() -> None:
    try:
        import unsloth  # noqa: F401  # runs unsloth/import_fixes.py
    except Exception:
        # unsloth missing (security-only suites) or import failed; drift
        # detectors will surface any pathology the patches would mask.
        pass


_apply_upstream_import_fixes_for_tests()
