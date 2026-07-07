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


def _preload_real_device_type(
    package: str = "unsloth_zoo",
    prereqs: tuple = ("utils",),
) -> bool:
    """Pre-load the REAL ``<package>.device_type`` module under a
    temporarily-mocked ``torch.cuda.is_available()`` so its
    ``DEVICE_TYPE = get_device_type()`` initialization succeeds without
    a real accelerator. Returns True on success; returns False if
    torch is not importable at all (the security-audit CI job runs
    tests/security/ without installing torch, and those tests don't
    need the preload), or if the target package isn't installed.

    Parameterised so the same harness works for both ``unsloth_zoo``
    (where ``utils.py`` defines ``Version`` before ``device_type``
    consumes it) and ``unsloth`` (which has no such prereq).
    """
    target = f"{package}.device_type"
    if target in sys.modules:
        return True
    pkg_spec = importlib.util.find_spec(package)
    if pkg_spec is None or not pkg_spec.submodule_search_locations:
        return False
    pkg_path = pkg_spec.submodule_search_locations[0]

    import os

    skeleton_already = package in sys.modules
    if not skeleton_already:
        pkg_mod = types.ModuleType(package)
        pkg_mod.__path__ = [pkg_path]
        pkg_mod.__spec__ = pkg_spec
        pkg_mod.__package__ = package
        sys.modules[package] = pkg_mod

    try:
        for prereq in prereqs:
            full = f"{package}.{prereq}"
            if full in sys.modules:
                continue
            prereq_path = os.path.join(pkg_path, f"{prereq}.py")
            prereq_spec = importlib.util.spec_from_file_location(full, prereq_path)
            prereq_mod = importlib.util.module_from_spec(prereq_spec)
            sys.modules[full] = prereq_mod
            try:
                prereq_spec.loader.exec_module(prereq_mod)
            except ModuleNotFoundError as exc:
                # Tests that don't need torch (e.g. the tests/security
                # subtree which only exercises scanner regex tables and
                # subprocess invocations) shouldn't be blocked by the
                # device-type preload when torch isn't installed. Pop
                # the half-built modules and bail out gracefully.
                if "torch" in str(exc):
                    sys.modules.pop(full, None)
                    if not skeleton_already:
                        sys.modules.pop(package, None)
                    return False
                raise

        device_type_path = os.path.join(pkg_path, "device_type.py")
        dt_spec = importlib.util.spec_from_file_location(target, device_type_path)
        dt_mod = importlib.util.module_from_spec(dt_spec)
        sys.modules[target] = dt_mod

        import torch
        _orig_is_avail = torch.cuda.is_available
        torch.cuda.is_available = lambda: True  # type: ignore[assignment]
        try:
            dt_spec.loader.exec_module(dt_mod)
        finally:
            torch.cuda.is_available = _orig_is_avail
    finally:
        if not skeleton_already:
            sys.modules.pop(package, None)

    return True


def _install_device_type_stub(name: str) -> None:
    """Last-resort stub when the real preload can't run (no torch / no
    package installed). Matches the surface ``unsloth`` and ``unsloth_zoo``
    consumers read at import time."""
    stub = types.ModuleType(name)
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
    sys.modules[name] = stub


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
    if not _preload_real_device_type("unsloth_zoo", prereqs=("utils",)):
        _install_device_type_stub("unsloth_zoo.device_type")
    # NOTE: we deliberately do NOT stub ``unsloth.device_type`` here.
    # Doing so makes ``import unsloth`` succeed on CPU-only CI, which
    # then runs ``unsloth/_gpu_init.py:_patch_trl_trainer()`` and
    # rebinds ``trl.trainer.sft_trainer.SFTTrainer`` /
    # ``transformers.models.ministral.MinistralAttention`` to Unsloth's
    # compiled wrappers. ``inspect.getsource(...)`` on those classes
    # then returns the wrapper source, which masks upstream and causes
    # zoo's drift detectors (test_MinistralAttention_forward_signature,
    # test_unsloth_rl_trainer_*) to fail. The cost is that the
    # ``test_unsloth_trainer_exec_marker`` smoke test fails on CPU-only
    # runners; that failure exists on main too and tracks a separate
    # ``unsloth.device_type`` consumer that needs its own CPU fallback.
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
    # Let `import unsloth` succeed on a CPU-only CI runner. The flag is
    # honoured by unsloth's get_device_type (returns "cuda" sentinel) and
    # by PatchFastRL / _patch_trl_trainer (early-return so trl.SFTTrainer
    # stays pristine for downstream inspect.getsource drift detectors).
    # Production hosts with a real accelerator skip both branches.
    import os
    os.environ.setdefault("UNSLOTH_ALLOW_CPU", "1")
    try:
        import unsloth  # noqa: F401  # runs unsloth/import_fixes.py
    except Exception:
        # unsloth missing (security-only suites) or import failed; drift
        # detectors will surface any pathology the patches would mask.
        pass


_apply_upstream_import_fixes_for_tests()
