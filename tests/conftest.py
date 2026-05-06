"""GPU-free test harness.

`unsloth_zoo.__init__` calls `device_type.get_device_type()` at import time,
which raises `NotImplementedError("Unsloth cannot find any torch accelerator")`
on CI runners without CUDA / XPU / HIP visible. This makes any test that
imports `unsloth_zoo` un-runnable on a GPU-free CI.

Most tests in this directory only exercise CPU-only logic (LoRA extractor
shape parity, registration coverage, dtype helpers). They do not need a real
accelerator. To unblock GPU-free CI, this conftest pre-installs a stub
`unsloth_zoo.device_type` into `sys.modules` BEFORE the package is imported,
exposing every name `unsloth_zoo/__init__.py` reads from it.

Behavior:
  - When a real accelerator is available (CUDA / XPU / HIP), the stub is
    NOT installed; the real `device_type.py` runs and reports the actual
    accelerator. CI on GPU runners still gets full fidelity.
  - When no accelerator is available, the stub claims `cuda` so the import
    chain in `__init__.py` does not raise. Downstream code that tries to
    call `torch.cuda.*` will still fail at *runtime*, but at *import* the
    package loads cleanly. Tests that stay on CPU run; tests that need
    GPU compute would fail on their own kernel calls and should be marked
    `@pytest.mark.skipif` separately.
  - The stub is a no-op when `unsloth_zoo` is already imported (some
    upstream pytest harness already loaded it).
"""

from __future__ import annotations

import importlib.util
import sys
import types


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
    """Pre-load the REAL `unsloth_zoo.device_type` module under a
    temporarily-mocked `torch.cuda.is_available()` so its
    `DEVICE_TYPE = get_device_type()` initialization succeeds without a
    real accelerator. Returns True on success.

    We need the real module (not a stub) so tests like
    `test_device_synchronize_xpu_calls_synchronize_when_present` keep
    exercising the real `device_synchronize` body.

    Strategy: build the minimal `unsloth_zoo` namespace package skeleton
    (so the relative `from .utils import Version` works), pre-load
    `unsloth_zoo.utils`, then pre-load `unsloth_zoo.device_type` with
    `torch.cuda.is_available` patched to True for the duration. The
    `@functools.cache` on `get_device_type` permanently captures the
    "cuda" result, so subsequent calls return "cuda" without needing
    the patch. Finally we drop the `unsloth_zoo` skeleton so the real
    `__init__.py` runs on the next `import unsloth_zoo`; it will find
    the already-loaded `device_type` and `utils` in `sys.modules` and
    skip re-execution.
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
        # Pre-load utils (device_type does `from .utils import Version`).
        if "unsloth_zoo.utils" not in sys.modules:
            utils_path = os.path.join(pkg_path, "utils.py")
            utils_spec = importlib.util.spec_from_file_location(
                "unsloth_zoo.utils", utils_path,
            )
            utils_mod = importlib.util.module_from_spec(utils_spec)
            sys.modules["unsloth_zoo.utils"] = utils_mod
            utils_spec.loader.exec_module(utils_mod)

        # Pre-load device_type with a temporarily-True is_available.
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
            # Drop our skeleton so the real package __init__.py executes
            # on the next `import unsloth_zoo`. The pre-loaded submodules
            # remain in sys.modules and will be reused by __init__.
            sys.modules.pop("unsloth_zoo", None)

    return True


def _patch_torch_cuda_for_import() -> None:
    """Monkey-patch concrete `torch.cuda.*` calls that other parts of
    `unsloth_zoo.temporary_patches.*` make at module IMPORT time. After
    this conftest finishes, `torch.cuda.is_available()` is back to its
    real value (False on a GPU-free CI), so transitive deps like torchao
    / dynamo correctly skip CUDA init when they are imported by other
    test modules.

    Specifically guards:
      gpt_oss.py:1141 -> torch.cuda.memory.mem_get_info(0)
    which runs at module top-level after `unsloth_zoo.device_type`'s
    `DEVICE_TYPE` is already "cuda" (cached above).
    """
    try:
        import torch.cuda.memory as _cuda_memory  # type: ignore
        _cuda_memory.mem_get_info = lambda *a, **k: (0, 80 * 1024 ** 3)
    except Exception:
        pass


if not _has_real_accelerator():
    if not _preload_real_device_type():
        # Fallback: if we cannot find the real device_type source (eg.
        # zipped install), fall back to a stub so tests at least import.
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
