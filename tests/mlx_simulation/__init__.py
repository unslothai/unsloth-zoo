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


def _spoof_apple_silicon_platform():
    """Make platform.system()=='Darwin' and platform.machine()=='arm64'.

    Idempotent.  PR-B's _IS_MLX gate in unsloth/__init__.py uses these
    to decide between MLX and CUDA dispatch.
    """
    global _PLATFORM_SPOOFED
    if _PLATFORM_SPOOFED:
        return
    _PLATFORM_SPOOFED = True

    import platform
    if not hasattr(platform, "_orig_system_for_mlx_shim"):
        platform._orig_system_for_mlx_shim = platform.system
        platform.system = lambda: "Darwin"
    if not hasattr(platform, "_orig_machine_for_mlx_shim"):
        platform._orig_machine_for_mlx_shim = platform.machine
        platform.machine = lambda: "arm64"
