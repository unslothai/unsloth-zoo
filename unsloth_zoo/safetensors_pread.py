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

"""safetensors ``backend="pread"`` for CPU opens whose tensors are each read once and copied to the GPU.

On a CUDA unified-memory GPU (DGX Spark / GB10, Jetson, Thor) copying an mmap view to the device is many times slower
than copying a pread buffer (measured on DGX Spark: two Qwen3-30B-A3B layers, 19.6 s mmap vs 0.84 s pread). Windows
reads with pread too (mmap charges the whole file to the commit limit). Discrete Linux GPUs keep mmap: there pread
only adds a host copy. Tensors that stay on the host keep mmap everywhere (pread turns each into a private copy).
``UNSLOTH_SAFETENSORS_PREAD`` = 0 never, 1 always; safetensors < 0.8.0 has no ``backend`` and keeps mmap.
"""

import functools
import os
import sys

__all__ = ["pread_backend_kwargs", "safetensors_supports_backend"]

PREAD_ENV = "UNSLOTH_SAFETENSORS_PREAD"


def _env():
    value = (os.environ.get(PREAD_ENV) or "").strip().lower()
    if value in ("0", "off", "false", "no"):
        return False
    if value in ("1", "on", "true", "yes"):
        return True
    return None


@functools.lru_cache(maxsize = None)
def safetensors_supports_backend():
    """safetensors >= 0.8.0 (the first release with ``safe_open(..., backend=)``)."""
    try:
        import inspect

        from safetensors import safe_open

        text = getattr(safe_open, "__text_signature__", None) or ""
        if text:
            return "backend" in text
        return "backend" in inspect.signature(safe_open).parameters
    except Exception:
        return False


_UMA = None


def _unified_memory_now():
    """Every visible device is an integrated CUDA GPU. Never creates a CUDA context: a process without one is not
    loading to the GPU yet. ROCm APUs are not included (not measured). ``UNSLOTH_FORCE_UMA=1|0`` overrides."""
    global _UMA
    forced = os.environ.get("UNSLOTH_FORCE_UMA")
    if forced in ("0", "1"):
        return forced == "1"
    if _UMA is not None:
        return _UMA
    from .integrated_device import _cuda_context_exists, _is_hip_build, device_is_integrated_part

    if not _cuda_context_exists() or _is_hip_build():
        return False
    try:
        import torch

        count = torch.cuda.device_count()
        _UMA = count > 0 and all(device_is_integrated_part(i) for i in range(count))
    except Exception:
        return False
    return _UMA


def pread_backend_kwargs(device = "cpu"):
    """``{"backend": "pread"}`` for ``safe_open(..., device="cpu")`` where pread beats mmap, else ``{}``."""
    value = _env()
    if value is False or str(device) != "cpu":
        return {}
    if value is None and sys.platform != "win32" and not _unified_memory_now():
        return {}
    return {"backend": "pread"} if safetensors_supports_backend() else {}
