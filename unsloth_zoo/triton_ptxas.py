# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Keep Triton on its own Blackwell ptxas when the system one is older.

Importing flashinfer sets ``TRITON_PTXAS_BLACKWELL_PATH`` to the system ``ptxas`` whenever that is CUDA 13 or
newer, even when it is older than the ptxas-blackwell Triton bundles. Triton then emits PTX for the older version
and every later sm_100+ compile uses it, so kernels needing a newer PTX ISA (the MXFP4 hardware e2m1 convert, PTX
9.2) silently fall back, or fail to compile once picked. flashinfer keeps an existing value, so pinning the bundled
path here, before vLLM pulls flashinfer in, keeps Triton's default. A newer system ptxas is left for flashinfer to
choose; a user-set value is never touched."""

from __future__ import annotations

import os
import re
import shutil
import subprocess

__all__ = ["keep_newest_blackwell_ptxas"]

_ENV = "TRITON_PTXAS_BLACKWELL_PATH"


def _version_tuple(text):
    match = re.search(r"(\d+)\.(\d+)", str(text or ""))
    return (int(match.group(1)), int(match.group(2))) if match else None


def _system_ptxas_version(path):
    out = subprocess.run([path, "--version"], capture_output = True, text = True, timeout = 30)
    match = re.search(r"release (\d+)\.(\d+)", (out.stdout or "") + (out.stderr or ""))
    return (int(match.group(1)), int(match.group(2))) if match else None


def keep_newest_blackwell_ptxas():
    """Returns the path pinned, else None. Never raises."""
    try:
        if os.environ.get(_ENV):
            return None  # user (or an earlier flashinfer import) chose; never override
        from importlib.util import find_spec
        if find_spec("flashinfer") is None or find_spec("triton") is None:
            return None
        import torch
        from unsloth_zoo.rocm_build import is_rocm_torch
        if is_rocm_torch(torch) or getattr(torch.version, "cuda", None) is None:
            return None
        from triton.backends.nvidia.compiler import get_ptxas
        tool = get_ptxas(100)  # env unset, so this is Triton's bundled ptxas-blackwell
        bundled, bundled_version = getattr(tool, "path", None), _version_tuple(getattr(tool, "version", None))
        if not bundled or bundled_version is None or not os.path.isfile(bundled):
            return None  # Triton without a ptxas-blackwell tool object (older layouts)
        system = shutil.which("ptxas")
        if not system or os.path.realpath(system) == os.path.realpath(bundled):
            return None
        system_version = _system_ptxas_version(system)
        # flashinfer only switches to a CUDA 13+ system ptxas; a newer or equal one is a fine choice.
        if system_version is None or system_version[0] < 13 or system_version >= bundled_version:
            return None
        os.environ[_ENV] = bundled
        return bundled
    except Exception:
        return None
