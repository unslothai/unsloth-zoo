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

"""Which board this is, for the two decisions that depend on unified memory.

Both questions used to be answered from different places with the same driver
probe, and one of them cannot afford that probe:

* ``expandable_segments_unsupported()`` is read by ``unsloth_zoo/__init__.py``
  inside the import-time allocator block, which runs BEFORE ``import torch``
  because the allocator reads its configuration during torch initialization.
  So this one is filesystem only: no torch, no CUDA context, no subprocess.
* ``_any_device_integrated()`` is the driver's own answer and costs a CUDA
  context (measured: ``torch.cuda.is_available()`` and
  ``torch.cuda.device_count()`` leave ``torch.cuda.is_initialized()`` False,
  ``torch.cuda.get_device_properties()`` flips it True). It moved here from
  ``gradient_checkpointing.py`` so one module owns the topic, and it is still
  called from there, on the first gradient-checkpointing init, after the caller
  has picked its device. It is NOT called at import.

Why the allocator needs any of this: ``expandable_segments:True`` does not go
through ``cudaMalloc``. It reserves a virtual range and backs it with the CUDA
virtual memory management driver calls (``cuMemAddressReserve``,
``cuMemCreate``, ``cuMemMap``, ``cuMemSetAccess``), each wrapped in
``C10_CUDA_DRIVER_CHECK``, which is the only thing in the caching allocator
that raises ``RuntimeError: CUDA driver error: ...``. A cudaMalloc shortage
raises ``torch.OutOfMemoryError`` with "CUDA out of memory. Tried to allocate"
instead, so the two are distinguishable from a traceback alone. Torch's own
"expandable_segments not supported on this platform" guard in
``c10/cuda/CUDAAllocatorConfig.h`` is a compile-time ``#if`` on
``PYTORCH_C10_DRIVER_API_SUPPORTED``, not a runtime device-capability query, so
a Linux aarch64 CUDA build (what the JetPack wheels are) enables the mode and
only finds out at allocation time.

Deliberately scoped to Tegra, not to every integrated device. On NVIDIA GB10
(DGX Spark) expandable segments is reported upstream as the mitigation that
makes a long prefill survive (vllm-project/vllm#55569), so taking it away there
would be the regression, and those boards are excluded by name.
"""

import os
import platform
import sys

__all__ = [
    "BOARD_IDENTITY_FILES",
    "read_board_identity",
    "is_tegra_board",
    "expandable_segments_unsupported",
]

_ENV_TRUE  = ("1", "true", "yes", "on")
_ENV_FALSE = ("0", "false", "no", "off")

# Set to 1 to keep the old behaviour on a board this module excludes, or to 0 to
# force the exclusion on a board it did not recognise. Same tri-state shape as
# UNSLOTH_DISABLE_DOUBLE_BUFFER and UNSLOTH_FORCE_UMA.
FORCE_ENV = "UNSLOTH_FORCE_EXPANDABLE_SEGMENTS"

# Read as bytes, not text: /proc/device-tree/compatible is a NUL separated list
# of strings and /proc/device-tree/model is NUL terminated. The DMI pair is here
# because ARM boards that boot via ACPI have no /proc/device-tree at all.
BOARD_IDENTITY_FILES = (
    "/etc/nv_tegra_release",
    "/proc/device-tree/model",
    "/proc/device-tree/compatible",
    "/sys/firmware/devicetree/base/model",
    "/sys/firmware/devicetree/base/compatible",
    "/sys/class/dmi/id/product_name",
    "/sys/class/dmi/id/board_name",
)

# "nvidia,tegra" is the device-tree compatible family (nvidia,tegra234 on Orin).
# "tegra" on its own also catches the /etc/nv_tegra_release file NAME, which is
# what L4T ships and what makes the existence of that file a signal by itself.
_TEGRA_MARKERS = ("nvidia,tegra", "tegra", "jetson")

# Checked BEFORE the Tegra markers and wins over them. GB10 is Tegra lineage and
# its device tree can say so, but expandable segments works there (see module
# docstring), so a Spark must keep it.
_VMM_CAPABLE_MARKERS = ("dgx spark", "dgx-spark", "dgx_spark", "gb10")

# Tegra is ARM only, so an x86_64 host answers with one string compare and never
# stats a path. This is also what makes a stray file on a normal host harmless.
_ARM_MACHINES = ("aarch64", "arm64", "armv8b", "armv8l", "armv7l")


def _env_tristate(name):
    """True, False, or None when the variable is unset or not a known word."""
    value = os.environ.get(name, "").strip().lower()
    if value in _ENV_TRUE:  return True
    if value in _ENV_FALSE: return False
    return None


def read_board_identity(paths = None):
    """Lowercased `name=contents` of every readable board identity file.

    Empty string when none of them can be read, which is the normal answer on
    x86_64, on macOS and on Windows. The file name is part of the text on
    purpose: it is the only signal /etc/nv_tegra_release carries reliably.
    """
    if paths is None: paths = BOARD_IDENTITY_FILES
    chunks = []
    for path in paths:
        try:
            with open(path, "rb") as file:
                raw = file.read(4096) # these are small; never read an unbounded file
        except OSError:
            continue # missing, a directory, or not readable by this user
        text = raw.replace(b"\x00", b" ").decode("utf-8", errors = "ignore")
        chunks.append(f"{os.path.basename(path)}={' '.join(text.lower().split())}")
    return " ".join(chunks)


def is_tegra_board(identity = None):
    """True on an NVIDIA Tegra SoC board (Jetson), False everywhere else.

    ``identity`` is only for tests; production reads the files.
    """
    if platform.machine().strip().lower() not in _ARM_MACHINES:
        return False
    if identity is None:
        identity = read_board_identity()
    if not identity:
        return False
    if any(marker in identity for marker in _VMM_CAPABLE_MARKERS):
        return False
    return any(marker in identity for marker in _TEGRA_MARKERS)


def _cuda_context_exists():
    """True only when torch is already imported AND its CUDA context is up.

    ``sys.modules``, not ``import torch``: this must never be the thing that
    imports torch, and importing it is what the caller is avoiding.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_initialized())
    except Exception:
        return False


def _any_device_integrated():
    # True if ANY visible CUDA/HIP device is integrated (unified memory). A single
    # static check on purpose: an integrated device anywhere makes double buffering
    # pure overhead, and a mixed integrated + discrete box is rare.
    #
    # Moved here from gradient_checkpointing.py, which still imports it under this
    # name. torch is imported inside the function so this module stays importable
    # before torch is, which is the whole reason it exists.
    try:
        import torch
        return any(
            bool(getattr(torch.cuda.get_device_properties(i), "is_integrated", 0))
            for i in range(torch.cuda.device_count())
        )
    except Exception:
        return False


def _tegra_device_visible_for_free():
    """The driver's opinion, but only when reading it costs nothing.

    A container can hide the board files (/proc/device-tree is not bind-mounted
    into the default namespace), so this is the second opinion for that case. It
    returns False rather than probing whenever no CUDA context exists yet, so
    ``import unsloth_zoo`` never initializes CUDA on its account.
    """
    if not _cuda_context_exists():
        return False
    if not _any_device_integrated():
        return False
    try:
        import torch
        names = " ".join(
            str(torch.cuda.get_device_name(i)).lower()
            for i in range(torch.cuda.device_count())
        )
    except Exception:
        return False
    if any(marker in names for marker in _VMM_CAPABLE_MARKERS):
        return False
    return any(marker in names for marker in ("tegra", "jetson", "orin", "xavier"))


def expandable_segments_unsupported():
    """Should PYTORCH_ALLOC_CONF's ``expandable_segments:True`` be kept off here?

    Answered without importing torch and without creating a CUDA context, so it
    is safe to call from the import-time allocator block. ``UNSLOTH_FORCE_EXPANDABLE_SEGMENTS``
    forces either answer (1 keeps expandable segments, 0 removes them).
    """
    forced = _env_tristate(FORCE_ENV)
    if forced is not None:
        return not forced
    if is_tegra_board():
        return True
    return _tegra_device_visible_for_free()
