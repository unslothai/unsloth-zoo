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

``expandable_segments_unsupported()`` is read BEFORE ``import torch`` (the allocator reads its
configuration during torch initialization), so it is filesystem only. ``_any_device_integrated()``
costs a CUDA context -- ``get_device_properties`` flips ``is_initialized`` True where
``device_count`` does not -- so only ``gradient_checkpointing.py`` calls it, never at import.

Torch's own "expandable_segments not supported on this platform" guard in
``c10/cuda/CUDAAllocatorConfig.h`` is a compile-time ``#if``, not a device query, so an aarch64
CUDA build enables the mode and its VMM driver calls fail only at allocation time (as
``RuntimeError: CUDA driver error``, never ``torch.OutOfMemoryError``). Scoped to Tegra, not every
integrated device: on GB10 expandable segments is the upstream mitigation (vllm-project/vllm#55569).
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

# 1 keeps the old behaviour on an excluded board, 0 forces the exclusion on an unrecognised one.
FORCE_ENV = "UNSLOTH_FORCE_EXPANDABLE_SEGMENTS"

# Bytes, not text: /proc/device-tree/compatible is a NUL separated list. The DMI pair is here
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

# Bare "tegra" also catches the /etc/nv_tegra_release file NAME, so its existence is a signal.
_TEGRA_MARKERS = ("nvidia,tegra", "tegra", "jetson")

# Checked BEFORE the Tegra markers and wins: GB10 is Tegra lineage but works.
_VMM_CAPABLE_MARKERS = ("dgx spark", "dgx-spark", "dgx_spark", "gb10")

# Tegra is ARM only, so an x86_64 host answers with one compare and a stray file is harmless.
_ARM_MACHINES = ("aarch64", "arm64", "armv8b", "armv8l", "armv7l")


def _env_tristate(name):
    value = os.environ.get(name, "").strip().lower()
    if value in _ENV_TRUE:  return True
    if value in _ENV_FALSE: return False
    return None


def read_board_identity(paths = None):
    """Lowercased `name=contents` of every readable board identity file. The file NAME is part
    of the text: it is the only signal /etc/nv_tegra_release carries reliably."""
    if paths is None: paths = BOARD_IDENTITY_FILES
    chunks = []
    for path in paths:
        try:
            with open(path, "rb") as file:
                raw = file.read(4096) # never read an unbounded file
        except OSError:
            continue
        text = raw.replace(b"\x00", b" ").decode("utf-8", errors = "ignore")
        chunks.append(f"{os.path.basename(path)}={' '.join(text.lower().split())}")
    return " ".join(chunks)


def is_tegra_board(identity = None):
    """True on an NVIDIA Tegra SoC board (Jetson). ``identity`` is only for tests."""
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
    """``sys.modules``, not ``import torch``: this must never be what imports torch."""
    torch = sys.modules.get("torch")
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_initialized())
    except Exception:
        return False


def _any_device_integrated():
    # ANY integrated device makes double buffering pure overhead, and a mixed box is rare.
    # torch is imported inside the function so this module stays importable before torch is.
    try:
        import torch
        return any(
            bool(getattr(torch.cuda.get_device_properties(i), "is_integrated", 0))
            for i in range(torch.cuda.device_count())
        )
    except Exception:
        return False


def _tegra_device_visible_for_free():
    """The driver's opinion, for the container case where the board files are hidden; False
    rather than a probe when no CUDA context exists, so import never initializes CUDA."""
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
    """Should ``expandable_segments:True`` be kept off here? Answered without importing torch
    or creating a CUDA context, so it is safe in the import-time allocator block."""
    forced = _env_tristate(FORCE_ENV)
    if forced is not None:
        return not forced
    if is_tegra_board():
        return True
    return _tegra_device_visible_for_free()
