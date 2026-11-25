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
    "cuda_props_are_integrated",
    "cuda_total_memory",
    "device_is_integrated_part",
    "device_memory_pool",
    "device_shares_host_memory",
    "unified_memory_device_indices",
]

_ENV_TRUE  = ("1", "true", "yes", "on")
_ENV_FALSE = ("0", "false", "no", "off")

# 1 keeps the old behaviour on an excluded board, 0 forces the exclusion on an unrecognised one.
FORCE_ENV = "UNSLOTH_FORCE_EXPANDABLE_SEGMENTS"

# Read as bytes: /proc/device-tree/compatible is NUL separated; the DMI pair is included because ACPI-booted ARM boards have no /proc/device-tree at all.
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
    # Any integrated device makes double buffering pure overhead; torch is imported lazily.
    try:
        import torch
        return any(
            bool(getattr(torch.cuda.get_device_properties(i), "is_integrated", 0))
            for i in range(torch.cuda.device_count())
        )
    except Exception:
        return False


# AMD APU graphics have no memory of their own, only a carve-out of system RAM; gcnArchName is read
# since hipDeviceProp_t::integrated is unreliable before ROCm 6.2.
_HIP_APU_ARCHS = frozenset((
    "gfx902", "gfx909", "gfx90c", "gfx1033", "gfx1035", "gfx1036", "gfx1037",
    "gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx1153",
))
# NVIDIA parts whose device memory is system memory, for wheels that do not publish the integrated field.
_NVIDIA_UNIFIED_NAME_MARKERS = ("gb10", "dgx spark", "thor", "orin", "xavier", "tegra", "jetson")

# 1 keeps checkpointed activations on the device everywhere, 0 offloads them even on unified memory (the old default).
KEEP_ON_DEVICE_ENV = "UNSLOTH_GC_KEEP_ON_DEVICE"


def _hip_version_at_least(major, minor):
    try:
        import torch
        parts = str(torch.version.hip or "").split(".")
        return (int(parts[0]), int(parts[1])) >= (major, minor)
    except Exception:
        return False


def device_is_integrated_part(index):
    """True when CUDA / HIP device ``index`` is an integrated GPU (an APU / SoC part with no memory chips of its own).

    Three signals, any one is enough: the driver's integrated flag (trusted on HIP only from ROCm 6.2, where clr
    started filling it in), an AMD APU architecture name, or (CUDA wheels without the flag only) an NVIDIA
    unified-memory part name. Never raises.
    """
    try:
        import torch
        props = torch.cuda.get_device_properties(index)
    except Exception:
        return False
    hip = _is_hip_build()
    if hip:
        arch = str(getattr(props, "gcnArchName", "") or "").split(":")[0].strip().lower()
        if arch in _HIP_APU_ARCHS:
            return True
        if not _hip_version_at_least(6, 2):
            return False
    for field in ("is_integrated", "integrated"):
        value = getattr(props, field, None)
        if value is not None and bool(value):
            return True
    if hip:
        return False
    # CUDA fills the flag reliably: when the wheel publishes it, 0 means discrete and a name never overrides it.
    if any(getattr(props, field, None) is not None for field in ("is_integrated", "integrated")):
        return False
    name = str(getattr(props, "name", "") or "").lower()
    return any(marker in name for marker in _NVIDIA_UNIFIED_NAME_MARKERS)


# AMD APU BIOS carve-out is hidden from the OS; past it, allocations use GTT host RAM; shared only
# if carve-out is at most CARVEOUT_MAX_SHARE of allocatable memory.
CARVEOUT_MAX_SHARE = 0.25


def _amdgpu_sysfs_vram_total(props, root = "/sys/class/drm"):
    """Bytes of dedicated VRAM (the carve-out) amdgpu reports for the card at ``props``' PCI address, or None."""
    import glob as _glob
    bus = getattr(props, "pci_bus_id", None)
    found = []
    for device in sorted(_glob.glob(os.path.join(root, "card*", "device"))):
        try:
            with open(os.path.join(device, "mem_info_vram_total"), "rb") as file:
                vram = int(file.read(64).strip())
        except (OSError, ValueError):
            continue
        address = os.path.basename(os.path.realpath(device)).lower() # 0000:c3:00.0
        try:
            card_bus = int(address.split(":")[1], 16)
        except (IndexError, ValueError):
            card_bus = None
        found.append((card_bus, vram))
    if bus is not None:
        for card_bus, vram in found:
            if card_bus == bus:
                return vram
    return found[0][1] if len(found) == 1 else None


def _windows_hidden_memory():
    """Installed RAM minus what Windows can use, i.e. the carve-out (plus firmware reservations), or None."""
    try:
        import ctypes
        installed_kib = ctypes.c_ulonglong(0)
        if not ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(installed_kib)):
            return None

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return max(int(installed_kib.value) * 1024 - int(status.ullTotalPhys), 0)
    except Exception:
        return None


def device_memory_pool(index):
    """Where device ``index``'s allocations come from, for the offload decision. A dict with ``pool`` one of
    "discrete", "shared" (device memory IS host RAM: offloading frees nothing), "carveout" (an integrated part
    whose allocations come from a dedicated carve-out the host cannot use: offloading frees device budget) or
    "unknown" (integrated, carve-out unreadable: treated like "carveout"), plus the numbers it was decided on.
    """
    info = {"pool": "discrete", "total_memory": None, "carveout": None, "carveout_source": None}
    if not device_is_integrated_part(index):
        return info
    try:
        import torch
        props = torch.cuda.get_device_properties(index)
        total = int(getattr(props, "total_memory", 0) or 0)
    except Exception:
        info["pool"] = "unknown"
        return info
    info["total_memory"] = total
    if not _is_hip_build():
        # GB10 / Tegra: one coherent LPDDR pool, no carve-out.
        info["pool"] = "shared"
        return info
    carveout, source = None, None
    if sys.platform.startswith("linux"):
        carveout, source = _amdgpu_sysfs_vram_total(props), "sysfs mem_info_vram_total"
    elif sys.platform == "win32":
        carveout, source = _windows_hidden_memory(), "installed minus usable RAM"
    info["carveout"], info["carveout_source"] = carveout, source
    if carveout is None or total <= 0:
        info["pool"] = "unknown"
    elif carveout <= CARVEOUT_MAX_SHARE * total:
        info["pool"] = "shared"
    else:
        info["pool"] = "carveout"
    return info


def device_shares_host_memory(index):
    """True when device ``index``'s allocations come out of the same RAM host offload would use (see
    ``device_memory_pool``): an integrated part with no carve-out or a small one. Never raises."""
    try:
        return device_memory_pool(index)["pool"] == "shared"
    except Exception:
        return False


def unified_memory_device_indices():
    """Visible CUDA / HIP device indices whose checkpointed activations should stay on the device.

    Offloading to the host there is a copy into the same DRAM: no memory is saved, and the pinned blocks plus the
    D2H and H2D copies are pure cost. ``UNSLOTH_GC_KEEP_ON_DEVICE`` = 1 selects every device, 0 none.
    """
    forced = _env_tristate(KEEP_ON_DEVICE_ENV)
    if forced is False:
        return frozenset()
    try:
        import torch
        count = torch.cuda.device_count()
    except Exception:
        return frozenset()
    if forced is True:
        return frozenset(range(count))
    return frozenset(i for i in range(count) if device_shares_host_memory(i))


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


def _is_hip_build():
    """True on a ROCm/HIP torch, where ``torch.cuda`` is the HIP namespace, so every CUDA-only
    probe below has to exclude HIP by name rather than by field presence."""
    try:
        import torch
        return bool(getattr(torch.version, "hip", None))
    except Exception:
        return False


def cuda_props_are_integrated(props):
    """True when ``props`` describes an integrated NVIDIA part, whose VRAM is system RAM.

    torch publishes ``cudaDeviceProp::integrated`` as ``is_integrated``, plain ``integrated`` on
    some wheels and absent on others, hence getattr. HIP is excluded BY NAME rather than by
    trusting the field to be missing: clr left it unassigned before ROCm 6.2, so False there means
    "unknown", not "discrete".
    """
    if _is_hip_build():
        return False
    for field in ("is_integrated", "integrated"):
        value = getattr(props, field, None)
        if value is not None:
            return bool(value)
    return False


def _driver_total_memory(index):
    """The CUDA runtime's own total for ``index``, or None if it cannot be had.

    ``mem_get_info`` pins a primary context for the life of the process, so this is only reached
    from a caller that has ALREADY read device properties and so initialized CUDA itself.
    """
    try:
        import torch
        total = torch.cuda.mem_get_info(index)[1]
    except Exception:
        return None
    try:
        total = int(total)
    except (TypeError, ValueError):
        return None
    return total if total > 0 else None


def cuda_total_memory(index = 0, props = None):
    """Bytes of device memory torch can actually use on CUDA device ``index``.

    ``props.total_memory`` MAY understate that on a unified-memory host, where it can be the
    dedicated carve-out while the runtime's total spans the shared pool too. May, not does, so the
    rule is the one the ROCm path already follows: compare, and adopt only a LARGER driver total.
    Equal, smaller, zero or unreadable leaves the properties figure untouched, so a discrete card
    cannot change behaviour, and only integrated parts pay for the comparison at all.
    """
    try:
        import torch
        props = torch.cuda.get_device_properties(index) if props is None else props
        total = int(getattr(props, "total_memory", 0) or 0)
    except Exception:
        return 0
    if total <= 0 or not cuda_props_are_integrated(props):
        return total
    driver_total = _driver_total_memory(index)
    if driver_total is None:
        return total
    return max(total, driver_total)


def expandable_segments_unsupported():
    """Should ``expandable_segments:True`` be kept off here? Answered without importing torch
    or creating a CUDA context, so it is safe in the import-time allocator block."""
    forced = _env_tristate(FORCE_ENV)
    if forced is not None:
        return not forced
    if is_tegra_board():
        return True
    return _tegra_device_visible_for_free()
