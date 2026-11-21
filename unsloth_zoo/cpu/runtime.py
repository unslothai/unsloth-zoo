# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = [
    "cpu_flags",
    "cpu_bf16_supported",
    "cpu_fp16_supported",
    "cpu_autocast_dtype",
    "cpu_bnb_4bit_supported",
    "cpu_capabilities",
    "apply_cpu_security_patches",
    "available_ram",
]

import functools
import os
import platform
import warnings


@functools.cache
def cpu_flags():
    """Lower-cased ISA feature flags of the host CPU (empty when unreadable)."""
    flags = set()
    try:
        with open("/proc/cpuinfo", "r", encoding = "utf-8", errors = "ignore") as f:
            for line in f:
                key = line.split(":", 1)[0].strip().lower()
                if key in ("flags", "features"):
                    flags.update(line.split(":", 1)[1].lower().split())
                    break
    except OSError:
        pass
    if not flags and platform.system() == "Darwin":
        try:
            import subprocess
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.features", "machdep.cpu.leaf7_features"],
                capture_output = True, text = True, timeout = 5,
            ).stdout
            flags.update(out.lower().split())
        except Exception:
            pass
    return frozenset(flags)


@functools.cache
def cpu_bf16_supported():
    """bfloat16 compute in hardware: AMX / AVX512-BF16 on x86, the BF16 extension on Arm.

    Without it torch emulates bf16 matmuls and bf16 training is several times slower than fp32,
    so this is the answer to "should CPU training default to bf16". UNSLOTH_CPU_BF16=0/1 overrides.
    """
    override = os.environ.get("UNSLOTH_CPU_BF16", "").strip().lower()
    if override in ("0", "false", "no", "off"):
        return False
    if override in ("1", "true", "yes", "on"):
        return True
    flags = cpu_flags()
    if not flags:
        # No /proc/cpuinfo (Windows, macOS on Arm): time it instead.
        return _bf16_autocast_is_fast()
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64", "i686", "x86"):
        return bool(flags & {"amx_bf16", "avx512_bf16"})
    if machine in ("aarch64", "arm64"):
        return "bf16" in flags
    return False


def _bf16_autocast_is_fast():
    """Probes bf16 autocast matmul vs fp32: native bf16 (AVX512-BF16, AMX, Arm BF16 via oneDNN or ACL) runs 2-3x faster,
    the reference fallback 30-200x slower; a 1.5x threshold separates them."""
    try:
        import time
        import torch
        a = torch.randn(512, 512)
        b = torch.randn(512, 512)

        def best(dtype):
            times = []
            for _ in range(3):
                start = time.perf_counter()
                with torch.autocast("cpu", dtype = dtype, enabled = dtype is not None):
                    a @ b
                times.append(time.perf_counter() - start)
            return min(times[1:])
        return best(torch.bfloat16) < 1.5 * best(None)
    except Exception:
        return False


@functools.cache
def cpu_fp16_supported():
    """float16 matmuls in hardware fast enough to train with fp16 autocast: AMX-FP16 (Granite Rapids).
    Measured fp16 autocast vs fp32 for a LoRA MLP fwd+bwd: AVX512-FP16 without AMX-FP16 1.1-1.2x
    (those CPUs have AMX-BF16 at 1.9-2.7x), Arm Neoverse N2 FP16 0.91x, AVX2 / Zen 4 / Windows on Arm
    50-200x slower. Only consulted when bf16 is not native. UNSLOTH_CPU_FP16=0/1 overrides.
    """
    override = os.environ.get("UNSLOTH_CPU_FP16", "").strip().lower()
    if override in ("0", "false", "no", "off"):
        return False
    if override in ("1", "true", "yes", "on"):
        return True
    return "amx_fp16" in cpu_flags()


def cpu_autocast_dtype():
    """Mixed-precision dtype for CPU training: bf16 when native, else fp16 when native, else None (fp32)."""
    import torch
    if cpu_bf16_supported():
        return torch.bfloat16
    if cpu_fp16_supported():
        return torch.float16
    return None


@functools.cache
def cpu_bnb_4bit_supported():
    """A real NF4 Linear4bit forward + backward on CPU, not an import check: bitsandbytes resolves
    every native symbol to a raising stub when its library is dead, so only a call proves it."""
    if os.environ.get("UNSLOTH_CPU_LOAD_IN_4BIT", "").strip().lower() in ("0", "false", "no", "off"):
        return False
    try:
        import torch
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import bitsandbytes as bnb
            dtype = torch.bfloat16 if cpu_bf16_supported() else torch.float32
            layer = bnb.nn.Linear4bit(128, 64, bias = False, compute_dtype = dtype, quant_type = "nf4")
            weight = layer.weight
            if not getattr(weight, "bnb_quantized", False):
                layer.weight = weight._quantize("cpu")
            x = torch.randn(4, 128, dtype = dtype, requires_grad = True)
            y = layer(x)
            y.float().sum().backward()
            return bool(torch.isfinite(y).all()) and x.grad is not None and bool(torch.isfinite(x.grad).all())
    except Exception:
        return False


_CGROUP_ROOT = "/sys/fs/cgroup"
_PROC_SELF_CGROUP = "/proc/self/cgroup"


def _cgroup_memory_headroom():
    """Smallest memory.max - memory.current over this process's cgroup v2 and every ancestor (a limit set on an
    enclosing job / slice applies even when the leaf says max), or None when no finite limit is readable."""
    try:
        with open(_PROC_SELF_CGROUP) as f:
            path = next((line.split(":", 2)[2].strip() for line in f if line.startswith("0::")), None)
    except OSError:
        return None
    if path is None:
        return None
    headroom = None
    parts = [part for part in path.strip("/").split("/") if part]
    for depth in range(len(parts), -1, -1):
        base = os.path.join(_CGROUP_ROOT, *parts[:depth])
        try:
            with open(os.path.join(base, "memory.max")) as f:
                limit = f.read().strip()
            if limit == "max":
                continue
            with open(os.path.join(base, "memory.current")) as f:
                used = int(f.read())
            room = max(0, int(limit) - used)
        except (OSError, ValueError):
            continue  # not mounted here (container without a cgroup namespace) or the root, which has no limit
        headroom = room if headroom is None else min(headroom, room)
    return headroom


def available_ram():
    """Bytes of RAM still available to this process, or None: MemAvailable capped by the cgroup v2 memory.max headroom
    of its cgroup and every ancestor, and by memory_budget's reading (cgroup v1, memory.high), whichever is lower."""
    available = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
                    break
    except OSError:
        try:
            import psutil
            available = psutil.virtual_memory().available
        except Exception:
            return None
    try:
        headroom = _cgroup_memory_headroom()
    except Exception:
        headroom = None
    if headroom is not None:
        available = min(available or 1 << 62, headroom)
    # memory_budget also knows cgroup v1 (a v1 container sees the host's MemAvailable), memory.high and the ancestors
    # it can read; the stricter of the two readings wins.
    try:
        from unsloth_zoo.memory_budget import host_memory_bytes
        budget = host_memory_bytes()[1]
        if isinstance(budget, int) and not isinstance(budget, bool) and budget >= 0:
            available = budget if available is None else min(available, budget)
    except Exception:
        pass
    return available


def cpu_capabilities():
    import torch
    return {
        "machine": platform.machine(),
        "threads": torch.get_num_threads(),
        "bf16": cpu_bf16_supported(),
        "fp16": cpu_fp16_supported(),
        "amx": "amx_bf16" in cpu_flags(),
        "bnb_4bit": cpu_bnb_4bit_supported(),
    }


def apply_cpu_security_patches():
    """Installs device-independent security guards from temporary_patches; each guard file loads alone, with a private
    TEMPORARY_PATCHES list standing in for .common, since the package also pulls GPU kernels. Idempotent."""
    import importlib.util
    import sys
    import types

    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temporary_patches")
    applied = []
    # deepseek_v41: device-independent training fixes for the DeepSeek-V4.1 port (no-ops without it).
    # fp8_partial_block_dequant: FP8 checkpoints are dequantized on load on CPU; keep partial edge blocks right.
    for name in ("datasets_folder_metadata", "deepseek_v41", "fp8_partial_block_dequant"):
        module_name = f"{__package__}._guard_{name}"
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, os.path.join(root, f"{name}.py"))
            if spec is None:
                continue
            common_name = f"{__package__}.common"
            common = types.ModuleType(common_name)
            common.TEMPORARY_PATCHES = []
            previous = sys.modules.get(common_name)
            sys.modules[common_name] = common
            try:
                module = importlib.util.module_from_spec(spec)
                module.__package__ = __package__
                spec.loader.exec_module(module)
            except Exception:
                continue
            finally:
                if previous is None:
                    sys.modules.pop(common_name, None)
                else:
                    sys.modules[common_name] = previous
            sys.modules[module_name] = module
            module._unsloth_cpu_patches = list(common.TEMPORARY_PATCHES)
        for patch in getattr(module, "_unsloth_cpu_patches", []):
            try:
                patch()
                applied.append(patch.__name__)
            except Exception:
                pass
    return applied
