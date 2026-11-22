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


# Runs in a child interpreter: a bf16 matmul on a CPU whose torch build picks an instruction the CPU lacks dies with
# an illegal instruction (Windows 0xc000001d, exit 132), which would otherwise kill the user's process.
_BF16_PROBE_CODE = """
import time
import torch
a = torch.randn(512, 512)
b = torch.randn(512, 512)

def best(dtype):
    times = []
    for _ in range(8):
        start = time.perf_counter()
        with torch.autocast("cpu", dtype = dtype, enabled = dtype is not None):
            a @ b
        times.append(time.perf_counter() - start)
    return min(times[1:])  # each matmul is ~1 ms: the minimum of 7 drops runs a busy host preempted

ratio = best(torch.bfloat16) / max(best(None), 1e-9)
print("UNSLOTH_BF16_RATIO=%.4f" % ratio)
print("UNSLOTH_BF16_FAST=" + str(int(ratio < 1.5)))
"""
# Only a clear answer is kept on disk: native bf16 measures 0.25-0.5x fp32, the reference fallback 30-200x, so a ratio
# between these bounds is a noisy run (a loaded host) that decides this process only.
_PROBE_CACHE_RATIO_BOUNDS = (1.0, 3.0)
_BF16_PROBE_TIMEOUT = 180
_ILLEGAL_INSTRUCTION_CODES = (132, -4, 0xC000001D, -1073741795)
_ILLEGAL_INSTRUCTION_WARNING = (
    "Unsloth: the bfloat16 matmul probe crashed with an illegal instruction; training in float32."
)
_PROBE_CACHE_NAME = "cpu_bf16_probe.json"
_PROBE_CACHE_MAX = 64


def _probe_cache_path():
    """~/.cache/unsloth/cpu_bf16_probe.json (UNSLOTH_CPU_PROBE_CACHE_DIR moves it), or None when disabled
    (UNSLOTH_CPU_PROBE_CACHE=0) or there is no usable home directory."""
    if os.environ.get("UNSLOTH_CPU_PROBE_CACHE", "").strip().lower() in ("0", "false", "no", "off"):
        return None
    root = os.path.expanduser(os.environ.get("UNSLOTH_CPU_PROBE_CACHE_DIR", "").strip())
    if root == "":
        home = os.path.expanduser("~")
        # No resolvable home (restricted containers): skip the cache rather than write a literal ~ into the CWD.
        if home == "~" or not os.path.isdir(home):
            return None
        root = os.path.join(home, ".cache", "unsloth")
    return os.path.join(root, _PROBE_CACHE_NAME)


def _probe_cache_key():
    """What the verdict depends on: the interpreter and torch the child runs, the CPU, and the probe itself."""
    import hashlib
    import json
    import sys
    torch = sys.modules.get("torch")
    torch_version = getattr(torch, "__version__", None)
    if not isinstance(torch_version, str):
        try:
            import importlib.metadata
            torch_version = importlib.metadata.version("torch")
        except Exception:
            torch_version = None
    system = platform.system()
    cpu = ""
    if system == "Windows":
        cpu = os.environ.get("PROCESSOR_IDENTIFIER", "").strip() or platform.processor()  # family, model, stepping
    elif system == "Darwin":
        try:
            import subprocess
            cpu = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output = True, text = True, timeout = 5,
            ).stdout.strip()
        except Exception:
            pass
    # The child inherits the environment, and oneDNN / ATen read these to pick (or cap) the bf16 kernels:
    # ONEDNN_MAX_CPU_ISA=AVX2 on an AMX CPU turns a 0.4x bf16 matmul into an 85x one.
    dispatch_env = sorted(
        (name, value) for name, value in os.environ.items()
        if name.startswith(("ONEDNN_", "DNNL_", "TORCH_MKLDNN_")) or name == "ATEN_CPU_CAPABILITY"
    )
    # Same version string, different build (a reinstall, conda vs pip variants): torch rewrites version.py on install.
    torch_stamp = None
    try:
        import importlib.util
        spec = importlib.util.find_spec("torch")
        if spec is not None and spec.origin:
            version_file = os.path.join(os.path.dirname(spec.origin), "version.py")
            torch_stamp = [spec.origin, os.path.getmtime(version_file), os.path.getsize(version_file)]
    except Exception:
        pass
    identity = [
        sys.executable, list(sys.version_info[:3]), torch_version, system, platform.machine(),
        platform.node(), cpu, _BF16_PROBE_CODE, dispatch_env, torch_stamp,
    ]
    return hashlib.sha256(json.dumps(identity).encode("utf-8")).hexdigest()


def _probe_cache_read(path):
    import json
    try:
        with open(path, "r", encoding = "utf-8") as f:
            data = json.load(f)
    except Exception:  # missing, unreadable or corrupt: probe as if there were no cache
        return {}
    return data if isinstance(data, dict) else {}


def _probe_cache_lookup(path, key):
    entry = _probe_cache_read(path).get(key)
    if isinstance(entry, dict) and isinstance(entry.get("fast"), bool):
        return entry["fast"], bool(entry.get("illegal_instruction"))
    return None


def _probe_cache_store(path, key, fast, illegal_instruction):
    """Atomic (temporary file + os.replace), so concurrent first runs never leave a torn file; two writers at once can
    drop each other's new entry, which only costs a later probe. Any failure (read-only dir, full disk) is ignored."""
    import json
    import tempfile
    tmp = None
    try:
        data = _probe_cache_read(path)
        data.pop(key, None)
        data[key] = {"fast": bool(fast), "illegal_instruction": bool(illegal_instruction)}
        while len(data) > _PROBE_CACHE_MAX:
            data.pop(next(iter(data)))
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok = True)
        fd, tmp = tempfile.mkstemp(prefix = ".cpu_bf16_probe_", suffix = ".tmp", dir = directory)
        with os.fdopen(fd, "w", encoding = "utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, path)
        tmp = None
    except Exception:
        pass
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _bf16_autocast_is_fast():
    """Probes bf16 autocast matmul vs fp32: native bf16 (AVX512-BF16, AMX, Arm BF16 via oneDNN or ACL) runs 2-3x faster,
    the reference fallback 30-200x slower; a 1.5x threshold separates them. Only reached where /proc/cpuinfo is
    unreadable (Windows, macOS). The probe runs in a short child process (once: cpu_bf16_supported is cached), so a
    crash, a timeout or any failure means bf16 is not fast and training stays fp32.

    Every import of unsloth on such a CPU-only host asks, and each new process (Studio workers) would pay a second
    torch import plus the benchmark, so settled answers (a verdict with a clear speed ratio, or an illegal-instruction
    crash) are kept on disk, keyed by interpreter, torch, CPU and probe. Timeouts, spawn errors, other failures and
    ratios too close to the threshold to trust are not. UNSLOTH_CPU_PROBE_CACHE=0 turns the disk cache off."""
    try:
        path = _probe_cache_path()
        key = _probe_cache_key() if path is not None else None
    except Exception:
        path = key = None
    if path is not None:
        cached = _probe_cache_lookup(path, key)
        if cached is not None:
            fast, illegal_instruction = cached
            if illegal_instruction:
                warnings.warn(_ILLEGAL_INSTRUCTION_WARNING)
            return fast
    verdict = _run_bf16_probe()
    if verdict is None:
        return False
    fast, illegal_instruction, settled = verdict
    if settled and path is not None:
        _probe_cache_store(path, key, fast, illegal_instruction)
    return fast


def _run_bf16_probe():
    """(fast, illegal_instruction, settled) when the child answered, None when it did not (timeout, spawn error, other
    failures), which reads as not fast. Only settled answers (a clear ratio, an illegal instruction) are cached."""
    import subprocess
    import sys
    import tempfile
    command = [sys.executable]
    if sys.version_info >= (3, 11):
        command.append("-P")  # do not put the working directory on sys.path
    command += ["-c", _BF16_PROBE_CODE]
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(name, "1")  # one thread is enough to tell 2x faster from 30x slower
    try:
        done = subprocess.run(
            command, capture_output = True, text = True, timeout = _BF16_PROBE_TIMEOUT, env = env,
            cwd = tempfile.gettempdir(), creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        return None
    if done.returncode != 0:
        if done.returncode in _ILLEGAL_INSTRUCTION_CODES:
            warnings.warn(_ILLEGAL_INSTRUCTION_WARNING)
            return False, True, True
        return None
    lines = (done.stdout or "").splitlines()
    fast = True if "UNSLOTH_BF16_FAST=1" in lines else False if "UNSLOTH_BF16_FAST=0" in lines else None
    if fast is None:
        return None
    ratio = None
    for line in lines:
        if line.startswith("UNSLOTH_BF16_RATIO="):
            try:
                ratio = float(line.split("=", 1)[1])
            except ValueError:
                pass
    low, high = _PROBE_CACHE_RATIO_BOUNDS
    settled = ratio is not None and (ratio <= low or ratio >= high)
    return fast, False, settled


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
