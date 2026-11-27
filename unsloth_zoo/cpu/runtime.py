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
    "cpu_compute_cores",
    "fit_cpu_threads",
    "available_ram",
]

import functools
import os
import platform
import sys
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


# Child interpreter: bf16 matmul on unsupported instructions (exit 132) would kill the user.
_BF16_PROBE_CODE = """
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

print("UNSLOTH_BF16_FAST=" + str(int(best(torch.bfloat16) < 1.5 * best(None))))
"""
_BF16_PROBE_TIMEOUT = 180


def _bf16_autocast_is_fast():
    """Probes bf16 autocast matmul vs fp32: native bf16 (AVX512-BF16, AMX, Arm BF16 via oneDNN or ACL) runs 2-3x faster,
    the reference fallback 30-200x slower; a 1.5x threshold separates them. Only reached where /proc/cpuinfo is
    unreadable (Windows, macOS). The probe runs in a short child process (once: cpu_bf16_supported is cached), so a
    crash, a timeout or any failure means bf16 is not fast and training stays fp32."""
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
        return False
    if done.returncode != 0:
        if done.returncode in (132, -4, 0xC000001D, -1073741795):
            warnings.warn("Unsloth: the bfloat16 matmul probe crashed with an illegal instruction; training in float32.")
        return False
    return "UNSLOTH_BF16_FAST=1" in (done.stdout or "").splitlines()


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


def _linux_physical_cores(cpus, root = "/sys/devices/system/cpu"):
    """Distinct (package, core) pairs among ``cpus``; None when sysfs topology is unreadable."""
    cores = set()
    try:
        for cpu in cpus:
            base = f"{root}/cpu{cpu}/topology/"
            with open(base + "core_id") as f:
                core = f.read().strip()
            with open(base + "physical_package_id") as f:
                package = f.read().strip()
            # Negative package/core: unknown topology; callers count CPUs, not one merged core.
            if int(package) < 0 or int(core) < 0:
                return None
            cores.add((package, core))
    except (OSError, ValueError):
        return None
    return len(cores) or None


def _cgroup_cpu_limit(proc_cgroup = "/proc/self/cgroup", root = "/sys/fs/cgroup"):
    """CPUs the cgroup quota allows (cgroup v2 cpu.max up the hierarchy, else v1 cfs), None when unlimited."""
    limits = []
    try:
        with open(proc_cgroup) as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    for line in lines:
        hierarchy, _, path = line.partition(":")[2].partition(":")
        # Own cgroup and ancestors up to the mount root bound it; absent paths are skipped.
        parts = [p for p in path.split("/") if p]
        if hierarchy == "":  # v2 unified
            for depth in range(len(parts), -1, -1):
                try:
                    with open(os.path.join(root, *parts[:depth], "cpu.max")) as f:
                        quota, period = (f.read().split() + ["100000"])[:2]
                except OSError:
                    continue
                if quota != "max" and int(period) > 0:
                    limits.append(int(quota) / int(period))
        elif "cpu" in hierarchy.split(","):
            for v1 in (f"{root}/cpu,cpuacct", f"{root}/cpu"):
                if not os.path.isdir(v1):
                    continue
                for depth in range(len(parts), -1, -1):
                    try:
                        with open(os.path.join(v1, *parts[:depth], "cpu.cfs_quota_us")) as f:
                            quota = int(f.read())
                        with open(os.path.join(v1, *parts[:depth], "cpu.cfs_period_us")) as f:
                            period = int(f.read())
                    except (OSError, ValueError):
                        continue
                    if quota > 0 and period > 0:
                        limits.append(quota / period)
                break
    if not limits:
        return None
    import math
    return max(1, math.ceil(min(limits)))


# OpenMP thread binding. With any of these set, the OpenMP runtime binds the main thread to its first place when torch
# loads (OMP_PLACES=cores: one core and its SMT sibling), so the main thread's affinity no longer shows the CPUs the
# process was given; the pool's worker threads hold the other places.
_OMP_BIND_ENV = ("OMP_PROC_BIND", "OMP_PLACES", "GOMP_CPU_AFFINITY", "KMP_AFFINITY")


def _omp_binding_requested():
    for name in _OMP_BIND_ENV:
        value = os.environ.get(name, "").strip().lower()
        if value and not (name == "OMP_PROC_BIND" and value == "false") and \
                not (name == "KMP_AFFINITY" and value in ("disabled", "none")):
            return True
    return False


_OMP_RUNTIMES = ("libgomp", "libiomp", "libomp")
_OMP_SONAMES = ("libgomp.so.1", "libiomp5.so", "libomp.so", "libomp.so.5")


def _torch_omp_runtime(paths):
    """The loaded OpenMP runtime that is torch's own, or None when that is ambiguous. Another wheel can bundle its own
    copy (auditwheel renames it libgomp-<hash>.so.1 under <package>.libs); its place list is not torch's."""
    if len(paths) == 1:
        return paths[0]
    try:
        import torch
        torch_dir = os.path.dirname(os.path.realpath(torch.__file__))
    except Exception:
        return None
    # pip wheels: torch/lib (x86) or torch.libs (aarch64)
    own = [p for p in paths if os.path.realpath(p).startswith((torch_dir + os.sep, torch_dir + ".libs" + os.sep))]
    if own:
        return own[0] if len(own) == 1 else None
    # conda / distro torch links the system runtime by its soname
    plain = [p for p in paths if os.path.basename(p) in _OMP_SONAMES]
    return plain[0] if len(plain) == 1 else None


def _omp_place_cpus():
    """CPUs in the place list of torch's OpenMP runtime, which it built from the process mask before binding the main
    thread (OMP_PLACES / GOMP_CPU_AFFINITY narrow it to what the user listed). Empty when unknown."""
    import ctypes
    try:
        with open("/proc/self/maps") as f:
            paths = sorted({line.split()[-1] for line in f if "/" in line
                            and os.path.basename(line.split()[-1]).startswith(_OMP_RUNTIMES)})
    except OSError:
        return set()
    path = _torch_omp_runtime(paths)
    if path is None:
        return set()
    try:
        lib = ctypes.CDLL(path)
        cpus = set()
        for place in range(int(lib.omp_get_num_places())):
            count = int(lib.omp_get_place_num_procs(place))
            if count <= 0:
                continue
            ids = (ctypes.c_int * count)()
            lib.omp_get_place_proc_ids(place, ids)
            cpus.update(int(cpu) for cpu in ids)
    except (OSError, AttributeError, ValueError):
        return set()
    return cpus


def _process_affinity():
    """CPUs this process may run on. Under OpenMP binding the main thread's mask is only its own place, so the place
    list of torch's OpenMP runtime is used; when that runtime is ambiguous, the union of every thread's mask after one
    small op has started torch's pool (never wider than where the pool runs)."""
    cpus = set(os.sched_getaffinity(0))
    if not _omp_binding_requested():
        return cpus
    try:
        import torch
        places = _omp_place_cpus()
        if places:
            return cpus | places
        threads = torch.get_num_threads()
        if threads > 1:
            torch.ones(threads * 65536).add_(1)
        for entry in os.listdir("/proc/self/task"):
            try:
                cpus |= set(os.sched_getaffinity(int(entry)))
            except (OSError, ValueError):
                pass
    except Exception:
        pass
    return cpus


def _binding_narrowed_default(current):
    """torch sized its default pool from the main thread's bound mask (aarch64 wheels: 1 thread under OMP_PROC_BIND)."""
    if not sys.platform.startswith("linux") or not _omp_binding_requested():
        return False
    try:
        return current <= len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return False


def cpu_compute_cores():
    """Physical cores (SMT siblings share units), affinity mask, quota-capped; None off Linux."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        cpus = sorted(_process_affinity())
    except (AttributeError, OSError):
        return None
    if not cpus:
        return None
    cores = _linux_physical_cores(cpus) or len(cpus)
    try:
        quota = _cgroup_cpu_limit()
    except Exception:
        quota = None
    return max(1, min(cores, quota) if quota else cores)


def _affinity_split_per_rank(ranks):
    """True if torchrun --numa-binding or mpirun --bind-to gave the rank a slice; not taskset."""
    if ranks <= 1 or not sys.platform.startswith("linux"):
        return False
    try:
        mine = _process_affinity()
        parent = os.sched_getaffinity(os.getppid())
    except (AttributeError, OSError):
        return False
    return bool(mine) and mine < parent and len(mine) * ranks <= len(parent)


def _local_world_size():
    # The variables Accelerate's get_cpu_distributed_information reads (torchrun, Intel MPI, Open MPI, MVAPICH).
    for name in ("LOCAL_WORLD_SIZE", "MPI_LOCALNRANKS", "OMPI_COMM_WORLD_LOCAL_SIZE", "MV2_COMM_WORLD_LOCAL_SIZE"):
        try:
            value = int(os.environ.get(name, ""))
        except ValueError:
            continue
        if value >= 1:
            return value
    return 1


def _launcher_set_one_thread():
    """torchrun (nproc > 1) and accelerate launch export OMP_NUM_THREADS=1 when the user did not set it, so CPU ranks
    run single-threaded; a 1 next to their markers is the launcher's, not the user's (UNSLOTH_CPU_THREADS=1 keeps 1)."""
    if os.environ.get("OMP_NUM_THREADS", "").strip() != "1":
        return False
    if os.environ.get("TORCHELASTIC_RUN_ID") and _local_world_size() > 1:
        return True
    return "ACCELERATE_MIXED_PRECISION" in os.environ


_THREADS_FITTED = None
# Default ceiling on the fitted count. A training step is thousands of small ops (a 0.6B LoRA step at 2 x 128 tokens
# is ~3000 GEMMs plus elementwise ops), each an OpenMP fork / join: past ~16 threads the barriers cost more than the
# cores add, and oneDNN's AMX bf16 matmul for 256 x 3072 x 1024 runs 10-15x slower at 32+ threads than at 16. Xeon
# 8559C (48 cores per node), Qwen3-0.6B: SFT LoRA 0.25 s/step at 16 threads vs 24-50 s at 48; GRPO and full
# finetuning best or within spread at 16 too. UNSLOTH_CPU_THREADS / OMP_NUM_THREADS still set any count.
_DEFAULT_THREAD_CAP = 16


def fit_cpu_threads():
    """Sizes torch's intra-op pool for CPU training. Idempotent; returns (threads, reason).
    UNSLOTH_CPU_THREADS wins; a user's OMP_NUM_THREADS / MKL_NUM_THREADS is kept; otherwise one thread per physical
    core in the affinity mask / cgroup quota, split across the local ranks of torchrun / accelerate, and at most
    _DEFAULT_THREAD_CAP. Never raises the count above torch's default except to undo a launcher's OMP_NUM_THREADS=1
    or an OpenMP binding (OMP_PROC_BIND / OMP_PLACES) that sized torch's default from the main thread's one place."""
    global _THREADS_FITTED
    import torch
    if _THREADS_FITTED is not None:
        return _THREADS_FITTED
    current = torch.get_num_threads()
    reason = "default"
    target = None
    explicit = os.environ.get("UNSLOTH_CPU_THREADS", "").strip()
    if explicit:
        try:
            target = max(1, int(explicit))
            reason = "UNSLOTH_CPU_THREADS"
        except ValueError:
            warnings.warn(f"Unsloth: UNSLOTH_CPU_THREADS = {explicit!r} is not an integer; ignoring it.")
    if target is None:
        launcher = _launcher_set_one_thread()
        # A launcher only ever exports OMP_NUM_THREADS, so an MKL_NUM_THREADS next to it is still the user's.
        user_mkl = bool(os.environ.get("MKL_NUM_THREADS", "").strip())
        user_set = user_mkl or bool(os.environ.get("OMP_NUM_THREADS", "").strip())
        if user_set and (user_mkl or not launcher):
            _THREADS_FITTED = (current, "OMP_NUM_THREADS / MKL_NUM_THREADS")
            _pin_omp_env(current)
            return _THREADS_FITTED
        cores = cpu_compute_cores()
        ranks = _local_world_size()
        if cores is None:
            if launcher:
                target, reason = max(1, (os.cpu_count() or 1) // ranks), "launcher OMP_NUM_THREADS=1 undone"
        else:
            fitted = max(1, cores // ranks)
            if _affinity_split_per_rank(ranks):
                # cores already counts this rank's own CPUs; only a cgroup quota is still shared by the ranks.
                try:
                    quota = _cgroup_cpu_limit() if sys.platform.startswith("linux") else None
                except Exception:
                    quota = None
                fitted = max(1, min(cores, quota // ranks) if quota else cores)
            capped = fitted > _DEFAULT_THREAD_CAP
            fitted = min(fitted, _DEFAULT_THREAD_CAP)
            if launcher:
                target, reason = fitted, "launcher OMP_NUM_THREADS=1 undone"
            elif fitted > current and _binding_narrowed_default(current):
                target, reason = fitted, "OpenMP binding narrowed torch's default"
            elif fitted < current:
                try:
                    quota = _cgroup_cpu_limit() if sys.platform.startswith("linux") else None
                except Exception:
                    quota = None
                basis = "cgroup CPU quota" if quota is not None and quota <= cores else "physical cores"
                target, reason = fitted, basis + (f" / {ranks} local ranks" if ranks > 1 else "")
                if capped:
                    reason += f", capped at {_DEFAULT_THREAD_CAP}"
    if target is not None and target != current:
        try:
            torch.set_num_threads(target)
            print(f"Unsloth (CPU): {current} -> {target} torch threads ({reason}); UNSLOTH_CPU_THREADS overrides.")
        except Exception:
            target = current
    _THREADS_FITTED = (torch.get_num_threads(), reason)
    _pin_omp_env(_THREADS_FITTED[0])
    return _THREADS_FITTED


def _pin_omp_env(threads):
    """Exports the fitted count as OMP_NUM_THREADS, since Accelerate's PartialState sets torch
    threads when unset (mpirun), undoing this fit and UNSLOTH_CPU_THREADS; workers inherit it."""
    if not os.environ.get("OMP_NUM_THREADS", "").strip():
        os.environ["OMP_NUM_THREADS"] = str(max(1, int(threads)))


@functools.cache
def _cpu_vendor():
    """Lower-cased CPU vendor string ("authenticamd", "genuineintel", ...; "" when unknown)."""
    try:
        with open("/proc/cpuinfo", "r", encoding = "utf-8", errors = "ignore") as f:
            for line in f:
                key, _, value = line.partition(":")
                if key.strip().lower() == "vendor_id":
                    return value.strip().lower()
    except OSError:
        pass
    # Windows: "AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD".
    return (platform.processor() or "").lower()


@functools.cache
def _cpu_family_model():
    """(x86 family, model) from /proc/cpuinfo or the Windows processor string; None when unknown."""
    family = model = None
    try:
        with open("/proc/cpuinfo", "r", encoding = "utf-8", errors = "ignore") as f:
            for line in f:
                key, _, value = line.partition(":")
                key = key.strip().lower()
                if key == "cpu family" and family is None:
                    family = int(value)
                elif key == "model" and model is None:
                    model = int(value)
                if family is not None and model is not None:
                    return family, model
    except (OSError, ValueError):
        pass
    import re
    found = re.search(r"family\s+(\d+)\s+model\s+(\d+)", platform.processor() or "", re.I)
    return (int(found.group(1)), int(found.group(2))) if found else None


# Zen 4 models of family 25 (Genoa / Raphael / Phoenix / Bergamo / Siena): their siblings slowed decode by 4-13%.
_ZEN4_MODELS = (range(0x10, 0x20), range(0x60, 0x80), range(0xA0, 0xB0))


def _zen3():
    """Family 25 outside the Zen 4 models (Milan, Vermeer, Cezanne, Rembrandt): the class the decode SMT gain was
    measured on. A cloud VM can hide a Zen 4 part's AVX-512 / bf16 (an EPYC 9V74 runner exposing only AVX2), so the
    family / model decides, not the flags."""
    fm = _cpu_family_model()
    return fm is not None and fm[0] == 25 and not any(fm[1] in r for r in _ZEN4_MODELS)


def _logical_cpus():
    """Logical CPUs (SMT siblings counted) in the affinity mask, quota-capped on Linux; os.cpu_count() elsewhere."""
    if sys.platform.startswith("linux"):
        try:
            cpus = len(_process_affinity())
        except (AttributeError, OSError):
            return None
        try:
            quota = _cgroup_cpu_limit()
        except Exception:
            quota = None
        return max(1, min(cpus, quota) if quota else cpus)
    return os.cpu_count()


# Fitted counts a rollout may widen: the defaults, not a user's, a launcher's or a quota's.
_ROLLOUT_WIDEN_REASONS = ("default", "physical cores")


def rollout_threads():
    """torch threads for CPU generate (decode, GRPO rollouts), or None to keep the training count.

    Training uses one thread per physical core (SMT siblings made SFT and GRPO 13-20% slower). Decoding is batched
    GEMVs that stream every weight per token, and on AMD Zen 3 without native bf16 (EPYC 7763) the siblings help it:
    2-core x 2-SMT EPYC 7763 runners decoded Qwen3-0.6B GRPO rollouts about 35% faster on 4 threads with training
    left at 2 (step -19 to -21% on Windows, -10% on Linux), while Zen 4 (EPYC 9V45) was 4-13% slower and Intel was
    not measured, so only Zen 3 widens. UNSLOTH_CPU_ROLLOUT_THREADS=N sets the count for every host, 0 keeps the training count."""
    import torch
    explicit = os.environ.get("UNSLOTH_CPU_ROLLOUT_THREADS", "").strip().lower()
    current = torch.get_num_threads()
    if explicit:
        try:
            target = int(explicit)
        except ValueError:
            return None
        return target if target > 0 and target != current else None
    fitted = _THREADS_FITTED
    if fitted is None or fitted[1] not in _ROLLOUT_WIDEN_REASONS or fitted[0] != current or _local_world_size() > 1:
        return None
    if platform.machine().lower() not in ("x86_64", "amd64") or "authenticamd" not in _cpu_vendor() or not _zen3():
        return None
    if cpu_bf16_supported():
        return None
    logical = _logical_cpus()
    if not logical:
        return None
    target = min(logical, 2 * current, _DEFAULT_THREAD_CAP)
    return target if target > current else None


_CGROUP_ROOT = "/sys/fs/cgroup"
_PROC_SELF_CGROUP = "/proc/self/cgroup"


def _cgroup_memory_headroom():
    """Min memory.max - memory.current over cgroup v2 and ancestors; None if no finite limit."""
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
    # memory_budget reads cgroup v1 and memory.high (v1 sees host MemAvailable); stricter wins.
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
    # Guards: deepseek_v41 (DeepSeek-V4.1 port fixes; no-op without it), fp8_partial_block_dequant
    # (CPU FP8 edge blocks), hybrid_sliding_masks (StaticCache on Inkling past sliding window).
    for name in ("datasets_folder_metadata", "deepseek_v41", "fp8_partial_block_dequant", "hybrid_sliding_masks"):
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
