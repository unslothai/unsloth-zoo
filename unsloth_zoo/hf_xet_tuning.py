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

"""Size ``hf_xet``'s download buffers and concurrency from the machine actually running.

hf_xet sizes its reconstruction buffers from constants, not available RAM: a 2GB floor plus 512MB
per concurrent file (8 of them) capped at 8GB, plus a 1GB prefetch floor. ``HF_XET_HIGH_PERFORMANCE``
(once Unsloth's default) raises that cap to 64GB and the stream count to 124, and is applied AFTER
the environment is read, so it DISCARDS any explicit ``HF_XET_RECONSTRUCTION_*`` cap; bounding memory
requires turning it off, not just setting the caps.

Values come from total RAM (cgroup-aware: inside a container ``psutil`` reports the HOST's RAM, which
is how a 16GB CI runner ends up with an 8GB buffer) and core count. They are emitted as environment
variables because ``hf_xet`` reads its config once, natively, before Python can reach it, with
``setdefault`` semantics so an explicit user setting wins.

Durations MUST carry a unit suffix: a bare ``"60"`` is IGNORED (the 300s default stands), ``"60s"``
is honoured.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = [
    "SystemProfile",
    "system_profile",
    "xet_env_overrides",
    "xet_log_env",
    "apply_xet_env",
    "scan_xet_log",
    "XET_HIGH_PERFORMANCE_VARS",
]

_GB = 1_000_000_000  # hf_xet's ByteSize renders/parses SI units, so stay in SI.
_MB = 1_000_000

# Both spellings enable high-performance mode in xet-core; both must be cleared for a cap to hold.
XET_HIGH_PERFORMANCE_VARS = ("HF_XET_HIGH_PERFORMANCE", "HF_XET_HP")

_TRUTHY = {"1", "true", "yes", "on"}

# (exclusive upper bound on total RAM, buffer_limit, buffer_size, perfile_size, max_concurrent_files)
_TIERS = (
    (12 * _GB, 1 * _GB, 512 * _MB, 128 * _MB, 4),
    (24 * _GB, 2 * _GB, 768 * _MB, 192 * _MB, 6),
    (None,     4 * _GB, 1 * _GB,   256 * _MB, 8),
)

# Below this much usable RAM, callers prefer HTTP over Xet (see hf_xet_health).
MIN_XET_RAM_BYTES = 4 * _GB


def _is_true(value: Optional[str]) -> bool:
    return value is not None and str(value).strip().lower() in _TRUTHY


def _read_first_line(path: Path) -> Optional[str]:
    try:
        with open(path, "r") as f:
            return f.readline().strip()
    except OSError:
        return None


# Module constant so tests can point the whole layer at a fixture tree.
CGROUP_ROOT = Path("/sys/fs/cgroup")


def _proc_self_cgroup() -> list[str]:
    """Lines of ``/proc/self/cgroup``, or empty if it is not readable (non-Linux, hidden procfs)."""
    try:
        with open("/proc/self/cgroup", "r") as f:
            return [line.strip() for line in f if line.strip()]
    except OSError:
        return []


def _walk_to_root(root: Path, rel: Optional[str]) -> list[Path]:
    """``root/rel`` and every ancestor up to *root*, innermost first: a parent slice's limit binds too."""
    dirs: list[Path] = []
    if rel and rel != "/":
        current = root / rel.lstrip("/")
        while True:
            dirs.append(current)
            if current == root or root not in current.parents:
                break
            current = current.parent
    dirs.append(root)
    return dirs


def _cgroup_v2_dirs() -> list[Path]:
    """Candidate cgroup v2 dirs for THIS process, innermost first.

    The root ``/sys/fs/cgroup/memory.max`` usually does not exist, so the real limit lives at the
    path named in ``/proc/self/cgroup``.
    """
    root = CGROUP_ROOT
    if not root.is_dir():
        return []
    rel = None
    # The v2 line is "0::<path>"; under systemd hybrid mode v1 lines share the file, so scan.
    for line in _proc_self_cgroup():
        if line.startswith("0::"):
            rel = line[3:].strip()
            break
    return _walk_to_root(root, rel)


def _cgroup_v1_dirs(controller: str) -> list[Path]:
    """Candidate cgroup v1 dirs for *controller*, innermost first, then the mount root.

    In a container the root read already suffices: runc bind-mounts the container's own cgroup dir
    onto ``/sys/fs/cgroup/<controller>``. Outside one it does not: a Slurm step
    (``/sys/fs/cgroup/memory/slurm/uid_<uid>/job_<id>/step_<n>``) or a systemd scope with
    ``MemoryLimit=``/``CPUQuota=`` reads the "unlimited" sentinel at the root, hiding the real
    ceiling -- which is how a 32 GB Slurm step on a 1 TB node gets OOM killed.
    """
    root = CGROUP_ROOT / controller
    if not root.is_dir():
        return []
    rel = None
    for line in _proc_self_cgroup():
        parts = line.split(":", 2)
        # "<id>:<controllers>:<path>"; mounts are often combined ("cpu,cpuacct"), so split.
        if len(parts) == 3 and controller in parts[1].split(","):
            rel = parts[2]
            break
    return _walk_to_root(root, rel)


def _parse_limit(raw: Optional[str]) -> Optional[int]:
    """``"max"`` (or an unparseable value) means unlimited."""
    if raw is None or raw == "max":
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    # cgroup v1 spells "unlimited" as a near-2^63 sentinel.
    return value if 0 < value < (1 << 62) else None


def cgroup_memory_limit() -> Optional[int]:
    """Smallest binding memory ceiling from cgroup v2 (``memory.max``/``memory.high``) or v1."""
    limits: list[int] = []
    for d in _cgroup_v2_dirs():
        for name in ("memory.max", "memory.high"):
            value = _parse_limit(_read_first_line(d / name))
            if value is not None:
                limits.append(value)
    for d in _cgroup_v1_dirs("memory"):
        value = _parse_limit(_read_first_line(d / "memory.limit_in_bytes"))
        if value is not None:
            limits.append(value)
    return min(limits) if limits else None


def cgroup_cpu_limit() -> Optional[float]:
    """CPU ceiling in whole cores from cgroup v2 ``cpu.max`` ("<quota> <period>") or v1 cfs quota."""
    quotas: list[float] = []
    for d in _cgroup_v2_dirs():
        raw = _read_first_line(d / "cpu.max")
        if not raw:
            continue
        parts = raw.split()
        if len(parts) == 2 and parts[0] != "max":
            try:
                quota, period = int(parts[0]), int(parts[1])
                if quota > 0 and period > 0:
                    quotas.append(quota / period)
            except ValueError:
                pass
    for d in _cgroup_v1_dirs("cpu"):
        quota = _parse_limit(_read_first_line(d / "cpu.cfs_quota_us"))
        period = _parse_limit(_read_first_line(d / "cpu.cfs_period_us"))
        if quota is not None and period:
            quotas.append(quota / period)
    return min(quotas) if quotas else None


@dataclass(frozen = True)
class SystemProfile:
    total_ram_bytes: int
    available_ram_bytes: int
    cpu_count: int
    ram_source: str
    cpu_source: str


def _psutil_memory() -> tuple[Optional[int], Optional[int]]:
    try:
        import psutil  # type: ignore
    except Exception:
        return (None, None)
    try:
        vm = psutil.virtual_memory()
        return (int(vm.total), int(vm.available))
    except Exception:
        return (None, None)


def _sysconf_memory() -> Optional[int]:
    """Fallback when psutil is absent."""
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (OSError, ValueError, AttributeError):
        return None


def system_profile() -> SystemProfile:
    """Usable RAM and cores for THIS process, preferring a cgroup limit over the host's totals."""
    host_total, host_available = _psutil_memory()
    if host_total is None:
        host_total = _sysconf_memory()
    ram_source = "psutil/sysconf"

    cg_mem = cgroup_memory_limit()
    if cg_mem is not None and (host_total is None or cg_mem < host_total):
        total = cg_mem
        available = cg_mem if host_available is None else min(host_available, cg_mem)
        ram_source = "cgroup"
    else:
        total = host_total or 0
        available = host_available if host_available is not None else total

    try:
        cpus = len(os.sched_getaffinity(0))  # respects taskset / CPU pinning
        cpu_source = "affinity"
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
        cpu_source = "cpu_count"
    cg_cpu = cgroup_cpu_limit()
    if cg_cpu is not None and cg_cpu > 0 and cg_cpu < cpus:
        # A fractional quota still binds: Kubernetes "cpu: 500m" is cpu.max "50000 100000" = 0.5.
        # Requiring >= 1 fell back to the host's core count, so a half-core pod opened 64 streams.
        cpus = max(1, int(cg_cpu))
        cpu_source = "cgroup"

    return SystemProfile(
        total_ram_bytes = int(total),
        available_ram_bytes = int(available),
        cpu_count = max(1, int(cpus)),
        ram_source = ram_source,
        cpu_source = cpu_source,
    )


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


# Shortened Xet client timeouts: safe in a download child we supervise, wrong process-wide.
_FAIL_FAST_KEYS = (
    "HF_XET_CLIENT_READ_TIMEOUT",
    "HF_XET_CLIENT_CONNECT_TIMEOUT",
    "HF_XET_CLIENT_RETRY_MAX_ATTEMPTS",
    "HF_XET_CLIENT_RETRY_MAX_DURATION",
)


def xet_env_overrides(
    profile: Optional[SystemProfile] = None,
    *,
    throttled: bool = False,
    fail_fast: bool = True,
) -> dict[str, str]:
    """RAM/CPU-derived ``HF_XET_*`` settings. Pure: returns a dict, touches no environment.

    *throttled* halves the stream ceiling; set after a logged "429 Too Many Requests", where the
    account rather than the machine is the limiting factor. *fail_fast* keeps the shortened Xet
    timeouts, which only suit callers whose failure our Xet -> HTTP ladder can act on. Unknown total
    RAM (0) yields the smallest tier: guessing low costs throughput, guessing high costs an OOM.
    """
    profile = profile or system_profile()
    total = profile.total_ram_bytes or _TIERS[0][0] - 1
    tier = next(t for t in _TIERS if t[0] is None or total < t[0])
    _, limit, size, perfile, max_files = tier

    cpus = profile.cpu_count
    # More files in flight than cores buys nothing and multiplies the per-file buffer.
    max_files = _clamp(max_files, 2, max(2, cpus))
    streams = _clamp(cpus * 2, 4, 64)
    if throttled:
        streams = max(4, streams // 2)

    # hf_xet grows the buffer to size + max_files * perfile and clamps it at limit; keeping limit at
    # or above that sum makes it state the true ceiling instead of truncating the other two.
    limit = max(limit, size + max_files * perfile)

    env = {
        # Memory. The effective buffer is size + max_files * perfile, capped by limit.
        "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT": str(limit),
        "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE": str(size),
        "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE": str(perfile),
        "HF_XET_RECONSTRUCTION_MIN_PREFETCH_BUFFER": str(min(size, 512 * _MB)),
        "HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS": str(max_files),
        # ac_* is the adaptive-concurrency band; the initial value stays under the ceiling so a slow
        # link ramps up instead of opening 16 streams into a stall.
        "HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY": str(streams),
        "HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY": str(_clamp(cpus, 2, 8)),
        "HF_XET_CLIENT_AC_MIN_DOWNLOAD_CONCURRENCY": "1",
        # Fail fast so OUR ladder decides instead of hf_xet retrying for ~6 minutes. Bare integers
        # are ignored; the unit suffix is required.
        "HF_XET_CLIENT_READ_TIMEOUT": "60s",
        "HF_XET_CLIENT_CONNECT_TIMEOUT": "20s",
        "HF_XET_CLIENT_RETRY_MAX_ATTEMPTS": "2",
        "HF_XET_CLIENT_RETRY_MAX_DURATION": "30s",
        # The chunk cache only pays off when re-fetching known chunks; upstream default is already
        # 0, pinned so a stray value cannot raise it.
        "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0",
        # Applied AFTER the env in xet-core, so leaving this on would discard every cap above.
        "HF_XET_HIGH_PERFORMANCE": "0",
        "HF_XET_HP": "0",
    }

    if not fail_fast:
        # xet-core reads its config once per process and these four apply to every CAS call in it,
        # including uploads and direct huggingface_hub downloads that our ladder does not catch.
        # Process-wide they turn a transient CAS 5xx into a hard error after ~12s of backoff instead
        # of the ~363s xet-core would have spent retrying.
        for key in _FAIL_FAST_KEYS:
            env.pop(key, None)
    return env


def xet_log_env(log_dir: "str | Path", *, diagnostics: bool = False) -> dict[str, str]:
    """Point hf_xet's own logger at *log_dir* so failures can be read back (see ``scan_xet_log``).

    The trailing separator is what makes hf_xet treat the value as a DIRECTORY, not a file.
    *diagnostics* also enables hf_xet's built-in CPU/RAM sampler.
    """
    log_dir = Path(log_dir)
    env = {
        "HF_XET_LOG_DEST": os.path.join(str(log_dir), ""),
        "HF_XET_LOG_FORMAT": "json",
        "HF_XET_LOG_PREFIX": "unsloth-xet",
    }
    if diagnostics or _is_true(os.environ.get("UNSLOTH_XET_DIAGNOSTICS")):
        env["HF_XET_SYSTEM_MONITOR_ENABLED"] = "1"
        env["HF_XET_SYSTEM_MONITOR_SAMPLE_INTERVAL"] = "5s"
        env["HF_XET_SYSTEM_MONITOR_LOG_PATH"] = str(log_dir / "sysmon_{PID}.log")
    return env


def apply_xet_env(
    env: "Optional[dict]" = None,
    *,
    profile: Optional[SystemProfile] = None,
    throttled: bool = False,
    force: bool = False,
    fail_fast: bool = False,
) -> dict[str, str]:
    """Apply the overrides to *env* (default: ``os.environ``) and return only what was written.

    ``setdefault`` semantics: a user-set variable is left alone, so a shell's
    ``HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT=16gb`` still wins. High-performance mode is the
    exception -- being applied AFTER the environment is read, an enabled ``HF_XET_HIGH_PERFORMANCE``
    discards every cap above rather than competing with it, so it is turned off even when already
    set; ``UNSLOTH_XET_ALLOW_HIGH_PERFORMANCE=1`` keeps it (and drops the caps it would have voided).

    *force* overwrites every variable, for callers building a fresh child environment. *fail_fast*
    defaults to False here, unlike ``xet_env_overrides``, because this runs at import: the shortened
    timeouts would otherwise apply to every direct ``huggingface_hub`` download and upload in the
    process, none of which our ladder supervises. Supervised children pass ``fail_fast = True``.
    """
    target = os.environ if env is None else env
    overrides = xet_env_overrides(profile, throttled = throttled, fail_fast = fail_fast)

    if _is_true(os.environ.get("UNSLOTH_XET_ALLOW_HIGH_PERFORMANCE")):
        for var in XET_HIGH_PERFORMANCE_VARS:
            overrides.pop(var, None)

    written: dict[str, str] = {}
    for key, value in overrides.items():
        if force or key not in target or (key in XET_HIGH_PERFORMANCE_VARS and _is_true(target.get(key))):
            target[key] = value
            written[key] = value
    return written


_LOG_ERROR_RE = re.compile(r'"level"\s*:\s*"(ERROR|WARN)"', re.IGNORECASE)


def scan_xet_log(log_dir: "Optional[str | Path]", *, max_messages: int = 5) -> list[str]:
    """Return up to *max_messages* ERROR/WARN lines from hf_xet's json logs in *log_dir*.

    Turns a quiet Xet failure into an explicit fallback reason. Best-effort: a missing or unreadable
    directory yields an empty list, never an exception.
    """
    if not log_dir:
        return []
    try:
        directory = Path(log_dir)
        if not directory.is_dir():
            return []
        files = sorted(
            (p for p in directory.glob("*.log") if p.is_file()),
            key = lambda p: p.stat().st_mtime,
            reverse = True,
        )
    except OSError:
        return []

    messages: list[str] = []
    for path in files[:4]:  # newest few; a log dir can accumulate across sessions
        try:
            with open(path, "r", errors = "replace") as f:
                for line in f:
                    if _LOG_ERROR_RE.search(line):
                        messages.append(line.strip()[:400])
                        if len(messages) >= max_messages:
                            return messages
        except OSError:
            continue
    return messages
