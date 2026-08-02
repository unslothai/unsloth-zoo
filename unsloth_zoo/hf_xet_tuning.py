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

hf_xet allocates its reconstruction buffers from constants, not from available RAM. Stock defaults
are a 2GB floor plus 512MB per concurrent file (8 of them) capped at 8GB, plus a 1GB prefetch floor
-- so a download can hold ~8GB on a laptop that does not have it. ``HF_XET_HIGH_PERFORMANCE`` (which
Unsloth used to default ON) raises that cap to 64GB and the stream count to 124, and it is applied
AFTER the environment is read, so it silently DISCARDS any explicit ``HF_XET_RECONSTRUCTION_*`` cap.
Bounding memory therefore requires turning high-performance mode off, not just setting the caps.

Values are chosen from total RAM (cgroup-aware: inside a container ``psutil`` reports the HOST's RAM,
which is how a 16GB CI runner ends up with an 8GB buffer) and core count. Everything is emitted as
environment variables because ``hf_xet`` reads its config once, natively, before Python can reach it
-- and applied with ``setdefault`` semantics so an explicit user setting always wins.

Durations MUST carry a unit suffix: hf_xet silently IGNORES a bare integer (``"60"`` leaves the
300s default in place) and only honours ``"60s"``.
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

# HF boolean env convention, matching hf_xet_fallback._is_true.
_TRUTHY = {"1", "true", "yes", "on"}

# (exclusive upper bound on total RAM, buffer_limit, buffer_size, perfile_size, max_concurrent_files)
_TIERS = (
    (12 * _GB, 1 * _GB, 512 * _MB, 128 * _MB, 4),
    (24 * _GB, 2 * _GB, 768 * _MB, 192 * _MB, 6),
    (None,     4 * _GB, 1 * _GB,   256 * _MB, 8),
)

# Below this much usable RAM, Xet's smallest sane working set is still a poor trade: callers treat
# this as "prefer HTTP" (see hf_xet_health).
MIN_XET_RAM_BYTES = 4 * _GB


def _is_true(value: Optional[str]) -> bool:
    return value is not None and str(value).strip().lower() in _TRUTHY


def _read_first_line(path: Path) -> Optional[str]:
    try:
        with open(path, "r") as f:
            return f.readline().strip()
    except OSError:
        return None


def _cgroup_v2_dirs() -> list[Path]:
    """Candidate cgroup v2 dirs for THIS process, innermost first.

    The root ``/sys/fs/cgroup/memory.max`` usually does not exist (the root cgroup has no limit
    file), so the real limit lives at the path named in ``/proc/self/cgroup``. Every ancestor is
    also checked because a limit set on a parent slice still binds us.
    """
    root = Path("/sys/fs/cgroup")
    if not root.is_dir():
        return []
    rel = None
    content = _read_first_line(Path("/proc/self/cgroup"))
    # cgroup v2 line is "0::<path>"; v1 lines carry a controller list instead.
    if content is not None and content.startswith("0::"):
        rel = content[3:].strip()
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


def _parse_limit(raw: Optional[str]) -> Optional[int]:
    """``"max"`` (or an unparseable value) means unlimited."""
    if raw is None or raw == "max":
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    # cgroup v1 spells "unlimited" as a near-2^63 sentinel rather than a word.
    return value if 0 < value < (1 << 62) else None


def cgroup_memory_limit() -> Optional[int]:
    """Smallest binding memory ceiling from cgroup v2 (``memory.max``/``memory.high``) or v1."""
    limits: list[int] = []
    for d in _cgroup_v2_dirs():
        for name in ("memory.max", "memory.high"):
            value = _parse_limit(_read_first_line(d / name))
            if value is not None:
                limits.append(value)
    v1 = _parse_limit(_read_first_line(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")))
    if v1 is not None:
        limits.append(v1)
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
    quota = _parse_limit(_read_first_line(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")))
    period = _parse_limit(_read_first_line(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")))
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
    """Fallback when psutil is absent (it is an optional dep of the lightweight download child)."""
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
        # A container's limit binds regardless of what the host reports.
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
    if cg_cpu is not None and cg_cpu >= 1 and int(cg_cpu) < cpus:
        cpus = int(cg_cpu)
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


def xet_env_overrides(
    profile: Optional[SystemProfile] = None,
    *,
    throttled: bool = False,
) -> dict[str, str]:
    """RAM/CPU-derived ``HF_XET_*`` settings. Pure: returns a dict, touches no environment.

    *throttled* halves the download stream ceiling; Unsloth sets it when a previous session logged
    "429 Too Many Requests", which means the account, not the machine, is the limiting factor.

    An unknown total RAM (0) yields the smallest tier: guessing low costs a little throughput,
    guessing high costs an OOM.
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

    # hf_xet grows the buffer to size + max_files * perfile and clamps it at limit. Keeping the
    # limit at or above that sum makes the three numbers describe ONE budget: the limit then states
    # the true ceiling instead of silently truncating the other two.
    limit = max(limit, size + max_files * perfile)

    return {
        # Memory. The effective buffer is size + max_files * perfile, capped by limit.
        "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT": str(limit),
        "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE": str(size),
        "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE": str(perfile),
        "HF_XET_RECONSTRUCTION_MIN_PREFETCH_BUFFER": str(min(size, 512 * _MB)),
        "HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS": str(max_files),
        # CPU / sockets. ac_* is the adaptive-concurrency band; the initial value stays under the
        # ceiling so a slow link ramps up instead of opening 16 streams into a stall.
        "HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY": str(streams),
        "HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY": str(_clamp(cpus, 2, 8)),
        "HF_XET_CLIENT_AC_MIN_DOWNLOAD_CONCURRENCY": "1",
        # Fail fast so OUR Xet -> HTTP ladder decides, instead of hf_xet retrying for ~6 minutes.
        # Bare integers are ignored here; the unit suffix is required.
        "HF_XET_CLIENT_READ_TIMEOUT": "60s",
        "HF_XET_CLIENT_CONNECT_TIMEOUT": "20s",
        "HF_XET_CLIENT_RETRY_MAX_ATTEMPTS": "2",
        "HF_XET_CLIENT_RETRY_MAX_DURATION": "30s",
        # The chunk cache only pays off when re-fetching known chunks; on a plain download it is
        # extra disk and RAM. Upstream default is already 0; pinned so a stray value cannot raise it.
        "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0",
        # Applied AFTER the env in xet-core, so leaving this on would discard every cap above.
        "HF_XET_HIGH_PERFORMANCE": "0",
        "HF_XET_HP": "0",
    }


def xet_log_env(log_dir: "str | Path", *, diagnostics: bool = False) -> dict[str, str]:
    """Point hf_xet's own logger at *log_dir* so failures can be read back (see ``scan_xet_log``).

    A trailing separator is what makes hf_xet treat the value as a DIRECTORY rather than a file.
    *diagnostics* additionally enables hf_xet's built-in CPU/RAM sampler.
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
) -> dict[str, str]:
    """Apply the overrides to *env* (default: ``os.environ``) and return only what was written.

    ``setdefault`` semantics: a variable the user already set is left alone, so
    ``HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT=16gb`` in a shell still wins.

    High-performance mode is the one exception. It is a preset applied AFTER the environment is
    read, so an enabled ``HF_XET_HIGH_PERFORMANCE`` silently discards every cap above rather than
    merely competing with it -- capping memory and leaving it on are contradictory requests. It is
    therefore turned off even when already set, and the only way to keep it is
    ``UNSLOTH_XET_ALLOW_HIGH_PERFORMANCE=1`` (which drops the caps it would have voided).

    *force* overwrites every variable regardless, for callers building a fresh child environment.
    """
    target = os.environ if env is None else env
    overrides = xet_env_overrides(profile, throttled = throttled)

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

    Turns "Xet quietly failed to fetch some terms" into an explicit fallback reason. Best-effort:
    a missing or unreadable directory yields an empty list, never an exception.
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
