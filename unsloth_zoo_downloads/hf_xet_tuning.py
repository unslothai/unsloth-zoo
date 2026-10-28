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

"""Sizes hf_xet buffers from cgroup-aware RAM and cores as HF_XET_* env vars with setdefault semantics;
HF_XET_HIGH_PERFORMANCE raises the cap to 64GB and discards explicit caps; bounding memory needs it off; durations
need a unit suffix"""

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
    "resize_for_cache_dir",
    "scan_xet_log",
    "XET_HIGH_PERFORMANCE_VARS",
]

_GB = 1_000_000_000  # hf_xet's ByteSize renders/parses SI units, so stay in SI.
_MB = 1_000_000

# Both spellings enable high-performance mode in xet-core; both must be cleared for a cap to hold.
XET_HIGH_PERFORMANCE_VARS = ("HF_XET_HIGH_PERFORMANCE", "HF_XET_HP")

_TRUTHY = {"1", "true", "yes", "on"}

# The high-performance preset is just knobs (xet_config.rs with_high_performance), so we write them scaled to RAM: an eighth of RAM, clamped to 1 to 64 GB.
_RAM_FRACTION = 8
_MIN_BUFFER_LIMIT = 1 * _GB
_MAX_BUFFER_LIMIT = 64 * _GB
# Only the buffer group moves throughput; streams and files exist to keep it fed, so they follow cores and never exceed what the budget can hold.
_MAX_STREAMS = 124
_MAX_CONCURRENT_FILES = 24
# xet-core's xorb size: the unit a single download stream has outstanding at any moment.
_XORB_BYTES = 64 * 1024 * 1024
# xet-core's own defaults (config/groups/reconstruction.rs): sizing down from a default is a cap, and sizing below one
# is just a slower download, which these two floors prevent.
_STOCK_PREFETCH_BUFFER = 1 * _GB
_STOCK_BUFFER_SIZE = 2 * _GB

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
    """Candidate cgroup v2 dirs for this process, innermost first; /sys/fs/cgroup/memory.max usually does not exist, so
    the real limit lives at the /proc/self/cgroup path."""
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
    """Candidate cgroup v1 dirs for controller, innermost first, then the mount root; outside a container the root reads
    the unlimited sentinel and hides Slurm or systemd limits."""
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
    # Free space where the download lands: a multi-GB buffer for a transfer the disk cannot hold just wastes RAM, and a
    # nearly full disk predicts a doomed download.
    free_disk_bytes: int = 0
    disk_source: str = "unknown"


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


def hf_cache_root(cache_dir: "Optional[str | Path]" = None) -> Path:
    """Root in huggingface_hub's precedence (HF_HUB_CACHE, HUGGINGFACE_HUB_CACHE, HF_HOME/hub, XDG_CACHE_HOME), so disk
    checks hit the right volume; cache_dir wins since hf_hub_download reads env only when it is None."""
    if cache_dir is not None:
        try:
            return Path(os.path.expanduser(os.fspath(cache_dir)))
        except (TypeError, ValueError):
            pass
    try:
        from .hf_cache import _active_caches

        hub_cache = _active_caches()[1]
        if hub_cache is not None:
            return hub_cache
        return Path.home() / ".cache" / "huggingface" / "hub"
    except Exception:  # noqa: BLE001 - a cache we cannot resolve must not stop a download
        # Home may be unresolvable on a locked-down machine; the caller walks up from here anyway.
        return Path(".")


def _free_disk(cache_dir: "Optional[str | Path]" = None) -> "tuple[int, str]":
    """Free bytes on the filesystem holding the HF cache, walking up to the first path that exists."""
    import shutil

    candidate = hf_cache_root(cache_dir)
    for path in (candidate, *candidate.parents):
        try:
            return int(shutil.disk_usage(path).free), str(path)
        except OSError:
            continue
    return 0, "unknown"


def system_profile(cache_dir: "Optional[str | Path]" = None) -> SystemProfile:
    """Usable RAM and cores for this process, preferring a cgroup limit over host totals; cache_dir is the real download
    destination, so the disk reading follows the bytes."""
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

    free_disk, disk_source = _free_disk(cache_dir)

    try:
        cpus = len(os.sched_getaffinity(0))  # respects taskset / CPU pinning
        cpu_source = "affinity"
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
        cpu_source = "cpu_count"
    cg_cpu = cgroup_cpu_limit()
    if cg_cpu is not None and cg_cpu > 0 and cg_cpu < cpus:
        # A fractional quota still binds (Kubernetes cpu: 500m is cpu.max 50000 100000 = 0.5), so it must not fall back to the host's core count.
        cpus = max(1, int(cg_cpu))
        cpu_source = "cgroup"

    return SystemProfile(
        total_ram_bytes = int(total),
        available_ram_bytes = int(available),
        cpu_count = max(1, int(cpus)),
        ram_source = ram_source,
        cpu_source = cpu_source,
        free_disk_bytes = free_disk,
        disk_source = disk_source,
    )


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


# Shortened Xet timeouts are safe only in a supervised child; the memory-bound caps are voided by high-performance mode,
# so applying them anyway shrinks the transfer while xet-core ignores the limit, slower than either clean choice.
_CAPS_VOIDED_BY_HIGH_PERFORMANCE = (
    "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT",
    "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE",
    "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE",
    "HF_XET_RECONSTRUCTION_MIN_PREFETCH_BUFFER",
    "HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS",
)

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
    cache_dir: "Optional[str | Path]" = None,
) -> dict[str, str]:
    """Pure: HF_XET_* settings from RAM, cores and disk; throttled halves the stream ceiling after a logged 429;
    fail_fast keeps shortened Xet timeouts, only for ladder-supervised callers; unknown RAM reads small (guessing low
    costs throughput, high costs an OOM); cache_dir ignored when profile given."""
    profile = profile or system_profile(cache_dir)
    # Unknown RAM reads as small: guessing low costs throughput, guessing high costs an OOM.
    total = profile.total_ram_bytes or 8 * _GB
    limit = _clamp(total // _RAM_FRACTION, _MIN_BUFFER_LIMIT, _MAX_BUFFER_LIMIT)

    # Clamp the buffer to a quarter of free disk; test disk_source rather than truthiness, since zero free bytes is a full disk, not an unmeasured one.
    free = profile.free_disk_bytes
    if profile.disk_source != "unknown":
        limit = _clamp(min(limit, free // 4), _MIN_BUFFER_LIMIT, _MAX_BUFFER_LIMIT)

    # The shared buffer is the knob that moves throughput, so it is a floor at xet-core's 2 GB default, held back by
    # half the budget (a bigger share leaves no room for per-file buffers) and a sixth of RAM (so the floor cannot
    # override the memory bound on a 2 GB VM), not a ratio.
    size = min(
        max(limit // 4, min(_STOCK_BUFFER_SIZE, limit // 2)),
        max(256 * _MB, (profile.total_ram_bytes or 8 * _GB) // 6),
    )
    perfile = max(limit // 32, 128 * _MB)
    # Prefetch approaches xet-core's 1 GB default from below and never exceeds it; only a budget under 4 GB gets less,
    # since going below the default just slows the download.
    prefetch = _clamp(limit // 4, 128 * _MB, _STOCK_PREFETCH_BUFFER)

    cpus = profile.cpu_count
    # Max files is bounded by the budget as well as cores, since size + max_files * perfile is what hf_xet can hold; the
    # absolute budget floor can exceed the proportional bound on small machines, so the smaller of limit and a third of
    # RAM is used.
    affordable = max(2, (min(limit, total // 3) - size) // perfile)
    max_files = _clamp(min(cpus, affordable), 2, _MAX_CONCURRENT_FILES)
    # A stream holds a 64 MiB xorb, so streams past the budget's xorb slots only queue; a 64-core, 8 GB container opened
    # 124 streams on cores alone, so both bounds apply.
    streams = _clamp(min(cpus * 2, max(4, limit // _XORB_BYTES)), 4, _MAX_STREAMS)
    if throttled:
        streams = max(4, streams // 2)
    # Start well under the ceiling and let xet-core's adaptive concurrency ramp, which keeps a slow or congested link from opening every stream into a stall.
    initial = min(_clamp(cpus, 2, 8), streams)

    env = {
        # Memory. The effective buffer is size + max_files * perfile, capped by limit.
        "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT": str(limit),
        "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE": str(size),
        "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE": str(perfile),
        "HF_XET_RECONSTRUCTION_MIN_PREFETCH_BUFFER": str(prefetch),
        "HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS": str(max_files),
        # ac_* is the adaptive-concurrency band; the initial value stays under the ceiling so a slow link ramps up instead of opening 16 streams into a stall.
        "HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY": str(streams),
        "HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY": str(initial),
        "HF_XET_CLIENT_AC_MIN_DOWNLOAD_CONCURRENCY": "1",
        # Fail fast so our ladder decides instead of hf_xet retrying for ~6 minutes; bare integers are ignored, so the unit suffix is required.
        "HF_XET_CLIENT_READ_TIMEOUT": "60s",
        "HF_XET_CLIENT_CONNECT_TIMEOUT": "20s",
        "HF_XET_CLIENT_RETRY_MAX_ATTEMPTS": "2",
        "HF_XET_CLIENT_RETRY_MAX_DURATION": "30s",
        # The chunk cache only pays off when re-fetching known chunks; upstream default is already 0, pinned so a stray value cannot raise it.
        "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0",
        # Applied AFTER the env in xet-core, so leaving this on would discard every cap above.
        "HF_XET_HIGH_PERFORMANCE": "0",
        "HF_XET_HP": "0",
    }

    if not fail_fast:
        # xet-core reads config once per process, so these fail-fast keys apply to every CAS call, including uploads and
        # direct huggingface_hub downloads our ladder does not catch; process-wide a transient CAS 5xx becomes a hard
        # error after ~12s instead of ~363s of retries.
        for key in _FAIL_FAST_KEYS:
            env.pop(key, None)
    return env


def xet_log_env(log_dir: "str | Path", *, diagnostics: bool = False) -> dict[str, str]:
    """Points hf_xet's logger at log_dir (read back via scan_xet_log); the trailing separator makes hf_xet treat it as a
    directory; diagnostics also enables its CPU/RAM sampler."""
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


_SEEDED_INTO_ENVIRON: dict[str, str] = {}
_SEEDED_THROTTLED = False


def apply_xet_env(
    env: "Optional[dict]" = None,
    *,
    profile: Optional[SystemProfile] = None,
    throttled: bool = False,
    force: bool = False,
    fail_fast: bool = False,
    cache_dir: "Optional[str | Path]" = None,
) -> dict[str, str]:
    """Apply the overrides to *env* (default: ``os.environ``) and return only what was written.

    ``setdefault`` semantics: a user-set variable is left alone, so a shell's
    ``HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT=16gb`` still wins -- and that now includes
    ``HF_XET_HIGH_PERFORMANCE``. Enabling it is a deliberate act by someone who knows their machine,
    and since xet-core applies it AFTER reading the environment it voids the caps rather than
    competing with them, so the honest response is to stand down: we drop the caps it would have
    discarded and leave the flag alone. Measured on a 192-core / 1996 GiB box, overriding it cost
    2.55x download throughput (16684 -> 6553 Mbit/s), to defend RAM that machine was never short of.
    Set ``UNSLOTH_XET_FORCE_CAPS=1`` to get the old behaviour and cap a machine regardless.

    *force* overwrites every variable we would otherwise leave alone. It does not revoke a user-set
    high-performance flag: that stand-down is our own sizing standing aside, not a default to force
    through. *fail_fast*
    defaults to False here, unlike ``xet_env_overrides``, because this runs at import: the shortened
    timeouts would otherwise apply to every direct ``huggingface_hub`` download and upload in the
    process, none of which our ladder supervises. Supervised children pass ``fail_fast = True``.

    *cache_dir* is the destination the sized process will pass huggingface_hub; naming it keeps the
    disk clamp off an unrelated filesystem in both directions.
    """
    target = os.environ if env is None else env
    overrides = xet_env_overrides(
        profile, throttled = throttled, fail_fast = fail_fast, cache_dir = cache_dir,
    )

    # A user's high-performance flag keeps the headroom, so our caps are dropped rather than left alongside it; read it
    # as xet-core does (configuration_utils.rs get_high_performance_flag): HF_XET_HIGH_PERFORMANCE wins if set at all,
    # HF_XET_HP only otherwise.
    primary, alias = XET_HIGH_PERFORMANCE_VARS
    user_wants_hp = _is_true(target[primary]) if primary in target else _is_true(target.get(alias))
    forcing_caps = _is_true(os.environ.get("UNSLOTH_XET_FORCE_CAPS"))
    # Either the user's flag wins and our sizing stands down, or the caps win and the flag must be turned off, since
    # xet-core reads it last and would void them; overwritable lists the values we may replace when already set.
    overwritable: tuple = ()
    if user_wants_hp and not forcing_caps:
        for var in (*XET_HIGH_PERFORMANCE_VARS, *_CAPS_VOIDED_BY_HIGH_PERFORMANCE):
            overrides.pop(var, None)
    elif forcing_caps:
        overwritable = XET_HIGH_PERFORMANCE_VARS

    written: dict[str, str] = {}
    for key, value in overrides.items():
        if force or key not in target or key in overwritable:
            target[key] = value
            written[key] = value
    # Record what we wrote into the real os.environ, and on what terms, so a later resize redoes only our own values,
    # the same way, and leaves user settings alone.
    if target is os.environ:
        global _SEEDED_THROTTLED
        _SEEDED_INTO_ENVIRON.update(written)
        _SEEDED_THROTTLED = throttled
    return written


def resize_for_cache_dir(
    env: dict,
    cache_dir: "Optional[str | Path]",
    *,
    fail_fast: bool = True,
    throttled: "Optional[bool]" = None,
) -> dict[str, str]:
    """Recomputes only the sizing values this process seeded into the environment, since setdefault would keep them for
    the wrong cache; values we never wrote, or that changed since, are left alone; throttled defaults to the seeding
    call's."""
    if throttled is None:
        throttled = _SEEDED_THROTTLED
    for key, value in _SEEDED_INTO_ENVIRON.items():
        if env.get(key) == value:
            env.pop(key, None)
    return apply_xet_env(env, fail_fast = fail_fast, cache_dir = cache_dir, throttled = throttled)


_LOG_ERROR_RE = re.compile(r'"level"\s*:\s*"(ERROR|WARN)"', re.IGNORECASE)


def scan_xet_log(log_dir: "Optional[str | Path]", *, max_messages: int = 5) -> list[str]:
    """Up to max_messages ERROR/WARN lines from hf_xet's json logs in log_dir, to explain a quiet Xet failure;
    best-effort: a missing or unreadable directory yields an empty list, never an exception."""
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
