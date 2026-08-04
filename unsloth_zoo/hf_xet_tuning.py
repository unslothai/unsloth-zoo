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
    "resize_for_cache_dir",
    "scan_xet_log",
    "XET_HIGH_PERFORMANCE_VARS",
]

_GB = 1_000_000_000  # hf_xet's ByteSize renders/parses SI units, so stay in SI.
_MB = 1_000_000

# Both spellings enable high-performance mode in xet-core; both must be cleared for a cap to hold.
XET_HIGH_PERFORMANCE_VARS = ("HF_XET_HIGH_PERFORMANCE", "HF_XET_HP")

_TRUTHY = {"1", "true", "yes", "on"}

# The high-performance preset is not a mode, just knobs (xet_config.rs with_high_performance:
# 64 GB limit, 16 GB buffer, 2 GB per file, 124 streams), so we write the same ones scaled to the
# machine. An eighth of RAM reproduces the table this replaces at every point it defined
# (8 GB -> 1 GB, 16 -> 2, 32 -> 4, 128 -> 16, 256 -> 32) and keeps going, to xet-core's own 64 GB.
_RAM_FRACTION = 8
_MIN_BUFFER_LIMIT = 1 * _GB
_MAX_BUFFER_LIMIT = 64 * _GB
# Measured on a 192-core / 20 Gbit/s host: the buffer group alone is worth 2.70x, the concurrency
# group alone nothing (0.92x). Streams and files exist to keep the buffer fed, so they follow cores
# and never exceed what the budget can hold.
_MAX_STREAMS = 124
_MAX_CONCURRENT_FILES = 24
# xet-core's xorb size: the unit a single download stream has outstanding at any moment.
_XORB_BYTES = 64 * 1024 * 1024
# xet-core's own defaults (config/groups/reconstruction.rs). Sizing DOWN from a default is a cap;
# sizing below one for no reason is just a slower download, which is what these two floors prevent.
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
    # Free space where the download will land. Sizing a multi-GB buffer for a transfer the disk
    # cannot hold just wastes RAM, and a nearly full disk is a better predictor of a doomed
    # download than anything else we can see cheaply.
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
    """Where downloads land, in the same precedence huggingface_hub itself uses.

    ``hf_cache._active_caches`` already mirrors that precedence -- ``HF_HUB_CACHE``, the legacy
    ``HUGGINGFACE_HUB_CACHE``, then ``HF_HOME/hub``, then ``XDG_CACHE_HOME/huggingface/hub`` -- and
    resolving anything less measures a filesystem the download may never touch: with
    ``XDG_CACHE_HOME`` or ``HUGGINGFACE_HUB_CACHE`` pointed at a data volume, the free space of the
    home directory decides the buffer instead.

    *cache_dir* wins over all of them: ``hf_hub_download`` only consults the variables when it is
    None, so a caller that names one has named the volume the bytes land on.
    """
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
    """Usable RAM and cores for THIS process, preferring a cgroup limit over the host's totals.

    *cache_dir* is the download's real destination, so the disk reading follows the bytes."""
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
        free_disk_bytes = free_disk,
        disk_source = disk_source,
    )


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


# Shortened Xet client timeouts: safe in a download child we supervise, wrong process-wide.
# Sizing knobs that exist only to bound memory. When the user has asked for high-performance mode
# they have opted out of that bound, and applying these anyway would shrink the transfer while
# xet-core ignores the limit -- slower than either choice made cleanly.
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
    """RAM/CPU/disk-derived ``HF_XET_*`` settings. Pure: returns a dict, touches no environment.

    The budget scales continuously with the machine rather than stepping through a table, so a
    laptop keeps a laptop's buffers and a large host reaches xet-core's own high-performance sizing
    without the flag that would discard every bound.

    *throttled* halves the stream ceiling; set after a logged "429 Too Many Requests", where the
    account rather than the machine is the limiting factor. *fail_fast* keeps the shortened Xet
    timeouts, which only suit callers whose failure our Xet -> HTTP ladder can act on. Unknown total
    RAM (0) reads as small: guessing low costs throughput, guessing high costs an OOM.

    *cache_dir* points the disk clamp at the volume the bytes land on. Ignored when *profile* is
    supplied, which already carries its own disk reading.
    """
    profile = profile or system_profile(cache_dir)
    # Unknown RAM reads as small: guessing low costs throughput, guessing high costs an OOM.
    total = profile.total_ram_bytes or 8 * _GB
    limit = _clamp(total // _RAM_FRACTION, _MIN_BUFFER_LIMIT, _MAX_BUFFER_LIMIT)

    # Never size a buffer for a transfer the disk cannot land. A quarter of free space is generous
    # for a buffer and still refuses to promise 64 GB of in-flight data to a disk with 20 GB left.
    # disk_source, not truthiness: a successful reading of zero free bytes is a full disk, which is
    # exactly when the clamp should bite, and must not read as "we could not measure".
    free = profile.free_disk_bytes
    if profile.disk_source != "unknown":
        limit = _clamp(min(limit, free // 4), _MIN_BUFFER_LIMIT, _MAX_BUFFER_LIMIT)

    # The shared buffer is the one knob that moves throughput (2.70x on a 2 TB host; 1.45x from
    # this value alone on an 8 GB laptop, where a quarter of the budget ran at 0.73x). So it is a
    # floor, not a ratio: reach for xet-core's 2 GB default, held back only by half the budget (a
    # bigger share leaves no room for per-file buffers) and a sixth of RAM (which keeps the floor
    # from overriding the memory bound on a 2 GB VM).
    size = min(
        max(limit // 4, min(_STOCK_BUFFER_SIZE, limit // 2)),
        max(256 * _MB, (profile.total_ram_bytes or 8 * _GB) // 6),
    )
    perfile = max(limit // 32, 128 * _MB)
    # The prefetch floor is the one knob where our old arithmetic went BELOW xet-core's own default
    # (1 GB) on every machine, which is not a cap, just a smaller transfer. Approach the default
    # from below and never exceed it: only a machine whose entire budget is under 4 GB gets less.
    prefetch = _clamp(limit // 4, 128 * _MB, _STOCK_PREFETCH_BUFFER)

    cpus = profile.cpu_count
    # size + max_files * perfile is what hf_xet can hold, so the count comes from the budget as
    # well as from cores: three numbers describing one allocation, not three that overshoot.
    # The budget floor is absolute, so on a small machine it can exceed the proportional bound: a
    # 2 GB container gets the 1 GB floor, and cores alone would then put 973 MB in flight, half its
    # RAM. Bound the count by whichever is smaller so the third-of-RAM promise holds there too.
    affordable = max(2, (min(limit, total // 3) - size) // perfile)
    max_files = _clamp(min(cpus, affordable), 2, _MAX_CONCURRENT_FILES)
    # A stream holds a 64 MiB xorb, so streams past the budget's xorb slots only queue behind the
    # semaphore. Both bounds matter: a 64-core container with 8 GB opened 124 under cores alone.
    streams = _clamp(min(cpus * 2, max(4, limit // _XORB_BYTES)), 4, _MAX_STREAMS)
    if throttled:
        streams = max(4, streams // 2)
    # Start well under the ceiling and let xet-core's adaptive concurrency ramp: on a slow or
    # congested link the ramp is what keeps us from opening every stream into a stall.
    initial = min(_clamp(cpus, 2, 8), streams)

    env = {
        # Memory. The effective buffer is size + max_files * perfile, capped by limit.
        "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT": str(limit),
        "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE": str(size),
        "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE": str(perfile),
        "HF_XET_RECONSTRUCTION_MIN_PREFETCH_BUFFER": str(prefetch),
        "HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS": str(max_files),
        # ac_* is the adaptive-concurrency band; the initial value stays under the ceiling so a slow
        # link ramps up instead of opening 16 streams into a stall.
        "HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY": str(streams),
        "HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY": str(initial),
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

    # A user who turned high-performance mode on keeps it, and keeps the headroom it implies:
    # leaving our caps in place alongside it would be the worst of both, since xet-core would void
    # the limit but still honour the smaller per-file and concurrency numbers.
    # Read the flag the way xet-core reads it (configuration_utils.rs get_high_performance_flag):
    # HF_XET_HIGH_PERFORMANCE wins if it is SET AT ALL, and only then does the HF_XET_HP alias get a
    # look. Taking any-of instead would stand our sizing down over an alias xet-core is ignoring.
    primary, alias = XET_HIGH_PERFORMANCE_VARS
    user_wants_hp = _is_true(target[primary]) if primary in target else _is_true(target.get(alias))
    forcing_caps = _is_true(os.environ.get("UNSLOTH_XET_FORCE_CAPS"))
    # Only one of these may hold: either the user's flag wins and our sizing stands down, or the
    # caps win and the flag has to be turned off, because xet-core reads it last and would void
    # them. Overwritable is the set we are allowed to write over an existing value.
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
    # Remember what WE put in the real environment, and on what terms, so a later resize can tell
    # our own numbers from a user's, redo only ours, and redo them the same way.
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
    """Re-size *env* for *cache_dir*, recomputing the values this process seeded and no others.

    ``apply_xet_env`` is setdefault, and ``unsloth_zoo`` sizes itself at import, so by the time a
    download names its destination every sizing key is already present and a second call writes
    nothing: the download would run on numbers computed for whichever cache the environment named,
    which is exactly what *cache_dir* is there to correct. Dropping our own seeded values first
    makes the recompute bite. A value we never wrote, or one something else has changed since, is
    left alone, so an explicit user setting still wins.

    *throttled* defaults to whatever the seeding call used, so a halved stream ceiling asked for by
    a logged 429 survives the recompute; the whole point of that reduction is that it reaches the
    process doing the downloading.
    """
    if throttled is None:
        throttled = _SEEDED_THROTTLED
    for key, value in _SEEDED_INTO_ENVIRON.items():
        if env.get(key) == value:
            env.pop(key, None)
    return apply_xet_env(env, fail_fast = fail_fast, cache_dir = cache_dir, throttled = throttled)


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
