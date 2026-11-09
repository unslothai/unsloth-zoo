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

"""Single source of truth for the ``dataset_num_proc`` used by ``Dataset.map()``.

Four drifted copies of this heuristic exist (three in ``dataset_utils``, one in
the trainer source ``unsloth/models/rl.py`` generates) and all share the two
defects behind https://github.com/unslothai/unsloth/issues/2693:

1. They ask ``multiprocessing.get_start_method()`` whether forking is available,
   but ``Dataset.map`` pools come from ``multiprocess`` (the dill fork), which
   keeps its own independent default context.

2. They use ``1`` as the "disable multiprocessing" sentinel, yet ``datasets``
   >= 4.1 pools for any ``num_proc >= 1``; only ``None`` is in-process on every
   supported version.

It lives here rather than in ``unsloth`` so the generated trainer source can
import it without importing back into ``unsloth`` itself. ``dataset_utils`` now
sizes its auto paths through it as well.
"""

from __future__ import annotations

import contextlib
import os
import sys
from typing import Iterator, Optional

__all__ = [
    "AUTO_NUM_PROC_CAP",
    "MEMORY_BUDGET_FRACTION",
    "NUM_PROC_ENV_VAR",
    "WORKER_MEMORY_BUDGET_GB",
    "ZOO_MIN_ROWS_FOR_MULTIPROC",
    "environment_override",
    "get_dataset_num_proc",
    "map_failure_diagnostics",
    "multiprocessing_start_method",
    "reset_warning_state",
    "resolve_responses_only_num_proc",
]

# Escape hatch: a positive integer forces that count verbatim (no cap, no start-method veto); 0 or none forces in-process tokenization.
NUM_PROC_ENV_VAR = "UNSLOTH_DATASET_NUM_PROC"

# Upper bound for the auto count only, raised via NUM_PROC_ENV_VAR; the old heuristic forked up to 64 workers, each
# handed a dill-pickled tokenizer closure and an Arrow shard, and more workers were slower than none at every size.
AUTO_NUM_PROC_CAP = 8

# ~680 MB peak RSS per worker on that run, flat across counts; 1 GB for headroom.
WORKER_MEMORY_BUDGET_GB = 1.0

# Share of available RAM tokenization may spend, the rest left for the run; bounds issue 2693, where OOM-killed workers
# surface only as 'One of the subprocesses has abruptly died during map operation', hiding the real cause.
MEMORY_BUDGET_FRACTION = 0.5

# Mirrors _MIN_ROWS_FOR_MULTIPROC, a local inside
# dataset_utils.train_on_responses_only (hence not importable): below it that
# helper maps a split in-process, unless handed an explicit count.
# resolve_responses_only_num_proc needs the threshold to keep that guard; a
# canary in tests/test_dataset_num_proc.py catches drift.
ZOO_MIN_ROWS_FOR_MULTIPROC = 5_000

# Warn at most once per process per distinct reason.
_WARNED: set = set()


def reset_warning_state() -> None:
    """Clear the once-per-process warning memo. Intended for tests."""
    _WARNED.clear()


def _warn_once(key: str, message: str) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    print(f"Unsloth: {message}")


def _unpinned_default_start_method(module) -> Optional[str]:
    """Read the name of the default context itself, since multiprocess lists spawn first yet its Pool forks; unlike
    get_context() it does not assign _actual_context; private attributes, hence a fallback."""
    try:
        methods = module.get_all_start_methods()
    except Exception:
        methods = []

    try:
        name = module.context._default_context._default_context._name
        # Only if the platform offers it: a Windows runner answered 'fork' while get_all_start_methods() was ['spawn'],
        # reading Windows as forkable and looping on spawn re-import (issues 3211, 3397).
        if isinstance(name, str) and name and (not methods or name in methods):
            return name
    except Exception:
        pass
    return methods[0] if methods else None


def _module_start_method(module_name: str) -> Optional[str]:
    """Raises if the module cannot be read; observing must not call get_start_method(allow_none = False), which pins
    _actual_context and makes a later set_start_method() raise."""
    module = __import__(module_name)
    method = module.get_start_method(allow_none = True)
    if method is None:
        method = _unpinned_default_start_method(module)
    return method


def multiprocessing_start_method() -> Optional[str]:
    """Start method datasets will use, or None: datasets imports Pool from multiprocess, not stdlib multiprocessing, so
    multiprocess decides, falling back to stdlib; None means the caller treats forking as unavailable."""
    for module_name in ("multiprocess", "multiprocessing"):
        try:
            return _module_start_method(module_name)
        except Exception:
            continue
    return None


def _workers_unusable_reason() -> Optional[str]:
    """Non-fork workers cannot re-import the dynamically generated trainer module; macOS stays in-process by stated
    policy, not as a probe side effect, since fork can crash there (CPython moved its default to spawn in 3.8,
    bpo-33725; multiprocess never copied that)."""
    start_method = multiprocessing_start_method()
    if start_method != "fork":
        return f"this process uses the {start_method!r} start method"
    if sys.platform == "darwin":
        return "'multiprocess' forks on macOS, which CPython documents as unsafe there (bpo-33725)"
    return None


def _cgroup_cpu_quota() -> Optional[float]:
    """cgroup CPU ceiling in cores, or None outside one; reuses hf_xet_tuning's reader (v1 and v2, /proc/self/cgroup
    walk, unlimited sentinels), imported lazily so the module loads alone."""
    try:
        from unsloth_zoo.hf_xet_tuning import cgroup_cpu_limit
        return cgroup_cpu_limit()
    except Exception:
        return None


# Module constant so tests can point the unaided reader at a fixture tree.
CGROUP_ROOT = "/sys/fs/cgroup"


def _cgroup_first_line(path: str) -> Optional[str]:
    try:
        # Encoding is explicit: these kernel files are ASCII, but a locale-dependent read crashes or produces mojibake
        # on a Windows console codepage, and CI checks every read and write for it.
        with open(path, "r", encoding = "utf-8") as f:
            return f.readline().strip()
    except OSError:
        return None


def _cgroup_int(raw: Optional[str]) -> Optional[int]:
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _cgroup_limit(raw: Optional[str]) -> Optional[int]:
    """``"max"`` (or an unparseable value) means unlimited."""
    value = _cgroup_int(raw)
    if value is None:
        return None
    # cgroup v1 spells "unlimited" as a near-2^63 sentinel.
    return value if 0 < value < (1 << 62) else None


def _cgroup_dirs(root: str, rel: Optional[str]) -> list:
    """root/rel and every ancestor up to root, innermost first; a parent slice's limit binds this process just as its own does."""
    dirs = []
    if rel and rel != "/":
        current = os.path.normpath(os.path.join(root, rel.lstrip("/")))
        while current.startswith(root + os.sep):
            dirs.append(current)
            current = os.path.dirname(current)
    dirs.append(root)
    return dirs


def _proc_self_cgroup() -> list:
    try:
        with open("/proc/self/cgroup", "r", encoding = "utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except OSError:
        return []  # not Linux, or procfs is hidden


def _cgroup_free_bytes_unaided() -> Optional[int]:
    """Fallback for an older unsloth_zoo without the private cgroup helpers, mirroring _cgroup_free_bytes; a bare public
    cgroup_memory_limit is last resort, as a limit is still a ceiling."""
    lines = _proc_self_cgroup()
    free = []

    if os.path.isdir(CGROUP_ROOT):
        rel = None
        # The v2 line starts with 0::<path>; under systemd hybrid mode v1 lines share the file, so scan for it rather than taking the first line.
        for line in lines:
            if line.startswith("0::"):
                rel = line[3:].strip()
                break
        for directory in _cgroup_dirs(CGROUP_ROOT, rel):
            used = _cgroup_int(_cgroup_first_line(os.path.join(directory, "memory.current")))
            for name in ("memory.max", "memory.high"):
                limit = _cgroup_limit(_cgroup_first_line(os.path.join(directory, name)))
                if limit is not None:
                    free.append(limit if used is None else limit - used)

    v1_root = os.path.join(CGROUP_ROOT, "memory")
    if os.path.isdir(v1_root):
        rel = None
        for line in lines:
            # "<id>:<controllers>:<path>"; mounts are often combined.
            parts = line.split(":", 2)
            if len(parts) == 3 and "memory" in parts[1].split(","):
                rel = parts[2]
                break
        for directory in _cgroup_dirs(v1_root, rel):
            used = _cgroup_int(_cgroup_first_line(os.path.join(directory, "memory.usage_in_bytes")))
            limit = _cgroup_limit(
                _cgroup_first_line(os.path.join(directory, "memory.limit_in_bytes"))
            )
            if limit is not None:
                free.append(limit if used is None else limit - used)

    if free:
        return max(min(free), 0)

    try:
        from unsloth_zoo.hf_xet_tuning import cgroup_memory_limit
        return cgroup_memory_limit()
    except Exception:
        return None


def _cgroup_free_bytes() -> Optional[int]:
    """Bytes free under the binding cgroup memory limit, or None; each limit pairs with the usage of the directory that
    set it, since an ancestor's usage counts unseen siblings; root memory.current is the whole machine, so never
    subtract it from a unit's MemoryMax."""
    try:
        from unsloth_zoo.hf_xet_tuning import (
            _cgroup_v1_dirs,
            _cgroup_v2_dirs,
            _parse_limit,
            _read_first_line,
        )
    except Exception:
        # An older unsloth_zoo lacks these helpers, so pair the limit with current usage: an 8GB cgroup with 6GB
        # resident would otherwise report 8GB free, and memory pressure is what this ceiling guards.
        return _cgroup_free_bytes_unaided()

    def _used(path):
        raw = _read_first_line(path)
        if not raw:
            return None
        try:
            return int(raw.strip())
        except ValueError:
            return None

    free = []
    for directory in _cgroup_v2_dirs():
        used = _used(directory / "memory.current")
        for name in ("memory.max", "memory.high"):
            limit = _parse_limit(_read_first_line(directory / name))
            if limit is not None:
                free.append(limit if used is None else limit - used)
    for directory in _cgroup_v1_dirs("memory"):
        used = _used(directory / "memory.usage_in_bytes")
        limit = _parse_limit(_read_first_line(directory / "memory.limit_in_bytes"))
        if limit is not None:
            free.append(limit if used is None else limit - used)

    return max(min(free), 0) if free else None


def _available_memory_gb() -> Optional[float]:
    """Free RAM usable by this process, or None; psutil's virtual_memory().available reports the host inside a container, so a 2GB pod on a 512GB box got OOM-killed."""
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
    except Exception:
        return None

    free = _cgroup_free_bytes()
    if free is not None:
        available_gb = min(available_gb, free / (1024**3))
    return available_gb


def _usable_cpus() -> Optional[int]:
    """CPUs this process may run on, or None: cpu_count() reports the host's, so under taskset, Slurm pinning or a
    Kubernetes CPU quota a one-core job would auto-size workers that contend for that core."""
    try:
        import psutil
        cpus = psutil.cpu_count()
    except Exception:
        cpus = os.cpu_count()
    if not cpus:
        return None

    try:
        cpus = min(cpus, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass  # not Linux, or the call is unavailable: the host count stands

    quota = _cgroup_cpu_quota()
    if quota is not None and quota > 0:
        # A fractional quota still binds: Kubernetes "cpu: 500m" is 0.5 cores.
        cpus = min(cpus, max(1, int(quota)))
    return cpus


def _affordable_workers() -> Optional[int]:
    """How many workers free RAM can cover, or None when it cannot be read."""
    available_gb = _available_memory_gb()
    if available_gb is None:
        return None
    budget_gb = available_gb * MEMORY_BUDGET_FRACTION
    return int(budget_gb / WORKER_MEMORY_BUDGET_GB)


def _clamp_by_memory(num_proc: int) -> Optional[int]:
    """Bound a worker count by RAM; None means do not use workers. Applies to explicit counts too, since a caller's
    number (Studio passes max(1, cpu_count // 4)) could OOM a machine with no room."""
    affordable = _affordable_workers()
    if affordable is None:
        # Without a memory reading, honour the requested count rather than serialise a machine that may be perfectly capable.
        return num_proc

    if affordable < 2:
        _warn_once(
            "memory_serial",
            "not enough free memory for dataset tokenization workers "
            f"(~{WORKER_MEMORY_BUDGET_GB:g}GB each); tokenizing in-process.",
        )
        return None

    if affordable < num_proc:
        _warn_once(
            "memory_clamp",
            f"reducing dataset_num_proc {num_proc} -> {affordable} to fit free "
            f"memory (~{WORKER_MEMORY_BUDGET_GB:g}GB per worker). Set "
            f"{NUM_PROC_ENV_VAR} to override.",
        )
        return affordable

    return num_proc


def _auto_num_proc() -> Optional[int]:
    """Worker count to use when the caller did not ask for a specific one."""
    try:
        import psutil  # noqa: F401  the memory clamp below needs it to mean anything
    except Exception:
        # No psutil means no memory reading; stay conservative.
        return None

    cpus = _usable_cpus()
    if cpus is None:
        return None
    if cpus <= 1:
        # Workers would only contend for the single core they are pinned to.
        return None

    return _clamp_by_memory(min(max(cpus // 2, 2), AUTO_NUM_PROC_CAP))


def _serial(serial_as_none: bool) -> Optional[int]:
    """The value meaning "run in-process", encoded for the calling layer.

    At a ``map()`` call site that is ``None``, the only value ``datasets`` runs
    in-process on every supported release. At the *config* layer it is ``1``,
    because a config ``None`` is read downstream as "auto-size me" and would
    inflate a serial request; the call site turns that ``1`` back into ``None``.

    That only holds while workers are usable, hence the veto check. Where they
    are not, nothing can inflate a config ``None`` -- every auto-sizer that reads
    it vetoes too -- while ``1`` is unsafe, since only the SFT map site is
    rewritten by ``rl_replacements.py``: TRL's DPO, KTO, CPO, ORPO, Reward and
    PRM trainers hand ``args.dataset_num_proc`` straight to ``Dataset.map``,
    whose ``Pool(1)`` child re-imports the user's ``__main__`` (#3211 / #3397).
    """
    if serial_as_none:
        return None
    return None if _workers_unusable_reason() is not None else 1


def _from_environment() -> "tuple[bool, Optional[int]]":
    """Read NUM_PROC_ENV_VAR. Returns (was_set_and_valid, value)."""
    raw = os.environ.get(NUM_PROC_ENV_VAR)
    if raw is None:
        return False, None

    text = raw.strip()
    if text.lower() in ("", "0", "none", "null", "false"):
        return True, None

    try:
        value = int(text)
    except ValueError:
        _warn_once(
            "env_invalid",
            f"{NUM_PROC_ENV_VAR}={raw!r} is not an integer, ignoring it.",
        )
        return False, None

    if value < 0:
        _warn_once(
            "env_negative",
            f"{NUM_PROC_ENV_VAR}={raw!r} is negative, ignoring it.",
        )
        return False, None
    if value <= 1:
        return True, None
    return True, value


def environment_override() -> "tuple[bool, Optional[int]]":
    """Returns (was_set_and_valid, value), value None meaning in-process; NUM_PROC_ENV_VAR is the uncapped hatch, so
    callers need this rather than reading it directly; an unparseable or negative value is not the hatch and is
    ignored with a warning."""
    return _from_environment()


def get_dataset_num_proc(
    desired: Optional[int] = None, *, serial_as_none: bool = True
) -> Optional[int]:
    """Return a safe ``num_proc`` for ``Dataset.map()`` / ``Dataset.filter()``.

    ``None`` means "run in-process" -- the only value that builds no worker pool
    on every supported ``datasets`` release, since 4.x pools for any
    ``num_proc >= 1``. Everything returned is bounded by free memory, explicit
    counts included; only ``NUM_PROC_ENV_VAR`` is exempt.

    Args:
        desired: The worker count the caller asked for, or None to auto-size.
        serial_as_none: How to encode "run in-process". True at a ``map()`` call
            site, yielding ``None``. **False when writing back to a config**,
            yielding ``1``, because downstream readers
            (``dataset_utils.sft_prepare_dataset``) treat a config ``None`` as
            "auto-size me" and would inflate a user's ``1``.

    Returns:
        A worker count >= 2, or this layer's in-process sentinel (``None`` at a
        call site, ``1`` at the config layer).
    """
    # 1. The environment override wins over everything, uncapped and unvetoed, so a user who knows their workload is fork-safe is never downgraded.
    env_set, env_value = _from_environment()
    if env_set:
        # In-process must still be encoded for this layer, or UNSLOTH_DATASET_NUM_PROC=0 (the hatch the dead-worker
        # message recommends) would inflate a config instead of removing workers.
        return _serial(serial_as_none) if env_value is None else env_value

    # 2. Workers are unusable whatever was requested; see _workers_unusable_reason for the two refusals.
    unusable = _workers_unusable_reason()
    if unusable is not None:
        if isinstance(desired, int) and not isinstance(desired, bool) and desired > 1:
            _warn_once(
                "start_method",
                f"dataset_num_proc = {desired} cannot be used because "
                f"{unusable}. Falling back to single-process tokenization. "
                f"Set {NUM_PROC_ENV_VAR} to override.",
            )
        return _serial(serial_as_none)

    # 3. Normalise no-multiprocessing requests: 1 is the trap, since callers pass it to mean serial but datasets >= 4.1 hands them a Pool(1).
    if isinstance(desired, int) and not isinstance(desired, bool) and desired <= 1:
        return _serial(serial_as_none)

    # 4. Auto-size when no usable request was made, then bound by memory; a non-int, or a bool (an int subclass), is not a request.
    if desired is None or not isinstance(desired, int) or isinstance(desired, bool):
        num_proc = _auto_num_proc()
    else:
        num_proc = _clamp_by_memory(desired)

    if num_proc is None:
        return _serial(serial_as_none)
    return num_proc


# datasets raises this message when a pool worker dies, never reads its exit status, so an OOM kill, a segfault and a
# real exception look identical (issue 2693).
_WORKER_DIED = "subprocesses has abruptly died"


@contextlib.contextmanager
def map_failure_diagnostics(num_proc: Optional[int]) -> "Iterator[None]":
    """Re-raise a dead-worker error from Dataset.map naming the start method and worker count, which the original message drops; the cause is chained."""
    try:
        yield
    except RuntimeError as exception:
        if _WORKER_DIED not in str(exception):
            raise

        workers = num_proc if isinstance(num_proc, int) else 1
        estimated_gb = workers * WORKER_MEMORY_BUDGET_GB
        raise RuntimeError(
            f"Unsloth: tokenization failed because a dataset worker died.\n"
            f"  dataset_num_proc = {num_proc!r} "
            f"({workers} worker{'' if workers == 1 else 's'}, "
            f"start method {multiprocessing_start_method()!r})\n"
            f"  Each worker holds its own copy of the tokenizer and its dataset "
            f"shard, so this run needed roughly {estimated_gb:g}GB on top of the "
            f"model. The most common cause is the out-of-memory killer.\n"
            f"  Retry with {NUM_PROC_ENV_VAR}=0 for the fewest workers this path "
            f"can use, or {NUM_PROC_ENV_VAR}=<n> to choose a count. That is "
            f"in-process everywhere, though on an unsloth_zoo older than the one "
            f"that reads 1 as in-process, train_on_responses_only still forks a "
            f"single worker on a split over {ZOO_MIN_ROWS_FOR_MULTIPROC:,} rows: "
            f"a bare None there reads as 'size it for me'.\n"
            f"  Original error: {exception}"
        ) from exception


def _largest_split_rows(trainer) -> Optional[int]:
    """Rows in the largest sized split train_on_responses_only maps over, or None if no split has a readable length;
    unsized splits are skipped so they cannot hide a sized sibling; eval_dataset may be a dict."""
    if trainer is None:
        return None

    largest = 0
    measured = False
    for attribute in ("train_dataset", "eval_dataset"):
        dataset = getattr(trainer, attribute, None)
        if dataset is None:
            continue
        splits = list(dataset.values()) if isinstance(dataset, dict) else [dataset]
        for split in splits:
            if split is None:
                continue
            try:
                rows = len(split)
            except Exception:
                # Skip rather than let it mask a sized sibling.
                continue
            measured = True
            largest = max(largest, rows)
    return largest if measured else None


def _zoo_auto_sizer_forks() -> bool:
    """Whether train_on_responses_only auto-sizes workers for a bare None; it asks stdlib multiprocessing, not
    multiprocess, so where the two disagree None means size-it-for-me and datasets builds that pool on the non-fork
    multiprocess context."""
    try:
        return _module_start_method("multiprocessing") == "fork"
    except Exception:
        return False


def _serial_for_the_zoo(value: Optional[int]) -> Optional[int]:
    """Serial is None only while train_on_responses_only's own check refuses workers, else 1: in-process on newer
    releases, a Pool(1) (not cpu_count + 4) on older ones."""
    if value is not None or not _zoo_auto_sizer_forks():
        return value
    _warn_once(
        "start_method_split",
        "'multiprocess' and 'multiprocessing' disagree about the start method "
        f"({multiprocessing_start_method()!r} against 'fork'), so "
        "train_on_responses_only is being held to one worker rather than left to "
        f"size itself. Set {NUM_PROC_ENV_VAR} to override.",
    )
    return 1


def resolve_responses_only_num_proc(trainer, num_proc):
    """Caps the auto worker count (uncapped in older releases, issue 2693); serial is 1 on fork but None on spawn, since
    a Pool(1) child re-imports __main__ (issues 3211, 3397); an explicit count skips its small-split guard, so it is
    worth it only for splits big enough to parallelize."""
    # Mirrors that helper's own test so explicit means the same on both sides; it treats bools as auto, since type(True) is not int.
    was_auto = num_proc is None or type(num_proc) is not int
    rows = _largest_split_rows(trainer)
    small = rows is None or rows < ZOO_MIN_ROWS_FOR_MULTIPROC

    if not was_auto:
        resolved = get_dataset_num_proc(num_proc, serial_as_none = False)
        if resolved == 1 and small:
            # Serial with every split under the threshold: the helper's guard runs them in-process, but an older
            # unsloth_zoo would not run that 1 in-process, so UNSLOTH_DATASET_NUM_PROC=0 would still build a Pool(1).
            return None
        return _serial_for_the_zoo(resolved)

    if small:
        # The escape hatch must win here too.
        env_set, env_value = _from_environment()
        if env_set and env_value is not None:
            return env_value
        # Otherwise its guard yields None, more in-process than the 1 expressible here; safe on any start method since every split is under the per-split guard.
        return num_proc

    return _serial_for_the_zoo(get_dataset_num_proc(None, serial_as_none = False))
