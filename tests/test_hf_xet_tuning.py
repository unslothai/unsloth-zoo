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

"""Tests for the RAM/CPU-derived hf_xet buffer caps."""

from __future__ import annotations

import dataclasses

import pytest

from unsloth_zoo import hf_xet_tuning as tuning

GB = 1_000_000_000
MB = 1_000_000


def _profile(ram_gb: float, cpus: int = 8, free_disk_gb: float = 0) -> tuning.SystemProfile:
    return tuning.SystemProfile(
        total_ram_bytes = int(ram_gb * GB),
        available_ram_bytes = int(ram_gb * GB),
        cpu_count = cpus,
        ram_source = "test",
        cpu_source = "test",
        free_disk_bytes = int(free_disk_gb * GB),
        # 0 GB means "not measured" for the tests that do not care; a REAL zero is a full disk and
        # gets a source, because the two take different branches.
        disk_source = "test" if free_disk_gb else "unknown",
    )


@pytest.mark.parametrize(
    "ram_gb, cpus, limit, size, perfile, files, streams",
    [
        # An eighth of RAM is the budget and a thirty-second is a per-file buffer, so the numbers
        # describe ONE allocation. The shared buffer is the exception: it reaches for xet-core's
        # own 2 GB default and only falls short where half the budget or a sixth of RAM says it
        # must, because a quarter of the budget measured 0.73x on an 8 GB laptop.
        (8, 4, 1 * GB, 500 * MB, 128 * MB, 3, 8),
        # No cliffs: 12 GB sits between the old table's 8 and 16 GB rows and gets a budget to match,
        # where the old step function gave it the 8 GB row's.
        (12, 4, 1500 * MB, 750 * MB, 128 * MB, 4, 8),
        (16, 8, 2 * GB, 1 * GB, 128 * MB, 7, 16),
        # 32 GB is the first machine that can hold xet-core's stock buffer outright.
        (32, 16, 4 * GB, 2 * GB, 128 * MB, 15, 32),
        # A big host is bounded by a budget proportional to what it has, not by a laptop's. Holding
        # a 2 TB server to the old flat 4 GB row cost 30% against xet-core's own defaults.
        (128, 64, 16 * GB, 4 * GB, 500 * MB, 24, 124),
        (2048, 192, 64 * GB, 16 * GB, 2000 * MB, 24, 124),
        # Cores bound concurrency independently of RAM: 8 cores on a 2 TB box still gets 8 files.
        (2048, 8, 64 * GB, 16 * GB, 2000 * MB, 8, 16),
        # ...and so does the budget. A 64-core container with 8 GB (CI, or a cgroup-limited pod)
        # cannot use 64 file buffers or 128 streams out of a 1 GB budget, and opening them anyway is
        # how a small machine on a thin link ends up worse off than with no tuning at all.
        (8, 64, 1 * GB, 500 * MB, 128 * MB, 3, 14),
    ],
)
def test_knobs_scale_with_the_machine(ram_gb, cpus, limit, size, perfile, files, streams):
    env = tuning.xet_env_overrides(_profile(ram_gb, cpus))
    assert int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"]) == limit
    assert int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE"]) == size
    assert int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE"]) == perfile
    assert int(env["HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"]) == files
    assert int(env["HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY"]) == streams


def test_a_big_host_lands_on_xet_cores_own_high_performance_numbers():
    """The point of scaling rather than gating: on the box where HF_XET_HIGH_PERFORMANCE was worth
    2.55x (16684 vs 6553 Mbit/s), the scaled knobs reach the same buffer sizing that preset uses,
    without the flag and without discarding the bound on smaller machines. Concurrent FILES is the
    one number we do not follow: the preset's 100 x 2 GB is 200 GB of per-file buffer against its
    own 64 GB limit, so most of it can never be allocated."""
    env = tuning.xet_env_overrides(_profile(2048, cpus = 192))
    preset = {  # xet_runtime/src/config/xet_config.rs, with_high_performance()
        "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT": str(64 * GB),
        "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE": str(16 * GB),
        "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE": str(2000 * MB),
        "HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY": "124",
    }
    for key, value in preset.items():
        assert env[key] == value, key


def test_the_budget_never_promises_more_than_the_disk_can_take():
    """A 2 TB host with a nearly full disk is not a 2 TB budget: buffering 64 GB of a download that
    will fail on ENOSPC helps nobody, and the free-space figure is the only signal we have."""
    roomy = tuning.xet_env_overrides(_profile(2048, cpus = 192, free_disk_gb = 8000))
    tight = tuning.xet_env_overrides(_profile(2048, cpus = 192, free_disk_gb = 40))
    assert int(roomy["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"]) == 64 * GB
    assert int(tight["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"]) == 10 * GB
    # Never below the floor, though: a full disk must not shrink the buffer to nothing, because the
    # download can still be the one that frees space (a resume, or a cache on another filesystem).
    full = tuning.xet_env_overrides(_profile(2048, cpus = 192, free_disk_gb = 0.5))
    assert int(full["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"]) == 1 * GB


def test_free_disk_is_measured_for_the_cache_not_the_cwd(monkeypatch, tmp_path):
    """The cache can sit on a different filesystem from wherever the process happens to be."""
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "not-created-yet" / "hub"))
    assert tuning.hf_cache_root() == tmp_path / "not-created-yet" / "hub"
    free, source = tuning._free_disk()
    # The leaf does not exist, so it walks up to the first parent that does rather than giving up.
    assert free > 0 and source in {str(p) for p in (tmp_path, *tmp_path.parents)}


def test_cache_root_follows_hubs_full_precedence(monkeypatch, tmp_path):
    """A cache moved with XDG_CACHE_HOME or the legacy HUGGINGFACE_HUB_CACHE lives on a different
    filesystem from ``~``, so measuring the home directory's free space sizes the buffer from a
    disk the download never writes to."""
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(var, raising = False)

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert tuning.hf_cache_root() == tmp_path / "xdg" / "huggingface" / "hub"

    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "legacy"))
    assert tuning.hf_cache_root() == tmp_path / "legacy"

    # HF_HOME names the home, not the hub cache: Hub appends "hub" to it.
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    assert tuning.hf_cache_root() == tmp_path / "home" / "hub"

    # HF_HUB_CACHE is still the most specific and wins over all of them.
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "explicit"))
    assert tuning.hf_cache_root() == tmp_path / "explicit"


def test_worst_case_buffer_stays_under_the_limit():
    """size + files*perfile is what hf_xet can hold, so it must not exceed the budget. The stock
    defaults (2GB + 8*512MB, capped at 8GB) are exactly how an 8GB spike happens."""
    for ram_gb, cpus in ((4, 2), (8, 8), (8, 96), (16, 8), (32, 64), (64, 16), (512, 192), (2048, 192)):
        env = tuning.xet_env_overrides(_profile(ram_gb, cpus))
        worst = (
            int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE"])
            + int(env["HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"])
            * int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE"])
        )
        limit = int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"])
        assert worst <= limit, f"{ram_gb}GB: worst case {worst} exceeds limit {limit}"
        # The bound that matters is proportional, not absolute: never more than a third of the
        # machine's RAM, so the buffer cannot be the reason a box OOMs. A large host is allowed a
        # large budget precisely because it has one to spare.
        assert worst <= ram_gb * GB / 3, f"{ram_gb}GB: worst case {worst} is too much of it"


def test_unknown_ram_is_read_as_a_small_machine():
    """Guessing low costs throughput; guessing high costs an OOM."""
    env = tuning.xet_env_overrides(_profile(0))
    assert int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"]) == 1 * GB


def test_durations_carry_a_unit_suffix():
    """hf_xet SILENTLY IGNORES a bare integer duration, leaving the 300s default in place."""
    env = tuning.xet_env_overrides(_profile(16))
    for key in (
        "HF_XET_CLIENT_READ_TIMEOUT",
        "HF_XET_CLIENT_CONNECT_TIMEOUT",
        "HF_XET_CLIENT_RETRY_MAX_DURATION",
    ):
        assert env[key][-1].isalpha(), f"{key}={env[key]!r} needs a unit suffix"


def test_high_performance_is_turned_off():
    """xet-core applies the preset AFTER reading the environment, so leaving it on discards every cap."""
    env = tuning.xet_env_overrides(_profile(16))
    assert env["HF_XET_HIGH_PERFORMANCE"] == "0"
    assert env["HF_XET_HP"] == "0"


def test_cpu_count_bounds_concurrency():
    small = tuning.xet_env_overrides(_profile(32, cpus = 2))
    large = tuning.xet_env_overrides(_profile(32, cpus = 128))
    assert int(small["HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"]) == 2
    assert int(small["HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY"]) == 4
    # Two streams per core, but no more than the 4 GB budget has 64 MiB xorb slots for, and never
    # more files than the budget affords buffers for.
    assert int(large["HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY"]) == 59
    assert int(large["HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"]) <= 24
    # The ramp always starts at or below the ceiling, whichever bound produced it.
    for env in (small, large):
        assert int(env["HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY"]) <= int(
            env["HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY"]
        )


def test_throttled_halves_the_stream_ceiling():
    """A previous 429 means the account, not the machine, is the limit."""
    normal = tuning.xet_env_overrides(_profile(32, cpus = 16))
    throttled = tuning.xet_env_overrides(_profile(32, cpus = 16), throttled = True)
    assert int(throttled["HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY"]) == max(
        4, int(normal["HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY"]) // 2
    )


def test_apply_never_overwrites_a_user_setting():
    env = {"HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT": "16000000000"}
    tuning.apply_xet_env(env, profile = _profile(16))
    assert env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"] == "16000000000"
    # ...but the rest is still filled in.
    assert env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE"] == str(128 * MB)


def test_a_user_set_high_performance_flag_is_left_alone():
    """We used to clear it, which is how a machine that had opted in lost 2.55x of its download
    throughput the moment it imported unsloth_zoo (16684 -> 6553 Mbit/s, measured on a 192-core
    1996 GiB box). Enabling it is a deliberate act; setdefault applies to it like everything else."""
    for var in ("HF_XET_HIGH_PERFORMANCE", "HF_XET_HP"):
        env = {var: "1"}
        written = tuning.apply_xet_env(env, profile = _profile(16))
        assert env[var] == "1", var
        assert var not in written, var


def test_high_performance_also_stands_our_sizing_down():
    """Leaving the caps on alongside it is the worst of both: xet-core reads the flag last and
    voids the LIMIT, but still honours the smaller per-file and concurrency numbers, so the
    transfer ends up smaller than either choice made cleanly."""
    env = {"HF_XET_HIGH_PERFORMANCE": "1"}
    tuning.apply_xet_env(env, profile = _profile(16))
    for key in tuning._CAPS_VOIDED_BY_HIGH_PERFORMANCE:
        assert key not in env, key
    # Non-sizing settings are unrelated to the memory bound and still apply.
    assert env["HF_XET_CHUNK_CACHE_SIZE_BYTES"] == "0"


def test_force_caps_still_overrides_high_performance(monkeypatch):
    """The escape hatch for someone who really does want a machine bounded: the flag has to be
    turned off for the caps to mean anything, so this is the one case that overwrites it."""
    monkeypatch.setenv("UNSLOTH_XET_FORCE_CAPS", "1")
    env = {"HF_XET_HIGH_PERFORMANCE": "1"}
    written = tuning.apply_xet_env(env, profile = _profile(16))
    assert env["HF_XET_HIGH_PERFORMANCE"] == "0"
    assert written["HF_XET_HIGH_PERFORMANCE"] == "0"
    assert env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"] == str(2 * GB)


def test_a_large_machine_is_not_held_to_a_laptops_buffer():
    """The table used to flat-line at 24 GB, so a 2 TB server got a 24 GB laptop's 4 GB buffer.
    Measured, that ran 30% BELOW xet-core's own defaults, because the buffer gates how much can be
    in flight. The budget must grow with the machine, monotonically and without a cliff."""
    seen = []
    for ram_gb in (8, 16, 32, 128, 512, 2048):
        env = tuning.xet_env_overrides(_profile(ram_gb, cpus = 64))
        seen.append((ram_gb, int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"])))
    for (small_ram, small), (big_ram, big) in zip(seen, seen[1:]):
        assert big >= small, f"{big_ram}GB got {big} but {small_ram}GB got {small}"
    assert seen[-1][1] > seen[2][1], f"a 2TB box still capped like a 32GB one: {seen}"
    # And the small end is untouched, which is where the bound earns its keep.
    assert seen[0][1] == 1 * GB


def test_cgroup_limit_beats_the_host_total(monkeypatch, tmp_path):
    """Inside a container psutil reports the HOST's RAM, which is how a 16GB runner gets an 8GB
    buffer, so the cgroup ceiling has to win."""
    monkeypatch.setattr(tuning, "_psutil_memory", lambda: (512 * GB, 500 * GB))
    monkeypatch.setattr(tuning, "cgroup_memory_limit", lambda: 8 * GB)
    monkeypatch.setattr(tuning, "cgroup_cpu_limit", lambda: 2.0)
    profile = tuning.system_profile()
    assert profile.total_ram_bytes == 8 * GB
    assert profile.ram_source == "cgroup"
    assert profile.cpu_count == 2
    env = tuning.xet_env_overrides(profile)
    # An eighth of 8 GB is the 1 GB floor, and the 2-core quota allows only 2 files in flight, so
    # the worst case (256MB + 2*128MB) sits well inside it.
    assert int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"]) == 1 * GB
    assert int(env["HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"]) == 2


def test_parse_limit_treats_sentinels_as_unlimited():
    assert tuning._parse_limit("max") is None
    assert tuning._parse_limit("9223372036854771712") is None  # cgroup v1 "unlimited"
    assert tuning._parse_limit("not-a-number") is None
    assert tuning._parse_limit(None) is None
    assert tuning._parse_limit("8000000000") == 8 * GB


def test_xet_log_env_marks_the_destination_as_a_directory(tmp_path):
    """hf_xet only treats HF_XET_LOG_DEST as a directory when it ends with a separator."""
    env = tuning.xet_log_env(tmp_path)
    assert env["HF_XET_LOG_DEST"].endswith(("/", "\\"))
    assert env["HF_XET_LOG_FORMAT"] == "json"
    assert "HF_XET_SYSTEM_MONITOR_ENABLED" not in env
    assert tuning.xet_log_env(tmp_path, diagnostics = True)["HF_XET_SYSTEM_MONITOR_ENABLED"] == "1"


def test_scan_xet_log_extracts_failures(tmp_path):
    (tmp_path / "a.log").write_text(
        '{"level":"INFO","message":"downloading"}\n'
        '{"level":"ERROR","message":"CAS fetch failed for term abc"}\n'
    )
    messages = tuning.scan_xet_log(tmp_path)
    assert len(messages) == 1
    assert "CAS fetch failed" in messages[0]


def test_scan_xet_log_is_best_effort(tmp_path):
    assert tuning.scan_xet_log(None) == []
    assert tuning.scan_xet_log(tmp_path / "does-not-exist") == []


def test_system_profile_is_sane_on_this_machine():
    profile = tuning.system_profile()
    assert profile.cpu_count >= 1
    assert profile.total_ram_bytes > 0
    assert dataclasses.asdict(profile)["ram_source"]


def test_a_nested_cgroup_v1_limit_is_found(monkeypatch, tmp_path):
    """Reading only the v1 controller ROOT is right inside a container (runc bind-mounts the
    container's own cgroup dir onto /sys/fs/cgroup/<controller>) and wrong elsewhere: a Slurm step at
    /sys/fs/cgroup/memory/slurm/uid_N/job_N/step_N reads the "unlimited" sentinel at the root, so a
    32 GB step on a 1 TB node sized its buffer from 1 TB and was OOM killed. Mounts are often
    combined ("cpu,cpuacct"), so the controller list has to be split rather than matched whole.
    """
    rel = "slurm/uid_2001/job_304876/step_0"
    monkeypatch.setattr(tuning, "_proc_self_cgroup", lambda: [
        "10:memory:/" + rel,
        "4:cpu,cpuacct:/" + rel,     # combined mount: matching the list whole would miss this
    ])
    monkeypatch.setattr(tuning, "CGROUP_ROOT", tmp_path)

    # Controller roots read "unlimited", exactly as they do on a real Slurm node.
    (tmp_path / "memory").mkdir(parents = True)
    (tmp_path / "memory" / "memory.limit_in_bytes").write_text("9223372036854771712\n")
    (tmp_path / "cpu").mkdir(parents = True)
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("-1\n")
    (tmp_path / "cpu" / "cpu.cfs_period_us").write_text("100000\n")

    nested_mem = tmp_path / "memory" / rel
    nested_mem.mkdir(parents = True)
    (nested_mem / "memory.limit_in_bytes").write_text(str(32 * GB) + "\n")
    nested_cpu = tmp_path / "cpu" / rel
    nested_cpu.mkdir(parents = True)
    (nested_cpu / "cpu.cfs_quota_us").write_text("400000\n")
    (nested_cpu / "cpu.cfs_period_us").write_text("100000\n")

    assert tuning.cgroup_memory_limit() == 32 * GB
    assert tuning.cgroup_cpu_limit() == 4.0


def test_a_hybrid_cgroup_v2_line_below_the_first_is_still_found(monkeypatch, tmp_path):
    """Under systemd hybrid mode v1 controller lines share the file, so scanning for "0::" rather
    than reading line 1 is what makes the v2 read version independent."""
    monkeypatch.setattr(tuning, "_proc_self_cgroup", lambda: [
        "4:cpu,cpuacct:/user.slice",
        "0::/user.slice/user-1000.slice/session-3.scope",
    ])
    monkeypatch.setattr(tuning, "CGROUP_ROOT", tmp_path)
    scope = tmp_path / "user.slice/user-1000.slice/session-3.scope"
    scope.mkdir(parents = True)
    (scope / "memory.max").write_text(str(6 * GB) + "\n")

    assert tuning.cgroup_memory_limit() == 6 * GB


def test_fractional_cgroup_cpu_quota_binds(monkeypatch):
    """Kubernetes "cpu: 500m" arrives as cpu.max "50000 100000" -> 0.5. Requiring >= 1 discarded it
    and fell back to the host's core count, so a half-core pod opened streams sized for the node."""
    monkeypatch.setattr(tuning, "cgroup_cpu_limit", lambda: 0.5)
    profile = tuning.system_profile()
    assert profile.cpu_count == 1
    assert profile.cpu_source == "cgroup"

    env = tuning.xet_env_overrides(profile)
    assert int(env["HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY"]) <= 4


def test_fail_fast_timeouts_are_opt_out():
    """The shortened Xet timeouts belong in a supervised download child, not process-wide: xet-core
    reads its config once per process and applies it to uploads and direct huggingface_hub downloads
    too, neither of which our HTTP ladder can catch."""
    supervised = tuning.xet_env_overrides(_profile(16))
    global_env = tuning.xet_env_overrides(_profile(16), fail_fast = False)

    for key in tuning._FAIL_FAST_KEYS:
        assert key in supervised
        assert key not in global_env

    # The memory caps -- the whole point of the module -- are unaffected either way.
    for key in ("HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT", "HF_XET_HIGH_PERFORMANCE"):
        assert global_env[key] == supervised[key]


def test_apply_xet_env_does_not_shorten_timeouts_process_wide():
    env: dict[str, str] = {}
    tuning.apply_xet_env(env, profile = _profile(16))
    assert "HF_XET_CLIENT_RETRY_MAX_ATTEMPTS" not in env
    assert env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"]

    child: dict[str, str] = {}
    tuning.apply_xet_env(child, profile = _profile(16), fail_fast = True)
    assert child["HF_XET_CLIENT_RETRY_MAX_ATTEMPTS"] == "2"


# Real machines Unsloth actually runs on, smallest first. A regression that only shows up on a
# 2-core Colab or a 0.5-core pod is one nobody here can reproduce, so they are pinned by name.
DEVICES = [
    # name,                     RAM GB, cpus, free disk GB
    ("tiny VM",                      2,    2,     8),
    ("k8s pod, 0.5 cpu quota",       4,    1,    20),
    ("MacBook Air M2",               8,    8,   100),
    ("GitHub CI runner",            16,    4,    14),
    ("Colab free T4",             12.7,    2,    78),
    ("Kaggle T4 x2",                29,    4,    57),
    ("desktop, RTX 4090",           32,   16,   500),
    ("CI container on a big host",   8,   64,   200),
    ("Slurm step on a 1TB node",    32,    8,  1000),
    ("A100 node",                  200,   32,  3000),
    ("8xH100 node",               2048,  192,  8000),
]


@pytest.mark.parametrize("name, ram_gb, cpus, disk_gb", DEVICES, ids = [d[0] for d in DEVICES])
def test_every_device_stays_within_its_own_means(name, ram_gb, cpus, disk_gb):
    """One rule per resource, applied to every device we know of. Being generous on a big host is
    only defensible if the same arithmetic is conservative on a small one, and the small ones are
    where nobody notices until a user reports an OOM or a stalled download on hotel wifi."""
    env = tuning.xet_env_overrides(_profile(ram_gb, cpus, disk_gb))
    limit = int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"])
    size = int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE"])
    perfile = int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE"])
    files = int(env["HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"])
    streams = int(env["HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY"])
    initial = int(env["HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY"])
    worst = size + files * perfile

    assert worst <= limit, f"{name}: {worst} in flight against a {limit} budget"
    assert worst <= ram_gb * GB / 3, f"{name}: {worst} is too much of {ram_gb}GB"
    assert limit <= max(1 * GB, disk_gb * GB / 4), f"{name}: {limit} buffered onto {disk_gb}GB free"
    assert 2 <= files <= max(2, cpus), f"{name}: {files} files on {cpus} cores"
    # A stream holds a xorb; more of them than the budget can hold is queueing, not parallelism.
    assert 4 <= streams <= max(4, min(cpus * 2, limit // (64 * 1024 * 1024))), f"{name}: {streams}"
    assert 2 <= initial <= streams, f"{name}: ramp starts at {initial} of {streams}"


def test_the_smallest_devices_get_the_smallest_numbers():
    """A 2 GB VM and a fractional-core pod get the smallest allocation we hand out: two files, four
    streams, and a shared buffer bounded by a sixth of RAM rather than by xet-core's default, which
    on a 2 GB machine would be the whole machine."""
    for name, ram_gb, cpus, disk_gb in DEVICES[:2]:
        env = tuning.xet_env_overrides(_profile(ram_gb, cpus, disk_gb))
        assert int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"]) == 1 * GB, name
        assert int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE"]) <= ram_gb * GB / 6, name
        assert int(env["HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"]) == 2, name
        assert int(env["HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY"]) == 4, name


def test_only_the_big_hosts_get_the_big_numbers():
    """The converse, and the point of the change: the devices that can afford xet-core's own
    high-performance sizing are the only ones that get it."""
    big = {
        name for name, ram_gb, cpus, disk_gb in DEVICES
        if int(tuning.xet_env_overrides(_profile(ram_gb, cpus, disk_gb))
               ["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"]) > 8 * GB
    }
    assert big == {"A100 node", "8xH100 node"}


# Every HF_XET_* name xet-core 1.5.x actually reads, filtered to the groups we touch. Regenerate
# with:  grep -rhno "HF_XET_[A-Z_]\+" <xet-core>/xet_runtime --include=*.rs | cut -d: -f2- | sort -u
# We used to set HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY, which xet-core has never had a variable of
# that name for (the real one is HF_XET_RECONSTRUCTION_USE_VECTORED_WRITE, default true): a silent
# no-op that read, in review and in the logs, exactly like a setting that was being honoured.
XET_CORE_VARS = frozenset({
    "HF_XET_CACHE",
    "HF_XET_CHUNK_CACHE_SIZE_BYTES",
    "HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY",
    "HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY",
    "HF_XET_CLIENT_AC_MIN_DOWNLOAD_CONCURRENCY",
    "HF_XET_CLIENT_CONNECT_TIMEOUT",
    "HF_XET_CLIENT_READ_TIMEOUT",
    "HF_XET_CLIENT_RETRY_BASE_DELAY",
    "HF_XET_CLIENT_RETRY_MAX_ATTEMPTS",
    "HF_XET_CLIENT_RETRY_MAX_DURATION",
    "HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS",
    "HF_XET_HIGH_PERFORMANCE",
    "HF_XET_HP",
    "HF_XET_LOG_DEST",
    "HF_XET_LOG_FILE",
    "HF_XET_LOG_FORMAT",
    "HF_XET_LOG_PREFIX",
    "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT",
    "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE",
    "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE",
    "HF_XET_RECONSTRUCTION_MAX_RECONSTRUCTION_FETCH_SIZE",
    "HF_XET_RECONSTRUCTION_MIN_PREFETCH_BUFFER",
    "HF_XET_RECONSTRUCTION_MIN_RECONSTRUCTION_FETCH_SIZE",
    "HF_XET_RECONSTRUCTION_TARGET_BLOCK_COMPLETION_TIME",
    "HF_XET_RECONSTRUCTION_USE_VECTORED_WRITE",
    "HF_XET_SYSTEM_MONITOR_ENABLED",
    "HF_XET_SYSTEM_MONITOR_LOG_PATH",
    "HF_XET_SYSTEM_MONITOR_SAMPLE_INTERVAL",
})


def test_we_only_set_variables_xet_core_reads(tmp_path):
    emitted = set(tuning.xet_env_overrides(_profile(16))) | set(tuning.xet_log_env(tmp_path, diagnostics = True))
    unknown = emitted - XET_CORE_VARS
    assert not unknown, f"not read by xet-core: {sorted(unknown)}"


def test_the_prefetch_floor_never_undercuts_xet_cores_own(tmp_path):
    """Sizing DOWN from a default is a cap; sizing below it for no reason is just a slower
    download. Only a machine whose whole budget is smaller than the 1 GB default gets less."""
    for ram_gb, cpus in ((8, 4), (16, 8), (32, 16), (64, 32), (2048, 192)):
        env = tuning.xet_env_overrides(_profile(ram_gb, cpus))
        prefetch = int(env["HF_XET_RECONSTRUCTION_MIN_PREFETCH_BUFFER"])
        limit = int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"])
        assert prefetch <= tuning._STOCK_PREFETCH_BUFFER, f"{ram_gb}GB raised it to {prefetch}"
        assert prefetch <= limit, f"{ram_gb}GB: {prefetch} prefetch inside a {limit} budget"
        if limit >= 4 * GB:
            assert prefetch == tuning._STOCK_PREFETCH_BUFFER, f"{ram_gb}GB undercut it: {prefetch}"


def test_no_module_sets_a_variable_xet_core_does_not_read():
    """The dead variable was set from unsloth_zoo/__init__.py at import time, not from here, so a
    check confined to this module would have missed it. Scan the package instead."""
    import pathlib
    import re

    root = pathlib.Path(tuning.__file__).parent
    unknown: dict[str, set] = {}
    for path in root.rglob("*.py"):
        # Only quoted names: a variable is only ever read or written as a string literal, so this
        # skips prose ("any explicit HF_XET_RECONSTRUCTION_* cap") without needing to parse it.
        for name in re.findall(r"[\"'](HF_XET_[A-Z_]+)[\"']", path.read_text(errors = "ignore")):
            if name not in XET_CORE_VARS:
                unknown.setdefault(name, set()).add(path.relative_to(root).as_posix())
    assert not unknown, f"names xet-core does not read: { {k: sorted(v) for k, v in unknown.items()} }"


@pytest.mark.parametrize("ram_gb, cpus", [(8, 8), (12.7, 2), (16, 4), (32, 16), (64, 32), (2048, 192)])
def test_a_machine_that_can_hold_xet_cores_own_buffer_gets_it(ram_gb, cpus):
    """The rule the small-device regression came down to. Our sizing exists to CAP a machine that
    needs capping; handing one less than it would have had by doing nothing is not a cap, it is
    just a slower download. Measured: a quarter of the budget on an 8 GB laptop ran at 0.73x of
    what shipped before, and restoring this one value took it to 1.45x."""
    env = tuning.xet_env_overrides(_profile(ram_gb, cpus))
    size = int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE"])
    limit = int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"])
    stock = tuning._STOCK_BUFFER_SIZE
    # Only two things may hold it under stock, and only as far as they reach: half the budget, and
    # a sixth of RAM. Anything smaller than both of those is a choice, not a constraint.
    assert size >= min(stock, limit // 2, max(256 * MB, ram_gb * GB / 6)), f"{ram_gb}GB got {size}"
    # ...so a machine with room for stock on both counts is never handed less than stock.
    if limit // 2 >= stock and ram_gb * GB / 6 >= stock:
        assert size >= stock, f"{ram_gb}GB undercut stock: {size}"


def test_a_small_container_with_many_cores_still_fits_a_third_of_its_ram():
    """The budget floor is absolute, so on a 2 GB machine it is half the RAM, and letting cores
    alone set the file count put 973 MB in flight there. The proportional promise has to win."""
    for cpus in (2, 8, 64):
        env = tuning.xet_env_overrides(_profile(2, cpus))
        worst = (
            int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE"])
            + int(env["HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"])
            * int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE"])
        )
        assert worst <= 2 * GB / 3, f"2GB/{cpus} cores: {worst} in flight"


def test_the_flag_is_read_the_way_xet_core_reads_it():
    """xet-core checks HF_XET_HIGH_PERFORMANCE and only falls back to the HF_XET_HP alias when the
    first is unset (configuration_utils.rs get_high_performance_flag), so an explicit "0" masks the
    alias. Standing our sizing down over an alias xet-core is ignoring would leave the machine on
    xet-core's stock constants instead of the ones we sized for it."""
    masked = {"HF_XET_HIGH_PERFORMANCE": "0", "HF_XET_HP": "1"}
    tuning.apply_xet_env(masked, profile = _profile(16))
    assert "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT" in masked, "sizing stood down for nothing"

    alias_only = {"HF_XET_HP": "1"}
    tuning.apply_xet_env(alias_only, profile = _profile(16))
    assert "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT" not in alias_only


def test_a_full_disk_is_not_read_as_an_unmeasured_one():
    """shutil.disk_usage returning exactly zero free bytes is the case the clamp exists for, so it
    must not share a sentinel with "we could not measure". A full cache filesystem otherwise kept
    the whole 64 GB budget while one byte free took it to the floor."""
    full = tuning.SystemProfile(
        total_ram_bytes = 2048 * GB, available_ram_bytes = 2048 * GB, cpu_count = 192,
        ram_source = "test", cpu_source = "test", free_disk_bytes = 0, disk_source = "/data",
    )
    env = tuning.xet_env_overrides(full)
    assert int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"]) == 1 * GB

    unknown = dataclasses.replace(full, disk_source = "unknown")
    env = tuning.xet_env_overrides(unknown)
    assert int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"]) == 64 * GB


def test_free_disk_reports_unknown_when_it_cannot_measure(monkeypatch):
    import shutil

    def _boom(_path):
        raise OSError("no such filesystem")

    monkeypatch.setattr(shutil, "disk_usage", _boom)
    assert tuning._free_disk() == (0, "unknown")


def _two_volumes(monkeypatch, roomy: str, tight: str) -> None:
    """Pretend *roomy* and *tight* are separate filesystems, and fix RAM so only the disk moves."""
    import collections
    import shutil

    usage = collections.namedtuple("usage", "total used free")

    def _disk_usage(path):
        text = str(path)
        if text.startswith(tight):
            return usage(1 * GB, 1 * GB, 0)
        if text.startswith(roomy):
            return usage(9000 * GB, 1000 * GB, 8000 * GB)
        raise OSError(f"unexpected filesystem: {text}")

    monkeypatch.setattr(shutil, "disk_usage", _disk_usage)
    monkeypatch.setattr(tuning, "_psutil_memory", lambda: (2048 * GB, 2048 * GB))
    monkeypatch.setattr(tuning, "cgroup_memory_limit", lambda: None)


def test_an_explicit_cache_dir_is_the_disk_that_gets_measured(monkeypatch, tmp_path):
    """``cache_dir=`` beats every cache environment variable in huggingface_hub
    (``file_download.py``: ``if cache_dir is None: cache_dir = constants.HF_HUB_CACHE``), so a
    caller who names one has named the volume the bytes land on. Sizing off the global cache
    instead promises a 64 GB buffer to a full target disk, and throttles a roomy target to the
    floor because an unrelated default cache is full."""
    roomy = tmp_path / "roomy"
    tight = tmp_path / "tight"
    for directory in (roomy, tight):
        directory.mkdir()
    _two_volumes(monkeypatch, str(roomy), str(tight))

    monkeypatch.setenv("HF_HUB_CACHE", str(roomy / "hub"))
    # A full target volume is not sized from the roomy default cache.
    assert tuning.hf_cache_root(tight / "hub") == tight / "hub"
    profile = tuning.system_profile(tight / "hub")
    assert profile.free_disk_bytes == 0 and profile.disk_source.startswith(str(tight))
    env = tuning.xet_env_overrides(cache_dir = tight / "hub")
    assert int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"]) == 1 * GB

    # And the reverse: a full default cache does not throttle a download aimed elsewhere.
    monkeypatch.setenv("HF_HUB_CACHE", str(tight / "hub"))
    applied: dict = {}
    tuning.apply_xet_env(applied, cache_dir = str(roomy / "hub"))
    assert int(applied["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"]) == 64 * GB
    # No cache_dir: unchanged behaviour, the environment still decides.
    assert int(tuning.xet_env_overrides()["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"]) == 1 * GB


def test_a_process_already_sized_at_import_still_resizes_for_its_destination(monkeypatch, tmp_path):
    """``apply_xet_env`` is setdefault and ``unsloth_zoo`` sizes itself at import, so by the time a
    download names its ``cache_dir`` every sizing key is already in the environment and a second
    apply writes nothing: the download would run on the global cache's numbers. ``resize_for_cache_dir``
    drops what we seeded so the recompute lands, and leaves anything we did not write alone."""
    import os as _os

    roomy = tmp_path / "roomy"
    tight = tmp_path / "tight"
    for directory in (roomy, tight):
        directory.mkdir()
    _two_volumes(monkeypatch, str(roomy), str(tight))

    fake_env = {"HF_HUB_CACHE": str(roomy / "hub")}
    monkeypatch.setattr(_os, "environ", fake_env)
    monkeypatch.setattr(tuning, "_SEEDED_INTO_ENVIRON", {})

    tuning.apply_xet_env()  # what import does: sizes against the roomy default cache
    assert int(fake_env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"]) == 64 * GB

    # A second apply is a no-op on the seeded keys, which is the bug the resize exists to fix.
    assert "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT" not in tuning.apply_xet_env(
        dict(fake_env), cache_dir = tight / "hub",
    )

    written = tuning.resize_for_cache_dir(dict(fake_env), tight / "hub")
    assert int(written["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"]) == 1 * GB

    # A value we never wrote is not ours to recompute.
    user_env = dict(fake_env)
    user_env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"] = "16000000000"
    assert "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT" not in tuning.resize_for_cache_dir(
        user_env, tight / "hub",
    )
    assert user_env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"] == "16000000000"


def test_a_throttle_asked_for_by_a_logged_429_survives_the_resize(monkeypatch, tmp_path):
    """``unsloth_zoo/__init__`` seeds the halved stream ceiling when the Xet logs show a 429. The
    resize drops what that call wrote, so recomputing on the default terms would hand the child the
    full ceiling exactly when the account asked for fewer streams."""
    import os as _os

    key = "HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY"
    roomy = tmp_path / "roomy"
    tight = tmp_path / "tight"
    for directory in (roomy, tight):
        directory.mkdir()
    _two_volumes(monkeypatch, str(roomy), str(tight))
    # Pin the cores too: on a 1-2 CPU runner the ceiling is already the floor of 4, so halving it
    # would be indistinguishable from not halving it and the test would prove nothing.
    monkeypatch.setattr(_os, "sched_getaffinity", lambda _pid: set(range(16)))
    monkeypatch.setattr(tuning, "cgroup_cpu_limit", lambda: None)

    fake_env = {"HF_HUB_CACHE": str(roomy / "hub")}
    monkeypatch.setattr(_os, "environ", fake_env)
    monkeypatch.setattr(tuning, "_SEEDED_INTO_ENVIRON", {})
    monkeypatch.setattr(tuning, "_SEEDED_THROTTLED", False, raising = False)

    tuning.apply_xet_env(throttled = True)  # what import does after finding a 429
    throttled_streams = int(fake_env[key])
    assert throttled_streams == max(4, int(tuning.xet_env_overrides()[key]) // 2)

    resized = tuning.resize_for_cache_dir(dict(fake_env), roomy / "hub")
    assert int(resized[key]) == throttled_streams, "the 429 reduction has to reach the downloader"
    # Still overridable, for a caller that knows the throttle has lifted.
    assert int(tuning.resize_for_cache_dir(dict(fake_env), roomy / "hub", throttled = False)[key]) \
        > throttled_streams
