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
        # 0 GB means "not measured"; a REAL zero is a full disk, gets a source, and branches apart.
        disk_source = "test" if free_disk_gb else "unknown",
    )


@pytest.mark.parametrize(
    "ram_gb, cpus, limit, size, perfile, files, streams",
    [
        # Budget is an eighth of RAM and per-file buffer a thirty-second; the shared buffer reaches for the 2 GB
        # default, and a quarter of it measured 0.73x on 8 GB.
        (8, 4, 1 * GB, 500 * MB, 128 * MB, 3, 8),
        # No cliffs: 12 GB sits between the old table's 8 and 16 GB rows and gets a budget to match.
        (12, 4, 1500 * MB, 750 * MB, 128 * MB, 4, 8),
        (16, 8, 2 * GB, 1 * GB, 128 * MB, 7, 16),
        # 32 GB is the first machine that can hold xet-core's stock buffer outright.
        (32, 16, 4 * GB, 2 * GB, 128 * MB, 15, 32),
        # A budget proportional to RAM: the old flat 4 GB row cost a 2 TB server 30%.
        (128, 64, 16 * GB, 4 * GB, 500 * MB, 24, 124),
        (2048, 192, 64 * GB, 16 * GB, 2000 * MB, 24, 124),
        # Cores bound concurrency independently of RAM: 8 cores on a 2 TB box still gets 8 files.
        (2048, 8, 64 * GB, 16 * GB, 2000 * MB, 8, 16),
        # The budget bounds concurrency too: a 64-core container with 8 GB cannot use 64 file buffers from a 1 GB budget, worse than no tuning on a thin link.
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
    """Scales buffer sizing to the with_high_performance() numbers by machine size without setting
    HF_XET_HIGH_PERFORMANCE, keeping the bound on small hosts; concurrent files are not followed, since the preset
    100 x 2 GB per-file buffer is 200 GB against its 64 GB limit."""
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
    """The budget is also capped by free disk space, not just host RAM: buffering 64 GB of a download that fails on
    ENOSPC helps nobody, and free space is the only signal we have."""
    roomy = tuning.xet_env_overrides(_profile(2048, cpus = 192, free_disk_gb = 8000))
    tight = tuning.xet_env_overrides(_profile(2048, cpus = 192, free_disk_gb = 40))
    assert int(roomy["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"]) == 64 * GB
    assert int(tight["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"]) == 10 * GB
    # Never below the floor: this download may be the one that frees space (a resume, say).
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
    """A cache moved by XDG_CACHE_HOME or the legacy HUGGINGFACE_HUB_CACHE can sit on another filesystem than ~, so free space must be measured on the cache root."""
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

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "explicit"))
    assert tuning.hf_cache_root() == tmp_path / "explicit"


def test_worst_case_buffer_stays_under_the_limit():
    """size + files*perfile is what hf_xet can hold and must stay under the budget; the stock defaults (2GB + 8*512MB, capped at 8GB) are how an 8GB spike happens."""
    for ram_gb, cpus in ((4, 2), (8, 8), (8, 96), (16, 8), (32, 64), (64, 16), (512, 192), (2048, 192)):
        env = tuning.xet_env_overrides(_profile(ram_gb, cpus))
        worst = (
            int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE"])
            + int(env["HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"])
            * int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE"])
        )
        limit = int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"])
        assert worst <= limit, f"{ram_gb}GB: worst case {worst} exceeds limit {limit}"
        # The bound is proportional, not absolute: the worst case stays under a third of RAM, so the buffer cannot be what OOMs a box.
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
    # Two streams per core, bounded by the 64 MiB xorb slots and file buffers the budget affords.
    assert int(large["HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY"]) == 59
    assert int(large["HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"]) <= 24
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
    assert env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE"] == str(128 * MB)


def test_a_user_set_high_performance_flag_is_left_alone():
    """A user-set HF_XET_HIGH_PERFORMANCE (or HF_XET_HP) is left alone, since enabling it is deliberate; clearing it on
    import once cost a large throughput loss, and setdefault applies."""
    for var in ("HF_XET_HIGH_PERFORMANCE", "HF_XET_HP"):
        env = {var: "1"}
        written = tuning.apply_xet_env(env, profile = _profile(16))
        assert env[var] == "1", var
        assert var not in written, var


def test_high_performance_also_stands_our_sizing_down():
    """With HF_XET_HIGH_PERFORMANCE set, our caps must also be dropped: xet-core voids the LIMIT but still honours the
    smaller per-file and concurrency caps, so the result is worse than either."""
    env = {"HF_XET_HIGH_PERFORMANCE": "1"}
    tuning.apply_xet_env(env, profile = _profile(16))
    for key in tuning._CAPS_VOIDED_BY_HIGH_PERFORMANCE:
        assert key not in env, key
    assert env["HF_XET_CHUNK_CACHE_SIZE_BYTES"] == "0"


def test_force_caps_still_overrides_high_performance(monkeypatch):
    """UNSLOTH_XET_FORCE_CAPS is the escape hatch: bounding a machine requires the flag off, so it is the one case that overwrites HF_XET_HIGH_PERFORMANCE to 0."""
    monkeypatch.setenv("UNSLOTH_XET_FORCE_CAPS", "1")
    env = {"HF_XET_HIGH_PERFORMANCE": "1"}
    written = tuning.apply_xet_env(env, profile = _profile(16))
    assert env["HF_XET_HIGH_PERFORMANCE"] == "0"
    assert written["HF_XET_HIGH_PERFORMANCE"] == "0"
    assert env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"] == str(2 * GB)


def test_a_large_machine_is_not_held_to_a_laptops_buffer():
    """The budget must grow monotonically with RAM, without a cliff: a 24 GB flat-line gave a 2 TB server a laptop's 4
    GB buffer, below xet-core's own defaults, which gate in-flight data."""
    seen = []
    for ram_gb in (8, 16, 32, 128, 512, 2048):
        env = tuning.xet_env_overrides(_profile(ram_gb, cpus = 64))
        seen.append((ram_gb, int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"])))
    for (small_ram, small), (big_ram, big) in zip(seen, seen[1:]):
        assert big >= small, f"{big_ram}GB got {big} but {small_ram}GB got {small}"
    assert seen[-1][1] > seen[2][1], f"a 2TB box still capped like a 32GB one: {seen}"
    assert seen[0][1] == 1 * GB


def test_cgroup_limit_beats_the_host_total(monkeypatch, tmp_path):
    """Inside a container psutil reports the HOST's RAM, so the cgroup memory limit must win over it."""
    monkeypatch.setattr(tuning, "_psutil_memory", lambda: (512 * GB, 500 * GB))
    monkeypatch.setattr(tuning, "cgroup_memory_limit", lambda: 8 * GB)
    monkeypatch.setattr(tuning, "cgroup_cpu_limit", lambda: 2.0)
    profile = tuning.system_profile()
    assert profile.total_ram_bytes == 8 * GB
    assert profile.ram_source == "cgroup"
    assert profile.cpu_count == 2
    env = tuning.xet_env_overrides(profile)
    # An eighth of 8 GB is the 1 GB floor and the 2-core quota allows 2 files: 256MB + 2*128MB fits.
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
    """The v1 controller root is only right inside a runc container; under Slurm it shows the unlimited sentinel, so
    read the nested cgroup and split combined lists like cpu,cpuacct rather than matching them whole."""
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
    """Under systemd hybrid mode v1 controller lines share the same cgroup file, so the 0:: v2 line must be found by scanning rather than reading line 1."""
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
    """A fractional cgroup CPU quota must bind (Kubernetes cpu 500m reads as cpu.max 50000 100000, i.e. 0.5); requiring
    >= 1 fell back to host cores, sizing streams for the node."""
    monkeypatch.setattr(tuning, "cgroup_cpu_limit", lambda: 0.5)
    profile = tuning.system_profile()
    assert profile.cpu_count == 1
    assert profile.cpu_source == "cgroup"

    env = tuning.xet_env_overrides(profile)
    assert int(env["HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY"]) <= 4


def test_fail_fast_timeouts_are_opt_out():
    """Fail-fast Xet timeouts belong only in the supervised download child: xet-core reads its config once per process,
    so they would also hit uploads and direct hub downloads."""
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


# Real machines Unsloth runs on, smallest first; pinned by name since regressions on a 2-core Colab or 0.5-core pod cannot be reproduced otherwise.
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
    """One sizing rule applies to every device: generosity on big hosts is only defensible if the same arithmetic stays
    conservative on small ones, where OOMs and stalls go unnoticed."""
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
    """Small devices (2 GB VM, fractional-core pod) get two files, four streams and a buffer capped at a sixth of RAM,
    since xet-core's default would take a 2 GB machine whole."""
    for name, ram_gb, cpus, disk_gb in DEVICES[:2]:
        env = tuning.xet_env_overrides(_profile(ram_gb, cpus, disk_gb))
        assert int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"]) == 1 * GB, name
        assert int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE"]) <= ram_gb * GB / 6, name
        assert int(env["HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"]) == 2, name
        assert int(env["HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY"]) == 4, name


def test_only_the_big_hosts_get_the_big_numbers():
    """Only devices that can afford xet-core's own high-performance sizing receive it; that is the point of the change."""
    big = {
        name for name, ram_gb, cpus, disk_gb in DEVICES
        if int(tuning.xet_env_overrides(_profile(ram_gb, cpus, disk_gb))
               ["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"]) > 8 * GB
    }
    assert big == {"A100 node", "8xH100 node"}


# Every HF_XET_* name xet-core 1.5.x reads (filtered to the groups we touch), regenerated by grepping its xet_runtime
# *.rs; HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY was never real, a no-op; the real one is
# HF_XET_RECONSTRUCTION_USE_VECTORED_WRITE, default true.
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
    """The prefetch floor never undercuts xet-core's default unless the whole budget is below 1 GB; sizing below the
    default for no reason only makes downloads slower."""
    for ram_gb, cpus in ((8, 4), (16, 8), (32, 16), (64, 32), (2048, 192)):
        env = tuning.xet_env_overrides(_profile(ram_gb, cpus))
        prefetch = int(env["HF_XET_RECONSTRUCTION_MIN_PREFETCH_BUFFER"])
        limit = int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"])
        assert prefetch <= tuning._STOCK_PREFETCH_BUFFER, f"{ram_gb}GB raised it to {prefetch}"
        assert prefetch <= limit, f"{ram_gb}GB: {prefetch} prefetch inside a {limit} budget"
        if limit >= 4 * GB:
            assert prefetch == tuning._STOCK_PREFETCH_BUFFER, f"{ram_gb}GB undercut it: {prefetch}"


def test_no_module_sets_a_variable_xet_core_does_not_read():
    """Scans the whole package, since a dead variable was set from unsloth_zoo/__init__.py at import time and a check in this module alone would miss it."""
    import pathlib
    import re

    light = pathlib.Path(tuning.__file__).parent
    unknown: dict[str, set] = {}
    for root in (light, light.parent / "unsloth_zoo"):
        for path in root.rglob("*.py"):
            # Only quoted names: this skips prose ("any explicit HF_XET_RECONSTRUCTION_* cap") unparsed.
            for name in re.findall(r"[\"'](HF_XET_[A-Z_]+)[\"']", path.read_text(errors = "ignore")):
                if name not in XET_CORE_VARS:
                    unknown.setdefault(name, set()).add(path.relative_to(root.parent).as_posix())
    assert not unknown, f"names xet-core does not read: { {k: sorted(v) for k, v in unknown.items()} }"


@pytest.mark.parametrize("ram_gb, cpus", [(8, 8), (12.7, 2), (16, 4), (32, 16), (64, 32), (2048, 192)])
def test_a_machine_that_can_hold_xet_cores_own_buffer_gets_it(ram_gb, cpus):
    """Sizing exists to cap machines that need it; giving a machine less buffer than it would get by default is just a
    slower download, not a cap (the small-device regression)."""
    env = tuning.xet_env_overrides(_profile(ram_gb, cpus))
    size = int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE"])
    limit = int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"])
    stock = tuning._STOCK_BUFFER_SIZE
    # Only half the budget and a sixth of RAM may hold the buffer under stock, and only as far as they reach; anything smaller is a choice, not a constraint.
    assert size >= min(stock, limit // 2, max(256 * MB, ram_gb * GB / 6)), f"{ram_gb}GB got {size}"
    if limit // 2 >= stock and ram_gb * GB / 6 >= stock:
        assert size >= stock, f"{ram_gb}GB undercut stock: {size}"


def test_a_small_container_with_many_cores_still_fits_a_third_of_its_ram():
    """Small containers with many cores must still fit their RAM budget: cores alone setting the file count put 973 MB
    in flight on a 2 GB box, so the proportional promise must win."""
    for cpus in (2, 8, 64):
        env = tuning.xet_env_overrides(_profile(2, cpus))
        worst = (
            int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE"])
            + int(env["HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"])
            * int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE"])
        )
        assert worst <= 2 * GB / 3, f"2GB/{cpus} cores: {worst} in flight"


def test_the_flag_is_read_the_way_xet_core_reads_it():
    """xet-core reads HF_XET_HP only when HF_XET_HIGH_PERFORMANCE is unset (configuration_utils.rs
    get_high_performance_flag), so an explicit 0 masks it; standing down over it leaves stock constants."""
    masked = {"HF_XET_HIGH_PERFORMANCE": "0", "HF_XET_HP": "1"}
    tuning.apply_xet_env(masked, profile = _profile(16))
    assert "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT" in masked, "sizing stood down for nothing"

    alias_only = {"HF_XET_HP": "1"}
    tuning.apply_xet_env(alias_only, profile = _profile(16))
    assert "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT" not in alias_only


def test_a_full_disk_is_not_read_as_an_unmeasured_one():
    """Zero free bytes (shutil.disk_usage) is a real full disk, not the unmeasured sentinel; else a full cache kept 64 GB while 1 byte free hit the floor."""
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
    """cache_dir= beats every cache env var in huggingface_hub (file_download.py), so size from that volume, not the global default cache."""
    roomy = tmp_path / "roomy"
    tight = tmp_path / "tight"
    for directory in (roomy, tight):
        directory.mkdir()
    _two_volumes(monkeypatch, str(roomy), str(tight))

    monkeypatch.setenv("HF_HUB_CACHE", str(roomy / "hub"))
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
    """apply_xet_env is setdefault and unsloth_zoo sizes at import, so a cache_dir apply would write nothing; resize_for_cache_dir drops only keys it seeded."""
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
    """A halved stream ceiling seeded by unsloth_zoo/__init__ after a logged 429 must survive the cache_dir resize, not recompute to the full ceiling."""
    import os as _os

    key = "HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY"
    roomy = tmp_path / "roomy"
    tight = tmp_path / "tight"
    for directory in (roomy, tight):
        directory.mkdir()
    _two_volumes(monkeypatch, str(roomy), str(tight))
    # Pin the cores too: on a 1-2 CPU runner the ceiling is already the floor of 4, so halving it would look the same as not halving it.
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
    assert int(tuning.resize_for_cache_dir(dict(fake_env), roomy / "hub", throttled = False)[key]) \
        > throttled_streams


# A fresh cloud VM has no environment and no Xet logs to learn from; Colab and Kaggle look like this on every run, so the old heuristic failed there forever.
CLOUD_VMS = [
    # name,                RAM GB, cpus, free disk GB
    ("Colab standard",       13.6,    2,   78),
    ("Colab high-RAM",       51.0,    8,  225),
    ("Kaggle T4 x2",         33.7,    4, 1100),
    ("Kaggle CPU",           30.0,    4,   20),
    ("SageMaker Studio Lab", 16.0,    4,   15),
    ("Paperspace free",      30.0,    8,   50),
]

# xet-core's high-performance preset (xet_runtime/src/config/xet_config.rs) is applied AFTER the environment, so it voids whatever we set.
_HIGH_PERFORMANCE_CEILING = 64 * GB


@pytest.mark.parametrize("name, ram_gb, cpus, disk_gb", CLOUD_VMS,
                         ids = [d[0] for d in CLOUD_VMS])
def test_a_fresh_cloud_vm_is_never_put_into_high_performance_mode(
        name, ram_gb, cpus, disk_gb, monkeypatch):
    """2026.6.7 regression: no prior 429 in Xet logs enabled high performance, and fresh VMs always have empty logs
    (29.5GB download OOM-killed on Colab); the flag matters since with_high_performance() discards our numbers."""
    for var in (*tuning.XET_HIGH_PERFORMANCE_VARS, "UNSLOTH_XET_FORCE_CAPS"):
        monkeypatch.delenv(var, raising = False)
    env: dict = {}
    tuning.apply_xet_env(env, profile = _profile(ram_gb, cpus, disk_gb))

    for var in tuning.XET_HIGH_PERFORMANCE_VARS:
        assert not tuning._is_true(env.get(var)), (
            f"{name}: {var}={env.get(var)!r} hands this VM a "
            f"{_HIGH_PERFORMANCE_CEILING // GB}GB ceiling on {ram_gb}GB of RAM"
        )


@pytest.mark.parametrize("name, ram_gb, cpus, disk_gb", CLOUD_VMS,
                         ids = [d[0] for d in CLOUD_VMS])
def test_a_fresh_cloud_vm_keeps_its_buffers_inside_its_own_ram(
        name, ram_gb, cpus, disk_gb, monkeypatch):
    """What hf_xet can hold at once must stay a modest share of RAM, a quarter at most; Colab peak RSS fell from 11.3GB (83%) to 1.3GB (10%) once sized."""
    for var in (*tuning.XET_HIGH_PERFORMANCE_VARS, "UNSLOTH_XET_FORCE_CAPS"):
        monkeypatch.delenv(var, raising = False)
    env: dict = {}
    tuning.apply_xet_env(env, profile = _profile(ram_gb, cpus, disk_gb))

    in_flight = (int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE"])
                 + int(env["HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"])
                 * int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE"]))
    budget = ram_gb * GB / 4
    assert in_flight <= budget, (
        f"{name}: {in_flight / GB:.2f}GB in flight on a {ram_gb}GB VM "
        f"({100 * in_flight / (ram_gb * GB):.0f}% of RAM, limit 25%)"
    )


def test_an_empty_xet_log_directory_is_not_read_as_permission_to_go_fast():
    """No 429 in the logs only means nothing has been tried yet, not that high performance is safe; absence of evidence must take the conservative branch."""
    quiet: dict = {}
    tuning.apply_xet_env(quiet, profile = _profile(13.6, 2, 78))
    throttled: dict = {}
    tuning.apply_xet_env(throttled, profile = _profile(13.6, 2, 78), throttled = True)

    for env, label in ((quiet, "no 429s logged"), (throttled, "a 429 was logged")):
        for var in tuning.XET_HIGH_PERFORMANCE_VARS:
            assert not tuning._is_true(env.get(var)), f"{label}: {var} enabled"
    # A logged 429 may fetch less hard, but a quiet log never fetches HARDER than one.
    assert (int(throttled["HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY"])
            <= int(quiet["HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY"]))
