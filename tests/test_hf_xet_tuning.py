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


def _profile(ram_gb: float, cpus: int = 8) -> tuning.SystemProfile:
    return tuning.SystemProfile(
        total_ram_bytes = int(ram_gb * GB),
        available_ram_bytes = int(ram_gb * GB),
        cpu_count = cpus,
        ram_source = "test",
        cpu_source = "test",
    )


@pytest.mark.parametrize(
    "ram_gb, limit, size, perfile, files",
    [
        # limit is raised to the worst case (size + files*perfile) where that exceeds the tier's
        # headline figure, so the three numbers describe one budget.
        (8, 1024 * MB, 512 * MB, 128 * MB, 4),
        (11.9, 1024 * MB, 512 * MB, 128 * MB, 4),
        (16, 2 * GB, 768 * MB, 192 * MB, 6),
        (32, 4 * GB, 1 * GB, 256 * MB, 8),
        (2048, 4 * GB, 1 * GB, 256 * MB, 8),  # a huge host is still capped
    ],
)
def test_tier_table(ram_gb, limit, size, perfile, files):
    env = tuning.xet_env_overrides(_profile(ram_gb))
    assert int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"]) == limit
    assert int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE"]) == size
    assert int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE"]) == perfile
    assert int(env["HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"]) == files


def test_worst_case_buffer_stays_under_the_limit():
    """size + files*perfile is what hf_xet can hold, so it must not exceed the tier cap. The stock
    defaults (2GB + 8*512MB, capped at 8GB) are exactly how an 8GB spike happens."""
    for ram_gb in (4, 8, 16, 32, 64, 512):
        env = tuning.xet_env_overrides(_profile(ram_gb))
        worst = (
            int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE"])
            + int(env["HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"])
            * int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE"])
        )
        limit = int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"])
        assert worst <= limit, f"{ram_gb}GB: worst case {worst} exceeds limit {limit}"
        assert limit <= 4 * GB  # never the stock 8GB, let alone high-performance's 64GB


def test_unknown_ram_picks_the_smallest_tier():
    """Guessing low costs throughput; guessing high costs an OOM."""
    env = tuning.xet_env_overrides(_profile(0))
    assert int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"]) == 1024 * MB


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
    assert int(large["HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY"]) == 64


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
    assert env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE"] == str(192 * MB)


def test_apply_clears_an_inherited_high_performance_flag():
    """The one deliberate exception to setdefault: an inherited "1" would void every cap."""
    env = {"HF_XET_HIGH_PERFORMANCE": "1"}
    tuning.apply_xet_env(env, profile = _profile(16))
    assert env["HF_XET_HIGH_PERFORMANCE"] == "0"


def test_user_can_opt_back_into_high_performance(monkeypatch):
    monkeypatch.setenv("UNSLOTH_XET_ALLOW_HIGH_PERFORMANCE", "1")
    env = {"HF_XET_HIGH_PERFORMANCE": "1"}
    written = tuning.apply_xet_env(env, profile = _profile(16))
    assert env["HF_XET_HIGH_PERFORMANCE"] == "1"
    assert "HF_XET_HIGH_PERFORMANCE" not in written


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
    # Smallest tier and only 2 files in flight, so the 1GB limit is already above the worst case
    # (512MB + 2*128MB) and stands unchanged.
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
