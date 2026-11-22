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

"""CPU thread fitting: physical cores, cgroup quota, launcher OMP_NUM_THREADS=1."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


_ROOT = Path(__file__).resolve().parents[1]


_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "UNSLOTH_CPU_THREADS", "TORCHELASTIC_RUN_ID",
               "LOCAL_WORLD_SIZE", "ACCELERATE_MIXED_PRECISION", "MPI_LOCALNRANKS", "OMPI_COMM_WORLD_LOCAL_SIZE")


def _fit_in_subprocess(extra, cores = None):
    """fit_cpu_threads() in a fresh interpreter (torch reads OMP_NUM_THREADS once), with cpu_compute_cores pinned."""
    env = {k: v for k, v in os.environ.items() if k not in _THREAD_ENV}
    env.update(extra)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_ROOT), env.get("PYTHONPATH")]))
    code = (
        "import json, torch\n"
        "from unsloth_zoo.cpu import runtime\n"
        f"cores = {cores!r}\n"
        "if cores is not None: runtime.cpu_compute_cores = lambda: cores\n"
        "before = torch.get_num_threads()\n"
        "first = runtime.fit_cpu_threads(); second = runtime.fit_cpu_threads()\n"
        "print(json.dumps([before, first[0], first[1], list(first) == list(second), torch.get_num_threads()]))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], env = env, capture_output = True, text = True, timeout = 300)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


# "user": the count torch derived from the user's OMP_NUM_THREADS / MKL_NUM_THREADS is kept as is. Newer torch builds
# cap those to the CPUs it sees (a 2-vCPU CI runner reports 2 for OMP_NUM_THREADS=8), so compare with torch's own count.
_USER = "user"


@pytest.mark.parametrize("extra, cores, expected", [
    ({"OMP_NUM_THREADS": "8"}, 3, _USER),  # a user's OMP_NUM_THREADS is kept
    ({"MKL_NUM_THREADS": "3"}, 2, _USER),  # so is MKL_NUM_THREADS
    ({"UNSLOTH_CPU_THREADS": "5", "OMP_NUM_THREADS": "3"}, 2, 5),  # UNSLOTH_CPU_THREADS wins
    ({"OMP_NUM_THREADS": "1"}, 4, 1),  # a plain OMP_NUM_THREADS=1 is the user's
    # torchrun with 2 local ranks exported OMP_NUM_THREADS=1: split the cores instead.
    ({"OMP_NUM_THREADS": "1", "TORCHELASTIC_RUN_ID": "r", "LOCAL_WORLD_SIZE": "2"}, 8, 4),
    ({"OMP_NUM_THREADS": "1", "TORCHELASTIC_RUN_ID": "r", "LOCAL_WORLD_SIZE": "2", "UNSLOTH_CPU_THREADS": "1"}, 8, 1),
    ({"OMP_NUM_THREADS": "4", "TORCHELASTIC_RUN_ID": "r", "LOCAL_WORLD_SIZE": "2"}, 8, _USER),
    ({"OMP_NUM_THREADS": "1", "ACCELERATE_MIXED_PRECISION": "no"}, 6, 6),  # accelerate launch's 1
    # A launcher's OMP_NUM_THREADS=1 next to the user's MKL_NUM_THREADS: the MKL setting stays authoritative.
    ({"MKL_NUM_THREADS": "2", "OMP_NUM_THREADS": "1", "TORCHELASTIC_RUN_ID": "r", "LOCAL_WORLD_SIZE": "2"}, 16, _USER),
])
def test_fit_cpu_threads_env_cases(extra, cores, expected):
    before, fitted, reason, idempotent, live = _fit_in_subprocess(extra, cores)
    if expected == _USER:
        assert fitted == before == live and reason == "OMP_NUM_THREADS / MKL_NUM_THREADS", (extra, before, fitted, reason)
    else:
        assert fitted == expected == live, (extra, before, fitted, reason)
    assert idempotent


def test_fit_cpu_threads_default_drops_smt_siblings_never_raises():
    before, fitted, reason, _, _ = _fit_in_subprocess({}, 1)
    assert fitted == 1 and (before == 1 or reason == "physical cores")
    big = 10 ** 6
    before, fitted, reason, _, _ = _fit_in_subprocess({}, big)
    assert fitted == before  # more cores than torch's default: left alone


def test_fit_cpu_threads_splits_across_local_ranks_without_launcher_marker():
    _, fitted, reason, _, _ = _fit_in_subprocess({"LOCAL_WORLD_SIZE": "2"}, 2)
    assert fitted == 1 and "local ranks" in reason


def test_physical_cores_from_sysfs(tmp_path):
    from unsloth_zoo.cpu.runtime import _linux_physical_cores
    # cpu0/cpu2 are SMT siblings of core 0, cpu1/cpu3 of core 1; cpu4 is core 0 on package 1.
    layout = {0: ("0", "0"), 1: ("0", "1"), 2: ("0", "0"), 3: ("0", "1"), 4: ("1", "0")}
    for cpu, (package, core) in layout.items():
        topo = tmp_path / f"cpu{cpu}" / "topology"
        topo.mkdir(parents = True)
        (topo / "physical_package_id").write_text(package + "\n")
        (topo / "core_id").write_text(core + "\n")
    assert _linux_physical_cores([0, 1, 2, 3, 4], root = str(tmp_path)) == 3
    assert _linux_physical_cores([0, 2], root = str(tmp_path)) == 1
    assert _linux_physical_cores([0, 9], root = str(tmp_path)) is None  # unreadable: caller keeps the CPU count


def test_cgroup_cpu_quota(tmp_path):
    from unsloth_zoo.cpu.runtime import _cgroup_cpu_limit
    proc = tmp_path / "cgroup"
    proc.write_text("0::/a/b\n")
    (tmp_path / "a" / "b").mkdir(parents = True)
    (tmp_path / "a" / "b" / "cpu.max").write_text("max 100000\n")
    assert _cgroup_cpu_limit(str(proc), str(tmp_path)) is None
    (tmp_path / "a" / "cpu.max").write_text("250000 100000\n")  # a parent's 2.5 CPUs bound the child
    assert _cgroup_cpu_limit(str(proc), str(tmp_path)) == 3
    v1 = tmp_path / "v1"
    (v1 / "cpu,cpuacct").mkdir(parents = True)
    (v1 / "cpu,cpuacct" / "cpu.cfs_quota_us").write_text("400000\n")
    (v1 / "cpu,cpuacct" / "cpu.cfs_period_us").write_text("100000\n")
    proc.write_text("4:cpu,cpuacct:/x\n")
    assert _cgroup_cpu_limit(str(proc), str(v1)) == 4


def test_cgroup_v1_quota_follows_the_process_path(tmp_path):
    """cgroup v1: the quota on the process's own cgroup (4:cpu,cpuacct:/jobs/train) applies even when the controller
    root is unlimited; ancestors bound it too, and a path missing under the mount falls back to the root."""
    from unsloth_zoo.cpu.runtime import _cgroup_cpu_limit
    proc = tmp_path / "cgroup"
    proc.write_text("12:memory:/jobs/train\n4:cpu,cpuacct:/jobs/train\n")
    mount = tmp_path / "v1" / "cpu,cpuacct"
    leaf = mount / "jobs" / "train"
    leaf.mkdir(parents = True)
    for d, quota in ((mount, "-1"), (mount / "jobs", "-1"), (leaf, "200000")):
        (d / "cpu.cfs_quota_us").write_text(quota + "\n")
        (d / "cpu.cfs_period_us").write_text("100000\n")
    root = str(tmp_path / "v1")
    assert _cgroup_cpu_limit(str(proc), root) == 2
    (leaf / "cpu.cfs_quota_us").write_text("-1\n")
    (mount / "jobs" / "cpu.cfs_quota_us").write_text("150000\n")
    assert _cgroup_cpu_limit(str(proc), root) == 2  # parent's 1.5 CPUs, rounded up
    (mount / "jobs" / "cpu.cfs_quota_us").write_text("-1\n")
    assert _cgroup_cpu_limit(str(proc), root) is None
    proc.write_text("4:cpu,cpuacct:/host/only/path\n")
    (mount / "cpu.cfs_quota_us").write_text("300000\n")
    assert _cgroup_cpu_limit(str(proc), root) == 3


def test_cpu_compute_cores_within_affinity():
    from unsloth_zoo.cpu.runtime import cpu_compute_cores
    cores = cpu_compute_cores()
    if not sys.platform.startswith("linux"):
        assert cores is None
        return
    assert 1 <= cores <= len(os.sched_getaffinity(0))


def test_fit_threads_names_the_cgroup_quota(monkeypatch):
    from unsloth_zoo.cpu import runtime
    if not sys.platform.startswith("linux"):
        pytest.skip("Linux only")
    for name in _THREAD_ENV:
        monkeypatch.delenv(name, raising = False)
    before = torch.get_num_threads()
    monkeypatch.setattr(runtime, "_THREADS_FITTED", None)
    monkeypatch.setattr(runtime, "_cgroup_cpu_limit", lambda *a, **k: 1)
    try:
        threads, reason = runtime.fit_cpu_threads()
        assert threads == 1 and (reason == "cgroup CPU quota" or before == 1)
    finally:
        torch.set_num_threads(before)
        runtime._THREADS_FITTED = None


def test_user_thread_env_kept_when_torch_caps_it(monkeypatch):
    """Spoof of a small CI runner: torch already capped OMP_NUM_THREADS=8 to 2; the fit keeps torch's 2 and never
    swaps in its own count, whatever cores it would pick."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_zoo_cpu_runtime_spoof", _ROOT / "unsloth_zoo" / "cpu" / "runtime.py")
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    for name in _THREAD_ENV:
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    calls = []
    monkeypatch.setattr(torch, "get_num_threads", lambda: 2)
    monkeypatch.setattr(torch, "set_num_threads", lambda n: calls.append(n))
    monkeypatch.setattr(runtime, "cpu_compute_cores", lambda: 1)
    assert runtime.fit_cpu_threads() == (2, "OMP_NUM_THREADS / MKL_NUM_THREADS")
    assert calls == []


def test_physical_cores_unknown_topology_falls_back(tmp_path):
    """sysfs reports unknown topology as physical_package_id -1, core_id 0: not one core, so None (count the CPUs)."""
    from unsloth_zoo.cpu.runtime import _linux_physical_cores
    for cpu in range(8):
        topo = tmp_path / f"cpu{cpu}" / "topology"
        topo.mkdir(parents = True)
        (topo / "physical_package_id").write_text("-1\n")
        (topo / "core_id").write_text("0\n")
    assert _linux_physical_cores(list(range(8)), root = str(tmp_path)) is None


def _fit_with(monkeypatch, cores, ranks, mine, parent, quota = None):
    from unsloth_zoo.cpu import runtime
    for name in _THREAD_ENV:
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", str(ranks))
    monkeypatch.setattr(runtime, "_THREADS_FITTED", None)
    monkeypatch.setattr(runtime, "cpu_compute_cores", lambda: cores)
    monkeypatch.setattr(runtime, "_cgroup_cpu_limit", lambda *a, **k: quota)
    monkeypatch.setattr(runtime.sys, "platform", "linux")
    monkeypatch.setattr(runtime.os, "sched_getaffinity", lambda pid: set(mine if pid == 0 else parent), raising = False)
    threads = [64]
    monkeypatch.setattr(torch, "get_num_threads", lambda: threads[0])
    monkeypatch.setattr(torch, "set_num_threads", lambda n: threads.__setitem__(0, n))
    try:
        return runtime.fit_cpu_threads()[0]
    finally:
        runtime._THREADS_FITTED = None


@pytest.mark.parametrize("mine, parent, quota, expected", [
    (range(4), range(8), None, 4),  # torchrun --numa-binding / mpirun --bind-to: each rank owns 4 cores
    (range(4), range(8), 4, 2),  # a cgroup quota is still shared by both ranks
    (range(4), range(4), None, 2),  # taskset -c 0-3 on the whole job: both ranks share the 4 cores
    (range(8), range(8), None, 2),  # 8 SMT CPUs = 4 cores shared by both ranks: split
])
def test_fit_threads_does_not_split_a_per_rank_affinity_again(monkeypatch, mine, parent, quota, expected):
    assert _fit_with(monkeypatch, 4, 2, mine, parent, quota) == expected
