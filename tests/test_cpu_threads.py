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


@pytest.mark.parametrize("extra, cores, expected", [
    # No user setting: one thread per physical core, never more than torch's default.
    ({"OMP_NUM_THREADS": "8"}, 3, 8),  # a user's OMP_NUM_THREADS is kept
    ({"MKL_NUM_THREADS": "3"}, 2, 3),  # so is MKL_NUM_THREADS
    ({"UNSLOTH_CPU_THREADS": "5", "OMP_NUM_THREADS": "3"}, 2, 5),  # UNSLOTH_CPU_THREADS wins
    ({"OMP_NUM_THREADS": "1"}, 4, 1),  # a plain OMP_NUM_THREADS=1 is the user's
    # torchrun with 2 local ranks exported OMP_NUM_THREADS=1: split the cores instead.
    ({"OMP_NUM_THREADS": "1", "TORCHELASTIC_RUN_ID": "r", "LOCAL_WORLD_SIZE": "2"}, 8, 4),
    ({"OMP_NUM_THREADS": "1", "TORCHELASTIC_RUN_ID": "r", "LOCAL_WORLD_SIZE": "2", "UNSLOTH_CPU_THREADS": "1"}, 8, 1),
    ({"OMP_NUM_THREADS": "4", "TORCHELASTIC_RUN_ID": "r", "LOCAL_WORLD_SIZE": "2"}, 8, 4),
    ({"OMP_NUM_THREADS": "1", "ACCELERATE_MIXED_PRECISION": "no"}, 6, 6),  # accelerate launch's 1
])
def test_fit_cpu_threads_env_cases(extra, cores, expected):
    before, fitted, reason, idempotent, live = _fit_in_subprocess(extra, cores)
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
