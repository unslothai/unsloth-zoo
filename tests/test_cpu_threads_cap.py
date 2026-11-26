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

"""The default CPU thread fit stops at 16 threads on many-core hosts; every explicit setting still wins."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "UNSLOTH_CPU_THREADS", "TORCHELASTIC_RUN_ID",
               "LOCAL_WORLD_SIZE", "ACCELERATE_MIXED_PRECISION", "MPI_LOCALNRANKS", "OMPI_COMM_WORLD_LOCAL_SIZE",
               "MV2_COMM_WORLD_LOCAL_SIZE")


def _fit(extra, cores, torch_threads):
    """fit_cpu_threads() in a fresh interpreter, with the core count and torch's starting pool size pinned."""
    env = {k: v for k, v in os.environ.items() if k not in _THREAD_ENV}
    env.update(extra)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_ROOT), env.get("PYTHONPATH")]))
    code = (
        "import json, os, torch\n"
        "from unsloth_zoo.cpu import runtime\n"
        f"runtime.cpu_compute_cores = lambda: {cores!r}\n"
        "runtime._cgroup_cpu_limit = lambda *a, **k: None\n"
        f"start = {torch_threads!r}\n"
        "if start is not None: torch.set_num_threads(start)\n"
        "before = torch.get_num_threads()\n"
        "threads, reason = runtime.fit_cpu_threads()\n"
        "print(json.dumps([before, threads, reason, torch.get_num_threads(), os.environ.get('OMP_NUM_THREADS')]))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], env = env, capture_output = True, text = True, timeout = 300)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("cores, start, expected", [
    (48, 48, 16),   # one 48-core NUMA node: 16, not 48
    (96, 96, 16),   # a whole 2-socket host
    (24, 24, 16),
    (17, 17, 16),
    (16, 16, 16),   # at the cap: unchanged
    (12, 24, 12),   # SMT host with 12 cores: one thread per core, as before
    (4, 8, 4),
    (2, 2, 2),
])
def test_default_fit_caps_at_16(cores, start, expected):
    before, threads, reason, live, omp = _fit({}, cores, start)
    assert before == start
    assert threads == live == expected, (cores, start, threads, reason)
    assert omp == str(expected)  # exported, so Accelerate's PartialState and workers keep it
    if cores > 16:
        assert "capped at 16" in reason


def test_cap_never_raises_a_smaller_torch_pool():
    before, threads, reason, live, _ = _fit({}, 48, 8)
    assert threads == live == 8 and reason == "default"


@pytest.mark.parametrize("extra, expected", [
    ({"UNSLOTH_CPU_THREADS": "40"}, 40),   # explicit count above the cap is honoured
])
def test_unsloth_cpu_threads_still_wins(extra, expected):
    _, threads, reason, live, _ = _fit(extra, 48, 48)
    assert threads == live == expected and reason == "UNSLOTH_CPU_THREADS"


def test_user_omp_env_kept_above_the_cap():
    before, threads, reason, live, omp = _fit({"OMP_NUM_THREADS": "32"}, 48, None)
    assert threads == live == before and reason == "OMP_NUM_THREADS / MKL_NUM_THREADS" and omp == "32"


def test_launcher_one_thread_undone_to_the_cap():
    # torchrun with 2 local ranks on 96 cores exported OMP_NUM_THREADS=1: 48 per rank, capped to 16.
    _, threads, reason, live, _ = _fit({"OMP_NUM_THREADS": "1", "TORCHELASTIC_RUN_ID": "r", "LOCAL_WORLD_SIZE": "2"},
                                       96, None)
    assert threads == live == 16 and reason == "launcher OMP_NUM_THREADS=1 undone"


def test_local_ranks_split_below_the_cap_unchanged():
    # 4 local ranks on 48 cores: 12 each, under the cap, so the split is what it was.
    _, threads, reason, live, _ = _fit({"LOCAL_WORLD_SIZE": "4"}, 48, 48)
    assert threads == live == 12 and "4 local ranks" in reason
