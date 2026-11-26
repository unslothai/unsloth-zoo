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

"""OMP_PROC_BIND / OMP_PLACES / GOMP_CPU_AFFINITY narrow the main thread to one place when torch loads; the CPU
thread fit still counts every core the process was given, not one."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from unsloth_zoo.cpu import runtime

_ROOT = Path(__file__).resolve().parents[1]
_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "UNSLOTH_CPU_THREADS", "TORCHELASTIC_RUN_ID", "LOCAL_WORLD_SIZE",
        "ACCELERATE_MIXED_PRECISION", "OMP_PROC_BIND", "OMP_PLACES", "GOMP_CPU_AFFINITY", "KMP_AFFINITY")


@pytest.mark.parametrize("env, expected", [
    ({}, False),
    ({"OMP_PROC_BIND": "close"}, True),
    ({"OMP_PROC_BIND": "TRUE"}, True),
    ({"OMP_PROC_BIND": "false"}, False),
    ({"OMP_PLACES": "cores"}, True),
    ({"GOMP_CPU_AFFINITY": "0-3"}, True),
    ({"KMP_AFFINITY": "granularity=fine,compact,1,0"}, True),
    ({"KMP_AFFINITY": "disabled"}, False),
    ({"OMP_PROC_BIND": " "}, False),
])
def test_binding_env_detection(monkeypatch, env, expected):
    for name in _ENV:
        monkeypatch.delenv(name, raising = False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert runtime._omp_binding_requested() is expected


def test_union_of_thread_masks_under_binding(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setenv("OMP_PROC_BIND", "close")
    masks = {0: {4, 100}, 11: {4, 100}, 12: {5, 101}, 13: {6, 102}, 14: {7, 103}}
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(masks[pid]), raising = False)
    real_listdir = os.listdir
    monkeypatch.setattr(os, "listdir", lambda p: ["11", "12", "13", "14"] if p == "/proc/self/task" else real_listdir(p))
    assert runtime._process_affinity() == {4, 5, 6, 7, 100, 101, 102, 103}


def test_no_binding_reads_main_thread_only(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising = False)
    calls = []
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: calls.append(pid) or {0, 1}, raising = False)
    assert runtime._process_affinity() == {0, 1}
    assert calls == [0]


def _fit_in_child(extra, start_threads = None):
    env = {k: v for k, v in os.environ.items() if k not in _ENV}
    env.update(extra)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_ROOT), env.get("PYTHONPATH")]))
    code = ("import json, torch\nfrom unsloth_zoo.cpu import runtime\n"
            + (f"torch.set_num_threads({start_threads})\n" if start_threads else "")
            + "print(json.dumps([runtime.cpu_compute_cores(), list(runtime.fit_cpu_threads()), torch.get_num_threads()]))\n")
    r = subprocess.run([sys.executable, "-c", code], env = env, capture_output = True, text = True, timeout = 600)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "the thread fit is Linux-only")
@pytest.mark.parametrize("bind", [
    {"OMP_PROC_BIND": "close", "OMP_PLACES": "cores"},
    {"OMP_PROC_BIND": "spread"},
    {"OMP_PROC_BIND": "true"},
    {"OMP_PROC_BIND": "close", "OMP_PLACES": "threads"},
])
def test_real_binding_keeps_core_count(bind):
    plain = _fit_in_child({})
    if plain[0] is None or plain[0] < 2:
        pytest.skip("needs at least 2 physical cores")
    bound = _fit_in_child(bind)
    assert bound[0] == plain[0]
    assert bound[1][0] == plain[1][0]
    assert bound[2] == plain[2] > 1


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "the thread fit is Linux-only")
def test_binding_narrowed_default_is_raised():
    # aarch64 torch wheels start at 1 thread under OMP_PROC_BIND (sized from the main thread's bound mask)
    plain = _fit_in_child({})
    if plain[0] is None or plain[0] < 2:
        pytest.skip("needs at least 2 physical cores")
    bound = _fit_in_child({"OMP_PROC_BIND": "close", "OMP_PLACES": "cores"}, start_threads = 1)
    assert bound[1] == [plain[1][0], "OpenMP binding narrowed torch's default"]
    assert bound[2] == plain[2]
    unbound = _fit_in_child({}, start_threads = 1)  # no binding: torch's smaller pool is kept, as before
    assert unbound[2] == 1


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "the thread fit is Linux-only")
def test_omp_places_cover_the_process_mask():
    env = {k: v for k, v in os.environ.items() if k not in _ENV}
    env.update({"OMP_PROC_BIND": "close", "OMP_PLACES": "threads"})
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_ROOT), env.get("PYTHONPATH")]))
    code = ("import json, os\nmask = sorted(os.sched_getaffinity(0))\nimport torch\n"
            "from unsloth_zoo.cpu import runtime\nprint(json.dumps([mask, sorted(runtime._omp_place_cpus())]))\n")
    r = subprocess.run([sys.executable, "-c", code], env = env, capture_output = True, text = True, timeout = 600)
    assert r.returncode == 0, r.stderr[-2000:]
    mask, places = json.loads(r.stdout.strip().splitlines()[-1])
    if not places:
        pytest.skip("no OpenMP runtime answered omp_get_num_places")
    assert places == mask
