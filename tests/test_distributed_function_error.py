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

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


_WORKER = r"""
import json, sys, datetime
import torch.distributed as dist
# Long PG timeout: a peer that is never told about the error would hang far past the test budget.
dist.init_process_group("gloo", timeout = datetime.timedelta(seconds = 600))
rank = dist.get_rank()
from unsloth_zoo.utils import distributed_function
out = {"rank": rank}
out["one"] = distributed_function(1, lambda: {"a": 1})
out["two"] = distributed_function(2, lambda: (3, "x"))
out["args"] = distributed_function(1, lambda a, b = 0: a + b, 4, b = 5)

def bad_cache_write(path):
    raise ValueError("disk full at " + path)

try:
    distributed_function(1, bad_cache_write, "/cache/x.py")
    out["raised"] = None
except Exception as e:
    out["raised"] = [type(e).__name__, str(e)]
# The group must still be usable after the failure.
out["after"] = distributed_function(1, lambda: 11)
print("RESULT " + json.dumps(out), flush = True)
dist.destroy_process_group()
"""


def test_rank0_raise_reaches_every_rank(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text(_WORKER)
    port = _free_port()
    procs = []
    start = time.time()
    for rank in range(2):
        env = dict(
            os.environ,
            RANK = str(rank), LOCAL_RANK = str(rank),
            WORLD_SIZE = "2", LOCAL_WORLD_SIZE = "2",
            MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port),
            UNSLOTH_IS_PRESENT = "1",
            UNSLOTH_ZOO_DISABLE_GPU_INIT = "1",
            CUDA_VISIBLE_DEVICES = "",
            PYTHONPATH = os.pathsep.join([ROOT, os.environ.get("PYTHONPATH", "")]),
        )
        procs.append(subprocess.Popen(
            [sys.executable, str(script)],
            env = env, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True,
        ))
    results = {}
    for p in procs:
        try:
            out, _ = p.communicate(timeout = 120)
        except subprocess.TimeoutExpired:
            for q in procs: q.kill()
            pytest.fail("a rank hung in distributed_function after rank 0 raised")
        lines = [l for l in out.splitlines() if l.startswith("RESULT ")]
        assert p.returncode == 0 and lines, out[-3000:]
        r = json.loads(lines[-1][len("RESULT "):])
        results[r["rank"]] = r
    assert time.time() - start < 120

    for rank in (0, 1):
        r = results[rank]
        assert r["one"] == {"a": 1}
        assert r["two"] == [3, "x"]
        assert r["args"] == 9
        assert r["after"] == 11
        assert r["raised"] is not None, f"rank {rank} did not raise"
    assert results[0]["raised"] == ["ValueError", "disk full at /cache/x.py"]
    assert results[1]["raised"][0] == "RuntimeError"
    assert "ValueError: disk full at /cache/x.py" in results[1]["raised"][1]
    assert "bad_cache_write" in results[1]["raised"][1]


def test_not_initialized_runs_locally_and_raises():
    from unsloth_zoo.utils import distributed_function
    assert distributed_function(1, lambda: 5) == 5
    with pytest.raises(KeyError):
        distributed_function(1, lambda: {}["k"])
