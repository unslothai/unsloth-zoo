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

"""distributed_any() must run on each rank's own device, not cuda:0 before set_device."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import types

import pytest
import torch
import torch.distributed as dist

from unsloth_zoo import utils as U

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


_WORKER = r"""
import json, os, sys, datetime
import torch, torch.distributed as dist
dist.init_process_group(sys.argv[1], timeout=datetime.timedelta(seconds=60))
rank = dist.get_rank()
from unsloth_zoo.utils import distributed_any, distributed_function
out = {"rank": rank, "any_one": distributed_any(rank == 1), "any_none": distributed_any(False)}
out["from_rank0"] = distributed_function(1, lambda: 7)
if sys.argv[1] == "nccl":
    torch.cuda.synchronize()
    out["reserved"] = [torch.cuda.memory_reserved(d) for d in range(torch.cuda.device_count())]
print("RESULT " + json.dumps(out), flush=True)
dist.destroy_process_group()
"""


def _run_two_ranks(tmp_path, backend):
    script = tmp_path / "worker.py"
    script.write_text(_WORKER)
    port = _free_port()
    procs = []
    for rank in range(2):
        env = dict(
            os.environ,
            RANK=str(rank),
            LOCAL_RANK=str(rank),
            WORLD_SIZE="2",
            LOCAL_WORLD_SIZE="2",
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
            UNSLOTH_IS_PRESENT="1",
            PYTHONPATH=os.pathsep.join([ROOT, os.environ.get("PYTHONPATH", "")]),
        )
        procs.append(subprocess.Popen(
            [sys.executable, str(script), backend],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        ))
    results = {}
    for p in procs:
        try:
            out, _ = p.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            for q in procs:
                q.kill()
            pytest.fail("a rank hung in distributed_any")
        lines = [l for l in out.splitlines() if l.startswith("RESULT ")]
        assert p.returncode == 0 and lines, out[-3000:]
        r = json.loads(lines[-1][len("RESULT "):])
        results[r["rank"]] = r
    return results


def test_gloo_two_ranks_agree(tmp_path):
    results = _run_two_ranks(tmp_path, "gloo")
    for rank in (0, 1):
        assert results[rank]["any_one"] is True
        assert results[rank]["from_rank0"] == 7
        assert results[rank]["any_none"] is False


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2 or not dist.is_nccl_available(),
    reason="needs 2 CUDA devices and NCCL",
)
def test_nccl_rank_one_stays_off_gpu_zero_before_set_device(tmp_path):
    results = _run_two_ranks(tmp_path, "nccl")
    for rank in (0, 1):
        assert results[rank]["any_one"] is True
        assert results[rank]["any_none"] is False
        assert results[rank]["from_rank0"] == 7
    # Rank 1 never called set_device: its buffers must still be on cuda:1, not cuda:0.
    assert results[1]["reserved"][0] == 0, results[1]["reserved"]
    assert results[1]["reserved"][1] > 0, results[1]["reserved"]


class _Group:
    def __init__(self, bound):
        self.bound_device_id = bound


def _fake_nccl(monkeypatch, bound=None, initialized=False, current=0, count=2, local_rank="1"):
    c10d = dist.distributed_c10d
    monkeypatch.setattr(c10d, "_get_object_coll_device", lambda group=None: "cuda")
    monkeypatch.setattr(c10d, "_get_default_group", lambda: _Group(bound))
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: initialized)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    if local_rank is None:
        monkeypatch.delenv("LOCAL_RANK", raising=False)
    else:
        monkeypatch.setenv("LOCAL_RANK", local_rank)


def test_local_rank_device_before_set_device(monkeypatch):
    _fake_nccl(monkeypatch)
    assert U._collective_device() == torch.device("cuda", 1)


def test_bound_device_wins(monkeypatch):
    _fake_nccl(monkeypatch, bound=torch.device("cuda", 3), count=4)
    assert U._collective_device() == torch.device("cuda", 3)


def test_device_already_selected_wins_over_local_rank(monkeypatch):
    _fake_nccl(monkeypatch, initialized=True, current=2, count=4)
    assert U._collective_device() == torch.device("cuda", 2)


def test_local_rank_outside_visible_devices_uses_current(monkeypatch):
    # One GPU per process (CUDA_VISIBLE_DEVICES set per rank): LOCAL_RANK 1 is not a local index.
    _fake_nccl(monkeypatch, count=1)
    assert U._collective_device() == torch.device("cuda", 0)


def test_cpu_backend_uses_cpu(monkeypatch):
    monkeypatch.setattr(dist.distributed_c10d, "_get_object_coll_device", lambda group=None: "cpu")
    assert U._collective_device() == torch.device("cpu")


def test_distributed_any_does_not_use_object_collectives(monkeypatch):
    seen = []

    def _all_reduce(tensor, op=None, **kwargs):
        seen.append((tensor.device, op))
        tensor.fill_(1)

    def _no_object_collective(*a, **k):
        raise AssertionError("object collective: lands on torch.cuda.current_device()")

    monkeypatch.setattr(U, "torch_distributed_is_initialized", lambda: True)
    monkeypatch.setattr(dist.distributed_c10d, "_get_object_coll_device", lambda group=None: "cpu")
    monkeypatch.setattr(dist, "all_gather_object", _no_object_collective)
    monkeypatch.setattr(dist, "all_reduce", _all_reduce)
    assert U.distributed_any(False) is True
    assert seen == [(torch.device("cpu"), dist.ReduceOp.MAX)]
