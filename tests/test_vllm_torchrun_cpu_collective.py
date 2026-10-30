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

"""fast_inference = True under torchrun: init_process_group(backend = None) on CUDA registers only nccl, so
_all_reduce_int on a CPU int raised No backend type associated with device type cpu."""

import json
import os
import pathlib
import socket
import subprocess
import sys
import textwrap

import pytest
import torch.distributed as dist

import unsloth_zoo.vllm_utils as vu

_ROOT = pathlib.Path(__file__).resolve().parents[1]

_WORKER = """
import json, os, sys
import torch.distributed as dist
backend, out_dir = sys.argv[1], sys.argv[2]
dist.init_process_group(backend)
import unsloth_zoo.vllm_utils as vu
rank = dist.get_rank()
res = {"takes_cpu": vu._group_takes_cpu_tensors() if hasattr(vu, "_group_takes_cpu_tensors") else None}
try:
    res["min"] = vu._all_reduce_int(rank + 1, dist.ReduceOp.MIN)
    res["sum"] = vu._all_reduce_int(rank + 1, dist.ReduceOp.SUM)
    # The load-time gate: one rank without vLLM turns it off on every rank.
    res["agree_all"] = vu.torchrun_ranks_agree(True)
    res["agree_one_off"] = vu.torchrun_ranks_agree(rank != 1)
except Exception as e:
    res["error"] = type(e).__name__ + ": " + str(e)
json.dump(res, open(os.path.join(out_dir, "rank%d.json" % rank), "w"))
dist.destroy_process_group()
"""


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _torchrun(tmp_path, backend, nproc = 2, timeout = 180):
    script = tmp_path / "worker.py"
    script.write_text(textwrap.dedent(_WORKER))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES = "", UNSLOTH_ALLOW_CPU = "1",
               PYTHONPATH = os.pathsep.join([str(_ROOT), os.environ.get("PYTHONPATH", "")]))
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes", "1", "--nproc-per-node", str(nproc),
           "--master-addr", "127.0.0.1", "--master-port", str(_free_port()), str(script), backend, str(tmp_path)]
    r = subprocess.run(cmd, env = env, capture_output = True, text = True, timeout = timeout, cwd = tmp_path)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return [json.loads((tmp_path / f"rank{i}.json").read_text()) for i in range(nproc)]


@pytest.mark.parametrize("backend", ["cuda:gloo", "gloo", "cpu:gloo,cuda:gloo"])
def test_cross_rank_int_reduce_without_a_cpu_backend(tmp_path, backend):
    out = _torchrun(tmp_path, backend)
    for res in out:
        assert "error" not in res, res
        assert res["min"] == 1 and res["sum"] == 3, res
        assert res["agree_all"] is True and res["agree_one_off"] is False, res
    if backend == "cuda:gloo":
        assert out[0]["takes_cpu"] is False


def test_takes_cpu_tensors_reads_the_backend_config(monkeypatch):
    for config, expected in (("cuda:nccl", False), ("cpu:gloo,cuda:nccl", True), ("cpu:gloo, cuda:gloo", True),
                             ("xpu:xccl", False), ("gloo", True)):
        monkeypatch.setattr(dist, "get_backend_config", lambda group = None, c = config: c)
        assert vu._group_takes_cpu_tensors() is expected, config
    def boom(group = None):
        raise RuntimeError("no config")
    monkeypatch.setattr(dist, "get_backend_config", boom)
    for name, expected in (("nccl", False), ("undefined", False), ("gloo", True)):
        monkeypatch.setattr(dist, "get_backend", lambda group = None, n = name: n)
        assert vu._group_takes_cpu_tensors() is expected, name


def test_torchrun_group_keeps_a_cpu_backend_on_cuda(monkeypatch):
    """The group load_vllm creates under torchrun: gloo beside nccl on a CUDA rank, torch's pick elsewhere."""
    calls = []
    monkeypatch.setattr(vu, "DEVICE_TYPE", "cuda")
    monkeypatch.setattr(vu.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(vu.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(vu.torch.cuda, "set_device", lambda index: None)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "is_nccl_available", lambda: True)
    monkeypatch.setattr(dist, "init_process_group", lambda **kwargs: calls.append(kwargs))
    assert vu._init_torchrun_process_group((1, 1, 2)) == 1
    assert calls[-1]["backend"] == "cpu:gloo,cuda:nccl", calls[-1]
    monkeypatch.setattr(dist, "is_nccl_available", lambda: False)
    vu._init_torchrun_process_group((1, 1, 2))
    assert calls[-1]["backend"] is None
    monkeypatch.setattr(vu.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(dist, "is_nccl_available", lambda: True)
    vu._init_torchrun_process_group((0, 0, 2))
    assert calls[-1]["backend"] is None
