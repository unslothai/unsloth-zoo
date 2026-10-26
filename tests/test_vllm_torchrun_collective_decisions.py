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

"""fast_inference under torchrun: decisions that change which collectives a rank enters are collective.

external_launcher puts every rank's vLLM engine in the job's process group. A rank that drops vLLM
(an old card) while another keeps it, or a rank retrying its engine alone after vLLM's cross-rank
memory MIN, leaves the ranks in different collectives until the group timeout. CPU only: gloo.
"""

import json
import os
import pathlib
import socket
import subprocess
import sys
import textwrap
import types

import pytest

import unsloth_zoo.vllm_utils as vu

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _torchrun(tmp_path, code, nproc = 2, timeout = 120):
    script = tmp_path / "worker.py"
    script.write_text(textwrap.dedent(code))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES = "", UNSLOTH_ALLOW_CPU = "1", UNSLOTH_DDP_TIMEOUT = "30",
               PYTHONPATH = os.pathsep.join([str(_ROOT), os.environ.get("PYTHONPATH", "")]))
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes", "1", "--nproc-per-node", str(nproc),
           "--master-addr", "127.0.0.1", "--master-port", str(_free_port()), str(script), str(tmp_path)]
    return subprocess.run(cmd, env = env, capture_output = True, text = True, timeout = timeout, cwd = tmp_path)


def _results(tmp_path, nproc = 2):
    return [json.loads((tmp_path / f"r{rank}.json").read_text()) for rank in range(nproc)]


def test_agreement_outside_torchrun_is_the_local_answer(monkeypatch):
    import torch.distributed as dist
    for key in vu._TORCHRUN_ENV_KEYS + ("TORCHELASTIC_RUN_ID",):
        monkeypatch.delenv(key, raising = False)
    assert vu.torchrun_ranks_agree(True) is True
    assert vu.torchrun_ranks_agree(False) is False
    assert not dist.is_initialized()


def test_one_rank_dropping_vllm_drops_it_on_every_rank(tmp_path):
    r = _torchrun(tmp_path, """
        import json, os, sys
        import torch.distributed as dist
        import unsloth_zoo.vllm_utils as vu
        rank = int(os.environ["RANK"])
        out = {
            "mixed": vu.torchrun_ranks_agree(rank == 0),  # rank 1 sits on an old card
            "all_ok": vu.torchrun_ranks_agree(True),
            "none_ok": vu.torchrun_ranks_agree(False),
            "world": dist.get_world_size(),
        }
        json.dump(out, open(os.path.join(sys.argv[1], "r%d.json" % rank), "w"))
        dist.destroy_process_group()
    """)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    for out in _results(tmp_path):
        assert out == {"mixed": False, "all_ok": True, "none_ok": False, "world": 2}


@pytest.mark.parametrize("failed, outcome", [
    ((False, False), ["ok", "ok"]),
    ((True, True), ["ok", "ok"]),            # symmetric: each rank keeps its usual retry / raise
    ((False, True), ["raise", "raise"]),     # one rank failed after the memory MIN: all stop
    ((True, False), ["raise", "raise"]),
])
def test_engine_attempt_outcome_is_the_same_on_every_rank(tmp_path, failed, outcome):
    r = _torchrun(tmp_path, f"""
        import json, os, sys, time
        import torch.distributed as dist
        import unsloth_zoo.vllm_utils as vu
        rank = int(os.environ["RANK"])
        vu._init_torchrun_process_group(vu._torchrun_launch_info())
        failed = {failed!r}[rank]
        start = time.time()
        try:
            vu._external_launcher_attempt_agreed(failed, "CUDA out of memory" if failed else None)
            out = {{"outcome": "ok"}}
        except RuntimeError as error:
            out = {{"outcome": "raise", "msg": str(error)}}
        out["seconds"] = time.time() - start
        json.dump(out, open(os.path.join(sys.argv[1], "r%d.json" % rank), "w"))
        dist.destroy_process_group()
    """)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    results = _results(tmp_path)
    assert [o["outcome"] for o in results] == outcome
    for rank, o in enumerate(results):
        assert o["seconds"] < 20
        if o["outcome"] == "raise":
            assert "1 of 2 torchrun ranks" in o["msg"]
            assert ("CUDA out of memory" in o["msg"]) == failed[rank]


def test_memory_reduce_counter_wraps_the_external_launcher_once(monkeypatch):
    calls = []

    class ExecutorWithExternalLauncher:
        def determine_available_memory(self):
            calls.append(1)
            return [123]

    fake = types.ModuleType("vllm.v1.executor.uniproc_executor")
    fake.ExecutorWithExternalLauncher = ExecutorWithExternalLauncher
    for name in ("vllm", "vllm.v1", "vllm.v1.executor"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.v1.executor.uniproc_executor", fake)
    monkeypatch.setattr(vu, "_EXTERNAL_LAUNCHER_MEMORY_REDUCES", 0)
    assert vu._count_external_launcher_memory_reduces()
    assert vu._count_external_launcher_memory_reduces()  # idempotent: one wrapper
    assert ExecutorWithExternalLauncher().determine_available_memory() == [123]
    assert calls == [1] and vu._EXTERNAL_LAUNCHER_MEMORY_REDUCES == 1


def test_memory_reduce_counter_absent_vllm_keeps_old_retries(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm.v1.executor.uniproc_executor", None)
    assert vu._count_external_launcher_memory_reduces() is False


def test_load_vllm_agrees_before_any_retry_and_after_success():
    """Source check: the failure report precedes every retry branch; success reports after the loop."""
    src = (_ROOT / "unsloth_zoo" / "vllm_utils.py").read_text(encoding = "utf-8")
    body = src[src.index("def load_vllm("):]
    body = body[: body.index("\ndef ", 10)]
    i_count = body.index("count_reduces = torchrun_info is not None and _count_external_launcher_memory_reduces()")
    i_loop = body.index("while True:", i_count)
    i_before = body.index("reduces_before = _EXTERNAL_LAUNCHER_MEMORY_REDUCES", i_loop)
    i_fail = body.index("_external_launcher_attempt_agreed(failed = True, error = error)")
    for retry in ("_is_vllm_replayssm_refusal(error)", "_is_memory_profiling_race_error(error)",
                  "trials += 1", "_shrink_for_memory_retry("):
        assert i_fail < body.index(retry), retry
    i_ok = body.index("_external_launcher_attempt_agreed(failed = False)")
    assert i_count < i_loop < i_before < i_fail < i_ok
    assert i_ok > body.index("if not loaded: _release_failed_vllm_engine(registries_before)")
