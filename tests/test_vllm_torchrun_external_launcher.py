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

"""fast_inference = True under plain torchrun (DDP).

vLLM's default executor rendezvouses a private world_size = 1 group over a fresh
tcp:// port. torchrun exports TORCHELASTIC_USE_AGENT_STORE=True, so that rendezvous
is a TCPStore client of a server that never exists and engine init hangs. load_vllm
now joins the torchrun group with vLLM's external_launcher executor instead. CPU only:
the torchrun cases use gloo and never build an engine.
"""

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
_TORCHRUN_ENV = {"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "2", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29500"}


def test_launch_info_needs_every_torchrun_variable():
    assert vu._torchrun_launch_info(_TORCHRUN_ENV) == (1, 1, 2)
    for key in _TORCHRUN_ENV:
        partial = {k: v for k, v in _TORCHRUN_ENV.items() if k != key}
        assert vu._torchrun_launch_info(partial) is None, key
    assert vu._torchrun_launch_info({**_TORCHRUN_ENV, "RANK": "x"}) is None
    assert vu._torchrun_launch_info({}) is None
    # A hand-set single process (no torchrun) keeps vLLM's executor; torchrun --nproc-per-node 1 does not.
    single = {**_TORCHRUN_ENV, "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"}
    assert vu._torchrun_launch_info(single) is None
    assert vu._torchrun_launch_info({**single, "TORCHELASTIC_RUN_ID": "abc"}) == (0, 0, 1)
    for bad in ({"LOCAL_RANK": "-1"}, {"WORLD_SIZE": "0"}, {"RANK": "2"}):
        assert vu._torchrun_launch_info({**_TORCHRUN_ENV, **bad}) is None, bad


def test_target_device_is_device_0_outside_torchrun(monkeypatch):
    """vLLM's own executor takes device 0 whatever is current; a torchrun rank uses its own."""
    for key in vu._TORCHRUN_ENV_KEYS + ("TORCHELASTIC_RUN_ID",):
        monkeypatch.delenv(key, raising = False)
    monkeypatch.setattr(vu, "_current_device_index", lambda: 1)
    assert vu.get_target_device().index == 0
    assert vu.get_target_device(3).index == 3
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    assert vu.get_target_device().index == 1


def test_current_device_index_is_an_int_on_cpu(monkeypatch):
    monkeypatch.setattr(vu, "DEVICE_TYPE", "cpu")
    assert vu._current_device_index() == 0


def test_process_group_timeout_env(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DDP_TIMEOUT", raising = False)
    assert vu._process_group_timeout() == 1800
    monkeypatch.setenv("UNSLOTH_DDP_TIMEOUT", "77")
    assert vu._process_group_timeout() == 77
    monkeypatch.setenv("UNSLOTH_DDP_TIMEOUT", "x")
    assert vu._process_group_timeout() == 1800


def test_scoped_environ_restores_set_and_absent_keys(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.delenv("UNSLOTH_TEST_ABSENT", raising = False)
    with vu._scoped_environ({"LOCAL_RANK": "0", "UNSLOTH_TEST_ABSENT": "1"}):
        assert os.environ["LOCAL_RANK"] == "0"
        assert os.environ["UNSLOTH_TEST_ABSENT"] == "1"
    assert os.environ["LOCAL_RANK"] == "3"
    assert "UNSLOTH_TEST_ABSENT" not in os.environ
    with pytest.raises(RuntimeError):
        with vu._scoped_environ({"LOCAL_RANK": "0"}):
            raise RuntimeError("engine init failed")
    assert os.environ["LOCAL_RANK"] == "3"


def test_load_vllm_wires_the_helper_before_the_engine():
    """Source check: the helper runs before the EngineArgs filter, and LLM() sits in the scoped env."""
    src = (_ROOT / "unsloth_zoo" / "vllm_utils.py").read_text(encoding = "utf-8")
    body = src[src.index("def load_vllm("):]
    body = body[: body.index("\ndef ", 10)]
    i_setup = body.index("_setup_vllm_external_launcher(engine_args, torchrun_info)")
    i_filter = body.index("good_keys = inspect.signature(AsyncEngineArgs if use_async else EngineArgs)")
    i_scope = body.index("with _scoped_environ(scoped):")
    i_llm = body.index("llm = LLM(**engine_args)")
    assert i_setup < i_filter < i_scope < i_llm
    assert "None if (return_args or use_async) else _torchrun_launch_info()" in body


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _torchrun(tmp_path, code, nproc = 2, timeout = 180):
    script = tmp_path / "worker.py"
    script.write_text(textwrap.dedent(code))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES = "", UNSLOTH_ALLOW_CPU = "1",
               PYTHONPATH = os.pathsep.join([str(_ROOT), os.environ.get("PYTHONPATH", "")]))
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes", "1", "--nproc-per-node", str(nproc),
           "--master-addr", "127.0.0.1", "--master-port", str(_free_port()), str(script), str(tmp_path)]
    return subprocess.run(cmd, env = env, capture_output = True, text = True, timeout = timeout, cwd = tmp_path)


def test_external_launcher_joins_the_torchrun_group_under_gloo(tmp_path):
    """Two torchrun ranks: one world of 2 (not two worlds of 1), agent-store flag untouched, seeds offset."""
    r = _torchrun(tmp_path, """
        import json, os, sys
        import torch.distributed as dist
        import unsloth_zoo.vllm_utils as vu
        assert os.environ["TORCHELASTIC_USE_AGENT_STORE"] == "True"
        info = vu._torchrun_launch_info()
        engine_args = {"seed": 3}
        local = vu._setup_vllm_external_launcher(engine_args, info)
        flag = os.environ["TORCHELASTIC_USE_AGENT_STORE"]
        out = {"world": dist.get_world_size(), "rank": dist.get_rank(), "args": engine_args, "local": local, "flag": flag}
        # The group is the job's: a collective spans both ranks.
        import torch
        t = torch.tensor([dist.get_rank() + 1])
        dist.all_reduce(t)
        out["sum"] = int(t.item())
        json.dump(out, open(os.path.join(sys.argv[1], "r%d.json" % dist.get_rank()), "w"))
        dist.destroy_process_group()
    """)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    import json
    outs = [json.load(open(tmp_path / f"r{i}.json")) for i in range(2)]
    for i, o in enumerate(outs):
        assert o["world"] == 2 and o["rank"] == i and o["sum"] == 3
        assert o["args"] == {"seed": 3 + i, "distributed_executor_backend": "external_launcher"}
        assert o["flag"] == "True"
        assert o["local"] is None


def test_save_dir_atomically_replaces_files_and_cleans_up(tmp_path):
    target = tmp_path / "lora"
    target.mkdir()
    (target / "adapter_model.safetensors").write_text("old")
    (target / "keep.txt").write_text("user file")

    def save_fn(tmp):
        os.makedirs(os.path.join(tmp, "sub"))
        open(os.path.join(tmp, "adapter_model.safetensors"), "w").write("new")
        open(os.path.join(tmp, "sub", "x.json"), "w").write("{}")

    vu._save_dir_atomically(save_fn, str(target), 1)
    assert (target / "adapter_model.safetensors").read_text() == "new"
    assert (target / "sub" / "x.json").read_text() == "{}"
    assert (target / "keep.txt").read_text() == "user file"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["lora"]


def test_save_lora_under_ddp_writes_through_a_private_folder(tmp_path, monkeypatch):
    calls = []

    class FakeModel:
        def state_dict(self):
            import torch
            return {"a.lora_A.weight": torch.ones(1), "b.weight": torch.ones(1)}

        def get_input_embeddings(self):
            import torch
            return types.SimpleNamespace(weight = torch.ones(1, dtype = torch.float16))

        def save_pretrained(self, save_directory, **kwargs):
            calls.append((save_directory, kwargs.get("push_to_hub"), sorted(kwargs["state_dict"])))
            os.makedirs(save_directory, exist_ok = True)
            open(os.path.join(save_directory, "adapter_config.json"), "w").write("{}")

    target = str(tmp_path / "grpo_saved_lora")
    monkeypatch.setattr(vu, "_distributed_rank_and_world", lambda: (1, 2))
    vu.save_lora(FakeModel(), target)
    assert calls[-1][0] != target and not calls[-1][1] and calls[-1][2] == ["a.lora_A.weight"]
    assert os.path.exists(os.path.join(target, "adapter_config.json"))
    assert sorted(os.listdir(tmp_path)) == ["grpo_saved_lora"]

    # A push still goes out from whichever rank the user calls it on, under the real path.
    vu.save_lora(FakeModel(), target, push_to_hub = True)
    assert calls[-1][0] == target and calls[-1][1] is True

    monkeypatch.setattr(vu, "_distributed_rank_and_world", lambda: (0, 1))
    vu.save_lora(FakeModel(), target)
    assert calls[-1][0] == target


@pytest.mark.parametrize("split_group", [None, "0", "1"])
def test_split_group_gets_a_device_bound_default_group(monkeypatch, split_group):
    import torch.distributed as dist

    calls = []
    monkeypatch.setattr(vu, "DEVICE_TYPE", "cuda")
    monkeypatch.setattr(vu.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(vu.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(vu.torch.cuda, "set_device", lambda index: None)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "init_process_group", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(dist, "get_rank", lambda: 1)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(vu, "get_target_device", lambda index = None: "cuda:1")
    monkeypatch.setattr(dist, "is_gloo_available", lambda: True)
    monkeypatch.setattr(dist, "is_nccl_available", lambda: True)
    if split_group is None:
        monkeypatch.delenv("VLLM_DISTRIBUTED_USE_SPLIT_GROUP", raising = False)
    else:
        monkeypatch.setenv("VLLM_DISTRIBUTED_USE_SPLIT_GROUP", split_group)
    vu._setup_vllm_external_launcher({}, (1, 1, 2))
    assert len(calls) == 1
    if split_group == "1":
        assert calls[0]["device_id"] == vu.torch.device("cuda", 1)
        assert calls[0]["backend"] == "cpu:gloo,cuda:nccl"
    else:
        # Not device bound, but gloo beside nccl: the job's CPU collectives need a CPU backend.
        assert "device_id" not in calls[0] and calls[0]["backend"] == "cpu:gloo,cuda:nccl"
