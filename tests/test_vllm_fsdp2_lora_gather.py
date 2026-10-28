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

"""save_lora / load_lora gather FSDP2-sharded LoRA adapters. CPU only: two gloo torchrun ranks, no vLLM."""

import json
import os
import pathlib
import socket
import subprocess
import sys
import textwrap

_ROOT = pathlib.Path(__file__).resolve().parents[1]

_WORKER = """
import json, os, sys
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
out_dir = sys.argv[1]
dist.init_process_group("gloo")
rank = dist.get_rank()
import unsloth_zoo.vllm_utils as vu

class Lora(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = torch.nn.Linear(8, 6, bias = False)
        self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(8, 4, bias = False)})
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(4, 6, bias = False)})

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(10, 8)
        self.layers = torch.nn.ModuleList([Lora(), Lora()])
        self.saved = None
    def get_input_embeddings(self):
        return self.embed
    def save_pretrained(self, save_directory, state_dict = None, **kwargs):
        os.makedirs(save_directory, exist_ok = True)
        self.saved = state_dict
        torch.save(state_dict, os.path.join(save_directory, "adapter.pt"))

torch.manual_seed(0)
model = Model()
full = {k: v.detach().clone() for k, v in model.state_dict().items() if ".lora_" in k}
# Checked before the Trainer shards the model: that False must not stick.
early_sharded = vu._lora_is_sharded(model) if hasattr(vu, "_lora_is_sharded") else None
mesh = init_device_mesh("cpu", (2,))
for layer in model.layers:
    fully_shard(layer, mesh = mesh)
fully_shard(model, mesh = mesh)
res = {"helper": hasattr(vu, "_lora_state_dict"), "early_sharded": early_sharded}
try:
    vu.save_lora(model, os.path.join(out_dir, "lora"))
    saved = model.saved
    res["save_plain"] = all(type(v) is torch.Tensor for v in saved.values())
    res["save_equal"] = sorted(saved) == sorted(full) and all(
        torch.equal(saved[k].to(full[k].dtype), full[k]) for k in full)
    on_disk = torch.load(os.path.join(out_dir, "lora", "adapter.pt"), weights_only = False)
    res["disk_plain"] = all(type(v) is torch.Tensor for v in on_disk.values())
except Exception as e:
    res["save_error"] = type(e).__name__ + ": " + str(e)[:300]
if res["helper"]:
    # load_lora(load_tensors = True) takes it under torch.inference_mode.
    try:
        with torch.inference_mode():
            gathered = vu._lora_state_dict(model)
        res["gather_equal"] = sorted(gathered) == sorted(full) and all(
            type(v) is torch.Tensor and torch.equal(v, full[k]) for k, v in gathered.items())
    except Exception as e:
        res["gather_error"] = type(e).__name__ + ": " + str(e)[:300]
if hasattr(vu, "_lora_is_sharded"):
    vu._merged_rollout_enabled = lambda: True
    res["sharded"] = vu._lora_is_sharded(model)
    res["merged_rollout"] = vu.prepare_merged_rollout(model) is None
json.dump(res, open(os.path.join(out_dir, "rank%d.json" % rank), "w"))
dist.destroy_process_group()
"""


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_save_lora_and_load_lora_tensors_gather_fsdp2_shards(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text(textwrap.dedent(_WORKER))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES = "", UNSLOTH_ALLOW_CPU = "1",
               PYTHONPATH = os.pathsep.join([str(_ROOT), os.environ.get("PYTHONPATH", "")]))
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes", "1", "--nproc-per-node", "2",
           "--master-addr", "127.0.0.1", "--master-port", str(_free_port()), str(script), str(tmp_path)]
    r = subprocess.run(cmd, env = env, capture_output = True, text = True, timeout = 300, cwd = tmp_path)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    for rank in range(2):
        res = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert "save_error" not in res, res
        assert res["save_plain"] and res["save_equal"] and res["disk_plain"], res
        assert res["helper"] and "gather_error" not in res and res["gather_equal"], res
        assert res["early_sharded"] is False, res
        assert res["sharded"] is True and res["merged_rollout"] is True, res


_SCAN_WORKER = """
import json, sys
import torch
import unsloth_zoo.vllm_utils as vu

class Lora(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = torch.nn.Linear(4, 4, bias = False)
        self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(4, 2, bias = False)})
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 4, bias = False)})

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Lora() for _ in range(400)])

model = Model()
seen = []
orig = model.named_parameters
def counting(*args, **kwargs):
    for item in orig(*args, **kwargs):
        seen.append(item[0])
        yield item
model.named_parameters = counting
res = {"sharded": vu._lora_is_sharded(model), "params_walked": len(seen), "params_total": len(list(orig()))}
json.dump(res, open(sys.argv[1], "w"))
"""


def test_unsharded_check_stops_at_the_first_adapter(tmp_path):
    script = tmp_path / "scan.py"
    script.write_text(textwrap.dedent(_SCAN_WORKER))
    out = tmp_path / "scan.json"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES = "", UNSLOTH_ALLOW_CPU = "1",
               PYTHONPATH = os.pathsep.join([str(_ROOT), os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run([sys.executable, str(script), str(out)], env = env, capture_output = True, text = True,
                       timeout = 300, cwd = tmp_path)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    res = json.loads(out.read_text())
    assert res["sharded"] is False, res
    assert res["params_total"] == 1200, res
    assert res["params_walked"] <= 3, res


_PLAIN_WORKER = """
import json, sys
import torch
import unsloth_zoo.vllm_utils as vu

class NoParams:
    def state_dict(self):
        return {"layers.0.lora_A.default.weight": torch.zeros(2, 4)}

model = torch.nn.Linear(4, 4)
walked = []
orig = model.named_parameters
model.named_parameters = lambda *a, **k: walked.append(1) or orig(*a, **k)
res = {
    "no_params": vu._lora_is_sharded(NoParams()),
    "no_params_state": sorted(vu._lora_state_dict(NoParams())),
    "no_group": vu._lora_is_sharded(model),
    "walked": len(walked),
}
json.dump(res, open(sys.argv[1], "w"))
"""


def test_no_process_group_or_no_named_parameters_is_unsharded(tmp_path):
    # save_lora / load_lora take module-like objects without named_parameters; outside a process
    # group nothing can be an FSDP2 DTensor, so the parameters are not walked either.
    script = tmp_path / "plain.py"
    script.write_text(textwrap.dedent(_PLAIN_WORKER))
    out = tmp_path / "plain.json"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES = "", UNSLOTH_ALLOW_CPU = "1",
               PYTHONPATH = os.pathsep.join([str(_ROOT), os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run([sys.executable, str(script), str(out)], env = env, capture_output = True, text = True,
                       timeout = 300, cwd = tmp_path)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    res = json.loads(out.read_text())
    assert res == {
        "no_params": False,
        "no_params_state": ["layers.0.lora_A.default.weight"],
        "no_group": False,
        "walked": 0,
    }, res
