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

"""UNSLOTH_FSDP2_MERGED_ROLLOUT=1: load_lora's gathered adapter copies feed the merged-rollout fold under native
FSDP2, any later in-place step makes them refused, and the default path caches nothing. CPU only: two gloo
torchrun ranks, no vLLM."""

import os
import pathlib
import socket
import subprocess
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_WORKER = r'''
import os

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
import torch
import torch.distributed as dist
import torch.nn as nn

dist.init_process_group("gloo")
from torch.distributed.fsdp import fully_shard  # noqa: E402
from torch.distributed.tensor import DTensor  # noqa: E402

import unsloth_zoo.vllm_utils as vu  # noqa: E402

rank = dist.get_rank()


class LoraLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = nn.Linear(8, 16, bias=False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(8, 4, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(4, 16, bias=False)})
        self.base_layer.weight.requires_grad_(False)

    def forward(self, x):
        return self.base_layer(x) + self.lora_B["default"](self.lora_A["default"](x))


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = LoraLinear()


def build():
    torch.manual_seed(0)
    m = M()
    fully_shard(m, ignored_params={m.layer.base_layer.weight})
    return m


def check(cond, what):
    if not cond:
        raise AssertionError(f"[rank {rank}] FAIL: {what}")
    print(f"[rank {rank}] ok: {what}", flush=True)


# 4. default (no opt-in)
os.environ.pop("UNSLOTH_FSDP2_MERGED_ROLLOUT", None)
m = build()
check(isinstance(m.layer.lora_A["default"].weight, DTensor), "adapters are DTensor shards")
check(vu._lora_is_sharded(m), "_lora_is_sharded sees the shards")
sd = vu._lora_state_dict(m)
check(sd["layer.lora_A.default.weight"].shape == (4, 8), "default gather returns full A")
check("_unsloth_fsdp2_lora_gather" not in m.__dict__ and m.layer not in vu._FSDP2_GATHERED_AB, "default path caches nothing")
try:
    vu._merged_rollout_adapter_weights(m.layer, "default")
    check(False, "default helper must refuse DTensor shards")
except RuntimeError:
    check(True, "default helper refuses DTensor shards")

# 1. opt-in gather
os.environ["UNSLOTH_FSDP2_MERGED_ROLLOUT"] = "1"
m = build()
vu._lora_state_dict(m)
A, B = vu._merged_rollout_adapter_weights(m.layer, "default")
check(A.shape == (4, 8) and B.shape == (16, 4), "helper returns full-shape A / B")
check(torch.equal(A, m.layer.lora_A["default"].weight.full_tensor()), "A equals full_tensor()")
check(torch.equal(B, m.layer.lora_B["default"].weight.full_tensor()), "B equals full_tensor()")

# 3. reuse without a step: count all_gathers through a wrapper
calls = {"n": 0}
orig = DTensor.full_tensor


def counting(self, *a, **k):
    calls["n"] += 1
    return orig(self, *a, **k)


DTensor.full_tensor = counting
vu._lora_state_dict(m, reuse_gathered=True)
check(calls["n"] == 0, "reuse_gathered returns load_lora's gather without a collective")
vu._lora_state_dict(m)
check(calls["n"] == 2, "a plain call (load_lora) always re-gathers")

# 2. in-place steps on the shards: the DTensor-level counter (non-foreach) ...
with torch.no_grad():
    for p in (m.layer.lora_A["default"].weight, m.layer.lora_B["default"].weight):
        p.add_(1.0)
for what, fn in (("fold helper", lambda: vu._merged_rollout_adapter_weights(m.layer, "default")),
                 ("reuse_gathered", lambda: vu._lora_state_dict(m, reuse_gathered=True))):
    try:
        fn()
        check(False, f"{what} must refuse a gather older than a DTensor-level step")
    except RuntimeError:
        check(True, f"{what} refuses a gather older than a DTensor-level step")
vu._lora_state_dict(m)
A2, B2 = vu._merged_rollout_adapter_weights(m.layer, "default")
check(torch.equal(A2, A + 1.0) and torch.equal(B2, B + 1.0), "re-gathered copies carry the step")
# ... and the local-shard counter (what CUDA foreach / fused AdamW bumps)
with torch.no_grad():
    m.layer.lora_B["default"].weight._local_tensor.add_(1.0)
try:
    vu._merged_rollout_adapter_weights(m.layer, "default")
    check(False, "fold helper must refuse a gather older than a local-shard step")
except RuntimeError:
    check(True, "fold helper refuses a gather older than a local-shard step")

# 5. no load_lora gather yet: reuse_gathered refuses instead of gathering inside one rank's generate
m = build()
calls["n"] = 0
try:
    vu._lora_state_dict(m, reuse_gathered=True)
    check(False, "reuse_gathered without a gather must refuse")
except RuntimeError:
    check(calls["n"] == 0, "reuse_gathered without a gather refuses and makes no collective")


# 6. load_lora's gather: once per vllm_rollout_scope, never where nothing can fold
class Engine:
    _unsloth_merged_rollout_wrapped = True


os.environ["UNSLOTH_VLLM_MERGED_ROLLOUT"] = "1"
m = build()
m.vllm_engine = Engine()
calls["n"] = 0
check(vu._fsdp2_merged_rollout_gather(m) and calls["n"] == 2, "load_lora gathers outside a scope")
check(vu._fsdp2_merged_rollout_gather(m) and calls["n"] == 4, "every load_lora outside a scope re-gathers")
scope = vu._RolloutScope()
check(vu._fsdp2_merged_rollout_gather(m, scope) and calls["n"] == 6, "the first load_lora in a scope gathers")
for turn in range(3):
    check(vu._fsdp2_merged_rollout_gather(m, scope) and calls["n"] == 6, f"scope turn {turn + 2} makes no collective")
vu._lora_state_dict(m, reuse_gathered=True)
check(calls["n"] == 6, "a fallback inside the scope reuses the gather")
m.vllm_engine = object()
check(not vu._fsdp2_merged_rollout_gather(m) and calls["n"] == 6, "an engine without the merged wrapper never gathers")
m.vllm_engine = Engine()
m.is_loaded_in_4bit = True
check(not vu._fsdp2_merged_rollout_gather(m) and calls["n"] == 6, "a bitsandbytes (QLoRA) base never gathers for a fold")
del m.is_loaded_in_4bit
os.environ.pop("UNSLOTH_FSDP2_MERGED_ROLLOUT")
check(not vu._fsdp2_merged_rollout_gather(m) and calls["n"] == 6, "without the opt-in load_lora adds no gather")
os.environ.pop("UNSLOTH_VLLM_MERGED_ROLLOUT")

# 7. a root group with reshard_after_forward=False (fully_shard's default for the root) keeps its adapters unsharded after a
# no-grad forward; the gather's pre-save hooks swap them back to DTensors, and the cache must match the swapped parameters.
os.environ["UNSLOTH_FSDP2_MERGED_ROLLOUT"] = "1"


class M2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer, self.layer2 = LoraLinear(), LoraLinear()

    def forward(self, x):
        return self.layer2(self.layer(x)[:, :8])


torch.manual_seed(0)
m = M2()
fully_shard(m.layer, ignored_params={m.layer.base_layer.weight})
fully_shard(m, ignored_params={m.layer2.base_layer.weight}, reshard_after_forward=False)
vu._lora_state_dict(m)
with torch.no_grad():
    m(torch.randn(2, 8))
check(not isinstance(m.layer2.lora_A["default"].weight, DTensor), "the root group's adapters stay unsharded after a no-grad forward")
vu._lora_state_dict(m)
try:
    vu._lora_state_dict(m, reuse_gathered=True)
    check(True, "reuse_gathered accepts the fresh gather")
except RuntimeError as e:
    check(False, f"reuse_gathered refused the fresh gather: {e}")
os.environ.pop("UNSLOTH_FSDP2_MERGED_ROLLOUT")
DTensor.full_tensor = orig
dist.barrier()
dist.destroy_process_group()
print(f"[rank {rank}] ALL PASS", flush=True)
'''


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_fsdp2_merged_rollout_gathered_adapters(tmp_path):
    pytest.importorskip("torch.distributed.fsdp")
    script = tmp_path / "worker.py"
    script.write_text(_WORKER)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES = "", UNSLOTH_ALLOW_CPU = "1",
               PYTHONPATH = os.pathsep.join([str(_ROOT), os.environ.get("PYTHONPATH", "")]))
    env.pop("UNSLOTH_FSDP2_MERGED_ROLLOUT", None)
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes", "1", "--nproc-per-node", "2",
           "--master-addr", "127.0.0.1", "--master-port", str(_free_port()), str(script)]
    r = subprocess.run(cmd, env = env, capture_output = True, text = True, timeout = 300, cwd = tmp_path)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert r.stdout.count("ALL PASS") == 2, r.stdout[-3000:]
