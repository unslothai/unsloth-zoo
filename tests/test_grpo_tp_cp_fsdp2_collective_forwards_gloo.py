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

"""GRPO under Unsloth's tensor parallel, Unsloth's DPO / GRPO context parallel and accelerate's FSDP2: every
forward is a collective (tp all-reduces per block, Ulysses cp all-to-alls per attention layer, FSDP2 all-gathers
per unit), so the ranks must run the same number of forwards. grpo_forwards_are_collective must say so, else the
mini-batch planner sizes each rank on its own (free memory, sequence length) and the rank with fewer chunks leaves
the other blocked in its next collective (a hang).

The toy below stands in for those layouts: one collective per forward. Each layout is marked the way Unsloth /
accelerate mark it: `_unsloth_tp` on the model levels (apply_tensor_parallel), an active context parallel
manager (unsloth.context_parallel.get_cp_manager inside compute_loss) and `_unsloth_cp_rl_manager` on the hooked
decoder (attach_rl_hooks), and a real fully_shard root (accelerate's fsdp2_prepare_model), which carries no
Unsloth marker."""

import json
import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

if not torch.distributed.is_available():
    pytest.skip("torch.distributed unavailable", allow_module_level = True)

REPO = Path(__file__).resolve().parents[1]

_WORKER = textwrap.dedent(
    r"""
    import datetime, json, os, sys, types
    import torch
    import torch.distributed as dist

    rank, out, layout = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    dist.init_process_group("gloo", rank = rank, world_size = 2, timeout = datetime.timedelta(seconds = 20))
    import unsloth_zoo.rl_replacements as rl
    from transformers.modeling_outputs import CausalLMOutput

    V, H = 32, 8
    class Toy(torch.nn.Module):
        def __init__(self, collective):
            super().__init__()
            torch.manual_seed(0)
            self.embed = torch.nn.Embedding(V, H)
            self.proj = torch.nn.Linear(H, H)
            self.lm_head = torch.nn.Linear(H, V, bias = False)
            self.lm_head.weight.requires_grad_(False)
            self.collective = collective
            self.forwards = 0
        def get_output_embeddings(self):
            return self.lm_head
        def forward(self, input_ids = None, attention_mask = None, logits_to_keep = None, **kwargs):
            self.forwards += 1
            if self.collective:  # the tp all-reduce / cp all-to-all: every rank must enter it
                dist.all_reduce(torch.ones(1))
            hidden = self.proj(self.embed(input_ids))
            if logits_to_keep:
                hidden = hidden[:, -logits_to_keep:]
            return CausalLMOutput(logits = hidden)

    def trainer_for(model):
        return types.SimpleNamespace(
            args = types.SimpleNamespace(unsloth_grpo_mini_batch = None, unsloth_logit_chunk_multiplier = 1),
            processing_class = types.SimpleNamespace(pad_token_id = 0),
            model = model,
            accelerator = types.SimpleNamespace(unwrap_model = lambda m, keep_fp32_wrapper = False: m, scaler = None),
            use_vllm = False, _autocast_dtype = None, beta = 0.0,
        )

    # Each rank plans its own split from its own free memory: rank 0 fits 1 row per forward, rank 1 all of
    # them, so rank 0 wants 4 forwards and rank 1 one.
    rl.autotune_batch_and_chunks = lambda rows, *a, **k: (1 if rank == 0 else rows, 1)

    # tp / cp peers read the same rows.
    g = torch.Generator().manual_seed(100)
    ids = torch.randint(1, V, (4, 6), generator = g)
    adv = torch.randn(4, generator = g)

    def run(model):
        mask = torch.ones(4, 3, dtype = torch.long)
        loss = rl.grpo_accumulated_loss(
            trainer_for(model), ids, (ids != 0).long(), 3, mask, adv, None, None,
            loss_type = "dapo", num_items_in_batch = int(mask.sum()), num_processes = 1,
            current_gradient_accumulation_steps = 1, max_completion_length = 3,
        )[0]
        loss.backward()
        return loss.item()

    result = {"rank": rank, "zoo": rl.__file__, "plain": rl.grpo_forwards_are_collective(Toy(collective = False))}
    model = Toy(collective = True)
    if layout == "tp":
        model.__dict__["_unsloth_tp"] = object()  # what apply_tensor_parallel stamps on each model level
        wrapper = torch.nn.Module()  # a wrapper that is not a model level: the mark sits one level down
        wrapper.module = Toy(collective = False)
        wrapper.module.__dict__["_unsloth_tp"] = object()
        result["wrapped"] = rl.grpo_forwards_are_collective(wrapper)
    elif layout == "cp":
        # unsloth.context_parallel keeps the active manager while compute_loss / the logps passes run.
        cp = types.ModuleType("unsloth.context_parallel")
        cp.get_cp_manager = lambda: cp_state["manager"]
        cp_state = {"manager": None}
        sys.modules["unsloth.context_parallel"] = cp
        hooked = torch.nn.Module()  # attach_rl_hooks marks the decoder
        hooked.model = Toy(collective = False)
        hooked.model.__dict__["_unsloth_cp_rl_manager"] = object()
        result["wrapped"] = rl.grpo_forwards_are_collective(hooked)
        cp_state["manager"] = object()
    elif layout == "fsdp2":
        try:
            from torch.distributed.fsdp import fully_shard
        except ImportError:
            from torch.distributed._composable.fsdp import fully_shard
        from torch.distributed.device_mesh import init_device_mesh
        mesh = init_device_mesh("cpu", (2,))
        model = Toy(collective = False)  # the real all-gathers are the collective
        # Before sharding: a cached negative must not survive accelerate's later fully_shard of the root.
        result["before_shard"] = rl.grpo_forwards_are_collective(model)
        fully_shard(model, mesh = mesh)  # accelerate's fsdp2_prepare_model shards the root
    result["collective"] = rl.grpo_forwards_are_collective(model)
    try:
        result["loss"] = run(model)
        result["forwards"] = model.forwards
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"[:600]
    with open(f"{out}.{rank}", "w") as f:
        json.dump(result, f)
    """
)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("layout", ["tp", "cp", "fsdp2"])
def test_collective_layouts_agree_on_the_forward_count(tmp_path, layout):
    script = tmp_path / "worker.py"
    script.write_text(_WORKER)
    env = dict(os.environ)
    for key in ("UNSLOTH_RETURN_HIDDEN_STATES", "UNSLOTH_GRPO_LOGPROB_RECOMPUTE"):
        env.pop(key, None)
    env.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(_free_port()),
        CUDA_VISIBLE_DEVICES = "",
        TORCHDYNAMO_DISABLE = "1",
        UNSLOTH_IS_PRESENT = "1",
        UNSLOTH_ZOO_DISABLE_GPU_INIT = "1",
        UNSLOTH_GRPO_SEQ_PACKING = "0",
        UNSLOTH_GRPO_PREFIX_GROUPER = "0",
        PYTHONPATH = os.pathsep.join([str(REPO), env.get("PYTHONPATH", "")]),
    )
    out = tmp_path / "r"
    procs = [
        subprocess.Popen(
            [sys.executable, str(script), str(rank), str(out), layout],
            env = env, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True,
        )
        for rank in range(2)
    ]
    logs = []
    for p in procs:
        try:
            logs.append(p.communicate(timeout = 300)[0])
        except subprocess.TimeoutExpired:
            for q in procs:
                q.kill()
            logs.append(p.communicate()[0])
    results = {}
    for rank in range(2):
        path = Path(f"{out}.{rank}")
        assert path.exists(), f"rank {rank} wrote nothing:\n{logs[rank][-4000:]}"
        results[rank] = json.loads(path.read_text())
    for rank, r in results.items():
        assert r["zoo"].startswith(str(REPO)), r
        assert r["plain"] is False, r
        assert r["collective"] is True, r
        if layout in ("tp", "cp"):
            assert r["wrapped"] is True, r
        else:
            assert r["before_shard"] is False, r
        # A forward-count mismatch blocks one rank in its next collective until the gloo timeout.
        assert "error" not in r, r
        assert r["forwards"] == 4, r
    assert results[0]["loss"] == pytest.approx(results[1]["loss"], abs = 1e-6)
