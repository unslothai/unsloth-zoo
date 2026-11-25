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

"""GRPO under Unsloth's expert parallel: every MoE forward is an all-to-all over the ep ranks, so the ranks must
run the same number of forwards. apply_expert_parallel stamps `_unsloth_ep` (the ep mesh) on each model level;
grpo_forwards_are_collective must read it, else the mini-batch planner sizes each rank on its own (free memory,
sequence length) and the rank with fewer chunks leaves the other blocked in its next all-to-all (a hang).

The model below stands in for an expert parallel MoE: one collective per forward, as the all-to-all is."""

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

    rank, out = int(sys.argv[1]), sys.argv[2]
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
            if self.collective:  # the expert all-to-all: every rank must enter it
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

    # Each rank plans its own split (free memory, its own sequence length): rank 0 fits 1 row per forward,
    # rank 1 all of them, so rank 0 wants 4 forwards and rank 1 one.
    autotune = rl.autotune_batch_and_chunks
    rl.autotune_batch_and_chunks = lambda rows, *a, **k: (1 if rank == 0 else rows, 1)

    g = torch.Generator().manual_seed(100 + rank)
    seq = 7 if rank == 0 else 5  # unequal sequence lengths across ranks
    ids = torch.randint(1, V, (4, seq), generator = g)
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

    ep = Toy(collective = True)
    ep.__dict__["_unsloth_ep"] = object()  # what apply_expert_parallel stamps (the ep mesh)
    wrapper = torch.nn.Module()  # a DDP-like wrapper: the mark sits one level down
    wrapper.module = Toy(collective = False)
    wrapper.module.__dict__["_unsloth_ep"] = object()
    result = {
        "rank": rank,
        "ep": rl.grpo_forwards_are_collective(ep),
        "wrapped": rl.grpo_forwards_are_collective(wrapper),
        "plain": rl.grpo_forwards_are_collective(Toy(collective = False)),
    }
    try:
        result["loss"] = run(ep)
        result["forwards"] = ep.forwards
        ref = Toy(collective = False)
        result["ref_loss"] = run(ref)
        result["grad_err"] = max(
            (p.grad - r.grad).abs().max().item()
            for p, r in zip(ep.parameters(), ref.parameters()) if p.grad is not None
        )
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


def test_expert_parallel_ranks_agree_on_the_forward_count(tmp_path):
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
            [sys.executable, str(script), str(rank), str(out)],
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
        assert (r["ep"], r["wrapped"], r["plain"]) == (True, True, False), r
        # A forward-count mismatch blocks one rank in its next collective until the gloo timeout.
        assert "error" not in r, r
        assert r["forwards"] == 4, r
        assert abs(r["loss"] - r["ref_loss"]) < 1e-6, r
        assert r["grad_err"] < 1e-6, r
