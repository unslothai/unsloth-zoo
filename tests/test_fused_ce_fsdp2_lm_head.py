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

"""unsloth-zoo#1751: fused CE under FSDP2 when lm_head is its own FSDP group.

accelerate >= 1.15 shards an untied lm_head (plus the final norm it can find) in a separate
group whose all-gather only fires when lm_head is called. The fused losses read lm_head.weight
instead, so on two gloo ranks the stock forward below died with "aten.mm.default got mixed
torch.Tensor and DTensor". Loss and the reduced lm_head gradient must match a plain reference.
"""

import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

pytest.importorskip("torch.distributed.fsdp")
if not torch.distributed.is_available() or not hasattr(torch.distributed.fsdp, "fully_shard"):
    pytest.skip("FSDP2 fully_shard unavailable", allow_module_level = True)

REPO = Path(__file__).resolve().parents[1]

_WORKER = textwrap.dedent(
    r"""
    import json, os, sys
    os.environ["UNSLOTH_FUSED_CE_COMPILE_DISABLE"] = "1"
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor

    rank, world, mode, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    dist.init_process_group("gloo", rank = rank, world_size = world)
    from unsloth_zoo.fused_losses import unsloth_fused_lm_head_loss

    V, H, T = 64, 16, 12
    torch.manual_seed(0)
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(V, H)
            self.proj = torch.nn.Linear(H, H)
            self.lm_head = torch.nn.Linear(H, V, bias = False)
        def forward(self, ids, labels):
            hidden = self.proj(self.embed(ids))
            return unsloth_fused_lm_head_loss(hidden, self.lm_head, labels, vocab_size = V)
    model = Toy()
    full_w = model.lm_head.weight.detach().clone()
    train_head = mode == "train_head"
    model.lm_head.weight.requires_grad_(train_head)

    # accelerate.utils.fsdp_utils.fsdp2_prepare_model: lm_head in its own unit, then the root.
    fully_shard(model.lm_head, reshard_after_forward = False)
    fully_shard(model)

    g = torch.Generator().manual_seed(100 + rank)
    ids = torch.randint(0, V, (2, T), generator = g)
    labels = torch.randint(0, V, (2, T), generator = g)
    labels[:, :3] = -100

    result = {"rank": rank}
    try:
        loss = model(ids, labels)
        loss.backward()
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    else:
        with torch.no_grad():
            ref_model = Toy()
            ref_model.load_state_dict({
                k: (v.full_tensor() if isinstance(v, DTensor) else v)
                for k, v in model.state_dict().items()
            })
        ref_model.lm_head.weight.requires_grad_(train_head)
        hidden = ref_model.proj(ref_model.embed(ids))
        logits = hidden @ ref_model.lm_head.weight.t()
        ref = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, V).float(), labels[:, 1:].reshape(-1), ignore_index = -100)
        if train_head:
            ref.backward()
            ref_grad = ref_model.lm_head.weight.grad.clone()
            dist.all_reduce(ref_grad)
            ref_grad /= world
            grad = model.lm_head.weight.grad
            result["sharded_grad"] = isinstance(grad, DTensor)
            result["grad_err"] = (grad.full_tensor() - ref_grad).abs().max().item()
        result["loss_err"] = abs(loss.item() - ref.item())
        result["weight_is_dtensor_after"] = isinstance(model.lm_head.weight, DTensor)
    with open(f"{out}.{rank}", "w") as f:
        json.dump(result, f)
    dist.destroy_process_group()
    """
)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(tmp_path, mode):
    import json

    script = tmp_path / "worker.py"
    script.write_text(_WORKER)
    env = dict(os.environ)
    env.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(_free_port()),
        CUDA_VISIBLE_DEVICES = "",
        UNSLOTH_IS_PRESENT = "1",
        PYTHONPATH = os.pathsep.join([str(REPO), env.get("PYTHONPATH", "")]),
    )
    out = tmp_path / f"r_{mode}"
    procs = [
        subprocess.Popen(
            [sys.executable, str(script), str(rank), "2", mode, str(out)],
            env = env, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True,
        )
        for rank in range(2)
    ]
    logs = [p.communicate(timeout = 600)[0] for p in procs]
    results = []
    for rank in range(2):
        path = Path(f"{out}.{rank}")
        assert path.exists(), f"rank {rank} wrote nothing:\n{logs[rank][-4000:]}"
        results.append(json.loads(path.read_text()))
    return results


@pytest.mark.parametrize("mode", ["frozen_head", "train_head"])
def test_fused_ce_with_separately_sharded_lm_head(tmp_path, mode):
    for r in _run(tmp_path, mode):
        assert "error" not in r, r
        assert r["loss_err"] < 1e-5, r
        if mode == "train_head":
            # The gradient went through FSDP's reduce-scatter, onto the sharded parameter.
            assert r["sharded_grad"], r
            assert r["grad_err"] < 1e-5, r
