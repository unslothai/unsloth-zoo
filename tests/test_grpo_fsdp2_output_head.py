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

"""unsloth-zoo#1850: grpo_accumulated_loss under FSDP2 must read the output head after the forward.

FSDP2 swaps a module's sharded DTensor parameters for plain tensors in its pre-forward hook and
back after backward. Capturing ``output_embeddings.weight`` before the forward handed the matmul a
DTensor ("aten.mm.default got mixed torch.Tensor and DTensor"). Two gloo ranks run two
accumulation steps; loss and reduced gradients must match an unsharded reference.
"""

import json
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
    import json, os, sys, types
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor

    rank, world, layout, bias, train_head, out = sys.argv[1:7]
    rank, world = int(rank), int(world)
    bias, train_head = bias == "1", train_head == "1"
    dist.init_process_group("gloo", rank = rank, world_size = world)
    import unsloth_zoo.rl_replacements as rl
    from transformers.modeling_outputs import CausalLMOutput

    V, H = 32, 8
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.embed = torch.nn.Embedding(V, H)
            self.proj = torch.nn.Linear(H, H)
            self.lm_head = torch.nn.Linear(H, V, bias = bias)
            self.lm_head.weight.requires_grad_(train_head)
            if bias:
                self.lm_head.bias.requires_grad_(train_head)
        def get_output_embeddings(self):
            return self.lm_head
        def forward(self, input_ids = None, attention_mask = None, logits_to_keep = None, **kwargs):
            hidden = self.proj(self.embed(input_ids))
            if logits_to_keep:
                hidden = hidden[:, -logits_to_keep:]
            # A pytree output: FSDP2 hooks its backward through the tensors it can find.
            return CausalLMOutput(logits = hidden)

    def trainer_for(model):
        return types.SimpleNamespace(
            args = types.SimpleNamespace(unsloth_grpo_mini_batch = 1, unsloth_logit_chunk_multiplier = 1),
            processing_class = types.SimpleNamespace(pad_token_id = 0),
            model = model,
            accelerator = types.SimpleNamespace(unwrap_model = lambda m, keep_fp32_wrapper = False: m, scaler = None),
            use_vllm = False, _autocast_dtype = None, beta = 0.0,
        )

    g = torch.Generator().manual_seed(100 + rank)
    batches = []
    for _ in range(2):
        ids = torch.randint(1, V, (3, 7), generator = g)
        ids[2, 0] = 0  # one left-padded prompt
        batches.append((ids, torch.randn(3, generator = g)))

    def run(model, trainer):
        losses = []
        for ids, adv in batches:
            mask = torch.ones(3, 4, dtype = torch.long)
            loss = rl.grpo_accumulated_loss(
                trainer, ids, (ids != 0).long(), 4, mask, adv, None, None,
                loss_type = "dapo", num_items_in_batch = int(mask.sum()), num_processes = 1,
                current_gradient_accumulation_steps = 2, max_completion_length = 4,
            )[0]
            (loss / 2).backward()
            losses.append(loss.item())
        return losses

    model = Toy()
    if layout == "head":
        # accelerate.utils.fsdp_utils.fsdp2_prepare_model: lm_head in its own unit, then the root.
        fully_shard(model.lm_head, reshard_after_forward = False)
    fully_shard(model)

    result = {"rank": rank}
    try:
        losses = run(model, trainer_for(model))
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    else:
        ref = Toy()
        ref_losses = run(ref, trainer_for(ref))
        result["loss_err"] = max(abs(a - b) for a, b in zip(losses, ref_losses))
        grad_err = 0.0
        for (name, p), (_, rp) in zip(model.named_parameters(), ref.named_parameters()):
            assert (p.grad is None) == (rp.grad is None), name
            if p.grad is None:
                continue
            rg = rp.grad.clone()
            dist.all_reduce(rg)
            rg /= world
            got = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
            grad_err = max(grad_err, (got - rg).abs().max().item())
        result["grad_err"] = grad_err
        result["head_grad"] = model.lm_head.weight.grad is not None
    with open(f"{out}.{rank}", "w") as f:
        json.dump(result, f)
    dist.destroy_process_group()
    """
)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize(
    "layout, bias, train_head",
    [("root", False, False), ("root", True, True), ("head", True, False), ("head", False, True)],
)
def test_grpo_loss_reads_the_unsharded_output_head(tmp_path, layout, bias, train_head):
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
    args = [layout, str(int(bias)), str(int(train_head)), str(out)]
    procs = [
        subprocess.Popen(
            [sys.executable, str(script), str(rank), "2", *args],
            env = env, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True,
        )
        for rank in range(2)
    ]
    logs = [p.communicate(timeout = 600)[0] for p in procs]
    for rank in range(2):
        path = Path(f"{out}.{rank}")
        assert path.exists(), f"rank {rank} wrote nothing:\n{logs[rank][-4000:]}"
        result = json.loads(path.read_text())
        assert "error" not in result, result["error"]
        assert result["loss_err"] < 1e-5, result
        assert result["grad_err"] < 1e-5, result
        assert result["head_grad"] == train_head, result
