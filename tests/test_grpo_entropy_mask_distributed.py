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

"""top_entropy_quantile under TRL 0.22 get_high_entropy_mask: a rank with no valid tokens returns before the gather
other ranks enter; 2 gloo ranks must both finish and match the TRL >= 0.24 result."""

import datetime
import os
import socket
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import unsloth_zoo.rl_replacements as rr


class _Accelerator:
    """The two collectives get_high_entropy_mask uses, on torch.distributed (pads with pad_index)."""

    def __init__(self):
        self.num_processes = dist.get_world_size()

    def pad_across_processes(self, tensor, dim = 0, pad_index = 0):
        size = torch.tensor([tensor.shape[0]])
        sizes = [torch.zeros_like(size) for _ in range(self.num_processes)]
        dist.all_gather(sizes, size)
        out = tensor.new_full((int(max(sizes)),), pad_index)
        out[:tensor.shape[0]] = tensor
        return out

    def gather(self, tensor):
        parts = [torch.zeros_like(tensor) for _ in range(self.num_processes)]
        dist.all_gather(parts, tensor.contiguous())
        return torch.cat(parts)


# TRL v0.22.2 GRPOTrainer.get_high_entropy_mask, verbatim body
def _trl_0_22_mask(self, entropies, mask, threshold, accelerator = None):
    non_pad_entropies = entropies[mask.bool()].float()
    if non_pad_entropies.numel() == 0:
        return torch.zeros_like(entropies, dtype=torch.bool)
    all_non_pad_entropies = self.accelerator.gather(non_pad_entropies)
    entropy_threshold = torch.quantile(all_non_pad_entropies, threshold)
    masked_entropies = entropies * mask.float()
    entropy_mask = masked_entropies >= entropy_threshold
    return entropy_mask & mask.bool()


# TRL >= 0.24 GRPOTrainer.get_high_entropy_mask, verbatim body
def _trl_0_24_mask(self, entropies, mask, threshold):
    local = entropies[mask.bool()].float()
    pad_value = -1e9
    padded = self.accelerator.pad_across_processes(local, dim=0, pad_index=pad_value)
    gathered = self.accelerator.gather(padded)
    gathered = gathered[gathered != pad_value]
    if gathered.numel() == 0:
        return torch.zeros_like(entropies, dtype=torch.bool)
    entropy_threshold = torch.quantile(gathered, threshold)
    masked_entropies = entropies * mask.float()
    entropy_mask = masked_entropies >= entropy_threshold
    return entropy_mask & mask.bool()


ENTROPIES = [
    torch.tensor([[0.4, 0.7, 0.9], [0.1, 0.2, 0.3]]),
    torch.tensor([[1.5, 0.6, 2.0], [0.8, 0.05, 1.1]]),
]
MASKS = [
    torch.zeros(2, 3),                          # rank 0: every completion masked (e.g. truncated)
    torch.tensor([[1., 1., 1.], [1., 0., 1.]]),
]


def _worker(rank, port, method_name, empty_rank0, out):
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(port)
    dist.init_process_group("gloo", rank = rank, world_size = 2, timeout = datetime.timedelta(seconds = 20))
    try:
        trainer = types.SimpleNamespace(accelerator = _Accelerator())
        trainer.get_high_entropy_mask = globals()[method_name].__get__(trainer)
        masks = MASKS if empty_rank0 else [MASKS[1], torch.ones(2, 3)]  # 5 vs 6 valid tokens
        got = rr.grpo_high_entropy_mask(trainer, ENTROPIES[rank], masks[rank], 0.5)
        want = _trl_0_24_mask(trainer, ENTROPIES[rank], masks[rank], 0.5)
        dist.barrier()
        out[rank] = bool(torch.equal(got, want))
    finally:
        dist.destroy_process_group()


def _run(method_name, empty_rank0 = True):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    out = mp.Manager().dict()
    ctx = mp.spawn(_worker, args = (port, method_name, empty_rank0, out), nprocs = 2, join = False)
    if not ctx.join(timeout = 120):
        ctx.join(timeout = 120)
    return dict(out)


@pytest.mark.parametrize("method_name", ["_trl_0_22_mask", "_trl_0_24_mask"])
def test_empty_rank_keeps_every_rank_in_the_collective(method_name):
    assert _run(method_name) == {0: True, 1: True}


def test_uneven_token_counts_match_padded_gather():
    assert _run("_trl_0_22_mask", empty_rank0 = False) == {0: True, 1: True}


def test_single_process_calls_the_trainer_method_unchanged():
    calls = []
    trainer = types.SimpleNamespace(accelerator = types.SimpleNamespace(num_processes = 1))
    trainer.get_high_entropy_mask = lambda *args: calls.append(args) or "trl"
    assert rr.grpo_high_entropy_mask(trainer, ENTROPIES[0], MASKS[1], 0.5) == "trl"
    assert len(calls) == 1


def test_detects_which_trl_methods_pad():
    assert rr._high_entropy_mask_pads(_trl_0_24_mask) is True
    assert rr._high_entropy_mask_pads(_trl_0_22_mask) is False
    assert rr._high_entropy_mask_pads(lambda *args: None) is True
