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

"""Under FSDP2's sharded layout every forward is collective, so GRPO ranks that ran a different number of
forwards hung (2xT4, Gemma3 1B GRPO, UNSLOTH_FSDP2_LAYOUT=sharded: one rank all-gathered a block while the other
was in backward). The mini batch is sized from the rank's own sequence length, so on such a model the ranks
agree on one (unsloth turns packing / PrefixGrouper off there, whose self-verify forwards are also per rank)."""

from __future__ import annotations

import inspect
import multiprocessing
import os
import socket


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _Model(reshard):
    import torch

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Sequential(torch.nn.Linear(2, 2)))
    model[0]._unsloth_reshard_after_forward = False  # the root and last block keep their weights
    if reshard is not None:
        model[1][0]._unsloth_reshard_after_forward = reshard
    return model


def _worker(rank, port, out):
    import datetime

    import torch
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port))
    dist.init_process_group("gloo", rank = rank, world_size = 2, timeout = datetime.timedelta(seconds = 60))
    try:
        from unsloth_zoo.rl_replacements import grpo_agree_mini_batch, grpo_forwards_are_collective

        sharded, fit, plain = _Model(True), _Model(False), _Model(None)
        out.put((rank, (
            grpo_agree_mini_batch(4 if rank == 0 else 2, torch.device("cpu")),
            grpo_forwards_are_collective(sharded),
            grpo_forwards_are_collective(fit),
            grpo_forwards_are_collective(plain),
        )))
    except Exception as exc:
        out.put((rank, repr(exc)))
    finally:
        dist.destroy_process_group()


def test_ranks_agree_on_the_smallest_mini_batch():
    ctx = multiprocessing.get_context("spawn")
    out = ctx.Queue()
    port = _port()
    procs = [ctx.Process(target = _worker, args = (r, port, out)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
    assert [p.exitcode for p in procs] == [0, 0]
    assert dict(out.get(timeout = 5) for _ in range(2)) == {0: (2, True, False, False), 1: (2, True, False, False)}


def test_single_process_forwards_are_never_collective():
    from unsloth_zoo.rl_replacements import grpo_forwards_are_collective

    assert grpo_forwards_are_collective(_Model(True)) is False


def test_grpo_accumulated_loss_agrees_on_collective_forwards():
    from unsloth_zoo.rl_replacements import grpo_accumulated_loss

    source = inspect.getsource(grpo_accumulated_loss)
    tuned = source.index("autotune_batch_and_chunks(")
    assert source.index("grpo_agree_mini_batch(B", tuned) > tuned
    collective = source.index("_collective_forwards = grpo_forwards_are_collective(")
    assert collective < tuned
