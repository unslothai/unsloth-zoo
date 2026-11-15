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

"""GRPO's per-call mini batch is sized from each rank's own sequence length. Under FSDP2's sharded layout every
forward is collective, so ranks that ran a different number of forward chunks hung (2xT4, Gemma3 1B GRPO,
UNSLOTH_FSDP2_LAYOUT=sharded: one rank all-gathered a block while the other gathered loss metrics). The ranks
must agree on one mini batch."""

from __future__ import annotations

import inspect
import multiprocessing
import os
import socket


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, port, out):
    import datetime

    import torch
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port))
    dist.init_process_group("gloo", rank = rank, world_size = 2, timeout = datetime.timedelta(seconds = 60))
    try:
        from unsloth_zoo.rl_replacements import _grpo_agree_mini_batch

        out.put((rank, _grpo_agree_mini_batch(4 if rank == 0 else 2, torch.device("cpu"))))
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
    assert dict(out.get(timeout = 5) for _ in range(2)) == {0: 2, 1: 2}


def test_single_process_keeps_its_mini_batch():
    import torch
    from unsloth_zoo.rl_replacements import _grpo_agree_mini_batch

    assert _grpo_agree_mini_batch(3, torch.device("cpu")) == 3


def test_grpo_accumulated_loss_agrees_after_autotuning():
    from unsloth_zoo.rl_replacements import grpo_accumulated_loss

    source = inspect.getsource(grpo_accumulated_loss)
    tuned = source.index("autotune_batch_and_chunks(")
    assert source.index("_grpo_agree_mini_batch(B", tuned) > tuned
