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

"""DeepSpeed ZeRO-3 GRPO: Core's old / ref scoring asks grpo_forwards_are_collective whether to agree the
mini-batch count across ranks. It only knew FSDP2, so under ZeRO-3 a rank with longer sequences ran more
forwards than its peer and the parameter all-gathers hung."""

from __future__ import annotations

import multiprocessing
import os
import socket


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Zero3Config:
    def is_zero3(self):
        return True


def _worker(rank, port, out):
    import datetime
    import weakref

    import torch
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port))
    dist.init_process_group("gloo", rank = rank, world_size = 2, timeout = datetime.timedelta(seconds = 60))
    try:
        import transformers.integrations.deepspeed as hf_deepspeed
        from unsloth_zoo.rl_replacements import grpo_agree_mini_batch, grpo_forwards_are_collective

        model = torch.nn.Sequential(torch.nn.Linear(2, 2))  # no FSDP2 markers: ZeRO-3 partitions in place
        # Asked once before DeepSpeed is live: the answer must not stick.
        before = grpo_forwards_are_collective(model)
        config = _Zero3Config()
        hf_deepspeed._hf_deepspeed_config_weak_ref = weakref.ref(config)

        # Core's old / ref path (unsloth/models/rl_replacements.py): split count sized from this
        # rank's sequence length, agreed only when the forwards are collective.
        B = 1 if rank == 0 else 2
        collective = grpo_forwards_are_collective(model)
        if collective:
            B = grpo_agree_mini_batch(B, torch.device("cpu"))
        counts = [torch.zeros(1, dtype = torch.int64) for _ in range(2)]
        dist.all_gather(counts, torch.tensor([B], dtype = torch.int64))
        hf_deepspeed._hf_deepspeed_config_weak_ref = None
        after = grpo_forwards_are_collective(model)
        out.put((rank, (before, collective, [int(c) for c in counts], after)))
    except Exception as exc:
        out.put((rank, repr(exc)))
    finally:
        dist.destroy_process_group()


def test_zero3_ranks_run_equal_forward_counts():
    ctx = multiprocessing.get_context("spawn")
    out = ctx.Queue()
    port = _port()
    procs = [ctx.Process(target = _worker, args = (r, port, out)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
    assert [p.exitcode for p in procs] == [0, 0]
    results = dict(out.get(timeout = 5) for _ in range(2))
    expected = (False, True, [2, 2], False)
    assert results == {0: expected, 1: expected}, results

