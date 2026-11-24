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
import sys
import time
from types import ModuleType

import pytest

_PARTITION_MODULE = "deepspeed.runtime.zero.partition_parameters"


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _ZeroConfig:
    def __init__(self, stage = 3):
        self.stage = stage

    def is_zero3(self):
        return self.stage == 3


def _worker(rank, port, out, ready):
    import datetime
    import weakref

    import torch
    import torch.distributed as dist

    try:
        # Imports before rendezvous: a slow import on one rank must not eat the first collective's timeout.
        import transformers.integrations.deepspeed as hf_deepspeed
        from unsloth_zoo.rl_replacements import grpo_agree_mini_batch, grpo_forwards_are_collective

        ready.wait()
        # USE_LIBUV=0: some Windows torch builds default the TCPStore to libuv without shipping it.
        os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port), USE_LIBUV = "0")
        dist.init_process_group("gloo", rank = rank, world_size = 2, timeout = datetime.timedelta(seconds = 60))

        model = torch.nn.Sequential(torch.nn.Linear(2, 2))  # no FSDP2 markers: ZeRO-3 partitions in place
        # Asked once before DeepSpeed is live: the answer must not stick.
        before = grpo_forwards_are_collective(model)
        zero2 = _ZeroConfig(stage = 2)
        hf_deepspeed._hf_deepspeed_config_weak_ref = weakref.ref(zero2)
        under_zero2 = grpo_forwards_are_collective(model)
        config = _ZeroConfig()
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

        # A partitioned model stays collective with the global config gone (or naming another stage).
        sys.modules.setdefault(_PARTITION_MODULE, ModuleType(_PARTITION_MODULE))
        next(model.parameters()).ds_id = 0
        hf_deepspeed._hf_deepspeed_config_weak_ref = weakref.ref(zero2)
        partitioned = grpo_forwards_are_collective(model)
        hf_deepspeed._hf_deepspeed_config_weak_ref = None
        out.put((rank, (before, under_zero2, collective, [int(c) for c in counts], after, partitioned)))
    except Exception as exc:
        out.put((rank, repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_zero3_ranks_run_equal_forward_counts():
    dist = pytest.importorskip("torch.distributed")
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("requires an available torch.distributed gloo backend")
    ctx = multiprocessing.get_context("spawn")
    out = ctx.Queue()
    ready = ctx.Barrier(2, timeout = 120)
    port = _port()
    procs = [ctx.Process(target = _worker, args = (r, port, out, ready)) for r in range(2)]
    started = []
    try:
        for p in procs:
            p.start()
            started.append(p)
        deadline = time.monotonic() + 240
        for p in started:
            p.join(timeout = max(0, deadline - time.monotonic()))
        assert [p.exitcode for p in started] == [0, 0]
        results = dict(out.get(timeout = 5) for _ in started)
        expected = (False, False, True, [2, 2], False, True)
        assert results == {0: expected, 1: expected}, results
    finally:
        for p in started:
            if p.is_alive():
                p.kill()
        for p in started:
            p.join()
        out.close()


def _fake_two_ranks(monkeypatch, world_size = 2):
    import torch.distributed as dist
    import transformers.integrations.deepspeed as hf_deepspeed

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda *a, **k: world_size)
    monkeypatch.setattr(hf_deepspeed, "_hf_deepspeed_config_weak_ref", None, raising = False)
    return hf_deepspeed


def test_partitioned_model_is_collective_without_live_zero3_config(monkeypatch):
    import weakref

    import torch
    from unsloth_zoo.rl_replacements import grpo_forwards_are_collective

    hf_deepspeed = _fake_two_ranks(monkeypatch)
    monkeypatch.setitem(sys.modules, _PARTITION_MODULE, ModuleType(_PARTITION_MODULE))
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    assert not grpo_forwards_are_collective(model)
    for index, parameter in enumerate(model.parameters()):  # partitioned after the first (negative) call
        parameter.ds_id = index
    assert grpo_forwards_are_collective(model)
    zero2 = _ZeroConfig(stage = 2)
    monkeypatch.setattr(hf_deepspeed, "_hf_deepspeed_config_weak_ref", weakref.ref(zero2))
    assert grpo_forwards_are_collective(model)


def test_no_parameter_scan_until_deepspeed_is_loaded(monkeypatch):
    import torch
    from unsloth_zoo.rl_replacements import grpo_forwards_are_collective

    _fake_two_ranks(monkeypatch)
    monkeypatch.delitem(sys.modules, _PARTITION_MODULE, raising = False)

    class _NoScan(torch.nn.Module):
        def parameters(self, *args, **kwargs):
            raise AssertionError("parameters scanned on a run without DeepSpeed")

    assert not grpo_forwards_are_collective(_NoScan())


def test_single_rank_zero3_is_not_collective(monkeypatch):
    import weakref

    import torch
    from unsloth_zoo.rl_replacements import grpo_forwards_are_collective

    hf_deepspeed = _fake_two_ranks(monkeypatch, world_size = 1)
    monkeypatch.setitem(sys.modules, _PARTITION_MODULE, ModuleType(_PARTITION_MODULE))
    config = _ZeroConfig()
    monkeypatch.setattr(hf_deepspeed, "_hf_deepspeed_config_weak_ref", weakref.ref(config))
    model = torch.nn.Linear(2, 2)
    model.weight.ds_id = 0
    assert not grpo_forwards_are_collective(model)
