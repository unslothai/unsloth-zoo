# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present the Unsloth team. All rights reserved.

"""``train_on_responses_only`` sizes its auto worker count from what this process may use.

It used ``min(cpu_count + 4, 64)`` capped by the HOST's available memory, so a job pinned to two cores (taskset,
Slurm, a container CPU quota) forked up to 64 tokenizer workers onto them, and a memory-limited cgroup on a large box
was sized from the box. It now goes through ``unsloth_zoo.dataset_num_proc``, like every other map site.

CPU-pure and offline.
"""

from __future__ import annotations

import multiprocessing
import os
import types

import psutil
import pytest
from datasets import Dataset

import unsloth_zoo.dataset_num_proc as dnp
from unsloth_zoo.dataset_utils import train_on_responses_only

USER_ID, ASSISTANT_ID = 1, 2
ROW = [USER_ID, 10, 11, ASSISTANT_ID, 20, 21]


class _StubTokenizer:
    def __call__(self, text, add_special_tokens = False):
        ids = {"<|user|>": [USER_ID], "<|assistant|>": [ASSISTANT_ID]}.get(text)
        return types.SimpleNamespace(input_ids = ids if ids is not None else [ord(c) for c in text])

    def convert_ids_to_tokens(self, ids):
        return [str(i) for i in ids]


def _requested_num_proc(monkeypatch, *, cpus, host_gb, cgroup_free_gb = None):
    monkeypatch.delenv(dnp.NUM_PROC_ENV_VAR, raising = False)
    dnp.reset_warning_state()
    # A forking Linux host with many CPUs and plenty of host RAM, as both the old and the new code see it.
    monkeypatch.setattr(multiprocessing, "get_start_method", lambda allow_none = False: "fork")
    monkeypatch.setattr(dnp, "multiprocessing_start_method", lambda: "fork")
    monkeypatch.setattr(dnp.sys, "platform", "linux")
    monkeypatch.setattr(psutil, "cpu_count", lambda *a, **k: 64)
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: types.SimpleNamespace(available = host_gb * 1024**3)
    )
    # What this process may actually use.
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(cpus)), raising = False)
    monkeypatch.setattr(dnp, "_cgroup_cpu_quota", lambda: None)
    monkeypatch.setattr(
        dnp,
        "_cgroup_free_bytes",
        lambda: None if cgroup_free_gb is None else int(cgroup_free_gb * 1024**3),
    )

    requested = []
    real_map = Dataset.map

    def _spy(self, *args, **kwargs):
        requested.append(kwargs.get("num_proc"))
        kwargs["num_proc"] = None  # assert on the request, never actually fork here
        return real_map(self, *args, **kwargs)

    monkeypatch.setattr(Dataset, "map", _spy)
    trainer = types.SimpleNamespace(
        # Over the 5,000-row small-split guard, so the auto count is what reaches map().
        train_dataset = Dataset.from_dict({"input_ids": [list(ROW)] * 5_001}),
        eval_dataset = None,
        args = types.SimpleNamespace(packing = False),
    )
    train_on_responses_only(trainer, "<|user|>", "<|assistant|>", tokenizer = _StubTokenizer())
    assert requested, "Dataset.map was never called"
    return requested


@pytest.mark.parametrize("cpus, expected", [(2, 2), (6, 3), (32, dnp.AUTO_NUM_PROC_CAP)])
def test_auto_count_follows_cpu_affinity(monkeypatch, cpus, expected):
    assert _requested_num_proc(monkeypatch, cpus = cpus, host_gb = 1024) == [expected]


def test_single_allowed_cpu_maps_in_process(monkeypatch):
    assert _requested_num_proc(monkeypatch, cpus = 1, host_gb = 1024) == [None]


def test_cgroup_memory_limit_beats_host_memory(monkeypatch):
    # 3 GB free under the cgroup on a 1 TB host: half of it covers one ~1 GB worker, so no pool at all.
    assert _requested_num_proc(monkeypatch, cpus = 32, host_gb = 1024, cgroup_free_gb = 3) == [None]
    # 8 GB free: four workers.
    assert _requested_num_proc(monkeypatch, cpus = 32, host_gb = 1024, cgroup_free_gb = 8) == [4]
