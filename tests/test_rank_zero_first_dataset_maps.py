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

"""Under DDP, dataset preparation on a file-backed split runs on rank 0 first and the other ranks
load its Arrow cache; an in-memory split (no cache file) keeps every rank running at once.
Two CPU ranks over gloo."""

import json
import os
import socket

import pytest

torch = pytest.importorskip("torch")
datasets = pytest.importorskip("datasets")
import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _double(batch):
    # Runs only when the map is computed, not when its cache file is loaded.
    with open(os.environ["S7_CALLS_FILE"], "a") as f:
        f.write("1")
    return {"y": [2 * x for x in batch["x"]]}


def _worker(rank, world, port, kind, data_dir, out_dir):
    os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port))
    dist.init_process_group("gloo", rank = rank, world_size = world)
    from datasets.arrow_dataset import Dataset
    from unsloth_zoo.dataset_utils import rank_zero_first, rank_zero_first_prepare

    calls_file = os.path.join(out_dir, f"calls{rank}")
    os.environ["S7_CALLS_FILE"] = calls_file
    barriers = []
    real_barrier = dist.barrier

    def counting_barrier(*args, **kwargs):
        barriers.append(1)
        return real_barrier(*args, **kwargs)

    dist.barrier = counting_barrier
    if kind == "file":
        dataset = datasets.load_from_disk(data_dir)
    else:
        dataset = Dataset.from_dict({"x": list(range(64))})
    record = dict(rank = rank, file_backed = bool(dataset.cache_files))
    try:
        if kind == "raise":
            dataset = datasets.load_from_disk(data_dir)
            try:
                with rank_zero_first(dataset):
                    raise ValueError("bad split")
            except ValueError:
                record["raised"] = True
        else:
            prepare = rank_zero_first_prepare(lambda ds, name: ds.map(_double, batched = True))
            out = prepare(dataset, "train")
            record["y"] = out["y"][:4]
            record["wrapped_twice_is_same"] = rank_zero_first_prepare(prepare) is prepare
    finally:
        dist.barrier = real_barrier
    record["computed"] = len(open(calls_file).read()) if os.path.exists(calls_file) else 0
    record["barriers"] = len(barriers)
    with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as f:
        json.dump(record, f)
    dist.barrier()
    dist.destroy_process_group()


def _run(kind, tmp_path):
    data_dir = str(tmp_path / "data")
    datasets.Dataset.from_dict({"x": list(range(64))}).save_to_disk(data_dir)
    out_dir = tmp_path / kind
    out_dir.mkdir()
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target = _worker, args = (rank, 2, PORT[kind], kind, data_dir, str(out_dir)))
        for rank in range(2)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
    for p in procs:
        if p.is_alive():
            p.kill()
            pytest.fail(f"{kind}: a rank hung")
        assert p.exitcode == 0, f"{kind}: rank exited with {p.exitcode}"
    return [json.loads((out_dir / f"rank{rank}.json").read_text()) for rank in range(2)]


PORT = {}


@pytest.fixture(autouse = True)
def _ports():
    for kind in ("file", "memory", "raise"):
        PORT[kind] = _free_port()


def test_file_backed_split_is_computed_once_and_read_from_cache(tmp_path):
    rank0, rank1 = _run("file", tmp_path)
    assert rank0["file_backed"] and rank1["file_backed"]
    assert rank0["computed"] == 1, rank0
    assert rank1["computed"] == 0, "rank 1 recomputed the map instead of loading rank 0's cache"
    assert rank0["y"] == rank1["y"] == [0, 2, 4, 6]
    assert rank0["wrapped_twice_is_same"] and rank1["wrapped_twice_is_same"]


def test_in_memory_split_keeps_ranks_concurrent(tmp_path):
    rank0, rank1 = _run("memory", tmp_path)
    assert not rank0["file_backed"]
    assert rank0["barriers"] == rank1["barriers"] == 0, "an in-memory split was serialised"
    assert rank0["computed"] == rank1["computed"] == 1
    assert rank0["y"] == rank1["y"] == [0, 2, 4, 6]


def test_rank_zero_raising_does_not_hang_the_others(tmp_path):
    rank0, rank1 = _run("raise", tmp_path)
    assert rank0["raised"] and rank1["raised"]
