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
"""Tiled MLP x expert parallel (gloo CPU, 2 ranks): ranks tile differently (8 vs 4), so EP collectives would
mismatch (gloo hang / RecursionError). EP blocks must run untiled in any order; dense modules stay tiled."""

import datetime
import multiprocessing
import os
import tempfile
import time

import pytest
import torch

H = 16
ROWS = (14, 10)  # rows:3 gives 8 tiles on rank 0 and 4 on rank 1
EP_ATTR = "_unsloth_ep"


class DenseMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(H, 2 * H)
        self.down = torch.nn.Linear(2 * H, H)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.down(torch.nn.functional.silu(self.up(x)))


class ToyExpertParallelMoE(torch.nn.Module):
    """One collective per call, like ep_native._ep_experts' count all_to_all."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(H, H)
        self.calls = 0

    def forward(self, x):
        import torch.distributed as dist

        self.calls += 1
        if dist.is_initialized() and getattr(self.proj.weight, EP_ATTR, None) is not None:
            rows = torch.tensor([x.reshape(-1, H).shape[0]])
            dist.all_reduce(rows)
        return self.proj(x)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.dense = DenseMLP()
        self.moe = ToyExpertParallelMoE()

    def forward(self, x):
        x = x + self.dense(x)
        return x + self.moe(x)


def _apply_ep(net):
    # Mimics apply_expert_parallel: tag expert params, wrap block.forward.
    for p in net.moe.parameters():
        setattr(p, EP_ATTR, ("ep layout",))
    inner = net.moe.forward

    def forward(*args, **kwargs):
        return inner(*args, **kwargs)

    net.moe.forward = forward


def _case(rank, order):
    import unsloth_zoo.tiling_registry as tr

    x = torch.randn(1, ROWS[rank], H, generator = torch.Generator().manual_seed(rank))
    ref = Net()
    _apply_ep(ref)
    want = ref(x)
    want.sum().backward()
    want_grads = {k: p.grad.clone() for k, p in ref.named_parameters()}

    net = Net()
    force = ["densemlp", "toyexpertparallelmoe"]
    if order == "ep_then_tile":
        _apply_ep(net)
        report = tr.install(net, "rows:3", force = force, verbose = False)
    else:
        report = tr.install(net, "rows:3", force = force, verbose = False)
        _apply_ep(net)
        if order == "tile_then_ep_revalidate":
            report = tr.revalidate(net, verbose = False)
    got = net(x)
    got.sum().backward()
    return {
        "tiled": sorted(name for name, _, _ in report.tiled),
        "moe_calls": net.moe.calls,
        "dense_calls": net.dense.calls,
        "out": float((got - want).abs().max()),
        "grads": max(float((p.grad - want_grads[k]).abs().max()) for k, p in net.named_parameters()),
        "zoo": tr.__file__,
    }


def _worker(rank, world, tmp, order):
    import torch.distributed as dist

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    dist.init_process_group(
        "gloo",
        store = dist.FileStore(os.path.join(tmp, "rendezvous"), world),
        rank = rank,
        world_size = world,
        timeout = datetime.timedelta(seconds = 20),
    )
    try:
        res = _case(rank, order)
    except BaseException as exc:
        res = {"error": f"{type(exc).__name__}: {exc}"[:500]}
    torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    try:
        dist.destroy_process_group()
    except Exception:
        pass


@pytest.mark.parametrize("order", ["tile_then_ep", "tile_then_ep_revalidate", "ep_then_tile"])
def test_expert_parallel_block_runs_untiled_on_every_rank(order):
    import unsloth_zoo.tiling_registry as tr

    ctx = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmp:
        procs = [ctx.Process(target = _worker, args = (r, 2, tmp, order)) for r in range(2)]
        for p in procs:
            p.start()
        deadline = time.monotonic() + 120
        while any(p.is_alive() for p in procs) and time.monotonic() < deadline:
            time.sleep(0.1)
        for p in procs:
            if p.is_alive():
                p.terminate()
            p.join(10)
        results = []
        for r in range(2):
            path = os.path.join(tmp, f"result_{r}.pt")
            results.append(torch.load(path, weights_only = False) if os.path.exists(path) else None)
    assert all(r is not None and "error" not in r for r in results), results
    assert all(r["zoo"] == tr.__file__ for r in results), (results, tr.__file__)
    for rank, r in enumerate(results):
        assert r["moe_calls"] == 1, (rank, r)
        assert r["dense_calls"] > 2, (rank, r)  # dense stays tiled
        assert "dense" in r["tiled"], (rank, r)
        if order != "tile_then_ep":  # tiled at load, before expert parallel existed: the report predates it
            assert "moe" not in r["tiled"], (rank, r)
        assert r["out"] < 1e-5 and r["grads"] < 1e-5, (rank, r)
    assert results[0]["dense_calls"] > results[1]["dense_calls"], results  # 8 vs 4 tiles: the ranks differ
