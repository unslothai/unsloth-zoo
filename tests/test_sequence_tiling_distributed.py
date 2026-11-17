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

import json
import os
import sys
import tempfile

import pytest
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _sequence_tiling_models import GatedMLP, lora_mlp, H  # noqa: E402

N_TILES = 4


def install(module):
    from types import MethodType
    from unsloth_zoo.sequence_tiling import tiled_call
    inner = type(module).forward.__get__(module)
    module.forward = MethodType(lambda self, x: tiled_call(inner, x, n_tiles = N_TILES, needs_rng = False), module)
    return module


class Net(torch.nn.Module):
    def __init__(self, tiled, seed = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.embed = torch.nn.Embedding(64, H)
        self.embed.weight.requires_grad_(False)
        self.mlp = lora_mlp(seed)
        self.head = torch.nn.Linear(H, 8)
        self.tiled = tiled
        if tiled == "new":
            install(self.mlp)

    def forward(self, ids):
        h = self.embed(ids)
        if self.tiled == "new_module_fn":
            from unsloth_zoo.sequence_tiling import tiled_call
            h = h + tiled_call(self.mlp, h, n_tiles = N_TILES, needs_rng = False)
        elif self.tiled == "new":
            h = h + self.mlp(h)
        else:
            h = h + self.mlp(h)
        return self.head(h).float().pow(2).mean()


def _data(rank):
    g = torch.Generator().manual_seed(100 + rank)
    return torch.randint(0, 64, (2, 37), generator = g)


def _reference(world, device):
    ref = Net(None).to(device)
    acc = {n: torch.zeros_like(p) for n, p in ref.named_parameters() if p.requires_grad}
    for r in range(world):
        ref.zero_grad(set_to_none = True)
        ref(_data(r).to(device)).backward()
        for n, p in ref.named_parameters():
            if p.requires_grad: acc[n] += p.grad / world
    return acc


def _hook_counter(model):
    fired = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.register_post_accumulate_grad_hook(lambda p, n = n: fired.__setitem__(n, fired.get(n, 0) + 1))
    return fired


class _CommCount:
    NAMES = tuple(k for k in ("all_reduce", "all_gather_into_tensor", "reduce_scatter_tensor", "all_gather",
                              "all_gather_single", "reduce_scatter_single") if hasattr(dist, k))

    def __enter__(self):
        self.counts = {k: 0 for k in self.NAMES}
        self.saved = {}
        for k in self.NAMES:
            real = getattr(dist, k)
            self.saved[k] = real
            def wrap(*a, _k = k, _real = real, **kw):
                self.counts[_k] += 1
                return _real(*a, **kw)
            setattr(dist, k, wrap)
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items(): setattr(dist, k, v)


def _close(a, b, what):
    torch.testing.assert_close(a, b, rtol = 1e-5, atol = 1e-6, msg = what)


def cell_ddp(rank, world, device, out):
    from torch.nn.parallel import DistributedDataParallel as DDP
    res = {}
    ref = _reference(world, device)
    for mode in ("untiled", "new"):
        model = Net(None if mode == "untiled" else "new").to(device)
        ddp = DDP(model, device_ids = None if device.type == "cpu" else [device.index], bucket_cap_mb = 0.0001)
        reduces = []
        def hook(state, bucket):
            reduces.append(1)
            fut = dist.all_reduce(bucket.buffer().div_(world), async_op = True).get_future()
            return fut.then(lambda f: f.value()[0])
        ddp.register_comm_hook(None, hook)
        fired = _hook_counter(model)
        for step in range(2):
            fired.clear()
            model.zero_grad(set_to_none = True)
            ddp(_data(rank).to(device)).backward()
            for n, p in model.named_parameters():
                if p.requires_grad: _close(p.grad, ref[n], f"ddp {mode} step {step} {n}")
            assert fired and all(v == 1 for v in fired.values()), fired
        res[mode] = {"hooks_max": max(fired.values()), "bucket_allreduces": len(reduces)}
    assert res["new"]["bucket_allreduces"] == res["untiled"]["bucket_allreduces"]
    out["ddp"] = res


def cell_fsdp2(rank, world, device, out):
    try:
        from torch.distributed.fsdp import fully_shard
    except ImportError:
        from torch.distributed._composable.fsdp import fully_shard
    from torch.distributed.device_mesh import init_device_mesh
    mesh = init_device_mesh(device.type, (world,))
    ref = _reference(world, device)
    res = {}
    for mode in ("untiled", "new", "new_module_fn"):
        for wrap in ("mlp_and_root", "root"):
            model = Net(None if mode == "untiled" else mode).to(device)
            if wrap == "mlp_and_root":
                fully_shard(model.mlp, mesh = mesh)
            fully_shard(model, mesh = mesh)
            with _CommCount() as cc:
                model(_data(rank).to(device)).backward()
            for n, p in model.named_parameters():
                if p.requires_grad:
                    _close(p.grad.full_tensor(), ref[n], f"fsdp2 {mode} {wrap} {n}")
            res[f"{mode}/{wrap}"] = dict(cc.counts)
    for wrap in ("mlp_and_root", "root"):
        assert res[f"new/{wrap}"] == res[f"untiled/{wrap}"], res
    out["fsdp2"] = res


def cell_tp(rank, world, device, out):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
    import torch.distributed._functional_collectives as funcol
    from unsloth_zoo.sequence_tiling import tiled_call
    mesh = init_device_mesh(device.type, (world,))
    torch.manual_seed(0)
    base = GatedMLP().to(device)
    x = torch.randn(2, 37, H, device = device)
    ref_x = x.clone().requires_grad_(True)
    base(ref_x).pow(2).mean().backward()
    ref = {n: p.grad.clone() for n, p in base.named_parameters()}
    res = {}
    for mode in ("untiled", "new"):
        torch.manual_seed(0)
        m = GatedMLP().to(device)
        parallelize_module(m, mesh, {"gate_proj": ColwiseParallel(), "up_proj": ColwiseParallel(),
                                     "down_proj": RowwiseParallel()})
        xi = x.clone().requires_grad_(True)
        counts, saved = {}, {}
        for k in ("all_reduce", "reduce_scatter_tensor", "all_gather_tensor", "all_to_all_single"):
            real = saved[k] = getattr(funcol, k)
            def wrap(*a, _k = k, _real = real, **kw):
                counts[_k] = counts.get(_k, 0) + 1
                return _real(*a, **kw)
            setattr(funcol, k, wrap)
        try:
            y = m(xi) if mode == "untiled" else tiled_call(m, xi, n_tiles = N_TILES, needs_rng = False)
            fwd = dict(counts)
            y.pow(2).mean().backward()
        finally:
            for k, v in saved.items(): setattr(funcol, k, v)
        for n, p in m.named_parameters():
            _close(p.grad.full_tensor(), ref[n], f"tp {mode} {n}")
        _close(xi.grad, ref_x.grad, f"tp {mode} input grad")
        res[mode] = {"forward": fwd, "forward+backward": counts}
    out["tp"] = res


def _worker(rank, world, backend, cells, path):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if backend == "nccl":
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank)))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if not dist.is_initialized():
        dist.init_process_group(backend, rank = rank, world_size = world)
    import unsloth_zoo
    assert unsloth_zoo.__file__.startswith(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), unsloth_zoo.__file__
    out = {}
    try:
        for c in cells:
            globals()["cell_" + c](rank, world, device, out)
        out["ok"] = True
    except Exception:
        import traceback
        out["ok"] = False
        out["error"] = traceback.format_exc()[-3000:]
    if rank == 0 or not out["ok"]:
        with open(f"{path}.{rank}", "w") as f: json.dump(out, f, default = str)
    dist.barrier() if out["ok"] else None
    dist.destroy_process_group()


def _spawn(world, cells):
    import socket
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(port)
    path = tempfile.mktemp(prefix = "seqtile_")
    torch.multiprocessing.spawn(_worker, args = (world, "gloo", cells, path), nprocs = world, join = True)
    results = {}
    for r in range(world):
        if os.path.exists(f"{path}.{r}"):
            with open(f"{path}.{r}") as f: results[r] = json.load(f)
    return results


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("cell", ["ddp", "fsdp2"])
def test_gloo(world, cell):
    if cell == "fsdp2" and not hasattr(getattr(torch, "cpu", None) and torch.cpu.Stream, "record_event"):
        # fully_shard's all-gather records an event on the CPU stream, which torch < 2.7 lacks; the untiled
        # reference fails the same way, so there is nothing to compare. The NCCL cells cover FSDP2 there.
        pytest.skip("FSDP2 on CPU needs torch.cpu.Stream.record_event (torch >= 2.7)")
    res = _spawn(world, [cell])
    print(world, cell, json.dumps({r: v for r, v in res.items()}, default = str))
    for r, v in res.items():
        assert v["ok"], v.get("error")


def test_gloo_tp2():
    res = _spawn(2, ["tp"])
    print("tp2", json.dumps(res, default = str))
    assert res[0]["ok"], res[0].get("error")


if __name__ == "__main__":
    # torchrun --nproc-per-node N tests/test_sequence_tiling_distributed.py  (SEQ_TILING_BACKEND=nccl for GPUs)
    backend = os.environ.get("SEQ_TILING_BACKEND", "gloo")
    cells = os.environ.get("SEQ_TILING_CELLS", "ddp,fsdp2,tp").split(",")
    path = os.environ.get("SEQ_TILING_OUT", tempfile.mktemp(prefix = "seqtile_"))
    _worker(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), backend, cells, path)
    if int(os.environ["RANK"]) == 0:
        with open(f"{path}.0") as f: r = json.load(f)
        print(json.dumps(r, default = str))
        sys.exit(0 if r["ok"] else 1)
