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

"""While the disk tier serves some blocks, the host RAM blocks are stored with the lossless codec too, and the RAM
that frees holds blocks that would otherwise be read from disk twice per micro-batch. GPU tests count disk blocks
and reads, and compare training, state_dict and remove() bitwise with no swap and with the codec kept off RAM."""

import copy
import importlib.util
import os
import sys

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The sibling modules are registered only while this copy of block_swap latches them, so test files loaded later
# register (and patch) their own copies.
_added = []
for _name, _file in (("unsloth_zoo.block_swap_codec", "block_swap_codec.py"),
                     ("unsloth_zoo.block_swap_disk", "block_swap_disk.py")):
    if _name not in sys.modules:
        _s = importlib.util.spec_from_file_location(_name, os.path.join(_HERE, "unsloth_zoo", _file))
        _m = importlib.util.module_from_spec(_s)
        _s.loader.exec_module(_m)
        sys.modules[_name] = _m
        _added.append(_name)
_spec = importlib.util.spec_from_file_location(
    "block_swap_disk_ram_codec_under_test", os.path.join(_HERE, "unsloth_zoo", "block_swap.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_disk = _mod._disk()
_codec = _mod._codec()
for _name in _added:
    sys.modules.pop(_name, None)
BlockSwap = _mod.BlockSwap


def _triton_decode():
    return (torch.cuda.is_available() and sys.platform.startswith("linux") and _codec.triton_decode_available()
            and _codec.self_check(torch.device("cuda", torch.cuda.current_device()))[1])


_codec_gpu = pytest.mark.skipif(not _triton_decode(), reason = "needs a Linux GPU with the Triton decode")


@pytest.fixture(autouse = True)
def _spill_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_OFFLOAD_DISK_DIR", str(tmp_path))
    for k in ("UNSLOTH_OFFLOAD_DISK", "UNSLOTH_OFFLOAD_HOST_LIMIT", "UNSLOTH_OFFLOAD_DISK_COMPRESS",
              "UNSLOTH_BLOCK_SWAP_COMPRESS", "UNSLOTH_OFFLOAD_DISK_PLACEMENT"):
        monkeypatch.delenv(k, raising = False)
    _disk._SPILL[:] = []
    yield
    _disk._SPILL[:] = []


class _Layer(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.base = nn.Linear(w, w, bias = False, dtype = torch.bfloat16)
        nn.init.normal_(self.base.weight, std = 0.02)
        self.norm = nn.Parameter(torch.rand(w, dtype = torch.bfloat16) + 0.5, requires_grad = False)
        self.lora_A = nn.Linear(w, 8, bias = False, dtype = torch.float32)
        self.lora_B = nn.Linear(8, w, bias = False, dtype = torch.float32)
        nn.init.normal_(self.lora_B.weight, std = 0.02)

    def forward(self, x):
        h = (x * self.norm).to(torch.bfloat16)
        lora = self.lora_B(self.lora_A(x.float())).to(torch.bfloat16)
        return x + torch.tanh(self.base(h) + lora)


def _stack(n = 12, w = 512, seed = 0):
    torch.manual_seed(seed)
    layers = nn.ModuleList([_Layer(w) for _ in range(n)])
    for name, p in layers.named_parameters():
        p.requires_grad_("lora_" in name)
    return layers.to("cuda")


class _NoSwap:
    def stats(self):
        return {"disk_reads": 0}


def _train(layers, sw, steps = 3, ga = 2, reentrant = False, seed = 1):
    g = torch.Generator(device = "cuda").manual_seed(seed)
    w = layers[0].norm.numel()
    xs = [torch.randn(4, 32, w, device = "cuda", generator = g, dtype = torch.bfloat16) for _ in range(steps * ga + 1)]
    params = [p for n, p in layers.named_parameters() if "lora_" in n]
    opt = torch.optim.SGD(params, lr = 0.5)
    losses, grads, reads = [], [], []
    sw.stats()
    for s in range(steps):
        for m in range(ga):
            h = xs[s * ga + m].detach().requires_grad_(True)
            for layer in layers:
                h = checkpoint(layer, h, use_reentrant = reentrant)
            loss = h.float().pow(2).mean() / ga
            loss.backward()
            losses.append(loss.detach().clone())
        grads.append([p.grad.detach().clone() for p in params])
        opt.step()
        opt.zero_grad(set_to_none = True)
        torch.cuda.synchronize()
        reads.append(sw.stats()["disk_reads"])
    with torch.no_grad():
        h = xs[-1]
        for layer in layers:
            h = layer(h)
    torch.cuda.synchronize()
    return (losses, grads, h.clone()), reads


def _equal(a, b):
    la, ga, ha = a
    lb, gb, hb = b
    assert all(torch.equal(x, y) for x, y in zip(la, lb)), (la, lb)
    for s, (x, y) in enumerate(zip(ga, gb)):
        assert all(torch.equal(p, q) for p, q in zip(x, y)), f"grads differ at step {s}"
    assert torch.equal(ha, hb), "logits differ"


def _per(layers):
    return _mod._Block(layers[0], {}, torch.device("cuda", torch.cuda.current_device())).nbytes()


def _limit_for(layers, ram_layers):
    # BlockSwap keeps the staging ring (two layers) and a layer of release lag out of the host budget.
    per = _per(layers)
    return 2 * _disk._round_up(per) + per + ram_layers * per + 1024


def _frozen(layers):
    return {k: v.clone() for k, v in layers.state_dict().items() if "lora_" not in k}


def _check_readback(base, layers, sw, want):
    sd = layers.state_dict()
    for k, v in want.items():
        assert torch.equal(sd[k].to(v.device), v), k
    sw.remove()
    for a, b in zip(base.parameters(), layers.parameters()):
        if not a.requires_grad:
            assert torch.equal(a, b)


@_codec_gpu
@pytest.mark.parametrize("reentrant", [False, True])
def test_install_keeps_disk_blocks_in_the_ram_the_host_codec_frees(monkeypatch, reentrant):
    base = _stack()
    want = _frozen(base)
    ref, _ = _train(copy.deepcopy(base), _NoSwap(), reentrant = reentrant)
    monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", str(_limit_for(base, 5)))
    seen = {}
    for mode in ("0", None):
        if mode is None:
            monkeypatch.delenv("UNSLOTH_BLOCK_SWAP_COMPRESS", raising = False)
        else:
            monkeypatch.setenv("UNSLOTH_BLOCK_SWAP_COMPRESS", mode)
        _disk._SPILL[:] = []
        layers = copy.deepcopy(base)
        sw = BlockSwap(layers, 11, prefetch_depth = 2, placement = "tail")
        disk = [b for b in sw.blocks if b.disk is not None]
        ram_encoded = sum(b.compressed() for b in sw.blocks if b.disk is None)
        got, reads = _train(layers, sw, reentrant = reentrant)
        _equal(ref, got)
        seen[mode] = (len(disk), ram_encoded, sw.stats()["disk_promoted"], reads[-1])
        _check_readback(base, layers, sw, want)
    # 5 raw-sized layers of host budget: raw, 5 in RAM and 6 on disk. Encoded (0.75 of raw with the fixed-width
    # codec), the room left holds at least a sixth; each layer kept is two fewer reads per micro-batch (it is not at
    # a turnaround), and a step has 2 micro-batches.
    assert seen["0"][:3] == (6, 0, 0), seen
    kept = seen[None][2]
    assert kept >= 1 and seen[None][0] == 6 - kept and seen[None][1] == 5 + kept, seen
    assert seen[None][3] == seen["0"][3] - 2 * 2 * kept, seen


@_codec_gpu
def test_load_time_spill_moves_back_into_ram_the_host_codec_frees(monkeypatch):
    base = _stack()
    want = _frozen(base)
    ref, _ = _train(copy.deepcopy(base), _NoSwap())
    monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", str(_limit_for(base, 5)))
    seen = {}
    for mode in ("0", None):
        if mode is None:
            monkeypatch.delenv("UNSLOTH_BLOCK_SWAP_COMPRESS", raising = False)
        else:
            monkeypatch.setenv("UNSLOTH_BLOCK_SWAP_COMPRESS", mode)
        _disk._SPILL[:] = []
        layers = copy.deepcopy(base)
        for p in layers.parameters():
            p.requires_grad_(False)
        state = _mod._HostLoad(11, "tail")
        state.layers, state.indices = layers, _mod.swap_indices(len(layers), 11, "tail")
        for i in state.indices:
            state.evict(i)
        spilled = len(state.disk)
        for name, p in layers.named_parameters():
            p.requires_grad_("lora_" in name)
        sw = BlockSwap(layers, state.indices, prefetch_depth = 2, placement = "tail")
        disk = [b for b in sw.blocks if b.disk is not None]
        moved = [b for li, b in zip(sw.indices, sw.blocks) if li in state.disk and b.disk is None]
        # Moved blocks keep their encoding and lose their disk tags.
        assert all(b.compressed() and b.disk_store is None for b in moved)
        assert all(getattr(p, "_unsloth_disk", None) is None for b in moved for p in b.params)
        got, reads = _train(layers, sw)
        _equal(ref, got)
        seen[mode] = (spilled, len(disk), len(moved), reads[-1])
        _check_readback(base, layers, sw, want)
    assert seen["0"][0] == seen[None][0] and seen["0"][1] == seen["0"][0] and seen["0"][2] == 0, seen
    assert seen[None][2] >= 1 and seen[None][1] == seen[None][0] - seen[None][2], seen
    assert seen[None][3] < seen["0"][3], seen


@_codec_gpu
def test_every_disk_block_kept_closes_the_disk_tier(monkeypatch):
    base = _stack(n = 19)
    want = _frozen(base)
    ref, _ = _train(copy.deepcopy(base), _NoSwap())
    layers = copy.deepcopy(base)
    # Raw, 14 of the 18 swapped layers fit host RAM and 4 go to disk; encoded, the 14 leave room for all 4.
    monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", str(_limit_for(base, 14)))
    monkeypatch.setenv("UNSLOTH_BLOCK_SWAP_COMPRESS", "0")
    raw = BlockSwap(copy.deepcopy(base), 18, prefetch_depth = 2, placement = "tail")
    assert sum(b.disk is not None for b in raw.blocks) == 4
    raw.remove()
    monkeypatch.delenv("UNSLOTH_BLOCK_SWAP_COMPRESS")
    _disk._SPILL[:] = []
    sw = BlockSwap(layers, 18, prefetch_depth = 2, placement = "tail")
    assert not any(b.disk is not None for b in sw.blocks)
    assert sw._disk_tier is None and sw.stats()["disk_layers"] == 0 and sw.stats()["disk_promoted"] == 4
    assert all(b.compressed() for b in sw.blocks)
    got, reads = _train(layers, sw)
    _equal(ref, got)
    assert reads[-1] == 0
    _check_readback(base, layers, sw, want)


@_codec_gpu
def test_disk_codec_off_keeps_ram_raw(monkeypatch):
    base = _stack()
    monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", str(_limit_for(base, 5)))
    monkeypatch.setenv("UNSLOTH_OFFLOAD_DISK_COMPRESS", "0")
    sw = BlockSwap(copy.deepcopy(base), 11, prefetch_depth = 2, placement = "tail")
    assert sum(b.disk is not None for b in sw.blocks) == 6
    assert not any(b.compressed() for b in sw.blocks)
    sw.remove()


def test_promotion_order_moves_twice_read_blocks_first():
    keep = _mod._KEEP_BOTTOM
    try:
        _mod._KEEP_BOTTOM = True
        # 10 swapped blocks, depth 2: positions 0, 1, 8, 9 stay on the card at a turnaround (read once).
        assert _mod._promotion_order([0, 1, 3, 4, 5, 6, 8, 9], 10, 2) == [3, 5, 4, 6, 0, 1, 8, 9]
        _mod._KEEP_BOTTOM = False
        assert _mod._promotion_order([0, 1, 3, 8, 9], 10, 2) == [0, 3, 1, 8, 9]
    finally:
        _mod._KEEP_BOTTOM = keep


@_codec_gpu
def test_host_recheck_runs_before_the_staging_ring_is_taken(monkeypatch):
    """The host check for the encoder's scratch counts the disk staging ring: it must run while the ring is not yet
    allocated (and so not yet out of free host RAM), else the ring is counted twice."""
    base = _stack()
    monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", str(_limit_for(base, 5)))
    seen = []
    real = BlockSwap._compress_beside_disk

    def probe(self, fit_with):
        seen.append(self._disk_tier)
        return real(self, fit_with)

    monkeypatch.setattr(BlockSwap, "_compress_beside_disk", probe)
    sw = BlockSwap(copy.deepcopy(base), 11, prefetch_depth = 2, placement = "tail")
    assert seen == [None], seen
    assert sw._disk_tier is not None and sw._compress
    sw.remove()
