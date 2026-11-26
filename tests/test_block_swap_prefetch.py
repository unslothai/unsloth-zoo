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

"""Fewer bytes per block swap step: the first prefetch_depth blocks stay on the card from one backward to the next
forward, and disk-tier blocks are stored with the lossless bf16 codec. GPU tests count copies, disk reads and
read bytes, and compare training bitwise with the path without each change."""

import copy
import importlib.util
import os
import sys

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cspec = importlib.util.spec_from_file_location(
    "unsloth_zoo.block_swap_codec", os.path.join(_HERE, "unsloth_zoo", "block_swap_codec.py"))
_cmod = importlib.util.module_from_spec(_cspec)
_cspec.loader.exec_module(_cmod)
sys.modules.setdefault("unsloth_zoo.block_swap_codec", _cmod)
_dspec = importlib.util.spec_from_file_location(
    "unsloth_zoo.block_swap_disk", os.path.join(_HERE, "unsloth_zoo", "block_swap_disk.py"))
_dmod = importlib.util.module_from_spec(_dspec)
_dspec.loader.exec_module(_dmod)
sys.modules.setdefault("unsloth_zoo.block_swap_disk", _dmod)
_spec = importlib.util.spec_from_file_location(
    "block_swap_prefetch_under_test", os.path.join(_HERE, "unsloth_zoo", "block_swap.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_disk = _mod._disk()
BlockSwap = _mod.BlockSwap

_cuda = pytest.mark.skipif(not torch.cuda.is_available() or not sys.platform.startswith("linux"),
                           reason = "needs a CUDA / ROCm GPU on Linux")
_timing = pytest.mark.skipif(not _mod._TIMING, reason = "copy counts need UNSLOTH_OFFLOAD_STATS")


@pytest.fixture(autouse = True)
def _spill_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_OFFLOAD_DISK_DIR", str(tmp_path))
    monkeypatch.delenv("UNSLOTH_OFFLOAD_DISK", raising = False)
    monkeypatch.delenv("UNSLOTH_OFFLOAD_HOST_LIMIT", raising = False)
    monkeypatch.delenv("UNSLOTH_OFFLOAD_DISK_COMPRESS", raising = False)
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


def _stack(n = 10, w = 256, seed = 0):
    torch.manual_seed(seed)
    layers = nn.ModuleList([_Layer(w) for _ in range(n)])
    for name, p in layers.named_parameters():
        p.requires_grad_("lora_" in name)
    return layers.to("cuda")


def _train(layers, sw, steps = 3, ga = 2, reentrant = False, seed = 1):
    """Losses, per-step LoRA grads, final logits and the swap's per-step copy / read counts."""
    g = torch.Generator(device = "cuda").manual_seed(seed)
    w = layers[0].norm.numel()
    xs = [torch.randn(4, 32, w, device = "cuda", generator = g, dtype = torch.bfloat16) for _ in range(steps * ga + 1)]
    params = [p for n, p in layers.named_parameters() if "lora_" in n]
    opt = torch.optim.SGD(params, lr = 0.5)
    losses, grads, counts = [], [], []
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
        st = sw.stats()
        counts.append((st["copies"], st["disk_reads"]))
    with torch.no_grad():
        h = xs[-1]
        for layer in layers:
            h = layer(h)
    torch.cuda.synchronize()
    return (losses, grads, h.clone()), counts


def _equal(a, b):
    la, ga, ha = a
    lb, gb, hb = b
    assert all(torch.equal(x, y) for x, y in zip(la, lb)), (la, lb)
    for s, (x, y) in enumerate(zip(ga, gb)):
        assert all(torch.equal(p, q) for p, q in zip(x, y)), f"grads differ at step {s}"
    assert torch.equal(ha, hb), "logits differ"


def _limit_for(layers, ram_layers):
    per = _mod._Block(layers[0], {}, torch.device("cuda", torch.cuda.current_device())).nbytes()
    return 2 * _disk._round_up(per) + per + ram_layers * per + 1024


@_cuda
@_timing
@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("reentrant", [False, True])
def test_first_blocks_stay_for_the_next_forward(monkeypatch, depth, reentrant):
    base = _stack()
    ref, _ = _train(copy.deepcopy(base), _NoSwap(), reentrant = reentrant)
    n, ga = 9, 2
    out = {}
    for keep in (False, True):
        monkeypatch.setattr(_mod, "_KEEP_BOTTOM", keep)
        layers = copy.deepcopy(base)
        sw = BlockSwap(layers, n, prefetch_depth = depth, placement = "tail")
        got, counts = _train(layers, sw, ga = ga, reentrant = reentrant)
        _equal(ref, got)
        out[keep] = counts
        sw.remove()
    # Per micro-batch every block is copied for its forward and again for its backward, less the last `depth` kept
    # across forward -> backward; keeping the first `depth` across backward -> next forward saves `depth` more.
    # Step 1 also holds the blocks armed at install; steps 2 and 3 are steady.
    for s in (1, 2):
        assert out[False][s][0] == ga * (2 * n - depth), out
        assert out[True][s][0] == ga * (2 * n - 2 * depth), out


@_cuda
@_timing
def test_first_blocks_kept_on_the_disk_tier_skip_their_reads(monkeypatch):
    base = _stack()
    ref, _ = _train(copy.deepcopy(base), _NoSwap())
    reads = {}
    for keep in (False, True):
        monkeypatch.setattr(_mod, "_KEEP_BOTTOM", keep)
        monkeypatch.setenv("UNSLOTH_OFFLOAD_DISK_COMPRESS", "0")
        _disk._SPILL[:] = []
        layers = copy.deepcopy(base)
        monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", str(_limit_for(base, 2)))
        sw = BlockSwap(layers, 9, prefetch_depth = 2, placement = "tail")
        disk_k = [k for k, b in enumerate(sw.blocks) if b.disk is not None]
        assert len(disk_k) == 7, disk_k
        got, counts = _train(layers, sw)
        _equal(ref, got)
        reads[keep] = [c[1] for c in counts]
        sw.remove()
        monkeypatch.delenv("UNSLOTH_OFFLOAD_HOST_LIMIT")
    # Disk blocks sit at both turnarounds (spread placement), so each kept block skips one read per micro-batch:
    # 2 x 7 - 2 (forward -> backward) without the keep, 2 x 7 - 4 with it, for both micro-batches (steady steps).
    for s in (1, 2):
        assert reads[False][s] == 2 * (2 * 7 - 2) and reads[True][s] == 2 * (2 * 7 - 4), reads


@_cuda
def test_disk_blocks_sit_at_both_turnarounds_and_between_ram_blocks(monkeypatch):
    base = _stack()
    monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", str(_limit_for(base, 2)))
    for placement, want in (("spread", [0, 1, 2, 4, 6, 7, 8]), ("tail", [2, 3, 4, 5, 6, 7, 8])):
        monkeypatch.setenv("UNSLOTH_OFFLOAD_DISK_PLACEMENT", placement)
        _disk._SPILL[:] = []
        sw = BlockSwap(copy.deepcopy(base), 9, prefetch_depth = 2, placement = "tail")
        assert [k for k, b in enumerate(sw.blocks) if b.disk is not None] == want, placement
        sw.remove()


class _NoSwap:
    def stats(self):
        return {"copies": 0, "disk_reads": 0}


def _triton_decode():
    return _cmod.triton_decode_available() and _cmod.self_check(torch.device("cuda", torch.cuda.current_device()))[1]


_codec_gpu = pytest.mark.skipif(not torch.cuda.is_available() or not sys.platform.startswith("linux")
                                or not _triton_decode(), reason = "needs a Linux GPU with the Triton decode")


@_codec_gpu
@pytest.mark.parametrize("reentrant", [False, True])
def test_disk_tier_reads_encoded_bytes_and_trains_bitwise(monkeypatch, reentrant):
    base = _stack()
    # Frozen weights only: training moves the LoRA ones.
    want = {k: v.clone() for k, v in base.state_dict().items() if "lora_" not in k}
    ref, _ = _train(copy.deepcopy(base), _NoSwap(), reentrant = reentrant)
    seen = {}
    for on in ("0", "1"):
        monkeypatch.setenv("UNSLOTH_OFFLOAD_DISK_COMPRESS", on)
        _disk._SPILL[:] = []
        layers = copy.deepcopy(base)
        monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", str(_limit_for(base, 2)))
        sw = BlockSwap(layers, 9, prefetch_depth = 2, placement = "tail")
        disk = [b for b in sw.blocks if b.disk is not None]
        assert len(disk) == 7
        got, counts = _train(layers, sw, reentrant = reentrant)
        _equal(ref, got)
        reads = sum(c[1] for c in counts)
        assert reads > 0
        seen[on] = (sum(b.compressed() for b in disk), sum(d.fetches for d in sw._disk_decoders.values()))
        # Read-back paths decode too: state_dict, then remove() restores the originals.
        sd = layers.state_dict()
        for k, v in want.items():
            assert torch.equal(sd[k].to(v.device), v), k
        sw.remove()
        for a, b in zip(base.parameters(), layers.parameters()):
            if not a.requires_grad:
                assert torch.equal(a, b)
        monkeypatch.delenv("UNSLOTH_OFFLOAD_HOST_LIMIT")
    # Engagement, kill switch N vs 0: every disk block encoded and decoded on fetch only with the codec on.
    assert seen["0"] == (0, 0) and seen["1"][0] == 7 and seen["1"][1] > 0, seen


@_codec_gpu
def test_encoded_disk_reads_are_smaller(monkeypatch):
    base = _stack()
    per = _mod._Block(base[0], {}, torch.device("cuda", torch.cuda.current_device())).nbytes()
    rb = {}
    for on in ("0", "1"):
        monkeypatch.setenv("UNSLOTH_OFFLOAD_DISK_COMPRESS", on)
        _disk._SPILL[:] = []
        layers = copy.deepcopy(base)
        monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", str(_limit_for(base, 2)))
        sw = BlockSwap(layers, 9, prefetch_depth = 1, placement = "tail")
        sw.stats()
        _train(layers, sw, steps = 2)
        st = sw._disk_tier.stats()
        rb[on] = st["disk_read_bytes"] / max(1, st["disk_reads"])
        sw.remove()
        monkeypatch.delenv("UNSLOTH_OFFLOAD_HOST_LIMIT")
    assert rb["0"] >= per and rb["1"] < 0.85 * rb["0"], (rb, per)


@_codec_gpu
def test_load_time_spill_is_encoded_in_place_at_install(monkeypatch):
    base = _stack()
    ref, _ = _train(copy.deepcopy(base), _NoSwap())
    layers = copy.deepcopy(base)
    for p in layers.parameters():
        p.requires_grad_(False)
    state = _mod._HostLoad(9, "tail")
    state.layers, state.indices = layers, _mod.swap_indices(len(layers), 9, "tail")
    monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", str(_limit_for(base, 2)))
    for i in state.indices:
        state.evict(i)
    assert len(state.disk) >= 6
    size = _disk._SPILL[0].size
    for name, p in layers.named_parameters():
        p.requires_grad_("lora_" in name)
    sw = BlockSwap(layers, state.indices, prefetch_depth = 1, placement = "tail")
    loaded = [b for li, b in zip(sw.indices, sw.blocks) if li in state.disk]
    assert loaded and all(b.compressed() for b in loaded)
    # Rewritten over their raw regions: the spill file did not grow for them.
    assert _disk._SPILL[0].size == size + sum(n for b in sw.blocks if b.disk is not None and b not in loaded
                                              for _, n, _ in b.disk_runs)
    got, _ = _train(layers, sw)
    _equal(ref, got)
    sw.remove()


@_codec_gpu
def test_install_under_inference_mode_encodes_disk_blocks(monkeypatch):
    base = _stack()
    with torch.no_grad():
        h = torch.randn(4, 32, 256, device = "cuda", dtype = torch.bfloat16)
        want = h
        for layer in base:
            want = layer(want)
    layers = copy.deepcopy(base)
    monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", str(_limit_for(base, 2)))
    with torch.inference_mode():
        sw = BlockSwap(layers, 9, prefetch_depth = 2, placement = "tail")
    assert sum(b.compressed() for b in sw.blocks if b.disk is not None) == 7
    with torch.no_grad():
        got = h
        for layer in layers:
            got = layer(got)
    assert torch.equal(got, want)
    sw.remove()


@_codec_gpu
def test_interrupted_step_with_encoded_disk_blocks_stays_bitwise(monkeypatch):
    """Prefetched encoded blocks released mid-step (reset) leave their decode's event on the slot; the next fetch
    into that slot waits for it, so later steps still match."""
    base = _stack()
    ref, _ = _train(copy.deepcopy(base), _NoSwap())
    layers = copy.deepcopy(base)
    monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", str(_limit_for(base, 2)))
    sw = BlockSwap(layers, 9, prefetch_depth = 2, placement = "tail")
    with torch.no_grad():
        h = torch.randn(4, 32, 256, device = "cuda", dtype = torch.bfloat16)
        for layer in layers[:5]:
            h = layer(h)
    held = [b for b in sw.blocks if b.resident and b.compressed() and b.slot is not None]
    sw.reset()
    # Engagement: a released encoded block left a fence on its slot (consumed by the re-arm or still waiting).
    assert held
    got, _ = _train(layers, sw)
    _equal(ref, got)
    sw.remove()


@_codec_gpu
def test_host_codec_keeps_the_disk_decoders_on_disk_blocks(monkeypatch):
    monkeypatch.setenv("UNSLOTH_BLOCK_SWAP_COMPRESS", "1")
    base = _stack()
    ref, _ = _train(copy.deepcopy(base), _NoSwap())
    layers = copy.deepcopy(base)
    for p in layers.parameters():
        p.requires_grad_(False)
    state = _mod._HostLoad(9, "tail")
    state.layers, state.indices = layers, _mod.swap_indices(len(layers), 9, "tail")
    monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", str(_limit_for(base, 2)))
    for i in state.indices:
        state.evict(i)
    for name, p in layers.named_parameters():
        p.requires_grad_("lora_" in name)
    monkeypatch.delenv("UNSLOTH_OFFLOAD_HOST_LIMIT")
    sw = BlockSwap(layers, state.indices, prefetch_depth = 1, placement = "tail")
    disk = [b for b in sw.blocks if b.disk is not None]
    assert disk and all(b.decoders is sw._disk_decoders for b in disk if b.compressed())
    assert all(b.decoders is not sw._disk_decoders for b in sw.blocks if b.disk is None)
    got, _ = _train(layers, sw)
    _equal(ref, got)
    sw.remove()


@_codec_gpu
def test_slot_fences_only_ever_hold_submitted_decodes(monkeypatch):
    """Cancelled disk reads (two interrupted steps in a row) never replace a slot's fence with an event no decode
    was recorded on: the fence is set when a decode is submitted, never on release."""
    base = _stack()
    ref, _ = _train(copy.deepcopy(base), _NoSwap())
    submitted = []
    real = _cmod.DeviceDecoder.fetch

    def fetch(self, codec, blob, out_u8, copy_stream, done):
        submitted.append(done)
        return real(self, codec, blob, out_u8, copy_stream, done)

    monkeypatch.setattr(_cmod.DeviceDecoder, "fetch", fetch)
    layers = copy.deepcopy(base)
    monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", str(_limit_for(base, 2)))
    sw = BlockSwap(layers, 9, prefetch_depth = 2, placement = "tail")
    fences = sw.blocks[0].fences
    for stop in (5, 3):
        with torch.no_grad():
            h = torch.randn(4, 32, 256, device = "cuda", dtype = torch.bfloat16)
            for layer in layers[:stop]:
                h = layer(h)
        sw.reset()
        assert fences and all(any(ev is s for s in submitted) for ev in fences.values())
    got, _ = _train(layers, sw)
    _equal(ref, got)
    sw.remove()


@_codec_gpu
def test_disk_blocks_stay_raw_without_gpu_room_for_the_decoders(monkeypatch):
    base = _stack()
    ref, _ = _train(copy.deepcopy(base), _NoSwap())
    layers = copy.deepcopy(base)
    # Host-loaded frozen weights: packing frees nothing on the card.
    for layer in layers[1:]:
        for name, p in layer.named_parameters():
            if "lora_" not in name:
                p.data = p.data.cpu()
    monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", str(_limit_for(base, 2)))
    total = torch.cuda.get_device_properties(0).total_memory
    real_info, real_reserved = torch.cuda.mem_get_info, torch.cuda.memory_reserved
    calls = []

    def no_room(device = None):
        calls.append(device)
        return (0, total)

    monkeypatch.setattr(torch.cuda, "mem_get_info", no_room)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device = None: torch.cuda.memory_allocated(device))
    try:
        sw = BlockSwap(layers, 9, prefetch_depth = 2, placement = "tail")
    finally:
        monkeypatch.setattr(torch.cuda, "mem_get_info", real_info)
        monkeypatch.setattr(torch.cuda, "memory_reserved", real_reserved)
    assert calls and sw._disk_decoders == {}
    assert not any(b.compressed() for b in sw.blocks)
    got, _ = _train(layers, sw)
    _equal(ref, got)
    sw.remove()


@_codec_gpu
def test_spill_rewrite_needs_host_room_for_both_buffers(monkeypatch):
    base = _stack()
    layers = copy.deepcopy(base)
    monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", str(_limit_for(base, 2)))
    monkeypatch.setenv("UNSLOTH_OFFLOAD_DISK_COMPRESS", "0")
    sw = BlockSwap(layers, 9, prefetch_depth = 2, placement = "tail")
    b = next(b for b in sw.blocks if b.disk is not None and not b.resident)
    dev = torch.device("cuda", torch.cuda.current_device())
    real = _mod._host_free
    monkeypatch.setattr(_mod, "_host_free", lambda: (real()[0], b.nbytes()))
    assert b.encode_disk(dev) is False and not b.compressed()
    monkeypatch.setattr(_mod, "_host_free", real)
    assert b.encode_disk(dev) is True and b.compressed()
    sw.remove()
