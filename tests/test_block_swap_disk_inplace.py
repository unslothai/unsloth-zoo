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

"""Disk tier layers read in place from the checkpoint's safetensors files instead of a spill copy.
CPU tests cover the byte matching and the region reads; GPU tests train through it bitwise against the spill path
and check that no layer bytes were written to disk."""

import copy
import importlib.util
import os
import sys

import pytest
import torch
import torch.nn as nn

safetensors_torch = pytest.importorskip("safetensors.torch")

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_dspec = importlib.util.spec_from_file_location(
    "unsloth_zoo.block_swap_disk", os.path.join(_HERE, "unsloth_zoo", "block_swap_disk.py"))
_dmod = importlib.util.module_from_spec(_dspec)
_dspec.loader.exec_module(_dmod)
sys.modules.setdefault("unsloth_zoo.block_swap_disk", _dmod)
_spec = importlib.util.spec_from_file_location(
    "block_swap_disk_inplace_under_test", os.path.join(_HERE, "unsloth_zoo", "block_swap.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_disk = _mod._disk()
BlockSwap = _mod.BlockSwap
# The disk tier's codec (on by default): standalone, block_swap gets the copy loaded here.
_cspec = importlib.util.spec_from_file_location(
    "block_swap_codec_inplace_under_test", os.path.join(_HERE, "unsloth_zoo", "block_swap_codec.py"))
_cmod = importlib.util.module_from_spec(_cspec)
_cspec.loader.exec_module(_cmod)
_mod._CODEC.append(_cmod)

_linux = pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "the disk tier is Linux only")
_cuda = pytest.mark.skipif(not torch.cuda.is_available() or not sys.platform.startswith("linux"),
                           reason = "needs a CUDA / ROCm GPU on Linux")


@pytest.fixture(autouse = True)
def _env(tmp_path, monkeypatch):
    spill = tmp_path / "spill"
    spill.mkdir()
    monkeypatch.setenv("UNSLOTH_OFFLOAD_DISK_DIR", str(spill))
    for k in ("UNSLOTH_OFFLOAD_DISK", "UNSLOTH_OFFLOAD_HOST_LIMIT", "UNSLOTH_OFFLOAD_DISK_INPLACE"):
        monkeypatch.delenv(k, raising = False)
    _disk._SPILL[:] = []
    yield
    _disk._SPILL[:] = []


def _save(folder, tensors, name = "model.safetensors"):
    os.makedirs(folder, exist_ok = True)
    safetensors_torch.save_file({k: v.contiguous().cpu() for k, v in tensors.items()}, os.path.join(folder, name))


class _Named(nn.Module):
    """A model whose checkpoint is `folder` (what from_pretrained records as name_or_path)."""

    def __init__(self, layers, folder):
        super().__init__()
        self.layers = layers
        self.name_or_path = folder


# CPU -------------------------------------------------------------------------------------------------------------

@_linux
def test_match_finds_whole_stacked_and_partial_tensors(tmp_path):
    torch.manual_seed(0)
    w1 = [torch.randint(0, 255, (37, 64), dtype = torch.uint8) for _ in range(3)]
    w3 = [torch.randint(0, 255, (37, 64), dtype = torch.uint8) for _ in range(3)]
    bf = torch.randn(33, 7, dtype = torch.bfloat16)
    tensors = {f"e{i}.w1": w1[i] for i in range(3)} | {f"e{i}.w3": w3[i] for i in range(3)} | {"bf": bf}
    _save(str(tmp_path / "ck"), tensors)
    model = _Named(nn.ModuleList(), str(tmp_path / "ck"))
    index = _disk.checkpoint_index(model)
    assert index is not None and len(index.files) == 1

    def flat(t):
        return t.contiguous().reshape(-1).view(torch.uint8)

    # One tensor, any dtype.
    ext = index.match(flat(bf))
    assert ext is not None and len(ext) == 1 and ext[0][2] == bf.nbytes
    # Experts stacked the way a loader restacks them (cat(w1_e, w3_e) per expert): several file ranges.
    stacked = torch.stack([torch.cat([w1[i], w3[i]]) for i in range(3)])
    ext = index.match(flat(stacked))
    assert ext is not None and sum(n for _, _, n, _ in ext) == stacked.nbytes
    # A leading part of one tensor.
    assert index.match(flat(w1[1])[:1000]) is not None
    # Transformed bytes are not in the file.
    assert index.match(flat(bf.float())) is None
    assert index.match(flat(w1[0].t())) is None


@_linux
@pytest.mark.parametrize("mode", ["auto", "buffered"])
def test_region_reads_mapped_bytes_zero_holes_and_redirects_writes(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("UNSLOTH_OFFLOAD_DISK_INPLACE", mode)
    torch.manual_seed(1)
    a = torch.randint(0, 255, (5000,), dtype = torch.uint8)
    b = torch.randn(1000, dtype = torch.bfloat16)
    _save(str(tmp_path / "ck"), {"a": a, "b": b})
    index = _disk.checkpoint_index(_Named(nn.ModuleList(), str(tmp_path / "ck")))
    if mode == "buffered":
        assert index.files[0].dfd is None
    pa, pb, pc = nn.Parameter(a.clone(), requires_grad = False), nn.Parameter(b.clone(), requires_grad = False), \
        nn.Parameter(torch.randn(300, dtype = torch.bfloat16), requires_grad = False)
    layout = [0, 8192, 12288]
    region = _disk.inplace_region(index, [pa, pb, pc], layout, 16384)
    assert isinstance(region, _disk.CheckpointRegion)
    # a and b come from the checkpoint; c (not in it) went to the spill file.
    assert region.ckpt_bytes == a.nbytes + b.nbytes
    assert region.written == _disk._round_up(pc.nbytes) and _disk.live_spill() is not None
    out = torch.full((16384,), 7, dtype = torch.uint8)
    region.read(0, out)
    assert torch.equal(out[:a.nbytes], a)
    assert torch.equal(out[a.nbytes:8192], torch.zeros(8192 - a.nbytes, dtype = torch.uint8))
    assert torch.equal(out[8192:8192 + b.nbytes], b.view(torch.uint8))
    assert torch.equal(out[12288:12288 + pc.nbytes], pc.data.view(torch.uint8))
    # Unaligned windows, as the reader threads split a region into chunks.
    piece = torch.empty(4321, dtype = torch.uint8)
    region.read(3333, piece)
    assert torch.equal(piece, out[3333:3333 + 4321])
    # A write over mapped bytes lands in the spill file; the checkpoint file is untouched.
    before = open(index.files[0].path, "rb").read()
    new = torch.randint(0, 255, (100,), dtype = torch.uint8)
    region.write_region([(new, 0)], new.numel(), at = 40)
    region.read(0, out)
    assert torch.equal(out[40:140], new) and torch.equal(out[:40], a[:40]) and torch.equal(out[140:a.nbytes], a[140:])
    assert open(index.files[0].path, "rb").read() == before


@_linux
def test_nothing_in_the_checkpoint_or_switched_off_means_the_usual_spill(tmp_path, monkeypatch):
    _save(str(tmp_path / "ck"), {"x": torch.randn(64, 64)})
    model = _Named(nn.ModuleList(), str(tmp_path / "ck"))
    index = _disk.checkpoint_index(model)
    p = nn.Parameter(torch.randn(64, 64), requires_grad = False)
    assert _disk.inplace_region(index, [p], [0], p.nbytes) is None
    monkeypatch.setenv("UNSLOTH_OFFLOAD_DISK_INPLACE", "0")
    assert _disk.checkpoint_index(model) is None
    # No checkpoint recorded on the model: no index.
    monkeypatch.delenv("UNSLOTH_OFFLOAD_DISK_INPLACE")
    assert _disk.checkpoint_index(_Named(nn.ModuleList(), str(tmp_path / "missing"))) is None


# GPU -------------------------------------------------------------------------------------------------------------

class _Layer(nn.Module):
    def __init__(self, w, quant = False):
        super().__init__()
        if quant:
            import bitsandbytes as bnb
            self.base = bnb.nn.Linear4bit(w, w, bias = False, compute_dtype = torch.bfloat16, quant_type = "nf4")
        else:
            self.base = nn.Linear(w, w, bias = False, dtype = torch.bfloat16)
        self.norm = nn.Parameter(torch.rand(w, dtype = torch.bfloat16) + 0.5, requires_grad = False)
        self.lora_A = nn.Linear(w, 8, bias = False, dtype = torch.float32)
        self.lora_B = nn.Linear(8, w, bias = False, dtype = torch.float32)
        nn.init.normal_(self.lora_B.weight, std = 0.02)

    def forward(self, x):
        h = (x * self.norm).to(torch.bfloat16)
        lora = self.lora_B(self.lora_A(x.float())).to(torch.bfloat16)
        return x + torch.tanh(self.base(h) + lora)


def _stack(n = 10, w = 256, quant = False, seed = 0):
    torch.manual_seed(seed)
    layers = nn.ModuleList([_Layer(w, quant) for _ in range(n)])
    for name, p in layers.named_parameters():
        p.requires_grad_("lora_" in name)
    return layers.to("cuda")


def _checkpoint(layers, folder):
    """The frozen weights as a checkpoint would hold them, split over two shards like a real upload."""
    frozen = {n: p.data for n, p in layers.named_parameters() if not p.requires_grad}
    names = sorted(frozen)
    _save(folder, {n: frozen[n] for n in names[:len(names) // 2]}, "model-00001-of-00002.safetensors")
    _save(folder, {n: frozen[n] for n in names[len(names) // 2:]}, "model-00002-of-00002.safetensors")


def _train(layers, steps = 3, seed = 1):
    from torch.utils.checkpoint import checkpoint
    g = torch.Generator(device = "cuda").manual_seed(seed)
    w = layers[0].norm.numel()
    xs = [torch.randn(4, 32, w, device = "cuda", generator = g, dtype = torch.bfloat16) for _ in range(steps + 1)]
    params = [p for n, p in layers.named_parameters() if "lora_" in n]
    opt = torch.optim.SGD(params, lr = 0.5)
    losses, grads = [], []
    for s in range(steps):
        h = xs[s].detach().requires_grad_(True)
        for layer in layers:
            h = checkpoint(layer, h, use_reentrant = False)
        loss = h.float().pow(2).mean()
        loss.backward()
        losses.append(loss.detach().clone())
        grads.append([p.grad.detach().clone() for p in params])
        opt.step()
        opt.zero_grad(set_to_none = True)
    with torch.no_grad():
        h = xs[-1]
        for layer in layers:
            h = layer(h)
    torch.cuda.synchronize()
    return losses, grads, h.clone()


def _equal(a, b):
    la, ga, ha = a
    lb, gb, hb = b
    assert all(torch.equal(x, y) for x, y in zip(la, lb)), (la, lb)
    for s, (x, y) in enumerate(zip(ga, gb)):
        assert all(torch.equal(p, q) for p, q in zip(x, y)), f"grads differ at step {s}"
    assert torch.equal(ha, hb), "logits differ"


def _limit_for(layers, ram_layers):
    per = _mod._Block(layers[0], {}, torch.device("cuda", torch.cuda.current_device())).nbytes()
    return 2 * _disk._round_up(per) + per + ram_layers * per + 1024, per


@_cuda
@pytest.mark.parametrize("compress", ["0", "1"])
def test_install_time_disk_layers_read_in_place_train_like_the_spill(monkeypatch, tmp_path, compress):
    monkeypatch.setenv("UNSLOTH_OFFLOAD_DISK_COMPRESS", compress)
    base = _stack()
    folder = str(tmp_path / "ck")
    _checkpoint(base, folder)
    limit, per = _limit_for(base, 2)
    monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", str(limit))

    spill_layers = copy.deepcopy(base)
    monkeypatch.setenv("UNSLOTH_OFFLOAD_DISK_INPLACE", "0")
    sw0 = BlockSwap(spill_layers, 9, prefetch_depth = 1, placement = "tail", model = _Named(spill_layers, folder))
    st0 = sw0.stats()
    assert st0["disk_layers"] == 7 and st0["disk_inplace_bytes"] == 0 and st0["disk_spill_written"] > 0, st0
    spill_out = _train(spill_layers)
    sw0.remove()
    _disk._SPILL[:] = []

    layers = copy.deepcopy(base)
    monkeypatch.delenv("UNSLOTH_OFFLOAD_DISK_INPLACE")
    sw = BlockSwap(layers, 9, prefetch_depth = 1, placement = "tail", model = _Named(layers, folder))
    st = sw.stats()
    assert st["disk_layers"] == 7, st
    if compress == "1":
        # bf16 blocks keep the codec (fewer bytes read per step): encoded into the spill file.
        assert any(b.compressed() for b in sw.blocks if b.disk is not None), st
    else:
        # Every disk layer's bytes come from the checkpoint files: nothing written.
        assert st["disk_inplace_bytes"] >= 7 * 256 * 256 * 2 and st["disk_spill_written"] == 0, st
        assert _disk.live_spill() is None, "no spill file should exist"
    out = _train(layers)
    st = sw.stats()
    assert st["disk_reads"] >= 3 * 7, st
    _equal(spill_out, out)
    sw.remove()
    for a, b in zip(base.parameters(), layers.parameters()):
        if not a.requires_grad:
            assert torch.equal(a, b)


@_cuda
def test_load_time_spill_reads_in_place_and_trains_like_the_spill(monkeypatch, tmp_path):
    """The from_pretrained path: _HostLoad finds the layer bytes in the checkpoint and writes only the small tail."""
    monkeypatch.setenv("UNSLOTH_OFFLOAD_DISK_COMPRESS", "0")
    base = _stack()
    folder = str(tmp_path / "ck")
    _checkpoint(base, folder)
    limit, per = _limit_for(base, 2)

    def run(inplace):
        if inplace:
            monkeypatch.delenv("UNSLOTH_OFFLOAD_DISK_INPLACE", raising = False)
        else:
            monkeypatch.setenv("UNSLOTH_OFFLOAD_DISK_INPLACE", "0")
        _disk._SPILL[:] = []
        layers = copy.deepcopy(base)
        for p in layers.parameters():
            p.requires_grad_(False)
        state = _mod._HostLoad(9, "tail")
        state.layers, state.indices = layers, _mod.swap_indices(len(layers), 9, "tail")
        state.model_ref = (lambda m: (lambda: m))(_Named(layers, folder))
        monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", str(limit))
        for i in state.indices:
            state.evict(i)
        assert state.disk
        for name, p in layers.named_parameters():
            p.requires_grad_("lora_" in name)
        sw = BlockSwap(layers, state.indices, prefetch_depth = 1, placement = "tail")
        st = sw.stats()
        out = _train(layers)
        sw.remove()
        return len(state.disk), st, out

    n0, st0, spill_out = run(False)
    n1, st1, out = run(True)
    assert n0 == n1
    assert st0["disk_inplace_bytes"] == 0 and st0["disk_spill_written"] >= n0 * per, st0
    # Only the norms (kept in RAM at load, one 4 KiB tail page per layer) were written.
    assert st1["disk_inplace_bytes"] >= n1 * 256 * 256 * 2 and st1["disk_spill_written"] <= n1 * 4096, st1
    _equal(spill_out, out)


@_cuda
def test_prequantized_4bit_layers_read_in_place(monkeypatch, tmp_path):
    """Packed 4-bit bytes as saved are what the layer holds: served from the checkpoint, quant state on the card."""
    pytest.importorskip("bitsandbytes")
    base = _stack(quant = True)
    folder = str(tmp_path / "ck")
    _checkpoint(base, folder)
    limit, _ = _limit_for(base, 1)
    monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", str(limit))
    monkeypatch.setenv("UNSLOTH_OFFLOAD_DISK_INPLACE", "0")
    ref_layers = copy.deepcopy(base)
    sw0 = BlockSwap(ref_layers, 9, prefetch_depth = 1, placement = "tail", model = _Named(ref_layers, folder))
    ref = _train(ref_layers)
    sw0.remove()
    _disk._SPILL[:] = []
    monkeypatch.delenv("UNSLOTH_OFFLOAD_DISK_INPLACE")
    layers = copy.deepcopy(base)
    sw = BlockSwap(layers, 9, prefetch_depth = 1, placement = "tail", model = _Named(layers, folder))
    st = sw.stats()
    assert st["disk_layers"] == 8 and st["disk_inplace_bytes"] > 0 and st["disk_spill_written"] == 0, st
    _equal(ref, _train(layers))
    sw.remove()


@_cuda
def test_load_time_in_place_layers_stay_raw_when_the_spill_has_no_room_to_encode(monkeypatch, tmp_path):
    """Disk compression on (the default): a layer read in place whose encoding has no spill room stays raw, read
    from the checkpoint, instead of failing the install."""
    monkeypatch.setenv("UNSLOTH_OFFLOAD_DISK_COMPRESS", "1")
    base = _stack()
    folder = str(tmp_path / "ck")
    _checkpoint(base, folder)
    limit, per = _limit_for(base, 2)
    ref = _train(copy.deepcopy(base))
    real = _disk._shared_spill

    def small_only(nbytes):
        # Room for the 4 KiB norm tails, not for a layer's encoding.
        if nbytes > 64 << 10:
            raise ValueError("Unsloth: offload_layers needs more disk (test)")
        return real(nbytes)

    layers = copy.deepcopy(base)
    for p in layers.parameters():
        p.requires_grad_(False)
    state = _mod._HostLoad(9, "tail")
    state.layers, state.indices = layers, _mod.swap_indices(len(layers), 9, "tail")
    state.model_ref = (lambda m: (lambda: m))(_Named(layers, folder))
    monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", str(limit))
    for i in state.indices:
        state.evict(i)
    assert state.disk
    for name, p in layers.named_parameters():
        p.requires_grad_("lora_" in name)
    monkeypatch.setattr(_disk, "_shared_spill", small_only)
    asked = []
    encode = _mod._Block.encode_disk
    monkeypatch.setattr(_mod._Block, "encode_disk", lambda self, d: asked.append(1) or encode(self, d))
    sw = BlockSwap(layers, state.indices, prefetch_depth = 1, placement = "tail")
    assert asked, "the disk codec was never engaged"
    st = sw.stats()
    disk = [b for b in sw.blocks if b.disk is not None]
    assert disk and not any(b.compressed() for b in disk), st
    assert st["disk_inplace_bytes"] >= len(state.disk) * 256 * 256 * 2, st
    _equal(ref, _train(layers))
    sw.remove()


@_linux
def test_direct_reads_span_the_bounce_buffer_at_odd_offsets(tmp_path, monkeypatch):
    monkeypatch.setattr(_disk, "_BOUNCE_BYTES", 8192)
    monkeypatch.setattr(_disk, "_BOUNCE", __import__("threading").local())
    big = torch.randint(0, 255, (100_003,), dtype = torch.uint8)
    _save(str(tmp_path / "ck"), {"big": big})
    index = _disk.checkpoint_index(_Named(nn.ModuleList(), str(tmp_path / "ck")))
    f = index.files[0]
    if f.dfd is None:
        pytest.skip("this filesystem refuses O_DIRECT")
    base = index.entries[0][1]
    for off, n in ((0, 100_003), (1, 4095), (4093, 20_000), (99_000, 1003)):
        out = torch.empty(n, dtype = torch.uint8)
        f.read_into(base + off, out, drop = True)
        assert torch.equal(out, big[off:off + n]), (off, n)
