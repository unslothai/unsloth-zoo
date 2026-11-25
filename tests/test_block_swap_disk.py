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

"""The block swap disk tier: layers host RAM cannot hold are served from an O_DIRECT spill file.
GPU tests compare training through the disk tier bitwise with the host RAM tier and with no swap at all."""

import copy
import importlib.util
import os
import sys

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "block_swap_disk_under_test", os.path.join(_HERE, "unsloth_zoo", "block_swap.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_disk = _mod._disk()
BlockSwap = _mod.BlockSwap

_cuda = pytest.mark.skipif(not torch.cuda.is_available() or not sys.platform.startswith("linux"),
                           reason = "needs a CUDA / ROCm GPU on Linux")


@pytest.fixture(autouse = True)
def _spill_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_OFFLOAD_DISK_DIR", str(tmp_path))
    monkeypatch.delenv("UNSLOTH_OFFLOAD_DISK", raising = False)
    monkeypatch.delenv("UNSLOTH_OFFLOAD_HOST_LIMIT", raising = False)
    # One spill file per process: each test gets a fresh one in its own directory.
    _disk._SPILL[:] = []
    yield
    _disk._SPILL[:] = []


# CPU -------------------------------------------------------------------------------------------------------------

def test_spill_file_round_trips_any_dtype_at_any_offset(tmp_path):
    if _disk.disk_tier_reason() is not None:
        pytest.skip(_disk.disk_tier_reason())
    store = _disk.SpillFile(1 << 20, directory = str(tmp_path))
    a = torch.randn(1000, dtype = torch.bfloat16)
    b = torch.randint(0, 255, (777,), dtype = torch.uint8)
    c = torch.randn(3, 5).to(torch.float8_e4m3fn) if hasattr(torch, "float8_e4m3fn") else torch.randn(15)
    base = store.write_region([(a, 0), (b, 2048), (c, 4096)], 4096 + c.nbytes)
    assert base % _disk.ALIGN == 0
    for t, off in ((a, 0), (b, 2048), (c, 4096)):
        buf, owner = _disk.read_bytes(store, base + off, t.nbytes)
        assert torch.equal(buf, t.reshape(-1).view(torch.uint8))
    # A second region starts aligned after the first.
    base2 = store.write_region([(b, 0)], b.nbytes)
    assert base2 >= base + 4096 + c.nbytes and base2 % _disk.ALIGN == 0
    buf, _ = _disk.read_bytes(store, base2, b.nbytes)
    assert torch.equal(buf, b)


def test_disk_tier_reason_and_host_limit_parse(monkeypatch):
    monkeypatch.setenv("UNSLOTH_OFFLOAD_DISK", "0")
    assert _disk.disk_tier_reason() == "UNSLOTH_OFFLOAD_DISK=0"
    monkeypatch.setenv("UNSLOTH_OFFLOAD_DISK", "auto")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    if sys.platform.startswith("linux"):
        assert "one process per host" in _disk.disk_tier_reason()
    monkeypatch.delenv("LOCAL_WORLD_SIZE")
    for text, want in (("40GB", 40 * 10**9), ("1.5GiB", int(1.5 * 2**30)), ("123", 123), ("2 MiB", 2 << 20)):
        monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", text)
        assert _disk.host_limit_bytes() == want
    monkeypatch.delenv("UNSLOTH_OFFLOAD_HOST_LIMIT")
    assert _disk.host_limit_bytes() is None


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "the disk tier is Linux only")
def test_spill_dir_skips_ram_filesystems_and_refuses_without_room(monkeypatch, tmp_path):
    monkeypatch.setattr(_disk, "_fs_type", lambda path: "tmpfs")
    with pytest.raises(ValueError, match = "no directory has that much free"):
        _disk.pick_spill_dir(1 << 20)
    monkeypatch.setattr(_disk, "_fs_type", lambda path: "ext4")
    assert _disk.pick_spill_dir(1 << 20) == str(tmp_path)
    with pytest.raises(ValueError, match = "UNSLOTH_OFFLOAD_DISK_DIR"):
        _disk.pick_spill_dir(1 << 60)


def test_host_limit_caps_the_host_fit(monkeypatch):
    monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", "1MB")
    _mod._check_host_fit([400_000, 400_000])
    with pytest.raises(ValueError, match = "At most 2 layers"):
        _mod._check_host_fit([400_000] * 3)


def test_auto_swap_indices_names_the_disk_tier():
    # CPU layers: auto_swap_indices plans CUDA cards only, so its picks go through the helper directly.
    layers = nn.ModuleList([nn.Linear(64, 64, bias = False).requires_grad_(False) for _ in range(10)])
    per = 64 * 64 * 4
    idx = list(range(10))
    assert _mod._disk_tier_indices(layers, idx, host_free_bytes = 10 * per) == []
    # Staging ring (2 layers) and a layer of lag come first: 6 layers of room keep 3 in RAM, the rest go to disk.
    assert _mod._disk_tier_indices(layers, idx, host_free_bytes = 6 * per) == list(range(3, 10))


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


def _train(layers, steps = 3, reentrant = False, seed = 1):
    """Losses, per-step LoRA grads and final logits of a few SGD steps with per-layer checkpointing."""
    g = torch.Generator(device = "cuda").manual_seed(seed)
    w = layers[0].norm.numel()
    xs = [torch.randn(4, 32, w, device = "cuda", generator = g, dtype = torch.bfloat16) for _ in range(steps + 1)]
    params = [p for n, p in layers.named_parameters() if "lora_" in n]
    opt = torch.optim.SGD(params, lr = 0.5)
    losses, grads = [], []
    for s in range(steps):
        # Reentrant checkpointing only tracks grads through inputs that need them.
        h = xs[s].detach().requires_grad_(True)
        for layer in layers:
            h = checkpoint(layer, h, use_reentrant = reentrant)
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


def _layer_bytes(layers):
    sw = _mod._Block(layers[0], {}, torch.device("cuda", torch.cuda.current_device()))
    return sw.nbytes()


def _limit_for(layers, ram_layers):
    """A host limit that holds the disk tier's staging ring plus `ram_layers` swapped layers."""
    per = _layer_bytes(layers)
    return 2 * _disk._round_up(per) + per + ram_layers * per + 1024


@_cuda
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("reentrant", [False, True])
def test_disk_tier_trains_bitwise_like_host_ram(monkeypatch, depth, reentrant):
    base = _stack()
    ref = _train(copy.deepcopy(base), reentrant = reentrant)
    ram_layers = copy.deepcopy(base)
    ram = BlockSwap(ram_layers, 9, prefetch_depth = depth, placement = "tail")
    assert ram.stats()["disk_layers"] == 0 and ram._disk_tier is None
    ram_out = _train(ram_layers, reentrant = reentrant)
    _equal(ref, ram_out)

    disk_layers = copy.deepcopy(base)
    monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", str(_limit_for(base, 2)))
    sw = BlockSwap(disk_layers, 9, prefetch_depth = depth, placement = "tail")
    st = sw.stats()
    assert st["disk_layers"] == 7 and st["disk_dir"] is not None, st
    assert [st["state"][i] for i in sw.indices].count("disk") >= 7 - depth
    disk_out = _train(disk_layers, reentrant = reentrant)
    _equal(ref, disk_out)
    st = sw.stats()
    per = _layer_bytes(base)
    # Engagement: every disk layer read at least once per forward and per backward (3 steps + one inference pass).
    assert st["disk_reads"] >= 3 * 7 and st["disk_read_bytes"] >= st["disk_reads"] * (per // 4096) * 4096, st
    sw.remove()
    ram.remove()
    # Restored weights are the originals.
    for a, b in zip(base.parameters(), disk_layers.parameters()):
        if not a.requires_grad:
            assert torch.equal(a, b)


@_cuda
def test_kill_switch_refuses_like_before(monkeypatch):
    base = _stack()
    monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", str(_limit_for(base, 2)))
    monkeypatch.setenv("UNSLOTH_OFFLOAD_DISK", "0")
    with pytest.raises(ValueError, match = "At most"):
        BlockSwap(copy.deepcopy(base), 9, prefetch_depth = 1, placement = "tail")
    assert not _disk._SPILL


@_cuda
def test_nothing_changes_when_host_ram_holds_the_layers(tmp_path):
    base = _stack()
    sw = BlockSwap(copy.deepcopy(base), 9, prefetch_depth = 1, placement = "tail")
    assert sw._disk_tier is None and not _disk._SPILL and os.listdir(tmp_path) == []
    assert all(b.disk is None for b in sw.blocks)
    sw.remove()


@_cuda
def test_qlora_4bit_layers_train_bitwise_from_disk(monkeypatch):
    pytest.importorskip("bitsandbytes")
    base = _stack(quant = True)
    ref = _train(copy.deepcopy(base))
    disk_layers = copy.deepcopy(base)
    monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", str(_limit_for(base, 1)))
    sw = BlockSwap(disk_layers, 9, prefetch_depth = 1, placement = "tail")
    assert sw.stats()["disk_layers"] == 8
    # The packed 4-bit bytes are what sits on disk: the quant state stays on the card.
    assert any(p.dtype == torch.uint8 for b in sw.blocks if b.disk is not None for p in b.params)
    _equal(ref, _train(disk_layers))
    sw.remove()


@_cuda
def test_state_dict_reads_disk_layers_back(monkeypatch):
    base = _stack()
    want = {k: v.clone() for k, v in base.state_dict().items()}
    layers = copy.deepcopy(base)
    monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", str(_limit_for(base, 1)))
    sw = BlockSwap(layers, 9, prefetch_depth = 1, placement = "tail")
    assert sw.stats()["disk_layers"] == 8
    sd = layers.state_dict()
    for k, v in want.items():
        assert torch.equal(sd[k].to(v.device), v), k
    sw.remove()


@_cuda
def test_load_time_spill_then_install(monkeypatch):
    """_HostLoad spills what host RAM cannot take as layers land; BlockSwap then serves those from disk."""
    base = _stack()
    ref = _train(copy.deepcopy(base))
    layers = copy.deepcopy(base)
    for p in layers.parameters():
        p.requires_grad_(False)
    state = _mod._HostLoad(9, "tail")
    state.layers, state.indices = layers, _mod.swap_indices(len(layers), 9, "tail")
    monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", str(_limit_for(base, 2)))
    for i in state.indices:
        state.evict(i)
    # Host load also counts the LoRA weights this fake layer carries (a real load has none yet): 6 to 8 spill.
    assert 6 <= len(state.disk) <= 8 and len(state.done) == 9
    assert state.disk == set(state.indices[-len(state.disk):])
    for i in state.disk:
        w = layers[i].base.weight
        assert w.untyped_storage().nbytes() == w.element_size() and w.shape == (256, 256)
    for i in set(state.indices) - state.disk:
        assert layers[i].base.weight.device.type == "cpu"
    for name, p in layers.named_parameters():
        p.requires_grad_("lora_" in name)
    sw = BlockSwap(layers, state.indices, prefetch_depth = 1, placement = "tail")
    assert sw.stats()["disk_layers"] >= len(state.disk)
    _equal(ref, _train(layers))
    sw.remove()


@_cuda
def test_full_finetuning_is_refused_on_the_disk_tier(monkeypatch):
    base = _stack()
    layers = copy.deepcopy(base)
    for p in layers.parameters():
        p.requires_grad_(False)
    state = _mod._HostLoad(9, "tail")
    state.layers, state.indices = layers, _mod.swap_indices(len(layers), 9, "tail")
    monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", str(_limit_for(base, 2)))
    for i in state.indices:
        state.evict(i)
    for p in layers.parameters():
        p.requires_grad_(True)
    with pytest.raises(ValueError, match = "Full finetuning"):
        BlockSwap(layers, state.indices, prefetch_depth = 1, placement = "tail")


@_cuda
def test_rewritten_spilled_weight_is_refused(monkeypatch):
    base = _stack()
    layers = copy.deepcopy(base)
    for p in layers.parameters():
        p.requires_grad_(False)
    state = _mod._HostLoad(9, "tail")
    state.layers, state.indices = layers, _mod.swap_indices(len(layers), 9, "tail")
    monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", str(_limit_for(base, 2)))
    for i in state.indices:
        state.evict(i)
    i = max(state.disk)
    layers[i].base.weight.data = layers[i].base.weight.data.float()
    with pytest.raises(RuntimeError, match = "rewritten"):
        BlockSwap(layers, state.indices, prefetch_depth = 1, placement = "tail")


@_cuda
def test_interrupted_forward_then_reset_stays_bitwise(monkeypatch):
    """A read in flight when the step is abandoned (reset) is cancelled; the next steps still match."""
    base = _stack()
    ref = _train(copy.deepcopy(base))
    layers = copy.deepcopy(base)
    monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", str(_limit_for(base, 1)))
    sw = BlockSwap(layers, 9, prefetch_depth = 2, placement = "tail")
    with torch.no_grad():
        h = torch.randn(4, 32, 256, device = "cuda", dtype = torch.bfloat16)
        for layer in layers[:6]:
            h = layer(h)
    sw.reset()
    _equal(ref, _train(layers))
    sw.remove()
