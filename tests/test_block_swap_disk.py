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
# block_swap finds its disk tier in sys.modules when it is loaded from file.
_dspec = importlib.util.spec_from_file_location(
    "unsloth_zoo.block_swap_disk", os.path.join(_HERE, "unsloth_zoo", "block_swap_disk.py"))
_dmod = importlib.util.module_from_spec(_dspec)
_dspec.loader.exec_module(_dmod)
sys.modules.setdefault("unsloth_zoo.block_swap_disk", _dmod)
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


def _open_spill_fds():
    out = []
    for fd in os.listdir("/proc/self/fd"):
        try:
            if ".unsloth_offload_" in os.readlink(f"/proc/self/fd/{fd}"):
                out.append(int(fd))
        except OSError:
            pass
    return out


def _live():
    # live_spill() on this tree; the bare list before it (so the test also runs against an older tree).
    fn = getattr(_disk, "live_spill", None)
    return fn() if fn is not None else (_disk._SPILL[0] if _disk._SPILL else None)


def _spilled_param(nbytes, need_bytes):
    p = nn.Parameter(torch.randn(nbytes // 4), requires_grad = False)
    _disk.spill_params([p], [0], p.data.nbytes, need_bytes = need_bytes)
    return p


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "the disk tier is Linux only")
def test_spill_file_closes_with_its_last_user():
    import gc
    import weakref
    if _disk.disk_tier_reason() is not None:
        pytest.skip(_disk.disk_tier_reason())
    before = _open_spill_fds()
    p = _spilled_param(4 << 20, 4 << 20)
    q = _spilled_param(1 << 20, 1 << 20)
    store = weakref.ref(_live())
    # Both models share the one file while either is alive.
    assert p._unsloth_disk.store is q._unsloth_disk.store is store()
    assert len(_open_spill_fds()) == len(before) + 1
    del p
    gc.collect()
    assert store() is not None and _live() is store()
    del q
    gc.collect()
    # The last user gone: the unlinked file is closed, so its disk blocks (and the bounce buffer) come back.
    assert store() is None and _live() is None
    assert _open_spill_fds() == before


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "the disk tier is Linux only")
def test_reused_spill_file_rechecks_free_space():
    if _disk.disk_tier_reason() is not None:
        pytest.skip(_disk.disk_tier_reason())
    p = _spilled_param(1 << 20, 1 << 20)
    store = _live()
    assert _disk._shared_spill(1 << 20) is store
    # A later load that needs more than the disk holds is refused up front, not by ENOSPC partway through.
    with pytest.raises(ValueError, match = "no directory has that much free"):
        _spilled_param(1 << 20, 1 << 60)
    assert _live() is store and p._unsloth_disk.store is store


def test_spill_file_close_survives_interpreter_teardown(monkeypatch, tmp_path):
    if _disk.disk_tier_reason() is not None:
        pytest.skip(_disk.disk_tier_reason())
    store = _disk.SpillFile(1 << 20, directory = str(tmp_path))
    # At exit, __del__ can run after module globals such as `os` are gone.
    monkeypatch.setattr(_disk, "os", None)
    store.close()
    assert store.fd is None


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


def test_host_limit_refusal_names_the_cap_not_the_system_reserve(monkeypatch):
    # Plenty of free RAM: only UNSLOTH_OFFLOAD_HOST_LIMIT can refuse, so the message must name it.
    monkeypatch.setattr(_mod, "_host_free", lambda: (64 << 30, 32 << 30))
    monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", "1MB")
    with pytest.raises(ValueError) as e:
        _mod._check_host_fit([400_000] * 3)
    msg = str(e.value)
    assert "UNSLOTH_OFFLOAD_HOST_LIMIT caps the offloaded layers at 1.0 MiB" in msg
    assert "free for the system" not in msg and "At most 2 layers" in msg


def test_free_ram_refusal_under_a_looser_cap_still_names_the_reserve(monkeypatch):
    monkeypatch.setattr(_mod, "_host_free", lambda: (64 << 30, 1 << 20))
    monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", "1GB")
    with pytest.raises(ValueError) as e:
        _mod._check_host_fit([400_000] * 3)
    msg = str(e.value)
    assert "free for the system" in msg and "caps the offloaded layers" not in msg


def test_auto_swap_indices_names_the_disk_tier():
    # CPU layers: auto_swap_indices plans CUDA cards only, so its picks go through the helper directly.
    layers = nn.ModuleList([nn.Linear(64, 64, bias = False).requires_grad_(False) for _ in range(10)])
    per = 64 * 64 * 4
    idx = list(range(10))
    assert _mod._disk_tier_indices(layers, idx, host_free_bytes = 10 * per) == []
    # Staging ring (2 layers) and a layer of lag come first: 6 layers of room keep 3 in RAM, the rest go to disk:
    # both turnarounds' blocks first, the rest spread between the RAM blocks.
    assert _mod._disk_tier_indices(layers, idx, host_free_bytes = 6 * per) == [0, 1, 3, 5, 7, 8, 9]
    os.environ["UNSLOTH_OFFLOAD_DISK_PLACEMENT"] = "tail"
    try:
        assert _mod._disk_tier_indices(layers, idx, host_free_bytes = 6 * per) == list(range(3, 10))
    finally:
        del os.environ["UNSLOTH_OFFLOAD_DISK_PLACEMENT"]


@pytest.mark.parametrize("world, want", [(1, []), (2, [0, 1, 3, 5, 7, 8, 9])])
def test_disk_tier_indices_plan_for_this_ranks_share_of_host_ram(monkeypatch, world, want):
    """Local ranks read the same free RAM: each plans its disk tier for free // LOCAL_WORLD_SIZE."""
    # Meta layers: none of their bytes are in host RAM yet.
    layers = nn.ModuleList([nn.Linear(64, 64, bias = False, device = "meta").requires_grad_(False) for _ in range(10)])
    per = 64 * 64 * 4
    monkeypatch.setattr(_mod, "_host_free", lambda: (64 * per, 12 * per))
    monkeypatch.delenv("UNSLOTH_OFFLOAD_HOST_LIMIT", raising = False)
    monkeypatch.delenv("UNSLOTH_OFFLOAD_DISK_PLACEMENT", raising = False)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", str(world))
    assert _mod._disk_tier_indices(layers, list(range(10))) == want


@pytest.mark.parametrize("world", [1, 2])
def test_plan_disk_budgets_this_ranks_share_of_host_ram(monkeypatch, world):
    """BlockSwap's disk fallback keeps in host RAM only what this local rank's share of free RAM holds."""
    from types import SimpleNamespace
    per = 1 << 20
    blocks = [SimpleNamespace(disk = None, sizes = [per], params = [object()], nbytes = lambda: per) for _ in range(10)]
    stub = SimpleNamespace(blocks = blocks, _host_credit = [], depth = 1)
    monkeypatch.setattr(_mod, "_host_free", lambda: (64 * per, 12 * per))
    monkeypatch.setattr(_disk, "disk_tier_reason", lambda: None)
    monkeypatch.setattr(_disk, "host_limit_bytes", lambda: None)
    monkeypatch.setattr(_disk, "_shared_spill", lambda nbytes: None)
    monkeypatch.setattr(_disk, "_round_up", lambda n: n)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", str(world))
    monkeypatch.setenv("UNSLOTH_OFFLOAD_DISK_PLACEMENT", "tail")
    assert BlockSwap._plan_disk(stub, "refused") is True
    # 12 blocks of free RAM (world 1) or 6 (world 2), less the staging ring (2) and a block of lag (1).
    assert len(stub._to_spill) == 10 - (12 // world - 3)


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
    # Each read moves the block's stored region: raw, or encoded when the codec took it.
    stored = min(n for b in sw.blocks if b.disk is not None for _, n, _ in b.disk_runs)
    assert stored <= -(-per // 4096) * 4096
    assert st["disk_reads"] >= 3 * 7 and st["disk_read_bytes"] >= st["disk_reads"] * stored, st
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


@_cuda
def test_failed_install_keeps_load_time_spills_for_a_retry(monkeypatch):
    """An install that fails after a load-time spill leaves the layers recoverable: a retry still reads them from
    disk instead of taking the placeholders for weights."""
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
    for name, p in layers.named_parameters():
        p.requires_grad_("lora_" in name)
    # Host RAM shrank between load and install: not even the staging ring fits, so the install refuses.
    monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", "4096")
    with pytest.raises(ValueError):
        BlockSwap(layers, state.indices, prefetch_depth = 1, placement = "tail")
    real = _mod.BlockSwap._pack

    def boom(self):
        raise RuntimeError("install failed")
    # Room again, but the install fails later (after the checks): rolled back, still recoverable.
    monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", str(_limit_for(base, 2)))
    monkeypatch.setattr(_mod.BlockSwap, "_pack", boom)
    with pytest.raises(RuntimeError, match = "install failed"):
        BlockSwap(layers, state.indices, prefetch_depth = 1, placement = "tail")
    monkeypatch.setattr(_mod.BlockSwap, "_pack", real)
    sw = BlockSwap(layers, state.indices, prefetch_depth = 1, placement = "tail")
    assert sw.stats()["disk_layers"] >= len(state.disk)
    _equal(ref, _train(layers))
    sw.remove()
