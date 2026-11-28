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

"""The disk tier's spill striped over several local disks: each region is split by the disks' read speeds and each
file is read on its own threads. CPU tests cover the split, reads at any offset, rewrites, the reader pools and which
directories qualify; GPU tests train through a striped spill bitwise like a single file and like no swap."""

import copy
import importlib.util
import os
import sys

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    "block_swap_stripe_under_test", os.path.join(_HERE, "unsloth_zoo", "block_swap.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_disk = _mod._disk()
_codec = _mod._codec()
for _name in _added:
    sys.modules.pop(_name, None)
BlockSwap = _mod.BlockSwap
ALIGN = _disk.ALIGN


@pytest.fixture(autouse = True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_OFFLOAD_DISK_DIR", str(tmp_path))
    for k in ("UNSLOTH_OFFLOAD_DISK", "UNSLOTH_OFFLOAD_HOST_LIMIT", "UNSLOTH_OFFLOAD_DISK_COMPRESS",
              "UNSLOTH_OFFLOAD_DISK_STRIPE", "UNSLOTH_BLOCK_SWAP_COMPRESS"):
        monkeypatch.delenv(k, raising = False)
    _disk._SPILL[:] = []
    yield
    _disk._SPILL[:] = []


def _two(tmp_path, weights = (2.0, 1.0)):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    tot = sum(weights)
    return _disk.StripedSpill([(str(a), weights[0] / tot), (str(b), weights[1] / tot)], 64 << 20)


def _bytes(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n,), dtype = torch.uint8, generator = g)


def test_regions_split_by_weight_and_read_back_at_any_offset(tmp_path):
    st = _two(tmp_path)
    regions = []
    for seed, n in enumerate((5 * ALIGN + 123, 3 << 20, 7)):
        x = _bytes(n, seed)
        y = _bytes(1000, seed + 10)
        base = st.write_region([(x, 0), (y, n + 512)], n + 512 + 1000)
        assert base % ALIGN == 0
        regions.append((base, x, y, n))
        ext = st.regions[base]
        total = _disk._round_up(n + 512 + 1000)
        assert sum(e[3] for e in ext) == total and all(e[3] % ALIGN == 0 and e[1] % ALIGN == 0 for e in ext)
        if total >= 3 * ALIGN:
            # About 2 : 1, rounded down to whole pages for the first file.
            assert abs(ext[0][3] - 2 * total / 3) <= ALIGN and len(ext) == 2
    for base, x, y, n in regions:
        for off, m in ((0, n), (1, n - 1), (n // 2 + 3, n - n // 2 - 3), (n + 512, 1000), (n - 1, 514)):
            if m <= 0:
                continue
            got, owner = _disk.read_bytes(st, base + off, m)
            want = torch.cat((x, torch.zeros(512, dtype = torch.uint8), y))[off:off + m]
            assert torch.equal(got, want), (base, off, m)
            del owner
    # Both files hold bytes; the virtual size counts each region once.
    assert all(f.written > 0 for f in st.files)
    assert st.written == sum(f.written for f in st.files)
    st.close()


def test_rewrite_of_a_whole_region_stays_balanced(tmp_path):
    st = _two(tmp_path)
    n = 8 << 20
    base = st.write_region([(_bytes(n, 1), 0)], n)
    old = [e[3] for e in st.regions[base]]
    enc = _bytes(6 << 20, 2)
    assert st.write_region([(enc, 0)], enc.numel(), at = base) == base
    new = [e[3] for e in st.regions[base]]
    assert sum(new) == enc.numel() and all(a <= b for a, b in zip(new, old))
    assert abs(new[0] / sum(new) - 2 / 3) < 0.01, new
    got, owner = _disk.read_bytes(st, base, enc.numel())
    assert torch.equal(got, enc)
    # Past the shortened rewrite: no file maps those bytes, they read as zeros.
    tail, owner2 = _disk.read_bytes(st, base + enc.numel(), ALIGN)
    assert not tail.any()
    st.close()


def test_rewrite_of_a_tail_inside_a_region(tmp_path):
    st = _two(tmp_path)
    n = 6 << 20
    x = _bytes(n, 3)
    base = st.write_region([(x, 0)], n)
    patch = _bytes(5 * ALIGN, 4)
    at = base + n - 5 * ALIGN
    st.write_region([(patch, 0)], patch.numel(), at = at)
    got, owner = _disk.read_bytes(st, base, n)
    assert torch.equal(got[:n - 5 * ALIGN], x[:n - 5 * ALIGN]) and torch.equal(got[n - 5 * ALIGN:], patch)
    st.close()


class _Stub:
    stage = None
    disk_futs = None


def test_disk_tier_reads_each_file_on_its_own_readers(tmp_path):
    st = _two(tmp_path)
    n = 12 << 20
    x = _bytes(n, 5)
    base = st.write_region([(x, 0)], n)
    tier = _disk.DiskTier(n, slots = 1)
    tier.chunk = 1 << 20
    b = _Stub()
    b.disk_store = st
    b.disk_runs = ((base, n, 0),)
    tier._start(b, 0)
    for f in b.disk_futs:
        f.result()
    assert torch.equal(tier.ring[0][1][:n], x)
    # Engagement: both files read, each on its own pool, in proportion to the split.
    assert len(tier.pools) == 2 and set(tier.file_bytes) == {f.directory for f in st.files}
    assert [tier.file_bytes[f.directory] for f in st.files] == [e[3] for e in st.regions[base]]
    tier.inflight.clear()
    tier.close()
    st.close()


def _fake_mounts(monkeypatch, tmp_path, table):
    dirs = {}
    for name, *_ in table:
        d = tmp_path / name
        d.mkdir(exist_ok = True)
        dirs[name] = str(d)
    monkeypatch.setattr(_disk, "_mounts", lambda: [(dirs[n], kind, src, opts.split(",")) for n, kind, src, opts in table])
    devs = {dirs[n]: k for k, (n, *_) in enumerate(table)}
    monkeypatch.setattr(_disk, "_dev", lambda p: devs.get(p, -1))
    src = {dirs[n]: (kind, s) for n, kind, s, _ in table}
    monkeypatch.setattr(_disk, "_fs_source", lambda p: src.get(p, (None, None)))
    return dirs


_TABLE = [
    ("nvme", "ext4", "/dev/nvmefake0n1", "rw,relatime"),
    ("boot", "ext4", "/dev/sdfake1", "rw,relatime"),
    ("boot2", "ext4", "/dev/sdfake1", "rw,relatime"),       # the same disk mounted again
    ("over", "overlay", "overlay", "rw,relatime"),
    ("loop", "ext4", "/dev/loop7", "rw"),
    ("ram", "tmpfs", "tmpfs", "rw"),
    ("ro", "ext4", "/dev/sdfake2", "ro,relatime"),
    ("samedisk", "xfs", "/dev/nvmefake0n1", "rw"),          # another mount of the primary's disk
]


def test_candidates_are_other_real_disks_only(monkeypatch, tmp_path):
    dirs = _fake_mounts(monkeypatch, tmp_path, _TABLE)
    assert _disk._stripe_candidates(dirs["nvme"]) == [dirs["boot"]]
    # A primary on an overlay (Colab L4, standard A100) gets no secondary disk.
    assert _disk._stripe_candidates(dirs["over"]) == []


def test_stripe_dirs_weights_kill_switch_and_list(monkeypatch, tmp_path):
    dirs = _fake_mounts(monkeypatch, tmp_path, _TABLE)
    monkeypatch.delenv("UNSLOTH_OFFLOAD_DISK_DIR")
    monkeypatch.setattr(_disk, "pick_spill_dir", lambda need: dirs["nvme"])
    speeds = {dirs["nvme"]: 0.74, dirs["boot"]: 0.38}
    monkeypatch.setattr(_disk, "_read_speed", lambda d: speeds.get(d))
    got = _disk._stripe_dirs(1 << 30)
    assert [d for d, _ in got] == [dirs["nvme"], dirs["boot"]]
    assert abs(got[0][1] - 0.74 / 1.12) < 1e-9 and abs(sum(w for _, w in got) - 1) < 1e-9
    # The primary pick itself is unchanged.
    assert got[0][0] == _disk.pick_spill_dir(1)
    monkeypatch.setenv("UNSLOTH_OFFLOAD_DISK_STRIPE", "0")
    assert _disk._stripe_dirs(1 << 30) == [(dirs["nvme"], 1.0)]
    monkeypatch.setenv("UNSLOTH_OFFLOAD_DISK_STRIPE", "auto")
    monkeypatch.setenv("UNSLOTH_OFFLOAD_DISK_DIR", dirs["nvme"])
    assert _disk._stripe_dirs(1 << 30) == [(dirs["nvme"], 1.0)]
    monkeypatch.delenv("UNSLOTH_OFFLOAD_DISK_DIR")
    # Too slow to help (under 15% of the summed speed), or no room for its share: left out.
    speeds[dirs["boot"]] = 0.05
    assert len(_disk._stripe_dirs(1 << 30)) == 1
    speeds[dirs["boot"]] = 0.38
    # A secondary disk (often the boot disk, holding the user's files) must also keep 20 GiB / 10% free after its share.
    monkeypatch.setattr(_disk, "_keeps_free", lambda d, n: d != dirs["boot"])
    assert len(_disk._stripe_dirs(1 << 30)) == 1
    monkeypatch.undo()


def test_secondary_disk_keeps_room_for_the_users_files(monkeypatch, tmp_path):
    class St:
        f_frsize, f_blocks = 4096, (400 << 30) // 4096
        def __init__(self, free):
            self.f_bavail = free // 4096
    # 400 GiB disk: keep max(20 GiB, 40 GiB) free beyond the share.
    monkeypatch.setattr(_disk.os, "statvfs", lambda d: St(60 << 30))
    assert _disk._keeps_free(str(tmp_path), 19 << 30) and not _disk._keeps_free(str(tmp_path), 21 << 30)


def test_list_env_stripes_over_exactly_those_dirs(monkeypatch, tmp_path):
    a, b = tmp_path / "x", tmp_path / "y"
    a.mkdir()
    b.mkdir()
    monkeypatch.setenv("UNSLOTH_OFFLOAD_DISK_STRIPE", os.pathsep.join((str(a), str(b))))
    monkeypatch.setitem(_disk._PROBE_SPEEDS, str(a), 3.0)
    monkeypatch.setitem(_disk._PROBE_SPEEDS, str(b), 1.0)
    st = _disk._shared_spill(16 << 20)
    assert isinstance(st, _disk.StripedSpill) and [f.directory for f in st.files] == [str(a), str(b)]
    assert _disk._shared_spill(1 << 20) is st and _disk.live_spill() is st
    monkeypatch.setenv("UNSLOTH_OFFLOAD_DISK_STRIPE", "0")
    _disk._SPILL[:] = []
    assert isinstance(_disk._shared_spill(16 << 20), _disk.SpillFile)


def test_read_speed_probe_measures_a_real_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(_disk, "_PROBE_BYTES", 32 << 20)
    v = _disk._read_speed(str(tmp_path))
    assert v is None or v > 0
    assert _disk._read_speed(str(tmp_path)) == v
    _disk._PROBE_SPEEDS.pop(str(tmp_path), None)


# GPU: training through a striped spill ------------------------------------------------------------------------------

def _triton_decode():
    return (torch.cuda.is_available() and sys.platform.startswith("linux") and _codec.triton_decode_available()
            and _codec.self_check(torch.device("cuda", torch.cuda.current_device()))[1])


_cuda = pytest.mark.skipif(not torch.cuda.is_available() or not sys.platform.startswith("linux"),
                           reason = "needs a CUDA / ROCm GPU on Linux")


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


def _stack(n = 10, w = 512, seed = 0):
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
    losses, grads = [], []
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
    with torch.no_grad():
        h = xs[-1]
        for layer in layers:
            h = layer(h)
    torch.cuda.synchronize()
    return losses, grads, h.clone()


def _equal(a, b):
    assert all(torch.equal(x, y) for x, y in zip(a[0], b[0])), (a[0], b[0])
    for s, (x, y) in enumerate(zip(a[1], b[1])):
        assert all(torch.equal(p, q) for p, q in zip(x, y)), f"grads differ at step {s}"
    assert torch.equal(a[2], b[2]), "logits differ"


def _limit_for(layers, ram_layers):
    per = _mod._Block(layers[0], {}, torch.device("cuda", torch.cuda.current_device())).nbytes()
    return 2 * _disk._round_up(per) + per + ram_layers * per + 1024


def _stripe_env(monkeypatch, tmp_path):
    a, b = tmp_path / "sa", tmp_path / "sb"
    a.mkdir(exist_ok = True)
    b.mkdir(exist_ok = True)
    monkeypatch.setenv("UNSLOTH_OFFLOAD_DISK_STRIPE", os.pathsep.join((str(a), str(b))))
    monkeypatch.setitem(_disk._PROBE_SPEEDS, str(a), 2.0)
    monkeypatch.setitem(_disk._PROBE_SPEEDS, str(b), 1.0)
    return str(a), str(b)


@_cuda
@pytest.mark.parametrize("compress", ["0", "1"])
@pytest.mark.parametrize("load_time", [False, True])
def test_training_through_a_striped_spill_is_bitwise(monkeypatch, tmp_path, compress, load_time):
    if compress == "1" and not _triton_decode():
        pytest.skip("needs the Triton decode")
    monkeypatch.setenv("UNSLOTH_OFFLOAD_DISK_COMPRESS", compress)
    base = _stack()
    want = {k: v.clone() for k, v in base.state_dict().items() if "lora_" not in k}
    ref = _train(copy.deepcopy(base), _NoSwap())
    got = {}
    for striped in (False, True):
        _disk._SPILL[:] = []
        if striped:
            dirs = _stripe_env(monkeypatch, tmp_path)
        else:
            monkeypatch.setenv("UNSLOTH_OFFLOAD_DISK_STRIPE", "0")
        layers = copy.deepcopy(base)
        monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", str(_limit_for(base, 2)))
        if load_time:
            for p in layers.parameters():
                p.requires_grad_(False)
            state = _mod._HostLoad(9, "tail")
            state.layers, state.indices = layers, _mod.swap_indices(len(layers), 9, "tail")
            for i in state.indices:
                state.evict(i)
            assert state.disk
            for name, p in layers.named_parameters():
                p.requires_grad_("lora_" in name)
            sw = BlockSwap(layers, state.indices, prefetch_depth = 2, placement = "tail")
        else:
            sw = BlockSwap(layers, 9, prefetch_depth = 2, placement = "tail")
        disk = [b for b in sw.blocks if b.disk is not None]
        assert disk
        if compress == "1":
            assert all(b.compressed() for b in disk)
        out = _train(layers, sw)
        _equal(ref, out)
        tier = sw._disk_tier
        if striped:
            assert all(isinstance(b.disk_store, _disk.StripedSpill) for b in disk)
            # Engagement: both files served reads on their own pools.
            assert set(tier.file_bytes) == set(dirs) and all(v > 0 for v in tier.file_bytes.values())
        else:
            assert not tier.file_bytes and not any(isinstance(b.disk_store, _disk.StripedSpill) for b in disk)
        got[striped] = out
        sd = layers.state_dict()
        for k, v in want.items():
            assert torch.equal(sd[k].to(v.device), v), k
        sw.remove()
        for a, b in zip(base.parameters(), layers.parameters()):
            if not a.requires_grad:
                assert torch.equal(a, b)
        monkeypatch.delenv("UNSLOTH_OFFLOAD_HOST_LIMIT")
    _equal(got[False], got[True])
