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

"""The Huffman block swap format: bit exact round trips (every 16 bit word, any bytes, length-limited codes,
odd chunk tails), the choice between it and the fixed-width format, the card decode (Triton and the torch
fallback), and disk-tier training that stays bitwise equal while reading fewer bytes (bf16 and fp8 blocks)."""

import copy
import importlib.util
import os
import sys

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, "unsloth_zoo", file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Private copies, never registered in sys.modules: other block swap test files patch the copies they register.
C = _load("block_swap_codec_huff_under_test", "block_swap_codec.py")
_dmod = _load("block_swap_disk_huff_under_test", "block_swap_disk.py")
_bs = _load("block_swap_huff_under_test", "block_swap.py")
_bs._CODEC[:] = [C]
_bs._DISK[:] = [_dmod]
_disk = _bs._disk()
assert _disk is _dmod and _bs._codec() is C


@pytest.fixture(autouse = True, scope = "module")
def _few_threads():
    # The encoder runs many small host ops: a few threads beat dozens contending on a busy host.
    n = torch.get_num_threads()
    torch.set_num_threads(min(n, 8))
    yield
    torch.set_num_threads(n)


@pytest.fixture(autouse = True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_BLOCK_SWAP_CODEC", "huff")
    monkeypatch.delenv("UNSLOTH_OFFLOAD_DISK_COMPRESS_FP8", raising = False)
    monkeypatch.delenv("UNSLOTH_OFFLOAD_DISK_COMPRESS", raising = False)
    monkeypatch.delenv("UNSLOTH_OFFLOAD_HOST_LIMIT", raising = False)
    monkeypatch.delenv("UNSLOTH_OFFLOAD_DISK", raising = False)
    monkeypatch.setenv("UNSLOTH_OFFLOAD_DISK_DIR", str(tmp_path))
    C._HUFF_BROKEN.clear()
    _disk._SPILL[:] = []
    yield
    C._HUFF_BROKEN.clear()
    _disk._SPILL[:] = []


def _weights(words, seed = 0, scale = 0.02):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(words, generator = g) * scale).to(torch.bfloat16).view(torch.uint8).clone()


def _blob(parts, codec):
    blob = torch.zeros(codec.nbytes, dtype = torch.uint8)
    C.write_parts(blob, parts)
    return blob


def _huff(raw, mode, chunk = C.GROUP, max_ratio = 10.0):
    enc = C._encode_huff(raw, mode, chunk, max_ratio, raw.device, None)
    assert enc is not None
    parts, codec = enc
    assert codec.kind == "huff" and codec.mode == mode
    return codec, _blob(parts, codec)


def _every_word(mode):
    """All 65536 16 bit words (NaN payloads, +-inf, subnormals, +-0) twice, the second time shuffled, inside
    data a chunk can still code below the staging size: weights ("bf16") or zero bytes ("bytes")."""
    words = torch.arange(-32768, 32768, dtype = torch.int32).to(torch.int16)
    g = torch.Generator().manual_seed(1)
    words = torch.cat((words, words[torch.randperm(words.numel(), generator = g)])).view(torch.uint8)
    fill = _weights(14 * C.GROUP) if mode == "bf16" else torch.zeros(28 * C.GROUP, dtype = torch.uint8)
    return torch.cat((fill[:fill.numel() // 2], words, fill[fill.numel() // 2:])).clone()


@pytest.mark.parametrize("mode", ["bf16", "bytes"])
def test_every_16_bit_word_round_trips(mode):
    raw = _every_word(mode)
    for chunk in (16 * C.GROUP, 4 * C.GROUP):
        enc = C._encode_huff(raw, mode, chunk, 10.0, raw.device, None)
        if enc is None:
            continue    # the shorter chunk holding only the uniform words would pass the staging size
        codec = enc[1]
        assert torch.equal(C.decode_region_cpu(codec, _blob(*enc)), raw)
    raw = _every_word(mode)
    codec, blob = _huff(raw, mode, chunk = 16 * C.GROUP)
    assert torch.equal(C.decode_region_cpu(codec, blob), raw)


@pytest.mark.parametrize("words", [C.GROUP + 768, 2 * C.GROUP + 1024 + 256, 3 * C.GROUP + 512])
@pytest.mark.parametrize("mode", ["bf16", "bytes"])
def test_odd_sizes_and_chunk_tails_round_trip(words, mode):
    raw = _weights(words, seed = words)
    codec, blob = _huff(raw, mode)
    assert codec.chunks[-1].n % codec.chunks[-1].nl != 0 or codec.n_chunks > 1
    assert torch.equal(C.decode_region_cpu(codec, blob), raw)


def test_small_regions_keep_the_fixed_width_format():
    # The 8 KiB decode table does not pay for itself below about 50K words: those regions stay fixed-width.
    raw = _weights(256)
    assert C.encode_region(raw, max_escape_fraction = 1.0)[1].kind == "v1"
    assert C.encode_region(raw, kind = "bytes") is None


def test_random_bytes_and_single_symbol_chunks_round_trip():
    g = torch.Generator().manual_seed(2)
    noise = torch.randint(0, 256, (3 * C.GROUP,), dtype = torch.uint8, generator = g)
    raw = torch.cat((noise, torch.zeros(5 * C.GROUP + 512, dtype = torch.uint8)))
    codec, blob = _huff(raw, "bytes", chunk = 8 * C.GROUP)
    assert torch.equal(C.decode_region_cpu(codec, blob), raw)
    # One symbol in the whole chunk: a 1 bit code, half the decode table unused.
    raw = torch.full((C.GROUP,), 0x3C, dtype = torch.uint8)
    for mode in ("bf16", "bytes"):
        codec, blob = _huff(raw, mode)
        assert torch.equal(C.decode_region_cpu(codec, blob), raw)


def test_code_lengths_are_limited_and_prefix_free():
    # Counts halving per symbol would give a 39 bit code without the limit.
    hist = [1 << max(0, 40 - i) for i in range(40)] + [0] * 216
    lengths = C._huff_lengths(hist)
    assert max(lengths) == C.HUFF_MAX_LEN and min(ln for ln in lengths if ln) >= 1
    assert sum(2.0 ** -ln for ln in lengths if ln) <= 1.0
    codes, lut = C._huff_tables(lengths)
    used = [(codes[i], ln) for i, ln in enumerate(lengths) if ln]
    for a, la in used:
        for b, lb in used:
            if (a, la) != (b, lb):
                # Read least significant bit first: no code is the low bits of another.
                assert not (la <= lb and (b & ((1 << la) - 1)) == a)
    # Data with that skew round trips (long codes exercised).
    g = torch.Generator().manual_seed(4)
    p = torch.tensor([2.0 ** -i for i in range(1, 30)])
    sym = torch.multinomial(p / p.sum(), 6 * C.GROUP, replacement = True, generator = g).to(torch.uint8)
    codec, blob = _huff(sym.clone(), "bytes")
    assert torch.equal(C.decode_region_cpu(codec, blob), sym)


def test_weights_pick_the_huffman_format_and_are_smaller():
    raw = _weights(24 * C.GROUP + 512)
    parts, codec = C.encode_region(raw, chunk = 16 * C.GROUP)
    assert codec.kind == "huff" and codec.mode == "bf16" and codec.n_chunks == 2
    assert codec.nbytes / codec.raw_nbytes < 0.70
    assert torch.equal(C.decode_region_cpu(codec, _blob(parts, codec)), raw)
    parts1, v1 = C._encode_v1(raw, 16 * C.GROUP, C.MAX_RATIO, raw.device, None, C.MAX_ESCAPE_FRACTION)
    assert v1.kind == "v1" and codec.nbytes < 0.92 * v1.nbytes


def test_huffman_format_is_the_default_and_v1_keeps_it_off(monkeypatch):
    raw = _weights(2 * C.GROUP)
    monkeypatch.setenv("UNSLOTH_BLOCK_SWAP_CODEC", "v1")
    assert C.encode_region(raw)[1].kind == "v1"
    assert C.encode_region(raw, kind = "bytes") is None
    for on in ("huff", None):
        if on is None:
            monkeypatch.delenv("UNSLOTH_BLOCK_SWAP_CODEC")
        else:
            monkeypatch.setenv("UNSLOTH_BLOCK_SWAP_CODEC", on)
        assert C.encode_region(raw)[1].kind == "huff"


def test_fixed_width_is_kept_where_it_is_smaller():
    # Exponents spread evenly over 15 values: the 4 bit table codes them as well as Huffman, without its tables.
    g = torch.Generator().manual_seed(5)
    w = _weights(2 * C.GROUP).view(torch.int16)
    exp = torch.randint(110, 125, (w.numel(),), generator = g, dtype = torch.int16)
    w.copy_((w & -32641) | (exp << 7))
    raw = w.view(torch.uint8)
    assert C.encode_region(raw)[1].kind == "v1"


def test_bytes_mode_compresses_fp8_like_bytes_and_rejects_random_ones():
    g = torch.Generator().manual_seed(6)
    x = (torch.randn(4 * C.GROUP, generator = g) * 40).to(torch.float8_e4m3fn).view(torch.uint8).clone()
    parts, codec = C.encode_region(x, kind = "bytes")
    assert codec.mode == "bytes" and codec.nbytes / codec.raw_nbytes < 0.9
    assert torch.equal(C.decode_region_cpu(codec, _blob(parts, codec)), x)
    noise = torch.randint(0, 256, (4 * C.GROUP,), dtype = torch.uint8, generator = g)
    assert C.encode_region(noise, kind = "bytes") is None


@pytest.mark.parametrize("mode", ["bf16", "bytes"])
def test_streaming_and_in_place_encodes_write_the_same_blob(mode):
    raw = _weights(5 * C.GROUP + 512) if mode == "bf16" else \
        (torch.randn(5 * C.GROUP + 512) * 40).to(torch.float8_e4m3fn).view(torch.uint8).clone()
    parts, codec = C.encode_region(raw, chunk = 2 * C.GROUP, kind = mode)
    assert codec.kind == "huff" and codec.n_chunks >= 2
    blob = _blob(parts, codec)
    out = torch.full((raw.numel(),), 0xEE, dtype = torch.uint8)
    p2, c2 = C.encode_region(raw, chunk = 2 * C.GROUP, out = out, kind = mode)
    C.write_parts(out, p2)
    assert c2.nbytes == codec.nbytes and torch.equal(out[:c2.nbytes], blob)
    # Over the region itself (the spill path): every chunk is read before its encoding lands.
    same = raw.clone()
    p3, c3 = C.encode_region(same, chunk = 2 * C.GROUP, out = same, kind = mode)
    C.write_parts(same, p3)
    assert torch.equal(same[:c3.nbytes], blob)


def test_a_corrupt_decode_table_is_detected():
    raw = _weights(C.GROUP)
    codec, blob = _huff(raw, "bf16")
    lut = blob[:C._HUFF_LUT * 2].view(torch.int16)
    lut[(lut != 0).nonzero()[0]] ^= 1   # one entry decodes to the wrong exponent
    assert not torch.equal(C.decode_region_cpu(codec, blob), raw)


# ------------------------------------------------------------------------------------------------ GPU
_gpu = pytest.mark.skipif(not torch.cuda.is_available() or not sys.platform.startswith("linux"),
                          reason = "needs a CUDA / ROCm GPU on Linux")
_triton = pytest.mark.skipif(not C.triton_decode_available(), reason = "needs Triton")


def _fetch(codec, blob, use_triton, dec = None):
    dev = torch.device("cuda", torch.cuda.current_device())
    dec = dec or C.DeviceDecoder(dev, codec.stage_nbytes, 16, use_triton = use_triton, ring = 2)
    out = torch.full((codec.raw_nbytes,), 0xAB, dtype = torch.uint8, device = dev)
    done = torch.cuda.Event()
    cs = torch.cuda.Stream(device = dev)
    cs.wait_stream(torch.cuda.current_stream(dev))
    dec.fetch(codec, blob.pin_memory(), out, cs, done)
    torch.cuda.current_stream(dev).wait_event(done)
    return out.cpu(), dec


@_gpu
@pytest.mark.parametrize("use_triton", [pytest.param(True, marks = _triton), False])
@pytest.mark.parametrize("mode", ["bf16", "bytes"])
def test_card_decode_round_trips_every_word(mode, use_triton):
    raw = torch.cat((_every_word(mode), _weights(40 * C.GROUP + 256)))
    codec, blob = _huff(raw, mode, chunk = 16 * C.GROUP)
    # More chunks than ring slots: staging slots are reused across chunks and across two fetches.
    assert codec.n_chunks > 2
    got, dec = _fetch(codec, blob, use_triton)
    assert torch.equal(got, raw) and dec.use_triton == use_triton
    got, _ = _fetch(codec, blob, use_triton, dec)
    assert torch.equal(got, raw)


@_gpu
@_triton
def test_card_decode_flags_a_corrupt_table():
    raw = _weights(2 * C.GROUP)
    codec, blob = _huff(raw, "bf16")
    lut = blob[:C._HUFF_LUT * 2].view(torch.int16)
    lut[(lut != 0).nonzero()[0]] ^= 1
    got, _ = _fetch(codec, blob, True)
    assert not torch.equal(got, raw)


@_gpu
@_triton
def test_self_check_turns_only_the_huffman_format_off_when_it_fails(monkeypatch):
    dev = torch.device("cuda", torch.cuda.current_device())
    assert C.self_check(dev)[:2] == (True, True) and not C._HUFF_BROKEN
    real = C._decode_huff_chunk

    def bad(ch, staged, mode, out_u8, use_triton):
        real(ch, staged, mode, out_u8, use_triton)
        out_u8[:1].add_(1)
    monkeypatch.setattr(C, "_decode_huff_chunk", bad)
    got = C.self_check(dev)
    assert got[:2] == (True, True), got
    assert C._HUFF_BROKEN and not C.huff_enabled()
    assert C.encode_region(_weights(C.GROUP))[1].kind == "v1"


@_gpu
@_triton
def test_decode_kernel_replays_bitwise_in_a_cuda_graph():
    # No host sync inside the decode launch: it captures, and the replay writes the same bytes.
    dev = torch.device("cuda", torch.cuda.current_device())
    raw = _weights(C.GROUP + 512)
    codec, blob = _huff(raw, "bf16")
    ch = codec.chunks[0]
    staged = blob[ch.host_off:ch.host_off + ch.nbytes].to(dev)
    out = torch.zeros(2 * ch.n, dtype = torch.uint8, device = dev)
    C._decode_huff_chunk(ch, staged, "bf16", out, True)   # compile outside the capture
    out.zero_()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream(device = dev)
    s.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(s), torch.cuda.graph(g, stream = s):
        C._decode_huff_chunk(ch, staged, "bf16", out, True)
    torch.cuda.current_stream(dev).wait_stream(s)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out.cpu(), raw[ch.raw_off:ch.raw_off + 2 * ch.n])


@_gpu
@_triton
def test_decode_kernel_compiles_once_per_mode():
    kernel = C._triton_kernels()[3]
    for n in (C.GROUP + 768, 2 * C.GROUP + 256, 3 * C.GROUP, 5 * C.GROUP + 512):
        for mode in ("bf16", "bytes"):
            raw = _weights(n, seed = n)
            codec, blob = _huff(raw, mode)
            got, _ = _fetch(codec, blob, True)
            assert torch.equal(got, raw)
    caches = getattr(kernel, "device_caches", None)
    if caches is None:
        pytest.skip("this Triton has no device_caches")
    dev = torch.cuda.current_device()
    entry = caches.get(dev) if hasattr(caches, "get") else None
    n_variants = len(entry[0]) if entry else sum(len(v[0]) for v in caches.values())
    assert n_variants <= 2, n_variants


# ------------------------------------------------------------------------------------- BlockSwap end to end
class _Layer(nn.Module):
    def __init__(self, w, fp8 = False):
        super().__init__()
        self.fp8 = fp8
        if fp8:
            g = torch.Generator().manual_seed(w)
            self.weight = nn.Parameter((torch.randn(w, w, generator = g) * 40).to(torch.float8_e4m3fn),
                                       requires_grad = False)
            self.scale = nn.Parameter(torch.full((w, 1), 0.02 / 40, dtype = torch.bfloat16), requires_grad = False)
        else:
            self.base = nn.Linear(w, w, bias = False, dtype = torch.bfloat16)
            nn.init.normal_(self.base.weight, std = 0.02)
        self.norm = nn.Parameter(torch.rand(w, dtype = torch.bfloat16) + 0.5, requires_grad = False)
        self.lora_A = nn.Linear(w, 8, bias = False, dtype = torch.float32)
        self.lora_B = nn.Linear(8, w, bias = False, dtype = torch.float32)
        nn.init.normal_(self.lora_B.weight, std = 0.02)

    def forward(self, x):
        h = (x * self.norm).to(torch.bfloat16)
        lora = self.lora_B(self.lora_A(x.float())).to(torch.bfloat16)
        if self.fp8:
            base = h @ (self.weight.to(torch.bfloat16) * self.scale).t()
        else:
            base = self.base(h)
        return x + torch.tanh(base + lora)


def _stack(n = 10, w = 512, fp8 = False):
    torch.manual_seed(0)
    layers = nn.ModuleList([_Layer(w, fp8) for _ in range(n)])
    for name, p in layers.named_parameters():
        p.requires_grad_("lora_" in name)
    return layers.to("cuda")


def _train(layers, sw = None, steps = 3, ga = 2):
    g = torch.Generator(device = "cuda").manual_seed(1)
    w = layers[0].norm.numel()
    xs = [torch.randn(4, 32, w, device = "cuda", generator = g, dtype = torch.bfloat16) for _ in range(steps * ga + 1)]
    params = [p for n, p in layers.named_parameters() if "lora_" in n]
    opt = torch.optim.SGD(params, lr = 0.5)
    losses, grads = [], []
    for s in range(steps):
        for m in range(ga):
            h = xs[s * ga + m].detach().requires_grad_(True)
            for layer in layers:
                h = checkpoint(layer, h, use_reentrant = False)
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


def _same(a, b):
    """Bitwise equality (float8 has no torch.equal on every build)."""
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.detach().contiguous().view(torch.uint8),
                                                                       b.detach().contiguous().view(torch.uint8))


def _limit_for(layers, ram_layers):
    per = _bs._Block(layers[0], {}, torch.device("cuda", torch.cuda.current_device())).nbytes()
    return 2 * _disk._round_up(per) + per + ram_layers * per + 1024


def _codec_gpu():
    return (torch.cuda.is_available() and sys.platform.startswith("linux") and C.triton_decode_available()
            and C.self_check(torch.device("cuda", torch.cuda.current_device()))[1])


_e2e = pytest.mark.skipif(not _codec_gpu(), reason = "needs a Linux GPU with the Triton decode")


def _disk_run(monkeypatch, base, codec_env, fp8_env = None):
    monkeypatch.setenv("UNSLOTH_BLOCK_SWAP_CODEC", codec_env)
    if fp8_env is not None:
        monkeypatch.setenv("UNSLOTH_OFFLOAD_DISK_COMPRESS_FP8", fp8_env)
    _disk._SPILL[:] = []
    layers = copy.deepcopy(base)
    monkeypatch.setenv("UNSLOTH_OFFLOAD_HOST_LIMIT", str(_limit_for(base, 2)))
    sw = _bs.BlockSwap(layers, 9, prefetch_depth = 2, placement = "tail")
    disk = [b for b in sw.blocks if b.disk is not None]
    kinds = [(c.kind, getattr(c, "mode", None)) for b in disk for c in b.codec.values() if c is not None]
    sw.stats()
    got = _train(layers, sw)
    st = sw._disk_tier.stats()
    per_read = st["disk_read_bytes"] / max(1, st["disk_reads"])
    want = {k: v for k, v in base.state_dict().items() if "lora_" not in k}
    sd = layers.state_dict()
    for k, v in want.items():
        assert _same(sd[k].to(v.device), v), k
    sw.remove()
    for a, b in zip(base.parameters(), layers.parameters()):
        if not a.requires_grad:
            assert _same(a, b)
    monkeypatch.delenv("UNSLOTH_OFFLOAD_HOST_LIMIT")
    return got, len(disk), kinds, per_read


@_e2e
def test_disk_tier_trains_bitwise_with_the_huffman_format_and_reads_fewer_bytes(monkeypatch):
    base = _stack()
    ref = _train(copy.deepcopy(base))
    got_h, n_h, kinds_h, read_h = _disk_run(monkeypatch, base, "huff")
    got_1, n_1, kinds_1, read_1 = _disk_run(monkeypatch, base, "v1")
    _equal(ref, got_h)
    _equal(ref, got_1)
    # Engagement, kill switch N vs 0.
    assert n_h == n_1 == 7
    assert kinds_h.count(("huff", "bf16")) == 7 and not any(k == "huff" for k, _ in kinds_1), (kinds_h, kinds_1)
    assert read_h < 0.93 * read_1, (read_h, read_1)


@_e2e
def test_fp8_disk_blocks_are_byte_coded_and_train_bitwise(monkeypatch):
    base = _stack(w = 1024, fp8 = True)
    ref = _train(copy.deepcopy(base))
    got, n, kinds, read_on = _disk_run(monkeypatch, base, "huff")
    _equal(ref, got)
    assert n == 7 and kinds.count(("huff", "bytes")) == 7, kinds
    got_off, _, kinds_off, read_off = _disk_run(monkeypatch, base, "huff", fp8_env = "0")
    _equal(ref, got_off)
    assert kinds_off == [] and read_on < 0.9 * read_off, (read_on, read_off)
