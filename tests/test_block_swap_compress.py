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

"""Lossless bf16 codec for block swap fetches (unsloth_zoo/block_swap_codec.py): exact round trips on CPU, and on
a GPU the decode kernel, its torch fallback and BlockSwap with UNSLOTH_BLOCK_SWAP_COMPRESS=1 against raw fetches.
Loaded from file, like test_block_swap.py."""

import importlib.util
import os

import pytest
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, "unsloth_zoo", file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("block_swap_codec_under_test", "block_swap_codec.py")
_bs = _load("block_swap_compress_under_test", "block_swap.py")


def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    try:
        torch.empty(1, device = "cuda")
    except Exception:
        pytest.skip("needs a real CUDA device")


def _encode(raw, **kw):
    enc = C.encode_region(raw, **kw)
    assert enc is not None
    parts, codec = enc
    blob = torch.zeros(codec.nbytes, dtype = torch.uint8)
    C.write_parts(blob, parts)
    return codec, blob


def _weights(words, seed = 0, scale = 0.02):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(words, generator = g) * scale).to(torch.bfloat16).view(torch.uint8).clone()


def test_every_16_bit_word_round_trips():
    # All 65536 words (NaN payloads, infs, subnormals, -0) in one region, twice in a shuffled order.
    words = torch.arange(-32768, 32768, dtype = torch.int32).to(torch.int16)
    g = torch.Generator().manual_seed(1)
    words = torch.cat((words, words[torch.randperm(words.numel(), generator = g)]))
    raw = words.view(torch.uint8).clone()
    codec, blob = _encode(raw, chunk = C.GROUP, max_ratio = 10.0)
    assert codec.esc_n > 0
    assert torch.equal(C.decode_region_cpu(codec, blob), raw)


def test_weight_like_words_compress_by_a_quarter_and_round_trip():
    raw = _weights(3 * C.GROUP + 512)
    codec, blob = _encode(raw, chunk = 2 * C.GROUP)
    assert codec.n_chunks == 2 and codec.words == raw.numel() // 2
    assert codec.nbytes / codec.raw_nbytes < 0.76
    assert torch.equal(C.decode_region_cpu(codec, blob), raw)


def test_patching_is_what_makes_escapes_exact():
    # Flip check: skipping the escape patch must break the round trip whenever escapes exist.
    raw = _weights(2 * C.GROUP)
    raw.view(torch.int16)[::1000] = 0x7F80   # inf: never among a weight group's 15 common exponents
    codec, blob = _encode(raw)
    assert codec.esc_n > 0
    out = torch.empty_like(raw)
    tables = blob[codec.aux_off:codec.aux_off + codec.tables_bytes]
    for j in range(codec.n_chunks):
        s, k, off = codec.chunk_span(j)
        C._decode_chunk_torch(blob[off:off + k * 3 // 2], tables, s, k, out[2 * s:2 * (s + k)])
    assert not torch.equal(out, raw)
    assert torch.equal(C.decode_region_cpu(codec, blob), raw)


def test_a_corrupt_table_is_detected():
    raw = _weights(C.GROUP)
    codec, blob = _encode(raw)
    blob[codec.aux_off] ^= 1
    assert not torch.equal(C.decode_region_cpu(codec, blob), raw)


def test_incompressible_words_stay_raw():
    g = torch.Generator().manual_seed(0)
    raw = torch.randint(0, 256, (4 * C.GROUP,), dtype = torch.uint8, generator = g)
    assert C.encode_region(raw) is None
    # fp16 weights: bits 14..7 mix exponent and mantissa, so the 4 bit table cannot cover them.
    fp16 = (torch.randn(2 * C.GROUP, generator = g) * 0.02).half().view(torch.uint8).clone()
    assert C.encode_region(fp16) is None
    # Not a whole number of 512 byte units: left alone.
    assert C.encode_region(torch.zeros(1000, dtype = torch.uint8)) is None


def test_positions_switch_to_int64_past_two_billion_words():
    assert C.RegionCodec(2**31 - 256, C.CHUNK, 3).pos_dtype == torch.int32
    big = C.RegionCodec(2**31 + 256, C.CHUNK, 3)
    assert big.pos_dtype == torch.int64 and big.pos_off % 8 == 0
    assert big.nbytes == big.pos_off + 3 * 8 + 3


def test_compression_is_off_by_default(monkeypatch):
    monkeypatch.delenv("UNSLOTH_BLOCK_SWAP_COMPRESS", raising = False)
    assert not _bs._compress_requested()
    monkeypatch.setenv("UNSLOTH_BLOCK_SWAP_COMPRESS", "1")
    assert _bs._compress_requested()
    monkeypatch.setenv("UNSLOTH_BLOCK_SWAP_COMPRESS", "0")
    assert not _bs._compress_requested()


# ----------------------------------------------------------------------------------------------------------- GPU
def _device_round_trip(raw, chunk, use_triton):
    codec, blob = _encode(raw, chunk = chunk, max_ratio = 10.0)
    dec = C.DeviceDecoder("cuda", min(chunk, codec.words) * 3 // 2, codec.aux_nbytes, use_triton = use_triton)
    out = torch.full((codec.raw_nbytes,), 0xCD, dtype = torch.uint8, device = "cuda")
    cs = torch.cuda.Stream()
    cs.wait_stream(torch.cuda.current_stream())
    done = torch.cuda.Event()
    dec.fetch(codec, blob.pin_memory(), out, cs, done)
    torch.cuda.current_stream().wait_event(done)
    return out.cpu(), dec


@pytest.mark.parametrize("use_triton", [True, False])
def test_device_decode_is_bitwise(use_triton):
    _cuda()
    if use_triton and not C.triton_decode_available():
        pytest.skip("no Triton")
    words = torch.arange(-32768, 32768, dtype = torch.int32).to(torch.int16).view(torch.uint8)
    for raw in (_weights(5 * C.GROUP + 256), torch.cat((words, _weights(3 * C.GROUP)))):
        got, dec = _device_round_trip(raw, 2 * C.GROUP, use_triton)
        assert dec.use_triton == use_triton
        assert torch.equal(got, raw)


def test_failed_kernel_launch_falls_back_to_the_torch_decode(monkeypatch):
    _cuda()
    if not C.triton_decode_available():
        pytest.skip("no Triton")
    raw = _weights(3 * C.GROUP)

    def boom(*a, **k):
        raise RuntimeError("simulated launch failure")

    monkeypatch.setattr(C, "_decode_chunk", boom)
    got, dec = _device_round_trip(raw, C.GROUP, True)
    assert not dec.use_triton and "simulated" in dec.failed
    assert torch.equal(got, raw)


def test_self_check_passes():
    _cuda()
    ok, triton_on, why = C.self_check("cuda")
    assert ok, why
    assert triton_on == C.triton_decode_available()


class _Layer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, d, bias = False, dtype = torch.bfloat16).requires_grad_(False)
        self.up = nn.Linear(d, 2 * d, bias = False, dtype = torch.bfloat16).requires_grad_(False)
        self.norm = nn.Parameter(torch.ones(d, dtype = torch.bfloat16), requires_grad = False)
        self.lora_A = nn.Parameter(torch.randn(d, 8, dtype = torch.bfloat16) * 0.01)

    def forward(self, x):
        h = x * self.norm
        h = self.w(h) + (h @ self.lora_A) @ self.lora_A.t()
        return x + self.up(torch.tanh(h)).view(*x.shape, 2).sum(-1)


def _run(layers, compress, monkeypatch, idx, mode = "non_reentrant", steps = 2):
    import torch.utils.checkpoint as cp
    monkeypatch.setenv("UNSLOTH_BLOCK_SWAP_COMPRESS", "1" if compress else "0")
    sw = _bs.BlockSwap(layers, idx, prefetch_depth = 1, device = "cuda") if idx is not None else None
    torch.manual_seed(0)
    x = torch.randn(4, 64, layers[0].w.in_features, device = "cuda", dtype = torch.bfloat16)
    outs = []
    for _ in range(steps):
        for b in layers:
            b.lora_A.grad = None
        h = x.requires_grad_(True)
        for b in layers:
            h = b(h) if mode == "none" else cp.checkpoint(b, h, use_reentrant = mode == "reentrant")
        loss = h.float().square().mean()
        loss.backward()
        torch.cuda.synchronize()
        outs.append([loss.detach().clone()] + [b.lora_A.grad.clone() for b in layers])
        x = x.detach()
    st = sw.stats() if sw is not None else None
    return outs, sw, st


def _layers(n = 6, d = 512):
    try:
        torch.ones(2, 2, device = "cuda", dtype = torch.bfloat16) @ torch.ones(2, 2, device = "cuda", dtype = torch.bfloat16)
    except RuntimeError:
        pytest.skip("no bf16 matmul on this card")
    torch.manual_seed(0)
    return nn.ModuleList([_Layer(d) for _ in range(n)]).cuda()


@pytest.mark.parametrize("mode", ["non_reentrant", "reentrant", "none"])
def test_compressed_swap_is_bitwise_to_raw_swap_and_no_swap(monkeypatch, mode):
    _cuda()
    layers = _layers()
    ref, _, _ = _run(layers, False, monkeypatch, None, mode)
    raw, sw, st = _run(layers, False, monkeypatch, [1, 3, 5], mode)
    assert st["compression"]["enabled"] is False and st["compression"]["fetches"] == 0
    sw.remove()
    got, sw, st = _run(layers, True, monkeypatch, [1, 3, 5], mode)
    comp = st["compression"]
    assert comp["enabled"] and comp["layers"] == 3 and comp["fetches"] > 0, comp
    assert comp["stored_bytes"] < 0.8 * comp["raw_bytes"], comp
    assert st["wire_bytes"] < 0.8 * st["copy_bytes"], st
    for a, b, c in zip(ref, raw, got):
        assert all(torch.equal(x, y) for x, y in zip(a, b))
        assert all(torch.equal(x, y) for x, y in zip(a, c))
    sw.remove()


def test_state_dict_and_remove_restore_the_exact_weights(monkeypatch):
    _cuda()
    layers = _layers(4, 256)
    for b in layers:
        b.lora_A.requires_grad_(False)
    ref = {k: v.clone() for k, v in layers.state_dict().items()}
    monkeypatch.setenv("UNSLOTH_BLOCK_SWAP_COMPRESS", "1")
    sw = _bs.BlockSwap(layers, [0, 2, 3], prefetch_depth = 1, device = "cuda")
    assert sw.compression_stats()["layers"] == 3
    assert any(b.host[0] is None for b in sw.blocks), "compressed blocks keep no raw host views"
    sd = layers.state_dict()
    for k, v in ref.items():
        assert torch.equal(sd[k].to(v.device), v), k
    sw.remove()
    for k, v in layers.state_dict().items():
        assert torch.equal(v, ref[k]) and v.device.type == "cuda", k
    # Each restored weight owns its storage (safetensors refuses shared ones).
    ptrs = [p.untyped_storage().data_ptr() for p in layers.parameters()]
    assert len(set(ptrs)) == len(ptrs)


def test_peft_adapter_state_dict_skips_decoding(monkeypatch):
    _cuda()
    layers = _layers(3, 256)
    monkeypatch.setenv("UNSLOTH_BLOCK_SWAP_COMPRESS", "1")
    sw = _bs.BlockSwap(layers, [1, 2], prefetch_depth = 1, device = "cuda")
    calls = []
    orig = _bs._Block.host_views
    monkeypatch.setattr(_bs._Block, "host_views", lambda self, *a, **k: calls.append(1) or orig(self, *a, **k))

    ns = {"__name__": "peft.utils.save_and_load", "layers": layers}
    exec("def get_peft_model_state_dict():\n    return layers.state_dict()", ns)
    sd = ns["get_peft_model_state_dict"]()
    assert calls == [] and sd["1.w.weight"].numel() == 0
    sd = layers.state_dict()
    assert len(calls) == 2 and sd["1.w.weight"].shape == (256, 256)
    sw.remove()


def test_compressed_swap_under_torch_compile_adds_no_graph_breaks(monkeypatch):
    _cuda()
    import torch._dynamo as dynamo
    from torch._dynamo.utils import counters

    def breaks(compress):
        layers = _layers(4, 256)
        monkeypatch.setenv("UNSLOTH_BLOCK_SWAP_COMPRESS", "1" if compress else "0")
        sw = _bs.BlockSwap(layers, [1, 3], prefetch_depth = 1, device = "cuda")
        dynamo.reset()
        counters.clear()
        x = torch.randn(2, 16, 256, device = "cuda", dtype = torch.bfloat16)

        def f(x):
            for b in layers:
                x = b(x)
            return x

        with torch.no_grad():
            out = torch.compile(f)(x)
            eager = f(x)
        n = sum(counters["graph_break"].values())
        st = sw.compression_stats()
        sw.remove()
        return out, eager, n, st

    o0, e0, n0, s0 = breaks(False)
    o1, e1, n1, s1 = breaks(True)
    assert s1["fetches"] > 0 and s0["fetches"] == 0
    assert n1 == n0, (n0, n1)
    assert torch.equal(o0, o1) and torch.equal(e0, e1)
