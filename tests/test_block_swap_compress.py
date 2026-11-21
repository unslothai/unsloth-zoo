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
# block_swap finds its codec through the package import; standalone, hand it the copy loaded above.
_bs._CODEC.append(C)


def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    try:
        torch.empty(1, device = "cuda")
    except Exception:
        pytest.skip("needs a real CUDA device")


def _require_compression():
    _cuda()
    if not C.triton_decode_available():
        pytest.skip("needs the Triton decode (compression stays off without it)")
    if not _bs._can_register():
        pytest.skip("needs host registration (the chunked fallback stays raw)")


def _encode(raw, **kw):
    kw.setdefault("max_escape_fraction", 1.0)
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


def test_escape_heavy_regions_stay_raw():
    raw = _weights(4 * C.GROUP)
    w = raw.view(torch.int16)
    g = torch.Generator().manual_seed(3)
    w[::50] = torch.randint(-32768, 32767, (w[::50].numel(),), generator = g, dtype = torch.int16)
    assert C.encode_region(raw, max_ratio = 10.0) is None, "2% escapes is past the 0.5% cap"
    assert C.encode_region(raw, max_ratio = 10.0, max_escape_fraction = 1.0) is not None
    assert C.encode_region(torch.zeros(0, dtype = torch.uint8)) is None


def test_streaming_encode_writes_the_same_blob():
    raw = _weights(3 * C.GROUP + 512)
    codec, blob = _encode(raw, chunk = C.GROUP)
    out = torch.full((raw.numel(),), 0xEE, dtype = torch.uint8)
    parts, c2 = C.encode_region(raw, chunk = C.GROUP, out = out, max_escape_fraction = 1.0)
    C.write_parts(out, parts)
    assert c2.nbytes == codec.nbytes and torch.equal(out[:c2.nbytes], blob)
    assert C.max_aux_nbytes(raw.numel(), 1.0) >= codec.aux_nbytes


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
    assert big.vals_off >= big.pos_off + 3 * 8 and big.nbytes == big.vals_off + 3


def test_aux_offsets_are_16_byte_aligned():
    # One patch-kernel variant: Triton specializes pointer arguments on 16 byte alignment.
    for esc in (0, 1, 3, 17, 1000):
        c = C.RegionCodec(5 * C.GROUP + 256, C.GROUP, esc)
        assert c.aux_off % 16 == 0 and c.pos_off % 16 == 0 and c.vals_off % 16 == 0


def test_bad_geometry_is_refused():
    for args in ((0, C.CHUNK, 0), (300, C.CHUNK, 0), (256, 0, 0), (256, C.GROUP + 1, 0), (256, C.GROUP, -1)):
        with pytest.raises(ValueError):
            C.RegionCodec(*args)
    with pytest.raises(ValueError):
        C.encode_region(_weights(C.GROUP), chunk = 0)
    with pytest.raises(ValueError):
        C.encode_region(_weights(C.GROUP), out = torch.zeros(10, dtype = torch.uint8))


def test_kernels_resolve_names_from_module_globals():
    # Triton 3.2 looks names in a kernel body up in the defining module's globals (not an enclosing function).
    if not C.triton_decode_available():
        pytest.skip("no Triton")
    for k in C._triton_kernels():
        assert k.fn.__globals__.get("tl") is not None, k.fn.__name__


def test_encode_works_under_strict_determinism():
    raw = _weights(2 * C.GROUP)
    torch.use_deterministic_algorithms(True)
    try:
        codec, blob = _encode(raw)
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(False)
    assert torch.equal(C.decode_region_cpu(codec, blob), raw)


def test_packing_keeps_the_encoder_scratch_in_the_published_reservation(monkeypatch):
    # Peers read the published bytes while this rank still has blocks to encode through one raw region of scratch.
    seen = []
    monkeypatch.setattr(_bs, "_publish_host_pending", lambda n: seen.append(n))
    monkeypatch.setattr(_bs, "_trim_host_heap", lambda: None)

    class B:
        def __init__(self, ok):
            self.ok = ok

        def compressible(self):
            return self.ok

    sw = _bs.BlockSwap.__new__(_bs.BlockSwap)
    sw.blocks = [B(True), B(True), B(False)]
    sw._host_credit = [0, 0, 0]
    sw._host_own = 1
    sw._host_walk = _bs._HostWalk([100, 200, 300], [0, 0, 0], 100)
    sw._encode_scratch = 1000
    for k in range(3):
        sw._publish_packed(k)
    assert seen == [sw._host_walk.pending(0) + 1000, sw._host_walk.pending(1), sw._host_walk.pending(2)]


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


@pytest.mark.parametrize("use_triton,vec", [(True, 1), (True, 0), (False, 1)])
def test_device_decode_is_bitwise(use_triton, vec, monkeypatch):
    _cuda()
    if use_triton and not C.triton_decode_available():
        pytest.skip("no Triton")
    monkeypatch.setattr(C, "_VEC", vec)
    words = torch.arange(-32768, 32768, dtype = torch.int32).to(torch.int16).view(torch.uint8)
    for raw in (_weights(5 * C.GROUP + 256), torch.cat((words, _weights(3 * C.GROUP)))):
        got, dec = _device_round_trip(raw, 2 * C.GROUP, use_triton)
        assert dec.use_triton == use_triton
        assert torch.equal(got, raw)


def test_a_failed_submission_leaves_the_decoder_usable(monkeypatch):
    # An OOM mid-fetch propagates; every ring / aux slot is released after queued work, so the next fetch is exact.
    _cuda()
    raw = _weights(4 * C.GROUP)
    codec, blob = _encode(raw, chunk = C.GROUP)
    dec = C.DeviceDecoder("cuda", C.GROUP * 3 // 2, codec.aux_nbytes, ring = 2)
    real, calls = C._decode_chunk, []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 3:
            raise torch.cuda.OutOfMemoryError("simulated")
        return real(*a, **k)

    monkeypatch.setattr(C, "_decode_chunk", flaky)
    out = torch.zeros(codec.raw_nbytes, dtype = torch.uint8, device = "cuda")
    cs = torch.cuda.Stream()
    with pytest.raises(torch.cuda.OutOfMemoryError):
        dec.fetch(codec, blob.pin_memory(), out, cs, torch.cuda.Event())
    monkeypatch.setattr(C, "_decode_chunk", real)
    done = torch.cuda.Event()
    cs.wait_stream(torch.cuda.current_stream())
    dec.fetch(codec, blob.pin_memory(), out, cs, done)
    torch.cuda.current_stream().wait_event(done)
    assert torch.equal(out.cpu(), raw)


def test_fetch_refuses_buffers_smaller_than_the_region():
    _cuda()
    raw = _weights(2 * C.GROUP)
    codec, blob = _encode(raw, chunk = C.GROUP)
    dec = C.DeviceDecoder("cuda", C.GROUP * 3 // 2, codec.aux_nbytes)
    out = torch.zeros(codec.raw_nbytes - 512, dtype = torch.uint8, device = "cuda")
    with pytest.raises(ValueError):
        dec.fetch(codec, blob, out, torch.cuda.Stream(), torch.cuda.Event())
    with pytest.raises(ValueError):
        C.decode_region_cpu(codec, blob[:-1])


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
    def __init__(self, d, bias = False):
        super().__init__()
        self.w = nn.Linear(d, d, bias = False, dtype = torch.bfloat16).requires_grad_(False)
        self.up = nn.Linear(d, 2 * d, bias = bias, dtype = torch.bfloat16).requires_grad_(False)
        # Named like a norm's `.weight` (real models: input_layernorm.weight).
        self.norm_weight = nn.Parameter(torch.ones(d, dtype = torch.bfloat16), requires_grad = False)
        self.lora_A = nn.Parameter(torch.randn(d, 8, dtype = torch.bfloat16) * 0.01)

    def forward(self, x):
        h = x * self.norm_weight
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


def _layers(n = 6, d = 512, bias = False):
    try:
        torch.ones(2, 2, device = "cuda", dtype = torch.bfloat16) @ torch.ones(2, 2, device = "cuda", dtype = torch.bfloat16)
    except RuntimeError:
        pytest.skip("no bf16 matmul on this card")
    torch.manual_seed(0)
    return nn.ModuleList([_Layer(d, bias) for _ in range(n)]).cuda()


@pytest.mark.parametrize("mode", ["non_reentrant", "reentrant", "none"])
def test_compressed_swap_is_bitwise_to_raw_swap_and_no_swap(monkeypatch, mode):
    _require_compression()
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
    _require_compression()
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
    evicted = [p for b in sw.blocks if not b.resident for p in b.params]
    assert evicted
    sw.remove()
    for k, v in layers.state_dict().items():
        assert torch.equal(v, ref[k]) and v.device.type == "cuda", k
    # Each restored weight owns its storage (safetensors refuses shared ones).
    ptrs = [p.untyped_storage().data_ptr() for p in evicted]
    assert len(set(ptrs)) == len(ptrs)


class _Net(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, x):
        for b in self.layers:
            x = b(x)
        return x


def _peft_state(compress, monkeypatch, tmp_path):
    peft = pytest.importorskip("peft")
    from safetensors.torch import load_file, save_file
    net = _Net(_layers(4, 256, bias = True))
    model = peft.get_peft_model(net, peft.LoraConfig(r = 4, target_modules = ["w"], modules_to_save = ["up"], bias = "all"))
    # Frozen copies peft still saves by key: modules_to_save weights and (bias = "all") every bias.
    for name, p in model.named_parameters():
        if "lora_" not in name:
            p.requires_grad_(False)
    ref = {k: v.detach().clone().cpu() for k, v in peft.get_peft_model_state_dict(model).items()}
    monkeypatch.setenv("UNSLOTH_BLOCK_SWAP_COMPRESS", "1" if compress else "0")
    sw = _bs.BlockSwap(model.base_model.model.layers, [1, 2, 3], prefetch_depth = 1, device = "cuda")
    evicted = sum(not b.resident for b in sw.blocks)
    state = peft.get_peft_model_state_dict(model)
    path = os.path.join(str(tmp_path), f"adapter_{int(compress)}.safetensors")
    # Own storage per entry, as peft's save_pretrained does for tensors that share a buffer.
    save_file({k: v.detach().cpu().clone() for k, v in state.items()}, path)
    back = load_file(path)
    stats = sw.compression_stats()
    sw.remove()
    return ref, back, evicted, stats


def test_real_peft_adapter_save_reloads_every_selected_frozen_weight(monkeypatch, tmp_path):
    _require_compression()
    ref, back, evicted, stats = _peft_state(True, monkeypatch, tmp_path)
    assert evicted >= 1 and stats["layers"] == 3
    assert any("modules_to_save" in k for k in ref) and any(k.endswith("bias") for k in ref)
    assert set(back) == set(ref)
    for k, v in ref.items():
        assert back[k].shape == v.shape and torch.equal(back[k], v), k
    ref0, back0, _, _ = _peft_state(False, monkeypatch, tmp_path)
    assert all(torch.equal(back0[k], ref0[k]) for k in ref0)


def test_state_dict_under_inference_mode_returns_normal_tensors(monkeypatch):
    _require_compression()
    layers = _layers(3, 256)
    monkeypatch.setenv("UNSLOTH_BLOCK_SWAP_COMPRESS", "1")
    sw = _bs.BlockSwap(layers, [1, 2], prefetch_depth = 1, device = "cuda")
    with torch.inference_mode():
        sd = layers.state_dict()
    assert not any(v.is_inference() for v in sd.values())
    sw.remove()


def test_install_falls_back_to_raw_when_only_the_encoder_scratch_does_not_fit(monkeypatch):
    _require_compression()
    real = _bs._check_host_fit

    def tight(sizes, on_host = 0, slack = 0):
        if slack > 0:
            raise ValueError("simulated: no host room for the scratch")
        return real(sizes, on_host = on_host, slack = slack)

    monkeypatch.setattr(_bs, "_check_host_fit", tight)
    ref, _, _ = _run(_layers(3, 256), False, monkeypatch, None)
    got, sw, st = _run(_layers(3, 256), True, monkeypatch, [1, 2])
    assert st["compression"]["enabled"] is False and st["compression"]["decoder_bytes"] == 0
    assert all(torch.equal(x, y) for a, b in zip(ref, got) for x, y in zip(a, b))
    sw.remove()


def test_install_stays_raw_when_decode_buffers_and_slots_do_not_fit(monkeypatch):
    _require_compression()
    layers = _layers(3, 256)
    for b in layers:   # host-loaded originals: packing frees nothing on the card
        b.w.weight.data, b.up.weight.data = b.w.weight.data.cpu(), b.up.weight.data.cpu()
        b.norm_weight.data = b.norm_weight.data.cpu()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (1 << 20, 1 << 30))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a, **k: 0)
    monkeypatch.setenv("UNSLOTH_BLOCK_SWAP_COMPRESS", "1")
    sw = _bs.BlockSwap(layers, [0, 1, 2], prefetch_depth = 1, device = "cuda")
    st = sw.compression_stats()
    assert st["enabled"] is False and st["decoder_bytes"] == 0, st
    sw.remove()


def test_no_decoders_are_kept_when_every_region_stays_raw(monkeypatch):
    _require_compression()
    layers = _layers(3, 256)
    g = torch.Generator().manual_seed(0)
    for p in layers.parameters():   # random bit patterns: bf16 dtype, nothing to compress
        if not p.requires_grad:
            p.data.view(torch.int16).copy_(torch.randint(-32768, 32767, p.shape, generator = g, dtype = torch.int16))
    monkeypatch.setenv("UNSLOTH_BLOCK_SWAP_COMPRESS", "1")
    sw = _bs.BlockSwap(layers, [1, 2], prefetch_depth = 1, device = "cuda")
    st = sw.compression_stats()
    assert st["layers"] == 0 and st["decoder_bytes"] == 0 and st["enabled"] is False, st
    sw.remove()


def test_install_under_strict_determinism(monkeypatch):
    _require_compression()
    torch.use_deterministic_algorithms(True, warn_only = False)
    try:
        monkeypatch.setenv("UNSLOTH_BLOCK_SWAP_COMPRESS", "1")
        sw = _bs.BlockSwap(_layers(3, 256), [1, 2], prefetch_depth = 1, device = "cuda")
        assert sw.compression_stats()["layers"] == 2
        sw.remove()
    finally:
        torch.use_deterministic_algorithms(False)


def test_remove_falls_back_to_a_host_decode_when_the_card_is_full(monkeypatch):
    _require_compression()
    layers = _layers(4, 256)
    ref = {k: v.clone() for k, v in layers.state_dict().items()}
    monkeypatch.setenv("UNSLOTH_BLOCK_SWAP_COMPRESS", "1")
    sw = _bs.BlockSwap(layers, [0, 2, 3], prefetch_depth = 1, device = "cuda")
    assert any(not b.resident for b in sw.blocks)

    def full(self, d, target):
        if torch.device(target).type == "cuda":
            raise torch.cuda.OutOfMemoryError("simulated")
        return orig(self, d, target)

    orig = _bs._Block._region_on
    monkeypatch.setattr(_bs._Block, "_region_on", full)
    sw.remove()
    for k, v in layers.state_dict().items():
        assert v.device.type == "cuda" and torch.equal(v, ref[k]), k


def test_compression_without_triton_stays_raw_and_exact(monkeypatch):
    _cuda()
    if not _bs._can_register():
        pytest.skip("needs host registration")
    monkeypatch.setattr(_bs._codec(), "triton_decode_available", lambda: False)
    layers = _layers(3, 256)
    ref, _, _ = _run(layers, False, monkeypatch, None)
    got, sw, st = _run(layers, True, monkeypatch, [0, 1, 2])
    assert st["compression"]["enabled"] is False and st["compression"]["fetches"] == 0
    assert st["wire_bytes"] == st["copy_bytes"]
    for a, b in zip(ref, got):
        assert all(torch.equal(x, y) for x, y in zip(a, b))
    sw.remove()


def test_compressed_swap_under_torch_compile_adds_no_graph_breaks(monkeypatch):
    _require_compression()
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
            before = sw.compression_stats()["fetches"]
            out = torch.compile(f)(x)
            fetched = sw.compression_stats()["fetches"] - before
            graphs = counters["stats"]["unique_graphs"]
            eager = f(x)
        n = sum(counters["graph_break"].values())
        sw.remove()
        return out, eager, n, fetched, graphs

    o0, e0, n0, f0, g0 = breaks(False)
    o1, e1, n1, f1, g1 = breaks(True)
    assert g0 > 0 and g1 > 0, "a graph was compiled"
    assert f1 > 0 and f0 == 0, "the compiled call itself ran compressed fetches"
    assert n1 == n0, (n0, n1)
    assert torch.equal(o0, o1) and torch.equal(e0, e1)
