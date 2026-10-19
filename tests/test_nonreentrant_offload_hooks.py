# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""saved_tensors_hooks offload for torch's non-reentrant checkpoint.

Gradients must match plain non-reentrant checkpointing bit for bit in every mode: side
stream pinned copies, unsloth_stream (keep last N on device, reload i-1 ahead), and the
pageable fallback once the pinned cap is hit. The engagement counters prove each path ran.
"""
import functools

import pytest
import torch
import torch.utils.checkpoint as ckpt


@pytest.fixture
def gc_module(monkeypatch):
    monkeypatch.setenv("UNSLOTH_ALLOW_CPU", "1")
    try:
        from unsloth_zoo import gradient_checkpointing as gc_module
    except ImportError as e:
        pytest.skip(f"unsloth_zoo import unavailable: {e}")
    monkeypatch.setattr(gc_module, "GC_PINNED_CAP", None, raising = False)
    monkeypatch.setattr(gc_module, "GC_PINNED_BYTES", 0, raising = False)
    monkeypatch.setattr(gc_module, "_PINNED_CACHED", {}, raising = False)
    monkeypatch.delenv("UNSLOTH_GC_PINNED_MAX_GB", raising = False)
    return gc_module


def test_cpu_tensors_and_frozen_inputs_pass_through(gc_module):
    offloader = gc_module.NonReentrantActivationOffloader()
    frozen = torch.ones(4)
    assert offloader.unpack(offloader.pack(frozen)) is frozen
    live = torch.ones(4, requires_grad = True)
    assert offloader.unpack(offloader.pack(live)) is live


def test_stream_backend_is_opt_in(gc_module, monkeypatch):
    monkeypatch.delenv("UNSLOTH_GC_OFFLOAD_BACKEND", raising = False)
    assert not gc_module.NonReentrantActivationOffloader().stream_mode
    monkeypatch.setenv("UNSLOTH_GC_OFFLOAD_BACKEND", "unsloth_stream")
    assert gc_module.NonReentrantActivationOffloader().stream_mode


def _run(hooks, layers, seed = 0):
    torch.manual_seed(seed)
    x = torch.randn(4, 64, 256, device = "cuda", requires_grad = True)
    h = x
    for layer in layers:
        if hooks is None:
            h = ckpt.checkpoint(layer, h, use_reentrant = False)
        else:
            with hooks():
                h = ckpt.checkpoint(layer, h, use_reentrant = False)
    loss = h.float().square().mean()
    loss.backward()
    grads = [x.grad.clone()] + [p.grad.clone() for layer in layers for p in layer.parameters()]
    for layer in layers: layer.zero_grad(set_to_none = True)
    return loss.detach(), grads


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
@pytest.mark.parametrize("mode", ["default", "side1", "side2", "stream_keep0", "stream_keep1", "stream_keep2", "pageable"])
def test_gradients_match_plain_checkpoint(gc_module, monkeypatch, mode):
    torch.manual_seed(0)
    layers = [
        torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.GELU(), torch.nn.Linear(512, 256)).cuda()
        for _ in range(6)
    ]
    if mode == "pageable":
        monkeypatch.setenv("UNSLOTH_GC_PINNED_MAX_GB", "0")
    if mode.startswith("stream"):
        monkeypatch.setenv("UNSLOTH_GC_STREAM_KEEP_LAST", mode[-1])
    if mode.startswith("side"):
        monkeypatch.setenv("UNSLOTH_GC_MAX_INFLIGHT_COPIES", mode[-1])
    offloader = gc_module.NonReentrantActivationOffloader(stream_mode = mode.startswith("stream"))
    ref_loss, ref_grads = _run(None, layers)
    for _ in range(2):  # the second step reuses cached pinned blocks
        loss, grads = _run(offloader.hooks, layers)
        assert torch.equal(loss, ref_loss)
        for a, b in zip(grads, ref_grads):
            assert torch.equal(a, b)
    stats = offloader.stats
    if mode == "pageable":
        assert stats["pageable"] > 0 and stats["pinned"] == 0
    else:
        assert stats["pinned"] > 0 and stats["pageable"] == 0
    if mode.startswith("stream"):
        assert stats["prefetch_hits"] == stats["prefetched"] > 0
    if mode.startswith("side"):
        assert offloader.d2h_streams and len(offloader.inflight[0]) <= int(mode[-1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
def test_side_stream_host_copy_is_pinned(gc_module):
    # Bytes are charged to the pinned cap only for a destination that really is pinned:
    # a non_blocking CUDA to CPU .to() allocates it from the pinned host allocator.
    offloader = gc_module.NonReentrantActivationOffloader(stream_mode = False)
    x = torch.randn(1024, 1024, device = "cuda", requires_grad = True)
    record = offloader.pack(x)
    torch.cuda.synchronize()
    assert offloader.stats["pinned"] == 1 and record.pinned_bytes > 0
    assert record.host.is_pinned()
    assert torch.equal(offloader.unpack(record), x.detach())


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
def test_pinned_charge_matches_the_caching_host_allocator(gc_module):
    if not hasattr(torch.cuda, "host_memory_stats"):
        pytest.skip("torch < 2.7 has no host_memory_stats")
    held = lambda: torch.cuda.host_memory_stats()["allocated_bytes.current"]
    offloader = gc_module.NonReentrantActivationOffloader(stream_mode = False)
    x = torch.randn(3, 1000, 1000, device = "cuda", requires_grad = True)  # 12 MB
    before = held()
    record = offloader.pack(x)
    torch.cuda.synchronize()
    # torch hands out a 16 MiB block for 12 MB, and the cap is charged the same.
    assert held() - before == gc_module.GC_PINNED_BYTES == 1 << 24
    del record
    # The freed block stays pinned in the host cache and stays charged.
    assert held() - before == gc_module.GC_PINNED_BYTES == 1 << 24
    record = offloader.pack(x)
    torch.cuda.synchronize()
    assert held() - before == gc_module.GC_PINNED_BYTES == 1 << 24  # block reused
    assert torch.equal(offloader.unpack(record), x.detach())


class _TwoInputLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(256, 256)
        self.b = torch.nn.Linear(256, 256)

    def forward(self, h, g):
        return torch.tanh(self.a(h) + self.b(g)), g * 1.5


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
@pytest.mark.parametrize("keep_last", ["0", "1"])
def test_stream_prefetch_skips_inputs_already_unpacked(gc_module, monkeypatch, keep_last):
    # Two saved inputs per checkpoint: unpacking the second must not reload the first again.
    monkeypatch.setenv("UNSLOTH_GC_STREAM_KEEP_LAST", keep_last)
    torch.manual_seed(0)
    layers = [_TwoInputLayer().cuda() for _ in range(5)]
    def run(hooks):
        torch.manual_seed(1)
        h = torch.randn(4, 64, 256, device = "cuda", requires_grad = True)
        g = torch.randn(4, 64, 256, device = "cuda", requires_grad = True)
        x = (h, g)
        for layer in layers:
            if hooks is None:
                h, g = ckpt.checkpoint(layer, h, g, use_reentrant = False)
            else:
                with hooks():
                    h, g = ckpt.checkpoint(layer, h, g, use_reentrant = False)
        (h.float().square().mean() + g.float().mean()).backward()
        grads = [x[0].grad.clone(), x[1].grad.clone()] + [p.grad.clone() for l in layers for p in l.parameters()]
        for l in layers: l.zero_grad(set_to_none = True)
        return grads
    ref = run(None)
    offloader = gc_module.NonReentrantActivationOffloader(stream_mode = True)
    for a, b in zip(run(offloader.hooks), ref):
        assert torch.equal(a, b)
    assert offloader.stats["prefetched"] > 0
    assert offloader.stats["prefetch_hits"] == offloader.stats["prefetched"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
@pytest.mark.parametrize("keep_last", ["0", "1"])
def test_stream_prefetch_reloads_every_input_of_the_previous_checkpoint(gc_module, monkeypatch, keep_last):
    # Each unpack queues the newest input still on the host, so both inputs of checkpoint i-1
    # are reloaded while checkpoint i recomputes; only the last checkpoint's inputs load cold.
    monkeypatch.setenv("UNSLOTH_GC_STREAM_KEEP_LAST", keep_last)
    torch.manual_seed(0)
    layers = [_TwoInputLayer().cuda() for _ in range(5)]
    offloader = gc_module.NonReentrantActivationOffloader(stream_mode = True)
    h = torch.randn(4, 64, 256, device = "cuda", requires_grad = True)
    g = torch.randn(4, 64, 256, device = "cuda", requires_grad = True)
    for layer in layers:
        with offloader.hooks():
            h, g = ckpt.checkpoint(layer, h, g, use_reentrant = False)
    (h.float().square().mean() + g.float().mean()).backward()
    torch.cuda.synchronize()
    assert offloader.stats["prefetched"] == 2 * (len(layers) - 1)
    assert offloader.stats["prefetch_hits"] == offloader.stats["prefetched"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
def test_low_ram_growth_fallback_frees_the_old_pinned_block(gc_module, monkeypatch):
    # One local rank, host RAM short: the grown slot goes pageable and its old pinned block must
    # be returned to the OS, not left cached in torch's host allocator beside the pageable copy.
    monkeypatch.setattr(gc_module, "PINNED_MEMORY_AVAILABLE", True, raising = False)
    monkeypatch.delenv("UNSLOTH_DISABLE_PINNED_MEMORY", raising = False)
    monkeypatch.setattr(gc_module, "GC_PINNED_CAP", gc_module._PIN_PER_BLOCK, raising = False)
    torch.cuda.init()
    torch._C._host_emptyCache()
    fits = {"ok": True}
    monkeypatch.setattr(gc_module, "_block_fits_host_ram", lambda block: fits["ok"])
    held = lambda: int(torch.cuda.host_memory_stats()["allocated_bytes.current"])
    before = held()
    slot = gc_module._new_host_buffer(1 << 20, torch.uint8)       # 1 MiB pinned block
    assert slot.is_pinned() and held() - before == 1 << 20
    slot.copy_(torch.ones(1 << 20, dtype = torch.uint8))
    fits["ok"] = False
    grown = gc_module._grow_host_buffer(slot, 4 << 20)
    del slot                                                      # the caller replaces the slot
    assert not grown.is_pinned() and grown.numel() == 4 << 20
    assert held() - before == 0
    assert gc_module.GC_PINNED_BYTES == 0 and not any(gc_module._PINNED_CACHED.values())


def _forward_peak_active(hooks, n_layers = 12):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_stats()["active_bytes.all.current"]
    torch.cuda.reset_peak_memory_stats()
    h = torch.randn(1, 4096, 2048, device = "cuda", dtype = torch.float16, requires_grad = True)  # 16 MiB
    for _ in range(n_layers):
        with hooks():
            h = ckpt.checkpoint(lambda a: torch.tanh(a * 0.9), h, use_reentrant = False)
    torch.cuda.synchronize()
    peak = torch.cuda.memory_stats()["active_bytes.all.peak"] - base
    h.float().mean().backward()
    return peak


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
def test_single_gpu_default_copies_on_the_compute_stream(gc_module, monkeypatch):
    # Copying on the compute stream, like the saved_tensors_hooks Unsloth used before, frees
    # each block in stream order, so offloading adds no VRAM over that baseline.
    monkeypatch.delenv("UNSLOTH_GC_MAX_INFLIGHT_COPIES", raising = False)
    monkeypatch.setattr(gc_module, "is_distributed", lambda: False)
    offloader = gc_module.NonReentrantActivationOffloader(stream_mode = False)
    assert offloader.max_inflight == 0

    def pack(x):
        if x.requires_grad and x.device.type != "cpu": return x.device, x.to("cpu", non_blocking = True)
        return None, x

    def unpack(packed):
        device, x = packed
        return x if device is None else x.to(device, non_blocking = True)

    baseline = _forward_peak_active(lambda: torch.autograd.graph.saved_tensors_hooks(pack, unpack))
    assert _forward_peak_active(offloader.hooks) <= baseline
    assert offloader.stats["pinned"] > 0 and not offloader.d2h_streams


def test_distributed_default_bounds_inflight_copies(gc_module, monkeypatch):
    monkeypatch.delenv("UNSLOTH_GC_MAX_INFLIGHT_COPIES", raising = False)
    monkeypatch.setattr(gc_module, "is_distributed", lambda: True)
    assert gc_module.NonReentrantActivationOffloader().max_inflight == 1
    monkeypatch.setenv("UNSLOTH_GC_MAX_INFLIGHT_COPIES", "3")
    assert gc_module.NonReentrantActivationOffloader().max_inflight == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
def test_capped_pinning_survives_blocks_freed_after_async_copies(gc_module, monkeypatch):
    # unpack's non_blocking H2D leaves each freed host block in torch's event queue, and
    # host_memory_stats() does not drain it, so the cached blocks read as not free. Once the
    # cap was reached, every later layer used to go pageable for the rest of training.
    if not hasattr(torch.cuda, "host_memory_stats"):
        pytest.skip("torch < 2.7 has no host_memory_stats")
    torch.cuda.synchronize()
    torch._C._host_emptyCache()
    base = gc_module._host_allocator_held_bytes()  # live pinned memory of earlier tests
    block = 1 << 20
    monkeypatch.setenv("UNSLOTH_GC_PINNED_MAX_GB", str(3 * block / (1 << 30)))
    monkeypatch.setattr(gc_module, "_WARNED_ABOUT_PINNED_CAP", True, raising = False)
    monkeypatch.setattr(gc_module, "_HOST_HELD_BASELINE", None, raising = False)
    layers = [torch.nn.Linear(1024, 1024).cuda() for _ in range(6)]
    offloader = gc_module.NonReentrantActivationOffloader(stream_mode = False)
    pinned = []
    for _ in range(4):
        before = offloader.stats["pinned"]
        h = torch.randn(256, 1024, device = "cuda", requires_grad = True)  # 1 MiB per layer input
        for layer in layers:
            with offloader.hooks():
                h = ckpt.checkpoint(layer, h, use_reentrant = False)
        h.sum().backward()
        torch.cuda.synchronize()
        pinned.append(offloader.stats["pinned"] - before)
    # 3 blocks fit the cap: every step reuses them, none pins past it.
    assert pinned == [3, 3, 3, 3]
    assert gc_module._host_allocator_held_bytes() - base <= 3 * block



@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
def test_a_probe_that_pins_a_new_block_releases_it(gc_module):
    # Another pinned user holds the cached block, so the probe pins a fresh one: it must not stay
    # cached past the cap.
    if not hasattr(torch.cuda, "host_memory_stats"):
        pytest.skip("torch < 2.7 has no host_memory_stats")
    torch.cuda.synchronize()
    torch._C._host_emptyCache()
    block = 1 << 20
    other = torch.empty(block, dtype = torch.uint8, pin_memory = True)  # holds the only cached block
    held = gc_module._host_allocator_held_bytes()
    assert gc_module._host_cache_reuses_block(block) is False
    assert gc_module._host_allocator_held_bytes() <= held
    del other


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
def test_a_probe_failure_other_than_oom_is_raised(gc_module, monkeypatch):
    if not hasattr(torch.cuda, "host_memory_stats"):
        pytest.skip("torch < 2.7 has no host_memory_stats")
    real_empty = torch.empty

    def broken(*args, **kwargs):
        if kwargs.get("pin_memory"):
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", broken)
    with pytest.raises(RuntimeError, match = "illegal memory access"):
        gc_module._host_cache_reuses_block(1 << 20)

    def oom(*args, **kwargs):
        raise RuntimeError("CUDA error: out of memory")

    monkeypatch.setattr(torch, "empty", oom)
    assert gc_module._host_cache_reuses_block(1 << 20) is False
