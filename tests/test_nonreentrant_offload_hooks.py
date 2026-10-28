# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""saved_tensors_hooks offload for torch's non-reentrant checkpoint.

Gradients must match plain non-reentrant checkpointing bit for bit in every mode: side
stream pinned copies, unsloth_stream (keep last N on device, reload i-1 ahead), and the
pageable fallback once the pinned cap is hit. The engagement counters prove each path ran.
"""
import functools
import weakref

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


class _ThreeInputLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(256, 256)

    def forward(self, h, g, k):
        return torch.tanh(self.a(h) + g + k)


def _max_prefetched_waiting(offloader, monkeypatch):
    # Largest number of reloads sitting on the device ahead of their unpack at any point.
    peak = [0]
    original = offloader._prefetch

    def counting(index):
        original(index)
        live = (ref() for ref in offloader.order)
        peak[0] = max(peak[0], sum(1 for r in live if r is not None and r.ready is not None))

    monkeypatch.setattr(offloader, "_prefetch", counting)
    return peak


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
def test_stream_prefetch_never_reaches_past_the_previous_checkpoint(gc_module, monkeypatch):
    # A three input checkpoint after single input ones: its three unpacks may only reload the
    # one input of the checkpoint before it, not walk on into older checkpoints.
    monkeypatch.setenv("UNSLOTH_GC_STREAM_KEEP_LAST", "0")
    torch.manual_seed(0)
    singles = [torch.nn.Linear(256, 256).cuda() for _ in range(3)]
    last = _ThreeInputLayer().cuda()
    offloader = gc_module.NonReentrantActivationOffloader(stream_mode = True)
    peak = _max_prefetched_waiting(offloader, monkeypatch)
    h = torch.randn(4, 64, 256, device = "cuda", requires_grad = True)
    g = torch.randn(4, 64, 256, device = "cuda", requires_grad = True)
    for layer in singles:
        with offloader.hooks():
            h = ckpt.checkpoint(layer, h, use_reentrant = False)
    with offloader.hooks():
        out = ckpt.checkpoint(last, h, g, h * 2, use_reentrant = False)
    out.float().square().mean().backward()
    torch.cuda.synchronize()
    assert peak[0] == 1
    assert offloader.stats["prefetch_hits"] == offloader.stats["prefetched"] == len(singles) - 1 + 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
def test_stream_prefetch_is_one_step_when_hooks_span_every_checkpoint(gc_module, monkeypatch):
    # One hooks() entry around all checkpoints gives no checkpoint boundaries: keep to index - 1,
    # so at most the next reload is queued while one still waits, however deep the stack is.
    monkeypatch.setenv("UNSLOTH_GC_STREAM_KEEP_LAST", "0")
    torch.manual_seed(0)
    layers = [_TwoInputLayer().cuda() for _ in range(8)]
    offloader = gc_module.NonReentrantActivationOffloader(stream_mode = True)
    peak = _max_prefetched_waiting(offloader, monkeypatch)
    h = torch.randn(4, 64, 256, device = "cuda", requires_grad = True)
    g = torch.randn(4, 64, 256, device = "cuda", requires_grad = True)
    with offloader.hooks():
        for layer in layers:
            h, g = ckpt.checkpoint(layer, h, g, use_reentrant = False)
    (h.float().square().mean() + g.float().mean()).backward()
    torch.cuda.synchronize()
    assert peak[0] <= 2


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


# Caching allocator model on CPU: a block freed on its allocation stream is reusable there at
# once, so a use on another stream is safe only if, at free time, the allocation stream waited on
# it or record_stream() named a stream that did (record_stream ignores the allocation stream).
class _Stream:
    def __init__(self, name): self.name, self.waited, self.ops = name, [], []
    def wait_event(self, event): self.waited.append(event)
    def wait_stream(self, other): pass
    def record_event(self): return _Event()
    def __repr__(self): return self.name


class _Event:
    def record(self, stream = None): pass
    def query(self): return False


class _Source:
    """A device tensor: `uses` is what record_stream() registered, `safe` is decided at free."""
    def __init__(self, stream, guards):
        self.alloc_stream, self.uses, self.guards, self.freed = stream, [], guards, []
        self.device = type("D", (), {"index": 0, "type": "cuda"})()
        weakref.finalize(self, _Source._free, self.alloc_stream, self.uses, self.guards, self.freed)

    @staticmethod
    def _free(alloc_stream, uses, guards, freed):
        # Every guard (a copy's event, or work queued on a consumer stream) must be ordered before
        # reuse on the allocation stream.
        for guard in guards:
            if isinstance(guard, _Stream):
                freed.append(guard is alloc_stream or guard in uses)
            else:
                freed.append(guard in alloc_stream.waited or any(guard in s.waited for s in uses))

    def record_stream(self, stream):
        if stream is not self.alloc_stream and stream not in self.uses: self.uses.append(stream)
    def numel(self): return 4
    def element_size(self): return 4
    def to(self, device, non_blocking = False): return "host copy"


@pytest.fixture
def fake_cuda(gc_module, monkeypatch):
    import contextlib
    state = {"current": _Stream("compute"), "side": _Stream("side")}
    monkeypatch.setattr(gc_module, "PINNED_MEMORY_AVAILABLE", True)
    monkeypatch.setattr(gc_module, "_pinned_memory_disabled", lambda: False)
    monkeypatch.setattr(gc_module, "_reserve_pinned", lambda n: True)
    monkeypatch.setattr(gc_module, "_release_pinned", lambda n: None)
    monkeypatch.setattr(gc_module, "_new_extra_stream", lambda i: state["side"])
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device = None: state["current"])
    monkeypatch.setattr(torch.cuda, "stream", lambda s: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    monkeypatch.setenv("UNSLOTH_GC_MAX_INFLIGHT_COPIES", "1")
    return state


def _pack_on_side_stream(gc_module, alloc_stream):
    offloader = gc_module.NonReentrantActivationOffloader(stream_mode = False)
    record = gc_module._OffloadedActivation(None, 0)
    guards = []
    x = _Source(alloc_stream, guards)
    offloader._to_host(record, x)
    guards.append(record.copied)
    return offloader, record, x.freed, x.uses, weakref.ref(x)


@pytest.mark.parametrize("same_stream", [True, False])
def test_d2h_source_is_reused_only_after_its_copy(gc_module, fake_cuda, same_stream):
    alloc = fake_cuda["current"] if same_stream else _Stream("producer")
    offloader, record, freed, uses, alive = _pack_on_side_stream(gc_module, alloc)
    offloader._drain()  # backward starts: the keepalive is retired
    assert alive() is None and freed == [True]
    # The usual case costs no delayed reuse: nothing is recorded on the side stream.
    assert fake_cuda["side"] not in uses
    assert uses == ([] if same_stream else [fake_cuda["current"]])


@pytest.mark.parametrize("same_stream", [True, False])
def test_a_dropped_graph_releases_its_d2h_source_after_the_copy(gc_module, fake_cuda, same_stream):
    alloc = fake_cuda["current"] if same_stream else _Stream("producer")
    offloader, record, freed, uses, alive = _pack_on_side_stream(gc_module, alloc)
    del record  # graph discarded without backward, offloader idle
    assert alive() is None and freed == [True]
    assert not offloader.inflight[0] or offloader.inflight[0][0][0] is None


def test_a_record_outliving_its_offloader_keeps_the_source_until_the_copy(gc_module, fake_cuda):
    offloader, record, freed, uses, alive = _pack_on_side_stream(gc_module, fake_cuda["current"])
    del offloader
    assert alive() is not None or freed == [True]
    del record
    assert alive() is None and freed == [True]


@pytest.mark.parametrize("same_stream", [True, False])
def test_a_consumed_prefetch_outlives_its_reload_and_its_consumer(gc_module, fake_cuda, same_stream):
    alloc = fake_cuda["current"] if same_stream else _Stream("prefetching")
    offloader = gc_module.NonReentrantActivationOffloader(stream_mode = False)
    record = gc_module._OffloadedActivation(type("D", (), {"index": 0})(), 0)
    guards = []
    x = _Source(alloc, guards)
    freed = x.freed
    ready = _Event()
    record.device_tensor, record.ready, record.ready_stream, record.host = x, ready, alloc, "host copy"
    guards.append(ready)
    del x
    out = offloader.unpack(record)
    guards.append(fake_cuda["current"])  # the consumer then reads it on the unpacking stream
    del record
    alive = weakref.ref(out)
    del out
    assert alive() is None and freed == [True, True]
