# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""Smart GC host slots under interleaved schedules.

A forward that starts after some backward used to rewind CPU_INDEX to 0, so under a 1F1B
pipeline schedule (F0 F1 B0 F2 B1 F3 B2 B3) the forward of micro batch 3 overwrote the
offloaded activations of micro batch 2, still waiting for its backward. Slots are now leased
per checkpoint and returned by its backward. The device side is replaced with no-op streams
so the host bookkeeping runs on CPU unchanged; grads must match a plain stack bit for bit,
and the usual one forward, one backward loop must keep its slot map and buffer count.
"""
import contextlib

import pytest
import torch


class _NoStream:
    def wait_stream(self, other): pass
    def wait_event(self, event): pass
    def record_event(self): return None


@pytest.fixture
def G(monkeypatch):
    monkeypatch.setenv("UNSLOTH_ALLOW_CPU", "1")
    monkeypatch.setenv("UNSLOTH_DISABLE_PINNED_MEMORY", "1")
    monkeypatch.setenv("UNSLOTH_DISABLE_DOUBLE_BUFFER", "1")
    try:
        from unsloth_zoo import gradient_checkpointing as G
    except ImportError as e:
        pytest.skip(f"unsloth_zoo import unavailable: {e}")
    G._double_buffer_disabled.cache_clear()
    names = ("CPU_BUFFERS", "CPU_INDEX", "GPU_BUFFERS", "GPU_BUFFERS_B", "USE_DOUBLE_BUFFER", "EXTRA_STREAMS",
             "MAIN_STREAMS", "NEXT_BUFFER_SLOT", "MINIMUM_SIZE", "FP32_OFFLOAD_EXACT", "BACKWARD_PASS", "FIRST_PASS",
             "LAST_GC_INDEX", "CURRENT_GC_INDEX", "USE_UNSLOTH_GC", "BUFFER_EVENTS_A", "BUFFER_EVENTS_B",
             "GPU_BUFFER_DTYPE")
    saved = {n: getattr(G, n, None) for n in names}
    G.initialize_unsloth_gradient_checkpointing(torch.float32)
    stream = _NoStream()
    monkeypatch.setattr(G, "_ensure_device_buffers", lambda i: None)
    monkeypatch.setattr(G, "_compute_stream", lambda i: stream)
    monkeypatch.setattr(G, "torch_gpu_stream", lambda s: contextlib.nullcontext(), raising = False)
    G.GPU_BUFFERS = {None: torch.empty(G.INITIAL_GPU_BUFFER_SIZE)}
    G.GPU_BUFFERS_B = None
    G.USE_DOUBLE_BUFFER = False
    G.EXTRA_STREAMS = {None: stream}
    G.NEXT_BUFFER_SLOT = {None: 0}
    G.MINIMUM_SIZE = 0
    G.FP32_OFFLOAD_EXACT = True
    yield G
    for n, v in saved.items(): setattr(G, n, v)
    getattr(G, "_reset_cpu_slots", lambda: None)()
    G._double_buffer_disabled.cache_clear()


H, S, L = 32, 16, 5


def _setup():
    g = torch.Generator().manual_seed(0)
    ws = [(torch.randn(H, H, generator = g) / H ** 0.5).requires_grad_() for _ in range(L)]
    xs = [torch.randn(1, S, H, generator = g) for _ in range(4)]
    return ws, xs


def _fwd(G, ws, x, ckpt):
    h = x * 1.0
    for w in ws:
        f = (lambda w: lambda h: h + torch.tanh(h @ w))(w)
        h = G.UnslothCheckpointFunction.apply(f, True, h) if ckpt else f(h)
    return h.square().mean()


def _run(G, schedule, ckpt):
    ws, xs = _setup()
    losses, grads = {}, {}
    for op, k in schedule:
        if op == "F":
            losses[k] = _fwd(G, ws, xs[k].requires_grad_(), ckpt)
        else:
            losses.pop(k).backward()
            grads[k] = [w.grad.clone() for w in ws] + [xs[k].grad.clone()]
            for w in ws: w.grad = None
    return grads


SCHEDULES = {
    "sequential": "F0 B0 F1 B1 F2 B2",
    "gpipe": "F0 F1 B0 B1",
    "1f1b": "F0 F1 B0 F2 B1 F3 B2 B3",
    "three_in_flight": "F0 F1 F2 B0 F3 B1 B2 B3",
}


@pytest.mark.parametrize("name", list(SCHEDULES))
def test_interleaved_schedules_match_plain_stack(G, name):
    schedule = [(op[0], int(op[1:])) for op in SCHEDULES[name].split()]
    _run(G, [("F", 0), ("B", 0)], True)  # settle FIRST_PASS / LAST_GC_INDEX like a real run
    ref = _run(G, schedule, False)
    got = _run(G, schedule, True)
    for k in ref:
        for a, b in zip(ref[k], got[k]):
            assert torch.equal(a, b), (name, k)
    # Every slot came back once each backward ran.
    if hasattr(G, "_FREE_CPU_SLOTS"):
        assert sorted(G._FREE_CPU_SLOTS) == list(range(G._CPU_SLOT_TOP))


def test_sequential_loop_keeps_slot_map_and_buffers(G):
    seen = []
    orig = G.UnslothCheckpointFunction.forward

    def spy(ctx, *a):
        out = orig(ctx, *a)
        seen.append(ctx._saved_metadata[2])
        return out
    G.UnslothCheckpointFunction.forward = staticmethod(spy)
    try:
        n = len(G.CPU_BUFFERS)
        _run(G, [("F", 0), ("B", 0), ("F", 1), ("B", 1), ("F", 2), ("B", 2)], True)
    finally:
        G.UnslothCheckpointFunction.forward = staticmethod(orig)
    # First pass offloads every layer, later passes skip the last one (LAST_GC_INDEX): same slots each step.
    assert seen == list(range(L)) + list(range(L - 1)) + [None] + list(range(L - 1)) + [None]
    assert len(G.CPU_BUFFERS) == n


def test_dropped_graph_returns_its_slots(G):
    ws, xs = _setup()
    loss = _fwd(G, ws, xs[0].requires_grad_(), True)
    held = G._CPU_SLOT_TOP
    assert held > 0 and not G._FREE_CPU_SLOTS
    del loss  # graph freed without a backward (e.g. an exception between forward and backward)
    assert sorted(G._FREE_CPU_SLOTS) == list(range(held))
    loss = _fwd(G, ws, xs[1].requires_grad_(), True)
    assert G._CPU_SLOT_TOP == held  # reused, not grown
    loss.backward()


class _FenceStream:
    def __init__(self, name):
        self.name, self.waited = name, []
    def record_event(self):
        return (self.name, "event")
    def wait_event(self, event):
        self.waited.append(event)


def test_slot_fence_is_kept_before_a_second_device_appears(G):
    G._reset_cpu_slots()
    s0, s1 = _FenceStream("cuda:0"), _FenceStream("cuda:1")
    lease = G._CpuSlotLease(0, s0)
    lease.release()  # only one device has offloaded so far
    assert G._SLOT_FENCES[0] == (0, ("cuda:0", "event"))
    again = G._CpuSlotLease(0, s0)  # same device: stream order already holds
    assert again.index == 0 and s0.waited == [] and 0 not in G._SLOT_FENCES
    again.release()
    other = G._CpuSlotLease(1, s1)  # first lease ever on device 1 still waits on device 0's release
    assert other.index == 0 and s1.waited == [("cuda:0", "event")]
    other.release()
    G._reset_cpu_slots()
    assert not G._SLOT_FENCES


class _Copied:
    def __init__(self, done): self.done = done
    def query(self): return self.done


def test_retire_waits_on_the_stream_that_allocated_the_source(G):
    import collections
    owner, current = _FenceStream("owner"), _FenceStream("current")
    first, second = _Copied(True), _Copied(False)
    inflight = collections.deque([(None, first, owner), (None, second, None)])
    G.NonReentrantActivationOffloader._retire(inflight, current, 0)
    assert owner.waited == [first] and current.waited == [second] and not inflight
