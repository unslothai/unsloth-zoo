# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""Smart GC host slots for a graph kept with retain_graph=True.

A backward returned its host slot right away, so in F0 B0(retain_graph) F1 B1 B0 the forward
of step 1 overwrote the activations that the second backward of step 0 reads back, and that
backward silently produced wrong gradients. The slot of a retained graph is now set aside until
the graph is freed. The device side is replaced with no-op streams so the host bookkeeping runs
on CPU unchanged; grads must match a plain stack bit for bit.
"""
import contextlib
import gc

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
    G._reset_cpu_slots()
    G._double_buffer_disabled.cache_clear()


H, S, L = 32, 16, 5


def _setup():
    g = torch.Generator().manual_seed(0)
    ws = [(torch.randn(H, H, generator = g) / H ** 0.5).requires_grad_() for _ in range(L)]
    xs = [torch.randn(1, S, H, generator = g) for _ in range(3)]
    return ws, xs


def _fwd(G, ws, x, ckpt):
    h = x * 1.0
    for w in ws:
        f = (lambda w: lambda h: h + torch.tanh(h @ w))(w)
        h = G.UnslothCheckpointFunction.apply(f, True, h) if ckpt else f(h)
    return h.square().mean()


def _run(G, schedule, ckpt):
    """schedule: (op, k) with op F (forward), R (backward, retain_graph) or B (backward)."""
    ws, xs = _setup()
    losses, grads = {}, []
    for op, k in schedule:
        if op == "F":
            losses[k] = _fwd(G, ws, xs[k].requires_grad_(), ckpt)
        else:
            losses[k].backward(retain_graph = op == "R")
            grads.append([w.grad.clone() for w in ws])
            for w in ws: w.grad = None
            if op == "B": del losses[k]
    return grads


def _parse(text):
    return [(op[0], int(op[1:])) for op in text.split()]


@pytest.mark.parametrize("schedule", [
    "F0 R0 F1 B1 B0",          # second backward of a retained graph after another step
    "F0 R0 B0 F1 B1",          # retain then backward again right away
    "F0 R0 F1 R1 F2 B2 B1 B0",  # several retained graphs in flight
])
def test_retained_graph_backward_matches_plain_stack(G, schedule):
    _run(G, _parse("F0 B0"), True)  # settle FIRST_PASS / LAST_GC_INDEX like a real run
    ref = _run(G, _parse(schedule), False)
    got = _run(G, _parse(schedule), True)
    assert len(ref) == len(got)
    for step, (a, b) in enumerate(zip(ref, got)):
        for x, y in zip(a, b):
            assert torch.equal(x, y), (schedule, step)


def test_retained_slots_return_once_the_graph_is_freed(G):
    _run(G, _parse("F0 B0"), True)
    ws, xs = _setup()
    loss = _fwd(G, ws, xs[0].requires_grad_(), True)
    loss.backward(retain_graph = True)
    held = G._CPU_SLOT_TOP
    loss2 = _fwd(G, ws, xs[1].requires_grad_(), True)  # must not reuse the retained slots
    assert G._CPU_SLOT_TOP > held
    loss2.backward()
    del loss, loss2
    gc.collect()
    top = G._CPU_SLOT_TOP
    for k in range(3):  # retained graph gone: plain steps reuse the pool, it does not grow
        loss = _fwd(G, ws, xs[k].requires_grad_(), True)
        loss.backward()
    assert G._CPU_SLOT_TOP == top
    assert not G._RETAINED_SLOTS


def test_plain_loop_reuses_the_same_slots(G):
    _run(G, _parse("F0 B0"), True)
    top = G._CPU_SLOT_TOP
    _run(G, _parse("F0 B0 F1 B1 F2 B2"), True)
    assert G._CPU_SLOT_TOP == top and not G._RETAINED_SLOTS
