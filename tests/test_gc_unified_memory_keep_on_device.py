# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""Unified-memory GPUs (AMD APUs gfx1150/1151, NVIDIA GB10): "unsloth" gradient checkpointing keeps the checkpointed
inputs on the device instead of copying them into host RAM, which is the same DRAM there. Detection runs on stub
device properties; the checkpoint paths run on CPU with streams stubbed (as test_gc_retained_graph_slots.py does)."""
import contextlib

import pytest
import torch

from unsloth_zoo import integrated_device as D


class _Props:
    def __init__(self, name = "stub", arch = None, **fields):
        self.name = name
        if arch is not None:
            self.gcnArchName = arch
        for k, v in fields.items():
            setattr(self, k, v)


def _install(monkeypatch, props_list, hip = None):
    monkeypatch.delenv(D.KEEP_ON_DEVICE_ENV, raising = False)
    monkeypatch.setattr(torch.version, "hip", hip, raising = False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(props_list))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: props_list[i])


@pytest.mark.parametrize("props, hip, expect", [
    # Strix Halo with the flag left unset (ROCm < 6.2 left it unassigned): the arch name decides.
    (_Props("Radeon 8060S Graphics", "gfx1151:xnack-", is_integrated = 0), "6.1.40091", True),
    (_Props("Radeon 890M", "gfx1150", is_integrated = 1), "7.2.0", True),
    # A discrete card with a garbage flag on old ROCm is not trusted.
    (_Props("Radeon RX 7900 XTX", "gfx1100", is_integrated = 1), "6.1.40091", False),
    (_Props("Radeon RX 7900 XTX", "gfx1100", is_integrated = 0), "7.2.0", False),
    # MI300A-style: discrete arch name, flag set on a ROCm that fills it in.
    (_Props("AMD Instinct MI300A", "gfx942:sramecc+:xnack-", is_integrated = 1), "7.0.1", True),
    (_Props("AMD Instinct MI300X", "gfx942:sramecc+:xnack-", is_integrated = 0), "7.0.1", False),
    (_Props("NVIDIA GB10", is_integrated = 1), None, True),
    (_Props("NVIDIA GB10"), None, True),  # wheel without the field
    (_Props("Orin", integrated = 1), None, True),
    (_Props("NVIDIA H100 80GB HBM3", is_integrated = 0), None, False),
    (_Props("Tesla T4"), None, False),
])
def test_device_shares_host_memory(monkeypatch, props, hip, expect):
    _install(monkeypatch, [props], hip)
    assert D.device_shares_host_memory(0) is expect
    assert D.unified_memory_device_indices() == (frozenset({0}) if expect else frozenset())


def test_mixed_box_only_selects_the_integrated_device(monkeypatch):
    # A Ryzen desktop with its iGPU visible next to a Radeon card: the discrete card keeps offloading.
    _install(monkeypatch, [_Props("Radeon RX 7900 XTX", "gfx1100", is_integrated = 0),
                           _Props("AMD Radeon Graphics", "gfx1036", is_integrated = 1)], "7.2.0")
    assert D.unified_memory_device_indices() == frozenset({1})


def test_env_override(monkeypatch):
    _install(monkeypatch, [_Props("Tesla T4"), _Props("NVIDIA GB10", is_integrated = 1)])
    monkeypatch.setenv(D.KEEP_ON_DEVICE_ENV, "1")
    assert D.unified_memory_device_indices() == frozenset({0, 1})
    monkeypatch.setenv(D.KEEP_ON_DEVICE_ENV, "0")
    assert D.unified_memory_device_indices() == frozenset()


def test_properties_error_is_not_unified(monkeypatch):
    def boom(i): raise RuntimeError("no device")
    monkeypatch.setattr(torch.cuda, "get_device_properties", boom)
    assert D.device_shares_host_memory(0) is False


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
    getattr(G._keep_on_device_indices, "cache_clear", lambda: None)()
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
    leases = []
    real = G._CpuSlotLease

    class _Counting(real):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            leases.append(1)
    monkeypatch.setattr(G, "_CpuSlotLease", _Counting)
    G._test_leases = leases
    yield G
    for n, v in saved.items(): setattr(G, n, v)
    G._reset_cpu_slots()
    G._double_buffer_disabled.cache_clear()
    getattr(G._keep_on_device_indices, "cache_clear", lambda: None)()


H, S, L = 32, 16, 5


def _grads(G, ckpt):
    g = torch.Generator().manual_seed(0)
    ws = [(torch.randn(H, H, generator = g) / H ** 0.5).requires_grad_() for _ in range(L)]
    x = torch.randn(2, S, H, generator = g).requires_grad_()
    for step in range(2):  # FIRST_PASS, then the steady state that skips the last layer
        h = x * 1.0
        for w in ws:
            f = (lambda w: lambda h: h + torch.tanh(h @ w))(w)
            h = G.UnslothCheckpointFunction.apply(f, True, h) if ckpt else f(h)
        h.square().mean().backward()
        G.BACKWARD_PASS = True
        G.FIRST_PASS = False
    return [w.grad.clone() for w in ws] + [x.grad.clone()]


def test_discrete_device_still_offloads(G, monkeypatch):
    # CPU tensors have device.index None; an empty set means nothing keeps on device.
    monkeypatch.setattr(G, "_keep_on_device_indices", lambda: frozenset())
    ref = _grads(G, ckpt = False)
    got = _grads(G, ckpt = True)
    assert len(G._test_leases) > 0
    for a, b in zip(ref, got): torch.testing.assert_close(a, b, rtol = 0, atol = 0)


def test_unified_device_keeps_inputs_on_device(G, monkeypatch):
    monkeypatch.setattr(G, "_keep_on_device_indices", lambda: frozenset({None}))
    ref = _grads(G, ckpt = False)
    got = _grads(G, ckpt = True)
    assert G._test_leases == []
    for a, b in zip(ref, got): torch.testing.assert_close(a, b, rtol = 0, atol = 0)


def test_unified_skips_initial_pinned_pool(G, monkeypatch):
    monkeypatch.setattr(G, "_keep_on_device_indices", lambda: frozenset({0}))
    monkeypatch.setattr(G, "_visible_device_count", lambda: 1)
    G.initialize_unsloth_gradient_checkpointing(torch.float32)
    assert len(G.CPU_BUFFERS) == 0
    monkeypatch.setattr(G, "_keep_on_device_indices", lambda: frozenset())
    G.initialize_unsloth_gradient_checkpointing(torch.float32)
    assert len(G.CPU_BUFFERS) == G.INITIAL_CPU_BUFFER_COUNT


def test_non_reentrant_offloader_keeps_unified(G, monkeypatch):
    monkeypatch.setattr(G, "_keep_on_device_indices", lambda: frozenset({None}))
    off = G.NonReentrantActivationOffloader()
    monkeypatch.setattr(off, "_on_host", lambda x: False)  # pretend the CPU tensor is a device tensor
    x = torch.randn(4, 8, requires_grad = True)
    packed = off.pack(x)
    assert packed[0] is None and packed[1].data_ptr() == x.data_ptr()
    assert off.stats["kept_unified"] == 1 and off.stats["kept_unified_bytes"] == x.numel() * x.element_size()
    assert off.unpack(packed) is packed[1]
