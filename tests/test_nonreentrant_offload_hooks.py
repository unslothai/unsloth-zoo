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
@pytest.mark.parametrize("mode", ["default", "stream_keep0", "stream_keep1", "stream_keep2", "pageable"])
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
