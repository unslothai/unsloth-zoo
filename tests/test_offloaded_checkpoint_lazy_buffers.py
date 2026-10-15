# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""Offload staging buffers and side streams are created per device on first use.

Allocating them on every visible GPU at init made each torchrun rank open a CUDA context
(and buffers) on every card. CPU only: device_count, torch.empty and the stream / event
constructors are spied, so a 4 GPU host is simulated and every touched device recorded.
"""
import pytest
import torch


@pytest.fixture
def spied(monkeypatch):
    monkeypatch.setenv("UNSLOTH_ALLOW_CPU", "1")
    try:
        from unsloth_zoo import gradient_checkpointing as gc_module
    except ImportError as e:
        pytest.skip(f"unsloth_zoo import unavailable: {e}")

    names = (
        "CPU_BUFFERS", "GPU_BUFFERS", "GPU_BUFFERS_B", "USE_DOUBLE_BUFFER",
        "BUFFER_EVENTS_A", "BUFFER_EVENTS_B", "NEXT_BUFFER_SLOT", "MINIMUM_SIZE",
        "CPU_INDEX", "FIRST_PASS", "LAST_GC_INDEX", "CURRENT_GC_INDEX",
        "BACKWARD_PASS", "USE_UNSLOTH_GC", "EXTRA_STREAMS", "MAIN_STREAMS",
        "GPU_BUFFER_DTYPE",
    )
    saved = {name : getattr(gc_module, name, None) for name in names}

    touched = []
    real_empty = torch.empty

    def empty(*args, device = None, **kwargs):
        if device is not None and str(device).startswith("cuda"):
            touched.append(("buffer", torch.device(device).index))
            device = "cpu"
        return real_empty(*args, device = device, **kwargs)

    class FakeStream:
        def __init__(self, device = None, **kwargs):
            self.device = torch.device(device)
            touched.append(("stream", self.device.index))

    class FakeEvent:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(gc_module, "DEVICE_TYPE", "cuda", raising = False)
    monkeypatch.setattr(gc_module, "DEVICE_TYPE_TORCH", "cuda", raising = False)
    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "Stream", FakeStream)
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "default_stream", lambda device = None: FakeStream(device))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setenv("UNSLOTH_DISABLE_PINNED_MEMORY", "1")  # pinning needs a real device
    monkeypatch.setattr(gc_module, "_double_buffer_disabled", lambda: False, raising = False)
    try:
        gc_module.initialize_unsloth_gradient_checkpointing(torch.bfloat16)
        yield gc_module, touched
    finally:
        for name, value in saved.items(): setattr(gc_module, name, value)


def test_init_touches_no_gpu(spied):
    _, touched = spied
    assert touched == [], touched


def test_only_the_used_device_is_allocated(spied):
    gc_module, touched = spied
    # A torchrun rank 2 offloading its first activation.
    gc_module._ensure_device_buffers(2)
    assert {index for _, index in touched} == {2}, touched
    assert sorted(kind for kind, _ in touched) == ["buffer", "buffer", "stream"], touched
    assert gc_module.GPU_BUFFERS[2].dtype == torch.bfloat16
    assert gc_module.EXTRA_STREAMS[2].device == torch.device("cuda:2")
    # Second use of the same device creates nothing new.
    gc_module._ensure_device_buffers(2)
    assert len(touched) == 3, touched
    assert len(gc_module.GPU_BUFFERS) == 4 and len(gc_module.NEXT_BUFFER_SLOT) == 4


def test_model_parallel_gets_each_used_card(spied):
    gc_module, touched = spied
    # device_map splitting layers over cards 0 and 3 in one process.
    gc_module._ensure_device_buffers(0)
    gc_module._ensure_device_buffers(3)
    assert {index for _, index in touched} == {0, 3}, touched
    assert gc_module.EXTRA_STREAMS[3].device == torch.device("cuda:3")


def test_reset_and_unpatch_do_not_allocate_untouched_devices(spied):
    gc_module, touched = spied
    gc_module._ensure_device_buffers(1)
    before = list(touched)
    gc_module.reset_unsloth_gradient_checkpointing_buffers()
    assert touched == before, touched
    gc_module._free_gpu_buffers_b()
    assert gc_module.GPU_BUFFERS_B is None and gc_module.USE_DOUBLE_BUFFER is False
    assert touched == before, touched
