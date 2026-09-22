# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""Offloaded gradient checkpointing on a model split over several cards.

The side stream that stages each card's activations was created with a bare
``torch.cuda.Stream()``, which belongs to the current device, so every card past the
first issued its offload copies against a cuda:0 stream. These pin each card's stream
to that card and check the offloaded result still matches plain checkpointing.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason = "Needs two accelerators",
)

DTYPE = torch.bfloat16
SHAPE = (4, 512, 1024)   # over MINIMUM_SIZE, so the offload branch engages


@pytest.fixture
def offload_module():
    from unsloth_zoo import gradient_checkpointing as gc_module
    names = (
        "CPU_BUFFERS", "GPU_BUFFERS", "GPU_BUFFERS_B", "USE_DOUBLE_BUFFER",
        "BUFFER_EVENTS_A", "BUFFER_EVENTS_B", "NEXT_BUFFER_SLOT", "MINIMUM_SIZE",
        "CPU_INDEX", "FIRST_PASS", "LAST_GC_INDEX", "CURRENT_GC_INDEX",
        "BACKWARD_PASS", "USE_UNSLOTH_GC", "EXTRA_STREAMS", "MAIN_STREAMS",
    )
    saved = {name : getattr(gc_module, name, None) for name in names}
    previous_device = torch.cuda.current_device()
    torch.cuda.set_device(0)
    gc_module.initialize_unsloth_gradient_checkpointing(DTYPE)
    try:
        yield gc_module
    finally:
        torch.cuda.set_device(previous_device)
        for name, value in saved.items(): setattr(gc_module, name, value)


def test_every_side_stream_lives_on_its_own_card(offload_module):
    n = torch.cuda.device_count()
    assert len(offload_module.EXTRA_STREAMS) == n
    for i in range(n):
        assert offload_module.EXTRA_STREAMS[i].device == torch.device(f"cuda:{i}"), i
        assert offload_module.MAIN_STREAMS[i].device == torch.device(f"cuda:{i}"), i


def _block(hidden, side):
    return torch.tanh(hidden * 2.0 + side) * side


def test_offload_on_a_card_that_is_not_current(offload_module):
    """Layers on cuda:1 while cuda:0 is current, as accelerate runs a split model."""
    pristine = getattr(
        torch.utils.checkpoint, "_unsloth_pristine_checkpoint", torch.utils.checkpoint.checkpoint,
    )
    device = torch.device("cuda:1")
    assert torch.cuda.current_device() == 0

    def run(checkpoint_fn):
        torch.manual_seed(0)
        hidden = torch.randn(SHAPE, dtype = DTYPE, device = device, requires_grad = True)
        side = torch.randn(SHAPE, dtype = DTYPE, device = device, requires_grad = True)
        out = hidden
        for _ in range(4):
            out = checkpoint_fn(_block, out, side, use_reentrant = True)
        out.float().sum().backward()
        torch.cuda.synchronize(device)
        return out.detach(), hidden.grad, side.grad

    expected = run(pristine)
    before = offload_module.CPU_INDEX
    actual = run(offload_module.unsloth_offloaded_gradient_checkpoint)
    assert offload_module.CPU_INDEX > before, "activation was not offloaded"
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want)
