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

"""Running out of PINNED HOST memory must degrade, not kill the run.

Pinned memory is a third budget, separate from VRAM and system RAM; exhausting it
surfaces as a bare "CUDA error: out of memory" while nvidia-smi shows the card
nearly empty. WSL2 caps it near 1-2GB and refuses allocations past a few MB
(unslothai/unsloth 338, 1552, 1744, 1797); resize_ on an already-pinned buffer is
the load-bearing call, not the 128KB startup allocation.
"""

import pytest
import torch

from unsloth_zoo import gradient_checkpointing as gc


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason = "pinned host memory requires a CUDA context",
)

HOST_OOM = "CUDA error: out of memory"


@pytest.fixture(autouse = True)
def _reset_pinned_state(monkeypatch):
    monkeypatch.setattr(gc, "PINNED_MEMORY_AVAILABLE", True, raising = False)
    monkeypatch.setattr(gc, "_WARNED_ABOUT_PINNED_MEMORY", False, raising = False)
    monkeypatch.delenv("UNSLOTH_DISABLE_PINNED_MEMORY", raising = False)


@pytest.mark.parametrize("message", [
    "CUDA error: out of memory",
    "CUDA driver error: out of memory",
    "cudaErrorMemoryAllocation",
    "CUDA out of memory. Tried to allocate 2.00 GiB",
])
def test_host_oom_is_recognised(message):
    assert gc._is_host_alloc_oom(RuntimeError(message))


@pytest.mark.parametrize("message", [
    "CUDA error: an illegal memory access was encountered",
    "CUDA error: device-side assert triggered",
    "cannot resize variables that require grad",
])
def test_unrelated_errors_are_not_swallowed(message):
    """A fallback that eats every RuntimeError would hide real corruption."""
    assert not gc._is_host_alloc_oom(RuntimeError(message))


@requires_cuda
def test_new_host_buffer_is_pinned_when_pinning_works():
    buffer = gc._new_host_buffer(1024, torch.float16)
    assert buffer.is_pinned()
    assert buffer.numel() == 1024


@requires_cuda
def test_new_host_buffer_falls_back_to_pageable(monkeypatch):
    real_empty = torch.empty

    def refuse_pinned(*args, **kwargs):
        if kwargs.get("pin_memory"):
            raise RuntimeError(HOST_OOM)
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", refuse_pinned)
    buffer = gc._new_host_buffer(1024, torch.float16)
    assert not buffer.is_pinned()
    assert buffer.numel() == 1024
    assert buffer.dtype == torch.float16
    # Latched off, so the next 32 layers do not each pay for a failed alloc.
    assert gc.PINNED_MEMORY_AVAILABLE is False


@requires_cuda
def test_new_host_buffer_reraises_unrelated_errors(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(torch, "empty", broken)
    with pytest.raises(RuntimeError, match = "illegal memory access"):
        gc._new_host_buffer(1024, torch.float16)


@requires_cuda
def test_grow_host_buffer_falls_back_when_pinned_resize_refuses(monkeypatch):
    """Issues 1744 and 1797 bisected here: growing a pinned buffer re-pins, and WSL refuses."""
    buffer = torch.empty(128 * 1024, dtype = torch.float16, device = "cpu", pin_memory = True)
    real_resize = torch.Tensor.resize_

    def refuse_growth(self, *size, **kwargs):
        if self.is_pinned():
            raise RuntimeError(HOST_OOM)
        return real_resize(self, *size, **kwargs)

    monkeypatch.setattr(torch.Tensor, "resize_", refuse_growth)
    grown = gc._grow_host_buffer(buffer, 4 * 1024 * 1024)
    assert grown.numel() == 4 * 1024 * 1024
    assert not grown.is_pinned()
    assert grown.dtype == torch.float16


@requires_cuda
def test_grow_host_buffer_keeps_pinning_when_the_resize_succeeds():
    buffer = torch.empty(128 * 1024, dtype = torch.float16, device = "cpu", pin_memory = True)
    grown = gc._grow_host_buffer(buffer, 1024 * 1024)
    assert grown.numel() == 1024 * 1024
    assert grown.is_pinned()


def test_grow_host_buffer_is_a_noop_when_already_large_enough():
    buffer = torch.empty(4096, dtype = torch.float16, device = "cpu")
    assert gc._grow_host_buffer(buffer, 1024) is buffer
    assert buffer.numel() == 4096


@requires_cuda
def test_env_var_skips_pinning_entirely(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DISABLE_PINNED_MEMORY", "1")

    def explode(*args, **kwargs):
        if kwargs.get("pin_memory"):
            pytest.fail("pinning was attempted despite UNSLOTH_DISABLE_PINNED_MEMORY=1")
        return torch.zeros(*args, **{k: v for k, v in kwargs.items() if k != "pin_memory"})

    monkeypatch.setattr(torch, "empty", explode)
    buffer = gc._new_host_buffer(1024, torch.float16)
    assert not buffer.is_pinned()


@requires_cuda
def test_initialisation_survives_a_platform_that_refuses_all_pinning(monkeypatch):
    """Startup buffers were allocated outside any try, so a host that refuses pinning
    outright never reached the first training step."""
    real_empty = torch.empty

    def refuse_pinned(*args, **kwargs):
        if kwargs.get("pin_memory"):
            raise RuntimeError(HOST_OOM)
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", refuse_pinned)
    gc.initialize_unsloth_gradient_checkpointing(dtype = torch.float16)
    assert len(gc.CPU_BUFFERS) == gc.INITIAL_CPU_BUFFER_COUNT
    assert all(not b.is_pinned() for b in gc.CPU_BUFFERS)


@requires_cuda
def test_checkpointed_block_survives_pinned_exhaustion_and_keeps_gradients():
    """The step completes with the gradients it would have had, pinning fully refused."""
    import unittest.mock as mock

    device = "cuda"
    dtype = torch.float16
    torch.manual_seed(3407)
    block = torch.nn.Linear(1024, 1024, device = device, dtype = dtype)

    def run(use_checkpoint, refuse_pinning):
        torch.manual_seed(3407)
        # 2048 x 1024 float16 = 4MB, comfortably over the 2MB offload threshold.
        x = torch.randn(2048, 1024, device = device, dtype = dtype, requires_grad = True)
        ctx = []
        if refuse_pinning:
            real_empty, real_resize = torch.empty, torch.Tensor.resize_

            def refuse_empty(*a, **k):
                if k.get("pin_memory"): raise RuntimeError(HOST_OOM)
                return real_empty(*a, **k)

            def refuse_resize(self, *size, **k):
                if self.is_pinned(): raise RuntimeError(HOST_OOM)
                return real_resize(self, *size, **k)

            ctx = [mock.patch.object(torch, "empty", refuse_empty),
                   mock.patch.object(torch.Tensor, "resize_", refuse_resize)]
        for c in ctx: c.start()
        try:
            gc.initialize_unsloth_gradient_checkpointing(dtype = dtype)
            gc.FIRST_PASS = False
            gc.LAST_GC_INDEX = 99          # never the "skip the last layer" case
            gc.CURRENT_GC_INDEX = 0
            gc.CPU_INDEX = 0
            gc.BACKWARD_PASS = True
            out = gc.unsloth_checkpoint(block, x) if use_checkpoint else block(x)
            out.float().pow(2).mean().backward()
        finally:
            for c in ctx: c.stop()
        return x.grad.detach().float().clone()

    reference = run(use_checkpoint = False, refuse_pinning = False)
    starved = run(use_checkpoint = True, refuse_pinning = True)

    assert torch.isfinite(starved).all()
    torch.testing.assert_close(starved, reference, rtol = 2e-3, atol = 2e-3)


@requires_cuda
def test_checkpointed_block_survives_a_per_allocation_pinned_cap():
    """WSL's shape: small pinned allocations succeed, large ones do not, so only the
    growth path saves the run."""
    import unittest.mock as mock

    CAP = 1 << 20  # 1MB, between the 128KB startup buffers and the 4MB activation
    dtype = torch.float16
    torch.manual_seed(3407)
    block = torch.nn.Linear(1024, 1024, device = "cuda", dtype = dtype)
    real_empty, real_resize = torch.empty, torch.Tensor.resize_

    def capped_empty(*a, **k):
        if k.get("pin_memory"):
            numel = a[0] if a and isinstance(a[0], int) else 0
            if numel * torch.finfo(k.get("dtype", dtype)).bits // 8 > CAP:
                raise RuntimeError(HOST_OOM)
        return real_empty(*a, **k)

    def capped_resize(self, *size, **k):
        if self.is_pinned():
            numel = size[0] if size and isinstance(size[0], int) else 0
            if numel * self.element_size() > CAP:
                raise RuntimeError(HOST_OOM)
        return real_resize(self, *size, **k)

    torch.manual_seed(3407)
    x = torch.randn(2048, 1024, device = "cuda", dtype = dtype, requires_grad = True)
    with mock.patch.object(torch, "empty", capped_empty), \
         mock.patch.object(torch.Tensor, "resize_", capped_resize):
        gc.initialize_unsloth_gradient_checkpointing(dtype = dtype)
        assert all(b.is_pinned() for b in gc.CPU_BUFFERS), \
            "startup buffers should still pin under a per-allocation cap"
        gc.FIRST_PASS = False
        gc.LAST_GC_INDEX = 99
        gc.CURRENT_GC_INDEX = 0
        gc.CPU_INDEX = 0
        gc.BACKWARD_PASS = True
        gc.unsloth_checkpoint(block, x).float().pow(2).mean().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()


@requires_cuda
def test_pageable_buffers_are_not_copied_with_a_non_blocking_claim(monkeypatch):
    """non_blocking must follow the buffer: async into pageable memory is really sync."""
    seen = {}
    real_copy = torch.Tensor.copy_

    def record(self, other, *args, **kwargs):
        if self.device.type == "cpu":
            seen["non_blocking"] = kwargs.get("non_blocking", args[0] if args else False)
            seen["pinned"] = self.is_pinned()
        return real_copy(self, other, *args, **kwargs)

    real_empty, real_resize = torch.empty, torch.Tensor.resize_

    def refuse_empty(*a, **k):
        if k.get("pin_memory"): raise RuntimeError(HOST_OOM)
        return real_empty(*a, **k)

    def refuse_resize(self, *size, **k):
        if self.is_pinned(): raise RuntimeError(HOST_OOM)
        return real_resize(self, *size, **k)

    monkeypatch.setattr(torch, "empty", refuse_empty)
    monkeypatch.setattr(torch.Tensor, "resize_", refuse_resize)
    gc.initialize_unsloth_gradient_checkpointing(dtype = torch.float16)
    monkeypatch.setattr(torch.Tensor, "copy_", record)

    gc.FIRST_PASS = False
    gc.LAST_GC_INDEX = 99
    gc.CURRENT_GC_INDEX = 0
    gc.CPU_INDEX = 0
    gc.BACKWARD_PASS = True
    block = torch.nn.Linear(1024, 1024, device = "cuda", dtype = torch.float16)
    x = torch.randn(2048, 1024, device = "cuda", dtype = torch.float16, requires_grad = True)
    gc.unsloth_checkpoint(block, x).float().sum().backward()

    assert seen, "the offload branch never ran, so this test proved nothing"
    assert seen["pinned"] is False
    assert seen["non_blocking"] is False
