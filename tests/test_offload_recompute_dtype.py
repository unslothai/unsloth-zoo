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

"""Offloaded activations must come back in their own dtype, not the staging buffer's
(FORCE_FLOAT32 loads init bf16 buffers but run fp16/fp32; ROCm gfx10 hit an LLVM abort)."""
import pytest

torch = pytest.importorskip("torch")

from unsloth_zoo import gradient_checkpointing as gc

_STATE = [
    "CPU_BUFFERS", "CPU_INDEX", "GPU_BUFFERS", "GPU_BUFFERS_B", "MAIN_STREAMS", "EXTRA_STREAMS",
    "BACKWARD_PASS", "LAST_GC_INDEX", "FIRST_PASS", "CURRENT_GC_INDEX", "USE_UNSLOTH_GC",
    "USE_DOUBLE_BUFFER", "MINIMUM_SIZE", "NEXT_BUFFER_SLOT", "BUFFER_EVENTS_A", "BUFFER_EVENTS_B",
]
_MISSING = object()


@pytest.fixture
def offload(request):
    saved = {name: getattr(gc, name, _MISSING) for name in _STATE}
    gc.initialize_unsloth_gradient_checkpointing(request.param)
    try:
        yield gc
    finally:
        for name, value in saved.items():
            if value is _MISSING:
                if hasattr(gc, name):
                    delattr(gc, name)
            else:
                setattr(gc, name, value)


def _round_trip(activation_dtype):
    seen = []
    handed_back = []

    def layer(hidden):
        seen.append(hidden.dtype)
        handed_back.append(hidden.detach().clone())
        return hidden * 2

    hidden = torch.randn(2, 1024, 2048, device = "cuda", dtype = activation_dtype, requires_grad = True)
    out = gc.UnslothCheckpointFunction.apply(layer, False, hidden)
    out = gc.UnslothCheckpointFunction.apply(layer, False, out)
    out.float().sum().backward()
    torch.cuda.synchronize()
    assert gc.CPU_INDEX >= 1, "nothing was offloaded, so this test proved nothing"
    # Recompute order: [2] = layer 2 (input [1]), [3] = layer 1 (input [0]).
    assert len(handed_back) == 4, "expected two forwards and two recomputes"
    return seen, handed_back, hidden.grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "drives the real offload path")
@pytest.mark.parametrize("offload", [torch.bfloat16], indirect = True)
def test_the_recompute_sees_the_dtype_the_forward_saw(offload):
    seen, handed_back, grad = _round_trip(torch.float16)
    assert set(seen) == {torch.float16}, f"the recompute ran in {sorted(map(str, set(seen)))}"
    assert torch.equal(handed_back[2], handed_back[1]), "layer 2's input was changed by the offload"
    assert torch.equal(handed_back[3], handed_back[0]), "layer 1's input was changed by the offload"
    assert grad.dtype == torch.float16
    assert torch.equal(grad, torch.full_like(grad, 4.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "drives the real offload path")
@pytest.mark.parametrize("offload", [torch.float16], indirect = True)
def test_a_wider_activation_is_still_offloaded_exactly(offload):
    seen, handed_back, grad = _round_trip(torch.float32)
    assert set(seen) == {torch.float32}, f"the recompute ran in {sorted(map(str, set(seen)))}"
    assert torch.equal(handed_back[2], handed_back[1]), "layer 2's input was changed by the offload"
    assert torch.equal(handed_back[3], handed_back[0]), "layer 1's input was changed by the offload"
    assert grad.dtype == torch.float32
    assert torch.equal(grad, torch.full_like(grad, 4.0))


def test_the_byte_view_reinterprets_without_casting():
    storage = torch.empty(4096, dtype = torch.bfloat16)
    for dtype in (torch.float16, torch.float32):
        values = torch.randn(1024, dtype = torch.float32).to(dtype).view(4, 256)
        nbytes = values.numel() * values.element_size()
        gc._view_bytes_as(storage, nbytes, dtype, values.shape).copy_(values)
        assert torch.equal(gc._view_bytes_as(storage, nbytes, dtype, values.shape), values)
        assert gc._elements_for(storage, nbytes) == nbytes // storage.element_size()
    values = torch.randn(1024).to(torch.float16)
    assert not torch.equal(values.to(torch.bfloat16).to(torch.float16), values)


def test_elements_for_rounds_up_so_an_odd_byte_count_still_fits():
    # Floor division would silently leave the byte view short of its shape.
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        storage = torch.empty(8, dtype = dtype)
        esize = storage.element_size()
        for nbytes in (1, esize - 1 or 1, esize, esize + 1, 3 * esize - 1):
            got = gc._elements_for(storage, nbytes)
            assert got * esize >= nbytes, f"{dtype} buffer sized {got} cannot hold {nbytes} bytes"
            assert (got - 1) * esize < nbytes, f"{dtype} buffer sized {got} for {nbytes} bytes is oversized"


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "initialisation allocates GPU buffers")
@pytest.mark.parametrize("offload", [torch.bfloat16], indirect = True)
def test_the_offload_cutoff_is_two_megabytes_whatever_the_dtype(offload):
    # Old cutoff was in init-dtype elements, so a 3MB fp32 activation (gemma3) stayed on GPU.
    assert gc.MINIMUM_SIZE == 2 * 1024 * 1024
    for dtype in (torch.float16, torch.float32):
        gc.initialize_unsloth_gradient_checkpointing(dtype)
        assert gc.MINIMUM_SIZE == 2 * 1024 * 1024, f"the cutoff moved with the init dtype {dtype}"

    gc.initialize_unsloth_gradient_checkpointing(torch.bfloat16)

    def layer(hidden):
        return hidden * 2

    # 786,432 float32 elements = 3MB: over 2MB of bytes, under 2MB/2 elements.
    hidden = torch.randn(1, 384, 2048, device = "cuda", dtype = torch.float32, requires_grad = True)
    out = gc.UnslothCheckpointFunction.apply(layer, False, hidden)
    out = gc.UnslothCheckpointFunction.apply(layer, False, out)
    old_cutoff_in_elements = 2 * 1024 * 1024 // torch.empty(0, dtype = torch.bfloat16).element_size()
    assert hidden.numel() * hidden.element_size() == 3 * 1024 * 1024
    assert hidden.numel() < old_cutoff_in_elements, "this activation no longer straddles the change"
    assert gc.CPU_INDEX >= 1, "a 3MB float32 activation was left on the GPU"
    out.float().sum().backward()
    torch.cuda.synchronize()
