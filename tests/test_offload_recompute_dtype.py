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
def offload(request, monkeypatch):
    saved = {name: getattr(gc, name, _MISSING) for name in _STATE + ["FP32_OFFLOAD_EXACT"]}
    dtype, mode = request.param if isinstance(request.param, tuple) else (request.param, "bf16")
    monkeypatch.setenv("UNSLOTH_OFFLOAD_FP32", mode)
    gc.initialize_unsloth_gradient_checkpointing(dtype)
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
@pytest.mark.parametrize("offload", [(torch.float16, "exact")], indirect = True)
def test_a_wider_activation_is_still_offloaded_exactly(offload):
    seen, handed_back, grad = _round_trip(torch.float32)
    assert set(seen) == {torch.float32}, f"the recompute ran in {sorted(map(str, set(seen)))}"
    assert torch.equal(handed_back[2], handed_back[1]), "layer 2's input was changed by the offload"
    assert torch.equal(handed_back[3], handed_back[0]), "layer 1's input was changed by the offload"
    assert grad.dtype == torch.float32
    assert torch.equal(grad, torch.full_like(grad, 4.0))


def test_the_byte_view_reinterprets_without_casting():
    storage = torch.empty(4096, dtype = torch.bfloat16)
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        values = torch.randn(1024, dtype = torch.float32).to(dtype).view(4, 256)
        nbytes = values.numel() * values.element_size()
        gc._view_bytes_as(storage, nbytes, dtype, values.shape).copy_(values)
        assert torch.equal(gc._view_bytes_as(storage, nbytes, dtype, values.shape), values)
    values = torch.randn(1024).to(torch.float16)
    assert not torch.equal(values.to(torch.bfloat16).to(torch.float16), values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "initialisation allocates GPU buffers")
@pytest.mark.parametrize("offload", [(torch.bfloat16, "bf16"), (torch.bfloat16, "exact")], indirect = True)
def test_the_offload_cutoff_counts_stored_bytes(offload):
    # bf16-stored float32 must offload exactly where the old bf16 buffers did (numel > 1M), so host memory
    # does not grow; exact float32 storage counts its own 4 bytes per element.
    assert gc.MINIMUM_SIZE == 2 * 1024 * 1024
    for dtype in (torch.float16, torch.float32):
        gc.initialize_unsloth_gradient_checkpointing(dtype)
        assert gc.MINIMUM_SIZE == 2 * 1024 * 1024, f"the cutoff moved with the init dtype {dtype}"
    gc.initialize_unsloth_gradient_checkpointing(torch.bfloat16)

    def layer(hidden):
        return hidden * 2

    # 786,432 float32 elements = 3MB of activation, 1.5MB when stored as bf16.
    hidden = torch.randn(1, 384, 2048, device = "cuda", dtype = torch.float32, requires_grad = True)
    out = gc.UnslothCheckpointFunction.apply(layer, False, hidden)
    out = gc.UnslothCheckpointFunction.apply(layer, False, out)
    if gc.FP32_OFFLOAD_EXACT:
        assert gc.CPU_INDEX >= 1, "exact storage holds 3MB, over the cutoff, yet nothing was offloaded"
    else:
        assert gc.CPU_INDEX == 0, "1.5MB of bf16 was offloaded; the old bf16 buffers kept it on the GPU"
    out.float().sum().backward()
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "drives the real offload path")
@pytest.mark.parametrize("offload", [(torch.bfloat16, "bf16")], indirect = True)
def test_a_float32_activation_is_stored_as_bf16_and_recomputed_in_float32(offload):
    seen = []
    handed_back = []

    def layer(hidden):
        seen.append(hidden.dtype)
        handed_back.append(hidden.detach().clone())
        return hidden * 2

    hidden = (torch.randn(2, 1024, 2048, device = "cuda") * 50).requires_grad_(True)
    with torch.no_grad():
        hidden[..., 3] = 2.0e5
    out = gc.UnslothCheckpointFunction.apply(layer, False, hidden)
    out = gc.UnslothCheckpointFunction.apply(layer, False, out)
    stored = [x for x in gc.CPU_BUFFERS[: gc.CPU_INDEX]]
    out.sum().backward()
    torch.cuda.synchronize()
    assert gc.CPU_INDEX >= 1, "nothing was offloaded, so this test proved nothing"
    assert set(seen) == {torch.float32}, f"the recompute ran in {sorted(map(str, set(seen)))}"
    for recomputed, original in ((handed_back[2], handed_back[1]), (handed_back[3], handed_back[0])):
        assert torch.equal(recomputed, original.to(torch.bfloat16).float()), "not the bf16 rounding of the input"
        assert torch.isfinite(recomputed).all()
    # Same bytes as the old bf16 buffers: 2 per element, not 4.
    assert all(x.untyped_storage().nbytes() >= hidden.numel() * 2 for x in stored)
    assert all(x.untyped_storage().nbytes() < hidden.numel() * 4 for x in stored)
    assert torch.equal(hidden.grad, torch.full_like(hidden.grad, 4.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA tensors")
def test_cuda_arg_fast_path_matches_torch_device_state_helpers():
    from torch.utils.checkpoint import _infer_device_type, get_device_states
    a = torch.randn(4, device = "cuda")
    p = torch.nn.Parameter(torch.randn(4, device = "cuda"))
    for args in ((a,), (a, p), (a, a)):
        devices = gc._cuda_tensor_arg_devices(args)
        assert _infer_device_type(*args) == "cuda"
        torch_devices, torch_states = get_device_states(*args)
        assert devices == torch_devices
        for device, state in zip(devices, torch_states):
            assert torch.equal(torch.cuda.get_rng_state(device), state)
    # Anything else keeps torch's pytree walk.
    for args in ((a, None), ((a,),), (a.cpu(),), (a, 1), ()):
        assert gc._cuda_tensor_arg_devices(args) is None
