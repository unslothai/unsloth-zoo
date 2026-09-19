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

"""An offloaded activation must come back in the dtype it went out in.

The staging buffers are allocated in the dtype checkpointing was initialised with. That is
not always the dtype the model runs in: a FORCE_FLOAT32 family (qwen3_5, gemma3) on a GPU
without bfloat16 initialises bfloat16 and runs float16. Forward stored the float16 hidden
states, backward copied them into the bfloat16 GPU buffer and handed THAT to the recompute.

So the recompute ran on bfloat16 hidden states against float16 weights. Usually that is
`RuntimeError: expected mat1 and mat2 to have the same dtype, but got: BFloat16 != Half`.
On ROCm gfx10 (RX 6500 XT, found there) the bf16 tensor reached fla's gated-delta Triton
kernel first, which LLVM cannot compile, and the training process died with no Python
exception: `LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.fdot2.bf16.bf16`.

Only offloaded activations are affected, so it needs a sequence length of 512 or more and an
activation over 2 MB. Every short test passes.
"""
import inspect

import pytest

torch = pytest.importorskip("torch")

from unsloth_zoo import gradient_checkpointing as gc


def _source(fn):
    return inspect.getsource(fn)


def test_backward_views_the_staging_buffers_in_the_saved_dtype():
    backward = _source(gc.UnslothCheckpointFunction.backward)
    assert "saved_dtype = ctx._saved_dtype" in backward
    for buffer in ("GPU_BUFFERS_B[device_index]", "GPU_BUFFERS[device_index]", "host_buffer"):
        assert f"{buffer}[:new_size].view(saved_dtype).view(shape)" in backward, (
            f"{buffer} is handed to the recompute in the buffer's dtype, not the activation's"
        )
        assert f"{buffer}[:new_size].view(shape)" not in backward


def test_forward_records_the_dtype_and_never_casts_into_the_host_slot():
    forward = _source(gc.UnslothCheckpointFunction.forward)
    assert "ctx._saved_dtype = arg.dtype" in forward
    assert "x[:new_size].view(arg.dtype).view(shape)" in forward
    assert "x = x[:new_size].view(shape)" not in forward


def test_only_same_width_activations_are_offloaded():
    """A view needs equal element sizes. A float32 activation over 16-bit buffers stays on the
    GPU rather than being cast, and the width check has to come before the offload decision."""
    forward = _source(gc.UnslothCheckpointFunction.forward)
    gate = forward.index("_same_width = arg.element_size() ==")
    decision = forward.index("use_gpu_buffer = True")
    assert gate < decision
    assert "if _same_width and new_size > MINIMUM_SIZE" in forward


def test_a_bf16_buffer_viewed_as_fp16_is_bit_exact():
    """What the fix relies on: reinterpreting 16-bit storage loses nothing, where the old
    float16 -> bfloat16 cast dropped mantissa bits on the way out and again on the way back."""
    storage = torch.empty(64, dtype = torch.bfloat16)
    values = torch.randn(64).to(torch.float16)
    storage.view(torch.float16).copy_(values)
    assert torch.equal(storage.view(torch.float16), values)
    assert not torch.equal(values.to(torch.bfloat16).to(torch.float16), values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "drives the real offload path")
def test_the_recompute_sees_the_dtype_the_forward_saw():
    names = [
        "CPU_BUFFERS", "CPU_INDEX", "GPU_BUFFERS", "GPU_BUFFERS_B", "MAIN_STREAMS", "EXTRA_STREAMS",
        "BACKWARD_PASS", "LAST_GC_INDEX", "FIRST_PASS", "CURRENT_GC_INDEX", "USE_UNSLOTH_GC",
        "USE_DOUBLE_BUFFER", "MINIMUM_SIZE", "NEXT_BUFFER_SLOT", "BUFFER_EVENTS_A", "BUFFER_EVENTS_B",
    ]
    missing = object()
    saved = {name: getattr(gc, name, missing) for name in names}
    try:
        # Initialised bfloat16, like a FORCE_FLOAT32 load; the model then runs float16.
        gc.initialize_unsloth_gradient_checkpointing(torch.bfloat16)
        seen = []

        def layer(hidden):
            seen.append(hidden.dtype)
            return hidden * 2

        # Two checkpointed layers: the last one is never offloaded, so the first one is.
        hidden = torch.randn(2, 1024, 2048, device = "cuda", dtype = torch.float16, requires_grad = True)
        out = gc.UnslothCheckpointFunction.apply(layer, False, hidden)
        out = gc.UnslothCheckpointFunction.apply(layer, False, out)
        out.float().sum().backward()
        torch.cuda.synchronize()

        assert gc.CPU_BUFFERS, "nothing was offloaded, so this test proved nothing"
        assert set(seen) == {torch.float16}, f"the recompute ran in {sorted(map(str, set(seen)))}"
        assert hidden.grad is not None and hidden.grad.dtype == torch.float16
        assert torch.equal(hidden.grad, torch.full_like(hidden.grad, 4.0))
    finally:
        for name, value in saved.items():
            if value is missing:
                if hasattr(gc, name):
                    delattr(gc, name)
            else:
                setattr(gc, name, value)
