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

"""Offloaded checkpointing on a non-default current stream, as an in-process vLLM engine sets
(vllm/utils/torch_utils.py current_stream): the offload copy must wait for that stream."""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason = "Needs a real accelerator to offload to",
)

DTYPE = torch.bfloat16
SHAPE = (4, 512, 1024)   # over MINIMUM_SIZE, so the activation is offloaded


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
    gc_module.initialize_unsloth_gradient_checkpointing(DTYPE)
    try:
        yield gc_module
    finally:
        for name, value in saved.items(): setattr(gc_module, name, value)


def _block(hidden, side):
    return torch.sin(hidden) * side


def _run(checkpoint_fn, stream):
    torch.manual_seed(0)
    base = torch.randn(SHAPE, dtype = DTYPE, device = "cuda", requires_grad = True)
    side = torch.randn(SHAPE, dtype = DTYPE, device = "cuda", requires_grad = True)
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        # Keep the compute stream busy so the activation is written well after the offload could start.
        torch.cuda._sleep(200_000_000)
        hidden = base * 3.0
        if checkpoint_fn is None:
            output = _block(hidden, side)
        else:
            output = checkpoint_fn(_block, hidden, side, use_reentrant = True)
        output.float().sum().backward()
    torch.cuda.synchronize()
    return base.grad, side.grad


def test_offload_waits_for_the_current_stream(offload_module):
    stream = torch.cuda.Stream()
    expected = _run(None, stream)
    before = offload_module.CPU_INDEX
    actual = _run(offload_module.unsloth_offloaded_gradient_checkpoint, stream)
    assert offload_module.CPU_INDEX > before, "activation was not offloaded"
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want)
