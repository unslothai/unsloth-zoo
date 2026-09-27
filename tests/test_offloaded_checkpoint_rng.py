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

"""Offloaded checkpointing against the real pinned buffers: the CPU coverage keeps
MINIMUM_SIZE above every tensor, so the D2H/H2D branch never runs there."""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason = "Needs a real accelerator to offload to",
)

DTYPE = torch.bfloat16
SHAPE = (4, 512, 1024)   # over MINIMUM_SIZE (2MB of elements) in bf16, so offload engages


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


def _pristine_checkpoint():
    return getattr(
        torch.utils.checkpoint, "_unsloth_pristine_checkpoint",
        torch.utils.checkpoint.checkpoint,
    )


def _block(hidden, side):
    return torch.nn.functional.dropout(hidden * 2.0 + side, p = 0.5, training = True)


def _inputs(device):
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    hidden = torch.randn(SHAPE, dtype = DTYPE, device = device, requires_grad = True)
    side   = torch.randn(SHAPE, dtype = DTYPE, device = device, requires_grad = True)
    return hidden, side


def _run(checkpoint_fn, kwargs, device):
    hidden, side = _inputs(device)
    torch.manual_seed(99)
    torch.cuda.manual_seed_all(99)
    output = checkpoint_fn(_block, hidden, side, use_reentrant = True, **kwargs)
    output.sum().backward()
    torch.cuda.synchronize()
    return output, hidden.grad, side.grad


@pytest.mark.parametrize("preserve_rng_state", [None, True, False])
def test_offloaded_matches_torch_with_real_offload(offload_module, preserve_rng_state):
    """Pre-fix the flag slot ate ``hidden``, so both grads matter."""
    device = torch.device("cuda")
    kwargs = {} if preserve_rng_state is None else {"preserve_rng_state" : preserve_rng_state}

    expected = _run(_pristine_checkpoint(), kwargs, device)

    before = offload_module.CPU_INDEX
    actual = _run(offload_module.unsloth_offloaded_gradient_checkpoint, kwargs, device)
    assert offload_module.CPU_INDEX > before, \
        "activation was not offloaded - this test did not exercise the path it claims to"

    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want)


def test_offloaded_preserve_rng_state_false_reaches_the_function(offload_module):
    """True forks and restores the RNG; False lets the recompute advance it."""
    device = torch.device("cuda")

    _run(offload_module.unsloth_offloaded_gradient_checkpoint, {"preserve_rng_state" : True}, device)
    after_true = (torch.get_rng_state(), torch.cuda.get_rng_state())

    _run(offload_module.unsloth_offloaded_gradient_checkpoint, {"preserve_rng_state" : False}, device)
    after_false = (torch.get_rng_state(), torch.cuda.get_rng_state())

    assert not (
        torch.equal(after_true[0], after_false[0]) and torch.equal(after_true[1], after_false[1])
    ), "preserve_rng_state made no difference - the flag never reached the Function"


def test_offloaded_binds_tensor_keywords_after_the_rng_flag(offload_module):
    """Reference is eager, not torch: torch's reentrant path rejects keywords
    outright, which is what _bind_checkpoint_kwargs deliberately replaces."""
    device = torch.device("cuda")

    def run(checkpoint_fn):
        hidden, side = _inputs(device)
        torch.manual_seed(7)
        torch.cuda.manual_seed_all(7)
        if checkpoint_fn is None:
            output = _block(hidden, side = side)
        else:
            output = checkpoint_fn(
                _block, hidden, side = side, use_reentrant = True, preserve_rng_state = True,
            )
        output.sum().backward()
        torch.cuda.synchronize()
        return output, hidden.grad, side.grad

    expected = run(None)
    actual   = run(offload_module.unsloth_offloaded_gradient_checkpoint)
    for got, want in zip(actual, expected):
        assert want is not None
        torch.testing.assert_close(got, want)


def test_offloaded_stays_correct_across_repeated_steps(offload_module):
    """Backward clears FIRST_PASS and sets BACKWARD_PASS, changing which layers
    offload next forward; one step never reaches that regime."""
    device = torch.device("cuda")
    checkpoint = offload_module.unsloth_offloaded_gradient_checkpoint

    for step in range(3):
        expected = _run(_pristine_checkpoint(), {"preserve_rng_state" : True}, device)
        actual   = _run(checkpoint, {"preserve_rng_state" : True}, device)
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, msg = lambda m, s = step: f"step {s}: {m}")
