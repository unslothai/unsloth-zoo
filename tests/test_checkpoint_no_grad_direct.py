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

"""A checkpoint entered with grad off runs the block directly (no Function, saved inputs, offload slot or counter step)."""

import pytest
import torch


@pytest.fixture
def gc_module(monkeypatch):
    from unsloth_zoo import gradient_checkpointing as module
    for name, value in (
        ("FIRST_PASS", False), ("LAST_GC_INDEX", 0), ("CURRENT_GC_INDEX", 0),
        ("MINIMUM_SIZE", 1 << 40), ("CPU_INDEX", 0),
        ("CPU_BUFFERS", [torch.empty(0)]), ("BACKWARD_PASS", False),
        ("USE_DOUBLE_BUFFER", False), ("GPU_BUFFERS", []), ("GPU_BUFFERS_B", None),
    ):
        monkeypatch.setattr(module, name, value, raising = False)
    return module


SHIMS = ("unsloth_checkpoint", "unsloth_offloaded_gradient_checkpoint", "unsloth_gradient_checkpoint")
FUNCTIONS = {
    "unsloth_checkpoint": "UnslothCheckpointFunction",
    "unsloth_offloaded_gradient_checkpoint": "UnslothCheckpointFunction",
    "unsloth_gradient_checkpoint": "Unsloth_Gradient_Checkpointer",
}


def _spy_forward(gc_module, monkeypatch, shim):
    cls = getattr(gc_module, FUNCTIONS[shim])
    real = cls.forward
    calls = []

    def forward(ctx, *args):
        calls.append(torch.is_grad_enabled())
        return real(ctx, *args)
    monkeypatch.setattr(cls, "forward", staticmethod(forward))
    return calls


def _block(weight):
    def block(hidden, scale = None):
        out = torch.tanh(hidden @ weight)
        return out * scale if scale is not None else out
    return block


@pytest.mark.parametrize("shim", SHIMS)
@pytest.mark.parametrize("mode", ["no_grad", "inference_mode"])
def test_grad_off_runs_block_directly(gc_module, monkeypatch, shim, mode):
    calls = _spy_forward(gc_module, monkeypatch, shim)
    weight = torch.randn(8, 8)
    # The input-grad hook leaves the first layer's input requires_grad under no_grad.
    hidden = torch.randn(2, 8).requires_grad_(True)
    scale = torch.randn(2, 8).requires_grad_(True)
    stats = getattr(gc_module, "NO_GRAD_CHECKPOINT_STATS", {})
    before = stats.get("calls", 0)
    context = torch.no_grad() if mode == "no_grad" else torch.inference_mode()
    with context:
        out = getattr(gc_module, shim)(_block(weight), hidden, scale = scale, use_reentrant = True)
        expected = _block(weight)(hidden, scale = scale)
    assert calls == []
    assert gc_module.CURRENT_GC_INDEX == 0 and gc_module.LAST_GC_INDEX == 0
    assert stats["calls"] == before + 1
    assert not out.requires_grad
    torch.testing.assert_close(out, expected, rtol = 0, atol = 0)


@pytest.mark.parametrize("shim", SHIMS)
def test_grad_on_still_checkpoints(gc_module, monkeypatch, shim):
    calls = _spy_forward(gc_module, monkeypatch, shim)
    weight = torch.randn(8, 8, requires_grad = True)
    hidden = torch.randn(2, 8, requires_grad = True)
    out = getattr(gc_module, shim)(_block(weight), hidden, use_reentrant = True)
    assert len(calls) == 1
    out.sum().backward()
    ref_weight = weight.detach().clone().requires_grad_(True)
    ref_hidden = hidden.detach().clone().requires_grad_(True)
    _block(ref_weight)(ref_hidden).sum().backward()
    torch.testing.assert_close(hidden.grad, ref_hidden.grad)


def test_grad_off_scopes_out_routing_frame(gc_module):
    """As the checkpoint's own first pass: an outer frame is not visible inside the block."""
    seen = []

    def block(hidden):
        seen.append(getattr(gc_module._GC_ROUTING_TLS, "frame", None))
        return hidden * 2
    outer = gc_module.CheckpointRoutingFrame()
    with gc_module._checkpoint_routing_scope(outer, False), torch.no_grad():
        gc_module.unsloth_checkpoint(block, torch.randn(2, 2).requires_grad_(True), use_reentrant = True)
    assert seen == [None]


def test_torch_reentrant_checkpoint_routed_to_unsloth_function(gc_module, monkeypatch):
    """A pre-bound torch checkpoint reaches UnslothCheckpointFunction via CheckpointFunction: grad off runs directly."""
    pristine = getattr(torch.utils.checkpoint, "_unsloth_pristine_checkpoint", None)
    if pristine is None:
        pristine = torch.utils.checkpoint.checkpoint
    if getattr(pristine, "__name__", "") in gc_module._UNSLOTH_CKPT_SHIM_NAMES:
        pytest.skip("torch's own checkpoint is not reachable")
    monkeypatch.setattr(torch.utils.checkpoint, "CheckpointFunction", gc_module.UnslothCheckpointFunction)
    calls = _spy_forward(gc_module, monkeypatch, "unsloth_checkpoint")
    weight = torch.randn(8, 8)
    hidden = torch.randn(2, 8).requires_grad_(True)
    with torch.no_grad():
        out = pristine(_block(weight), hidden, use_reentrant = True)
    assert calls == []
    torch.testing.assert_close(out, torch.tanh(hidden.detach() @ weight), rtol = 0, atol = 0)
    # Grad on: still the checkpoint Function, with a correct gradient.
    weight.requires_grad_(True)
    out = pristine(_block(weight), hidden, use_reentrant = True)
    assert len(calls) == 1
    out.sum().backward()
    ref = hidden.detach().clone().requires_grad_(True)
    torch.tanh(ref @ weight.detach()).sum().backward()
    torch.testing.assert_close(hidden.grad, ref.grad)
