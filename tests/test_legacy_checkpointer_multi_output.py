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

"""patch_gradient_checkpointing() swaps torch.utils.checkpoint.checkpoint for Unsloth_Gradient_Checkpointer, so every
caller of torch's checkpoint goes through it, including ones whose function returns several tensors (a chunked
log-prob / entropy pair, a block returning hidden states and an aux loss). Those must get the same gradients as
without checkpointing. CPU only."""
import torch

from unsloth_zoo.gradient_checkpointing import Unsloth_Gradient_Checkpointer


def _grads(fn, use_checkpointer, weights):
    torch.manual_seed(0)
    x = torch.randn(4, 8, requires_grad = True)
    w = torch.randn(8, 8, requires_grad = True)
    if use_checkpointer:
        outs = Unsloth_Gradient_Checkpointer.apply(fn, x, w)
    else:
        outs = fn(x, w)
    loss = sum(c * o.float().sum() for c, o in zip(weights, outs) if c and o.requires_grad)
    loss.backward()
    return x.grad, w.grad


def _two(h, w):
    y = h @ w
    return y.sin(), y.cos()


def _with_integer_output(h, w):
    y = h @ w
    return y.tanh(), y.argmax(dim = -1), y.exp()


def test_two_outputs_match_no_checkpoint():
    ref = _grads(_two, False, (1.0, 2.0))
    got = _grads(_two, True, (1.0, 2.0))
    for a, b in zip(got, ref):
        torch.testing.assert_close(a, b)


def test_unused_output_and_integer_output():
    # the integer output never gets a gradient, the last output is unused (its incoming grad is None)
    ref = _grads(_with_integer_output, False, (1.0, 0.0, 0.0))
    got = _grads(_with_integer_output, True, (1.0, 0.0, 0.0))
    for a, b in zip(got, ref):
        torch.testing.assert_close(a, b)


def test_single_tensor_and_one_tuple_unchanged():
    def one(h, w):
        return (h @ w).relu()

    def one_tuple(h, w):
        return ((h @ w).relu(),)

    for fn in (one, one_tuple):
        torch.manual_seed(0)
        x = torch.randn(4, 8, requires_grad = True)
        w = torch.randn(8, 8, requires_grad = True)
        out = Unsloth_Gradient_Checkpointer.apply(fn, x, w)
        out = out[0] if isinstance(out, tuple) else out
        out.sum().backward()
        gx, gw = x.grad.clone(), w.grad.clone()
        x.grad = w.grad = None
        (x @ w).relu().sum().backward()
        torch.testing.assert_close(gx, x.grad)
        torch.testing.assert_close(gw, w.grad)


def test_no_differentiable_output_still_raises():
    # torch's reentrant checkpoint raises when no recomputed output requires grad, and so did this shim for a single
    # output; silently returning no gradients would turn that error into a step that trains nothing.
    import pytest

    def constant(h, w):
        return torch.ones_like(h) * 1.0

    def constant_pair(h, w):
        return torch.ones_like(h) * 1.0, torch.zeros_like(h) * 1.0

    for fn in (constant, constant_pair):
        x = torch.randn(4, 8, requires_grad = True)
        w = torch.randn(8, 8, requires_grad = True)
        outs = Unsloth_Gradient_Checkpointer.apply(fn, x, w)
        outs = outs if isinstance(outs, tuple) else (outs,)
        with pytest.raises(RuntimeError):
            sum(o.sum() for o in outs).backward()
