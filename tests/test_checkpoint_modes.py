# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""What `_in_non_reentrant_checkpoint` answers for each `use_gradient_checkpointing`.

`training_utils.py` accepts exactly three values, and only ONE of them is a
region this has to defer a mode switch inside:

    "unsloth"  (the default)   False   correct, see below
    True                       True    the case this work exists for
    False                      False   nothing is recomputed

The `"unsloth"` answer is the one that looks wrong and is not.
`Unsloth_Offloaded_Gradient_Checkpointer` is a bare `autograd.Function`: its
backward recomputes the forward under `enable_grad` and then runs
`torch.autograd.backward(output, dY)` on that FRESH graph. It never unpacks
activations a compiled forward stashed, so the two halves cannot disagree and a
mode flip cannot strand it. Reentrant `torch.utils.checkpoint` is the same shape
and answers False for the same reason.

Non-reentrant checkpointing is the odd one out: it PACKS saved intermediates in
the forward via saved-tensor hooks and UNPACKS them in the backward, and those
two halves are exactly what a mid-step compile-mode flip desynchronises.

None of this was pinned by a test, so a future move of Unsloth's own
checkpointer toward the pack/unpack model would silently fall outside the guard
with nothing going red.

Answers checked on torch 2.6.0, 2.7.1, 2.8.0, 2.9.1, 2.10.0, 2.11.0 and 2.12.0,
identical on every one. 2.6 and 2.7 matter: they predate
`_top_saved_tensors_default_hooks` and answer via the frame walk, so both halves
of `_in_non_reentrant_checkpoint` are covered rather than only the fast one.
"""

import torch
import torch.utils.checkpoint as C

from unsloth_zoo.temporary_patches.utils import _in_non_reentrant_checkpoint


class _UnslothShapedCheckpointer(torch.autograd.Function):
    """The shape of `Unsloth_Offloaded_Gradient_Checkpointer`: save the input,
    recompute under `enable_grad`, run a fresh backward over the new graph."""

    @staticmethod
    def forward(ctx, forward_function, hidden_states):
        with torch.no_grad():
            output = forward_function(hidden_states)
        ctx.save_for_backward(hidden_states)
        ctx.forward_function = forward_function
        ctx.probed = _in_non_reentrant_checkpoint()
        _RESULTS["unsloth"] = ctx.probed
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (hidden_states,) = ctx.saved_tensors
        hidden_states = hidden_states.detach().requires_grad_(True)
        with torch.enable_grad():
            output = ctx.forward_function(hidden_states)
        torch.autograd.backward(output, grad_output)
        return None, hidden_states.grad


_RESULTS = {}


def _probe_inside(use_reentrant):
    seen = {}

    def inner(t):
        seen["answer"] = _in_non_reentrant_checkpoint()
        return t * 2

    x = torch.randn(4, 4, requires_grad = True)
    C.checkpoint(inner, x, use_reentrant = use_reentrant)
    return seen["answer"]


def test_unsloth_checkpointing_is_not_a_deferring_region():
    """It re-derives rather than unpacks, so there is nothing to strand."""
    _RESULTS.clear()
    x = torch.randn(4, 4, requires_grad = True)
    _UnslothShapedCheckpointer.apply(lambda t: t * 2, x)
    assert _RESULTS["unsloth"] is False


def test_gradient_checkpointing_true_is_a_deferring_region():
    """Non-reentrant packs in the forward and unpacks in the backward. This is
    the one the deferral exists for."""
    assert _probe_inside(use_reentrant = False) is True


def test_reentrant_checkpointing_is_not_a_deferring_region():
    assert _probe_inside(use_reentrant = True) is False


def test_gradient_checkpointing_false_is_not_a_deferring_region():
    assert _in_non_reentrant_checkpoint() is False


def test_the_unsloth_checkpointer_still_re_derives():
    """The reason `"unsloth"` is exempt, asserted rather than assumed. If its
    backward ever stops running a fresh `autograd.backward` over a recomputed
    graph, the exemption above stops being safe and this goes red first."""
    import inspect

    from unsloth_zoo.gradient_checkpointing import (
        Unsloth_Offloaded_Gradient_Checkpointer as Checkpointer,
    )

    backward = inspect.getsource(Checkpointer.backward)
    assert "torch.enable_grad()" in backward, "no longer recomputes under grad"
    assert "torch.autograd.backward(" in backward, "no longer re-derives"
    assert "saved_tensors_hooks" not in backward, "now packs, so it can be stranded"


def test_the_three_accepted_values_are_still_the_three_we_cover():
    """`training_utils` asserts on this set. A fourth value would need its own
    answer here, so fail rather than let it default."""
    import inspect

    from unsloth_zoo import training_utils

    source = inspect.getsource(training_utils.prepare_model_for_training)
    assert 'use_gradient_checkpointing in (True, False, "unsloth",)' in source, (
        "the accepted set moved; this file pins an answer for each value"
    )
