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
"""The eager fallback must not flip compile modes inside a checkpointed step.

Non-reentrant activation checkpointing (`use_reentrant = False`) runs the
region twice: once during the forward, where autograd packs every saved
intermediate, and once during the backward, where it recomputes them and
checks that the two agree. A compiled forward recomputed eagerly does not
agree, and torch aborts the backward with either

    torch.utils.checkpoint: Recomputed values for the following tensors have
    different metadata than during the forward pass.

or the sibling assertion

    AssertionError: Something went unexpectedly wrong in activation
    checkpoint. Please report this bug by filing an issue to PyTorch.

The eager fallback exists to survive an exhausted recompile cache, which is a
speed problem, and it must not turn one into a dead training step. When the
cache runs out mid-step it now buys a little more budget, finishes the call
compiled, and defers the switch to `apply_pending_eager_fallbacks()` at the
next step boundary.
"""
import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from unsloth_zoo.temporary_patches import utils as U


# aot_eager, not inductor: what this file tests is the compile-mode switch and
# what autograd saves, and AOTAutograd already changes both. Inductor adds a C++
# codegen step that fails for reasons of its own once an earlier test in the
# suite has touched the inductor config, which is how these two tests passed
# alone and failed in a full run.
_BACKEND = "aot_eager"


def _norm(w, x, k):
    # `k` is a plain int, so Dynamo specialises on its value and every new
    # block forces a fresh compilation. That is how a real vision run exhausts
    # the cache: one guard per shape rather than per constant, but the same
    # outcome.
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w * (1.0 + 0.0 * k)


class _Block(nn.Module):
    def __init__(self, d, fn, k):
        super().__init__()
        self.a = nn.Linear(d, d)
        self.b = nn.Linear(d, d)
        self.w = nn.Parameter(torch.ones(d))
        self.fn = fn
        self.k = k

    def forward(self, x):
        x = self.a(x).relu()
        x = self.fn(self.w, x, self.k)
        return self.b(x)


def _run_checkpointed_step(fn, n_blocks = 6, d = 16):
    torch.manual_seed(0)
    blocks = nn.ModuleList([_Block(d, fn, k) for k in range(n_blocks)])
    x = torch.randn(2, 4, d, requires_grad = True)
    h = x
    for block in blocks:
        h = checkpoint(block, h, use_reentrant = False)
    h.sum().backward()


def _reset_dynamo(limit):
    import torch._dynamo as dynamo
    dynamo.reset()
    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(dynamo.config, name):
            setattr(dynamo.config, name, limit)
    for name in ("accumulated_recompile_limit", "accumulated_cache_size_limit"):
        if hasattr(dynamo.config, name):
            setattr(dynamo.config, name, limit)


@pytest.fixture
def dynamo_limits():
    import torch._dynamo as dynamo
    names = ("recompile_limit", "cache_size_limit",
             "accumulated_recompile_limit", "accumulated_cache_size_limit")
    saved = {n: getattr(dynamo.config, n) for n in names
             if hasattr(dynamo.config, n)}
    n_wrappers = len(U._EAGER_FALLBACK_WRAPPERS)
    try:
        yield
    finally:
        for n, v in saved.items():
            setattr(dynamo.config, n, v)
        del U._EAGER_FALLBACK_WRAPPERS[n_wrappers:]
        dynamo.reset()


@pytest.mark.skipif(
    not U._recompile_limit_errors(),
    reason = "this torch has no recompile-limit exception to raise",
)
def test_exhausted_recompile_cache_does_not_break_the_backward(dynamo_limits):
    """The regression: cell 18 of Gemma4_(E2B)-Vision died here."""
    _reset_dynamo(2)
    compiled = torch.compile(_norm, fullgraph = True, dynamic = False,
                             backend = _BACKEND)
    fn = U._fall_back_to_eager_on_recompile_limit(compiled, _norm, "test_norm")

    # Before the fix this raises: the first blocks pack compiled activations,
    # the fallback flips to eager mid-forward, and the recompute disagrees.
    _run_checkpointed_step(fn)

    state = fn._unsloth_fallback_state
    assert state["pending_eager"], "the cache should have run out at all"
    assert not state["eager"], "the switch must wait for the step boundary"


@pytest.mark.skipif(
    not U._recompile_limit_errors(),
    reason = "this torch has no recompile-limit exception to raise",
)
def test_pending_switch_is_applied_at_the_step_boundary(dynamo_limits):
    _reset_dynamo(2)
    compiled = torch.compile(_norm, fullgraph = True, dynamic = False,
                             backend = _BACKEND)
    fn = U._fall_back_to_eager_on_recompile_limit(compiled, _norm, "test_norm")
    _run_checkpointed_step(fn)
    assert fn._unsloth_fallback_state["pending_eager"]

    assert U.apply_pending_eager_fallbacks() >= 1
    assert fn._unsloth_fallback_state["eager"]
    assert not fn._unsloth_fallback_state["pending_eager"]

    # A second call is a no-op, and the next step is now consistently eager,
    # so the backward keeps working.
    assert U.apply_pending_eager_fallbacks() == 0
    _run_checkpointed_step(fn)


def test_apply_pending_is_a_no_op_without_a_pending_switch(dynamo_limits):
    fn = U._fall_back_to_eager_on_recompile_limit(
        lambda *a, **k: None, lambda *a, **k: None, "untriggered",
    )
    if not hasattr(fn, "_unsloth_fallback_state"):
        pytest.skip("no fallback wrapper on this torch")
    assert U.apply_pending_eager_fallbacks() == 0
    assert not fn._unsloth_fallback_state["eager"]


def test_bump_recompile_limits_raises_whichever_names_torch_uses(dynamo_limits):
    import torch._dynamo as dynamo
    names = [n for n in ("recompile_limit", "cache_size_limit")
             if hasattr(dynamo.config, n)]
    assert names, "no recompile limit on this torch at all"
    before = getattr(dynamo.config, names[0])
    assert U._bump_recompile_limits(7)
    assert getattr(dynamo.config, names[0]) == before + 7
