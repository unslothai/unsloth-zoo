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

Non-reentrant activation checkpointing (`use_reentrant = False`) runs the region
twice -- packing every saved intermediate in the forward, recomputing them in the
backward -- and checks the two agree. A compiled forward recomputed eagerly does
not, and torch aborts the backward with either

    torch.utils.checkpoint: Recomputed values for the following tensors have
    different metadata than during the forward pass.

or the sibling

    AssertionError: Something went unexpectedly wrong in activation
    checkpoint. Please report this bug by filing an issue to PyTorch.

The fallback exists to survive an exhausted recompile cache, a speed problem,
and must not turn one into a dead training step. It now buys a little more
budget, finishes the call compiled, and defers the switch to
`apply_pending_eager_fallbacks()` at the next step boundary.
"""
import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from unsloth_zoo.temporary_patches import utils as U


# aot_eager, not inductor: this file tests the compile-mode switch and what
# autograd saves, both of which AOTAutograd already covers. Inductor adds a C++
# codegen step that fails for reasons of its own once an earlier test touched
# the inductor config -- why these passed alone and failed in a full run.
_BACKEND = "aot_eager"


def _norm(w, x, k):
    # `k` is a plain int, so Dynamo specialises on its value and every block
    # forces a fresh compilation. A real vision run exhausts the cache the same
    # way, one guard per shape rather than per constant.
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
    # These tests end with a bump deliberately outstanding, so the bookkeeping
    # has to come back too: left behind, a later fallback restores this test's
    # limit of 2 over the process default, or finds the allowance spent. Running
    # out also takes EVERY live borrower eager, including the package's own
    # patched kernels, which would stay latched for the rest of the worker.
    saved_states = [(_w, dict(_w._unsloth_fallback_state))
                    for _w in (_r() for _r in U._EAGER_FALLBACK_WRAPPERS)
                    if _w is not None]
    saved_global = U._GLOBAL_BUMPS
    saved_orig = dict(U._ORIGINAL_RECOMPILE_LIMITS)
    # Deep copy: the map is {bumped value: the value it came from} with lists
    # appended to and popped in place, so copying it as a set broke the restore
    # chain, and copying only the outer layers let a test that settles a debt
    # mutate the snapshot teardown restores from.
    saved_bumped = {k: {kk: list(vv) for kk, vv in v.items()}
                    for k, v in U._BUMPED_RECOMPILE_LIMITS.items()}
    # The give-up decision is kept by LABEL, so it outlives the wrapper. Every
    # test here reuses "test_norm", so an earlier latch would start the next
    # one already eager.
    U._LATCHED_EAGER_LABELS.discard("test_norm")
    U._PENDING_EAGER_LABELS.discard("test_norm")
    try:
        yield
    finally:
        for n, v in saved.items():
            setattr(dynamo.config, n, v)
        # Drop only what died: the registry is process-wide, and a module
        # imported during these tests appends its own kernels past the mark, so
        # truncating deregistered gemma/gemma4/qwen3 for the rest of the worker.
        _tail = [_r for _r in U._EAGER_FALLBACK_WRAPPERS[n_wrappers:]
                 if _r() is not None]
        del U._EAGER_FALLBACK_WRAPPERS[n_wrappers:]
        U._EAGER_FALLBACK_WRAPPERS.extend(_tail)
        for _w, _st in saved_states:
            _w._unsloth_fallback_state.clear()
            _w._unsloth_fallback_state.update(_st)
        U._GLOBAL_BUMPS = saved_global
        U._ORIGINAL_RECOMPILE_LIMITS.clear()
        U._ORIGINAL_RECOMPILE_LIMITS.update(saved_orig)
        U._BUMPED_RECOMPILE_LIMITS.clear()
        U._BUMPED_RECOMPILE_LIMITS.update(saved_bumped)
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
@pytest.mark.skipif(
    U._in_non_reentrant_checkpoint() is None,
    reason = "torch < 2.8 cannot report whether a checkpoint region is live",
)
def test_a_spent_budget_ends_the_step_instead_of_flipping_mid_region(dynamo_limits):
    """The same regression, on the path where there is no budget left to buy.

    Deferring needs a compiled result to defer with, and here the call raised
    and the retry could not be paid for. Running it eagerly finishes the forward
    but strands what the step already packed compiled, so the backward aborts or
    (when the shapes line up) returns silently wrong gradients. End the step and
    let the caller retry it consistently."""
    _reset_dynamo(2)
    compiled = torch.compile(_norm, fullgraph = True, dynamic = False,
                             backend = _BACKEND)
    fn = U._fall_back_to_eager_on_recompile_limit(compiled, _norm, "test_norm")

    # Deny the loan at its source: pinning `_GLOBAL_BUMPS` stopped working once
    # the allowance became a live measurement rather than a stored count, and
    # the retry silently succeeded.
    real_bump = U._bump_recompile_limits
    U._bump_recompile_limits = lambda *a, **k: False
    try:
        with pytest.raises(U._recompile_limit_errors()):
            _run_checkpointed_step(fn)
    finally:
        U._bump_recompile_limits = real_bump

    state = fn._unsloth_fallback_state
    assert state["eager"], "the wrapper must latch so the retry is consistent"
    assert state["bumps"] == 0, "no budget was borrowed"

    # The promise is that the caller can retry the step. torch holds
    # `with _checkpoint_hook(...)` open across the generator's yield, so our
    # raise abandons it with the hooks installed; the boundary settles that.
    U.apply_pending_eager_fallbacks()
    def plain(x): return torch.nn.functional.softmax(x * 2, dim = -1)
    checkpoint(plain, torch.randn(4, 4, requires_grad = True),
               use_reentrant = False).sum().backward()


def test_the_pre_2_8_fallback_answers_the_same_as_the_accessor():
    """torch 2.4 to 2.7 have no accessor, and answering None there would quietly
    restore the old behaviour on releases pyproject still supports. The frame
    walk must agree with the accessor where both exist, including False for
    reentrant checkpointing, which is not at risk."""
    assert U._walk_for_checkpoint_frame() is False, "outside any region"

    seen = []
    def probe(x):
        seen.append((U._in_non_reentrant_checkpoint(), U._walk_for_checkpoint_frame()))
        return torch.nn.functional.softmax(x * 2, dim = -1)

    for reentrant, expected in ((False, True), (True, False)):
        seen.clear()
        x = torch.randn(4, 4, requires_grad = True)
        checkpoint(probe, x, use_reentrant = reentrant).sum().backward()
        assert seen, "the probe never ran"
        for accessor, walked in seen:
            assert walked is expected, f"use_reentrant={reentrant}: {walked}"
            if accessor is not None:
                assert accessor == walked, "the two disagree"


def test_a_user_hook_on_top_does_not_hide_the_region():
    """`saved_tensors_hooks` / `save_on_cpu` entered inside a checkpointed
    function sits above ours, and the accessor reports only the top one. Reading
    that as "no region" sends the give-up path eager in exactly the case it
    exists to refuse."""
    seen = []
    def probe(x):
        seen.append(U._in_non_reentrant_checkpoint())
        return torch.nn.functional.softmax(x * 2, dim = -1)

    def with_user_hook(x):
        with torch.autograd.graph.save_on_cpu():
            return probe(x)

    x = torch.randn(4, 4, requires_grad = True)
    checkpoint(with_user_hook, x, use_reentrant = False).sum().backward()
    assert seen and all(s is True for s in seen), seen


def test_a_reentrant_region_does_not_hide_an_outer_non_reentrant_one():
    """The frame walk is all torch 2.4 to 2.7 have, and a reentrant checkpoint
    nested inside a non-reentrant one used to end it at the reentrant frames and
    answer False -- while the outer region is just as strandable by a flip."""
    seen = []
    def probe(x):
        seen.append((U._in_non_reentrant_checkpoint(), U._walk_for_checkpoint_frame()))
        return torch.nn.functional.softmax(x * 2, dim = -1)

    def outer(x):
        return checkpoint(probe, x, use_reentrant = True)

    x = torch.randn(4, 4, requires_grad = True)
    checkpoint(outer, x, use_reentrant = False).sum().backward()
    assert seen, "the probe never ran"
    assert any(walked for _, walked in seen), f"outer region missed: {seen}"
    for accessor, walked in seen:
        if accessor is not None:
            assert accessor == walked, f"the two disagree: {seen}"


def test_an_older_torch_falls_through_to_the_frame_walk():
    """Standing in for 2.4 to 2.7, where the accessor does not exist. Returning
    None there would silently restore the pre-fix behaviour."""
    seen = []
    def probe(x):
        seen.append(U._in_non_reentrant_checkpoint())
        return torch.nn.functional.softmax(x * 2, dim = -1)

    real = U._saved_tensor_hook_accessor
    U._saved_tensor_hook_accessor = lambda: None
    try:
        assert U._in_non_reentrant_checkpoint() is False, "outside any region"
        for reentrant, expected in ((False, True), (True, False)):
            seen.clear()
            x = torch.randn(4, 4, requires_grad = True)
            checkpoint(probe, x, use_reentrant = reentrant).sum().backward()
            assert seen and all(s is expected for s in seen), \
                f"use_reentrant={reentrant}: {seen}"
    finally:
        U._saved_tensor_hook_accessor = real


@pytest.mark.skipif(
    not U._recompile_limit_errors(),
    reason = "this torch has no recompile-limit exception to raise",
)
def test_a_spent_budget_still_falls_back_outside_a_checkpoint(dynamo_limits):
    """The majority case must not regress. With no checkpoint region live there
    is nothing packed to strand, so eager is safe and the run continues."""
    _reset_dynamo(2)
    compiled = torch.compile(_norm, fullgraph = True, dynamic = False,
                             backend = _BACKEND)
    fn = U._fall_back_to_eager_on_recompile_limit(compiled, _norm, "test_norm")

    real_bump = U._bump_recompile_limits
    U._bump_recompile_limits = lambda *a, **k: False
    try:
        torch.manual_seed(0)
        w = torch.randn(16, requires_grad = True)
        for k in range(8):                      # a fresh guard variant each time
            fn(w, torch.randn(2, 4, 16), k).sum().backward()
    finally:
        U._bump_recompile_limits = real_bump

    assert fn._unsloth_fallback_state["eager"], "it should have run out at all"


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


@pytest.mark.skipif(
    U._saved_tensor_hook_accessor() is None,
    reason = "torch < 2.8 cannot report whether the hooks are still installed",
)
def test_settlement_retries_until_the_generator_is_actually_collected():
    """A caller retrying from inside `except ... as exc` still roots the
    traceback, so the abandoned generator survives the collection and its hooks
    stay installed. Clearing the pending flag regardless meant the next boundary
    never tried again."""
    import gc

    def boom(x):
        raise RuntimeError("give up")

    saved = U._RAISED_INSIDE_CHECKPOINT
    try:
        try:
            checkpoint(boom, torch.randn(4, 4, requires_grad = True),
                       use_reentrant = False)
        except RuntimeError as exc:
            cycle = {"exc": exc}                # a cycle, as raising through
            cycle["self"] = cycle               # compiled frames leaves
            U._RAISED_INSIDE_CHECKPOINT = True
            # The step boundary the caller reaches before retrying: the
            # traceback is still live, so no collection can finalise anything.
            U.apply_pending_eager_fallbacks()
            assert U._in_non_reentrant_checkpoint() is True, "hooks already gone"

        del cycle                               # garbage now, but only to gc
        assert U._in_non_reentrant_checkpoint() is True, "refcounting freed it"

        U.apply_pending_eager_fallbacks()       # the next boundary must retry
        assert U._in_non_reentrant_checkpoint() is False, \
            "the abandoned generator was never settled"
    finally:
        U._RAISED_INSIDE_CHECKPOINT = saved
        U._CHECKPOINT_SETTLE_ATTEMPTS = 0
        gc.collect()


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


def test_the_packed_marker_does_not_survive_the_step():
    """`_PACKED_COMPILED_IN_CHECKPOINT` says "this step packed compiled". Only
    `_restore_recompile_limits` cleared it, which an ordinary successful step
    never reaches, so once set it stayed true for the rest of the run and a
    later call nowhere near a checkpoint had `_give_up` re-raise the compiler
    error instead of taking the safe eager fallback."""
    U._PACKED_COMPILED_IN_CHECKPOINT = True
    U.apply_pending_eager_fallbacks()          # nothing pending: still a boundary
    assert U._PACKED_COMPILED_IN_CHECKPOINT is False


def _sequential_walk(segments, use_reentrant):
    """What the frame walk answers inside each module of a sequential run."""
    from torch.utils.checkpoint import checkpoint_sequential

    seen = []

    class Probe(nn.Module):
        def forward(self, x):
            seen.append(U._walk_for_checkpoint_frame())
            return x * 1.0

    modules = [Probe() for _ in range(4)]
    x = torch.randn(2, 2, requires_grad = True)
    checkpoint_sequential(modules, segments, x, use_reentrant = use_reentrant)
    return seen


def test_a_reentrant_sequential_is_not_read_as_a_non_reentrant_region():
    """`checkpoint_sequential` keeps its own frame in `torch.utils.checkpoint`,
    and its per-segment closure is called `forward` in that same file. Counting
    the closure as a `CheckpointFunction` frame left the outer frame hitting the
    pack/recompute catch-all, so a fully reentrant sequence answered
    "non-reentrant region open" and `_give_up` re-raised instead of falling
    back."""
    assert _sequential_walk(2, True) == [False] * 4


def test_the_last_sequential_segment_is_outside_every_region():
    """The final segment is run directly, not through `checkpoint`, so nothing
    is packed while it executes and eager is safe there even when the earlier
    segments are non-reentrant."""
    assert _sequential_walk(2, False) == [True, True, False, False]
