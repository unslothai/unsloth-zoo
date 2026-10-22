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

"""`fullgraph = True` regions when the user has switched the compiler off.

`TORCH_COMPILE_DISABLE=1` (and `torch._dynamo.config.disable = True`, which is
the same flag by hand) takes torch.compile out of the picture while debugging.
Up to torch 2.13 a `fullgraph = True` region then ran eagerly. From torch 2.14
torch runs the body and raises afterwards, out of its post-call bookkeeping:

    RuntimeError: torch.compile with fullgraph=True found no compiled frames.
    Skipped frames: ... Dynamo tracing is disabled

Every `torch_compile_with_fallback` region inherits that, and there are many:
`rl_replacements.py` wraps the GRPO loss accumulator in one, and `compiler.py`
writes the decorator into every generated `unsloth_compiled_cache` module. So a
user who disables torch.compile on torch 2.14 loses GRPO training at the first
step, having asked only to turn the compiler off.

The flag is read before the call, never recovered from after one: torch has
already run the body by the time it raises, so recovering would run it twice.
And it is read at the FIRST call, not at decoration and not on every call. At
decoration is too early, since a flag set only while modules import would stick
for the run. Every call is too late to be consistent: a checkpoint's pack and
its recompute must agree, and by the time a recompute runs the forward that
packed it is gone.

`TORCHDYNAMO_DISABLE=1` is a different switch, handled inside
`torch._dynamo.optimize`, which returns the undecorated function. Nothing here
has to deal with it, and the tests below pin that too.
"""

import pytest
import torch

from unsloth_zoo.temporary_patches import utils as u


@pytest.fixture(autouse = True)
def clear_checkpoint_markers():
    # A compiled call under a checkpoint sets these process-wide, and only a
    # step boundary clears them. Left set, they make the recompile-limit arm
    # re-raise rather than fall back, in this file's tests and in any that run
    # after it.
    yield
    u._PACKED_COMPILED_IN_CHECKPOINT = False
    u._COMPILED_OK_LABELS.clear()


@pytest.fixture
def dynamo_off():
    prev = torch._dynamo.config.disable
    torch._dynamo.config.disable = True
    try:
        yield
    finally:
        torch._dynamo.config.disable = prev


def _add_one(x):
    return x + 1


def _counting_fn():
    """A fresh code object per test: torch skip marks attach to the code."""
    calls = []
    src = {}
    exec("def f(x):\n    calls.append(1)\n    return x + 1\n", {"calls": calls}, src)
    return src["f"], calls


def _checkpointed_wrapper():
    """One wrapper plus the list of its body executions."""
    calls = []
    src = {}
    exec("def block(x, w):\n"
         "    calls.append(1)\n"
         "    return torch.nn.functional.layer_norm(x @ w, (w.shape[-1],)).sin()\n",
         {"calls": calls, "torch": torch}, src)
    return u.torch_compile_with_fallback(fullgraph = True)(src["block"]), calls


def _pack(wrapped):
    """One non-reentrant checkpointed forward. Returns (output, weight)."""
    from torch.utils.checkpoint import checkpoint
    torch.manual_seed(0)
    x = torch.randn(4, 8, requires_grad = True)
    w = torch.randn(8, 8, requires_grad = True)
    return checkpoint(wrapped, x, w, use_reentrant = False), w


# --------------------------------------------------------------- the dispatch


def test_decorated_before_disabling_runs_eagerly_and_exactly_once():
    fn, calls = _counting_fn()
    wrapped = u.torch_compile_with_fallback(fullgraph = True)(fn)
    prev = torch._dynamo.config.disable
    torch._dynamo.config.disable = True
    try:
        del calls[:]
        got = wrapped(torch.zeros(3))
    finally:
        torch._dynamo.config.disable = prev
    assert torch.equal(got, torch.ones(3))
    # 2 here is the torch 2.14 double-run: body, then raise, then eager retry.
    assert len(calls) == 1


def test_decorated_while_disabled_runs_eagerly_and_exactly_once(dynamo_off):
    fn, calls = _counting_fn()
    wrapped = u.torch_compile_with_fallback(fullgraph = True)(fn)
    del calls[:]
    got = wrapped(torch.zeros(3))
    assert torch.equal(got, torch.ones(3))
    assert len(calls) == 1


def test_a_flag_held_only_over_decoration_does_not_stick():
    """Disabled while the module imports, restored before the first call.

    Deciding at decoration time pinned such a region to eager for the whole
    run, which is what reading the flag at the first call instead avoids.
    """
    fn, _ = _counting_fn()
    prev = torch._dynamo.config.disable
    torch._dynamo.config.disable = True
    try:
        wrapped = u.torch_compile_with_fallback(fullgraph = True)(fn)
    finally:
        torch._dynamo.config.disable = prev
    assert torch.equal(wrapped(torch.zeros(3)), torch.ones(3))
    assert wrapped._unsloth_fallback_state["compiler_off"] is False
    assert wrapped._unsloth_fallback_state["eager"] is False


def test_fullgraph_false_is_untouched(dynamo_off):
    wrapped = u.torch_compile_with_fallback(fullgraph = False)(_add_one)
    assert torch.equal(wrapped(torch.zeros(3)), torch.ones(3))


def test_a_real_graph_break_still_raises():
    # The eager dispatch is keyed on the flag, not on an error message, so a
    # genuine fullgraph failure is never absorbed.
    def breaks(x):
        print("graph break")
        return x + 1
    wrapped = u.torch_compile_with_fallback(fullgraph = True)(breaks)
    with pytest.raises(Exception):
        wrapped(torch.zeros(3))


@pytest.mark.skipif(
    not hasattr(getattr(torch, "compiler", None), "set_stance"),
    reason = "torch.compiler.set_stance is new in torch 2.6; the floor is 2.4",
)
def test_the_force_eager_stance_is_left_to_torch():
    # torch only raises "found no compiled frames" while the stance is
    # "default" (torch/_dynamo/eval_frame.py), so force_eager needs nothing.
    fn, calls = _counting_fn()
    wrapped = u.torch_compile_with_fallback(fullgraph = True)(fn)
    torch.compiler.set_stance("force_eager")
    try:
        del calls[:]
        got = wrapped(torch.zeros(3))
    finally:
        torch.compiler.set_stance("default")
    assert torch.equal(got, torch.ones(3))
    assert len(calls) == 1


def test_dynamo_tracing_disabled_reads_the_live_config():
    # The reader stays live; it is the wrapper that takes one snapshot.
    prev = torch._dynamo.config.disable
    try:
        torch._dynamo.config.disable = True
        assert u.dynamo_tracing_disabled() is True
        torch._dynamo.config.disable = False
        assert u.dynamo_tracing_disabled() is False
    finally:
        torch._dynamo.config.disable = prev


# ------------------------------------------------- checkpointing consistency
#
# Non-reentrant checkpointing recomputes the forward during backward and
# compares what each pass saved, so a pack and its recompute running in
# different modes raises CheckpointError and ends the step. Each of these was
# a live failure of a per-call read of the flag.


@pytest.mark.parametrize("disabled", [True, False])
def test_a_checkpointed_step_runs_with_the_compiler_either_way(disabled):
    prev = torch._dynamo.config.disable
    torch._dynamo.config.disable = disabled
    try:
        wrapped, calls = _checkpointed_wrapper()
        del calls[:]
        out, w = _pack(wrapped)
        out.sum().backward()
    finally:
        torch._dynamo.config.disable = prev
    assert w.grad is not None
    if disabled:
        # The forward and its recompute, each exactly once. Compiled, the
        # recompute replays a graph and never re-enters the Python body.
        assert len(calls) == 2


def test_the_snapshot_survives_a_flip_between_a_forward_and_its_backward():
    """Re-enabling mid-step must not recompute an eager pack compiled.

    Only this direction is ours. Disabling mid-step over a COMPILED pack is
    torch's own: it consults `config.disable` whenever it converts a frame, so
    the recompute's variant is skipped and runs eager whatever we do. That case
    raises `CheckpointError` identically on `main`.
    """
    prev = torch._dynamo.config.disable
    torch._dynamo.config.disable = True
    try:
        wrapped, _ = _checkpointed_wrapper()
        out, w = _pack(wrapped)
        torch._dynamo.config.disable = False
        out.sum().backward()
    finally:
        torch._dynamo.config.disable = prev
    assert w.grad is not None


def test_two_outstanding_packs_of_one_wrapper_agree():
    """Two live packs of one wrapper, with a flip between their forwards.

    A mode held per wrapper and updated per call described only the newest
    pack, so the older one was recomputed in the other mode. Fails on `main`
    as well, where the second forward is what changes mode.
    """
    prev = torch._dynamo.config.disable
    torch._dynamo.config.disable = True
    try:
        wrapped, _ = _checkpointed_wrapper()
        first, w1 = _pack(wrapped)
        torch._dynamo.config.disable = False
        second, w2 = _pack(wrapped)
        (first.sum() + second.sum()).backward()
    finally:
        torch._dynamo.config.disable = prev
    assert w1.grad is not None and w2.grad is not None


def test_a_completed_checkpointed_step_leaves_no_state_to_expire():
    """The decision is the wrapper's own, and it does not drift after a step.

    Nothing in this package calls `apply_pending_eager_fallbacks`, so a mode
    that had to be expired at a step boundary never would be. Checked away from
    a checkpoint: whether torch can still run a COMPILED region once the flag
    is set is torch's own frame-conversion behaviour, unchanged from `main`.
    """
    prev = torch._dynamo.config.disable
    torch._dynamo.config.disable = False
    try:
        wrapped, _ = _checkpointed_wrapper()
        out, w = _pack(wrapped)
        out.sum().backward()
        assert w.grad is not None
        torch._dynamo.config.disable = True
        assert wrapped._unsloth_fallback_state["compiler_off"] is False
        assert wrapped._unsloth_fallback_state["eager"] is False
    finally:
        torch._dynamo.config.disable = prev


def test_a_nested_wrapper_is_traceable_after_running_eager():
    # An outer fullgraph trace must not reach a checkpoint-hook probe: it is a
    # pybind builtin Dynamo refuses to enter, which is fatal under fullgraph.
    def inner(x):
        return x * 2
    wrapped = u.torch_compile_with_fallback(fullgraph = True)(inner)
    prev = torch._dynamo.config.disable
    torch._dynamo.config.disable = True
    try:
        wrapped(torch.zeros(3))
    finally:
        torch._dynamo.config.disable = prev

    def outer(x):
        return wrapped(x) + 1
    assert torch.equal(
        torch.compile(outer, fullgraph = True)(torch.zeros(3)), torch.ones(3),
    )
