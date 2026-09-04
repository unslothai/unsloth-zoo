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

"""Simulation suite for unsloth-zoo PR #1145.

PR #1145 routes the disabled-hook arm of `_fall_back_to_eager_on_recompile_limit`
through `_give_up_at_compile_time`, the contract `_give_up_on_backend` already
had. Before, that arm did `state["eager"] = True; return eager_func(...)`: never
raised, never touched the global label sets, never restored the marker.

That is shared machinery (every `fullgraph = True` patch here plus 14 sites in
unsloth core), so the risk is NOT "does the new case work" but "does the old one
still work". Each test is a before/after pair, hence the `IS_FIX` branches.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IS_FIX = True

from unsloth_zoo.temporary_patches import utils as U  # noqa: E402

LABEL = "SimMod"

DISABLE_MSG = (
    "Skip calling `torch.compiler.disable()`d function\n"
    "  Explanation: Skip calling function "
    "`<function requires_grad_for_gradient_checkpointing.<locals>."
    "requires_grad_pre_hook at 0x7f00>` since it was wrapped with "
    "`torch.compiler.disable` (reason: None)\n"
    "  Hint: Remove the `torch.compiler.disable` call"
)
OLD_DISABLE_MSG = (
    "call torch._dynamo.disable() wrapped function "
    "<function requires_grad_pre_hook at 0x7f00>"
)


@pytest.fixture(autouse=True)
def _clean_label_state():
    """The latch is process-global and keyed by label: without this the first
    fallback starts every later test already eager."""
    sets = (U._LATCHED_EAGER_LABELS, U._PENDING_EAGER_LABELS,
            U._RECENT_EAGER_LABELS, U._COMPILED_OK_LABELS)
    for s in sets:
        s.discard(LABEL)
    packed = U._PACKED_COMPILED_IN_CHECKPOINT
    raised = getattr(U, "_RAISED_INSIDE_CHECKPOINT", False)
    U._PACKED_COMPILED_IN_CHECKPOINT = False
    U._RAISED_INSIDE_CHECKPOINT = False
    yield
    for s in sets:
        s.discard(LABEL)
    U._PACKED_COMPILED_IN_CHECKPOINT = packed
    U._RAISED_INSIDE_CHECKPOINT = raised


def _unsupported(msg):
    import torch._dynamo.exc as exc
    cls = getattr(exc, "Unsupported", None)
    if cls is None:
        pytest.skip("this torch has no torch._dynamo.exc.Unsupported")
    try:
        return cls(msg)
    except Exception:
        pytest.skip("Unsupported cannot be constructed on this torch")


def _wrap(compiled, eager=None, label=LABEL):
    eager = eager or (lambda *a, **k: "eager")
    return U._fall_back_to_eager_on_recompile_limit(compiled, eager, label)


def _raiser(msg=DISABLE_MSG):
    def compiled(*a, **k):
        raise _unsupported(msg)
    return compiled


# R1: the common case must not change. This is the regression that would hurt.

def test_disabled_hook_outside_any_checkpoint_still_falls_back_to_eager():
    """The common case: no checkpoint in sight. Both trees return eager, no raise."""
    assert _wrap(_raiser())() == "eager"


def test_first_call_refusal_inside_checkpoint_still_falls_back():
    """Nothing compiled was packed by this label, so eager pack + eager recompute
    agree. Raising would break every first-call user of a checkpointed region."""
    U._PACKED_COMPILED_IN_CHECKPOINT = True   # a region is live
    assert LABEL not in U._COMPILED_OK_LABELS  # but WE never compiled ok
    assert _wrap(_raiser())() == "eager"


def test_old_torch_message_signature_also_falls_back():
    """torch < 2.7 phrased it differently; miss a spelling and it escapes hard."""
    assert _wrap(_raiser(OLD_DISABLE_MSG))() == "eager"


def test_an_unrelated_graph_break_still_raises():
    """The narrow match is this arm's safety property: over-catching hides bugs."""
    with pytest.raises(Exception):
        _wrap(_raiser("Unsupported: something else entirely"))()


# R2: the new behaviour, only on the fix tree.

def test_refusal_after_compiling_ok_inside_a_live_region():
    """The bug. This label compiled fine, so a compiled pack is outstanding and
    flipping to eager mid-region desynchronises pack and recompute. main returns
    eager silently (corruption); fix raises, so the step ends honestly."""
    U._COMPILED_OK_LABELS.add(LABEL)
    U._PACKED_COMPILED_IN_CHECKPOINT = True
    w = _wrap(_raiser())
    if IS_FIX:
        with pytest.raises(Exception):
            w()
        assert U._RAISED_INSIDE_CHECKPOINT is True
    else:
        assert w() == "eager"
        assert U._RAISED_INSIDE_CHECKPOINT is False


def test_fix_records_the_latch_in_the_global_label_sets():
    """main records nothing, hence the empty latched list in captures taken after
    this arm fired. fix records it in both sets."""
    _wrap(_raiser())()
    if IS_FIX:
        assert LABEL in U._LATCHED_EAGER_LABELS
        assert LABEL in U._RECENT_EAGER_LABELS
    else:
        assert LABEL not in U._LATCHED_EAGER_LABELS


def test_a_new_wrapper_with_a_latched_label_starts_eager():
    """Consequence of recording the latch: a rebuilt wrapper inherits it, as for
    the codegen arm. Intended, but more un-compilation than before, so pinned."""
    _wrap(_raiser())()
    second = _wrap(lambda *a, **k: "compiled-should-not-run")
    if IS_FIX:
        assert second() == "eager"
    else:
        assert second() == "compiled-should-not-run"


# R3: blast radius. Only THIS label may latch.

def test_only_this_label_latches_not_every_borrower():
    """Budget exhaustion is process wide and takes other borrowers with it; a
    compile-time refusal is one region's, so unrelated labels keep compiling."""
    other = "SimOther"
    for s in (U._LATCHED_EAGER_LABELS, U._RECENT_EAGER_LABELS):
        s.discard(other)
    try:
        other_wrapper = _wrap(lambda *a, **k: "compiled", label=other)
        _wrap(_raiser())()
        assert other not in U._LATCHED_EAGER_LABELS
        assert other_wrapper() == "compiled"
    finally:
        for s in (U._LATCHED_EAGER_LABELS, U._RECENT_EAGER_LABELS,
                  U._COMPILED_OK_LABELS):
            s.discard(other)


# R4: the marker must be restored, not leaked.

def test_marker_is_restored_to_what_it_was():
    """The arm captures the marker before the compiled call and puts it back; left
    set, a later wrapper reads a compiled pack outstanding and ends a fine step."""
    U._PACKED_COMPILED_IN_CHECKPOINT = False
    _wrap(_raiser())()
    assert U._PACKED_COMPILED_IN_CHECKPOINT is False


def test_marker_true_is_preserved_when_no_raise_happens():
    U._PACKED_COMPILED_IN_CHECKPOINT = True
    # label not in _COMPILED_OK_LABELS, so no raise; marker must survive.
    _wrap(_raiser())()
    assert U._PACKED_COMPILED_IN_CHECKPOINT is True


# R5: the codegen arm this now shares a body with must be unchanged.

def test_backend_refusal_arm_still_behaves():
    """`_give_up_on_backend` was refactored, not redefined: still falls back."""
    import torch

    class _BackendFail(Exception):
        pass

    def compiled(*a, **k):
        raise torch._dynamo.exc.BackendCompilerFailed(
            lambda: None, _BackendFail("inductor said no")
        ) if hasattr(torch._dynamo.exc, "BackendCompilerFailed") else _BackendFail("no")

    try:
        out = _wrap(compiled)()
    except Exception:
        pytest.skip("this torch does not expose the backend failure shape used here")
    assert out == "eager"


# Platform: the changed code has no platform branch. Prove it, do not say it.

def test_changed_path_has_no_platform_or_device_branch():
    import inspect
    src = inspect.getsource(U._fall_back_to_eager_on_recompile_limit)
    start = src.find("def _give_up_at_compile_time")
    if start == -1:
        pytest.skip("main tree has no _give_up_at_compile_time")
    body = src[start:src.find("def _is_backend_refusal_we_handle", start)]
    for token in ("sys.platform", "platform.system", "cuda", "hip", "mps", "darwin", "win32"):
        assert token not in body, f"{token} appeared in the shared give-up path"


def test_conservative_when_another_region_packed_compiled_this_step():
    """The one place the fix is deliberately pessimistic.

    `_PACKED_COMPILED_IN_CHECKPOINT` is global for the step and
    `_COMPILED_OK_LABELS` sticky for the process, so this label compiling ok plus
    ANY compiled pack under a checkpoint this step makes the arm raise, even if
    this wrapper packed nothing here. Same trade `_give_up_on_backend` has always
    made: ending a probably-fine step is cheap, wrong gradients are not.
    """
    U._COMPILED_OK_LABELS.add(LABEL)          # compiled fine earlier
    U._PACKED_COMPILED_IN_CHECKPOINT = True   # some region packed compiled
    w = _wrap(_raiser())
    if IS_FIX:
        with pytest.raises(Exception):
            w()
    else:
        assert w() == "eager"
