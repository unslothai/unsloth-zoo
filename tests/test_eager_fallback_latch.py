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

"""The eager fallback must latch, and it must be announceable.

`fullgraph = True` turns Dynamo's recompilation-cache exhaustion into a raise
instead of a fallback, so a performance problem becomes a hard training
failure. `_fall_back_to_eager_on_recompile_limit` catches that and runs the
eager original instead.

The subtlety is non-reentrant activation checkpointing: it recomputes each
packed forward during backward and asserts the two saved the same
intermediates, so a mode change between them aborts the backward. Latching
makes every call after the first failure eager, leaving one mixed step, which
`force_eager_fallback()` lets unsloth close. A non-latching wrapper does not
fix this -- see the note on `test_the_fallback_latches` in
test_recompile_limit_fallback.py before changing it back.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unsloth_zoo.temporary_patches import utils as U  # noqa: E402


class Boom(Exception):
    pass


@pytest.fixture(autouse=True)
def _isolate_registry(monkeypatch):
    """Each test gets its own wrapper registry.

    Without this the registry is module-global and tests see each other's
    wrappers, so `force_eager_fallback()`'s count depends on execution order.
    """
    monkeypatch.setattr(U, "_EAGER_FALLBACK_WRAPPERS", [])
    monkeypatch.setattr(U, "_recompile_limit_errors", lambda: (Boom,))
    monkeypatch.setattr(U, "_disabled_hook_graph_break_error", lambda: ())
    # The give-up decision is kept by LABEL, so it survives rebuilds -- and
    # tests, which all reuse "M.forward" / "M.f".
    monkeypatch.setattr(U, "_LATCHED_EAGER_LABELS", set())
    monkeypatch.setattr(U, "_PENDING_EAGER_LABELS", set())
    monkeypatch.setattr(U, "_RECENT_EAGER_LABELS", set())
    monkeypatch.setattr(U, "_EAGER_FALLBACK_PRUNE_AT", 64)


def _pair(fail_after=0):
    """A compiled func that raises from call `fail_after` on, and its eager twin."""
    calls = {"compiled": 0, "eager": 0}

    def compiled(x):
        calls["compiled"] += 1
        if calls["compiled"] > fail_after:
            raise Boom("recompile_limit reached with fullgraph=True")
        return ("compiled", x)

    def eager(x):
        calls["eager"] += 1
        return ("eager", x)

    return compiled, eager, calls


# ---- the latch -----------------------------------------------------------

def test_a_healthy_compiled_function_is_never_wrapped_out_of_the_way():
    compiled, eager, calls = _pair(fail_after=100)
    w = U._fall_back_to_eager_on_recompile_limit(compiled, eager, "M.forward")
    assert [w(i)[0] for i in range(5)] == ["compiled"] * 5
    assert calls["eager"] == 0


def test_it_falls_back_on_cache_exhaustion():
    compiled, eager, calls = _pair(fail_after=0)
    w = U._fall_back_to_eager_on_recompile_limit(compiled, eager, "M.forward")
    assert w(1) == ("eager", 1)


def test_the_fallback_latches():
    """The whole point. After the first failure the compiler is not consulted
    again, so every later pack and recompute agree."""
    compiled, eager, calls = _pair(fail_after=2)
    w = U._fall_back_to_eager_on_recompile_limit(compiled, eager, "M.forward")
    modes = [w(i)[0] for i in range(6)]
    assert modes == ["compiled", "compiled", "eager", "eager", "eager", "eager"]
    # 4 attempts: two that worked, the one that raised, and the single bumped
    # retry the wrapper makes before giving up. Never again.
    assert calls["compiled"] == 4


def test_the_warning_is_logged_once(monkeypatch):
    seen = []
    monkeypatch.setattr(U.logger, "warning", lambda m: seen.append(m))
    compiled, eager, _ = _pair(fail_after=0)
    w = U._fall_back_to_eager_on_recompile_limit(compiled, eager, "M.forward")
    for _ in range(4):
        w(1)
    assert len(seen) == 1
    assert "M.forward" in seen[0]


def test_an_unrelated_exception_still_propagates():
    def compiled(x):
        raise ValueError("a real bug")

    w = U._fall_back_to_eager_on_recompile_limit(compiled, lambda x: x, "M.f")
    with pytest.raises(ValueError):
        w(1)


def test_no_wrapper_at_all_when_torch_has_no_such_errors(monkeypatch):
    """On a torch with none of these exceptions there is nothing to catch, and
    the compiled callable must be returned untouched rather than wrapped.

    `_backend_compile_errors` is stubbed too: the wrapper also catches Inductor
    codegen failures now, so leaving that tuple populated gives it a real reason
    to exist and the assertion below would be testing the wrong thing."""
    monkeypatch.setattr(U, "_recompile_limit_errors", lambda: ())
    monkeypatch.setattr(U, "_backend_compile_errors", lambda: ())
    compiled, eager, _ = _pair()
    assert U._fall_back_to_eager_on_recompile_limit(
        compiled, eager, "M.f") is compiled


def test_the_compiled_callable_stays_reachable():
    compiled, eager, _ = _pair()
    w = U._fall_back_to_eager_on_recompile_limit(compiled, eager, "M.f")
    assert w._unsloth_compiled_func is compiled
    assert w.__wrapped__ is eager


# ---- the graph-break arm is unchanged ------------------------------------

def test_our_own_disabled_hook_falls_back_and_latches(monkeypatch):
    class GraphBreak(Exception):
        pass

    monkeypatch.setattr(U, "_disabled_hook_graph_break_error",
                        lambda: (GraphBreak,))
    calls = {"n": 0}

    def compiled(x):
        calls["n"] += 1
        raise GraphBreak("Skip calling `torch.compiler.disable()`d function")

    w = U._fall_back_to_eager_on_recompile_limit(compiled, lambda x: "eager",
                                                 "M.f")
    assert [w(1) for _ in range(3)] == ["eager"] * 3
    assert calls["n"] == 1, "latched, so the compiler is consulted once"


def test_someone_elses_graph_break_still_raises(monkeypatch):
    class GraphBreak(Exception):
        pass

    monkeypatch.setattr(U, "_disabled_hook_graph_break_error",
                        lambda: (GraphBreak,))

    def compiled(x):
        raise GraphBreak("Unsupported: some genuinely unsupported construct")

    w = U._fall_back_to_eager_on_recompile_limit(compiled, lambda x: "eager",
                                                 "M.f")
    with pytest.raises(GraphBreak):
        w(1)


# ---- force_eager_fallback ------------------------------------------------

def test_force_does_nothing_when_nothing_ever_fell_back():
    """The honest default. A caller getting 0 back knows the activation
    checkpoint assertion it caught was not caused by a mode flip, and must
    re-raise rather than retry."""
    compiled, eager, _ = _pair(fail_after=100)
    w = U._fall_back_to_eager_on_recompile_limit(compiled, eager, "M.f")
    w(1)
    assert U.force_eager_fallback() == 0
    assert w(1)[0] == "compiled", "must not switch a healthy function"


def test_force_switches_the_others_once_one_has_fallen_back():
    a_c, a_e, a_calls = _pair(fail_after=0)     # this one exhausts its cache
    b_c, b_e, b_calls = _pair(fail_after=100)   # this one is still fine
    a = U._fall_back_to_eager_on_recompile_limit(a_c, a_e, "A.f")
    b = U._fall_back_to_eager_on_recompile_limit(b_c, b_e, "B.f")
    a(1)
    assert U.force_eager_fallback() == 2
    assert b(1)[0] == "eager"
    assert b_calls["compiled"] == 0


def test_force_counts_wrappers_not_changes():
    """Deliberate: by the time unsloth calls this, the wrapper that caused the
    trouble has already latched itself, so "how many changed" can be 0 in
    exactly the case that matters and would read as "nothing happened"."""
    compiled, eager, _ = _pair(fail_after=0)
    # Bound, not called inline: the registry holds wrappers weakly, so an
    # unbound one is collected before force_eager_fallback() can see it.
    w = U._fall_back_to_eager_on_recompile_limit(compiled, eager, "A.f")
    w(1)
    assert U.force_eager_fallback() == 1


def test_force_ignores_the_guard_when_asked():
    compiled, eager, _ = _pair(fail_after=100)
    w = U._fall_back_to_eager_on_recompile_limit(compiled, eager, "M.f")
    assert U.force_eager_fallback(only_if_already_triggered=False) == 1
    assert w(1)[0] == "eager"


def test_force_is_idempotent():
    compiled, eager, _ = _pair(fail_after=0)
    w = U._fall_back_to_eager_on_recompile_limit(compiled, eager, "A.f")
    w(1)
    assert U.force_eager_fallback() == U.force_eager_fallback() == 1


def test_force_settles_a_deferral_whose_wrapper_is_already_gone():
    """GRPO's `accumulate_chunk` is built inside the forward, so by the time the
    backward dies the wrapper that deferred is collected and the pending label
    is the only evidence. Asking the live wrappers alone returned 0, the caller
    re-raised the failure this exists to retry past, and the rebuilt wrapper
    compiled again."""
    U._PENDING_EAGER_LABELS.add("chunk.f")

    assert U.force_eager_fallback() > 0, "the deferral was not seen"
    assert "chunk.f" in U._LATCHED_EAGER_LABELS
    assert not U._PENDING_EAGER_LABELS, "the deferral was left unsettled"

    compiled, eager, calls = _pair(fail_after=100)
    rebuilt = U._fall_back_to_eager_on_recompile_limit(compiled, eager, "chunk.f")
    assert rebuilt(1)[0] == "eager", "the rebuilt wrapper compiled again"
    assert calls["compiled"] == 0


def test_force_sees_a_latch_whose_wrapper_is_already_gone():
    """Same for a label that gave up outright rather than deferring.

    Through the real give-up, not by writing `_LATCHED_EAGER_LABELS`: that set
    is permanent by design, holds a previous model's labels, and cannot be the
    evidence alone. `_RECENT_EAGER_LABELS` says "in this step"; give-up writes
    both."""
    compiled, eager, _ = _pair(fail_after = 0)
    w = U._fall_back_to_eager_on_recompile_limit(compiled, eager, "chunk.f")
    w(1)
    del w
    import gc; gc.collect()

    assert "chunk.f" in U._LATCHED_EAGER_LABELS
    assert U.force_eager_fallback() > 0


def test_the_registry_does_not_keep_dead_wrappers_alive():
    """Weak, so a model that was patched and thrown away is not reported as a
    live compiled forward and cannot inflate the count."""
    import gc
    compiled, eager, _ = _pair()
    w = U._fall_back_to_eager_on_recompile_limit(compiled, eager, "gone.f")
    assert "gone.f" in U.eager_fallback_state()
    del w
    gc.collect()
    assert "gone.f" not in U.eager_fallback_state()


def test_state_reports_each_label():
    a_c, a_e, _ = _pair(fail_after=0)
    b_c, b_e, _ = _pair(fail_after=100)
    a = U._fall_back_to_eager_on_recompile_limit(a_c, a_e, "A.f")
    b = U._fall_back_to_eager_on_recompile_limit(b_c, b_e, "B.f")
    a(1)
    b(1)
    assert U.eager_fallback_state() == {"A.f": True, "B.f": False}


def test_the_recovery_hook_is_importable_from_the_package():
    """unsloth calls this from its activation-checkpoint retry path, through
    the `unsloth_zoo.temporary_patches` facade rather than the submodule."""
    from unsloth_zoo.temporary_patches import (
        eager_fallback_state,
        force_eager_fallback,
    )
    assert force_eager_fallback is U.force_eager_fallback
    assert eager_fallback_state is U.eager_fallback_state


def test_the_recovery_hook_is_declared_public():
    """`from ... import *` honours __all__, so a name missing from it is not
    public however it is re-exported."""
    assert "force_eager_fallback" in U.__all__
    assert "eager_fallback_state" in U.__all__


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_a_previous_models_latch_is_not_evidence_for_the_next():
    """`_LATCHED_EAGER_LABELS` is permanent on purpose, so it cannot double as
    "something fell back just now": train two models in one process and the
    first one's labels answered for the second, so `force_eager_fallback`
    reported a compile-mode flip where there was none -- the caller's cue to
    retry rather than re-raise a genuine checkpoint failure."""
    U._LATCHED_EAGER_LABELS.add("model_a.SomeNorm.forward")   # discarded model
    assert U.force_eager_fallback(only_if_already_triggered = True) == 0

    U._RECENT_EAGER_LABELS.add("model_b.SomeNorm.forward")    # this step
    assert U.force_eager_fallback(only_if_already_triggered = True) > 0


def test_the_settle_clears_the_recent_labels():
    """One flip is evidence for its own step, not for every step after it."""
    U._RECENT_EAGER_LABELS.add("x.forward")
    U.force_eager_fallback(only_if_already_triggered = True)
    assert U._RECENT_EAGER_LABELS == set()
    assert U.force_eager_fallback(only_if_already_triggered = True) == 0


def test_a_boundary_with_nothing_pending_still_expires_the_recent_labels():
    """A wrapper giving up OUTSIDE a checkpoint records its label with no
    pending entry, so every later boundary took the early return and the label
    never expired -- a genuine checkpoint failure in a later step then read it
    as a compile-mode flip and asked for a retry."""
    U._RECENT_EAGER_LABELS.add("gave_up_outside_a_checkpoint.forward")
    # Nothing pending: no wrapper was registered, so this is the early return.
    assert U.force_eager_fallback() == 0
    assert U._RECENT_EAGER_LABELS == set(), \
        "the label survived a step boundary it should not have"
    assert U.force_eager_fallback(only_if_already_triggered = True) == 0


def test_the_boundary_does_not_discard_a_pending_flip():
    """Clearing on the empty path must not reach the path that has work: the
    labels are what `_LATCHED_EAGER_LABELS` is updated from."""
    U._PENDING_EAGER_LABELS.add("real.forward")
    U._RECENT_EAGER_LABELS.add("real.forward")
    U.force_eager_fallback()
    assert "real.forward" in U._LATCHED_EAGER_LABELS


def test_the_wrapper_registry_does_not_grow_without_bound():
    """GRPO re-wraps `accumulate_chunk` inside every backward, so one dead weak
    reference per step accumulated forever and every scan walked them."""
    def build(n):
        for i in range(n):
            compiled, eager, _ = _pair(fail_after = 100)
            U._fall_back_to_eager_on_recompile_limit(compiled, eager, f"chunk{i}.f")

    # Control: with the threshold out of reach every dead reference stays,
    # exactly what a long GRPO run used to accumulate.
    U._EAGER_FALLBACK_PRUNE_AT = 10 ** 9
    build(500)
    assert len(U._EAGER_FALLBACK_WRAPPERS) == 500

    U._EAGER_FALLBACK_WRAPPERS.clear()
    U._EAGER_FALLBACK_PRUNE_AT = 64
    build(4000)
    assert len(U._EAGER_FALLBACK_WRAPPERS) < 200, len(U._EAGER_FALLBACK_WRAPPERS)
