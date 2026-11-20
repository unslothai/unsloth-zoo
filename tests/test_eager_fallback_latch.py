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

"""fullgraph=True makes recompile-cache exhaustion raise; the fallback must latch and be announceable, since a mode flip
between pack and recompute aborts backward; force_eager_fallback() closes the mixed step; non-latching wrappers do
not fix it"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unsloth_zoo.temporary_patches import utils as U  # noqa: E402


class Boom(Exception):
    pass


@pytest.fixture(autouse=True)
def _isolate_registry(monkeypatch):
    """Gives each test its own wrapper registry; the module-global registry otherwise leaks wrappers between tests,
    making force_eager_fallback()'s count order-dependent."""
    monkeypatch.setattr(U, "_EAGER_FALLBACK_WRAPPERS", [])
    monkeypatch.setattr(U, "_recompile_limit_errors", lambda: (Boom,))
    monkeypatch.setattr(U, "_disabled_hook_graph_break_error", lambda: ())
    # Give-up is keyed by label so it survives rebuilds, and tests all reuse M.forward / M.f, so these label sets are reset per test.
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
    """After the first failure the compiler is never consulted again, so every later pack and recompute agree on the mode."""
    compiled, eager, calls = _pair(fail_after=2)
    w = U._fall_back_to_eager_on_recompile_limit(compiled, eager, "M.forward")
    modes = [w(i)[0] for i in range(6)]
    assert modes == ["compiled", "compiled", "eager", "eager", "eager", "eager"]
    # 4 compiled attempts: two that worked, the one that raised, and the single bumped retry before giving up; never compiled again after that
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
    """With no recompile-limit or backend compile exceptions the compiled callable must be returned unwrapped; both
    tuples are stubbed, since the wrapper also catches Inductor codegen failures"""
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


def test_force_does_nothing_when_nothing_ever_fell_back():
    """force_eager_fallback() returning 0 means no mode flip caused the checkpoint assertion, so the caller must re-raise instead of retrying."""
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
    """force_eager_fallback() counts wrappers, not changes: the wrapper that caused the trouble has already latched, so a changes count can read 0 when it matters."""
    compiled, eager, _ = _pair(fail_after=0)
    # Keep the wrapper bound: the registry holds wrappers weakly, so an unbound one is collected before force_eager_fallback() can see it.
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
    """GRPO's accumulate_chunk is built inside the forward, so its deferring wrapper is collected before backward dies;
    the pending label is the only evidence, as live wrappers alone returned 0."""
    U._PENDING_EAGER_LABELS.add("chunk.f")

    assert U.force_eager_fallback() > 0, "the deferral was not seen"
    assert "chunk.f" in U._LATCHED_EAGER_LABELS
    assert not U._PENDING_EAGER_LABELS, "the deferral was left unsettled"

    compiled, eager, calls = _pair(fail_after=100)
    rebuilt = U._fall_back_to_eager_on_recompile_limit(compiled, eager, "chunk.f")
    assert rebuilt(1)[0] == "eager", "the rebuilt wrapper compiled again"
    assert calls["compiled"] == 0


def test_force_sees_a_latch_whose_wrapper_is_already_gone():
    """Uses the real give-up, not a direct write to _LATCHED_EAGER_LABELS (permanent, holds earlier models' labels, so
    not evidence alone); give-up writes it and _RECENT_EAGER_LABELS (this step)."""
    compiled, eager, _ = _pair(fail_after = 0)
    w = U._fall_back_to_eager_on_recompile_limit(compiled, eager, "chunk.f")
    w(1)
    del w
    import gc; gc.collect()

    assert "chunk.f" in U._LATCHED_EAGER_LABELS
    assert U.force_eager_fallback() > 0


def test_the_registry_does_not_keep_dead_wrappers_alive():
    """The registry holds wrappers weakly, so a discarded patched model is not counted as a live compiled forward and cannot inflate the count."""
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
    """unsloth calls this from its activation-checkpoint retry path via the unsloth_zoo.temporary_patches facade, not the submodule."""
    from unsloth_zoo.temporary_patches import (
        eager_fallback_state,
        force_eager_fallback,
    )
    assert force_eager_fallback is U.force_eager_fallback
    assert eager_fallback_state is U.eager_fallback_state


def test_the_recovery_hook_is_declared_public():
    """from ... import * honours __all__, so a name missing from it is not public however it is re-exported."""
    assert "force_eager_fallback" in U.__all__
    assert "eager_fallback_state" in U.__all__


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_a_previous_models_latch_is_not_evidence_for_the_next():
    """_LATCHED_EAGER_LABELS is permanent, so it cannot mean fell back just now: an earlier model's labels answered for
    the next, faking a mode flip that caused a retry instead of a re-raise."""
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
    """A give-up outside a checkpoint records its label with no pending entry, so later boundaries took the early return
    and the label never expired; a later real checkpoint failure was read as a mode flip."""
    U._RECENT_EAGER_LABELS.add("gave_up_outside_a_checkpoint.forward")
    # Nothing pending: no wrapper was registered, so this is the early return.
    assert U.force_eager_fallback() == 0
    assert U._RECENT_EAGER_LABELS == set(), \
        "the label survived a step boundary it should not have"
    assert U.force_eager_fallback(only_if_already_triggered = True) == 0


def test_the_boundary_does_not_discard_a_pending_flip():
    """Clearing on the empty path must not discard a pending flip: the recent labels are what _LATCHED_EAGER_LABELS is updated from."""
    U._PENDING_EAGER_LABELS.add("real.forward")
    U._RECENT_EAGER_LABELS.add("real.forward")
    U.force_eager_fallback()
    assert "real.forward" in U._LATCHED_EAGER_LABELS


def test_the_wrapper_registry_does_not_grow_without_bound():
    """GRPO re-wraps accumulate_chunk in every backward; dead weak references piled up one per step and every scan
    walked them, so the registry must not grow without bound."""
    def build(n):
        for i in range(n):
            compiled, eager, _ = _pair(fail_after = 100)
            U._fall_back_to_eager_on_recompile_limit(compiled, eager, f"chunk{i}.f")

    # Control: with _EAGER_FALLBACK_PRUNE_AT out of reach every dead reference stays, reproducing the unbounded growth that pruning prevents in long GRPO runs.
    U._EAGER_FALLBACK_PRUNE_AT = 10 ** 9
    build(500)
    assert len(U._EAGER_FALLBACK_WRAPPERS) == 500

    U._EAGER_FALLBACK_WRAPPERS.clear()
    U._EAGER_FALLBACK_PRUNE_AT = 64
    build(4000)
    assert len(U._EAGER_FALLBACK_WRAPPERS) < 200, len(U._EAGER_FALLBACK_WRAPPERS)
