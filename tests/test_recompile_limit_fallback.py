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

"""Exhausting the recompile cache must slow a run down, not end it.

`patch_function(..., fullgraph = True)` compiles with fullgraph, and Dynamo
raises on cache exhaustion under fullgraph instead of falling back:

    FailOnRecompileLimitHit: recompile_limit reached with fullgraph=True

Running out of compilation cache is a performance problem, and turning it into
a hard training failure is strictly worse than being slow.

The fallback is narrow on purpose. A genuine graph break under fullgraph
still raises, because that is a correctness signal about the patch itself.
"""

import contextlib
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unsloth_zoo.temporary_patches.utils import (  # noqa: E402
    _fall_back_to_eager_on_recompile_limit,
    _recompile_limit_errors,
)

import torch._dynamo.exc as _dynamo_exc  # noqa: E402

# Dynamo does not agree on these names across the torch>=2.4,<2.13 pyproject
# declares: 2.4 has none of them, 2.5 has only CacheLimitExceeded, 2.6+ has
# RecompileLimitExceeded and FailOnRecompileLimitHit. Importing one by name
# would kill collection on a supported torch, so look them up the way
# _recompile_limit_errors does.
_LIMIT_ERROR = next(
    (
        e for e in (
            getattr(_dynamo_exc, n, None)
            for n in ("FailOnRecompileLimitHit", "RecompileLimitExceeded", "CacheLimitExceeded")
        )
        if isinstance(e, type) and issubclass(e, BaseException)
    ),
    None,
)

# Dynamo renamed its cache-limit knobs in torch 2.7. 2.4 to 2.6, all inside the
# same declared range, only have the cache_size_limit spelling, and
# `config.patch` looks each key up and raises on one the installed torch does
# not define.
_LIMIT_KEYS = (
    ("recompile_limit", "accumulated_recompile_limit")
    if hasattr(torch._dynamo.config, "recompile_limit")
    else ("cache_size_limit", "accumulated_cache_size_limit")
)

pytestmark = pytest.mark.skipif(
    _LIMIT_ERROR is None,
    reason = "this torch exposes no Dynamo recompile-limit exception",
)


def _pair(compiled_raises = None, calls = None):
    calls = calls if calls is not None else {"c": 0, "e": 0}

    def compiled(x):
        calls["c"] += 1
        if compiled_raises is not None:
            raise compiled_raises
        return x * 2

    def eager(x):
        """Eager docstring."""
        calls["e"] += 1
        return x * 2

    return compiled, eager, calls


# ---- the failure that must stop being fatal -------------------------------

def test_recompile_limit_falls_back_instead_of_raising():
    c, e, calls = _pair(_LIMIT_ERROR("recompile_limit reached"))
    w = _fall_back_to_eager_on_recompile_limit(c, e, "M.forward")
    assert w(3) == 6
    # Two compiled attempts: the one that hit the limit and the one retry the
    # wrapper makes with a raised budget, so a step that is halfway through an
    # activation-checkpoint pack can still finish compiled. Then eager.
    assert calls == {"c": 2, "e": 1}


def test_the_fallback_latches():
    """Do not reverse this assertion. It has been settled twice, on live
    hardware rather than by argument.

    Retrying the compiler per call was tried, so that each activation
    checkpoint pack and its own recompute would agree on a mode. It does not
    work: the pack and the recompute run under different guards -- grad mode
    differs, and the recompute happens inside backward -- so the compiler can
    succeed for one and raise for the other in either direction, and the
    backward still aborts with "Something went unexpectedly wrong in activation
    checkpoint".

    Latching leaves exactly one inconsistent step, the one during which the
    latch flips, and `force_eager_fallback` exists for unsloth to catch that
    single assertion and retry that step.
    """
    c, e, calls = _pair(_LIMIT_ERROR("recompile_limit reached"))
    w = _fall_back_to_eager_on_recompile_limit(c, e, "M.forward")
    for _ in range(5):
        assert w(1) == 2
    # One failing attempt plus the single bumped retry, and nothing after the
    # latch: the compiler is not consulted again.
    assert calls["c"] == 2, "the compiler must not be re-entered after the latch"
    assert calls["e"] == 5


def test_the_latch_is_per_wrapper_not_global():
    """Two separately wrapped functions must not knock each other eager."""
    c1, e1, calls1 = _pair(_LIMIT_ERROR("recompile_limit reached"))
    c2, e2, calls2 = _pair()
    w1 = _fall_back_to_eager_on_recompile_limit(c1, e1, "A.forward")
    w2 = _fall_back_to_eager_on_recompile_limit(c2, e2, "B.forward")
    w1(1); w1(1)
    assert w2(5) == 10
    assert calls2 == {"c": 1, "e": 0}


def test_recompile_limit_exceeded_is_also_caught():
    RecompileLimitExceeded = getattr(_dynamo_exc, "RecompileLimitExceeded", None)
    if RecompileLimitExceeded is None:
        pytest.skip("this torch has no torch._dynamo.exc.RecompileLimitExceeded")
    c, e, _ = _pair(RecompileLimitExceeded("recompile_limit reached"))
    assert _fall_back_to_eager_on_recompile_limit(c, e, "M.forward")(4) == 8


# ---- what must still raise ------------------------------------------------

def test_a_real_graph_break_still_raises():
    from torch._dynamo.exc import Unsupported
    c, e, calls = _pair(Unsupported("call_function BuiltinVariable"))
    w = _fall_back_to_eager_on_recompile_limit(c, e, "M.forward")
    with pytest.raises(Unsupported):
        w(1)
    assert calls["e"] == 0, "a graph break is a correctness signal, not a perf one"


def test_an_ordinary_error_still_raises():
    c, e, calls = _pair(ValueError("something is actually wrong"))
    w = _fall_back_to_eager_on_recompile_limit(c, e, "M.forward")
    with pytest.raises(ValueError):
        w(1)
    assert calls["e"] == 0


# ---- the happy path is untouched ------------------------------------------

def test_compiled_path_is_used_when_nothing_goes_wrong():
    c, e, calls = _pair()
    w = _fall_back_to_eager_on_recompile_limit(c, e, "M.forward")
    assert w(5) == 10
    assert calls == {"c": 1, "e": 0}


# ---- introspection other code depends on ----------------------------------

def test_signature_and_metadata_survive():
    c, e, _ = _pair()
    w = _fall_back_to_eager_on_recompile_limit(c, e, "M.forward")
    import inspect
    assert list(inspect.signature(w).parameters) == ["x"]
    assert w.__doc__ == "Eager docstring."
    assert w.__wrapped__ is e


def test_unwrapping_still_reaches_the_eager_function():
    # patch_function unwraps anything carrying get_compiler_config via
    # __wrapped__, so double-patching must not nest compiles.
    c, e, _ = _pair()
    c.get_compiler_config = lambda: {}
    w = _fall_back_to_eager_on_recompile_limit(c, e, "M.forward")
    assert hasattr(w, "get_compiler_config")
    assert w.__wrapped__ is e


def test_compiled_callable_stays_reachable():
    c, e, _ = _pair()
    w = _fall_back_to_eager_on_recompile_limit(c, e, "M.forward")
    assert w._unsloth_compiled_func is c


def test_no_wrapper_when_torch_exposes_no_such_errors(monkeypatch):
    # Older torch without these names: return the compiled function
    # unchanged rather than guessing at which exception to catch.
    #
    # BOTH sources must be empty now. The wrapper also catches the graph
    # break Unsloth causes itself by registering a `torch.compiler.disable`d
    # gradient-checkpointing hook inside a fullgraph region, so a torch that
    # has `Unsupported` but no recompile-limit errors still needs wrapping.
    import unsloth_zoo.temporary_patches.utils as u
    monkeypatch.setattr(u, "_recompile_limit_errors", lambda: ())
    monkeypatch.setattr(u, "_disabled_hook_graph_break_error", lambda: ())
    c, e, _ = _pair()
    assert u._fall_back_to_eager_on_recompile_limit(c, e, "M.forward") is c


def test_wrapper_is_added_for_the_graph_break_alone(monkeypatch):
    """A torch with `Unsupported` but no recompile-limit names still gets the
    fallback, because the disabled-hook collision can still happen there."""
    import unsloth_zoo.temporary_patches.utils as u
    monkeypatch.setattr(u, "_recompile_limit_errors", lambda: ())
    c, e, _ = _pair()
    if not u._disabled_hook_graph_break_error():
        pytest.skip("this torch has no torch._dynamo.exc.Unsupported")
    assert u._fall_back_to_eager_on_recompile_limit(c, e, "M.forward") is not c


def test_no_version_specific_dynamo_name_is_imported_at_module_level():
    """torch 2.4 has none of the recompile-limit exceptions and 2.5 has only
    CacheLimitExceeded, both inside the torch>=2.4,<2.13 pyproject declares. A
    top-level `from torch._dynamo.exc import <name>` would therefore fail
    collection on a supported torch instead of exercising the compatibility
    path the helper under test exists for."""
    src = Path(__file__).read_text(encoding = "utf-8")
    for name in ("FailOnRecompileLimitHit", "RecompileLimitExceeded", "CacheLimitExceeded"):
        assert f"import {name}" not in src, f"{name} must be looked up with getattr"


def test_the_dynamo_limit_keys_exist_on_this_torch():
    """A hardcoded spelling turns the cache-exhaustion test below into an
    error on torch 2.4 to 2.6 rather than a run of it."""
    for key in _LIMIT_KEYS:
        assert hasattr(torch._dynamo.config, key), key


@pytest.mark.parametrize("message", [
    "cache_size_limit reached",
    "accumulated_cache_size_limit reached",
    "recompile_limit reached",
])
def test_cache_exhaustion_reported_as_a_bare_unsupported_falls_back(message):
    """torch 2.4 has no cache-limit exception class: `convert_frame` ends that
    branch in `unimplemented(f"{limit_type} reached")`, so exhaustion arrives
    as a plain `Unsupported` and re-raising it ends the run instead of slowing
    it. 2.5 and 2.6 reach the same `unimplemented` with
    skip_code_recursive_on_cache_limit_hit off."""
    from torch._dynamo.exc import Unsupported
    c, e, calls = _pair(Unsupported(message))
    w = _fall_back_to_eager_on_recompile_limit(c, e, "M.forward")
    assert w(3) == 6
    # Two compiled attempts: the one that hit the limit and the one retry the
    # wrapper makes with a raised budget, so a step that is halfway through an
    # activation-checkpoint pack can still finish compiled. Then eager.
    assert calls == {"c": 2, "e": 1}
    # And it latches, like every other fallback reason.
    assert w(3) == 6
    assert calls["c"] == 2


def test_error_tuple_is_non_empty_on_this_torch():
    errs = _recompile_limit_errors()
    assert _LIMIT_ERROR in errs
    assert all(issubclass(x, BaseException) for x in errs)


# ---- wiring into patch_function -------------------------------------------

def test_only_fullgraph_patches_get_the_wrapper():
    src = Path(
        sys.modules["unsloth_zoo.temporary_patches.utils"].__file__
    ).read_text(encoding = "utf-8")
    i = src.index("new_func = torch.compile(")
    tail = src[i:i + 800]
    assert "if fullgraph:" in tail, "fullgraph=False already falls back by itself"
    assert "_fall_back_to_eager_on_recompile_limit(" in tail


def test_real_cache_exhaustion_completes_with_correct_numerics():
    """The synthetic raises above prove the wrapper; this proves the premise.

    Exhausts a real Dynamo cache by giving each module instance a different
    python int attribute, which Dynamo guards on by value.
    """
    from unsloth_zoo.temporary_patches.utils import patch_function

    class M(torch.nn.Module):
        def __init__(self, k):
            super().__init__()
            self.k = k

        def forward(self, x):
            return x * self.k

    def forward(self, x):
        return x * self.k + 0.0

    torch._dynamo.reset()
    with torch._dynamo.config.patch({key: 2 for key in _LIMIT_KEYS}):
        assert patch_function(M, "forward", forward,
                              fullgraph = True, force = True)
        got = [M(k)(torch.tensor([1.0, 2.0])).tolist() for k in range(8)]
    assert got == [[k * 1.0, k * 2.0] for k in range(8)]


def test_end_to_end_through_patch_function():
    from unsloth_zoo.temporary_patches.utils import patch_function

    class M:
        def forward(self, x):
            return x + 1

    def forward(self, x):
        return x + 1

    assert patch_function(M, "forward", forward, fullgraph = True, force = True)
    assert M().forward(torch.tensor(1.0)).item() == 2.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# ---- the one case where falling back is the wrong answer ------------------

@contextlib.contextmanager
def _hard_failure(enabled):
    """Set whichever spelling this torch has, and put it back.

    2.6 only has fail_on_cache_limit_hit; 2.7 renamed it to
    fail_on_recompile_limit_hit and kept the old name as an alias. Reading a
    fixed name here would AttributeError on 2.6, which pyproject allows.
    """
    import torch._dynamo.config as config

    names = [n for n in ("fail_on_recompile_limit_hit", "fail_on_cache_limit_hit")
             if hasattr(config, n)]
    if not names:
        pytest.skip("this torch has no recompile-limit hard-failure flag")
    previous = {n: getattr(config, n) for n in names}
    # One name, since on 2.7+ the second is an alias of the first and writing
    # both would just set the same value twice.
    setattr(config, names[0], enabled)
    try:
        yield
    finally:
        for n, v in previous.items():
            setattr(config, n, v)


def test_the_hard_failure_flag_is_respected():
    """torch raises FailOnRecompileLimitHit from two branches with the same
    class: fullgraph=True, where falling back is exactly right, and
    `fail_on_recompile_limit_hit`, where the user asked for the run to stop.
    The exception cannot tell them apart, so the flag has to be read."""
    c, e, calls = _pair(_LIMIT_ERROR("recompile_limit reached"))
    w = _fall_back_to_eager_on_recompile_limit(c, e, "M.forward")
    with _hard_failure(True):
        with pytest.raises(_LIMIT_ERROR):
            w(3)
    # Nothing latched either, or a later call would silently go eager.
    assert calls == {"c": 1, "e": 0}


def test_the_flag_being_off_keeps_the_fallback():
    with _hard_failure(False):
        c, e, calls = _pair(_LIMIT_ERROR("recompile_limit reached"))
        w = _fall_back_to_eager_on_recompile_limit(c, e, "M.forward")
        assert w(3) == 6
    # Two compiled attempts: the one that hit the limit and the one retry the
    # wrapper makes with a raised budget, so a step that is halfway through an
    # activation-checkpoint pack can still finish compiled. Then eager.
    assert calls == {"c": 2, "e": 1}


def test_a_torch_without_the_flag_still_falls_back():
    """2.6 spells it fail_on_cache_limit_hit, and some builds have neither."""
    from unsloth_zoo.temporary_patches.utils import _wants_hard_recompile_failure
    import torch._dynamo.config as config

    saved = {n: getattr(config, n) for n in
             ("fail_on_recompile_limit_hit", "fail_on_cache_limit_hit")
             if hasattr(config, n)}
    for n in saved:
        setattr(config, n, False)
    try:
        assert _wants_hard_recompile_failure() is False
    finally:
        for n, v in saved.items():
            setattr(config, n, v)


def test_an_unrelated_error_from_the_retry_is_not_swallowed():
    """The retry must only absorb compiler failures.

    On cache exhaustion the wrapper retries the compiled function once with a
    raised budget. If that retry fails for a real reason -- a data-dependent op,
    a shape error, anything of the model's own -- falling through to eager runs
    the same call a second time, re-applying any mutation it already made, and
    buries the error. Only a recompile-limit failure may reach eager.
    """
    calls = {"c": 0, "e": 0}
    boom = RuntimeError("a real model failure, not a compiler one")

    def compiled(x):
        calls["c"] += 1
        if calls["c"] == 1:
            raise _LIMIT_ERROR("recompile_limit reached")
        raise boom

    def eager(x):
        calls["e"] += 1
        return x * 2

    with _hard_failure(False):
        w = _fall_back_to_eager_on_recompile_limit(compiled, eager, "M.forward")
        with pytest.raises(RuntimeError, match = "a real model failure"):
            w(3)

    # Two compiled attempts, and eager never ran: the caller sees its own error.
    assert calls == {"c": 2, "e": 0}


def test_the_recompile_budget_is_bounded_and_handed_back():
    """The budgets are process-global, so a per-wrapper cap bounds nothing.

    Every bump raises `torch._dynamo.config` for the whole process. Without a
    shared cap, N wrappers (or several models trained in one process) each spend
    their own allowance and the limit ends up hundreds higher for every unrelated
    compiled function; without a restore it stays there for the process's life.
    """
    from unsloth_zoo.temporary_patches import utils as u
    import torch._dynamo.config as config

    name = u._LIMIT_KEYS[0] if hasattr(u, "_LIMIT_KEYS") else _LIMIT_KEYS[0]
    before = getattr(config, name)
    saved_global, saved_orig = u._GLOBAL_BUMPS, dict(u._ORIGINAL_RECOMPILE_LIMITS)
    u._GLOBAL_BUMPS, u._ORIGINAL_RECOMPILE_LIMITS = 0, {}
    try:
        # Far more attempts than the cap allows, as many wrappers would make.
        granted = sum(bool(u._bump_recompile_limits()) for _ in range(50))
        assert granted == u._MAX_TOTAL_RECOMPILE_LIMIT_BUMPS, granted
        raised = getattr(config, name)
        assert raised == before + granted * u._RECOMPILE_LIMIT_BUMP, (before, raised)

        assert u._restore_recompile_limits() >= 1
        assert getattr(config, name) == before
        # And the allowance is available again for the next model.
        assert u._GLOBAL_BUMPS == 0
    finally:
        setattr(config, name, before)
        u._GLOBAL_BUMPS, u._ORIGINAL_RECOMPILE_LIMITS = saved_global, saved_orig


def test_exhausting_the_budget_takes_the_other_borrowers_with_it():
    """Wrappers in the same budget crisis must switch together, and only those.

    One checkpointed region routinely spans several patched functions. Letting
    only the wrapper that ran out go eager leaves the rest of the step half
    compiled, which is the mismatch this path exists to avoid. A wrapper that
    never borrowed budget was never in trouble and must stay compiled.
    """
    from unsloth_zoo.temporary_patches import utils as u

    borrower_calls = {"c": 0, "e": 0}
    bystander_calls = {"c": 0, "e": 0}

    def borrower(x):
        borrower_calls["c"] += 1
        raise _LIMIT_ERROR("recompile_limit reached")

    def borrower_eager(x):
        borrower_calls["e"] += 1
        return x * 2

    b_c, b_e, _ = _pair(_LIMIT_ERROR("recompile_limit reached"), borrower_calls)
    s_c, s_e, _ = _pair(None, bystander_calls)

    saved_global = u._GLOBAL_BUMPS
    with _hard_failure(False):
        first = _fall_back_to_eager_on_recompile_limit(b_c, b_e, "A.forward")
        second = _fall_back_to_eager_on_recompile_limit(borrower, borrower_eager,
                                                        "B.forward")
        bystander = _fall_back_to_eager_on_recompile_limit(s_c, s_e, "C.forward")
        # Both borrowers exhaust; the bystander never fails.
        first(1)
        u._GLOBAL_BUMPS = u._MAX_TOTAL_RECOMPILE_LIMIT_BUMPS   # budget gone
        second(1)

    assert second._unsloth_fallback_state["eager"] is True
    # first borrowed budget earlier, so it comes along.
    assert first._unsloth_fallback_state["eager"] is True
    # The bystander never bumped, so it stays compiled.
    assert bystander._unsloth_fallback_state["eager"] is False
    u._GLOBAL_BUMPS = saved_global


def _dead_ref():
    """A weakref whose referent is already gone, as the registry holds them."""
    import weakref
    def _gone(): pass
    ref = weakref.ref(_gone)
    del _gone
    import gc; gc.collect()
    return ref


def test_a_collected_borrower_does_not_strand_the_raised_limit():
    """The no-pending path must still settle debt.

    A wrapper can bump the budget, mark itself pending, and then be dropped
    before the next step boundary: training aborts, or the patched object is
    re-patched or replaced, and the registry holds it only weakly. The boundary
    hook then sees nothing pending and used to return early, leaving
    `torch._dynamo.config` raised for the life of the process and the shared
    bump allowance spent for every later model.
    """
    from unsloth_zoo.temporary_patches import utils as u
    import torch._dynamo.config as config

    name = u._LIMIT_KEYS[0] if hasattr(u, "_LIMIT_KEYS") else _LIMIT_KEYS[0]
    before = getattr(config, name)
    saved_global = u._GLOBAL_BUMPS
    saved_orig = dict(u._ORIGINAL_RECOMPILE_LIMITS)
    saved_registry = list(u._EAGER_FALLBACK_WRAPPERS)
    u._GLOBAL_BUMPS, u._ORIGINAL_RECOMPILE_LIMITS = 0, {}
    u._EAGER_FALLBACK_WRAPPERS.clear()
    try:
        assert u._bump_recompile_limits()
        assert getattr(config, name) > before
        # The borrower is gone: every registered ref is dead, as it would be
        # after the wrapper was collected.
        u._EAGER_FALLBACK_WRAPPERS.append(_dead_ref())

        u.apply_pending_eager_fallbacks()

        assert getattr(config, name) == before, "limit left raised for the process"
        assert u._GLOBAL_BUMPS == 0, "bump allowance left spent for later models"
    finally:
        setattr(config, name, before)
        u._GLOBAL_BUMPS = saved_global
        u._ORIGINAL_RECOMPILE_LIMITS.clear()
        u._ORIGINAL_RECOMPILE_LIMITS.update(saved_orig)
        u._EAGER_FALLBACK_WRAPPERS.clear()
        u._EAGER_FALLBACK_WRAPPERS.extend(saved_registry)


def test_a_live_borrower_still_keeps_its_headroom():
    """The control for the test above: do not hand the budget back underneath a
    wrapper that borrowed it and is still compiling against it."""
    from unsloth_zoo.temporary_patches import utils as u
    import torch._dynamo.config as config

    name = u._LIMIT_KEYS[0] if hasattr(u, "_LIMIT_KEYS") else _LIMIT_KEYS[0]
    before = getattr(config, name)
    saved_global = u._GLOBAL_BUMPS
    saved_orig = dict(u._ORIGINAL_RECOMPILE_LIMITS)
    saved_registry = list(u._EAGER_FALLBACK_WRAPPERS)
    u._GLOBAL_BUMPS, u._ORIGINAL_RECOMPILE_LIMITS = 0, {}
    u._EAGER_FALLBACK_WRAPPERS.clear()

    def _borrower(): pass
    _borrower._unsloth_fallback_state = {"eager": False, "pending_eager": False, "bumps": 1}
    _borrower._unsloth_fallback_label = "borrower"
    try:
        assert u._bump_recompile_limits()
        raised = getattr(config, name)
        import weakref
        u._EAGER_FALLBACK_WRAPPERS.append(weakref.ref(_borrower))

        u.apply_pending_eager_fallbacks()

        assert getattr(config, name) == raised, "took the headroom back mid-flight"
    finally:
        setattr(config, name, before)
        u._GLOBAL_BUMPS = saved_global
        u._ORIGINAL_RECOMPILE_LIMITS.clear()
        u._ORIGINAL_RECOMPILE_LIMITS.update(saved_orig)
        u._EAGER_FALLBACK_WRAPPERS.clear()
        u._EAGER_FALLBACK_WRAPPERS.extend(saved_registry)


@contextlib.contextmanager
def _isolated_budget():
    """Run with a private registry and a fresh, restored bump allowance.

    The bump state is process-global, so a test that leaves it dirty poisons
    every later test in the same worker.
    """
    from unsloth_zoo.temporary_patches import utils as u
    import torch._dynamo.config as config

    keys = u._LIMIT_KEYS if hasattr(u, "_LIMIT_KEYS") else _LIMIT_KEYS
    name = keys[0]
    # A bump raises the accumulated limit too, and one test leaves its bump
    # active on purpose, so restoring only the first name leaks +16 on the
    # second into every later test in this worker.
    before_all = {k: getattr(config, k) for k in keys if hasattr(config, k)}
    before = before_all[name]
    saved_global = u._GLOBAL_BUMPS
    saved_orig = dict(u._ORIGINAL_RECOMPILE_LIMITS)
    saved_bumped = dict(u._BUMPED_RECOMPILE_LIMITS)
    saved_registry = list(u._EAGER_FALLBACK_WRAPPERS)
    u._GLOBAL_BUMPS, u._ORIGINAL_RECOMPILE_LIMITS = 0, {}
    u._BUMPED_RECOMPILE_LIMITS.clear()
    u._EAGER_FALLBACK_WRAPPERS.clear()
    try:
        yield u, config, name, before
    finally:
        for key, value in before_all.items():
            setattr(config, key, value)
        u._BUMPED_RECOMPILE_LIMITS.clear()
        u._BUMPED_RECOMPILE_LIMITS.update(saved_bumped)
        u._GLOBAL_BUMPS = saved_global
        u._ORIGINAL_RECOMPILE_LIMITS.clear()
        u._ORIGINAL_RECOMPILE_LIMITS.update(saved_orig)
        u._EAGER_FALLBACK_WRAPPERS.clear()
        u._EAGER_FALLBACK_WRAPPERS.extend(saved_registry)


@pytest.mark.parametrize("boom", [
    RuntimeError("a real model failure, not a compiler one"),
    _dynamo_exc.Unsupported("a real graph break, not a budget problem"),
], ids = ["unrelated_error", "real_graph_break"])
def test_a_retry_that_raises_hands_the_borrowed_budget_back(boom):
    """A retry that dies must not keep the headroom it borrowed.

    The retry raises the budget for the whole process and counts a bump against
    the wrapper. If the call then fails for a reason of its own -- a bad batch
    the caller catches and skips, or a genuine graph break -- the wrapper stays
    non-eager with a bump outstanding, so the step boundary finds nothing
    pending and `_restore_recompile_limits_if_idle` refuses forever. The raised
    `torch._dynamo.config` and the spent shared allowance then outlive the run
    for every later model and unrelated compiled function.
    """
    with _isolated_budget() as (u, config, name, before):
        calls = {"c": 0}

        def compiled(x):
            calls["c"] += 1
            if calls["c"] == 1:
                raise _LIMIT_ERROR("recompile_limit reached")
            raise boom

        def eager(x):
            return x * 2

        with _hard_failure(False):
            w = _fall_back_to_eager_on_recompile_limit(compiled, eager, "M.forward")
            with pytest.raises(type(boom)):
                w(3)

        assert calls["c"] == 2, "the retry did not run"
        state = w._unsloth_fallback_state
        assert state["bumps"] == 0, "the failed call kept its bump"
        assert getattr(config, name) == before, "limit left raised for the process"
        assert u._GLOBAL_BUMPS == 0, "bump allowance left spent for later models"


def test_the_fixture_restores_every_limit_a_bump_raised():
    """`_bump_recompile_limits` raises the accumulated limit as well as the
    per-code one, and `test_a_successful_retry_keeps_its_bump` deliberately
    exits with its bump still active. Restoring one name left the other +16
    for every later test sharing this worker."""
    keys = [k for k in _LIMIT_KEYS if hasattr(torch._dynamo.config, k)]
    assert len(keys) > 1, "this torch exposes only one limit; nothing to leak"
    outer = {k: getattr(torch._dynamo.config, k) for k in keys}
    with _isolated_budget() as (mod, config, name, before):
        mod._bump_recompile_limits()
        assert all(getattr(config, k) > outer[k] for k in keys), "bump raised one only"
    assert {k: getattr(torch._dynamo.config, k) for k in keys} == outer


def test_a_bump_taken_inside_a_scoped_config_patch_is_not_written_back():
    """`torch._dynamo.config.patch` restores its outer value on exit, so the
    value captured inside it is stale by the time we settle up. Writing it back
    would change the process-wide limit for good."""
    with _isolated_budget() as (mod, config, name, before):
        with torch._dynamo.config.patch({name: 2}):
            mod._bump_recompile_limits()
            assert mod._ORIGINAL_RECOMPILE_LIMITS[name] == 2, "captured the temporary"
        assert getattr(config, name) == before, "dynamo restored the outer value"
        mod._restore_recompile_limits()
        assert getattr(config, name) == before, "clobbered the outer value"


def test_a_successful_retry_keeps_its_bump():
    """The control for the test above.

    The wrapper is genuinely compiling against the extra headroom until it goes
    eager, so a retry that succeeded must hold on to its bump; releasing it
    would take the budget away mid-flight.
    """
    with _isolated_budget() as (u, config, name, before):
        calls = {"c": 0}

        def compiled(x):
            calls["c"] += 1
            if calls["c"] == 1:
                raise _LIMIT_ERROR("recompile_limit reached")
            return x * 2

        def eager(x):
            return x * 2

        with _hard_failure(False):
            w = _fall_back_to_eager_on_recompile_limit(compiled, eager, "M.forward")
            assert w(3) == 6

        state = w._unsloth_fallback_state
        assert state["bumps"] == 1 and state["pending_eager"] is True
        assert getattr(config, name) > before, "headroom taken back mid-flight"
        assert u._GLOBAL_BUMPS == 1
