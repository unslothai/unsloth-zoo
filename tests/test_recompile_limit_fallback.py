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

"""Cache exhaustion must slow a run, not end it: patch_function(..., fullgraph=True) raises FailOnRecompileLimitHit,
which the wrapper handles by going eager; a genuine graph break still raises."""

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

# Dynamo names differ across torch>=2.4,<2.13: 2.4 has none, 2.5 only CacheLimitExceeded, 2.6+ RecompileLimitExceeded
# and FailOnRecompileLimitHit; importing one by name would kill collection on a supported torch, so use getattr as
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

# Dynamo renamed its cache-limit knobs in torch 2.7: 2.4 to 2.6 only have cache_size_limit, and config.patch raises on
# any key the installed torch does not define.
_LIMIT_KEYS = (
    ("recompile_limit", "accumulated_recompile_limit")
    if hasattr(torch._dynamo.config, "recompile_limit")
    else ("cache_size_limit", "accumulated_cache_size_limit")
)

pytestmark = pytest.mark.skipif(
    _LIMIT_ERROR is None,
    reason = "this torch exposes no Dynamo recompile-limit exception",
)


@pytest.fixture(autouse = True)
def _an_unspent_recompile_budget():
    """Wrappers share one process-wide budget: _bump_recompile_limits counts borrows in _GLOBAL_BUMPS up to
    _MAX_TOTAL_RECOMPILE_LIMIT_BUMPS, returned only once all settle to eager; a leaked borrow makes later tests
    order-dependent, so each test gets it back."""
    utils = sys.modules["unsloth_zoo.temporary_patches.utils"]
    config = torch._dynamo.config
    # Save every name the bump path writes, not just this torch's pair: _reset_bump_state restores _PRISTINE_LIMITS
    # across all of them, or the two views of the budget disagree.
    limits = {key: getattr(config, key) for key in _PRISTINE_LIMITS}
    saved = (
        utils._GLOBAL_BUMPS,
        dict(utils._ORIGINAL_RECOMPILE_LIMITS),
        dict(utils._BUMPED_RECOMPILE_LIMITS),
        set(utils._LATCHED_EAGER_LABELS),
        set(utils._PENDING_EAGER_LABELS),
        set(utils._RECENT_EAGER_LABELS),
    )
    utils._GLOBAL_BUMPS = 0
    utils._ORIGINAL_RECOMPILE_LIMITS.clear()
    utils._BUMPED_RECOMPILE_LIMITS.clear()
    # Also set the budget itself to _PRISTINE_LIMITS: a test entering on another value (a leak from an earlier file)
    # would otherwise see a number it never started from; teardown restores the outer value.
    for key, value in _PRISTINE_LIMITS.items():
        setattr(config, key, value)
    try:
        yield
    finally:
        for key, value in limits.items():
            setattr(config, key, value)
        utils._GLOBAL_BUMPS = saved[0]
        for table, restored in (
            (utils._ORIGINAL_RECOMPILE_LIMITS, saved[1]),
            (utils._BUMPED_RECOMPILE_LIMITS, saved[2]),
        ):
            table.clear()
            table.update(restored)
        for labels, restored in (
            (utils._LATCHED_EAGER_LABELS, saved[3]),
            (utils._PENDING_EAGER_LABELS, saved[4]),
            (utils._RECENT_EAGER_LABELS, saved[5]),
        ):
            labels.clear()
            labels.update(restored)


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


def test_recompile_limit_falls_back_instead_of_raising():
    c, e, calls = _pair(_LIMIT_ERROR("recompile_limit reached"))
    w = _fall_back_to_eager_on_recompile_limit(c, e, "M.forward")
    assert w(3) == 6
    # Two compiled attempts (the one that hit the limit plus one retry on a raised budget), so a step halfway through a
    # checkpoint pack can still finish compiled; then eager.
    assert calls == {"c": 2, "e": 1}


def test_the_fallback_latches():
    """Do not reverse, settled on live hardware: per-call compiler retry fails since pack and recompute run under
    different guards; latching leaves one step that force_eager_fallback lets unsloth retry."""
    c, e, calls = _pair(_LIMIT_ERROR("recompile_limit reached"))
    w = _fall_back_to_eager_on_recompile_limit(c, e, "M.forward")
    for _ in range(5):
        assert w(1) == 2
    # One failing attempt plus the single bumped retry, and nothing after the latch: the compiler is not consulted again.
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


def test_compiled_path_is_used_when_nothing_goes_wrong():
    c, e, calls = _pair()
    w = _fall_back_to_eager_on_recompile_limit(c, e, "M.forward")
    assert w(5) == 10
    assert calls == {"c": 1, "e": 0}


def test_signature_and_metadata_survive():
    c, e, _ = _pair()
    w = _fall_back_to_eager_on_recompile_limit(c, e, "M.forward")
    import inspect
    assert list(inspect.signature(w).parameters) == ["x"]
    assert w.__doc__ == "Eager docstring."
    assert w.__wrapped__ is e


def test_unwrapping_still_reaches_the_eager_function():
    # patch_function unwraps anything carrying get_compiler_config via __wrapped__, so double-patching must not nest compiles.
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
    # Older torch without these names returns the compiled function unchanged, but only if all three error sources are
    # empty: recompile limits, the graph break from Unsloth's torch.compiler.disable'd checkpoint hook under fullgraph,
    # and Inductor codegen failures.
    import unsloth_zoo.temporary_patches.utils as u
    monkeypatch.setattr(u, "_recompile_limit_errors", lambda: ())
    monkeypatch.setattr(u, "_disabled_hook_graph_break_error", lambda: ())
    monkeypatch.setattr(u, "_backend_compile_errors", lambda: ())
    c, e, _ = _pair()
    assert u._fall_back_to_eager_on_recompile_limit(c, e, "M.forward") is c


def test_wrapper_is_added_for_the_graph_break_alone(monkeypatch):
    """A torch with Unsupported but no recompile-limit names still gets the fallback, since the disabled-hook collision can still happen there."""
    import unsloth_zoo.temporary_patches.utils as u
    monkeypatch.setattr(u, "_recompile_limit_errors", lambda: ())
    c, e, _ = _pair()
    if not u._disabled_hook_graph_break_error():
        pytest.skip("this torch has no torch._dynamo.exc.Unsupported")
    assert u._fall_back_to_eager_on_recompile_limit(c, e, "M.forward") is not c


def test_no_version_specific_dynamo_name_is_imported_at_module_level():
    """torch 2.4 has no recompile-limit exceptions and 2.5 only CacheLimitExceeded, both inside torch>=2.4,<2.13; a
    top-level import would fail collection, so use getattr."""
    src = Path(__file__).read_text(encoding = "utf-8")
    for name in ("FailOnRecompileLimitHit", "RecompileLimitExceeded", "CacheLimitExceeded"):
        assert f"import {name}" not in src, f"{name} must be looked up with getattr"


def test_the_dynamo_limit_keys_exist_on_this_torch():
    """Limit key names must match the installed torch: a hardcoded spelling would error on torch 2.4 to 2.6 rather than run the cache-exhaustion test."""
    for key in _LIMIT_KEYS:
        assert hasattr(torch._dynamo.config, key), key


@pytest.mark.parametrize("message", [
    "cache_size_limit reached",
    "accumulated_cache_size_limit reached",
    "recompile_limit reached",
])
def test_cache_exhaustion_reported_as_a_bare_unsupported_falls_back(message):
    """torch 2.4 has no cache-limit exception: convert_frame ends that branch in unimplemented, so exhaustion arrives as
    a plain Unsupported and re-raising ends the run; 2.5 and 2.6 reach the same unimplemented with
    skip_code_recursive_on_cache_limit_hit off."""
    from torch._dynamo.exc import Unsupported
    c, e, calls = _pair(Unsupported(message))
    w = _fall_back_to_eager_on_recompile_limit(c, e, "M.forward")
    assert w(3) == 6
    # Two compiled attempts (the one that hit the limit plus one retry on a raised budget), so a step halfway through a
    # checkpoint pack can still finish compiled; then eager.
    assert calls == {"c": 2, "e": 1}
    # And it latches, like every other fallback reason.
    assert w(3) == 6
    assert calls["c"] == 2


def test_error_tuple_is_non_empty_on_this_torch():
    errs = _recompile_limit_errors()
    assert _LIMIT_ERROR in errs
    assert all(issubclass(x, BaseException) for x in errs)


def test_only_fullgraph_patches_get_the_wrapper():
    src = Path(
        sys.modules["unsloth_zoo.temporary_patches.utils"].__file__
    ).read_text(encoding = "utf-8")
    i = src.index("new_func = torch.compile(")
    tail = src[i:i + 800]
    assert "if fullgraph:" in tail, "fullgraph=False already falls back by itself"
    assert "_fall_back_to_eager_on_recompile_limit(" in tail


def test_real_cache_exhaustion_completes_with_correct_numerics():
    """The synthetic raises prove the wrapper; this exhausts a real Dynamo cache via a per-instance python int attribute, which Dynamo guards on by value."""
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


@contextlib.contextmanager
def _hard_failure(enabled):
    """Sets whichever of fail_on_recompile_limit_hit or fail_on_cache_limit_hit this torch has and restores it; 2.6 has only the latter, so a fixed name fails there."""
    import torch._dynamo.config as config

    names = [n for n in ("fail_on_recompile_limit_hit", "fail_on_cache_limit_hit")
             if hasattr(config, n)]
    if not names:
        pytest.skip("this torch has no recompile-limit hard-failure flag")
    previous = {n: getattr(config, n) for n in names}
    # Write one name: on 2.7+ the second is an alias of the first, so writing both would just set the same value twice.
    setattr(config, names[0], enabled)
    try:
        yield
    finally:
        for n, v in previous.items():
            setattr(config, n, v)


def test_the_hard_failure_flag_is_respected():
    """FailOnRecompileLimitHit comes from fullgraph=True (falling back is right) and from fail_on_recompile_limit_hit
    (user asked to stop); the exception cannot tell them apart, so read the flag."""
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
    # Two compiled attempts (the one that hit the limit plus one retry on a raised budget), so a step halfway through a
    # checkpoint pack can still finish compiled; then eager.
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
    """Only recompile-limit failures may reach eager: a model error in the single bumped retry would otherwise run the
    call twice, reapply mutations and bury the error."""
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
    """Budgets are process-global, so a per-wrapper cap bounds nothing: without a shared cap and a restore, the limit
    stays hundreds higher for every unrelated compiled function."""
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
    """Wrappers that borrowed from an exhausted budget switch together, since one checkpointed region spans several
    patched functions; a wrapper that never borrowed stays compiled."""
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
    """A bumped wrapper dropped before the step boundary must still settle: the registry holds it weakly, and an early
    return would leave torch._dynamo.config raised for the process."""
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
        # The borrower is gone: every registered ref is dead, as it would be after the wrapper was collected.
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
    """Control: do not hand the budget back underneath a wrapper that borrowed it and is still compiling against it."""
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
    """Gives a private registry and a fresh, restored bump allowance; the bump state is process-global, so a dirty test would poison the whole worker."""
    from unsloth_zoo.temporary_patches import utils as u
    import torch._dynamo.config as config

    keys = u._LIMIT_KEYS if hasattr(u, "_LIMIT_KEYS") else _LIMIT_KEYS
    name = keys[0]
    # A bump raises the accumulated limit too, and one test leaves its bump active on purpose; restoring only the first
    # name leaks +16 on the second into every later test.
    before_all = {k: getattr(config, k) for k in keys if hasattr(config, k)}
    before = before_all[name]
    saved_global = u._GLOBAL_BUMPS
    saved_orig = dict(u._ORIGINAL_RECOMPILE_LIMITS)
    saved_bumped = dict(u._BUMPED_RECOMPILE_LIMITS)
    saved_registry = list(u._EAGER_FALLBACK_WRAPPERS)
    # A wrapper run inside a real checkpoint sets the flags saved below; only a settled boundary clears them. Left true,
    # _give_up re-raises in every later test instead of falling back, reading as a broken kernel elsewhere.
    saved_packed = u._PACKED_COMPILED_IN_CHECKPOINT
    saved_raised = u._RAISED_INSIDE_CHECKPOINT
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
        u._PACKED_COMPILED_IN_CHECKPOINT = saved_packed
        u._RAISED_INSIDE_CHECKPOINT = saved_raised


@pytest.mark.parametrize("boom", [
    RuntimeError("a real model failure, not a compiler one"),
    _dynamo_exc.Unsupported("a real graph break, not a budget problem"),
], ids = ["unrelated_error", "real_graph_break"])
def test_a_retry_that_raises_hands_the_borrowed_budget_back(boom):
    """A failing retry must return its borrowed budget: otherwise the bump stays outstanding,
    _restore_recompile_limits_if_idle refuses forever, and the raised limit outlives the run."""
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
    """_bump_recompile_limits raises the accumulated limit as well as the per-code one; restoring only one leaves the other +16 for every later test in the worker."""
    keys = [k for k in _LIMIT_KEYS if hasattr(torch._dynamo.config, k)]
    assert len(keys) > 1, "this torch exposes only one limit; nothing to leak"
    outer = {k: getattr(torch._dynamo.config, k) for k in keys}
    with _isolated_budget() as (mod, config, name, before):
        mod._bump_recompile_limits()
        assert all(getattr(config, k) > outer[k] for k in keys), "bump raised one only"
    assert {k: getattr(torch._dynamo.config, k) for k in keys} == outer


def test_a_bump_nested_inside_a_scoped_config_patch_still_settles_up():
    """Bumps nest (8 to 24, patch to 2, then 18): each recorded value is ours to settle, and recording only the newest left 24 stuck forever."""
    with _isolated_budget() as (mod, config, name, before):
        mod._bump_recompile_limits()
        first = getattr(config, name)
        assert first > before
        with torch._dynamo.config.patch({name: 2}):
            mod._bump_recompile_limits()
            assert getattr(config, name) != first, "the nested bump is a new value"
        assert getattr(config, name) == first, "dynamo restored our earlier bump"
        mod._restore_recompile_limits()
        assert getattr(config, name) == before, "the first bump was stranded"


def test_a_bump_taken_inside_a_scoped_config_patch_is_not_written_back():
    """torch._dynamo.config.patch restores its outer value on exit, so a value captured inside it is stale; writing it
    back would change the process-wide limit for good."""
    with _isolated_budget() as (mod, config, name, before):
        with torch._dynamo.config.patch({name: 2}):
            mod._bump_recompile_limits()
            assert mod._ORIGINAL_RECOMPILE_LIMITS[name] == 2, "captured the temporary"
        assert getattr(config, name) == before, "dynamo restored the outer value"
        mod._restore_recompile_limits()
        assert getattr(config, name) == before, "clobbered the outer value"


def test_a_restore_underneath_an_active_config_patch_keeps_the_debt():
    """A step boundary may land inside a torch._dynamo.config.patch where our bump is not live; dropping the bookkeeping
    there loses the original, so the debt must be kept."""
    with _isolated_budget() as (mod, config, name, before):
        mod._bump_recompile_limits()
        bumped = getattr(config, name)
        assert bumped > before
        with torch._dynamo.config.patch({name: 2}):
            mod._restore_recompile_limits()     # our value is hidden right now
            assert getattr(config, name) == 2, "clobbered the patched value"
        assert getattr(config, name) == bumped, "dynamo restored our bump"
        mod._restore_recompile_limits()
        assert getattr(config, name) == before, "the bump was stranded forever"


def test_a_successful_retry_keeps_its_bump():
    """Inside a checkpoint a successful retry must hold its bump for the rest of the step; outside one it settles at
    once (test_a_successful_retry_outside_a_checkpoint_settles_itself)."""
    from torch.utils.checkpoint import checkpoint

    with _isolated_budget() as (u, config, name, before):
        if u._in_non_reentrant_checkpoint() is None:
            pytest.skip("this torch cannot report checkpoint regions")
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
            x = torch.randn(2, 2, requires_grad = True)
            checkpoint(lambda t: w(t).sum(), x, use_reentrant = False).backward()

        state = w._unsloth_fallback_state
        assert state["bumps"] == 1 and state["pending_eager"] is True
        assert getattr(config, name) > before, "headroom taken back mid-flight"
        assert u._GLOBAL_BUMPS == 1


def _utils():
    """The module itself, so module-level bookkeeping can be reset."""
    import unsloth_zoo.temporary_patches.utils as U
    return U


@pytest.fixture(autouse = True)
def _leave_the_packages_kernels_as_found():
    """_latch_all_to_eager latches every live borrower in the process-wide registry, so exhausting the budget here would
    also latch the gemma/gemma4/qwen3 kernels and skew their tests."""
    import unsloth_zoo.temporary_patches.utils as U
    saved = [(w, dict(w._unsloth_fallback_state))
             for w in (r() for r in U._EAGER_FALLBACK_WRAPPERS) if w is not None]
    yield
    for w, st in saved:
        w._unsloth_fallback_state.clear()
        w._unsloth_fallback_state.update(st)


@pytest.fixture(autouse = True)
def _forget_this_files_latches():
    """The give-up decision is keyed by label so it outlives the wrapper; it also outlives a test, so this file's labels are dropped around each test."""
    import unsloth_zoo.temporary_patches.utils as U
    U._LATCHED_EAGER_LABELS -= set(_OUR_LABELS)
    U._PENDING_EAGER_LABELS -= set(_OUR_LABELS)
    yield
    U._LATCHED_EAGER_LABELS -= set(_OUR_LABELS)
    U._PENDING_EAGER_LABELS -= set(_OUR_LABELS)


# Labels this file's own wrappers are built with, so the reset can leave the package's real kernels registered.
_OUR_LABELS = frozenset((
    "M.forward", "A.forward", "B.forward", "C.forward", "probe",
))


def _limit_names():
    import torch._dynamo.config as cfg
    return [n for group in _utils()._RECOMPILE_LIMIT_NAMES for n in group
            if isinstance(getattr(cfg, n, None), int)]


def _snapshot_limits():
    import torch._dynamo.config as cfg
    return {n: getattr(cfg, n) for n in _limit_names()}


# Captured at import, before any test bumps: pytest imports the module first.
_PRISTINE_LIMITS = _snapshot_limits()


def _reset_bump_state(U):
    import torch._dynamo.config as cfg
    U._ORIGINAL_RECOMPILE_LIMITS.clear()
    U._BUMPED_RECOMPILE_LIMITS.clear()
    U._GLOBAL_BUMPS = 0
    # Drop only this file's wrappers: the registry is process-wide, and clearing it deregistered gemma/gemma4/qwen3 so
    # their tests could not find themselves in eager_fallback_state().
    U._EAGER_FALLBACK_WRAPPERS[:] = [
        _r for _r in U._EAGER_FALLBACK_WRAPPERS if _r() is not None
        and getattr(_r(), "_unsloth_fallback_label", None) not in _OUR_LABELS
    ]
    # Same scoping rule for the labels the give-up decision now lives in.
    U._LATCHED_EAGER_LABELS -= set(_OUR_LABELS)
    U._PENDING_EAGER_LABELS -= set(_OUR_LABELS)
    # Clearing the bookkeeping alone left a real bump standing: an exhausted wrapper raised both budgets by 16 before
    # signalling, so later tests ran against enlarged limits and could stop reaching the exhaustion they test.
    for name, value in _PRISTINE_LIMITS.items():
        setattr(cfg, name, value)

def test_a_scoped_first_bump_does_not_strand_a_stale_original():
    """A first bump inside torch._dynamo.config.patch records the temporary value; setdefault kept it as the original,
    so bookkeeping must rebase or a restore lowers the process-wide limit."""
    import torch._dynamo.config as cfg
    U = _utils()
    name = "recompile_limit" if hasattr(cfg, "recompile_limit") else "cache_size_limit"
    _reset_bump_state(U)
    outer = getattr(cfg, name)
    try:
        with torch._dynamo.config.patch({name: 2}):
            U._bump_recompile_limits(16)          # records original 2
        assert getattr(cfg, name) == outer, "patch exit should restore the outer value"
        U._bump_recompile_limits(16)              # must rebase onto `outer`
        U._restore_recompile_limits()
        assert getattr(cfg, name) == outer, (
            f"limit left at {getattr(cfg, name)}, expected {outer}")
    finally:
        setattr(cfg, name, outer)
        _reset_bump_state(U)


def test_the_frame_walk_has_no_depth_cap():
    """No depth cap on the frame walk: checkpoint() can sit well over 60 frames up under nested module dispatch, so stopping early would wrongly report no region."""
    import inspect
    U = _utils()
    src = inspect.getsource(U._walk_for_checkpoint_frame)
    assert "_limit" not in src, "the depth cap is back"
    assert "seen" not in src


def test_deep_nesting_still_finds_the_region():
    U = _utils()
    if U._saved_tensor_hook_accessor() is None:
        pytest.skip("needs a torch that can build the region")
    seen = {}

    def probe():
        seen["walk"] = U._walk_for_checkpoint_frame()
        return torch.zeros(1, requires_grad=True).sum()

    def deep(n):
        return probe() if n == 0 else deep(n - 1)

    x = torch.randn(4, 4, requires_grad=True)
    torch.utils.checkpoint.checkpoint(lambda _: deep(120), x, use_reentrant=False)
    assert seen["walk"] is True, "120 frames deep and the walk lost the region"


def test_an_uninspectable_hook_stack_reads_as_unknown():
    """Before torch 2.8 nothing can be inspected, and that is when the settlement retry is needed, so an uninspectable hook stack must read as unknown, not False."""
    U = _utils()
    real = U._saved_tensor_hook_accessor
    U._saved_tensor_hook_accessor = lambda: None
    try:
        assert U._checkpoint_hooks_left_installed() is U._UNKNOWN
    finally:
        U._saved_tensor_hook_accessor = real


def test_settlement_stays_pending_when_it_cannot_be_inspected():
    U = _utils()
    real = U._saved_tensor_hook_accessor
    U._saved_tensor_hook_accessor = lambda: None
    try:
        U._RAISED_INSIDE_CHECKPOINT = True
        U._CHECKPOINT_SETTLE_ATTEMPTS = 0
        assert U._settle_abandoned_checkpoint_generator() is False
        assert U._RAISED_INSIDE_CHECKPOINT is True
    finally:
        U._saved_tensor_hook_accessor = real
        U._RAISED_INSIDE_CHECKPOINT = False
        U._CHECKPOINT_SETTLE_ATTEMPTS = 0


def test_settlement_still_gives_up_eventually():
    """Settlement retries are bounded, so a permanently rooted traceback cannot cost a collect a step forever."""
    U = _utils()
    real = U._saved_tensor_hook_accessor
    U._saved_tensor_hook_accessor = lambda: None
    try:
        U._RAISED_INSIDE_CHECKPOINT = True
        U._CHECKPOINT_SETTLE_ATTEMPTS = 0
        for _ in range(U._MAX_CHECKPOINT_SETTLE_ATTEMPTS + 2):
            if U._settle_abandoned_checkpoint_generator():
                break
        else:
            pytest.fail("settlement never gave up")
    finally:
        U._saved_tensor_hook_accessor = real
        U._RAISED_INSIDE_CHECKPOINT = False
        U._CHECKPOINT_SETTLE_ATTEMPTS = 0


def test_the_early_stop_signal_is_resolvable():
    U = _utils()
    errs = U._checkpoint_early_stop_errors()
    assert isinstance(errs, tuple)
    for cls in errs:
        assert issubclass(cls, BaseException)


def test_early_stop_counts_as_a_finished_retry():
    """_StopRecomputationError from checkpoint's recompute hook is a finished retry, not a failure: releasing the bump
    and re-raising reset the counters, letting variants borrow past both caps."""
    U = _utils()
    errs = U._checkpoint_early_stop_errors()
    if not errs:
        pytest.skip("this torch has no _StopRecomputationError")
    stop = errs[0]
    calls = {"n": 0}

    def eager(x):
        return x

    class _Compiled:
        def __call__(self, x):
            calls["n"] += 1
            if calls["n"] == 1:
                raise U._recompile_limit_errors()[0]("cache exhausted")
            raise stop()

    wrapped = U._fall_back_to_eager_on_recompile_limit(_Compiled(), eager, "probe")
    _reset_bump_state(U)
    with pytest.raises(stop):
        wrapped(torch.zeros(1))
    assert wrapped._unsloth_fallback_state["pending_eager"] is True, \
        "the retry finished; the wrapper must still be latched for next step"
    _reset_bump_state(U)


def test_a_hidden_branch_survives_restoring_a_visible_one():
    """Restoring a visible branch must keep hidden branches' debt: a restore that popped the whole per-name map lost the 24->8 debt and handed 24 back for good."""
    import torch._dynamo.config as cfg
    U = _utils()
    name = _limit_names()[0]
    _reset_bump_state(U)
    outer = getattr(cfg, name)
    try:
        U._bump_recompile_limits(16)                  # outer -> outer+16
        bumped = getattr(cfg, name)
        with torch._dynamo.config.patch({name: 2}):
            U._restore_recompile_limits()             # ours is hidden: keep it
            U._bump_recompile_limits(16)              # 2 -> 18
            U._restore_recompile_limits()             # settles 18 -> 2
            assert getattr(cfg, name) == 2
        assert getattr(cfg, name) == bumped, "patch exit hands our bump back"
        assert U._restore_recompile_limits() == 1, "the hidden debt was dropped"
        assert getattr(cfg, name) == outer, (
            f"limit left at {getattr(cfg, name)}, expected {outer}")
    finally:
        setattr(cfg, name, outer)
        _reset_bump_state(U)


def test_the_allowance_counts_what_is_actually_in_effect():
    """The allowance counts the chain depth under the live value: zeroing it let scoped patches re-borrow everything,
    while counting every recorded branch starved wrappers of budget."""
    import torch._dynamo.config as cfg
    U = _utils()
    name = _limit_names()[0]
    _reset_bump_state(U)
    outer = getattr(cfg, name)
    try:
        U._bump_recompile_limits(16)
        assert U._GLOBAL_BUMPS == 1
        with torch._dynamo.config.patch({name: 2}):
            U._restore_recompile_limits()             # nothing settled
            # Under the patch the limit is 2: our headroom is not in effect, so borrowing here is bounded by 2 and not by our raised value.
            assert U._GLOBAL_BUMPS == 0
        # ...and the moment the patch hands it back it is charged again.
        U._bump_recompile_limits(0)
        assert U._GLOBAL_BUMPS >= 1, "a live bump stopped being counted"
        U._restore_recompile_limits()
        U._restore_recompile_limits()
        assert U._GLOBAL_BUMPS == 0, "a settled bump must be repaid"
    finally:
        setattr(cfg, name, outer)
        _reset_bump_state(U)


def test_a_branch_a_completed_patch_rolled_back_stops_counting():
    """A bump taken inside a patch dies with the patch, so its entry must stop consuming the process-wide allowance (the starvation case)."""
    import torch._dynamo.config as cfg
    U = _utils()
    name = _limit_names()[0]
    _reset_bump_state(U)
    outer = getattr(cfg, name)
    try:
        with torch._dynamo.config.patch({name: 2}):
            U._bump_recompile_limits(16)              # 2 -> 18, dies on exit
        U._bump_recompile_limits(16)                  # outer -> outer+16
        U._restore_recompile_limits()
        assert getattr(cfg, name) == outer
        assert U._GLOBAL_BUMPS == 0, "an unreachable branch was still charged"
    finally:
        setattr(cfg, name, outer)
        _reset_bump_state(U)


def test_two_bumps_landing_on_the_same_value_are_two_debts():
    """Two bumps landing on the same value are two debts: one map entry per value collapsed them, so the patch exit
    resurrected the bumped value with nothing to restore."""
    import torch._dynamo.config as cfg
    U = _utils()
    name = _limit_names()[0]
    _reset_bump_state(U)
    outer = getattr(cfg, name)
    try:
        U._bump_recompile_limits(16)                  # outer -> outer+16
        bumped = getattr(cfg, name)
        with torch._dynamo.config.patch({name: outer}):
            U._bump_recompile_limits(16)              # outer -> outer+16 again
            U._restore_recompile_limits()             # settles ONE of them
            assert getattr(cfg, name) == outer
        assert getattr(cfg, name) == bumped, "patch exit hands the outer one back"
        assert U._restore_recompile_limits() == 1, "the second debt was lost"
        assert getattr(cfg, name) == outer
    finally:
        setattr(cfg, name, outer)
        _reset_bump_state(U)


def test_the_total_cap_still_holds_across_scoped_patches():
    """The counter keeps bumps bounded even when every restore lands under a patch that hides the live value."""
    import torch._dynamo.config as cfg
    U = _utils()
    name = _limit_names()[0]
    _reset_bump_state(U)
    outer = getattr(cfg, name)
    try:
        taken = 0
        for _ in range(U._MAX_TOTAL_RECOMPILE_LIMIT_BUMPS + 4):
            if U._bump_recompile_limits(16):
                taken += 1
            with torch._dynamo.config.patch({name: 2}):
                U._restore_recompile_limits()
        assert taken == U._MAX_TOTAL_RECOMPILE_LIMIT_BUMPS, taken
    finally:
        setattr(cfg, name, outer)
        _reset_bump_state(U)


def test_a_fully_settled_name_leaves_no_bookkeeping_behind():
    """The ordinary path must still clear bookkeeping, or the counter never reaches 0 and bumps stop being available at all."""
    import torch._dynamo.config as cfg
    U = _utils()
    name = _limit_names()[0]
    _reset_bump_state(U)
    outer = getattr(cfg, name)
    try:
        U._bump_recompile_limits(16)
        U._bump_recompile_limits(16)
        U._restore_recompile_limits()
        assert getattr(cfg, name) == outer
        assert not U._BUMPED_RECOMPILE_LIMITS
        assert not U._ORIGINAL_RECOMPILE_LIMITS
        assert U._GLOBAL_BUMPS == 0
    finally:
        setattr(cfg, name, outer)
        _reset_bump_state(U)


def test_the_reset_helper_puts_the_real_budgets_back():
    """The reset helper must restore the real budgets: clearing only bookkeeping left both limits +16 for everything after a test whose wrapper really bumped."""
    import torch._dynamo.config as cfg
    U = _utils()
    names = _limit_names()
    before = {n: getattr(cfg, n) for n in names}
    U._bump_recompile_limits(16)
    assert any(getattr(cfg, n) != before[n] for n in names), "nothing bumped"
    _reset_bump_state(U)
    assert {n: getattr(cfg, n) for n in names} == before


def test_giving_up_ends_the_step_when_an_earlier_layer_packed_compiled():
    """Give-up must end the step if an earlier layer packed compiled, since a checkpointed forward returns before
    backward and _in_non_reentrant_checkpoint() then reads False; eager recompute aborts or gives wrong grads."""
    U = _utils()
    _reset_bump_state(U)
    U._PACKED_COMPILED_IN_CHECKPOINT = False

    def eager(x):
        return x

    class _Exhausted:
        def __call__(self, x):
            raise U._recompile_limit_errors()[0]("cache exhausted")

    wrapped = U._fall_back_to_eager_on_recompile_limit(_Exhausted(), eager, "M.forward")
    real_in = U._in_non_reentrant_checkpoint
    real_bump = U._bump_recompile_limits
    U._in_non_reentrant_checkpoint = lambda: False   # the layer has returned
    U._bump_recompile_limits = lambda *a, **k: False # nothing left to borrow
    try:
        U._PACKED_COMPILED_IN_CHECKPOINT = True
        with pytest.raises(U._recompile_limit_errors()):
            wrapped(torch.zeros(1))
        assert U._RAISED_INSIDE_CHECKPOINT is True, \
            "the step has to be ended so the caller can retry it"
    finally:
        U._in_non_reentrant_checkpoint = real_in
        U._bump_recompile_limits = real_bump
        U._RAISED_INSIDE_CHECKPOINT = False
        U._CHECKPOINT_SETTLE_ATTEMPTS = 0
        U._PACKED_COMPILED_IN_CHECKPOINT = False
        _reset_bump_state(U)


def test_nothing_packed_this_step_still_just_goes_eager():
    """With nothing packed compiled this step, give-up must still go eager, which is survivable and cheaper than a retry, so ordinary steps must not start ending."""
    U = _utils()
    _reset_bump_state(U)
    calls = {"e": 0}

    def eager(x):
        calls["e"] += 1
        return x

    class _Exhausted:
        def __call__(self, x):
            raise U._recompile_limit_errors()[0]("cache exhausted")

    wrapped = U._fall_back_to_eager_on_recompile_limit(_Exhausted(), eager, "M.forward")
    real_in = U._in_non_reentrant_checkpoint
    real_bump = U._bump_recompile_limits
    U._in_non_reentrant_checkpoint = lambda: False
    U._bump_recompile_limits = lambda *a, **k: False
    try:
        U._PACKED_COMPILED_IN_CHECKPOINT = False
        assert wrapped(torch.zeros(1)) is not None
        assert calls["e"] == 1
        assert U._RAISED_INSIDE_CHECKPOINT is False
    finally:
        U._in_non_reentrant_checkpoint = real_in
        U._bump_recompile_limits = real_bump
        U._PACKED_COMPILED_IN_CHECKPOINT = False
        _reset_bump_state(U)


def test_the_packed_marker_is_cleared_at_the_step_boundary():
    """The packed-compiled marker is cleared at the step boundary, since last step's activations are freed; a sticky flag would end every later step."""
    U = _utils()
    _reset_bump_state(U)
    U._PACKED_COMPILED_IN_CHECKPOINT = True
    U._restore_recompile_limits()
    assert U._PACKED_COMPILED_IN_CHECKPOINT is False


def test_the_packed_probe_is_invisible_to_dynamo():
    """The probe must be invisible to Dynamo: the accessor is a pybind builtin it refuses to enter, fatal under
    fullgraph; Gemma4_(E2B)-Vision died at cell 15 with the probe."""
    U = _utils()
    real = U._dynamo_is_tracing
    U._dynamo_is_tracing = lambda: True
    try:
        U._PACKED_COMPILED_IN_CHECKPOINT = False
        U._note_packed_under_checkpoint()
        assert U._PACKED_COMPILED_IN_CHECKPOINT is False, \
            "the probe must not run, or even look, while tracing"
    finally:
        U._dynamo_is_tracing = real
        U._PACKED_COMPILED_IN_CHECKPOINT = False


def test_the_probe_still_runs_from_eager():
    U = _utils()
    real = U._dynamo_is_tracing
    U._dynamo_is_tracing = lambda: False
    saved = U._saved_tensor_hook_accessor
    def _pack(*a, **k): ...
    _pack.__qualname__ = "_checkpoint_hook.<locals>.pack_hook"
    _pack.__module__ = "torch.utils.checkpoint"
    U._saved_tensor_hook_accessor = lambda: (lambda _: [_pack, _pack])
    try:
        U._PACKED_COMPILED_IN_CHECKPOINT = False
        U._note_packed_under_checkpoint()
        assert U._PACKED_COMPILED_IN_CHECKPOINT is True
    finally:
        U._dynamo_is_tracing = real
        U._saved_tensor_hook_accessor = saved
        U._PACKED_COMPILED_IN_CHECKPOINT = False


def test_the_tracing_check_survives_a_torch_without_it():
    """torch 2.4 has neither accessor; answering False keeps the old path."""
    U = _utils()
    assert U._dynamo_is_tracing() in (True, False)


def test_a_real_compiled_region_traces_through_the_wrapper():
    """Compiles a function that calls a wrapped one under fullgraph; before the guard this raised `Attempted to call function marked as skipped`."""
    U = _utils()
    _reset_bump_state(U)

    def eager(x):
        return x * 2

    wrapped = U._fall_back_to_eager_on_recompile_limit(eager, eager, "M.forward")

    def outer(x):
        return wrapped(x) + 1

    compiled = torch.compile(outer, fullgraph = True, backend = "aot_eager")
    try:
        assert torch.equal(compiled(torch.ones(3)), torch.full((3,), 3.0))
    finally:
        _reset_bump_state(U)
        torch._dynamo.reset()


def test_a_pre_2_8_torch_does_not_mark_every_call_as_packed():
    """The saved-tensor-hook accessor arrived in torch 2.8; on the four supported 2.4-2.7 releases it is absent, so
    latching marked every compiled call packed and _give_up rethrew."""
    U = _utils()
    real = U._saved_tensor_hook_accessor
    U._saved_tensor_hook_accessor = lambda: None
    saved = U._PACKED_COMPILED_IN_CHECKPOINT
    try:
        U._PACKED_COMPILED_IN_CHECKPOINT = False
        U._note_packed_under_checkpoint()
        assert U._PACKED_COMPILED_IN_CHECKPOINT is False
    finally:
        U._saved_tensor_hook_accessor = real
        U._PACKED_COMPILED_IN_CHECKPOINT = saved


def test_a_pre_2_8_give_up_still_sees_a_region_around_the_failing_call():
    """Without the accessor only the already-returned earlier layer is lost: the give-up path walks the live stack itself, and that walk works on every torch."""
    U = _utils()
    real_accessor = U._saved_tensor_hook_accessor
    real_walk = U._walk_for_checkpoint_frame
    U._saved_tensor_hook_accessor = lambda: None
    U._walk_for_checkpoint_frame = lambda: True
    try:
        assert U._in_non_reentrant_checkpoint() is True
    finally:
        U._saved_tensor_hook_accessor = real_accessor
        U._walk_for_checkpoint_frame = real_walk


def test_an_uninspectable_accessor_that_raises_still_latches():
    """A present but broken accessor is not the same as an absent one: torch could answer and did not, so stay conservative and latch."""
    U = _utils()
    def _boom(*a, **k): raise RuntimeError("no")
    real = U._saved_tensor_hook_accessor
    U._saved_tensor_hook_accessor = lambda: _boom
    saved = U._PACKED_COMPILED_IN_CHECKPOINT
    try:
        U._PACKED_COMPILED_IN_CHECKPOINT = False
        U._note_packed_under_checkpoint()
        assert U._PACKED_COMPILED_IN_CHECKPOINT is True
    finally:
        U._saved_tensor_hook_accessor = real
        U._PACKED_COMPILED_IN_CHECKPOINT = saved


def test_an_unsettled_boundary_says_so(caplog):
    """`apply_pending_eager_fallbacks` dropped the settlement result, so a
    caller retrying from inside `except ... as exc` -- whose traceback roots the
    abandoned generator -- was told the boundary was clean while the checkpoint
    hooks were still installed."""
    import logging
    U = _utils()
    real = U._saved_tensor_hook_accessor
    U._saved_tensor_hook_accessor = lambda: None      # settlement stays pending
    try:
        U._RAISED_INSIDE_CHECKPOINT = True
        U._CHECKPOINT_SETTLE_ATTEMPTS = 0
        try: U.logger.warning_once.cache_clear()
        except Exception: pass
        with caplog.at_level(logging.WARNING):
            U.apply_pending_eager_fallbacks()
        assert "has not been finalised" in caplog.text
    finally:
        U._saved_tensor_hook_accessor = real
        U._RAISED_INSIDE_CHECKPOINT = False
        U._CHECKPOINT_SETTLE_ATTEMPTS = 0


def test_a_settled_boundary_is_quiet(caplog):
    import logging
    U = _utils()
    U._RAISED_INSIDE_CHECKPOINT = False
    with caplog.at_level(logging.WARNING):
        U.apply_pending_eager_fallbacks()
    assert "has not been finalised" not in caplog.text


def test_a_user_hook_above_the_checkpoints_does_not_hide_it():
    """The accessor sees only the top hook, so a save_on_cpu or saved_tensors_hooks entered inside the region must not
    hide the checkpoint beneath it; else give-up recomputes an earlier compiled activation eagerly."""
    U = _utils()
    def _mine(*a, **k): ...
    _mine.__qualname__ = "save_on_cpu.<locals>.pack_to_cpu"
    _mine.__module__ = "torch.autograd.graph"
    saved_accessor = U._saved_tensor_hook_accessor
    saved_walk = U._walk_for_checkpoint_frame
    saved_marker = U._PACKED_COMPILED_IN_CHECKPOINT
    U._saved_tensor_hook_accessor = lambda: (lambda _: [_mine, _mine])
    U._walk_for_checkpoint_frame = lambda: True     # a region IS open
    try:
        U._PACKED_COMPILED_IN_CHECKPOINT = False
        U._note_packed_under_checkpoint()
        assert U._PACKED_COMPILED_IN_CHECKPOINT is True
    finally:
        U._saved_tensor_hook_accessor = saved_accessor
        U._walk_for_checkpoint_frame = saved_walk
        U._PACKED_COMPILED_IN_CHECKPOINT = saved_marker


def test_a_user_hook_with_no_region_below_it_does_not_latch():
    """Frames are consulted, not assumed: an ordinary saved_tensors_hooks outside any checkpoint must stay unlatched, or such runs lose the eager fallback."""
    U = _utils()
    def _mine(*a, **k): ...
    saved_accessor = U._saved_tensor_hook_accessor
    saved_walk = U._walk_for_checkpoint_frame
    saved_marker = U._PACKED_COMPILED_IN_CHECKPOINT
    U._saved_tensor_hook_accessor = lambda: (lambda _: [_mine, _mine])
    U._walk_for_checkpoint_frame = lambda: False
    try:
        U._PACKED_COMPILED_IN_CHECKPOINT = False
        U._note_packed_under_checkpoint()
        assert U._PACKED_COMPILED_IN_CHECKPOINT is False
    finally:
        U._saved_tensor_hook_accessor = saved_accessor
        U._walk_for_checkpoint_frame = saved_walk
        U._PACKED_COMPILED_IN_CHECKPOINT = saved_marker


def test_an_empty_hook_stack_never_walks_the_frames():
    """No hooks means no region, so the per-call probe must return early and never pay for a full stack walk, since that is the common case."""
    U = _utils()
    walked = {"n": 0}
    saved_accessor = U._saved_tensor_hook_accessor
    saved_walk = U._walk_for_checkpoint_frame
    saved_marker = U._PACKED_COMPILED_IN_CHECKPOINT
    def _count():
        walked["n"] += 1
        return True
    U._saved_tensor_hook_accessor = lambda: (lambda _: [])
    U._walk_for_checkpoint_frame = _count
    try:
        U._PACKED_COMPILED_IN_CHECKPOINT = False
        U._note_packed_under_checkpoint()
        assert walked["n"] == 0
        assert U._PACKED_COMPILED_IN_CHECKPOINT is False
    finally:
        U._saved_tensor_hook_accessor = saved_accessor
        U._walk_for_checkpoint_frame = saved_walk
        U._PACKED_COMPILED_IN_CHECKPOINT = saved_marker


def test_a_rebuilt_wrapper_remembers_the_give_up():
    """The give-up decision belongs to the call site, keyed by label: GRPO's accumulate_chunk is rebuilt each forward
    and held weakly, so a latch on the wrapper never bounded the switch."""
    c1, e1, calls1 = _pair(_LIMIT_ERROR("recompile_limit reached"))
    w1 = _fall_back_to_eager_on_recompile_limit(c1, e1, "C.forward")
    w1(1)
    assert w1._unsloth_fallback_state["eager"], "the first one should have latched"
    del w1

    c2, e2, calls2 = _pair()
    w2 = _fall_back_to_eager_on_recompile_limit(c2, e2, "C.forward")
    assert w2(5) == 10
    assert calls2 == {"c": 0, "e": 1}, "the rebuilt wrapper compiled again"


def test_a_different_call_site_is_not_dragged_along():
    c1, e1, _ = _pair(_LIMIT_ERROR("recompile_limit reached"))
    _fall_back_to_eager_on_recompile_limit(c1, e1, "C.forward")(1)

    c2, e2, calls2 = _pair()
    assert _fall_back_to_eager_on_recompile_limit(c2, e2, "B.forward")(5) == 10
    assert calls2 == {"c": 1, "e": 0}


def test_a_deferred_switch_survives_the_wrapper_too():
    """The deferred switch (bumped retry succeeds) must be recorded by label, not in wrapper state: pending_eager died
    with GRPO's accumulate_chunk before the boundary."""
    U = _utils()
    c, e, _ = _pair(_LIMIT_ERROR("recompile_limit reached"))
    w = _fall_back_to_eager_on_recompile_limit(c, e, "C.forward")
    w(1)
    assert "C.forward" in (U._LATCHED_EAGER_LABELS | U._PENDING_EAGER_LABELS)


def test_the_boundary_settles_a_deferred_label_with_no_live_wrapper():
    """apply_pending_eager_fallbacks must check pending labels, not only live wrappers, so a label whose wrapper was already collected still settles."""
    U = _utils()
    U._PENDING_EAGER_LABELS.add("C.forward")
    U.apply_pending_eager_fallbacks()
    assert "C.forward" in U._LATCHED_EAGER_LABELS
    assert "C.forward" not in U._PENDING_EAGER_LABELS

    c, e, calls = _pair()
    assert _fall_back_to_eager_on_recompile_limit(c, e, "C.forward")(5) == 10
    assert calls == {"c": 0, "e": 1}, "the rebuilt wrapper compiled again"


def test_a_rebuilt_wrapper_inherits_the_deferral_not_the_latch():
    """A rebuilt wrapper inherits the pending deferral, not a latch: the label stays pending until the step boundary,
    because deferring means the switch waits for it."""
    U = _utils()
    U._PENDING_EAGER_LABELS.add("B.forward")
    c, e, _ = _pair()
    w = _fall_back_to_eager_on_recompile_limit(c, e, "B.forward")
    assert w._unsloth_fallback_state["pending_eager"] is True
    assert w._unsloth_fallback_state["eager"] is False


def test_an_untouched_call_site_is_not_settled_by_someone_elses_boundary():
    U = _utils()
    U._PENDING_EAGER_LABELS.add("A.forward")
    U.apply_pending_eager_fallbacks()
    c, e, calls = _pair()
    assert _fall_back_to_eager_on_recompile_limit(c, e, "B.forward")(5) == 10
    assert calls == {"c": 1, "e": 0}


def test_a_successful_retry_outside_a_checkpoint_settles_itself():
    """Defer only inside checkpointed training, where a mid-step flip breaks pack/recompute; outside a region no
    boundary comes, as only the trainer's step hook calls apply_pending_eager_fallbacks."""
    with _isolated_budget() as (u, config, name, before):
        calls = {"c": 0}

        def compiled(x):
            calls["c"] += 1
            if calls["c"] == 1:
                raise _LIMIT_ERROR("recompile_limit reached")
            return x * 2

        def eager(x):
            return x * 2

        w = u._fall_back_to_eager_on_recompile_limit(compiled, eager, "M.forward")
        assert w(3) == 6, "the retry must still finish this call"

        assert getattr(config, name) == before, "limit left raised for the process"
        assert u._GLOBAL_BUMPS == 0, "shared bump allowance left spent"
        assert w._unsloth_fallback_state["eager"] is True, "switch never happened"

        assert w(3) == 6
        assert calls["c"] == 2, "the compiler was consulted after the switch"


def test_a_successful_retry_inside_a_checkpoint_still_defers():
    """Inside a non-reentrant region the switch stays deferred: what this step already packed compiled is recomputed in backward, and flipping now would mismatch it."""
    from torch.utils.checkpoint import checkpoint

    with _isolated_budget() as (u, config, name, before):
        if u._in_non_reentrant_checkpoint() is None:
            pytest.skip("this torch cannot report checkpoint regions")
        calls = {"c": 0}

        def compiled(x):
            calls["c"] += 1
            if calls["c"] == 1:
                raise _LIMIT_ERROR("recompile_limit reached")
            return x * 2

        def eager(x):
            return x * 2

        w = u._fall_back_to_eager_on_recompile_limit(compiled, eager, "M.forward")
        x = torch.randn(2, 2, requires_grad = True)
        checkpoint(lambda t: w(t).sum(), x, use_reentrant = False).backward()

        state = w._unsloth_fallback_state
        assert state["pending_eager"] is True, "the debt was not recorded"
        assert state["eager"] is False, "flipped inside a half-packed region"


@contextlib.contextmanager
def _pre_2_8_probe(walk):
    """The per-call probe on a torch with no hook accessor, counting its walks."""
    U = _utils()
    real_accessor, real_walk = U._saved_tensor_hook_accessor, U._walk_for_checkpoint_frame
    saved_flag, saved_misses = U._PACKED_COMPILED_IN_CHECKPOINT, U._CHECKPOINT_PROBE_MISSES
    calls = {"n": 0}
    def _counting():
        calls["n"] += 1
        return walk()
    U._saved_tensor_hook_accessor = lambda: None
    U._walk_for_checkpoint_frame = _counting
    U._PACKED_COMPILED_IN_CHECKPOINT, U._CHECKPOINT_PROBE_MISSES = False, 0
    try:
        yield U, calls
    finally:
        U._saved_tensor_hook_accessor, U._walk_for_checkpoint_frame = real_accessor, real_walk
        U._PACKED_COMPILED_IN_CHECKPOINT = saved_flag
        U._CHECKPOINT_PROBE_MISSES = saved_misses


def test_the_probe_does_not_walk_the_stack_under_inference_mode():
    """The probe skips the stack walk under inference_mode: it records no autograd graph, so no backward is owed; generation runs under it for most GRPO calls."""
    with _pre_2_8_probe(lambda: True) as (U, calls):
        with torch.inference_mode():
            for _ in range(8): U._note_packed_under_checkpoint()
        assert calls["n"] == 0
        assert U._PACKED_COMPILED_IN_CHECKPOINT is False
        U._note_packed_under_checkpoint()
        assert calls["n"] == 1 and U._PACKED_COMPILED_IN_CHECKPOINT is True


def test_the_probe_still_fires_with_grad_off_inside_a_function_forward():
    """Do not gate the probe on is_grad_enabled(): autograd.Function.forward runs with grad off, and Unsloth's gradient
    checkpointing is such a Function, so gating skipped the probe."""
    with _pre_2_8_probe(lambda: True) as (U, calls):

        class _Region(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                assert not torch.is_grad_enabled(), "premise gone: grad is on here"
                U._note_packed_under_checkpoint()
                return x * 2

            @staticmethod
            def backward(ctx, g):
                return g

        _Region.apply(torch.randn(4, requires_grad = True)).sum().backward()
        assert calls["n"] == 1, "the probe never ran inside the Function forward"
        assert U._PACKED_COMPILED_IN_CHECKPOINT is True


def test_the_probe_stops_walking_after_a_step_of_fruitless_walks():
    """Without non-reentrant checkpointing the marker never latches, so with no accessor the walk would run on every
    call; the probe stops after a per-step miss budget."""
    with _pre_2_8_probe(lambda: False) as (U, calls):
        for _ in range(U._CHECKPOINT_PROBE_MISS_BUDGET * 4):
            U._note_packed_under_checkpoint()
        assert calls["n"] == U._CHECKPOINT_PROBE_MISS_BUDGET


def test_a_step_boundary_re_arms_the_probe():
    """The probe miss budget bounds one step, not the run: a model that only reaches its checkpointed layers later must still be seen next step."""
    with _pre_2_8_probe(lambda: False) as (U, calls):
        for _ in range(U._CHECKPOINT_PROBE_MISS_BUDGET + 5):
            U._note_packed_under_checkpoint()
        spent = calls["n"]
        U.apply_pending_eager_fallbacks()
        U._note_packed_under_checkpoint()
        assert calls["n"] == spent + 1


def test_a_latch_clears_the_probe_budget():
    """A latch clears the probe miss count, so misses spent before the region opened do not count against the next step."""
    with _pre_2_8_probe(lambda: U._CHECKPOINT_PROBE_MISSES >= 3) as (U, calls):
        for _ in range(4): U._note_packed_under_checkpoint()
        assert U._PACKED_COMPILED_IN_CHECKPOINT is True
        assert U._CHECKPOINT_PROBE_MISSES == 0


def test_the_budget_never_delays_a_checkpointed_run():
    """A checkpointed run latches on one of its first calls (layer 0 is already inside a region), so it never spends the probe budget."""
    with _pre_2_8_probe(lambda: True) as (U, calls):
        for _ in range(200): U._note_packed_under_checkpoint()
        assert calls["n"] == 1                          # latched, then returns early
        assert U._CHECKPOINT_PROBE_MISSES == 0
