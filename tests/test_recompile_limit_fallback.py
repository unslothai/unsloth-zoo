"""Exhausting the recompile cache must slow a run down, not end it.

`patch_function(..., fullgraph = True)` compiles with fullgraph, and Dynamo
raises on cache exhaustion under fullgraph instead of falling back:

    FailOnRecompileLimitHit: recompile_limit reached with fullgraph=True

Running out of compilation cache is a performance problem, and turning it into
a hard training failure is strictly worse than being slow.

The fallback is narrow on purpose. A genuine graph break under fullgraph
still raises, because that is a correctness signal about the patch itself.
"""

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
    assert calls == {"c": 1, "e": 1}


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
    assert calls["c"] == 1, "the compiler must not be re-entered after the latch"
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
    with torch._dynamo.config.patch(recompile_limit = 2,
                                    accumulated_recompile_limit = 2):
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
