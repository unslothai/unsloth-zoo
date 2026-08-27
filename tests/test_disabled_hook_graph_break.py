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

"""Two Unsloth fixes collide; the run should not die of it.

One wraps the gradient-checkpointing `requires_grad` hooks in
`torch.compiler.disable`, because Dynamo cannot trace
`Tensor.requires_grad_()`. The other compiles a forward with
`fullgraph = True`. The disabled hook is then invoked from inside that
fullgraph region and Dynamo refuses:

    Unsupported: Skip calling `torch.compiler.disable()`d function

Neither fix is wrong on its own and a user can do nothing about the
combination, so it should cost speed rather than the run.

The hazard guarded against here is over-catching. The fallback takes exactly
this one extra case, matched on the disable signature, and most of the tests
below are about the graph breaks that must STILL raise.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unsloth_zoo.temporary_patches import utils as U  # noqa: E402

DISABLE_MSG = (
    "Skip calling `torch.compiler.disable()`d function\n"
    "  Explanation: Skip calling function "
    "`<function requires_grad_for_gradient_checkpointing.<locals>."
    "requires_grad_pre_hook at 0x7f00>` since it was wrapped with "
    "`torch.compiler.disable` (reason: None)\n"
    "  Hint: Remove the `torch.compiler.disable` call"
)


# ---- the matcher ----------------------------------------------------------

def test_it_recognises_our_own_disabled_hook():
    assert U._is_our_own_disabled_hook(RuntimeError(DISABLE_MSG))


_HOOK = ("<function requires_grad_for_gradient_checkpointing.<locals>."
         "requires_grad_pre_hook at 0x7f00>")

# Dynamo's refusal to trace a disabled callable, once per wording inside the
# torch>=2.4,<2.13 pyproject declares. 2.4 to 2.6 emit the `_dynamo.disable`
# text from `variables/functions.py` (2.4.0 L604, 2.5.1 L655, 2.6.0 L620); 2.7
# replaced it with the structured `unimplemented_v2` block (2.7.0 L1173) and 2.8
# added the trailing reason. Missing a wording means the run dies on that torch
# instead of slowing down, which is exactly what this fallback exists to stop.
_OLD_WORDING = f"call torch._dynamo.disable() wrapped function {_HOOK}"
_NEW_WORDING = (
    "Skip calling `torch.compiler.disable()`d function\n"
    f"  Explanation: Skip calling function `{_HOOK}` since it was wrapped "
    "with `torch.compiler.disable`\n"
    "  Hint: Remove the `torch.compiler.disable` call\n"
    f"\n  Developer debug context: {_HOOK}\n"
)


@pytest.mark.parametrize("versions,message", [
    ("2.4 / 2.5 / 2.6", _OLD_WORDING),
    ("2.7", _NEW_WORDING),
    ("2.8+", DISABLE_MSG),
])
def test_every_supported_torch_wording_is_recognised(versions, message):
    assert U._is_our_own_disabled_hook(RuntimeError(message)), versions
    def compiled(*a, **k):
        raise _unsupported(message)
    assert _wrap(compiled)() == "eager", versions


def test_an_ordinary_graph_break_is_not_matched():
    """The whole point. These must keep raising."""
    for text in (
        "Unsupported: call_function BuiltinVariable(print)",
        "Unsupported Tensor.requires_grad_() call",
        "Dynamo failed to trace a data-dependent branch",
        "graph break in user code",
    ):
        assert not U._is_our_own_disabled_hook(RuntimeError(text)), text


def test_a_mention_of_disable_alone_is_not_enough():
    """Both halves of the signature are required, so a message that merely
    talks about torch.compiler.disable does not qualify."""
    assert not U._is_our_own_disabled_hook(
        RuntimeError("consider using torch.compiler.disable here"))
    # Same for the pre-2.7 wording: the whole literal is required, not the
    # decorator's name, so advice to apply it is not a disabled-hook break.
    assert not U._is_our_own_disabled_hook(
        RuntimeError("try torch._dynamo.disable() on this function"))


def test_an_unstringifiable_exception_does_not_crash():
    class Bad(Exception):
        def __str__(self):
            raise ValueError("nope")
    assert U._is_our_own_disabled_hook(Bad()) is False


# ---- the fallback ---------------------------------------------------------

def _wrap(compiled, eager=None):
    eager = eager or (lambda *a, **k: "eager")
    return U._fall_back_to_eager_on_recompile_limit(compiled, eager, "TestMod")


def _unsupported(msg):
    import torch._dynamo.exc as exc
    cls = getattr(exc, "Unsupported", None)
    if cls is None:
        pytest.skip("this torch has no torch._dynamo.exc.Unsupported")
    try:
        return cls(msg)
    except Exception:
        pytest.skip("Unsupported cannot be constructed on this torch")


def test_our_disabled_hook_falls_back_to_eager():
    def compiled(*a, **k):
        raise _unsupported(DISABLE_MSG)
    assert _wrap(compiled)() == "eager"


def test_the_fallback_latches():
    """Do not reverse this. Per-call retry was tried and does not keep a
    checkpoint pack and its own recompute in the same mode: they run under
    different guards, so the compiler can flip either way. Latching leaves
    exactly one inconsistent step, which unsloth catches and retries via
    `force_eager_fallback`, instead of an unbounded number.
    """
    calls = {"n": 0}

    def compiled(*a, **k):
        calls["n"] += 1
        raise _unsupported(DISABLE_MSG)

    w = _wrap(compiled)
    w(); w(); w()
    assert calls["n"] == 1, "the compiler must not be re-entered after the latch"


def test_every_later_call_takes_the_same_path():
    """The property the checkpoint depends on, in the form that survived: once
    the switch has happened, it is total. A build that sends some calls eager
    and some compiled is the configuration that aborts the backward."""
    outcomes = iter(["ok", "fail", "ok", "ok"])
    seen = []

    def compiled(*a, **k):
        which = next(outcomes)
        seen.append(which)
        if which == "fail":
            raise _unsupported(DISABLE_MSG)
        return "compiled"

    w = _wrap(compiled)
    assert w() == "compiled"
    assert w() == "eager"   # the call that exhausted the cache
    assert w() == "eager"   # latched: no second attempt at the compiler
    assert w() == "eager"
    assert seen == ["ok", "fail"]


def test_it_warns_once_and_not_per_call(caplog):
    """The condition repeats every call; the log must not."""
    import logging

    def compiled(*a, **k):
        raise _unsupported(DISABLE_MSG)

    w = _wrap(compiled)
    with caplog.at_level(logging.WARNING):
        w(); w(); w(); w()
    warnings = [r for r in caplog.records if "eagerly" in r.getMessage()]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]


def test_a_real_graph_break_still_raises():
    """The property the narrow match exists to protect."""
    def compiled(*a, **k):
        raise _unsupported("Unsupported: call_function on a data-dependent value")
    with pytest.raises(Exception) as ei:
        _wrap(compiled)()
    assert "data-dependent" in str(ei.value)


def test_an_unrelated_exception_still_raises():
    def compiled(*a, **k):
        raise ValueError("something else entirely")
    with pytest.raises(ValueError):
        _wrap(compiled)()


def test_a_successful_compile_is_untouched():
    assert _wrap(lambda *a, **k: "compiled")() == "compiled"


def test_arguments_reach_the_eager_function():
    def compiled(*a, **k):
        raise _unsupported(DISABLE_MSG)
    w = _wrap(compiled, eager=lambda x, y=0: x + y)
    assert w(3, y=4) == 7


# ---- what must be preserved ----------------------------------------------

def test_the_recompile_limit_fallback_still_works():
    errs = U._recompile_limit_errors()
    if not errs:
        pytest.skip("no recompile-limit exceptions on this torch")

    def compiled(*a, **k):
        raise errs[0]("recompile_limit reached with fullgraph=True")
    assert _wrap(compiled)() == "eager"


def test_the_compiled_callable_stays_reachable():
    """Anything that unwraps the wrapper must still find it."""
    def compiled(*a, **k):
        return "compiled"
    compiled.get_compiler_config = lambda: {}
    w = _wrap(compiled)
    assert w._unsloth_compiled_func is compiled
    assert hasattr(w, "get_compiler_config")


def test_it_degrades_to_the_compiled_function_when_torch_offers_none_of_them():
    """On a torch with no such exceptions at all, wrapping buys nothing and
    must not add a layer.

    All THREE families have to be absent. The wrapper now also catches Inductor
    codegen refusals, and `_backend_compile_errors()` is non-empty on every
    supported torch, so mocking only the two recompile families left the third
    live and the wrapper rightly kept its layer -- this asserted "no exceptions
    at all" while one family was still there.
    """
    import unittest.mock as mock
    with mock.patch.object(U, "_recompile_limit_errors", lambda: ()), \
         mock.patch.object(U, "_disabled_hook_graph_break_error", lambda: ()), \
         mock.patch.object(U, "_backend_compile_errors", lambda: ()):
        sentinel = object()
        assert U._fall_back_to_eager_on_recompile_limit(
            sentinel, lambda: None, "x") is sentinel


def test_backend_errors_alone_are_enough_to_keep_the_wrapper():
    """The other half of the contract, and the reason the test above changed.

    A torch that reports no recompile-limit exception and no graph-break
    exception can still refuse to generate code, so the layer has to stay --
    otherwise the codegen fallback would silently not exist there.
    """
    import unittest.mock as mock
    if not U._backend_compile_errors():
        pytest.skip("this torch exposes no backend compile exception")
    with mock.patch.object(U, "_recompile_limit_errors", lambda: ()), \
         mock.patch.object(U, "_disabled_hook_graph_break_error", lambda: ()):
        sentinel = object()
        assert U._fall_back_to_eager_on_recompile_limit(
            sentinel, lambda: None, "x") is not sentinel


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
