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

"""Retry stays atomic: re-running compiled_func is safe only because fullgraph=True raises at frame
entry before any op runs; non-atomic callables corrupt grads under non-reentrant recompute."""

import inspect
import sys
from pathlib import Path

import torch
import torch.utils.checkpoint as checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unsloth_zoo.temporary_patches import utils as U  # noqa: E402


def test_the_only_production_entry_point_compiles_the_function_directly():
    """Must compile the function itself, not a caller of one, or the retry stops being atomic."""
    source = inspect.getsource(U.compile_with_eager_fallback)
    assert "_raw_torch_compile(" in source, (
        "compile_with_eager_fallback no longer compiles the function itself"
    )
    assert "_fall_back_to_eager_on_recompile_limit(compiled, func, label)" in source, (
        "the fallback is no longer wrapped around the direct compile of `func`; "
        "if it now wraps a callable that CALLS a compiled one, the recompile-limit "
        "retry re-runs work that already happened -- see this module's docstring"
    )


def test_nothing_else_in_the_package_applies_the_fallback():
    """One production entry point, so one shape to keep atomic: a second call site would need its own check, and this test notices one arriving."""
    package = Path(U.__file__).resolve().parent
    callers = sorted(
        path.relative_to(package).as_posix()
        for path in package.rglob("*.py")
        if "_fall_back_to_eager_on_recompile_limit(" in path.read_text(encoding = "utf-8")
    )
    assert callers == ["utils.py"], (
        f"_fall_back_to_eager_on_recompile_limit is applied in {callers}; every "
        f"new call site has to hand it a direct compile of the eager function, "
        f"or the retry re-executes work inside a checkpoint recomputation"
    )


def _run_region(restart_once, trailing):
    """Checkpointed region whose inner call may restart; eager, so fast and deterministic."""
    state = {"recomputing": False, "fired": False}

    def inner(x):
        a = torch.sin(x)
        if state["recomputing"] and restart_once and not state["fired"]:
            state["fired"] = True
            raise RuntimeError("stand-in for FailOnRecompileLimitHit")
        return torch.cos(a) * 1.5

    def retrying(x):
        try:
            return inner(x)
        except RuntimeError:
            return inner(x)             # from the START, exactly as the retry does

    def region(x):
        out = retrying(retrying(x))
        for _ in range(trailing):
            out = torch.tanh(out)
        return out

    torch.manual_seed(0)
    x = torch.randn(4, 4).requires_grad_(True)
    out = checkpoint.checkpoint(region, x, use_reentrant = False)
    state["recomputing"] = True         # everything past here is the recompute
    out.sum().backward()
    return x.grad.clone()


def test_restarting_a_call_inside_a_recompute_silently_corrupts_gradients():
    """A restart inside recompute raises no error, yet gradients silently disagree."""
    for trailing in (0, 1, 3, 8):
        clean = _run_region(restart_once = False, trailing = trailing)
        restarted = _run_region(restart_once = True, trailing = trailing)
        assert not torch.allclose(clean, restarted), (
            f"trailing={trailing}: restarting a call during the recomputation "
            f"produced the same gradients as not restarting it. torch may have "
            f"started resetting recomp_counter, which would make the atomicity "
            f"requirement above unnecessary -- check before deleting anything."
        )
