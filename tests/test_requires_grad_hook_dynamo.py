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

"""Tests _run_eagerly_under_compile in peft_utils.py.

The gradient-checkpointing hooks call `requires_grad_()`, which Dynamo cannot
trace, so a compiled model carrying one dies with

    Unsupported: Unsupported Tensor.requires_grad_() call

Making the hooks opaque to Dynamo fixes that, but has a second-order hazard:
register_other_hooks() decides which hooks are OURS by matching
__name__/__qualname__ against "requires_grad_pre_hook" and
"requires_grad_post_hook". If wrapping lost those names, our hooks would stop
being recognised and would be re-registered on top of themselves. torch 2.9
carries them through torch._dynamo.disable, but that is an internal detail of
a private module and this has to hold from torch 2.6 up, so the wrapper copies
them explicitly.

No GPU needed.
"""

import ast
import sys
import types
from pathlib import Path

import pytest
import torch

PEFT_UTILS = Path(__file__).resolve().parents[1] / "unsloth_zoo" / "peft_utils.py"
_SRC = PEFT_UTILS.read_text(encoding = "utf-8")


def _load():
    for node in ast.parse(_SRC).body:
        if isinstance(node, ast.FunctionDef) and node.name == "_run_eagerly_under_compile":
            ns = {"torch": torch}
            exec(ast.get_source_segment(_SRC, node), ns)
            return ns[node.name]
    raise AssertionError("_run_eagerly_under_compile not found in peft_utils.py")


run_eagerly = _load()


def _hook(module, args, kwargs):
    return "called"


_hook.__name__ = "requires_grad_pre_hook"
_hook.__qualname__ = "requires_grad_for_gradient_checkpointing.<locals>.requires_grad_pre_hook"


def test_names_survive_wrapping():
    # register_other_hooks matches on these; losing them re-registers our
    # hooks on top of themselves.
    w = run_eagerly(_hook)
    assert w.__name__ == "requires_grad_pre_hook"
    assert "requires_grad_pre_hook" in w.__qualname__


def test_wrapped_hook_still_runs():
    assert run_eagerly(_hook)(None, (), {}) == "called"


def test_register_other_hooks_still_recognises_it():
    # Mirrors the matching in register_other_hooks exactly.
    w = run_eagerly(_hook)
    name1 = name2 = "requires_grad_pre_hook"
    qualname = getattr(w, "__qualname__", "")
    name = getattr(w, "__name__", "")
    recognised = (name1 in qualname or name2 in qualname) or (name2 in name)
    assert recognised, "our own hook would be treated as a foreign hook"


@pytest.fixture
def _fake_dynamo():
    # `import torch._dynamo as x` resolves the ATTRIBUTE on the torch package
    # once the submodule is in sys.modules, so both have to be swapped.
    saved_mod = sys.modules.get("torch._dynamo")
    saved_attr = getattr(torch, "_dynamo", None)

    def install(mod):
        sys.modules["torch._dynamo"] = mod
        torch._dynamo = mod

    yield install

    if saved_mod is None: sys.modules.pop("torch._dynamo", None)
    else: sys.modules["torch._dynamo"] = saved_mod
    if saved_attr is not None: torch._dynamo = saved_attr


def test_falls_back_when_disable_is_absent(_fake_dynamo):
    # Older / stripped torch builds: must be a no-op, not a crash.
    _fake_dynamo(types.ModuleType("torch._dynamo"))
    assert run_eagerly(_hook) is _hook


def test_falls_back_when_disable_raises(_fake_dynamo):
    mod = types.ModuleType("torch._dynamo")
    def _boom(fn): raise RuntimeError("nope")
    mod.disable = _boom
    _fake_dynamo(mod)
    assert run_eagerly(_hook) is _hook


def test_tolerates_unsettable_attributes(_fake_dynamo):
    # A C-implemented or slotted callable may refuse __name__ assignment.
    class _Callable:
        __slots__ = ()
        def __call__(self, *a, **k): return "called"
    mod = types.ModuleType("torch._dynamo")
    mod.disable = lambda fn: _Callable()
    _fake_dynamo(mod)
    w = run_eagerly(_hook)          # must not raise
    assert w(None, (), {}) == "called"


def test_both_hooks_are_decorated_in_source():
    for hook in ("requires_grad_pre_hook", "requires_grad_post_hook"):
        i = _SRC.index(f"def {hook}(")
        preceding = _SRC[:i].rstrip().splitlines()[-1]
        assert "_run_eagerly_under_compile" in preceding, hook


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
