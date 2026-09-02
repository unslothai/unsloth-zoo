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

"""CPU tests pinning Gemma4TextMLP.forward to the NON-compiled patch path.

``fullgraph = None`` reads as "do not compile" in `patch_function`, which only
compiles on a bool. It is a sentinel, so it can regress from either side: the call
site flipping to a bool, or `patch_function` redefining None. A test for each, plus
the fp16 clamp the patch exists for.

Not an eager boundary: a compiled caller still inlines this body, exactly as it
inlined the wrapped version. `test_compiled_caller_still_inlines_the_mlp` pins that
so it is not misread the other way.

Synthetic classes throughout, so this runs on any transformers, including 4.57.6
which has no gemma4 at all.
"""

import ast
import importlib
import inspect
import pathlib
import sys
import types

import pytest
import torch

from unsloth_zoo.temporary_patches import gemma4 as gemma4_patches
from unsloth_zoo.temporary_patches.utils import patch_function


def _is_dynamo_artifact(fn):
    """True if `fn` came back from torch.compile, however it is carried."""
    return (
        "dynamo" in (type(fn).__module__ or "")
        or hasattr(fn, "_torchdynamo_orig_callable")
        or type(fn).__name__ == "OptimizedModule"
    )


def _mlp_patch_call():
    """The ``patch_function(Gemma4TextMLP, "forward", forward, ...)`` call node.

    All three operands are matched, so a second Gemma4TextMLP patch call cannot
    satisfy this while the forward call site it guards has changed.
    """
    source = pathlib.Path(inspect.getfile(gemma4_patches)).read_text()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", None) != "patch_function":
            continue
        if len(node.args) < 3:
            continue
        target, attr, replacement = node.args[0], node.args[1], node.args[2]
        if getattr(target, "id", None) != "Gemma4TextMLP":
            continue
        if not (isinstance(attr, ast.Constant) and attr.value == "forward"):
            continue
        if getattr(replacement, "id", None) != "forward":
            continue
        return node
    return None


def test_call_site_passes_fullgraph_none():
    """The call site must keep asking for the eager path."""
    call = _mlp_patch_call()
    assert call is not None, (
        'patch_function(Gemma4TextMLP, "forward", forward, ...) call not found'
    )

    keywords = {kw.arg: kw.value for kw in call.keywords}
    assert "fullgraph" in keywords, (
        "Gemma4TextMLP.forward must pass fullgraph explicitly so the eager "
        "choice is visible at the call site"
    )
    value = keywords["fullgraph"]
    assert isinstance(value, ast.Constant) and value.value is None, (
        "Gemma4TextMLP.forward is deliberately not compiled; fullgraph must stay "
        f"None, got {ast.dump(value)}"
    )


def test_patch_function_treats_none_as_do_not_compile(monkeypatch):
    """`None` must keep meaning "leave it eager".

    Watches the call, not the result: `_is_dynamo_artifact` reports what an object
    looks like, not whether torch.compile ran.
    """

    class Target:
        def forward(self, x: torch.Tensor):
            return x

    def replacement(self, x: torch.Tensor):
        return x * 2

    calls = []
    monkeypatch.setattr(torch, "compile", lambda *a, **k: calls.append((a, k)))

    assert patch_function(Target, "forward", replacement, fullgraph=None)
    assert calls == [], (
        "fullgraph=None must not call torch.compile; the Gemma-4 MLP patch relies on this"
    )
    assert Target.forward is replacement, "the exact callable handed in must be installed"
    assert not _is_dynamo_artifact(Target.forward)
    assert Target.forward(None, torch.ones(2)).tolist() == [2.0, 2.0]


def test_patch_function_bool_still_compiles():
    """Guards the test above from passing because compilation broke everywhere.

    Asserts rather than skips: a skip here would go green on exactly the machine
    where the sentinel test proves nothing.
    """

    class Target:
        def forward(self, x: torch.Tensor):
            return x

    def replacement(self, x: torch.Tensor):
        return x * 2

    assert patch_function(Target, "forward", replacement, fullgraph=False)
    assert _is_dynamo_artifact(Target.forward), (
        "fullgraph=False did not produce a compiled callable, so None vs bool "
        "cannot be told apart and the sentinel test above is vacuous"
    )


def test_patch_function_none_does_not_unwrap_either():
    """`None` skips `unwrap_already_compiled` too, so it means "do not compile or
    unwrap", not "force eager". Pinned so a refactor cannot pass a pre-compiled
    callable and still look correct."""

    class Target:
        def forward(self, x: torch.Tensor):
            return x

    def replacement(self, x: torch.Tensor):
        return x * 2

    precompiled = torch.compile(replacement, dynamic=True)
    assert patch_function(Target, "forward", precompiled, fullgraph=None, force=True)
    assert Target.forward is precompiled


class _StockMLP(torch.nn.Module):
    """Mirrors transformers' Gemma4TextMLP shape closely enough to patch."""

    def __init__(self, hidden=8, intermediate=16):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = torch.nn.Linear(intermediate, hidden, bias=False)
        self.act_fn = torch.nn.functional.gelu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


PARENT = "transformers.models.gemma4"
DOTTED = f"{PARENT}.modeling_gemma4"
# The import chain itself, and nothing below it.
_ABSENT_OK = frozenset(("transformers", "transformers.models", PARENT, DOTTED))


def _install_mlp_patch(monkeypatch):
    """Run patch_Gemma4TextMLP against a synthetic MLP and return the class.

    Where gemma4 exists the synthetic class is swapped into the real module, since
    the patch imports that module by name. Where it does not (4.57.6), a stand-in
    module is registered instead.
    """
    try:
        modeling = importlib.import_module(DOTTED)
        injected = False
    except ModuleNotFoundError as e:
        # Only gemma4's own absence may reach the stand-in, matched on the exact
        # chain. A prefix match would also swallow a missing dependency *under*
        # gemma4, masking a real broken import with the synthetic class.
        if e.name not in _ABSENT_OK:
            raise
        modeling = types.ModuleType(DOTTED)
        injected = True

    monkeypatch.setattr(modeling, "Gemma4TextMLP", _StockMLP, raising=False)
    if injected:
        # `import a.b.c as m` walks the chain, so the parent has to resolve too or
        # the patch takes its ImportError early return and no-ops.
        package = types.ModuleType(PARENT)
        package.modeling_gemma4 = modeling
        monkeypatch.setitem(sys.modules, PARENT, package)
        monkeypatch.setitem(sys.modules, DOTTED, modeling)

    stock = _StockMLP.forward
    gemma4_patches.patch_Gemma4TextMLP()
    return _StockMLP, stock


def test_mlp_patch_installs_eager_and_clamps_fp16(monkeypatch):
    """The reason the patch exists must survive the eager path."""
    cls, stock = _install_mlp_patch(monkeypatch)
    try:
        assert cls.forward is not stock, "the fp16 clamp did not install"
        assert not _is_dynamo_artifact(cls.forward)

        model = cls().to(torch.float16)
        with torch.no_grad():
            for proj in (model.gate_proj, model.up_proj, model.down_proj):
                proj.weight.fill_(6.0)
            x = torch.full((4, 8), 20.0, dtype=torch.float16)
            assert not torch.isfinite(stock(model, x)).all(), (
                "test input no longer overflows fp16, so it proves nothing"
            )
            assert torch.isfinite(cls.forward(model, x)).all()
    finally:
        cls.forward = stock


def test_mlp_patch_is_bit_exact_off_the_fp16_path(monkeypatch):
    """bf16/fp32 take the early return and must be untouched."""
    cls, stock = _install_mlp_patch(monkeypatch)
    try:
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(0)
            model = cls().to(dtype)
            x = torch.randn(4, 8, dtype=dtype)
            with torch.no_grad():
                assert torch.equal(stock(model, x), cls.forward(model, x)), (
                    f"{dtype} output drifted from upstream"
                )
    finally:
        cls.forward = stock


def test_compiled_caller_still_inlines_the_mlp(monkeypatch):
    """Not compiling this function is not the same as excluding it from compilation.

    Dynamo inlines an undecorated callee, so under a compiled parent the MLP still
    runs compiled, as it did when wrapped. Also fails if anyone adds an explicit
    `torch.compiler.disable` here, which would break the graph in every layer.
    """
    cls, stock = _install_mlp_patch(monkeypatch)
    try:
        model = cls()
        graphs = []

        def backend(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        def parent(x):
            return x + cls.forward(model, torch.nn.functional.layer_norm(x, (8,)))

        torch._dynamo.reset()
        compiled = torch.compile(parent, backend=backend, dynamic=True)
        out = compiled(torch.randn(2, 8))

        assert torch.isfinite(out).all()
        assert len(graphs) == 1, f"expected one graph and no break, got {len(graphs)}"
        code = "\n".join(g.code for g in graphs)
        assert "linear" in code, "the MLP body is missing from the caller's graph"
    finally:
        cls.forward = stock



def test_stand_in_rejects_a_failure_below_gemma4(monkeypatch):
    """gemma4 being absent is a valid skip; gemma4 being broken is not.

    Both surface as ModuleNotFoundError, so the stand-in matches the exact import
    chain. A prefix match would substitute the synthetic class for a real broken
    import and every test here would pass on a module that never loaded.
    """
    def boom(name, *a, **k):
        raise ModuleNotFoundError(
            "no module named 'transformers.models.gemma4.some_dependency'",
            name="transformers.models.gemma4.some_dependency",
        )

    monkeypatch.setattr(importlib, "import_module", boom)
    with pytest.raises(ModuleNotFoundError):
        _install_mlp_patch(monkeypatch)
