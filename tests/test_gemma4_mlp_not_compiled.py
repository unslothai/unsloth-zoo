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

The patch is installed with ``fullgraph = None``, which `patch_function` reads as
"do not compile" (it only compiles when ``fullgraph`` is a bool). That drops the
direct `torch.compile` wrapper this call site used to install, so the MLP stops
being its own compile entry point with its own recompile cache.

It is deliberately NOT an eager boundary, and there is a test below pinning that
too: a compiled caller still inlines this body into its own graph, exactly as it
inlined the wrapped version before. Anything claiming this function is "outside
Dynamo" is wrong, and `test_compiled_caller_still_inlines_the_mlp` says so.

``None`` is a sentinel rather than an explicit argument, so two things can quietly
undo it: flipping the call site back to a bool, or redefining what `patch_function`
does with ``None``. There is a test here for each, plus one that the fp16 overflow
clamp the patch actually exists for still installs and still clamps.

Synthetic classes throughout, so this runs on any installed transformers, including
4.57.6 which has no gemma4 at all.
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

    Matched on all three positional operands, not just the class: a future second
    `Gemma4TextMLP` patch call would otherwise satisfy this test while the forward
    call site it is meant to guard had quietly changed.
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
    """`None` must keep meaning "leave it eager", whatever else changes.

    Watches the call rather than inspecting the result: `_is_dynamo_artifact`
    reports what an object looks like, which is not the same as whether
    torch.compile ran.
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

    A hard assertion, not a skip: if this file could skip here it would go green
    on a machine where compilation silently stopped working, which is exactly the
    condition under which the sentinel test above proves nothing.
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
    """`None` skips `unwrap_already_compiled` as well as the compile, so it means
    "do not compile or unwrap" rather than "force eager". Pinned so a future
    refactor cannot pass a pre-compiled callable and still look correct."""

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


def _install_mlp_patch(monkeypatch):
    """Run patch_Gemma4TextMLP against a synthetic MLP and return the class.

    On a transformers that ships gemma4 the synthetic class is swapped into the real
    module, because `patch_Gemma4TextMLP` imports that module by name and the real
    package would otherwise win over an injected one. Where gemma4 does not exist
    (4.57.6) a stand-in module is registered instead.
    """
    try:
        modeling = importlib.import_module(DOTTED)
        injected = False
    except ModuleNotFoundError as e:
        # Only the absence of gemma4 itself may fall through to the stand-in.
        # A broader except would let a genuinely broken gemma4 import be masked
        # by the synthetic class, and this file would pass without ever touching
        # the code it is meant to guard.
        if not (e.name or "").startswith(PARENT) and e.name != "transformers":
            raise
        modeling = types.ModuleType(DOTTED)
        injected = True

    monkeypatch.setattr(modeling, "Gemma4TextMLP", _StockMLP, raising=False)
    if injected:
        # `import a.b.c as m` walks the whole chain, so the parent package has to
        # resolve too or the patch takes its ImportError early return and no-ops.
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

    Dynamo inlines an undecorated callee into its caller's graph, so under a
    compiled parent the MLP still runs compiled -- as it did before this call site
    changed, because Dynamo traces straight through an inner torch.compile wrapper
    too. Pinned so nobody documents this patch as an eager boundary.
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


def test_checkpoint_pack_and_recompute_agree(monkeypatch):
    """The failure this patch was written against is a pack/recompute divergence,
    so check the two passes actually agree rather than only that nothing raised.
    """
    cls, stock = _install_mlp_patch(monkeypatch)
    try:
        torch.manual_seed(0)
        model = cls()
        passes = []

        def meta(t):
            return (tuple(t.shape), t.dtype, t.device.type, tuple(t.stride()))

        def block(x):
            # Registered before the body: with early stop the recompute is
            # abandoned partway through, so a trailing append would never run.
            seen = []
            passes.append(seen)
            out = x + cls.forward(model, x)
            seen.append(meta(out))
            return out

        x = torch.randn(4, 8, requires_grad=True)
        with torch.utils.checkpoint.set_checkpoint_early_stop(False):
            y = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            y.sum().backward()

        assert len(passes) == 2, f"expected a pack pass and a recompute, saw {len(passes)}"
        assert passes[0] == passes[1], f"pack/recompute diverged: {passes[0]} vs {passes[1]}"
        assert torch.isfinite(x.grad).all()
    finally:
        cls.forward = stock
