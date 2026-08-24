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
"do not compile" (it only compiles when ``fullgraph`` is a bool). That keeps the
function off a compile surface whose mode can differ between a checkpoint's
saved-tensor pack and its recompute, which is what `torch.utils.checkpoint`
reports as "Recomputed values ... have different metadata".

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
    """The ``patch_function(Gemma4TextMLP, "forward", ...)`` call node."""
    source = pathlib.Path(inspect.getfile(gemma4_patches)).read_text()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", None) != "patch_function":
            continue
        if not node.args or getattr(node.args[0], "id", None) != "Gemma4TextMLP":
            continue
        return node
    return None


def test_call_site_passes_fullgraph_none():
    """The call site must keep asking for the eager path."""
    call = _mlp_patch_call()
    assert call is not None, "patch_function(Gemma4TextMLP, ...) call not found"

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


def test_patch_function_treats_none_as_do_not_compile():
    """`None` must keep meaning "leave it eager", whatever else changes."""

    class Target:
        def forward(self, x: torch.Tensor):
            return x

    def replacement(self, x: torch.Tensor):
        return x * 2

    assert patch_function(Target, "forward", replacement, fullgraph=None)
    assert not _is_dynamo_artifact(Target.forward), (
        "fullgraph=None must not compile; the Gemma-4 MLP patch relies on this"
    )
    assert Target.forward(None, torch.ones(2)).tolist() == [2.0, 2.0]


def test_patch_function_bool_still_compiles():
    """Guards the test above from passing because compilation broke everywhere."""

    class Target:
        def forward(self, x: torch.Tensor):
            return x

    def replacement(self, x: torch.Tensor):
        return x * 2

    assert patch_function(Target, "forward", replacement, fullgraph=False)
    if not _is_dynamo_artifact(Target.forward):
        pytest.skip("torch.compile unavailable here, so None vs bool cannot be told apart")


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
    except Exception:
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
