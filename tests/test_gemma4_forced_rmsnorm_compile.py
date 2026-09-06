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


"""The forced-float32 Gemma4 RMSNorm replacement must survive the compiler.

`patch_Gemma4RMSNorm` swaps `Gemma4RMSNorm.forward` for a wrapper that returns
fp16 (the sub-layer dtype the forced-float32 design wants) instead of the input
dtype. The compiler then regenerates every modeling class from source, and it
has two ways of handling a class whose forward was replaced:

  * replacement still named `forward`: emit the patched body as a standalone
    `<Class>_forward`, then splice a call to it over the ORIGINAL method's
    source text. That text is no longer what the class attribute holds, so the
    splice matches nothing and the generated class keeps the upstream forward.
  * replacement with any other name: locate `def forward` in the class text and
    replace it with a call to the renamed function.

The RMSNorm patch used the first spelling. On a 4-bit gemma-4 load with
UNSLOTH_FORCE_FLOAT32=1 the generated class therefore returned the fp32
residual unchanged into a bf16 q_proj, and the first matmul raised
"expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16".

Two guards, CPU only. The first drives the real `create_standalone_class`
against a fake modeling module in both spellings, so it documents the
compiler's behaviour rather than assuming it, and executes the generated code
to show the dtype actually changes hands. The second reads the patch's own
source so the spelling cannot quietly revert.
"""

import ast
import importlib.util
import inspect
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from unsloth_zoo import compiler
from unsloth_zoo.temporary_patches import gemma4_float32

REPO_ROOT = Path(__file__).resolve().parents[1]

# A stand-in for Gemma4RMSNorm: the upstream forward returns the input dtype,
# the forced replacement returns float16. Both replacement spellings live in
# the module so `inspect.getsource` can read them exactly as it reads the real
# patch, which is defined inside a function in gemma4_float32.py.
_FAKE_MODELING_SOURCE = """
    import torch
    from torch import nn


    def _fake_norm_kernel(hidden_states_2d, eps):
        x = hidden_states_2d.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim = True)
        return (x * torch.pow(variance + eps, -0.5)).to(torch.float16)


    class FakeRMSNorm(nn.Module):
        def __init__(self, dim: int = 4, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def _norm(self, hidden_states):
            return hidden_states * torch.pow(hidden_states.pow(2).mean(-1, keepdim = True) + self.eps, -0.5)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            original_body_ran = self._norm(hidden_states.float()) * self.weight.float()
            return original_body_ran.type_as(hidden_states)


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        forced_body_ran = _fake_norm_kernel(hidden_states.reshape(-1, hidden_states.shape[-1]), self.eps)
        return forced_body_ran.reshape(hidden_states.shape)


    def _fake_forced_rmsnorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        forced_body_ran = _fake_norm_kernel(hidden_states.reshape(-1, hidden_states.shape[-1]), self.eps)
        return forced_body_ran.reshape(hidden_states.shape)
"""


def _load_fake_modeling_module(tmp_path, monkeypatch, name):
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent(_FAKE_MODELING_SOURCE), encoding = "utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(compiler, name, module, raising = False)
    return module


def _generate_with_replacement(tmp_path, monkeypatch, name, replacement_attr):
    module = _load_fake_modeling_module(tmp_path, monkeypatch, name)
    # What patch_function does: the class attribute becomes the replacement.
    module.FakeRMSNorm.forward = getattr(module, replacement_attr)
    generated = compiler.create_standalone_class(
        "FakeRMSNorm", name, dir(module), disable = True,
    )
    return module, generated


def _run_generated(generated, name, module):
    """Execute the generated class and report the dtype its forward returns
    for a float32 input, which is the whole point of the patch."""
    namespace = {"torch": torch, "nn": torch.nn, "__name__": name + "_generated"}
    namespace.update({k: getattr(module, k) for k in dir(module) if not k.startswith("__")})
    exec(compile(generated, f"<{name}>", "exec"), namespace)
    norm = namespace["FakeRMSNorm"](4)
    return norm(torch.ones(2, 4, dtype = torch.float32)).dtype


def test_a_renamed_replacement_is_what_the_generated_class_calls(tmp_path, monkeypatch):
    module, generated = _generate_with_replacement(
        tmp_path, monkeypatch, "fake_rmsnorm_renamed", "_fake_forced_rmsnorm_forward",
    )
    assert "return _fake_forced_rmsnorm_forward(" in generated
    assert "original_body_ran" not in generated, "the upstream forward survived the splice"
    assert _run_generated(generated, "fake_rmsnorm_renamed", module) == torch.float16


def test_a_replacement_still_named_forward_is_silently_dropped(tmp_path, monkeypatch):
    """The negative control, and the bug. If the compiler ever learns to handle
    this spelling, this test goes red and says the rename below is no longer
    load-bearing; until then it is."""
    module, generated = _generate_with_replacement(
        tmp_path, monkeypatch, "fake_rmsnorm_named_forward", "forward",
    )
    # The patched body is emitted, but nothing in the class calls it.
    assert "forced_body_ran" in generated
    assert "return FakeRMSNorm_forward(" not in generated
    assert "original_body_ran" in generated
    assert _run_generated(generated, "fake_rmsnorm_named_forward", module) == torch.float32


def _rmsnorm_patch_call():
    """The `patch_function(...)` call inside `patch_Gemma4RMSNorm`, found
    structurally so a second patch_function call elsewhere cannot stand in."""
    source = inspect.getsource(gemma4_float32)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "patch_Gemma4RMSNorm":
            calls = [
                n for n in ast.walk(node)
                if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "patch_function"
            ]
            assert len(calls) == 1, f"expected one patch_function call, found {len(calls)}"
            return node, calls[0]
    pytest.fail("patch_Gemma4RMSNorm is gone from gemma4_float32.py")


def test_the_real_patch_does_not_spell_its_replacement_forward():
    outer, call = _rmsnorm_patch_call()
    assert len(call.args) >= 3, "patch_function(target, attr, new_func, ...) expected"
    replacement = call.args[2]
    assert isinstance(replacement, ast.Name), "the replacement must be a named function"
    assert replacement.id != "forward", (
        "the RMSNorm replacement is named `forward` again; the compiler drops "
        "that spelling and the forced-float32 path returns fp32 into bf16 weights"
    )
    defined = {n.name for n in outer.body if isinstance(n, ast.FunctionDef)}
    assert replacement.id in defined, f"{replacement.id} is not defined inside patch_Gemma4RMSNorm"
    # And it still patches the attribute the model calls.
    assert isinstance(call.args[1], ast.Constant) and call.args[1].value == "forward"
