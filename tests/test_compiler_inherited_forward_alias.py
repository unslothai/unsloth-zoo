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

"""create_standalone_class on a subclass that inherits its forward (deprecated aliases such as
Ernie4_5_VL_MoeForConditionalGeneration) must emit a working override (CPU, source only)."""

import ast
import importlib.util
import sys
import types

import pytest

from unsloth_zoo import compiler

MODULE = '''
import torch


class Canonical(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = torch.nn.Linear(4, 8, bias = False)

    def forward(
        self,
        hidden_states,
        labels = None,
        **kwargs,
    ):
        return self.lm_head(hidden_states)


class DeprecatedAlias(Canonical):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
'''


@pytest.fixture
def fake_module(tmp_path, monkeypatch):
    path = tmp_path / "fake_alias_modeling.py"
    path.write_text(MODULE)
    spec = importlib.util.spec_from_file_location("fake_alias_modeling", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "fake_alias_modeling", module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(compiler, "fake_alias_pkg", types.SimpleNamespace(mod = module), raising = False)
    return module


def _class_methods(source, name):
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == name)
    return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}


def test_inherited_forward_gets_delegating_override(fake_module):
    source = compiler.create_standalone_class(
        "DeprecatedAlias", "fake_alias_pkg.mod", [], disable = True, add_loss_kwargs = True,
    )
    methods = _class_methods(source, "DeprecatedAlias")
    assert set(methods) == {"__init__", "forward"}
    call = methods["forward"].body[-1].value
    assert isinstance(call, ast.Call) and call.func.id == "DeprecatedAlias_forward"
    assert any(isinstance(n, ast.FunctionDef) and n.name == "DeprecatedAlias_forward" for n in ast.parse(source).body)


def test_own_forward_still_replaced_in_place(fake_module):
    source = compiler.create_standalone_class(
        "Canonical", "fake_alias_pkg.mod", [], disable = True, add_loss_kwargs = True,
    )
    methods = _class_methods(source, "Canonical")
    assert list(methods) == ["__init__", "forward"]
    assert methods["forward"].body[-1].value.func.id == "Canonical_forward"


def test_inherited_forward_without_fused_ce_is_left_alone(fake_module):
    # Only the fused CE call site (add_loss_kwargs) emits the override; other callers keep failing loudly.
    with pytest.raises(AttributeError):
        compiler.create_standalone_class("DeprecatedAlias", "fake_alias_pkg.mod", [], disable = True)
