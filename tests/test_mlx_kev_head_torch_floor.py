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

"""Kev's head.pt must not be unpickled by a torch that runs code under weights_only=True
(CVE-2025-32434); decision.py imports mlx at module level, so its guard is read with ast."""

import ast
import types
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "unsloth_zoo" / "mlx" / "decision.py"


def _tree():
    return ast.parse(SOURCE.read_text(encoding = "utf-8"))


def _guard():
    tree = _tree()
    node = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_require_safe_torch_load"), None)
    assert node is not None, "decision.py has no torch version guard for head.pt"
    namespace = {}
    exec(compile(ast.Module(body = [node], type_ignores = []), str(SOURCE), "exec"), namespace)
    return namespace["_require_safe_torch_load"]


def _kev_init_calls():
    tree = _tree()
    kev = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_KevModel")
    init = next(n for n in kev.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    return [ast.unparse(n.func) for n in ast.walk(init) if isinstance(n, ast.Call)]


@pytest.mark.parametrize("version", ["2.5.1", "2.4.0+cpu", "1.13.1"])
def test_old_torch_refused(version):
    with pytest.raises(ValueError, match = "cannot load one safely"):
        _guard()(types.SimpleNamespace(__version__ = version), Path("head.pt"))


@pytest.mark.parametrize("version", ["2.6.0", "2.6.0a0+git1", "2.9.1+cu128", "2.15.0"])
def test_safe_torch_allowed(version):
    _guard()(types.SimpleNamespace(__version__ = version), Path("head.pt"))


def test_guard_runs_before_torch_load():
    calls = _kev_init_calls()
    assert "_require_safe_torch_load" in calls
    assert calls.index("_require_safe_torch_load") < calls.index("torch.load")
