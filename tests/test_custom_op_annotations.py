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

"""torch.library.custom_op annotations resolve on the torch floor.

Under `from __future__ import annotations` every annotation is a string. torch 2.6's infer_schema
evals that string in its own module, not the function's, so only the names that module imports
resolve: `Tuple[torch.Tensor, torch.Tensor]` raised "Unsupported type annotation" at import and took
the whole module down on the torch 2.6 lane (#1934). Newer torch resolves the function's globals, so
only the floor sees it; this checks every custom op against the floor's namespace on any torch.
"""

import ast
import builtins
import pathlib
import typing

import pytest

torch = pytest.importorskip("torch")

ZOO = pathlib.Path(__file__).resolve().parents[1] / "unsloth_zoo"

# torch/_library/infer_schema.py (torch 2.6) module globals, which its eval() sees.
_FLOOR_NAMESPACE = {
    **vars(builtins),
    "inspect": __import__("inspect"),
    "typing": typing,
    "List": typing.List,
    "Optional": typing.Optional,
    "Sequence": typing.Sequence,
    "Union": typing.Union,
    "torch": torch,
    "device": torch.device,
    "dtype": torch.dtype,
    "Tensor": torch.Tensor,
    "types": torch.types,
}


def _is_custom_op(decorator):
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    return isinstance(target, (ast.Attribute, ast.Name)) and getattr(target, "attr", getattr(target, "id", None)) == "custom_op"


def _custom_op_annotations():
    for path in sorted(ZOO.rglob("*.py")):
        rel = path.relative_to(ZOO).as_posix()
        src = path.read_text(encoding = "utf-8")
        if "custom_op" not in src:
            continue
        tree = ast.parse(src)
        postponed = any(
            isinstance(n, ast.ImportFrom) and n.module == "__future__" and any(a.name == "annotations" for a in n.names)
            for n in tree.body
        )
        if not postponed:
            continue
        for func in ast.walk(tree):
            if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(map(_is_custom_op, func.decorator_list)):
                args = func.args.posonlyargs + func.args.args + func.args.kwonlyargs
                for node in [a.annotation for a in args] + [func.returns]:
                    if node is not None:
                        yield f"{rel}:{func.lineno}:{func.name}", ast.unparse(node)


def test_the_scan_sees_custom_ops():
    assert any("moe_router_triton.py" in where for where, _ in _custom_op_annotations())


def test_custom_op_annotations_resolve_in_the_torch_floor_namespace():
    bad = []
    for where, annotation in _custom_op_annotations():
        try:
            eval(annotation, dict(_FLOOR_NAMESPACE))
        except Exception as e:
            bad.append(f"{where} {annotation!r}: {type(e).__name__}: {e}")
    assert not bad, "torch 2.6 rejects these custom_op annotations at import; use builtin / torch names: " + "; ".join(bad)
