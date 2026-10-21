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

"""torch.library.custom_op functions live in modules without postponed annotations.

Under `from __future__ import annotations` every annotation is a string, and the torch floor cannot
read those: torch 2.4's infer_schema rejects a string `torch.Tensor` parameter outright, and torch
2.6 evals the string in its own module, where `Tuple` does not exist. #1934's fused router hit both
at import, which `temporary_patches/__init__.py` does unconditionally. Newer torch reads the strings
fine, so only the floor lanes see it; this static check catches it on any torch.
"""

import ast
import pathlib

ZOO = pathlib.Path(__file__).resolve().parents[1] / "unsloth_zoo"


def _is_custom_op(decorator):
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    name = getattr(target, "attr", getattr(target, "id", None))
    return name == "custom_op"


def _postponed(tree):
    return any(
        isinstance(n, ast.ImportFrom) and n.module == "__future__" and any(a.name == "annotations" for a in n.names)
        for n in tree.body
    )


def _custom_ops():
    for path in sorted(ZOO.rglob("*.py")):
        src = path.read_text(encoding = "utf-8")
        if "custom_op" not in src:
            continue
        tree = ast.parse(src)
        for func in ast.walk(tree):
            if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(map(_is_custom_op, func.decorator_list)):
                yield path.relative_to(ZOO).as_posix(), func, _postponed(tree)


def test_the_scan_sees_custom_ops():
    found = {rel for rel, _, _ in _custom_ops()}
    assert {"temporary_patches/moe_router_triton.py", "fp16_emulation.py"} <= found, found


def test_no_custom_op_has_string_annotations():
    bad = sorted(f"{rel}:{func.lineno}:{func.name}" for rel, func, postponed in _custom_ops() if postponed)
    assert not bad, (
        "torch.library.custom_op in a module with `from __future__ import annotations`: torch 2.4 rejects "
        "string annotations at definition (2.6 only resolves torch / some typing names), so the module "
        "fails to import on the floor: " + ", ".join(bad)
    )
