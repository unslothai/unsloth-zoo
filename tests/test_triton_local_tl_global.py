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

"""Triton < 3.4 resolves names in a @triton.jit body from the module globals, not the enclosing function: a kernel
defined inside a function that imports `triton.language as tl` locally fails to compile there (NameError) unless
that function declares `global tl`. Static, so it holds on any installed Triton."""

import ast
import pathlib

ZOO = pathlib.Path(__file__).resolve().parents[1] / "unsloth_zoo"


def _is_jit(decorator):
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    return "jit" in ast.unparse(target)


def _local_tl_without_global(tree):
    bad = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = list(ast.walk(fn))
        kernels = [n for n in body if n is not fn and isinstance(n, ast.FunctionDef) and any(map(_is_jit, n.decorator_list))]
        local_tl = any(
            isinstance(n, ast.Import) and any(a.asname == "tl" or a.name == "tl" for a in n.names)
            or isinstance(n, ast.ImportFrom) and any((a.asname or a.name) == "tl" for a in n.names)
            for n in body
        )
        declared = any(isinstance(n, ast.Global) and "tl" in n.names for n in fn.body)
        if kernels and local_tl and not declared:
            bad.append(f"{fn.name}:{fn.lineno}")
    return bad


def test_nested_kernels_with_a_local_tl_declare_it_global():
    bad = {}
    for path in sorted(ZOO.rglob("*.py")):
        if "_vendored" in path.parts:
            continue
        src = path.read_text(encoding = "utf-8")
        if "triton" not in src or "tl" not in src:
            continue
        found = _local_tl_without_global(ast.parse(src))
        if found:
            bad[path.relative_to(ZOO).as_posix()] = found
    assert not bad, f"add `global tl` before the local import (Triton < 3.4 compiles from module globals): {bad}"


def test_the_check_can_fail():
    tree = ast.parse(
        "def make():\n"
        "    import triton\n"
        "    import triton.language as tl\n"
        "    @triton.jit\n"
        "    def k(X, B: tl.constexpr):\n"
        "        tl.store(X, 1)\n"
        "    return k\n"
        "def ok():\n"
        "    global tl\n"
        "    import triton.language as tl\n"
        "    @triton.jit\n"
        "    def k(X):\n"
        "        tl.store(X, 1)\n"
        "    return k\n"
    )
    assert _local_tl_without_global(tree) == ["make:1"]
