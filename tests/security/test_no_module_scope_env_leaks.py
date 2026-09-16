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

"""No test module may switch an import-time global on at module scope and leave it on.

pytest imports every selected module before it runs anything, so a module-scope
`os.environ["X"] = ...` is not scoped to that file: it is set for the whole xdist worker,
for every test collected after it, in whatever directory. Those tests then measure a
configuration nobody chose.

This is not hypothetical. `test_higher_precision_layernorm_scope.py` needed
`UNSLOTH_ZOO_DISABLE_GPU_INIT=1` for one import and set it with a bare `setdefault` at
module scope. `__init__.py:179` reads that into `_SKIP_GPU_INIT`, which skips the
import-time allocator block and pins `DEVICE_TYPE` to "cpu", so
`test_alloc_conf_platform_matrix.py` reported all three allocator variables unset and
`test_temporary_patches_imports.py` reported `device_memory` as 0. Both blamed the
product, neither was about the product, and between them they failed the
`unsloth_zoo @ main` leg of unsloth's Core workflow on every open pull request.

Setting one of these for an import is fine. Leaving it set is not, so what is required
here is the restore, not the abstinence.
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]

# Read once at import and cached for the process, so a later change is not observed and a
# test that assumes the default silently measures something else.
IMPORT_TIME_GLOBALS = frozenset({
    "UNSLOTH_ZOO_DISABLE_GPU_INIT",
    "UNSLOTH_VLLM_STANDBY",
    "UNSLOTH_ALLOW_CPU",
    "UNSLOTH_COMPILE_DISABLE",
    "PYTORCH_ALLOC_CONF",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_HIP_ALLOC_CONF",
})


def _env_writes(node: ast.AST):
    """`os.environ[...] = ...`, `os.environ.setdefault(...)` and `os.environ.update(...)`."""
    for child in ast.walk(node):
        if isinstance(child, ast.Assign):
            for target in child.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and "environ" in ast.unparse(target.value)
                    and isinstance(target.slice, ast.Constant)
                ):
                    yield target.slice.value, child.lineno
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            if child.func.attr not in ("setdefault", "update"):
                continue
            if "environ" not in ast.unparse(child.func.value):
                continue
            if child.args and isinstance(child.args[0], ast.Constant):
                yield child.args[0].value, child.lineno
            else:
                # `update(a_dict)`: the names are not literals here, so report it as
                # unknown rather than pass it, and let the author name them.
                yield None, child.lineno


def _module_scope_statements(tree: ast.Module):
    """Top-level statements only, minus anything inside a `try`.

    A `try`/`finally` at module scope is the shape this file is asking for: set it, do the
    import, put it back. Descending into one would flag the fix as the bug.
    """
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Try)):
            continue
        yield node


def test_no_test_module_sets_an_import_time_global_and_leaves_it():
    offenders = []
    for path in sorted(TESTS_ROOT.rglob("test_*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding = "utf-8"))
        except SyntaxError:
            continue
        for node in _module_scope_statements(tree):
            for name, line in _env_writes(node):
                if name is None or name in IMPORT_TIME_GLOBALS:
                    offenders.append(
                        f"{path.relative_to(TESTS_ROOT)}:{line}: "
                        f"{name or 'os.environ.update(<not a literal>)'}"
                    )
    assert not offenders, (
        "these test modules set an import-time global at module scope, which pytest runs at "
        "COLLECTION, so it applies to every test on the worker rather than to this file. Set "
        "it around the import that needs it and restore it in a `finally`:\n  "
        + "\n  ".join(offenders)
    )


def test_the_scan_would_catch_the_regression_it_exists_for(tmp_path):
    """Hand it the exact shape that broke Core and it must object."""
    sample = tmp_path / "test_offender.py"
    sample.write_text(
        "import os\n"
        'os.environ.setdefault("UNSLOTH_ZOO_DISABLE_GPU_INIT", "1")\n'
        "from unsloth_zoo.compiler import higher_precision_layernorms\n",
        encoding = "utf-8",
    )
    tree = ast.parse(sample.read_text(encoding = "utf-8"))
    found = [
        name
        for node in _module_scope_statements(tree)
        for name, _ in _env_writes(node)
        if name in IMPORT_TIME_GLOBALS
    ]
    assert found == ["UNSLOTH_ZOO_DISABLE_GPU_INIT"], found


def test_the_scan_accepts_a_restored_write(tmp_path):
    """And the fix must read as clean, or the guard just bans the working pattern."""
    sample = tmp_path / "test_restored.py"
    sample.write_text(
        "import os\n"
        'previous = os.environ.get("UNSLOTH_ZOO_DISABLE_GPU_INIT")\n'
        "try:\n"
        '    os.environ.setdefault("UNSLOTH_ZOO_DISABLE_GPU_INIT", "1")\n'
        "    import unsloth_zoo\n"
        "finally:\n"
        '    os.environ.pop("UNSLOTH_ZOO_DISABLE_GPU_INIT", None)\n',
        encoding = "utf-8",
    )
    tree = ast.parse(sample.read_text(encoding = "utf-8"))
    found = [
        name
        for node in _module_scope_statements(tree)
        for name, _ in _env_writes(node)
        if name in IMPORT_TIME_GLOBALS
    ]
    # The write is inside the `try`, so the restore covers it however the import ends.
    # That is the one shape this guard calls clean, and the fix has to read as clean or
    # the guard bans the working pattern along with the broken one.
    assert found == [], found
