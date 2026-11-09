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

"""A module-scope os.environ write of an import-time global such as UNSLOTH_ZOO_DISABLE_GPU_INIT must be restored after
the import; otherwise it leaks to the whole xdist worker."""

from __future__ import annotations

import ast
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]

# Read once at import and cached for the process, so a later change is not observed and a test assuming the default silently measures something else.
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
                # An update(a_dict) call has no literal names here, so it is reported as unknown rather than passed, letting the author name them.
                yield None, child.lineno


def _module_scope_statements(tree: ast.Module):
    """Scans top-level statements only, skipping try blocks, since a try/finally that restores the variable is the fix and must not be flagged."""
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
    # A write inside the try is restored however the import ends; that is the one shape this guard calls clean, or the guard would ban the working pattern too.
    assert found == [], found
