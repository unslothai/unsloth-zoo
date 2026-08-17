# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""A test file may not reach a sibling through the `tests.` package.

This repo's `tests/` carries no `__init__.py`, so `tests.foo` does not name
anything here: it names whatever `tests` package is importable, and there is
one. Unsloth's Core CI runs these files with its own repo root on `sys.path`,
and unsloth DOES ship `tests/__init__.py`, so `from tests.x import y` bound to
that project's package and raised `ModuleNotFoundError: No module named
'tests.x'` -- taking the whole module out at collection and failing the job
with exit 2, on every unsloth pull request rather than on anything zoo changed.

A grep, because the import only misbehaves under a `sys.path` this suite does
not control, so no amount of running the files from this repo would catch it.
"""

import ast
import pathlib

TESTS = pathlib.Path(__file__).resolve().parent


def _imports_through_the_tests_package(source: str) -> list:
    """Every `import tests.x` / `from tests.x import y` in `source`."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            # `level` is the number of leading dots: a relative import names
            # this directory and is not the failure mode.
            if not node.level and (node.module or "").split(".")[0] == "tests":
                found.append(f"from {node.module} import ...")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "tests":
                    found.append(f"import {alias.name}")
    return found


def test_the_pattern_is_recognized():
    """The grep can fail, which is what makes the sweep below mean something."""
    assert _imports_through_the_tests_package("from tests.a import b") == [
        "from tests.a import ..."
    ]
    assert _imports_through_the_tests_package("import tests.a") == ["import tests.a"]
    # A relative import and an unrelated one are both fine.
    assert _imports_through_the_tests_package("from .a import b") == []
    assert _imports_through_the_tests_package("from testsuite import b") == []
    assert _imports_through_the_tests_package("from a import tests") == []


def _scan_the_suite() -> dict:
    """Every file under `tests/` that reaches a sibling through `tests.`.

    This file is scanned too. Its own examples are string literals and parse to
    no import node, so there is nothing to exempt - and exempting it would have
    let the one module whose whole job is refusing this pattern be the one
    module allowed to carry it.
    """
    offenders = {}
    for path in sorted(TESTS.rglob("*.py")):
        hits = _imports_through_the_tests_package(
            path.read_text(encoding = "utf-8", errors = "replace")
        )
        if hits:
            offenders[path.relative_to(TESTS).as_posix()] = hits
    return offenders


def test_the_guard_scans_itself():
    """The examples above are literals, so scanning this file finds nothing."""
    here = pathlib.Path(__file__).resolve()
    assert _imports_through_the_tests_package(here.read_text(encoding = "utf-8")) == []
    # And it really is in the walk, rather than passing by being skipped.
    assert here in set(TESTS.rglob("*.py"))


def test_no_test_file_imports_a_sibling_through_the_tests_package():
    offenders = _scan_the_suite()
    assert not offenders, (
        "these reach a sibling through `tests.`, which binds to another "
        f"project's package under unsloth's CI: {offenders}. Import the module "
        "by name after putting this directory on sys.path instead."
    )
