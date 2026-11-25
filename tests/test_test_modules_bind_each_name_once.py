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

"""No test module may bind a top-level name twice: the later binding silently shadows the earlier
one; two `_SHRUNK_VLM_STACKS` tables with different shapes broke nightly Mac Metal tests."""

import ast
import collections
from pathlib import Path

_TESTS = Path(__file__).resolve().parent


def _top_level_rebinds(source):
    seen = collections.defaultdict(list)
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                seen[target.id].append(node.lineno)
    return {name: lines for name, lines in seen.items() if len(lines) > 1}


def test_no_test_module_binds_a_top_level_name_twice():
    found = {}
    for path in sorted(_TESTS.rglob("test_*.py")):
        rebinds = _top_level_rebinds(path.read_text(encoding="utf-8"))
        if rebinds:
            found[str(path.relative_to(_TESTS))] = rebinds
    assert not found, f"top-level names bound more than once (the last binding wins for the whole file): {found}"


def test_the_check_sees_a_rebound_table():
    assert _top_level_rebinds("A = {1: (1, 2, 3)}\n\ndef f():\n    return A\n\nA = {1: (1, 2)}\n") == {"A": [1, 6]}
    assert _top_level_rebinds("A = 1\nB = 2\nif True:\n    A = 3\n") == {}
