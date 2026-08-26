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

import pytest

from unsloth_zoo.rl_environments import create_locked_down_function

BLOCKED = [
    ("import os", "def matmul(A, B):\n    import os\n    return os.getpid()\n", ImportError),
    ("dunder import", 'def matmul(A, B):\n    return __import__("os")\n', RuntimeError),
    ("subclasses walk", "def matmul(A, B):\n    return (1).__class__.__bases__[0].__subclasses__()\n", RuntimeError),
    ("open", 'def matmul(A, B):\n    return open("/etc/passwd").read()\n', NameError),
    ("eval", 'def matmul(A, B):\n    return eval("1+1")\n', NameError),
    ("subprocess", "def matmul(A, B):\n    import subprocess\n    return subprocess.run\n", ImportError),
    ("socket", "def matmul(A, B):\n    import socket\n    return socket.socket()\n", ImportError),
    # The dunder is a string constant here, so the AST check does not see it.
    # It fails anyway because getattr is not in the allowlist at all.
    ("getattr", 'def matmul(A, B):\n    return getattr(A, "__class__")\n', NameError),
]


@pytest.mark.parametrize("label,source,expected", BLOCKED, ids=[b[0] for b in BLOCKED])
def test_escape_is_blocked(label, source, expected):
    with pytest.raises(expected):
        create_locked_down_function(source)([], [])


def test_safe_stdlib_imports_still_work():
    source = (
        "def strategy(board):\n"
        "    import math\n"
        "    from typing import Callable\n"
        '    return "W"\n'
    )
    assert create_locked_down_function(source)([[0]]) == "W"


def test_generated_matmul_still_runs():
    source = (
        "def matmul(A, B):\n"
        "    z, s = zip, sum\n"
        "    Bt = list(z(*B))\n"
        "    return [[s(a*b for a, b in z(row, col)) for col in Bt] for row in A]\n"
    )
    fn = create_locked_down_function(source)
    assert fn([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[19, 22], [43, 50]]


def test_positional_defaults_survive_lockdown():
    fn = create_locked_down_function("def strategy(board, depth = 7):\n    return depth\n")
    assert fn([[0]]) == 7


def test_keyword_only_defaults_survive_lockdown():
    # A literal keyword-only default passes validation, so the locked down
    # function must keep it; otherwise calling it raises TypeError.
    fn = create_locked_down_function("def strategy(board, *, depth = 3):\n    return depth\n")
    assert fn.__kwdefaults__ == {"depth": 3}
    assert fn([[0]]) == 3


def test_full_policy_remains_available():
    fn = create_locked_down_function(
        "def matmul(A, B):\n    import os\n    return os.sep\n",
        builtins_policy = "full",
    )
    assert isinstance(fn([], []), str)
