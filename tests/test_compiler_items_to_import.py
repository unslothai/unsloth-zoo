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

"""create_new_function's import list must match the per-name `^def <x>\\(` search it replaced."""

import inspect
import random
import re

import pytest

from unsloth_zoo import compiler


def _reference(functions, new_source, name):
    return [
        x
        for x in functions
        if (
            (x in new_source)
            and (x != name)
            and not re.search(r"^def " + re.escape(x) + r"\(", new_source, flags = re.MULTILINE)
        )
    ]


SOURCE = (
    "def helper(x):\n"
    "    return other(x)\n"
    "class Foo(Base):\n"
    "    def method(self):\n"
    "        return helper(1) + method_two()\n"
    "def  spaced(y):\n"
    "    pass\n"
    "def other2 (z):\n"
    "    pass\n"
    "def café(a):\n"
    "    return a\n"
    "def x́y(a):\n"
    "    return a\n"
    "\r\ndef crlf(a):\r\n    return a\r\n"
    "  def nested(a):\n"
    "def help(a): helper_x(a)\n"
    "def a.b(c): pass\n"
)

NAMES = [
    "helper", "other", "Foo", "Base", "method", "method_two", "spaced", "other2",
    "café", "caf", "x́y", "x", "crlf", "nested", "help", "helper_x", "a.b",
    "a", "missing", "", "def", "(", "Foo(", "self",
]


@pytest.mark.parametrize("name", ["Foo", "helper", "nope", "café"])
def test_items_match_reference(name):
    assert compiler._items_to_import(NAMES, SOURCE, name) == _reference(NAMES, SOURCE, name)


def test_items_match_reference_random():
    rng = random.Random(0)
    alphabet = ["def ", "(", ")", "\n", " ", "a", "b", "_", "1", ".", "é", "́", ":", "\r"]
    for _ in range(300):
        source = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 120)))
        names = ["".join(rng.choice("ab_1.é́(") for _ in range(rng.randint(0, 3))) for _ in range(12)]
        name = rng.choice(names) if names else "a"
        assert compiler._items_to_import(names, source, name) == _reference(names, source, name)


def test_items_match_reference_real_module():
    source = inspect.getsource(compiler)
    names = dir(compiler)
    assert compiler._items_to_import(names, source, "create_new_function") == _reference(
        names, source, "create_new_function"
    )
