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

# Generated code must not reach os.environ via str.format field traversal.
import os
import pytest
from unsloth_zoo.rl_environments import create_locked_down_function

SECRET = "hf_FAKE_FORMAT_FIELD_SECRET"

ESCAPES = [
    'def f():\n    import random\n    return "{0._module._os.environ[UNSLOTH_T]}".format(random)\n',
    'def f():\n    import random\n    return "{r._module._os.environ[UNSLOTH_T]}".format_map({"r": random})\n',
    'def f():\n    import random\n    s = "{0." + "_module._os.environ[UNSLOTH_T]}"\n    return s.format(random)\n',
    'def f():\n    import random\n    return str.format("{0._module._os.environ[UNSLOTH_T]}", random)\n',
    'def f():\n    import random\n    return "{0:{1._module}}".format("a", random)\n',
    'def f():\n    import random\n    return "{0[_os]}".format({"_os": 1})\n',
    'def f(x="a"):\n    match x:\n        case str(format=g):\n            return g\n',
]

LEGIT = [
    ('def f(x=3.14159):\n    return "{} {:.2f} {name}".format(1, x, name="a")\n', "1 3.14 a"),
    ('def f():\n    return "{0[1]}-{0[0]}".format([1, 2])\n', "2-1"),
    ('def f(x=2):\n    return "{x:>{w}}".format_map({"x": x, "w": 3})\n', "  2"),
    ('def f(x=2):\n    return f"{x:>3}" + format(x, "02d") + "%d" % x\n', "  2022"),
]

@pytest.mark.parametrize("source", ESCAPES)
def test_format_field_traversal_refused(source, monkeypatch):
    monkeypatch.setenv("UNSLOTH_T", SECRET)
    try:
        out = create_locked_down_function(source)()
    except (RuntimeError, AttributeError, ImportError, TypeError):
        return
    assert SECRET not in str(out)
    pytest.fail(f"generated code ran: {out!r}")

@pytest.mark.parametrize("source,expected", LEGIT)
def test_literal_format_still_works(source, expected):
    assert create_locked_down_function(source)() == expected
