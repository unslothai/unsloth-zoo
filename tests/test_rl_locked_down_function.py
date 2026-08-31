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
    # Allowlisted modules re-export unsafe ones: random keeps a private handle
    # on os, and typing keeps sys. random._os is caught statically by the
    # private-attribute rule; typing.sys is not private, so the facade is what
    # stops it. Both layers are load bearing.
    ("random._os", 'def matmul(A, B):\n    import random\n    return random._os.system("echo pwned")\n', RuntimeError),
    ("typing.sys", 'def matmul(A, B):\n    import typing\n    return typing.sys.modules["os"].getpid()\n', AttributeError),
    # attrgetter takes a dotted string, so it walks dunders without the AST
    # check ever seeing them. The member is denied; operator itself stays.
    ("operator.attrgetter", 'def matmul(A, B):\n    import operator\n    return operator.attrgetter("__class__.__bases__")(1)\n', AttributeError),
    ("operator.methodcaller", 'def matmul(A, B):\n    import operator\n    return operator.methodcaller("__str__")(1)\n', AttributeError),
    # Formatter.get_field resolves a dotted string and returns the object, so
    # it recovers the real __import__. str.format walks attributes the same way
    # but only returns text, which is why only this one matters.
    ("string.Formatter", 'def matmul(A, B):\n    import string\n    q = string.Formatter()\n    return q.get_field("0.get_field.__globals__[__builtins__][__import__]", (q,), {})[0]\n', ImportError),
]


def test_typing_get_type_hints_is_denied():
    # Annotations are compiled to strings, and get_type_hints evaluates them,
    # so this executes without a single dunder appearing in the source.
    source = (
        "def matmul(A, B):\n"
        "    import typing\n"
        "    class C:\n"
        '        x: "__import__(chr(111)+chr(115)).system(chr(88))"\n'
        "    return typing.get_type_hints(C)\n"
    )
    with pytest.raises(AttributeError):
        create_locked_down_function(source)([], [])


def test_typing_forward_ref_evaluators_are_denied():
    # ForwardRef.evaluate and evaluate_forward_ref are the same evaluator, made
    # public in 3.14. On 3.13 the entry point is the private _evaluate, which
    # the private-attribute rule already covers, but this package supports 3.14
    # so the names are denied on every version.
    for source in (
        'def matmul(A, B):\n    import typing\n    return typing.ForwardRef("1+1")\n',
        'def matmul(A, B):\n    import typing\n    return typing.evaluate_forward_ref\n',
    ):
        with pytest.raises(AttributeError):
            create_locked_down_function(source)([], [])


def test_rest_of_typing_still_works():
    # typing stays allowlisted because the notebooks' own samples import it.
    source = (
        "def strategy(board):\n"
        "    from typing import Callable\n"
        '    return "W"\n'
    )
    assert create_locked_down_function(source)([[0]]) == "W"


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


def test_allowlisted_dotted_submodule_works_both_forms():
    from_form = (
        "def strategy(board):\n"
        "    from collections.abc import Iterable\n"
        "    return isinstance(board, Iterable)\n"
    )
    bare_form = (
        "def strategy(board):\n"
        "    import collections.abc\n"
        "    return isinstance(board, collections.abc.Iterable)\n"
    )
    assert create_locked_down_function(from_form)([[0]]) is True
    assert create_locked_down_function(bare_form)([[0]]) is True


def test_private_attribute_access_is_rejected():
    # Single underscore, not two: private attributes reach modules and
    # internals just as well as dunders do.
    source = "def matmul(A, B):\n    import abc\n    return abc.ABCMeta._abc_impl\n"
    with pytest.raises(RuntimeError):
        create_locked_down_function(source)([], [])


def test_underscore_locals_are_still_allowed():
    # The Name check stays at two underscores, so ordinary throwaway and
    # private-ish locals keep working.
    source = (
        "def strategy(board):\n"
        "    _total = 0\n"
        "    for _ in board:\n"
        "        _total += 1\n"
        "    return _total\n"
    )
    assert create_locked_down_function(source)([[0], [0], [0]]) == 3


def test_public_members_of_allowlisted_modules_still_work():
    source = (
        "def strategy(board):\n"
        "    import math\n"
        "    import random\n"
        "    random.seed(0)\n"
        "    return math.floor(2.7) + random.choice([0])\n"
    )
    assert create_locked_down_function(source)([[0]]) == 2


def test_operator_itemgetter_still_works():
    # itemgetter is what generated sorting code actually reaches for, and it
    # takes an index rather than an attribute path, so only attrgetter and
    # methodcaller are denied.
    source = (
        "def strategy(board):\n"
        "    import operator\n"
        '    return sorted([(2, "b"), (1, "a")], key = operator.itemgetter(0))[0][0]\n'
    )
    assert create_locked_down_function(source)([[0]]) == 1


def test_class_helpers_still_work():
    # Generated helpers are occasionally written as a small class.
    source = (
        "def matmul(A, B):\n"
        "    class Base:\n"
        "        def go(self):\n"
        "            return 1\n"
        "    class C(Base):\n"
        "        def go(self):\n"
        "            return super().go() + 1\n"
        "    return C().go()\n"
    )
    assert create_locked_down_function(source)([], []) == 2


def test_hasattr_and_time_are_available():
    source = (
        "def matmul(A, B):\n"
        "    import time\n"
        "    t = time.monotonic()\n"
        '    return hasattr(A, "shape") is False and t > 0\n'
    )
    assert create_locked_down_function(source)([], []) is True


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


def test_dunder_method_definitions_are_denied():
    # A method name is FunctionDef.name, not a Name or Attribute node, so the
    # walk cannot see it. Without the fail-closed rule, a custom __setattr__
    # captures what functools.update_wrapper copies.
    source = (
        "def matmul(A, B):\n"
        "    import functools\n"
        "    import statistics\n"
        "    box = []\n"
        "    class W:\n"
        "        def __setattr__(self, k, v):\n"
        "            box.append(v)\n"
        "    w = W()\n"
        '    functools.update_wrapper(w, statistics.mean, assigned=("__globals__",), updated=())\n'
        '    return box[0]["__builtins__"]["__import__"]("os")\n'
    )
    with pytest.raises(RuntimeError):
        create_locked_down_function(source)([], [])


def test_ordinary_dunder_methods_still_allowed():
    # __init__ and the comparison protocol are ordinary in helper classes.
    source = (
        "def matmul(A, B):\n"
        "    class Node:\n"
        "        def __init__(self, v):\n"
        "            self.v = v\n"
        "        def __lt__(self, o):\n"
        "            return self.v < o.v\n"
        "    return sorted([Node(3), Node(1)])[0].v\n"
    )
    assert create_locked_down_function(source)([], []) == 1


def test_functools_reduce_still_works():
    source = (
        "def matmul(A, B):\n"
        "    import functools\n"
        "    return functools.reduce(lambda a, b: a + b, [1, 2, 3])\n"
    )
    assert create_locked_down_function(source)([], []) == 6


def test_frame_traversal_is_denied():
    # A running generator reaches gi_frame.f_back.f_back.f_builtins, which is
    # the trusted caller's real builtins. No module is involved and no name
    # starts with an underscore, so this needs its own rule.
    source = (
        "def matmul(A, B):\n"
        "    holder = []\n"
        "    def g():\n"
        "        fr = holder[0].gi_frame.f_back.f_back\n"
        "        yield fr.f_builtins\n"
        "    gen = g()\n"
        "    holder.append(gen)\n"
        '    return next(gen)["__import__"]("os")\n'
    )
    with pytest.raises(RuntimeError):
        create_locked_down_function(source)([], [])


def test_ordinary_generators_still_work():
    source = (
        "def matmul(A, B):\n"
        "    def g():\n"
        "        yield 1\n"
        "        yield 2\n"
        "    return sum(g())\n"
    )
    assert create_locked_down_function(source)([], []) == 3


def test_clock_setters_denied_but_clocks_readable():
    with pytest.raises(AttributeError):
        create_locked_down_function(
            "def matmul(A, B):\n    import time\n    return time.clock_settime\n"
        )([], [])
    readable = "def matmul(A, B):\n    import time\n    return time.monotonic() > 0\n"
    assert create_locked_down_function(readable)([], []) is True


def test_not_implemented_is_available():
    # The comparison dunders we allow are expected to return NotImplemented
    # for operands they do not handle.
    source = (
        "def matmul(A, B):\n"
        "    class N:\n"
        "        def __eq__(self, o):\n"
        "            if not isinstance(o, N):\n"
        "                return NotImplemented\n"
        "            return True\n"
        "    return N() == 5\n"
    )
    assert create_locked_down_function(source)([], []) is False
