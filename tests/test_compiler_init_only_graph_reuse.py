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

"""The init-only caller graph is built once per model source and queried per helper.
Every answer from the shared graph must equal a fresh per-call analysis."""
import ast
import importlib
import inspect
import textwrap

import pytest

_INIT_ONLY_CALLERS = frozenset(("__init__", "__post_init__", "_init_weights", "post_init"))


def _reference(module_source, name):
    # Verbatim copy of the single-call analysis before the graph was shared.
    try:
        tree = ast.parse(textwrap.dedent(module_source))
    except Exception:
        return False
    callers_of = {}
    def visit(node, owner):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if isinstance(node, ast.ClassDef):
                    visit(child, (child.name, True))
                else:
                    visit(child, (child.name, False) if owner is None else owner)
                continue
            if isinstance(child, ast.Call):
                func = child.func
                called = func.id if isinstance(func, ast.Name) else None
                if called is not None:
                    callers_of.setdefault(called, set()).add(owner)
            visit(child, owner)
    visit(tree, None)
    seen = set()
    def init_only(fn):
        if fn in seen:
            return True
        seen.add(fn)
        owners = callers_of.get(fn)
        if not owners:
            return False
        for owner in owners:
            if owner is None:
                return False
            owner_name, is_method = owner
            if is_method:
                if owner_name in _INIT_ONLY_CALLERS:
                    continue
                return False
            if not init_only(owner_name):
                return False
        return True
    return init_only(name)


FIXTURE = '''
def leaf(x):
    return x

def mid(x):
    return leaf(x)

def top(x):
    return mid(x)

def ping(n):
    return pong(n - 1)

def pong(n):
    return ping(n)

def only_cycle(n):
    return only_cycle(n)

def forward_ref(x):
    return later(x)

def later(x):
    return x

def outer(x):
    def nested(y):
        return inner(y)
    return nested(x)

def inner(x):
    return x

def module_level(x):
    return x

VALUE = module_level(1)

async def async_helper(x):
    return leaf(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.a = top(config)
        self.b = ping(3)
        self.c = outer(1)
        self.d = forward_ref(2)

    def _init_weights(self, module):
        module.w = only_cycle(4)

    def forward(self, x):
        return leaf(x) + async_helper(x)

class Cfg:
    table = inner(5)
'''

CHAIN = '''
def inner(x):
    return x

def outer(x):
    return inner(x)

class Block(nn.Module):
    def forward(self, x):
        return outer(x)
'''


def _names(source):
    tree = ast.parse(textwrap.dedent(source))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.add(node.func.id)
    return sorted(names) + ["missing_function"]


def _check_shared_graph(source):
    from unsloth_zoo.compiler import (
        _init_only_callers_of,
        _only_called_at_init,
        function_only_called_at_init,
    )
    names = _names(source)
    expected = {name: _reference(source, name) for name in names}
    graph = _init_only_callers_of(source)
    for order in (names, names[::-1], names + names):
        for name in order:
            assert _only_called_at_init(graph, name) == expected[name], name
            assert function_only_called_at_init(source, name) == expected[name], name
    return expected


def test_shared_graph_matches_fresh_analysis_on_fixture():
    expected = _check_shared_graph(FIXTURE)
    assert expected["top"] and expected["mid"]  # transitive through a function chain
    assert not expected["leaf"]  # also reached from forward
    assert expected["ping"] and expected["pong"]  # cycle reached only from __init__
    assert expected["only_cycle"]
    assert expected["forward_ref"] and expected["later"]
    assert expected["outer"]
    assert not expected["inner"]  # also called from a class body
    assert not expected["module_level"]
    assert not expected["missing_function"]


def test_query_state_is_not_shared_between_names():
    # Asking for `outer` first must not make `inner` look init-only (a shared `seen` would).
    from unsloth_zoo.compiler import _init_only_callers_of, _only_called_at_init
    graph = _init_only_callers_of(CHAIN)
    assert not _only_called_at_init(graph, "outer")
    assert not _only_called_at_init(graph, "inner")
    assert not _only_called_at_init(graph, "outer")


def test_unparseable_source_is_never_init_only():
    from unsloth_zoo.compiler import (
        _init_only_callers_of,
        _only_called_at_init,
        function_only_called_at_init,
    )
    graph = _init_only_callers_of("not python (")
    assert graph is None
    assert not _only_called_at_init(graph, "_mix")
    assert not function_only_called_at_init("not python (", "_mix")


@pytest.mark.parametrize(
    "module_name",
    [
        "transformers.models.llama.modeling_llama",
        "transformers.models.mpt.modeling_mpt",
        "transformers.models.gemma3.modeling_gemma3",
        "transformers.models.qwen3_next.modeling_qwen3_next",
        "transformers.models.whisper.modeling_whisper",
        "transformers.models.gpt_oss.modeling_gpt_oss",
    ],
)
def test_shared_graph_matches_fresh_analysis_on_transformers(module_name):
    try:
        modeling = importlib.import_module(module_name)
    except Exception:
        pytest.skip(f"{module_name} is newer than the installed transformers")
    _check_shared_graph(inspect.getsource(modeling))
