# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""utils._class_source == inspect.getsource; compiler._def_prefix_checker == the old per-name regex."""

import ast
import importlib
import inspect
import linecache
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "unsloth_zoo"
_WANTED = {
    ROOT / "utils.py": {"_CLASS_LINES", "_class_lines", "_class_source"},
    ROOT / "compiler.py": {"_def_prefix_checker"},
}


def _lift():
    namespace = {"ast": ast, "inspect": inspect, "linecache": linecache, "re": re, "sys": sys}
    for path, wanted in _WANTED.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        body = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                body.append(node)
            elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in wanted for t in node.targets
            ):
                body.append(node)
        exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
        assert wanted <= set(namespace), wanted - set(namespace)
    return namespace


_NS = _lift()
_class_source = _NS["_class_source"]
_def_prefix_checker = _NS["_def_prefix_checker"]

MODELING_MODULES = (
    "llama",
    "mistral",
    "qwen2",
    "qwen3",
    "qwen3_moe",
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
    "gemma",
    "gemma2",
    "gemma3",
    "gemma3n",
    "siglip",
    "mllama",
    "phi3",
    "granite",
    "cohere",
    "olmo2",
)


def _modeling_modules():
    pytest.importorskip("transformers")
    modules = []
    for name in MODELING_MODULES:
        try:
            modules.append(importlib.import_module(f"transformers.models.{name}.modeling_{name}"))
        except Exception:
            continue
    if not modules:
        pytest.skip("no transformers modeling module importable")
    return modules


def _count_parses(monkeypatch):
    calls = []
    real = _NS["_class_lines"]

    def counted(lines):
        calls.append(lines)
        return real(lines)

    _NS["_CLASS_LINES"].clear()
    monkeypatch.setitem(_NS, "_class_lines", counted)
    return calls


def test_transformers_classes_match_getsource(monkeypatch):
    modules = _modeling_modules()
    expected = {}
    for module in modules:
        for cls in vars(module).values():
            if isinstance(cls, type) and cls not in expected:
                try:
                    expected[cls] = inspect.getsource(cls)
                except (OSError, TypeError):
                    continue
    parses = _count_parses(monkeypatch)
    for cls, source in expected.items():
        assert _class_source(cls) == source, f"{cls.__module__}.{cls.__qualname__}"
    if sys.version_info < (3, 13):
        files = {inspect.getsourcefile(cls) for cls in expected}
        assert 0 < len(parses) <= len(files) < len(expected)


SYNTHETIC = '''
import functools


def deco(*args, **kwargs):
    if args and isinstance(args[0], type):
        return args[0]
    return lambda cls: cls


class Plain:
    def method(self):
        return 1


@deco
@functools.total_ordering
class Decorated:
    def __lt__(self, other):
        return True


@deco("RMSNorm")
@deco('x)#', key = "y(")
class StringDecorator:
    def method(self):
        return 1


@deco(
    1,
    key = 2,
)
class MultiLineDecorator:
    def method(self):
        return 1


@deco

class BlankAfterDecorator:
    def method(self):
        return 1


@deco
# a comment between decorator and class
class CommentAfterDecorator:
    def method(self):
        return 1


class Outer:
    class Nested:
        def method(self):
            return 1


class Duplicate:
    def method(self):
        return 1


Duplicate_first = Duplicate


class Duplicate:
    def method(self):
        return 2


DOC = """
class InString:
    pass
"""


if True:
    class InString:
        def method(self):
            return 3


class NoMethods:
    pass


TABLE = dict(
    a = 1,
)


class AfterCall:
    @functools.lru_cache
    def method(self):
        return 4


try:
    VALUE = 1

    class InTry:
        def method(self):
            return 5

    @deco
    @deco("x")
    class InTryDecorated:
        def method(self):
            return 6

    @deco(
        1,
    )
    class InTryMultiLineDecorator:
        def method(self):
            return 7

    class InTryFirst2:
        def method(self):
            return 8
except Exception:
    pass


try:
    class InTryFirst:
        def method(self):
            return 9

    @deco(
1)
    class InTryDedentedTail:
        def method(self):
            return 10
except Exception:
    pass
'''


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    directory = tmp_path_factory.mktemp("zoo_class_source")
    (directory / "zoo_class_source_fixture.py").write_text(SYNTHETIC, encoding="utf-8")
    sys.path.insert(0, str(directory))
    try:
        yield importlib.import_module("zoo_class_source_fixture")
    finally:
        sys.path.remove(str(directory))
        sys.modules.pop("zoo_class_source_fixture", None)


SYNTHETIC_NAMES = (
    "Plain", "Decorated", "StringDecorator", "AfterCall", "InTry", "InTryDecorated", "InTryFirst",
    "InTryFirst2", "MultiLineDecorator", "BlankAfterDecorator", "CommentAfterDecorator", "Duplicate",
    "InString", "NoMethods", "InTryMultiLineDecorator", "InTryDedentedTail",
)


@pytest.mark.parametrize("name", SYNTHETIC_NAMES)
def test_synthetic_matches_getsource(synthetic, name):
    cls = getattr(synthetic, name)
    assert _class_source(cls) == inspect.getsource(cls)


def test_synthetic_nested_and_first_duplicate(synthetic):
    for cls in (synthetic.Outer.Nested, synthetic.Duplicate_first):
        assert _class_source(cls) == inspect.getsource(cls)


def test_edited_file_is_reread(tmp_path, monkeypatch):
    path = tmp_path / "zoo_class_source_edit.py"
    path.write_text("class A:\n    def f(self):\n        return 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module("zoo_class_source_edit")
    try:
        assert _class_source(module.A) == inspect.getsource(module.A)
        path.write_text("x = 1\n\nclass A:\n    def f(self):\n        return 22\n", encoding="utf-8")
        os.utime(path, (1, 1))
        assert _class_source(module.A) == inspect.getsource(module.A)
        if sys.version_info < (3, 13):  # 3.13+ reads the stale __firstlineno__
            assert "return 22" in _class_source(module.A)
    finally:
        sys.modules.pop("zoo_class_source_edit", None)


def _old_defined(name, source):
    return re.search(r"\bdef[\s]{1,}" + re.escape(name), source, flags=re.DOTALL) is not None


TRICKY_SOURCE = (
    "def def foo(x):\n"
    "    pass\n"
    "xdef bar(): pass\n"
    "_def baz(): pass\n"
    "def\tqux_long(): pass\n"
    "def\n\n   spaced(): pass\n"
    "async def coro(): pass\n"
    "def café_ü(): pass\n"
    "x = 'def in_string(' \n"
    "lambda: undef_thing()\n"
    "def"
)
TRICKY_NAMES = (
    "foo", "fo", "def", "bar", "baz", "qux", "qux_long", "qux_longer", "spaced", "coro",
    "café", "café_ü", "in_string", "undef_thing", "thing", "x", "_", "a.b",
    "foo(", "q$", "", " foo", "ü",
)


def test_def_prefix_checker_matches_regex_tricky():
    check = _def_prefix_checker(TRICKY_SOURCE)
    for name in TRICKY_NAMES:
        assert check(name) == _old_defined(name, TRICKY_SOURCE), repr(name)


def test_def_prefix_checker_matches_regex_on_modeling_files():
    checked = 0
    for module in _modeling_modules():
        source = inspect.getsource(module)
        check = _def_prefix_checker(source)
        names = set(dir(module)) | set(re.findall(r"\w+", source)[:3000])
        for name in sorted(names):
            assert check(name) == _old_defined(name, source), (module.__name__, name)
            checked += 1
    assert checked > 1000


def test_gpt_oss_config_classes():
    pytest.importorskip("transformers.models.gpt_oss.configuration_gpt_oss")
    try:
        gpt_oss = importlib.import_module("unsloth_zoo.temporary_patches.gpt_oss")
    except Exception as exception:
        pytest.skip(f"gpt_oss patches not importable: {exception}")
    from transformers.models.gpt_oss import configuration_gpt_oss

    classes = [getattr(gpt_oss, n) for n in ("Old_GptOssConfig", "GptOssConfig") if hasattr(gpt_oss, n)]
    classes.append(configuration_gpt_oss.GptOssConfig)
    assert [_class_source(c) for c in classes] == [inspect.getsource(c) for c in classes]
