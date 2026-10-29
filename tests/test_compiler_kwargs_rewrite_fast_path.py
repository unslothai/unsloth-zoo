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

"""_rewrite_kwargs_param skips tokenize when no `def <name>` exists and only returns new text on a rename."""

import time
import tokenize

import pytest

import unsloth_zoo.compiler as compiler


def _count_tokenize(monkeypatch):
    calls = []
    real = tokenize.generate_tokens

    def counting(readline):
        calls.append(1)
        return real(readline)

    monkeypatch.setattr(compiler.tokenize, "generate_tokens", counting)
    return calls


_CLASS_SOURCE = (
    "class UnslothSFTTrainer(SFTTrainer):\n"
    "    def __init__(self, model, args = None, **kwargs):\n"
    '        x = f"{{literal}} {model}"\n'
    "        y = 1 + \\\n"
    "            2\n"
    "        super().__init__(model, args, **kwargs)\n"
)


def test_class_source_does_not_tokenize(monkeypatch):
    calls = _count_tokenize(monkeypatch)
    assert compiler._rewrite_kwargs_param(_CLASS_SOURCE, "UnslothSFTTrainer") == (
        _CLASS_SOURCE,
        None,
    )
    assert calls == []


@pytest.mark.parametrize(
    "source",
    [
        "def f(a, kwargs, **kwargs):\n    return a\n",
        "def  f (a, kwargs, **kwargs):\n    return a\n",
        "def \\\nf(a, kwargs, **kwargs):\n    return a\n",
        "def\t\\\r\n f(a, kwargs, **kwargs):\r\n    return a\r\n",
        "async def f(kwargs, **kwargs):\n    pass\n",
        "class A:\n    def f(self, kwargs, **kwargs):\n        pass\n",
        "def f(a,\tkwargs, **kwargs):\n    pass\n",
        "def f(\n    a,\n    kwargs,\n    **kwargs,\n):\n    '''doc'''\n    return a\n",
    ],
)
def test_renames(source):
    new_source, alias = compiler._rewrite_kwargs_param(source, "f")
    assert alias == "kwargs_"
    assert "kwargs_" in new_source


@pytest.mark.parametrize(
    "source",
    [
        "def g(kwargs, **kwargs):\n    pass\n",
        "s = 'def f1(kwargs, **kwargs)'\n",
        "def f(a, **kwargs):\n    x = f'{{a}} {b}'  \n    y = 1 + \\\n\\\n 2\n",
        "def f(a, kwargs, **kwargs",
        "x = 1\n   ",
        "def forward(self, x, **kwargs):\n    " + "#" * 5000 + "\n    return x\n",
    ],
)
def test_no_rename_returns_source_unchanged(source):
    # untokenize is not the identity on every Python, so no rename must mean the input back.
    start = time.perf_counter()
    assert compiler._rewrite_kwargs_param(source, "f") == (source, None)
    assert time.perf_counter() - start < 1.0


def test_later_def_without_collision_keeps_alias():
    source = (
        "def f(a, kwargs, **kwargs):\n    return kwargs\n"
        "def f(b, **kwargs):\n    return b\n"
    )
    new_source, alias = compiler._rewrite_kwargs_param(source, "f")
    assert alias == "kwargs_"
    assert "def f(a, kwargs_, **kwargs)" in new_source
