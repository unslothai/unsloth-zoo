# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
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
"""What the sys.modules leak guard in conftest must and must not report."""

import sys
from importlib.machinery import ModuleSpec
from types import ModuleType

import pytest

import conftest


class _Item:
    """The one attribute the hooks read off a pytest item."""

    nodeid = "tests/test_fake.py::test_fake"


def _apply(entries):
    for name, module in entries.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _run_guard(before, after):
    """Run the guard over a test that started with `before` and left `after` behind.

    Both states are set explicitly rather than inherited, because by the time this file
    runs another test may already have imported unsloth_zoo and put the stub in place.
    """
    saved = {name: sys.modules.get(name) for name in conftest._GUARDED_MODULES}
    item = _Item()
    try:
        _apply(before)
        conftest.pytest_runtest_setup(item)
        _apply(after)
        conftest.pytest_runtest_teardown(item, None)
    finally:
        _apply(saved)


def _unsloth_stub(name):
    """What `_BnbFinder` builds: a bare module carrying the stub's spec origin."""
    module = ModuleType(name)
    module.__spec__ = ModuleSpec(name, None, origin = "bitsandbytes_stub")
    return module


def _partial_stub(name):
    """The shape that motivated the guard: a bare module carrying one chosen symbol."""
    module = ModuleType(name)
    module.dequantize_4bit = lambda *args, **kwargs: None
    return module


def _real_module(name):
    module = ModuleType(name)
    module.__file__ = "/somewhere/%s/__init__.py" % name.replace(".", "/")
    return module


@pytest.mark.parametrize("name", ["bitsandbytes.nn", "bitsandbytes.functional"])
def test_unsloth_bitsandbytes_stub_is_not_a_leak(name):
    # `import unsloth_zoo` installs this itself on a host with no real bitsandbytes, so
    # the first test to import the package was reported as having swapped a module out.
    _run_guard({name: None}, {name: _unsloth_stub(name)})


def test_importing_the_real_package_is_not_a_leak():
    _run_guard({"bitsandbytes.nn": None}, {"bitsandbytes.nn": _real_module("bitsandbytes.nn")})


def test_a_partial_substitute_is_still_a_leak():
    # test_vllm_to_hf_conversion left a bitsandbytes.functional carrying only
    # dequantize_4bit, and every later `from bitsandbytes.functional import QuantState`
    # failed. That is the case this guard exists for.
    with pytest.raises(AssertionError, match = "bitsandbytes.functional"):
        _run_guard(
            {"bitsandbytes.functional": None},
            {"bitsandbytes.functional": _partial_stub("bitsandbytes.functional")},
        )


def test_replacing_a_module_that_was_there_is_still_a_leak():
    # Even by the unsloth stub: nothing installs it over a real bitsandbytes, so a test
    # is the only thing that could have.
    with pytest.raises(AssertionError, match = "bitsandbytes.nn"):
        _run_guard(
            {"bitsandbytes.nn": _real_module("bitsandbytes.nn")},
            {"bitsandbytes.nn": _unsloth_stub("bitsandbytes.nn")},
        )
