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

"""Checks which moe_utils the conftest sys.modules leak gate flags: pytest tmp_path copies (directory deleted later) are
leaks, the CI .lanes compile folder is not; both directions broke tests."""

from __future__ import annotations

import types

from conftest import _GUARDED_BARE_MODULES, _is_from_pytest_tmp


def _module(path):
    module = types.ModuleType("moe_utils")
    if path is not None:
        module.__file__ = path
    return module


def test_moe_utils_is_the_guarded_bare_name():
    assert "moe_utils" in _GUARDED_BARE_MODULES


def test_a_copy_from_a_pytest_temp_directory_is_a_leak(tmp_path_factory):
    # Uses the real pytest temp layout rather than a hand-written path, so the test tracks pytest if its layout ever changes.
    real = tmp_path_factory.mktemp("compile_location") / "moe_utils.py"
    assert _is_from_pytest_tmp(_module(str(real)))


def test_the_ci_lane_compile_folder_is_not_a_leak():
    # consolidated-tests-ci.yml sets UNSLOTH_COMPILE_LOCATION to $PWD/.lanes/compiled-$id; that copy stays valid for the
    # whole job, so flagging it fails every MoE compile test.
    assert not _is_from_pytest_tmp(
        _module("/home/runner/work/unsloth-zoo/unsloth-zoo/.lanes/compiled-pyproject/moe_utils.py")
    )


def test_the_default_compiled_cache_is_not_a_leak():
    assert not _is_from_pytest_tmp(_module("unsloth_compiled_cache/moe_utils.py"))


def test_a_module_with_no_file_is_not_reported():
    # A namespace package or a bare ModuleType substitute: nothing to go stale.
    assert not _is_from_pytest_tmp(_module(None))
    assert not _is_from_pytest_tmp(None)


# Bitsandbytes names watch for a bare ModuleType: unsloth_zoo mints one on Apple Silicon, which failed teardown on macOS (unsloth-zoo issue 1217).

from conftest import _GUARDED_MODULES, _is_module_stub, _is_unsloth_stub


def _stub_module(name, marked):
    module = types.ModuleType(name)
    if marked:
        module.IS_UNSLOTH_STUB = True
    return module


def test_the_bitsandbytes_names_are_guarded():
    assert "bitsandbytes.nn" in _GUARDED_MODULES


def test_the_zoo_stub_is_not_a_leak():
    stub = _stub_module("bitsandbytes.nn", marked = True)
    # It is a bare ModuleType, so the file test alone cannot tell it from a swap.
    assert _is_module_stub(stub)
    assert _is_unsloth_stub(stub)


def test_an_unmarked_substitute_is_still_a_leak():
    """The narrowing must still flag an unmarked hand rolled substitute, such as a test that leaves its own bitsandbytes.functional module in sys.modules."""
    assert not _is_unsloth_stub(_stub_module("bitsandbytes.functional", marked = False))
    assert not _is_unsloth_stub(None)


def test_the_real_wheel_is_not_a_stub():
    assert not _is_unsloth_stub(_module("/usr/lib/python3/bitsandbytes/nn/__init__.py"))
