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

"""Which `moe_utils` the sys.modules leak gate in conftest.py treats as a leak.

The gate has to separate two things that look identical in sys.modules. Compiling a
MoE architecture imports moe_utils out of the compile folder as ordinary production
behaviour, and CI points that folder at `$PWD/.lanes/compiled-<id>`, which lives for
the whole job. A test that loads a copy from tmp_path_factory leaves behind a module
whose directory is deleted underneath it, and every later bare `from moe_utils import
...` silently resolves to that copy.

Getting this backwards is not hypothetical in either direction: a rule with no
qualification failed test_compiler_dynamic_exec.py for compiling a MoE model, and no
rule at all is what let the tmp_path copy from
test_moe_weight_preprocessor_registry_shared.py break the recovery test in
test_compiled_cache_collective.py.
"""

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
    # The real thing rather than a hand-written path, so this keeps tracking
    # pytest's layout if it ever changes.
    real = tmp_path_factory.mktemp("compile_location") / "moe_utils.py"
    assert _is_from_pytest_tmp(_module(str(real)))


def test_the_ci_lane_compile_folder_is_not_a_leak():
    # consolidated-tests-ci.yml: UNSLOTH_COMPILE_LOCATION="$PWD/.lanes/compiled-$id".
    # This copy stays valid for the whole job, and flagging it fails every test that
    # compiles a MoE architecture.
    assert not _is_from_pytest_tmp(
        _module("/home/runner/work/unsloth-zoo/unsloth-zoo/.lanes/compiled-pyproject/moe_utils.py")
    )


def test_the_default_compiled_cache_is_not_a_leak():
    assert not _is_from_pytest_tmp(_module("unsloth_compiled_cache/moe_utils.py"))


def test_a_module_with_no_file_is_not_reported():
    # A namespace package or a bare ModuleType substitute: nothing to go stale.
    assert not _is_from_pytest_tmp(_module(None))
    assert not _is_from_pytest_tmp(None)
