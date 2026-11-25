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

"""Import-time test setup for spawned workers: a spawn child is a fresh interpreter, so conftest's
device-type preload and CPU shims never ran; call bootstrap() before importing unsloth_zoo."""

import importlib.util
import os
import pathlib
import sys

_NAME = "_unsloth_zoo_tests_conftest"


def bootstrap() -> None:
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    if _NAME in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        _NAME, pathlib.Path(__file__).resolve().with_name("conftest.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_NAME] = module
    spec.loader.exec_module(module)
