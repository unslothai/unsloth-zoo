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

"""patch_torch_compile must not write dynamo config keys torch marks deprecated (torch 2.13 warns on each write)."""

import pytest

torch = pytest.importorskip("torch")
import torch._dynamo.config as dynamo_config

from unsloth_zoo.patching_utils import patch_torch_compile


class _DeprecatedEntry:
    deprecated = True

    def __init__(self, entry):
        self._entry = entry

    def __getattr__(self, name):
        return getattr(self._entry, name)


def test_deprecated_dynamo_key_is_not_written(monkeypatch):
    key = "numpy_default_float"
    assert key in dynamo_config._config
    monkeypatch.setattr(dynamo_config, key, "float64")
    monkeypatch.setitem(dynamo_config._config, key, _DeprecatedEntry(dynamo_config._config[key]))
    monkeypatch.setattr(dynamo_config, "recompile_limit", 8)

    patch_torch_compile(debug = False, O3 = False, ignore_errors = True)

    assert getattr(dynamo_config, key) == "float64"
    assert dynamo_config.recompile_limit == 1024

