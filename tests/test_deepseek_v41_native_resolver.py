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

"""With no transformers.models.deepseek_v41, the DeepSeek-V4.1 patches resolve Unsloth's native implementation
(unsloth/models/deepseek_v41/modeling.py) from sys.modules, and never import unsloth themselves."""

import sys
import types

import pytest

pytest.importorskip("torch")
from unsloth_zoo.temporary_patches import deepseek_v41 as patches  # noqa: E402


@pytest.fixture
def no_port(monkeypatch):
    real = patches.importlib.util.find_spec
    monkeypatch.setattr(
        patches.importlib.util, "find_spec",
        lambda name, *a: None if name == "transformers.models.deepseek_v41" else real(name, *a),
    )


def test_resolves_the_native_module_once_imported(no_port, monkeypatch):
    for name in [n for n in sys.modules if n.endswith("deepseek_v41.modeling")]:
        monkeypatch.delitem(sys.modules, name)
    assert patches._modeling() is None
    native = types.ModuleType("unsloth.models.deepseek_v41.modeling")
    native.DeepseekV41DecoderLayer = type("DeepseekV41DecoderLayer", (), {})
    monkeypatch.setitem(sys.modules, "unsloth.models.deepseek_v41.modeling", native)
    assert patches._modeling() is native


def test_ignores_unrelated_modules_with_the_same_suffix(no_port, monkeypatch):
    for name in [n for n in sys.modules if n.endswith("deepseek_v41.modeling")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "other.deepseek_v41.modeling", types.ModuleType("other.deepseek_v41.modeling"))
    assert patches._modeling() is None
