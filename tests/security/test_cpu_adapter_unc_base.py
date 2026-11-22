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

"""A Hub adapter whose adapter_config.json names a UNC base must not make the Windows CPU loader probe that share."""
import types

import pytest

from unsloth_zoo import vision_utils
from unsloth_zoo.cpu import loader


def _fake_peft(monkeypatch, base):
    import peft
    cfg = types.SimpleNamespace(base_model_name_or_path = base, revision = None)
    monkeypatch.setattr(peft.PeftConfig, "from_pretrained", classmethod(lambda cls, *a, **k: cfg))


@pytest.mark.parametrize("base", [r"\\evil.example\share\model", "//evil.example/share/model"])
def test_hub_adapter_unc_base_refused(monkeypatch, base):
    monkeypatch.setattr(vision_utils, "_WINDOWS", True)
    _fake_peft(monkeypatch, base)
    with pytest.raises(ValueError, match = "network share"):
        loader._adapter_base_model("someone/adapter", None, None)


def test_hub_adapter_normal_base_kept(monkeypatch):
    monkeypatch.setattr(vision_utils, "_WINDOWS", True)
    _fake_peft(monkeypatch, "unsloth/Llama-3.2-1B-Instruct")
    assert loader._adapter_base_model("someone/adapter", None, None) == ("unsloth/Llama-3.2-1B-Instruct", None)


def test_local_adapter_keeps_user_chosen_base(monkeypatch, tmp_path):
    # A folder on the user's own disk is their choice, as before.
    (tmp_path / "adapter_config.json").write_text("{}")
    monkeypatch.setattr(vision_utils, "_WINDOWS", True)
    _fake_peft(monkeypatch, r"\\fileserver\models\base")
    assert loader._adapter_base_model(str(tmp_path), None, None) == (r"\\fileserver\models\base", None)


def test_not_windows_unchanged(monkeypatch):
    monkeypatch.setattr(vision_utils, "_WINDOWS", False)
    _fake_peft(monkeypatch, "//evil.example/share/model")
    assert loader._adapter_base_model("someone/adapter", None, None)[0] == "//evil.example/share/model"
