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

"""Decision-model loads never run a config.json `model_file` (mlx-vlm / mlx-lm exec it unchecked)."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_mlx_shim():
    if importlib.util.find_spec("mlx") is None:
        from mlx_simulation import simulate_mlx_on_torch

        simulate_mlx_on_torch()


@pytest.fixture
def vlm_utils(monkeypatch):
    # Same signature and model_file behaviour as mlx-vlm 0.7.x get_model_and_args.
    module = types.ModuleType("mlx_vlm.utils")

    def get_model_and_args(config, model_path=None):
        if model_path is not None and (model_file := config.get("model_file")):
            spec = importlib.util.spec_from_file_location("custom_model", Path(model_path) / model_file)
            spec.loader.exec_module(importlib.util.module_from_spec(spec))
            return None, "custom"
        return None, config["model_type"]

    module.get_model_and_args = get_model_and_args
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", module)
    return module


@pytest.mark.parametrize("loader", ["load_decision_model", "load_language_model_as_clef", "load_language_model_as_d1"])
def test_decision_loaders_switch_on_the_model_file_guard(loader, vlm_utils, tmp_path):
    from unsloth_zoo.mlx import decision

    marker = tmp_path / "ran.marker"
    (tmp_path / "custom_arch.py").write_text(f"open({str(marker)!r}, 'w').write('ran')\n")
    config = {"model_type": "lfm3_vl", "model_file": "custom_arch.py"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(Exception):   # the guard, or a later load step: either way model_file never runs
        getattr(decision, loader)(str(tmp_path))
    assert getattr(vlm_utils.get_model_and_args, "_unsloth_model_file_guard", False)
    with pytest.raises(ValueError, match="trust_remote_code=True"):
        vlm_utils.get_model_and_args(config, model_path=tmp_path)
    assert not marker.exists()
