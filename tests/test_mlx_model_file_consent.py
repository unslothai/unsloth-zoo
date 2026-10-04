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

"""config.json `model_file` must not run repository code unless trust_remote_code=True."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_mlx_shim():
    from mlx_simulation import simulate_mlx_on_torch

    simulate_mlx_on_torch()


def _exec_model_file(model_path, model_file):
    spec = importlib.util.spec_from_file_location("custom_model", Path(model_path) / model_file)
    arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arch)
    return arch


def _fake_mlx_lm_utils():
    # Same signature and model_file behaviour as mlx-lm 0.31.3 load_model.
    module = types.ModuleType("mlx_lm.utils")

    def load_model(model_path, lazy=False, strict=True, model_config={}, get_model_classes=None):
        config = json.loads((Path(model_path) / "config.json").read_text())
        config.update(model_config)
        if (model_file := config.get("model_file")) is not None:
            _exec_model_file(model_path, model_file)
        return "model", config

    def load(path_or_hf_repo):
        # mlx-lm's load() looks load_model up in its module globals at call time.
        return module.load_model(path_or_hf_repo)

    module.load_model = load_model
    module.load = load
    return module


def _fake_mlx_vlm_utils():
    # Same signature and model_file behaviour as mlx-vlm 0.7.4 get_model_and_args.
    module = types.ModuleType("mlx_vlm.utils")

    def get_model_and_args(config, model_path=None):
        if model_path is not None and (model_file := config.get("model_file")):
            return _exec_model_file(model_path, model_file), "custom"
        return None, config["model_type"]

    module.get_model_and_args = get_model_and_args
    return module


@pytest.fixture
def fake_loaders(monkeypatch):
    lm_utils, vlm_utils = _fake_mlx_lm_utils(), _fake_mlx_vlm_utils()
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", lm_utils)
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", vlm_utils)
    return lm_utils, vlm_utils


def _model_dir(tmp_path, *, model_file):
    marker = tmp_path / "ran.marker"
    config = {"model_type": "llama"}
    if model_file:
        config["model_file"] = "custom_arch.py"
        (tmp_path / "custom_arch.py").write_text(
            f"open({str(marker)!r}, 'w').write('ran')\n"
        )
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path, marker


def test_mlx_lm_model_file_refused_without_trust(fake_loaders, tmp_path):
    from unsloth_zoo.mlx.loader import _install_mlx_model_file_guard

    lm_utils, _ = fake_loaders
    model_path, marker = _model_dir(tmp_path, model_file=True)
    _install_mlx_model_file_guard()
    with pytest.raises(ValueError, match="trust_remote_code=True"):
        lm_utils.load_model(model_path)
    # mlx_lm.load resolves load_model through the module, so it is covered too.
    with pytest.raises(ValueError, match="custom_arch.py"):
        lm_utils.load(model_path)
    assert not marker.exists()


def test_mlx_vlm_model_file_refused_without_trust(fake_loaders, tmp_path):
    from unsloth_zoo.mlx.loader import _install_mlx_model_file_guard

    _, vlm_utils = fake_loaders
    model_path, marker = _model_dir(tmp_path, model_file=True)
    config = json.loads((model_path / "config.json").read_text())
    _install_mlx_model_file_guard()
    with pytest.raises(ValueError, match="trust_remote_code=True"):
        vlm_utils.get_model_and_args(config, model_path=model_path)
    assert not marker.exists()
    # Without model_path mlx-vlm never execs model_file, so nothing is refused.
    assert vlm_utils.get_model_and_args(config) == (None, "llama")


def test_model_file_allowed_when_caller_trusts(fake_loaders, tmp_path):
    from unsloth_zoo.mlx.loader import _scoped_mlx_model_file_trust

    lm_utils, _ = fake_loaders
    model_path, marker = _model_dir(tmp_path, model_file=True)

    @_scoped_mlx_model_file_trust
    def from_pretrained(model_name, token=None, trust_remote_code=False):
        return lm_utils.load_model(model_name)

    with pytest.raises(ValueError):
        from_pretrained(model_path)
    assert not marker.exists()
    from_pretrained(model_path, trust_remote_code=True)
    assert marker.exists()
    # Trust is scoped to the call: a later untrusted load is refused again.
    marker.unlink()
    with pytest.raises(ValueError):
        from_pretrained(model_path, None, False)
    assert not marker.exists()


def test_plain_config_unaffected(fake_loaders, tmp_path):
    from unsloth_zoo.mlx.loader import _install_mlx_model_file_guard

    lm_utils, vlm_utils = fake_loaders
    model_path, _ = _model_dir(tmp_path, model_file=False)
    _install_mlx_model_file_guard()
    model, config = lm_utils.load_model(model_path, lazy=True)
    assert model == "model" and config["model_type"] == "llama"
    assert vlm_utils.get_model_and_args(config, model_path=model_path) == (None, "llama")


def test_guard_is_idempotent(fake_loaders):
    from unsloth_zoo.mlx.loader import _install_mlx_model_file_guard

    lm_utils, vlm_utils = fake_loaders
    _install_mlx_model_file_guard()
    first_lm, first_vlm = lm_utils.load_model, vlm_utils.get_model_and_args
    _install_mlx_model_file_guard()
    assert lm_utils.load_model is first_lm
    assert vlm_utils.get_model_and_args is first_vlm
    assert first_lm._unsloth_model_file_guard
    assert not getattr(first_lm.__wrapped__, "_unsloth_model_file_guard", False)


def test_releases_that_gate_model_file_are_left_alone(monkeypatch):
    from unsloth_zoo.mlx.loader import _install_mlx_model_file_guard

    # mlx-lm >= 0.32 takes trust_remote_code itself; mlx-vlm < 0.7 has no model_path.
    lm_utils = types.ModuleType("mlx_lm.utils")
    vlm_utils = types.ModuleType("mlx_vlm.utils")

    def load_model(model_path, lazy=False, strict=True, model_config={}, trust_remote_code=False):
        return "model", {}

    def get_model_and_args(config):
        return None, config["model_type"]

    lm_utils.load_model = load_model
    vlm_utils.get_model_and_args = get_model_and_args
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", lm_utils)
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", vlm_utils)
    _install_mlx_model_file_guard()
    assert lm_utils.load_model is load_model
    assert vlm_utils.get_model_and_args is get_model_and_args


def test_from_pretrained_is_scoped():
    from unsloth_zoo.mlx.loader import FastMLXModel

    assert hasattr(FastMLXModel.from_pretrained, "__wrapped__")
    import inspect

    assert inspect.signature(FastMLXModel.from_pretrained).parameters["trust_remote_code"].default is False


def test_explicit_model_file_none_override_loads(fake_loaders, tmp_path):
    # mlx-lm applies model_config over config.json, so model_file=None means nothing is executed.
    from unsloth_zoo.mlx.loader import _install_mlx_model_file_guard

    lm_utils, _ = fake_loaders
    model_path, marker = _model_dir(tmp_path, model_file=True)
    _install_mlx_model_file_guard()
    model, config = lm_utils.load_model(model_path, model_config={"model_file": None})
    assert model == "model" and config["model_file"] is None
    assert not marker.exists()


def test_teacher_load_does_not_inherit_student_trust(fake_loaders, tmp_path, monkeypatch):
    from unsloth_zoo.mlx import distill
    from unsloth_zoo.mlx.loader import _MLX_MODEL_FILE_TRUST

    lm_utils, _ = fake_loaders
    model_path, marker = _model_dir(tmp_path, model_file=True)
    fake_mlx_lm = types.ModuleType("mlx_lm")
    fake_mlx_lm.load = lambda path: lm_utils.load(path)
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    token = _MLX_MODEL_FILE_TRUST.set(True)
    try:
        with pytest.raises(ValueError, match="trust_remote_code=True"):
            distill.load_teacher(str(model_path))
    finally:
        _MLX_MODEL_FILE_TRUST.reset(token)
    assert not marker.exists()
