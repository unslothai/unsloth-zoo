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

from __future__ import annotations

import sys
import types

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_mlx_shim():
    from mlx_simulation import simulate_mlx_on_torch

    simulate_mlx_on_torch()


# Verbatim from mlx-vlm 0.6.3.
_VLM_MSG = (
    "ModelConfig.__init__() missing 1 required positional argument: 'text_config'"
)


def test_vlm_message_shape_is_recognized():
    from unsloth_zoo.mlx.loader import _missing_mlx_config_keys

    assert _missing_mlx_config_keys(_VLM_MSG) == ["text_config"]
    assert _missing_mlx_config_keys(
        "ModelConfig.__init__() missing 3 required positional arguments: "
        "'text_config', 'vision_config', and 'model_type'"
    ) == ["text_config", "vision_config", "model_type"]


def test_library_named_in_the_message():
    from unsloth_zoo.mlx.loader import _raise_if_incomplete_mlx_config

    with pytest.raises(ValueError) as exc:
        _raise_if_incomplete_mlx_config(
            "unsloth/Example-VL", "qwen2_5_vl", _VLM_MSG, TypeError(_VLM_MSG),
            library="mlx-vlm",
        )
    msg = str(exc.value)
    assert "mlx-vlm" in msg
    assert "mlx-lm" not in msg
    assert "unsloth/Example-VL" in msg
    assert "text_config" in msg
    assert "qwen2_5_vl" in msg
    assert "Apple Silicon" in msg


def test_nested_sub_config_is_named():
    # Real mlx-vlm 0.7.4 on a qwen2_5_vl config.json with text_config dropped.
    from unsloth_zoo.mlx.loader import _raise_if_incomplete_mlx_config

    message = (
        "TextConfig.__init__() missing 2 required positional arguments: "
        "'hidden_size' and 'vocab_size'"
    )
    with pytest.raises(ValueError) as exc:
        _raise_if_incomplete_mlx_config(
            "unsloth/Example-VL", "qwen2_5_vl", message, TypeError(message),
            library="mlx-vlm",
        )
    assert "'hidden_size', 'vocab_size' (fields of TextConfig)" in str(exc.value)


def test_default_library_still_mlx_lm():
    from unsloth_zoo.mlx.loader import _raise_if_incomplete_mlx_config

    with pytest.raises(ValueError) as exc:
        _raise_if_incomplete_mlx_config(
            "unsloth/LFM2.5-230M", "lfm2",
            "ModelArgs.__init__() missing 1 required positional argument: "
            "'block_ff_dim'",
            TypeError("x"),
        )
    assert "mlx-lm" in str(exc.value)


def test_extra_weight_filter_converts_the_type_error():
    from unsloth_zoo.mlx.loader import _load_mlx_vlm_with_extra_weight_filter

    def _vlm_load(model_name, **kwargs):
        raise TypeError(_VLM_MSG)

    with pytest.raises(ValueError) as exc:
        _load_mlx_vlm_with_extra_weight_filter(
            "unsloth/Example-VL", "qwen2_5_vl", _vlm_load, {},
        )
    msg = str(exc.value)
    assert "text_config" in msg
    assert "mlx-vlm" in msg
    assert "config.json" in msg
    assert isinstance(exc.value.__cause__, TypeError)


def test_extra_weight_filter_passes_through_other_type_errors():
    from unsloth_zoo.mlx.loader import _load_mlx_vlm_with_extra_weight_filter

    drift = "load() got an unexpected keyword argument 'lazy'"

    def _vlm_load(model_name, **kwargs):
        raise TypeError(drift)

    with pytest.raises(TypeError) as exc:
        _load_mlx_vlm_with_extra_weight_filter(
            "unsloth/Example-VL", "qwen2_5_vl", _vlm_load, {},
        )
    assert str(exc.value) == drift


def test_successful_vlm_load_is_untouched():
    from unsloth_zoo.mlx.loader import _load_mlx_vlm_with_extra_weight_filter

    def _vlm_load(model_name, **kwargs):
        return ("model", "processor")

    assert _load_mlx_vlm_with_extra_weight_filter(
        "unsloth/Example-VL", "qwen2_5_vl", _vlm_load, {},
    ) == ("model", "processor")


def _patch_vlm_distributed(monkeypatch, error):
    from unsloth_zoo.mlx import loader

    def _sharded_load(path, **kwargs):
        raise error

    utils = types.ModuleType("mlx_vlm.utils")
    utils.get_model_path = lambda name, revision=None: "/nonexistent"
    utils.sharded_load = _sharded_load
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", utils)
    monkeypatch.setattr(loader, "_mlx_active_distributed_groups", lambda p, t: (None, object()))
    monkeypatch.setattr(loader, "_bind_mlx_vlm_processor_loader", lambda f, **k: f)
    monkeypatch.setattr(loader, "_bind_mlx_vlm_quantized_projector_loader", lambda f: f)
    monkeypatch.setattr(
        loader, "_materialize_mlx_vlm_config_override", lambda path, cfg, **k: (path, None)
    )
    return loader


def test_distributed_vlm_converts_the_type_error(monkeypatch):
    loader = _patch_vlm_distributed(monkeypatch, TypeError(_VLM_MSG))
    with pytest.raises(ValueError) as exc:
        loader._load_mlx_vlm_distributed(
            "unsloth/Example-VL", "qwen2_5_vl", config_override_data={"model_type": "x"}
        )
    assert "text_config" in str(exc.value) and "mlx-vlm" in str(exc.value)


def test_distributed_vlm_signature_drift_still_reported(monkeypatch):
    drift = "sharded_load() got an unexpected keyword argument 'tensor_group'"
    loader = _patch_vlm_distributed(monkeypatch, TypeError(drift))
    with pytest.raises(ImportError, match="newer mlx-vlm"):
        loader._load_mlx_vlm_distributed(
            "unsloth/Example-VL", "qwen2_5_vl", config_override_data={"model_type": "x"}
        )


def test_non_config_init_type_error_keeps_its_traceback():
    # A processor / model constructor missing an argument is not a config.json problem.
    from unsloth_zoo.mlx.loader import _raise_if_incomplete_mlx_config

    for message in (
        "Qwen2VLProcessor.__init__() missing 1 required positional argument: 'image_processor'",
        "Model.__init__() missing 1 required positional argument: 'config'",
    ):
        _raise_if_incomplete_mlx_config(
            "unsloth/Example-VL", "qwen2_5_vl", message, TypeError(message),
            library="mlx-vlm",
        )


def test_bare_init_form_needs_a_from_dict_frame():
    # Python 3.9 drops the class name; only a TypeError raised in from_dict is a config gap.
    from unsloth_zoo.mlx.loader import _raise_if_incomplete_mlx_config

    message = "__init__() missing 1 required positional argument: 'text_config'"

    def from_dict():
        raise TypeError(message)

    def build_processor():
        raise TypeError(message)

    try:
        build_processor()
    except TypeError as error:
        _raise_if_incomplete_mlx_config("m", "t", message, error, library="mlx-vlm")
    with pytest.raises(ValueError, match="text_config"):
        try:
            from_dict()
        except TypeError as error:
            _raise_if_incomplete_mlx_config("m", "t", message, error, library="mlx-vlm")
