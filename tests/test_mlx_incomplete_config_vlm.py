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

import inspect

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


def test_retry_path_is_deliberately_not_guarded():
    from unsloth_zoo.mlx.loader import _load_mlx_vlm_with_extra_weight_filter

    source = inspect.getsource(_load_mlx_vlm_with_extra_weight_filter)
    assert source.count("_raise_if_incomplete_mlx_config") == 1


def test_distributed_guard_precedes_the_signature_drift_branch():
    from unsloth_zoo.mlx.loader import _load_mlx_vlm_distributed

    source = inspect.getsource(_load_mlx_vlm_distributed)
    assert "_raise_if_incomplete_mlx_config" in source
    guard_at = source.index("_raise_if_incomplete_mlx_config")
    drift_at = source.index('"tensor_group" not in message')
    assert guard_at < drift_at
    from unsloth_zoo.mlx.loader import _missing_mlx_config_keys

    assert _missing_mlx_config_keys(
        "sharded_load() got an unexpected keyword argument 'tensor_group'"
    ) == []
    assert _missing_mlx_config_keys(
        "sharded_load() missing 1 required positional argument: 'tensor_group'"
    ) == []


def test_runtime_quant_vlm_path_is_guarded():
    from unsloth_zoo.mlx import loader

    source = inspect.getsource(loader)
    # Window = runtime-quant VLM branch: its opening print to its QK-norm marker comment.
    branch_start = 'via mlx-vlm (VLM, "'
    branch_end = "Pre-quantize load bypasses the extra-weight filter"
    assert source.count(branch_start) == 1
    assert source.count(branch_end) == 1
    window = source[source.index(branch_start):source.index(branch_end)]
    assert "_raise_if_incomplete_mlx_config" in window
    assert 'library="mlx-vlm"' in window


def test_every_vlm_load_entry_point_is_guarded():
    # 3 VLM + 2 mlx-lm call sites; a new unguarded load path should update this.
    from unsloth_zoo.mlx import loader

    source = inspect.getsource(loader)
    calls = source.count("_raise_if_incomplete_mlx_config(")
    definition = source.count("def _raise_if_incomplete_mlx_config(")
    assert calls - definition == 5
    assert source.count('library="mlx-vlm"') == 3
