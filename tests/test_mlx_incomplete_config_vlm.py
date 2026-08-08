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

# The mlx-vlm side of the incomplete-config diagnostic. mlx-vlm's
# BaseModelConfig.from_dict (mlx_vlm/models/base.py) filters the config on
# inspect.signature(cls).parameters before constructing, exactly as mlx-lm's
# from_dict does, so a required field missing from a mirrored config.json fails
# in ModelConfig.__init__ with a bare TypeError. Without a guard the VLM paths
# surface that raw error, which reads as missing Apple Silicon support.

from __future__ import annotations

import inspect

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_mlx_shim():
    from mlx_simulation import simulate_mlx_on_torch

    simulate_mlx_on_torch()


# Verbatim from mlx-vlm 0.6.3. 43 of its architectures declare at least one
# required (no-default) ModelConfig field -- text_config, vision_config,
# projector_config and model_type are the common ones -- so this is the shape a
# mirrored VLM config.json produces when transformers drops a key.
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
    # A reader sent to the wrong project wastes the same time the bare
    # TypeError did, so the VLM paths must say mlx-vlm, not mlx-lm.
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
    # #1004's three text-side call sites pass no library; changing their message
    # is out of scope for the VLM change.
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
    # The real behavioural test: vlm_load is injected, so the whole path runs.
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
    # The original TypeError stays reachable for anyone reading the traceback.
    assert isinstance(exc.value.__cause__, TypeError)


def test_extra_weight_filter_passes_through_other_type_errors():
    # mlx-vlm signature drift at our own call site must keep its traceback,
    # exactly as on the text side.
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
    # The retry inside the extra-weight filter runs only after a ValueError,
    # which means ModelConfig already constructed -- a missing-key TypeError
    # cannot originate there. Guarding it would be dead code, so assert the
    # first load is the only guarded one.
    from unsloth_zoo.mlx.loader import _load_mlx_vlm_with_extra_weight_filter

    source = inspect.getsource(_load_mlx_vlm_with_extra_weight_filter)
    assert source.count("_raise_if_incomplete_mlx_config") == 1


def test_distributed_guard_precedes_the_signature_drift_branch():
    # Ordering is load-bearing in _load_mlx_vlm_distributed: it already owns a
    # TypeError handler for sharded_load signature drift. The incomplete-config
    # check has to run first (it is the precise, __init__-anchored test) while
    # still leaving the drift branch reachable.
    from unsloth_zoo.mlx.loader import _load_mlx_vlm_distributed

    source = inspect.getsource(_load_mlx_vlm_distributed)
    assert "_raise_if_incomplete_mlx_config" in source
    guard_at = source.index("_raise_if_incomplete_mlx_config")
    drift_at = source.index('"tensor_group" not in message')
    assert guard_at < drift_at
    # The drift branch must survive: an __init__-anchored pattern cannot match
    # "sharded_load() got an unexpected keyword argument 'tensor_group'".
    from unsloth_zoo.mlx.loader import _missing_mlx_config_keys

    assert _missing_mlx_config_keys(
        "sharded_load() got an unexpected keyword argument 'tensor_group'"
    ) == []
    assert _missing_mlx_config_keys(
        "sharded_load() missing 1 required positional argument: 'tensor_group'"
    ) == []


def test_runtime_quant_vlm_path_is_guarded():
    # The runtime-quant branch calls vlm_load directly and so bypasses
    # _load_mlx_vlm_with_extra_weight_filter -- the same bypass the QK-norm
    # guard already had to be repeated for. Without its own guard a
    # load_in_4bit VLM load keeps the bare TypeError.
    from unsloth_zoo.mlx import loader

    source = inspect.getsource(loader)
    marker = "Pre-quantize load bypasses the extra-weight filter"
    assert marker in source
    window = source[source.index(marker) - 1200:source.index(marker)]
    assert "_raise_if_incomplete_mlx_config" in window


def test_every_vlm_load_entry_point_is_guarded():
    # Guard against a fourth VLM load path being added later without the
    # diagnostic. Counts the guard's call sites: three VLM ones plus the two
    # text-side ones #1004 installed.
    from unsloth_zoo.mlx import loader

    source = inspect.getsource(loader)
    calls = source.count("_raise_if_incomplete_mlx_config(")
    definition = source.count("def _raise_if_incomplete_mlx_config(")
    assert calls - definition == 5
    assert source.count('library="mlx-vlm"') == 3
