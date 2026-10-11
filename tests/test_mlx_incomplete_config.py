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

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_mlx_shim():
    from mlx_simulation import simulate_mlx_on_torch

    simulate_mlx_on_torch()


# Verbatim from mlx-lm 0.31.3 on unsloth/LFM2.5-230M (unslothai/unsloth#7306).
_LFM2_MSG = (
    "ModelArgs.__init__() missing 1 required positional argument: 'block_ff_dim'"
)


def test_missing_config_key_raises_actionable_error():
    from unsloth_zoo.mlx.loader import _raise_if_incomplete_mlx_config

    with pytest.raises(ValueError) as exc:
        _raise_if_incomplete_mlx_config(
            "unsloth/LFM2.5-230M", "lfm2", _LFM2_MSG, TypeError(_LFM2_MSG)
        )
    msg = str(exc.value)
    assert "unsloth/LFM2.5-230M" in msg
    assert "block_ff_dim" in msg
    assert "config.json" in msg
    assert "lfm2" in msg
    assert "Apple Silicon" in msg


def test_multiple_missing_keys_all_named():
    from unsloth_zoo.mlx.loader import _raise_if_incomplete_mlx_config

    message = (
        "ModelArgs.__init__() missing 2 required positional arguments: "
        "'block_ff_dim' and 'block_multiple_of'"
    )
    with pytest.raises(ValueError) as exc:
        _raise_if_incomplete_mlx_config(
            "unsloth/Example", "lfm2", message, TypeError(message)
        )
    msg = str(exc.value)
    assert "block_ff_dim" in msg and "block_multiple_of" in msg
    assert "keys" in msg


def test_unrelated_type_error_falls_through():
    from unsloth_zoo.mlx.loader import _raise_if_incomplete_mlx_config

    for message in (
        "unsupported operand type(s) for +: 'int' and 'str'",
        "load_model() got an unexpected keyword argument 'strict'",
    ):
        _raise_if_incomplete_mlx_config(
            "unsloth/Example", "lfm2", message, TypeError(message)
        )


def test_mlx_lm_signature_drift_is_not_a_config_problem():
    from unsloth_zoo.mlx.loader import (
        _missing_mlx_config_keys,
        _raise_if_incomplete_mlx_config,
    )

    for message in (
        "load_model() missing 1 required positional argument: 'model_path'",
        "_download() missing 1 required positional argument: 'repo'",
        "load_tokenizer() missing 2 required positional arguments: 'a' and 'b'",
    ):
        assert _missing_mlx_config_keys(message) == []
        _raise_if_incomplete_mlx_config(
            "unsloth/Example", "lfm2", message, TypeError(message)
        )


def test_key_extraction_shapes():
    from unsloth_zoo.mlx.loader import _missing_mlx_config_keys

    assert _missing_mlx_config_keys(_LFM2_MSG) == ["block_ff_dim"]
    assert _missing_mlx_config_keys(
        "ModelArgs.__init__() missing 3 required positional arguments: "
        "'a', 'b', and 'c'"
    ) == ["a", "b", "c"]
    assert _missing_mlx_config_keys(
        "__init__() missing 1 required positional argument: 'block_ff_dim'"
    ) == ["block_ff_dim"]
    assert _missing_mlx_config_keys(
        "TextConfig.__init__() missing 1 required positional argument: 'vocab_size'"
    ) == ["vocab_size"]
    assert _missing_mlx_config_keys(
        "Linear.__init__() missing 1 required positional argument: 'output_dims'"
    ) == []
    assert _missing_mlx_config_keys("something else entirely") == []


def _stub_text_load(monkeypatch, tmp_path, config, error):
    import json
    import mlx_lm.utils as lm_utils

    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(lm_utils, "_download", lambda *a, **k: tmp_path)

    def load_model(*args, **kwargs):
        raise TypeError(error)

    monkeypatch.setattr(lm_utils, "load_model", load_model)


def test_text_load_names_missing_config_key(monkeypatch, tmp_path):
    import unsloth_zoo.mlx.loader as loader

    _stub_text_load(monkeypatch, tmp_path, {"model_type": "lfm2"}, _LFM2_MSG)
    with pytest.raises(ValueError, match="block_ff_dim"):
        loader._load_mlx_lm_with_strict_fallback("unsloth/LFM2.5-230M", "lfm2", None, {})


def test_text_load_keeps_non_config_init_type_error(monkeypatch, tmp_path):
    import unsloth_zoo.mlx.loader as loader

    message = "Linear.__init__() missing 1 required positional argument: 'output_dims'"
    _stub_text_load(monkeypatch, tmp_path, {"model_type": "lfm2"}, message)
    with pytest.raises(TypeError, match="output_dims"):
        loader._load_mlx_lm_with_strict_fallback("unsloth/Example", "lfm2", None, {})


def _stub_model_args(monkeypatch, *required):
    import dataclasses
    import mlx_lm.utils as lm_utils

    fields = [(name, object) for name in ("model_type", *required)]
    fields.append(("hidden_size", int, dataclasses.field(default=0)))
    args_class = dataclasses.make_dataclass("ModelArgs", fields)
    monkeypatch.setattr(
        lm_utils, "_get_classes", lambda config: (object, args_class), raising=False
    )


def _backfill(tmp_path, config, model_config=None):
    import json
    from unsloth_zoo.mlx.loader import _mlx_lm_config_backfill

    (tmp_path / "config.json").write_text(json.dumps(config))
    return _mlx_lm_config_backfill("unsloth/Example", str(tmp_path), model_config or {})


def test_backfill_restores_keys_dropped_by_resaves(monkeypatch, tmp_path):
    # Values the transformers re-saves of unsloth/Qwen3-Coder-Next, LFM2.5-230M and ERNIE-4.5 dropped (#8126).
    _stub_model_args(monkeypatch, "rope_theta", "block_ff_dim", "tie_word_embeddings")
    filled = _backfill(tmp_path, {
        "model_type": "ernie4_5",
        "rope_parameters": {"rope_theta": 5000000, "rope_type": "default"},
        "intermediate_size": 2560,
    })
    assert filled == {"rope_theta": 5000000, "block_ff_dim": 2560, "tie_word_embeddings": True}


def test_backfill_leaves_complete_and_underivable_configs(monkeypatch, tmp_path):
    _stub_model_args(monkeypatch, "rope_theta", "num_hidden_layers")
    complete = {"temperature": 0.5}
    assert _backfill(tmp_path, {"model_type": "qwen3_next", "rope_theta": 1e6}, complete) is complete
    assert _backfill(tmp_path, {"model_type": "nemotron_h"}) == {}
    # Explicit values and the model_config override always win.
    assert _backfill(
        tmp_path, {"model_type": "qwen3_next", "rope_parameters": {"rope_theta": 1}},
        {"rope_theta": 7},
    ) == {"rope_theta": 7}


def test_text_load_passes_backfilled_config(monkeypatch, tmp_path):
    import json
    import mlx_lm.utils as lm_utils
    import unsloth_zoo.mlx.loader as loader

    _stub_model_args(monkeypatch, "block_ff_dim")
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "lfm2", "intermediate_size": 2560}))
    monkeypatch.setattr(lm_utils, "_download", lambda *a, **k: tmp_path)
    seen = {}

    def load_model(path, lazy=False, strict=True, model_config=None):
        seen.update(model_config)
        raise TypeError(_LFM2_MSG)

    monkeypatch.setattr(lm_utils, "load_model", load_model)
    with pytest.raises(ValueError, match="block_ff_dim"):
        loader._load_mlx_lm_with_strict_fallback("unsloth/LFM2.5-230M", "lfm2", None, {})
    assert seen == {"block_ff_dim": 2560}
