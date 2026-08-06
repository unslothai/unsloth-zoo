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

# A required mlx-lm ModelArgs field absent from config.json raises a bare
# TypeError naming neither the model nor the file, which reads as missing Apple
# Silicon support; the guard must name the config as the cause instead.

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_mlx_shim():
    from mlx_simulation import simulate_mlx_on_torch

    simulate_mlx_on_torch()


# Verbatim from mlx-lm 0.31.3 loading unsloth/LFM2.5-230M, whose mirrored
# config.json lost upstream's "block_ff_dim": 2560 (unslothai/unsloth#7306).
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
    # The whole point: stop readers concluding MLX is unsupported.
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
    # Only the missing-required-argument shape is ours; anything else must keep
    # its original traceback rather than be relabelled a config problem.
    from unsloth_zoo.mlx.loader import _raise_if_incomplete_mlx_config

    for message in (
        "unsupported operand type(s) for +: 'int' and 'str'",
        "load_model() got an unexpected keyword argument 'strict'",
    ):
        _raise_if_incomplete_mlx_config(
            "unsloth/Example", "lfm2", message, TypeError(message)
        )


def test_mlx_lm_signature_drift_is_not_a_config_problem():
    # This loader deliberately tolerates mlx-lm changing load_model/_download's
    # signature between releases (see the bypass comment in
    # _load_mlx_lm_with_strict_fallback). Those failures have the same
    # "missing N required positional arguments" wording as a genuinely
    # incomplete config, so anchoring on __init__ is what keeps them apart:
    # misreporting one as "config.json is missing 'model_path'" would send a
    # reader to the wrong repo entirely.
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
    # Bare __init__ (no owning class in the message) still counts.
    assert _missing_mlx_config_keys(
        "__init__() missing 1 required positional argument: 'block_ff_dim'"
    ) == ["block_ff_dim"]
    assert _missing_mlx_config_keys("something else entirely") == []
