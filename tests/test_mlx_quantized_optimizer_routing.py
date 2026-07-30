# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Which optimizer names route to the 8-bit path, and what the user is told.

Pure name logic, so this runs under the torch shim on Linux CI instead of skipping.
The numerics live in test_mlx_quantized_optimizer_state.py (real MLX only), so
keeping the routing contract here means the gate still executes something.
"""

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


def _normalize(name):
    from unsloth_zoo.mlx.trainer import _normalize_mlx_optimizer_name
    return _normalize_mlx_optimizer_name(name)


@pytest.mark.parametrize("name", ["adamw_8bit", "adam_8bit"])
def test_8bit_names_reach_the_quantized_path(name):
    """These were rewritten to fp32 adamw with no message."""
    assert _normalize(name) == name, (
        f"{name} was collapsed to {_normalize(name)!r}; the 8-bit request is being "
        "silently downgraded to full-precision optimizer state"
    )


@pytest.mark.parametrize("name", [
    "paged_adamw_8bit",
    "adamw_bnb_8bit",
    "paged_adamw_32bit",
    "adamw_torch",
    "adamw_hf",
])
def test_paged_and_bnb_names_still_collapse(name):
    """Deliberate limit: "paged" promises CPU offload, "bnb" a library MLX lacks."""
    assert _normalize(name) == "adamw"


@pytest.mark.parametrize("name,expected", [("adamw", "adamw"), ("adam", "adam"), ("sgd", "sgd")])
def test_plain_names_unchanged(name, expected):
    assert _normalize(name) == expected


def test_unsupported_name_still_raises():
    with pytest.raises(ValueError, match="Unsupported MLX optimizer"):
        _normalize("definitely_not_an_optimizer")


def test_8bit_names_are_advertised_as_supported():
    from unsloth_zoo.mlx.trainer import SUPPORTED_MLX_OPTIMIZERS
    assert "adamw_8bit" in SUPPORTED_MLX_OPTIMIZERS
    assert "adam_8bit" in SUPPORTED_MLX_OPTIMIZERS


def test_description_states_what_is_and_is_not_quantized():
    """The message must name the actual shape of the saving."""
    from unsloth_zoo.mlx.optimizers_quantized import describe_quantized_optimizer
    message = describe_quantized_optimizer("adamw_8bit")

    assert "adamw_8bit" in message
    assert "FIRST moment" in message
    assert "group_size=64" in message
    assert "second moment stays float32" in message
    assert "multiple of 64" in message
