# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""_reject_unsupported_hf_quantization_fields must accept an explicitly-passed
bnb_4bit_use_double_quant (nested/double-quant is supported on the MLX load
path) while still failing loud on genuinely unsupported BitsAndBytesConfig
fields. Pure-logic; runs under the torch shim so Linux CI covers it.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


def _reject():
    from unsloth_zoo.mlx.loader import _reject_unsupported_hf_quantization_fields
    return _reject_unsupported_hf_quantization_fields


@pytest.mark.parametrize("cfg", [
    {"bnb_4bit_use_double_quant": True},
    {"bnb_4bit_use_double_quant": False},
    {"bnb_4bit_quant_type": "fp4", "bnb_4bit_use_double_quant": True},
    {"load_in_4bit": True, "bnb_4bit_use_double_quant": True},
])
def test_double_quant_is_accepted(cfg):
    """double_quant (True or False) no longer triggers rejection."""
    _reject()(cfg)  # must not raise


def test_double_quant_not_in_error_message():
    """When another field is bad, double_quant=True is not among the rejected."""
    with pytest.raises(ValueError) as exc:
        _reject()({
            "bnb_4bit_quant_type": "nf4",          # still unsupported
            "bnb_4bit_use_double_quant": True,     # now supported
        })
    msg = str(exc.value)
    assert "bnb_4bit_quant_type" in msg
    assert "bnb_4bit_use_double_quant" not in msg


@pytest.mark.parametrize("cfg,bad_field", [
    ({"bnb_4bit_quant_type": "nf4"}, "bnb_4bit_quant_type"),
    ({"llm_int8_threshold": 3.0}, "llm_int8_threshold"),
    ({"bnb_4bit_quant_storage": "float16"}, "bnb_4bit_quant_storage"),
    ({"llm_int8_enable_fp32_cpu_offload": True}, "llm_int8_enable_fp32_cpu_offload"),
    ({"bnb_4bit_compute_dtype": "float16"}, "bnb_4bit_compute_dtype"),
])
def test_genuinely_unsupported_still_fails_loud(cfg, bad_field):
    """Every other bnb-specific field still raises a clear error."""
    with pytest.raises(ValueError) as exc:
        _reject()(cfg)
    assert bad_field in str(exc.value)


def test_empty_and_default_configs_pass():
    """No fields, or only defaulted/accepted fields, never raise."""
    _reject()({})
    _reject()({"load_in_4bit": True})
    _reject()({"bnb_4bit_compute_dtype": "float32", "bnb_4bit_use_double_quant": True})
