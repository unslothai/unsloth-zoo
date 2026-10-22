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

"""bnb_4bit_use_double_quant is accepted (bool only) while every other
BitsAndBytesConfig field still fails loud. Pure logic, runs under the torch shim.
"""

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


def _reject(cfg):
    from unsloth_zoo.mlx.loader import _reject_unsupported_hf_quantization_fields
    return _reject_unsupported_hf_quantization_fields(cfg)


@pytest.mark.parametrize("cfg", [
    {}, {"load_in_4bit": True},
    {"bnb_4bit_use_double_quant": True},
    {"bnb_4bit_use_double_quant": False},
    {"bnb_4bit_use_double_quant": None},
])
def test_double_quant_accepted(cfg):
    _reject(cfg)  # must not raise


@pytest.mark.parametrize("bad", ["true", 0, 1, 1.0, []])
def test_non_bool_double_quant_rejected(bad):
    """0/1 are ints, not bools, so numeric spellings raise too."""
    with pytest.raises(ValueError, match="bnb_4bit_use_double_quant"):
        _reject({"bnb_4bit_use_double_quant": bad})


@pytest.mark.parametrize("field,value", [
    ("bnb_4bit_quant_type", "nf4"),
    ("bnb_4bit_compute_dtype", "float16"),
    ("bnb_4bit_quant_storage", "float16"),
    ("llm_int8_threshold", 3.0),
    ("llm_int8_enable_fp32_cpu_offload", True),
])
def test_other_fields_still_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        _reject({field: value})


def test_double_quant_not_blamed_for_another_field():
    """nf4 is rejected; the now-supported double_quant is not named."""
    with pytest.raises(ValueError) as exc:
        _reject({"bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True})
    assert "bnb_4bit_use_double_quant" not in str(exc.value)
