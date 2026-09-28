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


"""Kimi K2.5 / K2.7 name their bf16 vision projector `mm_projector`; 4-bit loads must skip it."""

import pytest

from unsloth_zoo.peft_utils import SKIP_QUANTIZATION_MODULES

# Remote-code and native transformers 5 layouts.
PROJECTOR = ["mm_projector.proj.0", "mm_projector.proj.2",
             "model.mm_projector.in_proj", "model.mm_projector.out_proj"]
OTHERS_CONVERT = ["language_model.model.layers.1.mlp.shared_experts.gate_proj",
                  "language_model.model.layers.0.self_attn.q_a_proj"]


def _should_convert_module():
    module = pytest.importorskip("transformers.quantizers.quantizers_utils")
    function = getattr(module, "should_convert_module", None)
    if function is None:
        pytest.skip(reason = "transformers < 5.0 has no should_convert_module")
    return function


def _legacy_skip(full_name, keys):
    # transformers 4.x replace_with_bnb_linear matching.
    return any((key + "." in full_name) or (key == full_name) for key in keys)


@pytest.mark.parametrize("name", PROJECTOR)
def test_mm_projector_is_kept_in_full_precision_legacy_match(name):
    assert _legacy_skip(name, SKIP_QUANTIZATION_MODULES), name


@pytest.mark.parametrize("name", PROJECTOR)
def test_mm_projector_is_kept_in_full_precision(name):
    assert not _should_convert_module()(name, SKIP_QUANTIZATION_MODULES), name


@pytest.mark.parametrize("name", OTHERS_CONVERT)
def test_decoder_linears_still_convert_legacy_match(name):
    assert not _legacy_skip(name, SKIP_QUANTIZATION_MODULES), name


@pytest.mark.parametrize("name", OTHERS_CONVERT)
def test_decoder_linears_still_convert(name):
    assert _should_convert_module()(name, SKIP_QUANTIZATION_MODULES), name
