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


"""Old-layout bnb skip entries must match through Qwen2-VL's nested-lookahead conversion mapping; otherwise the 16-bit Linear is quantized and loading fails."""

import pytest

from unsloth_zoo.patching_utils import check_conversion_mappings

# transformers 4.56 Qwen2(.5)-VL.
QWEN_VL_MAPPING = {
    "^visual": "model.visual",
    r"^model(?!\.(language_model|visual))": "model.language_model",
}

# From unsloth/Qwen2.5-VL-72B-Instruct-unsloth-bnb-4bit.
SKIP = [
    "lm_head",
    "visual.merger.mlp",
    "model.layers.7.self_attn.o_proj",
    "model.layers.33.self_attn.o_proj",
]


class _QwenVL:
    _checkpoint_conversion_mapping = QWEN_VL_MAPPING


@pytest.mark.parametrize("name", [
    "model.language_model.layers.7.self_attn.o_proj",
    "model.language_model.layers.33.self_attn.o_proj",
])
def test_old_layout_language_skip_module_is_matched(name):
    assert check_conversion_mappings(_QwenVL, name, SKIP)


@pytest.mark.parametrize("name", [
    "model.language_model.layers.8.self_attn.o_proj",
    "model.language_model.layers.7.self_attn.q_proj",
    "model.language_model.layers.17.self_attn.o_proj",
])
def test_other_language_linears_still_convert(name):
    assert not check_conversion_mappings(_QwenVL, name, SKIP)


def test_old_layout_visual_skip_module_is_still_matched():
    assert check_conversion_mappings(_QwenVL, "model.visual.merger.mlp.0", SKIP)


def test_real_class_mapping_if_present():
    transformers = pytest.importorskip("transformers")
    cls = getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None)
    if not getattr(cls, "_checkpoint_conversion_mapping", None):
        pytest.skip(reason = "transformers 5 moved conversion mappings out of the class")
    assert check_conversion_mappings(cls, "model.language_model.layers.7.self_attn.o_proj", SKIP)
