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

"""ShareGPT turns that carry a numeric or null field next to the role and the text, such as the per-turn "weight" in
Open-Orca/SlimOrca and teknium/OpenHermes-2.5, are still standardized; a non-text content column still raises."""

import pytest
from datasets import Dataset

from unsloth_zoo.dataset_utils import standardize_data_formats


SLIMORCA_ROWS = [
    {"conversations": [
        {"from": "system", "value": "You are an AI assistant.", "weight": None},
        {"from": "human", "value": "Write an article.", "weight": 0.0},
        {"from": "gpt", "value": "Title: An article.", "weight": 1.0},
    ]},
    {"conversations": [
        {"from": "human", "value": "Every day, a tree drops 7 leaves.", "weight": None},
        {"from": "gpt", "value": "Here's the logic behind this.", "weight": None},
    ]},
]


def test_a_numeric_turn_weight_is_ignored():
    out = standardize_data_formats(Dataset.from_list(SLIMORCA_ROWS))
    assert out[0]["conversations"] == [
        {"role": "system", "content": "You are an AI assistant."},
        {"role": "user", "content": "Write an article."},
        {"role": "assistant", "content": "Title: An article."},
    ]
    assert out[1]["conversations"] == [
        {"role": "user", "content": "Every day, a tree drops 7 leaves."},
        {"role": "assistant", "content": "Here's the logic behind this."},
    ]


def test_an_all_null_extra_field_is_ignored():
    rows = [
        {"conversations": [
            {"from": "human", "value": "hi", "weight": None},
            {"from": "gpt", "value": "yo", "weight": None},
        ]},
    ]
    out = standardize_data_formats(Dataset.from_list(rows))
    assert out[0]["conversations"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]


def test_non_text_content_still_raises():
    rows = [
        {"conversations": [
            {"from": "human", "value": [{"type": "text", "text": "hi"}]},
            {"from": "gpt", "value": [{"type": "text", "text": "yo"}]},
        ]},
    ]
    with pytest.raises(RuntimeError, match = "Cannot standardize non text datasets"):
        standardize_data_formats(Dataset.from_list(rows))
