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

"""A message with more or fewer than two text fields raises an error that names the fields it found,
instead of a bare AssertionError with no message (follow-up to unsloth-zoo#2567)."""

import pytest
from datasets import Dataset

from unsloth_zoo.dataset_utils import standardize_data_formats


def test_a_string_turn_field_is_named_in_the_error():
    rows = [
        {"conversations": [
            {"from": "human", "value": "hi", "weight": "0.0"},
            {"from": "gpt", "value": "yo", "weight": "1.0"},
        ]},
    ]
    with pytest.raises(RuntimeError) as error:
        standardize_data_formats(Dataset.from_list(rows))
    message = str(error.value)
    assert "exactly 2" in message
    assert "'from', 'value', 'weight'" in message


def test_a_single_field_is_named_in_the_error():
    rows = [
        {"conversations": [{"value": "hi"}, {"value": "yo"}]},
    ]
    with pytest.raises(RuntimeError, match = r"found 1: \['value'\]"):
        standardize_data_formats(Dataset.from_list(rows))


def test_two_fields_still_standardize():
    rows = [
        {"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "yo"}]},
    ]
    out = standardize_data_formats(Dataset.from_list(rows))
    assert out[0]["conversations"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
