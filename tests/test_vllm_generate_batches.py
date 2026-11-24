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


import pytest

import unsloth_zoo.vllm_utils as vllm_utils
from unsloth_zoo.vllm_utils import create_batches, generate_batches


class _Engine:
    def __init__(self, approx_max_num_seqs):
        self.approx_max_num_seqs = approx_max_num_seqs
        self.calls = []

    def generate(self, batch, *args, **kwargs):
        self.calls.append(list(batch))
        return list(batch)


@pytest.fixture
def t4(monkeypatch):
    # A T4 reports about 14.58 GiB, inside the <= 16 GiB band (width = approx_max_num_seqs // 5).
    # setenv first so the variable is removed again after the test, even if the code under test sets it.
    monkeypatch.setenv("UNSLOTH_VLLM_BATCHES", "0")
    monkeypatch.delenv("UNSLOTH_VLLM_BATCHES")
    monkeypatch.setattr(vllm_utils, "get_mem_info", lambda: (0, int(14.58 * 2**30)))


def test_small_engine_width_is_at_least_one(t4):
    llm = _Engine(4)
    assert generate_batches(llm, list(range(10))) == list(range(10))
    assert all(len(batch) == 1 for batch in llm.calls)


def test_auto_width_is_per_engine_not_process_wide(t4):
    import os
    small, big = _Engine(50), _Engine(200)
    generate_batches(small, list(range(25)))
    assert "UNSLOTH_VLLM_BATCHES" not in os.environ
    assert generate_batches(big, list(range(100))) == list(range(100))
    generate_batches(small, list(range(25)))
    assert [len(batch) for batch in big.calls] == [40, 40, 20]
    assert [len(batch) for batch in small.calls] == [10, 10, 5] * 2


def test_explicit_env_override_still_wins(t4, monkeypatch):
    monkeypatch.setenv("UNSLOTH_VLLM_BATCHES", "3")
    llm = _Engine(200)
    generate_batches(llm, list(range(7)))
    assert [len(batch) for batch in llm.calls] == [3, 3, 1]


def test_empty_input(t4):
    assert create_batches([], 8) == []
    assert generate_batches(_Engine(200), []) == []


def test_create_batches_splits_like_before():
    assert create_batches(list(range(10)), 4) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert create_batches(list(range(8)), 4) == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert create_batches(list(range(3)), 64) == [[0, 1, 2]]
