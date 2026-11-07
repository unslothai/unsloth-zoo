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

"""standardize_data_formats stays in process for an in-memory dataset unless a worker count was asked for (an int,
the env override) or the dataset is backed by cache files; the output is the same either way."""

import pytest
from datasets import Dataset, load_from_disk

import unsloth_zoo.dataset_num_proc as num_proc_policy
from unsloth_zoo.dataset_num_proc import NUM_PROC_ENV_VAR
from unsloth_zoo.dataset_utils import standardize_data_formats

ROWS = [
    {"conversations": [{"from": "human", "value": f"q{i}"}, {"from": "gpt", "value": f"a{i}"}]}
    for i in range(40)
]
POLICY_COUNT = 3


@pytest.fixture
def seen(monkeypatch):
    """Records the num_proc each map is called with, then runs it in process so no pool is built here."""
    calls = []
    original = Dataset.map

    def spy(self, *args, **kwargs):
        if kwargs.get("desc") == "Unsloth: Standardizing formats":
            calls.append(kwargs.get("num_proc"))
            kwargs["num_proc"] = None
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Dataset, "map", spy)
    monkeypatch.setattr(num_proc_policy, "get_dataset_num_proc", lambda *a, **k: POLICY_COUNT)
    monkeypatch.delenv(NUM_PROC_ENV_VAR, raising = False)
    return calls


def _expected():
    return [
        {"conversations": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]}
        for i in range(40)
    ]


def test_in_memory_default_runs_in_process(seen):
    out = standardize_data_formats(Dataset.from_list(ROWS))
    assert seen == [None]
    assert out.to_list() == _expected()


def test_explicit_count_goes_through_the_policy(seen):
    out = standardize_data_formats(Dataset.from_list(ROWS), num_proc = 4)
    assert seen == [POLICY_COUNT]
    assert out.to_list() == _expected()


def test_env_override_goes_through_the_policy(seen, monkeypatch):
    monkeypatch.setenv(NUM_PROC_ENV_VAR, "2")
    out = standardize_data_formats(Dataset.from_list(ROWS))
    assert seen == [POLICY_COUNT]
    assert out.to_list() == _expected()


def test_disk_backed_dataset_goes_through_the_policy(seen, tmp_path):
    Dataset.from_list(ROWS).save_to_disk(str(tmp_path / "ds"))
    ds = load_from_disk(str(tmp_path / "ds"))
    assert ds.cache_files
    out = standardize_data_formats(ds)
    assert seen == [POLICY_COUNT]
    assert out.to_list() == _expected()
