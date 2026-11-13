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

"""renumber_safetensor_files must preserve every shard's bytes when a surviving input filename collides with a clean
name claimed by an earlier list position (a local dir can carry a stale or foreign 'model-i-of-N' name)."""

import os

from unsloth_zoo.saving_utils import renumber_safetensor_files


def _write(folder, name, data):
    with open(os.path.join(folder, name), "wb") as f:
        f.write(data)


def _read_all(folder):
    on_disk = {}
    for name in os.listdir(folder):
        with open(os.path.join(folder, name), "rb") as f:
            on_disk[name] = f.read()
    return on_disk


def test_renumber_safetensor_files_does_not_lose_a_shard_on_name_collision(tmp_path):
    """file_list mixes temp_split_* names with a surviving shard whose own name equals clean_names[1]; the
    single-pass rename overwrote it before its turn, so a returned name pointed nowhere and the bytes were gone."""
    save_directory = str(tmp_path)
    victim, t0, t1 = (
        "model-00002-of-00003.safetensors",
        "temp_split_aaaa_000.safetensors",
        "temp_split_aaaa_001.safetensors",
    )
    _write(save_directory, victim, b"VICTIM-SHARD-WEIGHTS")
    _write(save_directory, t0, b"SPLIT-PART-0")
    _write(save_directory, t1, b"SPLIT-PART-1")

    returned = renumber_safetensor_files([t0, t1, victim], save_directory)

    on_disk = _read_all(save_directory)
    assert all(name in on_disk for name in returned), (
        f"returned names missing on disk: {[n for n in returned if n not in on_disk]}"
    )
    assert sorted(on_disk.values()) == sorted(
        [b"VICTIM-SHARD-WEIGHTS", b"SPLIT-PART-0", b"SPLIT-PART-1"]
    ), "a shard's weights were silently overwritten"


def test_renumber_safetensor_files_survives_a_staging_named_input(tmp_path):
    """A crashed earlier run leaves `renaming_<i>_<clean>` files behind, and the local-dir path re-lists every
    *.safetensors as an input. Staging names drawn from the input namespace clobbered that leftover mid-renumber
    (then crashed in pass 2); scoped temp names cannot collide with any name fixed before the call."""
    save_directory = str(tmp_path)
    leftover, other = "renaming_0_model-00001-of-00002.safetensors", "chunk.safetensors"
    _write(save_directory, other, b"REAL-SHARD")
    _write(save_directory, leftover, b"STALE-LOOKING-SHARD")

    returned = renumber_safetensor_files([other, leftover], save_directory)

    on_disk = _read_all(save_directory)
    assert all(name in on_disk for name in returned), (
        f"returned names missing on disk: {[n for n in returned if n not in on_disk]}"
    )
    assert sorted(on_disk.values()) == sorted([b"REAL-SHARD", b"STALE-LOOKING-SHARD"]), (
        "a shard's weights were silently overwritten"
    )


def test_renumber_safetensor_files_round_trips_an_ordinary_list(tmp_path):
    """No collision: the temp names must not survive and every file still lands on its clean name."""
    save_directory = str(tmp_path)
    names = ["aaa.safetensors", "zzz.safetensors"]
    for i, name in enumerate(names):
        _write(save_directory, name, f"payload-{i}".encode())

    returned = renumber_safetensor_files(names, save_directory)

    on_disk = _read_all(save_directory)
    assert sorted(on_disk) == sorted(returned)
    assert sorted(on_disk.values()) == [b"payload-0", b"payload-1"]
