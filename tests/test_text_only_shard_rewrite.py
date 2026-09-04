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

"""What happens to the files when a text_only export drops a whole shard (#969).

`unsloth/gemma-3-4b-it` keeps text weights in both of its two shards, so the end-to-end run
never deletes one, never renumbers, and never has to decide whether the index should still
exist. Those are the branches most likely to leave a directory that no longer loads, and a
checkpoint whose index names a file that is not there fails at load with no useful message.

So drive them directly, on real safetensors files small enough to be free. The shard layout
here is the one the drop is for: some shards mixed, at least one pure vision.
"""

from __future__ import annotations

import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from unsloth_zoo.saving_utils import (
    _rewrite_shards_text_only,
    _write_text_only_index,
    renumber_safetensor_files,
)

TEXT = ["model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight",
        "model.norm.weight"]
KEY_MAP = {k : "language_model." + k for k in TEXT}


def _write_shards(directory, layout):
    """Write one tiny tensor per key and return the filenames, in order."""
    os.makedirs(directory, exist_ok = True)
    names = []
    for i, keys in enumerate(layout):
        name = f"model-{i+1:05d}-of-{len(layout):05d}.safetensors"
        save_file({k : torch.ones(2) for k in keys},
                  os.path.join(directory, name), metadata = {"format" : "pt"})
        names.append(name)
    return names


def _vision(n):
    return [f"vision_tower.vision_model.encoder.layers.{i}.q_proj.weight" for i in range(n)]


def _keys_in(path):
    with safe_open(path, framework = "pt", device = "cpu") as f:
        return set(f.keys())


def test_a_shard_holding_only_vision_weights_is_deleted(tmp_path):
    d = str(tmp_path)
    names = _write_shards(d, [
        [KEY_MAP[TEXT[0]]] + _vision(1),
        _vision(3),
        [KEY_MAP[TEXT[1]], KEY_MAP[TEXT[2]]],
    ])

    kept = _rewrite_shards_text_only(d, names, KEY_MAP)

    assert kept == [names[0], names[2]], f"kept {kept}"
    assert not os.path.exists(os.path.join(d, names[1])), "the all-vision shard is still on disk"
    assert _keys_in(os.path.join(d, names[0])) == {TEXT[0]}, "the kept tensors were not renamed"
    assert _keys_in(os.path.join(d, names[2])) == {TEXT[1], TEXT[2]}


def test_the_index_follows_the_shards_through_the_renumber(tmp_path):
    """The renumber closes the gap the deletion left, and the index has to move with it."""
    d = str(tmp_path)
    names = _write_shards(d, [
        [KEY_MAP[TEXT[0]]],
        _vision(2),
        [KEY_MAP[TEXT[1]], KEY_MAP[TEXT[2]]],
    ])
    # A pre-drop index, exactly as Step 6 or the copied original would have left it.
    with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata" : {}, "weight_map" : {
            **{KEY_MAP[TEXT[0]] : names[0]},
            **{k : names[1] for k in _vision(2)},
            **{KEY_MAP[TEXT[1]] : names[2], KEY_MAP[TEXT[2]] : names[2]},
        }}, f)

    kept = _rewrite_shards_text_only(d, names, KEY_MAP)
    final = renumber_safetensor_files(kept, d)
    _write_text_only_index(d, final)

    assert final == ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    on_disk = sorted(f for f in os.listdir(d) if f.endswith(".safetensors"))
    assert on_disk == final, f"directory holds {on_disk}, index was written for {final}"

    with open(os.path.join(d, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    assert set(weight_map) == set(TEXT), f"index lists {sorted(weight_map)}"
    for key, filename in weight_map.items():
        assert key in _keys_in(os.path.join(d, filename)), f"{key} is not in {filename}"


def test_a_single_surviving_shard_loses_the_index_entirely(tmp_path):
    """One file needs no index, and a leftover one would name shards that no longer exist."""
    d = str(tmp_path)
    names = _write_shards(d, [_vision(2), [KEY_MAP[k] for k in TEXT]])
    index_path = os.path.join(d, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump({"metadata" : {}, "weight_map" : {k : names[0] for k in _vision(2)}}, f)

    final = renumber_safetensor_files(_rewrite_shards_text_only(d, names, KEY_MAP), d)
    _write_text_only_index(d, final)

    assert final == ["model.safetensors"], f"final list is {final}"
    assert sorted(f for f in os.listdir(d) if f.endswith(".safetensors")) == ["model.safetensors"]
    assert not os.path.exists(index_path), "a one-shard checkpoint kept its index"
    assert _keys_in(os.path.join(d, "model.safetensors")) == set(TEXT)
