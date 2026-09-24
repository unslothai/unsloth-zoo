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

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import unsloth_zoo.saving_utils as saving_utils
from unsloth_zoo.saving_utils import (
    TextOnlyRemapError,
    _stage_shards_text_only,
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


def _file_hashes(directory):
    import hashlib
    hashes = {}
    for name in os.listdir(directory):
        with open(os.path.join(directory, name), "rb") as f:
            hashes[name] = hashlib.sha256(f.read()).hexdigest()
    return hashes


def _staging_files(directory):
    return [f for f in os.listdir(directory) if f.startswith(".unsloth-shard-")]


def _commit(directory, staged):
    """Mirror the commit half of the post-merge block: replace, delete, return kept names."""
    kept = []
    for filename, staging_path in staged.items():
        file_path = os.path.join(directory, filename)
        if staging_path is None:
            if os.path.exists(file_path): os.remove(file_path)
            continue
        os.replace(staging_path, file_path)
        kept.append(filename)
    return kept


def test_a_shard_holding_only_vision_weights_is_deleted(tmp_path):
    d = str(tmp_path)
    names = _write_shards(d, [
        [KEY_MAP[TEXT[0]]] + _vision(1),
        _vision(3),
        [KEY_MAP[TEXT[1]], KEY_MAP[TEXT[2]]],
    ])

    kept = _commit(d, _stage_shards_text_only(d, names, KEY_MAP))

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

    kept = _commit(d, _stage_shards_text_only(d, names, KEY_MAP))
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

    kept = _commit(d, _stage_shards_text_only(d, names, KEY_MAP))
    final = renumber_safetensor_files(kept, d)
    _write_text_only_index(d, final)

    assert final == ["model.safetensors"], f"final list is {final}"
    assert sorted(f for f in os.listdir(d) if f.endswith(".safetensors")) == ["model.safetensors"]
    assert not os.path.exists(index_path), "a one-shard checkpoint kept its index"
    assert _keys_in(os.path.join(d, "model.safetensors")) == set(TEXT)


def test_a_write_failure_mid_staging_leaves_every_shard_untouched(tmp_path, monkeypatch):
    """Regression for the #1097 review: a failure must not have rewritten anything yet.

    Drives `_stage_shards_text_only` directly rather than the full merge function, which
    needs a real model and hub plumbing this unit test has no reason to construct; the
    helper is exactly the stage/verify unit the review asked to make crash-safe, and the
    commit half (`_commit` above) is proven inert on a raised exception by never running.
    """
    d = str(tmp_path)
    names = _write_shards(d, [
        [KEY_MAP[TEXT[0]]] + _vision(1),
        [KEY_MAP[TEXT[1]], KEY_MAP[TEXT[2]]],
    ])
    before = _file_hashes(d)
    calls = {"n" : 0}
    real_save_file = saving_utils.save_file

    def _flaky_save_file(tensors, path, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError(28, "No space left on device")
        return real_save_file(tensors, path, *args, **kwargs)

    monkeypatch.setattr(saving_utils, "save_file", _flaky_save_file)

    with pytest.raises(OSError):
        _stage_shards_text_only(d, names, KEY_MAP)

    assert _file_hashes(d) == before, "an original shard was modified before the raise"
    assert _staging_files(d) == [], "a staging file survived the failure"


def test_a_verification_mismatch_leaves_every_shard_untouched(tmp_path):
    """A plan the shards cannot satisfy must not commit a partial rewrite either."""
    d = str(tmp_path)
    names = _write_shards(d, [[KEY_MAP[TEXT[0]]], [KEY_MAP[TEXT[1]], KEY_MAP[TEXT[2]]]])
    before = _file_hashes(d)
    # A key whose base tensor is on none of the shards: staging can never satisfy this plan.
    bad_plan = dict(KEY_MAP, **{"model.layers.1.self_attn.q_proj.weight" : "language_model.model.layers.1.self_attn.q_proj.weight"})

    with pytest.raises(TextOnlyRemapError):
        _stage_shards_text_only(d, names, bad_plan)

    assert _file_hashes(d) == before, "an original shard was modified on a plan mismatch"
    assert _staging_files(d) == [], "a staging file survived the mismatch"


@pytest.mark.skipif(os.name == "nt", reason = "POSIX permission bits are not meaningful on Windows")
def test_a_committed_shard_keeps_the_original_mode(tmp_path):
    """mkstemp stages at 0o600; the commit must not narrow a kept shard to owner-only."""
    d = str(tmp_path)
    names = _write_shards(d, [[KEY_MAP[TEXT[0]]], [KEY_MAP[TEXT[1]], KEY_MAP[TEXT[2]]]])
    for name in names: os.chmod(os.path.join(d, name), 0o644)

    kept = _commit(d, _stage_shards_text_only(d, names, KEY_MAP))

    for name in kept:
        mode = os.stat(os.path.join(d, name)).st_mode & 0o777
        assert mode == 0o644, f"{name} ended up {oct(mode)} instead of the shard's original 0o644"
