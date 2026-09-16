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

"""A shard name out of `model.safetensors.index.json` must never leave `save_directory`.

`merge_and_overwrite_lora` builds its shard list from four places, and three of them
already reduce every name to its last component: the local listing uses `os.listdir`,
the Hub listing uses `os.path.split(x["name"])[-1]` and the local stale-shard filter
uses `os.path.split(v)[-1]`. The fourth, the local index branch, took
`index_data["weight_map"].values()` raw.

Those names are joined onto `save_directory` and onto the base directory and handed to
`shutil.copy2`, so a `weight_map` value like `../../escaped/x.safetensors` wrote a file
outside the directory the user asked to export to.

The branch needs `model_name` to be a directory with an index and no top level
`.safetensors`, which `check_local_model_exists` never returns: it requires a
`.safetensors`. It is reached anyway because `is_local_path` is recomputed from
`os.path.isdir(model_name)` after resolution, so a repo id that also names a directory
in the working directory takes the local branch with that directory's index. That is
what the fixture below sets up, with the Hub answers stubbed so the test is offline and
deterministic.

Everything the test touches stays under `tmp_path`, including the simulated escape
target, and the payload is an obviously fake non-shard file.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil

import pytest

import _merge_e2e_helpers as H
from unsloth_zoo import saving_utils


FAMILY = "llama"

PAYLOAD = "unsloth-not-a-real-shard.safetensors"

# Every one of these is a `weight_map` value that must not survive as written.
HOSTILE_NAMES = [
    pytest.param(f"../../escaped/{PAYLOAD}", id = "relative-traversal"),
    pytest.param(f"/{PAYLOAD}",              id = "absolute-path"),
    pytest.param(f"..\\..\\escaped\\{PAYLOAD}", id = "windows-traversal"),
    pytest.param("..",                       id = "parent-directory"),
    pytest.param(".",                        id = "current-directory"),
    pytest.param("",                         id = "empty-string"),
]


@pytest.fixture(autouse = True)
def _merge_on_the_cpu(monkeypatch):
    """Same reason as `test_merge_e2e_hub_unreachable`: only control flow is under
    test, so the device should not become a hardware dependency."""
    monkeypatch.setattr(saving_utils, "_active_merge_device", lambda: "cpu")


def _stub_the_hub(monkeypatch):
    """Stand in for a repo id that resolves on the Hub and is not quantized, without
    a network call. Resolution then returns the name itself, and the shadowing
    directory in the working directory is what the merge reads."""
    monkeypatch.setattr(
        saving_utils, "check_hf_model_exists", lambda *a, **k: True, raising = True,
    )
    monkeypatch.setattr(
        saving_utils, "check_model_quantization_status",
        lambda *a, **k: (False, None), raising = True,
    )


def _shadow_directory(tmp_path, weight_map_value):
    """`ns/base`: a repo id shape that is also a real directory holding a hostile index
    and no shards. Returns its path relative to `tmp_path`."""
    real_base = os.path.join(str(tmp_path), "real_base")
    shadow = os.path.join(str(tmp_path), "ns", "base")
    os.makedirs(shadow, exist_ok = True)
    shutil.copy2(
        os.path.join(real_base, "config.json"), os.path.join(shadow, "config.json"),
    )
    index = {
        "metadata"   : {"total_size" : 8},
        "weight_map" : {"model.embed_tokens.weight" : weight_map_value},
    }
    with open(
        os.path.join(shadow, "model.safetensors.index.json"), "w", encoding = "utf-8",
    ) as f:
        json.dump(index, f)
    return os.path.join("ns", "base")


def _plant_payload(tmp_path, base_rel, weight_map_value):
    """Make the hostile name point at a file that exists, so the copy is not skipped
    for the uninteresting reason that its source is missing. Only paths that stay
    under `tmp_path` are planted, which is every case except the absolute one."""
    source = os.path.normpath(os.path.join(str(tmp_path), base_rel, weight_map_value))
    if not source.startswith(str(tmp_path) + os.sep) or os.path.isdir(source):
        return None
    os.makedirs(os.path.dirname(source), exist_ok = True)
    with open(source, "wb") as f:
        f.write(b"not a real shard")
    return source


def _record_copies(monkeypatch):
    """Every `shutil.copy2` destination the merge asks for."""
    destinations = []
    real_copy2 = shutil.copy2

    def recording_copy2(src, dst, *args, **kwargs):
        destinations.append(dst)
        return real_copy2(src, dst, *args, **kwargs)
    monkeypatch.setattr(shutil, "copy2", recording_copy2, raising = True)
    return destinations


def _record_shard_names(monkeypatch):
    """The resolved shard list, caught at the first module level function it is handed
    to after the branch under test."""
    names = []
    real = saving_utils.is_hf_sharded_safetensors

    def recording(safetensors_list, *args, **kwargs):
        names.extend(safetensors_list)
        return real(safetensors_list, *args, **kwargs)
    monkeypatch.setattr(
        saving_utils, "is_hf_sharded_safetensors", recording, raising = True,
    )
    return names


@pytest.mark.parametrize("weight_map_value", HOSTILE_NAMES)
def test_a_hostile_weight_map_entry_cannot_escape_the_save_directory(
    monkeypatch, tmp_path, weight_map_value,
):
    if not H.family_available(FAMILY):
        pytest.skip(f"{FAMILY} unavailable in this transformers")
    H.set_offline_cpu_env()

    spec = H.make_spec(FAMILY)
    model = H.build_and_save_base(spec, os.path.join(str(tmp_path), "real_base"))
    peft_model = H.attach_lora(model, spec, "full")

    base_rel = _shadow_directory(tmp_path, weight_map_value)
    # Deeper than the base directory, so a traversal out of one is not the same file
    # as a traversal out of the other.
    save_directory = os.path.join("out", "deep", "merged")
    # An existing directory next to the output, standing in for any directory a
    # traversal could land in on a real machine.
    escaped = os.path.join(str(tmp_path), "out", "escaped")
    os.makedirs(escaped, exist_ok = True)
    _plant_payload(tmp_path, base_rel, weight_map_value)

    monkeypatch.chdir(tmp_path)
    _stub_the_hub(monkeypatch)
    destinations = _record_copies(monkeypatch)
    shard_names = _record_shard_names(monkeypatch)

    try:
        saving_utils.merge_and_overwrite_lora(
            get_model_name  = lambda *a, **k: base_rel,
            model           = peft_model,
            tokenizer       = None,
            save_directory  = save_directory,
            save_method     = "merged_16bit",
            push_to_hub     = False,
        )
    except Exception:
        # A directory whose index names no readable shard cannot merge. Failing is
        # fine, escaping is not, and the assertions below are about the escape.
        pass

    inside = os.path.realpath(os.path.join(str(tmp_path), save_directory))
    for destination in destinations:
        resolved = os.path.realpath(destination)
        assert resolved == inside or resolved.startswith(inside + os.sep), (
            f"{weight_map_value!r} made the merge write to {destination!r}, "
            f"outside {save_directory!r}"
        )

    # The invariant the shard names themselves have to satisfy, stated as what they
    # resolve to rather than as a character set, so it reads the same on both
    # separators: `..\\x` is one ordinary file name on POSIX and a traversal on
    # Windows, and this catches it exactly where it is one.
    for name in shard_names:
        joined = os.path.normpath(os.path.join(inside, name))
        assert joined.startswith(inside + os.sep), (
            f"{name!r} survived as a shard name and resolves to {joined!r}, "
            f"outside {inside!r}"
        )

    assert os.listdir(escaped) == [], (
        f"{weight_map_value!r} planted {os.listdir(escaped)} in a directory outside "
        f"the requested output directory"
    )


def _shadow_directory_for(tmp_path, weight_map_value):
    """`_shadow_directory` with the index value supplied verbatim."""
    return _shadow_directory(tmp_path, weight_map_value)


def _existing_external_shard(tmp_path):
    """A real safetensors file outside the output directory, for the merge to find.

    The parametrised cases deliberately leave the absolute target missing, which
    makes the vulnerable code fail on the size check before the shard list is even
    used. That is a pass for the wrong reason, and it hides the second sink: the
    merge opens a shard it believes is its own with `open(..., "r+b")` and an
    `mmap` write, so an absolute entry naming a file that DOES exist is modified in
    place and never goes near `shutil.copy2` or its `not os.path.exists` gate.
    """
    real_base = os.path.join(str(tmp_path), "real_base")
    shards = [f for f in os.listdir(real_base) if f.endswith(".safetensors")]
    assert shards, "the base model produced no safetensors to copy"
    outside = os.path.join(str(tmp_path), "outside")
    os.makedirs(outside, exist_ok = True)
    victim = os.path.join(outside, "victim.safetensors")
    shutil.copy2(os.path.join(real_base, shards[0]), victim)
    return victim


def test_an_absolute_entry_cannot_overwrite_an_existing_file_outside_the_output(
    monkeypatch, tmp_path,
):
    """The overwrite variant, which the missing-target cases cannot reach."""
    if not H.family_available(FAMILY):
        pytest.skip(f"{FAMILY} unavailable in this transformers")
    H.set_offline_cpu_env()

    spec = H.make_spec(FAMILY)
    model = H.build_and_save_base(spec, os.path.join(str(tmp_path), "real_base"))
    peft_model = H.attach_lora(model, spec, "full")

    victim = _existing_external_shard(tmp_path)
    before = hashlib.sha256(pathlib.Path(victim).read_bytes()).hexdigest()

    base_rel = _shadow_directory(tmp_path, victim)
    save_directory = os.path.join("out", "deep", "merged")

    monkeypatch.chdir(tmp_path)
    _stub_the_hub(monkeypatch)
    shard_names = _record_shard_names(monkeypatch)

    try:
        saving_utils.merge_and_overwrite_lora(
            get_model_name  = lambda *a, **k: base_rel,
            model           = peft_model,
            tokenizer       = None,
            save_directory  = save_directory,
            save_method     = "merged_16bit",
            push_to_hub     = False,
        )
    except Exception:
        # Same as the parametrised cases: failing to merge is fine, touching a file
        # outside the output directory is not.
        pass

    after = hashlib.sha256(pathlib.Path(victim).read_bytes()).hexdigest()
    assert after == before, (
        f"an absolute weight_map entry modified {victim!r}, a file outside "
        f"{save_directory!r}"
    )

    inside = os.path.realpath(os.path.join(str(tmp_path), save_directory))
    for name in shard_names:
        joined = os.path.normpath(os.path.join(inside, name))
        assert joined.startswith(inside + os.sep), (
            f"{name!r} survived as a shard name and resolves to {joined!r}, "
            f"outside {inside!r}"
        )
