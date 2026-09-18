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
import torch

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


def _nested_layout(tmp_path, subdirectories):
    """A local model directory whose only shards live in subdirectories.

    The index names them the way they are laid out, e.g. `weights/model-....safetensors`.
    Such a name is contained: it resolves under the directory it is joined onto, so it
    is not the traversal this file guards against, and it has to keep working. The
    merge reaches this shape through the same shadowing route as the hostile cases,
    because there is no top level `.safetensors` for `os.listdir` to find.

    `subdirectories` is one name per shard, so passing the same shard count with two
    different directories puts two shards under two parents.
    """
    from safetensors import safe_open

    real_base = os.path.join(str(tmp_path), "real_base")
    shards = sorted(f for f in os.listdir(real_base) if f.endswith(".safetensors"))
    assert shards, "the base model produced no safetensors"

    nested_root = os.path.join(str(tmp_path), "ns", "base")
    os.makedirs(nested_root, exist_ok = True)
    shutil.copy2(
        os.path.join(real_base, "config.json"), os.path.join(nested_root, "config.json"),
    )

    weight_map = {}
    total_size = 0
    for index, shard in enumerate(shards):
        directory = subdirectories[index % len(subdirectories)]
        # Posix separators, which is what a real index carries on every platform.
        relative = f"{directory}/{shard}"
        destination = os.path.join(nested_root, directory, shard)
        os.makedirs(os.path.dirname(destination), exist_ok = True)
        shutil.copy2(os.path.join(real_base, shard), destination)
        total_size += os.path.getsize(destination)
        with safe_open(destination, framework = "pt") as f:
            for key in f.keys():
                weight_map[key] = relative

    with open(
        os.path.join(nested_root, "model.safetensors.index.json"), "w", encoding = "utf-8",
    ) as f:
        json.dump({"metadata" : {"total_size" : total_size}, "weight_map" : weight_map}, f)

    return nested_root, os.path.join("ns", "base"), weight_map


def _flattened(directory, into):
    """Every `.safetensors` under `directory`, copied flat, for the helpers that read
    a merged output with `os.listdir`."""
    os.makedirs(into, exist_ok = True)
    for root, _directories, files in os.walk(directory):
        for name in files:
            if name.endswith(".safetensors"):
                shutil.copy2(os.path.join(root, name), os.path.join(into, name))
    return into


@pytest.mark.parametrize("in_place", [False, True], ids = ["out-of-place", "in-place"])
def test_a_contained_nested_shard_name_still_merges(monkeypatch, tmp_path, in_place):
    """A contained nested name is kept, not rewritten, and the merge completes.

    Collapsing such a name to its last component pointed size discovery at
    `model_name/model-....safetensors`, which does not exist, so an in-place merge of
    this layout died on the bare `assert(max_size_in_bytes != 0 and ...)` instead of
    merging. Out of place the shard has to be staged under a parent that nothing had
    created yet. Both are asserted here against the real merge, with no swallowed
    exception: a recorder that runs before the copy proves nothing about the export.
    """
    if not H.family_available(FAMILY):
        pytest.skip(f"{FAMILY} unavailable in this transformers")
    H.set_offline_cpu_env()

    spec = H.make_spec(FAMILY)
    real_base = os.path.join(str(tmp_path), "real_base")
    model = H.build_and_save_base(spec, real_base)
    base_tensors = H.read_safetensors_dir(real_base)
    peft_model = H.attach_lora(model, spec, "full")
    adapted = H.extract_adapted(peft_model)

    nested_root, base_rel, weight_map = _nested_layout(tmp_path, ["weights"])
    save_directory = base_rel if in_place else os.path.join("out", "deep", "merged")

    monkeypatch.chdir(tmp_path)
    _stub_the_hub(monkeypatch)
    shard_names = _record_shard_names(monkeypatch)

    saving_utils.merge_and_overwrite_lora(
        get_model_name  = lambda *a, **k: base_rel,
        model           = peft_model,
        tokenizer       = None,
        save_directory  = save_directory,
        save_method     = "merged_16bit",
        output_dtype    = torch.float32,
        push_to_hub     = False,
    )

    # The name reached the merge as written, rather than as its last component.
    assert set(shard_names) == set(weight_map.values()), (
        f"the nested names did not survive: {shard_names!r}"
    )

    output = os.path.join(str(tmp_path), save_directory)
    for relative in set(weight_map.values()):
        assert os.path.exists(os.path.join(output, relative)), (
            f"{relative!r} was never written under {output!r}"
        )

    # And the tensors it wrote are the merge, not a copy of the base.
    H.assert_merge_correct(
        family       = FAMILY,
        base_tensors = base_tensors,
        out_dir      = _flattened(output, os.path.join(str(tmp_path), "flat")),
        save_dtype   = torch.float32,
        adapted      = adapted,
        base_dir     = real_base,
    )


def test_two_nested_shards_sharing_a_basename_are_not_merged_into_one(
    monkeypatch, tmp_path,
):
    """Collapsing to the last component also loses the difference between two shards.

    `a/x.safetensors` and `b/x.safetensors` are two files; their basename is one name,
    so a shard list built from basenames holds a single entry and half the tensors are
    never merged.
    """
    if not H.family_available(FAMILY):
        pytest.skip(f"{FAMILY} unavailable in this transformers")
    H.set_offline_cpu_env()

    spec = H.make_spec(FAMILY)
    real_base = os.path.join(str(tmp_path), "real_base")
    # Small enough shards that the base is written as more than one file.
    model = H.build_and_save_base(spec, real_base, max_shard_size = "32KB")
    shards = [f for f in os.listdir(real_base) if f.endswith(".safetensors")]
    if len(shards) < 2:
        pytest.skip("the tiny base model did not shard into more than one file")
    peft_model = H.attach_lora(model, spec, "full")

    # One shard per parent directory, every parent holding the same file name.
    nested_root = os.path.join(str(tmp_path), "ns", "base")
    os.makedirs(nested_root, exist_ok = True)
    shutil.copy2(
        os.path.join(real_base, "config.json"), os.path.join(nested_root, "config.json"),
    )
    from safetensors import safe_open
    weight_map = {}
    for index, shard in enumerate(sorted(shards)):
        relative = f"part{index}/x.safetensors"
        destination = os.path.join(nested_root, f"part{index}", "x.safetensors")
        os.makedirs(os.path.dirname(destination), exist_ok = True)
        shutil.copy2(os.path.join(real_base, shard), destination)
        with safe_open(destination, framework = "pt") as f:
            for key in f.keys():
                weight_map[key] = relative
    with open(
        os.path.join(nested_root, "model.safetensors.index.json"), "w", encoding = "utf-8",
    ) as f:
        json.dump({"metadata" : {"total_size" : 0}, "weight_map" : weight_map}, f)

    base_rel = os.path.join("ns", "base")
    monkeypatch.chdir(tmp_path)
    _stub_the_hub(monkeypatch)
    shard_names = _record_shard_names(monkeypatch)

    saving_utils.merge_and_overwrite_lora(
        get_model_name  = lambda *a, **k: base_rel,
        model           = peft_model,
        tokenizer       = None,
        save_directory  = base_rel,
        save_method     = "merged_16bit",
        push_to_hub     = False,
    )

    assert len(set(shard_names)) == len(set(weight_map.values())), (
        f"{len(set(weight_map.values()))} distinct shards became {set(shard_names)!r}"
    )


@pytest.mark.parametrize("path", ["dequant", "splitting"])
def test_an_unsafe_index_is_refused_on_the_dequant_and_splitting_paths(
    monkeypatch, tmp_path, path,
):
    """The guard must not be conditional on the copy block running.

    An MXFP4/FP8 dequant or a splitting save skips that block entirely, so an
    in-place export left the hostile index sitting in the output directory, and
    `regenerate_index` does not fire once a single non-HF-named shard remains.
    """
    if not H.family_available(FAMILY):
        pytest.skip(f"{FAMILY} unavailable in this transformers")
    H.set_offline_cpu_env()

    spec = H.make_spec(FAMILY)
    model = H.build_and_save_base(spec, os.path.join(str(tmp_path), "real_base"))
    peft_model = H.attach_lora(model, spec, "full")

    # A mixed index: one real shard so the merge gets as far as the index handling,
    # plus the hostile entry. The real shard is deliberately not HF-sharded-named, so
    # `safe_tensor_index_files` is empty and `regenerate_index` stays false -- the
    # exact shape in which nothing else would touch the index.
    base_rel = _shadow_directory(tmp_path, f"../../escaped/{PAYLOAD}")
    shadow = os.path.join(str(tmp_path), base_rel)
    real_base = os.path.join(str(tmp_path), "real_base")
    shards = [f for f in os.listdir(real_base) if f.endswith(".safetensors")]
    shutil.copy2(
        os.path.join(real_base, shards[0]), os.path.join(shadow, "model.safetensors"),
    )
    index_path = os.path.join(shadow, "model.safetensors.index.json")
    with open(index_path, "r", encoding = "utf-8") as f:
        index = json.load(f)
    index["weight_map"]["model.norm.weight"] = "model.safetensors"
    with open(index_path, "w", encoding = "utf-8") as f:
        json.dump(index, f)

    monkeypatch.chdir(tmp_path)
    _stub_the_hub(monkeypatch)
    if path == "splitting":
        monkeypatch.setattr(
            saving_utils, "should_split_shards", lambda *a, **k: True, raising = True,
        )
    else:
        monkeypatch.setattr(
            saving_utils, "check_model_quantization_status",
            lambda *a, **k: (True, "mxfp4"), raising = True,
        )

    with pytest.raises(RuntimeError, match = "outside the model directory"):
        saving_utils.merge_and_overwrite_lora(
            get_model_name  = lambda *a, **k: base_rel,
            model           = peft_model,
            tokenizer       = None,
            save_directory  = base_rel,   # in place: the index is already in the output
            save_method     = "merged_16bit",
            push_to_hub     = False,
        )


def test_a_mixed_index_is_not_exported_verbatim(monkeypatch, tmp_path):
    """An index with good shards plus one escaping entry must not be copied out.

    Filtering the in-memory list still left `model.safetensors.index.json` itself
    copied verbatim whenever the remaining shards kept it nonempty, so the export
    carried a map that a later `from_pretrained` would join raw. The whole index
    is refused instead.
    """
    if not H.family_available(FAMILY):
        pytest.skip(f"{FAMILY} unavailable in this transformers")

    shadow = tmp_path / "ns" / "base"
    shadow.mkdir(parents = True)
    index = {
        "metadata"   : {"total_size" : 8},
        "weight_map" : {
            "a.weight" : "model-00001-of-00002.safetensors",
            "b.weight" : "weights/model-00002-of-00002.safetensors",
            "c.weight" : f"../../escaped/{PAYLOAD}",
        },
    }
    index_path = shadow / "model.safetensors.index.json"
    with open(index_path, "w", encoding = "utf-8") as f:
        json.dump(index, f)

    with pytest.raises(RuntimeError, match = "outside the model directory"):
        saving_utils._reject_unsafe_shard_index(str(index_path))

    # The two contained names, including the nested one, are not what triggered it.
    del index["weight_map"]["c.weight"]
    with open(index_path, "w", encoding = "utf-8") as f:
        json.dump(index, f)
    saving_utils._reject_unsafe_shard_index(str(index_path))


def test_an_index_that_is_not_valid_utf8_is_still_checked(tmp_path):
    """The guard must read what the eventual consumer reads.

    `transformers.utils.hub.get_checkpoint_shard_files` opens the index with a bare
    `open(index_filename)`, so it decodes with the locale encoding. Under cp1252,
    which is the Windows default, bytes that strict UTF-8 rejects still parse and the
    traversal is still followed. A guard that gave up on those bytes would wave the
    index through to a reader that does not.
    """
    index_path = tmp_path / "model.safetensors.index.json"
    # One stray 0xff inside metadata the attacker controls; the entry stays ASCII.
    index_path.write_bytes(
        b'{"metadata": {"n": "\xff"}, "weight_map": {"a": "../../' +
        PAYLOAD.encode() + b'"}}'
    )
    # The premise: strict UTF-8 cannot read this, cp1252 can.
    with pytest.raises(UnicodeDecodeError):
        index_path.read_text(encoding = "utf-8")
    assert "../../" in json.loads(index_path.read_text(encoding = "cp1252"))["weight_map"]["a"]

    with pytest.raises(RuntimeError, match = "outside the model directory"):
        saving_utils._reject_unsafe_shard_index(str(index_path))


@pytest.mark.parametrize("name, expected", [
    ("model-00001-of-00004.safetensors", True),
    ("weights/model-00001-of-00002.safetensors", True),
    ("..\\..\\escaped\\x.safetensors", False),   # one filename on POSIX, traversal on Windows
    ("C:\\x.safetensors", False),
    ("\\\\server\\share\\x.safetensors", False),
    ("../x.safetensors", False),
    ("/x.safetensors", False),
    ("\x00evil.safetensors", False),
])
def test_containment_is_judged_under_both_path_flavours(name, expected):
    """The index we vouch for is exported and read back somewhere else.

    Judging a backslash name by the host alone blesses an index that traverses on
    whichever machine opens it, so both rule sets have to agree before it is kept.
    """
    assert saving_utils._shard_name_stays_inside(name) is expected


def test_the_exported_index_is_the_one_that_was_validated(monkeypatch, tmp_path):
    """Validate one read and export another and the gap between them is exploitable.

    The guard used to check the file and `shutil.copy2` then opened it again; an
    index swapped in between was exported unchecked. Modelled deterministically by
    swapping it the instant validation returns, which is the race, won every time.
    """
    if not H.family_available(FAMILY):
        pytest.skip(f"{FAMILY} unavailable in this transformers")
    H.set_offline_cpu_env()

    spec = H.make_spec(FAMILY)
    base_dir = os.path.join(str(tmp_path), "base")
    model = H.build_and_save_base(spec, base_dir, max_shard_size = "40KB")
    peft_model = H.attach_lora(model, spec, "full")
    assert len([f for f in os.listdir(base_dir) if f.endswith(".safetensors")]) > 1

    poisoned = {
        "metadata": {"total_size": 8},
        "weight_map": {"a.weight": f"../../escaped/{PAYLOAD}"},
    }
    index_path = os.path.join(base_dir, "model.safetensors.index.json")
    real_guard = saving_utils._reject_unsafe_shard_index

    def swap_after_validating(path):
        validated = real_guard(path)
        with open(path, "w", encoding = "utf-8") as f:
            json.dump(poisoned, f)
        return validated
    monkeypatch.setattr(
        saving_utils, "_reject_unsafe_shard_index", swap_after_validating, raising = True,
    )
    _stub_the_hub(monkeypatch)

    save_directory = os.path.join(str(tmp_path), "exported")
    try:
        saving_utils.merge_and_overwrite_lora(
            get_model_name  = lambda *a, **k: base_dir,
            model           = peft_model,
            tokenizer       = None,
            save_directory  = save_directory,
            save_method     = "merged_16bit",
            push_to_hub     = False,
        )
    except Exception:
        pass

    exported = os.path.join(save_directory, "model.safetensors.index.json")
    if os.path.exists(exported):
        with open(exported, encoding = "utf-8") as f:
            values = list(json.load(f)["weight_map"].values())
        assert all(saving_utils._shard_name_stays_inside(v) for v in values), (
            f"the swapped index reached the export: {values!r}"
        )


def test_a_lone_nested_shard_keeps_its_index(monkeypatch, tmp_path):
    """One shard under a subdirectory is not a layout a loader can find unaided.

    `safe_tensor_index_files` is set for several shards or for the HF
    `model-0000n-of-0000m` naming, and a single `weights/model.safetensors` is
    neither. Without the index the export holds that one file, no root
    `model.safetensors`, and nothing naming what it does hold.
    """
    if not H.family_available(FAMILY):
        pytest.skip(f"{FAMILY} unavailable in this transformers")
    H.set_offline_cpu_env()

    spec = H.make_spec(FAMILY)
    real_base = os.path.join(str(tmp_path), "real_base")
    model = H.build_and_save_base(spec, real_base)
    shards = sorted(f for f in os.listdir(real_base) if f.endswith(".safetensors"))
    if len(shards) != 1:
        pytest.skip("the tiny base model did not fit in a single shard")
    peft_model = H.attach_lora(model, spec, "full")

    # Deliberately not an HF shard name, so only the nested test can carry the index.
    relative = "weights/model.safetensors"
    nested_root = os.path.join(str(tmp_path), "ns", "base")
    os.makedirs(os.path.join(nested_root, "weights"), exist_ok = True)
    shutil.copy2(
        os.path.join(real_base, "config.json"), os.path.join(nested_root, "config.json"),
    )
    destination = os.path.join(nested_root, "weights", "model.safetensors")
    shutil.copy2(os.path.join(real_base, shards[0]), destination)

    from safetensors import safe_open
    weight_map = {}
    with safe_open(destination, framework = "pt") as f:
        for key in f.keys():
            weight_map[key] = relative
    with open(
        os.path.join(nested_root, "model.safetensors.index.json"), "w", encoding = "utf-8",
    ) as f:
        json.dump(
            {"metadata" : {"total_size" : os.path.getsize(destination)},
             "weight_map" : weight_map}, f,
        )

    base_rel = os.path.join("ns", "base")
    save_directory = os.path.join("out", "deep", "merged")
    monkeypatch.chdir(tmp_path)
    _stub_the_hub(monkeypatch)

    saving_utils.merge_and_overwrite_lora(
        get_model_name  = lambda *a, **k: base_rel,
        model           = peft_model,
        tokenizer       = None,
        save_directory  = save_directory,
        save_method     = "merged_16bit",
        push_to_hub     = False,
    )

    output = os.path.join(str(tmp_path), save_directory)
    exported_index = os.path.join(output, "model.safetensors.index.json")
    assert os.path.exists(exported_index), (
        f"a lone nested shard was exported without an index: {os.listdir(output)}"
    )
    with open(exported_index, encoding = "utf-8") as f:
        exported_map = json.load(f)["weight_map"]
    assert exported_map, "the exported index names no shard"
    for value in set(exported_map.values()):
        assert os.path.exists(os.path.join(output, value)), (
            f"the exported index names {value!r}, which is not in the export"
        )


@pytest.mark.parametrize("payload", ["[]", "null", '"index"', "3"], ids = list("lnsi"))
def test_a_stale_index_that_is_not_an_object_does_not_abort_the_merge(
    monkeypatch, tmp_path, payload,
):
    """Valid JSON that is not an object carries no weight_map to traverse with.

    `json.loads` does not raise on any of these, so the guard's `except` never sees
    them and an unconditional `.get` turned a stale file beside a perfectly good
    single-shard model into an AttributeError out of the whole merge.
    """
    if not H.family_available(FAMILY):
        pytest.skip(f"{FAMILY} unavailable in this transformers")
    H.set_offline_cpu_env()

    spec = H.make_spec(FAMILY)
    real_base = os.path.join(str(tmp_path), "real_base")
    model = H.build_and_save_base(spec, real_base)
    base_tensors = H.read_safetensors_dir(real_base)
    peft_model = H.attach_lora(model, spec, "full")
    adapted = H.extract_adapted(peft_model)

    # The shards stay where they are; only the stale index is bogus.
    with open(
        os.path.join(real_base, "model.safetensors.index.json"), "w", encoding = "utf-8",
    ) as f:
        f.write(payload)

    monkeypatch.chdir(tmp_path)
    _stub_the_hub(monkeypatch)
    save_directory = os.path.join("out", "merged")

    saving_utils.merge_and_overwrite_lora(
        get_model_name  = lambda *a, **k: "real_base",
        model           = peft_model,
        tokenizer       = None,
        save_directory  = save_directory,
        save_method     = "merged_16bit",
        output_dtype    = torch.float32,
        push_to_hub     = False,
    )

    H.assert_merge_correct(
        family       = FAMILY,
        base_tensors = base_tensors,
        out_dir      = os.path.join(str(tmp_path), save_directory),
        save_dtype   = torch.float32,
        adapted      = adapted,
        base_dir     = real_base,
    )


def test_an_inert_index_beside_usable_shards_does_not_block_the_export(
    monkeypatch, tmp_path,
):
    """An index the export never carries must not be able to veto it.

    When `os.listdir` already found the shards, the index values are never used for
    the shard list, and a single non-HF-named shard means the index is not copied
    either. Refusing on its contents would fail an export that has always worked over
    a file nothing downstream reads.
    """
    if not H.family_available(FAMILY):
        pytest.skip(f"{FAMILY} unavailable in this transformers")
    H.set_offline_cpu_env()

    spec = H.make_spec(FAMILY)
    base_dir = os.path.join(str(tmp_path), "base")
    model = H.build_and_save_base(spec, base_dir)
    shards = [f for f in os.listdir(base_dir) if f.endswith(".safetensors")]
    if len(shards) != 1 or shards[0] != "model.safetensors":
        pytest.skip("the tiny base model is not the single-flat-shard layout under test")
    base_tensors = H.read_safetensors_dir(base_dir)
    peft_model = H.attach_lora(model, spec, "full")
    adapted = H.extract_adapted(peft_model)

    # Stale, and as bad as it gets: a traversal and a value that is not even a string.
    with open(
        os.path.join(base_dir, "model.safetensors.index.json"), "w", encoding = "utf-8",
    ) as f:
        json.dump({"weight_map": {"a": "../../x.safetensors", "b": None}}, f)

    monkeypatch.chdir(tmp_path)
    save_directory = os.path.join(str(tmp_path), "merged")
    H.run_merge(peft_model, base_dir, save_directory, save_dtype = torch.float32)

    # It merged, and the hostile index did not travel with it.
    assert not os.path.exists(
        os.path.join(save_directory, "model.safetensors.index.json")
    ), "the inert index was exported after all, so refusing it would have been right"
    H.assert_merge_correct(
        family = FAMILY, base_tensors = base_tensors, out_dir = save_directory,
        save_dtype = torch.float32, adapted = adapted, base_dir = base_dir,
    )


def test_the_exported_index_keeps_the_mode_and_time_copy2_gave_it(
    monkeypatch, tmp_path,
):
    """The validated-byte write replaced `copy2`, which is copy plus copystat.

    Only the copy half was replaced, so without the copystat the exported index takes
    the process creation mode rather than the source's: a `0600` index lands world
    readable, a widening in the one file this change exists to keep honest.
    """
    if not H.family_available(FAMILY):
        pytest.skip(f"{FAMILY} unavailable in this transformers")
    H.set_offline_cpu_env()

    spec = H.make_spec(FAMILY)
    base_dir = os.path.join(str(tmp_path), "base")
    # Small shards, so the export is multi-shard and the index is really carried.
    model = H.build_and_save_base(spec, base_dir, max_shard_size = "32KB")
    if len([f for f in os.listdir(base_dir) if f.endswith(".safetensors")]) < 2:
        pytest.skip("the tiny base model did not shard into more than one file")
    peft_model = H.attach_lora(model, spec, "full")

    source_index = os.path.join(base_dir, "model.safetensors.index.json")
    assert os.path.exists(source_index), "the sharded base wrote no index"
    os.chmod(source_index, 0o600)
    backdated = os.stat(source_index).st_mtime - 100000
    os.utime(source_index, (backdated, backdated))

    monkeypatch.chdir(tmp_path)
    save_directory = os.path.join(str(tmp_path), "merged")
    H.run_merge(peft_model, base_dir, save_directory, save_dtype = torch.float32)

    exported = os.path.join(save_directory, "model.safetensors.index.json")
    assert os.path.exists(exported), "the index was not exported"
    source_stat, exported_stat = os.stat(source_index), os.stat(exported)
    assert oct(exported_stat.st_mode & 0o777) == oct(source_stat.st_mode & 0o777), (
        f"the exported index widened from {oct(source_stat.st_mode & 0o777)} to "
        f"{oct(exported_stat.st_mode & 0o777)}"
    )
    assert int(exported_stat.st_mtime) == int(source_stat.st_mtime), (
        "the exported index did not keep the source mtime that copy2 preserved"
    )


def _symlinked_model(tmp_path, spec, *, link_the_parent = False):
    """A model directory whose shard is a link out of it, and no index at all.

    This is the shape a Hugging Face cache snapshot has, every shard a link into a
    shared `blobs/` directory, and it reaches the merge through the `os.listdir`
    branch without any `weight_map` involved.
    """
    real_base = os.path.join(str(tmp_path), "real_base")
    shard = sorted(f for f in os.listdir(real_base) if f.endswith(".safetensors"))[0]

    outside = os.path.join(str(tmp_path), "outside")
    os.makedirs(outside, exist_ok = True)
    victim = os.path.join(outside, "victim.safetensors")
    shutil.copy2(os.path.join(real_base, shard), victim)

    shadow = os.path.join(str(tmp_path), "ns", "base")
    os.makedirs(shadow, exist_ok = True)
    shutil.copy2(
        os.path.join(real_base, "config.json"), os.path.join(shadow, "config.json"),
    )
    if link_the_parent:
        # The shard itself is an ordinary file; a PARENT component is the link, which
        # replacing the file cannot repair.
        holder = os.path.join(str(tmp_path), "outside_holder")
        os.makedirs(holder, exist_ok = True)
        shutil.copy2(os.path.join(real_base, shard), os.path.join(holder, "model.safetensors"))
        os.symlink(holder, os.path.join(shadow, "weights"))
        with open(
            os.path.join(shadow, "model.safetensors.index.json"), "w", encoding = "utf-8",
        ) as f:
            from safetensors import safe_open
            weight_map = {}
            with safe_open(os.path.join(holder, "model.safetensors"), framework = "pt") as g:
                for key in g.keys():
                    weight_map[key] = "weights/model.safetensors"
            json.dump({"metadata": {"total_size": 1}, "weight_map": weight_map}, f)
    else:
        os.symlink(victim, os.path.join(shadow, "model.safetensors"))
    assert not os.path.exists(os.path.join(shadow, "model.safetensors")) or \
        os.path.islink(os.path.join(shadow, "model.safetensors")) or link_the_parent
    return os.path.join("ns", "base"), victim


def test_a_symlinked_shard_is_not_written_through(monkeypatch, tmp_path):
    """The in-place merge must not write into whatever a shard links to.

    Nothing copies the shard first, because `os.path.exists` is true THROUGH the link,
    so the `r+b` overwrite went straight into the target. No index is involved, which
    is why none of the `weight_map` guarding catches it. It is materialised rather than
    refused: this is the ordinary shape of an HF cache snapshot, and writing through
    would corrupt a blob other models share.
    """
    if not H.family_available(FAMILY):
        pytest.skip(f"{FAMILY} unavailable in this transformers")
    H.set_offline_cpu_env()

    spec = H.make_spec(FAMILY)
    real_base = os.path.join(str(tmp_path), "real_base")
    model = H.build_and_save_base(spec, real_base)
    base_tensors = H.read_safetensors_dir(real_base)
    peft_model = H.attach_lora(model, spec, "full")
    adapted = H.extract_adapted(peft_model)

    base_rel, victim = _symlinked_model(tmp_path, spec)
    before = hashlib.sha256(open(victim, "rb").read()).hexdigest()

    monkeypatch.chdir(tmp_path)
    _stub_the_hub(monkeypatch)

    saving_utils.merge_and_overwrite_lora(
        get_model_name  = lambda *a, **k: base_rel,
        model           = peft_model,
        tokenizer       = None,
        save_directory  = base_rel,          # in place: the case that escaped
        save_method     = "merged_16bit",
        output_dtype    = torch.float32,
        push_to_hub     = False,
    )

    after = hashlib.sha256(open(victim, "rb").read()).hexdigest()
    assert before == after, (
        f"the merge wrote through the link into {victim!r}, outside the output directory"
    )
    output = os.path.join(str(tmp_path), base_rel)
    shard = os.path.join(output, "model.safetensors")
    assert not os.path.islink(shard), "the shard is still a link, so the next write escapes"
    # And it really merged, into its own copy.
    H.assert_merge_correct(
        family = FAMILY, base_tensors = base_tensors, out_dir = output,
        save_dtype = torch.float32, adapted = adapted, base_dir = real_base,
    )


def test_a_shard_behind_a_symlinked_parent_is_refused(monkeypatch, tmp_path):
    """Replacing the file cannot repair a path that escapes through a parent link.

    So the writer refuses instead of materialising, rather than quietly writing out of
    the directory the user asked to export to.
    """
    if not H.family_available(FAMILY):
        pytest.skip(f"{FAMILY} unavailable in this transformers")
    H.set_offline_cpu_env()

    spec = H.make_spec(FAMILY)
    real_base = os.path.join(str(tmp_path), "real_base")
    model = H.build_and_save_base(spec, real_base)
    peft_model = H.attach_lora(model, spec, "full")

    base_rel, _victim = _symlinked_model(tmp_path, spec, link_the_parent = True)
    holder = os.path.join(str(tmp_path), "outside_holder", "model.safetensors")
    before = hashlib.sha256(open(holder, "rb").read()).hexdigest()

    monkeypatch.chdir(tmp_path)
    _stub_the_hub(monkeypatch)

    with pytest.raises(RuntimeError, match = "outside the output directory"):
        saving_utils.merge_and_overwrite_lora(
            get_model_name  = lambda *a, **k: base_rel,
            model           = peft_model,
            tokenizer       = None,
            save_directory  = base_rel,
            save_method     = "merged_16bit",
            push_to_hub     = False,
        )

    after = hashlib.sha256(open(holder, "rb").read()).hexdigest()
    assert before == after, "the file behind the linked parent was modified anyway"


def test_a_symlink_that_stays_inside_the_output_is_left_alone(tmp_path):
    """Only a link OUT is repaired; one that resolves back inside is not touched."""
    output = os.path.join(str(tmp_path), "out")
    os.makedirs(os.path.join(output, "real"), exist_ok = True)
    target = os.path.join(output, "real", "shard.safetensors")
    with open(target, "wb") as f:
        f.write(b"inside")
    link = os.path.join(output, "model.safetensors")
    os.symlink(target, link)

    saving_utils._materialize_shard_that_resolves_outside(link, output)

    assert os.path.islink(link), "a contained link was needlessly replaced"
    assert saving_utils._resolves_inside(link, output)
