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

    Counting the shard names is what pins the collapse, but a count alone is satisfied
    by a merge that kept both names and wrote nothing useful into either, so the
    tensors are checked against the independent reference as well.
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
    base_tensors = H.read_safetensors_dir(real_base)
    peft_model = H.attach_lora(model, spec, "full")
    adapted = H.extract_adapted(peft_model)

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

    # Both shards are still where the index says they are, and both really hold the
    # merge. Without this the test passes on a run that kept two names and left half
    # the tensors unmerged, which is the failure it exists to catch.
    output = os.path.join(str(tmp_path), base_rel)
    for relative in sorted(set(weight_map.values())):
        assert os.path.exists(os.path.join(output, relative)), (
            f"{relative!r} is missing from {output!r} after the merge"
        )
    # `_flattened` is no use here: both shards are named `x.safetensors`, which is the
    # whole point of the case, so it would copy one over the other and hide exactly the
    # loss being tested. Number them by their parent instead.
    flat = os.path.join(str(tmp_path), "flat")
    os.makedirs(flat, exist_ok = True)
    for relative in sorted(set(weight_map.values())):
        shutil.copy2(
            os.path.join(output, relative),
            os.path.join(flat, relative.replace("/", "_").replace(os.sep, "_")),
        )
    H.assert_merge_correct(
        family       = FAMILY,
        base_tensors = base_tensors,
        out_dir      = flat,
        save_dtype   = torch.float32,
        adapted      = adapted,
        base_dir     = real_base,
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


@pytest.mark.parametrize("quant_type", ["mxfp4", "fp8"])
def test_a_dequantized_nested_singleton_still_gets_an_index(
    monkeypatch, tmp_path, quant_type,
):
    """A dequant export skips the index-copy block, so regeneration is the only path.

    `regenerate_index` also needed more than one shard, or an HF-sharded name, so a lone
    `weights/model.safetensors` got neither. `from_pretrained` looks for exactly
    `model.safetensors` then `model.safetensors.index.json` at the ROOT of the directory
    (`modeling_utils`, transformers 5.17.0), and a nested singleton is neither, so the
    export had no discoverable weights at all.

    The tensor-level dequant arithmetic is stubbed, deliberately: it is irrelevant to
    which index the export ends up with, and it is the only part that needs real
    quantized weights. Everything deciding and writing the index is the real code.
    """
    if not H.family_available(FAMILY):
        pytest.skip(f"{FAMILY} unavailable in this transformers")
    H.set_offline_cpu_env()

    spec = H.make_spec(FAMILY)
    real_base = os.path.join(str(tmp_path), "real_base")
    model = H.build_and_save_base(spec, real_base)
    peft_model = H.attach_lora(model, spec, "full")

    nested_root, base_rel, weight_map = _nested_layout(tmp_path, ["weights"])
    if len(set(weight_map.values())) != 1:
        pytest.skip("the tiny base model did not fit in a single nested shard")

    monkeypatch.chdir(tmp_path)
    # A quantized base, so the export takes the dequant route.
    monkeypatch.setattr(
        saving_utils, "check_hf_model_exists", lambda *a, **k: True, raising = True,
    )
    monkeypatch.setattr(
        saving_utils, "check_model_quantization_status",
        lambda *a, **k: (True, quant_type), raising = True,
    )
    # The shard rewrite runs for real, as an ordinary 16bit merge: only the tensor-level
    # dequant needs weights this tiny model does not have, and it has no bearing on which
    # index the export ends up with. Everything deciding and writing the index is real.
    _real_merge = saving_utils._merge_and_overwrite_lora
    def _merge_without_dequant(*a, **k):
        k["base_model_is_quantized"] = False
        k["quant_type"] = None
        return _real_merge(*a, **k)
    monkeypatch.setattr(
        saving_utils, "_merge_and_overwrite_lora", _merge_without_dequant, raising = True,
    )

    save_directory = os.path.join("out", "merged")
    saving_utils.merge_and_overwrite_lora(
        get_model_name  = lambda *a, **k: base_rel,
        model           = peft_model,
        tokenizer       = None,
        save_directory  = save_directory,
        save_method     = "merged_16bit",
        push_to_hub     = False,
    )

    output = os.path.join(str(tmp_path), save_directory)
    root_shard = os.path.join(output, "model.safetensors")
    index = os.path.join(output, "model.safetensors.index.json")
    assert os.path.exists(root_shard) or os.path.exists(index), (
        f"a dequantized nested singleton was exported with neither a root "
        f"model.safetensors nor an index, so nothing can load it: {os.listdir(output)}"
    )
    if os.path.exists(index):
        with open(index, encoding = "utf-8") as f:
            exported = json.load(f)["weight_map"]
        assert exported, "the regenerated index names no shard"
        for value in set(exported.values()):
            assert os.path.exists(os.path.join(output, value)), (
                f"the regenerated index names {value!r}, which is not in the export"
            )


@pytest.mark.parametrize("directory, path, expected", [
    ("/",     "/model.safetensors",          True),
    ("/",     "/sub/model.safetensors",      True),
    ("/tmp",  "/tmp/model.safetensors",      True),
    ("/tmp",  "/tmp",                        True),
    ("/tmp",  "/etc/passwd",                 False),
    ("/tmp",  "/tmpfoo/model.safetensors",   False),
])
def test_containment_holds_at_a_filesystem_root(directory, path, expected):
    """A `root + os.sep` prefix test is wrong at a filesystem root.

    `os.path.realpath("/")` is `/`, so the prefix becomes `//` and `/model.safetensors`
    reads as outside the directory that holds it. Every guarded writer then refuses a
    shard that is genuinely inside, after the config files have already been written.
    The `/tmpfoo` case pins that the fix does not go the other way and accept a sibling
    whose name merely starts with the directory's.
    """
    assert saving_utils._resolves_inside(path, directory) is expected


@pytest.mark.parametrize("name, nested", [
    ("model.safetensors",                    False),
    ("model-00001-of-00002.safetensors",     False),
    ("weights/model.safetensors",            True),
    ("weights\\model.safetensors",           True),
    ("a/b/model.safetensors",                True),
    ("a\\b\\model.safetensors",              True),
])
def test_nesting_is_detected_under_both_separator_rules(name, nested):
    """`_shard_name_stays_inside` admits a name if it is contained under BOTH rules, so
    `weights\\model.safetensors` survives and is written verbatim. Detecting nesting with
    only the native separator misses it on POSIX, and the export then carries neither a
    root `model.safetensors` nor an index naming the file it does have."""
    assert saving_utils._shard_name_stays_inside(name) is True
    assert saving_utils._has_directory_component(name) is nested


def test_a_backslash_named_singleton_still_gets_an_index(monkeypatch, tmp_path):
    """End to end for the case above: the export must be loadable, not just consistent."""
    if not H.family_available(FAMILY):
        pytest.skip(f"{FAMILY} unavailable in this transformers")
    if os.name == "nt":
        pytest.skip("a backslash is a separator on Windows, so this is the nested case")
    H.set_offline_cpu_env()

    spec = H.make_spec(FAMILY)
    real_base = os.path.join(str(tmp_path), "real_base")
    model = H.build_and_save_base(spec, real_base)
    shards = [f for f in os.listdir(real_base) if f.endswith(".safetensors")]
    if len(shards) != 1:
        pytest.skip("the tiny base model did not fit in a single shard")
    peft_model = H.attach_lora(model, spec, "full")

    # One shard, named with a backslash: an ordinary POSIX filename, no directory.
    relative = "weights\\model.safetensors"
    shadow = os.path.join(str(tmp_path), "ns", "base")
    os.makedirs(shadow, exist_ok = True)
    shutil.copy2(os.path.join(real_base, "config.json"), os.path.join(shadow, "config.json"))
    shutil.copy2(os.path.join(real_base, shards[0]), os.path.join(shadow, relative))
    from safetensors import safe_open
    weight_map = {}
    with safe_open(os.path.join(shadow, relative), framework = "pt") as f:
        for key in f.keys():
            weight_map[key] = relative
    with open(
        os.path.join(shadow, "model.safetensors.index.json"), "w", encoding = "utf-8",
    ) as f:
        json.dump({"metadata": {"total_size": 1}, "weight_map": weight_map}, f)

    base_rel = os.path.join("ns", "base")
    save_directory = os.path.join("out", "merged")
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
    index = os.path.join(output, "model.safetensors.index.json")
    assert os.path.exists(os.path.join(output, "model.safetensors")) or os.path.exists(index), (
        f"nothing in the export is discoverable by a loader: {os.listdir(output)}"
    )
    if os.path.exists(index):
        with open(index, encoding = "utf-8") as f:
            exported = json.load(f)["weight_map"]
        for value in set(exported.values()):
            assert os.path.exists(os.path.join(output, value)), (
                f"the exported index names {value!r}, which is not in the export"
            )


# Names that leave a directory and re-enter a sibling of the same name. A containment
# test that joins onto a STAND-IN root and checks the prefix accepts every one of these,
# because they normalise back under the stand-in while the real join lands outside.
COLLIDING_NAMES = [
    pytest.param("../unsloth_shard_root/victim.safetensors", id = "re-enter-the-stand-in"),
    pytest.param("..///unsloth_shard_root/victim.safetensors", id = "extra-separators"),
    pytest.param("sub/../../unsloth_shard_root/victim.safetensors", id = "via-a-subdirectory"),
    pytest.param("..\\unsloth_shard_root\\victim.safetensors", id = "windows-spelling"),
]


@pytest.mark.parametrize("name", COLLIDING_NAMES)
def test_a_name_that_re_enters_a_same_named_sibling_is_refused(name):
    """The predicate must not be defeated by knowing the stand-in root's name."""
    assert saving_utils._shard_name_stays_inside(name) is False, (
        f"{name!r} was accepted; joined onto a real output it resolves to "
        f"{os.path.normpath(os.path.join('/tmp/out/merged', name))!r}"
    )


@pytest.mark.parametrize("name", COLLIDING_NAMES)
def test_a_colliding_name_never_reaches_the_filesystem(monkeypatch, tmp_path, name):
    """End to end: nothing may be created outside the requested output directory.

    The write sinks refuse this name, but the shard preparation loop runs first and its
    `os.makedirs` plus `shutil.copy2` had already created the file outside before the
    refusal. Asserting on the filesystem rather than on the exception is the point.
    """
    if not H.family_available(FAMILY):
        pytest.skip(f"{FAMILY} unavailable in this transformers")
    H.set_offline_cpu_env()

    spec = H.make_spec(FAMILY)
    real_base = os.path.join(str(tmp_path), "real_base")
    model = H.build_and_save_base(spec, real_base)
    peft_model = H.attach_lora(model, spec, "full")

    base_rel = _shadow_directory(tmp_path, name)
    _plant_payload(tmp_path, base_rel, name)
    save_directory = os.path.join("out", "deep", "merged")

    monkeypatch.chdir(tmp_path)
    _stub_the_hub(monkeypatch)
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
        # Refusing is fine. Escaping is not, and that is what is asserted below.
        pass

    inside = os.path.realpath(os.path.join(str(tmp_path), save_directory))
    escaped = []
    for root, _dirs, files in os.walk(os.path.join(str(tmp_path), "out")):
        for entry in files:
            resolved = os.path.realpath(os.path.join(root, entry))
            if resolved != inside and not resolved.startswith(inside + os.sep):
                escaped.append(os.path.relpath(resolved, str(tmp_path)))
    assert not escaped, f"{name!r} put {escaped} outside {save_directory!r}"


@pytest.mark.parametrize("name", [
    pytest.param("C:..\\victim.safetensors",  id = "drive-relative-parent"),
    pytest.param("C:victim.safetensors",      id = "drive-relative-plain"),
    pytest.param("C:../victim.safetensors",   id = "drive-relative-posix-sep"),
    pytest.param("c:..\\victim.safetensors",  id = "lowercase-drive"),
    pytest.param("Z:..\\victim.safetensors",  id = "other-drive"),
])
def test_a_drive_relative_name_is_refused(name):
    """`ntpath.isabs('C:..\\x')` is False and normpath keeps the drive ahead of the
    `..`, so neither the absolute test nor the leading-`..` test sees it. On Windows it
    joins relative to that drive's working directory: `C:\\out\\merged` + `C:..\\x` is
    `C:\\out\\x`, and another drive leaves the output tree altogether."""
    assert saving_utils._shard_name_stays_inside(name) is False


def test_a_symlinked_output_component_is_refused_before_the_copy(monkeypatch, tmp_path):
    """An out-of-place export whose OUTPUT already contains a linked component.

    `save_directory/weights -> /outside` with a legitimate index entry
    `weights/model.safetensors`: `makedirs` accepts the existing link and `copy2` writes
    straight through it. The write sinks refuse the path, but only much later, so the
    file was already outside by the time the export reported a refusal. Asserting on the
    filesystem, not on the exception.
    """
    if not H.family_available(FAMILY):
        pytest.skip(f"{FAMILY} unavailable in this transformers")
    H.set_offline_cpu_env()

    spec = H.make_spec(FAMILY)
    real_base = os.path.join(str(tmp_path), "real_base")
    model = H.build_and_save_base(spec, real_base)
    peft_model = H.attach_lora(model, spec, "full")
    _nested_root, base_rel, _weight_map = _nested_layout(tmp_path, ["weights"])

    outside = os.path.join(str(tmp_path), "outside")
    os.makedirs(outside, exist_ok = True)
    save_directory = os.path.join("out", "merged")
    os.makedirs(os.path.join(str(tmp_path), save_directory), exist_ok = True)
    os.symlink(outside, os.path.join(str(tmp_path), save_directory, "weights"))

    monkeypatch.chdir(tmp_path)
    _stub_the_hub(monkeypatch)
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
        pass

    assert os.listdir(outside) == [], (
        f"the export wrote {os.listdir(outside)} through the linked output component"
    )


def test_a_failed_materialization_leaves_the_input_checkpoint_intact(tmp_path):
    """The shard can be gigabytes; a failed copy must not destroy what it replaces.

    Unlinking first and copying second leaves the user's own checkpoint with its link
    gone and a partial file in its place on ENOSPC or an interruption.
    """
    output = os.path.join(str(tmp_path), "model_dir")
    os.makedirs(output, exist_ok = True)
    outside = os.path.join(str(tmp_path), "outside")
    os.makedirs(outside, exist_ok = True)
    target = os.path.join(outside, "victim.safetensors")
    with open(target, "wb") as f:
        f.write(b"the real weights")
    link = os.path.join(output, "model.safetensors")
    os.symlink(target, link)

    # Injected where the staging copy actually happens, so this cannot quietly stop
    # exercising a failure if the copy mechanism changes again.
    boom = RuntimeError("no space left on device")
    real_copyfileobj = shutil.copyfileobj
    def failing_copyfileobj(src, dst, *a, **k):
        dst.write(src.read(4))            # a partial file exists, as it would on ENOSPC
        dst.flush()
        raise boom
    shutil.copyfileobj = failing_copyfileobj
    try:
        with pytest.raises(RuntimeError):
            saving_utils._materialize_shard_that_resolves_outside(link, output)
    finally:
        shutil.copyfileobj = real_copyfileobj

    assert os.path.islink(link), "the link was removed before the copy succeeded"
    assert os.path.realpath(link) == os.path.realpath(target)
    with open(target, "rb") as f:
        assert f.read() == b"the real weights", "the link target was modified"
    leftovers = [n for n in os.listdir(output) if ".unsloth-materializing" in n]
    assert leftovers == [], f"a staging file was left behind: {leftovers}"


@pytest.mark.parametrize("name", [
    pytest.param(".. \\victim.safetensors",     id = "parent-plus-trailing-space"),
    pytest.param(".. /victim.safetensors",      id = "parent-plus-space-posix-sep"),
    pytest.param("   \\victim.safetensors",     id = "spaces-only-becomes-rooted"),
    # Windows trims the trailing run of spaces AND periods, so each of these opens as
    # `..`. Stripping only the spaces left the first two looking like ordinary names,
    # and stripping the run with a plain rstrip(" .") takes them to the empty string,
    # which normpath drops: either way the traversal is accepted.
    pytest.param(".. .\\victim.safetensors",    id = "parent-space-period"),
    pytest.param(".. ./victim.safetensors",     id = "parent-space-period-posix-sep"),
    pytest.param("...\\victim.safetensors",     id = "three-periods"),
    pytest.param(". .\\victim.safetensors",     id = "period-space-period"),
])
def test_a_dots_and_spaces_component_is_refused(name):
    """Windows strips trailing spaces and periods from a name at the object manager
    layer, so `.. ` is created and opened as `..`, while `ntpath.normpath` preserves the
    space and the leading-`..` test never matches. `...` is a legal Windows name rather
    than a traversal, but no shard is called that, so the whole family is refused rather
    than modelled. See learn.microsoft.com/en-us/dotnet/standard/io/file-path-formats"""
    assert saving_utils._shard_name_stays_inside(name) is False


def test_the_staging_path_is_never_written_through_a_symlink(tmp_path):
    """The staging file sits in the same attacker-supplied directory as the shard.

    Opening it by name let `copy2` follow a planted symlink and overwrite its target: a
    second write sink introduced by the staging step that was meant to make the first
    one safe.
    """
    model_dir = os.path.join(str(tmp_path), "model")
    os.makedirs(model_dir, exist_ok = True)
    outside = os.path.join(str(tmp_path), "outside")
    os.makedirs(outside, exist_ok = True)

    shard_target = os.path.join(outside, "real.safetensors")
    with open(shard_target, "wb") as f:
        f.write(b"REAL SHARD BYTES")
    shard = os.path.join(model_dir, "model.safetensors")
    os.symlink(shard_target, shard)

    # A second external file, reachable only through the staging pathname that the
    # earlier fixed-name staging would have used.
    second = os.path.join(outside, "second_victim.bin")
    with open(second, "wb") as f:
        f.write(b"DO NOT TOUCH ME")
    planted = shard + ".unsloth-materializing"
    os.symlink(second, planted)

    saving_utils._materialize_shard_that_resolves_outside(shard, model_dir)

    with open(second, "rb") as f:
        assert f.read() == b"DO NOT TOUCH ME", (
            "the copy followed the planted staging symlink and overwrote its target"
        )
    # The shard was still materialised correctly.
    assert not os.path.islink(shard)
    with open(shard, "rb") as f:
        assert f.read() == b"REAL SHARD BYTES"
    # The planted path is left exactly as it was found. Staging picks a name of its own,
    # so there is nothing here it has to clear, and clearing it would be a deletion of
    # someone else's file rather than a cleanup of ours.
    assert os.path.islink(planted) and os.path.realpath(planted) == os.path.realpath(second)
    leftovers = [
        n for n in os.listdir(model_dir)
        if ".unsloth-materializing" in n and os.path.join(model_dir, n) != planted
    ]
    assert leftovers == [], f"a staging file was left behind: {leftovers}"


def test_a_file_sitting_at_the_old_staging_name_is_not_deleted(tmp_path):
    """Staging used to unlink `<shard>.unsloth-materializing` before creating it, to
    survive a leftover from an interrupted run. That unlink deletes whatever is really
    there, and for an in-place merge of a symlinked checkpoint the directory is the
    user's own: an unrelated file, or another shard this same index lists.
    """
    model_dir = os.path.join(str(tmp_path), "model")
    os.makedirs(model_dir, exist_ok = True)
    outside = os.path.join(str(tmp_path), "outside")
    os.makedirs(outside, exist_ok = True)

    shard_target = os.path.join(outside, "real.safetensors")
    with open(shard_target, "wb") as f:
        f.write(b"REAL SHARD BYTES")
    shard = os.path.join(model_dir, "model.safetensors")
    os.symlink(shard_target, shard)

    bystander = shard + ".unsloth-materializing"
    with open(bystander, "wb") as f:
        f.write(b"SOMEBODY ELSE'S CHECKPOINT")

    saving_utils._materialize_shard_that_resolves_outside(shard, model_dir)

    assert os.path.exists(bystander), "staging deleted a file it did not create"
    with open(bystander, "rb") as f:
        assert f.read() == b"SOMEBODY ELSE'S CHECKPOINT"
    assert not os.path.islink(shard)
    with open(shard, "rb") as f:
        assert f.read() == b"REAL SHARD BYTES"


@pytest.mark.parametrize("name", [
    "model.safetensors",
    "./model.safetensors",
    "a/./model.safetensors",
    "a/../model.safetensors",
    "weights/model.safetensors",
    "a\\.. \\victim.safetensors",
    ". \\victim.safetensors",
])
def test_a_contained_name_is_not_refused_by_the_windows_view(name):
    """The Windows-visible spelling must not reject names that stay inside.

    A single `.` is the current directory however it is spelled, and an interior `..`
    only cancels the component before it, so neither leaves the directory. An earlier,
    blunter rule that refused every dots-and-spaces component rejected all of these,
    each of which merges on main.

    `...` used to be listed here as a legal Windows name. It is not: its trailing
    periods come off with the rest of the trailing run and nothing legal is left, and
    telling it apart from `.. .` after that strip is guesswork on a predicate that
    decides whether a shard can be written outside the export. No shard is named `...`,
    so the whole two-or-more-dots family is refused instead.
    """
    assert saving_utils._shard_name_stays_inside(name) is True


def test_a_linked_parent_and_linked_shard_leaves_the_external_directory_alone(tmp_path):
    """Materialising is a write, so it must not happen in a directory that is not ours.

    With `save_directory/weights -> /outside/dir` AND
    `/outside/dir/model.safetensors -> /victim`, the repair replaced the link inside
    `/outside/dir`, mutating a directory the export has no business touching, and only
    afterwards did the containment check refuse. The victim's bytes survived, because
    the repair copies content in, but an external directory had still been rewritten.
    """
    save_directory = os.path.join(str(tmp_path), "out", "merged")
    os.makedirs(save_directory, exist_ok = True)
    outside_dir = os.path.join(str(tmp_path), "outside", "dir")
    os.makedirs(outside_dir, exist_ok = True)
    victim = os.path.join(str(tmp_path), "victim.safetensors")
    with open(victim, "wb") as f:
        f.write(b"VICTIM BYTES")

    external_shard = os.path.join(outside_dir, "model.safetensors")
    os.symlink(victim, external_shard)
    os.symlink(outside_dir, os.path.join(save_directory, "weights"))

    before = sorted(os.listdir(outside_dir))
    file_path = os.path.join(save_directory, "weights", "model.safetensors")
    saving_utils._materialize_shard_that_resolves_outside(file_path, save_directory)

    assert os.path.islink(external_shard), (
        "the link inside the external directory was replaced with a regular file"
    )
    assert sorted(os.listdir(outside_dir)) == before
    with open(victim, "rb") as f:
        assert f.read() == b"VICTIM BYTES"
    # And the export still refuses, which is what should have happened all along.
    with pytest.raises(RuntimeError, match = "outside the output directory"):
        saving_utils._assert_shard_is_inside(file_path, save_directory)


def test_an_external_parent_with_a_leaf_pointing_back_inside_is_refused(tmp_path):
    """The entry is what a writer replaces, so resolving the leaf is not enough.

    Layout, all of it something an untrusted model directory can contain:

        out/merged/weights        -> ../../outside
        outside/model.safetensors -> ../out/merged/real.safetensors
        out/merged/real.safetensors = a real shard

    `_resolves_inside` on the whole path follows the leaf and lands on
    `out/merged/real.safetensors`, which IS inside, so the guard used to approve it.
    `_materialize_shard_that_resolves_outside` correctly declines, because the parent
    is external, and deliberately leaves the refusal to this check -- which then did
    not refuse.

    `os.replace`, the mxfp4 and fp8 sinks, replaces the directory ENTRY and never
    follows the last component, so the merge wrote a new regular file into `outside/`.
    Asserted here as the guard's contract rather than through a dequant merge, because
    reaching those two writers needs real quantized weights the tiny base has not got.
    """
    save_directory = os.path.join(str(tmp_path), "out", "merged")
    os.makedirs(save_directory, exist_ok = True)
    outside = os.path.join(str(tmp_path), "outside")
    os.makedirs(outside, exist_ok = True)

    real = os.path.join(save_directory, "real.safetensors")
    with open(real, "wb") as f:
        f.write(b"INSIDE BYTES")
    os.symlink(outside, os.path.join(save_directory, "weights"))
    os.symlink(real, os.path.join(outside, "model.safetensors"))

    file_path = os.path.join(save_directory, "weights", "model.safetensors")
    # The leaf really does resolve inside; that is the whole trap.
    assert saving_utils._resolves_inside(file_path, save_directory)
    # Materialisation declines, by design, because the parent is external.
    saving_utils._materialize_shard_that_resolves_outside(file_path, save_directory)

    with pytest.raises(RuntimeError, match = "outside the output directory"):
        saving_utils._assert_shard_is_inside(file_path, save_directory)

    # Nothing may have been created in the external directory on the way to refusing.
    assert sorted(os.listdir(outside)) == ["model.safetensors"]
    assert os.path.islink(os.path.join(outside, "model.safetensors"))


@pytest.mark.parametrize("length", [200, 223, 230, 250])
def test_a_long_shard_basename_still_materializes(tmp_path, length):
    """The staging name must fit NAME_MAX whatever the shard is called.

    `mkstemp` builds `prefix` + 8 random characters. Carrying the whole basename in the
    prefix made the staging component exceed the filesystem's limit for a shard whose own
    name was near it: on ext4 a 230-character basename raised
    `OSError: [Errno 36] File name too long`, failing a shard that is valid and is exactly
    what materialisation exists to repair. Every one of these names is legal on the
    filesystem, which is the point: creating the shard succeeds, so materialising it must.
    """
    output = os.path.join(str(tmp_path), "out")
    os.makedirs(output, exist_ok = True)
    outside = os.path.join(str(tmp_path), "outside")
    os.makedirs(outside, exist_ok = True)
    victim = os.path.join(outside, "victim.safetensors")
    with open(victim, "wb") as f:
        f.write(b"VICTIM BYTES")

    suffix = ".safetensors"
    name = ("s" * (length - len(suffix))) + suffix
    # `os.pathconf` does not exist on Windows, the same reason the code under test
    # guards it; 255 is the component limit there too.
    try:
        name_max = os.pathconf(output, "PC_NAME_MAX")
    except (AttributeError, OSError, ValueError):
        name_max = 255
    if length > name_max:
        pytest.skip("this filesystem cannot hold a name that long")
    shard = os.path.join(output, name)
    try:
        os.symlink(victim, shard)
    except (OSError, NotImplementedError) as error:
        # Windows needs Developer Mode or admin rights to create a symlink.
        pytest.skip(f"cannot create a symlink here: {error}")

    saving_utils._materialize_shard_that_resolves_outside(shard, output)

    assert not os.path.islink(shard), f"a {length}-character shard was not materialized"
    with open(shard, "rb") as f:
        assert f.read() == b"VICTIM BYTES"
    with open(victim, "rb") as f:
        assert f.read() == b"VICTIM BYTES", "the link target was written through"
    leftovers = [n for n in os.listdir(output) if ".unsloth-materializing" in n]
    assert leftovers == [], f"staging files left behind: {leftovers}"


def test_a_linked_index_in_the_output_is_not_written_through(tmp_path):
    """The index was the one write left outside the containment the shards get.

    `open(destination, "wb")`, and `shutil.copy2` which uses it, follow a symlink sitting
    at the destination and truncate its TARGET. An output directory already carrying a
    linked `model.safetensors.index.json` therefore had that external file overwritten by
    the export. `os.replace` swaps the directory entry instead and never follows the last
    component, so the link is replaced rather than written through.
    """
    output = os.path.join(str(tmp_path), "out")
    os.makedirs(output, exist_ok = True)
    outside = os.path.join(str(tmp_path), "outside")
    os.makedirs(outside, exist_ok = True)

    victim = os.path.join(outside, "victim.json")
    with open(victim, "w", encoding = "utf-8") as f:
        f.write("VICTIM CONTENT")
    destination = os.path.join(output, "model.safetensors.index.json")
    os.symlink(victim, destination)

    source = os.path.join(str(tmp_path), "source.index.json")
    with open(source, "w", encoding = "utf-8") as f:
        json.dump({"metadata": {}, "weight_map": {"a": "model.safetensors"}}, f)
    os.chmod(source, 0o600)

    payload = b'{"metadata": {}, "weight_map": {"a": "model.safetensors"}}'
    saving_utils._export_index_atomically(source, destination, payload)

    with open(victim, encoding = "utf-8") as f:
        assert f.read() == "VICTIM CONTENT", "the link target was written through"
    assert not os.path.islink(destination), "the link was left in place of the index"
    with open(destination, "rb") as f:
        assert f.read() == payload
    # copystat still applies, and to the staging file, so the mode is never widened.
    assert oct(os.stat(destination).st_mode & 0o777) == oct(0o600)
    leftovers = [n for n in os.listdir(output) if n.startswith(".unsloth-index-")]
    assert leftovers == [], f"staging files left behind: {leftovers}"


def test_a_regenerated_index_does_not_follow_a_link_either(tmp_path):
    """Regeneration writes the index too, and used `open(path, "w")`.

    `_final_has_nested_shard` brings a dequantizing nested singleton down the
    regeneration arm, where the write followed a symlink at the destination and
    truncated its target, exactly as the copy path did before it was staged.
    """
    output = os.path.join(str(tmp_path), "out")
    os.makedirs(output, exist_ok = True)
    outside = os.path.join(str(tmp_path), "outside")
    os.makedirs(outside, exist_ok = True)
    victim = os.path.join(outside, "victim.json")
    with open(victim, "w", encoding = "utf-8") as f:
        f.write("VICTIM CONTENT")
    destination = os.path.join(output, "model.safetensors.index.json")
    os.symlink(victim, destination)

    payload = json.dumps(
        {"metadata": {}, "weight_map": {"a": "weights/model.safetensors"}}, indent = 4,
    ).encode("utf-8")
    saving_utils._export_index_atomically(None, destination, payload)

    with open(victim, encoding = "utf-8") as f:
        assert f.read() == "VICTIM CONTENT", "the link target was written through"
    assert not os.path.islink(destination)
    with open(destination, "rb") as f:
        assert json.loads(f.read()) == json.loads(payload)
    # A regenerated index has no source to copy a mode from, and must not land 0600:
    # `open(..., "w")` gave it the umask, and readers other than the owner need it.
    mode = os.stat(destination).st_mode & 0o777
    assert mode & 0o044, f"regenerated index landed unreadable: {oct(mode)}"
    assert [n for n in os.listdir(output) if n.startswith(".unsloth-index-")] == []
