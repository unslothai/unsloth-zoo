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

"""An unreachable Hub must not block a 16bit merge of a local FP8 base (`outputs/mymodel` reaches the probe):
_merge_and_overwrite_lora_fp8 dequantizes it offline; nf4/fp4 keep raising, as falling back there is a silent no-op."""

import json
import os
import warnings

import pytest
import requests
from huggingface_hub.errors import OfflineModeIsEnabled

from unsloth_zoo import saving_utils


FP8_CONFIG        = {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic"}
FBGEMM_FP8_CONFIG = {"quant_method": "fbgemm_fp8"}
NF4_CONFIG        = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4"}
FP4_CONFIG        = {"load_in_4bit": True, "bnb_4bit_quant_type": "fp4"}
BNB_CONFIG        = {"load_in_4bit": True}
MXFP4_CONFIG      = {"quant_method": "mxfp4"}


def _make_local_model(directory, quantization_config = None):
    """Writes the files the checks key off: a .safetensors for `check_local_model_exists` and config.json for `check_model_quantization_status`."""
    os.makedirs(directory, exist_ok = True)
    open(os.path.join(directory, "model.safetensors"), "wb").close()
    config = {"model_type": "llama"}
    if quantization_config is not None:
        config["quantization_config"] = quantization_config
    with open(os.path.join(directory, "config.json"), "w", encoding = "utf-8") as f:
        json.dump(config, f)
    return directory


def _hub_raises(monkeypatch, error):
    def fake_ls(self, path, detail = True, **kwargs):
        raise error
    monkeypatch.setattr(saving_utils.HfFileSystem, "ls", fake_ls, raising = True)


def _forbid_hub(monkeypatch):
    def no_network(self, path, detail = True, **kwargs):
        raise AssertionError(
            f"HfFileSystem.ls({path!r}) was called; this path must resolve "
            f"without probing the Hub"
        )
    monkeypatch.setattr(saving_utils.HfFileSystem, "ls", no_network, raising = True)


_TRANSPORT_ERRORS = [
    pytest.param(
        OfflineModeIsEnabled("Cannot reach https://huggingface.co: offline mode is enabled."),
        id = "offline-mode",
    ),
    pytest.param(requests.exceptions.ConnectionError("dns failure"), id = "dns-failure"),
    pytest.param(requests.exceptions.ReadTimeout("read timed out"),  id = "read-timeout"),
    pytest.param(requests.exceptions.ProxyError("proxy refused"),    id = "proxy-error"),
]


def _same_path(a, b):
    return os.path.realpath(str(a)) == os.path.realpath(str(b))


@pytest.mark.parametrize("error", _TRANSPORT_ERRORS)
@pytest.mark.parametrize("quant_config", [
    pytest.param(FP8_CONFIG,        id = "finegrained-fp8"),
    pytest.param(FBGEMM_FP8_CONFIG, id = "fbgemm-fp8"),
])
def test_local_fp8_resolves_when_the_hub_is_unreachable(monkeypatch, tmp_path, error, quant_config):
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "mymodel"), quant_config)
    _hub_raises(monkeypatch, error)

    name, is_local, source, is_quantized, quant_type = \
        saving_utils.determine_base_model_source("outputs/mymodel", save_method = "merged_16bit")

    assert _same_path(name, tmp_path / "outputs" / "mymodel")
    assert is_local is True
    assert source == "local_fp8"
    assert is_quantized is True
    assert quant_type == "fp8"


@pytest.mark.parametrize("error", _TRANSPORT_ERRORS)
def test_local_fp8_fallback_never_lands_on_the_silent_no_op(monkeypatch, tmp_path, error):
    """Only nf4/fp4 make the 16bit merge write nothing, so the quant type returned here must not be one of those."""
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "mymodel"), FP8_CONFIG)
    _hub_raises(monkeypatch, error)

    _, _, _, is_quantized, quant_type = saving_utils.determine_base_model_source(
        "outputs/mymodel", save_method = "merged_16bit",
    )
    assert is_quantized is True
    assert quant_type not in ("nf4", "fp4", "bitsandbytes")


@pytest.mark.parametrize("error", _TRANSPORT_ERRORS)
def test_local_fp8_fallback_emits_no_warning(monkeypatch, tmp_path, error):
    """A resolution that succeeds is not something to warn about."""
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "mymodel"), FP8_CONFIG)
    _hub_raises(monkeypatch, error)

    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        saving_utils.determine_base_model_source("outputs/mymodel", save_method = "merged_16bit")
    assert [str(w.message) for w in caught
            if issubclass(w.category, UserWarning)] == []


@pytest.mark.parametrize("error", _TRANSPORT_ERRORS)
@pytest.mark.parametrize("quant_config, label", [
    pytest.param(NF4_CONFIG, "nf4", id = "nf4"),
    pytest.param(FP4_CONFIG, "fp4", id = "fp4"),
    pytest.param(BNB_CONFIG, "bitsandbytes", id = "bitsandbytes"),
])
def test_local_4bit_still_raises_when_the_hub_is_unreachable(
    monkeypatch, tmp_path, error, quant_config, label,
):
    """Falling back here would put a 16bit merge onto the path that writes nothing."""
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "mymodel"), quant_config)
    _hub_raises(monkeypatch, error)

    with pytest.raises(RuntimeError) as excinfo:
        saving_utils.determine_base_model_source("outputs/mymodel")
    assert "connectivity" in str(excinfo.value)


@pytest.mark.parametrize("error", _TRANSPORT_ERRORS)
@pytest.mark.parametrize("save_method", [
    pytest.param(None,            id = "unknown"),
    pytest.param("mxfp4",         id = "mxfp4"),
    pytest.param("lora",          id = "lora"),
    pytest.param("merged_16bit_forced", id = "merged_16bit_forced"),
])
def test_local_fp8_only_falls_back_for_the_16bit_merge(
    monkeypatch, tmp_path, error, save_method,
):
    """FP8 is dequantized only for `merged_16bit`; other save methods write the stored FP8 tensor back, so the delta
    lands in scaled space without its companion scale; falling back would turn an outage into a wrong export."""
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "mymodel"), FP8_CONFIG)
    _hub_raises(monkeypatch, error)

    with pytest.raises(RuntimeError) as excinfo:
        saving_utils.determine_base_model_source("outputs/mymodel", save_method = save_method)
    assert "connectivity" in str(excinfo.value) or "rate limiting" in str(excinfo.value)


@pytest.mark.parametrize("save_method", ["merged_4bit", "forced_merged_4bit"])
def test_the_4bit_merges_still_fall_back_at_any_quantization(monkeypatch, tmp_path, save_method):
    """The 4bit merges read no base weights from disk, so nothing on disk can be missing for them and FP8 is not special to them."""
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "mymodel"), NF4_CONFIG)
    _hub_raises(monkeypatch, ConnectionError("dns failure"))

    _, is_local, source, _, quant_type = saving_utils.determine_base_model_source(
        "outputs/mymodel", save_method = save_method,
    )
    assert is_local is True
    assert source == "local_nf4"


def _shard(directory, name):
    open(os.path.join(directory, name), "wb").close()


def _write_index(directory, shards):
    index = {"weight_map": {f"layer{i}.weight": shard for i, shard in enumerate(shards)}}
    with open(os.path.join(directory, "model.safetensors.index.json"), "w", encoding = "utf-8") as f:
        json.dump(index, f)


@pytest.mark.parametrize("error", _TRANSPORT_ERRORS)
def test_a_partial_fp8_snapshot_does_not_satisfy_the_fallback(monkeypatch, tmp_path, error):
    """`check_local_model_exists` passes on the first `.safetensors`, so an interrupted download passes; the merge then
    rebuilds the index from present shards and reports success with missing layers."""
    monkeypatch.chdir(tmp_path)
    directory = _make_local_model(os.path.join("outputs", "mymodel"), FP8_CONFIG)
    os.remove(os.path.join(directory, "model.safetensors"))
    _shard(directory, "model-00001-of-00002.safetensors")
    _write_index(directory, ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"])
    _hub_raises(monkeypatch, error)

    with pytest.raises(RuntimeError) as excinfo:
        saving_utils.determine_base_model_source("outputs/mymodel", save_method = "merged_16bit")
    assert "connectivity" in str(excinfo.value) or "rate limiting" in str(excinfo.value)


def test_a_whole_sharded_fp8_snapshot_still_satisfies_the_fallback(monkeypatch, tmp_path):
    """The narrowing must cost nothing when every shard is there."""
    monkeypatch.chdir(tmp_path)
    directory = _make_local_model(os.path.join("outputs", "mymodel"), FP8_CONFIG)
    os.remove(os.path.join(directory, "model.safetensors"))
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    for shard in shards: _shard(directory, shard)
    _write_index(directory, shards)
    _hub_raises(monkeypatch, ConnectionError("dns failure"))

    _, is_local, source, _, quant_type = saving_utils.determine_base_model_source(
        "outputs/mymodel", save_method = "merged_16bit",
    )
    assert is_local is True
    assert source == "local_fp8"
    assert quant_type == "fp8"


def test_an_unsharded_fp8_snapshot_needs_no_index(monkeypatch, tmp_path):
    """No index means one file, which the caller has already seen."""
    monkeypatch.chdir(tmp_path)
    directory = _make_local_model(os.path.join("outputs", "mymodel"), FP8_CONFIG)
    assert saving_utils._local_snapshot_is_complete(directory) is True


@pytest.mark.parametrize("error", _TRANSPORT_ERRORS)
def test_a_shard_name_declares_the_total_even_with_no_index(monkeypatch, tmp_path, error):
    """A shard name such as `model-00001-of-00002` declares the set size, so a download interrupted before the index arrived is still caught."""
    monkeypatch.chdir(tmp_path)
    directory = _make_local_model(os.path.join("outputs", "mymodel"), FP8_CONFIG)
    os.remove(os.path.join(directory, "model.safetensors"))
    _shard(directory, "model-00001-of-00002.safetensors")
    _hub_raises(monkeypatch, error)

    with pytest.raises(RuntimeError) as excinfo:
        saving_utils.determine_base_model_source("outputs/mymodel", save_method = "merged_16bit")
    assert "connectivity" in str(excinfo.value) or "rate limiting" in str(excinfo.value)


def test_every_shard_present_needs_no_index(monkeypatch, tmp_path):
    """Once every named shard is present, a missing index alone is not a reason to refuse."""
    monkeypatch.chdir(tmp_path)
    directory = _make_local_model(os.path.join("outputs", "mymodel"), FP8_CONFIG)
    os.remove(os.path.join(directory, "model.safetensors"))
    for shard in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        _shard(directory, shard)
    _hub_raises(monkeypatch, ConnectionError("dns failure"))

    _, is_local, source, _, _ = saving_utils.determine_base_model_source(
        "outputs/mymodel", save_method = "merged_16bit",
    )
    assert is_local is True
    assert source == "local_fp8"


def test_shards_that_disagree_on_the_total_are_not_complete(tmp_path):
    """Two totals cannot both be right, and neither can be checked against the other."""
    directory = str(tmp_path)
    for shard in ("model-00001-of-00002.safetensors", "model-00001-of-00005.safetensors"):
        open(os.path.join(directory, shard), "wb").close()
    assert saving_utils._local_snapshot_is_complete(directory) is False


def test_an_index_naming_one_shard_of_five_is_not_complete(monkeypatch, tmp_path):
    """A stale index naming one of five shards still passes the file-exists check, so the shard totals in the filenames are checked too."""
    monkeypatch.chdir(tmp_path)
    directory = _make_local_model(os.path.join("outputs", "mymodel"), FP8_CONFIG)
    os.remove(os.path.join(directory, "model.safetensors"))
    _shard(directory, "model-00001-of-00005.safetensors")
    _write_index(directory, ["model-00001-of-00005.safetensors"])

    assert saving_utils._local_snapshot_is_complete(directory) is False
    _hub_raises(monkeypatch, ConnectionError("dns failure"))
    with pytest.raises(RuntimeError):
        saving_utils.determine_base_model_source("outputs/mymodel", save_method = "merged_16bit")


def test_an_index_over_a_whole_set_is_complete(monkeypatch, tmp_path):
    """The complement, so the name check cannot quietly refuse well formed snapshots."""
    monkeypatch.chdir(tmp_path)
    directory = _make_local_model(os.path.join("outputs", "mymodel"), FP8_CONFIG)
    os.remove(os.path.join(directory, "model.safetensors"))
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    for shard in shards: _shard(directory, shard)
    _write_index(directory, shards)
    assert saving_utils._local_snapshot_is_complete(directory) is True


def test_an_index_may_carry_paths_rather_than_bare_names(tmp_path):
    """`weight_map` values are compared as basenames, as the merge's stale shard filter does, so path-prefixed values still match."""
    directory = str(tmp_path)
    open(os.path.join(directory, "model-00001-of-00001.safetensors"), "wb").close()
    with open(os.path.join(directory, "model.safetensors.index.json"), "w", encoding = "utf-8") as f:
        json.dump({"weight_map": {"a.weight": "./model-00001-of-00001.safetensors"}}, f)
    assert saving_utils._local_snapshot_is_complete(directory) is True


def test_shards_from_different_sets_do_not_complete_each_other(tmp_path):
    """Shards from different sets (`model` and `backup`) sharing a total do not complete each other; `model-00002-of-00002` is still missing."""
    directory = str(tmp_path)
    for shard in ("model-00001-of-00002.safetensors", "backup-00002-of-00002.safetensors"):
        open(os.path.join(directory, shard), "wb").close()
    assert saving_utils._local_snapshot_is_complete(directory) is False


def test_a_partial_set_beside_a_whole_one_is_still_not_complete(tmp_path):
    """A partial set beside a whole one is still incomplete: with no index the merge reads every top-level
    `.safetensors`, and mismatched shapes raise the 'Bad in-place call' that the stale-shard filter exists to
    prevent."""
    directory = str(tmp_path)
    for shard in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                  "backup-00001-of-00002.safetensors"):
        open(os.path.join(directory, shard), "wb").close()
    assert saving_utils._local_snapshot_is_complete(directory) is False


@pytest.mark.parametrize("error", _TRANSPORT_ERRORS)
def test_an_unnumbered_stale_copy_is_ambiguous_too(monkeypatch, tmp_path, error):
    """Unnumbered stale copies are ambiguous too: `model.safetensors` beside a stale `backup.safetensors` is read in
    full and the index keeps whichever file was visited last."""
    monkeypatch.chdir(tmp_path)
    directory = _make_local_model(os.path.join("outputs", "mymodel"), FP8_CONFIG)
    _shard(directory, "backup.safetensors")
    _hub_raises(monkeypatch, error)

    assert saving_utils._local_snapshot_is_complete(directory) is False
    with pytest.raises(RuntimeError):
        saving_utils.determine_base_model_source("outputs/mymodel", save_method = "merged_16bit")


def _write_index_named(directory, names):
    index = {"weight_map": {f"layer{i}.weight": name for i, name in enumerate(names)}}
    with open(os.path.join(directory, "model.safetensors.index.json"), "w", encoding = "utf-8") as f:
        json.dump(index, f)


@pytest.mark.parametrize("error", _TRANSPORT_ERRORS)
def test_an_index_the_merge_will_not_honour_settles_nothing(monkeypatch, tmp_path, error):
    """An index the merge will not honour settles nothing: `consolidated.safetensors` is dropped beside a partial shard,
    and the empty index intersection keeps that shard."""
    monkeypatch.chdir(tmp_path)
    directory = _make_local_model(os.path.join("outputs", "mymodel"), FP8_CONFIG)
    os.remove(os.path.join(directory, "model.safetensors"))
    _shard(directory, "consolidated.safetensors")
    _shard(directory, "model-00001-of-00005.safetensors")
    _write_index_named(directory, ["consolidated.safetensors"])

    assert saving_utils._local_snapshot_is_complete(directory) is False
    _hub_raises(monkeypatch, error)
    with pytest.raises(RuntimeError):
        saving_utils.determine_base_model_source("outputs/mymodel", save_method = "merged_16bit")


def test_an_indexed_consolidated_file_on_its_own_is_honoured(tmp_path):
    """Nothing else is there for the merge to prefer, so it reads the file the index names."""
    directory = str(tmp_path)
    open(os.path.join(directory, "consolidated.safetensors"), "wb").close()
    _write_index_named(directory, ["consolidated.safetensors"])
    assert saving_utils._local_snapshot_is_complete(directory) is True


def test_an_index_still_filters_a_stale_extra(tmp_path):
    """When survivors intersect the index, the merge keeps only that intersection, so a stale `backup.safetensors` beside an indexed set is filtered out."""
    directory = str(tmp_path)
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    for name in shards + ["backup.safetensors"]:
        open(os.path.join(directory, name), "wb").close()
    _write_index_named(directory, shards)
    assert saving_utils._local_snapshot_is_complete(directory) is True


def test_an_unnumbered_extra_beside_a_whole_set_is_ambiguous(tmp_path):
    """The same, in the shape the numbered rules alone did not see."""
    directory = str(tmp_path)
    for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                 "backup.safetensors"):
        open(os.path.join(directory, name), "wb").close()
    assert saving_utils._local_snapshot_is_complete(directory) is False


def test_a_consolidated_duplicate_beside_its_shards_is_still_one_snapshot(tmp_path):
    """Mistral-7B-v0.3, Codestral, Nemo and Small ship a duplicate consolidated.safetensors that the merge drops when
    proper shards coexist; counting it would call those repos incomplete."""
    directory = str(tmp_path)
    for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                 "consolidated.safetensors"):
        open(os.path.join(directory, name), "wb").close()
    assert saving_utils._local_snapshot_is_complete(directory) is True


def test_a_lone_consolidated_file_is_one_snapshot(tmp_path):
    """The merge keeps it when it is the only file, so this is a base it can read."""
    directory = str(tmp_path)
    open(os.path.join(directory, "consolidated.safetensors"), "wb").close()
    assert saving_utils._local_snapshot_is_complete(directory) is True


def test_files_that_are_not_safetensors_are_not_extras(tmp_path):
    """config.json and the tokenizer are not weights and the merge never reads them as such."""
    directory = str(tmp_path)
    for name in ("model.safetensors", "config.json", "tokenizer.json"):
        open(os.path.join(directory, name), "wb").close()
    assert saving_utils._local_snapshot_is_complete(directory) is True


def test_two_whole_sets_are_ambiguous_rather_than_complete(tmp_path):
    """Two complete sets are worse than one complete plus one partial because nothing fails: both share tensor keys, and
    the index is regenerated from every top-level file, so the export can silently be the stale copy."""
    directory = str(tmp_path)
    for shard in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                  "backup-00001-of-00002.safetensors", "backup-00002-of-00002.safetensors"):
        open(os.path.join(directory, shard), "wb").close()
    assert saving_utils._local_snapshot_is_complete(directory) is False


def test_an_unreadable_index_is_not_proof_of_completeness(tmp_path):
    """Unproven completeness answers False: the unreachable Hub is the honest failure."""
    directory = str(tmp_path)
    with open(os.path.join(directory, "model.safetensors.index.json"), "w", encoding = "utf-8") as f:
        f.write("{not json")
    assert saving_utils._local_snapshot_is_complete(directory) is False


@pytest.mark.parametrize("error", _TRANSPORT_ERRORS)
def test_a_partial_snapshot_still_serves_the_4bit_merges(monkeypatch, tmp_path, error):
    """Completeness is needed only where base weights are read; 4bit merges take them from memory, so refusing a partial snapshot would drop a working fallback."""
    monkeypatch.chdir(tmp_path)
    directory = _make_local_model(os.path.join("outputs", "mymodel"), NF4_CONFIG)
    _write_index(directory, ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"])
    _hub_raises(monkeypatch, error)

    _, is_local, source, _, _ = saving_utils.determine_base_model_source(
        "outputs/mymodel", save_method = "merged_4bit",
    )
    assert is_local is True
    assert source == "local_nf4"


def test_reachable_hub_16bit_repo_still_outranks_the_local_fp8_copy(monkeypatch, tmp_path):
    """Catching the failure rather than hoisting priority 5 preserves this: with the Hub up, the 16bit repo still wins at priority 3."""
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "mymodel"), FP8_CONFIG)

    def fake_ls(self, path, detail = True, **kwargs):
        return [{"name": f"{path}/model.safetensors"}]
    monkeypatch.setattr(saving_utils.HfFileSystem, "ls", fake_ls, raising = True)
    monkeypatch.setattr(
        saving_utils, "check_model_quantization_status",
        # The local copy is looked up by absolute path and the Hub by the name as given, which tells the calls apart;
        # **kwargs is needed since the Hub call passes `local_ok = False` for a name that is also a directory.
        lambda name, token = None, **kwargs: (True, "fp8") if os.path.isabs(str(name)) else (False, None),
    )

    name, is_local, source, is_quantized, quant_type = \
        saving_utils.determine_base_model_source("outputs/mymodel")
    assert name == "outputs/mymodel"
    assert is_local is False
    assert source == "HF_unquantized"


def test_reachable_hub_absent_repo_still_falls_to_the_local_fp8_copy(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "mymodel"), FP8_CONFIG)
    # fsspec converts RepositoryNotFoundError into a plain FileNotFoundError.
    _hub_raises(monkeypatch, FileNotFoundError("outputs/mymodel (repository not found)"))

    name, is_local, source, _, quant_type = \
        saving_utils.determine_base_model_source("outputs/mymodel")
    assert _same_path(name, tmp_path / "outputs" / "mymodel")
    assert source == "local_fp8"
    assert quant_type == "fp8"


@pytest.mark.parametrize("shape", ["absolute", "dot-relative", "deep-relative"])
def test_path_shaped_local_fp8_never_probes_the_hub_at_all(monkeypatch, tmp_path, shape):
    """The string gate already spares path-shaped names, so the new fallback is reached only by the repo-id shaped case that needs it."""
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "nested", "mymodel"), FP8_CONFIG)
    given = {
        "absolute":      str(tmp_path / "outputs" / "nested" / "mymodel"),
        "dot-relative":  os.path.join(".", "outputs", "nested", "mymodel"),
        "deep-relative": os.path.join("outputs", "nested", "mymodel"),
    }[shape]
    _forbid_hub(monkeypatch)

    name, is_local, source, _, quant_type = saving_utils.determine_base_model_source(given)
    assert _same_path(name, tmp_path / "outputs" / "nested" / "mymodel")
    assert is_local is True
    assert source == "local_fp8"
    assert quant_type == "fp8"


def test_local_mxfp4_still_resolves_before_the_probe(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "mymodel"), MXFP4_CONFIG)
    _forbid_hub(monkeypatch)
    _, _, source, _, quant_type = saving_utils.determine_base_model_source("outputs/mymodel")
    assert source == "local_mxfp4"
    assert quant_type == "mxfp4"


@pytest.mark.parametrize("error", _TRANSPORT_ERRORS)
def test_sibling_lookup_says_so_when_the_hub_is_unreachable(monkeypatch, tmp_path, error):
    """None stays right since the merge can still dequantize FP8; silently falling back is not, as the base used differs from what a reachable Hub would choose."""
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "GLM-5.2-FP8"), FP8_CONFIG)
    _hub_raises(monkeypatch, error)

    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        assert saving_utils._resolve_fp8_16bit_sibling("outputs/GLM-5.2-FP8") is None
    messages = [str(w.message) for w in caught]
    assert any("16bit sibling" in m for m in messages), messages
    assert any("FP8 weights instead" in m for m in messages), messages


def test_sibling_lookup_is_silent_and_network_free_when_a_local_sibling_exists(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "GLM-5.2-FP8"), FP8_CONFIG)
    _make_local_model(os.path.join("outputs", "GLM-5.2"), None)
    _forbid_hub(monkeypatch)

    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        sibling = saving_utils._resolve_fp8_16bit_sibling("outputs/GLM-5.2-FP8")
    assert _same_path(sibling, tmp_path / "outputs" / "GLM-5.2")
    assert [str(w.message) for w in caught
            if issubclass(w.category, UserWarning)] == []


def test_sibling_lookup_stays_silent_for_a_genuinely_absent_sibling(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "GLM-5.2-FP8"), FP8_CONFIG)
    # fsspec converts RepositoryNotFoundError into a plain FileNotFoundError.
    _hub_raises(monkeypatch, FileNotFoundError("outputs/mymodel (repository not found)"))

    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        assert saving_utils._resolve_fp8_16bit_sibling("outputs/GLM-5.2-FP8") is None
    assert [str(w.message) for w in caught
            if issubclass(w.category, UserWarning)] == []


def test_the_sibling_lookup_keeps_asking_the_hub_after_an_unreadable_local_config(
    monkeypatch, tmp_path,
):
    """An unreadable local config.json says nothing about the Hub: skip that candidate and keep asking the Hub, or FP8
    would be dequantized while a full-precision sibling sits on the Hub."""
    monkeypatch.chdir(tmp_path)
    # `unsloth/GLM-Air-FP8` strips to `unsloth/GLM-Air`, which also exists locally, broken.
    local_sibling = os.path.join("unsloth", "GLM-Air")
    os.makedirs(local_sibling, exist_ok = True)
    open(os.path.join(local_sibling, "model.safetensors"), "wb").close()
    with open(os.path.join(local_sibling, "config.json"), "w", encoding = "utf-8") as f:
        f.write("{ this is not json")

    def fake_ls(self, path, detail = True, **kwargs):
        return [{"name": f"{path}/model.safetensors"}]
    monkeypatch.setattr(saving_utils.HfFileSystem, "ls", fake_ls, raising = True)
    # Stubs only the download so the real classifier runs on both sides: the broken local dir raises while the Hub
    # serves a readable config; without the local skip, the broken dir answers for the Hub too.
    good_config = tmp_path / "hub-config.json"
    good_config.write_text(json.dumps({"model_type": "llama"}), encoding = "utf-8")
    import huggingface_hub
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download",
        lambda *args, **kwargs: str(good_config), raising = True,
    )
    # The sibling is also HEADed anonymously to prove it is readable without a token; stub that too or the lookup asks the real Hub.
    monkeypatch.setattr(
        huggingface_hub, "get_hf_file_metadata",
        lambda *args, **kwargs: None, raising = True,
    )

    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        sibling = saving_utils._resolve_fp8_16bit_sibling("unsloth/GLM-Air-FP8")
    assert sibling == "unsloth/GLM-Air"
    assert [str(w.message) for w in caught
            if "could not check the Hugging Face Hub" in str(w.message)] == [], (
        "a local config problem must not be reported as an unreachable Hub"
    )
