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

"""An unreachable Hub must not block a 16bit merge of a local FP8 base.

`outputs/mymodel` survives the string gate in
`test_local_base_model_resolution_offline.py`, being both a valid repo id and an
ordinary directory, and an FP8 directory is neither priority 1 nor 2, so it reaches
the probe. With the Hub down the probe raised and priority 5 was never consulted, even
though `merge_and_overwrite_lora` dequantizes an FP8 base for `merged_16bit` via
`_merge_and_overwrite_lora_fp8` and so needs nothing from the network.

Measured on huggingface_hub 1.24.0 with `outputs/fp8` on disk and `ls` raising
OfflineModeIsEnabled:

    local unquantized   ('.../outputs/unquantized', True, 'local_unquantized', ...)
    local mxfp4         ('.../outputs/mxfp4',       True, 'local_mxfp4', ...)
    local fp8           RuntimeError                <- the regression
    local nf4           RuntimeError                <- correct, see below

nf4/fp4 deliberately keeps raising: there the merge writes nothing, so falling back
would trade a loud failure for the silent no-op. Falling back is not automatically the
safer choice.
"""

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
    """`check_local_model_exists` keys off a `.safetensors` file,
    `check_model_quantization_status` off config.json."""
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


# The regression: a repo-id shaped local FP8 directory with the Hub down.

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
    """The merge writes nothing only for nf4/fp4 on a 16bit merge, so the quant type
    handed back here must not be one of those."""
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


# Everything the merge cannot complete offline must keep raising.

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
    pytest.param("merged_16bit_forced", id = "merged_16bit_forced"),
])
def test_local_fp8_only_falls_back_for_the_16bit_merge(
    monkeypatch, tmp_path, error, save_method,
):
    """FP8 is dequantized only on the `merged_16bit` path. Every other save method reaches
    the in place writer, which takes the stored tensor with `file.get_tensor(key)` and
    writes it back at its FP8 dtype, so the delta lands in scaled space without the
    companion scale. Falling back there would turn an outage into a wrong export, which is
    worse than the no-op this branch removes, so it keeps raising."""
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "mymodel"), FP8_CONFIG)
    _hub_raises(monkeypatch, error)

    with pytest.raises(RuntimeError) as excinfo:
        saving_utils.determine_base_model_source("outputs/mymodel", save_method = save_method)
    assert "connectivity" in str(excinfo.value) or "rate limiting" in str(excinfo.value)


@pytest.mark.parametrize("save_method", ["merged_4bit", "forced_merged_4bit"])
def test_the_4bit_merges_still_fall_back_at_any_quantization(monkeypatch, tmp_path, save_method):
    """The complement of the narrowing: these read no base weights at all, so nothing on
    disk can be missing and FP8 is not special to them."""
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "mymodel"), NF4_CONFIG)
    _hub_raises(monkeypatch, ConnectionError("dns failure"))

    _, is_local, source, _, quant_type = saving_utils.determine_base_model_source(
        "outputs/mymodel", save_method = save_method,
    )
    assert is_local is True
    assert source == "local_nf4"


# A half downloaded snapshot is not a base the merge can read.

def _shard(directory, name):
    open(os.path.join(directory, name), "wb").close()


def _write_index(directory, shards):
    index = {"weight_map": {f"layer{i}.weight": shard for i, shard in enumerate(shards)}}
    with open(os.path.join(directory, "model.safetensors.index.json"), "w", encoding = "utf-8") as f:
        json.dump(index, f)


@pytest.mark.parametrize("error", _TRANSPORT_ERRORS)
def test_a_partial_fp8_snapshot_does_not_satisfy_the_fallback(monkeypatch, tmp_path, error):
    """`check_local_model_exists` answers on the first `.safetensors`, so an interrupted
    download passes it. The merge would then rebuild the index from the shards present and
    write a base missing layers, reported as a success."""
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
    """The index is not the only thing that knows: `model-00001-of-00002.safetensors` says
    one more should be here, and a download interrupted before the index arrived is exactly
    the case an index check cannot see."""
    monkeypatch.chdir(tmp_path)
    directory = _make_local_model(os.path.join("outputs", "mymodel"), FP8_CONFIG)
    os.remove(os.path.join(directory, "model.safetensors"))
    _shard(directory, "model-00001-of-00002.safetensors")
    _hub_raises(monkeypatch, error)

    with pytest.raises(RuntimeError) as excinfo:
        saving_utils.determine_base_model_source("outputs/mymodel", save_method = "merged_16bit")
    assert "connectivity" in str(excinfo.value) or "rate limiting" in str(excinfo.value)


def test_every_shard_present_needs_no_index(monkeypatch, tmp_path):
    """The complement: the naming is satisfied, so the missing index is not itself a
    reason to refuse."""
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
    """A stale or half rebuilt index can name a single shard whose own filename says four
    more belong to it. Every file the index names exists, and the snapshot is still partial,
    so the names it gives are checked as names too."""
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
    """`weight_map` values are compared as basenames, the way the merge's own stale shard
    filter compares them."""
    directory = str(tmp_path)
    open(os.path.join(directory, "model-00001-of-00001.safetensors"), "wb").close()
    with open(os.path.join(directory, "model.safetensors.index.json"), "w", encoding = "utf-8") as f:
        json.dump({"weight_map": {"a.weight": "./model-00001-of-00001.safetensors"}}, f)
    assert saving_utils._local_snapshot_is_complete(directory) is True


def test_shards_from_different_sets_do_not_complete_each_other(tmp_path):
    """One shard of `model` and one stale shard of `backup` declare the same total and are
    not the same set, so `model-00002-of-00002` is still missing."""
    directory = str(tmp_path)
    for shard in ("model-00001-of-00002.safetensors", "backup-00002-of-00002.safetensors"):
        open(os.path.join(directory, shard), "wb").close()
    assert saving_utils._local_snapshot_is_complete(directory) is False


def test_a_partial_set_beside_a_whole_one_is_still_not_complete(tmp_path):
    """One whole `model` set is not enough while a partial `backup` set sits next to it. The
    merge takes every top-level `.safetensors` in the directory and filters stale shards only
    against an index, so with no index the leftover is read too, and mismatched shapes are
    the "Bad in-place call" that filter exists for."""
    directory = str(tmp_path)
    for shard in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                  "backup-00001-of-00002.safetensors"):
        open(os.path.join(directory, shard), "wb").close()
    assert saving_utils._local_snapshot_is_complete(directory) is False


def test_two_whole_sets_are_ambiguous_rather_than_complete(tmp_path):
    """Two complete sets are worse than one complete and one partial, because nothing fails.
    Both carry the same tensor keys, the merge reads every top-level `.safetensors` and
    regenerates the index from them, so each key ends up pointing at whichever file was
    visited last and the export can quietly be the stale copy."""
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
    """Completeness is required only where base weights are read. The 4bit merges take
    them from memory, so a missing shard is not their problem and refusing there would
    withdraw a fallback that works."""
    monkeypatch.chdir(tmp_path)
    directory = _make_local_model(os.path.join("outputs", "mymodel"), NF4_CONFIG)
    _write_index(directory, ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"])
    _hub_raises(monkeypatch, error)

    _, is_local, source, _, _ = saving_utils.determine_base_model_source(
        "outputs/mymodel", save_method = "merged_4bit",
    )
    assert is_local is True
    assert source == "local_nf4"


@pytest.mark.parametrize("error", _TRANSPORT_ERRORS)
def test_no_local_copy_still_raises(monkeypatch, tmp_path, error):
    monkeypatch.chdir(tmp_path)
    _hub_raises(monkeypatch, error)
    with pytest.raises(RuntimeError):
        saving_utils.determine_base_model_source("unsloth/does-not-exist")


# A reachable Hub stays authoritative: nothing that resolved before changed.

def test_reachable_hub_16bit_repo_still_outranks_the_local_fp8_copy(monkeypatch, tmp_path):
    """Catching the failure rather than hoisting priority 5 is what preserves this: with
    the Hub up, the 16bit repo still wins at priority 3."""
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "mymodel"), FP8_CONFIG)

    def fake_ls(self, path, detail = True, **kwargs):
        return [{"name": f"{path}/model.safetensors"}]
    monkeypatch.setattr(saving_utils.HfFileSystem, "ls", fake_ls, raising = True)
    monkeypatch.setattr(
        saving_utils, "check_model_quantization_status",
        # The local copy is consulted by resolved absolute path, the Hub by the name as
        # given, which is what tells the two calls apart.
        lambda name, token = None: (True, "fp8") if os.path.isabs(str(name)) else (False, None),
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
    """The string gate already spares these, so the new fallback is only ever reached by
    the repo-id shaped case that needs it."""
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


# The FP8 16bit sibling lookup must not report "no sibling" for a Hub it could not
# reach.

@pytest.mark.parametrize("error", _TRANSPORT_ERRORS)
def test_sibling_lookup_says_so_when_the_hub_is_unreachable(monkeypatch, tmp_path, error):
    """None stays right, because the merge can still dequantize the FP8 weights, but
    silently is not: the base used is not the one a reachable Hub would have chosen."""
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
    """A local candidate whose config.json cannot be read cannot be classified, and that is
    not an answer about the Hub. Giving up there would dequantize the FP8 weights while a
    full precision sibling sat on the Hub, so the local candidate is skipped and the lookup
    continues."""
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
    # Stubbed at the download so the real classifier runs on both sides: the directory
    # reaches the real parser and raises, the Hub serves a readable config for the same
    # name. Without the local skip, the broken directory answers for the Hub too.
    good_config = tmp_path / "hub-config.json"
    good_config.write_text(json.dumps({"model_type": "llama"}), encoding = "utf-8")
    import huggingface_hub
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download",
        lambda *args, **kwargs: str(good_config), raising = True,
    )

    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        sibling = saving_utils._resolve_fp8_16bit_sibling("unsloth/GLM-Air-FP8")
    assert sibling == "unsloth/GLM-Air"
    assert [str(w.message) for w in caught
            if "could not check the Hugging Face Hub" in str(w.message)] == [], (
        "a local config problem must not be reported as an unreachable Hub"
    )
