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

"""Gated and offline Hub config errors: HfFileSystem.ls wraps GatedRepoError in a FileNotFoundError that
_HUB_ABSENT_ERRORS swallows (no gated warning); offline LocalEntryNotFoundError was swallowed too, giving
HF_unquantized for nf4 bases, skipping the nf4/fp4 guard; nothing touches the network."""

import json
import os
import warnings

import huggingface_hub
import pytest
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    RepositoryNotFoundError,
)

from unsloth_zoo import saving_utils


_REPO = "ns/base-bnb-4bit"
_NF4 = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4"}

# Same shape as test_quant_status_transport_errors.py: the Hub branch is reached only when os.path.exists(name) is False while check_local_model_exists hits.
_REQUESTED = "Outputs/MyModel"
_ON_DISK = ("outputs", "mymodel")


def _require_a_case_sensitive_filesystem(tmp_path):
    """Probed, not assumed: only a case difference makes os.path.exists miss while check_local_model_exists hits; on
    case-insensitive APFS/NTFS, macos-14 and windows-latest fail without the skip."""
    probe = tmp_path / "case_probe"
    probe.mkdir(exist_ok = True)
    if os.path.exists(str(tmp_path / "CASE_PROBE")):
        pytest.skip(
            "case insensitive filesystem: os.path.exists would resolve "
            f"{_REQUESTED!r} locally and skip the Hub branch under test"
        )
    pass


class _StubResponse:
    """The attributes `HfHubHTTPError.__init__` actually reads on 1.x."""
    status_code = 403
    reason      = "Forbidden"
    url         = "https://huggingface.co/api/models/test"
    headers     = {}
    content     = b""
    text        = ""
    request     = None


def _hub_error(cls, message):
    """Detects by signature, not version: response is keyword-only and required from 1.0 for HTTP-backed Hub error classes, and absent on 0.x."""
    try:
        return cls(message)
    except TypeError:
        return cls(message, response = _StubResponse())


def _gated():
    return _hub_error(GatedRepoError, "403 Client Error. Cannot access gated repo")


def _local_entry_not_found():
    """What hf_hub_download raises when the Hub is unreachable and the file is not cached; the test skips, not fails, if a release drops the class."""
    errors = pytest.importorskip("huggingface_hub.errors")
    cls = getattr(errors, "LocalEntryNotFoundError", None)
    if cls is None:
        pytest.skip("huggingface_hub has no LocalEntryNotFoundError")
    return _hub_error(
        cls,
        "An error happened while trying to locate the file on the Hub and we "
        "cannot find the requested files in the local cache.",
    )


def _patch_ls_present(monkeypatch):
    """The existence probe succeeds; everything here is about the round trip after."""
    def fake_ls(self, path, detail = True, **kwargs):
        return [{"name": f"{path}/model.safetensors"}]
    monkeypatch.setattr(saving_utils.HfFileSystem, "ls", fake_ls, raising = True)


def _patch_config_fetch(monkeypatch, side_effect):
    def fake_download(*args, **kwargs):
        raise side_effect
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download, raising = True)


def _patch_repo_info(monkeypatch, side_effect):
    """Fail underneath the *real* `ls`, so its own error conversion is under test."""
    def fake_repo_info(self, *args, **kwargs):
        raise side_effect
    monkeypatch.setattr(huggingface_hub.HfApi, "repo_info", fake_repo_info, raising = True)


def _make_local_model(directory, quant_config = None):
    directory.mkdir(parents = True, exist_ok = True)
    (directory / "model.safetensors").write_bytes(b"")
    config = {"model_type": "llama"}
    if quant_config is not None:
        config["quantization_config"] = quant_config
    (directory / "config.json").write_text(json.dumps(config), encoding = "utf-8")
    return directory


def _gated_warnings(messages):
    return [m for m in messages if "gated on the Hugging Face Hub" in m]


# 1. The gated message has to survive the Hub's own error conversion.

def test_gated_repo_warns_through_the_real_hf_file_system(monkeypatch):
    """The production shape: real `ls`, gated `repo_info` underneath it."""
    _patch_repo_info(monkeypatch, _gated())
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        result = saving_utils.check_hf_model_exists("meta-llama/Llama-3.2-1B")
    messages = [str(w.message) for w in caught]

    # False is unchanged: a local copy must still win.
    assert result is False
    assert _gated_warnings(messages), (
        f"a gated repo produced no gated warning; `ls` converts GatedRepoError to "
        f"FileNotFoundError so matching on the type alone never fires. Got {messages}"
    )


def test_the_real_ls_really_does_hide_the_gated_error(monkeypatch):
    """Pins the upstream behaviour the fix depends on: ls raises FileNotFoundError with the GatedRepoError as __cause__,
    so a change fails here, not as a dead clause."""
    _patch_repo_info(monkeypatch, _gated())
    with pytest.raises(FileNotFoundError) as excinfo:
        huggingface_hub.HfFileSystem().ls("meta-llama/Llama-3.2-1B", detail = True)
    assert isinstance(excinfo.value.__cause__, GatedRepoError)


def test_a_directly_raised_gated_error_still_warns(monkeypatch):
    """The other shape stays covered: not every caller is `HfFileSystem.ls`."""
    def fake_ls(self, path, detail = True, **kwargs):
        raise _gated()
    monkeypatch.setattr(saving_utils.HfFileSystem, "ls", fake_ls, raising = True)
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        assert saving_utils.check_hf_model_exists(_REPO) is False
    assert _gated_warnings([str(w.message) for w in caught])


def test_a_genuinely_absent_repo_does_not_claim_to_be_gated(monkeypatch):
    """Control. The unwrap must not turn every absent repo into a licence problem."""
    _patch_repo_info(monkeypatch, _hub_error(RepositoryNotFoundError, "404 repo not found"))
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        assert saving_utils.check_hf_model_exists("ns/definitely-not-there") is False
    assert not _gated_warnings([str(w.message) for w in caught])


# 2. An unreadable config is never an unquantized model.

def test_offline_config_download_raises_instead_of_reporting_unquantized(monkeypatch):
    _patch_ls_present(monkeypatch)
    _patch_config_fetch(monkeypatch, _local_entry_not_found())
    with pytest.raises(RuntimeError) as excinfo:
        saving_utils.check_model_quantization_status(_REPO)
    assert "not an unquantized model" in str(excinfo.value)


def test_offline_config_download_never_selects_hf_unquantized(monkeypatch, tmp_path):
    """Offline config download must never select HF_unquantized for an nf4 base: that skips the nf4/fp4 guard and merges 16bit against quantized weights."""
    monkeypatch.chdir(tmp_path)
    _patch_ls_present(monkeypatch)
    _patch_config_fetch(monkeypatch, _local_entry_not_found())
    with pytest.raises(RuntimeError):
        saving_utils.determine_base_model_source(_REPO, None, "merged_16bit")


@pytest.mark.parametrize("save_method", [
    "merged_4bit", "forced_merged_4bit", "merged_16bit", None,
])
def test_an_unreadable_local_config_raises_for_every_save_method(
    monkeypatch, tmp_path, save_method,
):
    """An unreadable base config must raise for every save method: guessing unquantized for an mxfp4 base makes
    _merge_and_overwrite_lora take the in place writer, not the full rewrite."""
    directory = tmp_path / "base"
    directory.mkdir()
    (directory / "model.safetensors").write_bytes(b"")
    (directory / "config.json").write_text("{ not json", encoding = "utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError) as excinfo:
        saving_utils.determine_base_model_source("base", None, save_method)

    message = str(excinfo.value)
    assert "could not read the quantization config" in message, message
    assert "config.json" in message, "the raise has to name the thing to repair"


def test_gated_config_download_warns_and_raises(monkeypatch):
    """A gated `config.json` is unread, not unquantized, and the user gets told why."""
    _patch_ls_present(monkeypatch)
    _patch_config_fetch(monkeypatch, _gated())
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError):
            saving_utils.check_model_quantization_status(_REPO)
    assert _gated_warnings([str(w.message) for w in caught])


def test_a_config_error_chained_behind_a_generic_one_is_still_gated(monkeypatch):
    """The download path gets the same `__cause__` unwrap as the probe."""
    chained = FileNotFoundError("ns/base-bnb-4bit (repository not found)")
    chained.__cause__ = _gated()
    _patch_ls_present(monkeypatch)
    _patch_config_fetch(monkeypatch, chained)
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError):
            saving_utils.check_model_quantization_status(_REPO)
    assert _gated_warnings([str(w.message) for w in caught])


# Controls: the absent cases must not start raising.

def test_a_repo_without_a_config_json_still_reports_unquantized(monkeypatch):
    """`EntryNotFoundError` is a fact about the repo, not about the network."""
    _patch_ls_present(monkeypatch)
    _patch_config_fetch(monkeypatch, _hub_error(EntryNotFoundError, "404 no config.json"))
    assert saving_utils.check_model_quantization_status(_REPO) == (False, None)


def test_an_absent_repo_still_reports_unquantized(monkeypatch):
    _patch_ls_present(monkeypatch)
    _patch_config_fetch(monkeypatch, _hub_error(RepositoryNotFoundError, "404 repo not found"))
    assert saving_utils.check_model_quantization_status(_REPO) == (False, None)


def test_a_reachable_hub_still_detects_quantization(monkeypatch, tmp_path):
    """The happy path is untouched: a real config still classifies as nf4."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"quantization_config": _NF4}), encoding = "utf-8")
    _patch_ls_present(monkeypatch)
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download",
        lambda *a, **k: str(config_path), raising = True,
    )
    is_quantized, config = saving_utils.check_model_quantization_status(_REPO)
    assert is_quantized is True
    assert config is not None


@pytest.mark.parametrize("save_method", ["merged_4bit", "forced_merged_4bit"])
def test_a_local_4bit_copy_still_merges_when_the_config_fetch_is_offline(
    monkeypatch, tmp_path, save_method,
):
    """The raise must reach the local fallback like a 429 does; both 4bit merges fold LoRA into weights already in memory."""
    _require_a_case_sensitive_filesystem(tmp_path)
    directory = _make_local_model(tmp_path.joinpath(*_ON_DISK), quant_config = _NF4)
    monkeypatch.chdir(tmp_path)
    _patch_ls_present(monkeypatch)
    _patch_config_fetch(monkeypatch, _local_entry_not_found())

    resolved = saving_utils.determine_base_model_source(_REQUESTED, None, save_method)
    assert os.path.realpath(resolved[0]) == os.path.realpath(str(directory))
    assert resolved[1] is True
    assert resolved[2] == "local_nf4"


def test_an_exactly_named_local_directory_never_reaches_the_config_fetch(
    monkeypatch, tmp_path,
):
    """When the requested name is the directory itself, the config is read off disk and no socket is opened."""
    directory = _make_local_model(tmp_path / "mymodel", quant_config = _NF4)
    _patch_ls_present(monkeypatch)

    def explode(*args, **kwargs):
        raise AssertionError("the config fetch must not be reached for a local directory")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", explode, raising = True)

    is_quantized, config = saving_utils.check_model_quantization_status(str(directory))
    assert is_quantized is True
    assert config is not None


# An unreadable local config is not an answer about the Hub.

def test_a_healthy_hub_base_is_used_when_the_local_config_cannot_be_read(monkeypatch, tmp_path):
    """An unreadable local config says nothing about the Hub: the same name resolves there to a readable base, so the Hub base is used rather than raising."""
    _require_a_case_sensitive_filesystem(tmp_path)
    monkeypatch.chdir(tmp_path)
    directory = os.path.join("outputs", "mymodel")
    os.makedirs(directory, exist_ok = True)
    open(os.path.join(directory, "model.safetensors"), "wb").close()
    with open(os.path.join(directory, "config.json"), "w", encoding = "utf-8") as f:
        f.write("{ truncated mid write")

    def fake_ls(self, path, detail = True, **kwargs):
        return [{"name": f"{path}/model.safetensors"}]
    monkeypatch.setattr(saving_utils.HfFileSystem, "ls", fake_ls, raising = True)
    # Hub serves a readable config for the same name; stubbed at the download, not at check_model_quantization_status,
    # so the directory still reaches the real raise.
    good_config = tmp_path / "hub-config.json"
    good_config.write_text(json.dumps({"model_type": "llama"}), encoding = "utf-8")
    import huggingface_hub
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download",
        lambda *args, **kwargs: str(good_config), raising = True,
    )

    name, is_local, source, is_quantized, quant_type = \
        saving_utils.determine_base_model_source("outputs/mymodel")
    assert name == "outputs/mymodel"
    assert is_local is False
    assert source == "HF_unquantized"


def test_the_local_parse_error_is_what_surfaces_when_the_hub_has_nothing(monkeypatch, tmp_path):
    """When the Hub has nothing, surface the local parse error: 'not found locally or on Hugging Face' would be false, since the directory exists but is unreadable."""
    _require_a_case_sensitive_filesystem(tmp_path)
    monkeypatch.chdir(tmp_path)
    directory = os.path.join("outputs", "mymodel")
    os.makedirs(directory, exist_ok = True)
    open(os.path.join(directory, "model.safetensors"), "wb").close()
    with open(os.path.join(directory, "config.json"), "w", encoding = "utf-8") as f:
        f.write("{ truncated mid write")

    def absent(self, path, detail = True, **kwargs):
        raise FileNotFoundError(f"{path} (repository not found)")
    monkeypatch.setattr(saving_utils.HfFileSystem, "ls", absent, raising = True)

    with pytest.raises(RuntimeError) as excinfo:
        saving_utils.determine_base_model_source("outputs/mymodel")
    message = str(excinfo.value)
    assert "config.json" in message, message
    assert "not found locally or on Hugging Face" not in message


def test_an_unreachable_hub_still_wins_over_an_unreadable_local_config(monkeypatch, tmp_path):
    """Two failures occurred and only the transport one is retryable; it stopped the resolution, so it stays the one reported."""
    _require_a_case_sensitive_filesystem(tmp_path)
    monkeypatch.chdir(tmp_path)
    directory = os.path.join("outputs", "mymodel")
    os.makedirs(directory, exist_ok = True)
    open(os.path.join(directory, "model.safetensors"), "wb").close()
    with open(os.path.join(directory, "config.json"), "w", encoding = "utf-8") as f:
        f.write("{ truncated mid write")

    def unreachable(self, path, detail = True, **kwargs):
        raise ConnectionError("Temporary failure in name resolution")
    monkeypatch.setattr(saving_utils.HfFileSystem, "ls", unreachable, raising = True)

    with pytest.raises(RuntimeError) as excinfo:
        saving_utils.determine_base_model_source("outputs/mymodel", save_method = "merged_16bit")
    assert "connectivity" in str(excinfo.value) or "rate limiting" in str(excinfo.value)


def test_a_local_quantized_copy_does_not_answer_for_an_unquantized_hub_repo(monkeypatch, tmp_path):
    """A local nf4 copy beside a 16bit Hub repo must not answer the Hub-side check: check_model_quantization_status
    prefers the directory (HF_nf4, priority 4 not 3), so merged_16bit refused a base the Hub serves in full
    precision."""
    _require_a_case_sensitive_filesystem(tmp_path)
    monkeypatch.chdir(tmp_path)
    directory = os.path.join("outputs", "mymodel")
    os.makedirs(directory, exist_ok = True)
    open(os.path.join(directory, "model.safetensors"), "wb").close()
    with open(os.path.join(directory, "config.json"), "w", encoding = "utf-8") as f:
        json.dump({"model_type": "llama",
                   "quantization_config": {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4"}}, f)

    def fake_ls(self, path, detail = True, **kwargs):
        return [{"name": f"{path}/model.safetensors"}]
    monkeypatch.setattr(saving_utils.HfFileSystem, "ls", fake_ls, raising = True)
    # The Hub's own config, reached only if the directory is not consulted for it.
    hub_config = tmp_path / "hub-config.json"
    hub_config.write_text(json.dumps({"model_type": "llama"}), encoding = "utf-8")
    import huggingface_hub
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download",
        lambda *args, **kwargs: str(hub_config), raising = True,
    )

    name, is_local, source, is_quantized, quant_type = \
        saving_utils.determine_base_model_source("outputs/mymodel", save_method = "merged_16bit")
    assert source == "HF_unquantized", source
    assert is_local is False
    assert is_quantized is False
    assert quant_type is None
