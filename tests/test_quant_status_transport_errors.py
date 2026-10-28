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

"""A Hub failure fetching config.json must raise, not return (False, None) as unquantized, which skipped the nf4/fp4 16bit merge refusal; the tests mock the Hub."""

import json
import os

import huggingface_hub
import pytest
import requests
from huggingface_hub.errors import (
    EntryNotFoundError,
    HfHubHTTPError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
)

from huggingface_hub.utils import validate_repo_id
from unsloth_zoo import saving_utils


_REPO = "ns/base-bnb-4bit"


def _patch_ls_present(monkeypatch):
    """The existence probe succeeds; everything here is about the round trip after."""
    def fake_ls(self, path, detail = True, **kwargs):
        return [{"name": f"{path}/model.safetensors"}]
    monkeypatch.setattr(saving_utils.HfFileSystem, "ls", fake_ls, raising = True)


def _patch_config_fetch(monkeypatch, side_effect):
    def fake_download(*args, **kwargs):
        raise side_effect
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download, raising = True)


def _make_local_model(directory, quant_config = None):
    import json
    directory.mkdir(parents = True, exist_ok = True)
    (directory / "model.safetensors").write_bytes(b"")
    config = {"model_type": "llama"}
    if quant_config is not None:
        config["quantization_config"] = quant_config
    (directory / "config.json").write_text(json.dumps(config), encoding = "utf-8")
    return directory


class _StubResponse:
    """The attributes `HfHubHTTPError.__init__` actually reads on 1.x."""
    status_code = 429
    reason      = "Too Many Requests"
    url         = "https://huggingface.co/api/models/test"
    headers     = {}
    content     = b""
    text        = ""
    request     = None


def _hub_http_error(message):
    """Builds a real HfHubHTTPError by signature, not version: 0.x takes a bare message, while 1.0+ requires a
    keyword-only response; pyproject allows huggingface_hub>=0.34.0."""
    try:
        return HfHubHTTPError(message)
    except TypeError:
        return HfHubHTTPError(message, response = _StubResponse())


def _hub_error(cls, message):
    """Same signature-based construction for the other Hub errors: RepositoryNotFoundError and GatedRepoError need
    response on 1.x; EntryNotFoundError and OfflineModeIsEnabled do not."""
    try:
        return cls(message)
    except TypeError:
        return cls(message, response = _StubResponse())


def _RateLimited(message = "429 Client Error: Too Many Requests"):
    return _hub_http_error(message)


def _ServerError(message = "503 Server Error: Service Unavailable"):
    return _hub_http_error(message)


_TRANSPORT_ERRORS = [
    pytest.param(_RateLimited, id = "rate-limited-429"),
    pytest.param(_ServerError, id = "server-error-503"),
    pytest.param(lambda: ConnectionError("Temporary failure in name resolution"), id = "dns-failure"),
    pytest.param(lambda: TimeoutError("Read timed out"), id = "read-timeout"),
    pytest.param(lambda: OfflineModeIsEnabled("Offline mode is enabled"), id = "hf-hub-offline"),
]


@pytest.mark.parametrize("make_error", _TRANSPORT_ERRORS)
def test_transport_failure_on_the_config_fetch_raises(monkeypatch, make_error):
    error = make_error()
    _patch_ls_present(monkeypatch)
    _patch_config_fetch(monkeypatch, error)

    with pytest.raises(RuntimeError) as excinfo:
        saving_utils.check_model_quantization_status(_REPO)

    message = str(excinfo.value)
    assert "connectivity" in message or "rate limiting" in message
    assert _REPO in message
    assert excinfo.value.__cause__ is error


def test_the_error_says_it_is_not_a_statement_about_quantization(monkeypatch):
    """The RuntimeError must say it is not a statement about quantization; 'not a missing model' is the wrong
    reassurance for a caller asking whether weights are quantized."""
    _patch_ls_present(monkeypatch)
    _patch_config_fetch(monkeypatch, _RateLimited())

    with pytest.raises(RuntimeError) as excinfo:
        saving_utils.check_model_quantization_status(_REPO)

    message = str(excinfo.value)
    assert "unquantized" in message
    assert "quantization config" in message


@pytest.mark.parametrize("make_error", _TRANSPORT_ERRORS)
def test_transport_failure_does_not_return_false(monkeypatch, make_error):
    """A rate-limited config fetch must never come back as 'this model is not quantized': that answer caused the wrong merge."""
    _patch_ls_present(monkeypatch)
    _patch_config_fetch(monkeypatch, make_error())

    with pytest.raises(RuntimeError):
        result = saving_utils.check_model_quantization_status(_REPO)
        pytest.fail(f"answered {result!r} for an unreachable Hub instead of raising")


# The absent cases must keep answering (False, None).

_ABSENT_ERRORS = [
    pytest.param(lambda: _hub_error(EntryNotFoundError, "config.json not found"), id = "no-config-in-repo"),
    pytest.param(lambda: _hub_error(RepositoryNotFoundError, "404 repo not found"), id = "repo-absent"),
    pytest.param(lambda: FileNotFoundError("config.json"), id = "fsspec-file-not-found"),
    # A bare ValueError from the download is deliberately not in the absent set: treating a malformed body as
    # deterministic let a mangling proxy answer unquantized.
]


def test_a_mangled_proxy_response_on_the_download_raises(monkeypatch):
    """Fetch and parse are classified separately: both JSONDecodeError classes subclass ValueError, so one except
    ValueError would forgive a proxy corrupting the download."""
    _patch_ls_present(monkeypatch)
    _patch_config_fetch(
        monkeypatch,
        requests.exceptions.JSONDecodeError("Expecting value", "<html>", 0),
    )
    with pytest.raises(RuntimeError) as excinfo:
        saving_utils.check_model_quantization_status(_REPO)
    assert "connectivity" in str(excinfo.value) or "rate limiting" in str(excinfo.value)


@pytest.mark.parametrize("body, label", [
    pytest.param("{not json",                 "truncated",  id = "truncated-json"),
    pytest.param("<html>proxy error</html>",  "proxy-page", id = "proxy-error-page"),
])
def test_a_config_that_cannot_be_parsed_raises(monkeypatch, tmp_path, body, label):
    """A downloaded config that cannot be parsed must raise: (False, None) would skip the nf4/fp4 guard and merge 16bit
    over quantized weights; a proxy HTML error page lands here."""
    blob = tmp_path / "config.json"
    blob.write_text(body, encoding = "utf-8")
    _patch_ls_present(monkeypatch)
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download",
        lambda *args, **kwargs: str(blob), raising = True,
    )
    with pytest.raises(RuntimeError) as excinfo:
        saving_utils.check_model_quantization_status(_REPO)
    assert "cannot" in str(excinfo.value) or "could not" in str(excinfo.value)


def test_an_offline_cache_miss_on_the_config_raises(monkeypatch):
    """LocalEntryNotFoundError subclasses EntryNotFoundError, so its clause must come before the absent set or an
    offline run reports every remote base as unquantized."""
    from huggingface_hub.errors import LocalEntryNotFoundError
    assert issubclass(LocalEntryNotFoundError, EntryNotFoundError), "premise of the ordering"
    _patch_ls_present(monkeypatch)
    _patch_config_fetch(monkeypatch, LocalEntryNotFoundError("offline and not cached"))
    with pytest.raises(RuntimeError):
        saving_utils.check_model_quantization_status(_REPO)


@pytest.mark.parametrize("make_error", _ABSENT_ERRORS)
def test_absent_or_malformed_config_still_reports_unquantized(monkeypatch, make_error):
    """Absent-config errors still return (False, None): nothing readable says the weights are quantized and none of these errors is transient."""
    _patch_ls_present(monkeypatch)
    _patch_config_fetch(monkeypatch, make_error())
    assert saving_utils.check_model_quantization_status(_REPO) == (False, None)


def test_a_reachable_hub_still_detects_quantization(monkeypatch, tmp_path):
    """The success path is untouched: a real config.json still classifies."""
    config = _make_local_model(
        tmp_path / "cached",
        quant_config = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4"},
    ) / "config.json"
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download",
        lambda *args, **kwargs: str(config), raising = True,
    )
    assert saving_utils.check_model_quantization_status(_REPO) == (True, "nf4")


# A malformed *local* config.json takes the same path as a downloaded one.

def test_malformed_local_config_raises(tmp_path):
    """A local config.json that exists but cannot be parsed must raise: unknown quantization must not be reported as not quantized."""
    directory = tmp_path / "base"
    directory.mkdir()
    (directory / "model.safetensors").write_bytes(b"")
    (directory / "config.json").write_text("{not json", encoding = "utf-8")
    with pytest.raises(RuntimeError):
        saving_utils.check_model_quantization_status(str(directory))


def test_an_absent_local_config_still_reports_unquantized(tmp_path):
    """An absent local config still returns (False, None), unlike an unreadable one: an empty directory says nothing, and the parse is never reached."""
    directory = tmp_path / "base"
    directory.mkdir()
    (directory / "model.safetensors").write_bytes(b"")
    assert saving_utils.check_model_quantization_status(str(directory)) == (False, None)


# Offline fallbacks still apply on the config fetch: it is reached only when os.path.exists(name) is False but
# check_local_model_exists finds a case-differing copy (Outputs/MyModel vs outputs/mymodel).

_REQUESTED = "Outputs/MyModel"
_ON_DISK = ("outputs", "mymodel")


def _require_a_case_sensitive_filesystem(tmp_path):
    """Probes case sensitivity and skips otherwise: only a case-only name difference makes os.path.exists miss while
    check_local_model_exists hits; on APFS or NTFS these tests would test nothing."""
    probe = tmp_path / "case_probe"
    probe.mkdir(exist_ok = True)
    if os.path.exists(str(tmp_path / "CASE_PROBE")):
        pytest.skip(
            "case insensitive filesystem: os.path.exists would resolve "
            f"{_REQUESTED!r} locally and skip the Hub branch under test"
        )
    pass


@pytest.mark.parametrize("save_method", ["merged_4bit", "forced_merged_4bit"])
def test_local_4bit_still_resolves_when_only_the_config_fetch_fails(
    monkeypatch, tmp_path, save_method,
):
    """A config fetch failure with ls succeeding must reach the same local fallback as a total outage, since both 4bit
    merges fold LoRA into weights already in memory."""
    directory = _make_local_model(
        tmp_path.joinpath(*_ON_DISK),
        quant_config = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4"},
    )
    _require_a_case_sensitive_filesystem(tmp_path)
    monkeypatch.chdir(tmp_path)
    _patch_ls_present(monkeypatch)
    _patch_config_fetch(monkeypatch, _RateLimited())

    resolved = saving_utils.determine_base_model_source(_REQUESTED, None, save_method)
    import os
    assert os.path.realpath(resolved[0]) == os.path.realpath(str(directory))
    assert resolved[1] is True
    assert resolved[2] == "local_nf4"


def test_local_4bit_plus_16bit_merge_still_raises(monkeypatch, tmp_path):
    """merged_16bit from an nf4 base cannot complete from local weights, so the fallback does not apply and the outage propagates."""
    _make_local_model(
        tmp_path.joinpath(*_ON_DISK),
        quant_config = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4"},
    )
    _require_a_case_sensitive_filesystem(tmp_path)
    monkeypatch.chdir(tmp_path)
    _patch_ls_present(monkeypatch)
    _patch_config_fetch(monkeypatch, _RateLimited())

    with pytest.raises(RuntimeError):
        saving_utils.determine_base_model_source(_REQUESTED, None, "merged_16bit")


def test_local_fp8_resolves_for_a_16bit_merge(monkeypatch, tmp_path):
    """FP8 keeps its fallback: the 16bit merge dequantizes the local FP8 weights."""
    directory = _make_local_model(
        tmp_path.joinpath(*_ON_DISK),
        quant_config = {"quant_method": "fp8"},
    )
    _require_a_case_sensitive_filesystem(tmp_path)
    monkeypatch.chdir(tmp_path)
    _patch_ls_present(monkeypatch)
    _patch_config_fetch(monkeypatch, _RateLimited())

    resolved = saving_utils.determine_base_model_source(_REQUESTED, None, "merged_16bit")
    import os
    assert os.path.realpath(resolved[0]) == os.path.realpath(str(directory))
    assert resolved[2] == "local_fp8"


def test_no_local_copy_propagates(monkeypatch, tmp_path):
    """With no local copy the outage must propagate, not return (None, ...), which would silently write no files."""
    _require_a_case_sensitive_filesystem(tmp_path)
    monkeypatch.chdir(tmp_path)
    _patch_ls_present(monkeypatch)
    _patch_config_fetch(monkeypatch, _RateLimited())

    with pytest.raises(RuntimeError):
        saving_utils.determine_base_model_source("ns/absent", None, "merged_16bit")


def test_an_exactly_named_local_directory_never_reaches_the_config_fetch(
    monkeypatch, tmp_path,
):
    """When the requested name is the local directory itself, quantization is read from disk and the config fetch is never attempted."""
    _make_local_model(
        tmp_path.joinpath(*_ON_DISK),
        quant_config = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4"},
    )
    _require_a_case_sensitive_filesystem(tmp_path)
    monkeypatch.chdir(tmp_path)
    _patch_ls_present(monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("the config fetch must not be reached for a local directory")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbidden, raising = True)

    resolved = saving_utils.determine_base_model_source(
        "outputs/mymodel", None, "merged_4bit",
    )
    assert resolved[3] is True
    assert resolved[4] == "nf4"


def test_a_disabled_repo_is_not_announced_as_a_connectivity_problem(monkeypatch):
    """DisabledRepoError does not subclass RepositoryNotFoundError, so unless named it hits the catch-all and is reported as connectivity; retrying cannot help."""
    from huggingface_hub.errors import DisabledRepoError, RepositoryNotFoundError
    assert not issubclass(DisabledRepoError, RepositoryNotFoundError), (
        "premise: this is why it needs naming rather than inheriting its way in"
    )

    # Built via _hub_error: a bare DisabledRepoError(...) raises TypeError on hub 1.0 (required keyword-only response),
    # which would read like the bug under test.
    error = _hub_error(DisabledRepoError, "403 disabled")

    def fake_ls(self, path, detail = True, **kwargs):
        raise error
    monkeypatch.setattr(saving_utils.HfFileSystem, "ls", fake_ls, raising = True)

    assert saving_utils.check_hf_model_exists("ns/disabled") is False


@pytest.mark.parametrize("given, expected_repo, expected_revision", [
    pytest.param("ns/name",           "ns/name", None,  id = "plain"),
    pytest.param("hf://ns/name",      "ns/name", None,  id = "uri"),
    pytest.param("ns/name@dev",       "ns/name", "dev", id = "revision"),
    pytest.param("hf://ns/name@dev",  "ns/name", "dev", id = "uri-and-revision"),
])
def test_the_download_is_addressed_as_a_repo_and_a_revision(
    monkeypatch, given, expected_repo, expected_revision,
):
    """hf_hub_download rejects hf://ns/name and ns/name@dev with HFValidationError, which the absent set would read as
    unquantized; pass repo_id and revision separately."""
    seen = {}

    def fake_download(*args, **kwargs):
        seen.update(kwargs)
        # Built through the helper: `response` is required on 1.x for some of these classes.
        raise _hub_error(EntryNotFoundError, "no config.json")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download, raising = True)

    saving_utils.check_model_quantization_status(given)
    assert seen.get("repo_id") == expected_repo
    assert seen.get("revision") == expected_revision


def test_a_quantized_base_behind_a_uri_is_not_reported_unquantized(monkeypatch, tmp_path):
    """The consequence, end to end at this function: the address must not decide the answer."""
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"quantization_config": {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4"}}),
        encoding = "utf-8",
    )
    def fake_download(*args, **kwargs):
        # What the real one does with anything that is not a bare repo id.
        validate_repo_id(kwargs["repo_id"])
        return str(config)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download, raising = True)

    for given in ("ns/name", "hf://ns/name", "ns/name@dev"):
        assert saving_utils.check_model_quantization_status(given) == (True, "nf4"), given
