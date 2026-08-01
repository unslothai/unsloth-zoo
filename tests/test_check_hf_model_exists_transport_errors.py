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

"""`check_hf_model_exists` must not report "absent" for a Hub it cannot reach.

The bare `except: return False` around `HfFileSystem(...).ls(...)` mapped a 429, a
5xx, a DNS or proxy failure, a read timeout and `HF_HUB_OFFLINE` onto "this repo does
not exist". `determine_base_model_source` then returned `(None, ...)` and
`merge_and_overwrite_lora` warned and returned None, so `save_pretrained_merged`
created no output directory at all and only warned about it.

`HfFileSystem` already separates the two: `_raise_file_not_found` (a plain
FileNotFoundError) is reached only after `_repo_and_revision_exists` catches
RepositoryNotFoundError / RevisionNotFoundError / HFValidationError, while every
transport failure propagates out of `ls` untouched.
"""

import warnings

import pytest
import requests
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from unsloth_zoo import saving_utils


def _response(status_code):
    """A real requests.Response, which is what HfHubHTTPError introspects."""
    response = requests.Response()
    response.status_code = status_code
    response.reason = "test"
    response.url = "https://huggingface.co/api/models/test"
    return response


def _patch_ls(monkeypatch, side_effect):
    """No test in this file touches the network."""
    def fake_ls(self, path, detail = True, **kwargs):
        if isinstance(side_effect, BaseException):
            raise side_effect
        return side_effect
    monkeypatch.setattr(saving_utils.HfFileSystem, "ls", fake_ls, raising = True)


# Unreachable Hub must raise, never answer False.

_TRANSPORT_ERRORS = [
    pytest.param(
        HfHubHTTPError("429 Too Many Requests", response = _response(429)),
        id = "rate-limited-429",
    ),
    pytest.param(
        HfHubHTTPError("503 Service Unavailable", response = _response(503)),
        id = "server-error-503",
    ),
    pytest.param(ConnectionError("Temporary failure in name resolution"), id = "dns-failure"),
    pytest.param(TimeoutError("read timed out"), id = "read-timeout"),
    pytest.param(OSError("proxy refused the connection"), id = "proxy-error"),
    pytest.param(
        OfflineModeIsEnabled("Cannot reach https://huggingface.co: offline mode is enabled."),
        id = "hf-hub-offline",
    ),
]


@pytest.mark.parametrize("error", _TRANSPORT_ERRORS)
def test_transport_failure_raises_instead_of_reporting_absent(monkeypatch, error):
    _patch_ls(monkeypatch, error)

    with pytest.raises(RuntimeError) as excinfo:
        saving_utils.check_hf_model_exists("unsloth/Llama-3.2-1B-Instruct")

    message = str(excinfo.value)
    # Names the real cause, not the model.
    assert "connectivity" in message or "rate limiting" in message
    assert "unsloth/Llama-3.2-1B-Instruct" in message
    assert excinfo.value.__cause__ is error


def test_transport_failure_does_not_return_false(monkeypatch):
    """`return False` here is the single step that makes the whole export no-op."""
    _patch_ls(monkeypatch, HfHubHTTPError("429", response = _response(429)))

    result = None
    try:
        result = saving_utils.check_hf_model_exists("unsloth/Llama-3.2-1B-Instruct")
    except RuntimeError:
        return  # correct behaviour
    pytest.fail(
        f"check_hf_model_exists swallowed a 429 and returned {result!r}; "
        f"the caller would then export nothing and warn only."
    )


# A genuinely absent or inaccessible repo must still answer False.

_ABSENT_ERRORS = [
    # What ls actually raises for a missing repo: fsspec converts
    # RepositoryNotFoundError into a plain FileNotFoundError.
    pytest.param(
        FileNotFoundError("unslothai/nope (repository not found)"), id = "fsspec-file-not-found",
    ),
    pytest.param(
        RepositoryNotFoundError("404 repo not found", response = _response(404)),
        id = "repository-not-found",
    ),
    pytest.param(
        RevisionNotFoundError("404 revision not found", response = _response(404)),
        id = "revision-not-found",
    ),
    pytest.param(
        GatedRepoError("403 gated repo", response = _response(403)), id = "gated-repo",
    ),
]


@pytest.mark.parametrize("error", _ABSENT_ERRORS)
def test_absent_repo_still_returns_false(monkeypatch, error):
    _patch_ls(monkeypatch, error)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert saving_utils.check_hf_model_exists("unslothai/definitely-not-a-real-repo") is False


# A gated repo answers False like an absent one, but it is not absent, and that
# difference is the only actionable thing the user has.

def _ls_gated_the_way_fsspec_delivers_it(monkeypatch):
    """What a gated repo ACTUALLY looks like coming out of `HfFileSystem.ls`.

    Not a GatedRepoError: `_repo_and_revision_exist` catches
    `RepositoryNotFoundError` (which GatedRepoError subclasses) and
    `_raise_file_not_found` re-raises it as `FileNotFoundError(msg) from err`, so the
    real reason is in `__cause__` and an `except GatedRepoError` handler would never
    fire. Read off the installed huggingface_hub source, not assumed.
    """
    def fake_ls(self, path, detail = True, **kwargs):
        try:
            raise GatedRepoError("403 gated repo", response = _response(403))
        except GatedRepoError as err:
            raise FileNotFoundError(f"{path} (repository not found)") from err
    monkeypatch.setattr(saving_utils.HfFileSystem, "ls", fake_ls, raising = True)


def test_gated_repo_says_it_is_gated_rather_than_absent(monkeypatch):
    """False on purpose, so a local copy still wins, but a gated repo demonstrably
    exists and reporting it missing sends the user hunting a typo they do not have."""
    _ls_gated_the_way_fsspec_delivers_it(monkeypatch)

    with pytest.warns(UserWarning) as caught:
        assert saving_utils.check_hf_model_exists("meta-llama/Llama-3.2-1B") is False

    message = "\n".join(str(w.message) for w in caught)
    assert "gated" in message
    assert "meta-llama/Llama-3.2-1B" in message
    assert "token" in message


def test_gated_detection_survives_the_fsspec_conversion(monkeypatch):
    """Pinned so a refactor back to `except GatedRepoError` cannot pass: the type
    never reaches the handler."""
    assert issubclass(GatedRepoError, RepositoryNotFoundError)
    wrapped = None
    try:
        try:
            raise GatedRepoError("403", response = _response(403))
        except GatedRepoError as err:
            raise FileNotFoundError("ns/x (repository not found)") from err
    except FileNotFoundError as e:
        wrapped = e
    assert not isinstance(wrapped, GatedRepoError), "the type is gone by this point"
    assert saving_utils._gated_repo_cause(wrapped) is not None, (
        "the reason has to be recovered from __cause__"
    )
    assert saving_utils._gated_repo_cause(FileNotFoundError("plain 404")) is None


def test_a_plain_absent_repo_does_not_warn_about_gating(monkeypatch):
    """The complement: nothing about a 404 should mention accepting terms."""
    _patch_ls(monkeypatch, RepositoryNotFoundError("404", response = _response(404)))
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        assert saving_utils.check_hf_model_exists("ns/nope") is False
    assert [w for w in caught if "gated" in str(w.message)] == []


# The success paths are unchanged.

def test_repo_with_safetensors_returns_true(monkeypatch):
    _patch_ls(monkeypatch, [
        {"name": "unsloth/Llama-3.2-1B-Instruct/config.json"},
        {"name": "unsloth/Llama-3.2-1B-Instruct/model.safetensors"},
    ])
    assert saving_utils.check_hf_model_exists("unsloth/Llama-3.2-1B-Instruct") is True


def test_repo_without_safetensors_returns_false(monkeypatch):
    _patch_ls(monkeypatch, [
        {"name": "unsloth/Llama-3.2-1B-Instruct/config.json"},
        {"name": "unsloth/Llama-3.2-1B-Instruct/pytorch_model.bin"},
    ])
    assert saving_utils.check_hf_model_exists("unsloth/Llama-3.2-1B-Instruct") is False


def test_no_warning_is_used_to_report_an_unreachable_hub(monkeypatch):
    """A warning does not stop the caller believing the export happened."""
    _patch_ls(monkeypatch, HfHubHTTPError("429", response = _response(429)))

    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError):
            saving_utils.check_hf_model_exists("unsloth/Llama-3.2-1B-Instruct")

    assert not [w for w in caught if "not found" in str(w.message)]


# The downstream consequence: no silent no-op merge.

def test_determine_base_model_source_propagates_the_transport_error(monkeypatch):
    """The path that produced `(None, ...)` and hence the silent no-op export."""
    _patch_ls(monkeypatch, HfHubHTTPError("429", response = _response(429)))
    monkeypatch.setattr(saving_utils, "check_local_model_exists", lambda *a, **k: None)

    with pytest.raises(RuntimeError):
        saving_utils.determine_base_model_source("unsloth/Llama-3.2-1B-Instruct")


def test_determine_base_model_source_still_reports_nothing_found(monkeypatch):
    """A truly absent repo with no local copy keeps the old (None, ...) answer."""
    _patch_ls(monkeypatch, FileNotFoundError("unslothai/nope (repository not found)"))
    monkeypatch.setattr(saving_utils, "check_local_model_exists", lambda *a, **k: None)

    final_model_name, is_local, source_info, is_quantized, quant_type = (
        saving_utils.determine_base_model_source("unslothai/definitely-not-a-real-repo")
    )
    assert final_model_name is None
    assert is_local is False
    assert source_info == ""
