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

"""An unreachable Hub must not be reported as an *unquantized* model either.

Narrowing `check_hf_model_exists` closed the first of two Hub round trips on the
16bit merge path. `determine_base_model_source` makes a second one immediately
after: once the existence probe says the repo is there, it asks
`check_model_quantization_status` whether the weights are quantized, and that
function fetches `config.json`.

That fetch was wrapped in a bare `except:` returning `(False, None)`, so a 429, a
5xx, a read timeout or a Xet stall on the config alone produced:

    ls("ns/base-bnb-4bit")            -> succeeds, repo exists
    hf_hub_download("config.json")    -> 429, swallowed, config = None
    check_model_quantization_status   -> (False, None)
    determine_base_model_source       -> ("ns/base-bnb-4bit", False,
                                          "HF_unquantized", False, None)

which is Priority 3 for an nf4 base. `merge_and_overwrite_lora` gates its
nf4/fp4 refusal on `base_model_is_quantized`, so that guard was skipped and a
`merged_16bit` export proceeded against weights it believed were already 16bit.

This is strictly worse than the silent no-op the rest of this branch removes: a
no-op writes nothing, and this writes something wrong. Same failure of reasoning,
one call later, so it is classified the same way. Every test here monkeypatches
the two Hub entry points, so nothing touches the network.
"""

import huggingface_hub
import pytest
import requests
from huggingface_hub.errors import (
    EntryNotFoundError,
    HfHubHTTPError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
)

from unsloth_zoo import saving_utils


_REPO = "ns/base-bnb-4bit"


def _patch_ls_present(monkeypatch):
    """The existence probe succeeds: the repo is there and carries safetensors.
    Everything here is about what happens on the round trip *after* that."""
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
    """Build a real `HfHubHTTPError` whichever way the installed version wants.

    0.x takes a bare message. From 1.0 `response` is keyword-only and required,
    and the constructor reads `.headers` and `.request` off it. Doing this by
    signature rather than by version keeps these cells describing the class the
    Hub really raises across the whole supported range, and `pyproject.toml`
    allows huggingface_hub>=0.34.0, so the range is wide.
    """
    try:
        return HfHubHTTPError(message)
    except TypeError:
        return HfHubHTTPError(message, response = _StubResponse())


def _hub_error(cls, message):
    """Same signature-not-version handling for the other HTTP-backed classes.
    `RepositoryNotFoundError` and `GatedRepoError` inherit the required `response`
    on 1.x; `EntryNotFoundError` and `OfflineModeIsEnabled` do not."""
    try:
        return cls(message)
    except TypeError:
        return cls(message, response = _StubResponse())


def _RateLimited(message = "429 Client Error: Too Many Requests"):
    return _hub_http_error(message)


def _ServerError(message = "503 Server Error: Service Unavailable"):
    return _hub_http_error(message)


# ---------------------------------------------------------------------------
# A transport failure on the config fetch must raise, not answer "unquantized".
# ---------------------------------------------------------------------------

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
    """"not a missing model" is the wrong reassurance for this caller, which was
    asking whether the weights are quantized, not whether the repo is there."""
    _patch_ls_present(monkeypatch)
    _patch_config_fetch(monkeypatch, _RateLimited())

    with pytest.raises(RuntimeError) as excinfo:
        saving_utils.check_model_quantization_status(_REPO)

    message = str(excinfo.value)
    assert "unquantized" in message
    assert "quantization config" in message


@pytest.mark.parametrize("make_error", _TRANSPORT_ERRORS)
def test_transport_failure_does_not_return_false(monkeypatch, make_error):
    """The single step that caused the wrong merge, asserted bluntly: a rate
    limited config fetch must never come back as "this model is not quantized"."""
    _patch_ls_present(monkeypatch)
    _patch_config_fetch(monkeypatch, make_error())

    with pytest.raises(RuntimeError):
        result = saving_utils.check_model_quantization_status(_REPO)
        pytest.fail(f"answered {result!r} for an unreachable Hub instead of raising")


# ---------------------------------------------------------------------------
# The absent cases must keep answering (False, None).
# ---------------------------------------------------------------------------

_ABSENT_ERRORS = [
    pytest.param(lambda: _hub_error(EntryNotFoundError, "config.json not found"), id = "no-config-in-repo"),
    pytest.param(lambda: _hub_error(RepositoryNotFoundError, "404 repo not found"), id = "repo-absent"),
    pytest.param(lambda: FileNotFoundError("config.json"), id = "fsspec-file-not-found"),
    # A bare ValueError out of the *download* is deliberately NOT here. It was,
    # on the assumption that a malformed body is deterministic, and that is what
    # let a mangling proxy answer "unquantized". Whether an unreadable config is
    # transient depends on which step failed, so the two steps get their own
    # tests: test_a_mangled_proxy_response_on_the_download_raises and
    # test_a_malformed_downloaded_config_does_not_raise.
]


def test_a_mangled_proxy_response_on_the_download_raises(monkeypatch):
    """The subtle half, and the one an earlier revision of this fix got wrong.

    `json.JSONDecodeError` and `requests.exceptions.JSONDecodeError` are both
    ValueError subclasses, so a single `except ValueError` that forgives a
    malformed config.json ALSO forgives a captive portal or mangling middlebox
    corrupting the API response mid-download. That is a transport failure being
    reported as "this model is not quantized", which is the exact defect this
    function was narrowed to stop, reintroduced one layer down.

    The fetch and the parse are therefore classified separately: this must raise,
    while `test_absent_or_malformed_config_still_reports_unquantized` below pins
    that an unreadable file already in hand must not.
    """
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
    """The download succeeded and the bytes are not a config.

    An earlier revision of this file asserted the opposite, that this should stay
    `(False, None)` because nothing about a malformed file is transient. That reads
    the wrong question. Transience is not what the caller uses the answer for: it
    uses it to decide whether the base is quantized, and "I could not read the
    config" is not evidence that the weights are full precision. Answering
    `(False, None)` skips the nf4/fp4 guard and merges 16bit over quantized weights,
    which is the same class of wrong answer as the rest of this file, just sourced
    from the disk rather than the socket. A mangling proxy also reaches here by
    serving an HTML error page as the file body.
    """
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
    """`LocalEntryNotFoundError` means "the network is unavailable and this file is
    not cached". It subclasses `EntryNotFoundError`, which is in the absent set, so
    without its own clause ahead of that tuple an offline run reports every remote
    base as unquantized."""
    from huggingface_hub.errors import LocalEntryNotFoundError
    assert issubclass(LocalEntryNotFoundError, EntryNotFoundError), "premise of the ordering"
    _patch_ls_present(monkeypatch)
    _patch_config_fetch(monkeypatch, LocalEntryNotFoundError("offline and not cached"))
    with pytest.raises(RuntimeError):
        saving_utils.check_model_quantization_status(_REPO)


@pytest.mark.parametrize("make_error", _ABSENT_ERRORS)
def test_absent_or_malformed_config_still_reports_unquantized(monkeypatch, make_error):
    """Nothing readable says the weights are quantized, and none of these is
    transient, so `(False, None)` stays the honest answer."""
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


# ---------------------------------------------------------------------------
# A malformed *local* config.json is not a transport failure and must not raise.
# ---------------------------------------------------------------------------

def test_malformed_local_config_raises(tmp_path):
    """Same reasoning on the local branch, and the same code path: a config.json
    that exists but cannot be parsed leaves the quantization of the base unknown,
    and unknown must not be reported as "not quantized"."""
    directory = tmp_path / "base"
    directory.mkdir()
    (directory / "model.safetensors").write_bytes(b"")
    (directory / "config.json").write_text("{not json", encoding = "utf-8")
    with pytest.raises(RuntimeError):
        saving_utils.check_model_quantization_status(str(directory))


def test_an_absent_local_config_still_reports_unquantized(tmp_path):
    """Absent is genuinely different from unreadable and keeps its answer. Nothing
    in an empty directory says the weights are quantized, and the parse is never
    reached because the file does not exist."""
    directory = tmp_path / "base"
    directory.mkdir()
    (directory / "model.safetensors").write_bytes(b"")
    assert saving_utils.check_model_quantization_status(str(directory)) == (False, None)


# ---------------------------------------------------------------------------
# The offline fallbacks still apply on this second round trip.
#
# Reaching the config fetch at all needs care, and the reason is worth stating.
# `check_model_quantization_status` branches on `os.path.exists(name)`, so when
# the requested name is *itself* an existing directory it reads that directory's
# config.json and never opens a socket. The Hub branch is therefore reached only
# when `os.path.exists(name)` is False while `check_local_model_exists(name)`
# still finds a copy, and it does that case insensitively. `Outputs/MyModel`
# against an on-disk `outputs/mymodel` is exactly that shape, and it is an
# ordinary thing for a user to type.
# ---------------------------------------------------------------------------

_REQUESTED = "Outputs/MyModel"
_ON_DISK = ("outputs", "mymodel")


@pytest.mark.parametrize("save_method", ["merged_4bit", "forced_merged_4bit"])
def test_local_4bit_still_resolves_when_only_the_config_fetch_fails(
    monkeypatch, tmp_path, save_method,
):
    """A partial outage that lets `ls` through but rate limits the config fetch
    has to reach the same fallback as a total one. Both 4bit merges fold LoRA
    into the weights already in memory, so they need nothing from the Hub."""
    directory = _make_local_model(
        tmp_path.joinpath(*_ON_DISK),
        quant_config = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4"},
    )
    monkeypatch.chdir(tmp_path)
    _patch_ls_present(monkeypatch)
    _patch_config_fetch(monkeypatch, _RateLimited())

    resolved = saving_utils.determine_base_model_source(_REQUESTED, None, save_method)
    import os
    assert os.path.realpath(resolved[0]) == os.path.realpath(str(directory))
    assert resolved[1] is True
    assert resolved[2] == "local_nf4"


def test_local_4bit_plus_16bit_merge_still_raises(monkeypatch, tmp_path):
    """`merged_16bit` off an nf4 base cannot complete from local weights, so the
    fallback deliberately does not answer for it and the outage propagates."""
    _make_local_model(
        tmp_path.joinpath(*_ON_DISK),
        quant_config = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4"},
    )
    monkeypatch.chdir(tmp_path)
    _patch_ls_present(monkeypatch)
    _patch_config_fetch(monkeypatch, _RateLimited())

    with pytest.raises(RuntimeError):
        saving_utils.determine_base_model_source(_REQUESTED, None, "merged_16bit")


def test_local_fp8_resolves_for_a_16bit_merge(monkeypatch, tmp_path):
    """FP8 keeps its fallback here too: the 16bit merge dequantizes the local FP8
    weights and needs nothing from the network."""
    directory = _make_local_model(
        tmp_path.joinpath(*_ON_DISK),
        quant_config = {"quant_method": "fp8"},
    )
    monkeypatch.chdir(tmp_path)
    _patch_ls_present(monkeypatch)
    _patch_config_fetch(monkeypatch, _RateLimited())

    resolved = saving_utils.determine_base_model_source(_REQUESTED, None, "merged_16bit")
    import os
    assert os.path.realpath(resolved[0]) == os.path.realpath(str(directory))
    assert resolved[2] == "local_fp8"


def test_no_local_copy_propagates(monkeypatch, tmp_path):
    """Nothing local can answer, so the outage is the whole story and must be
    what the caller sees, rather than a `(None, ...)` that writes no files."""
    monkeypatch.chdir(tmp_path)
    _patch_ls_present(monkeypatch)
    _patch_config_fetch(monkeypatch, _RateLimited())

    with pytest.raises(RuntimeError):
        saving_utils.determine_base_model_source("ns/absent", None, "merged_16bit")


def test_an_exactly_named_local_directory_never_reaches_the_config_fetch(
    monkeypatch, tmp_path,
):
    """The complement, pinned so the reasoning above cannot silently rot: when the
    requested name IS the directory, quantization is read off disk and the config
    fetch is not attempted, so an outage there cannot affect the answer."""
    _make_local_model(
        tmp_path.joinpath(*_ON_DISK),
        quant_config = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4"},
    )
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


# ---------------------------------------------------------------------------
# A disabled repo is a fact about the repo, not about the network.
# ---------------------------------------------------------------------------

def test_a_disabled_repo_is_not_announced_as_a_connectivity_problem(monkeypatch):
    """`DisabledRepoError` is the one Hub 4xx that does NOT subclass
    `RepositoryNotFoundError`, so without being named explicitly it reached the
    catch-all and was announced as "a connectivity or rate limiting problem, not a
    missing model". It is neither, and no amount of retrying changes it."""
    from huggingface_hub.errors import DisabledRepoError, RepositoryNotFoundError
    assert not issubclass(DisabledRepoError, RepositoryNotFoundError), (
        "premise: this is why it needs naming rather than inheriting its way in"
    )

    # Built through `_hub_error`, not `DisabledRepoError("...")`. It inherits
    # HfHubHTTPError, whose `response` is keyword-only and required from hub 1.0, so
    # the bare call raises TypeError there. That TypeError then reaches the
    # catch-all and the test fails announcing a connectivity problem, which reads
    # exactly like the bug under test rather than a broken fixture.
    error = _hub_error(DisabledRepoError, "403 disabled")

    def fake_ls(self, path, detail = True, **kwargs):
        raise error
    monkeypatch.setattr(saving_utils.HfFileSystem, "ls", fake_ls, raising = True)

    assert saving_utils.check_hf_model_exists("ns/disabled") is False
