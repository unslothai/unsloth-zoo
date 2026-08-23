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

"""A local base model must still resolve when the Hub is unreachable.

The other half of `test_check_hf_model_exists_transport_errors.py`, which the first
fix broke: `determine_base_model_source` probed the Hub *before* looking on disk, and
a local directory is not a repo id, so making every non-absent error fatal made local
base models unresolvable.

What `HfFileSystem.ls` actually raises for a local path (measured, not guessed):

    huggingface_hub 1.24              huggingface_hub 0.36
    'base'        ValueError          NotImplementedError
    './base'      HfUriError          FileNotFoundError
    '/abs/base'   FileNotFoundError online, OfflineModeIsEnabled offline

Only two of those six are in the absent set and the type is not stable across
versions, so the obvious shapes are classified from the string instead: a leading `.`,
`/` or `~`, or more than one `/`, is never offered to the Hub at all.

A single segment name is deliberately NOT rejected that way: `gpt2` and
`bert-base-uncased` are canonical repos 0.x lists happily (measured: 17 entries with
safetensors on 0.36.2), so they are still probed, and 1.x's plain `ValueError` counts
as absent only for names with no `/`.

The string gate is necessary but not sufficient, because `outputs/mymodel` is a valid
repo id *and* an ordinary local directory. Hence the second half: resolve local first.
Priorities 1 and 2 outrank every Hub answer, so returning them before the probe
changes no result while an unreachable Hub still propagates past them.
"""

import json
import os

import pytest
import requests
from huggingface_hub.errors import (
    HFValidationError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
)

from unsloth_zoo import saving_utils

try:
    from huggingface_hub.errors import HfUriError
except ImportError:      # huggingface_hub < 1.0 has no hf:// URI parser
    HfUriError = None


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


def _forbid_hub(monkeypatch):
    """Fail loudly if anything reaches for the Hub at all, which pins that resolution
    is network free rather than merely tolerant of a failure."""
    def no_network(self, path, detail = True, **kwargs):
        raise AssertionError(
            f"HfFileSystem.ls({path!r}) was called; a local base model must "
            f"resolve without probing the Hub"
        )
    monkeypatch.setattr(saving_utils.HfFileSystem, "ls", no_network, raising = True)


def _hub_raises(monkeypatch, error):
    def fake_ls(self, path, detail = True, **kwargs):
        raise error
    monkeypatch.setattr(saving_utils.HfFileSystem, "ls", fake_ls, raising = True)


def _response(status_code):
    response = requests.Response()
    response.status_code = status_code
    response.reason = "test"
    response.url = "https://huggingface.co/api/models/test"
    return response


_OFFLINE = OfflineModeIsEnabled(
    "Cannot reach https://huggingface.co: offline mode is enabled."
)


# 1. check_hf_model_exists: a string that cannot name a repo is absent, not
#    unreachable, and must cost no network call.

_NOT_REPO_IDS = [
    pytest.param("./base", id = "dot-slash-relative-path"),
    pytest.param("/home/user/models/base", id = "absolute-path"),
    pytest.param("/base", id = "absolute-single-segment"),
    pytest.param("~/models/base", id = "home-relative-path"),
    pytest.param("models/base/checkpoint-500", id = "three-segment-path"),
    pytest.param("C:\\models\\base", id = "windows-path"),
    pytest.param("", id = "empty-string"),
    pytest.param(".", id = "cwd"),
]


@pytest.mark.parametrize("model_name", _NOT_REPO_IDS)
def test_local_path_is_absent_not_unreachable(monkeypatch, model_name):
    """These raised RuntimeError("could not reach the Hub"), which is both wrong and
    fatal."""
    _forbid_hub(monkeypatch)
    assert saving_utils.check_hf_model_exists(model_name) is False


@pytest.mark.parametrize("model_name", _NOT_REPO_IDS)
@pytest.mark.parametrize("typed_uris", [False, True])
def test_a_path_stays_a_path_on_either_huggingface_hub(monkeypatch, model_name, typed_uris):
    """The same contract under both answers the Hub parser can give.

    `models/base/checkpoint-500` is classified by logic gated on whether the installed
    release maps the `models/` type prefix, and a job installs only one version, so
    running under whichever one that is leaves the other branch untested: a
    version-gated strip landed green here and still broke this file on 1.25.1.
    """
    monkeypatch.setattr(
        saving_utils, "_hub_addresses_typed_model_uris", lambda: typed_uris,
    )
    _forbid_hub(monkeypatch)
    assert saving_utils.check_hf_model_exists(model_name) is False


def test_valid_repo_id_is_still_probed(monkeypatch):
    """The gate must not quietly stop checking the Hub for real repo ids.

    Includes the canonical single segment ids, which 0.x lists happily (measured:
    `ls("gpt2")` returns 17 entries with safetensors on 0.36.2, and pyproject supports
    >= 0.34). Rejecting them from the string would turn a working `gpt2` merge into
    "not found locally or on Hugging Face" on every 0.3x install.
    """
    seen = []
    def fake_ls(self, path, detail = True, **kwargs):
        seen.append(path)
        return [{"name": f"{path}/model.safetensors"}]
    monkeypatch.setattr(saving_utils.HfFileSystem, "ls", fake_ls, raising = True)

    assert saving_utils.check_hf_model_exists("unsloth/Llama-3.2-1B-Instruct") is True
    # A two segment relative directory is indistinguishable from a repo id, so it must
    # still be probed rather than assumed local.
    assert saving_utils.check_hf_model_exists("outputs/mymodel") is True
    assert saving_utils.check_hf_model_exists("gpt2") is True
    assert saving_utils.check_hf_model_exists("bert-base-uncased") is True
    assert seen == [
        "unsloth/Llama-3.2-1B-Instruct", "outputs/mymodel", "gpt2", "bert-base-uncased",
    ]


_SINGLE_SEGMENT_REJECTIONS = [
    pytest.param(
        ValueError("Repository id must be 'namespace/name', got 'base'. "
                   "Single-segment ids (e.g. 'gpt2') are no longer supported."),
        id = "hf-1.x-plain-ValueError",
    ),
    pytest.param(
        NotImplementedError("Access to repositories lists is not implemented."),
        id = "hf-0.x-NotImplementedError-namespace-listing",
    ),
]


@pytest.mark.parametrize("error", _SINGLE_SEGMENT_REJECTIONS)
def test_single_segment_name_the_hub_will_not_list_is_absent(monkeypatch, error):
    """`base` is probed (it could be a canonical id) but both refusals mean "not a repo
    id I can list", never "the Hub is unreachable"."""
    _hub_raises(monkeypatch, error)
    assert saving_utils.check_hf_model_exists("base") is False


_INVALID_REPO_ID_ERRORS = [
    pytest.param(
        HFValidationError("Repo id must use alphanumeric chars, '-', '_' or '.'."),
        id = "HFValidationError",
    ),
    pytest.param(
        NotImplementedError("Access to repositories lists is not implemented."),
        id = "hf-0.x-NotImplementedError",
    ),
]
if HfUriError is not None:
    _INVALID_REPO_ID_ERRORS.append(pytest.param(
        HfUriError("hf://./base", "Repo id must use alphanumeric chars, '-', '_' or '.'."),
        id = "hf-1.x-HfUriError",
    ))


@pytest.mark.parametrize("error", _INVALID_REPO_ID_ERRORS)
def test_invalid_repo_id_errors_are_classified_absent(monkeypatch, error):
    """Backstop for a repo-id-shaped string the installed release still rejects as
    malformed: that is about the argument, never about connectivity."""
    _hub_raises(monkeypatch, error)
    assert saving_utils.check_hf_model_exists("unsloth/Llama-3.2-1B-Instruct") is False


def test_a_plain_ValueError_from_the_transport_is_not_swallowed(monkeypatch):
    """A bare `ValueError` is absent only for a single segment name. Adding `ValueError`
    to the absent set outright would also swallow a transport failure surfacing as one
    and reopen the silent no-op, so the catch is scoped to names with no `/`."""
    _hub_raises(monkeypatch, ValueError("malformed chunked response from proxy"))
    with pytest.raises(RuntimeError):
        saving_utils.check_hf_model_exists("unsloth/Llama-3.2-1B-Instruct")


def test_transport_error_on_a_valid_repo_id_still_raises(monkeypatch):
    """Widening the absent set must not let the silent no-op back in."""
    _hub_raises(monkeypatch, _OFFLINE)
    with pytest.raises(RuntimeError):
        saving_utils.check_hf_model_exists("unsloth/Llama-3.2-1B-Instruct")


# 2. determine_base_model_source: a local base model resolves with no network.

@pytest.mark.parametrize("shape", ["relative", "dot-slash", "absolute"])
def test_local_base_model_resolves_without_network(monkeypatch, tmp_path, shape):
    """The three shapes a user actually passes for a local base model."""
    monkeypatch.chdir(tmp_path)
    _make_local_model("base")
    model_name = {
        "relative":  "base",
        "dot-slash": "./base",
        "absolute":  str(tmp_path / "base"),
    }[shape]

    _forbid_hub(monkeypatch)
    final_name, is_local, source_info, is_quantized, quant_type = (
        saving_utils.determine_base_model_source(model_name)
    )

    assert is_local is True
    assert source_info == "local_unquantized"
    assert is_quantized is False and quant_type is None
    assert os.path.realpath(final_name) == os.path.realpath(str(tmp_path / "base"))


def test_absolute_local_path_resolves_under_hf_hub_offline(monkeypatch, tmp_path):
    """The reported shape: absolute path plus HF_HUB_OFFLINE=1.

    Both offline devices are deliberate. `HF_HUB_OFFLINE` is read into module constants
    at import time, so setting it cannot take an already-imported huggingface_hub
    offline, and the `OfflineModeIsEnabled` fake is never reached because an absolute
    path is not probed. Keeping both means removing either protection still fails.
    """
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.chdir(tmp_path)
    _make_local_model("base")
    _hub_raises(monkeypatch, _OFFLINE)

    final_name, is_local, source_info, _, _ = (
        saving_utils.determine_base_model_source(str(tmp_path / "base"))
    )
    assert is_local is True
    assert source_info == "local_unquantized"
    assert os.path.realpath(final_name) == os.path.realpath(str(tmp_path / "base"))


def test_repo_id_shaped_local_dir_resolves_when_hub_is_unreachable(monkeypatch, tmp_path):
    """`outputs/mymodel` is a valid repo id AND an ordinary local directory, so no
    string test can spare it from the Hub. Only resolving local first saves it, which is
    what makes the reordering necessary and the string gate insufficient."""
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("outputs", "mymodel"))
    _hub_raises(monkeypatch, _OFFLINE)

    final_name, is_local, source_info, _, _ = (
        saving_utils.determine_base_model_source("outputs/mymodel")
    )
    assert is_local is True
    assert source_info == "local_unquantized"
    assert os.path.realpath(final_name) == os.path.realpath(
        str(tmp_path / "outputs" / "mymodel")
    )


def test_local_mxfp4_also_outranks_the_hub_without_network(monkeypatch, tmp_path):
    """Priority 2 outranks every Hub answer just as priority 1 does."""
    monkeypatch.chdir(tmp_path)
    _make_local_model("base", quantization_config = {"quant_method": "mxfp4"})
    _forbid_hub(monkeypatch)

    _, is_local, source_info, is_quantized, quant_type = (
        saving_utils.determine_base_model_source("base")
    )
    assert (is_local, source_info, is_quantized, quant_type) == (
        True, "local_mxfp4", True, "mxfp4",
    )


# 3. The original bug stays fixed: no local fallback plus an unreachable Hub must still
#    be loud.

_UNREACHABLE = [
    pytest.param(_OFFLINE, id = "hf-hub-offline"),
    pytest.param(ConnectionError("Temporary failure in name resolution"), id = "dns-failure"),
    pytest.param(TimeoutError("read timed out"), id = "read-timeout"),
]


@pytest.mark.parametrize("error", _UNREACHABLE)
def test_unreachable_hub_with_no_local_copy_still_raises(monkeypatch, tmp_path, error):
    """Nothing on disk can answer, so silence here is what exported nothing."""
    monkeypatch.chdir(tmp_path)     # genuinely empty: no local candidate at all
    _hub_raises(monkeypatch, error)

    with pytest.raises(RuntimeError) as excinfo:
        saving_utils.determine_base_model_source("unsloth/Llama-3.2-1B-Instruct")
    assert "unsloth/Llama-3.2-1B-Instruct" in str(excinfo.value)


def test_unreachable_hub_does_not_fall_through_to_nothing_found(monkeypatch, tmp_path):
    """The old behaviour returned `(None, ...)` here, which the caller turned into a
    no-op export plus a warning."""
    monkeypatch.chdir(tmp_path)
    _hub_raises(monkeypatch, _OFFLINE)

    try:
        result = saving_utils.determine_base_model_source("unsloth/Llama-3.2-1B-Instruct")
    except RuntimeError:
        return  # correct
    pytest.fail(f"unreachable Hub reported as 'nothing found': {result!r}")


def test_absent_repo_with_no_local_copy_still_reports_nothing_found(monkeypatch, tmp_path):
    """A genuinely missing repo keeps the old answer, so the caller can raise its own
    message naming the model."""
    monkeypatch.chdir(tmp_path)
    _hub_raises(monkeypatch, RepositoryNotFoundError("404", response = _response(404)))

    assert saving_utils.determine_base_model_source("unslothai/nope") == (
        None, False, "", False, None,
    )


# 4. A local copy that only wins at priority 5 must NOT rescue an unreachable Hub,
#    because the merge cannot use it for a 16bit export.

def test_unreachable_hub_does_not_fall_back_to_a_priority_5_local_copy(
    monkeypatch, tmp_path,
):
    """A bnb-4bit local copy is priority 5, below the Hub's 3 and 4.

    Substituting it when the Hub is unreachable is not graceful degradation:
    `merge_and_overwrite_lora` writes nothing for an nf4/fp4 base under
    `merged_16bit`, which is the silent no-op itself, so the RuntimeError propagates.
    Priorities 1 and 2 legitimately resolve without the Hub and are covered above.
    """
    monkeypatch.chdir(tmp_path)
    _make_local_model(os.path.join("unsloth", "mymodel"), quantization_config = {
        "load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
    })
    _hub_raises(monkeypatch, _OFFLINE)

    with pytest.raises(RuntimeError) as excinfo:
        saving_utils.determine_base_model_source("unsloth/mymodel")
    assert "unsloth/mymodel" in str(excinfo.value)


def test_a_non_repo_id_local_quantized_copy_still_resolves_at_priority_5(
    monkeypatch, tmp_path,
):
    """A local path is never probed, so priority 5 is reached with no Hub call and a
    `forced_merged_4bit` export still works offline. Only a repo-id-shaped name can be
    blocked by an unreachable Hub."""
    monkeypatch.chdir(tmp_path)
    _make_local_model("base", quantization_config = {
        "load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
    })
    _forbid_hub(monkeypatch)

    final_name, is_local, source_info, is_quantized, quant_type = (
        saving_utils.determine_base_model_source("./base")
    )
    assert (is_local, source_info, is_quantized, quant_type) == (
        True, "local_nf4", True, "nf4",
    )
    assert os.path.realpath(final_name) == os.path.realpath(str(tmp_path / "base"))


def test_reachable_hub_still_outranks_a_quantized_local_copy(monkeypatch, tmp_path):
    """The priority order is unchanged: the fallback fires only when the Hub failed."""
    monkeypatch.chdir(tmp_path)
    _make_local_model("mymodel", quantization_config = {
        "load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
    })
    monkeypatch.setattr(saving_utils.HfFileSystem, "ls",
        lambda self, path, detail = True, **kw: [{"name": f"{path}/model.safetensors"}],
        raising = True)
    # Unquantized on the Hub -> priority 3 beats the local nf4 copy at priority 5.
    monkeypatch.setattr(saving_utils, "check_model_quantization_status",
        lambda name, token = None:
            (False, None) if str(name) == "unsloth/mymodel" else (True, "nf4"))

    final_name, is_local, source_info, _, _ = (
        saving_utils.determine_base_model_source("unsloth/mymodel")
    )
    assert (final_name, is_local, source_info) == ("unsloth/mymodel", False, "HF_unquantized")
