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

"""A transport failure on a single-segment id must not read as absent: only the huggingface_hub 1.16+ guard raises a
bare ValueError with its wording; on 1.0-1.15 a bare ValueError can come off a live socket, and on 0.x a transport
failure can surface as a ValueError subclass."""

import http.server
import inspect
import json
import sys
import threading

import huggingface_hub
import pytest
import requests
from huggingface_hub.errors import OfflineModeIsEnabled

from unsloth_zoo import saving_utils

try:
    from huggingface_hub.errors import HfUriError
except ImportError:      # huggingface_hub < 1.0 has no hf:// URI parser
    HfUriError = None

def _hub_has_single_segment_guard():
    """Whether the resolver refuses a slashless id before sending a request; feature-detected, since the major version does not answer it (guard first in 1.16.0)."""
    try:
        source = inspect.getsource(huggingface_hub.HfFileSystem.resolve_path)
    except Exception:
        return False
    return 'path.count("/") == 0' in source and "Repository id must be" in source


_HUB_HAS_SINGLE_SEGMENT_GUARD = _hub_has_single_segment_guard()


# The exact text 1.24.0 uses, so these tests describe the real rejection.
_REJECTION_MESSAGE = (
    "Repository id must be 'namespace/name', got 'gpt2'. "
    "Single-segment ids (e.g. 'gpt2') are no longer supported."
)


def _patch_ls(monkeypatch, side_effect):
    def fake_ls(self, path, detail = True, **kwargs):
        if isinstance(side_effect, BaseException):
            raise side_effect
        return side_effect
    monkeypatch.setattr(saving_utils.HfFileSystem, "ls", fake_ls, raising = True)


def _requests_json_decode_error():
    """requests JSONDecodeError on a non-JSON body (0.36.2) is a ValueError and OSError but not FileNotFoundError, so it reaches the ValueError catch."""
    return requests.exceptions.JSONDecodeError("Expecting value", "<html>", 0)


def _stdlib_json_decode_error():
    """What httpx (huggingface_hub 1.x) raises for the same body."""
    return json.JSONDecodeError("Expecting value", "<html>", 0)


_SUBCLASS_TRANSPORT_ERRORS = [
    pytest.param(_requests_json_decode_error, id = "requests-JSONDecodeError"),
    pytest.param(_stdlib_json_decode_error,   id = "stdlib-JSONDecodeError"),
]


@pytest.mark.parametrize("make_error", _SUBCLASS_TRANSPORT_ERRORS)
@pytest.mark.parametrize("name", ["gpt2", "bert-base-uncased", "distilgpt2"])
def test_json_decode_error_on_single_segment_id_raises(monkeypatch, make_error, name):
    """A proxy mangling the response body is a connectivity failure; on 0.x it lands on a slashless name, so answering False would silently hide it."""
    error = make_error()
    assert isinstance(error, ValueError), "premise: it reaches the ValueError catch"
    _patch_ls(monkeypatch, error)
    with pytest.raises(RuntimeError) as excinfo:
        saving_utils.check_hf_model_exists(name)
    assert "connectivity" in str(excinfo.value)
    assert excinfo.value.__cause__ is error


@pytest.mark.parametrize("make_error", _SUBCLASS_TRANSPORT_ERRORS)
def test_json_decode_error_never_answers_false(monkeypatch, make_error):
    _patch_ls(monkeypatch, make_error())
    try:
        result = saving_utils.check_hf_model_exists("gpt2")
    except RuntimeError:
        return
    pytest.fail(f"reported {result!r} for an unreachable Hub instead of raising")


@pytest.mark.parametrize("make_error", _SUBCLASS_TRANSPORT_ERRORS)
def test_determine_base_model_source_propagates_single_segment_transport_error(
    monkeypatch, make_error,
):
    """determine_base_model_source must raise rather than return (None, ...), since merge_and_overwrite_lora turns that into a merge that writes nothing."""
    _patch_ls(monkeypatch, make_error())
    monkeypatch.setattr(saving_utils, "check_local_model_exists", lambda path: None)
    with pytest.raises(RuntimeError):
        saving_utils.determine_base_model_source("gpt2")


@pytest.mark.parametrize("name", ["gpt2", "bert-base-uncased", "distilgpt2"])
def test_single_segment_rejection_still_reports_absent(monkeypatch, name):
    """1.16+ rejects these ids before opening a socket, so the check stays False and callers fall through to their local priorities."""
    message = _REJECTION_MESSAGE.replace("gpt2", name)
    _patch_ls(monkeypatch, ValueError(message))
    assert saving_utils.check_hf_model_exists(name) is False


def test_single_segment_rejection_is_recognised_by_its_message(monkeypatch):
    """The message is recognised on 0.x too, so a backported guard would not report a phantom connectivity failure."""
    assert saving_utils._is_single_segment_id_rejection("gpt2", ValueError(_REJECTION_MESSAGE))


@pytest.mark.parametrize("name", ["hf://gpt2", "hf://bert-base-uncased"])
def test_a_single_segment_id_wearing_the_uri_scheme_is_still_absent(monkeypatch, name):
    """hf://gpt2 is unaddressable like gpt2; 1.16+ names the stripped id, so counting slashes before stripping would
    misreport it as a connectivity failure and skip local priorities."""
    _patch_ls(monkeypatch, ValueError(_REJECTION_MESSAGE))
    assert saving_utils.check_hf_model_exists(name) is False


def test_the_uri_scheme_does_not_smuggle_a_transport_failure_past_the_check(monkeypatch):
    """Stripping the scheme must not make bare ValueErrors that are not id rejections count as absent; a transport failure must still raise."""
    _patch_ls(monkeypatch, ValueError("Connection reset by peer"))
    with pytest.raises(RuntimeError):
        saving_utils.check_hf_model_exists("hf://gpt2")


@pytest.mark.parametrize("name", [
    "openai-community/gpt2@main",
    "hf://openai-community/gpt2@main",
    "openai-community/gpt2@refs/pr/1",
])
def test_a_revision_suffix_is_still_addressed_as_a_repo(name):
    """namespace/name@revision is documented HfFileSystem syntax; validate_repo_id rejects the @ and refs/pr/1 trips the
    depth rule, so split the revision before any prefilter."""
    assert saving_utils._is_hub_repo_id(name) is True


@pytest.mark.parametrize("name", ["./base@old", "/abs/base@old", "~/base@old", "a/b/c@old"])
def test_a_revision_suffix_does_not_launder_a_filesystem_path(name):
    """Splitting on `@` must not turn a path into a repo id."""
    assert saving_utils._is_hub_repo_id(name) is False


@pytest.mark.parametrize("name", ["gpt2@main", "hf://gpt2@main", "bert-base-uncased@main"])
def test_a_single_segment_id_keeps_its_revision(name):
    """A revision on a single-segment id (gpt2@main) must survive the split, which must not require one slash; on 1.16+ its rejection must not read as connectivity."""
    assert saving_utils._is_hub_repo_id(name) is True


@pytest.mark.parametrize("name", [
    "hf://models/openai-community/gpt2",
    "hf://models/openai-community/gpt2@main",
])
def test_an_explicit_typed_model_uri_tracks_what_the_parser_does(name):
    """Under hf://, models/namespace/name is the typed form but resolves only on some releases, so classification must equal the probed capability, not always strip."""
    supported = saving_utils._hub_addresses_typed_model_uris()
    assert saving_utils._is_hub_repo_id(name) is supported


@pytest.mark.parametrize("name", [
    "models/base/checkpoint-500",
    "models/outputs/final",
    "models/openai-community/gpt2",
])
@pytest.mark.parametrize("capability", [False, True])
def test_a_bare_models_prefix_stays_a_local_path(monkeypatch, name, capability):
    """A bare models/ prefix stays a local path (what a Trainer writes) under both capability answers; stripping it would send a local base to the Hub."""
    monkeypatch.setattr(
        saving_utils, "_hub_addresses_typed_model_uris", lambda: capability,
    )
    assert saving_utils._is_hub_repo_id(name) is False


def test_the_capability_probe_asks_about_the_resolver_not_just_the_parser(monkeypatch):
    """The probe must ask the resolver, not just parse_hf_uri: models/ resolves only from 1.16.0, though parse_hf_uri
    exists from 1.15, so a parser-only probe would claim support on 1.15."""
    if "models/" in tuple(getattr(huggingface_hub.constants,
                                  "REPO_TYPES_URL_PREFIXES", {}).values()):
        pytest.skip("this release advertises models/ in the prefix table")

    # The parser is injected on every version so the cell discriminates whatever is installed; the real parse_hf_uri would be vacuous on 0.x.
    import types
    stub = types.ModuleType("huggingface_hub.utils._hf_uris")

    class _Uri:
        type = "model"
        id = "namespace/name"
        revision = None
        path_in_repo = ""

    stub.parse_hf_uri = lambda uri, endpoint = None: _Uri()
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils._hf_uris", stub)

    def resolver_without_the_parser(self, path, revision = None):
        # References no URI parser: the 1.15 and 0.x shape.
        raise AssertionError("never called")

    monkeypatch.setattr(
        saving_utils.HfFileSystem, "resolve_path", resolver_without_the_parser,
        raising = True,
    )
    assert saving_utils._hub_addresses_typed_model_uris() is False, (
        "a parser that maps the prefix does not mean the resolver reaches it"
    )

    def resolver_with_the_parser(self, path, revision = None):
        # The capability probe reads the code object, so parse_hf_uri must be referenced for real; a mention in a comment must not count.
        parse_hf_uri  # noqa: F821
        raise AssertionError("never called")

    monkeypatch.setattr(
        saving_utils.HfFileSystem, "resolve_path", resolver_with_the_parser,
        raising = True,
    )
    assert saving_utils._hub_addresses_typed_model_uris() is True


@pytest.mark.parametrize("name", [
    "/models/org/repo", "./models/org/repo", "~/models/org/repo", "models/a/b/c",
])
def test_the_typed_prefix_never_launders_a_path(monkeypatch, name):
    """With the capability forced on, a path that merely starts with models stays a path: parse_hf_uri would read
    models/a/b/c as repo a/b, so path rules are not delegated to it."""
    monkeypatch.setattr(saving_utils, "_hub_addresses_typed_model_uris", lambda: True)
    assert saving_utils._is_hub_repo_id(name) is False


@pytest.mark.parametrize("name", [
    "datasets/stanfordnlp/imdb", "spaces/org/demo", "kernels/org/k",
])
def test_repo_types_that_cannot_be_a_base_model_stay_rejected(monkeypatch, name):
    """Dataset, Space and kernel ids resolve too but are rejected on purpose: none can be the base model of a LoRA merge."""
    monkeypatch.setattr(saving_utils, "_hub_addresses_typed_model_uris", lambda: True)
    assert saving_utils._is_hub_repo_id(name) is False


@pytest.mark.parametrize("given, expected", [
    pytest.param("hf://openai-community/gpt2/",  "openai-community/gpt2", id = "root-uri"),
    pytest.param("hf://openai-community/gpt2//", "openai-community/gpt2", id = "doubled"),
    pytest.param("hf://gpt2/",                   "gpt2",                  id = "single-segment"),
])
def test_a_trailing_slash_on_an_hf_uri_is_not_depth(given, expected):
    """A trailing slash on hf://org/repo/ is not depth: resolve_path normalizes it to repo org/repo with an empty path, so ls addresses the repository."""
    assert saving_utils._as_hub_addressed(given) == expected
    assert saving_utils._is_hub_repo_id(given) is True


@pytest.mark.parametrize("given", [
    "outputs/mymodel/",
    "./outputs/mymodel/",
    "outputs/nested/mymodel/",
])
def test_a_trailing_slash_without_the_scheme_still_reads_as_a_directory(given):
    """Only an explicit hf:// makes a trailing slash a Hub root; bare, it marks a local directory, and stripping it
    would send outputs/mymodel/ to the Hub as a repo id."""
    assert saving_utils._is_hub_repo_id(given) is False


def test_a_root_uri_is_probed_rather_than_reported_absent(monkeypatch):
    """The root hf:// URI must be probed: a depth rule on the trailing slash would report an existing repo absent with no Hub round trip to disagree."""
    seen = []

    def fake_ls(self, path, detail = True, **kwargs):
        seen.append(path)
        return [{"name": f"{path}/model.safetensors"}]
    monkeypatch.setattr(saving_utils.HfFileSystem, "ls", fake_ls, raising = True)

    assert saving_utils.check_hf_model_exists("hf://openai-community/gpt2/") is True
    assert seen == ["hf://openai-community/gpt2/"], (
        "the name must reach `ls` unedited: `_as_hub_addressed` decides shape, not the argument"
    )


@pytest.mark.parametrize("save_method", ["merged_4bit", "forced_merged_4bit", "merged_16bit", None])
def test_an_at_sign_in_a_local_directory_changes_nothing(monkeypatch, tmp_path, save_method):
    """Splitting on @ offers local outputs/my@model to the Hub as outputs/my, no different from an ordinary one-slash
    directory, so the outcome must match before and after the split; asserted as an equivalence, so it survives a
    policy change for local bases."""
    import huggingface_hub as _hub

    def _local_nf4(directory):
        directory.mkdir(parents = True, exist_ok = True)
        (directory / "model.safetensors").write_bytes(b"")
        (directory / "config.json").write_text(json.dumps({
            "model_type": "llama",
            "quantization_config": {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4"},
        }), encoding = "utf-8")

    _patch_ls(monkeypatch, ConnectionError("Temporary failure in name resolution"))
    def _dead_download(*args, **kwargs):
        raise ConnectionError("Temporary failure in name resolution")
    monkeypatch.setattr(_hub, "hf_hub_download", _dead_download, raising = True)

    outcomes = []
    for relative in ("outputs/mymodel", "outputs/my@model"):
        root = tmp_path / relative.replace("/", "_").replace("@", "at")
        _local_nf4(root / "outputs" / relative.split("/")[1])
        monkeypatch.chdir(root)
        try:
            resolved = saving_utils.determine_base_model_source(relative, None, save_method)
            outcomes.append(("resolved", resolved[2]))
        except RuntimeError as e:
            outcomes.append(("raised", "connectivity" in str(e) or "rate limiting" in str(e)))

    assert outcomes[0] == outcomes[1], (
        f"`outputs/my@model` diverged from `outputs/mymodel`: {outcomes}"
    )


@pytest.mark.parametrize("name", ["gpt2@main", "hf://gpt2@main"])
def test_a_revisioned_single_segment_id_is_absent_not_unreachable(monkeypatch, name):
    """1.25.1 names the whole gpt2@main in its rejection, so the classifier must strip the revision before counting segments or it reads a transport failure."""
    message = _REJECTION_MESSAGE.replace("gpt2", "gpt2@main", 1)
    _patch_ls(monkeypatch, ValueError(message))
    assert saving_utils.check_hf_model_exists(name) is False


def test_rejection_message_on_a_namespaced_name_still_raises(monkeypatch):
    """namespace/name is addressable on every supported version, so a rejection message naming it is a real failure and must still raise."""
    _patch_ls(monkeypatch, ValueError(_REJECTION_MESSAGE))
    with pytest.raises(RuntimeError):
        saving_utils.check_hf_model_exists("openai-community/gpt2")


def test_a_bare_valueerror_with_no_rejection_wording_always_raises(monkeypatch):
    """A bare ValueError without the guard's wording must raise: a major >= 1 gate answered False for live-socket
    ValueErrors on 1.0-1.15, so the merge wrote nothing."""
    _patch_ls(monkeypatch, ValueError("boom"))
    with pytest.raises(RuntimeError) as excinfo:
        saving_utils.check_hf_model_exists("gpt2")
    message = str(excinfo.value)
    assert "connectivity" in message or "rate limiting" in message


def test_classification_never_consults_the_installed_version(monkeypatch):
    """Classification uses the message alone, never huggingface_hub.__version__: a version check hid the 1.0-1.15 gap
    from a test that recomputed the same expression."""
    _patch_ls(monkeypatch, ValueError(_REJECTION_MESSAGE))
    assert saving_utils.check_hf_model_exists("gpt2") is False

    _patch_ls(monkeypatch, ValueError("Expecting value: line 1 column 1 (char 0)"))
    with pytest.raises(RuntimeError):
        saving_utils.check_hf_model_exists("gpt2")

    assert not hasattr(saving_utils, "_HUB_REJECTS_SINGLE_SEGMENT_IDS"), (
        "recognition is by the rejection wording, which upstream keeps byte "
        "identical across 1.16.0 - 1.25.1, not by a version boundary that has "
        "already moved once"
    )


@pytest.mark.skipif(HfUriError is None, reason = "huggingface_hub < 1.0 has no HfUriError")
def test_hf_uri_error_is_still_absent_not_transport(monkeypatch):
    """HfUriError subclasses ValueError but is an id error matched by type in the absent set, so it never reaches the transport branch."""
    _patch_ls(monkeypatch, HfUriError("hf://gpt2", "Invalid HF URI 'hf://gpt2'"))
    assert saving_utils.check_hf_model_exists("gpt2") is False


@pytest.mark.parametrize("name", ["gpt2", "bert-base-uncased", "distilgpt2"])
def test_listable_single_segment_id_still_returns_true(monkeypatch, name):
    """A listable single-segment id still returns True: 0.36.2 lists gpt2; the narrowing lives only in the except branch, so no True became False."""
    _patch_ls(monkeypatch, [
        {"name": f"{name}/model.safetensors"},
        {"name": f"{name}/config.json"},
    ])
    assert saving_utils.check_hf_model_exists(name) is True


def test_absent_single_segment_id_still_returns_false(monkeypatch):
    """0.x answers NotImplementedError for a slashless name with no canonical repo behind it; that is still absent, with no exception raised."""
    _patch_ls(monkeypatch, NotImplementedError("Access to repositories lists is not implemented."))
    assert saving_utils.check_hf_model_exists("definitely-not-a-real-model-xyz") is False


def test_offline_mode_on_a_single_segment_id_raises(monkeypatch):
    _patch_ls(monkeypatch, OfflineModeIsEnabled("Cannot reach https://huggingface.co"))
    with pytest.raises(RuntimeError):
        saving_utils.check_hf_model_exists("gpt2")


class _GarbageHandler(http.server.BaseHTTPRequestHandler):
    """200 with HTML, which is what a captive portal does to an API call."""

    _BODY = b"<html><body>proxy error: not json at all</body></html>"

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self._BODY)))
        self.end_headers()
        self.wfile.write(self._BODY)

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def garbage_endpoint():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _GarbageHandler)
    thread = threading.Thread(target = server.serve_forever, daemon = True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.skipif(
    _HUB_HAS_SINGLE_SEGMENT_GUARD,
    reason = "this huggingface_hub refuses single segment ids before it sends a request, "
             "so no transport failure can reach that branch on this version",
)
def test_real_proxy_failure_on_a_single_segment_id_raises(monkeypatch, garbage_endpoint):
    """No stubbed exception: a real socket, a real 200 and a body the client cannot parse, as a captive proxy would send."""
    base_fs = saving_utils.HfFileSystem

    class _PinnedEndpoint(base_fs):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("endpoint", garbage_endpoint)
            kwargs["token"] = False
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(saving_utils, "HfFileSystem", _PinnedEndpoint, raising = True)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)

    with pytest.raises(RuntimeError) as excinfo:
        saving_utils.check_hf_model_exists("gpt2")
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert type(excinfo.value.__cause__) is not ValueError, (
        "premise: the parse failure arrives as a ValueError subclass"
    )
