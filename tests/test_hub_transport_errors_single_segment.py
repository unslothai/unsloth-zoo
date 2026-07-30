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

"""A transport failure on a single segment id must not be reported as "absent".

Third file in the set. `test_check_hf_model_exists_transport_errors.py` pins
"an unreachable Hub is never a missing model";
`test_local_base_model_resolution_offline.py` pins "a local base model resolves
with the Hub down". This one pins the seam between them: the `ValueError` catch
that exists so `gpt2` can answer False on huggingface_hub >= 1.0 must not also
swallow a genuine connectivity failure.

Why the seam exists, measured rather than guessed:

    huggingface_hub 1.24.0   `HfFileSystem.resolve_path` starts with
                             `if path.count("/") == 0: raise ValueError(
                                 "Repository id must be 'namespace/name', ...
                                  Single-segment ids ... no longer supported.")`
                             That guard sits before `parse_hf_uri` and before
                             `_repo_and_revision_exist`, so it is raised without
                             any network I/O and can never be a transport
                             failure.

    huggingface_hub 0.36.2   has no such guard. `ls("gpt2")` really does reach
                             the Hub (17 entries including safetensors), so the
                             network IS touched and a failure there can surface
                             as a ValueError subclass.

Measured against a local HTTP server answering 200 with a non-JSON body, which
is what a captive proxy or a mangling middlebox does:

    huggingface_hub 0.36.2   ls("gpt2")                  requests.exceptions.JSONDecodeError
    huggingface_hub 0.36.2   ls("openai-community/gpt2") requests.exceptions.JSONDecodeError
    huggingface_hub 1.24.0   ls("gpt2")                  ValueError (the guard, no request sent)
    huggingface_hub 1.24.0   ls("openai-community/gpt2") json.JSONDecodeError

Both JSONDecodeError classes are `isinstance(..., ValueError)`, so on 0.36.2 a
proxy failure on `gpt2` landed in the single segment branch and answered False.
`determine_base_model_source` then fell through every priority, and the merge
wrote nothing. Neither is `type(e) is ValueError`, which is the first of the two
discriminators used: only the guard raises the bare class.

The second is the wording. The guard is NOT present on every 1.x release, so the
bare class alone does not identify it: upstream `hf_file_system.py` has no such
check at 0.36.2, 1.0.0, 1.5.0, 1.10.0 or 1.15.0 and grows one at 1.16.0. Across
1.0 - 1.15 a slashless name still reaches the network, so a bare ValueError there
can be a transport failure, and a `major >= 1` gate reported it as absent. The
message is what tells the two apart, and it is byte identical at 1.16.0, 1.20.0,
1.24.0 and 1.25.1.
"""

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
    """Whether the installed resolver refuses a slashless id before sending a
    request, feature detected rather than inferred from the version number.

    The major number does not answer this: upstream `hf_file_system.py` has no
    such guard at 0.36.2, 1.0.0, 1.5.0, 1.10.0 or 1.15.0 and grows one at 1.16.0.
    Reading the installed resolver is exact, needs no network, and cannot drift
    the way a hardcoded boundary already did.
    """
    try:
        source = inspect.getsource(huggingface_hub.HfFileSystem.resolve_path)
    except Exception:
        return False
    return 'path.count("/") == 0' in source and "Repository id must be" in source


_HUB_HAS_SINGLE_SEGMENT_GUARD = _hub_has_single_segment_guard()


# The exact text huggingface_hub 1.24.0 uses, so these tests describe the real
# rejection rather than a convenient stand-in.
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
    """The concrete class requests raises for a non-JSON body (measured on
    0.36.2). It is a ValueError *and* an OSError, but not a FileNotFoundError,
    so it reaches the ValueError catch rather than the absent set."""
    return requests.exceptions.JSONDecodeError("Expecting value", "<html>", 0)


def _stdlib_json_decode_error():
    """What httpx (huggingface_hub 1.x) raises for the same body."""
    return json.JSONDecodeError("Expecting value", "<html>", 0)


# ---------------------------------------------------------------------------
# A ValueError *subclass* on a single segment name is a transport failure.
# ---------------------------------------------------------------------------

_SUBCLASS_TRANSPORT_ERRORS = [
    pytest.param(_requests_json_decode_error, id = "requests-JSONDecodeError"),
    pytest.param(_stdlib_json_decode_error,   id = "stdlib-JSONDecodeError"),
]


@pytest.mark.parametrize("make_error", _SUBCLASS_TRANSPORT_ERRORS)
@pytest.mark.parametrize("name", ["gpt2", "bert-base-uncased", "distilgpt2"])
def test_json_decode_error_on_single_segment_id_raises(monkeypatch, make_error, name):
    """A proxy mangling the response body is a connectivity problem, and on
    huggingface_hub 0.x it lands on a slashless name. Reporting False here is
    the original silent no-op wearing a different hat."""
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
    """The whole point: the caller must never see `(None, ...)`, because
    `merge_and_overwrite_lora` turns that into a merge that writes nothing."""
    _patch_ls(monkeypatch, make_error())
    monkeypatch.setattr(saving_utils, "check_local_model_exists", lambda path: None)
    with pytest.raises(RuntimeError):
        saving_utils.determine_base_model_source("gpt2")


# ---------------------------------------------------------------------------
# The real single segment rejection still answers False.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["gpt2", "bert-base-uncased", "distilgpt2"])
def test_single_segment_rejection_still_reports_absent(monkeypatch, name):
    """huggingface_hub >= 1.16 cannot address these ids at all. That is a
    statement about the argument, raised before a socket is opened, so it stays
    False and callers fall through to their local priorities."""
    message = _REJECTION_MESSAGE.replace("gpt2", name)
    _patch_ls(monkeypatch, ValueError(message))
    assert saving_utils.check_hf_model_exists(name) is False


def test_single_segment_rejection_is_recognised_by_its_message(monkeypatch):
    """Recognised on 0.x too, so a backport of the guard would not start
    reporting a phantom connectivity failure."""
    assert saving_utils._is_single_segment_id_rejection("gpt2", ValueError(_REJECTION_MESSAGE))


@pytest.mark.parametrize("name", ["hf://gpt2", "hf://bert-base-uncased"])
def test_a_single_segment_id_wearing_the_uri_scheme_is_still_absent(monkeypatch, name):
    """`hf://gpt2` is the same unaddressable id with a scheme on it.

    `HfFileSystem` strips the protocol first, so 1.16+ raises the rejection naming
    the *stripped* `gpt2`. Verified on 1.25.1. Counting slashes on the unstripped
    string sees two, concludes `namespace/name`, and reports the rejection as a
    connectivity failure, which also stops `determine_base_model_source` reaching
    its local priorities. Caught by Codex on this branch.
    """
    _patch_ls(monkeypatch, ValueError(_REJECTION_MESSAGE))
    assert saving_utils.check_hf_model_exists(name) is False


def test_the_uri_scheme_does_not_smuggle_a_transport_failure_past_the_check(monkeypatch):
    """The other direction of the same normalisation: stripping the scheme must not
    start swallowing bare ValueErrors that say nothing about an id rejection."""
    _patch_ls(monkeypatch, ValueError("Connection reset by peer"))
    with pytest.raises(RuntimeError):
        saving_utils.check_hf_model_exists("hf://gpt2")


@pytest.mark.parametrize("name", [
    "openai-community/gpt2@main",
    "hf://openai-community/gpt2@main",
    "openai-community/gpt2@refs/pr/1",
])
def test_a_revision_suffix_is_still_addressed_as_a_repo(name):
    """`namespace/name@revision` is documented HfFileSystem syntax.

    `resolve_path` splits the revision off and resolves the repo; verified on
    1.25.1, where `hf://openai-community/gpt2@main` answers repo_id
    `openai-community/gpt2` at revision `main`. `validate_repo_id` rejects the `@`,
    so a prefilter that consults it without splitting first calls an existing remote
    model absent, and `refs/pr/1` additionally trips the depth rule. Before this
    file grew a prefilter these went straight to `ls` and worked. Caught by Codex.
    """
    assert saving_utils._is_hub_repo_id(name) is True


@pytest.mark.parametrize("name", ["./base@old", "/abs/base@old", "~/base@old", "a/b/c@old"])
def test_a_revision_suffix_does_not_launder_a_filesystem_path(name):
    """Splitting on `@` must not turn a path into a repo id."""
    assert saving_utils._is_hub_repo_id(name) is False


@pytest.mark.parametrize("name", ["gpt2@main", "hf://gpt2@main", "bert-base-uncased@main"])
def test_a_single_segment_id_keeps_its_revision(name):
    """A revision on a canonical single segment id, which the first version of the
    split dropped by requiring exactly one slash.

    The zero-slash branch of `resolve_path` splits `@` as well, so this is valid
    syntax on every release before the 1.16 guard. Verified on 0.36.2:

        resolve_path("gpt2@main")             -> repo_id 'gpt2',              rev 'main'
        resolve_path("bert-base-uncased@main")-> repo_id 'bert-base-uncased', rev 'main'

    On 1.16+ the id is unaddressable with or without the revision, and there the
    rejection has to be recognised rather than reported as connectivity, which is
    the cell below. Caught by Codex on this branch.
    """
    assert saving_utils._is_hub_repo_id(name) is True


@pytest.mark.parametrize("name", [
    "hf://models/openai-community/gpt2",
    "hf://models/openai-community/gpt2@main",
])
def test_an_explicit_typed_model_uri_tracks_what_the_parser_does(name):
    """Under `hf://`, `models/namespace/name` is unambiguously the typed form.

    Whether it resolves is a property of the installed release, not of a version
    number, so this asserts the classification *equals the capability* rather than
    hardcoding either answer:

        1.25.1   resolve_path("models/openai-community/gpt2") -> model openai-community/gpt2
        0.36.2   resolve_path("models/openai-community/gpt2") -> FileNotFoundError

    Stripping the prefix unconditionally would be worse than not stripping it, since
    on 0.36.2 that probes `openai-community/gpt2` and would answer True for an
    address that version cannot reach. Caught by Codex, twice: I rejected it the
    first time from 0.36.2's prefix table, which is the wrong version to check.
    """
    supported = saving_utils._hub_addresses_typed_model_uris()
    assert saving_utils._is_hub_repo_id(name) is supported


@pytest.mark.parametrize("name", [
    "models/base/checkpoint-500",
    "models/outputs/final",
    "models/openai-community/gpt2",
])
@pytest.mark.parametrize("capability", [False, True])
def test_a_bare_models_prefix_stays_a_local_path(monkeypatch, name, capability):
    """Without the scheme the same string is ambiguous, and the commoner reading wins.

    `models/base/checkpoint-500` is what a Trainer writes. Stripping the prefix there
    turns a local directory into the plausible repo id `base/checkpoint-500` and sends
    a local base to the Hub, which is why the scheme is what distinguishes "I mean a
    URI" from "I mean a folder". Asserted under both capability answers, because the
    reading must not depend on the installed release: Codex found that it did, by
    running the gate on 1.25.1 where `test_local_path_is_absent_not_unreachable
    [three-segment-path]` then reached `HfFileSystem.ls`.
    """
    monkeypatch.setattr(
        saving_utils, "_hub_addresses_typed_model_uris", lambda: capability,
    )
    assert saving_utils._is_hub_repo_id(name) is False


def test_the_capability_probe_asks_about_the_resolver_not_just_the_parser(monkeypatch):
    """`parse_hf_uri` mapping the prefix does not mean `resolve_path` reaches it.

    Measured against a repo that exists, `resolve_path("models/openai-community/
    gpt2")` fails on 0.34.6, 0.36.2 and 1.15.0 and succeeds on 1.16.0, 1.20.0 and
    1.25.1. `parse_hf_uri` exists from 1.15, so a probe that asks only the parser
    claims support on 1.15 that the resolver does not have. Both questions are
    required, and this pins that: with the parser mapping the prefix but the resolver
    not delegating to it, the answer must be False.
    """
    if "models/" in tuple(getattr(huggingface_hub.constants,
                                  "REPO_TYPES_URL_PREFIXES", {}).values()):
        pytest.skip("this release advertises models/ in the prefix table")

    # A parser that maps the prefix is injected on every version, so this cell has
    # the same discriminating power whether or not the installed release ships one.
    # Asserting against the real `parse_hf_uri` would make the test vacuous on 0.x,
    # where it is absent, which is how the parser-only probe passed review here.
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
        # References no URI parser, which is the 1.15 and 0.x shape.
        raise AssertionError("never called")

    monkeypatch.setattr(
        saving_utils.HfFileSystem, "resolve_path", resolver_without_the_parser,
        raising = True,
    )
    assert saving_utils._hub_addresses_typed_model_uris() is False, (
        "a parser that maps the prefix does not mean the resolver reaches it"
    )

    def resolver_with_the_parser(self, path, revision = None):
        # The probe reads the code object, so the name has to be referenced for
        # real; a mention in a comment must not count, and once did.
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
    """With the capability forced on, a path that merely begins with `models` stays a
    path. `parse_hf_uri` itself would read `models/a/b/c` as repo `a/b` plus file `c`,
    and `/abs/base` as repo `abs/base`, which is why this file does not delegate its
    path rules to that parser."""
    monkeypatch.setattr(saving_utils, "_hub_addresses_typed_model_uris", lambda: True)
    assert saving_utils._is_hub_repo_id(name) is False


@pytest.mark.parametrize("name", [
    "datasets/stanfordnlp/imdb", "spaces/org/demo", "kernels/org/k",
])
def test_repo_types_that_cannot_be_a_base_model_stay_rejected(monkeypatch, name):
    """These resolve too, and are rejected deliberately: a dataset, a Space and a
    kernel are not the base model of a LoRA merge, so a Hub probe for one is a
    round trip that can only answer a question nobody asked."""
    monkeypatch.setattr(saving_utils, "_hub_addresses_typed_model_uris", lambda: True)
    assert saving_utils._is_hub_repo_id(name) is False


@pytest.mark.parametrize("save_method", ["merged_4bit", "forced_merged_4bit", "merged_16bit", None])
def test_an_at_sign_in_a_local_directory_changes_nothing(monkeypatch, tmp_path, save_method):
    """The argument the revision split rests on, which nothing else pins.

    Splitting on `@` means a local directory named `outputs/my@model` is now offered
    to the Hub as the repo id `outputs/my`. That sounds like it could turn a local
    base into a Hub probe, so the question is whether it differs from an ordinary
    one-slash directory name, and it does not: `outputs/mymodel` is already a valid
    `namespace/name` shape, so it was already probed both before and after the
    split. Asserted as an equivalence rather than as a fixed outcome, so it keeps
    holding if the surrounding policy for local bases changes.

    With every Hub entry point dead, both names must answer identically, which for a
    local nf4 base means the 4bit merges resolve locally and the 16bit ones raise.
    """
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
    """1.25.1 names the whole `gpt2@main` in the rejection, so the classifier has to
    normalise before counting segments or it calls this a transport failure."""
    message = _REJECTION_MESSAGE.replace("gpt2", "gpt2@main", 1)
    _patch_ls(monkeypatch, ValueError(message))
    assert saving_utils.check_hf_model_exists(name) is False


def test_rejection_message_on_a_namespaced_name_still_raises(monkeypatch):
    """Scoping stays: `namespace/name` is addressable on every supported
    version, so nothing about it is an id rejection."""
    _patch_ls(monkeypatch, ValueError(_REJECTION_MESSAGE))
    with pytest.raises(RuntimeError):
        saving_utils.check_hf_model_exists("openai-community/gpt2")


def test_a_bare_valueerror_with_no_rejection_wording_always_raises(monkeypatch):
    """The regression a version test cannot catch, on every supported release.

    The guard carries its own wording, so recognising it by that wording is exact
    wherever it exists. A bare ValueError that does NOT carry it was not raised by
    the guard, which leaves a transport failure as the only thing it can be.

    A `major >= 1` gate got this wrong for the entire 1.0 - 1.15 range, where no
    guard exists and the network IS reached: such a ValueError came off a live
    socket and was still answered False, so `determine_base_model_source` fell
    through every priority and the merge wrote nothing. That is the exact defect
    this file exists to pin, reached through a different door.
    """
    _patch_ls(monkeypatch, ValueError("boom"))
    with pytest.raises(RuntimeError) as excinfo:
        saving_utils.check_hf_model_exists("gpt2")
    message = str(excinfo.value)
    assert "connectivity" in message or "rate limiting" in message


def test_classification_never_consults_the_installed_version(monkeypatch):
    """Both answers are decided by the message alone, whatever is installed.

    Pinned by identity as well as by behaviour: the previous implementation asked
    `huggingface_hub.__version__`, and the test that covered it recomputed the
    same expression, so the two agreed with each other rather than with upstream
    and the 1.0 - 1.15 hole stayed invisible.
    """
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
    """HfUriError subclasses ValueError but is an id error, and it is matched by
    type in the absent set, so it never reaches the transport branch."""
    _patch_ls(monkeypatch, HfUriError("hf://gpt2", "Invalid HF URI 'hf://gpt2'"))
    assert saving_utils.check_hf_model_exists("gpt2") is False


# ---------------------------------------------------------------------------
# No previously working input regressed: True stays True.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["gpt2", "bert-base-uncased", "distilgpt2"])
def test_listable_single_segment_id_still_returns_true(monkeypatch, name):
    """0.36.2 lists `gpt2` happily (17 entries, safetensors among them). The
    narrowing is in the except branch only, so a successful listing is
    untouched and no True became False."""
    _patch_ls(monkeypatch, [
        {"name": f"{name}/model.safetensors"},
        {"name": f"{name}/config.json"},
    ])
    assert saving_utils.check_hf_model_exists(name) is True


def test_absent_single_segment_id_still_returns_false(monkeypatch):
    """0.x answers NotImplementedError for a slashless name with no canonical
    repo behind it. Still absent, still no exception."""
    _patch_ls(monkeypatch, NotImplementedError("Access to repositories lists is not implemented."))
    assert saving_utils.check_hf_model_exists("definitely-not-a-real-model-xyz") is False


def test_offline_mode_on_a_single_segment_id_raises(monkeypatch):
    _patch_ls(monkeypatch, OfflineModeIsEnabled("Cannot reach https://huggingface.co"))
    with pytest.raises(RuntimeError):
        saving_utils.check_hf_model_exists("gpt2")


# ---------------------------------------------------------------------------
# End to end over real HTTP: a proxy answering 200 with a non-JSON body.
# ---------------------------------------------------------------------------

class _GarbageHandler(http.server.BaseHTTPRequestHandler):
    """Answers every request 200 with HTML, which is what a captive portal or a
    misconfigured corporate proxy does to an API call."""

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
    """The reproduction with no stubbed exception anywhere: a real socket, a
    real HTTP 200, a real body the client cannot parse."""
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
