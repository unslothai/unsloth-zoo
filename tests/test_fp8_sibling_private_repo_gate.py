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

"""The FP8 16bit sibling lookup must not widen what the caller's token reaches.

`_resolve_fp8_16bit_sibling` rewrites the requested repo id (`org/model-FP8` ->
`org/model`), so the repo it lands on is not the one the caller asked for. A caller
holding a token broader than the request, a service merging on behalf of a user, would
otherwise spend that token on the rewritten name and fold weights the requester could
never have fetched into the merged output. `token = None` makes that easy to hit by
accident: huggingface_hub reads it as "go find one" and picks up the ambient HF_TOKEN.

So the sibling is resolved with no credentials at all, and a sibling that only a token
can reach is dropped. The merge then dequantizes the FP8 weights it was actually given,
which is still a correct 16bit merge.

Listing is not the test for readability: a gated repo lists publicly and still refuses
its content anonymously (measured on `meta-llama/Llama-3.2-1B`, anonymous `ls` returns
the file list and the anonymous `config.json` fetch answers 401 GatedRepoError), so
these tests drive the content fetch, not just `ls`.
"""

import json
import warnings
from types import SimpleNamespace

import pytest
from huggingface_hub.errors import GatedRepoError

from unsloth_zoo import saving_utils


RESTRICTED_ENV = "UNSLOTH_ALLOW_RESTRICTED_FP8_SIBLING"
SERVICE_TOKEN  = "SERVICE_TOKEN"


def _gated_repo_error(message):
    """A `GatedRepoError` under either supported signature. huggingface_hub 1.x makes
    `response` keyword only AND required, and reads `.headers` / `.request` off it in
    `__init__`; 0.x defaults it to None. The project allows both (`>=0.34.0`)."""
    response = SimpleNamespace(headers = {}, request = None)
    try:
        return GatedRepoError(message, response = response)
    except TypeError:
        return GatedRepoError(message)


def _repo_not_found(path):
    """What a private repo looks like to an anonymous reader: the Hub reports it absent
    rather than admitting it exists, and fsspec flattens that to FileNotFoundError."""
    return FileNotFoundError(f"{path} (repository not found)")


def _hub(monkeypatch, tmp_path, *, anonymous_content, listed = True):
    """One repo on the Hub, `unsloth/GLM-5.2`, unquantized, and a record of every
    request made against it.

    `anonymous_content` is what a credential free CONTENT fetch does: `None` serves the
    config (public), or a callable returning the error it raises. A token bearing fetch
    always succeeds, which is what makes "did we need the token?" observable.
    """
    seen = SimpleNamespace(list_tokens = [], download_tokens = [])

    def fake_ls(self, path, detail = True, **kwargs):
        seen.list_tokens.append(self.token)
        if not listed:
            raise _repo_not_found(path)
        return [{"name": f"{path}/model.safetensors"}]
    monkeypatch.setattr(saving_utils.HfFileSystem, "ls", fake_ls, raising = True)

    config = tmp_path / "hub-config.json"
    config.write_text(json.dumps({"model_type": "llama"}), encoding = "utf-8")

    def fake_download(*args, **kwargs):
        token = kwargs.get("token")
        seen.download_tokens.append(token)
        if not token and anonymous_content is not None:
            raise anonymous_content(kwargs.get("repo_id"))
        return str(config)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download, raising = True)
    return seen


def _resolve(monkeypatch, tmp_path):
    """No local copy of anything, so only the Hub branch can answer."""
    monkeypatch.chdir(tmp_path)
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        sibling = saving_utils._resolve_fp8_16bit_sibling(
            "unsloth/GLM-5.2-FP8", token = SERVICE_TOKEN,
        )
    return sibling, [str(w.message) for w in caught]


@pytest.fixture(autouse = True)
def _no_opt_in(monkeypatch):
    monkeypatch.delenv(RESTRICTED_ENV, raising = False)


def test_a_public_sibling_still_resolves(monkeypatch, tmp_path):
    """The overwhelmingly common case, `unsloth/GLM-5.2-FP8` -> `unsloth/GLM-5.2`, is
    untouched: no warning, and the 16bit sibling is still the merge base."""
    _hub(monkeypatch, tmp_path, anonymous_content = None)

    sibling, messages = _resolve(monkeypatch, tmp_path)

    assert sibling == "unsloth/GLM-5.2"
    assert [m for m in messages if "sibling" in m] == [], messages


def test_a_gated_sibling_is_not_merged_onto(monkeypatch, tmp_path):
    """A gated repo lists publicly, so `ls` alone would wave it through, but only the
    token has accepted its terms. Resolving it would redistribute access controlled
    weights to a requester who cannot download them."""
    _hub(monkeypatch, tmp_path, anonymous_content = lambda p: _gated_repo_error(f"403 {p}"))

    sibling, messages = _resolve(monkeypatch, tmp_path)

    assert sibling is None
    assert any("gated or private" in m for m in messages), messages
    assert any(RESTRICTED_ENV in m for m in messages), messages


def test_a_private_sibling_is_not_merged_onto(monkeypatch, tmp_path):
    """A private repo is simply absent to an anonymous reader, so the lookup answers
    "no sibling" without ever asking the token about it."""
    seen = _hub(monkeypatch, tmp_path, anonymous_content = _repo_not_found, listed = False)

    sibling, _ = _resolve(monkeypatch, tmp_path)

    assert sibling is None
    assert SERVICE_TOKEN not in seen.list_tokens + seen.download_tokens


def test_a_restricted_sibling_resolves_when_explicitly_opted_in(monkeypatch, tmp_path):
    """Callers who hold access to both repos are not blocked, they just have to say so,
    and only then does the token go anywhere near the rewritten name."""
    seen = _hub(monkeypatch, tmp_path, anonymous_content = _repo_not_found)
    monkeypatch.setenv(RESTRICTED_ENV, "1")

    sibling, messages = _resolve(monkeypatch, tmp_path)

    assert sibling == "unsloth/GLM-5.2"
    assert [m for m in messages if "gated or private" in m] == []
    assert SERVICE_TOKEN in seen.list_tokens


def test_the_lookup_never_spends_the_callers_token_on_the_rewritten_name(monkeypatch, tmp_path):
    """The property the whole gate exists for. Not just "the weights are not downloaded":
    no request about `unsloth/GLM-5.2` carries the token at all, so the rewrite cannot
    disclose the repo's existence or cache its config either."""
    seen = _hub(monkeypatch, tmp_path, anonymous_content = None)

    sibling, _ = _resolve(monkeypatch, tmp_path)

    assert sibling == "unsloth/GLM-5.2"
    assert seen.list_tokens and seen.download_tokens, "the Hub was never asked"
    assert all(not t for t in seen.list_tokens + seen.download_tokens), seen


def test_an_unreachable_hub_is_not_reported_as_a_restricted_sibling(monkeypatch, tmp_path):
    """A 429 or a proxy error on the anonymous read says nothing about who may read the
    repo. Answering "gated or private" there would send users chasing an access problem
    they do not have."""
    _hub(monkeypatch, tmp_path,
         anonymous_content = lambda p: TimeoutError(f"read timed out for {p}"))

    sibling, messages = _resolve(monkeypatch, tmp_path)

    assert sibling is None
    assert any("could not check the Hugging Face Hub" in m for m in messages), messages
    assert [m for m in messages if "gated or private" in m] == []
