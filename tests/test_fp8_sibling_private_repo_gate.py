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

"""Sibling lookup must be anonymous, never token = None (which reads ambient HF_TOKEN); gated/private siblings are
dropped so caller tokens cannot widen what they reach."""

import json
import warnings
from types import SimpleNamespace

import pytest
from huggingface_hub.errors import EntryNotFoundError, GatedRepoError

from unsloth_zoo import saving_utils


RESTRICTED_ENV = "UNSLOTH_ALLOW_RESTRICTED_FP8_SIBLING"
SERVICE_TOKEN  = "SERVICE_TOKEN"


def _gated_repo_error(message):
    """GatedRepoError under both signatures: hf_hub 1.x needs a required keyword response that it reads .headers and
    .request from; 0.x defaults it to None; project allows >=0.34.0."""
    response = SimpleNamespace(headers = {}, request = None)
    try:
        return GatedRepoError(message, response = response)
    except TypeError:
        return GatedRepoError(message)


def _repo_not_found(path):
    """For an anonymous reader a private repo is absent, not forbidden: fsspec raises FileNotFoundError."""
    return FileNotFoundError(f"{path} (repository not found)")


def _hub(monkeypatch, tmp_path, *, anonymous_head, listed = True):
    """Fake unsloth/GLM-5.2: anonymous_head None means public, else the HEAD error; downloads always succeed, even
    anonymously, mimicking the hf_hub_download cache fallback, so only the HEAD decides."""
    seen = SimpleNamespace(list_tokens = [], download_tokens = [], head_tokens = [])

    def fake_ls(self, path, detail = True, **kwargs):
        seen.list_tokens.append(self.token)
        if not listed:
            raise _repo_not_found(path)
        return [{"name": f"{path}/model.safetensors"}]
    monkeypatch.setattr(saving_utils.HfFileSystem, "ls", fake_ls, raising = True)

    config = tmp_path / "hub-config.json"
    config.write_text(json.dumps({"model_type": "llama"}), encoding = "utf-8")

    import huggingface_hub
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download",
        lambda *a, **kw: (seen.download_tokens.append(kw.get("token")), str(config))[1],
        raising = True,
    )

    def fake_head(url, token = None, **kwargs):
        seen.head_tokens.append(token)
        if anonymous_head is not None:
            raise anonymous_head(url)
        return None
    monkeypatch.setattr(
        huggingface_hub, "get_hf_file_metadata", fake_head, raising = True,
    )
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
    """Public sibling (unsloth/GLM-5.2-FP8 to unsloth/GLM-5.2) resolves silently as the merge base."""
    _hub(monkeypatch, tmp_path, anonymous_head = None)

    sibling, messages = _resolve(monkeypatch, tmp_path)

    assert sibling == "unsloth/GLM-5.2"
    assert [m for m in messages if "sibling" in m] == [], messages


def test_a_gated_sibling_is_not_merged_onto(monkeypatch, tmp_path):
    """A gated repo lists publicly, but only the token has accepted its terms; merging would expose it."""
    _hub(monkeypatch, tmp_path, anonymous_head = lambda url: _gated_repo_error(f"403 {url}"))

    sibling, messages = _resolve(monkeypatch, tmp_path)

    assert sibling is None
    assert any("gated or private" in m for m in messages), messages
    assert any(RESTRICTED_ENV in m for m in messages), messages


def test_a_private_sibling_is_not_merged_onto(monkeypatch, tmp_path):
    """Private repos are absent to anonymous readers, so the lookup says no sibling without the token."""
    seen = _hub(monkeypatch, tmp_path, anonymous_head = _repo_not_found, listed = False)

    sibling, _ = _resolve(monkeypatch, tmp_path)

    assert sibling is None
    assert SERVICE_TOKEN not in seen.list_tokens + seen.download_tokens


def test_a_restricted_sibling_resolves_when_explicitly_opted_in(monkeypatch, tmp_path):
    """Callers with access to both opt in via RESTRICTED_ENV; only then is the token used on the rewrite."""
    seen = _hub(monkeypatch, tmp_path, anonymous_head = _repo_not_found)
    monkeypatch.setenv(RESTRICTED_ENV, "1")

    sibling, messages = _resolve(monkeypatch, tmp_path)

    assert sibling == "unsloth/GLM-5.2"
    assert [m for m in messages if "gated or private" in m] == []
    assert SERVICE_TOKEN in seen.list_tokens


def test_the_lookup_never_spends_the_callers_token_on_the_rewritten_name(monkeypatch, tmp_path):
    """No request for unsloth/GLM-5.2 carries the token, so the rewrite cannot reveal its existence or cache its config."""
    seen = _hub(monkeypatch, tmp_path, anonymous_head = None)

    sibling, _ = _resolve(monkeypatch, tmp_path)

    assert sibling == "unsloth/GLM-5.2"
    assert seen.list_tokens and seen.download_tokens, "the Hub was never asked"
    assert all(not t for t in seen.list_tokens + seen.download_tokens), seen


def test_an_unreachable_hub_is_not_reported_as_a_restricted_sibling(monkeypatch, tmp_path):
    """A 429 or proxy error on anonymous read says nothing about access; never report it as gated/private."""
    _hub(monkeypatch, tmp_path,
         anonymous_head = lambda url: TimeoutError(f"read timed out for {url}"))

    sibling, messages = _resolve(monkeypatch, tmp_path)

    assert sibling is None
    assert any("could not check the Hugging Face Hub" in m for m in messages), messages
    assert [m for m in messages if "gated or private" in m] == []


def test_a_warm_cache_cannot_make_a_gated_sibling_look_public(monkeypatch, tmp_path):
    """Gated check must use HEAD, not download: hf_hub_download falls back to the cache when HEAD fails."""
    seen = _hub(monkeypatch, tmp_path,
                anonymous_head = lambda url: _gated_repo_error(f"403 {url}"))

    sibling, messages = _resolve(monkeypatch, tmp_path)

    assert sibling is None
    assert any("gated or private" in m for m in messages), messages
    assert seen.head_tokens and all(not t for t in seen.head_tokens), seen.head_tokens


def test_a_sibling_that_ships_no_config_json_still_resolves(monkeypatch, tmp_path):
    """A 404 on config.json still proves anonymous read access; a missing config reads as unquantized."""
    _hub(monkeypatch, tmp_path,
         anonymous_head = lambda url: EntryNotFoundError(f"404 no config.json at {url}"))

    sibling, messages = _resolve(monkeypatch, tmp_path)

    assert sibling == "unsloth/GLM-5.2"
    assert [m for m in messages if "gated or private" in m] == [], messages
