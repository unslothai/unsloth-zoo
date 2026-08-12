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
holding a token broader than the request, a hosted service merging on behalf of a user,
would otherwise spend that token on the rewritten name and fold weights the requester
could never have fetched into the merged output.

Public and gated siblings keep resolving, so every normal merge is unchanged. Only a
*private* sibling, one that exists solely because of the token, is dropped, and then
the merge dequantizes the FP8 weights it was actually given.
"""

import json
import warnings

import pytest
from huggingface_hub.errors import GatedRepoError

from unsloth_zoo import saving_utils


PRIVATE_ENV = "UNSLOTH_ALLOW_PRIVATE_FP8_SIBLING"


def _repo_not_found(path):
    """What a private (or absent) repo looks like coming out of `HfFileSystem.ls`:
    fsspec's `_raise_file_not_found` converts the 401 into a plain FileNotFoundError."""
    return FileNotFoundError(f"{path} (repository not found)")


def _gated(path):
    """A gated repo arrives the same way, with the real cause chained underneath."""
    error = FileNotFoundError(f"{path} (gated)")
    error.__cause__ = GatedRepoError(f"403 for {path}")
    return error


def _hub(monkeypatch, tmp_path, *, anonymous):
    """One repo on the Hub, `unsloth/GLM-5.2`, unquantized. `anonymous` is what a
    token free listing of it does: return files (public), or raise."""
    def fake_ls(self, path, detail = True, **kwargs):
        if not self.token and anonymous is not None:
            raise anonymous(path)
        return [{"name": f"{path}/model.safetensors"}]
    monkeypatch.setattr(saving_utils.HfFileSystem, "ls", fake_ls, raising = True)

    config = tmp_path / "hub-config.json"
    config.write_text(json.dumps({"model_type": "llama"}), encoding = "utf-8")
    import huggingface_hub
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download",
        lambda *args, **kwargs: str(config), raising = True,
    )


def _resolve(monkeypatch, tmp_path):
    """No local copy of anything, so only the Hub branch can answer."""
    monkeypatch.chdir(tmp_path)
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        sibling = saving_utils._resolve_fp8_16bit_sibling(
            "unsloth/GLM-5.2-FP8", token = "SERVICE_TOKEN",
        )
    return sibling, [str(w.message) for w in caught]


@pytest.fixture(autouse = True)
def _no_opt_in(monkeypatch):
    monkeypatch.delenv(PRIVATE_ENV, raising = False)


def test_a_private_sibling_is_not_merged_onto(monkeypatch, tmp_path):
    """The token can read `unsloth/GLM-5.2`, nobody else can, and the caller asked for
    `unsloth/GLM-5.2-FP8`. Resolving it would put weights the requester has no access to
    into the output, so the merge stays on the FP8 base it was given."""
    _hub(monkeypatch, tmp_path, anonymous = _repo_not_found)

    sibling, messages = _resolve(monkeypatch, tmp_path)

    assert sibling is None
    assert any("private 16bit sibling" in m for m in messages), messages
    assert any(PRIVATE_ENV in m for m in messages), messages


def test_a_private_sibling_resolves_when_explicitly_opted_in(monkeypatch, tmp_path):
    """Single tenant callers who own both repos are not blocked, they just have to say so."""
    _hub(monkeypatch, tmp_path, anonymous = _repo_not_found)
    monkeypatch.setenv(PRIVATE_ENV, "1")

    sibling, messages = _resolve(monkeypatch, tmp_path)

    assert sibling == "unsloth/GLM-5.2"
    assert [m for m in messages if "private 16bit sibling" in m] == []


def test_a_public_sibling_still_resolves(monkeypatch, tmp_path):
    """The overwhelmingly common case, `unsloth/GLM-5.2-FP8` -> `unsloth/GLM-5.2`,
    is untouched: no warning, and the 16bit sibling is still the merge base."""
    _hub(monkeypatch, tmp_path, anonymous = None)

    sibling, messages = _resolve(monkeypatch, tmp_path)

    assert sibling == "unsloth/GLM-5.2"
    assert [m for m in messages if isinstance(m, str) and "sibling" in m] == [], messages


def test_a_gated_sibling_still_resolves(monkeypatch, tmp_path):
    """Gated is public-but-licensed: the repo is listed for everyone and only the
    download is gated, so it discloses nothing private and `meta-llama/...-FP8` ->
    `meta-llama/...` keeps working."""
    _hub(monkeypatch, tmp_path, anonymous = _gated)

    sibling, messages = _resolve(monkeypatch, tmp_path)

    assert sibling == "unsloth/GLM-5.2"
    assert [m for m in messages if "private 16bit sibling" in m] == []


def test_the_visibility_probe_never_sends_the_callers_token(monkeypatch, tmp_path):
    """The whole gate rests on the probe being anonymous; `token = None` would pick the
    ambient `HF_TOKEN` back up out of the environment and answer "public" about a repo
    only the service can see."""
    seen = []
    def fake_ls(self, path, detail = True, **kwargs):
        seen.append(self.token)
        return [{"name": f"{path}/model.safetensors"}]
    monkeypatch.setattr(saving_utils.HfFileSystem, "ls", fake_ls, raising = True)

    assert saving_utils._hub_repo_reads_without_a_token("unsloth/GLM-5.2") is True
    assert seen == [False], seen
