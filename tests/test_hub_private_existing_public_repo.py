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

"""create_repo(private=True, exist_ok=True) keeps an existing public repo public, so a private=True push must stop
before the model card or any weight is uploaded. Mocked HfApi / ModelCard: nothing reaches the Hub."""

from types import SimpleNamespace

import huggingface_hub
import pytest

from unsloth_zoo import saving_utils


def _setup(monkeypatch, existing_private):
    calls = []

    class FakeApi:
        def __init__(self, token = None):
            pass

        def create_repo(self, repo_id, private = None, exist_ok = False, **kw):
            calls.append(("create_repo", repo_id, private))
            return f"https://huggingface.co/{repo_id}"

        def repo_info(self, repo_id, repo_type = None, **kw):
            calls.append(("repo_info", repo_id))
            return SimpleNamespace(private = existing_private)

        def update_repo_settings(self, *a, **kw):
            calls.append(("update_repo_settings",))

    class FakeCard:
        def __init__(self, content):
            pass

        def push_to_hub(self, repo_id, **kw):
            calls.append(("card_push", repo_id))

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "ModelCard", FakeCard)
    model = SimpleNamespace(config = SimpleNamespace(_name_or_path = "unsloth/tiny", model_type = "llama"))
    return model, calls


def test_private_true_refuses_existing_public_repo(monkeypatch):
    model, calls = _setup(monkeypatch, existing_private = False)
    with pytest.raises(RuntimeError, match = "public"):
        saving_utils.create_huggingface_repo(model, "user/existing", private = True, token = "t")
    assert not any(c[0] == "card_push" for c in calls), calls
    assert not any(c[0] == "update_repo_settings" for c in calls), calls


@pytest.mark.parametrize("private, existing_private", [(True, True), (False, False), (None, False)])
def test_allowed_pushes_unchanged(monkeypatch, private, existing_private):
    model, calls = _setup(monkeypatch, existing_private = existing_private)
    saving_utils.create_huggingface_repo(model, "user/repo", private = private, token = "t")
    assert ("card_push", "user/repo") in calls
    assert not any(c[0] == "update_repo_settings" for c in calls), calls
