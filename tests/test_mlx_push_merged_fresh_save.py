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

"""model.push_to_hub_merged(repo_id) on MLX must upload the current model, not a save left behind by
an earlier push; mlx/loader.py and mlx/utils.py import mlx, so they are read with ast."""

import ast
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "unsloth_zoo" / "mlx"


def _function(path, name, namespace):
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body = [node], type_ignores = []), str(path), "exec"), namespace)
    return namespace[name]


def _wire(monkeypatch, tmp_path):
    saves, uploads = [], []

    def save_merged_model(model, tokenizer, path, **kwargs):
        path = Path(path)
        path.mkdir(parents = True, exist_ok = True)
        (path / "model.safetensors.index.json").write_text("{}")
        (path / "model.safetensors").write_text(model.weights)
        saves.append(model.weights)

    class Api:
        def __init__(self, token = None):
            pass

        def upload_large_folder(self, folder_path, **kwargs):
            uploads.append((Path(folder_path) / "model.safetensors").read_text())

        upload_folder = upload_large_folder

    hub = types.ModuleType("huggingface_hub")
    hub.HfApi = Api
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    utils_ns = {
        "Path": Path,
        "save_merged_model": save_merged_model,
        "_ensure_hub_repo_visibility": lambda api, repo_id, private: None,
        "_PUSH_MERGED_DEFAULT_COMMIT_MESSAGE": "default",
        "_PUSH_MERGED_DEFAULT_COMMIT_DESCRIPTION": "default",
    }
    push = _function(ROOT / "utils.py", "push_to_hub_merged", utils_ns)
    pkg = types.ModuleType("fakemlxpkg")
    pkg.__path__ = []
    utils_mod = types.ModuleType("fakemlxpkg.utils")
    utils_mod.push_to_hub_merged = push
    monkeypatch.setitem(sys.modules, "fakemlxpkg", pkg)
    monkeypatch.setitem(sys.modules, "fakemlxpkg.utils", utils_mod)
    loader_ns = {"__name__": "fakemlxpkg.loader", "__package__": "fakemlxpkg", "_refuse_offloaded": lambda *a: None}
    method = _function(ROOT / "loader.py", "_mlx_push_to_hub_merged", loader_ns)
    monkeypatch.chdir(tmp_path)
    return method, saves, uploads


def test_second_push_uploads_the_current_model(monkeypatch, tmp_path):
    method, saves, uploads = _wire(monkeypatch, tmp_path)
    model = types.SimpleNamespace(weights = "first", _tokenizer = object())
    method(model, "me/model")
    model.weights = "trained more"
    method(model, "me/model")
    assert uploads == ["first", "trained more"]


def test_explicit_save_directory_is_reused(monkeypatch, tmp_path):
    method, saves, uploads = _wire(monkeypatch, tmp_path)
    model = types.SimpleNamespace(weights = "saved", _tokenizer = object())
    method(model, "me/model", save_directory = "out")
    model.weights = "not saved"
    method(model, "me/model", save_directory = "out")
    assert saves == ["saved"] and uploads == ["saved", "saved"]
