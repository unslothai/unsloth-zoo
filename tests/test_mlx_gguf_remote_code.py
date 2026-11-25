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

"""MLX GGUF export, like CPU, lets the llama.cpp converter import auto_map code only if loaded with
trust_remote_code=True; runs on the mlx-on-torch shim (no Apple hardware)."""

import ast
import os
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True, scope="module")
def _install_mlx_torch_shim():
    pytest.importorskip("torch")
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


def _export(monkeypatch, tmp_path, model, tokenizer):
    tmp_path = tmp_path / str(len(list(tmp_path.iterdir())))
    tmp_path.mkdir()
    import test_mlx_save_export_regressions as regressions
    import unsloth_zoo.llama_cpp as llama_cpp
    import unsloth_zoo.mlx.utils as mutils
    assert Path(mutils.__file__).resolve() == ROOT / "unsloth_zoo" / "mlx" / "utils.py"  # this arm's own tree

    regressions._stub_gguf_export(monkeypatch, tmp_path)
    recorded = llama_cpp.convert_to_gguf
    seen = []

    def convert_to_gguf(**kwargs):
        seen.append(os.environ.get("HF_MODULES_CACHE"))
        return recorded(**kwargs)

    monkeypatch.setattr(llama_cpp, "convert_to_gguf", convert_to_gguf)
    monkeypatch.setenv("HF_MODULES_CACHE", "/user/modules")
    mutils.save_pretrained_gguf(model, tokenizer, tmp_path / "out", quantization_method = "q8_0")
    assert os.environ["HF_MODULES_CACHE"] == "/user/modules"  # restored after the export
    return seen


def test_mlx_converter_cannot_import_auto_map_code_the_load_never_approved(monkeypatch, tmp_path):
    model = types.SimpleNamespace(_hf_repo = "org/Model")
    seen = _export(monkeypatch, tmp_path, model, object())
    assert len(seen) == 1 and seen[0].endswith("no_remote_code"), seen


def test_mlx_load_that_approved_remote_code_keeps_it_for_the_converter(monkeypatch, tmp_path):
    stamped = types.SimpleNamespace(_hf_repo = "org/Model", _unsloth_trust_remote_code = True)
    assert _export(monkeypatch, tmp_path, stamped, object()) == ["/user/modules"]
    custom = type("GLMTokenizer", (), {"__module__": "transformers_modules.org.repo.tokenization"})()
    wrapper = types.SimpleNamespace(_tokenizer = custom)  # mlx-lm's TokenizerWrapper
    assert _export(monkeypatch, tmp_path, types.SimpleNamespace(_hf_repo = "org/Model"), wrapper) == ["/user/modules"]


def test_mlx_loader_stamps_trust_remote_code_on_every_load():
    from unsloth_zoo.mlx import loader
    model = types.SimpleNamespace()
    loader._stamp_remote_code_trust(model)
    assert not hasattr(model, "_unsloth_trust_remote_code")
    token = loader._MLX_MODEL_FILE_TRUST.set(True)
    try:
        loader._stamp_remote_code_trust(model)
    finally:
        loader._MLX_MODEL_FILE_TRUST.reset(token)
    assert model._unsloth_trust_remote_code is True
    # _finish_load is the single exit of FastMLXModel.from_pretrained, so every load is stamped.
    tree = ast.parse((ROOT / "unsloth_zoo" / "mlx" / "loader.py").read_text(encoding = "utf-8"))
    finish = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_finish_load")
    calls = {n.func.id for n in ast.walk(finish) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "_stamp_remote_code_trust" in calls
