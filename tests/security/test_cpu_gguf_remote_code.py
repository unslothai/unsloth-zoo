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

import os
import types
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "unsloth_zoo" / "cpu" / "loader.py"


def _export(monkeypatch, tmp_path, model, tokenizer = None):
    pytest.importorskip("torch")
    from unsloth_zoo.cpu import loader
    import unsloth_zoo.llama_cpp as llama_cpp
    assert Path(loader.__file__).resolve() == SRC  # this arm's own tree

    seen = []

    def convert_to_gguf(output, merged, **kwargs):
        seen.append(os.environ.get("HF_MODULES_CACHE"))
        return [f"{output}.{kwargs['quantization_type'].upper()}.gguf"], None

    monkeypatch.setattr(loader, "_save_merged", lambda *a, **k: None)
    monkeypatch.setattr(llama_cpp, "install_llama_cpp", lambda **k: ("quantize", "convert"))
    monkeypatch.setattr(llama_cpp, "convert_to_gguf", convert_to_gguf)
    monkeypatch.setattr(llama_cpp, "quantize_gguf", lambda *a, **k: None)
    monkeypatch.setenv("HF_MODULES_CACHE", "/user/modules")
    loader.save_pretrained_gguf(model, str(tmp_path / "out"), tokenizer, quantization_method = ["q8_0", "f16"])
    assert os.environ["HF_MODULES_CACHE"] == "/user/modules"  # restored after the export
    return seen


def test_converter_cannot_import_auto_map_code_the_load_never_approved(monkeypatch, tmp_path):
    seen = _export(monkeypatch, tmp_path, types.SimpleNamespace())
    assert len(seen) == 2 and all(path.endswith("no_remote_code") for path in seen)


def test_a_load_that_ran_remote_code_keeps_it_for_the_converter(monkeypatch, tmp_path):
    custom = type(
        "GLMTokenizer", (), {"__module__": "transformers_modules.org.repo.tokenization", "save_pretrained": lambda self, d: None}
    )()
    assert _export(monkeypatch, tmp_path, types.SimpleNamespace(), custom) == ["/user/modules"] * 2
    stamped = types.SimpleNamespace(_unsloth_trust_remote_code = True)
    assert _export(monkeypatch, tmp_path, stamped) == ["/user/modules"] * 2
