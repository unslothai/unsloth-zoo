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

"""convert_to_gguf must not torch.load (MTP inspection) or hand to the converter (which torch.loads it in the same
interpreter) a `.bin`-only checkpoint on torch < 2.6, where weights_only=True still runs code (CVE-2025-32434)."""

import importlib.util
import json
import sys
import textwrap
import types
from pathlib import Path

import pytest


@pytest.fixture(scope = "module")
def llama_cpp():
    path = Path(__file__).resolve().parents[2] / "unsloth_zoo" / "llama_cpp.py"
    spec = importlib.util.spec_from_file_location("llama_cpp_under_test_bin_old_torch", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _converter(tmp_path):
    log = tmp_path / "ran.txt"
    script = tmp_path / "fake_convert.py"
    script.write_text(textwrap.dedent(f"""
        import argparse
        from pathlib import Path
        p = argparse.ArgumentParser()
        p.add_argument("--outfile"); p.add_argument("--outtype"); p.add_argument("--split-max-size")
        p.add_argument("--no-mtp", action="store_true"); p.add_argument("--mmproj", action="store_true")
        p.add_argument("model_dir")
        a = p.parse_args()
        Path({str(log)!r}).write_text("ran")
        Path(a.outfile).write_bytes(b"GGUF")
    """), encoding = "utf-8")
    return script, log


def _model(tmp_path, mtp, safetensors = False):
    d = tmp_path / "model"
    d.mkdir()
    cfg = {"architectures": ["Qwen3_5ForCausalLM"], "num_hidden_layers": 24}
    if mtp:
        cfg["mtp_num_hidden_layers"] = 1
    (d / "config.json").write_text(json.dumps(cfg), encoding = "utf-8")
    (d / "pytorch_model.bin").write_bytes(b"not a real pickle")
    if safetensors:
        (d / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"model.layers.0.w": "model-00001-of-00001.safetensors"}}), encoding = "utf-8")
        (d / "model-00001-of-00001.safetensors").touch()
    return d


def _fake_torch(version, loads):
    def load(*args, **kwargs):
        loads.append(args[0])
        return {"model.layers.0.w": None}
    return types.SimpleNamespace(__version__ = version, load = load)


@pytest.mark.parametrize("mtp", (False, True))
def test_old_torch_bin_only_is_refused_before_any_load(llama_cpp, tmp_path, monkeypatch, mtp):
    loads = []
    monkeypatch.setattr(llama_cpp, "torch", _fake_torch("2.5.1", loads))
    script, ran = _converter(tmp_path)
    with pytest.raises(ValueError, match = "cannot load one safely"):
        llama_cpp.convert_to_gguf(
            model_name = str(tmp_path / "out"), input_folder = str(_model(tmp_path, mtp)),
            converter_location = str(script), quantization_type = "bf16",
        )
    assert loads == [], "MTP inspection torch.load ran on torch < 2.6"
    assert not ran.exists(), "the converter (which torch.loads the .bin) ran on torch < 2.6"


@pytest.mark.parametrize("version,safetensors", (("2.6.0", False), ("2.5.1", True)))
def test_new_torch_or_safetensors_unchanged(llama_cpp, tmp_path, monkeypatch, version, safetensors):
    loads = []
    monkeypatch.setattr(llama_cpp, "torch", _fake_torch(version, loads))
    script, ran = _converter(tmp_path)
    llama_cpp.convert_to_gguf(
        model_name = str(tmp_path / "out"), input_folder = str(_model(tmp_path, False, safetensors)),
        converter_location = str(script), quantization_type = "bf16",
    )
    assert ran.exists()
