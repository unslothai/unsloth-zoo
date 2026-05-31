"""Tests for convert_to_gguf's self-heal on a broken converter environment.

A stale or missing converter package (usually `gguf`) makes the converter
subprocess exit 1. convert_to_gguf should reinstall the deps into the
converter's own interpreter and retry once, surface the real traceback if the
failure persists, and never reinstall on a genuine model error.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import textwrap
import types
from pathlib import Path

import pytest


def _load_llama_cpp_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "unsloth_zoo" / "llama_cpp.py"
    spec = importlib.util.spec_from_file_location("llama_cpp_under_test_self_heal", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))
    return model_dir


def _write_converter(tmp_path: Path, body: str) -> Path:
    fake = tmp_path / "fake_convert.py"
    fake.write_text("import sys, argparse, os\n" + textwrap.dedent(body))
    return fake


def _convert(mod, tmp_path, converter):
    return mod.convert_to_gguf(
        model_name="llama-3.2-3b-instruct",
        input_folder=str(_make_model_dir(tmp_path)),
        quantization_type="bf16",
        converter_location=str(converter),
        print_output=False,
    )


def test_stale_package_self_heals(tmp_path, monkeypatch):
    mod = _load_llama_cpp_module()
    monkeypatch.chdir(tmp_path)

    # Fail with an ImportError until a "healed" marker exists, then succeed.
    converter = _write_converter(tmp_path, '''
        p = argparse.ArgumentParser()
        p.add_argument("--outfile"); p.add_argument("--outtype"); p.add_argument("--split-max-size")
        a, _ = p.parse_known_args()
        if os.path.exists("healed"):
            open(a.outfile, "wb").write(b"GGUF"); sys.exit(0)
        sys.stderr.write("ImportError: cannot import name 'GGUFWriter' from 'gguf'\\n")
        sys.exit(1)
    ''')

    calls = {"n": 0}
    def fake_reinstall(python_exe, print_output=False):
        calls["n"] += 1
        open("healed", "w").close()
        return types.SimpleNamespace(returncode=0, stdout="reinstalled gguf")
    monkeypatch.setattr(mod, "_reinstall_converter_deps", fake_reinstall)

    files, _is_vlm = _convert(mod, tmp_path, converter)
    assert calls["n"] == 1
    assert files and os.path.exists(files[0])


def test_reinstall_failure_surfaces_error(tmp_path, monkeypatch):
    mod = _load_llama_cpp_module()
    monkeypatch.chdir(tmp_path)

    converter = _write_converter(tmp_path, '''
        sys.stderr.write("ModuleNotFoundError: No module named 'gguf'\\n")
        sys.exit(1)
    ''')
    monkeypatch.setattr(
        mod, "_reinstall_converter_deps",
        lambda python_exe, print_output=False: types.SimpleNamespace(returncode=1, stdout="ERROR: offline"),
    )

    with pytest.raises(RuntimeError) as excinfo:
        _convert(mod, tmp_path, converter)
    msg = str(excinfo.value)
    assert "No module named 'gguf'" in msg
    assert "dependency reinstall failed" in msg


def test_genuine_model_error_not_repaired(tmp_path, monkeypatch):
    mod = _load_llama_cpp_module()
    monkeypatch.chdir(tmp_path)

    converter = _write_converter(tmp_path, '''
        sys.stderr.write("NotImplementedError: Architecture FooForCausalLM not supported\\n")
        sys.exit(1)
    ''')
    calls = {"n": 0}
    def spy(python_exe, print_output=False):
        calls["n"] += 1
        return types.SimpleNamespace(returncode=0, stdout="")
    monkeypatch.setattr(mod, "_reinstall_converter_deps", spy)

    with pytest.raises(RuntimeError) as excinfo:
        _convert(mod, tmp_path, converter)
    assert "Architecture FooForCausalLM not supported" in str(excinfo.value)
    assert calls["n"] == 0
