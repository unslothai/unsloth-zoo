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


def test_single_retry_no_infinite_loop(tmp_path, monkeypatch):
    # Reinstall "succeeds" but the converter stays broken: stop after one retry.
    mod = _load_llama_cpp_module()
    monkeypatch.chdir(tmp_path)
    converter = _write_converter(tmp_path, '''
        open("calls.txt", "a").write("x")
        sys.stderr.write("ImportError: still broken\\n")
        sys.exit(1)
    ''')
    calls = {"n": 0}
    def spy(python_exe, print_output=False):
        calls["n"] += 1
        return types.SimpleNamespace(returncode=0, stdout="")
    monkeypatch.setattr(mod, "_reinstall_converter_deps", spy)

    with pytest.raises(RuntimeError):
        _convert(mod, tmp_path, converter)
    assert calls["n"] == 1
    assert (tmp_path / "calls.txt").read_text() == "xx"  # original + one retry


def test_reinstall_raises_is_caught(tmp_path, monkeypatch):
    mod = _load_llama_cpp_module()
    monkeypatch.chdir(tmp_path)
    converter = _write_converter(tmp_path, 'sys.stderr.write("ImportError: x\\n"); sys.exit(1)\n')
    def boom(python_exe, print_output=False):
        raise RuntimeError("pip subprocess exploded")
    monkeypatch.setattr(mod, "_reinstall_converter_deps", boom)

    with pytest.raises(RuntimeError) as excinfo:
        _convert(mod, tmp_path, converter)
    assert "pip subprocess exploded" in str(excinfo.value)
    assert "ImportError: x" in str(excinfo.value)


def test_non_utf8_output_does_not_crash(tmp_path, monkeypatch):
    # Non-UTF8 bytes on stderr must surface as RuntimeError, not UnicodeDecodeError.
    mod = _load_llama_cpp_module()
    monkeypatch.chdir(tmp_path)
    converter = _write_converter(tmp_path, '''
        os.write(2, bytes([0x81, 0xff, 0xfe]))
        sys.stderr.flush()
        sys.stderr.write("\\nNotImplementedError: weird bytes\\n")
        sys.exit(1)
    ''')
    with pytest.raises(RuntimeError) as excinfo:
        _convert(mod, tmp_path, converter)
    assert "weird bytes" in str(excinfo.value)


def test_reinstall_bootstraps_pip_via_ensurepip(monkeypatch):
    # pip absent (eg uv venv): bootstrap via ensurepip, then the install retries.
    mod = _load_llama_cpp_module()
    calls = []
    def fake_run(cmd, **kw):
        calls.append(cmd)
        if "install" in cmd and sum(1 for c in calls if "install" in c) == 1:
            return types.SimpleNamespace(returncode=1, stdout="No module named pip")
        return types.SimpleNamespace(returncode=0, stdout="ok")
    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    res = mod._reinstall_converter_deps("/fake/python")
    assert res.returncode == 0
    assert any("ensurepip" in c for c in calls)


def test_reinstall_no_ensurepip_when_pip_present(monkeypatch):
    mod = _load_llama_cpp_module()
    calls = []
    monkeypatch.setattr(mod.subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd) or types.SimpleNamespace(returncode=0, stdout="ok"))
    res = mod._reinstall_converter_deps("/fake/python")
    assert res.returncode == 0
    assert not any("ensurepip" in c for c in calls)
