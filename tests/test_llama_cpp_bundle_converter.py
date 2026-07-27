"""Tests for the llama.cpp bundle converter resolver (_resolve_bundle_convert_script).

A prebuilt llama.cpp bundle ships convert_hf_to_gguf.py co-versioned with its own
conversion/ package. When UNSLOTH_LLAMA_CPP_SCRIPTS_DIR is unset, the resolver
prefers the bundle's converter only when the paired conversion/ package (both
__init__.py and base.py) is present; otherwise it returns None (monolith installs
/ hosts without a paired package fall through to the network).

Loads llama_cpp.py in isolation (spec_from_file_location), matching the idiom in
tests/test_convert_hf_to_gguf_patcher.py, and monkeypatches LLAMA_CPP_DEFAULT_DIR
to a tmp tree. No network, no GPU.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest


def _load_llama_cpp_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "unsloth_zoo" / "llama_cpp.py"
    spec = importlib.util.spec_from_file_location("llama_cpp_bundle_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_llama_cpp_module()


def _make_bundle(root, *, conversion=True, base=True,
                 converter="convert_hf_to_gguf.py"):
    """Create a fake bundle dir. `conversion` writes conversion/__init__.py,
    `base` writes conversion/base.py; `converter` (or None) writes that converter
    filename at the bundle root."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if conversion:
        conv = root / "conversion"
        conv.mkdir(exist_ok=True)
        (conv / "__init__.py").write_text("# co-versioned conversion package\n")
        if base:
            (conv / "base.py").write_text("class ModelBase:\n    pass\n")
    if converter is not None:
        (root / converter).write_text("# bundle converter entrypoint\n")
    return root


def test_returns_none_when_dir_missing(mod, monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(tmp_path / "nope"))
    assert mod._resolve_bundle_convert_script() is None


def test_returns_none_when_default_dir_empty_or_unset(mod, monkeypatch):
    # An unset / empty default dir must not crash and must fall through.
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", "")
    assert mod._resolve_bundle_convert_script() is None


def test_returns_none_without_conversion_package(mod, monkeypatch, tmp_path):
    # Monolith install: converter present but no conversion/__init__.py.
    bundle = _make_bundle(tmp_path / "monolith", conversion=False)
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(bundle))
    assert mod._resolve_bundle_convert_script() is None


def test_returns_none_with_conversion_but_no_converter(mod, monkeypatch, tmp_path):
    bundle = _make_bundle(tmp_path / "noconv", conversion=True, converter=None)
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(bundle))
    assert mod._resolve_bundle_convert_script() is None


def test_returns_none_when_base_py_missing(mod, monkeypatch, tmp_path):
    # __init__.py present but base.py absent: must match _detect_converter_layout
    # (which needs both) and fall through to the network instead of mis-selecting.
    bundle = _make_bundle(tmp_path / "nobase", conversion=True, base=False)
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(bundle))
    assert mod._resolve_bundle_convert_script() is None


def test_resolves_underscore_converter_with_conversion(mod, monkeypatch, tmp_path):
    bundle = _make_bundle(tmp_path / "bundle")
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(bundle))

    result = mod._resolve_bundle_convert_script()
    assert result is not None
    path, mtime_ns, size = result
    expected = bundle / "convert_hf_to_gguf.py"
    assert path == str(expected)
    st = os.stat(expected)
    assert mtime_ns == st.st_mtime_ns
    assert size == st.st_size


def test_resolves_hyphenated_converter(mod, monkeypatch, tmp_path):
    bundle = _make_bundle(tmp_path / "hyphen", converter="convert-hf-to-gguf.py")
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(bundle))

    result = mod._resolve_bundle_convert_script()
    assert result is not None
    assert result[0] == str(bundle / "convert-hf-to-gguf.py")


def test_underscore_wins_when_both_converters_present(mod, monkeypatch, tmp_path):
    # LLAMA_CPP_CONVERTER_FILENAMES orders convert_hf_to_gguf.py first.
    bundle = _make_bundle(tmp_path / "both")
    (bundle / "convert-hf-to-gguf.py").write_text("# legacy name\n")
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(bundle))

    result = mod._resolve_bundle_convert_script()
    assert result is not None
    assert result[0] == str(bundle / "convert_hf_to_gguf.py")


def test_source_or_new_layout_checkout_also_resolves(mod, monkeypatch, tmp_path):
    # The resolver keys purely on (conversion/{__init__,base}.py + converter), so a
    # new-layout source checkout under LLAMA_CPP_DEFAULT_DIR also resolves locally
    # rather than hitting the network. This is intended (its converter is
    # genuinely co-versioned with its own conversion/) and documented here.
    src = _make_bundle(tmp_path / "src_checkout")
    (src / "ggml").mkdir()  # extra source-tree noise, ignored by the resolver
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(src))

    result = mod._resolve_bundle_convert_script()
    assert result is not None
    assert result[0] == str(src / "convert_hf_to_gguf.py")


def test_env_pin_resolver_takes_precedence_over_bundle(mod, monkeypatch, tmp_path):
    # _download_convert_hf_to_gguf consults _resolve_local_convert_script first
    # and only falls back to the bundle when it returns None. Here the env pin
    # resolves, so the bundle is never reached.
    scripts_dir = _make_bundle(tmp_path / "pinned", conversion=False)
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", str(scripts_dir))

    local = mod._resolve_local_convert_script()
    assert local is not None
    assert local[0] == str(scripts_dir / "convert_hf_to_gguf.py")
