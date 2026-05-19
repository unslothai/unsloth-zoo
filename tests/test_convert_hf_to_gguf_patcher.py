"""Tests for the layout-aware convert_hf_to_gguf.py patcher.

Covers the helpers that distinguish upstream llama.cpp's old monolithic
convert_hf_to_gguf.py from the new conversion/ package layout, plus the
in-place branding patch on conversion/base.py and the Qwen2MoE-skip path.

Two flavours:

  - synthetic_*: hand-crafted fixture trees that match the upstream layouts
    structurally; no network. These are the load-bearing CI gates.
  - latest_*  : pulls the current files from raw.githubusercontent.com and
    asserts the patcher still understands master. Skipped when offline.
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
    spec = importlib.util.spec_from_file_location("llama_cpp_under_test_patcher", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --- Synthetic fixtures matching upstream layouts ---------------------------

# A minimal but realistic stand-in for the new `convert_hf_to_gguf.py`
# entrypoint. The structural anchor we detect on is `from conversion import`.
_PACKAGE_ENTRYPOINT = b"""\
#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf

from conversion import (
    ModelBase,
    ModelType,
    get_model_architecture,
    get_model_class,
    logger,
    print_registered_models,
)
"""

# A minimal stand-in for conversion/base.py containing the canonical
# Metadata.load call site at 8-space indent (matches conversion/base.py:912).
_PACKAGE_BASE_PY = b"""\
import gguf
from enum import IntEnum


class ModelType(IntEnum):
    TEXT = 0
    MMPROJ = 1


class ModelBase:
    _model_classes = {ModelType.TEXT: {}, ModelType.MMPROJ: {}}

    def prepare_metadata(self, vocab_only):
        total_params, shared_params, expert_params, expert_count = (0, 0, 0, 0)

        self.metadata = gguf.Metadata.load(self.metadata_override, self.dir_model_card, self.model_name, total_params)

        if self.remote_hf_model_id:
            self.metadata.name = self.remote_hf_model_id
"""

# A minimal stand-in for conversion/__init__.py with realistic TEXT_MODEL_MAP
# and MMPROJ_MODEL_MAP dict literals (matches __init__.py:19-231,234-283).
_PACKAGE_INIT_PY = b"""\
from __future__ import annotations
from .base import ModelBase, ModelType


TEXT_MODEL_MAP: dict[str, str] = {
    "LlamaForCausalLM": "llama",
    "MistralForCausalLM": "llama",
    "Qwen3ForCausalLM": "qwen",
    "Qwen2MoeForCausalLM": "qwen",
    "Qwen3MoeForCausalLM": "qwen",
    "Gemma3ForCausalLM": "gemma",
}


MMPROJ_MODEL_MAP: dict[str, str] = {
    "LlavaForConditionalGeneration": "llava",
    "Gemma3ForConditionalGeneration": "gemma",
}


def load_all_models() -> None:
    pass


def get_model_class(name, mmproj=False):
    return ModelBase
"""

# Stand-in for the new conversion/qwen.py: contains both expert-key literals
# in the same find_hparam call (upstream already handles the alias).
_PACKAGE_QWEN_PY = b"""\
from .base import ModelBase


class Qwen2MoeModel(ModelBase):
    def set_gguf_parameters(self):
        n_experts = self.find_hparam(["num_local_experts", "num_experts"])
        return n_experts
"""

# A minimal stand-in for the OLD monolithic convert_hf_to_gguf.py. Note: NO
# `from conversion import` anywhere; that is the structural anchor for layout
# detection. ModelBase and ModelType are defined inline.
_MONOLITH = b"""\
import argparse
import gguf
from enum import IntEnum


class ModelType(IntEnum):
    TEXT = 0
    MMPROJ = 1


class ModelBase:
    _model_classes = {ModelType.TEXT: {"LlamaForCausalLM": object}, ModelType.MMPROJ: {}}

    def prepare_metadata(self):
        self.metadata = gguf.Metadata.load(override, card, name, params)

        if self.remote_hf_model_id:
            self.metadata.name = self.remote_hf_model_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outfile", default=None)
"""


@pytest.fixture
def package_layout(tmp_path):
    """Build a synthetic new-layout llama.cpp tree on disk and return its root."""
    root = tmp_path / "llama.cpp"
    root.mkdir()
    (root / "convert_hf_to_gguf.py").write_bytes(_PACKAGE_ENTRYPOINT)
    conv = root / "conversion"
    conv.mkdir()
    (conv / "__init__.py").write_bytes(_PACKAGE_INIT_PY)
    (conv / "base.py").write_bytes(_PACKAGE_BASE_PY)
    (conv / "qwen.py").write_bytes(_PACKAGE_QWEN_PY)
    return root


@pytest.fixture
def monolith_layout(tmp_path):
    """Build a synthetic old-layout llama.cpp tree on disk and return its root."""
    root = tmp_path / "llama.cpp"
    root.mkdir()
    (root / "convert_hf_to_gguf.py").write_bytes(_MONOLITH)
    return root


# --- Layout detection -------------------------------------------------------


def test_detect_layout_returns_package_for_new_tree(package_layout):
    llama_cpp = _load_llama_cpp_module()
    entry_bytes = (package_layout / "convert_hf_to_gguf.py").read_bytes()
    assert llama_cpp._detect_converter_layout(entry_bytes, str(package_layout)) == "package"


def test_detect_layout_returns_monolith_for_old_tree(monolith_layout):
    llama_cpp = _load_llama_cpp_module()
    entry_bytes = (monolith_layout / "convert_hf_to_gguf.py").read_bytes()
    assert llama_cpp._detect_converter_layout(entry_bytes, str(monolith_layout)) == "monolith"


def test_detect_layout_falls_back_to_monolith_when_conversion_dir_missing(tmp_path):
    """Entrypoint has the `from conversion import` anchor but the package dir is
    absent on disk -> treat as monolith (defensive)."""
    llama_cpp = _load_llama_cpp_module()
    assert llama_cpp._detect_converter_layout(_PACKAGE_ENTRYPOINT, str(tmp_path)) == "monolith"


# --- Arch enumeration from conversion/__init__.py ---------------------------


def test_extract_text_model_map_keys(package_layout):
    llama_cpp = _load_llama_cpp_module()
    init_py = package_layout / "conversion" / "__init__.py"
    keys = llama_cpp._extract_dict_keys_from_conversion_init(str(init_py), "TEXT_MODEL_MAP")
    assert {"LlamaForCausalLM", "Qwen3ForCausalLM", "Gemma3ForCausalLM"} <= keys
    assert "Qwen2MoeForCausalLM" in keys


def test_extract_mmproj_model_map_keys(package_layout):
    llama_cpp = _load_llama_cpp_module()
    init_py = package_layout / "conversion" / "__init__.py"
    keys = llama_cpp._extract_dict_keys_from_conversion_init(str(init_py), "MMPROJ_MODEL_MAP")
    assert "LlavaForConditionalGeneration" in keys
    assert "Gemma3ForConditionalGeneration" in keys


def test_extract_returns_empty_for_missing_dict(package_layout):
    llama_cpp = _load_llama_cpp_module()
    init_py = package_layout / "conversion" / "__init__.py"
    keys = llama_cpp._extract_dict_keys_from_conversion_init(str(init_py), "NON_EXISTENT_MAP")
    assert keys == set()


def test_extract_returns_empty_for_unparseable_file(tmp_path):
    """If conversion/__init__.py is missing or unparseable, we get an empty set
    rather than raising — patcher then warns but does not abort."""
    llama_cpp = _load_llama_cpp_module()
    assert llama_cpp._extract_dict_keys_from_conversion_init(str(tmp_path / "nope.py"), "TEXT_MODEL_MAP") == set()


# --- Branding patch on conversion/base.py -----------------------------------


def test_branding_patch_applies_and_is_idempotent(package_layout):
    llama_cpp = _load_llama_cpp_module()
    base_py = package_layout / "conversion" / "base.py"

    # First call: applies.
    assert llama_cpp._apply_branding_patch_to_base(str(base_py)) == "applied"
    content = base_py.read_bytes()
    assert b"# UNSLOTH_BRANDING_APPLIED" in content
    assert b"self.metadata.quantized_by = 'Unsloth'" in content
    assert b"self.metadata.repo_url = 'https://huggingface.co/unsloth'" in content
    assert b"self.metadata.tags = ['unsloth', 'llama.cpp']" in content

    # Second call: no-op (idempotent).
    assert llama_cpp._apply_branding_patch_to_base(str(base_py)) == "already-applied"
    # File contents should be unchanged after the second call.
    assert base_py.read_bytes() == content


def test_branding_patch_pattern_missing_when_metadata_load_absent(tmp_path):
    """A conversion/base.py without the Metadata.load call returns 'pattern-missing'."""
    llama_cpp = _load_llama_cpp_module()
    base_py = tmp_path / "base.py"
    base_py.write_bytes(b"# completely different file content\n")
    assert llama_cpp._apply_branding_patch_to_base(str(base_py)) == "pattern-missing"


def test_branding_patch_preserves_lines_around_target(package_layout):
    llama_cpp = _load_llama_cpp_module()
    base_py = package_layout / "conversion" / "base.py"
    original = base_py.read_bytes()
    llama_cpp._apply_branding_patch_to_base(str(base_py))
    patched = base_py.read_bytes()

    # The Metadata.load line itself is preserved verbatim.
    assert b"self.metadata = gguf.Metadata.load(" in patched
    # Code that followed the target (the if self.remote_hf_model_id... block)
    # is still present after the patch (we only inserted lines, not deleted).
    assert b"if self.remote_hf_model_id:" in patched
    assert b"self.metadata.name = self.remote_hf_model_id" in patched
    # File grew (we added 4 branding lines + marker), not shrank.
    assert len(patched) > len(original)


# --- Qwen expert-key alias detection ---------------------------------------


def test_qwen_aliases_detected_when_both_keys_present(package_layout):
    llama_cpp = _load_llama_cpp_module()
    qwen_py = package_layout / "conversion" / "qwen.py"
    assert llama_cpp._qwen_already_handles_expert_aliases(str(qwen_py)) is True


def test_qwen_aliases_not_detected_when_only_one_key_present(tmp_path):
    llama_cpp = _load_llama_cpp_module()
    qwen_py = tmp_path / "qwen.py"
    qwen_py.write_bytes(b'n = self.hparams["num_experts"]\n')  # only num_experts
    assert llama_cpp._qwen_already_handles_expert_aliases(str(qwen_py)) is False


# --- Cache-key invalidation (sibling info) ---------------------------------


def test_conversion_sibling_info_changes_when_base_py_changes(package_layout):
    llama_cpp = _load_llama_cpp_module()
    info_before = llama_cpp._conversion_sibling_info(str(package_layout))
    assert info_before is not None

    # Touch base.py with new content (mtime + size both change).
    base_py = package_layout / "conversion" / "base.py"
    base_py.write_bytes(base_py.read_bytes() + b"\n# extra trailing comment\n")

    info_after = llama_cpp._conversion_sibling_info(str(package_layout))
    assert info_after is not None
    assert info_after != info_before, (
        "_conversion_sibling_info must change when conversion/base.py changes, "
        "so the @lru_cache(1) entry is invalidated"
    )


def test_conversion_sibling_info_none_for_monolith(monolith_layout):
    llama_cpp = _load_llama_cpp_module()
    assert llama_cpp._conversion_sibling_info(str(monolith_layout)) is None


# --- _get_llama_cpp_dir resolution (addresses PR #667 review) ---------------


def test_llama_cpp_dir_defaults_when_no_local_script():
    llama_cpp = _load_llama_cpp_module()
    assert llama_cpp._get_llama_cpp_dir(None) == llama_cpp.LLAMA_CPP_DEFAULT_DIR


def test_llama_cpp_dir_resolves_to_source_dir_when_local_script_set(tmp_path):
    """UNSLOTH_LLAMA_CPP_SCRIPTS_DIR override: the patcher must operate
    against the directory containing the selected converter, not the
    hard-coded default. Mirrors `_resolve_local_convert_script`'s 3-tuple
    return shape `(abs_path, mtime_ns, size)`."""
    llama_cpp = _load_llama_cpp_module()
    custom = tmp_path / "custom_llama_cpp"
    custom.mkdir()
    src = custom / "convert_hf_to_gguf.py"
    src.write_bytes(b"# placeholder\n")
    local_info = (str(src), src.stat().st_mtime_ns, src.stat().st_size)
    assert llama_cpp._get_llama_cpp_dir(local_info) == str(custom)


def test_package_layout_does_not_require_module_import(tmp_path, monkeypatch):
    """Regression for Codex P1 on 3a9a23c: when UNSLOTH_LLAMA_CPP_SCRIPTS_DIR
    points at a package-layout checkout, the patcher must NOT call
    `_load_module_from_path` on the entrypoint. Importing it would resolve
    `from conversion import ...` against LLAMA_CPP_DEFAULT_DIR (a different
    dir than the override) and raise ModuleNotFoundError, aborting the
    patcher before AST arch extraction + branding could run.

    We assert the contract by replacing `_load_module_from_path` with a
    sentinel that fails the test if called, then driving the patcher end-
    to-end with `UNSLOTH_LLAMA_CPP_SCRIPTS_DIR` set."""
    llama_cpp = _load_llama_cpp_module()

    # Build a custom package-layout checkout in tmp_path. We extend the
    # shared fixture with a parse_args() stub so the end-to-end pipeline can
    # finish its flag-parsing step on the patched file (the real upstream
    # entrypoint has these calls; the shared fixture omits them because no
    # other test exercises the full pipeline).
    entry_with_args = _PACKAGE_ENTRYPOINT + (
        b"\n"
        b"def parse_args():\n"
        b"    parser = argparse.ArgumentParser()\n"
        b"    parser.add_argument(\"model\")\n"
        b"    parser.add_argument(\"--outfile\", default=None)\n"
        b"    parser.add_argument(\"--outtype\", default=\"f16\")\n"
        b"    parser.add_argument(\"--vocab-only\", action=\"store_true\")\n"
        b"    return parser.parse_args()\n"
    )
    root = tmp_path / "custom_llama_cpp"
    root.mkdir()
    (root / "convert_hf_to_gguf.py").write_bytes(entry_with_args)
    conv = root / "conversion"
    conv.mkdir()
    (conv / "__init__.py").write_bytes(_PACKAGE_INIT_PY)
    (conv / "base.py").write_bytes(_PACKAGE_BASE_PY)
    (conv / "qwen.py").write_bytes(_PACKAGE_QWEN_PY)

    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", str(root))

    # Sentinel: any call here means the patcher fell through to the
    # module-load path on package layout, which is the bug we're guarding.
    called = {"hit": False}
    def _trap(*a, **kw):
        called["hit"] = True
        raise AssertionError("monolith-only _load_module_from_path called on package layout")
    monkeypatch.setattr(llama_cpp, "_load_module_from_path", _trap)

    # Cache must be cleared between runs because @lru_cache(1) keys include
    # the resolved local_script_info -- but a stale entry from a previous
    # test would short-circuit the new call.
    llama_cpp._download_convert_hf_to_gguf_cached.cache_clear()

    patched_path, text_archs, vision_archs = llama_cpp._download_convert_hf_to_gguf("regression_no_module_import")

    assert called["hit"] is False
    assert patched_path.endswith(".py")
    assert "LlamaForCausalLM" in text_archs
    assert text_archs == frozenset(text_archs)
    # base.py was patched in place under the override dir.
    assert b"# UNSLOTH_BRANDING_APPLIED" in (conv / "base.py").read_bytes()
    # Cleanup for follow-on tests.
    llama_cpp._download_convert_hf_to_gguf_cached.cache_clear()


def test_patcher_anchors_on_custom_dir_when_override_set(tmp_path):
    """Build a custom llama.cpp tree with the new package layout in a temp
    dir, point a synthetic local_script_info at it, and confirm sibling
    info + layout detection target THAT dir, not the hardcoded default."""
    llama_cpp = _load_llama_cpp_module()
    root = tmp_path / "custom_llama_cpp"
    root.mkdir()
    (root / "convert_hf_to_gguf.py").write_bytes(_PACKAGE_ENTRYPOINT)
    conv = root / "conversion"
    conv.mkdir()
    (conv / "__init__.py").write_bytes(_PACKAGE_INIT_PY)
    (conv / "base.py").write_bytes(_PACKAGE_BASE_PY)
    (conv / "qwen.py").write_bytes(_PACKAGE_QWEN_PY)

    local_info = (
        str(root / "convert_hf_to_gguf.py"),
        (root / "convert_hf_to_gguf.py").stat().st_mtime_ns,
        (root / "convert_hf_to_gguf.py").stat().st_size,
    )
    resolved = llama_cpp._get_llama_cpp_dir(local_info)
    assert resolved == str(root)
    sib = llama_cpp._conversion_sibling_info(resolved)
    assert sib is not None
    assert sib[1][0] == str(conv / "base.py")  # base.py path in sibling tuple
    layout = llama_cpp._detect_converter_layout(_PACKAGE_ENTRYPOINT, resolved)
    assert layout == "package"


# --- Network smoke against current upstream llama.cpp ----------------------


@pytest.fixture
def latest_llama_cpp(tmp_path):
    """Fetch the current convert_hf_to_gguf.py + conversion/{__init__,base,qwen}.py
    from raw.githubusercontent.com. Skips the test cleanly when offline or rate-
    limited (raw.githubusercontent.com is documented at 60 req/hour unauthed)."""
    requests = pytest.importorskip("requests")
    root = tmp_path / "llama.cpp"
    (root / "conversion").mkdir(parents=True)
    base_url = "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/"
    files = {
        "convert_hf_to_gguf.py": root / "convert_hf_to_gguf.py",
        "conversion/__init__.py": root / "conversion" / "__init__.py",
        "conversion/base.py": root / "conversion" / "base.py",
        "conversion/qwen.py": root / "conversion" / "qwen.py",
    }
    headers = {}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    for rel, dest in files.items():
        try:
            r = requests.get(base_url + rel, timeout=15, headers=headers)
        except requests.exceptions.RequestException as exc:
            pytest.skip(f"network unreachable: {exc}")
        if r.status_code in (403, 429, 503):
            pytest.skip(f"upstream rate-limited / unavailable: HTTP {r.status_code}")
        if r.status_code != 200:
            pytest.skip(f"upstream missing {rel}: HTTP {r.status_code}")
        dest.write_bytes(r.content)
    return root


def test_latest_upstream_detected_as_package_layout(latest_llama_cpp):
    llama_cpp = _load_llama_cpp_module()
    entry_bytes = (latest_llama_cpp / "convert_hf_to_gguf.py").read_bytes()
    layout = llama_cpp._detect_converter_layout(entry_bytes, str(latest_llama_cpp))
    assert layout == "package", "current llama.cpp master should match the new layout"


def test_latest_upstream_branding_patch_applies(latest_llama_cpp):
    """Against the live upstream conversion/base.py, the branding regex must
    still match. If upstream changes the indentation or arguments of
    Metadata.load, this test fails fast so we can update the regex."""
    llama_cpp = _load_llama_cpp_module()
    base_py = latest_llama_cpp / "conversion" / "base.py"
    status = llama_cpp._apply_branding_patch_to_base(str(base_py))
    assert status == "applied", f"branding patch did not apply to upstream base.py: {status}"
    content = base_py.read_bytes()
    assert b"# UNSLOTH_BRANDING_APPLIED" in content
    assert b"self.metadata.quantized_by = 'Unsloth'" in content


def test_latest_upstream_qwen_already_handles_aliases(latest_llama_cpp):
    """Upstream Qwen module is expected to call find_hparam with both keys."""
    llama_cpp = _load_llama_cpp_module()
    qwen_py = latest_llama_cpp / "conversion" / "qwen.py"
    if not qwen_py.exists():
        pytest.skip("upstream conversion/qwen.py absent")
    assert llama_cpp._qwen_already_handles_expert_aliases(str(qwen_py)) is True


def test_latest_upstream_arch_enumeration_non_empty(latest_llama_cpp):
    """TEXT_MODEL_MAP in upstream conversion/__init__.py must produce a non-empty
    architecture allowlist. This is the assertion that would have caught the
    original 'No supported architectures' warning if it had been a test."""
    llama_cpp = _load_llama_cpp_module()
    init_py = latest_llama_cpp / "conversion" / "__init__.py"
    text_archs = llama_cpp._extract_dict_keys_from_conversion_init(str(init_py), "TEXT_MODEL_MAP")
    assert "LlamaForCausalLM" in text_archs, (
        f"upstream TEXT_MODEL_MAP missing LlamaForCausalLM; "
        f"got {sorted(text_archs)[:10]}..."
    )
    # The set should also contain at least some Qwen entries since this is the
    # user's reported architecture family.
    qwen_keys = {k for k in text_archs if k.startswith("Qwen")}
    assert qwen_keys, f"upstream TEXT_MODEL_MAP has no Qwen* entries: {sorted(text_archs)[:20]}..."
