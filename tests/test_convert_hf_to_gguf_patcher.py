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



# New-layout entrypoint. The structural anchor we detect on is `from conversion import`.
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

# conversion/base.py: canonical Metadata.load at 8-space indent (base.py:912).
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

# conversion/__init__.py TEXT_MODEL_MAP + MMPROJ_MODEL_MAP (__init__.py:19-231,234-283).
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

# conversion/qwen.py: both expert-key literals in one find_hparam call (upstream
# already handles the alias).
_PACKAGE_QWEN_PY = b"""\
from .base import ModelBase


class Qwen2MoeModel(ModelBase):
    def set_gguf_parameters(self):
        n_experts = self.find_hparam(["num_local_experts", "num_experts"])
        return n_experts
"""

# The OLD monolith: NO `from conversion import`, the anchor for layout detection.
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




def test_branding_patch_applies_and_is_idempotent(package_layout):
    llama_cpp = _load_llama_cpp_module()
    base_py = package_layout / "conversion" / "base.py"

    assert llama_cpp._apply_branding_patch_to_base(str(base_py)) == "applied"
    content = base_py.read_bytes()
    assert b"# UNSLOTH_BRANDING_APPLIED" in content
    assert b"self.metadata.quantized_by = 'Unsloth'" in content
    assert b"self.metadata.repo_url = 'https://huggingface.co/unsloth'" in content
    assert b"self.metadata.tags = ['unsloth', 'llama.cpp']" in content

    assert llama_cpp._apply_branding_patch_to_base(str(base_py)) == "already-applied"
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

    assert b"self.metadata = gguf.Metadata.load(" in patched
    # Code after the target survives: the patch only inserts lines.
    assert b"if self.remote_hf_model_id:" in patched
    assert b"self.metadata.name = self.remote_hf_model_id" in patched
    assert len(patched) > len(original)




def test_qwen_aliases_detected_when_both_keys_present(package_layout):
    llama_cpp = _load_llama_cpp_module()
    qwen_py = package_layout / "conversion" / "qwen.py"
    assert llama_cpp._qwen_already_handles_expert_aliases(str(qwen_py)) is True


def test_qwen_aliases_not_detected_when_only_one_key_present(tmp_path):
    llama_cpp = _load_llama_cpp_module()
    qwen_py = tmp_path / "qwen.py"
    qwen_py.write_bytes(b'n = self.hparams["num_experts"]\n')  # only num_experts
    assert llama_cpp._qwen_already_handles_expert_aliases(str(qwen_py)) is False




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


def test_conversion_sibling_info_covers_a_module_the_patcher_never_edits(package_layout):
    """The key decides whether the package is RESCANNED, not just re-patched.

    _scan_conversion_package reads every module in conversion/, so a key built
    from only __init__.py, base.py and qwen.py left a changed fourth module
    invisible: in a long-lived process the next export returned the cached
    converter without rescanning, and then executed the file that had changed.
    """
    llama_cpp = _load_llama_cpp_module()
    other = package_layout / "conversion" / "zz_helper.py"
    other.write_bytes(b"VALUE = 1\n")
    before = llama_cpp._conversion_sibling_info(str(package_layout))
    assert before is not None

    other.write_bytes(b"VALUE = 2  # and a payload\n")
    after = llama_cpp._conversion_sibling_info(str(package_layout))
    assert after != before, (
        "a module outside the patched three changed without moving the cache key"
    )


def test_conversion_sibling_info_notices_a_module_appearing(package_layout):
    """A new module is a change too, and one an mtime on the old files misses."""
    llama_cpp = _load_llama_cpp_module()
    before = llama_cpp._conversion_sibling_info(str(package_layout))
    (package_layout / "conversion" / "zz_new.py").write_bytes(b"VALUE = 1\n")
    assert llama_cpp._conversion_sibling_info(str(package_layout)) != before


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

    We assert the contract by replacing the monolith-only arch extractor with
    a sentinel that fails the test if called, then driving the patcher end-
    to-end with `UNSLOTH_LLAMA_CPP_SCRIPTS_DIR` set."""
    llama_cpp = _load_llama_cpp_module()

    # A parse_args() stub so the end-to-end pipeline can finish its flag parsing.
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

    # A call here means the patcher fell through to the monolith path on package
    # layout, which is the bug being guarded.
    called = {"hit": False}
    def _trap(*a, **kw):
        called["hit"] = True
        raise AssertionError("monolith-only arch extraction ran on package layout")
    monkeypatch.setattr(llama_cpp, "_extract_archs_from_monolith_source", _trap)

    # @lru_cache(1) keys include local_script_info; a stale entry short-circuits.
    llama_cpp._download_convert_hf_to_gguf_cached.cache_clear()

    patched_path, text_archs, vision_archs = llama_cpp._download_convert_hf_to_gguf("regression_no_module_import")

    assert called["hit"] is False
    assert patched_path.endswith(".py")
    assert "LlamaForCausalLM" in text_archs
    assert text_archs == frozenset(text_archs)
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
    # By membership, not by index: the tuple now carries every module in
    # conversion/, so the scan is re-run when any of them changes, and its first
    # element is the file count. What this row is about is the DIRECTORY.
    paths = {entry[0] for entry in sib[1:]}
    assert str(conv / "base.py") in paths, paths
    assert str(conv / "__init__.py") in paths, paths
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
    # Qwen* entries: the user's reported architecture family.
    qwen_keys = {k for k in text_archs if k.startswith("Qwen")}
    assert qwen_keys, f"upstream TEXT_MODEL_MAP has no Qwen* entries: {sorted(text_archs)[:20]}..."


# ---------------------------------------------------------------------------
# num_experts patch: indentation and the write-time syntax gate (unsloth#4557)
# ---------------------------------------------------------------------------

def _converter_at_indent(indent, quote = b'"', newline = b"\n", pad = b" "):
    outer = pad * (indent - 4) if pad == b" " else pad
    inner = pad * indent if pad == b" " else pad * 2
    return (
        b"class Model:" + newline + outer + b"def set_gguf_parameters(self):" + newline +
        inner + b"n_experts = self.hparams[" + quote + b"num_experts" + quote + b"]" + newline +
        inner + b"return n_experts" + newline
    )


@pytest.mark.parametrize("indent", [8, 12, 16, 20])
@pytest.mark.parametrize("quote", [b'"', b"'"])
def test_num_experts_patch_preserves_indentation(indent, quote):
    import ast

    module = _load_llama_cpp_module()
    patched, applied = module._patch_num_experts(_converter_at_indent(indent, quote))
    assert applied
    assert b"num_local_experts" in patched
    ast.parse(patched.decode())


def test_num_experts_patch_is_a_noop_when_absent():
    module = _load_llama_cpp_module()
    source = b"class Model:\n    def set_gguf_parameters(self):\n        return 1\n"
    patched, applied = module._patch_num_experts(source)
    assert not applied
    assert patched == source


def test_patched_content_parses_gate():
    module = _load_llama_cpp_module()
    assert module._patched_content_parses(b"x = 1\n") is True
    assert module._patched_content_parses(b"def f(:\n") is False


@pytest.mark.parametrize("indent", [8, 12, 16, 20])
@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_num_experts_patch_preserves_the_line_ending(indent, newline):
    """The inserted lines must use the file's own ending, not a bare LF, or a CRLF
    checkout comes out with mixed endings."""
    import ast

    module = _load_llama_cpp_module()
    source = _converter_at_indent(indent, newline = newline)
    patched, applied = module._patch_num_experts(source)

    assert applied
    assert b"num_local_experts" in patched
    ast.parse(patched)
    # No LF that is not part of the file's own ending.
    assert patched.count(b"\n") == patched.count(newline), patched
    assert patched.replace(newline, b"\n").count(b"\n") == source.count(newline) + 1


def test_num_experts_patch_preserves_tab_indentation():
    import ast

    module = _load_llama_cpp_module()
    source = _converter_at_indent(2, pad = b"\t")
    patched, applied = module._patch_num_experts(source)

    assert applied
    ast.parse(patched)
    lines = patched.split(b"\n")
    inserted = [line for line in lines if b"num_local_experts" in line]
    assert inserted and all(line.startswith(b"\t\t") for line in inserted), lines


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_num_experts_patch_on_a_file_with_no_trailing_newline(newline):
    import ast

    module = _load_llama_cpp_module()
    # The target is the last line and the buffer has no trailing newline.
    source = (
        b"class Model:" + newline +
        b"    def set_gguf_parameters(self):" + newline +
        b'        n_experts = self.hparams["num_experts"]'
    )
    patched, applied = module._patch_num_experts(source)

    assert applied
    ast.parse(patched)
    assert patched.rstrip().endswith(b"self.hparams.get('num_local_experts')")


def test_num_experts_patch_is_not_fooled_by_a_similar_line():
    module = _load_llama_cpp_module()
    source = (
        b"class Model:\n"
        b"    def set_gguf_parameters(self):\n"
        b'        n_experts = self.other_hparams["num_experts"]\n'
        b'        m_experts = self.hparams["num_experts_extra"]\n'
    )
    patched, applied = module._patch_num_experts(source)

    assert not applied
    assert patched == source


@pytest.mark.parametrize(
    "content, expected",
    [
        (b"x = 1\n", True),
        (b"def f(:\n", False),
        # CRLF source is valid Python.
        (b"def f():\r\n    return 1\r\n", True),
        # A utf-8 BOM must be accepted; parsing a decoded str would reject it.
        (b"\xef\xbb\xbfx = 1\n", True),
        # PEP 263 coding cookie, honoured by ast.parse on bytes.
        (b"# -*- coding: utf-8 -*-\nx = 1\n", True),
        (b"# -*- coding: no-such-codec -*-\nx = 1\n", False),
        # Undecodable bytes and embedded NULs must be rejected, not raised.
        (b"x = '\xff\xfe'\n", False),
        (b"x = 1\x00\n", False),
    ],
)
def test_patched_content_parses_gate_cases(content, expected):
    module = _load_llama_cpp_module()
    assert module._patched_content_parses(content) is expected


def test_the_gate_rejects_the_hardcoded_indent_failure_mode():
    """What the old hardcoded replacement produced at 16 spaces: the statement landed
    at 12, one dedent below the comment, and was written without a syntax check."""
    module = _load_llama_cpp_module()
    broken = (
        b"class Model:\n"
        b"            def set_gguf_parameters(self):\n"
        b"                # Qwen3MoE seems to use num_local_experts instead of num_experts\n"
        b"            n_experts = self.hparams.get('num_experts', None)\n"
    )
    assert module._patched_content_parses(broken) is False


# ---------------------------------------------------------------------------
# Verification pass: a target line with a trailer, and the staged fallback
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "trailer",
    [b"  ", b"\t", b"  # noqa", b"  # keep this comment", b" #x"],
)
def test_num_experts_patch_keeps_a_trailing_comment_or_spaces(trailer):
    """Anchoring the pattern on the whole line must not lose the checkouts the old
    unanchored regex covered: a trailer is preserved and the patch still applies."""
    import ast

    module = _load_llama_cpp_module()
    source = (
        b"class Model:\n"
        b"    def set_gguf_parameters(self):\n"
        b'        n_experts = self.hparams["num_experts"]' + trailer + b"\n"
        b"        return n_experts\n"
    )
    patched, applied = module._patch_num_experts(source)

    assert applied, source
    ast.parse(patched)
    assert b"num_local_experts" in patched
    assert trailer.strip() in patched or not trailer.strip()
    assert b"        return n_experts\n" in patched


def test_num_experts_patch_ignores_a_commented_out_target():
    """A commented-out target must stay a comment, not become a statement."""
    module = _load_llama_cpp_module()
    source = b'class Model:\n    # n_experts = self.hparams["num_experts"]\n    pass\n'
    patched, applied = module._patch_num_experts(source)
    assert not applied
    assert patched == source


def _stage_list(*pairs):
    return list(pairs)


def test_choose_content_keeps_the_fully_patched_content_when_it_parses():
    module = _load_llama_cpp_module()
    label, content, dropped = module._choose_content_to_write(
        _stage_list(("base", b"x = 1\n"), ("p1", b"x = 1\ny = 2\n"), ("p2", b"x = 1\ny = 2\nz = 3\n"))
    )
    assert (label, dropped) == ("p2", [])
    assert content == b"x = 1\ny = 2\nz = 3\n"


def test_choose_content_drops_only_the_patch_that_broke_the_file():
    module = _load_llama_cpp_module()
    label, content, dropped = module._choose_content_to_write(
        _stage_list(("base", b"x = 1\n"), ("p1", b"x = 1\ny = 2\n"), ("p2", b"x = 1\ny = 2\n   oops(\n"))
    )
    assert label == "p1"
    assert dropped == ["p2"]
    assert content == b"x = 1\ny = 2\n"


def test_choose_content_falls_back_to_the_original_when_the_first_patch_broke_it():
    module = _load_llama_cpp_module()
    label, content, dropped = module._choose_content_to_write(
        _stage_list(("base", b"x = 1\n"), ("p1", b"x = 1\n  oops(\n"), ("p2", b"x = 1\n  oops(\ny = 2\n"))
    )
    assert label == "base"
    assert dropped == ["p1", "p2"]
    assert content == b"x = 1\n"


def test_choose_content_does_not_blame_our_patches_when_upstream_needs_a_newer_python():
    """If the untouched converter does not parse on the running interpreter, dropping
    our patches fixes nothing and would silently disable them. Keep them."""
    module = _load_llama_cpp_module()
    broken_base = b"class C[T]:\n    pass\n" if sys.version_info < (3, 12) else b"def f(:\n"
    stages = _stage_list(("base", broken_base), ("p1", broken_base + b"x = 1\n"))
    label, content, dropped = module._choose_content_to_write(stages)
    assert label == "p1"
    assert dropped == []
    assert content == stages[-1][1]


def test_choose_content_reports_no_drops_when_no_patch_applied():
    module = _load_llama_cpp_module()
    same = b"x = 1\n"
    label, content, dropped = module._choose_content_to_write(
        _stage_list(("base", same), ("p1", same))
    )
    assert dropped == []
    assert content == same


# --- end to end through the patcher, monolith layout -----------------------

_MONOLITH_HEAD = b"""\
#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf

logger = None


class ModelBase:
    def prepare_metadata(self):
        self.metadata = gguf.Metadata.load(None, None, None, None)
        return self.metadata


class TextModel(ModelBase):
    model_arch = gguf.MODEL_ARCH.LLAMA


@ModelBase.register("LlamaForCausalLM")
class LlamaModel(TextModel):
    model_arch = gguf.MODEL_ARCH.LLAMA


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--outfile", default=None)
    parser.add_argument("--outtype", default="f16")
    parser.add_argument("--vocab-only", action="store_true")
    return parser.parse_args()
"""


def _monolith_with_num_experts(indent, trailer = b""):
    body = (
        b"\n\n@ModelBase.register(\"Qwen3MoeForCausalLM\")\n"
        b"class Qwen3MoeModel(TextModel):\n"
        b"    class Inner:\n"
        b"        def modify_tensors(self):\n"
        + b" " * indent + b"n_experts = self.hparams[\"num_experts\"]" + trailer + b"\n"
        + b" " * indent + b"return n_experts\n"
    )
    return _MONOLITH_HEAD + body


def _drive_patcher(llama_cpp, tmp_path, monkeypatch, source, name = "convert_hf_to_gguf"):
    root = tmp_path / "llama_cpp_monolith"
    root.mkdir(exist_ok = True)
    (root / f"{name}.py").write_bytes(source)
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", str(root))
    out_dir = tmp_path / "patched_out"
    out_dir.mkdir(exist_ok = True)
    monkeypatch.setattr(llama_cpp, "LLAMA_CPP_DEFAULT_DIR", str(out_dir))
    llama_cpp._download_convert_hf_to_gguf_cached.cache_clear()
    try:
        patched_path, text_archs, vision_archs = llama_cpp._download_convert_hf_to_gguf(name)
    finally:
        llama_cpp._download_convert_hf_to_gguf_cached.cache_clear()
    return Path(patched_path).read_bytes(), text_archs


@pytest.mark.parametrize("indent", [12, 16])
@pytest.mark.parametrize("trailer", [b"", b"  # inline"])
def test_end_to_end_monolith_patch_is_written_and_parses(tmp_path, monkeypatch, indent, trailer):
    """Drive the whole patcher over a monolith whose target sits at a depth the old
    hardcoded replacement could not produce: the written file parses and is patched."""
    import ast

    llama_cpp = _load_llama_cpp_module()
    written, text_archs = _drive_patcher(
        llama_cpp, tmp_path, monkeypatch, _monolith_with_num_experts(indent, trailer)
    )
    ast.parse(written)
    assert b"num_local_experts" in written
    assert b" " * indent + b"n_experts = self.hparams.get(" in written
    assert b"self.metadata.quantized_by = 'Unsloth'" in written
    assert "LlamaForCausalLM" in text_archs


def test_end_to_end_keeps_earlier_patches_when_the_last_one_breaks_the_file(tmp_path, monkeypatch):
    """A broken num_experts patch must not cost the gguf guards and the branding."""
    import ast

    llama_cpp = _load_llama_cpp_module()

    def _broken(content):
        return content + b"\n   this is not python(\n", True
    monkeypatch.setattr(llama_cpp, "_patch_num_experts", _broken)

    written, _ = _drive_patcher(
        llama_cpp, tmp_path, monkeypatch, _monolith_with_num_experts(12)
    )
    ast.parse(written)
    assert b"this is not python(" not in written
    assert b"self.metadata.quantized_by = 'Unsloth'" in written
    assert b"try: gguf.MODEL_ARCH" in written


def test_end_to_end_writes_the_original_when_every_patch_stage_is_broken(tmp_path, monkeypatch):
    import ast

    llama_cpp = _load_llama_cpp_module()
    source = _monolith_with_num_experts(12)

    def _broken_first(content, *args, **kwargs):
        return content + b"\n  not python(\n"
    # Break the very first patch stage: everything after it inherits the breakage.
    monkeypatch.setattr(llama_cpp.re, "sub", lambda *a, **k: _broken_first(a[2]))

    written, _ = _drive_patcher(llama_cpp, tmp_path, monkeypatch, source)
    ast.parse(written)
    assert written == source


@pytest.mark.parametrize("prefix", [b"", b"\xef\xbb\xbf"])
def test_end_to_end_crlf_and_bom_monolith(tmp_path, monkeypatch, prefix):
    """CRLF throughout, optionally with a utf-8 BOM: the written converter must parse
    and must not gain a lone LF."""
    import ast

    llama_cpp = _load_llama_cpp_module()
    source = prefix + _monolith_with_num_experts(12).replace(b"\n", b"\r\n")
    written, _ = _drive_patcher(llama_cpp, tmp_path, monkeypatch, source)
    ast.parse(written)
    assert b"num_local_experts" in written
    assert written.startswith(prefix) if prefix else True
    inserted = [line for line in written.split(b"\r\n") if b"num_local_experts" in line]
    assert inserted, written
    assert b"\n" not in written.replace(b"\r\n", b"")


# --- network: the real monolith converter from a pinned upstream tag --------

@pytest.fixture
def real_monolith_converter(tmp_path):
    """The real pre-package convert_hf_to_gguf.py from llama.cpp b4600, a monolith
    whose two Qwen MoE sites use the patch target's spelling."""
    requests = pytest.importorskip("requests")
    url = "https://raw.githubusercontent.com/ggml-org/llama.cpp/b4600/convert_hf_to_gguf.py"
    try:
        response = requests.get(url, timeout = 30)
    except requests.exceptions.RequestException as exc:
        pytest.skip(f"network unreachable: {exc}")
    if response.status_code in (403, 429, 503):
        pytest.skip(f"upstream rate-limited / unavailable: HTTP {response.status_code}")
    if response.status_code != 200:
        pytest.skip(f"upstream missing the pinned converter: HTTP {response.status_code}")
    if b'n_experts = self.hparams["num_experts"]' not in response.content:
        pytest.skip("pinned converter no longer contains the patch target")
    return response.content


def test_real_monolith_converter_patch_applies_to_every_site(real_monolith_converter):
    import ast

    llama_cpp = _load_llama_cpp_module()
    expected = real_monolith_converter.count(b'n_experts = self.hparams["num_experts"]')
    patched, applied = llama_cpp._patch_num_experts(real_monolith_converter)

    assert applied
    ast.parse(patched)
    assert patched.count(b"or self.hparams.get('num_local_experts')") == expected
    assert b'n_experts = self.hparams["num_experts"]\n' not in patched
    # Indentation of every rewritten site is the one the file used (12 spaces).
    for line in patched.split(b"\n"):
        if b"or self.hparams.get('num_local_experts')" in line:
            assert line.startswith(b" " * 12) and not line.startswith(b" " * 13), line


def test_real_monolith_converter_patch_survives_a_crlf_checkout(real_monolith_converter):
    import ast

    llama_cpp = _load_llama_cpp_module()
    crlf = real_monolith_converter.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    patched, applied = llama_cpp._patch_num_experts(crlf)

    assert applied
    ast.parse(patched)
    assert b"\n" not in patched.replace(b"\r\n", b"")


# --- line endings of the inserted lines -------------------------------------

@pytest.mark.parametrize(
    "content, expected",
    [
        (b"a = 1\nb = 2\n", b"\n"),
        (b"a = 1\r\nb = 2\r\n", b"\r\n"),
        (b"", b"\n"),
        (b"a = 1", b"\n"),
        # Mixed, majority wins.
        (b"a\r\nb\r\nc\n", b"\r\n"),
        (b"a\nb\nc\r\n", b"\n"),
    ],
)
def test_dominant_newline(content, expected):
    module = _load_llama_cpp_module()
    assert module._dominant_newline(content) == expected


def test_branding_patch_on_a_crlf_base_py_stays_crlf(tmp_path):
    """The branding lines inserted into a CRLF base.py must use CRLF, not a bare LF."""
    module = _load_llama_cpp_module()
    base_py = tmp_path / "base.py"
    base_py.write_bytes(_PACKAGE_BASE_PY.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))

    assert module._apply_branding_patch_to_base(str(base_py)) == "applied"
    content = base_py.read_bytes()
    assert b"self.metadata.quantized_by = 'Unsloth'" in content
    assert b"\n" not in content.replace(b"\r\n", b"")
    assert module._apply_branding_patch_to_base(str(base_py)) == "already-applied"


def test_gguf_attribute_patch_keeps_the_blank_lines_after_the_import(tmp_path, monkeypatch):
    """The guards go straight after `import gguf`; the lines that followed it stay
    where they were."""
    module = _load_llama_cpp_module()
    source = _MONOLITH_HEAD.replace(b"import gguf\n", b"import gguf\n\n\n") + b"\n"
    written, _ = _drive_patcher(module, tmp_path, monkeypatch, source)
    assert b"try: gguf.MODEL_ARCH" in written
    assert b"\n\n\nlogger = None" in written


@pytest.mark.parametrize("indent", [12, 16])
def test_patching_an_already_patched_monolith_converges(tmp_path, monkeypatch, indent):
    """Patching the patcher's own output must be a no-op, not a second insertion.

    The monolith branding patch has no marker inside the file it edits, so before the
    guard it matched `Metadata.load(...)` again and appended another copy every time."""
    import ast

    llama_cpp = _load_llama_cpp_module()
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir(); second.mkdir()
    once, _ = _drive_patcher(
        llama_cpp, first, monkeypatch, _monolith_with_num_experts(indent)
    )
    twice, _ = _drive_patcher(llama_cpp, second, monkeypatch, once)

    ast.parse(twice)
    assert twice == once, "patching an already patched converter changed it"
    assert once.count(b"self.metadata.quantized_by = 'Unsloth'") == 1
    assert twice.count(b"self.metadata.quantized_by = 'Unsloth'") == 1
    assert twice.count(b"num_local_experts") == once.count(b"num_local_experts")
    # Same for the gguf guards: the arch scan finds the `gguf.X` names inside them.
    guards = once.count(b"except AttributeError: gguf.")
    assert guards > 0, "fixture did not exercise the gguf attribute guard patch"
    assert twice.count(b"except AttributeError: gguf.") == guards


def test_a_pristine_converter_is_still_branded(tmp_path, monkeypatch):
    """The idempotency guard keys on a line upstream never ships, so a fresh
    checkout is branded exactly as before."""
    llama_cpp = _load_llama_cpp_module()
    source = _monolith_with_num_experts(12)
    assert b"quantized_by" not in source
    written, _ = _drive_patcher(llama_cpp, tmp_path, monkeypatch, source)
    assert written.count(b"self.metadata.quantized_by = 'Unsloth'") == 1
    assert b"self.metadata.repo_url = 'https://huggingface.co/unsloth'" in written
    assert b"self.metadata.tags = ['unsloth', 'llama.cpp']" in written


def test_a_new_gguf_reference_is_guarded_on_a_converter_already_patched_once(
    tmp_path, monkeypatch
):
    """One pre-existing guard must not suppress guarding for every other name.

    The old convergence guard keyed on a single substring, so a converter patched once
    and then updated to reference a new `gguf` enum kept that enum unguarded, which is
    the exact AttributeError this patch exists to prevent.
    """
    import ast

    llama_cpp = _load_llama_cpp_module()
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir(); second.mkdir()

    once, _ = _drive_patcher(llama_cpp, first, monkeypatch, _monolith_with_num_experts(12))
    assert b"except AttributeError: gguf." in once, "fixture did not reach the guard patch"
    assert b"gguf.LATER_ENUM" not in once

    # The update: the same patched converter, now referencing a new enum from a body line.
    updated = once.replace(
        b"        return n_experts\n",
        b"        _later = gguf.LATER_ENUM\n        return n_experts\n",
        1,
    )
    assert updated != once, "the fixture rewrite did not apply"

    twice, _ = _drive_patcher(llama_cpp, second, monkeypatch, updated)
    ast.parse(twice)
    assert b"except AttributeError: gguf.LATER_ENUM = None" in twice, (
        "the new reference was left unguarded because the file already held a guard"
    )
    # And only that one is added: names already covered are not guarded twice.
    assert twice.count(b"except AttributeError: gguf.MODEL_ARCH.LLAMA = None") == 1
    assert twice.count(b"except AttributeError: gguf.LATER_ENUM = None") == 1

    # A third pass changes nothing: the convergence the substring check bought survives.
    third = tmp_path / "third"
    third.mkdir()
    thrice, _ = _drive_patcher(llama_cpp, third, monkeypatch, twice)
    assert thrice == twice


def test_a_converter_with_nothing_new_is_still_byte_identical(tmp_path, monkeypatch):
    """Per-attribute detection is only safe if re-patching a fully covered file writes
    the same bytes; otherwise every conversion grows the converter."""
    llama_cpp = _load_llama_cpp_module()
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir(); second.mkdir()
    once, _ = _drive_patcher(llama_cpp, first, monkeypatch, _monolith_with_num_experts(12))
    twice, _ = _drive_patcher(llama_cpp, second, monkeypatch, once)
    assert twice == once


def test_an_unterminated_final_target_keeps_the_files_line_ending():
    """A converter whose LAST line is the match carries no terminator to reuse, so it
    must inherit the file's dominant ending. A `\\n` fallback would leave a CRLF
    checkout with mixed endings, which still parses and so is easy to miss."""
    llama_cpp = _load_llama_cpp_module()
    crlf = (
        b"class M:\r\n"
        b"    def modify_tensors(self):\r\n"
        b"        n_experts = self.hparams[\"num_experts\"]"
    )
    patched, applied = llama_cpp._patch_num_experts(crlf)
    assert applied is True
    assert b"num_local_experts" in patched
    # Every ending in the result is CRLF: no bare LF anywhere.
    assert patched.replace(b"\r\n", b"") .count(b"\n") == 0, patched

    # And an LF file is untouched by the change: its dominant ending is LF.
    lf = crlf.replace(b"\r\n", b"\n")
    patched_lf, applied_lf = llama_cpp._patch_num_experts(lf)
    assert applied_lf is True
    assert b"\r" not in patched_lf, patched_lf


def test_a_terminated_target_still_reuses_its_own_ending():
    """The fallback must only apply where there is nothing to reuse. A line that HAS a
    terminator keeps it, even in a file whose dominant ending is the other one."""
    llama_cpp = _load_llama_cpp_module()
    mostly_lf = (
        b"class M:\n"
        b"    def modify_tensors(self):\n"
        b"        n_experts = self.hparams[\"num_experts\"]\r\n"
        b"        return n_experts\n"
    )
    patched, applied = llama_cpp._patch_num_experts(mostly_lf)
    assert applied is True
    assert b"num_local_experts')\r\n" in patched, patched
