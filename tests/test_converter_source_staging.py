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

"""Tests for co-versioned llama.cpp converter source staging.

No network, GPU or llama.cpp toolchain: downloads are monkeypatched to build a real
tarball on disk.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import shutil
import sys
import tarfile
import threading
from pathlib import Path

import pytest


def _load_llama_cpp_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "unsloth_zoo" / "llama_cpp.py"
    spec = importlib.util.spec_from_file_location("llama_cpp_staging_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mod():
    return _load_llama_cpp_module()


@pytest.fixture(autouse = True)
def _hermetic(mod, tmp_path, monkeypatch):
    """Every test starts with no converter env vars and no llama.cpp install."""
    for name in (
        "UNSLOTH_LLAMA_CPP_SCRIPTS_DIR",
        "UNSLOTH_LLAMA_CPP_CONVERTER_TAG",
        "UNSLOTH_CONVERTER_STAGE",
        "UNSLOTH_LLAMA_TAG",
        "UNSLOTH_LLAMA_CPP_OFFLINE",
        "UNSLOTH_OFFLINE",
        "HF_HUB_OFFLINE",
    ):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(tmp_path / "no_llama_cpp"))


_SHIM_ENTRYPOINT = b"""\
#!/usr/bin/env python3
import os
import sys
from pathlib import Path

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf

from conversion import (
    ModelBase,
    ModelType,
    get_model_architecture,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a model to GGUF")
    parser.add_argument("model", type=Path)
    parser.add_argument("--outfile", type=Path, default=None)
    parser.add_argument("--outtype", type=str, default="f16")
    return parser.parse_args()
"""

_MONOLITH_ENTRYPOINT = b"""\
#!/usr/bin/env python3
import gguf


@ModelBase.register("LlamaForCausalLM")
class LlamaModel(TextModel):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a model to GGUF")
    parser.add_argument("model", type=Path)
    parser.add_argument("--outfile", type=Path, default=None)
    parser.add_argument("--outtype", type=str, default="f16")
    return parser.parse_args()
"""

_CONVERSION_INIT = """\
TEXT_MODEL_MAP = {
    'LlamaForCausalLM': 'llama',
    'Qwen3MoeForCausalLM': 'qwen',
}
MMPROJ_MODEL_MAP = {
    'Gemma3ForConditionalGeneration': 'gemma3',
}
"""

_CONVERSION_BASE = """\
import gguf


class ModelBase:
    def set_metadata(self):
        self.metadata = gguf.Metadata.load(a, b, c)
        return self.metadata
"""

_CONVERSION_QWEN = """\
class Qwen3MoeModel(TextModel):
    def modify_tensors(self):
        n_experts = self.find_hparam(["num_local_experts", "num_experts"])
        return n_experts
"""


def _write_source_tree(root, *, entrypoint = _SHIM_ENTRYPOINT, conversion = True,
                       gguf_py = True):
    """Write what a llama.cpp source tarball unpacks to, or an install directory."""
    root = Path(root)
    root.mkdir(parents = True, exist_ok = True)
    (root / "convert_hf_to_gguf.py").write_bytes(entrypoint)
    if gguf_py:
        pkg = root / "gguf-py" / "gguf"
        pkg.mkdir(parents = True, exist_ok = True)
        (pkg / "__init__.py").write_text("# gguf\n")
        (pkg / "tensor_mapping.py").write_text("class TensorNameMap:\n    pass\n")
    if conversion:
        conv = root / "conversion"
        conv.mkdir(parents = True, exist_ok = True)
        (conv / "__init__.py").write_text(_CONVERSION_INIT)
        (conv / "base.py").write_text(_CONVERSION_BASE)
        (conv / "qwen.py").write_text(_CONVERSION_QWEN)
    return root


def _build_source_tarball(path, *, entrypoint = _SHIM_ENTRYPOINT, conversion = True,
                          gguf_py = True, tag = "b9000"):
    """Tarball nested under llama.cpp-{tag}/, the way codeload serves one."""
    root = Path(path).parent / f"_src_{tag}"
    inner = _write_source_tree(
        root / f"llama.cpp-{tag}", entrypoint = entrypoint,
        conversion = conversion, gguf_py = gguf_py,
    )
    with tarfile.open(path, "w:gz") as archive:
        archive.add(inner, arcname = f"llama.cpp-{tag}")
    return path


@pytest.fixture
def staging_env(mod, tmp_path, monkeypatch):
    """Cache at tmp_path, downloads served from a real tarball; `downloads` counts them."""
    cache = tmp_path / "converter-cache"
    monkeypatch.setattr(mod, "LLAMA_CPP_CONVERTER_CACHE_DIR", str(cache))
    state = {"downloads": 0, "entrypoint": _SHIM_ENTRYPOINT, "conversion": True,
             "gguf_py": True, "fail": None}

    def fake_download(url, dest_path):
        state["downloads"] += 1
        if state["fail"] is not None:
            raise state["fail"]
        _build_source_tarball(
            dest_path, entrypoint = state["entrypoint"],
            conversion = state["conversion"], gguf_py = state["gguf_py"],
        )
    monkeypatch.setattr(mod, "_download_archive", fake_download)
    monkeypatch.setattr(
        mod, "_resolve_llama_cpp_release",
        lambda *a, **k: pytest.fail("release discovery should not run here"),
    )
    state["cache"] = cache
    return state


# --- the happy path -----------------------------------------------------------

def test_staging_produces_all_three_trees_from_one_tarball(mod, staging_env):
    stage = mod._stage_converter_sources("b9000")
    assert stage is not None
    assert os.path.isfile(os.path.join(stage, "convert_hf_to_gguf.py"))
    assert os.path.isfile(os.path.join(stage, "conversion", "__init__.py"))
    assert os.path.isfile(os.path.join(stage, "conversion", "base.py"))
    assert os.path.isfile(os.path.join(stage, "gguf-py", "gguf", "__init__.py"))
    assert staging_env["downloads"] == 1


def test_the_manifest_records_the_revision_and_is_marked_complete(mod, staging_env):
    stage = mod._stage_converter_sources("b9000", repo = "unslothai/llama.cpp")
    manifest = json.loads(
        Path(stage, mod.UNSLOTH_CONVERTER_STAGE_FILENAME).read_text(encoding = "utf-8")
    )
    assert manifest["tag"] == "b9000"
    assert manifest["repo"] == "unslothai/llama.cpp"
    assert manifest["completed"] is True
    assert manifest["schema"] == mod.UNSLOTH_CONVERTER_STAGE_SCHEMA


def test_a_cache_hit_makes_zero_downloads(mod, staging_env):
    first = mod._stage_converter_sources("b9000")
    assert staging_env["downloads"] == 1
    second = mod._stage_converter_sources("b9000")
    assert second == first
    assert staging_env["downloads"] == 1


def test_a_different_tag_is_a_different_entry(mod, staging_env):
    a = mod._stage_converter_sources("b9000")
    b = mod._stage_converter_sources("b9001")
    assert a != b
    assert staging_env["downloads"] == 2


def test_a_revision_predating_the_split_needs_no_conversion_package(mod, staging_env):
    """The conversion/ requirement is read off the entrypoint."""
    staging_env["entrypoint"] = _MONOLITH_ENTRYPOINT
    staging_env["conversion"] = False
    stage = mod._stage_converter_sources("b7000")
    assert stage is not None
    assert not os.path.exists(os.path.join(stage, "conversion"))


# --- transactional behaviour --------------------------------------------------

def test_a_failed_download_publishes_nothing(mod, staging_env):
    staging_env["fail"] = RuntimeError("connection reset")
    assert mod._stage_converter_sources("b9000") is None
    assert not os.path.exists(mod._converter_stage_dir("ggml-org/llama.cpp", "b9000"))


def test_a_tarball_missing_gguf_py_is_refused(mod, staging_env):
    staging_env["gguf_py"] = False
    assert mod._stage_converter_sources("b9000") is None
    assert not os.path.exists(mod._converter_stage_dir("ggml-org/llama.cpp", "b9000"))


def test_a_shim_tarball_without_its_conversion_package_is_refused(mod, staging_env):
    staging_env["conversion"] = False
    assert mod._stage_converter_sources("b9000") is None
    assert not os.path.exists(mod._converter_stage_dir("ggml-org/llama.cpp", "b9000"))


def test_an_unwritable_cache_root_is_a_miss_not_a_raise(mod, staging_env, monkeypatch):
    """A cache root that cannot be created must return None, not raise."""
    def refuse(*args, **kwargs):
        raise PermissionError(13, "Permission denied")
    monkeypatch.setattr(mod.os, "makedirs", refuse)
    assert mod._stage_converter_sources("b9000") is None
    assert staging_env["downloads"] == 0


def test_a_cache_root_that_cannot_hold_a_temp_dir_is_a_miss_not_a_raise(
    mod, staging_env, monkeypatch,
):
    """A failed mkdtemp must return None, not raise."""
    def refuse(*args, **kwargs):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(mod.tempfile, "mkdtemp", refuse)
    assert mod._stage_converter_sources("b9000") is None
    assert staging_env["downloads"] == 0


def test_an_unwritable_cache_root_still_serves_an_existing_entry(mod, staging_env, monkeypatch):
    stage = mod._stage_converter_sources("b9000")
    assert stage is not None
    def refuse(*args, **kwargs):
        raise PermissionError(13, "Permission denied")
    monkeypatch.setattr(mod.os, "makedirs", refuse)
    monkeypatch.setattr(mod.tempfile, "mkdtemp", refuse)
    assert mod._stage_converter_sources("b9000") == stage


def test_a_failed_stage_leaves_a_working_entry_intact(mod, staging_env):
    good = mod._stage_converter_sources("b9000")
    marker = Path(good, "conversion", "__init__.py").read_text()
    staging_env["fail"] = RuntimeError("network died mid-refresh")
    # Force a miss by damaging only the manifest, as an interrupted publish would.
    Path(good, mod.UNSLOTH_CONVERTER_STAGE_FILENAME).unlink()
    assert mod._stage_converter_sources("b9000") is None
    assert Path(good, "conversion", "__init__.py").read_text() == marker


def test_a_damaged_entry_repairs_itself_instead_of_wedging(mod, staging_env):
    """A directory that exists but fails the probe must be replaced, not blocked."""
    stage = mod._stage_converter_sources("b9000")
    os.unlink(os.path.join(stage, "gguf-py", "gguf", "__init__.py"))
    assert mod._converter_stage_is_usable(stage) is False

    repaired = mod._stage_converter_sources("b9000")
    assert repaired == stage
    assert mod._converter_stage_is_usable(stage) is True
    downloads_after_repair = staging_env["downloads"]
    assert mod._stage_converter_sources("b9000") == stage
    assert staging_env["downloads"] == downloads_after_repair


def test_an_entry_missing_its_manifest_repairs_itself(mod, staging_env):
    stage = mod._stage_converter_sources("b9000")
    Path(stage, mod.UNSLOTH_CONVERTER_STAGE_FILENAME).unlink()
    assert mod._stage_converter_sources("b9000") == stage
    assert mod._converter_stage_is_usable(stage) is True


def test_the_superseded_wreck_is_not_left_behind(mod, staging_env):
    stage = mod._stage_converter_sources("b9000")
    os.unlink(os.path.join(stage, "gguf-py", "gguf", "__init__.py"))
    mod._stage_converter_sources("b9000")
    cache = Path(mod.LLAMA_CPP_CONVERTER_CACHE_DIR)
    assert list(cache.glob(".llama_cpp_converter_*")) == []


def test_an_entry_without_its_manifest_is_not_a_hit(mod, staging_env):
    """The manifest is written last, so its absence means a part-way attempt."""
    stage = mod._stage_converter_sources("b9000")
    Path(stage, mod.UNSLOTH_CONVERTER_STAGE_FILENAME).unlink()
    assert mod._converter_stage_is_usable(stage) is False


def test_an_incomplete_manifest_is_not_a_hit(mod, staging_env):
    stage = mod._stage_converter_sources("b9000")
    path = Path(stage, mod.UNSLOTH_CONVERTER_STAGE_FILENAME)
    manifest = json.loads(path.read_text(encoding = "utf-8"))
    manifest["completed"] = False
    path.write_text(json.dumps(manifest), encoding = "utf-8")
    assert mod._converter_stage_is_usable(stage) is False


def test_an_older_schema_is_not_a_hit(mod, staging_env):
    stage = mod._stage_converter_sources("b9000")
    path = Path(stage, mod.UNSLOTH_CONVERTER_STAGE_FILENAME)
    manifest = json.loads(path.read_text(encoding = "utf-8"))
    manifest["schema"] = mod.UNSLOTH_CONVERTER_STAGE_SCHEMA - 1
    path.write_text(json.dumps(manifest), encoding = "utf-8")
    assert mod._converter_stage_is_usable(stage) is False


def test_a_complete_manifest_over_a_gutted_tree_is_not_a_hit(mod, staging_env):
    """The probe checks the trees, not just the manifest."""
    stage = mod._stage_converter_sources("b9000")
    os.unlink(os.path.join(stage, "gguf-py", "gguf", "__init__.py"))
    assert mod._converter_stage_is_usable(stage) is False


def test_corrupt_manifest_json_is_not_a_hit(mod, staging_env):
    stage = mod._stage_converter_sources("b9000")
    Path(stage, mod.UNSLOTH_CONVERTER_STAGE_FILENAME).write_text("{not json", encoding = "utf-8")
    assert mod._converter_stage_is_usable(stage) is False


def test_the_private_staging_directory_is_always_cleaned_up(mod, staging_env, tmp_path):
    staging_env["fail"] = RuntimeError("boom")
    mod._stage_converter_sources("b9000")
    cache = Path(mod.LLAMA_CPP_CONVERTER_CACHE_DIR)
    leftovers = list(cache.glob(".llama_cpp_converter_*")) if cache.exists() else []
    assert leftovers == []


def test_losing_the_publish_race_adopts_the_winners_entry(mod, staging_env, monkeypatch):
    real_move = mod.shutil.move
    def racing_move(src, dst):
        real_move(src, dst)
        raise OSError("destination appeared first")
    monkeypatch.setattr(mod.shutil, "move", racing_move)
    stage = mod._stage_converter_sources("b9000")
    assert stage is not None
    assert mod._converter_stage_is_usable(stage)


def test_two_sequential_stagers_converge_on_one_entry(mod, staging_env):
    a = mod._stage_converter_sources("b9000")
    b = mod._stage_converter_sources("b9000")
    assert a == b
    assert staging_env["downloads"] == 1


# --- offline ------------------------------------------------------------------

@pytest.mark.parametrize(
    "var", ["UNSLOTH_LLAMA_CPP_OFFLINE", "UNSLOTH_OFFLINE", "HF_HUB_OFFLINE"],
)
def test_each_offline_switch_blocks_staging(mod, staging_env, monkeypatch, var):
    monkeypatch.setenv(var, "1")
    assert mod._stage_converter_sources("b9000") is None
    assert staging_env["downloads"] == 0


def test_offline_still_serves_a_complete_cache_entry(mod, staging_env, monkeypatch):
    stage = mod._stage_converter_sources("b9000")
    assert staging_env["downloads"] == 1
    monkeypatch.setenv("UNSLOTH_OFFLINE", "1")
    assert mod._stage_converter_sources("b9000") == stage
    assert staging_env["downloads"] == 1


def test_offline_reuses_a_revision_this_process_already_resolved(mod, monkeypatch, tmp_path):
    """Offline resolution consults the in-process memo of an already resolved tag."""
    mod._latest_converter_release_tag.cache_clear()
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_CONVERTER_TAG", raising = False)
    monkeypatch.delenv("UNSLOTH_LLAMA_TAG", raising = False)
    monkeypatch.setattr(mod, "_resolve_llama_cpp_release", lambda *a, **k: ("b9000", None))
    assert mod._resolve_converter_revision(str(tmp_path)) == ("ggml-org/llama.cpp", "b9000")
    monkeypatch.setenv("UNSLOTH_OFFLINE", "1")
    monkeypatch.setattr(
        mod, "_resolve_llama_cpp_release",
        lambda *a, **k: pytest.fail("offline must not reach the releases API"),
    )
    assert mod._resolve_converter_revision(str(tmp_path)) == ("ggml-org/llama.cpp", "b9000")
    mod._latest_converter_release_tag.cache_clear()


def test_offline_with_nothing_resolved_yet_still_declines(mod, monkeypatch, tmp_path):
    """The memo is consulted, not invented."""
    mod._latest_converter_release_tag.cache_clear()
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_CONVERTER_TAG", raising = False)
    monkeypatch.delenv("UNSLOTH_LLAMA_TAG", raising = False)
    monkeypatch.setenv("UNSLOTH_OFFLINE", "1")
    assert mod._resolve_converter_revision(str(tmp_path)) == (None, None)


def test_offline_does_not_reuse_a_revision_resolved_under_another_pin(mod, monkeypatch, tmp_path):
    """UNSLOTH_LLAMA_TAG is part of the memo key."""
    mod._latest_converter_release_tag.cache_clear()
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_CONVERTER_TAG", raising = False)
    monkeypatch.delenv("UNSLOTH_LLAMA_TAG", raising = False)
    monkeypatch.setattr(mod, "_resolve_llama_cpp_release", lambda *a, **k: ("b9000", None))
    assert mod._resolve_converter_revision(str(tmp_path)) == ("ggml-org/llama.cpp", "b9000")
    monkeypatch.setenv("UNSLOTH_LLAMA_TAG", "b1234")
    monkeypatch.setenv("UNSLOTH_OFFLINE", "1")
    assert mod._resolve_converter_revision(str(tmp_path)) == (None, None)
    mod._latest_converter_release_tag.cache_clear()


def test_offline_is_read_at_the_call_not_at_import(mod, monkeypatch):
    assert mod._converter_network_allowed() is True
    monkeypatch.setenv("UNSLOTH_OFFLINE", "1")
    assert mod._converter_network_allowed() is False
    monkeypatch.setenv("UNSLOTH_OFFLINE", "0")
    assert mod._converter_network_allowed() is True


# --- revision resolution ------------------------------------------------------

def test_an_explicit_tag_pin_wins(mod, monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_CONVERTER_TAG", "b1234")
    monkeypatch.setattr(
        mod, "_resolve_llama_cpp_release",
        lambda *a, **k: pytest.fail("an explicit pin must not consult the releases API"),
    )
    mod._write_prebuilt_marker(str(tmp_path), "b5678", "asset.tar.gz")
    assert mod._resolve_converter_revision(str(tmp_path)) == ("ggml-org/llama.cpp", "b1234")


def test_the_prebuilt_marker_tag_is_used_when_there_is_no_pin(mod, monkeypatch, tmp_path):
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_CONVERTER_TAG", raising = False)
    monkeypatch.setattr(
        mod, "_resolve_llama_cpp_release",
        lambda *a, **k: pytest.fail("the marker should have answered this"),
    )
    mod._write_prebuilt_marker(str(tmp_path), "b5678", "asset.tar.gz", repo = "unslothai/llama.cpp")
    assert mod._resolve_converter_revision(str(tmp_path)) == ("unslothai/llama.cpp", "b5678")


def test_a_marker_without_a_tag_falls_through(mod, monkeypatch, tmp_path):
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_CONVERTER_TAG", raising = False)
    (tmp_path / mod.UNSLOTH_PREBUILT_INFO_FILENAME).write_text('{"repo": "x"}', encoding = "utf-8")
    monkeypatch.setattr(mod, "_resolve_llama_cpp_release", lambda *a, **k: ("b9999", {}))
    assert mod._resolve_converter_revision(str(tmp_path)) == ("ggml-org/llama.cpp", "b9999")


def test_a_corrupt_marker_falls_through_rather_than_raising(mod, monkeypatch, tmp_path):
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_CONVERTER_TAG", raising = False)
    (tmp_path / mod.UNSLOTH_PREBUILT_INFO_FILENAME).write_text("{not json", encoding = "utf-8")
    monkeypatch.setattr(mod, "_resolve_llama_cpp_release", lambda *a, **k: ("b9999", {}))
    assert mod._resolve_converter_revision(str(tmp_path)) == ("ggml-org/llama.cpp", "b9999")


def test_offline_resolution_never_reaches_the_releases_api(mod, monkeypatch, tmp_path):
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_CONVERTER_TAG", raising = False)
    monkeypatch.setenv("UNSLOTH_OFFLINE", "1")
    monkeypatch.setattr(
        mod, "_resolve_llama_cpp_release",
        lambda *a, **k: pytest.fail("offline must not consult the releases API"),
    )
    assert mod._resolve_converter_revision(str(tmp_path)) == (None, None)


def test_a_failed_release_lookup_resolves_to_nothing(mod, monkeypatch, tmp_path):
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_CONVERTER_TAG", raising = False)
    monkeypatch.setattr(mod, "_resolve_llama_cpp_release", lambda *a, **k: None)
    mod._latest_converter_release_tag.cache_clear()
    assert mod._resolve_converter_revision(str(tmp_path)) == (None, None)


def test_release_discovery_runs_once_not_once_per_export(mod, monkeypatch, tmp_path):
    """Release discovery is memoized, so a warm cache pays no API round-trip."""
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_CONVERTER_TAG", raising = False)
    monkeypatch.delenv("UNSLOTH_LLAMA_TAG", raising = False)
    calls = []
    monkeypatch.setattr(
        mod, "_resolve_llama_cpp_release",
        lambda *a, **k: (calls.append(1), ("b9000", {}))[1],
    )
    mod._latest_converter_release_tag.cache_clear()
    for _ in range(5):
        assert mod._resolve_converter_revision(str(tmp_path)) == ("ggml-org/llama.cpp", "b9000")
    assert len(calls) == 1


def test_changing_the_llama_tag_pin_re_resolves(mod, monkeypatch, tmp_path):
    """The pin is the memo key, so a change mid-process is honoured."""
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_CONVERTER_TAG", raising = False)
    seen = []
    def fake_release(*a, **k):
        seen.append(os.environ.get("UNSLOTH_LLAMA_TAG", ""))
        return (f"tag-for-{os.environ.get('UNSLOTH_LLAMA_TAG', 'latest')}", {})
    monkeypatch.setattr(mod, "_resolve_llama_cpp_release", fake_release)
    mod._latest_converter_release_tag.cache_clear()
    monkeypatch.setenv("UNSLOTH_LLAMA_TAG", "b100")
    assert mod._resolve_converter_revision(str(tmp_path))[1] == "tag-for-b100"
    monkeypatch.setenv("UNSLOTH_LLAMA_TAG", "b200")
    assert mod._resolve_converter_revision(str(tmp_path))[1] == "tag-for-b200"
    assert seen == ["b100", "b200"]


# --- cache keys ---------------------------------------------------------------

def test_the_stage_key_cannot_escape_the_cache_root(mod):
    """Neither half of the key is trusted to stay inside the cache."""
    root = os.path.abspath(mod.LLAMA_CPP_CONVERTER_CACHE_DIR)
    for repo, tag in (
        ("ggml-org/llama.cpp", "../../etc"),
        ("../../evil", "b9000"),
        ("ggml-org/llama.cpp", "..%s.." % os.sep),
        ("ggml-org/llama.cpp", ".."),
        ("ggml-org/llama.cpp", "."),
    ):
        staged = os.path.abspath(mod._converter_stage_dir(repo, tag))
        assert os.path.commonpath([root, staged]) == root
        assert os.path.dirname(staged) == root


def test_the_stage_key_separates_the_two_repos(mod):
    upstream = mod._converter_stage_dir("ggml-org/llama.cpp", "b9000")
    fork = mod._converter_stage_dir("unslothai/llama.cpp", "b9000")
    assert upstream != fork


def test_an_empty_tag_stages_nothing(mod, staging_env):
    assert mod._stage_converter_sources("") is None
    assert mod._stage_converter_sources(None) is None
    assert staging_env["downloads"] == 0


# --- fork "mix" tag URL selection is shared with the prebuilt install ----------

def test_the_fork_source_asset_is_preferred(mod):
    assets = {"llama.cpp-source-b9739-mix-2d6bd50.tar.gz": "https://fork.invalid/src.tar.gz"}
    assert mod._converter_source_url("b9739-mix-2d6bd50", assets) == "https://fork.invalid/src.tar.gz"


def test_a_mix_tag_without_a_fork_asset_strips_the_suffix(mod):
    assert mod._converter_source_url("b9739-mix-2d6bd50", None) == \
        mod.LLAMA_CPP_SOURCE_TARBALL.format(tag = "b9739")


def test_a_plain_tag_is_unchanged(mod):
    assert mod._converter_source_url("b9000", None) == \
        mod.LLAMA_CPP_SOURCE_TARBALL.format(tag = "b9000")


# --- atomic writes ------------------------------------------------------------

def test_atomic_write_replaces_the_whole_file(mod, tmp_path):
    target = tmp_path / "converter.py"
    target.write_bytes(b"old" * 100)
    mod._atomic_write_bytes(str(target), b"new")
    assert target.read_bytes() == b"new"


def test_atomic_write_creates_missing_parents(mod, tmp_path):
    target = tmp_path / "a" / "b" / "converter.py"
    mod._atomic_write_bytes(str(target), b"hello")
    assert target.read_bytes() == b"hello"


def test_a_failed_atomic_write_leaves_the_original_and_no_temp_file(mod, tmp_path, monkeypatch):
    target = tmp_path / "converter.py"
    target.write_bytes(b"original")
    monkeypatch.setattr(mod.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("no space")))
    with pytest.raises(OSError):
        mod._atomic_write_bytes(str(target), b"replacement")
    assert target.read_bytes() == b"original"
    assert list(tmp_path.glob(".unsloth_tmp_*")) == []


# --- the resolver the patcher actually calls ----------------------------------

def test_the_staged_resolver_returns_the_entrypoint_stat_tuple(mod, staging_env, monkeypatch):
    monkeypatch.setattr(mod, "_resolve_converter_revision", lambda d: ("ggml-org/llama.cpp", "b9000"))
    result = mod._resolve_staged_convert_script()
    assert result is not None
    path, mtime_ns, size = result
    assert path.endswith("convert_hf_to_gguf.py")
    stat = os.stat(path)
    assert (mtime_ns, size) == (stat.st_mtime_ns, stat.st_size)


def test_the_staged_resolver_returns_none_when_no_revision_resolves(mod, monkeypatch):
    monkeypatch.setattr(mod, "_resolve_converter_revision", lambda d: (None, None))
    assert mod._resolve_staged_convert_script() is None


def test_the_staged_resolver_returns_none_when_staging_fails(mod, monkeypatch):
    monkeypatch.setattr(mod, "_resolve_converter_revision", lambda d: ("ggml-org/llama.cpp", "b9000"))
    monkeypatch.setattr(mod, "_stage_converter_sources", lambda *a, **k: None)
    assert mod._resolve_staged_convert_script() is None


def test_an_explicit_scripts_dir_is_never_supplemented_by_staging(mod, tmp_path, monkeypatch):
    """UNSLOTH_LLAMA_CPP_SCRIPTS_DIR stays authoritative; staging is never consulted."""
    pinned = tmp_path / "pinned"
    pinned.mkdir()
    (pinned / "convert_hf_to_gguf.py").write_bytes(_MONOLITH_ENTRYPOINT)
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", str(pinned))
    monkeypatch.setattr(
        mod, "_resolve_staged_convert_script",
        lambda: pytest.fail("an explicit scripts dir must not be supplemented"),
    )
    local = mod._resolve_local_convert_script()
    assert local is not None
    assert local[0] == str(pinned / "convert_hf_to_gguf.py")


# --- the misdetection, end to end --------------------------------------------

def test_a_shim_without_its_package_names_the_problem(mod, tmp_path, monkeypatch):
    """A shim with no conversion/ raises an actionable error instead of being patched."""
    pinned = tmp_path / "shim_only"
    pinned.mkdir()
    (pinned / "convert_hf_to_gguf.py").write_bytes(_SHIM_ENTRYPOINT)
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", str(pinned))
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(tmp_path / "out"))
    mod._download_convert_hf_to_gguf_cached.cache_clear()
    try:
        with pytest.raises(RuntimeError) as excinfo:
            mod._download_convert_hf_to_gguf("convert_hf_to_gguf")
    finally:
        mod._download_convert_hf_to_gguf_cached.cache_clear()
    message = str(excinfo.value)
    assert "conversion/" in message
    assert "same llama.cpp revision" in message
    # The pin is what selected this directory, and it outranks the tag, so naming
    # the tag here sends the user to a knob the pin keeps overriding.
    assert "UNSLOTH_LLAMA_CPP_SCRIPTS_DIR" in message
    assert str(pinned) in message
    assert "UNSLOTH_LLAMA_CPP_CONVERTER_TAG" not in message.split("outranks")[0], (
        "the tag is offered as the remedy while the scripts dir pin outranks it"
    )


def test_the_incomplete_error_is_not_wrapped_in_the_generic_one(mod, tmp_path, monkeypatch):
    """Its own type, so the broad `except Exception` re-raises it unchanged."""
    pinned = tmp_path / "shim_only2"
    pinned.mkdir()
    (pinned / "convert_hf_to_gguf.py").write_bytes(_SHIM_ENTRYPOINT)
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", str(pinned))
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(tmp_path / "out2"))
    mod._download_convert_hf_to_gguf_cached.cache_clear()
    try:
        with pytest.raises(mod._ConverterSourcesIncomplete) as excinfo:
            mod._download_convert_hf_to_gguf("convert_hf_to_gguf")
    finally:
        mod._download_convert_hf_to_gguf_cached.cache_clear()
    assert "Failed during loading/introspection" not in str(excinfo.value)


def test_a_staged_revision_patches_as_a_package(mod, staging_env, tmp_path, monkeypatch):
    """A staged tree behaves as a prebuilt bundle for every downstream step."""
    monkeypatch.setattr(mod, "_resolve_converter_revision", lambda d: ("ggml-org/llama.cpp", "b9000"))
    mod._download_convert_hf_to_gguf_cached.cache_clear()
    try:
        patched_path, text_archs, vision_archs = mod._download_convert_hf_to_gguf("unsloth_convert_hf_to_gguf")
    finally:
        mod._download_convert_hf_to_gguf_cached.cache_clear()
    stage = mod._converter_stage_dir("ggml-org/llama.cpp", "b9000")
    assert os.path.dirname(patched_path) == stage
    assert os.path.isfile(os.path.join(stage, "conversion", "__init__.py"))
    assert os.path.isfile(os.path.join(stage, "gguf-py", "gguf", "__init__.py"))
    assert "LlamaForCausalLM" in text_archs
    assert "Gemma3ForConditionalGeneration" in vision_archs
    assert mod._UNSLOTH_BRANDING_MARKER in Path(stage, "conversion", "base.py").read_bytes()
    ast.parse(Path(patched_path).read_bytes())


def test_the_staged_gguf_py_is_the_one_offered_the_qwen35_mapping(mod, staging_env, monkeypatch):
    """The qwen35 mapping patch is anchored on the staged converter's own directory."""
    monkeypatch.setattr(mod, "_resolve_converter_revision", lambda d: ("ggml-org/llama.cpp", "b9000"))
    seen = []
    monkeypatch.setattr(mod, "_patch_tensor_mapping_for_qwen35", lambda d: seen.append(d))
    mod._download_convert_hf_to_gguf_cached.cache_clear()
    try:
        mod._download_convert_hf_to_gguf("unsloth_convert_hf_to_gguf")
    finally:
        mod._download_convert_hf_to_gguf_cached.cache_clear()
    assert seen == [mod._converter_stage_dir("ggml-org/llama.cpp", "b9000")]


def test_two_staged_revisions_do_not_share_one_patcher_cache_entry(mod, staging_env, monkeypatch):
    """The patcher cache is keyed on (path, mtime, size), so tags cannot collide."""
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_CONVERTER_TAG", "b9000")
    mod._download_convert_hf_to_gguf_cached.cache_clear()
    try:
        first, _, _ = mod._download_convert_hf_to_gguf("unsloth_convert_hf_to_gguf")
        monkeypatch.setenv("UNSLOTH_LLAMA_CPP_CONVERTER_TAG", "b9001")
        second, _, _ = mod._download_convert_hf_to_gguf("unsloth_convert_hf_to_gguf")
    finally:
        mod._download_convert_hf_to_gguf_cached.cache_clear()
    assert first != second
    assert "b9000" in first and "b9001" in second


# --- resolver precedence ------------------------------------------------------
#
# UNSLOTH_LLAMA_CPP_SCRIPTS_DIR -> bundle with conversion/ -> installed
# self-contained monolith -> staged co-versioned sources -> the lone master
# entrypoint. The first three are offline.

def _no_network(mod, monkeypatch, why):
    """Turn every route to the network into a test failure."""
    def _trap(*args, **kwargs):
        raise AssertionError(why)
    monkeypatch.setattr(mod, "_download_archive", _trap)
    monkeypatch.setattr(mod, "_resolve_llama_cpp_release", _trap)
    monkeypatch.setattr(mod, "_resolve_converter_revision", _trap)
    monkeypatch.setattr(mod.requests, "get", _trap)


def test_a_bundle_with_its_conversion_package_never_reaches_staging(mod, tmp_path, monkeypatch):
    bundle = _write_source_tree(tmp_path / "bundle")
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(bundle))
    _no_network(mod, monkeypatch, "a co-versioned bundle must not trigger staging")
    info = mod._resolve_bundle_convert_script()
    assert info is not None
    assert Path(info[0]).parent == bundle


def _write_sibling_gguf_py(bundle):
    """The gguf-py tree every real llama.cpp install ships beside its converter."""
    pkg = Path(bundle) / "gguf-py" / "gguf"
    pkg.mkdir(parents = True, exist_ok = True)
    (pkg / "__init__.py").write_text("# gguf\n")
    (pkg / "tensor_mapping.py").write_text("class TensorNameMap:\n    pass\n")
    return pkg


def test_an_installed_self_contained_converter_is_used_instead_of_a_download(mod, tmp_path, monkeypatch):
    bundle = tmp_path / "old_install"
    bundle.mkdir()
    (bundle / "convert_hf_to_gguf.py").write_bytes(_MONOLITH_ENTRYPOINT)
    _write_sibling_gguf_py(bundle)
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(bundle))
    info = mod._resolve_monolith_bundle_convert_script()
    assert info is not None
    assert Path(info[0]) == bundle / "convert_hf_to_gguf.py"


def test_a_shim_with_no_package_beside_it_is_not_offered_as_self_contained(mod, tmp_path, monkeypatch):
    """An entrypoint importing conversion/ is half a revision, not a complete converter."""
    bundle = tmp_path / "half_install"
    bundle.mkdir()
    (bundle / "convert_hf_to_gguf.py").write_bytes(_SHIM_ENTRYPOINT)
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(bundle))
    assert mod._resolve_monolith_bundle_convert_script() is None


def test_the_monolith_resolver_never_shadows_a_real_co_versioned_bundle(mod, tmp_path, monkeypatch):
    bundle = _write_source_tree(tmp_path / "bundle")
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(bundle))
    assert mod._resolve_bundle_convert_script() is not None
    assert mod._resolve_monolith_bundle_convert_script() is None


def test_an_empty_install_offers_no_self_contained_converter(mod, tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(tmp_path / "nothing_here"))
    assert mod._resolve_monolith_bundle_convert_script() is None


def test_an_old_install_patches_its_own_converter_and_touches_no_network(mod, tmp_path, monkeypatch):
    """Drives the real entry point on a pre-split install."""
    bundle = tmp_path / "old_install"
    bundle.mkdir()
    (bundle / "convert_hf_to_gguf.py").write_bytes(_MONOLITH_ENTRYPOINT)
    _write_sibling_gguf_py(bundle)
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(bundle))
    _no_network(mod, monkeypatch, "an installed self-contained converter must not download")
    mod._download_convert_hf_to_gguf_cached.cache_clear()
    try:
        patched_path, text_archs, _ = mod._download_convert_hf_to_gguf("unsloth_convert_hf_to_gguf")
    finally:
        mod._download_convert_hf_to_gguf_cached.cache_clear()
    assert "LlamaForCausalLM" in text_archs
    assert Path(patched_path).parent == bundle


def test_staging_is_reached_only_when_the_three_local_rows_decline(mod, staging_env, monkeypatch):
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_CONVERTER_TAG", "b9000")
    assert mod._resolve_local_convert_script() is None
    assert mod._resolve_bundle_convert_script() is None
    assert mod._resolve_monolith_bundle_convert_script() is None
    info = mod._resolve_staged_convert_script()
    assert info is not None
    assert Path(info[0]).name == "convert_hf_to_gguf.py"
    assert staging_env["downloads"] == 1


def test_a_machine_with_no_llama_cpp_stages_a_complete_tree(mod, staging_env, monkeypatch):
    monkeypatch.setattr(mod, "_resolve_converter_revision", lambda d: ("ggml-org/llama.cpp", "b9000"))
    info = mod._resolve_staged_convert_script()
    assert info is not None
    staged_dir = str(Path(info[0]).parent)
    assert mod._detect_converter_layout(Path(info[0]).read_bytes(), staged_dir) == "package"


# --- the staging kill switch --------------------------------------------------

def test_staging_can_be_switched_off(mod, staging_env, monkeypatch):
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_CONVERTER_TAG", "b9000")
    monkeypatch.setenv("UNSLOTH_CONVERTER_STAGE", "0")
    assert mod._resolve_staged_convert_script() is None
    assert staging_env["downloads"] == 0


@pytest.mark.parametrize("value", ["0", "off", "FALSE", "no"])
def test_every_spelling_of_off_switches_staging_off(mod, monkeypatch, value):
    monkeypatch.setenv("UNSLOTH_CONVERTER_STAGE", value)
    assert mod._converter_staging_enabled() is False


@pytest.mark.parametrize("value", ["1", "on", "true", "yes", ""])
def test_anything_else_leaves_staging_on(mod, monkeypatch, value):
    monkeypatch.setenv("UNSLOTH_CONVERTER_STAGE", value)
    assert mod._converter_staging_enabled() is True


def test_the_staging_switch_is_read_at_the_call_not_at_import(mod, monkeypatch):
    assert mod._converter_staging_enabled() is True
    monkeypatch.setenv("UNSLOTH_CONVERTER_STAGE", "0")
    assert mod._converter_staging_enabled() is False
    monkeypatch.setenv("UNSLOTH_CONVERTER_STAGE", "1")
    assert mod._converter_staging_enabled() is True


def test_switching_staging_off_does_not_disable_the_prebuilt_hydration(mod, staging_env, tmp_path, monkeypatch):
    """The switch covers the export-time cache only, not install hydration."""
    monkeypatch.setenv("UNSLOTH_CONVERTER_STAGE", "0")
    install = tmp_path / "install"
    install.mkdir()
    mod._hydrate_converter_sources("b9000", str(install))
    assert (install / "convert_hf_to_gguf.py").is_file()
    assert (install / "conversion" / "base.py").is_file()
    assert (install / "gguf-py" / "gguf" / "__init__.py").is_file()


# --- staging robustness that the unit probes do not reach ---------------------

def test_a_failed_stage_for_one_tag_leaves_another_tags_entry_intact(mod, staging_env):
    good = mod._stage_converter_sources("b9000")
    assert mod._converter_stage_is_usable(good)
    staging_env["fail"] = RuntimeError("network died")
    assert mod._stage_converter_sources("b9999") is None
    assert mod._converter_stage_is_usable(good)


def test_concurrent_stagers_converge_on_one_entry(mod, staging_env):
    """Concurrency is handled by the immutable key, not by a lock."""
    results = []
    errors = []

    def _run():
        try:
            results.append(mod._stage_converter_sources("b9000"))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target = _run) for _ in range(4)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()

    assert errors == []
    assert len(set(results)) == 1, results
    assert results[0] is not None
    assert mod._converter_stage_is_usable(results[0])


def test_staging_uses_the_fork_source_asset_when_the_release_carries_one(mod, tmp_path, monkeypatch):
    """Staging resolves the same URL the prebuilt install would have."""
    monkeypatch.setattr(mod, "LLAMA_CPP_CONVERTER_CACHE_DIR", str(tmp_path / "cache"))
    seen = []

    def _record(url, dest_path):
        seen.append(url)
        raise RuntimeError("stop after URL resolution")
    monkeypatch.setattr(mod, "_download_archive", _record)

    mod._stage_converter_sources(
        "b9739-mix-2d6bd50", repo = "unslothai/llama.cpp",
        source_assets = {"llama.cpp-source-b9739-mix-2d6bd50.tar.gz": "https://fork.invalid/src.tar.gz"},
    )
    assert seen[-1] == "https://fork.invalid/src.tar.gz"

    mod._stage_converter_sources("b9739-mix-2d6bd50", repo = "unslothai/llama.cpp")
    assert seen[-1] == mod.LLAMA_CPP_SOURCE_TARBALL.format(tag = "b9739")


def test_a_read_only_pinned_checkout_still_resolves(mod, tmp_path, monkeypatch):
    """Resolution only stats, so a read-only checkout works."""
    checkout = _write_source_tree(tmp_path / "read_only")
    paths = sorted(checkout.rglob("*"), reverse = True)
    for path in paths:
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
    os.chmod(checkout, 0o555)
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", str(checkout))
    try:
        info = mod._resolve_local_convert_script()
        assert info is not None
        assert Path(info[0]).parent == checkout
    finally:
        os.chmod(checkout, 0o755)
        for path in paths:
            os.chmod(path, 0o755 if path.is_dir() else 0o644)


# --- the cost of pinning to the installed tag, made self-serviceable ----------
#
# An arch that only exists on master will not convert until a release cuts; the
# error must name the revision that refused and the knob that moves it.

def _staged_tree_at(mod, tag, root):
    """A published stage for `tag` under `root`, without any download."""
    stage = Path(mod._converter_stage_dir("ggml-org/llama.cpp", tag))
    _write_source_tree(stage)
    (stage / mod.UNSLOTH_CONVERTER_STAGE_FILENAME).write_text(json.dumps({
        "schema"    : mod.UNSLOTH_CONVERTER_STAGE_SCHEMA,
        "repo"      : "ggml-org/llama.cpp",
        "tag"       : tag,
        "completed" : True,
    }), encoding = "utf-8")
    assert mod._converter_stage_is_usable(str(stage))
    return stage


def test_the_staged_tag_is_readable_back_off_a_published_stage(mod, tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "LLAMA_CPP_CONVERTER_CACHE_DIR", str(tmp_path / "cache"))
    stage = _staged_tree_at(mod, "b9000", tmp_path)
    assert mod._staged_converter_tag(str(stage / "unsloth_convert_hf_to_gguf.py")) == "b9000"


def test_an_unstaged_converter_reports_no_tag(mod, tmp_path):
    checkout = _write_source_tree(tmp_path / "checkout")
    assert mod._staged_converter_tag(str(checkout / "convert_hf_to_gguf.py")) is None
    assert mod._staged_converter_tag(None) is None


def test_an_unsupported_arch_names_the_staged_tag_and_the_escape_hatch(mod, tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "LLAMA_CPP_CONVERTER_CACHE_DIR", str(tmp_path / "cache"))
    stage = _staged_tree_at(mod, "b9000", tmp_path)
    message = mod._unsupported_arch_message(
        "BrandNewForCausalLM", str(stage / "unsloth_convert_hf_to_gguf.py"),
    )
    assert "BrandNewForCausalLM" in message
    assert "b9000" in message
    assert "UNSLOTH_LLAMA_CPP_CONVERTER_TAG" in message


def test_an_unsupported_arch_on_an_unstaged_converter_still_names_the_escape_hatch(mod, tmp_path):
    """The remedy does not depend on which row of the ladder answered.

    This ladder newly prefers an installed converter over downloading master, so a
    model that converted yesterday can stop converting today against a copy that
    carries no manifest. Withholding the escape hatch from exactly that case leaves
    the population this change created with no way out."""
    checkout = _write_source_tree(tmp_path / "checkout")
    converter = str(checkout / "convert_hf_to_gguf.py")
    message = mod._unsupported_arch_message("BrandNewForCausalLM", converter)
    assert "BrandNewForCausalLM" in message
    assert "UNSLOTH_LLAMA_CPP_CONVERTER_TAG" in message, (
        "an unstaged converter gave the user no remedy at all"
    )
    assert "UNSLOTH_LLAMA_CPP_SCRIPTS_DIR" in message
    assert converter in message, "the message does not say which converter answered"


def test_convert_to_gguf_hands_the_user_the_staged_tag_and_the_escape_hatch(mod, tmp_path, monkeypatch):
    """The pin cost surfaces through the public entry point, not only in a helper."""
    monkeypatch.setattr(mod, "LLAMA_CPP_CONVERTER_CACHE_DIR", str(tmp_path / "cache"))
    stage = _staged_tree_at(mod, "b9000", tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"architectures": ["BrandNewForCausalLM"], "num_hidden_layers": 4}),
        encoding = "utf-8",
    )
    with pytest.raises(NotImplementedError) as excinfo:
        mod.convert_to_gguf(
            "model", str(model_dir),
            converter_location = str(stage / "unsloth_convert_hf_to_gguf.py"),
            supported_text_archs = {"LlamaForCausalLM"},
            supported_vision_archs = set(),
        )
    message = str(excinfo.value)
    assert "BrandNewForCausalLM" in message
    assert "b9000" in message
    assert "UNSLOTH_LLAMA_CPP_CONVERTER_TAG" in message


# --- Which spellings of the conversion import count. ---

_CONVERSION_IMPORT_SPELLINGS = (
    (b"from conversion import ModelBase\n",            True),
    (b"from conversion import (\n    ModelBase,\n)\n", True),
    (b"import conversion\n",                           True),
    (b"import conversion as _c\n",                     True),
    (b"from conversion.base import ModelBase\n",       True),
    (b"import conversion.base\n",                      True),
    (b"class ModelBase:\n    pass\n",                  False),
    (b"import convert_helpers\n",                      False),
    (b"# from conversion import ModelBase\n",          False),
    (b'CONVERSION_DOC = "from conversion import X"\n',  False),
)


@pytest.mark.parametrize("source, imports_it", _CONVERSION_IMPORT_SPELLINGS)
def test_every_spelling_of_the_conversion_import_is_recognised(source, imports_it):
    llama_cpp = _load_llama_cpp_module()
    assert llama_cpp._source_imports_conversion_package(source) is imports_it


@pytest.mark.parametrize("source, imports_it", _CONVERSION_IMPORT_SPELLINGS)
def test_layout_detection_follows_the_same_rule(tmp_path, source, imports_it):
    """Needs-the-package-but-cannot-see-it is 'incomplete' for every spelling."""
    llama_cpp = _load_llama_cpp_module()
    expected = "incomplete" if imports_it else "monolith"
    assert llama_cpp._detect_converter_layout(source, str(tmp_path)) == expected


def test_an_entrypoint_that_will_not_parse_is_not_called_self_contained(tmp_path):
    """Unparseable source falls back to the substring, not to 'no import'."""
    llama_cpp = _load_llama_cpp_module()
    broken = b"from conversion import ModelBase\nthis is not python(\n"
    assert llama_cpp._source_imports_conversion_package(broken) is True


def test_a_shim_using_a_submodule_import_is_not_offered_as_self_contained(tmp_path):
    llama_cpp = _load_llama_cpp_module()
    converter = tmp_path / "convert_hf_to_gguf.py"
    converter.write_bytes(b"from conversion.base import ModelBase\n")
    assert llama_cpp._entrypoint_needs_conversion_package(str(converter)) is True


# --- Cache identity, publication, and where a staged monolith's patched file lands. ---

_ARGPARSE_BLOCK = b"""

def parse_args():
    parser = argparse.ArgumentParser(description="Convert a model to GGUF")
    parser.add_argument("model", type=Path)
    parser.add_argument("--outfile", type=Path, default=None)
    return parser.parse_args()
"""


def _complete_stage(root, *, package, repo, tag, schema = None, manifest_repo = None,
                    manifest_tag = None):
    """A stage directory that passes every structural check."""
    import json as _json
    llama_cpp = _load_llama_cpp_module()
    os.makedirs(os.path.join(root, "gguf-py", "gguf"), exist_ok = True)
    open(os.path.join(root, "gguf-py", "gguf", "__init__.py"), "w").close()
    if package:
        conv = os.path.join(root, "conversion"); os.makedirs(conv, exist_ok = True)
        open(os.path.join(conv, "__init__.py"), "w").close()
        open(os.path.join(conv, "base.py"), "w").close()
        body = b"from conversion import ModelBase\n" + _ARGPARSE_BLOCK
    else:
        body = (b"@ModelBase.register(\"LlamaForCausalLM\")\nclass LlamaModel:\n    pass\n"
                + _ARGPARSE_BLOCK)
    with open(os.path.join(root, "convert_hf_to_gguf.py"), "wb") as f: f.write(body)
    with open(os.path.join(root, llama_cpp.UNSLOTH_CONVERTER_STAGE_FILENAME), "w") as f:
        _json.dump({
            "schema"    : schema if schema is not None else llama_cpp.UNSLOTH_CONVERTER_STAGE_SCHEMA,
            "repo"      : manifest_repo if manifest_repo is not None else repo,
            "tag"       : manifest_tag  if manifest_tag  is not None else tag,
            "completed" : True,
        }, f)
    return root


def test_a_stage_whose_manifest_names_another_revision_is_not_a_cache_hit(tmp_path):
    llama_cpp = _load_llama_cpp_module()
    stage = _complete_stage(str(tmp_path / "s"), package = True,
                            repo = "ggml-org/llama.cpp", tag = "b1111",
                            manifest_tag = "b9999")
    assert llama_cpp._converter_stage_is_usable(stage) is True
    assert llama_cpp._converter_stage_is_usable(
        stage, repo = "ggml-org/llama.cpp", tag = "b1111") is False
    assert llama_cpp._converter_stage_is_usable(
        stage, repo = "ggml-org/llama.cpp", tag = "b9999") is True


def test_a_stage_from_another_repo_is_not_a_cache_hit(tmp_path):
    llama_cpp = _load_llama_cpp_module()
    stage = _complete_stage(str(tmp_path / "s"), package = True,
                            repo = "ggml-org/llama.cpp", tag = "b1111",
                            manifest_repo = "someone/else")
    assert llama_cpp._converter_stage_is_usable(
        stage, repo = "ggml-org/llama.cpp", tag = "b1111") is False


def test_two_tags_that_sanitise_alike_do_not_serve_each_others_sources(tmp_path, monkeypatch):
    """Second line of defence: the manifest comparison, not just the digest."""
    llama_cpp = _load_llama_cpp_module()
    monkeypatch.setattr(llama_cpp, "LLAMA_CPP_CONVERTER_CACHE_DIR", str(tmp_path / "cache"))
    a = llama_cpp._converter_stage_dir("ggml-org/llama.cpp", "v/a")
    os.makedirs(a, exist_ok = True)
    _complete_stage(a, package = True, repo = "ggml-org/llama.cpp", tag = "v/a")
    assert llama_cpp._converter_stage_is_usable(a, repo = "ggml-org/llama.cpp", tag = "v/a") is True
    assert llama_cpp._converter_stage_is_usable(a, repo = "ggml-org/llama.cpp", tag = "v_a") is False


def test_publishing_onto_a_directory_that_appeared_does_not_nest_the_tree(mod, staging_env, monkeypatch):
    """Losing the publish race must not bury the loser's tree at <stage>/sources/."""
    stage_dir = mod._converter_stage_dir("ggml-org/llama.cpp", "b9000")
    winner_listing = {}

    real_download = mod._download_archive

    def download_then_lose_the_race(url, dest_path):
        real_download(url, dest_path)
        _write_source_tree(stage_dir)
        Path(stage_dir, mod.UNSLOTH_CONVERTER_STAGE_FILENAME).write_text(json.dumps({
            "schema": mod.UNSLOTH_CONVERTER_STAGE_SCHEMA,
            "repo": "ggml-org/llama.cpp", "tag": "b9000", "completed": True,
        }), encoding = "utf-8")
        winner_listing["files"] = sorted(os.listdir(stage_dir))
    monkeypatch.setattr(mod, "_download_archive", download_then_lose_the_race)

    real_exists = os.path.exists
    seen = {"n": 0}

    def exists_lying_on_the_move_guard(path):
        if os.path.abspath(path) == os.path.abspath(stage_dir):
            seen["n"] += 1
            if seen["n"] == 2: return False
        return real_exists(path)
    monkeypatch.setattr(mod.os.path, "exists", exists_lying_on_the_move_guard)

    stage = mod._stage_converter_sources("b9000")

    monkeypatch.undo()
    assert seen["n"] >= 2, "the move guard was never reached, so this proved nothing"
    assert winner_listing, "the winner never published, so there was no race"
    assert stage == stage_dir
    assert sorted(os.listdir(stage_dir)) == winner_listing["files"], (
        f"publication nested the loser's tree into the winner's: "
        f"{sorted(os.listdir(stage_dir))}"
    )
    assert not os.path.exists(os.path.join(stage_dir, "sources"))


def test_a_staged_monolith_keeps_its_patched_file_beside_its_own_gguf_py(mod, staging_env):
    staging_env["entrypoint"] = _MONOLITH_ENTRYPOINT
    staging_env["conversion"] = False
    os.environ["UNSLOTH_LLAMA_CPP_CONVERTER_TAG"] = "b7000"
    try:
        mod._download_convert_hf_to_gguf.cache_clear()
        patched, _text, _vision = mod._download_convert_hf_to_gguf()
    finally:
        os.environ.pop("UNSLOTH_LLAMA_CPP_CONVERTER_TAG", None)
        mod._download_convert_hf_to_gguf.cache_clear()

    stage_dir = mod._converter_stage_dir("ggml-org/llama.cpp", "b7000")
    assert os.path.dirname(patched) == stage_dir, (
        f"staged monolith patched file landed in {os.path.dirname(patched)}, "
        f"away from the gguf-py staged with it at {stage_dir}"
    )
    assert os.path.isdir(os.path.join(os.path.dirname(patched), "gguf-py")), \
        "the tree the entrypoint self-locates must be the one staged beside it"


def test_a_users_pinned_checkout_is_never_treated_as_our_cache(tmp_path, monkeypatch):
    llama_cpp = _load_llama_cpp_module()
    monkeypatch.setattr(llama_cpp, "LLAMA_CPP_CONVERTER_CACHE_DIR", str(tmp_path / "cache"))
    assert llama_cpp._is_inside_converter_cache(str(tmp_path / "my-llama.cpp")) is False


def test_the_converter_tag_pin_outranks_a_leftover_self_contained_converter(mod, staging_env, monkeypatch):
    install = tmp = Path(staging_env["cache"]).parent / "install"
    _write_source_tree(install, entrypoint = _MONOLITH_ENTRYPOINT, conversion = False)
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(install))

    mod._download_convert_hf_to_gguf.cache_clear()
    chosen, _t, _v = mod._download_convert_hf_to_gguf()
    assert os.path.dirname(chosen) == str(install)
    assert staging_env["downloads"] == 0

    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_CONVERTER_TAG", "b9000")
    mod._download_convert_hf_to_gguf.cache_clear()
    chosen, _t, _v = mod._download_convert_hf_to_gguf()
    mod._download_convert_hf_to_gguf.cache_clear()
    assert os.path.dirname(chosen) == mod._converter_stage_dir("ggml-org/llama.cpp", "b9000")
    assert staging_env["downloads"] == 1


@pytest.mark.skipif(os.name == "nt", reason = "Windows has no POSIX mode bits; chmod there only toggles the read-only flag, so st_mode & 0o777 is never 0o644. The Windows-relevant property is covered by the readability check below.")
def test_replacing_a_file_keeps_the_mode_it_had(tmp_path):
    mod = _load_llama_cpp_module()
    target = tmp_path / "converter.py"
    target.write_bytes(b"old\n")
    os.chmod(str(target), 0o644)
    mod._atomic_write_bytes(str(target), b"new\n")
    assert target.read_bytes() == b"new\n"
    assert os.stat(str(target)).st_mode & 0o777 == 0o644


@pytest.mark.skipif(os.name == "nt", reason = "umask is POSIX only")
def test_a_brand_new_file_is_not_created_owner_only(tmp_path):
    """No destination mode to copy: fall back to the umask, not mkstemp's 0600."""
    mod = _load_llama_cpp_module()
    target = tmp_path / "fresh.py"
    mod._atomic_write_bytes(str(target), b"x\n")
    umask = os.umask(0); os.umask(umask)
    assert os.stat(str(target)).st_mode & 0o777 == 0o666 & ~umask


def test_an_atomically_replaced_file_stays_readable_on_every_platform(tmp_path):
    mod = _load_llama_cpp_module()
    target = tmp_path / "converter.py"
    target.write_bytes(b"old\n")
    mod._atomic_write_bytes(str(target), b"new\n")
    assert target.read_bytes() == b"new\n"
    assert os.access(str(target), os.R_OK)
    assert os.access(str(target), os.W_OK)


# --- Two items Codex raised on ad88db33. ---

def test_offline_does_not_fall_through_to_the_legacy_master_download(mod, tmp_path, monkeypatch):
    install = tmp_path / "llama.cpp"
    (install / "gguf-py" / "gguf").mkdir(parents = True)
    (install / "gguf-py" / "gguf" / "__init__.py").write_text("")
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(install))
    monkeypatch.setattr(mod, "LLAMA_CPP_CONVERTER_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_OFFLINE", "1")

    attempts = []
    def trap(*args, **kwargs):
        attempts.append(args[0] if args else kwargs.get("url"))
        raise AssertionError("offline, nothing should reach the network")
    monkeypatch.setattr(mod.requests, "get", trap)

    mod._download_convert_hf_to_gguf.cache_clear()
    with pytest.raises(RuntimeError) as excinfo:
        mod._download_convert_hf_to_gguf()
    mod._download_convert_hf_to_gguf.cache_clear()

    assert attempts == [], f"offline still hit the network: {attempts}"
    message = str(excinfo.value)
    assert "offline" in message.lower()
    assert "UNSLOTH_LLAMA_CPP_SCRIPTS_DIR" in message


def test_a_damaged_converter_is_not_mistaken_for_a_self_contained_monolith(mod, tmp_path, monkeypatch):
    """A real monolith registers model classes; a truncated file does not."""
    install = tmp_path / "llama.cpp"
    (install / "gguf-py" / "gguf").mkdir(parents = True)
    (install / "gguf-py" / "gguf" / "__init__.py").write_text("")
    (install / "convert_hf_to_gguf.py").write_bytes(
        b"#!/usr/bin/env python3\nimport sys\n# truncated mid-download\n"
    )
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(install))
    assert mod._resolve_monolith_bundle_convert_script() is None

    (install / "convert_hf_to_gguf.py").write_bytes(_MONOLITH_ENTRYPOINT)
    resolved = mod._resolve_monolith_bundle_convert_script()
    assert resolved is not None
    assert resolved[0] == str(install / "convert_hf_to_gguf.py")


def test_an_empty_converter_file_is_not_offered_either(mod, tmp_path, monkeypatch):
    install = tmp_path / "llama.cpp"
    install.mkdir(parents = True)
    (install / "convert_hf_to_gguf.py").write_bytes(b"")
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(install))
    assert mod._resolve_monolith_bundle_convert_script() is None


def test_two_tags_that_sanitise_alike_get_different_directories(mod, tmp_path, monkeypatch):
    """Sanitising is not injective, so identity must not be the sanitised name."""
    monkeypatch.setattr(mod, "LLAMA_CPP_CONVERTER_CACHE_DIR", str(tmp_path / "cache"))
    a = mod._converter_stage_dir("ggml-org/llama.cpp", "feature/foo")
    b = mod._converter_stage_dir("ggml-org/llama.cpp", "feature_foo")
    assert a != b
    assert "feature_foo" in os.path.basename(a)
    assert a == mod._converter_stage_dir("ggml-org/llama.cpp", "feature/foo")
    assert a != mod._converter_stage_dir("unslothai/llama.cpp", "feature/foo")
    assert mod._converter_stage_dir("a", "b_c") != mod._converter_stage_dir("a_b", "c")


def test_one_revision_never_overwrites_another_revisions_live_tree(mod, staging_env, monkeypatch):
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_CONVERTER_TAG", "feature/foo")
    staging_env["entrypoint"] = _MONOLITH_ENTRYPOINT
    staging_env["conversion"] = False
    first = mod._stage_converter_sources("feature/foo")
    assert first is not None
    assert mod._converter_stage_is_usable(first, repo = "ggml-org/llama.cpp", tag = "feature/foo")

    second = mod._stage_converter_sources("feature_foo")
    assert second is not None and second != first
    assert mod._converter_stage_is_usable(first, repo = "ggml-org/llama.cpp", tag = "feature/foo"), \
        "staging the colliding tag destroyed the first revision's live tree"
    assert mod._converter_stage_is_usable(second, repo = "ggml-org/llama.cpp", tag = "feature_foo")


def test_a_truncated_cached_converter_is_not_a_cache_hit(mod, staging_env):
    staging_env["entrypoint"] = _MONOLITH_ENTRYPOINT
    staging_env["conversion"] = False
    stage = mod._stage_converter_sources("b9000")
    assert mod._converter_stage_is_usable(stage, repo = "ggml-org/llama.cpp", tag = "b9000")

    Path(stage, "convert_hf_to_gguf.py").write_bytes(b"")
    assert not mod._converter_stage_is_usable(stage, repo = "ggml-org/llama.cpp", tag = "b9000")

    Path(stage, "convert_hf_to_gguf.py").write_bytes(b"import sys\n# truncated\n")
    assert not mod._converter_stage_is_usable(stage, repo = "ggml-org/llama.cpp", tag = "b9000")

    staging_env["entrypoint"] = _SHIM_ENTRYPOINT
    staging_env["conversion"] = True
    pkg = mod._stage_converter_sources("b9500")
    assert mod._converter_stage_is_usable(pkg, repo = "ggml-org/llama.cpp", tag = "b9500")


def test_the_revision_pin_beats_a_modern_bundle_install_too(mod, staging_env, monkeypatch):
    install = Path(staging_env["cache"]).parent / "bundle"
    _write_source_tree(install)
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(install))

    mod._download_convert_hf_to_gguf.cache_clear()
    chosen, _t, _v = mod._download_convert_hf_to_gguf()
    assert os.path.dirname(chosen) == str(install)
    assert staging_env["downloads"] == 0

    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_CONVERTER_TAG", "b9000")
    mod._download_convert_hf_to_gguf.cache_clear()
    chosen, _t, _v = mod._download_convert_hf_to_gguf()
    mod._download_convert_hf_to_gguf.cache_clear()
    assert os.path.dirname(chosen) == mod._converter_stage_dir("ggml-org/llama.cpp", "b9000")
    assert staging_env["downloads"] == 1


def test_an_explicit_scripts_dir_still_outranks_the_revision_pin(mod, staging_env, monkeypatch, tmp_path):
    pinned_dir = tmp_path / "my-llama.cpp"
    _write_source_tree(pinned_dir)
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", str(pinned_dir))
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_CONVERTER_TAG", "b9000")
    mod._download_convert_hf_to_gguf.cache_clear()
    chosen, _t, _v = mod._download_convert_hf_to_gguf()
    mod._download_convert_hf_to_gguf.cache_clear()
    assert os.path.dirname(chosen) == str(pinned_dir)
    assert staging_env["downloads"] == 0


def test_a_package_entrypoint_truncated_after_its_import_is_not_a_cache_hit(mod, staging_env):
    """Truncated after the import, so the text still reads as package-based."""
    stage = mod._stage_converter_sources("b9500")
    assert mod._converter_stage_is_usable(stage, repo = "ggml-org/llama.cpp", tag = "b9500")

    Path(stage, "convert_hf_to_gguf.py").write_bytes(b"from conversion import (\n    ModelBase,\n")
    assert not mod._converter_stage_is_usable(stage, repo = "ggml-org/llama.cpp", tag = "b9500")

    Path(stage, "convert_hf_to_gguf.py").write_bytes(_SHIM_ENTRYPOINT)
    assert mod._converter_stage_is_usable(stage, repo = "ggml-org/llama.cpp", tag = "b9500")


def test_a_failed_release_lookup_is_not_remembered(mod, monkeypatch):
    """Successes are memoized, failures are retried."""
    mod._latest_converter_release_tag.cache_clear()
    calls = {"n": 0}
    outcomes = [None, None, ("b11037", {})]

    def flaky():
        calls["n"] += 1
        return outcomes[min(calls["n"] - 1, len(outcomes) - 1)]
    monkeypatch.setattr(mod, "_resolve_llama_cpp_release", flaky)

    assert mod._latest_converter_release_tag("") is None
    assert mod._latest_converter_release_tag("") is None
    assert calls["n"] == 2, "a failed lookup was cached and never retried"
    assert mod._latest_converter_release_tag("") == "b11037"
    assert calls["n"] == 3
    assert mod._latest_converter_release_tag("") == "b11037"
    assert calls["n"] == 3
    mod._latest_converter_release_tag.cache_clear()


def test_a_stage_repaired_by_another_process_is_adopted_not_destroyed(mod, staging_env, monkeypatch):
    """Asserted on the operation: stage_dir must not be moved at all."""
    stage_dir = mod._converter_stage_dir("ggml-org/llama.cpp", "b9000")
    _write_source_tree(stage_dir)
    Path(stage_dir, "convert_hf_to_gguf.py").write_bytes(b"")   # damaged: enters repair

    real_usable = mod._converter_stage_is_usable
    seen = {"n": 0}

    def repaired_after_the_condition(path, repo = None, tag = None):
        if os.path.abspath(path) == os.path.abspath(stage_dir):
            seen["n"] += 1
            if seen["n"] <= 2:
                return False
            if seen["n"] == 3:
                return True
        return real_usable(path, repo = repo, tag = tag)
    monkeypatch.setattr(mod, "_converter_stage_is_usable", repaired_after_the_condition)

    moved = []
    real_rename = mod.os.rename
    def spy(src, dst, *a, **k):
        moved.append(os.path.abspath(src))
        return real_rename(src, dst, *a, **k)
    monkeypatch.setattr(mod.os, "rename", spy)

    mod._stage_converter_sources("b9000")
    monkeypatch.undo()

    assert seen["n"] >= 3, "the repair window was never entered, so this proved nothing"
    assert os.path.abspath(stage_dir) not in moved, (
        "a replacement published between the condition and the move was moved aside, "
        "which deletes a live tree in the finally while another export holds the path"
    )


def test_a_replacement_published_during_the_move_is_restored_not_deleted(mod, staging_env, monkeypatch):
    """A valid tree moved aside between guard and move must be restored."""
    stage_dir = mod._converter_stage_dir("ggml-org/llama.cpp", "b9000")
    _write_source_tree(stage_dir)
    Path(stage_dir, "convert_hf_to_gguf.py").write_bytes(b"")   # damaged: enters repair

    real_rename = mod.os.rename
    swapped = {"done": False}

    def publish_a_winner_then_rename(src, dst, *a, **k):
        if not swapped["done"] and os.path.abspath(src) == os.path.abspath(stage_dir):
            swapped["done"] = True
            shutil.rmtree(stage_dir)
            _write_source_tree(stage_dir)
            Path(stage_dir, "WINNER").write_text("published by the other process\n")
            Path(stage_dir, mod.UNSLOTH_CONVERTER_STAGE_FILENAME).write_text(json.dumps({
                "schema": mod.UNSLOTH_CONVERTER_STAGE_SCHEMA,
                "repo": "ggml-org/llama.cpp", "tag": "b9000",
                "archive_sha256": "winner", "completed": True,
            }))
        return real_rename(src, dst, *a, **k)
    monkeypatch.setattr(mod.os, "rename", publish_a_winner_then_rename)

    result = mod._stage_converter_sources("b9000")
    monkeypatch.undo()

    assert swapped["done"], "the repair move never ran, so this proved nothing"
    assert result == stage_dir
    assert os.path.isdir(stage_dir), "the winner's tree was deleted in the finally"
    assert Path(stage_dir, "WINNER").is_file(), (
        "the winner's tree was replaced rather than restored, so the export still "
        "holding that path lost the files underneath it"
    )
    assert mod._converter_stage_is_usable(stage_dir, repo = "ggml-org/llama.cpp", tag = "b9000")


def test_a_monolith_without_its_gguf_py_is_left_to_staging(mod, tmp_path, monkeypatch):
    bundle = tmp_path / "no_gguf_py"
    bundle.mkdir()
    (bundle / "convert_hf_to_gguf.py").write_bytes(_MONOLITH_ENTRYPOINT)
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(bundle))
    assert mod._resolve_monolith_bundle_convert_script() is None


def test_a_monolith_with_a_damaged_gguf_py_is_left_to_staging(mod, tmp_path, monkeypatch):
    bundle = tmp_path / "damaged_gguf_py"
    bundle.mkdir()
    (bundle / "convert_hf_to_gguf.py").write_bytes(_MONOLITH_ENTRYPOINT)
    (bundle / "gguf-py" / "gguf").mkdir(parents = True)   # no __init__.py
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(bundle))
    assert mod._resolve_monolith_bundle_convert_script() is None


def test_a_monolith_without_gguf_py_is_still_used_when_staging_cannot_answer(mod, tmp_path, monkeypatch):
    bundle = tmp_path / "no_gguf_py_offline"
    bundle.mkdir()
    (bundle / "convert_hf_to_gguf.py").write_bytes(_MONOLITH_ENTRYPOINT)
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(bundle))
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_OFFLINE", "1")
    # Only real transfers are barred: _resolve_converter_revision must still run.
    def _trap(*args, **kwargs):
        raise AssertionError("offline must not download")
    monkeypatch.setattr(mod, "_download_archive", _trap)
    monkeypatch.setattr(mod, "_resolve_llama_cpp_release", _trap)
    monkeypatch.setattr(mod.requests, "get", _trap)
    mod._download_convert_hf_to_gguf_cached.cache_clear()
    try:
        patched_path, text_archs, _ = mod._download_convert_hf_to_gguf("unsloth_convert_hf_to_gguf")
    finally:
        mod._download_convert_hf_to_gguf_cached.cache_clear()
    assert "LlamaForCausalLM" in text_archs
    assert Path(patched_path).parent == bundle


# --- a read-only or shared cache ---

def _read_only(path):
    os.chmod(path, 0o555)


def test_a_read_only_cache_hit_is_copied_somewhere_writable(mod, tmp_path, monkeypatch, staging_env):
    stage = mod._stage_converter_sources("b9000")
    assert stage is not None
    home = tmp_path / "home"
    monkeypatch.setattr(mod, "UNSLOTH_HOME", str(home))
    _read_only(stage)
    try:
        resolved = mod._writable_stage(stage, repo = "ggml-org/llama.cpp", tag = "b9000")
        assert resolved is not None
        assert os.path.abspath(resolved) != os.path.abspath(stage)
        assert os.access(resolved, os.W_OK)
        assert mod._converter_stage_is_usable(
            resolved, repo = "ggml-org/llama.cpp", tag = "b9000",
        )
        Path(resolved, "unsloth_convert_hf_to_gguf.py").write_bytes(b"# patched\n")
    finally:
        os.chmod(stage, 0o755)


def test_the_writable_copy_is_made_once_and_then_reused(mod, tmp_path, monkeypatch, staging_env):
    stage = mod._stage_converter_sources("b9000")
    home = tmp_path / "home"
    monkeypatch.setattr(mod, "UNSLOTH_HOME", str(home))
    _read_only(stage)
    try:
        first = mod._writable_stage(stage, repo = "ggml-org/llama.cpp", tag = "b9000")
        Path(first, "MARKER").write_text("first copy\n")
        second = mod._writable_stage(stage, repo = "ggml-org/llama.cpp", tag = "b9000")
        assert second == first
        assert Path(second, "MARKER").is_file(), "it copied again instead of reusing"
    finally:
        os.chmod(stage, 0o755)


def test_a_writable_stage_is_returned_untouched(mod, tmp_path, monkeypatch, staging_env):
    stage = mod._stage_converter_sources("b9000")
    home = tmp_path / "home"
    monkeypatch.setattr(mod, "UNSLOTH_HOME", str(home))
    assert mod._writable_stage(stage, repo = "ggml-org/llama.cpp", tag = "b9000") == stage
    assert not home.exists(), "a writable stage was copied anyway"


def test_a_read_only_default_cache_is_reported_rather_than_copied_onto_itself(
    mod, tmp_path, monkeypatch, staging_env,
):
    stage = mod._stage_converter_sources("b9000")
    monkeypatch.setattr(mod, "UNSLOTH_HOME", str(tmp_path))
    monkeypatch.setattr(
        mod, "LLAMA_CPP_CONVERTER_CACHE_DIR",
        os.path.join(str(tmp_path), "llama.cpp-converter"),
    )
    _read_only(stage)
    try:
        assert mod._writable_stage(stage, repo = "ggml-org/llama.cpp", tag = "b9000") is None
    finally:
        os.chmod(stage, 0o755)


def test_the_resolver_hands_back_a_writable_directory_from_a_read_only_cache(
    mod, tmp_path, monkeypatch, staging_env,
):
    """Pins the production entry point, not the helper."""
    stage = mod._stage_converter_sources("b9000")
    assert stage is not None
    monkeypatch.setattr(mod, "UNSLOTH_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(mod, "_resolve_converter_revision",
                        lambda *a, **k: ("ggml-org/llama.cpp", "b9000"))
    _read_only(stage)
    try:
        info = mod._resolve_staged_convert_script()
        assert info is not None, "a valid but read-only cache hit was abandoned"
        resolved_dir = os.path.dirname(info[0])
        assert os.access(resolved_dir, os.W_OK), (
            "the resolver named a directory the patched converter cannot be written to"
        )
        Path(resolved_dir, "unsloth_convert_hf_to_gguf.py").write_bytes(b"# patched\n")
        assert os.path.isfile(os.path.join(resolved_dir, "conversion", "__init__.py")), (
            "the writable copy lost the co-versioned conversion/ the child imports"
        )
        assert os.path.isfile(
            os.path.join(resolved_dir, "gguf-py", "gguf", "__init__.py")
        ), "the writable copy lost the co-versioned gguf-py"
    finally:
        os.chmod(stage, 0o755)


# --- an explicit pin that cannot be staged ---

def test_a_pinned_revision_that_cannot_stage_fails_instead_of_downloading_master(
    mod, tmp_path, monkeypatch, staging_env,
):
    bundle = _write_source_tree(tmp_path / "install")
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(bundle))
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_CONVERTER_TAG", "b-does-not-exist")
    staging_env["fail"] = RuntimeError("404 while fetching the source tarball")
    def _trap(*a, **k):
        raise AssertionError("a failed pin must not fall through to the master download")
    monkeypatch.setattr(mod, "_download_convert_hf_to_gguf_file", _trap, raising = False)
    monkeypatch.setattr(mod.requests, "get", _trap)
    mod._download_convert_hf_to_gguf_cached.cache_clear()
    try:
        with pytest.raises(mod._ConverterSourcesIncomplete, match = "b-does-not-exist"):
            mod._download_convert_hf_to_gguf("unsloth_convert_hf_to_gguf")
    finally:
        mod._download_convert_hf_to_gguf_cached.cache_clear()


def test_an_unpinned_export_still_falls_back_when_staging_fails(mod, tmp_path, monkeypatch, staging_env):
    bundle = tmp_path / "old_install"
    bundle.mkdir()
    (bundle / "convert_hf_to_gguf.py").write_bytes(_MONOLITH_ENTRYPOINT)
    _write_sibling_gguf_py(bundle)
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(bundle))
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_CONVERTER_TAG", raising = False)
    staging_env["fail"] = RuntimeError("codeload is down")
    mod._download_convert_hf_to_gguf_cached.cache_clear()
    try:
        patched_path, text_archs, _ = mod._download_convert_hf_to_gguf("unsloth_convert_hf_to_gguf")
    finally:
        mod._download_convert_hf_to_gguf_cached.cache_clear()
    assert "LlamaForCausalLM" in text_archs
    assert Path(patched_path).parent == bundle


def test_a_package_entrypoint_cut_to_one_statement_is_not_a_cache_hit(mod, tmp_path):
    """Parsing is not evidence: the entrypoint must show a real CLI parser."""
    stage = _complete_stage(str(tmp_path / "s"), package = True,
                            repo = "ggml-org/llama.cpp", tag = "b9000")
    assert mod._converter_stage_is_usable(
        stage, repo = "ggml-org/llama.cpp", tag = "b9000") is True
    Path(stage, "convert_hf_to_gguf.py").write_bytes(b"from conversion import ModelBase\n")
    assert mod._converter_stage_is_usable(
        stage, repo = "ggml-org/llama.cpp", tag = "b9000") is False


def test_a_damaged_package_entrypoint_is_restaged_rather_than_served_forever(mod, staging_env):
    """Driven through the cache, not the predicate."""
    stage = mod._stage_converter_sources("b9000")
    Path(stage, "convert_hf_to_gguf.py").write_bytes(b"from conversion import ModelBase\n")
    before = staging_env["downloads"]
    repaired = mod._stage_converter_sources("b9000")
    assert repaired == stage
    assert staging_env["downloads"] == before + 1, "the damaged entry was served warm"
    assert mod._CONVERTER_ADD_ARGUMENT_RE.search(
        Path(stage, "convert_hf_to_gguf.py").read_bytes()
    ), "the entry was not actually restaged"


def test_a_real_shim_is_still_accepted(mod, tmp_path):
    stage = _write_source_tree(tmp_path / "real")
    Path(stage, mod.UNSLOTH_CONVERTER_STAGE_FILENAME).write_text(json.dumps({
        "schema": mod.UNSLOTH_CONVERTER_STAGE_SCHEMA,
        "repo": "ggml-org/llama.cpp", "tag": "b9000",
        "archive_sha256": "x", "completed": True,
    }))
    assert mod._converter_stage_is_usable(
        str(stage), repo = "ggml-org/llama.cpp", tag = "b9000") is True


def test_a_cached_mirror_that_lost_its_write_bit_is_restored(mod, tmp_path, monkeypatch, staging_env):
    stage = mod._stage_converter_sources("b9000")
    monkeypatch.setattr(mod, "UNSLOTH_HOME", str(tmp_path / "home"))
    _read_only(stage)
    try:
        mirror = mod._writable_stage(stage, repo = "ggml-org/llama.cpp", tag = "b9000")
        assert mirror is not None
        _read_only(mirror)
        again = mod._writable_stage(stage, repo = "ggml-org/llama.cpp", tag = "b9000")
        assert again == mirror
        assert os.access(again, os.W_OK), "an unwritable mirror was handed back"
        Path(again, "unsloth_convert_hf_to_gguf.py").write_bytes(b"# patched\n")
    finally:
        os.chmod(stage, 0o755)


def test_a_mirror_whose_write_bit_cannot_be_restored_is_reported(mod, tmp_path, monkeypatch, staging_env):
    stage = mod._stage_converter_sources("b9000")
    monkeypatch.setattr(mod, "UNSLOTH_HOME", str(tmp_path / "home"))
    _read_only(stage)
    try:
        mirror = mod._writable_stage(stage, repo = "ggml-org/llama.cpp", tag = "b9000")
        _read_only(mirror)
        def refuse(*a, **k):
            raise PermissionError(1, "Operation not permitted")
        monkeypatch.setattr(mod.os, "chmod", refuse)
        assert mod._writable_stage(stage, repo = "ggml-org/llama.cpp", tag = "b9000") is None
    finally:
        monkeypatch.undo()
        os.chmod(stage, 0o755)


def test_a_mirror_tightened_recursively_is_restored_throughout(mod, tmp_path, monkeypatch, staging_env):
    stage = mod._stage_converter_sources("b9000")
    monkeypatch.setattr(mod, "UNSLOTH_HOME", str(tmp_path / "home"))
    _read_only(stage)
    try:
        mirror = mod._writable_stage(stage, repo = "ggml-org/llama.cpp", tag = "b9000")
        assert mirror is not None
        # Not the mirror root: a read-only root makes os.access short-circuit the
        # `or`, so _tree_is_writable is never consulted and this stops testing it.
        for relative in ("conversion", os.path.join("gguf-py", "gguf")):
            os.chmod(os.path.join(mirror, relative), 0o555)
        again = mod._writable_stage(stage, repo = "ggml-org/llama.cpp", tag = "b9000")
        assert again == mirror
        Path(again, "unsloth_convert_hf_to_gguf.py").write_bytes(b"# patched\n")
        Path(again, "conversion", "base.py").write_bytes(b"# patched\n")
        Path(again, "gguf-py", "gguf", "probe.tmp").write_bytes(b"x")
    finally:
        os.chmod(stage, 0o755)


def test_a_second_converter_filename_is_tried_when_the_first_is_a_bare_shim(mod, tmp_path, monkeypatch):
    bundle = tmp_path / "both_names"
    bundle.mkdir()
    (bundle / "convert_hf_to_gguf.py").write_bytes(_SHIM_ENTRYPOINT)
    (bundle / "convert-hf-to-gguf.py").write_bytes(_MONOLITH_ENTRYPOINT)
    _write_sibling_gguf_py(bundle)
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(bundle))
    info = mod._resolve_monolith_bundle_convert_script()
    assert info is not None, "the usable second spelling was never examined"
    assert Path(info[0]).name == "convert-hf-to-gguf.py"


def test_a_second_filename_that_is_also_unusable_is_still_refused(mod, tmp_path, monkeypatch):
    bundle = tmp_path / "both_bad"
    bundle.mkdir()
    (bundle / "convert_hf_to_gguf.py").write_bytes(_SHIM_ENTRYPOINT)
    (bundle / "convert-hf-to-gguf.py").write_bytes(b"# truncated\n")
    _write_sibling_gguf_py(bundle)
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(bundle))
    assert mod._resolve_monolith_bundle_convert_script() is None


def test_a_monolith_truncated_after_its_registrations_is_not_a_cache_hit(mod, tmp_path):
    """Registrations without a parser are not enough."""
    stage = _complete_stage(str(tmp_path / "s"), package = False,
                            repo = "ggml-org/llama.cpp", tag = "b9000")
    assert mod._converter_stage_is_usable(
        stage, repo = "ggml-org/llama.cpp", tag = "b9000") is True
    Path(stage, "convert_hf_to_gguf.py").write_bytes(
        b"@ModelBase.register(\"LlamaForCausalLM\")\nclass LlamaModel:\n    pass\n"
    )
    assert mod._converter_stage_is_usable(
        stage, repo = "ggml-org/llama.cpp", tag = "b9000") is False


def test_a_real_monolith_is_still_accepted(mod, tmp_path):
    stage = _write_source_tree(tmp_path / "pre_split",
                               entrypoint = _MONOLITH_ENTRYPOINT, conversion = False)
    Path(stage, mod.UNSLOTH_CONVERTER_STAGE_FILENAME).write_text(json.dumps({
        "schema": mod.UNSLOTH_CONVERTER_STAGE_SCHEMA,
        "repo": "ggml-org/llama.cpp", "tag": "b9000",
        "archive_sha256": "x", "completed": True,
    }))
    assert mod._converter_stage_is_usable(
        str(stage), repo = "ggml-org/llama.cpp", tag = "b9000") is True


def test_an_installed_monolith_without_a_parser_is_left_to_staging(mod, tmp_path, monkeypatch):
    bundle = tmp_path / "truncated_install"
    bundle.mkdir()
    (bundle / "convert_hf_to_gguf.py").write_bytes(
        b"@ModelBase.register(\"LlamaForCausalLM\")\nclass LlamaModel:\n    pass\n"
    )
    _write_sibling_gguf_py(bundle)
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(bundle))
    assert mod._resolve_monolith_bundle_convert_script() is None


def test_an_installed_monolith_with_a_parser_is_still_taken(mod, tmp_path, monkeypatch):
    bundle = tmp_path / "good_install"
    bundle.mkdir()
    (bundle / "convert_hf_to_gguf.py").write_bytes(_MONOLITH_ENTRYPOINT)
    _write_sibling_gguf_py(bundle)
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(bundle))
    info = mod._resolve_monolith_bundle_convert_script()
    assert info is not None
    assert Path(info[0]) == bundle / "convert_hf_to_gguf.py"


def test_a_driveable_second_spelling_wins_over_an_undriveable_first(mod, tmp_path, monkeypatch):
    bundle = tmp_path / "mixed_names"
    bundle.mkdir()
    (bundle / "convert_hf_to_gguf.py").write_bytes(
        b"@ModelBase.register(\"LlamaForCausalLM\")\nclass LlamaModel:\n    pass\n"
    )
    (bundle / "convert-hf-to-gguf.py").write_bytes(_MONOLITH_ENTRYPOINT)
    _write_sibling_gguf_py(bundle)
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(bundle))
    info = mod._resolve_monolith_bundle_convert_script()
    assert info is not None
    assert Path(info[0]).name == "convert-hf-to-gguf.py"


def test_a_user_checkout_under_the_cache_root_is_not_treated_as_a_stage(mod, tmp_path, monkeypatch):
    """Ancestry under the cache root is not ownership."""
    cache = tmp_path / "shared-cache"
    checkout = cache / "my-llama.cpp"
    _write_source_tree(checkout, entrypoint = _MONOLITH_ENTRYPOINT, conversion = False)
    monkeypatch.setattr(mod, "LLAMA_CPP_CONVERTER_CACHE_DIR", str(cache))
    assert mod._is_inside_converter_cache(str(checkout)) is True
    assert mod._is_converter_stage_dir(str(checkout)) is False


def test_a_real_stage_under_the_cache_root_is_still_a_stage(mod, staging_env):
    stage = mod._stage_converter_sources("b9000")
    assert mod._is_converter_stage_dir(stage) is True


def test_a_stage_whose_manifest_was_deleted_is_no_longer_claimed(mod, staging_env):
    stage = mod._stage_converter_sources("b9000")
    Path(stage, mod.UNSLOTH_CONVERTER_STAGE_FILENAME).unlink()
    assert mod._is_converter_stage_dir(stage) is False


def test_a_user_checkout_under_the_cache_root_is_not_written_into(mod, tmp_path, monkeypatch):
    """The patched converter must land in LLAMA_CPP_DEFAULT_DIR."""
    cache = tmp_path / "shared-cache"
    checkout = cache / "my-llama.cpp"
    _write_source_tree(checkout, entrypoint = _MONOLITH_ENTRYPOINT, conversion = False)
    default_dir = tmp_path / "default"
    default_dir.mkdir()
    monkeypatch.setattr(mod, "LLAMA_CPP_CONVERTER_CACHE_DIR", str(cache))
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(default_dir))
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", str(checkout))
    _no_network(mod, monkeypatch, "an explicit checkout must not download")
    mod._download_convert_hf_to_gguf_cached.cache_clear()
    try:
        patched_path, text_archs, _ = mod._download_convert_hf_to_gguf("unsloth_convert_hf_to_gguf")
    finally:
        mod._download_convert_hf_to_gguf_cached.cache_clear()
    assert "LlamaForCausalLM" in text_archs
    assert Path(patched_path).parent == default_dir, (
        "the patched converter was written into the user's own checkout because it "
        "sits under the cache root"
    )
    assert not (checkout / "unsloth_convert_hf_to_gguf.py").exists()


def test_atomic_writes_follow_the_current_umask_without_reading_it(mod, tmp_path, monkeypatch):
    """The mode must track a umask the application changes, and os.umask must never be called.

    Reading the umask means setting it to 0 and putting it back, which two threads can
    interleave into leaving the process at 0 forever; caching it instead goes stale."""
    import stat

    def _forbidden(*args, **kwargs):
        raise AssertionError(
            "os.umask was called: reading it is the race, caching it is the staleness"
        )

    monkeypatch.setattr(os, "umask", _forbidden)

    real_umask = os.umask.__wrapped__ if hasattr(os.umask, "__wrapped__") else None
    for mask in (0o022, 0o077, 0o002):
        monkeypatch.undo()
        old = os.umask(mask)
        try:
            monkeypatch.setattr(os, "umask", _forbidden)
            target = tmp_path / f"fresh_{mask:o}.txt"
            mod._atomic_write_bytes(str(target), b"payload")
            assert target.read_bytes() == b"payload"
            got = stat.S_IMODE(target.stat().st_mode)
            assert got == 0o666 & ~mask, (
                f"umask {mask:04o} produced {got:04o}, expected {0o666 & ~mask:04o}: "
                f"the mode is not following the umask in force at the write"
            )
        finally:
            monkeypatch.undo()
            os.umask(old)


def test_an_atomic_write_keeps_an_existing_files_mode(mod, tmp_path):
    """Replacing a shared 0644 converter must not silently make it owner-only."""
    import stat

    target = tmp_path / "converter.py"
    target.write_bytes(b"old")
    os.chmod(target, 0o640)
    mod._atomic_write_bytes(str(target), b"new")
    assert target.read_bytes() == b"new"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


def test_the_temp_sibling_retries_a_name_collision(mod, tmp_path, monkeypatch):
    """A colliding random name must be retried, not raised at the caller."""
    names = iter([b"\x01" * 8, b"\x01" * 8, b"\x02" * 8])
    monkeypatch.setattr(os, "urandom", lambda n: next(names))
    first_fd, first_path = mod._open_new_sibling(str(tmp_path), 0o666)
    os.close(first_fd)
    second_fd, second_path = mod._open_new_sibling(str(tmp_path), 0o666)
    os.close(second_fd)
    assert first_path != second_path


def test_an_incomplete_install_is_not_synthesized_as_an_authoritative_pin(mod, tmp_path):
    """The MLX path must let a shim-without-conversion install reach the staged resolver."""
    incomplete = _write_source_tree(tmp_path / "incomplete", conversion = False)
    assert mod._converter_dir_is_incomplete(str(incomplete)), (
        "a shim entrypoint with no conversion/ beside it is the case staging exists for"
    )

    complete = _write_source_tree(tmp_path / "complete", conversion = True)
    assert not mod._converter_dir_is_incomplete(str(complete))

    monolith = _write_source_tree(
        tmp_path / "monolith", entrypoint = _MONOLITH_ENTRYPOINT, conversion = False,
    )
    assert not mod._converter_dir_is_incomplete(str(monolith)), (
        "a self-contained converter needs no conversion/ and is not incomplete"
    )

    empty = tmp_path / "empty"
    empty.mkdir()
    assert not mod._converter_dir_is_incomplete(str(empty)), (
        "a directory with no converter at all is empty, not incomplete"
    )


def test_an_unusable_prebuilt_marker_is_announced_not_silently_ignored(mod, tmp_path, caplog):
    """A broken marker must not silently downgrade to 'stage whatever is latest'."""
    for name, payload in (
        ("not_json", "{not json at all"),
        ("not_an_object", '["b7062"]'),
        ("no_tag", '{"repo": "ggml-org/llama.cpp"}'),
        ("blank_tag", '{"tag": "   "}'),
    ):
        install = tmp_path / name
        install.mkdir()
        (install / mod.UNSLOTH_PREBUILT_INFO_FILENAME).write_text(payload)
        caplog.clear()
        with caplog.at_level("WARNING"):
            repo, tag = mod._read_prebuilt_marker(str(install))
        assert (repo, tag) == (None, None)
        assert any("Ignoring" in r.message for r in caplog.records), (
            f"a {name} marker was ignored with no warning, so the export silently "
            f"stops matching the installed binaries"
        )

    absent = tmp_path / "absent"
    absent.mkdir()
    caplog.clear()
    with caplog.at_level("WARNING"):
        assert mod._read_prebuilt_marker(str(absent)) == (None, None)
    assert not caplog.records, "an absent marker is normal and must stay silent"


def test_the_converter_cache_env_var_is_read_at_the_call_not_at_import(mod, tmp_path, monkeypatch):
    """Setting the cache root after `import unsloth` must work, as the siblings promise."""
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_CONVERTER_CACHE", raising = False)
    assert mod._converter_cache_root() == mod.LLAMA_CPP_CONVERTER_CACHE_DIR

    late = tmp_path / "set-after-import"
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_CONVERTER_CACHE", str(late))
    assert mod._converter_cache_root() == str(late), (
        "the cache root was frozen at import, so a notebook that sets it in a later "
        "cell silently keeps staging into the default location"
    )

    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_CONVERTER_CACHE", "   ")
    assert mod._converter_cache_root() == mod.LLAMA_CPP_CONVERTER_CACHE_DIR


def test_a_stale_mirror_is_replaced_rather_than_blocking_every_retry(
    mod, tmp_path, monkeypatch, staging_env,
):
    """A mirror written under an older stage schema must not wedge the read-only path.

    UNSLOTH_CONVERTER_STAGE_SCHEMA exists to invalidate entries an older unsloth_zoo
    wrote, so an unusable mirror is a state a release is expected to create. Skipping
    publication because the path merely exists left the good copy deleted and every
    later call returning None."""
    stage = mod._stage_converter_sources("b9000")
    home = tmp_path / "home"
    monkeypatch.setattr(mod, "UNSLOTH_HOME", str(home))
    _read_only(stage)
    try:
        mirror = mod._writable_stage(stage, repo = "ggml-org/llama.cpp", tag = "b9000")
        assert mirror is not None

        manifest = Path(mirror, mod.UNSLOTH_CONVERTER_STAGE_FILENAME)
        stale = json.loads(manifest.read_text())
        stale["schema"] = mod.UNSLOTH_CONVERTER_STAGE_SCHEMA + 1
        manifest.write_text(json.dumps(stale))
        assert not mod._converter_stage_is_usable(
            mirror, repo = "ggml-org/llama.cpp", tag = "b9000",
        )

        for attempt in range(3):
            again = mod._writable_stage(stage, repo = "ggml-org/llama.cpp", tag = "b9000")
            assert again == mirror, (
                f"attempt {attempt + 1} returned {again}: a stale mirror blocks the "
                f"read-only cache path forever instead of being replaced"
            )
        assert mod._converter_stage_is_usable(
            mirror, repo = "ggml-org/llama.cpp", tag = "b9000",
        )
        assert not list(Path(mirror).parent.glob("*.superseded_*")), (
            "the displaced tree was left behind"
        )
    finally:
        os.chmod(stage, 0o755)


def test_the_incomplete_remedy_names_the_knob_actually_in_force(mod, tmp_path, monkeypatch):
    """Advice the user can follow, not advice a higher-priority knob will override."""
    pinned = tmp_path / "pinned_incomplete"
    pinned.mkdir()

    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", str(pinned))
    pinned_remedy = mod._incomplete_sources_remedy(str(pinned))
    assert "UNSLOTH_LLAMA_CPP_SCRIPTS_DIR" in pinned_remedy
    assert "outranks" in pinned_remedy
    assert str(pinned) in pinned_remedy

    # A pin pointing somewhere else did not select this directory, so the tag is
    # still the right advice.
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", str(tmp_path / "elsewhere"))
    assert "UNSLOTH_LLAMA_CPP_CONVERTER_TAG" in mod._incomplete_sources_remedy(str(pinned))

    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", raising = False)
    unpinned_remedy = mod._incomplete_sources_remedy(str(pinned))
    assert "UNSLOTH_LLAMA_CPP_CONVERTER_TAG" in unpinned_remedy
    assert "outranks" not in unpinned_remedy
