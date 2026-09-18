"""Tests for co-versioned llama.cpp converter source staging.

Upstream split convert_hf_to_gguf.py: the entrypoint is now a shim that does
`from conversion import ...` and inserts its sibling gguf-py onto sys.path. Unsloth
used to download that one file from refs/heads/master and pair it with whatever
conversion/ and gguf-py/ happened to be in LLAMA_CPP_DEFAULT_DIR, so the entrypoint
and its libraries were free to be different revisions.

These tests cover the staging that removes that skew, and specifically the failure
modes it has to survive: an interrupted stage, two processes racing, a hand-damaged
cache entry, and a machine with no network.

No network, no GPU, no llama.cpp toolchain. Every download is monkeypatched to build
a tarball on disk, so the archive handling is real while the transfer is not.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
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
    """Every test starts on a machine with no converter env vars and no llama.cpp.

    LLAMA_CPP_DEFAULT_DIR defaults to ~/.unsloth/llama.cpp, and the resolvers read it
    at the call, so without this a real install on the developer's box decides what
    the precedence tests see. Tests that want an install point it somewhere real."""
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


# The shim shape upstream ships today: no model classes, imports the package.
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

# A revision predating the split: self-contained, legitimately has no conversion/.
# Carries the argparse block too, because the patcher refuses a converter it cannot
# read flags out of, and a pre-split install has to drive it end to end.
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

# conversion/ as upstream ships it, cut down to the three things the patcher looks
# for: the two arch maps, the metadata line the branding patch anchors on, and a
# qwen.py that already handles the expert aliases.
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
    """Write a llama.cpp source tarball nested under llama.cpp-{tag}/, the way
    codeload serves one."""
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
    """Point the cache at tmp_path and serve every download from a real tarball.
    Returns a dict with a `downloads` counter so tests can assert on network use."""
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
    # Nothing here should ever reach the releases API; make it loud if it does.
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
    """The point of keying on an immutable tag: no freshness check, no request."""
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
    """A monolith entrypoint does not import conversion/, so a tarball without one
    is complete rather than broken. The requirement is read off the entrypoint."""
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
    """Staging half a revision is the exact defect this cache exists to remove, so
    it is refused in staging rather than published and discovered later."""
    staging_env["conversion"] = False
    assert mod._stage_converter_sources("b9000") is None
    assert not os.path.exists(mod._converter_stage_dir("ggml-org/llama.cpp", "b9000"))


def test_a_failed_stage_leaves_a_working_entry_intact(mod, staging_env):
    good = mod._stage_converter_sources("b9000")
    marker = Path(good, "conversion", "__init__.py").read_text()
    staging_env["fail"] = RuntimeError("network died mid-refresh")
    # Same tag: a cache hit, so it never even tries. Force the miss by damaging
    # only the manifest, which is what an interrupted publish looks like.
    Path(good, mod.UNSLOTH_CONVERTER_STAGE_FILENAME).unlink()
    assert mod._stage_converter_sources("b9000") is None
    # The trees are still there; only the manifest, written last, was missing.
    assert Path(good, "conversion", "__init__.py").read_text() == marker


def test_a_damaged_entry_repairs_itself_instead_of_wedging(mod, staging_env):
    """A directory that exists but fails the probe used to block its own
    replacement: exists() skipped the publish, the probe then refused the entry, and
    every later export re-downloaded and re-refused forever."""
    stage = mod._stage_converter_sources("b9000")
    os.unlink(os.path.join(stage, "gguf-py", "gguf", "__init__.py"))
    assert mod._converter_stage_is_usable(stage) is False

    repaired = mod._stage_converter_sources("b9000")
    assert repaired == stage
    assert mod._converter_stage_is_usable(stage) is True
    # And it is a cache hit again afterwards, not a permanent re-download.
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
    """`completed` is written last, so its absence is precisely the signal that a
    previous attempt died part-way through."""
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
    """An entry staged by an older unsloth_zoo is re-staged, not trusted."""
    stage = mod._stage_converter_sources("b9000")
    path = Path(stage, mod.UNSLOTH_CONVERTER_STAGE_FILENAME)
    manifest = json.loads(path.read_text(encoding = "utf-8"))
    manifest["schema"] = mod.UNSLOTH_CONVERTER_STAGE_SCHEMA - 1
    path.write_text(json.dumps(manifest), encoding = "utf-8")
    assert mod._converter_stage_is_usable(stage) is False


def test_a_complete_manifest_over_a_gutted_tree_is_not_a_hit(mod, staging_env):
    """The probe checks the trees too, so a hand-deleted gguf-py re-stages instead
    of silently reproducing the skew."""
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
    """Two processes staging the same immutable tag produce equivalent trees, so the
    loser takes the winner's rather than failing or overwriting it."""
    real_move = mod.shutil.move
    def racing_move(src, dst):
        # Simulate the other process publishing between our check and our move.
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
    # A prebuilt marker is present and still loses to the explicit pin.
    mod._write_prebuilt_marker(str(tmp_path), "b5678", "asset.tar.gz")
    assert mod._resolve_converter_revision(str(tmp_path)) == ("ggml-org/llama.cpp", "b1234")


def test_the_prebuilt_marker_tag_is_used_when_there_is_no_pin(mod, monkeypatch, tmp_path):
    """_write_prebuilt_marker has always recorded this and nothing read it back."""
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
    """_resolve_converter_revision runs on every export. Without memoization a warm
    converter cache still paid a releases API round-trip each time, which is both a
    latency cost and a way for an intermittent network to fail an export that needed
    nothing from it."""
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
    """The pin is the memo key rather than being read inside, so a change mid-process
    is still honoured."""
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
    """A tag arrives from a release API and from an env var, so neither half of the
    key is trusted to stay inside the cache."""
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
    """A fork tag and an upstream tag can collide by name; the repo keeps them apart."""
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
    """UNSLOTH_LLAMA_CPP_SCRIPTS_DIR stays authoritative: it resolves first, so the
    staging resolver is never consulted and no request is made."""
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
    """The whole point: a shim with no conversion/ used to be patched as a monolith
    and written where its own import cannot resolve. Now it says what is wrong."""
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
    assert "UNSLOTH_LLAMA_CPP_CONVERTER_TAG" in message
    assert "same llama.cpp revision" in message


def test_the_incomplete_error_is_not_wrapped_in_the_generic_one(mod, tmp_path, monkeypatch):
    """Its own type, so the broad `except Exception` re-raises it unchanged rather
    than burying the actionable text under an introspection failure."""
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
    """End to end: with nothing co-versioned on disk, the patcher stages all three
    trees and then sees a package layout, so the patched entrypoint lands beside the
    conversion/ it imports rather than in a directory where it cannot resolve.

    Also the compatibility claim in full: a staged tree is indistinguishable from a
    prebuilt bundle, so every existing downstream step lands on it with no special
    case. Arch extraction reads the staged conversion/__init__.py, the branding patch
    edits the staged conversion/base.py, and the result still parses."""
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
    # Archs come from the staged conversion/__init__.py, not from an empty scan.
    assert "LlamaForCausalLM" in text_archs
    assert "Gemma3ForConditionalGeneration" in vision_archs
    # Branding went into the staged conversion/base.py, not into a bundle elsewhere.
    assert mod._UNSLOTH_BRANDING_MARKER in Path(stage, "conversion", "base.py").read_bytes()
    ast.parse(Path(patched_path).read_bytes())


def test_the_staged_gguf_py_is_the_one_offered_the_qwen35_mapping(mod, staging_env, monkeypatch):
    """_patch_tensor_mapping_for_qwen35 is anchored on the converter's own directory,
    so staging points it at the gguf-py the child will actually import."""
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
    """The patcher cache is lru_cache(1) keyed on (path, mtime, size). Two tags live
    at different paths, so switching tags cannot be served the other one's result."""
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
# entrypoint. The first three are offline, and the tests below prove it by making
# any network call an immediate failure rather than by inspecting logs.

def _no_network(mod, monkeypatch, why):
    """Turn every route to the network into a test failure."""
    def _trap(*args, **kwargs):
        raise AssertionError(why)
    monkeypatch.setattr(mod, "_download_archive", _trap)
    monkeypatch.setattr(mod, "_resolve_llama_cpp_release", _trap)
    monkeypatch.setattr(mod, "_resolve_converter_revision", _trap)
    monkeypatch.setattr(mod.requests, "get", _trap)


def test_a_bundle_with_its_conversion_package_never_reaches_staging(mod, tmp_path, monkeypatch):
    """A prebuilt bundle is already co-versioned, so nothing new may run."""
    bundle = _write_source_tree(tmp_path / "bundle")
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(bundle))
    _no_network(mod, monkeypatch, "a co-versioned bundle must not trigger staging")
    info = mod._resolve_bundle_convert_script()
    assert info is not None
    assert Path(info[0]).parent == bundle


def test_an_installed_self_contained_converter_is_used_instead_of_a_download(mod, tmp_path, monkeypatch):
    """A pre-split install has a working converter that matches its binaries. It
    used to be ignored in favour of a master shim it could never run."""
    bundle = tmp_path / "old_install"
    bundle.mkdir()
    (bundle / "convert_hf_to_gguf.py").write_bytes(_MONOLITH_ENTRYPOINT)
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(bundle))
    info = mod._resolve_monolith_bundle_convert_script()
    assert info is not None
    assert Path(info[0]) == bundle / "convert_hf_to_gguf.py"


def test_a_shim_with_no_package_beside_it_is_not_offered_as_self_contained(mod, tmp_path, monkeypatch):
    """The distinction _resolve_bundle_convert_script cannot make on directory
    contents alone: this entrypoint imports conversion/, so it is half a revision
    and must not be served as a complete converter."""
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
    """Behavioural rather than structural: drive the real entry point on a pre-split
    install. Without the monolith row this population resolves a revision, downloads
    a tarball and only then discovers that revision has no conversion/."""
    bundle = tmp_path / "old_install"
    bundle.mkdir()
    (bundle / "convert_hf_to_gguf.py").write_bytes(_MONOLITH_ENTRYPOINT)
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(bundle))
    _no_network(mod, monkeypatch, "an installed self-contained converter must not download")
    mod._download_convert_hf_to_gguf_cached.cache_clear()
    try:
        patched_path, text_archs, _ = mod._download_convert_hf_to_gguf("unsloth_convert_hf_to_gguf")
    finally:
        mod._download_convert_hf_to_gguf_cached.cache_clear()
    # The registrations came out of the installed monolith, not from an empty scan.
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
    """The population this change exists for: nothing on disk, so what used to happen
    was a lone shim download. What happens now is a tree that can actually run."""
    monkeypatch.setattr(mod, "_resolve_converter_revision", lambda d: ("ggml-org/llama.cpp", "b9000"))
    info = mod._resolve_staged_convert_script()
    assert info is not None
    staged_dir = str(Path(info[0]).parent)
    assert mod._detect_converter_layout(Path(info[0]).read_bytes(), staged_dir) == "package"


# --- the staging kill switch --------------------------------------------------

def test_staging_can_be_switched_off(mod, staging_env, monkeypatch):
    """The escape hatch for anyone the new path surprises: behave exactly as before,
    with no edit to any file."""
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
    """The switch is about the export-time cache. A prebuilt install still hydrates
    its own converter, because those sources are part of the install."""
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
    """Two exports racing on the same revision end up with one usable tree, not a
    half-copied one. Concurrency is handled by the immutable key, not by a lock."""
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
    """Staging must resolve the same URL the prebuilt install would have, so the two
    paths cannot drift on what a fork "mix" tag means."""
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
    """Someone pinning a read-only checkout keeps working: resolution only stats."""
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
# Sources are pinned to the revision of the installed binaries, so an architecture
# that only exists on master will not convert until a release cuts. That is the
# accepted trade. It is only acceptable if the person who hits it can see which
# revision refused them and which knob moves it.

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
    """A pinned checkout, a bundle or an old monolith was never staged, so there is
    no revision to name and the short message is the honest one."""
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


def test_an_unsupported_arch_on_an_unstaged_converter_keeps_the_short_message(mod, tmp_path):
    checkout = _write_source_tree(tmp_path / "checkout")
    message = mod._unsupported_arch_message(
        "BrandNewForCausalLM", str(checkout / "convert_hf_to_gguf.py"),
    )
    assert message.endswith("converting model types of `BrandNewForCausalLM`.")
    assert "UNSLOTH_LLAMA_CPP_CONVERTER_TAG" not in message


def test_convert_to_gguf_hands_the_user_the_staged_tag_and_the_escape_hatch(mod, tmp_path, monkeypatch):
    """End to end through the public entry point: the pin cost surfaces where the
    user meets it, not only in a helper."""
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


# ---------------------------------------------------------------------------
# Which spellings of the conversion import count.
#
# Three decisions read this: whether an installed converter is offered as already
# self-contained, whether a staged tree is missing a tree it needs, and whether a
# layout is package / monolith / incomplete. A substring test for the one spelling
# upstream happens to use today reads every other spelling as self-contained, and
# the child then dies on an import the entrypoint really does make.
# ---------------------------------------------------------------------------

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
    """A converter that needs the package and cannot see it is 'incomplete', for
    every spelling. Reading only `from conversion import` sent the others back to
    'monolith', which is the exact misdetection this branch exists to remove."""
    llama_cpp = _load_llama_cpp_module()
    expected = "incomplete" if imports_it else "monolith"
    assert llama_cpp._detect_converter_layout(source, str(tmp_path)) == expected


def test_an_entrypoint_that_will_not_parse_is_not_called_self_contained(tmp_path):
    """Unparseable source falls back to the substring rather than to 'no import',
    because guessing 'self-contained' there serves a converter that cannot run."""
    llama_cpp = _load_llama_cpp_module()
    broken = b"from conversion import ModelBase\nthis is not python(\n"
    assert llama_cpp._source_imports_conversion_package(broken) is True


def test_a_shim_using_a_submodule_import_is_not_offered_as_self_contained(tmp_path):
    """End to end over the resolver: the installed converter imports conversion.base
    and the package is absent, so it must not be served as a complete monolith."""
    llama_cpp = _load_llama_cpp_module()
    converter = tmp_path / "convert_hf_to_gguf.py"
    converter.write_bytes(b"from conversion.base import ModelBase\n")
    assert llama_cpp._entrypoint_needs_conversion_package(str(converter)) is True


# ---------------------------------------------------------------------------
# Cache identity, publication, and where a staged monolith's patched file lands.
# All four of these were found by driving the staging code rather than reading it.
# ---------------------------------------------------------------------------

def _complete_stage(root, *, package, repo, tag, schema = None, manifest_repo = None,
                    manifest_tag = None):
    """A stage directory that passes every structural check, so the only thing
    under test is the identity comparison."""
    import json as _json
    llama_cpp = _load_llama_cpp_module()
    os.makedirs(os.path.join(root, "gguf-py", "gguf"), exist_ok = True)
    open(os.path.join(root, "gguf-py", "gguf", "__init__.py"), "w").close()
    if package:
        conv = os.path.join(root, "conversion"); os.makedirs(conv, exist_ok = True)
        open(os.path.join(conv, "__init__.py"), "w").close()
        open(os.path.join(conv, "base.py"), "w").close()
        body = b"from conversion import ModelBase\n"
    else:
        body = b"class ModelBase:\n    pass\n"
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
    """Both halves of the directory name are sanitised, so distinct tags can collide
    on one directory. Serving whatever is in it would hand the caller a different
    revision than it asked for, under a cache hit, with no download to notice."""
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
    """`v/a` and `v_a` name the same directory. The identity check is what stops the
    second one being handed the first one's tree."""
    llama_cpp = _load_llama_cpp_module()
    monkeypatch.setattr(llama_cpp, "LLAMA_CPP_CONVERTER_CACHE_DIR", str(tmp_path / "cache"))
    a = llama_cpp._converter_stage_dir("ggml-org/llama.cpp", "v/a")
    b = llama_cpp._converter_stage_dir("ggml-org/llama.cpp", "v_a")
    assert a == b, "precondition: these tags collide"
    os.makedirs(a, exist_ok = True)
    _complete_stage(a, package = True, repo = "ggml-org/llama.cpp", tag = "v/a")
    assert llama_cpp._converter_stage_is_usable(a, repo = "ggml-org/llama.cpp", tag = "v/a") is True
    assert llama_cpp._converter_stage_is_usable(b, repo = "ggml-org/llama.cpp", tag = "v_a") is False


def test_publishing_onto_a_directory_that_appeared_does_not_nest_the_tree(mod, staging_env, monkeypatch):
    """Another process publishes while we are downloading, and we still reach the move.

    Two things have to be true to enter that window, and both are arranged here
    rather than hoped for: the entry probe must miss (so the winner publishes only
    once our download has started), and the existence check guarding the move must
    answer False (a one-statement window that no thread can be made to hit
    reliably, so it is simulated precisely).

    shutil.move onto an EXISTING directory moves the source INSIDE it and raises
    nothing, so the loser's whole tree would be buried at <stage>/sources/ and would
    stay there for every later cache hit. os.rename raises instead, which is the
    signal the loser needs to discard its copy and adopt the published tree."""
    stage_dir = mod._converter_stage_dir("ggml-org/llama.cpp", "b9000")
    winner_listing = {}

    real_download = mod._download_archive

    def download_then_lose_the_race(url, dest_path):
        real_download(url, dest_path)
        # The winner publishes now: after our entry probe missed, before our move.
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
            # 1st: the repair check. 2nd: the guard immediately before the move.
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
    """A staged revision predating the split has no conversion/, so its layout is
    'monolith'. Writing its patched entrypoint to the default install directory
    would make the entrypoint's own sys.path.insert(1, __file__.parent / 'gguf-py')
    resolve against the INSTALL's gguf-py instead of the co-versioned tree staged
    right beside it, which is the revision skew this staging exists to remove."""
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
    """The anchoring rule must not start writing into a directory the user owns."""
    llama_cpp = _load_llama_cpp_module()
    monkeypatch.setattr(llama_cpp, "LLAMA_CPP_CONVERTER_CACHE_DIR", str(tmp_path / "cache"))
    assert llama_cpp._is_inside_converter_cache(str(tmp_path / "my-llama.cpp")) is False


def test_the_converter_tag_pin_outranks_a_leftover_self_contained_converter(mod, staging_env, monkeypatch):
    """The unsupported-architecture message tells the user to set
    UNSLOTH_LLAMA_CPP_CONVERTER_TAG. If a leftover self-contained converter on disk
    won over that pin, the documented way out of "this revision does not know your
    architecture" would do nothing, and the user would have no way to act on the
    error they were just handed."""
    install = tmp = Path(staging_env["cache"]).parent / "install"
    _write_source_tree(install, entrypoint = _MONOLITH_ENTRYPOINT, conversion = False)
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(install))

    # Without the pin the installed converter answers, offline.
    mod._download_convert_hf_to_gguf.cache_clear()
    chosen, _t, _v = mod._download_convert_hf_to_gguf()
    assert os.path.dirname(chosen) == str(install)
    assert staging_env["downloads"] == 0

    # With the pin, the named revision is staged instead.
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_CONVERTER_TAG", "b9000")
    mod._download_convert_hf_to_gguf.cache_clear()
    chosen, _t, _v = mod._download_convert_hf_to_gguf()
    mod._download_convert_hf_to_gguf.cache_clear()
    assert os.path.dirname(chosen) == mod._converter_stage_dir("ggml-org/llama.cpp", "b9000")
    assert staging_env["downloads"] == 1


@pytest.mark.skipif(os.name == "nt", reason = "Windows has no POSIX mode bits; chmod there only toggles the read-only flag, so st_mode & 0o777 is never 0o644. The Windows-relevant property is covered by the readability check below.")
def test_replacing_a_file_keeps_the_mode_it_had(tmp_path):
    """mkstemp creates 0600 and os.replace carries that mode onto the destination,
    so replacing a world-readable converter would quietly make it owner-only and
    every other user of a shared install would lose the read access the plain
    open(path, 'wb') this replaced had left them."""
    mod = _load_llama_cpp_module()
    target = tmp_path / "converter.py"
    target.write_bytes(b"old\n")
    os.chmod(str(target), 0o644)
    mod._atomic_write_bytes(str(target), b"new\n")
    assert target.read_bytes() == b"new\n"
    assert os.stat(str(target)).st_mode & 0o777 == 0o644


@pytest.mark.skipif(os.name == "nt", reason = "umask is POSIX only")
def test_a_brand_new_file_is_not_created_owner_only(tmp_path):
    """No destination to copy a mode from: fall back to what an ordinary create
    would have produced under the active umask, not to mkstemp's 0600."""
    mod = _load_llama_cpp_module()
    target = tmp_path / "fresh.py"
    mod._atomic_write_bytes(str(target), b"x\n")
    umask = os.umask(0); os.umask(umask)
    assert os.stat(str(target)).st_mode & 0o777 == 0o666 & ~umask


def test_an_atomically_replaced_file_stays_readable_on_every_platform(tmp_path):
    """The portable half of the same guarantee, and the one that holds on Windows:
    whatever the mode representation, the replaced file must still be readable and
    writable by the process that owns the install."""
    mod = _load_llama_cpp_module()
    target = tmp_path / "converter.py"
    target.write_bytes(b"old\n")
    mod._atomic_write_bytes(str(target), b"new\n")
    assert target.read_bytes() == b"new\n"
    assert os.access(str(target), os.R_OK)
    assert os.access(str(target), os.W_OK)
