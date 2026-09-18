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

import importlib.util
import json
import os
import sys
import tarfile
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
_MONOLITH_ENTRYPOINT = b"""\
#!/usr/bin/env python3
import gguf


@ModelBase.register("LlamaForCausalLM")
class LlamaModel(TextModel):
    pass
"""


def _build_source_tarball(path, *, entrypoint = _SHIM_ENTRYPOINT, conversion = True,
                          gguf_py = True, tag = "b9000"):
    """Write a llama.cpp source tarball nested under llama.cpp-{tag}/, the way
    codeload serves one."""
    root = Path(path).parent / f"_src_{tag}"
    inner = root / f"llama.cpp-{tag}"
    (inner).mkdir(parents = True, exist_ok = True)
    (inner / "convert_hf_to_gguf.py").write_bytes(entrypoint)
    if gguf_py:
        pkg = inner / "gguf-py" / "gguf"
        pkg.mkdir(parents = True, exist_ok = True)
        (pkg / "__init__.py").write_text("# gguf\n")
        (pkg / "tensor_mapping.py").write_text("class TensorNameMap:\n    pass\n")
    if conversion:
        conv = inner / "conversion"
        conv.mkdir(parents = True, exist_ok = True)
        (conv / "__init__.py").write_text("TEXT_MODEL_MAP = {'LlamaForCausalLM': 'llama'}\n")
        (conv / "base.py").write_text("class ModelBase:\n    pass\n")
        (conv / "qwen.py").write_text("# qwen\n")
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
    conversion/ it imports rather than in a directory where it cannot resolve."""
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", raising = False)
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(tmp_path / "empty_install"))
    monkeypatch.setattr(mod, "_resolve_converter_revision", lambda d: ("ggml-org/llama.cpp", "b9000"))
    mod._download_convert_hf_to_gguf_cached.cache_clear()
    try:
        patched_path, text_archs, vision_archs = mod._download_convert_hf_to_gguf("unsloth_convert_hf_to_gguf")
    finally:
        mod._download_convert_hf_to_gguf_cached.cache_clear()
    stage = mod._converter_stage_dir("ggml-org/llama.cpp", "b9000")
    assert os.path.dirname(patched_path) == stage
    assert os.path.isfile(os.path.join(stage, "conversion", "__init__.py"))
    # Archs come from the staged conversion/__init__.py, not from an empty scan.
    assert "LlamaForCausalLM" in text_archs
