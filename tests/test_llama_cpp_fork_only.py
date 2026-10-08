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

"""llama.cpp prebuilts and converter sources come only from unslothai/llama.cpp
releases, never from ggml-org. No network: every GitHub call is stubbed."""

import hashlib
import importlib.util
import json
import os
import platform
import sys
import tarfile
from pathlib import Path

import pytest

FORK = "unslothai/llama.cpp"


def _load():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "llama_cpp_fork_only_under_test", root / "unsloth_zoo" / "llama_cpp.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mod(tmp_path, monkeypatch):
    for name in (
        "UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", "UNSLOTH_LLAMA_CPP_CONVERTER_TAG",
        "UNSLOTH_CONVERTER_STAGE", "UNSLOTH_LLAMA_TAG", "UNSLOTH_LLAMA_CPP_OFFLINE",
        "UNSLOTH_OFFLINE", "UNSLOTH_LLAMA_FORCE_COMPILE", "UNSLOTH_LLAMA_CPP_CONVERTER_CACHE",
    ):
        monkeypatch.delenv(name, raising = False)
    module = _load()
    monkeypatch.setattr(module, "LLAMA_CPP_CONVERTER_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(module, "LLAMA_CPP_DEFAULT_DIR", str(tmp_path / "no_llama_cpp"))
    return module


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


def _release(tag, draft = False, prerelease = False):
    return {"tag_name": tag, "draft": draft, "prerelease": prerelease, "assets": []}


def _serve_release_list(mod, monkeypatch, pages):
    """Answer the fork's paginated release list; refuse every other URL."""
    calls = []

    def _get(url, **kwargs):
        calls.append(url)
        assert url.startswith(mod.LLAMA_CPP_PUBLISHED_RELEASES_API + "?"), url
        page = int(url.rsplit("page=", 1)[1])
        return _Response(pages[page - 1] if page <= len(pages) else [])
    monkeypatch.setattr(mod, "_requests_get_with_retries", _get)
    return calls


# --- prebuilt binaries ---------------------------------------------------------

@pytest.mark.parametrize("fork_release", [None, ("b9585-mix-abc", {})])
def test_a_cpu_install_never_falls_back_to_ggml_org(mod, monkeypatch, tmp_path, fork_release):
    """Fork unreachable, or a fork release with no bundle for this host: compile
    from source. Before, both cases went on to download ggml-org's CPU build."""
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    apis = []

    def _resolve(releases_api = None):
        apis.append(releases_api)
        if releases_api == mod.LLAMA_CPP_PUBLISHED_RELEASES_API:
            return fork_release
        return ("b9000", {"llama-b9000-bin-ubuntu-x64.tar.gz": "https://upstream.invalid/x.tar.gz"})
    monkeypatch.setattr(mod, "_resolve_llama_cpp_release", _resolve)
    monkeypatch.setattr(mod, "_fetch_release_json_asset", lambda assets, name: {})
    staged = []
    monkeypatch.setattr(
        mod, "_stage_prebuilt_install",
        lambda *a, **k: staged.append((k.get("repo"), a[2])) or pytest.fail("nothing should install"),
    )
    assert mod._install_llama_cpp_prebuilt(str(tmp_path / "llama.cpp"), gpu_support = False) is None
    assert staged == []
    assert all("unslothai/llama.cpp" in (api or "") for api in apis), apis


def test_every_releases_api_constant_is_the_fork(mod):
    assert "unslothai/llama.cpp" in mod.LLAMA_CPP_RELEASES_API
    assert "unslothai/llama.cpp" in mod.LLAMA_CPP_PUBLISHED_RELEASES_API
    assert "unslothai/llama.cpp" in mod.LLAMA_CPP_SOURCE_TARBALL
    assert "ggml-org" not in mod.LLAMA_CPP_SOURCE_TARBALL


def test_a_new_marker_records_the_fork_by_default(mod, tmp_path):
    mod._write_prebuilt_marker(str(tmp_path), "b9585-mix-abc", "asset.tar.gz")
    info = json.loads((tmp_path / mod.UNSLOTH_PREBUILT_INFO_FILENAME).read_text())
    assert info["repo"] == FORK


# --- converter revision resolution ----------------------------------------------

def test_a_pinned_upstream_tag_maps_to_the_newest_fork_release_built_on_it(mod, monkeypatch):
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_CONVERTER_TAG", "b1234")
    _serve_release_list(mod, monkeypatch, [[
        _release("b1300-mix-0000000"),
        _release("b1234-mix-draft00", draft = True),
        _release("b1234-mix-prerel0", prerelease = True),
        _release("b1234-mix-abc1234"),
        _release("b1234-mix-old0000"),
    ]])
    assert mod._resolve_converter_revision("/nonexistent") == (FORK, "b1234-mix-abc1234")


def test_the_mapping_reads_past_the_first_page(mod, monkeypatch):
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_CONVERTER_TAG", "b1000")
    first = [_release(f"b{2000 + i}-mix-aaaaaaa") for i in range(100)]
    calls = _serve_release_list(mod, monkeypatch, [first, [_release("b1000-mix-bbbbbbb")]])
    assert mod._resolve_converter_revision("/nonexistent") == (FORK, "b1000-mix-bbbbbbb")
    assert len(calls) == 2


def test_a_pinned_tag_with_no_fork_release_resolves_to_nothing(mod, monkeypatch):
    """No upstream fallback: the caller then fails the export naming the pin."""
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_CONVERTER_TAG", "b1234")
    _serve_release_list(mod, monkeypatch, [[_release("b1300-mix-0000000"),
                                            _release("b1234-mix-draft00", draft = True)]])
    assert mod._resolve_converter_revision("/nonexistent") == (None, None)


def test_an_unstageable_pin_names_the_fork_releases(mod, monkeypatch):
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_CONVERTER_TAG", "b1234")
    _serve_release_list(mod, monkeypatch, [[]])
    with pytest.raises(mod._ConverterSourcesIncomplete) as error:
        mod._download_convert_hf_to_gguf()
    assert "https://github.com/unslothai/llama.cpp/releases" in str(error.value)
    assert "ggml-org" not in str(error.value)


def test_a_legacy_ggml_org_marker_maps_to_the_matching_fork_release(mod, monkeypatch, tmp_path):
    """An old install of an upstream prebuilt keeps working, its converter from the fork."""
    mod._write_prebuilt_marker(str(tmp_path), "b1234", "llama-b1234-bin-ubuntu-x64.tar.gz",
                               repo = "ggml-org/llama.cpp")
    _serve_release_list(mod, monkeypatch, [[_release("b1234-mix-abc1234")]])
    monkeypatch.setattr(mod, "_resolve_llama_cpp_release",
                        lambda *a, **k: pytest.fail("the matching release answers this"))
    assert mod._resolve_converter_revision(str(tmp_path)) == (FORK, "b1234-mix-abc1234")


def test_a_legacy_marker_without_a_match_uses_the_latest_fork_release(mod, monkeypatch, tmp_path):
    mod._write_prebuilt_marker(str(tmp_path), "b1234", "x.tar.gz", repo = "ggml-org/llama.cpp")
    _serve_release_list(mod, monkeypatch, [[_release("b9999-mix-fffffff")]])
    apis = []
    monkeypatch.setattr(mod, "_resolve_llama_cpp_release",
                        lambda releases_api = None: apis.append(releases_api) or ("b9999-mix-fffffff", {}))
    assert mod._resolve_converter_revision(str(tmp_path)) == (FORK, "b9999-mix-fffffff")
    assert apis == [mod.LLAMA_CPP_PUBLISHED_RELEASES_API]


def test_the_latest_release_row_asks_the_fork(mod, monkeypatch, tmp_path):
    apis = []
    monkeypatch.setattr(mod, "_resolve_llama_cpp_release",
                        lambda releases_api = None: apis.append(releases_api) or ("b11443-mix-d65395f", {}))
    assert mod._resolve_converter_revision(str(tmp_path)) == (FORK, "b11443-mix-d65395f")
    assert apis == [mod.LLAMA_CPP_PUBLISHED_RELEASES_API]
    monkeypatch.setenv("UNSLOTH_OFFLINE", "1")
    assert mod._resolve_converter_revision(str(tmp_path)) == (FORK, "b11443-mix-d65395f")


def test_offline_a_pinned_upstream_tag_finds_its_staged_fork_release(mod, monkeypatch):
    monkeypatch.setattr(mod, "_requests_get_with_retries",
                        lambda *a, **k: pytest.fail("offline must not list releases"))
    stage = Path(mod._converter_stage_dir(FORK, "b1234-mix-abc1234"))
    stage.mkdir(parents = True)
    (stage / mod.UNSLOTH_CONVERTER_STAGE_FILENAME).write_text(json.dumps({
        "schema": mod.UNSLOTH_CONVERTER_STAGE_SCHEMA, "repo": FORK,
        "tag": "b1234-mix-abc1234", "completed": True,
    }))
    monkeypatch.setenv("UNSLOTH_OFFLINE", "1")
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_CONVERTER_TAG", "b1234")
    assert mod._resolve_converter_revision("/nonexistent") == (FORK, "b1234-mix-abc1234")
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_CONVERTER_TAG", "b4321")
    assert mod._resolve_converter_revision("/nonexistent") == (None, None)


# --- converter sources ------------------------------------------------------------

_SHIM = b"""\
from conversion import ModelBase


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
"""


def _source_tarball(path, tag, lora = True):
    root = Path(path).parent / "_src" / f"llama.cpp-{tag}"
    (root / "gguf-py" / "gguf").mkdir(parents = True)
    (root / "gguf-py" / "gguf" / "__init__.py").write_text("# gguf\n")
    (root / "conversion").mkdir()
    (root / "conversion" / "__init__.py").write_text("")
    (root / "conversion" / "base.py").write_text("")
    (root / "convert_hf_to_gguf.py").write_bytes(_SHIM)
    if lora:
        (root / "convert_lora_to_gguf.py").write_text("# lora converter\n")
    with tarfile.open(path, "w:gz") as archive:
        archive.add(root, arcname = root.name)
    return Path(path)


def _serve_source(mod, monkeypatch, tmp_path, tag, *, lora = True, digest = "match"):
    """Serve the fork release's source asset and its sha256 list; record downloads."""
    name = f"llama.cpp-source-{tag}.tar.gz"
    (tmp_path / "blob").mkdir()
    blob = _source_tarball(tmp_path / "blob" / "source.tar.gz", tag, lora = lora)
    sha = hashlib.sha256(blob.read_bytes()).hexdigest()
    published = {"match": sha, "wrong": "00" * 32, "none": None}[digest]
    url = f"https://github.com/{FORK}/releases/download/{tag}/{name}"
    assets = {name: url, "llama-prebuilt-sha256.json": url + ".sha"}
    downloads = []

    def _get(request_url, **kwargs):
        downloads.append(request_url)
        assert request_url == f"{mod.LLAMA_CPP_PUBLISHED_RELEASES_API}/tags/{tag}", request_url
        return _Response({"tag_name": tag, "draft": False, "assets": [
            {"name": k, "browser_download_url": v} for k, v in assets.items()]})
    monkeypatch.setattr(mod, "_requests_get_with_retries", _get)
    artifacts = {} if published is None else {name: {"sha256": published, "kind": "upstream-source"}}
    monkeypatch.setattr(mod, "_fetch_release_json_asset",
                        lambda a, n: {"source_repo": FORK, "artifacts": artifacts})

    def _download(download_url, dest):
        downloads.append(download_url)
        Path(dest).write_bytes(blob.read_bytes())
    monkeypatch.setattr(mod, "_download_archive", _download)
    return downloads


def test_staging_downloads_the_verified_fork_source_asset_with_the_lora_converter(
    mod, monkeypatch, tmp_path,
):
    tag = "b11443-mix-d65395f"
    downloads = _serve_source(mod, monkeypatch, tmp_path, tag)
    stage = mod._stage_converter_sources(tag)
    assert stage is not None
    assert (Path(stage) / "convert_lora_to_gguf.py").is_file()
    assert (Path(stage) / "convert_hf_to_gguf.py").is_file()
    manifest = json.loads((Path(stage) / mod.UNSLOTH_CONVERTER_STAGE_FILENAME).read_text())
    assert manifest["repo"] == FORK
    assert all("unslothai/llama.cpp" in url for url in downloads), downloads
    assert not any("ggml-org" in url or "codeload" in url for url in downloads)


def test_a_cached_stage_without_the_lora_converter_is_restaged(mod, monkeypatch, tmp_path):
    tag = "b11443-mix-d65395f"
    downloads = _serve_source(mod, monkeypatch, tmp_path, tag)
    stage = Path(mod._stage_converter_sources(tag))
    (stage / "convert_lora_to_gguf.py").unlink()
    assert mod._converter_stage_is_usable(str(stage), repo = FORK, tag = tag) is False
    n = len(downloads)
    assert mod._stage_converter_sources(tag) == str(stage)
    assert len(downloads) > n
    assert (stage / "convert_lora_to_gguf.py").is_file()


@pytest.mark.parametrize("digest", ["wrong", "none"])
def test_an_unverifiable_source_asset_is_never_staged(mod, monkeypatch, tmp_path, digest):
    tag = "b11443-mix-d65395f"
    _serve_source(mod, monkeypatch, tmp_path, tag, digest = digest)
    assert mod._stage_converter_sources(tag) is None
    assert not os.path.exists(mod._converter_stage_dir(FORK, tag))


def test_hydration_copies_the_lora_converter_into_a_prebuilt_install(mod, monkeypatch, tmp_path):
    tag = "b11443-mix-d65395f"
    _serve_source(mod, monkeypatch, tmp_path, tag)
    install = tmp_path / "install"
    mod._hydrate_converter_sources(tag, str(install))
    assert (install / "convert_lora_to_gguf.py").is_file()
    assert (install / "convert_hf_to_gguf.py").is_file()


def test_single_file_converter_fallback_comes_from_the_fork():
    module = _load()
    assert module.LLAMA_CPP_CONVERT_FILE.startswith("https://github.com/unslothai/llama.cpp/")


def test_hydration_refuses_a_source_archive_without_the_lora_converter(mod, monkeypatch, tmp_path):
    tag = "b11443-mix-d65395f"
    _serve_source(mod, monkeypatch, tmp_path, tag, lora = False)
    with pytest.raises(Exception):
        mod._hydrate_converter_sources(tag, str(tmp_path / "install"))
    assert not (tmp_path / "install" / "convert_hf_to_gguf.py").exists()


def test_upstream_tag_mapping_scans_past_three_release_pages(mod, monkeypatch):
    pages = [[_release(f"b{20000 - p * 100 - i}-mix-x") for i in range(100)] for p in range(4)]
    pages.append([_release("b9000-mix-old")])
    calls = _serve_release_list(mod, monkeypatch, pages)
    mod._FORK_RELEASE_TAGS.clear()
    assert mod._fork_release_tag_for("b9000") == "b9000-mix-old"
    assert len(calls) == 5
