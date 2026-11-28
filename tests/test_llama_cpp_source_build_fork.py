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

"""The llama.cpp source build (no prebuilt installed) and the chat.cpp check use unslothai/llama.cpp only: the source
archive of the release the prebuilt resolution chose, else a clone of the fork's default branch. Every network call
and subprocess is stubbed."""

import hashlib
import io
import os
import tarfile
import types

import pytest

from unsloth_zoo import llama_cpp

FORK_GIT = "https://github.com/unslothai/llama.cpp"
TAG = "b9999-mix-abc1234"


def _source_tarball(path, tag = TAG, tree = ("src", "ggml", "common")):
    with tarfile.open(path, "w:gz") as tar:
        for d in tree:
            data = b"// stub\n"
            info = tarfile.TarInfo(f"llama.cpp-{tag}/{d}/stub.c")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


@pytest.fixture
def net(monkeypatch, tmp_path):
    """A fork release TAG with a source archive and its published sha256; records every URL asked for."""
    for name in ("UNSLOTH_LLAMA_TAG", "UNSLOTH_LLAMA_FORCE_COMPILE", "UNSLOTH_AUTO_INSTALL"):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(llama_cpp, "_PREBUILT_RESOLVED_RELEASE", {}, raising = False)
    archive = str(tmp_path / "served.tar.gz")
    state = types.SimpleNamespace(urls = [], sha256 = _source_tarball(archive), resolves = [], tag = TAG)
    source_name = f"llama.cpp-source-{TAG}.tar.gz"
    assets = {source_name: f"{FORK_GIT}/releases/download/{TAG}/{source_name}",
              "llama-prebuilt-sha256.json": f"{FORK_GIT}/releases/download/{TAG}/llama-prebuilt-sha256.json"}

    def resolve(releases_api = None):
        state.resolves.append((releases_api, os.environ.get("UNSLOTH_LLAMA_TAG", "")))
        return (state.tag, dict(assets)) if state.tag else None

    def get(url, **kwargs):
        state.urls.append(url)
        if url.endswith("llama-prebuilt-sha256.json"):
            return _Response({"artifacts": {source_name: {"sha256": state.sha256}}})
        if url.endswith(f"/tags/{TAG}"):
            return _Response({"tag_name": TAG, "assets": [{"name": k, "browser_download_url": v} for k, v in assets.items()]})
        raise AssertionError(url)

    def download(url, dest):
        state.urls.append(url)
        with open(archive, "rb") as src, open(dest, "wb") as out:
            out.write(src.read())

    monkeypatch.setattr(llama_cpp, "_resolve_llama_cpp_release", resolve)
    monkeypatch.setattr(llama_cpp, "_requests_get_with_retries", get)
    monkeypatch.setattr(llama_cpp, "_download_archive", download)
    return state


@pytest.fixture
def build(monkeypatch, tmp_path):
    """install_llama_cpp's system probes, pip and shell steps recorded; stops right after the source is in place."""
    calls = []
    monkeypatch.chdir(tmp_path)

    def rec(kind, ret = None):
        def f(*a, **k):
            calls.append((kind, a[0] if a else None))
            return ret
        return f

    monkeypatch.setattr(llama_cpp, "check_build_requirements", rec("check_build_requirements", ([], "debian")))
    monkeypatch.setattr(llama_cpp, "do_we_need_sudo", rec("do_we_need_sudo", False))
    monkeypatch.setattr(llama_cpp, "check_pip", rec("check_pip", "pip"))
    monkeypatch.setattr(llama_cpp, "try_execute", rec("try_execute"))
    monkeypatch.setattr(llama_cpp, "try_execute_with_auto_install", rec("try_execute_with_auto_install"))
    monkeypatch.setattr(llama_cpp, "_maybe_install_llama_cpp_prebuilt", rec("prebuilt"))
    return calls


def _clones(calls):
    return [c for k, c in calls if k == "try_execute_with_auto_install" and "git clone" in (c or "")]


def test_no_release_clones_the_fork_default_branch(net, build, tmp_path):
    net.tag = None
    folder = str(tmp_path / "llama.cpp")
    assert llama_cpp.install_llama_cpp(llama_cpp_folder = folder, just_clone_repo = True) == folder
    clones = _clones(build)
    assert len(clones) == 1 and clones[0].startswith(f"git clone {FORK_GIT} ")
    assert "ggml-org" not in clones[0] and "--branch" not in clones[0]


def test_the_release_source_archive_is_the_source_tree(net, build, tmp_path):
    folder = str(tmp_path / "llama.cpp")
    assert llama_cpp.install_llama_cpp(llama_cpp_folder = folder, just_clone_repo = True) == folder
    assert _clones(build) == []
    assert all(os.path.isfile(os.path.join(folder, d, "stub.c")) for d in ("src", "ggml", "common"))
    assert net.urls and all(u.startswith(FORK_GIT + "/releases/download/" + TAG + "/") for u in net.urls), net.urls
    assert not [n for n in os.listdir(tmp_path) if n.startswith(".llama_cpp_source_")]  # staging cleaned


def test_the_same_release_the_prebuilt_resolution_chose(net, monkeypatch, tmp_path):
    # the prebuilt path resolved TAG and then failed to install a bundle: the source build compiles TAG, no re-resolve
    with monkeypatch.context() as m:
        m.setattr(llama_cpp, "_fetch_release_json_asset", lambda assets, name: {})
        m.setattr(llama_cpp, "_detect_gpu_target", lambda: None)
        m.setattr(llama_cpp, "_select_cpu_assets", lambda *a, **k: [])
        assert llama_cpp._install_llama_cpp_prebuilt(str(tmp_path / "llama.cpp")) is None
    assert len(net.resolves) == 1
    folder = str(tmp_path / "llama.cpp")
    assert llama_cpp._install_fork_release_source(folder) == TAG
    assert len(net.resolves) == 1


def test_a_pinned_tag_is_honoured(net, monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_LLAMA_TAG", "b9999")
    assert llama_cpp._install_fork_release_source(str(tmp_path / "llama.cpp")) == TAG
    assert net.resolves == [(llama_cpp.LLAMA_CPP_PUBLISHED_RELEASES_API, "b9999")]


@pytest.mark.parametrize("case", ["sha256_mismatch", "no_sha256", "not_a_source_tree"])
def test_an_unverifiable_archive_writes_nothing_and_clones_the_fork(net, build, tmp_path, case):
    if case == "sha256_mismatch":
        net.sha256 = "0" * 64
    elif case == "no_sha256":
        net.sha256 = None
    else:
        net.sha256 = _source_tarball(str(tmp_path / "served.tar.gz"), tree = ("docs",))
    folder = str(tmp_path / "llama.cpp")
    assert llama_cpp._install_fork_release_source(folder) is None
    assert not os.path.exists(folder)
    assert llama_cpp.install_llama_cpp(llama_cpp_folder = folder, just_clone_repo = True) == folder
    clones = _clones(build)
    assert len(clones) == 1 and clones[0].startswith(f"git clone {FORK_GIT} ")


def test_a_non_empty_folder_is_never_touched(net, tmp_path):
    folder = tmp_path / "llama.cpp"
    folder.mkdir()
    (folder / "mine.txt").write_text("x")
    assert llama_cpp._install_fork_release_source(str(folder)) is None
    assert os.listdir(folder) == ["mine.txt"] and net.urls == []


@pytest.mark.parametrize("marker", [None, ("unslothai/llama.cpp", TAG), ("ggml-org/llama.cpp", "b9000")])
def test_chat_cpp_is_read_from_the_fork(monkeypatch, tmp_path, marker):
    folder = tmp_path / "llama.cpp"
    folder.mkdir()
    if marker is not None:
        llama_cpp._write_prebuilt_marker(str(folder), marker[1], "asset.tar.gz", repo = marker[0])
    monkeypatch.setattr(llama_cpp, "LLAMA_CPP_DEFAULT_DIR", str(folder))
    asked = []

    class Reply:
        content = b"// no appended message\n"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(llama_cpp.requests, "get", lambda url, **kw: asked.append(url) or Reply())
    llama_cpp._check_llama_cpp_appended_system_message.cache_clear()
    try:
        assert llama_cpp._check_llama_cpp_appended_system_message() == ""
    finally:
        llama_cpp._check_llama_cpp_appended_system_message.cache_clear()
    ref = TAG if marker and marker[0] == "unslothai/llama.cpp" else "refs/heads/master"
    assert asked == [f"https://raw.githubusercontent.com/unslothai/llama.cpp/{ref}/common/chat.cpp"]
