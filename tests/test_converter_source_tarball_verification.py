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
"""The converter sources are a second download, and they were unverified.

`_stage_prebuilt_install` checks the sha256 of the BINARY asset.
`_hydrate_converter_sources` then downloads the source archive separately and
copies convert_hf_to_gguf.py, conversion/ and gguf-py/ out of it, and those are
the files the converter subprocess executes. The fork's own
llama-prebuilt-sha256.json carries an entry for that archive, so there is
something to check against on the path Unsloth actually uses.
"""

import hashlib
import importlib.util
import logging
import os
import pathlib
import sys
import tarfile

import pytest


def _load(module_name, relative_path):
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(module_name, root / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _source_archive(tmp_path, tag, payload = "X = 1\n"):
    """A source tarball shaped like the real one: root/, converter, gguf-py."""
    root = tmp_path / f"llama.cpp-{tag}"
    (root / "gguf-py" / "gguf").mkdir(parents = True)
    (root / "conversion").mkdir(parents = True)
    (root / "convert_hf_to_gguf.py").write_text(payload, encoding = "utf-8")
    (root / "conversion" / "__init__.py").write_text(payload, encoding = "utf-8")
    (root / "gguf-py" / "gguf" / "__init__.py").write_text(payload, encoding = "utf-8")
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(root, arcname = root.name)
    return archive


def _install(llama_cpp, tmp_path, archive, checksums, monkeypatch, tag = "b1-mix-abc"):
    install = tmp_path / "install"
    install.mkdir()

    def _fake_download(url, destination):
        destination_path = pathlib.Path(destination)
        destination_path.write_bytes(pathlib.Path(archive).read_bytes())

    monkeypatch.setattr(llama_cpp, "_download_archive", _fake_download)
    llama_cpp._hydrate_converter_sources(
        tag,
        str(install),
        source_assets = {f"llama.cpp-source-{tag}.tar.gz": "https://example.invalid/a"},
        checksums = checksums,
    )
    return install


def test_a_source_archive_that_fails_its_published_digest_is_refused(
    tmp_path, monkeypatch
):
    """These bytes become the converter. A mismatch has to stop the install, the
    same way the binary asset's mismatch does, rather than be copied into place.
    """
    llama_cpp = _load("llama_cpp_tarball_probe", "unsloth_zoo/llama_cpp.py")

    tag = "b1-mix-abc"
    archive = _source_archive(tmp_path, tag)
    checksums = {f"llama.cpp-source-{tag}.tar.gz": {"sha256": "00" * 32}}

    with pytest.raises(RuntimeError, match = "sha256 mismatch"):
        _install(llama_cpp, tmp_path, archive, checksums, monkeypatch, tag = tag)


def test_a_matching_source_archive_installs(tmp_path, monkeypatch):
    llama_cpp = _load("llama_cpp_tarball_ok_probe", "unsloth_zoo/llama_cpp.py")

    tag = "b1-mix-abc"
    archive = _source_archive(tmp_path, tag)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksums = {f"llama.cpp-source-{tag}.tar.gz": {"sha256": digest}}

    install = _install(llama_cpp, tmp_path, archive, checksums, monkeypatch, tag = tag)
    assert (install / "convert_hf_to_gguf.py").is_file()
    assert (install / "gguf-py" / "gguf" / "__init__.py").is_file()


def test_a_release_with_no_published_digest_says_so(tmp_path, monkeypatch, caplog):
    """There is nothing to check against, so it installs. Saying nothing at all
    was the problem: the comment beside the scan claimed the bundle's own sha256
    covered these files, and it never did.
    """
    llama_cpp = _load("llama_cpp_tarball_nodigest_probe", "unsloth_zoo/llama_cpp.py")

    tag = "b1-mix-abc"
    archive = _source_archive(tmp_path, tag)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        install = _install(llama_cpp, tmp_path, archive, {}, monkeypatch, tag = tag)
    assert (install / "convert_hf_to_gguf.py").is_file()
    assert any("cannot be verified" in record.message for record in caplog.records), (
        [r.message for r in caplog.records]
    )
