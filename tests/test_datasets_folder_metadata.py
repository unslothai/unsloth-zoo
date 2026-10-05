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

"""CVE-2026-66007: folder dataset metadata `file_name` must stay inside the dataset dir."""

import json
import os
import struct
import zlib

import pytest

datasets = pytest.importorskip("datasets")

from unsloth_zoo.temporary_patches.datasets_folder_metadata import (  # noqa: E402
    _file_name_escapes,
    patch_datasets_folder_metadata_file_name,
)

SECRET = b"UNSLOTH-SECRET-TOKEN"


def _png():
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
        + chunk(b"IEND", b"")
    )


def _folder(tmp_path, rows):
    (tmp_path / "secret.txt").write_bytes(SECRET)
    train = tmp_path / "ds" / "train"
    (train / "sub").mkdir(parents=True)
    (train / "ok.png").write_bytes(_png())
    (train / "sub" / "nested.png").write_bytes(_png())
    (train / "metadata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return tmp_path / "ds"


def _load(tmp_path, rows):
    patch_datasets_folder_metadata_file_name()
    data_dir = _folder(tmp_path, rows)
    return datasets.load_dataset("imagefolder", data_dir=str(data_dir), cache_dir=str(tmp_path / "cache"))


@pytest.mark.parametrize(
    "rows",
    [
        [{"file_name": "ok.png"}, {"file_name": "../../secret.txt"}],
        [{"file_name": "ok.png"}, {"file_name": "ABSOLUTE"}],
        [{"file_name": "ok.png"}, {"file_name": "file://../../secret.txt"}],
        [{"file_names": ["ok.png"]}, {"file_names": ["ok.png", "../../secret.txt"]}],
        [{"file_name": "ok.png", "image_file_name": "ok.png"}, {"file_name": "ok.png", "image_file_name": "../../secret.txt"}],
    ],
    ids=["relative", "absolute", "scheme", "file_names_list", "prefixed_key"],
)
def test_escaping_file_name_is_refused(tmp_path, rows):
    rows = json.loads(json.dumps(rows).replace("ABSOLUTE", str((tmp_path / "secret.txt").resolve())))
    with pytest.raises(ValueError, match="Invalid metadata"):
        ds = _load(tmp_path, rows)
        ds.save_to_disk(str(tmp_path / "saved"))
    saved = b"".join(p.read_bytes() for p in (tmp_path / "saved").rglob("*.arrow")) if (tmp_path / "saved").exists() else b""
    assert SECRET not in saved


def test_files_inside_the_folder_still_load(tmp_path):
    rows = [{"file_name": "ok.png", "text": "a"}, {"file_name": "sub/nested.png", "text": "b"}, {"file_name": "./sub/../ok.png", "text": "c"}]
    ds = _load(tmp_path, rows)["train"]
    assert ds.num_rows == 3
    assert ds["text"] == ["a", "b", "c"]
    assert all(img.size == (1, 1) for img in ds["image"])


def test_patch_is_idempotent():
    from datasets.packaged_modules.folder_based_builder import folder_based_builder

    patch_datasets_folder_metadata_file_name()
    first = folder_based_builder.FolderBasedBuilder._read_metadata
    patch_datasets_folder_metadata_file_name()
    assert folder_based_builder.FolderBasedBuilder._read_metadata is first
    inner = getattr(first, "__wrapped__", None)
    assert inner is None or not getattr(inner, "_unsloth_file_name_guard", False)


@pytest.mark.parametrize(
    "value, escapes",
    [
        ("ok.png", False),
        ("sub/nested.png", False),
        ("./sub/../ok.png", False),
        ("a..b.png", False),
        ("..", True),
        ("../x.png", True),
        ("sub/../../x.png", True),
        ("..\\..\\x.png", True),
        ("/etc/passwd", True),
        ("file://x", True),
        ("hf://datasets/a/b", True),
    ],
)
def test_file_name_escapes(value, escapes):
    assert _file_name_escapes(value) is escapes


@pytest.mark.skipif(os.name != "nt", reason="drive and UNC prefixes only mean something on Windows")
@pytest.mark.parametrize("value", ["C:x.png", "C:\\x.png", "\\x.png", "\\\\server\\share\\x.png"])
def test_windows_roots_escape(value):
    assert _file_name_escapes(value) is True
