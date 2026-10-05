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

# CVE-2026-66007 (GHSA-379c-qx7v-6h59): datasets < 5.0.1 joins a folder dataset's
# metadata `file_name` to the dataset directory unchecked, so "../../x" or an absolute
# path makes imagefolder/audiofolder read any local file, and save_to_disk/push_to_hub
# then embed its bytes. Same refusal as the upstream fix (huggingface/datasets f989ef9),
# applied to the metadata tables before any path is built.

import functools
import inspect
import os

from .common import TEMPORARY_PATCHES

__all__ = ["patch_datasets_folder_metadata_file_name"]

_GUARD_FLAG = "_unsloth_file_name_guard"
# Superset of every value the exact check below can refuse; only these reach Python.
_CANDIDATE_REGEX = r"://|\.\.|^[/\\]|^[A-Za-z]:"


def _file_name_escapes(value):
    if "://" in value:
        return True
    relpath = os.path.normpath(value).replace("\\", "/")
    # Windows: Python 3.13 isabs() no longer counts "/x", and "C:x" is drive-relative,
    # yet os.path.join(dir, either) leaves dir, so check the root and drive directly.
    return (
        os.path.isabs(value)
        or os.path.isabs(relpath)
        or relpath.startswith("/")
        or bool(os.path.splitdrive(value)[0])
        or relpath == ".."
        or relpath.startswith("../")
    )


def _file_name_columns(table):
    import pyarrow as pa

    def is_text(t):
        return pa.types.is_string(t) or pa.types.is_large_string(t)

    for field in table.schema:
        name = field.name
        if (name == "file_name" or name.endswith("_file_name")) and is_text(field.type):
            yield name, table.column(name).chunks
        elif (name == "file_names" or name.endswith("_file_names")) and (
            (pa.types.is_list(field.type) or pa.types.is_large_list(field.type))
            and is_text(field.type.value_type)
        ):
            yield name, [chunk.flatten() for chunk in table.column(name).chunks]


def _check_metadata_table(table):
    import pyarrow.compute as pc

    for name, chunks in _file_name_columns(table):
        for chunk in chunks:
            try:
                values = chunk.filter(pc.match_substring_regex(chunk, _CANDIDATE_REGEX)).to_pylist()
            except Exception:
                values = chunk.to_pylist()
            for value in values:
                if value is not None and _file_name_escapes(value):
                    raise ValueError(
                        f"Invalid metadata {name} '{value}': `{name}` must be a relative path "
                        f"pointing inside the directory containing the metadata file. Absolute paths, "
                        f"URL schemes and parent-directory ('..') traversal are not allowed."
                    )


def _datasets_has_upstream_fix(module):
    try:
        return "Invalid metadata file_name" in inspect.getsource(module)
    except Exception:
        pass
    try:
        import datasets
        from packaging.version import Version

        return Version(datasets.__version__) >= Version("5.0.1")
    except Exception:
        return False


def patch_datasets_folder_metadata_file_name():
    try:
        from datasets.packaged_modules.folder_based_builder import folder_based_builder
    except Exception:
        return
    builder = getattr(folder_based_builder, "FolderBasedBuilder", None)
    read_metadata = getattr(builder, "_read_metadata", None)
    if read_metadata is None or getattr(read_metadata, _GUARD_FLAG, False):
        return
    if _datasets_has_upstream_fix(folder_based_builder):
        return

    @functools.wraps(read_metadata)
    def _read_metadata(self, *args, **kwargs):
        for table in read_metadata(self, *args, **kwargs):
            _check_metadata_table(table)
            yield table

    setattr(_read_metadata, _GUARD_FLAG, True)
    builder._read_metadata = _read_metadata


TEMPORARY_PATCHES.append(patch_datasets_folder_metadata_file_name)
