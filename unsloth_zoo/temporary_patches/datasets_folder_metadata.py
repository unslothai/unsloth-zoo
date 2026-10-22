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
# applied to the metadata tables before any path is built. Installed on 5.0.1+ too: the
# upstream check runs before "\" becomes "/", so "sub\..\..\x" still escapes there.

import functools
import os
import posixpath

from .common import TEMPORARY_PATCHES

__all__ = ["patch_datasets_folder_metadata_file_name"]

_GUARD_FLAG = "_unsloth_file_name_guard"
# Superset of every value the exact check below can refuse; only these reach Python.
_CANDIDATE_REGEX = r"://|\.\.|^[/\\]|^[A-Za-z]:"


def _file_name_escapes(value):
    if "://" in value:
        return True
    # Backslashes first: datasets turns them into "/" before joining, so a POSIX normpath
    # of the raw value would miss "sub\..\..\x" written with backslash separators.
    relpath = posixpath.normpath(value.replace("\\", "/"))
    # Windows: Python 3.13 isabs() no longer counts "/x", and "C:x" is drive-relative,
    # yet os.path.join(dir, either) leaves dir, so check the root and drive directly.
    return (
        os.path.isabs(value)
        or relpath.startswith("/")
        or bool(os.path.splitdrive(value)[0])
        or relpath == ".."
        or relpath.startswith("../")
    )


def _is_path_key(name, list_depth):
    # Only the shapes datasets resolves: a singular key holding a string, a plural key
    # holding a list of strings. A list under "*_file_name" stays ordinary metadata.
    if list_depth == 0:
        return name == "file_name" or name.endswith("_file_name")
    if list_depth == 1:
        return name == "file_names" or name.endswith("_file_names")
    return False


def _file_name_arrays(name, array, list_depth = 0):
    # Folder builders resolve these keys at any depth, and Parquet metadata can wrap them in
    # dictionary encoding or any list flavour, so unwrap by type rather than by layout.
    import pyarrow as pa

    def is_any(kind, *checks):
        return any(getattr(pa.types, check, lambda _: False)(kind) for check in checks)

    kind = array.type
    if is_any(kind, "is_dictionary"):
        yield from _file_name_arrays(name, array.dictionary_decode(), list_depth)
    elif is_any(kind, "is_struct"):
        for field, child in zip(kind, array.flatten()):
            yield from _file_name_arrays(field.name, child)
    elif is_any(kind, "is_list", "is_large_list", "is_fixed_size_list", "is_list_view", "is_large_list_view"):
        yield from _file_name_arrays(name, array.flatten(), list_depth + 1)
    elif _is_path_key(name, list_depth) and is_any(kind, "is_string", "is_large_string", "is_string_view"):
        yield name, array


def _file_name_columns(table):
    for field in table.schema:
        for chunk in table.column(field.name).chunks:
            yield from _file_name_arrays(field.name, chunk)


def _check_metadata_table(table):
    import pyarrow.compute as pc

    for name, chunk in _file_name_columns(table):
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


def patch_datasets_folder_metadata_file_name():
    try:
        from datasets.packaged_modules.folder_based_builder import folder_based_builder
    except Exception:
        return
    builder = getattr(folder_based_builder, "FolderBasedBuilder", None)
    read_metadata = getattr(builder, "_read_metadata", None)
    if read_metadata is None or getattr(read_metadata, _GUARD_FLAG, False):
        return

    @functools.wraps(read_metadata)
    def _read_metadata(self, *args, **kwargs):
        for table in read_metadata(self, *args, **kwargs):
            _check_metadata_table(table)
            yield table

    setattr(_read_metadata, _GUARD_FLAG, True)
    builder._read_metadata = _read_metadata


TEMPORARY_PATCHES.append(patch_datasets_folder_metadata_file_name)
