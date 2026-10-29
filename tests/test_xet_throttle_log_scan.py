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

"""The import-time hf_xet 429 log scan, executed alone via ast (no torch import)."""
import ast
import os
import pathlib
import tracemalloc

import pytest

_INIT = pathlib.Path(__file__).resolve().parents[1] / "unsloth_zoo" / "__init__.py"
_MARKER = b"429 Too Many Requests"
MIB = 1 << 20


def _load_scanner():
    tree = ast.parse(_INIT.read_text(encoding = "utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "has_429_exact_full_read")
    namespace = {"Path": pathlib.Path}
    exec(compile(ast.fix_missing_locations(ast.Module(body = [node], type_ignores = [])), str(_INIT), "exec"), namespace)
    return namespace["has_429_exact_full_read"]


scan = _load_scanner()


def _reference(log_dir):
    for log_file in pathlib.Path(log_dir).glob("*.log"):
        try:
            if _MARKER in log_file.read_bytes():
                return True
        except OSError:
            continue
    return False


def _cases():
    yield "empty", {}
    yield "empty_file", {"a.log": b""}
    yield "marker_only", {"a.log": _MARKER}
    yield "partial_marker_at_eof", {"a.log": b"x" * 100 + _MARKER[:-1]}
    yield "non_log_suffix", {"a.txt": _MARKER, "b.log": b"nothing"}
    yield "non_utf8", {"a.log": bytes(range(256)) * 9000 + _MARKER + b"\xff\xfe\x80"}
    yield "second_file", {"a.log": b"ok\n" * 1000, "b.log": b"x" * 10 + _MARKER}
    yield "near_miss", {"a.log": b"429 Too Many Request " + b"429 Too  Many Requests"}
    # Marker straddling every split point of the 1 MiB block boundary.
    for shift in range(-2, len(_MARKER) + 2):
        yield f"boundary_{shift}", {"a.log": b"y" * (MIB - shift) + _MARKER + b"z" * 7}
    yield "third_block", {"a.log": b"q" * (2 * MIB + 12345) + _MARKER}
    yield "prefix_then_garbage", {"a.log": b"y" * (MIB - 5) + _MARKER[:5] + b"#" * MIB}


@pytest.mark.parametrize("name,files", list(_cases()), ids = [c[0] for c in _cases()])
def test_scan_matches_whole_file_read(tmp_path, name, files):
    for file_name, data in files.items():
        (tmp_path / file_name).write_bytes(data)
    assert scan(tmp_path) == _reference(tmp_path)


def test_missing_dir_and_directory_named_log(tmp_path):
    assert scan(tmp_path / "missing") is False
    (tmp_path / "dir.log").mkdir()
    (tmp_path / "z.log").write_bytes(_MARKER)
    assert scan(tmp_path) is True


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason = "root reads mode 000 files")
def test_unreadable_log_is_skipped(tmp_path):
    bad = tmp_path / "a.log"
    bad.write_bytes(_MARKER)
    bad.chmod(0)
    try:
        assert scan(tmp_path) is False
        (tmp_path / "b.log").write_bytes(_MARKER)
        assert scan(tmp_path) is True
    finally:
        bad.chmod(0o600)


@pytest.mark.parametrize("with_marker", [False, True])
def test_large_log_is_not_loaded_whole(tmp_path, with_marker):
    with open(tmp_path / "big.log", "wb") as f:
        for _ in range(24):
            f.write(b"2026-10-28T00:00:00Z INFO xet chunk fetched\n" * (MIB // 45))
        if with_marker:
            f.write(_MARKER)
    tracemalloc.start()
    try:
        result = scan(tmp_path)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert result is with_marker
    assert peak < 6 * MIB, peak
