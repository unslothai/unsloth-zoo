"""The import-time hf_xet 429 log scan streams each log in bounded blocks.

The scanner runs inside `import unsloth_zoo` before the device probe, so it is pulled out of
`__init__.py` with `ast` and executed alone (no torch import needed).
"""
import ast
import os
import pathlib
import tracemalloc

import pytest

_INIT = pathlib.Path(__file__).resolve().parents[1] / "unsloth_zoo" / "__init__.py"
_MARKER = b"429 Too Many Requests"


def _load_scanner():
    tree = ast.parse(_INIT.read_text(encoding = "utf-8"))
    node = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "has_429_exact_full_read"
    )
    module = ast.Module(body = [node], type_ignores = [])
    namespace = {"Path": pathlib.Path}
    exec(compile(ast.fix_missing_locations(module), str(_INIT), "exec"), namespace)
    return namespace["has_429_exact_full_read"]


def _reference(log_dir):
    # The whole-file read the scanner replaced.
    log_dir = pathlib.Path(log_dir).expanduser()
    if not log_dir.is_dir():
        return False
    for log_file in log_dir.glob("*.log"):
        try:
            if _MARKER in log_file.read_bytes():
                return True
        except OSError:
            continue
    return False


scan = _load_scanner()
MIB = 1 << 20


def _cases():
    yield "empty", {}
    yield "empty_file", {"a.log": b""}
    yield "marker_only", {"a.log": _MARKER}
    yield "partial_marker_at_eof", {"a.log": b"x" * 100 + _MARKER[:-1]}
    yield "non_log_suffix", {"a.txt": _MARKER, "b.log": b"nothing"}
    yield "non_utf8", {"a.log": bytes(range(256)) * 9000 + _MARKER + b"\xff\xfe\x80"}
    yield "non_utf8_absent", {"a.log": bytes(range(256)) * 9000}
    yield "second_file", {"a.log": b"ok\n" * 1000, "b.log": b"x" * 10 + _MARKER}
    yield "near_miss", {"a.log": b"429 Too Many Request " + b"429 Too  Many Requests"}
    # Marker straddling every split point of the 1 MiB block boundary, plus just inside / after it.
    for shift in range(-2, len(_MARKER) + 2):
        yield f"boundary_{shift}", {"a.log": b"y" * (MIB - shift) + _MARKER + b"z" * 7}
    # Marker in the third block, and a prefix in one block with the rest never arriving.
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
    assert scan(tmp_path) == _reference(tmp_path) is True


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason = "root reads mode 000 files")
def test_unreadable_log_is_skipped(tmp_path):
    bad = tmp_path / "a.log"
    bad.write_bytes(_MARKER)
    bad.chmod(0)
    try:
        assert scan(tmp_path) == _reference(tmp_path) is False
        (tmp_path / "b.log").write_bytes(_MARKER)
        assert scan(tmp_path) == _reference(tmp_path) is True
    finally:
        bad.chmod(0o600)


@pytest.mark.parametrize("with_marker", [False, True])
def test_large_log_is_not_loaded_whole(tmp_path, with_marker):
    size = 48 * MIB
    with open(tmp_path / "big.log", "wb") as f:
        block = b"2026-10-28T00:00:00Z INFO xet chunk fetched\n" * (MIB // 45)
        written = 0
        while written < size:
            f.write(block)
            written += len(block)
        if with_marker:
            f.write(_MARKER)
    tracemalloc.start()
    try:
        result = scan(tmp_path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result is with_marker
    # Whole-file read allocates the full 48 MiB; the streamed scan holds about two blocks.
    assert peak < 8 * MIB, peak
