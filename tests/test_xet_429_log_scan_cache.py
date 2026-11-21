# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""B12: the hf_xet 429 log scan reuses its answer for unchanged files and rescans changed / new ones.
Function loaded by AST from unsloth_zoo/__init__.py ; CPU only."""
import ast, builtins, os, pathlib, time
import pytest

_INIT = pathlib.Path(__file__).resolve().parents[1] / "unsloth_zoo" / "__init__.py"
MARK = b"429 Too Many Requests"


def _fn():
    tree = ast.parse(_INIT.read_text(encoding = "utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "has_429_exact_full_read")
    ns = {"Path": pathlib.Path, "os": os}
    exec(compile(ast.Module(body = [node], type_ignores = []), str(_INIT), "exec"), ns)
    return ns["has_429_exact_full_read"]


@pytest.fixture
def opened(monkeypatch):
    """Counts opens of *.log files (the expensive read)."""
    count = {"n": 0}
    real = builtins.open

    def spy(file, *a, **k):
        if str(file).endswith(".log"):
            count["n"] += 1
        return real(file, *a, **k)

    monkeypatch.setattr(builtins, "open", spy)
    return count


def test_unchanged_logs_are_not_reread(tmp_path, opened):
    f = _fn()
    for i in range(3):
        (tmp_path / f"x{i}.log").write_bytes(b"ok\n" * 1000)
    assert f(tmp_path) is False
    first = opened["n"]
    assert first == 3
    assert f(tmp_path) is False
    assert opened["n"] == first, "second scan re-read unchanged logs"


def test_appended_marker_is_seen(tmp_path, opened):
    f = _fn()
    p = tmp_path / "x.log"
    p.write_bytes(b"ok\n" * 1000)
    assert f(tmp_path) is False
    with open(p, "ab") as h:
        h.write(b"error: " + MARK + b"\n")
    assert f(tmp_path) is True
    assert f(tmp_path) is True


def test_new_file_and_corrupt_manifest(tmp_path, opened):
    f = _fn()
    (tmp_path / "a.log").write_bytes(b"ok\n")
    assert f(tmp_path) is False
    (tmp_path / "b.log").write_bytes(MARK)
    assert f(tmp_path) is True
    for m in tmp_path.glob(".unsloth_429_scan.json"):
        m.write_text("{not json")
    assert f(tmp_path) is True
    (tmp_path / "b.log").unlink()
    assert f(tmp_path) is False


def test_same_size_rewrite_is_rescanned(tmp_path, opened):
    f = _fn()
    p = tmp_path / "x.log"
    p.write_bytes(b"A" * len(MARK))
    assert f(tmp_path) is False
    time.sleep(0.01)
    p.write_bytes(MARK)
    assert f(tmp_path) is True


def test_readonly_dir_falls_back_to_full_scan(tmp_path, opened, monkeypatch):
    f = _fn()
    (tmp_path / "x.log").write_bytes(b"ok\n")
    real_replace = os.replace
    monkeypatch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(PermissionError("ro")))
    assert f(tmp_path) is False
    assert f(tmp_path) is False
    monkeypatch.setattr(os, "replace", real_replace)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.log"], "temp manifest left behind"
