# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression tests for scripts/lint_dynamic_exec.py.

The lint is what stops PR #1083's bug class coming back: an `exec`/`eval`/`compile`
whose first argument is built by interpolation, which is the shape that turns a value
into syntax. Everything already in the tree is allowlisted with a written
justification, so the live tree must pass and any new call must not.

CPU-only and network-free: the lint is stdlib only and runs as a subprocess.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "lint_dynamic_exec.py"
ALLOWLIST = REPO_ROOT / "scripts" / "dynamic_exec_allowlist.json"


def _run(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output = True, text = True, cwd = REPO_ROOT,
    )


def test_the_script_exists():
    assert SCRIPT.is_file()
    assert ALLOWLIST.is_file()


def test_self_test_passes():
    proc = _run("--self-test")
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"


def test_live_tree_passes():
    """Every interpolated dynamic-execution call in the tree is reviewed."""
    proc = _run()
    assert proc.returncode == 0, (
        f"the live tree fails the lint:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


# --- the lint actually catches things ----------------------------------------

@pytest.mark.parametrize("body, description", [
    ('exec(f"import {x}")', "f-string"),
    ('eval("torch." + x)', "concatenation"),
    ('compile("y = %s" % x, "<x>", "exec")', "%-format"),
    ('exec("a.{}".format(x))', ".format()"),
])
def test_a_new_interpolated_call_fails(body, description, tmp_path):
    offender = tmp_path / "offender.py"
    offender.write_text(f"def f(x):\n    {body}\n")
    proc = _run("--paths", str(offender))
    assert proc.returncode == 1, f"{description} was not caught:\n{proc.stdout}"
    assert "offender.py" in proc.stderr


@pytest.mark.parametrize("body", [
    "exec(source, globals())",
    "eval(name)",
    'exec("literal source")',
    'module.exec(f"{x}")',
])
def test_non_interpolated_calls_are_not_flagged(body, tmp_path):
    """Bare exec of generated source is the normal case here and must stay quiet."""
    clean = tmp_path / "clean.py"
    clean.write_text(f"def f(source, name, x, module):\n    {body}\n")
    proc = _run("--paths", str(clean))
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"


# --- the allowlist has to stay honest ----------------------------------------

def test_every_allowlist_entry_has_a_justification():
    # An empty allowlist is a legitimate state: it means no interpolated dynamic
    # execution is left in this repo at all.
    entries = json.loads(ALLOWLIST.read_text())["allowed"]
    unjustified = [
        e for e in entries
        if e.get("reason", "").strip().upper() in ("", "REVIEW ME")
    ]
    assert not unjustified, unjustified


def test_allowlist_entries_are_keyed_on_content_not_lines():
    """Line numbers drift; a justification keyed on one would silently detach."""
    entries = json.loads(ALLOWLIST.read_text())["allowed"]
    assert all("hash" in e for e in entries)
    assert all("line" not in e for e in entries)


def test_editing_an_allowlisted_call_revokes_its_justification(tmp_path):
    """The property that makes the allowlist safe: change the call, lose the pass."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import lint_dynamic_exec as lint
    finally:
        sys.path.pop(0)

    original = tmp_path / "a.py"
    original.write_text('def f(x):\n    exec(f"import {x}")\n')
    edited = tmp_path / "b.py"
    edited.write_text('def f(x):\n    exec(f"import os; {x}")\n')
    reflowed = tmp_path / "c.py"
    reflowed.write_text('def f(x):\n    exec(\n        f"import {x}"\n    )\n')

    hashes = {p.name: lint.scan_file(p)[0]["hash"] for p in (original, edited, reflowed)}
    assert hashes["a.py"] != hashes["b.py"], "changing the payload kept the hash"
    assert hashes["a.py"] == hashes["c.py"], "reformatting invalidated the hash"


def _isolated_lint(tmp_path, allowlist):
    """A copy of the lint whose allowlist is `allowlist` (it reads the one beside it)."""
    script = tmp_path / "lint_dynamic_exec.py"
    script.write_text(SCRIPT.read_text())
    (tmp_path / "dynamic_exec_allowlist.json").write_text(json.dumps(allowlist))
    return script


def test_a_stale_allowlist_entry_fails(tmp_path):
    """An entry matching nothing means the tree moved on; --update is required."""
    script = _isolated_lint(tmp_path, {"allowed": [{
        "path": "gone.py", "qualname": "f", "sink": "exec",
        "kind": "f-string", "hash": "0" * 16, "reason": "was reviewed once",
    }]})
    # A full scan, not --paths: staleness is only meaningful over the whole tree,
    # and this copy's tree contains none of the package directories.
    proc = subprocess.run(
        [sys.executable, str(script)], capture_output = True, text = True,
    )
    assert proc.returncode == 1
    assert "no longer matches any call" in proc.stderr


def test_paths_mode_does_not_report_the_rest_of_the_tree_as_stale(tmp_path):
    """Scanning one file must not invalidate every justification outside it."""
    clean = tmp_path / "clean.py"
    clean.write_text("def f(source):\n    exec(source)\n")
    proc = _run("--paths", str(clean))
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"


def test_an_unjustified_allowlist_entry_fails(tmp_path):
    """`--update` seeds entries with REVIEW ME; leaving one is a failure, not a pass."""
    offender = tmp_path / "offender.py"
    offender.write_text('def f(x):\n    exec(f"import {x}")\n')

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import lint_dynamic_exec as lint
    finally:
        sys.path.pop(0)
    finding = lint.scan_file(offender)[0]

    script = _isolated_lint(tmp_path, {"allowed": [{
        "path": finding["path"], "qualname": finding["qualname"],
        "sink": finding["sink"], "kind": finding["reason"],
        "hash": finding["hash"], "reason": "REVIEW ME",
    }]})
    proc = subprocess.run(
        [sys.executable, str(script), "--paths", str(offender)],
        capture_output = True, text = True,
    )
    assert proc.returncode == 1
    assert "no justification" in proc.stderr
