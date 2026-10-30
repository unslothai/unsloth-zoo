#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Fail CI when `exec`, `eval` or `compile` is handed something that is not a literal.

Two of the holes fixed in unsloth-zoo#1108 and unsloth#9777 were exactly this shape:
`exec` of an HTTPS response body, and `exec` of a string built from a downloaded
config. GitHub's CodeQL runs on these repositories with Python enabled and raises
other injection findings here, `py/path-injection` among them, but it has never once
raised `py/code-injection` - so this class is not covered by what is already running.

The rule is deliberately the blunt one: the first argument to one of those three
builtins must be a written-out string. Anything else - a name, a call, an f-string
with a placeholder - is reported. That is coarse, and it is meant to be. The realistic
failure is a contributor writing `exec(f"...")` without thinking about where the
pieces came from, not somebody engineering a way past a checker they could equally
well delete.

Existing call sites are recorded in a baseline beside this script, so the gate starts
green and only new ones fail. A baseline entry carries the call's own text rather than
its line number, so moving code does not churn it, and it carries a count, so a new
call cannot hide behind a removed one.

    python scripts/lint_exec_literals.py             # check, exit 1 on a new site
    python scripts/lint_exec_literals.py --update    # rewrite the baseline
    python scripts/lint_exec_literals.py --self-test # prove the rule still fires
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

# Bare names only: re.compile(pattern) and model.eval() are not these builtins, and matching on attributes gave many false positives.
SINKS = ("exec", "eval", "compile")

# Notebooks are scanned too: in the notebooks repository they are the maintained source, and a cell runs under the same interpreter as a module.
SUFFIXES = (".py", ".ipynb")

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = Path(__file__).resolve().parent / "exec_literals_baseline.json"
# Repo-relative: the printed commands run from the root, where the bare filename is not a file.
_BASELINE_REL = BASELINE_PATH.relative_to(REPO_ROOT).as_posix()

# Last line, so a caller names which of the three red outcomes fired without parsing the prose.
_VERDICT = "verdict: "

# Paths not ours to fail on; `tests` is excluded because a test legitimately keeps a real `exec` around as the thing it asserts about.
EXCLUDED_PARTS = frozenset({
    "tests", "node_modules", "build", "dist", ".venv", "venv", "site-packages",
    ".git", ".tox", ".mypy_cache", ".pytest_cache", "__pycache__", ".ipynb_checkpoints",
    ".eggs",
})


def _is_written_out(node: ast.AST) -> bool:
    """Whether this argument is a string the file spells out in full."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (str, bytes))
    if isinstance(node, ast.JoinedStr):
        # An f-string without placeholders is a plain string; one with a placeholder is the whole point of this check and is not written out.
        return not any(isinstance(part, ast.FormattedValue) for part in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        # `"import " + "torch"` is still written out; `"import " + name` is not.
        return _is_written_out(node.left) and _is_written_out(node.right)
    return False


def _key(relative: str, call: ast.Call) -> dict:
    """Entry for a call: its file, sink, and a digest of ast.unparse(call), so reformatting or moving the call keeps it; changing its arguments does not."""
    return {
        "file": relative,
        "sink": call.func.id,
        "digest": hashlib.sha256(ast.unparse(call).encode()).hexdigest()[:16],
    }


def _notebook_source(path: Path, relative: str) -> str:
    """Joins a notebook's code cells, since a cell is not a module, and blanks %magic and !command lines so reported line numbers stay accurate."""
    try:
        document = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"{relative}: could not be read ({error.__class__.__name__})")
    lines: list[str] = []
    for cell in document.get("cells") or []:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source") or []
        text = source if isinstance(source, str) else "".join(source)
        for line in text.splitlines():
            lines.append("" if line.lstrip().startswith(("%", "!")) else line)
        lines.append("")
    return "\n".join(lines)


def scan_file(path: Path, relative: str) -> list[dict]:
    if path.suffix == ".ipynb":
        source: str | bytes = _notebook_source(path, relative)
    else:
        source = path.read_bytes()
    try:
        tree = ast.parse(source, filename = str(path))
    except (SyntaxError, ValueError, MemoryError, RecursionError) as error:
        if path.suffix == ".ipynb":
            # A notebook that does not parse as one module is ordinary and not failed: a cell may be mid-edit or depend on an earlier `%%capture`.
            return []
        # An unparseable .py file was not checked, and reporting it clean is the bypass this gate exists to avoid.
        raise SystemExit(f"{relative}: could not be parsed ({error.__class__.__name__})")
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if not isinstance(node.func, ast.Name) or node.func.id not in SINKS:
            continue
        if _is_written_out(node.args[0]):
            continue
        entry = _key(relative, node)
        entry["line"] = node.lineno
        found.append(entry)
    return found


def collect(targets: list[str]) -> list[dict]:
    found = []
    for target in targets:
        root = REPO_ROOT / target
        if root.is_file() and root.suffix in SUFFIXES:
            paths = [root]
        elif root.is_dir():
            paths = sorted(q for s in SUFFIXES for q in root.rglob(f"*{s}"))
        else:
            # A scan target that does not exist would silently cover less than claimed, so it is an error rather than a passing run.
            raise SystemExit(f"{target}: scan target does not exist, so nothing under it was checked")
        for path in paths:
            if not path.is_file():
                continue
            try:
                relative = path.relative_to(REPO_ROOT).as_posix()
            except ValueError:
                # `--paths` may name a file outside the checkout (how the tests drive the gate), so it is reported by the name given.
                relative = path.as_posix()
            if EXCLUDED_PARTS & set(Path(relative).parts):
                continue
            found.extend(scan_file(path, relative))
    return found


def _counted(entries: list[dict]) -> dict:
    """Maps identity to occurrence count; matching both identity and count stops a new call taking the place of a removed one."""
    counts: dict = {}
    for entry in entries:
        counts[(entry["file"], entry["sink"], entry["digest"])] = (
            counts.get((entry["file"], entry["sink"], entry["digest"]), 0) + 1
        )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--update", action = "store_true", help = "rewrite the baseline")
    parser.add_argument("--self-test", action = "store_true", help = "check the rule still fires")
    parser.add_argument("--paths", nargs = "*", help = "scan these instead of the defaults")
    arguments = parser.parse_args()

    if arguments.self_test:
        return self_test()

    document = json.loads(BASELINE_PATH.read_text(encoding = "utf-8"))
    targets = arguments.paths or document["targets"]
    found = collect(targets)

    if arguments.update:
        # --update carries existing reasons over; rebuilding from scratch would reset every hand-written justification, silently unreviewing entries.
        reasons = {(e["file"], e["sink"], e["digest"]): e.get("reason", "") for e in document["entries"]}
        document["entries"] = sorted(
            (
                {
                    "file": f, "sink": s, "digest": d, "count": n,
                    "reason": reasons.get((f, s, d), "REVIEW ME"),
                }
                for (f, s, d), n in _counted(found).items()
            ),
            key = lambda e: (e["file"], e["sink"], e["digest"]),
        )
        BASELINE_PATH.write_text(json.dumps(document, indent = 2) + "\n", encoding = "utf-8")
        print(f"baseline: {len(document['entries'])} entries, {len(found)} call sites")
        return 0

    allowed = {(e["file"], e["sink"], e["digest"]): e["count"] for e in document["entries"]}
    observed = _counted(found)
    lines = {(e["file"], e["sink"], e["digest"]): e["line"] for e in found}

    new = [k for k, n in observed.items() if n > allowed.get(k, 0)]
    if new:
        print(f"{len(new)} dynamic-execution call site(s) not in the baseline:\n")
        for f, s, d in sorted(new):
            print(f"  {f}:{lines[(f, s, d)]}  {s}(...) is handed a value that is not written out")
        print(
            "\nEither pass a written-out string, or - if the value really is trusted - "
            "record it, in two steps:\n"
            "  1. python scripts/lint_exec_literals.py --update\n"
            f"  2. replace the new entr(y/ies)' \"REVIEW ME\" reason in\n"
            f"     {_BASELINE_REL} with why the value is trusted.\n"
            "Step 2 is not optional: `--update` writes \"REVIEW ME\", and this gate "
            "stays red on it (see the next check below)."
        )
        print(f"\n{_VERDICT}new-call-site")
        return 1

    unreviewed = sorted(
        (e["file"], e["sink"]) for e in document["entries"]
        if not e.get("reason") or e["reason"] == "REVIEW ME"
    )
    if unreviewed:
        print(f"{len(unreviewed)} baseline entr(y/ies) carry no justification:\n")
        for f, s in unreviewed:
            print(f"  {f}  {s}")
        print(
            f"\nSay why the value is trusted in the entry's `reason` field in "
            f"{_BASELINE_REL}. This is step 2 after `--update`."
        )
        print(f"\n{_VERDICT}unreviewed-entry")
        return 1

    stale = sorted(k for k in allowed if k not in observed)
    if stale:
        # A baseline entry outliving its call site would re-permit whatever lands on that digest next, so stale entries are errors.
        print(f"{len(stale)} baseline entr(y/ies) no longer match any call:\n")
        for f, s, d in stale:
            print(f"  {f}  {s}  {d}")
        print(
            "\nYou removed or rewrote a dynamic-execution call, which is a good thing and "
            "needs no review. Just run `python scripts/lint_exec_literals.py --update` to "
            "drop the entries; nothing here asks you to justify anything."
        )
        print(f"\n{_VERDICT}stale-entry")
        return 1

    print(f"ok: {len(found)} dynamic-execution call site(s), all recorded")
    print(f"\n{_VERDICT}ok")
    return 0


_BAD = '''
def f(name):
    exec(f"import {name}")
    eval("torch." + name)
    compile(source, "<x>", "exec")
    exec("import MODULE".replace("MODULE", name))
'''

_GOOD = '''
import re
def f(name, source):
    exec("import torch")
    exec(f"no placeholders here")
    eval("1 + 1")
    re.compile(name)          # not the builtin
    model.eval()              # not the builtin
    exec()                    # no arguments
'''


def self_test() -> int:
    """Checks the rule fires on bad shapes and stays quiet on good ones; kept inline so the gate can self-verify on a runner with only Python."""
    import tempfile

    failures = []
    with tempfile.TemporaryDirectory() as directory:
        for label, source, expected in (("bad", _BAD, 4), ("good", _GOOD, 0)):
            path = Path(directory) / f"{label}.py"
            path.write_text(source, encoding = "utf-8")
            count = len(scan_file(path, f"{label}.py"))
            if count != expected:
                failures.append(f"{label}: expected {expected} finding(s), got {count}")
    if failures:
        print("self-test FAILED:\n  " + "\n  ".join(failures))
        return 1
    print("self-test: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
