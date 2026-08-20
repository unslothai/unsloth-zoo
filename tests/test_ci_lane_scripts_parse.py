# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""
The concurrent CI lanes must be syntactically whole shell, and their pytest
invocations must still carry every file.

Fusing matrix cells onto one runner turns YAML into shell, and shell that is
assembled rather than written has a failure mode YAML validation cannot see. The
fused Core job shipped with

    "$venv/bin/python" -m pytest -v --tb=short -rs \\
                tests/test_upstream_pinned_symbols_transformers.py \\\\
                tests/test_torchvision_video_removed_message.py \\\\

-- double backslashes, from a generator whose f-string escaped one too many. The
YAML parsed, actionlint passed, and every lane reported `install=0 tests=missing`
because the continuation ended the command and the remaining files ran as their
own (nonexistent) commands. Twenty test files silently stopped being a test run.

`bash -n` catches the class in a second, which is the whole point: it is cheaper
than a CI round trip and it is what was missing.

The file-count assertion is the other half. A command that parses is not the same
as a command that still runs everything; a dropped continuation can leave a
perfectly valid pytest call that collects three files instead of twenty.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

# Steps that build shell by hand for concurrent lanes. Matched on content rather
# than a hardcoded list, so a new fused job is covered the day it lands.
_LANE_MARKERS = ("lane()", ".lanes/", "venv-")


def _lane_steps() -> list[tuple[str, str, str, str]]:
    """(workflow, job, step name, run body) for every step that drives lanes."""
    out = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        doc = yaml.safe_load(path.read_text(encoding = "utf-8"))
        for job_id, job in (doc.get("jobs") or {}).items():
            if not isinstance(job, dict):
                continue
            for step in job.get("steps") or []:
                if not isinstance(step, dict):
                    continue
                run = step.get("run")
                if isinstance(run, str) and any(m in run for m in _LANE_MARKERS):
                    out.append((path.name, job_id, step.get("name", "<unnamed>"), run))
    return out


def test_there_are_lane_steps_to_check() -> None:
    """Otherwise every test below passes by finding nothing."""
    steps = _lane_steps()
    assert len(steps) >= 2, f"only {len(steps)} lane steps found; the detector looks broken"


@pytest.mark.parametrize(
    "workflow,job,name,run",
    _lane_steps(),
    ids = lambda v: v if isinstance(v, str) and len(v) < 40 else "",
)
def test_every_lane_step_is_valid_shell(workflow: str, job: str, name: str, run: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix = ".sh", delete = False) as handle:
        handle.write(run)
        path = handle.name
    try:
        result = subprocess.run([ "bash", "-n", path ], capture_output = True, text = True)
    finally:
        Path(path).unlink(missing_ok = True)
    assert result.returncode == 0, (
        f"{workflow}: job '{job}' step '{name}' is not valid shell:\n{result.stderr}"
    )


@pytest.mark.parametrize(
    "workflow,job,name,run",
    _lane_steps(),
    ids = lambda v: v if isinstance(v, str) and len(v) < 40 else "",
)
def test_no_double_backslash_continuations(workflow: str, job: str, name: str, run: str) -> None:
    """
    `\\\\` at end of line is an escaped backslash, not a continuation. It ends the
    command and hands the next line to the shell as its own -- which is how twenty
    pytest files became twenty "command not found" lines that nothing checked.
    Valid shell, wrong program, and `bash -n` alone would not object.
    """
    bad = [
        i + 1
        for i, line in enumerate(run.splitlines())
        if re.search(r"[^\\]\\\\$", line)
    ]
    assert not bad, (
        f"{workflow}: job '{job}' step '{name}' has escaped backslashes where line "
        f"continuations were meant, at line(s) {bad}"
    )


def test_the_fused_core_job_still_runs_every_drift_file() -> None:
    """
    A command can parse and still have lost most of its arguments. This pins the
    count against the workflow the cells were fused from, so a continuation that
    silently drops files fails here rather than going green having run three.
    """
    doc = yaml.safe_load((WORKFLOWS / "consolidated-tests-ci.yml").read_text(encoding = "utf-8"))
    steps = doc["jobs"]["core-upstream-matrix"]["steps"]
    run = next(s["run"] for s in steps if "concurrently" in (s.get("name") or ""))

    body = run[run.index("-m pytest"):]
    body = body[: body.index("|| trc=$?")]
    # Join the continuations the way the shell does, then count what is left.
    joined = body.replace("\\\n", " ")
    files = re.findall(r"tests/\S+\.py", joined)

    assert len(files) == 20, (
        f"the fused Core lane invokes pytest on {len(files)} files; the three matrix "
        f"cells it replaced each ran 20. Found: {files}"
    )
    assert '>> "$log" 2>&1' in joined, (
        "the pytest output is no longer redirected into the lane log, so a failing "
        "lane would report a status with nothing to read"
    )
