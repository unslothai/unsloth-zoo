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


# The twenty drift files the fused Core lane inherited from the three matrix cells it
# replaced. Named rather than counted: the risk this guard exists for is a continuation
# that silently DROPS files, and a name tells you which one went missing where a count
# only tells you that one did. Adding a drift file is routine and must not fail here --
# an exact count made #1114 (which legitimately added the mxfp4 load-path file) turn this
# guard red, so the fix for a real bug was blocked by the test meant to protect it.
# Removing a file from the lane is the deliberate act: drop it from this set too, in the
# same commit, and the reviewer sees the coverage shrink.
_FUSED_CORE_LANE_REQUIRED = frozenset({
    "tests/test_compiler_dynamic_exec.py",
    "tests/test_compiler_rewriter_exhaustive.py",
    "tests/test_extended_dep_api_pins.py",
    "tests/test_gemma4_dtype_drift_guards.py",
    "tests/test_merge_e2e_hub_unreachable.py",
    "tests/test_missing_parent_package_is_drift.py",
    "tests/test_moe_merge_e2e_cpu.py",
    "tests/test_peft_paramwrapper_layout_drift.py",
    "tests/test_temporary_patches_exhaustive.py",
    "tests/test_torchvision_video_removed_message.py",
    "tests/test_transformers_moe_structure_drift.py",
    "tests/test_unsloth_zoo_lora_merge.py",
    "tests/test_upstream_import_fixes_drift.py",
    "tests/test_upstream_pinned_symbols_accelerator.py",
    "tests/test_upstream_pinned_symbols_transformers.py",
    "tests/test_upstream_pinned_symbols_trl_vllm.py",
    "tests/test_upstream_signatures.py",
    "tests/test_upstream_source_patterns.py",
    "tests/test_zoo_history_regressions_deep.py",
    "tests/test_zoo_source_upstream_refs.py",
})


def test_the_fused_core_job_still_runs_every_drift_file() -> None:
    """
    A command can parse and still have lost most of its arguments. This pins the
    files against the workflow the cells were fused from, so a continuation that
    silently drops files fails here rather than going green having run three.
    """
    doc = yaml.safe_load((WORKFLOWS / "consolidated-tests-ci.yml").read_text(encoding = "utf-8"))
    steps = doc["jobs"]["core-upstream-matrix"]["steps"]
    run = next(s["run"] for s in steps if "concurrently" in (s.get("name") or ""))

    body = run[run.index("-m pytest"):]
    body = body[: body.index("|| trc=$?")]
    # Join the continuations the way the shell does, then read off what is left.
    joined = body.replace("\\\n", " ")
    files = re.findall(r"tests/\S+\.py", joined)

    missing = _FUSED_CORE_LANE_REQUIRED - set(files)
    assert not missing, (
        f"the fused Core lane no longer invokes pytest on {sorted(missing)}. The three "
        f"matrix cells it replaced each ran all of them, so dropping one silently "
        f"narrows upstream-drift coverage. Lane currently runs: {sorted(files)}"
    )
    # A name repeated across continuations parses as two entries and would mask a drop
    # from the set check above, which is order- and multiplicity-blind.
    duplicated = sorted({name for name in files if files.count(name) > 1})
    assert not duplicated, (
        f"the fused Core lane names {duplicated} more than once; pytest would collect "
        f"the file twice and the duplicate could be hiding a file that went missing"
    )
    assert '>> "$log" 2>&1' in joined, (
        "the pytest output is no longer redirected into the lane log, so a failing "
        "lane would report a status with nothing to read"
    )


# A lane records its own exit status in the background, and that status is all the
# job ever learns about it. The capture is `{ cmd1; ...; cmdN } > "$log" || rc=$?`,
# and a brace group reports only cmdN -- so every earlier failure was recorded as
# rc=0, the "could not be built" guard never fired, and the job continued with a
# half-built venv. A truncated mlx download surfaced two steps later as
# ModuleNotFoundError, with the build step green.
#
# `set -e` inside the group does NOT help: it sits on the left of `|| rc=$?`, where
# POSIX ignores errexit. `&&`-chaining does, so the first failure becomes rc.

def _capture_groups(run: str) -> list[str]:
    """Bodies of every `{ ... } > ... || rc=$?` status-capture group in a lane."""
    lines = run.splitlines()
    groups, buf, depth = [], [], 0
    for line in lines:
        stripped = line.strip()
        if depth == 0 and stripped == "{":
            depth, buf = 1, []
            continue
        if depth:
            if stripped.startswith("}") and "rc=$?" in stripped:
                groups.append("\n".join(buf))
                depth = 0
            else:
                buf.append(line)
    return groups


def _statements(body: str) -> list[str]:
    """Logical statements: continuations joined, blanks and comment lines dropped.

    A comment between two `&&`-chained commands is legal shell (the newline after
    `&&` may be followed by comments), so it is not a statement for this purpose.
    """
    joined = body.replace("\\\n", " ")
    return [
        ln.strip() for ln in joined.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


def test_lane_status_capture_sees_every_failure() -> None:
    """
    Every command in a status-capture group must be `&&`-chained to the next, so a
    failure anywhere becomes the lane's recorded status rather than being masked by
    a later command that happens to succeed.
    """
    checked = 0
    for workflow, job, name, run in _lane_steps():
        for body in _capture_groups(run):
            statements = _statements(body)
            assert statements, f"{workflow}: job '{job}' step '{name}' has an empty capture group"
            checked += 1
            for stmt in statements[:-1]:
                assert stmt.endswith("&&"), (
                    f"{workflow}: job '{job}' step '{name}': the status-capture group has a "
                    f"statement that is not chained to the next one:\n    {stmt}\n"
                    "A brace group reports only its last command, so this failure would be "
                    "recorded as success. Chain it with `&&` (`set -e` does not work here: "
                    "errexit is ignored on the left of `|| rc=$?`)."
                )
            assert not statements[-1].endswith("&&"), (
                f"{workflow}: job '{job}' step '{name}': capture group ends with a dangling `&&`"
            )
    assert checked >= 3, f"only {checked} status-capture groups found; the detector looks broken"


def test_the_chaining_idiom_actually_propagates_failure() -> None:
    """
    Pins the shell semantics the test above relies on, so the reasoning is checked
    rather than asserted: the old form swallows, `set -e` does not rescue it, and
    `&&`-chaining does, while still tolerating a deliberate `|| true` step.
    """
    def rc_of(script: str) -> int:
        return subprocess.run(
            [ "bash", "-c", f"rc=0; {script} || rc=$?; exit $rc" ],
            capture_output = True, text = True,
        ).returncode

    assert rc_of("{ false; true; }") == 0, "brace group should mask an early failure"
    assert rc_of("( set -e; false; true; )") == 0, "errexit is ignored left of ||"
    assert rc_of("{ false && true; }") != 0, "&&-chaining must propagate the failure"
    assert rc_of("{ true && { false || true; } && false; }") != 0, (
        "a real failure after a tolerated `|| true` step must still propagate"
    )
    assert rc_of("{ true && { false || true; } && true; }") == 0, (
        "an all-succeeding chain with a tolerated step must stay green"
    )


def test_network_bound_installs_are_retried() -> None:
    """
    pip's `--retries` only covers establishing a connection, so a body truncated
    mid-download (ProtocolError / IncompleteRead) is not retried and the install
    dies. The lanes wrap those installs in retry().
    """
    for workflow, job, name, run in _lane_steps():
        for body in _capture_groups(run):
            installs = [
                stmt for stmt in _statements(body)
                if "pip" in stmt and " install " in stmt
                and "--upgrade pip" not in stmt
                and "--no-deps" not in stmt      # deliberately `|| true`
            ]
            for stmt in installs:
                assert stmt.startswith("retry "), (
                    f"{workflow}: job '{job}' step '{name}': network-bound install is not "
                    f"wrapped in retry():\n    {stmt}"
                )
