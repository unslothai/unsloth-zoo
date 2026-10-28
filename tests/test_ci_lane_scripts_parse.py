# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Concurrent CI lanes must parse as whole shell with every pytest file present: a double-backslash continuation
silently left twenty files out of the run; bash -n plus a file-count check guard it."""

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

# Steps that hand-build shell for concurrent lanes, matched by content rather than a hardcoded list so new fused jobs are covered automatically.
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
    """A double backslash at line end is an escaped backslash, not a continuation: the next line runs as its own command, which bash -n does not flag."""
    bad = [
        i + 1
        for i, line in enumerate(run.splitlines())
        if re.search(r"[^\\]\\\\$", line)
    ]
    assert not bad, (
        f"{workflow}: job '{job}' step '{name}' has escaped backslashes where line "
        f"continuations were meant, at line(s) {bad}"
    )


# Twenty drift files from the three fused matrix cells, named not counted: an exact count went red when ref 1114
# legitimately added a file; removing one means dropping it from this set in the same commit.
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
    """Pins the fused Core lane's files against the workflow it was fused from, so a continuation that silently drops files fails instead of passing with three."""
    doc = yaml.safe_load((WORKFLOWS / "consolidated-tests-ci.yml").read_text(encoding = "utf-8"))
    steps = doc["jobs"]["core-upstream-matrix"]["steps"]
    run = next(s["run"] for s in steps if "concurrently" in (s.get("name") or ""))

    body = run[run.index("-m pytest"):]
    body = body[: body.index("|| trc=$?")]
    joined = body.replace("\\\n", " ")
    files = re.findall(r"tests/\S+\.py", joined)

    missing = _FUSED_CORE_LANE_REQUIRED - set(files)
    assert not missing, (
        f"the fused Core lane no longer invokes pytest on {sorted(missing)}. The three "
        f"matrix cells it replaced each ran all of them, so dropping one silently "
        f"narrows upstream-drift coverage. Lane currently runs: {sorted(files)}"
    )
    # A repeated name would mask a dropped file, since the set check above is blind to multiplicity.
    duplicated = sorted({name for name in files if files.count(name) > 1})
    assert not duplicated, (
        f"the fused Core lane names {duplicated} more than once; pytest would collect "
        f"the file twice and the duplicate could be hiding a file that went missing"
    )
    assert '>> "$log" 2>&1' in joined, (
        "the pytest output is no longer redirected into the lane log, so a failing "
        "lane would report a status with nothing to read"
    )


# A brace group captured with || rc=$? reports only its last command, so earlier failures read as rc=0 and a half-built
# venv goes on; set -e inside the group does not help, since POSIX ignores errexit left of ||.

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
    """Joins line continuations and drops blank and comment lines; a comment between && chained commands is legal shell, so it is not a statement here."""
    joined = body.replace("\\\n", " ")
    return [
        ln.strip() for ln in joined.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


def test_lane_status_capture_sees_every_failure() -> None:
    """Every command in a status-capture group must be &&-chained, so any failure becomes the lane's recorded status instead of being masked by a later success."""
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
    """Pins the shell semantics: the old form swallows failures even under set -e, && chaining propagates them, and a deliberate || true step still passes."""
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
    """pip --retries covers only connecting, not a body truncated mid-download (IncompleteRead), so network installs are wrapped in retry(), bootstrap included."""
    for workflow, job, name, run in _lane_steps():
        for body in _capture_groups(run):
            installs = [
                stmt for stmt in _statements(body)
                if "pip" in stmt and " install " in stmt
                and "--no-deps" not in stmt      # deliberately `|| true`
            ]
            for stmt in installs:
                assert stmt.startswith("retry "), (
                    f"{workflow}: job '{job}' step '{name}': network-bound install is not "
                    f"wrapped in retry():\n    {stmt}"
                )
