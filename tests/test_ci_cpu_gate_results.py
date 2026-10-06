# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Exercise the actual two-lane CI shell with controlled pytest outcomes."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml


WORKFLOW = (
    Path(__file__).resolve().parents[1] / ".github/workflows/consolidated-tests-ci.yml"
)
pytestmark = pytest.mark.skipif(not shutil.which("bash"), reason = "requires bash")


def run_gates(tmp_path, failing = "", barrier = False):
    steps = yaml.safe_load(WORKFLOW.read_text())["jobs"]["repo-tests-cpu"]["steps"]
    script = next(
        s["run"]
        for s in steps
        if s.get("name") == "pytest CPU gates (two process lanes)"
    )
    # The function intercepts only the child pytest processes. The controller is
    # the workflow's real shell, including wait, logs, and exit-code collection.
    stub = r"""
python() {
  printf 'executed %s\n' "$*"
  if [ "$BARRIER" = 1 ]; then
    case "$*" in
      *tests/security*) touch "$RUNNER_TEMP/security-ready"; peer=behavior-ready ;;
      *tests/test_ci_cpu_gate_results.py*) touch "$RUNNER_TEMP/behavior-ready"; peer=security-ready ;;
      *) peer= ;;
    esac
    if [ -n "$peer" ]; then
      for attempt in $(seq 1 100); do
        [ -e "$RUNNER_TEMP/$peer" ] && break
        sleep 0.02
      done
      [ -e "$RUNNER_TEMP/$peer" ] || return 9
    fi
  fi
  if [ "$FAIL_FILE" = crash ] && [[ "$*" == *tests/test_ci_cpu_gate_results.py* ]]; then
    kill -TERM "$BASHPID"
  fi
  if [ -n "$FAIL_FILE" ] && [[ "$*" == *"$FAIL_FILE"* ]]; then
    return 7
  fi
}
"""
    return subprocess.run(
        ["bash", "-c", stub + script],
        env = {
            **os.environ,
            "RUNNER_TEMP": str(tmp_path),
            "FAIL_FILE": failing,
            "BARRIER": str(int(barrier)),
        },
        capture_output = True,
        text = True,
        timeout = 15,
    )


@pytest.mark.parametrize(
    "failing",
    [
        "tests/security",
        "tests/test_top_level_imports_are_declared.py",
        "tests/test_mlx_save_export_edge_cases.py",
    ],
)
def test_hard_failure_keeps_all_eight_invocations_visible(tmp_path, failing):
    result = run_gates(tmp_path, failing)
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout.count("executed -m pytest") == 8
    assert "::error::" in result.stdout
    assert "exited 7" in result.stdout


def test_advisory_failure_remains_nonblocking(tmp_path):
    result = run_gates(tmp_path, "tests/test_pypi_version_sync.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "::warning::cpu-advisory exited 7" in result.stdout
    assert result.stdout.count("executed -m pytest") == 8


def test_both_lanes_run_concurrently(tmp_path):
    result = run_gates(tmp_path, barrier = True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "security-ready").exists()
    assert (tmp_path / "behavior-ready").exists()
    assert result.stdout.count("executed -m pytest") == 8


def test_missing_result_after_a_lane_crash_is_a_failure(tmp_path):
    result = run_gates(tmp_path, "crash")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "::error::behavioral exited missing" in result.stdout
    assert result.stdout.count("executed -m pytest") == 8
