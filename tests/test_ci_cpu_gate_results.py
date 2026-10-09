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

"""Exercise the actual two-lane CI shell with controlled pytest outcomes."""

import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import pytest
import yaml


WORKFLOW = (
    Path(__file__).resolve().parents[1] / ".github/workflows/consolidated-tests-ci.yml"
)
pytestmark = pytest.mark.skipif(not shutil.which("bash"), reason = "requires bash")


def step_script(name):
    steps = yaml.safe_load(WORKFLOW.read_text())["jobs"]["repo-tests-cpu"]["steps"]
    return next(s["run"] for s in steps if s.get("name") == name)


def run_gates(tmp_path, failing = "", barrier = False):
    script = step_script("pytest CPU gates (two process lanes)")
    # Stubs only the child pytest calls; waiting, logs and exit codes run as the real workflow shell.
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
  if [ "$FAIL_FILE" = hang ] && [[ "$*" == *tests/test_ci_cpu_gate_results.py* ]]; then
    sleep 60
  fi
  if [ "$FAIL_FILE" = crash ] && [[ "$*" == *tests/test_ci_cpu_gate_results.py* ]]; then
    kill -TERM "$BASHPID"
  fi
  if [ -n "$FAIL_FILE" ] && [[ "$*" == *"$FAIL_FILE"* ]]; then
    return 7
  fi
}
"""
    env = {
        **os.environ,
        "RUNNER_TEMP": str(tmp_path),
        "FAIL_FILE": failing,
        "BARRIER": str(int(barrier)),
    }
    if failing != "hang":
        return subprocess.run(
            ["bash", "-c", stub + script],
            env = env,
            capture_output = True,
            text = True,
            timeout = 15,
        )
    # Stand-in for the step timeout: kill the whole step once the other lane is done.
    proc = subprocess.Popen(
        ["bash", "-c", stub + script],
        env = env,
        stdout = subprocess.DEVNULL,
        stderr = subprocess.DEVNULL,
        start_new_session = True,
    )
    try:
        for _ in range(500):
            if (tmp_path / "zoo-cpu-gates" / "cpu-advisory.status").exists():
                break
            time.sleep(0.02)
    finally:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def print_logs_after_timeout(tmp_path):
    return subprocess.run(
        ["bash", "-c", step_script("Print CPU gate logs after a step timeout")],
        env = {**os.environ, "RUNNER_TEMP": str(tmp_path)},
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


def test_lane_logs_survive_a_step_timeout(tmp_path):
    run_gates(tmp_path, "hang")
    assert not (tmp_path / "zoo-cpu-gates" / ".reported").exists()
    result = print_logs_after_timeout(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "executed -m pytest tests/test_ci_cpu_gate_results.py" in result.stdout
    assert result.stdout.count("executed -m pytest") == 8


def test_logs_are_not_printed_twice(tmp_path):
    run_gates(tmp_path, "tests/security")
    result = print_logs_after_timeout(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == ""
