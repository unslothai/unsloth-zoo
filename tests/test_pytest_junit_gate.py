# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""
scripts/pytest_junit_gate.py replaces a second pytest run with a read of the
first run's JUnit XML, so it has to reach the same verdict the greps did.

The reports here are produced by really running pytest on generated files rather
than by hand-writing XML: the whole point of the change is that pytest's report
format, not our idea of it, decides the gate. The malformed cases (empty file,
truncated XML, wrong root, count mismatch) are written by hand because pytest
cannot be asked to emit them.

The case worth naming is the module skipped at import -- `pytest.importorskip`
at module level, which is how every MLX file behaves without the real runtime.
It exits 0, prints "1 skipped", and pytest records it as ONE testcase carrying
<skipped>. A gate that only asked "were there failures?" calls that green; both
policies here refuse it.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
_GATE_PATH = REPO_ROOT / "scripts" / "pytest_junit_gate.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location("zoo_pytest_junit_gate", _GATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


# name -> (source, expected pytest exit status). The statuses are asserted so a
# future pytest that stops writing a report for one of these shows up here.
_SOURCES = {
    "all_pass": ("def test_a():\n    assert True\n\ndef test_b():\n    assert True\n", 0),
    "one_skip": (
        "import pytest\n\n"
        "def test_a():\n    assert True\n\n"
        "@pytest.mark.skip(reason = 'metal only')\n"
        "def test_b():\n    pass\n",
        0,
    ),
    "module_skipped": (
        "import pytest\n"
        "pytest.importorskip('unsloth_zoo_absent_dependency')\n\n"
        "def test_a():\n    assert True\n",
        5,
    ),
    "collection_error": (
        "import unsloth_zoo_absent_dependency\n\n"
        "def test_a():\n    assert True\n",
        2,
    ),
    "genuine_failure": ("def test_a():\n    assert False\n", 1),
    "no_tests": ("# nothing collectable here\n", 5),
    "xfail": (
        "import pytest\n\n"
        "@pytest.mark.xfail(reason = 'known')\n"
        "def test_a():\n    assert False\n\n"
        "def test_b():\n    assert True\n",
        0,
    ),
}


@pytest.fixture(scope = "module")
def reports(tmp_path_factory) -> dict[str, Path]:
    """Real `--junitxml` output for each source, keyed by name."""
    workdir = tmp_path_factory.mktemp("junit_gate")
    out = {}
    for name, (source, expected_status) in _SOURCES.items():
        test_file = workdir / f"t_{name}.py"
        test_file.write_text(source, encoding = "utf-8")
        report = workdir / f"{name}.xml"
        completed = subprocess.run(
            [
                sys.executable, "-m", "pytest", str(test_file),
                "-q", "-rs", "-p", "no:cacheprovider",
                f"--junitxml={report}",
            ],
            cwd = workdir, capture_output = True, text = True,
        )
        assert completed.returncode == expected_status, (
            f"{name}: pytest exited {completed.returncode}, expected "
            f"{expected_status}\n{completed.stdout}\n{completed.stderr}"
        )
        assert report.exists(), f"{name}: pytest wrote no report\n{completed.stdout}"
        out[name] = report
    return out


def _run(report: Path, policy: str) -> int:
    return gate.main([ "--report", str(report), "--require", policy ])


@pytest.mark.parametrize("policy", list(gate.POLICIES))
def test_a_clean_run_satisfies_every_policy(reports, policy: str) -> None:
    assert _run(reports["all_pass"], policy) == 0


def test_no_skips_rejects_a_single_skip_that_one_passed_accepts(reports) -> None:
    """The two policies are not interchangeable, which is why both exist."""
    assert _run(reports["one_skip"], "no-skips") == 1
    assert _run(reports["one_skip"], "one-passed") == 0


@pytest.mark.parametrize("policy", list(gate.POLICIES))
def test_a_module_skipped_at_import_fails_every_policy(reports, policy: str) -> None:
    """The case the whole gate exists for: exit 0, nothing ran."""
    assert _run(reports["module_skipped"], policy) == 1


@pytest.mark.parametrize("policy", list(gate.POLICIES))
def test_collection_error_fails_every_policy(reports, policy: str) -> None:
    assert _run(reports["collection_error"], policy) == 1


@pytest.mark.parametrize("policy", list(gate.POLICIES))
def test_a_genuine_failure_is_never_a_pass(reports, policy: str) -> None:
    assert _run(reports["genuine_failure"], policy) == 1


@pytest.mark.parametrize("policy", list(gate.POLICIES))
def test_a_report_with_no_tests_fails_every_policy(reports, policy: str) -> None:
    assert _run(reports["no_tests"], policy) == 1


def test_xfail_is_neither_a_skip_nor_a_pass(reports) -> None:
    """
    pytest writes an xfail as <skipped type="pytest.xfail"> but summarises it as
    "xfailed". The greps read the summary, so an xfail neither tripped no-skips
    nor satisfied one-passed; keep it that way.
    """
    counts = gate.read_report(reports["xfail"])
    assert (counts.xfailed, counts.skipped, counts.passed) == (1, 0, 1)
    assert _run(reports["xfail"], "no-skips") == 0


def test_counts_match_the_report(reports) -> None:
    counts = gate.read_report(reports["one_skip"])
    assert (counts.passed, counts.skipped, counts.failed, counts.errored) == (1, 1, 0, 0)
    assert counts.total == 2


def test_a_missing_report_fails_rather_than_passing(tmp_path) -> None:
    assert _run(tmp_path / "never-written.xml", "no-skips") == 1
    with pytest.raises(gate.ReportUnusable):
        gate.read_report(tmp_path / "never-written.xml")


@pytest.mark.parametrize(
    "name,content",
    [
        ("empty", ""),
        ("whitespace", "   \n"),
        ("truncated", '<?xml version="1.0"?><testsuites><testsuite tests="2"'),
        ("wrong_root", "<coverage line-rate='1'/>"),
        (
            "count_mismatch",
            # Declares two cases, carries one: a partial write must not be read as
            # a clean run of the one case that made it to disk.
            '<testsuites><testsuite tests="2" errors="0" failures="0" skipped="0">'
            '<testcase classname="t" name="test_a"/></testsuite></testsuites>',
        ),
        ("no_suite", "<testsuites/>"),
        (
            "suite_without_count",
            '<testsuite><testcase classname="t" name="test_a"/></testsuite>',
        ),
    ],
)
@pytest.mark.parametrize("policy", list(gate.POLICIES))
def test_unusable_reports_fail_loudly(
    tmp_path, policy: str, name: str, content: str,
) -> None:
    report = tmp_path / f"{name}.xml"
    report.write_text(content, encoding = "utf-8")
    assert _run(report, policy) == 1, f"{name} was treated as a satisfied gate"
    with pytest.raises(gate.ReportUnusable):
        gate.read_report(report)


def test_a_directory_is_not_a_report(tmp_path) -> None:
    assert _run(tmp_path, "no-skips") == 1


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(ValueError):
        gate.check(gate.Counts(1, 0, 0, 0), "whatever")
    with pytest.raises(SystemExit):
        gate.main([ "--report", "x.xml", "--require", "no-skips-at-all" ])


def test_the_workflow_gates_use_this_script_and_do_not_re_run_pytest() -> None:
    """
    The saving is the point: if a gate step ever grows a second pytest
    invocation of the file it is gating, this change has been undone.
    """
    yaml = pytest.importorskip("yaml")
    workflow = REPO_ROOT / ".github" / "workflows" / "consolidated-tests-ci.yml"
    doc = yaml.safe_load(workflow.read_text(encoding = "utf-8"))
    steps = doc["jobs"]["mlx-cpu-linux"]["steps"]

    gates = [ s for s in steps if (s.get("name") or "").startswith("Fail if") ]
    assert len(gates) == 5, f"expected 5 gate steps, found {len(gates)}"

    reports_written = {
        token.split("=", 1)[1].strip('"')
        for step in steps
        for token in (step.get("run") or "").split()
        if token.startswith("--junitxml=")
    }
    assert len(reports_written) == 5, (
        f"the gated pytest steps write {len(reports_written)} distinct reports, "
        f"expected 5: {sorted(reports_written)}"
    )

    for step in gates:
        run = step.get("run") or ""
        assert "scripts/pytest_junit_gate.py" in run, (
            f"gate step '{step['name']}' does not read a report"
        )
        assert "-m pytest" not in run, (
            f"gate step '{step['name']}' re-runs pytest; the verdict must come "
            f"from the first run's report"
        )
        assert any(
            Path(written).name in run for written in reports_written
        ), f"gate step '{step['name']}' names a report no pytest step writes"
