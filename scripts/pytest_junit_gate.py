#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Decide a CI skip gate from a pytest JUnit XML report instead of re-running pytest.

Several lanes gate on "the suite must not have skipped" or "at least one test
must really have run". That was expressed as a second pytest invocation of the
same file whose stdout was grepped:

    python -m pytest tests/x.py -q -rs | tee out.txt
    if grep -qiE '[0-9]+ skipped' out.txt; then ...

which runs the whole file twice. The first run already knows the answer, so it
writes `--junitxml=<report>` and this script reads the report.

Two policies, matching the two the greps expressed. They are NOT the same check
and must not be merged:

    no-skips    -- fail if any test skipped (the `grep -qiE '[0-9]+ skipped'`
                   gates: any skip at all means the numeric gate gated nothing).
    one-passed  -- fail unless at least one test passed (the
                   `grep -qiE '[1-9][0-9]* passed'` gates: these files skip
                   wholesale when an optional import is missing, and a file that
                   skips everything still exits 0).

A skip recorded at collection time -- the whole module skipped at import, which
is precisely the case these gates exist for -- appears in the report as a
testcase carrying a <skipped> element, so both policies see it.

xfail is bucketed on its own and satisfies neither policy, matching the terminal
summary the greps read (an xfail is reported as "xfailed", not as "skipped" or
"passed"). A non-strict xpass is the one outcome JUnit XML cannot distinguish
from a plain pass; none of the gated files use xfail markers, so the two agree
today, and a strict xpass is a failure in both worlds.

Anything that stops the report from being a truthful record of a real run is a
failure, never a pass: a missing file, an empty file, unparseable XML, no
testsuite element, zero testcases, or a testcase count that disagrees with the
suite's own `tests` attribute. A gate that cannot read its evidence has not
been satisfied.

Failures and errors also fail the gate. In the workflow they cannot reach it --
the pytest step that wrote the report already returned non-zero and stopped the
job -- but a report containing red tests must never be reported as a pass by
this script, whoever calls it.

Exit codes: 0 = gate satisfied; 1 = gate violated or report unusable;
2 = bad usage.

Usage:
    python3 scripts/pytest_junit_gate.py --report "$RUNNER_TEMP/x.xml" \
        --require no-skips --message "real mlx is installed but ... skipped"
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path


POLICIES = ("no-skips", "one-passed")


class ReportUnusable(Exception):
    """The report cannot answer the question. Always fatal, never a pass."""


class Counts:
    def __init__(
        self, passed: int, skipped: int, failed: int, errored: int, xfailed: int = 0
    ) -> None:
        self.passed = passed
        self.skipped = skipped
        self.failed = failed
        self.errored = errored
        self.xfailed = xfailed

    @property
    def total(self) -> int:
        return self.passed + self.skipped + self.failed + self.errored + self.xfailed

    def __str__(self) -> str:
        return (
            f"{self.total} test(s): {self.passed} passed, {self.skipped} skipped, "
            f"{self.failed} failed, {self.errored} errored, {self.xfailed} xfailed"
        )


def read_report(path: Path) -> Counts:
    """Count outcomes in a pytest --junitxml report, or raise ReportUnusable."""
    if not path.exists():
        raise ReportUnusable(
            f"no report at {path}. The pytest step should have written it with "
            f"--junitxml; either it did not run, it crashed before writing, or "
            f"the two paths disagree."
        )
    if not path.is_file():
        raise ReportUnusable(f"{path} is not a file")
    raw = path.read_bytes()
    if not raw.strip():
        raise ReportUnusable(f"the report at {path} is empty ({len(raw)} bytes)")

    try:
        root = ElementTree.fromstring(raw)
    except ElementTree.ParseError as error:
        raise ReportUnusable(f"the report at {path} is not parseable XML: {error}")

    if root.tag == "testsuite":
        suites = [root]
    elif root.tag == "testsuites":
        suites = list(root.findall("testsuite"))
    else:
        raise ReportUnusable(
            f"the report at {path} has root element <{root.tag}>, which is not a "
            f"pytest JUnit XML report"
        )
    if not suites:
        raise ReportUnusable(
            f"the report at {path} contains no <testsuite>, so nothing recorded a run"
        )

    passed = skipped = failed = errored = xfailed = 0
    declared = 0
    for suite in suites:
        # pytest writes the case count on the suite. A file that parses can still
        # be a partial write or a report merged from elsewhere, and then the
        # elements below are not the run we think we are gating on.
        attribute = suite.get("tests")
        if attribute is None:
            raise ReportUnusable(
                f"the <testsuite> in {path} has no `tests` attribute; this is not "
                f"a pytest JUnit XML report"
            )
        try:
            declared += int(attribute)
        except ValueError:
            raise ReportUnusable(
                f"the <testsuite> in {path} declares tests={attribute!r}, not a number"
            )
        for case in suite.iter("testcase"):
            skip = case.find("skipped")
            if case.find("error") is not None:
                errored += 1
            elif case.find("failure") is not None:
                failed += 1
            elif skip is not None:
                # An xfail is written as <skipped type="pytest.xfail">, but pytest's
                # terminal summary counts it as "xfailed", not "skipped" and not
                # "passed". The greps these gates replace were reading that summary,
                # so an xfail satisfied neither policy and violated neither. Keep it
                # in its own bucket to stay exactly as strict as before.
                if skip.get("type") == "pytest.xfail":
                    xfailed += 1
                else:
                    skipped += 1
            else:
                passed += 1

    counts = Counts(passed, skipped, failed, errored, xfailed)
    if counts.total != declared:
        raise ReportUnusable(
            f"the report at {path} declares tests={declared} but carries "
            f"{counts.total} <testcase> element(s); it is truncated or was not "
            f"written by this run"
        )
    if counts.total == 0:
        raise ReportUnusable(
            f"the report at {path} records no tests at all. pytest collected "
            f"nothing, so neither a skip gate nor a pass gate has any evidence."
        )
    return counts


def check(counts: Counts, policy: str) -> str | None:
    """Return the violation for `policy`, or None when the gate is satisfied."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
    if counts.failed or counts.errored:
        return (
            f"the report records {counts.failed} failed and {counts.errored} "
            f"errored test(s); a red run is not a satisfied gate"
        )
    if policy == "no-skips" and counts.skipped:
        return f"{counts.skipped} test(s) skipped, and this gate forbids any skip"
    if policy == "one-passed" and counts.passed == 0:
        return "no test passed, so nothing in this file actually ran"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description = "Assert a pytest gate from a JUnit XML report.",
    )
    parser.add_argument("--report", required = True, help = "path to the --junitxml file")
    parser.add_argument("--require", required = True, choices = POLICIES)
    parser.add_argument(
        "--message",
        default = "",
        help = "extra context printed on failure, e.g. the old gate's echo line",
    )
    arguments = parser.parse_args(argv)

    report = Path(arguments.report)
    try:
        counts = read_report(report)
    except ReportUnusable as error:
        print(f"::error::gate '{arguments.require}': {error}", file = sys.stderr)
        if arguments.message:
            print(arguments.message, file = sys.stderr)
        return 1

    violation = check(counts, arguments.require)
    if violation is None:
        print(f"gate '{arguments.require}' satisfied by {report}: {counts}")
        return 0
    print(f"::error::gate '{arguments.require}' violated by {report}: {violation}", file = sys.stderr)
    print(f"report says: {counts}", file = sys.stderr)
    if arguments.message:
        print(arguments.message, file = sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
