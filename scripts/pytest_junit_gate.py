#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Decide CI gates from a pytest --junitxml report: no-skips fails on any skip, one-passed needs a pass; they must not
be merged; xfail satisfies neither; failures or errors must never pass, nor a missing, empty or inconsistent report;
exit 0 satisfied, 1 violated or unusable, 2 bad usage."""

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
        # The suite's tests attribute is checked against the testcase count, since a file that parses can still be a partial or merged report.
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
                # An xfail is recorded as a skipped element with type pytest.xfail, but pytest counts it as xfailed, so
                # it satisfies neither policy and gets its own bucket.
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
