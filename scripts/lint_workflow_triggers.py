#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Refuse dangerous GitHub Actions trigger patterns at PR time.

Flags three patterns from the TanStack GHSA-g7cv-rxg3-hmpx compromise:

1.  `pull_request_target` -- runs a fork's workflow YAML against the BASE
    repo's secrets/permissions, letting the fork inject code into the base
    context. No safe use for public projects; use `pull_request` instead.

2.  `workflow_run` chained to a PR-triggered workflow -- same trust-boundary
    problem one hop later: a PR can poison artifacts/caches that the
    elevated-permission run then consumes.

3.  Shared cache keys between PR-triggered and publish/release/push
    workflows -- a fork PR can poison the cache the release run restores.
    Keys must be partitioned so secrets-holding workflows never read what a
    PR can write.

Exit codes: 0 = no findings; 1 = findings (stderr lists each with file path).

Run from repo root:
    python3 scripts/lint_workflow_triggers.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print(
        "ERROR: PyYAML is required. Install with 'pip install pyyaml'", file = sys.stderr
    )
    sys.exit(2)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

BANNED_TRIGGERS: tuple[str, ...] = ("pull_request_target",)
RESTRICTED_TRIGGERS: tuple[str, ...] = ("workflow_run",)
PUBLISH_WORKFLOW_NAMES: tuple[str, ...] = ("release-desktop.yml",)


def _normalise_on(on_field):
    if isinstance(on_field, str):
        return {on_field}
    if isinstance(on_field, list):
        return set(on_field)
    if isinstance(on_field, dict):
        return set(on_field.keys())
    return set()


def _load_workflow(path: Path):
    try:
        return yaml.safe_load(path.read_text())
    except Exception as exc:
        print(f"ERROR: failed to parse {path}: {exc}", file = sys.stderr)
        sys.exit(2)


def _extract_restore_key_prefixes(path: Path) -> list[str]:
    """Every prefix a `restore-keys:` block offers as a fallback.

    The exact-key comparison below cannot see these. `restore-keys` restores the newest
    entry whose key merely STARTS WITH the prefix, so a publish workflow can adopt an
    entry a pull request wrote without the two keys ever being equal.
    """
    text = path.read_text()
    prefixes: list[str] = []
    for m in re.finditer(r"(?:^|\n)([ \t]*)restore-keys:[ \t]*(\|-?|>-?)?[ \t]*([^\n]*)\n", text):
        indent, block, inline = m.group(1), m.group(2), m.group(3).strip()
        if inline and not block:
            prefixes.append(inline)
            continue
        for line in text[m.end():].split("\n"):
            if not line.strip():
                break
            if len(line) - len(line.lstrip()) <= len(indent):
                break
            prefixes.append(line.strip())
    return [x for x in prefixes if x]


def _literal_prefix(key: str) -> str:
    """The fixed-text head of a key, i.e. everything before the first expression.

    Keys are mostly `literal-${{ something }}`, so comparing whole strings compares the
    expressions too and almost never matches. The literal head is what decides whether
    one key can satisfy another's prefix restore.
    """
    return re.split(r"\$\{\{", key, maxsplit = 1)[0].strip().strip("'\"")


def _extract_cache_keys(path: Path) -> list[str]:
    text = path.read_text()
    keys: list[str] = []
    for m in re.finditer(r"(?:^|\n)\s*key:\s*([^\n]+)", text):
        keys.append(m.group(1).strip())
    return keys


def _trigger_set(yaml_doc) -> set[str]:
    on = yaml_doc.get(True)
    if on is None:
        on = yaml_doc.get("on")
    return _normalise_on(on)


def main() -> int:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "--workflows-dir",
        type = Path,
        default = DEFAULT_WORKFLOWS_DIR,
        help = "Override the workflows directory (used by tests).",
    )
    args = parser.parse_args()
    workflows_dir = args.workflows_dir

    findings: list[str] = []
    workflows = sorted(workflows_dir.glob("*.yml"))
    pr_triggered: list[tuple[Path, list[str]]] = []
    publish_triggered: list[tuple[Path, list[str]]] = []
    publish_restore_prefixes: list[tuple[Path, list[str]]] = []

    for path in workflows:
        doc = _load_workflow(path)
        triggers = _trigger_set(doc)

        for t in BANNED_TRIGGERS:
            if t in triggers:
                findings.append(
                    f"{path.name}: BANNED trigger '{t}' (GHSA-g7cv-rxg3-hmpx "
                    "pattern: fork PRs run in base-repo context). Switch to "
                    "'pull_request' and use a deploy-on-merge workflow for "
                    "any privileged step."
                )

        for t in RESTRICTED_TRIGGERS:
            if t in triggers:
                text = path.read_text()
                if "lint:workflow_triggers-allow-workflow_run" not in text:
                    findings.append(
                        f"{path.name}: RESTRICTED trigger '{t}' requires an "
                        "explicit `# lint:workflow_triggers-allow-workflow_run` "
                        "comment somewhere in the file, with a justification."
                    )

        if "pull_request" in triggers:
            pr_triggered.append((path, _extract_cache_keys(path)))
        is_dispatch_only = "workflow_dispatch" in triggers and not (
            "push" in triggers or "pull_request" in triggers
        )
        if path.name in PUBLISH_WORKFLOW_NAMES or is_dispatch_only:
            publish_triggered.append((path, _extract_cache_keys(path)))
            publish_restore_prefixes.append((path, _extract_restore_key_prefixes(path)))

    # A PR-triggered workflow usually delegates its key to a composite action, so the
    # literal key lives in .github/actions/*/action.yml while the workflow carries only
    # `${{ steps.x.outputs.key }}`. Those count as pull-request-reachable, because the
    # workflow using them runs on pull requests. Without this the prefix rule below would
    # compare against opaque expressions and match nothing.
    composite_dir = workflows_dir.parent / "actions"
    composite_keys: list[str] = []
    if composite_dir.is_dir():
        for action_path in sorted(composite_dir.rglob("action.y*ml")):
            composite_keys.extend(_extract_cache_keys(action_path))

    pr_keys = {key for _, keys in pr_triggered for key in keys}
    for pub_path, pub_keys in publish_triggered:
        for k in pub_keys:
            if k in pr_keys:
                findings.append(
                    f"{pub_path.name}: cache key {k!r} is also declared in a "
                    "PR-triggered workflow. A fork PR could poison this cache "
                    "and the publish workflow would restore it on next run. "
                    "Add a unique suffix (e.g. '-publish-only') to partition "
                    "the namespaces."
                )

    # The same trust boundary, reached by prefix instead of by an equal key.
    pr_key_prefixes = {
        _literal_prefix(k) for k in list(pr_keys) + composite_keys if _literal_prefix(k)
    }
    for pub_path, prefixes in publish_restore_prefixes:
        for prefix in prefixes:
            literal = _literal_prefix(prefix)
            if not literal:
                findings.append(
                    f"{pub_path.name}: restore-keys entry {prefix!r} begins with an "
                    "expression, so what it can restore is not decidable here. Give it "
                    "a literal prefix."
                )
                continue
            for pr_prefix in sorted(pr_key_prefixes):
                if pr_prefix.startswith(literal):
                    findings.append(
                        f"{pub_path.name}: restore-keys prefix {literal!r} matches "
                        f"{pr_prefix!r}, a cache key namespace a PR-triggered workflow "
                        "writes. A prefix restore takes the newest matching entry, so "
                        "this publish workflow could adopt a cache a pull request "
                        "produced even though no key is equal. Partition the "
                        "namespaces, or drop the restore-keys fallback here."
                )

    if findings:
        print(
            "Workflow trigger lint failed with the following issues:", file = sys.stderr
        )
        for f in findings:
            print(f"  - {f}", file = sys.stderr)
        return 1

    print(
        f"OK: scanned {len(workflows)} workflow file(s); "
        f"no pull_request_target, no unjustified workflow_run, "
        f"no PR/publish cache-key collision."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
