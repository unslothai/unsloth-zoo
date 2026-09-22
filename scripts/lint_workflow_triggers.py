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

    A blank line does NOT end the block. A YAML block scalar runs until the indentation
    drops, blank lines included, and actions/cache reads the value as a newline-delimited
    list and skips empty entries. So `safe-`, a blank line, then `shared-` really does
    offer `shared-`; treating the blank line as the end silently dropped every prefix
    after it, which is the half a reviewer is least likely to have looked at.
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
                continue
            if len(line) - len(line.lstrip()) <= len(indent):
                break
            prefixes.append(line.strip())
    return [x for x in prefixes if x]


def _literal_prefix(key: str) -> str:
    """The fixed-text head of a key: everything before the first expression.

    Keys are mostly `literal-${{ something }}`, so comparing whole strings compares the
    expressions too and almost never matches.
    """
    return re.split(r"\$\{\{", key, maxsplit = 1)[0].strip().strip("'\"")


# `runner.os` is the only expression that routinely LEADS a cache key, and it takes
# exactly three values, so expanding it turns the common undecidable case into three
# decidable ones. Confirmed against GitHub's docs: the values are Linux, Windows and
# macOS, exact and case-sensitive.
_RUNNER_OS_VALUES = ("Linux", "Windows", "macOS")
_RUNNER_OS_EXPR = re.compile(r"\$\{\{\s*runner\.os\s*\}\}")

# A key that is nothing but one expression referring to a step output or an action input
# delegates its namespace rather than declaring one.
_DELEGATED_KEY = re.compile(r"\$\{\{\s*(steps|inputs|needs)\.[^}]*\}\}")


def _prefix_candidates(key: str) -> list[str]:
    """Every literal head this key could have at runtime.

    A key beginning with an expression has no literal head at all, and dropping it was a
    hole: a PR writing `${{ runner.os }}-shared-abc` and a publish job restoring
    `Linux-shared-` would never be compared, because the PR side reduced to the empty
    string and was filtered out. Expanding `runner.os` first gives `Linux-shared-abc`,
    which is comparable. Anything still expression-led afterwards is genuinely
    undecidable and is reported rather than dropped.
    """
    keys = [_RUNNER_OS_EXPR.sub(v, key) for v in _RUNNER_OS_VALUES] if _RUNNER_OS_EXPR.search(key) else [key]
    return [h for h in (_literal_prefix(k) for k in keys) if h]


def _prefix_compatible(pr_head: str, publish_prefix: str) -> bool:
    """Can a key with this literal head be restored by this prefix?

    `restore-keys` matching is left-anchored and exact, with no globbing, so the two are
    compatible when either is a prefix of the other. The second direction is the one that
    was missing: a PR key `pip-v2-${{ runner.os }}-abc` reduces to the head `pip-v2-`,
    and a publish prefix `pip-v2-Linux-` is LONGER than that head, so a one-directional
    `head.startswith(prefix)` test says no while the runtime key `pip-v2-Linux-abc` does
    start with the prefix and would be restored.
    """
    return pr_head.startswith(publish_prefix) or publish_prefix.startswith(pr_head)


def _shell_built_key_prefixes(text: str, inputs: set | None = None) -> list[str]:
    """Literal key heads assembled in a composite action's shell, not in its YAML.

    The pip and uv caches build their key in a `run:` step and expose it as an output, so
    the YAML `key:` is only `${{ steps.probe.outputs.key }}` and carries no namespace at
    all. Reading YAML alone therefore learned nothing about the very composites this
    check exists to cover: pip-cache-restore's real namespace is the `pip-v2-` in
    `prefix="pip-v2-${name}-..."`, several lines away from any `key:`.

    `inputs` are the values callers actually pass for the first shell variable in such a
    prefix, which keeps the recorded namespace as narrow as the real one. See
    `_local_action_inputs` for why a broader one is not the safe direction.

    Only `key`-ish and `prefix`-ish variables are read. Taking every shell assignment
    would invent namespaces that no cache uses, and each invented one is a potential
    false rejection of a publish prefix.
    """
    heads: list[str] = []
    pattern = re.compile(
        r"""(?:^|[\s;(])(?:[A-Za-z_]*_)?(?:key|prefix|KEY|PREFIX)\s*=\s*["']?"""
        r"""([A-Za-z0-9][A-Za-z0-9._-]*?-)(?=\$|\{)""",
        re.M,
    )
    for m in pattern.finditer(text):
        heads.append(m.group(1))
    # `echo "key=pip-v2-${hash}" >> "$GITHUB_OUTPUT"` is the same thing written inline.
    for m in re.finditer(
        r"""echo\s+["']?(?:key|prefix)=([A-Za-z0-9][A-Za-z0-9._-]*?-)(?=\$|\{)""", text
    ):
        heads.append(m.group(1))
    if inputs:
        return [f"{h}{v}-" for h in heads for v in sorted(inputs)]
    return heads



def _local_action_inputs(pr_paths: list, action_dir: str) -> set:
    """The `name:`-style values PR workflows actually pass to a local action.

    Read off the `with:` block of each `uses: ./...<action_dir>` call site. Needed because
    a namespace recorded more broadly than the real one is a false rejection: this repo's
    pip cache builds `prefix="pip-${name}-..."`, so reading the shell alone records the
    bare head `pip-`, which then collides with any publish prefix beginning `pip-`
    including a properly partitioned `pip-release-` that no pull request can write.
    Substituting the values callers pass gives `pip-mlx-`, `pip-collect-` and so on.
    """
    values: set = set()
    pattern = re.compile(
        r"uses:\s*['\"]?\./[\w./-]*" + re.escape(action_dir) + r"[^\n]*\n((?:[ \t]+[^\n]*\n)*)"
    )
    for pth in pr_paths:
        try:
            text = pth.read_text()
        except OSError:
            continue
        for m in pattern.finditer(text):
            for nm in re.finditer(r"^\s+name:\s*['\"]?([A-Za-z0-9][\w.-]*)", m.group(1), re.M):
                values.add(nm.group(1))
    return values

def _pr_reachable_action_dirs(workflows_dir: Path, pr_paths: list) -> set:
    """Composite actions a PR-triggered workflow actually uses.

    Scanning every action under .github/actions treated a publish-only composite's keys
    as a namespace pull requests write, so a publish workflow restoring its OWN action's
    prefix was rejected as PR-poisonable. That is a false failure on a safe
    configuration, and a security lint that cries wolf gets switched off.
    """
    root = workflows_dir.parent
    dirs: set = set()
    seen: set = set()
    queue = [pth for pth in pr_paths]
    while queue:
        pth = queue.pop()
        if pth in seen:
            continue
        seen.add(pth)
        try:
            text = pth.read_text()
        except OSError:
            continue
        for m in re.finditer(r"uses:\s*['\"]?(\./[\w./-]+)", text):
            rel = m.group(1)[2:]
            cand = root.parent / rel
            # A local reusable workflow reference names the .yml file itself rather than a
            # directory containing an action.yml, so it has to be followed on its own.
            if cand.is_file() and cand.suffix in (".yml", ".yaml"):
                dirs.add(cand)
                queue.append(cand)
                continue
            for action in (cand / "action.yml", cand / "action.yaml"):
                if action.is_file():
                    dirs.add(action)
                    queue.append(action)
    return dirs

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
    pr_workflow_paths = [pth for pth, _ in pr_triggered]
    composite_keys: list[str] = []
    for action_path in sorted(_pr_reachable_action_dirs(workflows_dir, pr_workflow_paths)):
        composite_keys.extend(_extract_cache_keys(action_path))
        composite_keys.extend(
            _shell_built_key_prefixes(
                action_path.read_text(),
                _local_action_inputs(pr_workflow_paths, action_path.parent.name),
            )
        )

    # Composite keys belong in the exact comparison as well, not only the prefix one. A
    # PR-reachable action declaring `key: shared-key`, against a publish workflow using
    # that same key and no restore-keys at all, is the original cache-poisoning shape, and
    # it was invisible while this set held workflow-declared keys only.
    pr_keys = {key for _, keys in pr_triggered for key in keys} | set(composite_keys)
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
    pr_heads: set = set()
    undecidable_pr_keys: list[str] = []
    for k in list(pr_keys) + composite_keys:
        cands = _prefix_candidates(k)
        if cands:
            pr_heads.update(cands)
        elif _DELEGATED_KEY.fullmatch(k.strip()):
            # `key: ${{ steps.pip-cache.outputs.key }}` names no namespace of its own; it
            # hands the decision to a composite action, whose real prefix was collected
            # above from that action's own YAML and shell. Reporting it as undecidable
            # would flag every workflow that factors its cache out into an action.
            continue
        else:
            undecidable_pr_keys.append(k)

    for pub_path, prefixes in publish_restore_prefixes:
        for prefix in prefixes:
            pub_heads = _prefix_candidates(prefix)
            if not pub_heads:
                findings.append(
                    f"{pub_path.name}: restore-keys entry {prefix!r} begins with an "
                    "expression, so what it can restore is not decidable here. Give it a "
                    "literal prefix."
                )
                continue
            for pub_head in pub_heads:
                hit = next(
                    (h for h in sorted(pr_heads) if _prefix_compatible(h, pub_head)), None
                )
                if hit is not None:
                    findings.append(
                        f"{pub_path.name}: restore-keys prefix {pub_head!r} matches "
                        f"{hit!r}, a cache key namespace a PR-triggered workflow writes. "
                        "A prefix restore takes the newest matching entry, so this "
                        "publish workflow could adopt a cache a pull request produced "
                        "even though no key is equal. Partition the namespaces, or drop "
                        "the restore-keys fallback on the publish side."
                    )
                    break
            else:
                # Only reachable when nothing matched: an undecidable PR key could still
                # expand into this prefix, so say so rather than passing silently.
                for k in undecidable_pr_keys:
                    findings.append(
                        f"{pub_path.name}: restore-keys prefix {prefix!r} cannot be "
                        f"compared against PR cache key {k!r}, which begins with an "
                        "expression this check cannot expand, so whether the prefix "
                        "reaches that namespace is undecidable. Give the PR key a "
                        "literal prefix."
                    )
                    break

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
